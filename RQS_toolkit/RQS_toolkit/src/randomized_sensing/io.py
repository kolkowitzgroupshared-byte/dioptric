"""Portable randomized plans, pulse tables, physical count templates, and hashes."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .experiment import Experiment
from .pulses import Pool, Sequence, audit
from .readout import (AcquisitionPlan, TimedCounts, AveragedOutcomes,
                      validate_plan, validate_counts, validate_averages)

FILES = {"bundle.json", "pool.npz", "sequences.csv", "pulses.csv", "intervals.csv",
         "acquisition.csv", "counts_template.csv", "RUN.txt"}
COUNT_COLUMNS = ["plan_id", "row_id", "shots", "plus"]
AVERAGE_TEMPLATE = "averages_template.csv"
AVERAGE_COLUMNS = ["plan_id", "row_id", "repetitions", "mean"]


def _csv(path, columns, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _audit_pool(pool):
    for i in range(len(pool.sequences)):
        check = audit(pool, i)
        if not check["timing_valid"] or check["response_error"] > 1e-9*max(1., np.max(abs(pool.Z[i]))):
            raise ValueError(f"Invalid timing or response for pool entry {i}.")
    np.testing.assert_allclose(pool.Z[len(pool.sequences):], -pool.Z[:len(pool.sequences)], rtol=0, atol=0)


def export_bundle(plan, directory, *, overwrite=False):
    validate_plan(plan); _audit_pool(plan.pool)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError("Output directory is nonempty; choose a new folder or overwrite=True.")
    directory.mkdir(parents=True, exist_ok=True)
    p, identity = plan.pool, plan.plan_id
    metadata = dict(schema_version=1, kind="randomized", experiment=p.experiment.to_dict(),
                    duration_s=p.duration_s, start_time_s=p.start_time_s,
                    design_clock_s=p.design_clock_s, settings=p.settings, rows=plan.rows, seed=plan.seed)
    running = plan.field_mode == "free_running"
    if running:
        metadata["field_mode"] = plan.field_mode
    (directory/"bundle.json").write_text(json.dumps(metadata, indent=2)+"\n")
    lengths = [len(q.switch_times_s) for q in p.sequences]
    np.savez_compressed(directory/"pool.npz", responses=p.Z,
                        offsets=np.r_[0, np.cumsum(lengths)],
                        switch_times_s=np.array([t for q in p.sequences for t in q.switch_times_s]),
                        initial_signs=np.array([q.initial_sign for q in p.sequences], dtype=np.int8))
    summary, pulses, intervals = [], [], []
    for index in sorted({r["sequence_index"] for r in plan.rows}):
        q = p.sequence(index)
        edges = np.r_[0., q.switch_times_s, p.duration_s]
        summary.append(dict(sequence_index=index, duration_s=p.duration_s, start_time_s="" if running else p.start_time_s,
                            virtual_initial_sign=q.initial_sign, final_frame_sign=(-1)**len(q.switch_times_s),
                            pulse_count=len(q.switch_times_s), minimum_interval_s=float(np.diff(edges).min())))
        pulses.extend(dict(sequence_index=index, pulse_index=j, time_s=t,
                           reference_time_s="" if running else p.start_time_s+t, axis="X", angle_rad=np.pi)
                      for j, t in enumerate(q.switch_times_s))
        intervals.extend(dict(sequence_index=index, begin_s=float(left), end_s=float(right),
                              physical_modulation=(-1)**j, virtual_modulation=q.initial_sign*(-1)**j)
                         for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])))
    _csv(directory/"sequences.csv", list(summary[0]), summary)
    _csv(directory/"pulses.csv", ["sequence_index", "pulse_index", "time_s", "reference_time_s", "axis", "angle_rad"], pulses)
    _csv(directory/"intervals.csv", ["sequence_index", "begin_s", "end_s", "physical_modulation", "virtual_modulation"], intervals)
    acquisition = [dict(plan_id=identity, **r) for r in plan.rows]
    columns = ["plan_id", "row_id", "sequence_index", "basis", "shots", "outcome_sign"]
    count_columns = COUNT_COLUMNS+["start_time_s"] if running else COUNT_COLUMNS
    template = [dict(plan_id=identity, row_id=r["row_id"], shots=r["shots"], plus="") for r in plan.rows]
    if running:
        columns += ["planned_start_time_s"]
        for row in acquisition:
            row["planned_start_time_s"] = row.pop("start_time_s")
        for row in template:
            row["start_time_s"] = ""
    _csv(directory/"acquisition.csv", columns, acquisition)
    _csv(directory/"counts_template.csv", count_columns, template)
    files = FILES
    if not running:
        _csv(directory/AVERAGE_TEMPLATE, AVERAGE_COLUMNS,
             [dict(plan_id=identity, row_id=r["row_id"], repetitions=r["shots"], mean="") for r in plan.rows])
        files = FILES | {AVERAGE_TEMPLATE}
    timing = """The field runs continuously. Each acquisition row is ONE physical shot;
planned_start_time_s is a proposed schedule, not a measured timestamp.
Record EVERY actual sensing-start time in counts.start_time_s, in seconds from
the SAME stable clock origin used to define the Fourier amplitudes. Use elapsed
clock times, not Unix timestamps. Choose timing precision relative to the fastest
tone. Timestamps account for source phase evolution; source phase reset is not
required. The source amplitudes and frequencies must remain fixed.
Pulses.csv contains reusable relative times; absolute reference times are blank.
Never combine outcomes taken at different start times into one count row.
The decoder uses actual timestamps, including delays from the planned schedule.""" if running else """Play acquisition rows with the listed repetitions; grouping iid draws is valid
only for a fixed field. The tone phases at EVERY trigger must match start_time_s
relative to the same phase origin. Phase-lock/reset the source; trigger waiting
time is not predicted. Do not reuse a plan at arbitrary unknown starting phases."""
    estimation = """RQS is the default: whiten each response in the bank frame, then rotate
that back-projection to its actual start. Apply the shared X-mean guard and
per-tone correction. At least one X and one Y shot are required; there is no
6N shot floor for the moment estimate. Finite-pool/moment-law bias may remain.
TIMESTAMPED-ML optionally refines amplitudes and contrast with a constrained
binary likelihood, adding the moment estimate to the legacy three starts.
Below 3*2N shots or with insufficient Y-response rank it returns RQS and sets
ml_applied=False with a refinement_skipped_reason. Actual fit failures raise.
Neither method certifies accuracy, confidence, or a unique/global optimum.""" if running else """RQS uses approximate moment-law inversion with per-tone correction; RQS-ML
adds local likelihood refinement. No uniform accuracy/confidence is certified.
T2/visibility set simulation contrast and the small-X-mean guard only; contrast
cancels in the unguarded moment ratio."""
    (directory/"RUN.txt").write_text(f"""Prepare |+X>; physical modulation starts at +1 for EVERY sequence.
Apply only the internal X(pi) pulses listed in pulses.csv. No initial sign flip
or extra endpoint pulse is needed. Ideal instantaneous pulses are assumed.
Use physical X or Y analysis with the convention that a pulse-free shot has
mean_X=bias+C*cos(phi), mean_Y=bias+C*sin(phi), phi=2*pi*coupling*integral(b).
Odd pulse counts invert physical Y readout. Antithetic pool entries use the
same physical pulses with a virtual response sign. Both effects are recorded
in outcome_sign: corrected_mean=outcome_sign*(physical_mean-bias).
The supplied decoder applies this correction; fill counts with RAW physical
+1 counts, without changing signs or subtracting bias yourself.
The initial and final preparation/analysis operations are outside sensing time.
Every sensing interval conservatively obeys max(min_gap,edge_guard), rounded UP
to the design grid. A specified hardware clock overrides the numerical grid.
{timing}
Counts template: save a COPY with integer physical +1 counts, preserving IDs.
Choose observable weights after acquisition and pass them to decode. Different
queries reuse the same pulses and raw counts; new bundles contain no queries.
{estimation}
Calibrate the additive readout bias.
Finite pulse evolution, unrecorded timing jitter, field drift and calibration uncertainty
are not modeled. Active sensing and specified per-shot overhead are separate.
""")
    if not running:
        with (directory/"RUN.txt").open("a") as stream:
            stream.write("""Alternatively, fill a COPY of averages_template.csv with raw physical means
in [-1,1], one mean per sequence/basis row. Repetitions can record actual
averaging counts and need not equal planned shots. Preserve row and plan IDs.
read_averages/decode accept rounded means without integer-count conversion.
RQS keeps the plan's random-draw weights; actual repetitions weight RQS-ML.
Retain identical tone phases for every repetition of a row. Means from shots
at different unknown phases cannot replace per-shot timestamped data.
""")
    (directory/"manifest.json").write_text(json.dumps(dict(schema_version=1, plan_id=identity,
        sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in sorted(files)}), indent=2)+"\n")
    return directory


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or
            set(manifest.get("sha256", {})) not in (FILES, FILES | {AVERAGE_TEMPLATE})):
        raise ValueError("Unknown or incomplete randomized bundle manifest.")
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Bundle file changed: {name}")
    data = json.loads((directory/"bundle.json").read_text())
    if data.get("schema_version") != 1 or data.get("kind") != "randomized":
        raise ValueError("Unknown randomized bundle schema.")
    with np.load(directory/"pool.npz", allow_pickle=False) as arrays:
        offsets, signs, times = arrays["offsets"], arrays["initial_signs"], arrays["switch_times_s"]
        if (len(offsets) != len(signs)+1 or offsets[0] != 0 or offsets[-1] != len(times)
                or np.any(np.diff(offsets) < 0) or np.any(abs(signs) != 1)):
            raise ValueError("Invalid packed pulse sequences.")
        sequences = tuple(Sequence(tuple(times[left:right].tolist()), int(sign))
                          for left, right, sign in zip(offsets[:-1], offsets[1:], signs))
        if "cosine_weights" in data["experiment"]:
            raise ValueError("This bundle was saved by an earlier toolkit version that stored query "
                             "weights; load it with the full Algorithmic Sensing Toolkit.")
        pool = Pool(Experiment(**data["experiment"]), data["duration_s"], data["start_time_s"],
                    data["design_clock_s"], sequences, arrays["responses"].copy(), data["settings"])
    plan = AcquisitionPlan(pool, tuple(data["rows"]), data["seed"], data.get("field_mode", "phase_locked"))
    validate_plan(plan); _audit_pool(pool)
    if plan.plan_id != manifest["plan_id"]:
        raise ValueError("Bundle identity mismatch.")
    return plan


def write_counts(path, plan, counts):
    if Path(path).name in FILES | {"manifest.json", AVERAGE_TEMPLATE}:
        raise ValueError("Write measured counts to a separate file.")
    validate_counts(plan, counts)
    identity = plan.plan_id
    running = plan.field_mode == "free_running"
    positive = counts.positive_counts if running else counts
    rows = [dict(plan_id=identity, row_id=r["row_id"], shots=r["shots"], plus=positive[r["row_id"]]) for r in plan.rows]
    if running:
        for row in rows:
            row["start_time_s"] = counts.start_times_s[row["row_id"]]
    _csv(path, COUNT_COLUMNS+(["start_time_s"] if running else []), rows)


def read_counts(path, plan):
    identity, expected, counts = plan.plan_id, {r["row_id"]: r for r in plan.rows}, {}
    running, times = plan.field_mode == "free_running", {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COUNT_COLUMNS+(["start_time_s"] if running else []):
            raise ValueError("Use the counts_template.csv columns unchanged.")
        for row in reader:
            key = row["row_id"]
            if key not in expected or key in counts or row["plan_id"] != identity:
                raise ValueError("Unknown, duplicate, or mismatched count row.")
            try:
                shots, plus = int(row["shots"]), int(row["plus"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Fill every count with an integer; the blank template is not data.") from exc
            if shots != expected[key]["shots"]:
                raise ValueError("Shot totals do not match the acquisition.")
            counts[key] = plus
            if running:
                try:
                    times[key] = float(row["start_time_s"])
                except (ValueError, TypeError) as exc:
                    raise ValueError("Record the actual start_time_s of every shot.") from exc
    result = TimedCounts(counts, times) if running else counts
    validate_counts(plan, result)
    return result


def write_averages(path, plan, means, *, repetitions=None):
    """Save raw means, including rounded averages, separately from plan files."""
    if Path(path).name in FILES | {"manifest.json", AVERAGE_TEMPLATE}:
        raise ValueError("Write measured averages to a separate file.")
    if isinstance(means, AveragedOutcomes):
        if repetitions is not None:
            raise ValueError("Repetitions are already part of AveragedOutcomes.")
        averages = means
    else:
        averages = AveragedOutcomes(means, repetitions)
    averages = validate_averages(plan, averages)
    identity = plan.plan_id
    _csv(path, AVERAGE_COLUMNS,
         [dict(plan_id=identity, row_id=r["row_id"], repetitions=averages.repetitions[r["row_id"]],
               mean=averages.means[r["row_id"]]) for r in plan.rows])


def read_averages(path, plan):
    """Load complete row-ID means/repetitions, preserving the saved acquisition."""
    identity, expected, means, repetitions = plan.plan_id, {r["row_id"] for r in plan.rows}, {}, {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != AVERAGE_COLUMNS:
            raise ValueError("Use the averages_template.csv columns unchanged.")
        for row in reader:
            key = row["row_id"]
            if key not in expected or key in means or row["plan_id"] != identity:
                raise ValueError("Unknown, duplicate, or mismatched average row.")
            try:
                means[key] = float(row["mean"])
                repetitions[key] = int(row["repetitions"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Fill every mean with a finite number and repetitions with a positive integer.") from exc
    return validate_averages(plan, AveragedOutcomes(means, repetitions))
