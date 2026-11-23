#!/usr/bin/env python3
"""Convert ORIGINAL RQS toolkit exports into finite-width OPX timing tables.

Reads an unchanged export bundle containing pulses.csv, sequences.csv,
acquisition.csv and bundle.json. Creates separate, new CSVs (including explicit pi/2 gates) without changing
any original RQS files, response vectors, signs, or the acquisition plan.

Toolkit conventions:
    pulses.csv time_s = ideal pi-pulse CENTER, referenced to sensing start.
    sensing starts immediately AFTER the preparation pi/2 pulse.
    sensing duration is measured from the end of preparation to the START
    of the final analysis pi/2 pulse.

Physical microwave timeline:
    pi/2 preparation (pi2_duration_ns)
    free wait
    [pi pulse (pi_duration_ns) + free wait] repeated
    pi/2 analysis (pi2_duration_ns, basis X or Y from acquisition.csv)

All times refer to ideal pulse centers. The finite-width calculation here
checks geometry; it does not compensate for evolution during microwave pulses.
Actual phase signs, macro durations and trigger latencies require validation.

Example, in a notebook (after running Step 4):
    from rqs_opx_export import convert_rqs_bundle_to_opx
    generated = convert_rqs_bundle_to_opx(
        BUNDLE_DIR, pi_duration_ns=88, pi2_duration_ns=44, clock_ns=4
    )
    print(generated)

Command line:
    python rqs_opx_export.py outputs/rqs_phase_locked --pi-ns 88
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _read_rows(file: Path):
    with file.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _save_rows(file: Path, rows, headers):
    with file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _seconds_to_grid_ns(seconds, clock_ns: int, label: str):
    ns = float(seconds) * 1e9
    ticks = round(ns / clock_ns)
    rounded = int(ticks * clock_ns)
    if not math.isfinite(ns) or abs(rounded - ns) > 0.1:
        raise ValueError(f"{label}: {ns} ns is not aligned to the {clock_ns}-ns clock")
    return rounded


def convert_rqs_bundle_to_opx(
    bundle_dir,
    pi_duration_ns: int,
    *,
    pi2_duration_ns: int = 44,
    clock_ns: int = 4,
    min_wait_cycles: int = 4,
    output_dir=None,
):
    """Create OPX-compatible timing CSVs for an original, phase-locked RQS plan.

    Parameters
    ----------
    bundle_dir : str or pathlib.Path
        Directory exported using R.export_bundle(...) in the original toolkit.
    pi_duration_ns : int
        Physical duration of the refocusing pi pulse, in nanoseconds.
    pi2_duration_ns : int
        Duration of BOTH preparation and analysis pi/2 pulses.
    clock_ns : int
        QUA time grid, usually 4 ns.
    min_wait_cycles : int
        Minimum wait supported by the lab sequence/macro (default 4 cycles).
    output_dir : str or Path, optional
        Default is a sibling directory named <bundle>_opx_timing.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths of newly created CSV and summary files.

    Each acquisition row retains plan_id, row_id, basis, shots, and
    outcome_sign. The sign is METADATA FOR THE ORIGINAL DECODER, not a hardware
    operation. OPX should save RAW X/Y readout results against row_id.
    """
    bundle = Path(bundle_dir).expanduser().resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"RQS bundle folder does not exist: {bundle}")
    expected = ["bundle.json", "pulses.csv", "sequences.csv", "acquisition.csv"]
    missing = [name for name in expected if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing original RQS files in {bundle}: {missing}")

    for name, val in (("pi_duration_ns", pi_duration_ns),
                      ("pi2_duration_ns", pi2_duration_ns),
                      ("clock_ns", clock_ns)):
        if isinstance(val, bool) or int(val) != val or val <= 0:
            raise ValueError(f"{name} must be a positive integer in ns")
    pi_duration_ns = int(pi_duration_ns)
    pi2_duration_ns = int(pi2_duration_ns)
    clock_ns = int(clock_ns)
    if pi_duration_ns % clock_ns or pi2_duration_ns % clock_ns:
        raise ValueError("pi and pi/2 durations must be multiples of clock_ns")
    if (pi_duration_ns // 2) % clock_ns:
        raise ValueError("pi half-duration must align to the OPX clock grid")
    if min_wait_cycles < 0 or int(min_wait_cycles) != min_wait_cycles:
        raise ValueError("min_wait_cycles must be a nonnegative integer")
    min_gap_ns = int(min_wait_cycles) * clock_ns

    bundle_info = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    if bundle_info.get("field_mode", "phase_locked") != "phase_locked":
        raise ValueError("This converter supports phase_locked RQS bundles only")
    design_clock_s = bundle_info.get("experiment", {}).get("clock_s")
    if design_clock_s is not None and abs(float(design_clock_s) * 1e9 - clock_ns) > 1e-4:
        raise ValueError("Toolkit's clock_s does not match clock_ns")

    duration_ns = _seconds_to_grid_ns(bundle_info["duration_s"], clock_ns, "duration_s")
    pulse_rows = _read_rows(bundle / "pulses.csv")
    sequence_rows = _read_rows(bundle / "sequences.csv")
    acquisition_rows = _read_rows(bundle / "acquisition.csv")
    if not sequence_rows or not acquisition_rows:
        raise ValueError("sequences.csv and acquisition.csv must be nonempty")

    # Collect pulse centers by physical sequence index; antithetic/virtual
    # response signs are NOT additional gates.
    pulse_by_seq = defaultdict(list)
    for p in pulse_rows:
        seq_id = int(p["sequence_index"])
        idx = int(p["pulse_index"])
        if p["axis"].upper() != "X":
            raise ValueError(f"Unexpected refocusing axis for sequence {seq_id}")
        if abs(float(p["angle_rad"]) - math.pi) > 1e-6:
            raise ValueError(f"Unexpected refocusing angle for sequence {seq_id}")
        center = _seconds_to_grid_ns(p["time_s"], clock_ns,
                                     f"sequence {seq_id}, pulse {idx}")
        pulse_by_seq[seq_id].append((idx, center))

    sequence_ids = {int(s["sequence_index"]) for s in sequence_rows}
    if len(sequence_ids) != len(sequence_rows):
        raise ValueError("Duplicate sequence_index in sequences.csv")
    if set(pulse_by_seq) - sequence_ids:
        raise ValueError("pulses.csv references an unknown sequence_index")

    half_pi_ns = pi_duration_ns // 2
    schedules = {}
    detailed_pulses = []

    for s in sequence_rows:
        sid = int(s["sequence_index"])
        indexed = sorted(pulse_by_seq[sid], key=lambda x: x[0])
        if [idx for idx, _ in indexed] != list(range(len(indexed))):
            raise ValueError(f"Sequence {sid}: missing or repeated pulse_index")
        centers_ns = [center for _, center in indexed]
        if len(centers_ns) != int(s["pulse_count"]):
            raise ValueError(f"Sequence {sid}: pulse_count mismatch")
        if any(b <= a for a, b in zip(centers_ns, centers_ns[1:])):
            raise ValueError(f"Sequence {sid}: pulse centers not increasing")

        # t=0 at the BEGINNING of first pi/2; sensing starts at t=pi2_ns.
        # A pulse centered c ns into sensing starts at pi2_ns+c-pi_ns/2.
        pi_starts = [pi2_duration_ns + c - half_pi_ns for c in centers_ns]
        last_analysis_start = pi2_duration_ns + duration_ns
        wait_ns = []
        previous_gate_end = pi2_duration_ns
        for pi_idx, (center, start) in enumerate(zip(centers_ns, pi_starts)):
            gap = start - previous_gate_end
            if gap < min_gap_ns:
                raise ValueError(
                    f"Sequence {sid}: only {gap} ns before pi pulse {pi_idx}; "
                    f"need at least {min_gap_ns} ns. Adjust compiler gaps."
                )
            if gap % clock_ns:
                raise ValueError(f"Sequence {sid}: wait {gap} off clock grid")
            wait_ns.append(gap)
            previous_gate_end = start + pi_duration_ns
            detailed_pulses.append({
                "sequence_index": sid,
                "pulse_index": pi_idx,
                "center_ns_from_sensing_start": center,
                "start_ns_from_sensing_start": center - half_pi_ns,
                "end_ns_from_sensing_start": center + half_pi_ns,
                "start_ns_from_first_pi2_start": start,
                "duration_ns": pi_duration_ns,
                "end_ns_from_first_pi2_start": start + pi_duration_ns,
            })

        last_gap = last_analysis_start - previous_gate_end
        if last_gap < min_gap_ns:
            raise ValueError(
                f"Sequence {sid}: only {last_gap} ns before final pi/2; "
                f"need at least {min_gap_ns} ns. Adjust compiler edge guard."
            )
        if last_gap % clock_ns:
            raise ValueError(f"Sequence {sid}: final wait off clock grid")
        wait_ns.append(last_gap)

        full_duration = pi2_duration_ns * 2 + pi_duration_ns * len(centers_ns) + sum(wait_ns)
        if full_duration != duration_ns + 2 * pi2_duration_ns:
            raise AssertionError(f"Sequence {sid}: total microwave duration mismatch")
        schedules[sid] = dict(
            sequence_index=sid,
            sensing_duration_ns=duration_ns,
            pi_duration_ns=pi_duration_ns,
            pi2_duration_ns=pi2_duration_ns,
            pulse_count=len(centers_ns),
            pi_centers_ns_from_sensing_start=";".join(map(str, centers_ns)),
            pi_starts_ns_from_first_pi2=";".join(map(str, pi_starts)),
            waits_ns=";".join(map(str, wait_ns)),
            waits_cc=";".join(str(g // clock_ns) for g in wait_ns),
            final_pi2_start_ns_from_first_pi2=last_analysis_start,
            full_mw_duration_ns=full_duration,
        )

    seen_row_ids = set()
    ready_rows = []
    shots_by_basis = Counter()
    for row in acquisition_rows:
        sid = int(row["sequence_index"])
        basis = row["basis"].upper()
        sign = int(row["outcome_sign"])
        shots = int(row["shots"])
        row_id = row["row_id"]
        if sid not in schedules:
            raise ValueError(f"Acquisition row {row_id}: unknown sequence {sid}")
        if row_id in seen_row_ids:
            raise ValueError(f"Duplicate acquisition row_id: {row_id}")
        seen_row_ids.add(row_id)
        if basis not in ("X", "Y") or sign not in (-1, 1) or shots <= 0:
            raise ValueError(f"Acquisition row {row_id}: invalid basis/sign/shots")
        shots_by_basis[basis] += shots
        ready_rows.append({
            "plan_id": row["plan_id"],
            "row_id": row_id,
            "sequence_index": sid,
            "basis": basis,
            "shots": shots,
            "outcome_sign_DECODER_ONLY": sign,
            **schedules[sid],
        })

    if output_dir is None:
        dest = bundle.with_name(bundle.name + "_opx_timing")
    else:
        dest = Path(output_dir).expanduser().resolve()
    if dest == bundle or bundle in dest.parents:
        raise ValueError("Output must be outside the original RQS bundle")
    dest.mkdir(parents=True, exist_ok=True)

    acquisition_path = dest / "opx_ready_acquisition.csv"
    pulses_path = dest / "opx_pulses_with_duration.csv"
    _save_rows(acquisition_path, ready_rows, list(ready_rows[0]))
    _save_rows(pulses_path, detailed_pulses, [
        "sequence_index", "pulse_index", "center_ns_from_sensing_start",
        "start_ns_from_sensing_start", "end_ns_from_sensing_start",
        "start_ns_from_first_pi2_start", "duration_ns",
        "end_ns_from_first_pi2_start",
    ])
    # Full microwave gate schedule for every acquisition row, with BOTH pi/2
    # pulses explicitly represented. Rows are repeated per acquisition row so
    # that measurement basis, row_id, and shot grouping remain unambiguous.
    # Each schedule is for ONE sensing shot (repeat the schedule row["shots"] times).
    # The analysis rotation phases below are nominal active-rotation axes;
    # they must be verified against the lab's IQ/microwave and readout signs.
    full_event_rows = []
    for row in ready_rows:
        sid = row["sequence_index"]
        basis = row["basis"]
        n_pi = row["pulse_count"]
        waits = list(map(int, row["waits_ns"].split(";")))
        if len(waits) != n_pi + 1:
            raise AssertionError(f"Acquisition row {row['row_id']}: wrong wait count")
        starts = ([int(s) for s in row["pi_starts_ns_from_first_pi2"].split(";")]
                  if n_pi else [])
        ends = [s + pi_duration_ns for s in starts]
        events = [(0, "PREPARE_PLUS_X", "PI_OVER_2", "+Y", 0,
                   pi2_duration_ns, 0)]
        for i, start in enumerate(starts):
            events.append((i + 1, "REFOCUS", "PI", "+X", start,
                           pi_duration_ns, waits[i]))
        events.append((n_pi + 1, f"MEASURE_{basis}", "PI_OVER_2",
                       "-Y" if basis == "X" else "+X",
                       last_analysis_start, pi2_duration_ns, waits[-1]))
        previous_end = 0
        for gate_idx, name, angle, axis, start, width, wait_before in events:
            if start - previous_end != wait_before:
                raise AssertionError(
                    f"Acquisition row {row['row_id']}, gate {gate_idx}: "
                    f"expected wait {wait_before}, got {start - previous_end}")
            if (start % clock_ns or width % clock_ns
                    or wait_before % clock_ns):
                raise AssertionError("Gate event is off the hardware clock grid")
            if gate_idx and wait_before < min_gap_ns:
                raise AssertionError("Wait shorter than OPX min_wait_cycles")
            full_event_rows.append({
                "row_id": row["row_id"],
                "sequence_index": sid,
                "measurement_basis": basis,
                "shots": row["shots"],
                "gate_index": gate_idx,
                "operation": name,
                "pulse_type": angle,
                "nominal_rotation_axis": axis,
                "start_ns_from_first_pi2_start": start,
                "duration_ns": width,
                "end_ns_from_first_pi2_start": start + width,
                "wait_before_ns": wait_before,
                "wait_before_cc": wait_before // clock_ns,
            })
            previous_end = start + width
        if previous_end != duration_ns + 2 * pi2_duration_ns:
            raise AssertionError(f"Acquisition row {row['row_id']}: wrong end time")

    full_events_path = dest / "opx_full_microwave_sequence.csv"
    _save_rows(full_events_path, full_event_rows, list(full_event_rows[0]))

    summary_path = dest / "README_TIMING.txt"
    summary_path.write_text(
        f"SOURCE ORIGINAL RQS BUNDLE: {bundle}\n"
        f"UNMODIFIED: pulses.csv, sequences.csv, acquisition.csv, pool.npz, bundle.json\n"
        f"ACQUISITION ROWS: {len(ready_rows)}\n"
        f"PHYSICAL SEQUENCES: {len(schedules)}\n"
        f"TOTAL SHOTS: {sum(shots_by_basis.values())}\n"
        f"X / Y SHOTS: {shots_by_basis['X']} / {shots_by_basis['Y']}\n"
        f"SENSING DURATION: {duration_ns} ns\n"
        f"PI / PI2 DURATION: {pi_duration_ns} / {pi2_duration_ns} ns\n"
        f"CLOCK: {clock_ns} ns, MIN FREE WAIT: {min_gap_ns} ns\n"
        f"FIRST-PI2 START TO FINAL-PI2 END: {duration_ns + 2*pi2_duration_ns} ns\n\n"
        "TIMING CONVENTIONS:\n"
        "  sensing t=0 is immediately after initial pi/2 (not at its beginning)\n"
        "  pi center c (sensing-relative) => physical pi starts at c - pi_ns/2\n"
        "  in time relative to first pi/2 start, pi starts at pi2_ns+c-pi_ns/2\n"
        "  waits_ns and waits_cc contain one wait after initial pi/2 and one\n"
        "  after EVERY refocusing pi before final analysis pi/2\n"
        "  final pi/2 analysis basis comes from each acquisition row (X/Y)\n"
        "  opx_full_microwave_sequence.csv explicitly lists INITIAL/FINAL pi/2\n"
        "  and every refocusing pi for every row_id, in physical gate order.\n"
        "  Nominal axes: PREPARE +Y, REFOCUS +X, ANALYZE X: -Y, Y: +X.\n"
        "  Calibrate axes/signs against the lab's actual microwave convention.\n\n"
        "IMPORTANT: outcome_sign_DECODER_ONLY is not a microwave command.\n"
        "  Preserve RAW physical X/Y measurements keyed to original row_id.\n"
        "  The ORIGINAL RQS decoder applies virtual-sign/parity corrections.\n"
        "  Full gate schedule is per acquisition row, NOT per individual shot.\n"
        "  Repeat each row's schedule by the specified shots count.\n"
        "  Do not modify the original bundle or its saved response vectors.\n"
        "  Finite-pulse evolution is NOT corrected in the original ideal response.\n"
        "  Verify QUA macro durations, X/Y phases, trigger/latency and SCC timing.\n",
        encoding="utf-8",
    )
    return {
        "acquisition_csv": acquisition_path,
        "pulses_csv": pulses_path,
        "full_microwave_csv": full_events_path,
        "summary_txt": summary_path,
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path, help="Original RQS export bundle folder")
    parser.add_argument("--pi-ns", type=int, default=88, help="Refocusing pi duration (ns)")
    parser.add_argument("--pi2-ns", type=int, default=44, help="Preparation/analysis pi/2 duration (ns)")
    parser.add_argument("--clock-ns", type=int, default=4, help="Hardware clock period (ns)")
    parser.add_argument("--min-wait-cycles", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None, help="Output folder")
    args = parser.parse_args()
    output = convert_rqs_bundle_to_opx(
        args.bundle_dir,
        pi_duration_ns=args.pi_ns,
        pi2_duration_ns=args.pi2_ns,
        clock_ns=args.clock_ns,
        min_wait_cycles=args.min_wait_cycles,
        output_dir=args.output,
    )
    print("Created OPX timing tables:")
    for description, path in output.items():
        print(f"  {description}: {path}")


if __name__ == "__main__":
    _main()
