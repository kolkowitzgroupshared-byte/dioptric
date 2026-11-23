"""Uniform random acquisition, raw counts or means, and reusable observables."""

from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
import hashlib
import json
import warnings

import numpy as np
from scipy.linalg import cho_solve

from .pulses import Pool, integer
from ._estimator import moment_estimate, finish_moment, polish
from ._timestamped import validate_times, fit as fit_timestamped, RefinementUnavailable


@dataclass(frozen=True)
class AcquisitionPlan:
    pool: Pool
    rows: tuple[dict, ...]
    seed: int
    field_mode: str = "phase_locked"

    @property
    def total_shots(self):
        return sum(r["shots"] for r in self.rows)

    @property
    def active_time_s(self):
        return self.total_shots*self.pool.duration_s

    @property
    def time_with_overhead_s(self):
        return self.active_time_s+self.total_shots*self.pool.experiment.overhead_s

    @property
    def scheduled_span_s(self):
        """First sensing start to last sensing end; excludes final readout overhead."""
        if self.field_mode == "phase_locked":
            return None
        return self.rows[-1]["start_time_s"]-self.rows[0]["start_time_s"]+self.pool.duration_s

    @property
    def plan_id(self):
        p = self.pool
        metadata = dict(experiment=p.experiment.to_dict(), duration_s=p.duration_s,
                        start_time_s=p.start_time_s, design_clock_s=p.design_clock_s,
                        settings=p.settings, rows=self.rows, seed=self.seed,
                        sequences=[(s.switch_times_s, s.initial_sign) for s in p.sequences])
        if self.field_mode != "phase_locked":
            metadata["field_mode"] = self.field_mode
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(p.Z.astype("<f8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class Estimate:
    amplitudes: tuple[float, ...]
    values: tuple[float, ...]
    method: str
    x_mean: float
    guard_floor: float | None
    floor_active: bool
    ml_applied: bool
    fitted_contrast: float | None
    fit_converged: bool | None = None
    response_condition: float | None = None
    refinement_skipped_reason: str | None = None


@dataclass(frozen=True)
class TimedCounts:
    """Raw per-shot positive counts and ACTUAL starts, keyed by acquisition row ID."""

    positive_counts: dict[str, int]
    start_times_s: dict[str, float]


@dataclass(frozen=True)
class TimedOutcomes:
    """Raw values in [-1,1] and actual starts, one value per acquisition row.

    Each value must belong to its one recorded starting phase. This does not
    permit averaging shots taken at different phases into a single timestamp.
    """

    values: Mapping[str, float] | Sequence[float] | np.ndarray
    start_times_s: Mapping[str, float] | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class AveragedOutcomes:
    """Raw physical means in [-1,1], keyed by row ID or in plan.rows order.

    Repetitions default to the planned shot counts; a scalar, vector, or row-ID
    mapping can supply actual counts for likelihood weighting. RQS moments keep
    the plan's random-draw multiplicities, independently of repetition allocation.
    """

    means: Mapping[str, float] | Sequence[float] | np.ndarray
    repetitions: int | Mapping[str, int] | Sequence[int] | np.ndarray | None = None


def _row_values(plan, values, name, *, scalar=False):
    if isinstance(values, Mapping):
        if set(values) != {r["row_id"] for r in plan.rows}:
            raise ValueError(f"{name} must match every acquisition row exactly.")
        values = [values[r["row_id"]] for r in plan.rows]
    elif scalar and np.ndim(values) == 0:
        values = [values]*len(plan.rows)
    array = np.asarray(values)
    if array.shape != (len(plan.rows),) or np.iscomplexobj(array):
        raise ValueError(f"Supply one real {name} per acquisition row, in plan.rows order.")
    return array.tolist()


def validate_averages(plan, averages):
    """Resolve means/repetitions without rounding averaged outcomes to counts."""
    validate_plan(plan)
    if plan.field_mode != "phase_locked":
        raise ValueError("Averaged outcomes require phase_locked acquisition: freely running shots at different phases need individual actual timestamps and values.")
    values = _row_values(plan, averages.means, "mean")
    try:
        values = np.asarray(values, float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Means must be finite numbers in [-1, 1].") from exc
    if not np.isfinite(values).all() or np.any(abs(values) > 1):
        raise ValueError("Means must be finite numbers in [-1, 1].")
    repetitions = ([r["shots"] for r in plan.rows] if averages.repetitions is None else
                   _row_values(plan, averages.repetitions, "repetitions", scalar=True))
    repetitions = [integer(n, "repetitions", 1) for n in repetitions]
    ids = [r["row_id"] for r in plan.rows]
    return AveragedOutcomes(dict(zip(ids, values.tolist())), dict(zip(ids, repetitions)))


def _times(plan, start_times_s=None):
    if start_times_s is None:
        start_times_s = [r["start_time_s"] for r in plan.rows]
    elif isinstance(start_times_s, Mapping):
        if set(start_times_s) != {r["row_id"] for r in plan.rows}:
            raise ValueError("Start timestamps must match every acquisition row exactly.")
        start_times_s = [start_times_s[r["row_id"]] for r in plan.rows]
    return validate_times(start_times_s, len(plan.rows), plan.pool.duration_s)


def validate_plan(plan):
    if plan.field_mode not in ("phase_locked", "free_running"):
        raise ValueError("field_mode must be 'phase_locked' or 'free_running'.")
    seen = set()
    for i, row in enumerate(plan.rows):
        index = integer(row["sequence_index"], "sequence_index")
        if index >= len(plan.pool.Z):
            raise ValueError("sequence_index is outside the pool.")
        expected = int(plan.pool.y_outcome_signs[index]) if row["basis"] == "Y" else 1
        expected_id = f"shot_{i}" if plan.field_mode == "free_running" else f"{row['basis']}_{index}"
        if (row["basis"] not in ("X", "Y") or row["row_id"] in seen
                or row["row_id"] != expected_id
                or row["outcome_sign"] != expected):
            raise ValueError("Invalid acquisition basis, row identity, or outcome sign.")
        integer(row["shots"], "shots", 1)
        if plan.field_mode == "free_running" and row["shots"] != 1:
            raise ValueError("Freely running acquisitions require one timestamped outcome per shot.")
        seen.add(row["row_id"])
    if {r["basis"] for r in plan.rows} != {"X", "Y"}:
        raise ValueError("Acquisition requires both X and Y shots.")
    if plan.field_mode == "free_running":
        _times(plan)


def plan_acquisition(pool, *, n_shots, x_fraction=.2, seed=5,
                     field_mode="phase_locked", start_times_s=None, shot_period_s=None):
    """Freeze uniform draws WITH replacement.

    n_shots is a user-chosen budget, not an accuracy/confidence certificate.
    Physical pulses always start in the +1 frame. outcome_sign converts the
    physical Y mean (after bias subtraction) to the signed pool convention.
    phase_locked groups repeated sequence/basis shots. free_running retains one
    row per shot, with interleaved random X/Y bases and nominal start timestamps.
    Supply a full start schedule or a shot period; actual timestamps go in counts.
    """
    n_shots = integer(n_shots, "n_shots", 2 if field_mode == "free_running" else 4)
    if not np.isfinite(x_fraction) or not 0 < x_fraction < 1:
        raise ValueError("x_fraction must lie strictly between 0 and 1.")
    nx = min(n_shots-1, max(1, int(round(n_shots*x_fraction))))
    seed = integer(seed, "seed")
    rng = np.random.default_rng(seed)
    rows = []
    if field_mode == "free_running":
        if start_times_s is not None and shot_period_s is not None:
            raise ValueError("Supply start_times_s or shot_period_s, not both.")
        if start_times_s is None:
            period = pool.duration_s+pool.experiment.overhead_s if shot_period_s is None else shot_period_s
            if not np.isfinite(period) or period < pool.duration_s+pool.experiment.overhead_s:
                raise ValueError("shot_period_s must include sensing duration and specified per-shot overhead.")
            cycles = np.asarray(pool.experiment.frequencies_hz)*period
            if np.max(abs(cycles-np.rint(cycles))) < 1e-9:
                warnings.warn("The shot period repeats every tone phase: timestamps add no phase diversity. Pulse patterns may still give a full-rank design; consider a noncommensurate period if phase diversity is desired.", RuntimeWarning, stacklevel=2)
            start_times_s = pool.start_time_s+np.arange(n_shots)*period
        starts = validate_times(start_times_s, n_shots, pool.duration_s)
        indices = rng.integers(len(pool.Z), size=n_shots)
        bases = np.full(n_shots, "Y"); bases[rng.choice(n_shots, nx, replace=False)] = "X"
        for i, (index, basis, start) in enumerate(zip(indices, bases, starts)):
            index = int(index)
            sign = int(pool.y_outcome_signs[index]) if basis == "Y" else 1
            rows.append(dict(row_id=f"shot_{i}", sequence_index=index, basis=str(basis),
                             shots=1, outcome_sign=sign, start_time_s=float(start)))
        plan = AcquisitionPlan(pool, tuple(rows), seed, field_mode)
        validate_plan(plan)
        return plan
    if field_mode != "phase_locked":
        raise ValueError("field_mode must be 'phase_locked' or 'free_running'.")
    if start_times_s is not None or shot_period_s is not None:
        raise ValueError("Per-shot timing options require field_mode='free_running'.")
    for basis, shots in (("Y", n_shots-nx), ("X", nx)):
        multiplicities = np.bincount(rng.integers(len(pool.Z), size=shots), minlength=len(pool.Z))
        for index in np.flatnonzero(multiplicities):
            index = int(index)
            sign = int(pool.y_outcome_signs[index]) if basis == "Y" else 1
            rows.append(dict(row_id=f"{basis}_{index}", sequence_index=index, basis=basis,
                             shots=int(multiplicities[index]), outcome_sign=sign))
    plan = AcquisitionPlan(pool, tuple(rows), int(seed))
    validate_plan(plan)
    return plan


def validate_counts(plan, counts):
    validate_plan(plan)
    if plan.field_mode == "free_running":
        if not isinstance(counts, TimedCounts):
            raise ValueError("Freely running data require TimedCounts with actual start timestamps.")
        if counts.start_times_s is None:
            raise ValueError("Supply actual start timestamps for every freely running outcome.")
        _times(plan, counts.start_times_s)
        positive = counts.positive_counts
    else:
        if isinstance(counts, TimedCounts):
            raise ValueError("Timestamped data require field_mode='free_running'.")
        positive = counts
    if set(positive) != {r["row_id"] for r in plan.rows}:
        raise ValueError("Counts must match every acquisition row exactly.")
    for row in plan.rows:
        value = integer(positive[row["row_id"]], "positive count")
        if value > row["shots"]:
            raise ValueError("A positive count exceeds its shot total.")


def _amplitudes(p, amplitudes):
    a = np.asarray(amplitudes, float)
    if a.shape != (2*p.N,) or not np.isfinite(a).all() or np.linalg.norm(a) > p.experiment.amplitude_bound*(1+1e-12):
        raise ValueError("amplitudes must have length 2N and obey amplitude_bound.")
    return a


def evaluate_observables(amplitudes, *, cosine_weights, sine_weights):
    """Evaluate any later-chosen linear queries from an amplitude estimate."""
    a = np.asarray(amplitudes, float)
    if a.ndim != 1 or not a.size or a.size % 2 or not np.isfinite(a).all():
        raise ValueError("amplitudes must be a finite, nonempty vector of length 2N.")
    c, s = (np.asarray(w, float) for w in (cosine_weights, sine_weights))
    if (c.ndim not in (1, 2) or c.shape != s.shape or not c.size
            or c.shape[-1] != len(a)//2 or not np.isfinite(c).all() or not np.isfinite(s).all()):
        raise ValueError("Weights must be matching finite (N,) vectors or (M,N) arrays.")
    return tuple((np.atleast_2d(np.concatenate((c, s), axis=-1))@a).tolist())


def _values(pool, amplitudes, cosine_weights, sine_weights):
    if cosine_weights is None and sine_weights is None:
        # Earlier bundles may carry default queries; new acquisitions have none.
        if not hasattr(pool.experiment, "weights"):
            return ()
        cosine_weights, sine_weights = pool.experiment.cosine_weights, pool.experiment.sine_weights
    elif cosine_weights is None or sine_weights is None:
        raise ValueError("Supply both cosine_weights and sine_weights after acquisition.")
    return evaluate_observables(amplitudes, cosine_weights=cosine_weights, sine_weights=sine_weights)


def _timed_estimate(plan, positive, times, method, cosine_weights, sine_weights):
    method = "RQS" if method is None else method
    if method not in ("RQS", "TIMESTAMPED-ML"):
        raise ValueError("Freely running data use method='RQS' or 'TIMESTAMPED-ML'.")
    a, floor, mu = _timed_moment(plan.pool, plan.rows, positive, times)
    contrast, condition, converged, skipped = None, None, None, None
    applied = False
    if method == "TIMESTAMPED-ML":
        try:
            a, contrast, _, condition = fit_timestamped(plan.pool, plan.rows, positive, times,
                                                       moment_start=a)
            applied, converged = True, True
        except RefinementUnavailable as exc:
            skipped = str(exc)
    values = _values(plan.pool, a, cosine_weights, sine_weights)
    return Estimate(tuple(a.tolist()), values, method, mu, float(floor), bool(mu < floor),
                    applied, contrast, converged, condition, skipped)


def _timed_moment(pool, rows, positive, times):
    """Whiten in the bank frame, THEN rotate each Y back-projection to its start."""
    is_y = np.array([r["basis"] == "Y" for r in rows])
    indices = np.array([r["sequence_index"] for r in rows])
    shots = np.array([r["shots"] for r in rows])
    raw = np.array([2*positive[r["row_id"]]/r["shots"]-1 for r in rows])-pool.experiment.readout_bias
    signs = np.array([r["outcome_sign"] for r in rows])
    mu = float(np.average(raw[~is_y], weights=shots[~is_y]))
    selected = np.flatnonzero(is_y)
    back = np.zeros(2*pool.N)
    half = len(pool.sequences)
    # Repeated acquisitions reuse a <=128 MB bank-frame solve; larger banks
    # retain the bounded per-batch solve. Negative entries need no extra cache.
    if half*2*pool.N <= 16_000_000 and not hasattr(pool, "_back_projection"):
        pool._back_projection = cho_solve(pool.chol, pool.Z[:half].T).T
    for begin in range(0, len(selected), 512):
        take = selected[begin:begin+512]
        if hasattr(pool, "_back_projection"):
            ii = indices[take]
            u = pool._back_projection[ii % half]*np.where(ii < half, 1., -1.)[:, None]
        else:
            u = cho_solve(pool.chol, pool.Z[indices[take]].T).T
        rotated = pool.rotate_vectors(u, np.asarray(times)[take])
        back += (raw[take]*signs[take]*shots[take])@rotated
    a, floor = finish_moment(pool, back/shots[is_y].sum(), mu)
    return a, floor, mu


def population_estimate(pool, amplitudes, *, cosine_weights=None, sine_weights=None,
                        start_times_s=None, method=None):
    """Noise-free inversion diagnostic for a pool or timestamped acquisition plan.

    Freely running plans use exact per-shot probabilities in RQS (default) or
    optional TIMESTAMPED-ML; waveform/schedule sampling variation remains.
    """
    if isinstance(pool, AcquisitionPlan):
        plan = pool
        validate_plan(plan)
        pool = plan.pool
        if plan.field_mode == "free_running":
            a = _amplitudes(pool, amplitudes)
            times = _times(plan, start_times_s)
            probabilities = _timed_probabilities(plan, a, times)
            positive = {r["row_id"]: float(p) for r, p in zip(plan.rows, probabilities)}
            result = _timed_estimate(plan, positive, times, method, cosine_weights, sine_weights)
            return replace(result, method=result.method+"-population")
    if method not in (None, "RQS"):
        raise ValueError("A phase-locked population check uses method='RQS'.")
    if start_times_s is not None:
        raise ValueError("Per-shot timestamps require a freely running acquisition plan.")
    a = _amplitudes(pool, amplitudes)
    phase = pool.Z@a
    c = pool.experiment.contrast(pool.duration_s)
    mu = float(c*np.mean(np.cos(phase)))
    estimated, floor = moment_estimate(pool, c*np.sin(phase)@pool.Z/len(pool.Z), mu)
    values = _values(pool, estimated, cosine_weights, sine_weights)
    return Estimate(tuple(estimated.tolist()), values, "RQS-population", mu,
                    float(floor), bool(mu < floor), False, None)


def _timed_probabilities(plan, amplitudes, times):
    p = plan.pool
    z = p.responses_at(np.array([r["sequence_index"] for r in plan.rows]), times)
    phase = z@amplitudes
    is_y = np.array([r["basis"] == "Y" for r in plan.rows])
    signs = np.array([r["outcome_sign"] for r in plan.rows])
    value = np.where(is_y, signs*np.sin(phase), np.cos(phase))
    return np.clip((1+p.experiment.readout_bias+p.experiment.contrast(p.duration_s)*value)/2, 0, 1)


def simulate(plan, amplitudes, *, seed=42, start_times_s=None):
    """Raw physical +1 counts, with dephasing, visibility, offset, and pulse parity."""
    validate_plan(plan)
    p = plan.pool
    a = _amplitudes(p, amplitudes)
    rng, counts = np.random.default_rng(seed), {}
    if plan.field_mode == "free_running":
        times = _times(plan, start_times_s)
        plus = rng.binomial(1, _timed_probabilities(plan, a, times))
        return TimedCounts({r["row_id"]: int(k) for r, k in zip(plan.rows, plus)},
                           {r["row_id"]: float(t) for r, t in zip(plan.rows, times)})
    if start_times_s is not None:
        raise ValueError("Per-shot timestamps require field_mode='free_running'.")
    for row in plan.rows:
        phase = p.Z[row["sequence_index"]]@a
        value = row["outcome_sign"]*np.sin(phase) if row["basis"] == "Y" else np.cos(phase)
        probability = (1+p.experiment.readout_bias+p.experiment.contrast(p.duration_s)*value)/2
        counts[row["row_id"]] = int(rng.binomial(row["shots"], np.clip(probability, 0, 1)))
    return counts


def decode(plan, counts, *, method=None, cosine_weights=None, sine_weights=None):
    """Decode counts, timestamped raw values, or phase-locked averaged outcomes.

    Weights do not alter the saved acquisition or amplitude reconstruction.
    Without weights, new query-independent plans return values=().
    Untimestamped mean arrays require phase_locked acquisition; freely running
    values use TimedOutcomes. For mean mappings use decode_averages, so a +1
    count is never confused with a mean of +1.
    """
    if isinstance(counts, TimedOutcomes):
        validate_plan(plan)
        if plan.field_mode != "free_running":
            raise ValueError("TimedOutcomes require field_mode='free_running'.")
        if counts.start_times_s is None:
            raise ValueError("Supply actual start timestamps for every freely running outcome.")
        values = np.asarray(_row_values(plan, counts.values, "value"), float)
        if not np.isfinite(values).all() or np.any(abs(values) > 1):
            raise ValueError("Timed raw values must be finite numbers in [-1, 1].")
        positive = {r["row_id"]: float((1+v)/2) for r, v in zip(plan.rows, values)}
        return _timed_estimate(plan, positive, _times(plan, counts.start_times_s), method,
                               cosine_weights, sine_weights)
    if isinstance(counts, AveragedOutcomes) or not isinstance(counts, (Mapping, TimedCounts)):
        averages = counts if isinstance(counts, AveragedOutcomes) else AveragedOutcomes(counts)
        averages = validate_averages(plan, averages)
        return _decode_means(plan, averages.means, averages.repetitions, method,
                             cosine_weights, sine_weights)
    validate_counts(plan, counts)
    if plan.field_mode == "free_running":
        return _timed_estimate(plan, counts.positive_counts, _times(plan, counts.start_times_s),
                               method, cosine_weights, sine_weights)
    means = {r["row_id"]: 2*counts[r["row_id"]]/r["shots"]-1 for r in plan.rows}
    repetitions = {r["row_id"]: r["shots"] for r in plan.rows}
    return _decode_means(plan, means, repetitions, method, cosine_weights, sine_weights)


def decode_averages(plan, means, *, repetitions=None, method=None,
                    cosine_weights=None, sine_weights=None):
    """Decode raw means in plan.rows order or a row-ID mapping.

    Rounded averages are accepted. Actual repetitions weight optional RQS-ML;
    omitted repetitions use the plan's counts. No confidence interval is claimed.
    """
    return decode(plan, AveragedOutcomes(means, repetitions), method=method,
                  cosine_weights=cosine_weights, sine_weights=sine_weights)


def _decode_means(plan, means, repetitions, method, cosine_weights, sine_weights):
    method = "RQS" if method is None else method
    if method not in ("RQS", "RQS-ML"):
        raise ValueError("method must be 'RQS' or 'RQS-ML'.")
    p, v, mu, ny, nx = plan.pool, np.zeros(2*plan.pool.N), 0., 0, 0
    for row in plan.rows:
        weighted = row["shots"]*(means[row["row_id"]]-p.experiment.readout_bias)
        if row["basis"] == "Y":
            v += row["outcome_sign"]*weighted*p.Z[row["sequence_index"]]; ny += row["shots"]
        else:
            mu += weighted; nx += row["shots"]
    mu /= nx
    a, floor = moment_estimate(p, v/ny, mu)
    applied, contrast = False, None
    if method == "RQS-ML":
        rows = [dict(r, shots=repetitions[r["row_id"]]) for r in plan.rows]
        # These are exact sufficient statistics for unrounded binary averages,
        # and a weighted Bernoulli objective for rounded/analog averages.
        positive = {r["row_id"]: r["shots"]*(1+means[r["row_id"]])/2 for r in rows}
        a, applied, contrast = polish(p, a, rows, positive)
    values = _values(p, a, cosine_weights, sine_weights)
    return Estimate(tuple(a.tolist()), values, method, float(mu), float(floor),
                    bool(mu < floor), applied, contrast)
