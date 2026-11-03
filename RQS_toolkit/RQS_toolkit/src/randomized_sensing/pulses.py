"""Random-phase multisine controls; literal Fourier integration after timing repair."""

from dataclasses import dataclass
from numbers import Integral
import warnings

import numpy as np
from scipy.linalg import cho_factor

from ._response import response
from .experiment import Experiment
from ._grid import CellGrid, fft_grid


def integer(value, name, minimum=0):
    if type(value) is int and value >= minimum:
        return value
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


@dataclass(frozen=True)
class Sequence:
    switch_times_s: tuple[float, ...]
    initial_sign: int = 1


class Pool:
    def __init__(self, experiment, duration_s, start_time_s, design_clock_s, sequences,
                 responses, settings, *, allow_ill_conditioned=True):
        self.experiment = experiment
        self.duration_s, self.start_time_s = float(duration_s), float(start_time_s)
        self.design_clock_s = float(design_clock_s)
        self.sequences, self.settings = tuple(sequences), settings
        self.Z = np.asarray(responses, float)
        self.N = len(experiment.frequencies_hz)
        if self.Z.shape != (2*len(sequences), 2*self.N) or not np.isfinite(self.Z).all():
            raise ValueError("Invalid pool response dimensions.")
        self.Sigma = self.Z.T @ self.Z / len(self.Z)
        eigenvalues = np.linalg.eigvalsh(self.Sigma)
        if eigenvalues[0] <= 1e-10*eigenvalues[-1]:
            raise ValueError("Pool covariance is singular or ill-conditioned; increase duration/size or relax timing.")
        self.condition = float(eigenvalues[-1]/eigenvalues[0])
        if self.condition > 1e3 and not allow_ill_conditioned:
            raise ValueError(f"Pool covariance condition {self.condition:.3g} exceeds 1000; increase duration, separate tones, or relax timing. Set allow_ill_conditioned=True only to inspect this unstable design.")
        if self.condition > 20:
            warnings.warn(f"Pool covariance condition is {self.condition:.3g} (>20); weak response directions amplify noise. Increase duration, separate tones, or relax timing.", RuntimeWarning, stacklevel=2)
        layout = fft_grid(np.asarray(experiment.frequencies_hz), self.duration_s,
                          int(round(self.duration_s/self.design_clock_s)))
        self.synthesis_method = "fft" if layout is not None else "dense"
        self.chol = cho_factor(self.Sigma)
        self.lam_max = float(eigenvalues[-1])
        signs = np.array([q.initial_sign*(-1)**len(q.switch_times_s) for q in self.sequences], dtype=np.int8)
        self.y_outcome_signs = np.r_[signs, -signs]

    def sequence(self, index):
        index = integer(index, "sequence_index")
        if index >= len(self.Z):
            raise ValueError("sequence_index is outside the pool.")
        half = len(self.sequences)
        q = self.sequences[index % half]
        return Sequence(q.switch_times_s, q.initial_sign * (1 if index < half else -1))

    def responses_at(self, sequence_indices, start_times_s):
        """Rotate stored responses to actual starts, in seconds from one common origin.

        Times are elapsed clock times, not Unix timestamps. The same relative
        pulse waveform is used; no source reset or pulse re-compilation is needed.
        Scalar inputs return one vector; matching arrays return a response matrix.
        """
        indices = np.asarray(sequence_indices)
        times = np.asarray(start_times_s, float)
        if (not np.issubdtype(indices.dtype, np.integer) or indices.shape != times.shape
                or np.any(indices < 0) or np.any(indices >= len(self.Z))
                or not np.isfinite(times).all() or np.any(times < 0)):
            raise ValueError("Supply matching valid sequence indices and nonnegative finite start times.")
        return self.rotate_vectors(self.Z[indices], times)

    def rotate_vectors(self, vectors, start_times_s):
        """Rotate arbitrary (..., 2N) bank-frame vectors to matching actual starts."""
        base = np.asarray(vectors, float)
        times = np.asarray(start_times_s, float)
        if (base.shape != times.shape+(2*self.N,) or not np.isfinite(base).all()
                or not np.isfinite(times).all() or np.any(times < 0)):
            raise ValueError("Supply finite (..., 2N) vectors and matching nonnegative start times.")
        delta = times.astype(np.longdouble)-np.longdouble(self.start_time_s)
        cycles = np.remainder(delta[..., None]*np.asarray(self.experiment.frequencies_hz, np.longdouble), 1)
        angle = 2*np.arccos(np.longdouble(-1))*cycles
        c, s = np.asarray(np.cos(angle), float), np.asarray(np.sin(angle), float)
        x, y = base[..., :self.N], base[..., self.N:]
        return np.concatenate((x*c-y*s, x*s+y*c), axis=-1)


def audit(pool, index):
    """Check actual pulses, edge guards, design clock, and stored signed response."""
    e, q = pool.experiment, pool.sequence(index)
    edges = np.r_[0., q.switch_times_s, pool.duration_s]
    gaps = np.diff(edges)
    ticks = edges/pool.design_clock_s
    tol = 1e-8*pool.design_clock_s
    valid = bool(np.all(gaps >= -tol) and np.allclose(ticks, np.rint(ticks), rtol=0, atol=1e-7))
    if len(q.switch_times_s):
        valid &= bool(min(gaps[0], gaps[-1])+tol >= e.edge_guard_s
                      and np.all(gaps[1:-1]+tol >= e.min_gap_s))
    cap = pool.settings.get("max_pulses")
    valid &= cap is None or len(q.switch_times_s) <= cap
    v = q.initial_sign * response(e, q.switch_times_s, pool.duration_s, pool.start_time_s)
    error = float(np.max(abs(v-pool.Z[index])))
    return dict(timing_valid=valid, minimum_interval_s=float(gaps.min()),
                pulse_count=len(q.switch_times_s), response_error=error,
                y_outcome_sign=q.initial_sign * (-1)**len(q.switch_times_s))


def _repair(s, cells):
    # Flip non-adjacent short runs, merging them with their neighbours.
    for _ in range(256):
        edges = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1, len(s)]
        short = np.flatnonzero(np.diff(edges) < cells)
        if not len(short) or len(edges) == 2:
            return s
        pick = short[np.r_[True, np.diff(short) > 1]]
        flip = np.ones(len(edges)-1, np.int8); flip[pick] = -1
        s *= np.repeat(flip, np.diff(edges))
    raise RuntimeError("Minimum-dwell repair did not converge.")


def build_pool(experiment: Experiment, *, duration_s, pool_size=None, max_pulses=None,
               start_time_s=0., design_clock_s=2.5e-9, seed=4, allow_ill_conditioned=False):
    """Build independent repaired random multisines plus their exact negatives.

    A hardware clock takes precedence over design_clock_s. Without a hardware
    clock, design_clock_s is an explicit numerical synthesis grid. Every interval
    conservatively obeys max(min_gap_s, edge_guard_s), rounded UP to that grid.
    Arbitrary positive frequencies are supported; no commensurability is required.
    Commensurate tones use FFTs; other tones use bounded-memory dense cell tables.
    Warn for covariance condition >20; reject >1000 unless explicitly overridden.
    """
    f = np.asarray(experiment.frequencies_hz)
    if np.any(f <= 0):
        raise ValueError("RQS tone correction requires positive frequencies; DC is not supported.")
    tick = experiment.clock_s if experiment.clock_s is not None else design_clock_s
    if not np.isfinite([duration_s, start_time_s, tick]).all() or min(duration_s, tick) <= 0 or start_time_s < 0:
        raise ValueError("Use a positive duration/clock and a nonnegative phase-reference start time.")
    length = int(round(duration_s/tick))
    if length < 1 or abs(length*tick-duration_s) > 1e-7*tick:
        raise ValueError("duration_s must be a whole number of design/hardware clock cells.")
    if f.max()*tick >= .5:
        raise ValueError("Clock too coarse for the highest input frequency.")
    size = integer(max(8192, 64*len(f)) if pool_size is None else pool_size, "pool_size", 2)
    if size % 2 or size//2 < 2*len(f)+1:
        raise ValueError("pool_size must be even, with more than 2N distinct entries.")
    if max_pulses is not None:
        max_pulses = integer(max_pulses, "max_pulses")
    if not isinstance(allow_ill_conditioned, bool):
        raise ValueError("allow_ill_conditioned must be a bool.")
    dwell = max(1, int(np.ceil(max(experiment.min_gap_s, experiment.edge_guard_s)/tick)))
    if length*tick+1e-15 < experiment.edge_guard_s:
        raise ValueError("duration_s is shorter than edge_guard_s.")
    seed = integer(seed, "seed")
    if len(f) > 1 and np.min(np.diff(np.sort(f)))*duration_s < 1-1e-12:
        warnings.warn("Minimum tone spacing * duration_s is below 1: tones may be unresolved and estimates ill-conditioned. Increase duration or separate the tones; inspect pool.condition.", RuntimeWarning, stacklevel=2)
    rng = np.random.default_rng(seed)
    grid = CellGrid(experiment, duration_s, start_time_s, tick, length)
    sequences, vectors, attempts = [], [], 0
    while len(sequences) < size//2 and attempts < 20*(size//2):
        theta = rng.uniform(0, 2*np.pi, (min(grid.batch_size, size//2-len(sequences)), len(f)))
        patterns = np.where(grid.samples(theta) >= 0, 1, -1).astype(np.int8)
        accepted = []
        for s in patterns:
            attempts += 1
            s = _repair(s, dwell)
            pulses = np.flatnonzero(s[1:] != s[:-1])+1
            if max_pulses is not None and len(pulses) > max_pulses:
                continue
            q = Sequence(tuple((pulses*tick).tolist()), int(s[0]))
            sequences.append(q)
            accepted.append(s)
        if accepted:
            vectors.extend(grid.responses(np.asarray(accepted)))
    if len(sequences) != size//2:
        raise ValueError("Too few sequences meet max_pulses; increase the cap or shorten duration_s.")
    z = np.asarray(vectors)
    settings = dict(seed=int(seed), pool_size=size, max_pulses=max_pulses,
                    attempts=attempts, minimum_dwell_cells=dwell)
    return Pool(experiment, duration_s, start_time_s, tick, sequences, np.r_[z, -z], settings,
                allow_ill_conditioned=allow_ill_conditioned)
