"""Actual-start-time validation and bounded binary likelihood reconstruction."""

import numpy as np
from scipy.optimize import minimize


class RefinementUnavailable(ValueError):
    """Insufficient shots or Y rank for a full unregularized likelihood fit."""


def validate_times(times, count, duration_s):
    times = np.asarray(times, float)
    if times.shape != (count,) or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError("Supply one finite nonnegative start_time_s per shot, from a common clock origin.")
    tolerance = 16*np.finfo(float).eps*max(1., float(times.max(initial=0.)))
    if np.any(np.diff(times) <= 0) or np.any(np.diff(times) < duration_s-tolerance):
        raise ValueError("Shot start times must increase without overlapping sensing windows.")
    return times


def objective(z, is_y, signs, shots, plus, bias):
    """Mean negative binomial log-likelihood and its analytical gradient."""
    minus, total = shots-plus, shots.sum()

    def evaluate(parameters):
        a, contrast = parameters[:-1], parameters[-1]
        phase = z@a
        f = np.where(is_y, signs*np.sin(phase), np.cos(phase))
        df = np.where(is_y, signs*np.cos(phase), -np.sin(phase))
        mean = bias+contrast*f
        value = -(plus@np.log1p(mean)+minus@np.log1p(-mean))/total
        derivative = (minus/(1-mean)-plus/(1+mean))/total
        gradient = np.r_[(contrast*df*derivative)@z, f@derivative]
        return float(value), gradient

    return evaluate


def fit(pool, rows, positive_counts, start_times_s, *, moment_start=None):
    """Local constrained ML from measured outcomes and actual per-shot responses.

    Three legacy starts and an optional moment start use only measured data.
    No Gaussian moment identity or per-tone law correction is assumed here.
    """
    indices = np.array([r["sequence_index"] for r in rows])
    is_y = np.array([r["basis"] == "Y" for r in rows])
    signs = np.array([r["outcome_sign"] for r in rows])
    shots = np.array([r["shots"] for r in rows], float)
    plus = np.array([positive_counts[r["row_id"]] for r in rows], float)
    d, e = 2*pool.N, pool.experiment
    if len(rows) < 3*d or is_y.sum() < d:
        raise RefinementUnavailable("Timestamped ML requires at least 3*2N shots and 2N Y shots.")
    z = pool.responses_at(indices, start_times_s)
    singular = np.linalg.svd(z[is_y], compute_uv=False)
    if singular[-1] <= 1e-10*singular[0]:
        raise RefinementUnavailable("Timestamped Y responses do not span all amplitudes; change the schedule or add shots.")
    condition = float(singular[0]/singular[-1])
    contrast = e.contrast(pool.duration_s)
    if contrast <= 1e-12:
        raise ValueError("Calibrated contrast is too small for timestamped likelihood initialization.")
    measured = 2*plus/shots-1-e.readout_bias
    linear = np.linalg.lstsq(z[is_y], signs[is_y]*measured[is_y]/contrast, rcond=None)[0]
    mu = float(np.average(measured[~is_y], weights=shots[~is_y]))
    floor = contrast*np.exp(-.5*singular[0]**2/is_y.sum()*e.amplitude_bound**2)
    ratio = linear*contrast/max(mu, floor, 1e-12)
    upper = 1-abs(e.readout_bias)-1e-10
    lower = min(1e-8, upper/100)
    evaluate = objective(z, is_y, signs, shots, plus, e.readout_bias)
    radius = e.amplitude_bound
    constraint = dict(type="ineq", fun=lambda x: 1-np.dot(x[:-1], x[:-1])/radius**2,
                      jac=lambda x: np.r_[-2*x[:-1]/radius**2, 0.])
    results = []
    messages = []
    starts = (linear, ratio, np.zeros(d))
    if moment_start is not None:
        starts += (np.asarray(moment_start, float),)
    for a0 in starts:
        a0 = a0*min(1., .9*radius/max(np.linalg.norm(a0), 1e-300))
        initial = np.r_[a0, np.clip(contrast, lower, upper)]
        result = minimize(evaluate, initial, jac=True, method="SLSQP",
                          bounds=[(-radius, radius)]*d+[(lower, upper)],
                          constraints=[constraint], options=dict(maxiter=250, ftol=1e-11))
        messages.append(str(result.message))
        if result.success and np.isfinite(result.fun) and constraint["fun"](result.x) >= -1e-7:
            results.append(result)
    if not results:
        raise RuntimeError("Timestamped likelihood fit did not converge: "+"; ".join(messages))
    best = min(results, key=lambda r: r.fun)
    return best.x[:-1], float(best.x[-1]), mu, condition
