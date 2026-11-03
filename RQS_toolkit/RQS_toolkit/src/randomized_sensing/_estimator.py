"""RQS moment inversion, per-tone radial correction, and likelihood polish."""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import cho_solve
from scipy.special import j0, j1


def moment_estimate(pool, v, mu):
    """Phase-locked back-projection, with the shared guard and radial correction."""
    return finish_moment(pool, cho_solve(pool.chol, v), mu)


def finish_moment(pool, back_projection, mu):
    """Guard a contrast-normalized back-projection and correct each tone's radius."""
    contrast = pool.experiment.contrast(pool.duration_s)
    floor = max(1e-12, contrast*np.exp(-pool.lam_max*pool.experiment.amplitude_bound**2/2))
    a = back_projection/max(mu, floor)
    return correct_tones(pool, a), floor


def correct_tones(pool, a):
    """Shared per-tone Bessel-map inversion, without changing the original map."""
    if not hasattr(pool, "_H"):
        r = np.hypot(pool.Z[:len(pool.sequences), :pool.N], pool.Z[:len(pool.sequences), pool.N:])
        variance = np.mean(r*r, axis=0)/2
        grid = np.linspace(0., max(2.4, 1.2*pool.experiment.amplitude_bound), 256)
        h = np.empty((len(grid), pool.N))
        def block(start):
            rr = r[:, start:start+8]
            arg = grid[:, None, None]*rr[None]
            numerator = np.einsum("pj,xpj->xj", rr, j1(arg))/len(rr)
            denominator = variance[start:start+8]*np.mean(j0(arg), axis=1)
            return start, numerator/np.maximum(denominator, 1e-12)

        starts = range(0, pool.N, 8)
        if getattr(pool, "_map_workers", 1) > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                blocks = list(executor.map(block, starts))
        else:
            blocks = map(block, starts)
        for start, values in blocks:
            h[:, start:start+8] = values
        h[0] = 0.
        pool._grid, pool._H = grid, np.maximum.accumulate(h, axis=0)
    magnitude = np.hypot(a[:pool.N], a[pool.N:])
    fixed = np.array([np.interp(magnitude[j], pool._H[:, j], pool._grid) for j in range(pool.N)])
    scale = np.divide(fixed, magnitude, out=np.ones_like(fixed), where=magnitude > 1e-15)
    return a*np.r_[scale, scale]


def polish(pool, a, rows, counts, iterations=20):
    """Count-weighted joint (amplitudes, contrast) likelihood; known readout offset."""
    if sum(r["shots"] for r in rows) < 3*len(a):
        return a, False, None
    z = pool.Z[[r["sequence_index"] for r in rows]]
    is_y = np.array([r["basis"] == "Y" for r in rows])
    signs = np.array([r["outcome_sign"] for r in rows])
    n = np.array([r["shots"] for r in rows])
    plus = np.array([counts[r["row_id"]] for r in rows]); minus = n-plus
    bias = pool.experiment.readout_bias
    upper = 1-abs(bias)-1e-10

    def evaluate(amplitudes, contrast):
        phase = z@amplitudes
        f = np.where(is_y, signs*np.sin(phase), np.cos(phase))
        df = np.where(is_y, signs*np.cos(phase), -np.sin(phase))
        mean = bias+contrast*f
        likelihood = float(plus@np.log1p(mean) + minus@np.log1p(-mean))
        return likelihood, f, df, mean

    phase = z@a
    x = ~is_y
    empirical = (2*plus[x].sum()/n[x].sum()-1)-bias
    c = float(np.clip(empirical/max(np.average(np.cos(phase[x]), weights=n[x]), .05),
                      min(.02, upper/2), upper))
    for _ in range(iterations):
        likelihood, f, df, mean = evaluate(a, c)
        derivative = plus/(1+mean)-minus/(1-mean)
        variance = np.maximum(1-mean*mean, 1e-12)
        step_a = cho_solve(pool.chol, (c*df*derivative)@z)/max(np.sum(n*c*c*df*df/variance), 1e-12)
        step_c = np.sum(f*derivative)/max(np.sum(n*f*f/variance), 1e-12)
        for power in range(12):
            t = 2.**(-power)
            candidate, candidate_c = a+t*step_a, float(np.clip(c+t*step_c, 1e-8*upper, upper))
            if evaluate(candidate, candidate_c)[0] >= likelihood:
                break
        else:
            break
        change = np.max(abs(candidate-a))
        a, c = candidate, candidate_c
        if change < 1e-5:
            break
    return a, True, c
