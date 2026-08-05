# -*- coding: utf-8 -*-
"""
GPU-assisted bimodal / multi-NV charge-state histogram fitting.

This module is intentionally separate from analysis/bimodal_histogram.py so the
existing CPU/SciPy fitter can stay unchanged.

Supported distributions:
    - POISSON
    - COMPOUND_POISSON

Main bimodal path:
    - Build integer-count histograms on CPU.
    - Evaluate many candidate bimodal models on GPU with CuPy.
    - Return the best coarse fit for each histogram.

Main multi-NV path:
    - Fit a physical equal-brightness binomial multi-NV model on GPU.
    - For each forced N=1..max_nvs:
        * coarse grid search on GPU
        * local multi-start / coordinate-style refinement on GPU
        * NLL and BIC evaluation on integer histograms
    - Select N using either ordinary BIC or strict physical rules.
    - Optionally fit a 2-NV unequal-brightness diagnostic on GPU.

Physical equal-brightness model:
    k = number of NV- centers in one pillar, k=0..N
    P(k) = Binomial(N, p_minus)
    lambda_k = bg + N*rate0 + k*delta

2-NV unequal diagnostic model:
    parameters = [p1, p2, bg, rate0, delta1, delta2]
    lambda_00 = bg + 2*rate0
    lambda_10 = bg + 2*rate0 + delta1
    lambda_01 = bg + 2*rate0 + delta2
    lambda_11 = bg + 2*rate0 + delta1 + delta2

Created: July 2026
@author: Saroj Chand / updated with GPU-refined multi-NV protocol
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import warnings

import numpy as np

try:
    from scipy.special import gammaln as sp_gammaln
except Exception:
    sp_gammaln = None


try:
    import cupy as cp
    from cupyx.scipy.special import gammaln as cp_gammaln
    from cupyx.scipy.special import erf as cp_erf

    GPU_AVAILABLE = True
except Exception:
    cp = None
    cp_gammaln = None
    cp_erf = None
    GPU_AVAILABLE = False

SUPPORTED_PROB_DIST_NAMES = {
    "POISSON",
    "COMPOUND_POISSON",
    "BROADENED_COMPOUND_POISSON",
}

@dataclass(frozen=True)
class GpuFitConfig:
    """Configuration for the GPU bimodal coarse search."""

    num_ratio: int = 17
    num_rate: int = 56
    ratio_min: float = 0.05
    ratio_max: float = 0.95
    fit_chunk_size: int = 512
    candidate_chunk_size: int = 2048
    min_samples: int = 50
    max_count_padding: int = 0
    compound_nsig: float = 5.0
    compound_min_lim: int = 10
    compound_max_lim: int = 50_000
    eps: float = 1e-12


@dataclass(frozen=True)
class GpuMultimodeFitConfig:
    """Configuration for GPU multi-NV model search and refinement."""

    # Model family
    max_nvs: int = 3

    # Coarse grid search
    num_p: int = 13
    num_bg: int = 5
    num_rate0: int = 18
    num_delta: int = 18
    p_min: float = 0.02
    p_max: float = 0.98

    # GPU chunking
    fit_chunk_size: int = 512
    candidate_chunk_size: int = 512
    refine_fit_chunk_size: int = 64

    # Data cleaning / distributions
    min_samples: int = 50
    max_count_padding: int = 0
    compound_nsig: float = 5.0
    compound_min_lim: int = 10
    compound_max_lim: int = 50_000
    eps: float = 1e-12

    # Ordinary BIC model penalty
    bic_extra_nv_penalty: float = 8.0

    # GPU local refinement around the coarse grid optimum
    use_refinement: bool = True
    refine_iters: int = 8
    refine_shrink: float = 0.55
    refine_p_step: float = 0.08
    refine_bg_frac_step: float = 0.20
    refine_rate0_frac_step: float = 0.20
    refine_delta_frac_step: float = 0.20
    refine_min_abs_step: float = 0.25

    # Strict physical model selection
    strict_extra_nv_penalty: float = 80.0
    strict_bic_margin: float = 25.0
    strict_min_mode_weight: float = 0.05
    strict_min_mode_shots: int = 75
    strict_min_adjacent_dprime: float = 1.50
    strict_require_all_modes: bool = True

    # Optional GPU diagnostic that mirrors the CPU 2nv_unequal check.
    # For speed, keep False during broad screening, then True for final checks.
    include_2nv_unequal: bool = False
    unequal_refine_iters: int = 8
    unequal_refine_shrink: float = 0.55
    unequal_bic_extra_penalty: float = 0.0


    broadened_sigma0: float = 3.0
    broadened_fano: float = 0.0
    broadened_sigma_frac: float = 0.0
    broadened_min_sigma: float = 0.5
    broadened_max_sigma: float = 50.0

    # Hierarchical model search
    hierarchical_n1_red_chi_sq_stop: float = 1.35
    hierarchical_n2_red_chi_sq_stop: float = 1.50
    hierarchical_min_bic_improvement: float = 25.0
    hierarchical_min_red_chi_sq_improvement: float = 0.08


# =============================================================================
# Basic helpers
# =============================================================================


def _prob_dist_name(prob_dist) -> str:
    """Accept an Enum member or a plain string."""
    if hasattr(prob_dist, "name"):
        return str(prob_dist.name).upper()
    return str(prob_dist).upper()


def _clean_counts(counts, min_samples: int) -> np.ndarray | None:
    counts = np.asarray(counts, dtype=float).ravel()
    counts = counts[np.isfinite(counts)]
    counts = counts[counts >= 0]

    if counts.size < min_samples:
        return None

    median = np.median(counts)
    std = np.std(counts)
    if np.isfinite(std) and std > 0:
        counts = counts[counts < median + 10 * std]

    if counts.size < min_samples:
        return None

    return counts


def _make_integer_histogram(counts: np.ndarray, x_max: int):
    xi = np.rint(counts).astype(int)
    xi = xi[(xi >= 0) & (xi <= x_max)]

    bin_counts = np.bincount(xi, minlength=x_max + 1).astype(float)
    num_samples = float(np.sum(bin_counts))
    if num_samples <= 0:
        raise ValueError("empty histogram")

    hist = bin_counts / num_samples
    hist_errs = np.sqrt(np.maximum(bin_counts, 1.0)) / num_samples

    local_max = int(np.max(xi)) if xi.size else 0
    valid = np.zeros(x_max + 1, dtype=bool)
    valid[: local_max + 1] = True

    return hist, hist_errs, valid, bin_counts


def _initial_stats(cleaned_counts: Sequence[np.ndarray]):
    q02 = []
    q15 = []
    q50 = []
    q65 = []
    q85 = []
    q98 = []
    mean = []
    for counts in cleaned_counts:
        q02.append(float(np.quantile(counts, 0.02)))
        q15.append(float(np.quantile(counts, 0.15)))
        q50.append(float(np.quantile(counts, 0.50)))
        q65.append(float(np.quantile(counts, 0.65)))
        q85.append(float(np.quantile(counts, 0.85)))
        q98.append(float(np.quantile(counts, 0.98)))
        mean.append(float(np.mean(counts)))
    return {
        "q02": np.asarray(q02, dtype=float),
        "q15": np.asarray(q15, dtype=float),
        "q50": np.asarray(q50, dtype=float),
        "q65": np.asarray(q65, dtype=float),
        "q85": np.asarray(q85, dtype=float),
        "q98": np.asarray(q98, dtype=float),
        "mean": np.asarray(mean, dtype=float),
    }


def _as_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _as_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# =============================================================================
# Probability tables on GPU
# =============================================================================

# =============================================================================
# Probability tables on GPU
# =============================================================================


def _make_rate_grid(stats, config: GpuFitConfig) -> np.ndarray:
    lo = float(np.nanmin(stats["q02"]))
    hi = float(np.nanmax(stats["q98"]))

    lo = max(lo, config.eps)
    hi = max(hi, lo * 1.05, 1.0)

    return np.linspace(lo, hi, int(config.num_rate), dtype=np.float64)


def _poisson_pdf_table(x_vals_gpu, rates_gpu, eps: float):
    """Return pdf[r, x] = Pois(x | rate_r)."""
    rates_gpu = cp.maximum(rates_gpu.astype(cp.float64), eps)
    x_vals_gpu = x_vals_gpu.astype(cp.float64)

    log_rate = cp.log(rates_gpu)[:, None]
    x = x_vals_gpu[None, :]

    log_pdf = (
        cp.where(x == 0, 0.0, x * log_rate)
        - rates_gpu[:, None]
        - cp_gammaln(x + 1.0)
    )
    return cp.exp(log_pdf)


def _normal_cdf_gpu(x):
    """Standard normal CDF on GPU."""
    return 0.5 * (1.0 + cp_erf(x / cp.sqrt(2.0)))


def _safe_upper_lim_gpu(rate_grid: np.ndarray, config) -> int:
    rmax = float(np.nanmax(rate_grid))
    if not np.isfinite(rmax) or rmax < 0:
        raise ValueError(f"invalid rate grid max: {rmax}")

    upper = rmax + float(config.compound_nsig) * math.sqrt(max(rmax, 0.0))
    return int(
        min(
            max(int(math.ceil(upper)), int(config.compound_min_lim)),
            int(config.compound_max_lim),
        )
    )


def _broadened_sigma_from_rate_gpu(rates_gpu, config):
    """
    Technical broadening model for BROADENED_COMPOUND_POISSON.

    This keeps the existing fitter parameterization unchanged:
        popt = [p_minus, bg, rate0, delta]

    The extra broadening is not fitted per pillar yet. It is controlled globally
    by optional config attributes:

        broadened_sigma0       fixed additive broadening in counts
        broadened_fano         variance term proportional to rate
        broadened_sigma_frac   fractional broadening term proportional to rate
        broadened_min_sigma    lower bound
        broadened_max_sigma    upper bound

    Effective variance:
        sigma_eff^2 = sigma0^2 + fano * rate + (sigma_frac * rate)^2
    """
    sigma0 = float(getattr(config, "broadened_sigma0", 3.0))
    fano = float(getattr(config, "broadened_fano", 0.0))
    sigma_frac = float(getattr(config, "broadened_sigma_frac", 0.0))
    min_sigma = float(getattr(config, "broadened_min_sigma", 0.5))
    max_sigma = float(getattr(config, "broadened_max_sigma", 50.0))

    rates_gpu = cp.maximum(cp.asarray(rates_gpu, dtype=cp.float64), 0.0)

    var = (
        sigma0**2
        + fano * rates_gpu
        + (sigma_frac * rates_gpu) ** 2
    )
    sigma = cp.sqrt(cp.maximum(var, min_sigma**2))
    sigma = cp.minimum(sigma, max_sigma)

    return sigma


def _compound_poisson_latent_table(x_vals_gpu, rates_gpu, rate_grid_cpu, config):
    """
    Return p_m_given_rate[r, m] and m_vals for latent ideal count m.

    This is shared by ordinary compound-Poisson and broadened compound-Poisson.
    """
    upper_lim = _safe_upper_lim_gpu(rate_grid_cpu, config)

    # Include upper_lim itself.
    m_vals_gpu = cp.arange(0, upper_lim + 1, dtype=cp.float64)

    p_m_given_rate = _poisson_pdf_table(
        m_vals_gpu,
        rates_gpu,
        config.eps,
    )  # (R, M)

    return p_m_given_rate, m_vals_gpu


def _compound_poisson_pdf_table(x_vals_gpu, rates_gpu, rate_grid_cpu, config):
    """
    Return pdf[r, z] = sum_m Pois(z | m) Pois(m | rate_r).

    Here:
        rate_r is the mean photon-generation rate.
        m is the latent Poisson-distributed photon number.
        z is the observed integer count.
    """
    p_m_given_rate, m_vals_gpu = _compound_poisson_latent_table(
        x_vals_gpu,
        rates_gpu,
        rate_grid_cpu,
        config,
    )

    # p_z_given_m[m, z]
    m_safe = cp.maximum(m_vals_gpu, config.eps)
    z = x_vals_gpu.astype(cp.float64)[None, :]

    log_m = cp.log(m_safe)[:, None]
    log_p_z_given_m = (
        cp.where(z == 0, 0.0, z * log_m)
        - m_vals_gpu[:, None]
        - cp_gammaln(z + 1.0)
    )

    # Correct m=0 exactly: Pois(z | 0) is 1 for z=0 and 0 otherwise.
    if int(m_vals_gpu.size) > 0:
        log_p_z_given_m = log_p_z_given_m.copy()
        log_p_z_given_m[0, :] = -cp.inf
        log_p_z_given_m[0, 0] = 0.0

    p_z_given_m = cp.exp(log_p_z_given_m)  # (M, X)

    pdf = p_m_given_rate @ p_z_given_m  # (R, X)
    pdf = cp.maximum(pdf, config.eps)
    pdf = pdf / cp.maximum(cp.sum(pdf, axis=1, keepdims=True), config.eps)

    return pdf


def _broadened_compound_poisson_pdf_table(
    x_vals_gpu,
    rates_gpu,
    rate_grid_cpu,
    config,
):
    """
    BROADENED_COMPOUND_POISSON probability table.

    Physical model:
        m ~ compound-Poisson latent ideal count
        observed count z = m + Gaussian technical noise
        then binned into integer count bins

    P(z | rate, sigma)
        = sum_m P(m | compound-Poisson rate)
              * Integral[z-0.5, z+0.5] Normal(y | m, sigma) dy

    This is useful when pure COMPOUND_POISSON peaks are too narrow and the
    fitter incorrectly invents fake middle NV modes.
    """
    rates_gpu = cp.maximum(cp.asarray(rates_gpu, dtype=cp.float64), config.eps)

    # Increase latent sum range a bit because Gaussian broadening can move
    # probability into larger observed-count bins.
    sigma_gpu = _broadened_sigma_from_rate_gpu(rates_gpu, config)
    sigma_cpu = cp.asnumpy(sigma_gpu)

    broadened_rate_grid_cpu = np.asarray(rate_grid_cpu, dtype=float)
    if broadened_rate_grid_cpu.size > 0:
        broadened_rate_grid_cpu = broadened_rate_grid_cpu + 6.0 * float(
            np.nanmax(sigma_cpu)
        )

    p_m_given_rate, m_vals_gpu = _compound_poisson_latent_table(
        x_vals_gpu,
        rates_gpu,
        broadened_rate_grid_cpu,
        config,
    )

    # Gaussian bin probability P(z bin | m, sigma)
    #
    # Shapes:
    #   m_vals_gpu: (M,)
    #   x_vals_gpu: (X,)
    #   sigma_gpu:  (R,)
    #
    # We compute in chunks over rates to avoid huge R x M x X allocations.
    R = int(rates_gpu.size)
    X = int(x_vals_gpu.size)
    M = int(m_vals_gpu.size)

    out = cp.empty((R, X), dtype=cp.float64)

    chunk_size = int(getattr(config, "candidate_chunk_size", 512))
    chunk_size = max(chunk_size, 1)

    x_lo = x_vals_gpu.astype(cp.float64)[None, :] - 0.5  # (1, X)
    x_hi = x_vals_gpu.astype(cp.float64)[None, :] + 0.5  # (1, X)
    m = m_vals_gpu.astype(cp.float64)[:, None]           # (M, 1)

    for start in range(0, R, chunk_size):
        stop = min(start + chunk_size, R)

        p_m = p_m_given_rate[start:stop, :]              # (Rc, M)
        sig = sigma_gpu[start:stop]                      # (Rc,)

        mix = cp.zeros((stop - start, X), dtype=cp.float64)

        # Loop over rate candidates in this chunk. This avoids making
        # a huge 3D array. It is slower than pure compound-Poisson, but much
        # safer for memory.
        for local_ind in range(stop - start):
            s = cp.maximum(sig[local_ind], 1e-9)

            cdf_hi = _normal_cdf_gpu((x_hi - m) / s)     # (M, X)
            cdf_lo = _normal_cdf_gpu((x_lo - m) / s)     # (M, X)
            p_z_given_m = cp.clip(cdf_hi - cdf_lo, 0.0, 1.0)

            mix[local_ind, :] = p_m[local_ind:local_ind + 1, :] @ p_z_given_m

        mix = cp.maximum(mix, config.eps)
        mix = mix / cp.maximum(cp.sum(mix, axis=1, keepdims=True), config.eps)
        out[start:stop, :] = mix

    return out


def _single_mode_pdf_table(prob_name: str, x_vals_gpu, rates_gpu, rate_grid_cpu, config):
    prob_name = str(prob_name).upper()

    if prob_name == "POISSON":
        return _poisson_pdf_table(x_vals_gpu, rates_gpu, config.eps)

    if prob_name == "COMPOUND_POISSON":
        return _compound_poisson_pdf_table(
            x_vals_gpu,
            rates_gpu,
            rate_grid_cpu,
            config,
        )

    if prob_name == "BROADENED_COMPOUND_POISSON":
        return _broadened_compound_poisson_pdf_table(
            x_vals_gpu,
            rates_gpu,
            rate_grid_cpu,
            config,
        )

    raise ValueError(f"Unsupported prob_dist for GPU fitting: {prob_name}")


# =============================================================================
# Old bimodal GPU coarse path
# =============================================================================


def _candidate_arrays(num_rate: int, config: GpuFitConfig):
    ratio_grid = np.linspace(
        float(config.ratio_min),
        float(config.ratio_max),
        int(config.num_ratio),
        dtype=np.float64,
    )

    dark_inds = []
    bright_inds = []
    ratios = []
    for dark_ind in range(num_rate):
        for bright_ind in range(dark_ind, num_rate):
            for ratio in ratio_grid:
                dark_inds.append(dark_ind)
                bright_inds.append(bright_ind)
                ratios.append(float(ratio))

    return (
        np.asarray(dark_inds, dtype=np.int32),
        np.asarray(bright_inds, dtype=np.int32),
        np.asarray(ratios, dtype=np.float64),
    )


def fit_bimodal_histograms_gpu_batch(
    counts_batch: Iterable[Sequence[float]],
    prob_dist="COMPOUND_POISSON",
    config: GpuFitConfig | None = None,
    return_debug: bool = False,
):
    """
    Fit many bimodal histograms using one batched GPU coarse search.

    Returns a list of (popt, pcov, red_chi_sq), matching the CPU fitter shape.
    popt = [dark_mode_weight, dark_rate, bright_rate]
    """
    if config is None:
        config = GpuFitConfig()

    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy GPU backend is not available.")

    prob_name = _prob_dist_name(prob_dist)
    if prob_name not in SUPPORTED_PROB_DIST_NAMES:
        raise ValueError(
            f"GPU fitting currently supports {sorted(SUPPORTED_PROB_DIST_NAMES)}, got {prob_name}"
        )

    raw_items = list(counts_batch)
    if len(raw_items) == 0:
        return ([], {}) if return_debug else []

    cleaned = []
    valid_input_inds = []
    for ind, counts in enumerate(raw_items):
        clean = _clean_counts(counts, config.min_samples)
        if clean is None:
            continue
        cleaned.append(clean)
        valid_input_inds.append(ind)

    results = [(None, None, None) for _ in raw_items]
    if not cleaned:
        return (results, {"reason": "no_valid_histograms"}) if return_debug else results

    stats = _initial_stats(cleaned)
    x_max = int(max(np.max(c) for c in cleaned)) + int(config.max_count_padding)
    x_max = max(x_max, 1)
    x_vals = np.arange(x_max + 1, dtype=np.float64)

    hists = np.zeros((len(cleaned), x_max + 1), dtype=np.float64)
    hist_errs = np.ones_like(hists, dtype=np.float64)
    valid = np.zeros_like(hists, dtype=bool)

    for row, counts in enumerate(cleaned):
        hist, err, mask, _bin_counts = _make_integer_histogram(counts, x_max)
        hists[row, :] = hist
        hist_errs[row, :] = err
        valid[row, :] = mask

    rate_grid = _make_rate_grid(stats, config)
    dark_inds, bright_inds, ratios = _candidate_arrays(rate_grid.size, config)

    x_gpu = cp.asarray(x_vals, dtype=cp.float64)
    rates_gpu = cp.asarray(rate_grid, dtype=cp.float64)
    pdf_by_rate = _single_mode_pdf_table(prob_name, x_gpu, rates_gpu, rate_grid, config)

    best_loss_cpu = np.full(len(cleaned), np.inf, dtype=np.float64)
    best_candidate_cpu = np.full(len(cleaned), -1, dtype=np.int64)

    num_candidates = ratios.size
    candidate_chunk = int(config.candidate_chunk_size)
    fit_chunk = int(config.fit_chunk_size)

    for row_start in range(0, len(cleaned), fit_chunk):
        row_stop = min(row_start + fit_chunk, len(cleaned))

        h_gpu = cp.asarray(hists[row_start:row_stop], dtype=cp.float64)
        err_gpu = cp.asarray(hist_errs[row_start:row_stop], dtype=cp.float64)
        valid_gpu = cp.asarray(valid[row_start:row_stop], dtype=cp.float64)

        w_gpu = valid_gpu / cp.maximum(err_gpu, config.eps) ** 2
        a_gpu = cp.sum((h_gpu**2) * w_gpu, axis=1)

        chunk_best_loss = cp.full(row_stop - row_start, cp.inf, dtype=cp.float64)
        chunk_best_candidate = cp.full(row_stop - row_start, -1, dtype=cp.int64)

        for cand_start in range(0, num_candidates, candidate_chunk):
            cand_stop = min(cand_start + candidate_chunk, num_candidates)

            d_ind_gpu = cp.asarray(dark_inds[cand_start:cand_stop], dtype=cp.int32)
            b_ind_gpu = cp.asarray(bright_inds[cand_start:cand_stop], dtype=cp.int32)
            r_gpu = cp.asarray(ratios[cand_start:cand_stop], dtype=cp.float64)

            dark_pdf = pdf_by_rate[d_ind_gpu, :]
            bright_pdf = pdf_by_rate[b_ind_gpu, :]
            model_gpu = r_gpu[:, None] * dark_pdf + (1.0 - r_gpu[:, None]) * bright_pdf
            model_gpu = cp.maximum(model_gpu, config.eps)
            model_gpu = model_gpu / cp.maximum(cp.sum(model_gpu, axis=1, keepdims=True), config.eps)

            b_gpu = w_gpu @ (model_gpu.T**2)
            c_gpu = (h_gpu * w_gpu) @ model_gpu.T
            loss_gpu = a_gpu[:, None] + b_gpu - 2.0 * c_gpu

            local_ind = cp.argmin(loss_gpu, axis=1)
            local_loss = loss_gpu[cp.arange(loss_gpu.shape[0]), local_ind]

            improve = local_loss < chunk_best_loss
            chunk_best_loss = cp.where(improve, local_loss, chunk_best_loss)
            chunk_best_candidate = cp.where(
                improve,
                local_ind.astype(cp.int64) + cand_start,
                chunk_best_candidate,
            )

        best_loss_cpu[row_start:row_stop] = cp.asnumpy(chunk_best_loss)
        best_candidate_cpu[row_start:row_stop] = cp.asnumpy(chunk_best_candidate)

    for local_row, input_ind in enumerate(valid_input_inds):
        cand = int(best_candidate_cpu[local_row])
        if cand < 0 or not np.isfinite(best_loss_cpu[local_row]):
            continue

        ratio = float(ratios[cand])
        dark_rate = float(rate_grid[int(dark_inds[cand])])
        bright_rate = float(rate_grid[int(bright_inds[cand])])
        popt = np.asarray([ratio, dark_rate, bright_rate], dtype=float)

        dof = max(int(np.sum(valid[local_row])) - 3, 1)
        red_chi_sq = float(best_loss_cpu[local_row] / dof)
        results[input_ind] = (popt, None, red_chi_sq)

    debug = {
        "used_gpu": True,
        "prob_dist": prob_name,
        "num_input": len(raw_items),
        "num_valid": len(cleaned),
        "x_max": x_max,
        "num_ratio": int(config.num_ratio),
        "num_rate": int(config.num_rate),
        "num_candidates": int(num_candidates),
        "rate_grid_min": float(rate_grid[0]),
        "rate_grid_max": float(rate_grid[-1]),
    }

    return (results, debug) if return_debug else results


def fit_bimodal_histogram_gpu(
    counts_list,
    prob_dist="COMPOUND_POISSON",
    config: GpuFitConfig | None = None,
    no_print: bool = True,
    no_plot: bool = True,
):
    """Single-histogram wrapper matching fit_bimodal_histogram()."""
    if not no_plot:
        warnings.warn("GPU fitter does not implement plotting; ignoring no_plot=False.")

    results, debug = fit_bimodal_histograms_gpu_batch(
        [counts_list],
        prob_dist=prob_dist,
        config=config,
        return_debug=True,
    )

    popt, pcov, red_chi_sq = results[0]
    if not no_print:
        print(f"[gpu coarse] popt={popt} red_chi_sq={red_chi_sq}")
        print(f"[gpu coarse] debug={debug}")

    return popt, pcov, red_chi_sq


# =============================================================================
# Equal-brightness multi-NV GPU model
# =============================================================================


def _binom_coeffs(n_nvs: int) -> np.ndarray:
    return np.asarray([math.comb(n_nvs, k) for k in range(n_nvs + 1)], dtype=np.float64)


def _binom_weights_cpu(n_nvs: int, p_minus: float) -> np.ndarray:
    p_minus = float(np.clip(p_minus, 0.0, 1.0))
    ks = np.arange(n_nvs + 1, dtype=float)
    coeff = _binom_coeffs(n_nvs)
    w = coeff * (p_minus**ks) * ((1.0 - p_minus) ** (n_nvs - ks))
    s = float(np.sum(w))
    if s <= 0 or not np.isfinite(s):
        out = np.zeros(n_nvs + 1, dtype=float)
        out[0] = 1.0
        return out
    return w / s


def _lambda_equal_cpu(popt, n_nvs: int) -> np.ndarray:
    p_minus, bg, rate0, delta = [float(v) for v in np.asarray(popt, dtype=float)]
    ks = np.arange(n_nvs + 1, dtype=float)
    base = max(bg, 0.0) + int(n_nvs) * max(rate0, 1e-12)
    return base + ks * max(delta, 0.0)


def _adjacent_dprime_cpu(lambdas) -> np.ndarray:
    lambdas = np.asarray(lambdas, dtype=float)
    if lambdas.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(lambdas) / np.sqrt(lambdas[:-1] + lambdas[1:] + 1e-12)


def _multimode_bounds(stats, n_nvs: int, config: GpuMultimodeFitConfig):
    q02 = stats["q02"]
    q15 = stats["q15"]
    q65 = stats["q65"]
    q98 = stats["q98"]

    bg_hi = max(
        1.0,
        0.5 * float(np.nanmedian(q15)),
        float(np.nanpercentile(q02, 90)),
    )
    rate0_hi = max(
        1e-3,
        float(np.nanpercentile(q65, 90)) / max(n_nvs, 1),
        float(np.nanpercentile(q98, 75)) / max(n_nvs + 1, 1),
    )
    delta_hi = max(
        1e-3,
        (float(np.nanpercentile(q98, 95)) - float(np.nanpercentile(q15, 5)))
        / max(n_nvs, 1),
    )

    lo = np.asarray([config.p_min, 0.0, config.eps, 0.0], dtype=np.float64)
    hi = np.asarray([config.p_max, bg_hi, rate0_hi, delta_hi], dtype=np.float64)
    return lo, hi


def _make_multimode_candidates(stats, n_nvs: int, config: GpuMultimodeFitConfig):
    lo, hi = _multimode_bounds(stats, n_nvs, config)

    p_grid = np.linspace(lo[0], hi[0], config.num_p, dtype=np.float64)
    bg_grid = np.linspace(lo[1], hi[1], config.num_bg, dtype=np.float64)
    rate0_grid = np.linspace(lo[2], hi[2], config.num_rate0, dtype=np.float64)
    delta_grid = np.linspace(lo[3], hi[3], config.num_delta, dtype=np.float64)

    p, bg, rate0, delta = np.meshgrid(
        p_grid,
        bg_grid,
        rate0_grid,
        delta_grid,
        indexing="ij",
    )
    candidates = np.column_stack([p.ravel(), bg.ravel(), rate0.ravel(), delta.ravel()])
    return candidates.astype(np.float64), lo, hi


def _multimode_pdf_for_candidates_gpu(
    prob_name: str,
    x_gpu,
    candidates,
    n_nvs: int,
    config: GpuMultimodeFitConfig,
):
    """Return model[candidate, x] for equal-brightness N-NV candidates."""
    cand_gpu = cp.asarray(candidates, dtype=cp.float64)
    p_minus = cp.clip(cand_gpu[:, 0], config.eps, 1.0 - config.eps)
    bg = cp.maximum(cand_gpu[:, 1], 0.0)
    rate0 = cp.maximum(cand_gpu[:, 2], config.eps)
    delta = cp.maximum(cand_gpu[:, 3], 0.0)

    ks_cpu = np.arange(n_nvs + 1, dtype=np.float64)
    coeff_cpu = _binom_coeffs(n_nvs)
    ks_gpu = cp.asarray(ks_cpu, dtype=cp.float64)
    coeff_gpu = cp.asarray(coeff_cpu, dtype=cp.float64)

    weights = (
        coeff_gpu[None, :]
        * (p_minus[:, None] ** ks_gpu[None, :])
        * ((1.0 - p_minus[:, None]) ** (n_nvs - ks_gpu[None, :]))
    )
    weights = weights / cp.maximum(cp.sum(weights, axis=1, keepdims=True), config.eps)

    rates = bg[:, None] + n_nvs * rate0[:, None] + ks_gpu[None, :] * delta[:, None]
    rates_flat = rates.reshape(-1)
    rates_cpu = cp.asnumpy(rates_flat)

    pdf_flat = _single_mode_pdf_table(prob_name, x_gpu, rates_flat, rates_cpu, config)
    pdf = pdf_flat.reshape(cand_gpu.shape[0], n_nvs + 1, x_gpu.size)

    model = cp.sum(weights[:, :, None] * pdf, axis=1)
    model = cp.maximum(model, config.eps)
    model = model / cp.maximum(cp.sum(model, axis=1, keepdims=True), config.eps)
    return model


def _evaluate_equal_candidates_nll_chi_gpu(
    prob_name,
    x_gpu,
    candidates,
    n_nvs,
    counts_gpu,
    hist_gpu,
    w_gpu,
    hist_sq_gpu,
    config,
):
    model_gpu = _multimode_pdf_for_candidates_gpu(
        prob_name,
        x_gpu,
        candidates,
        int(n_nvs),
        config,
    )
    log_model = cp.log(cp.maximum(model_gpu, config.eps))
    nll = -(counts_gpu @ log_model.T)

    model_sq = w_gpu @ (model_gpu.T**2)
    cross = (hist_gpu * w_gpu) @ model_gpu.T
    chi = hist_sq_gpu[:, None] + model_sq - 2.0 * cross
    return nll, chi


def _coarse_fit_equal_for_n(
    prob_name,
    x_vals,
    hists,
    hist_errs,
    valid,
    bin_counts,
    n_samples,
    stats,
    n_nvs,
    config,
):
    candidates, lo, hi = _make_multimode_candidates(stats, int(n_nvs), config)
    x_gpu = cp.asarray(x_vals, dtype=cp.float64)

    num_rows = hists.shape[0]
    best_nll = np.full(num_rows, np.inf, dtype=np.float64)
    best_chi = np.full(num_rows, np.inf, dtype=np.float64)
    best_cand = np.full(num_rows, -1, dtype=np.int64)

    fit_chunk = int(config.fit_chunk_size)
    cand_chunk = int(config.candidate_chunk_size)

    for row_start in range(0, num_rows, fit_chunk):
        row_stop = min(row_start + fit_chunk, num_rows)
        rows = row_stop - row_start

        hist_gpu = cp.asarray(hists[row_start:row_stop], dtype=cp.float64)
        counts_gpu = cp.asarray(bin_counts[row_start:row_stop], dtype=cp.float64)
        err_gpu = cp.asarray(hist_errs[row_start:row_stop], dtype=cp.float64)
        valid_gpu = cp.asarray(valid[row_start:row_stop], dtype=cp.float64)

        w_gpu = valid_gpu / cp.maximum(err_gpu, config.eps) ** 2
        hist_sq_gpu = cp.sum((hist_gpu**2) * w_gpu, axis=1)

        chunk_best_nll = cp.full(rows, cp.inf, dtype=cp.float64)
        chunk_best_chi = cp.full(rows, cp.inf, dtype=cp.float64)
        chunk_best_cand = cp.full(rows, -1, dtype=cp.int64)

        for cand_start in range(0, candidates.shape[0], cand_chunk):
            cand_stop = min(cand_start + cand_chunk, candidates.shape[0])
            cand_cpu = candidates[cand_start:cand_stop]

            nll, chi = _evaluate_equal_candidates_nll_chi_gpu(
                prob_name,
                x_gpu,
                cand_cpu,
                int(n_nvs),
                counts_gpu,
                hist_gpu,
                w_gpu,
                hist_sq_gpu,
                config,
            )

            local_ind = cp.argmin(nll, axis=1)
            local_nll = nll[cp.arange(rows), local_ind]
            local_chi = chi[cp.arange(rows), local_ind]

            improve = local_nll < chunk_best_nll
            chunk_best_nll = cp.where(improve, local_nll, chunk_best_nll)
            chunk_best_chi = cp.where(improve, local_chi, chunk_best_chi)
            chunk_best_cand = cp.where(
                improve,
                local_ind.astype(cp.int64) + cand_start,
                chunk_best_cand,
            )

        best_nll[row_start:row_stop] = cp.asnumpy(chunk_best_nll)
        best_chi[row_start:row_stop] = cp.asnumpy(chunk_best_chi)
        best_cand[row_start:row_stop] = cp.asnumpy(chunk_best_cand)

    best_popt = np.full((num_rows, 4), np.nan, dtype=np.float64)
    valid_best = best_cand >= 0
    best_popt[valid_best, :] = candidates[best_cand[valid_best], :]

    return best_popt, best_nll, best_chi, lo, hi, int(candidates.shape[0])


def _local_offsets_equal_cpu():
    vals = [-1.0, 0.0, 1.0]
    offsets = []
    for a in vals:
        for b in vals:
            for c in vals:
                for d in vals:
                    offsets.append([a, b, c, d])
    return np.asarray(offsets, dtype=np.float64)


def _refine_equal_for_n(
    prob_name,
    x_vals,
    hists,
    hist_errs,
    valid,
    bin_counts,
    current_popt,
    n_nvs,
    lo,
    hi,
    config,
):
    if not bool(config.use_refinement) or int(config.refine_iters) <= 0:
        # Evaluate current candidates once to return consistent values.
        pass

    num_rows = hists.shape[0]
    x_gpu = cp.asarray(x_vals, dtype=cp.float64)
    offsets_gpu = cp.asarray(_local_offsets_equal_cpu(), dtype=cp.float64)
    num_offsets = int(offsets_gpu.shape[0])

    cur = np.asarray(current_popt, dtype=np.float64).copy()
    finite = np.all(np.isfinite(cur), axis=1)

    lo_gpu = cp.asarray(lo, dtype=cp.float64)
    hi_gpu = cp.asarray(hi, dtype=cp.float64)

    # Initial step sizes are row-specific for scale parameters.
    step = np.zeros_like(cur)
    step[:, 0] = float(config.refine_p_step)
    step[:, 1] = np.maximum(np.abs(cur[:, 1]) * float(config.refine_bg_frac_step), float(config.refine_min_abs_step))
    step[:, 2] = np.maximum(np.abs(cur[:, 2]) * float(config.refine_rate0_frac_step), float(config.refine_min_abs_step))
    step[:, 3] = np.maximum(np.abs(cur[:, 3]) * float(config.refine_delta_frac_step), float(config.refine_min_abs_step))

    step[~finite, :] = np.nan

    refine_chunk = max(1, int(config.refine_fit_chunk_size))
    n_iters = int(config.refine_iters) if bool(config.use_refinement) else 0

    for _iter in range(n_iters):
        shrink = float(config.refine_shrink) ** _iter
        step_iter = step * shrink

        for row_start in range(0, num_rows, refine_chunk):
            row_stop = min(row_start + refine_chunk, num_rows)
            rows = row_stop - row_start

            good_rows = finite[row_start:row_stop]
            if not np.any(good_rows):
                continue

            cur_gpu = cp.asarray(cur[row_start:row_stop], dtype=cp.float64)
            step_gpu = cp.asarray(step_iter[row_start:row_stop], dtype=cp.float64)

            cand_gpu = cur_gpu[:, None, :] + offsets_gpu[None, :, :] * step_gpu[:, None, :]
            cand_gpu = cp.minimum(cp.maximum(cand_gpu, lo_gpu[None, None, :]), hi_gpu[None, None, :])
            cand_flat = cand_gpu.reshape(rows * num_offsets, 4)

            hist_gpu = cp.asarray(hists[row_start:row_stop], dtype=cp.float64)
            counts_gpu = cp.asarray(bin_counts[row_start:row_stop], dtype=cp.float64)
            err_gpu = cp.asarray(hist_errs[row_start:row_stop], dtype=cp.float64)
            valid_gpu = cp.asarray(valid[row_start:row_stop], dtype=cp.float64)

            w_gpu = valid_gpu / cp.maximum(err_gpu, config.eps) ** 2
            hist_sq_gpu = cp.sum((hist_gpu**2) * w_gpu, axis=1)

            model_flat = _multimode_pdf_for_candidates_gpu(
                prob_name,
                x_gpu,
                cand_flat,
                int(n_nvs),
                config,
            )
            model = model_flat.reshape(rows, num_offsets, x_gpu.size)
            log_model = cp.log(cp.maximum(model, config.eps))
            nll = -cp.sum(counts_gpu[:, None, :] * log_model, axis=2)

            best_off = cp.argmin(nll, axis=1)
            best_cand = cand_gpu[cp.arange(rows), best_off, :]
            cur[row_start:row_stop, :] = cp.asnumpy(best_cand)

    # Final evaluation.
    best_nll = np.full(num_rows, np.inf, dtype=np.float64)
    best_chi = np.full(num_rows, np.inf, dtype=np.float64)

    for row_start in range(0, num_rows, refine_chunk):
        row_stop = min(row_start + refine_chunk, num_rows)
        rows = row_stop - row_start
        if not np.any(finite[row_start:row_stop]):
            continue

        hist_gpu = cp.asarray(hists[row_start:row_stop], dtype=cp.float64)
        counts_gpu = cp.asarray(bin_counts[row_start:row_stop], dtype=cp.float64)
        err_gpu = cp.asarray(hist_errs[row_start:row_stop], dtype=cp.float64)
        valid_gpu = cp.asarray(valid[row_start:row_stop], dtype=cp.float64)

        w_gpu = valid_gpu / cp.maximum(err_gpu, config.eps) ** 2
        hist_sq_gpu = cp.sum((hist_gpu**2) * w_gpu, axis=1)

        candidates = cur[row_start:row_stop]
        model = _multimode_pdf_for_candidates_gpu(
            prob_name,
            x_gpu,
            candidates,
            int(n_nvs),
            config,
        )
        log_model = cp.log(cp.maximum(model, config.eps))
        nll = -cp.sum(counts_gpu * log_model, axis=1)

        model_sq = cp.sum(w_gpu * (model**2), axis=1)
        cross = cp.sum(hist_gpu * w_gpu * model, axis=1)
        chi = hist_sq_gpu + model_sq - 2.0 * cross

        best_nll[row_start:row_stop] = cp.asnumpy(nll)
        best_chi[row_start:row_stop] = cp.asnumpy(chi)

    return cur, best_nll, best_chi


def _visibility_diagnostics_equal(fit, config: GpuMultimodeFitConfig):
    if not isinstance(fit, dict) or not fit.get("ok", False):
        return {"visibility_ok": False, "reason": "fit_not_ok"}

    n_nvs = int(fit.get("n_nvs", 1))
    popt = np.asarray(fit.get("popt", []), dtype=float)
    n_samples = float(fit.get("num_samples", fit.get("n_samp", 0)))

    if popt.size != 4 or not np.all(np.isfinite(popt)):
        return {"visibility_ok": False, "reason": "bad_popt"}

    weights = _binom_weights_cpu(n_nvs, popt[0])
    expected_shots = weights * n_samples
    lambdas = _lambda_equal_cpu(popt, n_nvs)
    dprimes = _adjacent_dprime_cpu(lambdas)

    if n_nvs <= 1:
        return {
            "visibility_ok": True,
            "reason": "N1_allowed",
            "weights": weights.tolist(),
            "expected_shots": expected_shots.tolist(),
            "lambdas": lambdas.tolist(),
            "adjacent_dprime": dprimes.tolist(),
            "min_adjacent_dprime": float(np.nanmin(dprimes)) if dprimes.size else np.nan,
        }

    if bool(config.strict_require_all_modes):
        modes_to_check = np.arange(n_nvs + 1, dtype=int)
    else:
        modes_to_check = np.arange(1, n_nvs, dtype=int)
        if modes_to_check.size == 0:
            modes_to_check = np.arange(n_nvs + 1, dtype=int)

    weights_ok = bool(np.all(weights[modes_to_check] >= float(config.strict_min_mode_weight)))
    shots_ok = bool(np.all(expected_shots[modes_to_check] >= float(config.strict_min_mode_shots)))
    sep_ok = bool(np.all(dprimes >= float(config.strict_min_adjacent_dprime))) if dprimes.size else False

    if not weights_ok:
        reason = "mode_weight_too_small"
    elif not shots_ok:
        reason = "mode_expected_shots_too_small"
    elif not sep_ok:
        reason = "adjacent_peaks_not_separated"
    else:
        reason = "visible"

    return {
        "visibility_ok": bool(weights_ok and shots_ok and sep_ok),
        "reason": reason,
        "weights": weights.tolist(),
        "expected_shots": expected_shots.tolist(),
        "lambdas": lambdas.tolist(),
        "adjacent_dprime": dprimes.tolist(),
        "modes_checked": modes_to_check.tolist(),
        "min_weight_checked": float(np.nanmin(weights[modes_to_check])),
        "min_expected_shots_checked": float(np.nanmin(expected_shots[modes_to_check])),
        "min_adjacent_dprime": float(np.nanmin(dprimes)) if dprimes.size else np.nan,
        "weights_ok": weights_ok,
        "shots_ok": shots_ok,
        "sep_ok": sep_ok,
    }


def _adjusted_bic_strict(fit, config: GpuMultimodeFitConfig):
    if not isinstance(fit, dict) or not fit.get("ok", False):
        return np.inf
    bic = float(fit.get("bic", np.inf))
    n_nvs = int(fit.get("n_nvs", 1))
    return bic + float(config.strict_extra_nv_penalty) * max(n_nvs - 1, 0)


# =============================================================================
# Optional 2-NV unequal-brightness GPU diagnostic
# =============================================================================


def _unequal2_bounds_from_equal(equal_popt, hi_equal, config: GpuMultimodeFitConfig):
    # [p1, p2, bg, rate0, delta1, delta2]
    lo = np.asarray([config.p_min, config.p_min, 0.0, config.eps, 0.0, 0.0], dtype=np.float64)
    hi = np.asarray([
        config.p_max,
        config.p_max,
        hi_equal[1],
        hi_equal[2],
        max(hi_equal[3] * 1.75, config.refine_min_abs_step),
        max(hi_equal[3] * 1.75, config.refine_min_abs_step),
    ], dtype=np.float64)
    return lo, hi


def _unequal2_initial_from_equal(best_equal_2):
    p, bg, rate0, delta = [float(v) for v in np.asarray(best_equal_2, dtype=float)]
    delta = max(delta, 0.0)
    starts = [
        [p, p, bg, rate0, 0.75 * delta, 1.25 * delta],
        [p, p, bg, rate0, 0.50 * delta, 1.50 * delta],
        [min(max(p - 0.10, 0.02), 0.98), min(max(p + 0.10, 0.02), 0.98), bg, rate0, 0.75 * delta, 1.25 * delta],
        [min(max(p + 0.10, 0.02), 0.98), min(max(p - 0.10, 0.02), 0.98), bg, rate0, 1.25 * delta, 0.75 * delta],
    ]
    return np.asarray(starts, dtype=np.float64)


def _unequal2_pdf_for_candidates_gpu(prob_name, x_gpu, candidates, config: GpuMultimodeFitConfig):
    cand_gpu = cp.asarray(candidates, dtype=cp.float64)
    p1 = cp.clip(cand_gpu[:, 0], config.eps, 1.0 - config.eps)
    p2 = cp.clip(cand_gpu[:, 1], config.eps, 1.0 - config.eps)
    bg = cp.maximum(cand_gpu[:, 2], 0.0)
    rate0 = cp.maximum(cand_gpu[:, 3], config.eps)
    d1 = cp.maximum(cand_gpu[:, 4], 0.0)
    d2 = cp.maximum(cand_gpu[:, 5], 0.0)

    weights = cp.stack([
        (1.0 - p1) * (1.0 - p2),
        p1 * (1.0 - p2),
        (1.0 - p1) * p2,
        p1 * p2,
    ], axis=1)
    weights = weights / cp.maximum(cp.sum(weights, axis=1, keepdims=True), config.eps)

    base = bg + 2.0 * rate0
    rates = cp.stack([base, base + d1, base + d2, base + d1 + d2], axis=1)
    rates_flat = rates.reshape(-1)
    rates_cpu = cp.asnumpy(rates_flat)

    pdf_flat = _single_mode_pdf_table(prob_name, x_gpu, rates_flat, rates_cpu, config)
    pdf = pdf_flat.reshape(cand_gpu.shape[0], 4, x_gpu.size)
    model = cp.sum(weights[:, :, None] * pdf, axis=1)
    model = cp.maximum(model, config.eps)
    model = model / cp.maximum(cp.sum(model, axis=1, keepdims=True), config.eps)
    return model


def _local_offsets_unequal2_cpu():
    # Coordinate-style neighborhood: center plus +/- each parameter.
    offsets = [np.zeros(6, dtype=np.float64)]
    for i in range(6):
        v = np.zeros(6, dtype=np.float64)
        v[i] = -1.0
        offsets.append(v.copy())
        v[i] = 1.0
        offsets.append(v.copy())
    return np.vstack(offsets).astype(np.float64)


def _fit_unequal2_diagnostic_gpu(
    prob_name,
    x_vals,
    hists,
    hist_errs,
    valid,
    bin_counts,
    equal2_popt,
    equal_hi,
    n_samples,
    config,
):
    num_rows = hists.shape[0]
    x_gpu = cp.asarray(x_vals, dtype=cp.float64)
    offsets_gpu = cp.asarray(_local_offsets_unequal2_cpu(), dtype=cp.float64)
    num_offsets = int(offsets_gpu.shape[0])

    lo, hi = _unequal2_bounds_from_equal(None, equal_hi, config)
    lo_gpu = cp.asarray(lo, dtype=cp.float64)
    hi_gpu = cp.asarray(hi, dtype=cp.float64)

    # Pick best among a few deterministic starts per row.
    current = np.full((num_rows, 6), np.nan, dtype=np.float64)
    best_nll = np.full(num_rows, np.inf, dtype=np.float64)
    best_chi = np.full(num_rows, np.inf, dtype=np.float64)

    starts_per_row = []
    for row in range(num_rows):
        if not np.all(np.isfinite(equal2_popt[row])):
            starts_per_row.append(None)
            continue
        starts = _unequal2_initial_from_equal(equal2_popt[row])
        starts = np.minimum(np.maximum(starts, lo[None, :]), hi[None, :])
        starts_per_row.append(starts)

    refine_chunk = max(1, int(config.refine_fit_chunk_size))

    # Initialize by evaluating starts.
    for row_start in range(0, num_rows, refine_chunk):
        row_stop = min(row_start + refine_chunk, num_rows)
        rows = row_stop - row_start

        max_starts = 4
        cand = np.full((rows, max_starts, 6), np.nan, dtype=np.float64)
        for rr in range(rows):
            starts = starts_per_row[row_start + rr]
            if starts is not None:
                cand[rr, : starts.shape[0], :] = starts

        finite = np.all(np.isfinite(cand), axis=2)
        if not np.any(finite):
            continue

        cand_flat = cand.reshape(rows * max_starts, 6)
        # Replace bad candidates with safe values; they will be masked after evaluation.
        bad = ~np.all(np.isfinite(cand_flat), axis=1)
        cand_flat[bad] = lo

        hist_gpu = cp.asarray(hists[row_start:row_stop], dtype=cp.float64)
        counts_gpu = cp.asarray(bin_counts[row_start:row_stop], dtype=cp.float64)
        err_gpu = cp.asarray(hist_errs[row_start:row_stop], dtype=cp.float64)
        valid_gpu = cp.asarray(valid[row_start:row_stop], dtype=cp.float64)
        w_gpu = valid_gpu / cp.maximum(err_gpu, config.eps) ** 2
        hist_sq_gpu = cp.sum((hist_gpu**2) * w_gpu, axis=1)

        model_flat = _unequal2_pdf_for_candidates_gpu(prob_name, x_gpu, cand_flat, config)
        model = model_flat.reshape(rows, max_starts, x_gpu.size)
        log_model = cp.log(cp.maximum(model, config.eps))
        nll = -cp.sum(counts_gpu[:, None, :] * log_model, axis=2)
        nll = cp.where(cp.asarray(finite), nll, cp.inf)

        best_s = cp.argmin(nll, axis=1)
        best_local = cand[cp.asnumpy(np.arange(rows)), cp.asnumpy(best_s)] if False else None
        best_s_cpu = cp.asnumpy(best_s)
        for rr in range(rows):
            if not np.isfinite(cp.asnumpy(nll[rr, best_s[rr]])):
                continue
            current[row_start + rr, :] = cand[rr, int(best_s_cpu[rr]), :]

    finite_rows = np.all(np.isfinite(current), axis=1)
    if not np.any(finite_rows):
        return None

    step = np.zeros_like(current)
    step[:, 0] = float(config.refine_p_step)
    step[:, 1] = float(config.refine_p_step)
    step[:, 2] = np.maximum(np.abs(current[:, 2]) * float(config.refine_bg_frac_step), float(config.refine_min_abs_step))
    step[:, 3] = np.maximum(np.abs(current[:, 3]) * float(config.refine_rate0_frac_step), float(config.refine_min_abs_step))
    step[:, 4] = np.maximum(np.abs(current[:, 4]) * float(config.refine_delta_frac_step), float(config.refine_min_abs_step))
    step[:, 5] = np.maximum(np.abs(current[:, 5]) * float(config.refine_delta_frac_step), float(config.refine_min_abs_step))

    for it in range(int(config.unequal_refine_iters)):
        shrink = float(config.unequal_refine_shrink) ** it
        step_iter = step * shrink

        for row_start in range(0, num_rows, refine_chunk):
            row_stop = min(row_start + refine_chunk, num_rows)
            rows = row_stop - row_start
            if not np.any(finite_rows[row_start:row_stop]):
                continue

            cur_gpu = cp.asarray(current[row_start:row_stop], dtype=cp.float64)
            step_gpu = cp.asarray(step_iter[row_start:row_stop], dtype=cp.float64)
            cand_gpu = cur_gpu[:, None, :] + offsets_gpu[None, :, :] * step_gpu[:, None, :]
            cand_gpu = cp.minimum(cp.maximum(cand_gpu, lo_gpu[None, None, :]), hi_gpu[None, None, :])
            cand_flat = cand_gpu.reshape(rows * num_offsets, 6)

            counts_gpu = cp.asarray(bin_counts[row_start:row_stop], dtype=cp.float64)
            model_flat = _unequal2_pdf_for_candidates_gpu(prob_name, x_gpu, cand_flat, config)
            model = model_flat.reshape(rows, num_offsets, x_gpu.size)
            log_model = cp.log(cp.maximum(model, config.eps))
            nll = -cp.sum(counts_gpu[:, None, :] * log_model, axis=2)
            best_o = cp.argmin(nll, axis=1)
            best_cand = cand_gpu[cp.arange(rows), best_o, :]
            current[row_start:row_stop, :] = cp.asnumpy(best_cand)

    # Final evaluation.
    for row_start in range(0, num_rows, refine_chunk):
        row_stop = min(row_start + refine_chunk, num_rows)
        rows = row_stop - row_start
        if not np.any(finite_rows[row_start:row_stop]):
            continue

        hist_gpu = cp.asarray(hists[row_start:row_stop], dtype=cp.float64)
        counts_gpu = cp.asarray(bin_counts[row_start:row_stop], dtype=cp.float64)
        err_gpu = cp.asarray(hist_errs[row_start:row_stop], dtype=cp.float64)
        valid_gpu = cp.asarray(valid[row_start:row_stop], dtype=cp.float64)
        w_gpu = valid_gpu / cp.maximum(err_gpu, config.eps) ** 2
        hist_sq_gpu = cp.sum((hist_gpu**2) * w_gpu, axis=1)

        cand = current[row_start:row_stop]
        model = _unequal2_pdf_for_candidates_gpu(prob_name, x_gpu, cand, config)
        log_model = cp.log(cp.maximum(model, config.eps))
        nll = -cp.sum(counts_gpu * log_model, axis=1)
        model_sq = cp.sum(w_gpu * (model**2), axis=1)
        cross = cp.sum(hist_gpu * w_gpu * model, axis=1)
        chi = hist_sq_gpu + model_sq - 2.0 * cross

        best_nll[row_start:row_stop] = cp.asnumpy(nll)
        best_chi[row_start:row_stop] = cp.asnumpy(chi)

    k_free = 6
    bic = (
        2.0 * best_nll
        + k_free * np.log(np.maximum(n_samples, 1.0))
        + float(config.unequal_bic_extra_penalty)
    )
    dof = np.maximum(np.sum(valid, axis=1) - k_free, 1)
    red = best_chi / dof

    return {
        "popt": current,
        "nll": best_nll,
        "chi": best_chi,
        "bic": bic,
        "red_chi_sq": red,
        "k_free": k_free,
    }


# =============================================================================
# Public multi-NV GPU batch fitter
# =============================================================================


def fit_binomial_multinv_histograms_gpu_batch(
    counts_batch: Iterable[Sequence[float]],
    prob_dist="COMPOUND_POISSON",
    config: GpuMultimodeFitConfig | None = None,
    max_nvs: int | None = None,
    force_nvs: int | None = None,
    strict_selection: bool = False,
    return_debug: bool = False,
):
    """
    GPU search/refinement for a physical equal-brightness multi-NV model.

    Model:
        P(k) = Binomial(N, p_minus)
        lambda_k = bg + N*rate0 + k*delta

    If strict_selection=True and force_nvs is None, N>1 is accepted only when
    BIC improves enough and the fitted modes are physically visible.
    """
    if config is None:
        config = GpuMultimodeFitConfig()

    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy GPU backend is not available.")

    prob_name = _prob_dist_name(prob_dist)
    if prob_name not in SUPPORTED_PROB_DIST_NAMES:
        raise ValueError(
            f"GPU fitting currently supports {sorted(SUPPORTED_PROB_DIST_NAMES)}, got {prob_name}"
        )

    raw_items = list(counts_batch)
    if len(raw_items) == 0:
        return ([], {}) if return_debug else []

    cleaned = []
    valid_input_inds = []
    for ind, counts in enumerate(raw_items):
        clean = _clean_counts(counts, config.min_samples)
        if clean is None:
            continue
        cleaned.append(clean)
        valid_input_inds.append(ind)

    failed = {"ok": False, "reason": "fit_failed"}
    results = [dict(failed) for _ in raw_items]
    if not cleaned:
        debug = {"reason": "no_valid_histograms", "num_input": len(raw_items)}
        return (results, debug) if return_debug else results

    stats = _initial_stats(cleaned)
    x_max = int(max(np.max(c) for c in cleaned)) + int(config.max_count_padding)
    x_max = max(x_max, 1)
    x_vals = np.arange(x_max + 1, dtype=np.float64)

    num_clean = len(cleaned)
    hists = np.zeros((num_clean, x_max + 1), dtype=np.float64)
    bin_counts = np.zeros_like(hists, dtype=np.float64)
    hist_errs = np.ones_like(hists, dtype=np.float64)
    valid = np.zeros_like(hists, dtype=bool)
    n_samples = np.zeros(num_clean, dtype=np.float64)

    for row, counts in enumerate(cleaned):
        hist, err, mask, bc = _make_integer_histogram(counts, x_max)
        hists[row, :] = hist
        hist_errs[row, :] = err
        valid[row, :] = mask
        bin_counts[row, :] = bc
        n_samples[row] = float(np.sum(bc))

    if force_nvs is not None:
        n_values = [int(force_nvs)]
    else:
        n_stop = int(config.max_nvs if max_nvs is None else max_nvs)
        n_values = list(range(1, n_stop + 1))

    # Fit each forced N separately.
    fits_by_n: dict[int, list[dict]] = {}
    candidate_counts = {}
    equal_hi_by_n = {}

    for n_nvs in n_values:
        print(f"GPU fitting forced N={n_nvs}...")

        coarse_popt, coarse_nll, coarse_chi, lo, hi, num_candidates = _coarse_fit_equal_for_n(
            prob_name=prob_name,
            x_vals=x_vals,
            hists=hists,
            hist_errs=hist_errs,
            valid=valid,
            bin_counts=bin_counts,
            n_samples=n_samples,
            stats=stats,
            n_nvs=int(n_nvs),
            config=config,
        )
        candidate_counts[int(n_nvs)] = int(num_candidates)
        equal_hi_by_n[int(n_nvs)] = hi

        if bool(config.use_refinement):
            popt, nll, chi = _refine_equal_for_n(
                prob_name=prob_name,
                x_vals=x_vals,
                hists=hists,
                hist_errs=hist_errs,
                valid=valid,
                bin_counts=bin_counts,
                current_popt=coarse_popt,
                n_nvs=int(n_nvs),
                lo=lo,
                hi=hi,
                config=config,
            )
        else:
            popt, nll, chi = coarse_popt, coarse_nll, coarse_chi

        k_free = 4 + max(int(n_nvs) - 1, 0)
        dof = np.maximum(np.sum(valid, axis=1) - k_free, 1)
        red = chi / dof
        bic = (
            2.0 * nll
            + k_free * np.log(np.maximum(n_samples, 1.0))
            + float(config.bic_extra_nv_penalty) * max(int(n_nvs) - 1, 0)
        )

        fits_n = []
        for row in range(num_clean):
            if not np.all(np.isfinite(popt[row])) or not np.isfinite(bic[row]):
                fits_n.append({"ok": False, "model": f"{n_nvs}nv_equal", "n_nvs": int(n_nvs)})
                continue

            fit = {
                "ok": True,
                "model": f"{n_nvs}nv_equal",
                "n_nvs": int(n_nvs),
                "popt": np.asarray(popt[row], dtype=float),
                "pcov": None,
                "red_chi_sq": float(red[row]),
                "nll": float(nll[row]),
                "ll": float(-nll[row]),
                "bic": float(bic[row]),
                "k_free": int(k_free),
                "num_samples": int(n_samples[row]),
                "n_samp": float(n_samples[row]),
                "x_max": int(x_max),
                "refined_gpu": bool(config.use_refinement),
            }
            fit["visibility"] = _visibility_diagnostics_equal(fit, config)
            fit["strict_adjusted_bic"] = float(_adjusted_bic_strict(fit, config))
            fits_n.append(fit)

        fits_by_n[int(n_nvs)] = fits_n

    # Optional 2NV unequal diagnostic, using equal N=2 as the seed.
    unequal2_by_row = [None for _ in range(num_clean)]
    if bool(config.include_2nv_unequal) and (2 in fits_by_n):
        print("GPU fitting 2NV unequal diagnostic...")
        equal2_popt = np.full((num_clean, 4), np.nan, dtype=np.float64)
        for row, fit in enumerate(fits_by_n[2]):
            if isinstance(fit, dict) and fit.get("ok", False):
                equal2_popt[row, :] = np.asarray(fit["popt"], dtype=float)

        unequal = _fit_unequal2_diagnostic_gpu(
            prob_name=prob_name,
            x_vals=x_vals,
            hists=hists,
            hist_errs=hist_errs,
            valid=valid,
            bin_counts=bin_counts,
            equal2_popt=equal2_popt,
            equal_hi=equal_hi_by_n[2],
            n_samples=n_samples,
            config=config,
        )
        if unequal is not None:
            for row in range(num_clean):
                if not np.all(np.isfinite(unequal["popt"][row])) or not np.isfinite(unequal["bic"][row]):
                    continue
                p1, p2, bg, rate0, d1, d2 = [float(v) for v in unequal["popt"][row]]
                if d2 < d1:
                    p1, p2, d1, d2 = p2, p1, d2, d1
                unequal2_by_row[row] = {
                    "ok": True,
                    "model": "2nv_unequal",
                    "n_nvs": 2,
                    "popt": np.asarray([p1, p2, bg, rate0, d1, d2], dtype=float),
                    "p1": p1,
                    "p2": p2,
                    "bg": bg,
                    "rate0": rate0,
                    "delta1": d1,
                    "delta2": d2,
                    "red_chi_sq": float(unequal["red_chi_sq"][row]),
                    "nll": float(unequal["nll"][row]),
                    "ll": float(-unequal["nll"][row]),
                    "bic": float(unequal["bic"][row]),
                    "k_free": 6,
                    "num_samples": int(n_samples[row]),
                    "x_max": int(x_max),
                }

    # Select a main equal-brightness model per row.
    for row, input_ind in enumerate(valid_input_inds):
        candidate_results = []
        equal_candidates = []
        for n_nvs in n_values:
            fit = fits_by_n[int(n_nvs)][row]
            if isinstance(fit, dict) and fit.get("ok", False):
                candidate_results.append(fit)
                equal_candidates.append(fit)

        if unequal2_by_row[row] is not None:
            candidate_results.append(unequal2_by_row[row])

        if not equal_candidates:
            results[input_ind] = {"ok": False, "reason": "no_ok_equal_candidate"}
            continue

        if force_nvs is not None:
            selected = equal_candidates[0]
            selection_reason = f"forced_N{int(force_nvs)}"
        elif strict_selection:
            # Default to N=1 when available; upgrade only if physically justified.
            n1 = [f for f in equal_candidates if int(f["n_nvs"]) == 1]
            if n1:
                selected = n1[0]
                best_adj = float(selected["strict_adjusted_bic"])
                selection_reason = "N1_default"
            else:
                selected = min(equal_candidates, key=lambda f: float(f["strict_adjusted_bic"]))
                best_adj = float(selected["strict_adjusted_bic"])
                selection_reason = "fallback_lowest_adjusted_bic"

            for fit in equal_candidates:
                n_val = int(fit["n_nvs"])
                if n_val <= 1:
                    continue
                vis = fit.get("visibility", {})
                if not bool(vis.get("visibility_ok", False)):
                    continue
                cand_adj = float(fit["strict_adjusted_bic"])
                if cand_adj < best_adj - float(config.strict_bic_margin):
                    selected = fit
                    best_adj = cand_adj
                    selection_reason = f"accepted_N{n_val}_visible_adjusted_BIC"
        else:
            selected = min(equal_candidates, key=lambda f: float(f["bic"]))
            selection_reason = "lowest_BIC_equal"

        # Best diagnostic among equal + optional unequal.
        best_any = min(candidate_results, key=lambda f: float(f.get("bic", np.inf)))
        best_equal = min(equal_candidates, key=lambda f: float(f.get("bic", np.inf)))

        out = dict(selected)
        out["candidate_results"] = _as_json_safe(candidate_results)
        out["best_candidate_model"] = best_any.get("model", None)
        out["best_candidate_bic"] = float(best_any.get("bic", np.nan))
        out["best_equal_model"] = best_equal.get("model", None)
        out["best_equal_bic"] = float(best_equal.get("bic", np.nan))
        out["unequal_2nv_beats_equal"] = bool(
            best_any.get("model", None) == "2nv_unequal"
            and float(best_any.get("bic", np.inf)) < float(best_equal.get("bic", np.inf))
        )
        out["strict_reason"] = selection_reason
        out["strict_candidates"] = _as_json_safe({
            int(f["n_nvs"]): {
                "bic": float(f.get("bic", np.nan)),
                "strict_adjusted_bic": float(f.get("strict_adjusted_bic", np.nan)),
                "visibility": f.get("visibility", None),
            }
            for f in equal_candidates
        })
        results[input_ind] = out

    debug = {
        "used_gpu": True,
        "prob_dist": prob_name,
        "model_family": "binomial_multinv_equal_brightness_gpu_refined",
        "num_input": len(raw_items),
        "num_valid": len(cleaned),
        "x_max": int(x_max),
        "n_values": [int(v) for v in n_values],
        "candidate_counts": candidate_counts,
        "fit_chunk_size": int(config.fit_chunk_size),
        "candidate_chunk_size": int(config.candidate_chunk_size),
        "use_refinement": bool(config.use_refinement),
        "refine_iters": int(config.refine_iters),
        "strict_selection": bool(strict_selection),
        "include_2nv_unequal": bool(config.include_2nv_unequal),
    }

    return (results, debug) if return_debug else results


# =============================================================================
# Unified fitter + thresholds
# =============================================================================

def _hierarchical_is_bad(fit_res, red_chi_sq_stop):
    if not isinstance(fit_res, dict) or not fit_res.get("ok", False):
        return True
    red = float(fit_res.get("red_chi_sq", np.nan))
    return (not np.isfinite(red)) or red > float(red_chi_sq_stop)


def _hierarchical_accept_higher_n(old_res, new_res, config):
    if not isinstance(new_res, dict) or not new_res.get("ok", False):
        return False
    if not isinstance(old_res, dict) or not old_res.get("ok", False):
        return True

    old_bic = float(old_res.get("bic", np.nan))
    new_bic = float(new_res.get("bic", np.nan))
    old_red = float(old_res.get("red_chi_sq", np.nan))
    new_red = float(new_res.get("red_chi_sq", np.nan))

    bic_improvement = old_bic - new_bic
    red_improvement = old_red - new_red

    return (
        np.isfinite(bic_improvement)
        and np.isfinite(red_improvement)
        and bic_improvement >= float(config.hierarchical_min_bic_improvement)
        and red_improvement >= float(config.hierarchical_min_red_chi_sq_improvement)
    )


def fit_binomial_multinv_histograms_gpu_hierarchical_batch(
    counts_batch,
    prob_dist="COMPOUND_POISSON",
    config=None,
    max_nvs=3,
    return_debug=False,
):
    if config is None:
        config = GpuMultimodeFitConfig()

    raw_items = list(counts_batch)

    # Step 1: fit N=1 for every pillar.
    n1_results, n1_debug = fit_binomial_multinv_histograms_gpu_batch(
        raw_items,
        prob_dist=prob_dist,
        config=config,
        max_nvs=1,
        force_nvs=1,
        strict_selection=False,
        return_debug=True,
    )

    final_results = list(n1_results)

    tried_n2 = []
    tried_n3 = []
    accepted_n2 = []
    accepted_n3 = []

    # Step 2: only bad N=1 pillars get N=2.
    if int(max_nvs) >= 2:
        n2_inds = [
            ind for ind, res in enumerate(final_results)
            if _hierarchical_is_bad(
                res,
                config.hierarchical_n1_red_chi_sq_stop,
            )
        ]
        tried_n2 = [int(v) for v in n2_inds]

        if n2_inds:
            n2_counts = [raw_items[ind] for ind in n2_inds]
            n2_results, n2_debug = fit_binomial_multinv_histograms_gpu_batch(
                n2_counts,
                prob_dist=prob_dist,
                config=config,
                max_nvs=2,
                force_nvs=2,
                strict_selection=False,
                return_debug=True,
            )

            for global_ind, new_res in zip(n2_inds, n2_results):
                if _hierarchical_accept_higher_n(
                    final_results[global_ind],
                    new_res,
                    config,
                ):
                    new_res = dict(new_res)
                    new_res["strict_reason"] = "hierarchical_accepted_N2"
                    final_results[global_ind] = new_res
                    accepted_n2.append(int(global_ind))
        else:
            n2_debug = {"skipped": True}
    else:
        n2_debug = {"skipped": True}

    # Step 3: only still-bad pillars get N=3.
    if int(max_nvs) >= 3:
        n3_inds = [
            ind for ind, res in enumerate(final_results)
            if _hierarchical_is_bad(
                res,
                config.hierarchical_n2_red_chi_sq_stop,
            )
        ]
        tried_n3 = [int(v) for v in n3_inds]

        if n3_inds:
            n3_counts = [raw_items[ind] for ind in n3_inds]
            n3_results, n3_debug = fit_binomial_multinv_histograms_gpu_batch(
                n3_counts,
                prob_dist=prob_dist,
                config=config,
                max_nvs=3,
                force_nvs=3,
                strict_selection=False,
                return_debug=True,
            )

            for global_ind, new_res in zip(n3_inds, n3_results):
                if _hierarchical_accept_higher_n(
                    final_results[global_ind],
                    new_res,
                    config,
                ):
                    new_res = dict(new_res)
                    new_res["strict_reason"] = "hierarchical_accepted_N3"
                    final_results[global_ind] = new_res
                    accepted_n3.append(int(global_ind))
        else:
            n3_debug = {"skipped": True}
    else:
        n3_debug = {"skipped": True}

    debug = {
        "used_gpu": True,
        "prob_dist": _prob_dist_name(prob_dist),
        "model_family": "hierarchical_binomial_multinv_equal_brightness",
        "strategy": "N=1 first; only bad fits try N=2/N=3",
        "num_input": len(raw_items),
        "max_nvs": int(max_nvs),
        "n1_debug": n1_debug,
        "n2_debug": n2_debug,
        "n3_debug": n3_debug,
        "num_tried_n2": len(tried_n2),
        "num_tried_n3": len(tried_n3),
        "num_accepted_n2": len(accepted_n2),
        "num_accepted_n3": len(accepted_n3),
        "tried_n2_indices": tried_n2,
        "tried_n3_indices": tried_n3,
        "accepted_n2_indices": accepted_n2,
        "accepted_n3_indices": accepted_n3,
    }

    return (final_results, debug) if return_debug else final_results

def fit_charge_histograms_gpu_batch(
    counts_batch: Iterable[Sequence[float]],
    prob_dist="COMPOUND_POISSON",
    model_mode: str = "auto",
    bimodal_config: GpuFitConfig | None = None,
    multimode_config: GpuMultimodeFitConfig | None = None,
    max_nvs: int = 3,
    force_nvs: int | None = None,
    return_debug: bool = False,
):
    """
    Unified GPU fitter.

    model_mode:
        "bimodal"
            Old two-mode coarse search. Output is mapped to N=1 form.

        "multimode" or "auto"
            Equal-brightness multi-NV model. Uses GPU coarse+refined fit and
            ordinary BIC selection, unless force_nvs is set.

        "strict_auto" or "gpu_refined"
            Equal-brightness multi-NV model with GPU refinement and strict
            physical model selection. This is the recommended pure-GPU path.
    """
    mode = str(model_mode).lower()

    if mode == "bimodal":
        fits, debug = fit_bimodal_histograms_gpu_batch(
            counts_batch,
            prob_dist=prob_dist,
            config=bimodal_config,
            return_debug=True,
        )

        results = []
        for fit in fits:
            popt, pcov, red_chi_sq = fit
            if popt is None:
                results.append({"ok": False, "reason": "fit_failed"})
                continue

            dark_weight, dark_rate, bright_rate = [float(v) for v in popt]
            rate0 = min(dark_rate, bright_rate)
            delta = abs(bright_rate - dark_rate)
            p_minus = 1.0 - dark_weight if bright_rate >= dark_rate else dark_weight
            results.append(
                {
                    "ok": True,
                    "model": "bimodal",
                    "n_nvs": 1,
                    "popt": np.asarray([p_minus, 0.0, rate0, delta], dtype=float),
                    "bimodal_popt": np.asarray(popt, dtype=float),
                    "pcov": pcov,
                    "red_chi_sq": float(red_chi_sq),
                    "nll": np.nan,
                    "bic": float(red_chi_sq) if np.isfinite(red_chi_sq) else np.nan,
                    "k_free": 3,
                    "strict_reason": "bimodal_mode",
                }
            )

        return (results, debug) if return_debug else results

    if mode not in {"auto", "multimode", "strict_auto", "gpu_refined", "hierarchical"}:
        raise ValueError(
            "model_mode must be 'auto', 'bimodal', 'multimode', 'strict_auto', 'gpu_refined', or 'hierarchical'."
        )

    if multimode_config is None:
        multimode_config = GpuMultimodeFitConfig(max_nvs=int(max_nvs))

    strict_selection = mode in {"strict_auto", "gpu_refined"}
    
    if mode == "hierarchical":
        return fit_binomial_multinv_histograms_gpu_hierarchical_batch(
            counts_batch,
            prob_dist=prob_dist,
            config=multimode_config,
            max_nvs=max_nvs,
            return_debug=return_debug,
        )

    return fit_binomial_multinv_histograms_gpu_batch(
        counts_batch,
        prob_dist=prob_dist,
        config=multimode_config,
        max_nvs=max_nvs,
        force_nvs=force_nvs,
        strict_selection=strict_selection,
        return_debug=return_debug,
    )


def determine_thresholds_any_minus_gpu(
    popt_arr,
    n_nvs_arr,
    prob_dist="COMPOUND_POISSON",
    x_max=None,
    config: GpuMultimodeFitConfig | None = None,
    chunk_size: int = 1024,
):
    """
    GPU threshold search for multi-NV equal-brightness fits.

    popt_arr[..., :] must be [p_minus, bg, rate0, delta].
    n_nvs_arr gives the selected number of NVs for each fit.

    Returned threshold classifies:
        counts <= threshold  -> all NV0
        counts > threshold   -> any NV-
    """
    if config is None:
        config = GpuMultimodeFitConfig()

    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy GPU backend is not available.")

    prob_name = _prob_dist_name(prob_dist)
    popt = np.asarray(popt_arr, dtype=float)
    n_nvs = np.asarray(n_nvs_arr, dtype=int)
    original_shape = popt.shape[:-1]

    flat = popt.reshape(-1, 4)
    n_flat = n_nvs.reshape(-1)
    valid = np.all(np.isfinite(flat), axis=1) & (n_flat > 0)

    threshold = np.full(flat.shape[0], np.nan, dtype=float)
    fidelity = np.full(flat.shape[0], np.nan, dtype=float)

    if x_max is None:
        rates = []
        for params, n_val in zip(flat[valid], n_flat[valid]):
            p_minus, bg, rate0, delta = [float(v) for v in params]
            ks = np.arange(int(n_val) + 1, dtype=float)
            rates.extend((bg + int(n_val) * rate0 + ks * delta).tolist())
        x_max = int(max(np.nanmax(rates) * 1.5 + 20, 20)) if rates else 20

    x_max = int(max(x_max, 1))
    x_gpu = cp.arange(0, x_max + 1, dtype=cp.float64)
    threshold_candidates = np.arange(-0.5, x_max + 0.5 + 1e-12, 1.0)
    valid_inds = np.where(valid)[0]

    for start in range(0, valid_inds.size, int(chunk_size)):
        stop = min(start + int(chunk_size), valid_inds.size)
        inds = valid_inds[start:stop]
        params_cpu = flat[inds]
        n_cpu = n_flat[inds]

        for n_val in np.unique(n_cpu):
            group_mask = n_cpu == n_val
            group_inds = inds[group_mask]
            params_group = params_cpu[group_mask]
            n_int = int(n_val)

            cand_gpu = cp.asarray(params_group, dtype=cp.float64)
            p_minus = cp.clip(cand_gpu[:, 0], config.eps, 1.0 - config.eps)
            bg = cp.maximum(cand_gpu[:, 1], 0.0)
            rate0 = cp.maximum(cand_gpu[:, 2], config.eps)
            delta = cp.maximum(cand_gpu[:, 3], 0.0)

            ks_cpu = np.arange(n_int + 1, dtype=np.float64)
            ks_gpu = cp.asarray(ks_cpu, dtype=cp.float64)
            coeff_gpu = cp.asarray(_binom_coeffs(n_int), dtype=cp.float64)

            weights = (
                coeff_gpu[None, :]
                * (p_minus[:, None] ** ks_gpu[None, :])
                * ((1.0 - p_minus[:, None]) ** (n_int - ks_gpu[None, :]))
            )
            weights = weights / cp.maximum(cp.sum(weights, axis=1, keepdims=True), config.eps)

            rates = bg[:, None] + n_int * rate0[:, None] + ks_gpu[None, :] * delta[:, None]
            rates_flat = rates.reshape(-1)
            rates_cpu = cp.asnumpy(rates_flat)

            pdf_flat = _single_mode_pdf_table(prob_name, x_gpu, rates_flat, rates_cpu, config)
            pdf = pdf_flat.reshape(params_group.shape[0], n_int + 1, x_max + 1)
            cdf = cp.clip(cp.cumsum(pdf, axis=2), 0.0, 1.0)
            zeros = cp.zeros((params_group.shape[0], n_int + 1, 1), dtype=cp.float64)
            cdf_ext = cp.concatenate([zeros, cdf], axis=2)

            w0 = weights[:, 0]
            wrest = cp.maximum(1.0 - w0, config.eps)
            cdf0 = cdf_ext[:, 0, :]

            if n_int == 1:
                cdf_rest = cdf_ext[:, 1, :]
            else:
                cdf_rest = cp.sum(weights[:, 1:, None] * cdf_ext[:, 1:, :], axis=1) / wrest[:, None]

            fid = w0[:, None] * cdf0 + wrest[:, None] * (1.0 - cdf_rest)
            best = cp.argmax(fid, axis=1)
            best_fid = fid[cp.arange(fid.shape[0]), best]

            threshold[group_inds] = threshold_candidates[cp.asnumpy(best)]
            fidelity[group_inds] = cp.asnumpy(best_fid)

    return threshold.reshape(original_shape), fidelity.reshape(original_shape)


def summarize_gpu():
    """Small diagnostic helper."""
    if not GPU_AVAILABLE:
        return {"gpu_available": False}

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    return {
        "gpu_available": True,
        "gpu_name": name,
        "cupy_version": cp.__version__,
    }


if __name__ == "__main__":
    print(summarize_gpu())

    rng = np.random.default_rng(0)
    test_counts = []
    for _ in range(32):
        dark = rng.poisson(8, size=250)
        bright = rng.poisson(22, size=750)
        counts = np.concatenate([dark, bright])
        rng.shuffle(counts)
        test_counts.append(counts)

    cfg = GpuFitConfig(num_ratio=15, num_rate=48)
    out, info = fit_bimodal_histograms_gpu_batch(
        test_counts,
        prob_dist="POISSON",
        config=cfg,
        return_debug=True,
    )
    print(info)
    print(out[0])

    mm_cfg = GpuMultimodeFitConfig(
        max_nvs=3,
        num_p=9,
        num_bg=4,
        num_rate0=10,
        num_delta=10,
        use_refinement=True,
        refine_iters=3,
    )
    mm_out, mm_info = fit_charge_histograms_gpu_batch(
        test_counts,
        prob_dist="POISSON",
        model_mode="strict_auto",
        multimode_config=mm_cfg,
        return_debug=True,
    )
    print(mm_info)
    print(mm_out[0])
