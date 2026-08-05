# -*- coding: utf-8 -*-
"""
GPU reference-only multi-NV charge-state histogram analysis.

This is the GPU counterpart of the CPU reference-only script.

Experiment layout expected:
    counts[0] = signal / with ionization pulse
    counts[1] = reference / without ionization pulse

Important physical rule:
    All fitted parameters come from the reference/no-ionization branch only.
    The signal/ionization branch is plotted only for visual comparison.

GPU path:
    - Fits all reference histograms in one batched CuPy call using the
      equal-brightness multi-NV binomial model.
    - Selects N = 1..max_nvs_per_position by BIC, unless force_nvs is set.
    - Computes any-NV- thresholds on GPU.
    - Computes multi-class thresholds from the saved GPU fit parameters.

Output compatibility:
    raw_data["charge_hist_multinv_binomial"] is filled with the same style of
    analysis dictionary as the CPU script, so the plotting/feedback functions can
    be used in the same way.

Created: July 2026
@author: Saroj Chand
"""

from __future__ import annotations

import os
import sys
import math
import traceback
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.special import gammaln as cp_gammaln
except Exception:
    cp = None
    cp_gammaln = None

# Compatibility patch for old labrad with newer NumPy.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from analysis import bimodal_histogram
from analysis.bimodal_histogram import ProbDist
from analysis.sc_gpu_bimodal_fitting import (
    GpuMultimodeFitConfig,
    determine_thresholds_any_minus_gpu,
    fit_charge_histograms_gpu_batch,
    summarize_gpu,
)

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield
from utils.constants import VirtualLaserKey


# =============================================================================
# Constants / compact plotting style
# =============================================================================

SCATTER_FIGSIZE = (6.5, 5)
SPATIAL_FIGSIZE = (6.5, 5)
BAR_FIGSIZE = (6.5, 5)
HIST_FIGSIZE = (6.5, 5)

POINT_SIZE = 10
POINT_ALPHA = 0.75

METRIC_INFO = {
    "n_nvs_est": (
        "Estimated NVs per pillar",
        "Estimated total number of NVs contributing to this pillar/spot.",
    ),
    "threshold_any": (
        "Any-NV$^{-}$ threshold",
        "Binary threshold: counts above this mean at least one NV is NV$^{-}$.",
    ),
    "readout_fidelity_any": (
        "Binary fidelity",
        "Readout fidelity for distinguishing k=0 from k≥1 NV$^{-}$.",
    ),
    "fidelity_multiclass": (
        "Multi-class fidelity",
        "Readout fidelity for distinguishing k=0,1,...,N NV$^{-}$.",
    ),
    "prep_fidelity_any_ref": (
        "P(any NV$^{-}$), fit",
        "From fitted weights: 1 - P(k=0).",
    ),
    "ref_p_any_minus": (
        "P(any NV$^{-}$), classified",
        "Fraction of reference shots classified as having at least one NV$^{-}$.",
    ),
    "ref_mean_num_minus": (
        "Mean k",
        "Mean estimated number of NV$^{-}$ in the reference branch.",
    ),
    "p_minus": (
        "p_minus",
        "Fitted single-NV probability of being NV$^{-}$ in the reference branch.",
    ),
    "rate0": (
        "rate0",
        "Fitted per-NV NV$^{0}$ readout counts.",
    ),
    "delta": (
        "delta",
        "Extra counts when one NV changes from NV$^{0}$ to NV$^{-}$.",
    ),
    "red_chi_sq": (
        "Reduced $\\chi^2$",
        "Goodness of fit; smaller is generally better.",
    ),
    "bic": (
        "BIC",
        "Model-selection score; smaller is better.",
    ),
}


# =============================================================================
# Small helpers
# =============================================================================


def safe_float(x):
    """Convert to float safely."""
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


def make_json_safe(obj):
    """Convert analysis objects to JSON/orjson-safe objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _as_array_float(x):
    return np.asarray(x, dtype=float)


def _binom_weights(n_nvs: int, p_minus: float) -> np.ndarray:
    n_nvs = int(max(n_nvs, 1))
    p_minus = float(np.clip(p_minus, 0.0, 1.0))
    ks = np.arange(n_nvs + 1, dtype=float)
    coeff = np.asarray([math.comb(n_nvs, int(k)) for k in ks], dtype=float)
    w = coeff * (p_minus**ks) * ((1.0 - p_minus) ** (n_nvs - ks))
    s = float(np.sum(w))
    if s <= 0 or not np.isfinite(s):
        out = np.zeros(n_nvs + 1, dtype=float)
        out[0] = 1.0
        return out
    return w / s


def _fit_equal_model_pdf_tables(prob_dist: ProbDist, n_nvs: int, popt, x_max: int):
    """
    Build per-class PDF/CDF arrays for saved GPU equal-brightness fit.

    popt = [p_minus, bg, rate0, delta]
    lambda_k = bg + N*rate0 + k*delta
    """
    p_minus, bg, rate0, delta = [float(v) for v in np.asarray(popt, dtype=float)]
    n_nvs = int(max(n_nvs, 1))
    x_max = int(max(x_max, 1))

    x_vals = np.arange(0, x_max + 1, dtype=float)
    weights = _binom_weights(n_nvs, p_minus)
    single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)

    pdf = np.zeros((n_nvs + 1, x_vals.size), dtype=float)
    base = max(bg, 0.0) + n_nvs * max(rate0, 1e-12)
    delta = max(delta, 0.0)

    for k in range(n_nvs + 1):
        lam_k = base + k * delta
        vals = np.asarray(single_pdf(x_vals, lam_k), dtype=float)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        vals = np.clip(vals, 0.0, None)
        s = float(np.sum(vals))
        if s > 0 and np.isfinite(s):
            vals = vals / s
        pdf[k, :] = vals

    cdf = np.cumsum(pdf, axis=1)
    cdf = np.clip(cdf, 0.0, 1.0)
    zeros = np.zeros((n_nvs + 1, 1), dtype=float)
    cdf_ext = np.concatenate([zeros, cdf], axis=1)
    thresholds = np.arange(-0.5, x_max + 0.5 + 1e-12, 1.0)

    return x_vals, weights, pdf, cdf_ext, thresholds


def determine_multithreshold_equal_multinv(
    popt,
    prob_dist: ProbDist,
    n_nvs: int,
    x_max: int,
    ret_fidelity: bool = True,
):
    """
    Multi-class Bayes thresholds for classes k=0..N.

    Returns thresholds of length N, separating k=0|1, k=1|2, ... k=N-1|N.
    """
    if popt is None:
        return (None, None) if ret_fidelity else None

    try:
        x_vals, weights, pdf, cdf_ext, T = _fit_equal_model_pdf_tables(
            prob_dist,
            n_nvs,
            popt,
            x_max,
        )
    except Exception:
        return (None, None) if ret_fidelity else None

    K = int(n_nvs) + 1
    L = T.size

    if K == 2:
        fid = weights[0] * cdf_ext[0] + weights[1] * (1.0 - cdf_ext[1])
        best = int(np.nanargmax(fid))
        out_thresholds = [float(T[best])]
        out_fid = float(fid[best])
        return (out_thresholds, out_fid) if ret_fidelity else out_thresholds

    # Dynamic programming over ordered thresholds.
    # dp[i, j] means classes 0..i are assigned using i+1-th threshold at j.
    dp = np.full((K - 1, L), -np.inf, dtype=float)
    back = np.full((K - 1, L), -1, dtype=int)

    dp[0, :] = weights[0] * cdf_ext[0, :]

    for class_ind in range(1, K - 1):
        for j in range(L):
            if j == 0:
                continue
            prev = np.arange(0, j, dtype=int)
            vals = dp[class_ind - 1, prev] + weights[class_ind] * (
                cdf_ext[class_ind, j] - cdf_ext[class_ind, prev]
            )
            best_prev_local = int(np.nanargmax(vals))
            dp[class_ind, j] = float(vals[best_prev_local])
            back[class_ind, j] = int(prev[best_prev_local])

    best_total = -np.inf
    best_j = 0
    for j in range(L):
        val = dp[K - 2, j] + weights[K - 1] * (1.0 - cdf_ext[K - 1, j])
        if val > best_total:
            best_total = float(val)
            best_j = int(j)

    idxs = [best_j]
    for class_ind in range(K - 2, 0, -1):
        best_j = int(back[class_ind, best_j])
        idxs.append(best_j)
    idxs = list(reversed(idxs))

    out_thresholds = [float(T[i]) for i in idxs]
    return (out_thresholds, float(best_total)) if ret_fidelity else out_thresholds


def classify_multinv_counts(counts, thresholds):
    """
    Classify each shot into k = number of NV- at this pillar.

    thresholds length = N, separating k=0|1|...|N.
    """
    counts = np.asarray(counts, dtype=float).flatten()
    if thresholds is None:
        return np.full(counts.shape, -1, dtype=int)

    thresholds = np.asarray(thresholds, dtype=float).flatten()
    if thresholds.size == 0 or np.any(~np.isfinite(thresholds)):
        return np.full(counts.shape, -1, dtype=int)

    return np.searchsorted(thresholds, counts, side="right").astype(int)


def summarize_ref_classification(ref_counts, threshold_any, thresholds_multiclass, n_nvs):
    """Analyze only the reference/no-ionization branch."""
    ref_counts = np.asarray(ref_counts, dtype=float).flatten()
    n_nvs = int(n_nvs)

    if threshold_any is None or not np.isfinite(threshold_any):
        p_any_minus = np.nan
    else:
        p_any_minus = float(np.mean(ref_counts > threshold_any))

    k_est = classify_multinv_counts(ref_counts, thresholds_multiclass)

    prob_k = np.zeros(n_nvs + 1, dtype=float)
    for k in range(n_nvs + 1):
        prob_k[k] = float(np.mean(k_est == k))

    good = k_est >= 0
    mean_num_minus = float(np.mean(k_est[good])) if np.any(good) else np.nan

    return {
        "p_any_minus": p_any_minus,
        "mean_num_minus": mean_num_minus,
        "prob_k": prob_k,
        "k_est": k_est,
    }


def feedback_classify_count(counts, threshold_any, thresholds_multiclass, n_nvs_est):
    """Helper for future feedback experiments."""
    counts = np.asarray(counts, dtype=float)
    is_any_nv_minus = counts > threshold_any
    k_est = np.searchsorted(thresholds_multiclass, counts, side="right")
    is_fully_initialized = k_est >= int(n_nvs_est)
    return {
        "is_any_nv_minus": is_any_nv_minus,
        "k_est": k_est,
        "is_fully_initialized": is_fully_initialized,
    }


# =============================================================================
# Histogram plotting
# =============================================================================


def plot_histograms(sig_counts_list, ref_counts_list, no_title=True, ax=None, density=False):
    """
    Plot signal/reference histograms.

    Red: with ionization pulse, visual only.
    Green: without ionization pulse, used for fitting.
    """
    readout_ms = None
    try:
        laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
        laser_dict = tb.get_virtual_laser_dict(laser_key)
        readout = laser_dict.get("duration", None)
        if readout is not None:
            readout_ms = int(readout / 1e6)
    except Exception:
        readout_ms = None

    labels = ["With ionization pulse", "Without ionization pulse"]
    colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
    counts_lists = [sig_counts_list, ref_counts_list]

    if ax is None:
        fig, ax = plt.subplots(figsize=HIST_FIGSIZE)
    else:
        fig = None

    if not no_title:
        if readout_ms is None:
            ax.set_title("Charge-state histograms")
        else:
            ax.set_title(f"Charge-state histograms, readout = {readout_ms} ms")

    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Probability" if density else "Number of occurrences")

    for ind in range(2):
        counts_list = counts_lists[ind]
        if counts_list is None or len(counts_list) == 0:
            continue
        kpl.histogram(
            ax,
            counts_list,
            color=colors[ind],
            density=density,
            label=labels[ind],
        )

    ax.set_xlim(-0.5, None)
    ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=7)

    if fig is not None:
        return fig


# =============================================================================
# Main GPU reference-only multi-NV analysis
# =============================================================================


def _fit_results_to_arrays(gpu_fit_results, num_positions):
    ok_arr = np.full(num_positions, False, dtype=bool)
    n_nvs_est_arr = np.full(num_positions, np.nan)
    p_minus_arr = np.full(num_positions, np.nan)
    bg_arr = np.full(num_positions, np.nan)
    rate0_arr = np.full(num_positions, np.nan)
    delta_arr = np.full(num_positions, np.nan)
    red_chi_sq_arr = np.full(num_positions, np.nan)
    bic_arr = np.full(num_positions, np.nan)
    nll_arr = np.full(num_positions, np.nan)
    model_list = [None for _ in range(num_positions)]
    fit_params_arr = np.full((num_positions, 4), np.nan, dtype=float)

    for ind, fit in enumerate(gpu_fit_results):
        if not isinstance(fit, dict) or not fit.get("ok", False):
            continue
        popt = np.asarray(fit.get("popt", []), dtype=float)
        if popt.size != 4 or not np.all(np.isfinite(popt)):
            continue
        ok_arr[ind] = True
        n_nvs_est_arr[ind] = int(fit.get("n_nvs", 1))
        fit_params_arr[ind, :] = popt
        p_minus_arr[ind], bg_arr[ind], rate0_arr[ind], delta_arr[ind] = popt
        red_chi_sq_arr[ind] = safe_float(fit.get("red_chi_sq", np.nan))
        bic_arr[ind] = safe_float(fit.get("bic", np.nan))
        nll_arr[ind] = safe_float(fit.get("nll", np.nan))
        model_list[ind] = fit.get("model", None)

    return {
        "ok": ok_arr,
        "n_nvs_est": n_nvs_est_arr,
        "fit_params": fit_params_arr,
        "p_minus": p_minus_arr,
        "bg": bg_arr,
        "rate0": rate0_arr,
        "delta": delta_arr,
        "red_chi_sq": red_chi_sq_arr,
        "bic": bic_arr,
        "nll": nll_arr,
        "model": model_list,
    }



def _adjacent_dprimes_for_equal_fit(popt, n_nvs):
    """
    Adjacent-peak separability for the equal-brightness multi-NV model.

    lambda_k = bg + N*rate0 + k*delta
    d'_k = delta / sqrt(lambda_k + lambda_{k+1})
    """
    try:
        p_minus, bg, rate0, delta = [float(v) for v in np.asarray(popt, dtype=float)]
        n_nvs = int(max(n_nvs, 1))
        base = max(bg, 0.0) + n_nvs * max(rate0, 1e-12)
        delta = max(delta, 0.0)
        out = []
        for k in range(n_nvs):
            lam0 = base + k * delta
            lam1 = base + (k + 1) * delta
            out.append(float(delta / np.sqrt(max(lam0 + lam1, 1e-12))))
        return np.asarray(out, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _strict_candidate_diagnostics(
    fit,
    ref_counts,
    strict_min_mode_weight=0.05,
    strict_min_mode_shots=75,
    strict_min_adjacent_dprime=1.5,
    strict_require_all_modes=True,
):
    """
    Physical checks for whether an N>1 fit is believable.

    The goal is to avoid fake intermediate modes. A higher-N model is only
    allowed when its components are visible in the data and adjacent peaks are
    separated enough to be meaningful.
    """
    diag = {
        "ok": False,
        "reasons": [],
        "weights": None,
        "expected_shots": None,
        "adjacent_dprime": None,
        "min_checked_weight": np.nan,
        "min_checked_shots": np.nan,
        "min_adjacent_dprime": np.nan,
    }

    if not isinstance(fit, dict) or not fit.get("ok", False):
        diag["reasons"].append("fit_not_ok")
        return diag

    n_nvs = int(fit.get("n_nvs", 1))
    popt = np.asarray(fit.get("popt", []), dtype=float)
    if popt.size != 4 or not np.all(np.isfinite(popt)):
        diag["reasons"].append("invalid_fit_params")
        return diag

    p_minus = float(np.clip(popt[0], 0.0, 1.0))
    weights = _binom_weights(n_nvs, p_minus)
    nshots = int(np.asarray(ref_counts, dtype=float).size)
    expected_shots = weights * max(nshots, 1)
    dprimes = _adjacent_dprimes_for_equal_fit(popt, n_nvs)

    diag["weights"] = weights.tolist()
    diag["expected_shots"] = expected_shots.tolist()
    diag["adjacent_dprime"] = dprimes.tolist()

    if n_nvs <= 1:
        diag["ok"] = True
        return diag

    if strict_require_all_modes:
        checked = np.arange(0, n_nvs + 1, dtype=int)
    else:
        # Less conservative: only require the intermediate modes to be visible.
        checked = np.arange(1, n_nvs, dtype=int)
        if checked.size == 0:
            checked = np.arange(0, n_nvs + 1, dtype=int)

    checked_weights = weights[checked]
    checked_shots = expected_shots[checked]

    diag["min_checked_weight"] = float(np.nanmin(checked_weights))
    diag["min_checked_shots"] = float(np.nanmin(checked_shots))

    if np.any(checked_weights < float(strict_min_mode_weight)):
        diag["reasons"].append(
            f"mode_weight_too_small_min={diag['min_checked_weight']:.4g}"
        )

    if np.any(checked_shots < float(strict_min_mode_shots)):
        diag["reasons"].append(
            f"mode_expected_shots_too_small_min={diag['min_checked_shots']:.4g}"
        )

    if dprimes.size == 0 or not np.all(np.isfinite(dprimes)):
        diag["reasons"].append("invalid_adjacent_dprime")
    else:
        diag["min_adjacent_dprime"] = float(np.nanmin(dprimes))
        if np.nanmin(dprimes) < float(strict_min_adjacent_dprime):
            diag["reasons"].append(
                f"adjacent_dprime_too_small_min={diag['min_adjacent_dprime']:.4g}"
            )

    diag["ok"] = len(diag["reasons"]) == 0
    return diag


def _adjusted_bic_for_strict(fit, strict_extra_nv_penalty=60.0):
    if not isinstance(fit, dict) or not fit.get("ok", False):
        return np.inf
    bic = safe_float(fit.get("bic", np.nan))
    if not np.isfinite(bic):
        return np.inf
    n_nvs = int(fit.get("n_nvs", 1))
    return float(bic + float(strict_extra_nv_penalty) * max(n_nvs - 1, 0))



# =============================================================================
# Pure-GPU refined multimodal fitting
# =============================================================================


def _require_cupy_for_refined_gpu():
    if cp is None or cp_gammaln is None:
        raise RuntimeError(
            "model_mode='gpu_refined' requires CuPy and cupyx.scipy.special.gammaln."
        )


def _gpu_poisson_pdf_rates_x(rates, x_vals):
    """
    Vectorized Poisson PMF on GPU.

    rates: shape (C,)
    x_vals: shape (X,)
    returns: shape (C, X)
    """
    rates = cp.asarray(rates, dtype=cp.float64).reshape(-1, 1)
    x_vals = cp.asarray(x_vals, dtype=cp.float64).reshape(1, -1)

    safe_rates = cp.maximum(rates, 1e-300)
    xlogr = cp.where(x_vals == 0, 0.0, x_vals * cp.log(safe_rates))
    logp = xlogr - safe_rates - cp_gammaln(x_vals + 1.0)
    return cp.exp(logp)


def _gpu_compound_poisson_pdf_rates_x(
    rates,
    x_vals,
    candidate_chunk_size=2048,
    nsig=8.0,
    min_upper=25,
    max_upper=5000,
):
    """
    Compound Poisson PMF used by your CPU code:

        P(z | rate) = sum_k Pois(z | k) Pois(k | rate)

    This GPU version evaluates the k-sum as a matrix product per chunk:

        [Pois(k | rate)] @ [Pois(z | k)]

    rates: shape (C,)
    x_vals: shape (X,)
    returns: shape (C, X)
    """
    rates = cp.asarray(rates, dtype=cp.float64).reshape(-1)
    x_vals = cp.asarray(x_vals, dtype=cp.float64).reshape(-1)

    C = int(rates.size)
    X = int(x_vals.size)
    out = cp.empty((C, X), dtype=cp.float64)

    for start in range(0, C, int(candidate_chunk_size)):
        stop = min(start + int(candidate_chunk_size), C)
        r = rates[start:stop]

        rmax = float(cp.asnumpy(cp.nanmax(r))) if r.size else 0.0
        if not np.isfinite(rmax) or rmax < 0:
            rmax = 0.0

        upper = int(np.ceil(rmax + float(nsig) * np.sqrt(max(rmax, 0.0))))
        upper = int(min(max(upper, int(min_upper)), int(max_upper)))

        k_vals = cp.arange(0, upper + 1, dtype=cp.float64)

        # P(k | rate): (Cc, K)
        pk_given_r = _gpu_poisson_pdf_rates_x(r, k_vals)

        # P(z | k): (K, X)
        pz_given_k = _gpu_poisson_pdf_rates_x(k_vals, x_vals)

        vals = pk_given_r @ pz_given_k
        vals = cp.clip(vals, 1e-300, None)
        out[start:stop, :] = vals

    return out


def _gpu_single_mode_pdf_rates_x(prob_dist_name, rates, x_vals, candidate_chunk_size=2048):
    prob_dist_name = str(prob_dist_name)

    if prob_dist_name == "POISSON":
        return _gpu_poisson_pdf_rates_x(rates, x_vals)

    if prob_dist_name == "COMPOUND_POISSON":
        return _gpu_compound_poisson_pdf_rates_x(
            rates,
            x_vals,
            candidate_chunk_size=candidate_chunk_size,
        )

    raise NotImplementedError(
        "gpu_refined currently supports ProbDist.POISSON and "
        "ProbDist.COMPOUND_POISSON. Use COMPOUND_POISSON for your charge data."
    )


def _build_integer_histogram_matrix_for_gpu(ref_counts_lists):
    """
    Build integer histograms with the same trimming spirit as the CPU fitter.

    Returns:
        hist_counts: (P, X) float64 raw bin counts
        x_vals:      (X,) integer count values
        nshots:      (P,) number of shots after trimming
        qstats:      dict of quantiles for initialization/bounds
    """
    cleaned = []
    q02 = []
    q15 = []
    q50 = []
    q65 = []
    q98 = []
    means = []
    nshots = []
    max_count = 1

    for arr in ref_counts_lists:
        x = np.asarray(arr, dtype=float).ravel()
        x = x[np.isfinite(x)]
        x = x[x >= 0]

        if x.size > 0:
            med = np.median(x)
            std = np.std(x)
            if np.isfinite(std) and std > 0:
                x = x[x < med + 10.0 * std]

        if x.size < 1:
            x = np.asarray([0.0], dtype=float)

        xi = np.rint(np.clip(x, 0, None)).astype(int)
        cleaned.append(xi)
        max_count = max(max_count, int(np.max(xi)))

        q02.append(float(np.quantile(x, 0.02)))
        q15.append(float(np.quantile(x, 0.15)))
        q50.append(float(np.quantile(x, 0.50)))
        q65.append(float(np.quantile(x, 0.65)))
        q98.append(float(np.quantile(x, 0.98)))
        means.append(float(np.mean(x)))
        nshots.append(int(x.size))

    P = len(cleaned)
    hist_counts = np.zeros((P, max_count + 1), dtype=np.float64)

    for ind, xi in enumerate(cleaned):
        bc = np.bincount(xi, minlength=max_count + 1).astype(np.float64)
        hist_counts[ind, :] = bc[: max_count + 1]

    x_vals = np.arange(max_count + 1, dtype=np.float64)

    qstats = {
        "q02": np.asarray(q02, dtype=np.float64),
        "q15": np.asarray(q15, dtype=np.float64),
        "q50": np.asarray(q50, dtype=np.float64),
        "q65": np.asarray(q65, dtype=np.float64),
        "q98": np.asarray(q98, dtype=np.float64),
        "mean": np.asarray(means, dtype=np.float64),
    }

    return hist_counts, x_vals, np.asarray(nshots, dtype=np.float64), qstats


def _gpu_binomial_weights(n_nvs, p_minus):
    p_minus = cp.clip(cp.asarray(p_minus, dtype=cp.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    ks_np = np.arange(int(n_nvs) + 1, dtype=np.float64)
    coeff_np = np.asarray([math.comb(int(n_nvs), int(k)) for k in ks_np], dtype=np.float64)
    ks = cp.asarray(ks_np, dtype=cp.float64).reshape(1, -1)
    coeff = cp.asarray(coeff_np, dtype=cp.float64).reshape(1, -1)
    p = p_minus.reshape(-1, 1)
    w = coeff * (p ** ks) * ((1.0 - p) ** (int(n_nvs) - ks))
    w = w / cp.maximum(cp.sum(w, axis=1, keepdims=True), 1e-300)
    return w


def _gpu_eval_nll_fixed_n(
    params_gpu,
    hist_gpu,
    pillar_idx_gpu,
    x_vals_gpu,
    n_nvs,
    prob_dist_name,
    pdf_candidate_chunk_size=2048,
):
    """
    Evaluate multinomial negative log likelihood for equal-brightness N-NV model.

        lambda_k = bg + N*rate0 + k*delta

    params_gpu:     (C, 4) columns [p_minus, bg, rate0, delta]
    hist_gpu:       (P, X)
    pillar_idx_gpu: (C,)
    x_vals_gpu:     (X,)
    returns:        (C,)
    """
    params_gpu = cp.asarray(params_gpu, dtype=cp.float64)
    C = int(params_gpu.shape[0])
    X = int(x_vals_gpu.size)

    out = cp.empty(C, dtype=cp.float64)
    chunk_size = int(pdf_candidate_chunk_size)
    ks = cp.arange(int(n_nvs) + 1, dtype=cp.float64)

    for start in range(0, C, chunk_size):
        stop = min(start + chunk_size, C)
        par = params_gpu[start:stop, :]
        p_minus = cp.clip(par[:, 0], 1e-6, 1.0 - 1e-6)
        bg = cp.clip(par[:, 1], 0.0, None)
        rate0 = cp.clip(par[:, 2], 1e-9, None)
        delta = cp.clip(par[:, 3], 0.0, None)

        weights = _gpu_binomial_weights(int(n_nvs), p_minus)
        lambdas = bg[:, None] + int(n_nvs) * rate0[:, None] + ks[None, :] * delta[:, None]

        mix = cp.zeros((stop - start, X), dtype=cp.float64)

        for k in range(int(n_nvs) + 1):
            pdf_k = _gpu_single_mode_pdf_rates_x(
                prob_dist_name,
                lambdas[:, k],
                x_vals_gpu,
                candidate_chunk_size=pdf_candidate_chunk_size,
            )
            mix += weights[:, k:k + 1] * pdf_k

        mix = cp.clip(mix, 1e-300, None)
        mix = mix / cp.maximum(cp.sum(mix, axis=1, keepdims=True), 1e-300)

        h = hist_gpu[pillar_idx_gpu[start:stop], :]
        out[start:stop] = -cp.sum(h * cp.log(mix), axis=1)

    return out


def _make_refined_gpu_initial_candidates(qstats, n_nvs, n_starts=6, seed=0):
    """
    CPU-side lightweight initialization only. Optimization is GPU-side.
    """
    rng = np.random.default_rng(seed)

    q02 = qstats["q02"]
    q15 = qstats["q15"]
    q98 = qstats["q98"]
    mean = qstats["mean"]

    P = q02.size
    S = int(max(n_starts, 1))
    N = int(n_nvs)

    bg0 = np.maximum(0.0, 0.25 * q02)
    rate0 = np.maximum(1e-3, (q15 - bg0) / max(N, 1))
    delta0 = np.maximum(1e-3, (q98 - q15) / max(N, 1))
    p0 = (mean - bg0 - N * rate0) / np.maximum(N * delta0, 1e-9)
    p0 = np.clip(p0, 0.05, 0.95)

    params = np.zeros((P, S, 4), dtype=np.float64)

    # Deterministic first few starts plus random jitter for extra starts.
    base_jitters = [
        (1.00, 1.00, 1.00, 1.00),
        (0.75, 0.75, 1.10, 0.80),
        (1.25, 1.25, 0.90, 1.20),
        (1.00, 0.50, 1.25, 1.50),
        (0.60, 1.50, 0.75, 0.65),
        (1.40, 1.00, 1.00, 1.80),
    ]

    for s in range(S):
        if s < len(base_jitters):
            pf, bgf, r0f, df = base_jitters[s]
            p = np.clip(p0 * pf, 0.02, 0.98)
            bg = bg0 * bgf
            r0 = rate0 * r0f
            d = delta0 * df
        else:
            p = np.clip(p0 + 0.15 * rng.standard_normal(P), 0.02, 0.98)
            bg = bg0 * np.exp(0.35 * rng.standard_normal(P))
            r0 = rate0 * np.exp(0.30 * rng.standard_normal(P))
            d = delta0 * np.exp(0.40 * rng.standard_normal(P))

        params[:, s, 0] = p
        params[:, s, 1] = bg
        params[:, s, 2] = np.maximum(r0, 1e-6)
        params[:, s, 3] = np.maximum(d, 0.0)

    # Candidate-specific bounds.
    lower = np.zeros_like(params)
    upper = np.zeros_like(params)

    lower[:, :, 0] = 1e-4
    upper[:, :, 0] = 1.0 - 1e-4

    lower[:, :, 1] = 0.0
    upper[:, :, 1] = np.maximum(1.0, q15[:, None])

    lower[:, :, 2] = 1e-6
    upper[:, :, 2] = np.maximum(1e-3, q98[:, None] / max(N, 1))

    lower[:, :, 3] = 0.0
    upper[:, :, 3] = np.maximum(1e-3, q98[:, None] - q02[:, None])

    step = np.zeros_like(params)
    step[:, :, 0] = 0.08
    step[:, :, 1] = np.maximum(1.0, 0.20 * q15[:, None])
    step[:, :, 2] = np.maximum(1.0, 0.25 * rate0[:, None])
    step[:, :, 3] = np.maximum(1.0, 0.25 * delta0[:, None])

    return params, lower, upper, step


def _gpu_coordinate_refine_fixed_n(
    hist_gpu,
    x_vals_gpu,
    qstats,
    n_nvs,
    prob_dist_name,
    n_starts=6,
    n_refine_iters=8,
    step_shrink=0.55,
    seed=0,
    pdf_candidate_chunk_size=2048,
):
    """
    Multi-start coordinate refinement on GPU for one fixed N.
    """
    P = int(hist_gpu.shape[0])
    S = int(max(n_starts, 1))

    params0, lower0, upper0, step0 = _make_refined_gpu_initial_candidates(
        qstats,
        n_nvs=int(n_nvs),
        n_starts=S,
        seed=seed,
    )

    params = cp.asarray(params0.reshape(P * S, 4), dtype=cp.float64)
    lower = cp.asarray(lower0.reshape(P * S, 4), dtype=cp.float64)
    upper = cp.asarray(upper0.reshape(P * S, 4), dtype=cp.float64)
    step = cp.asarray(step0.reshape(P * S, 4), dtype=cp.float64)

    pillar_idx = np.repeat(np.arange(P, dtype=np.int64), S)
    pillar_idx_gpu = cp.asarray(pillar_idx, dtype=cp.int64)

    nll = _gpu_eval_nll_fixed_n(
        params,
        hist_gpu,
        pillar_idx_gpu,
        x_vals_gpu,
        int(n_nvs),
        prob_dist_name,
        pdf_candidate_chunk_size=pdf_candidate_chunk_size,
    )

    for it in range(int(n_refine_iters)):
        for par_ind in range(4):
            plus = params.copy()
            minus = params.copy()
            plus[:, par_ind] = cp.minimum(upper[:, par_ind], plus[:, par_ind] + step[:, par_ind])
            minus[:, par_ind] = cp.maximum(lower[:, par_ind], minus[:, par_ind] - step[:, par_ind])

            nll_plus = _gpu_eval_nll_fixed_n(
                plus,
                hist_gpu,
                pillar_idx_gpu,
                x_vals_gpu,
                int(n_nvs),
                prob_dist_name,
                pdf_candidate_chunk_size=pdf_candidate_chunk_size,
            )
            nll_minus = _gpu_eval_nll_fixed_n(
                minus,
                hist_gpu,
                pillar_idx_gpu,
                x_vals_gpu,
                int(n_nvs),
                prob_dist_name,
                pdf_candidate_chunk_size=pdf_candidate_chunk_size,
            )

            use_plus = nll_plus < nll
            use_minus = (nll_minus < nll) & (nll_minus < nll_plus)

            params = cp.where(use_plus[:, None], plus, params)
            nll = cp.where(use_plus, nll_plus, nll)

            params = cp.where(use_minus[:, None], minus, params)
            nll = cp.where(use_minus, nll_minus, nll)

        step = step * float(step_shrink)

    params_np = cp.asnumpy(params).reshape(P, S, 4)
    nll_np = cp.asnumpy(nll).reshape(P, S)

    best_start = np.nanargmin(nll_np, axis=1)
    best_params = params_np[np.arange(P), best_start, :]
    best_nll = nll_np[np.arange(P), best_start]

    return best_params, best_nll


def fit_charge_histograms_gpu_batch_refined_auto(
    ref_counts_lists,
    prob_dist,
    gpu_fit_config=None,
    max_nvs_per_position=3,
    force_nvs=None,
    bic_extra_nv_penalty=8.0,
    strict_extra_nv_penalty=60.0,
    strict_bic_margin=25.0,
    strict_min_mode_weight=0.04,
    strict_min_mode_shots=50,
    strict_min_adjacent_dprime=1.35,
    strict_require_all_modes=True,
    gpu_refined_n_starts=6,
    gpu_refined_iters=8,
    gpu_refined_step_shrink=0.55,
    gpu_refined_pdf_chunk_size=2048,
):
    """
    Pure-GPU version of the careful CPU protocol.

    It does not call scipy curve_fit. It does:
        1. integer histograms,
        2. forced N=1,2,3 fits,
        3. multi-start continuous coordinate refinement on GPU,
        4. BIC + physical mode visibility selection.
    """
    _require_cupy_for_refined_gpu()

    prob_dist_name = str(prob_dist.name if hasattr(prob_dist, "name") else prob_dist)
    max_nvs = int(max_nvs_per_position)

    hist_counts, x_vals, nshots, qstats = _build_integer_histogram_matrix_for_gpu(
        ref_counts_lists
    )

    hist_gpu = cp.asarray(hist_counts, dtype=cp.float64)
    x_vals_gpu = cp.asarray(x_vals, dtype=cp.float64)

    if force_nvs is None:
        Ns = list(range(1, max_nvs + 1))
    else:
        Ns = [int(force_nvs)]

    all_results_by_n = {}
    all_debug_by_n = {}

    for N in Ns:
        print(f"GPU-refined fit: forced N={N}")

        best_params, best_nll = _gpu_coordinate_refine_fixed_n(
            hist_gpu=hist_gpu,
            x_vals_gpu=x_vals_gpu,
            qstats=qstats,
            n_nvs=N,
            prob_dist_name=prob_dist_name,
            n_starts=gpu_refined_n_starts,
            n_refine_iters=gpu_refined_iters,
            step_shrink=gpu_refined_step_shrink,
            seed=1000 + 17 * int(N),
            pdf_candidate_chunk_size=gpu_refined_pdf_chunk_size,
        )

        results_n = []
        for ind in range(len(ref_counts_lists)):
            popt = np.asarray(best_params[ind], dtype=float)
            nll = float(best_nll[ind])
            n_samp = float(max(nshots[ind], 1.0))

            # Same spirit as your CPU selection: more structural penalty for larger N.
            k_free = 4 + (int(N) - 1)
            bic = float(k_free * np.log(n_samp) + 2.0 * nll + float(bic_extra_nv_penalty) * (int(N) - 1))

            if not np.all(np.isfinite(popt)):
                results_n.append({"ok": False, "reason": "nonfinite_popt", "n_nvs": int(N)})
                continue

            results_n.append(
                {
                    "ok": True,
                    "model": f"{int(N)}nv_equal_gpu_refined",
                    "n_nvs": int(N),
                    "popt": popt,
                    "p_minus": float(popt[0]),
                    "bg": float(popt[1]),
                    "rate0": float(popt[2]),
                    "delta": float(popt[3]),
                    "nll": nll,
                    "bic": bic,
                    "red_chi_sq": np.nan,
                    "k_free": int(k_free),
                    "n_samp": n_samp,
                }
            )

        all_results_by_n[int(N)] = results_n
        all_debug_by_n[int(N)] = {
            "N": int(N),
            "median_nll": float(np.nanmedian(best_nll)),
            "median_bic": float(np.nanmedian([r.get("bic", np.nan) for r in results_n])),
        }

    # If forced N was requested, return those fits directly.
    if force_nvs is not None:
        debug = {
            "gpu_refined": True,
            "forced_nvs": int(force_nvs),
            "prob_dist": prob_dist_name,
            "debug_by_n": make_json_safe(all_debug_by_n),
        }
        return all_results_by_n[int(force_nvs)], debug

    selected = []
    selected_n = []
    strict_records = []
    P = len(ref_counts_lists)

    for ind in range(P):
        candidates = []
        for N in Ns:
            fit = all_results_by_n[int(N)][ind]
            diag = _strict_candidate_diagnostics(
                fit,
                ref_counts_lists[ind],
                strict_min_mode_weight=strict_min_mode_weight,
                strict_min_mode_shots=strict_min_mode_shots,
                strict_min_adjacent_dprime=strict_min_adjacent_dprime,
                strict_require_all_modes=strict_require_all_modes,
            )
            adj_bic = _adjusted_bic_for_strict(
                fit,
                strict_extra_nv_penalty=strict_extra_nv_penalty,
            )
            candidates.append(
                {
                    "n_nvs": int(N),
                    "ok_fit": bool(isinstance(fit, dict) and fit.get("ok", False)),
                    "physical_ok": bool(diag["ok"]),
                    "bic": safe_float(fit.get("bic", np.nan)) if isinstance(fit, dict) else np.nan,
                    "adjusted_bic": safe_float(adj_bic),
                    "diagnostics": make_json_safe(diag),
                    "model": fit.get("model", None) if isinstance(fit, dict) else None,
                    "popt": make_json_safe(fit.get("popt", None)) if isinstance(fit, dict) else None,
                }
            )

        # Start from N=1 if available.
        best_n = 1
        best_fit = all_results_by_n[1][ind]
        if not (isinstance(best_fit, dict) and best_fit.get("ok", False)):
            best_fit = None
            best_n = None
            for N in Ns:
                f = all_results_by_n[int(N)][ind]
                if isinstance(f, dict) and f.get("ok", False):
                    best_fit = f
                    best_n = int(N)
                    break

        if best_fit is None:
            selected.append({"ok": False, "reason": "all_gpu_refined_fits_failed"})
            selected_n.append(0)
            strict_records.append({"selected_n": 0, "candidates": candidates})
            continue

        best_adj_bic = _adjusted_bic_for_strict(
            best_fit,
            strict_extra_nv_penalty=strict_extra_nv_penalty,
        )
        selected_reason = f"kept N={best_n} simplest valid gpu-refined model"

        for N in range(max(int(best_n) + 1, 2), max_nvs + 1):
            fit = all_results_by_n[int(N)][ind]
            diag = _strict_candidate_diagnostics(
                fit,
                ref_counts_lists[ind],
                strict_min_mode_weight=strict_min_mode_weight,
                strict_min_mode_shots=strict_min_mode_shots,
                strict_min_adjacent_dprime=strict_min_adjacent_dprime,
                strict_require_all_modes=strict_require_all_modes,
            )
            cand_adj_bic = _adjusted_bic_for_strict(
                fit,
                strict_extra_nv_penalty=strict_extra_nv_penalty,
            )
            bic_ok = cand_adj_bic < (best_adj_bic - float(strict_bic_margin))
            if diag["ok"] and bic_ok:
                best_n = int(N)
                best_fit = fit
                best_adj_bic = cand_adj_bic
                selected_reason = (
                    f"accepted N={N}: gpu-refined adjusted BIC and physical checks passed"
                )

        out_fit = dict(best_fit)
        out_fit["strict_selected_n"] = int(best_n)
        out_fit["strict_selected_reason"] = selected_reason
        out_fit["strict_adjusted_bic"] = safe_float(best_adj_bic)
        out_fit["best_candidate_model"] = out_fit.get("model", None)
        out_fit["best_candidate_bic"] = out_fit.get("bic", np.nan)
        out_fit["best_equal_bic"] = out_fit.get("bic", np.nan)
        out_fit["unequal_2nv_beats_equal"] = False
        selected.append(out_fit)
        selected_n.append(int(best_n))
        strict_records.append(
            {
                "selected_n": int(best_n),
                "selected_reason": selected_reason,
                "selected_adjusted_bic": safe_float(best_adj_bic),
                "candidates": candidates,
            }
        )

    debug = {
        "gpu_refined": True,
        "prob_dist": prob_dist_name,
        "max_nvs_per_position": int(max_nvs),
        "bic_extra_nv_penalty": float(bic_extra_nv_penalty),
        "strict_extra_nv_penalty": float(strict_extra_nv_penalty),
        "strict_bic_margin": float(strict_bic_margin),
        "strict_min_mode_weight": float(strict_min_mode_weight),
        "strict_min_mode_shots": int(strict_min_mode_shots),
        "strict_min_adjacent_dprime": float(strict_min_adjacent_dprime),
        "strict_require_all_modes": bool(strict_require_all_modes),
        "gpu_refined_n_starts": int(gpu_refined_n_starts),
        "gpu_refined_iters": int(gpu_refined_iters),
        "gpu_refined_step_shrink": float(gpu_refined_step_shrink),
        "gpu_refined_pdf_chunk_size": int(gpu_refined_pdf_chunk_size),
        "selected_n_counts": {
            str(n): int(np.sum(np.asarray(selected_n) == n)) for n in range(0, max_nvs + 1)
        },
        "debug_by_n": make_json_safe(all_debug_by_n),
        "strict_records": make_json_safe(strict_records),
    }

    return selected, debug


def fit_charge_histograms_gpu_batch_strict_auto(
    ref_counts_lists,
    prob_dist,
    gpu_fit_config,
    max_nvs_per_position=3,
    force_nvs=None,
    strict_extra_nv_penalty=60.0,
    strict_bic_margin=20.0,
    strict_min_mode_weight=0.04,
    strict_min_mode_shots=50,
    strict_min_adjacent_dprime=1.35,
    strict_require_all_modes=True,
):
    """
    Careful multimodal GPU fitting.

    This runs separate forced-N fits for N=1..max_nvs_per_position. Then it
    starts from N=1 and only accepts N>1 when:
        - the higher-N model beats the simpler model by a clear adjusted-BIC margin,
        - all required modes have enough probability and expected shots,
        - adjacent peaks are separated enough in shot-noise units.

    This avoids the common artifact where a flexible multimodal fit invents a
    weak intermediate mode between the real NV0 and NV- peaks.
    """
    if force_nvs is not None:
        results, debug = fit_charge_histograms_gpu_batch(
            ref_counts_lists,
            prob_dist=prob_dist,
            model_mode="multimode",
            multimode_config=gpu_fit_config,
            max_nvs=int(max_nvs_per_position),
            force_nvs=int(force_nvs),
            return_debug=True,
        )
        return results, {
            "strict_auto": False,
            "reason": "force_nvs_used",
            "forced_nvs": int(force_nvs),
            "forced_debug": make_json_safe(debug),
        }

    max_nvs = int(max_nvs_per_position)
    all_results_by_n = {}
    all_debug_by_n = {}

    for n in range(1, max_nvs + 1):
        print(f"Strict-auto GPU fit: forced N={n}")
        res_n, debug_n = fit_charge_histograms_gpu_batch(
            ref_counts_lists,
            prob_dist=prob_dist,
            model_mode="multimode",
            multimode_config=gpu_fit_config,
            max_nvs=max_nvs,
            force_nvs=n,
            return_debug=True,
        )
        all_results_by_n[n] = res_n
        all_debug_by_n[n] = make_json_safe(debug_n)

    selected = []
    selected_n = []
    strict_records = []

    num_items = len(ref_counts_lists)
    for ind in range(num_items):
        candidates = []
        for n in range(1, max_nvs + 1):
            fit = all_results_by_n[n][ind]
            diag = _strict_candidate_diagnostics(
                fit,
                ref_counts_lists[ind],
                strict_min_mode_weight=strict_min_mode_weight,
                strict_min_mode_shots=strict_min_mode_shots,
                strict_min_adjacent_dprime=strict_min_adjacent_dprime,
                strict_require_all_modes=strict_require_all_modes,
            )
            adj_bic = _adjusted_bic_for_strict(
                fit,
                strict_extra_nv_penalty=strict_extra_nv_penalty,
            )
            candidates.append(
                {
                    "n_nvs": int(n),
                    "ok_fit": bool(isinstance(fit, dict) and fit.get("ok", False)),
                    "physical_ok": bool(diag["ok"]),
                    "bic": safe_float(fit.get("bic", np.nan)) if isinstance(fit, dict) else np.nan,
                    "adjusted_bic": safe_float(adj_bic),
                    "diagnostics": make_json_safe(diag),
                    "model": fit.get("model", None) if isinstance(fit, dict) else None,
                    "popt": make_json_safe(fit.get("popt", None)) if isinstance(fit, dict) else None,
                }
            )

        # Start from the simplest valid fit, ideally N=1.
        best_n = None
        best_fit = None
        for n in range(1, max_nvs + 1):
            fit = all_results_by_n[n][ind]
            if isinstance(fit, dict) and fit.get("ok", False):
                best_n = n
                best_fit = fit
                break

        if best_fit is None:
            selected.append({"ok": False, "reason": "all_forced_N_fits_failed"})
            selected_n.append(0)
            strict_records.append(
                {
                    "selected_n": 0,
                    "selected_reason": "all_forced_N_fits_failed",
                    "candidates": candidates,
                }
            )
            continue

        best_adj_bic = _adjusted_bic_for_strict(
            best_fit,
            strict_extra_nv_penalty=strict_extra_nv_penalty,
        )
        selected_reason = f"kept N={best_n} simplest valid model"

        for n in range(max(best_n + 1, 2), max_nvs + 1):
            fit = all_results_by_n[n][ind]
            diag = _strict_candidate_diagnostics(
                fit,
                ref_counts_lists[ind],
                strict_min_mode_weight=strict_min_mode_weight,
                strict_min_mode_shots=strict_min_mode_shots,
                strict_min_adjacent_dprime=strict_min_adjacent_dprime,
                strict_require_all_modes=strict_require_all_modes,
            )
            cand_adj_bic = _adjusted_bic_for_strict(
                fit,
                strict_extra_nv_penalty=strict_extra_nv_penalty,
            )

            # Smaller BIC is better. Candidate must be clearly better after
            # extra-N penalty and by a margin.
            bic_ok = cand_adj_bic < (best_adj_bic - float(strict_bic_margin))

            if diag["ok"] and bic_ok:
                best_n = n
                best_fit = fit
                best_adj_bic = cand_adj_bic
                selected_reason = (
                    f"accepted N={n}: adjusted BIC improved by >= {strict_bic_margin} "
                    "and physical mode checks passed"
                )

        out_fit = dict(best_fit)
        out_fit["strict_selected_n"] = int(best_n)
        out_fit["strict_selected_reason"] = selected_reason
        out_fit["strict_adjusted_bic"] = safe_float(best_adj_bic)
        selected.append(out_fit)
        selected_n.append(int(best_n))
        strict_records.append(
            {
                "selected_n": int(best_n),
                "selected_reason": selected_reason,
                "selected_adjusted_bic": safe_float(best_adj_bic),
                "candidates": candidates,
            }
        )

    debug = {
        "strict_auto": True,
        "prob_dist": str(prob_dist.name if hasattr(prob_dist, "name") else prob_dist),
        "max_nvs_per_position": int(max_nvs),
        "strict_extra_nv_penalty": float(strict_extra_nv_penalty),
        "strict_bic_margin": float(strict_bic_margin),
        "strict_min_mode_weight": float(strict_min_mode_weight),
        "strict_min_mode_shots": int(strict_min_mode_shots),
        "strict_min_adjacent_dprime": float(strict_min_adjacent_dprime),
        "strict_require_all_modes": bool(strict_require_all_modes),
        "forced_fit_debug_by_n": all_debug_by_n,
        "selected_n_counts": {
            str(n): int(np.sum(np.asarray(selected_n) == n)) for n in range(0, max_nvs + 1)
        },
        "strict_records": strict_records,
    }

    return selected, debug

def _make_compatible_gpu_multimode_config(
    max_nvs_per_position: int,
    bic_extra_nv_penalty: float,
    strict_extra_nv_penalty: float,
    strict_bic_margin: float,
    strict_min_mode_weight: float,
    strict_min_mode_shots: int,
    strict_min_adjacent_dprime: float,
    strict_require_all_modes: bool,
    gpu_refined_n_starts: int,
    gpu_refined_iters: int,
    gpu_refined_step_shrink: float,
    gpu_refined_pdf_chunk_size: int,
    include_2nv_unequal: bool,
    user_config: Optional[GpuMultimodeFitConfig] = None,
):
    """
    Build a GpuMultimodeFitConfig that is compatible with both older and newer
    versions of analysis.sc_gpu_bimodal_fitting.

    The updated backend may contain extra fields such as use_refinement,
    refine_iters, strict_* and include_2nv_unequal. Older versions do not.
    This helper only passes fields that exist in the dataclass.
    """
    if user_config is not None:
        return user_config

    desired = {
        # Existing/base fields
        "max_nvs": int(max_nvs_per_position),
        "num_p": 13,
        "num_bg": 5,
        "num_rate0": 18,
        "num_delta": 18,
        "fit_chunk_size": 512,
        "candidate_chunk_size": 512,
        "bic_extra_nv_penalty": float(bic_extra_nv_penalty),

        # New refined-GPU fields, if available in your backend
        "use_refinement": True,
        "refine_iters": int(gpu_refined_iters),
        "refine_fit_chunk_size": 64,
        "gpu_refined_n_starts": int(gpu_refined_n_starts),
        "gpu_refined_iters": int(gpu_refined_iters),
        "gpu_refined_step_shrink": float(gpu_refined_step_shrink),
        "gpu_refined_pdf_chunk_size": int(gpu_refined_pdf_chunk_size),

        # Strict physical model-selection fields, if available in your backend
        "strict_extra_nv_penalty": float(strict_extra_nv_penalty),
        "strict_bic_margin": float(strict_bic_margin),
        "strict_min_mode_weight": float(strict_min_mode_weight),
        "strict_min_mode_shots": int(strict_min_mode_shots),
        "strict_min_adjacent_dprime": float(strict_min_adjacent_dprime),
        "strict_require_all_modes": bool(strict_require_all_modes),

        # 2-NV unequal diagnostic, if available in your backend
        "include_2nv_unequal": bool(include_2nv_unequal),
    }

    try:
        import dataclasses

        valid_names = {f.name for f in dataclasses.fields(GpuMultimodeFitConfig)}
        kwargs = {k: v for k, v in desired.items() if k in valid_names}
        return GpuMultimodeFitConfig(**kwargs)
    except Exception:
        # Very old backend fallback.
        return GpuMultimodeFitConfig(
            max_nvs=int(max_nvs_per_position),
            num_p=13,
            num_bg=5,
            num_rate0=18,
            num_delta=18,
            fit_chunk_size=512,
            candidate_chunk_size=512,
        )


def _extract_candidate_results_from_fits_and_debug(fit_results, fit_debug, num_positions):
    """Return one candidate/diagnostic object per pillar when available."""
    out = [None for _ in range(num_positions)]

    if isinstance(fit_debug, dict):
        for key in ["strict_records", "candidate_results", "records"]:
            val = fit_debug.get(key, None)
            if isinstance(val, list) and len(val) == num_positions:
                return val

    for ind, fit in enumerate(fit_results):
        if not isinstance(fit, dict):
            continue
        for key in ["candidate_results", "strict_candidates", "strict_records"]:
            if key in fit:
                out[ind] = make_json_safe(fit.get(key))
                break

    return out


# =============================================================================
# Post-fit physical pruning
# =============================================================================


def _count_empirical_histogram_peaks(
    counts,
    smooth_window: int = 5,
    min_peak_prominence_frac: float = 0.025,
    min_peak_distance_counts: int = 8,
):
    """
    Count prominent peaks in the measured integer-count histogram.

    This is deliberately empirical. It does not ask whether a higher-N model can
    improve likelihood; it asks whether the measured histogram visibly supports
    that many charge manifolds.
    """
    x = np.asarray(counts, dtype=float).ravel()
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size < 10:
        return 0, np.asarray([], dtype=int), np.asarray([], dtype=float)

    # Same outlier spirit as the fitter.
    med = np.median(x)
    std = np.std(x)
    if np.isfinite(std) and std > 0:
        x = x[x < med + 10.0 * std]
    if x.size < 10:
        return 0, np.asarray([], dtype=int), np.asarray([], dtype=float)

    xi = np.rint(x).astype(int)
    hist = np.bincount(xi, minlength=int(np.max(xi)) + 1).astype(float)

    if hist.size < 3 or np.nanmax(hist) <= 0:
        return 0, np.asarray([], dtype=int), hist

    w = int(max(1, smooth_window))
    if w > 1:
        kernel = np.ones(w, dtype=float) / float(w)
        smooth = np.convolve(hist, kernel, mode="same")
    else:
        smooth = hist.copy()

    prominence = float(min_peak_prominence_frac) * float(np.nanmax(smooth))
    distance = int(max(1, min_peak_distance_counts))

    try:
        from scipy.signal import find_peaks
        peaks, props = find_peaks(smooth, prominence=prominence, distance=distance)
    except Exception:
        # Minimal fallback if scipy.signal is unavailable.
        peaks = []
        for i in range(1, smooth.size - 1):
            if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] >= prominence:
                if len(peaks) == 0 or (i - peaks[-1]) >= distance:
                    peaks.append(i)
                elif smooth[i] > smooth[peaks[-1]]:
                    peaks[-1] = i
        peaks = np.asarray(peaks, dtype=int)

    return int(len(peaks)), np.asarray(peaks, dtype=int), smooth


def _candidate_record_list(candidate_obj):
    """Normalize candidate diagnostics to a list of candidate dictionaries."""
    if candidate_obj is None:
        return []

    if isinstance(candidate_obj, dict):
        if "candidates" in candidate_obj and isinstance(candidate_obj["candidates"], list):
            return candidate_obj["candidates"]
        # Some backends use {"1": {...}, "2": {...}}.
        out = []
        for key, val in candidate_obj.items():
            if isinstance(val, dict):
                rec = dict(val)
                if "n_nvs" not in rec:
                    try:
                        rec["n_nvs"] = int(key)
                    except Exception:
                        pass
                out.append(rec)
        return out

    if isinstance(candidate_obj, list):
        return [x for x in candidate_obj if isinstance(x, dict)]

    return []


def _find_candidate_record(candidate_obj, n_nvs):
    for rec in _candidate_record_list(candidate_obj):
        try:
            if int(rec.get("n_nvs", rec.get("N", -1))) == int(n_nvs):
                return rec
        except Exception:
            continue
    return None


def _fit_from_candidate_record(rec, fallback_fit=None):
    """Build a fit dictionary from a strict/candidate record if it has popt."""
    if rec is None:
        return None

    popt = rec.get("popt", None)
    if popt is None and isinstance(rec.get("fit", None), dict):
        popt = rec["fit"].get("popt", None)

    try:
        popt = np.asarray(popt, dtype=float)
    except Exception:
        return None

    if popt.size != 4 or not np.all(np.isfinite(popt)):
        return None

    n_nvs = int(rec.get("n_nvs", rec.get("N", 1)))
    out = {}
    if isinstance(fallback_fit, dict):
        out.update(fallback_fit)

    out.update(
        {
            "ok": True,
            "model": rec.get("model", f"{n_nvs}nv_equal_posthoc"),
            "n_nvs": int(n_nvs),
            "popt": popt,
            "p_minus": float(popt[0]),
            "bg": float(popt[1]),
            "rate0": float(popt[2]),
            "delta": float(popt[3]),
            "bic": safe_float(rec.get("bic", np.nan)),
            "nll": safe_float(rec.get("nll", np.nan)),
            "red_chi_sq": safe_float(rec.get("red_chi_sq", np.nan)),
            "posthoc_from_candidate": True,
        }
    )
    return out



def _clean_counts_for_posthoc(counts):
    """Clean counts with the same outlier spirit as the fitter."""
    x = np.asarray(counts, dtype=float).ravel()
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return x
    med = np.median(x)
    std = np.std(x)
    if np.isfinite(std) and std > 0:
        x = x[x < med + 10.0 * std]
    return x


def _pearson_red_chi_sq_for_equal_fit(
    counts,
    popt,
    prob_dist: ProbDist,
    n_nvs: int,
    min_expected_per_bin: float = 2.0,
):
    """
    Pearson reduced chi-square for the equal-brightness fit.

    This is used only as a posthoc model-selection sanity check. The fitter itself
    uses likelihood/BIC, which can still prefer a higher-N model if that model
    explains broad tails. Here we ask a different question: did the extra mode
    substantially improve the binned residuals?
    """
    x = _clean_counts_for_posthoc(counts)
    if x.size < 10:
        return np.nan

    try:
        xi = np.rint(np.clip(x, 0, None)).astype(int)
        x_max = int(max(np.max(xi), 1))
        obs = np.bincount(xi, minlength=x_max + 1).astype(float)
        nshots = float(np.sum(obs))

        _x_vals, weights, pdf, _cdf_ext, _T = _fit_equal_model_pdf_tables(
            prob_dist,
            int(n_nvs),
            popt,
            x_max=x_max,
        )
        model_prob = np.sum(weights[:, None] * pdf, axis=0)
        model_prob = np.clip(model_prob, 1e-300, None)
        model_prob = model_prob / float(np.sum(model_prob))
        expected = nshots * model_prob

        # Ignore bins with no statistical support in both data and model.
        mask = (expected >= float(min_expected_per_bin)) | (obs > 0)
        if not np.any(mask):
            return np.nan

        chi2 = np.sum(((obs[mask] - expected[mask]) ** 2) / np.maximum(expected[mask], 1e-9))
        k_free = 4 + max(int(n_nvs) - 1, 0)
        dof = max(int(np.sum(mask)) - int(k_free), 1)
        return float(chi2 / dof)
    except Exception:
        return np.nan


def _middle_mode_support_diagnostics(
    counts,
    popt,
    prob_dist: ProbDist,
    n_nvs: int,
    min_middle_obs_to_expected_ratio: float = 0.35,
    middle_window_sigma: float = 1.25,
    middle_window_max_frac_delta: float = 0.35,
    min_middle_observed_fraction: float = 0.01,
):
    """
    Check whether interior modes k=1..N-1 have real measured support.

    This targets the failure mode you observed: a clean two-mode single-NV
    histogram is fit as N=2 by inventing a k=1 component between two far peaks.
    For a real N=2 equal-brightness pillar, the k=1 component should leave
    visible counts around its predicted center.
    """
    diag = {
        "ok": True,
        "reason": "no_interior_modes" if int(n_nvs) <= 1 else "ok",
        "interior": [],
    }

    n_nvs = int(n_nvs)
    if n_nvs <= 1:
        return diag

    x = _clean_counts_for_posthoc(counts)
    if x.size < 10:
        diag["ok"] = False
        diag["reason"] = "too_few_counts"
        return diag

    try:
        p_minus, bg, rate0, delta = [float(v) for v in np.asarray(popt, dtype=float)]
        base = max(bg, 0.0) + n_nvs * max(rate0, 1e-12)
        delta = max(delta, 0.0)
        centers = base + np.arange(n_nvs + 1, dtype=float) * delta
        x_max = int(max(np.nanmax(x), np.nanmax(centers) + 8.0 * np.sqrt(max(np.nanmax(centers), 1.0)), 1))

        _x_vals, weights, pdf, _cdf_ext, _T = _fit_equal_model_pdf_tables(
            prob_dist,
            n_nvs,
            popt,
            x_max=x_max,
        )

        bad_reasons = []
        for k in range(1, n_nvs):
            lam = float(centers[k])
            sigma = float(np.sqrt(max(lam, 1.0)))
            half_width = float(middle_window_sigma) * sigma
            if delta > 0:
                half_width = min(half_width, float(middle_window_max_frac_delta) * delta)
            half_width = max(half_width, 2.0)

            lo = lam - half_width
            hi = lam + half_width
            obs_frac = float(np.mean((x >= lo) & (x <= hi)))

            x_inds = np.arange(pdf.shape[1], dtype=float)
            mask = (x_inds >= lo) & (x_inds <= hi)
            expected_component_frac = float(weights[k] * np.sum(pdf[k, mask]))
            expected_total_frac = float(np.sum(weights[:, None] * pdf[:, mask]))
            ratio = obs_frac / max(expected_component_frac, 1e-12)

            rec = {
                "k": int(k),
                "center": lam,
                "half_width": half_width,
                "weight": float(weights[k]),
                "obs_frac_in_window": obs_frac,
                "expected_component_frac_in_window": expected_component_frac,
                "expected_total_frac_in_window": expected_total_frac,
                "obs_to_expected_component_ratio": ratio,
            }
            diag["interior"].append(rec)

            if obs_frac < float(min_middle_observed_fraction):
                bad_reasons.append(
                    f"middle_k{k}_obs_frac_too_small={obs_frac:.4g}"
                )
            if ratio < float(min_middle_obs_to_expected_ratio):
                bad_reasons.append(
                    f"middle_k{k}_obs_to_expected_too_small={ratio:.3g}"
                )

        if bad_reasons:
            diag["ok"] = False
            diag["reason"] = "; ".join(bad_reasons)
        else:
            diag["ok"] = True
            diag["reason"] = "interior_modes_have_empirical_support"
        return diag

    except Exception as exc:
        diag["ok"] = False
        diag["reason"] = f"middle_support_check_failed: {exc}"
        return diag


def _posthoc_demote_unphysical_fits(
    gpu_fit_results,
    candidate_results_list,
    ref_counts_lists,
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
    max_nvs_per_position=3,
    enable=True,
    require_empirical_peak_count=True,
    min_peak_prominence_frac=0.025,
    min_peak_distance_counts=8,
    min_mode_weight=0.08,
    min_mode_shots=100,
    min_adjacent_dprime=1.8,
    use_chi_square_gate=True,
    min_red_chi_sq_improvement=0.08,
    min_expected_per_bin_for_chi=2.0,
    use_middle_mode_support_gate=True,
    min_middle_obs_to_expected_ratio=0.35,
    middle_window_sigma=1.25,
    middle_window_max_frac_delta=0.35,
    min_middle_observed_fraction=0.01,
):
    """
    Post-fit pruning layer that demotes N=2/3 results when the data do not
    visibly support extra charge manifolds.

    This is intentionally more conservative than BIC. BIC can prefer a flexible
    model that explains shoulders/noise; for feedback/control we want only modes
    that are physically visible.
    """
    if not enable:
        return gpu_fit_results, {
            "enabled": False,
            "num_demoted": 0,
            "records": [None for _ in gpu_fit_results],
        }

    out_results = list(gpu_fit_results)
    records = []
    num_demoted = 0

    for ind, fit in enumerate(gpu_fit_results):
        rec = {
            "pillar_index": int(ind),
            "original_n": None,
            "final_n": None,
            "demoted": False,
            "reason": "kept",
            "num_empirical_peaks": None,
            "empirical_peak_positions": None,
            "selected_red_chi_sq": None,
            "lower_red_chi_sq": None,
            "red_chi_sq_improvement": None,
            "middle_mode_support": None,
        }

        if not isinstance(fit, dict) or not fit.get("ok", False):
            rec["reason"] = "fit_not_ok"
            records.append(rec)
            continue

        try:
            original_n = int(fit.get("n_nvs", 1))
        except Exception:
            original_n = 1

        rec["original_n"] = int(original_n)
        rec["final_n"] = int(original_n)

        if original_n <= 1:
            records.append(rec)
            continue

        peak_count, peaks, _smooth = _count_empirical_histogram_peaks(
            ref_counts_lists[ind],
            min_peak_prominence_frac=min_peak_prominence_frac,
            min_peak_distance_counts=min_peak_distance_counts,
        )
        rec["num_empirical_peaks"] = int(peak_count)
        rec["empirical_peak_positions"] = peaks.astype(int).tolist()

        # Decide the maximum N that the empirical histogram can justify.
        # Equal-brightness N requires N+1 visible charge manifolds.
        if require_empirical_peak_count:
            max_n_by_peaks = max(1, int(peak_count) - 1)
        else:
            max_n_by_peaks = int(max_nvs_per_position)

        target_n = min(original_n, max_n_by_peaks)

        # Also apply stricter parameter-space checks to the selected fit.
        diag = _strict_candidate_diagnostics(
            fit,
            ref_counts_lists[ind],
            strict_min_mode_weight=min_mode_weight,
            strict_min_mode_shots=min_mode_shots,
            strict_min_adjacent_dprime=min_adjacent_dprime,
            strict_require_all_modes=True,
        )

        if not diag.get("ok", False):
            target_n = min(target_n, original_n - 1)
            rec["reason"] = "failed_posthoc_physical_checks: " + "; ".join(diag.get("reasons", []))
        elif require_empirical_peak_count and peak_count < original_n + 1:
            rec["reason"] = f"empirical_peak_count_{peak_count}_less_than_required_{original_n + 1}"

        # Chi-square improvement gate. BIC/NLL can prefer an extra middle mode
        # because it explains tails. Require that the selected higher-N model
        # also improves binned residuals relative to the next-lower candidate.
        if use_chi_square_gate and original_n > 1:
            selected_chi = _pearson_red_chi_sq_for_equal_fit(
                ref_counts_lists[ind],
                fit.get("popt", None),
                prob_dist,
                original_n,
                min_expected_per_bin=min_expected_per_bin_for_chi,
            )
            rec["selected_red_chi_sq"] = safe_float(selected_chi)

            lower_fit = None
            for n_try in range(original_n - 1, 0, -1):
                cand = _find_candidate_record(candidate_results_list[ind], n_try)
                lower_fit = _fit_from_candidate_record(cand, fallback_fit=fit)
                if lower_fit is not None:
                    break

            if lower_fit is not None:
                lower_n = int(lower_fit.get("n_nvs", max(original_n - 1, 1)))
                lower_chi = _pearson_red_chi_sq_for_equal_fit(
                    ref_counts_lists[ind],
                    lower_fit.get("popt", None),
                    prob_dist,
                    lower_n,
                    min_expected_per_bin=min_expected_per_bin_for_chi,
                )
                improvement = safe_float(lower_chi) - safe_float(selected_chi)
                rec["lower_red_chi_sq"] = safe_float(lower_chi)
                rec["red_chi_sq_improvement"] = safe_float(improvement)

                if (not np.isfinite(improvement)) or improvement < float(min_red_chi_sq_improvement):
                    target_n = min(target_n, original_n - 1)
                    chi_reason = (
                        f"red_chi_sq_improvement_too_small="
                        f"{safe_float(improvement):.4g}< {float(min_red_chi_sq_improvement):.4g}"
                    )
                    rec["reason"] = chi_reason if rec["reason"] == "kept" else rec["reason"] + "; " + chi_reason

        # Interior-mode support gate. This specifically rejects the fake middle
        # k=1 component when the measured histogram only has two real peaks.
        if use_middle_mode_support_gate and original_n > 1:
            mid_diag = _middle_mode_support_diagnostics(
                ref_counts_lists[ind],
                fit.get("popt", None),
                prob_dist,
                original_n,
                min_middle_obs_to_expected_ratio=min_middle_obs_to_expected_ratio,
                middle_window_sigma=middle_window_sigma,
                middle_window_max_frac_delta=middle_window_max_frac_delta,
                min_middle_observed_fraction=min_middle_observed_fraction,
            )
            rec["middle_mode_support"] = make_json_safe(mid_diag)
            if not mid_diag.get("ok", False):
                target_n = min(target_n, original_n - 1)
                mid_reason = "failed_middle_mode_support: " + str(mid_diag.get("reason", "unknown"))
                rec["reason"] = mid_reason if rec["reason"] == "kept" else rec["reason"] + "; " + mid_reason

        if target_n < 1:
            target_n = 1

        if target_n < original_n:
            replacement = None
            # Try target_n, then smaller N if needed.
            for n_try in range(int(target_n), 0, -1):
                cand = _find_candidate_record(candidate_results_list[ind], n_try)
                replacement = _fit_from_candidate_record(cand, fallback_fit=fit)
                if replacement is not None:
                    replacement["posthoc_demoted_from_n"] = int(original_n)
                    replacement["posthoc_demoted_to_n"] = int(n_try)
                    replacement["posthoc_reason"] = rec["reason"]
                    target_n = int(n_try)
                    break

            # If candidate details are missing, keep the original fit but flag it.
            if replacement is not None:
                out_results[ind] = replacement
                rec["demoted"] = True
                rec["final_n"] = int(target_n)
                num_demoted += 1
            else:
                rec["reason"] += "; candidate_for_lower_N_missing_so_original_kept"

        records.append(rec)

    debug = {
        "enabled": True,
        "num_demoted": int(num_demoted),
        "require_empirical_peak_count": bool(require_empirical_peak_count),
        "min_peak_prominence_frac": float(min_peak_prominence_frac),
        "min_peak_distance_counts": int(min_peak_distance_counts),
        "min_mode_weight": float(min_mode_weight),
        "min_mode_shots": int(min_mode_shots),
        "min_adjacent_dprime": float(min_adjacent_dprime),
        "use_chi_square_gate": bool(use_chi_square_gate),
        "min_red_chi_sq_improvement": float(min_red_chi_sq_improvement),
        "min_expected_per_bin_for_chi": float(min_expected_per_bin_for_chi),
        "use_middle_mode_support_gate": bool(use_middle_mode_support_gate),
        "min_middle_obs_to_expected_ratio": float(min_middle_obs_to_expected_ratio),
        "middle_window_sigma": float(middle_window_sigma),
        "middle_window_max_frac_delta": float(middle_window_max_frac_delta),
        "min_middle_observed_fraction": float(min_middle_observed_fraction),
        "records": make_json_safe(records),
    }
    return out_results, debug


def process_and_plot(
    raw_data,
    do_plot_histograms=False,
    prob_dist: ProbDist = ProbDist.BROADENED_POISSON,
    max_nvs_per_position: int = 3,
    force_nvs: Optional[int] = None,
    bic_extra_nv_penalty: float = 8.0,
    save_analysis: bool = False,
    save_hist_figs: bool = False,
    n_jobs: int = 12,  # kept for API compatibility; backend GPU path ignores this
    joblib_verbose: int = 10,  # kept for API compatibility
    gpu_fit_config: Optional[GpuMultimodeFitConfig] = None,
    model_mode: str = "gpu_refined",
    strict_extra_nv_penalty: float = 100.0,
    strict_bic_margin: float = 30.0,
    strict_min_mode_weight: float = 0.05,
    strict_min_mode_shots: int = 75,
    strict_min_adjacent_dprime: float = 1.5,
    strict_require_all_modes: bool = True,
    gpu_refined_n_starts: int = 6,
    gpu_refined_iters: int = 8,
    gpu_refined_step_shrink: float = 0.55,
    gpu_refined_pdf_chunk_size: int = 2048,
    include_2nv_unequal: bool = False,

    # Extra conservative layer after GPU fitting.
    # This demotes N=2/3 if the measured histogram does not visibly support
    # the extra charge manifolds.
    posthoc_physical_pruning: bool = True,
    posthoc_require_empirical_peak_count: bool = True,
    posthoc_min_peak_prominence_frac: float = 0.025,
    posthoc_min_peak_distance_counts: int = 8,
    posthoc_min_mode_weight: float = 0.08,
    posthoc_min_mode_shots: int = 100,
    posthoc_min_adjacent_dprime: float = 1.8,

    # Extra posthoc gates for rejecting fake middle modes.
    # These only affect N=2/3 models after the backend fit is done.
    posthoc_use_chi_square_gate: bool = True,
    posthoc_min_red_chi_sq_improvement: float = 0.08,
    posthoc_min_expected_per_bin_for_chi: float = 2.0,
    posthoc_use_middle_mode_support_gate: bool = True,
    posthoc_min_middle_obs_to_expected_ratio: float = 0.35,
    posthoc_middle_window_sigma: float = 1.25,
    posthoc_middle_window_max_frac_delta: float = 0.35,
    posthoc_min_middle_observed_fraction: float = 0.01,
):
    """
    GPU reference-only multi-NV charge-state histogram analysis.

    This wrapper is compatible with the updated backend file:
        analysis/sc_gpu_bimodal_fitting.py

    Important physical rule:
        counts[0] = signal / with ionization pulse, visual only.
        counts[1] = reference / no-ionization branch, used for all fitting.

    Recommended model_mode:
        "gpu_refined"  -- pure-GPU refined multimodal fitting from backend.

    Other useful model_mode values supported by the backend:
        "strict_auto", "auto", "multimode", "bimodal".
    """
    nv_list = raw_data["nv_list"]
    num_positions = len(nv_list)

    counts = np.asarray(raw_data["counts"])
    if counts.shape[0] < 2:
        raise ValueError(
            "Expected raw_data['counts'][0] = signal and raw_data['counts'][1] = reference."
        )

    sig_counts_lists = [counts[0, ind].flatten() for ind in range(num_positions)]
    ref_counts_lists = [counts[1, ind].flatten() for ind in range(num_positions)]

    gpu_fit_config = _make_compatible_gpu_multimode_config(
        max_nvs_per_position=max_nvs_per_position,
        bic_extra_nv_penalty=bic_extra_nv_penalty,
        strict_extra_nv_penalty=strict_extra_nv_penalty,
        strict_bic_margin=strict_bic_margin,
        strict_min_mode_weight=strict_min_mode_weight,
        strict_min_mode_shots=strict_min_mode_shots,
        strict_min_adjacent_dprime=strict_min_adjacent_dprime,
        strict_require_all_modes=strict_require_all_modes,
        gpu_refined_n_starts=gpu_refined_n_starts,
        gpu_refined_iters=gpu_refined_iters,
        gpu_refined_step_shrink=gpu_refined_step_shrink,
        gpu_refined_pdf_chunk_size=gpu_refined_pdf_chunk_size,
        include_2nv_unequal=include_2nv_unequal,
        user_config=gpu_fit_config,
    )

    model_mode = str(model_mode).lower()

    print("\n=== Starting GPU multi-NV reference-only fits ===")
    print("GPU info:", summarize_gpu())
    print(f"Number of pillars: {num_positions}")
    print(f"prob_dist: {prob_dist.name}")
    print(f"model_mode: {model_mode}")
    print(f"max_nvs_per_position: {max_nvs_per_position}")
    print(f"force_nvs: {force_nvs}")
    print(f"bic_extra_nv_penalty: {bic_extra_nv_penalty}")
    print(f"strict_extra_nv_penalty: {strict_extra_nv_penalty}")
    print(f"strict_bic_margin: {strict_bic_margin}")
    print(f"strict_min_mode_weight: {strict_min_mode_weight}")
    print(f"strict_min_mode_shots: {strict_min_mode_shots}")
    print(f"strict_min_adjacent_dprime: {strict_min_adjacent_dprime}")
    print(f"strict_require_all_modes: {strict_require_all_modes}")
    print(f"gpu_refined_n_starts: {gpu_refined_n_starts}")
    print(f"gpu_refined_iters: {gpu_refined_iters}")
    print(f"gpu_refined_step_shrink: {gpu_refined_step_shrink}")
    print(f"gpu_refined_pdf_chunk_size: {gpu_refined_pdf_chunk_size}")
    print(f"include_2nv_unequal: {include_2nv_unequal}")
    print(f"posthoc_physical_pruning: {posthoc_physical_pruning}")
    print(f"posthoc_require_empirical_peak_count: {posthoc_require_empirical_peak_count}")
    print(f"posthoc_min_peak_prominence_frac: {posthoc_min_peak_prominence_frac}")
    print(f"posthoc_min_peak_distance_counts: {posthoc_min_peak_distance_counts}")
    print(f"posthoc_min_mode_weight: {posthoc_min_mode_weight}")
    print(f"posthoc_min_mode_shots: {posthoc_min_mode_shots}")
    print(f"posthoc_min_adjacent_dprime: {posthoc_min_adjacent_dprime}")
    print(f"posthoc_use_chi_square_gate: {posthoc_use_chi_square_gate}")
    print(f"posthoc_min_red_chi_sq_improvement: {posthoc_min_red_chi_sq_improvement}")
    print(f"posthoc_use_middle_mode_support_gate: {posthoc_use_middle_mode_support_gate}")
    print(f"posthoc_min_middle_obs_to_expected_ratio: {posthoc_min_middle_obs_to_expected_ratio}")
    print("Fitting reference/no-ionization branch only: counts[1]")

    # -------------------------------------------------------------------------
    # Main backend call. The updated sc_gpu_bimodal_fitting.py should implement
    # model_mode="gpu_refined" and keep older modes compatible.
    # -------------------------------------------------------------------------
    try:
        gpu_fit_results, gpu_debug = fit_charge_histograms_gpu_batch(
            ref_counts_lists,
            prob_dist=prob_dist.name,
            model_mode=model_mode,
            multimode_config=gpu_fit_config,
            max_nvs=int(max_nvs_per_position),
            force_nvs=force_nvs,
            return_debug=True,
        )
    except TypeError:
        # Compatibility for a backend that does not have all keyword names.
        gpu_fit_results, gpu_debug = fit_charge_histograms_gpu_batch(
            ref_counts_lists,
            prob_dist=prob_dist.name,
            model_mode=model_mode,
            multimode_config=gpu_fit_config,
            max_nvs=int(max_nvs_per_position),
            force_nvs=force_nvs,
            return_debug=True,
        )

    print("GPU fit debug:", gpu_debug)

    candidate_results_list = _extract_candidate_results_from_fits_and_debug(
        gpu_fit_results,
        gpu_debug,
        num_positions,
    )

    gpu_fit_results, posthoc_pruning_debug = _posthoc_demote_unphysical_fits(
        gpu_fit_results=gpu_fit_results,
        candidate_results_list=candidate_results_list,
        ref_counts_lists=ref_counts_lists,
        prob_dist=prob_dist,
        max_nvs_per_position=max_nvs_per_position,
        enable=posthoc_physical_pruning,
        require_empirical_peak_count=posthoc_require_empirical_peak_count,
        min_peak_prominence_frac=posthoc_min_peak_prominence_frac,
        min_peak_distance_counts=posthoc_min_peak_distance_counts,
        min_mode_weight=posthoc_min_mode_weight,
        min_mode_shots=posthoc_min_mode_shots,
        min_adjacent_dprime=posthoc_min_adjacent_dprime,
        use_chi_square_gate=posthoc_use_chi_square_gate,
        min_red_chi_sq_improvement=posthoc_min_red_chi_sq_improvement,
        min_expected_per_bin_for_chi=posthoc_min_expected_per_bin_for_chi,
        use_middle_mode_support_gate=posthoc_use_middle_mode_support_gate,
        min_middle_obs_to_expected_ratio=posthoc_min_middle_obs_to_expected_ratio,
        middle_window_sigma=posthoc_middle_window_sigma,
        middle_window_max_frac_delta=posthoc_middle_window_max_frac_delta,
        min_middle_observed_fraction=posthoc_min_middle_observed_fraction,
    )

    if posthoc_pruning_debug.get("enabled", False):
        print(
            "Posthoc physical pruning demoted",
            posthoc_pruning_debug.get("num_demoted", 0),
            "pillars."
        )

    fit_arrays = _fit_results_to_arrays(gpu_fit_results, num_positions)

    ok_arr = fit_arrays["ok"]
    n_nvs_est_arr = fit_arrays["n_nvs_est"]
    p_minus_arr = fit_arrays["p_minus"]
    bg_arr = fit_arrays["bg"]
    rate0_arr = fit_arrays["rate0"]
    delta_arr = fit_arrays["delta"]
    red_chi_sq_arr = fit_arrays["red_chi_sq"]
    bic_arr = fit_arrays["bic"]
    nll_arr = fit_arrays["nll"]
    model_list = fit_arrays["model"]
    fit_params_arr = fit_arrays["fit_params"]

    # -------------------------------------------------------------------------
    # GPU binary threshold: all NV0 vs any NV-
    # -------------------------------------------------------------------------
    ref_x_max = int(np.nanmax(counts[1])) if np.isfinite(counts[1]).any() else 20
    threshold_any_arr, fidelity_any_arr = determine_thresholds_any_minus_gpu(
        fit_params_arr,
        np.where(np.isfinite(n_nvs_est_arr), n_nvs_est_arr, 0).astype(int),
        prob_dist=prob_dist.name,
        x_max=ref_x_max,
        config=gpu_fit_config,
    )

    threshold_any_arr = np.asarray(threshold_any_arr, dtype=float)
    fidelity_any_arr = np.asarray(fidelity_any_arr, dtype=float)

    # -------------------------------------------------------------------------
    # Output arrays compatible with the CPU script
    # -------------------------------------------------------------------------
    fidelity_multiclass_arr = np.full(num_positions, np.nan)
    prep_fidelity_any_ref_arr = np.full(num_positions, np.nan)

    thresholds_multiclass_list = [None for _ in range(num_positions)]
    weights_list = [None for _ in range(num_positions)]
    ref_prob_k_list = [None for _ in range(num_positions)]
    ref_k_est_list = [None for _ in range(num_positions)]
    ref_p_any_minus_arr = np.full(num_positions, np.nan)
    ref_mean_num_minus_arr = np.full(num_positions, np.nan)
    feedback_params = [None for _ in range(num_positions)]

    best_candidate_model_list = [None for _ in range(num_positions)]
    best_candidate_bic_arr = np.full(num_positions, np.nan)
    best_equal_bic_arr = np.full(num_positions, np.nan)
    unequal_2nv_beats_equal_arr = np.full(num_positions, False, dtype=bool)

    # candidate_results_list was extracted before posthoc pruning.
    hist_figs = [None for _ in range(num_positions)]

    for ind in range(num_positions):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        fit_i = gpu_fit_results[ind] if isinstance(gpu_fit_results[ind], dict) else {}

        if not ok_arr[ind]:
            print(f"\nGPU multi-NV fit not OK for pillar index {ind}")
            reason = fit_i.get("reason", None)
            if reason is not None:
                print("Reason:", reason)
            continue

        n_est = int(n_nvs_est_arr[ind])
        p_minus = safe_float(p_minus_arr[ind])
        bg = safe_float(bg_arr[ind])
        rate0 = safe_float(rate0_arr[ind])
        delta = safe_float(delta_arr[ind])
        red_chi_sq = safe_float(red_chi_sq_arr[ind])
        threshold_any = safe_float(threshold_any_arr[ind])
        fidelity_any = safe_float(fidelity_any_arr[ind])

        popt = fit_params_arr[ind]
        x_max = int(np.nanmax(ref_counts_list)) if len(ref_counts_list) else ref_x_max

        thresholds_multiclass, fidelity_multiclass = determine_multithreshold_equal_multinv(
            popt,
            prob_dist,
            n_est,
            x_max=x_max,
            ret_fidelity=True,
        )

        if thresholds_multiclass is None:
            thresholds_multiclass = [threshold_any]
            fidelity_multiclass = np.nan

        thresholds_multiclass = np.asarray(thresholds_multiclass, dtype=float)
        weights = _binom_weights(n_est, p_minus)

        # Prefer backend-supplied values if present.
        fit_thresholds = fit_i.get("thresholds", None)
        if fit_thresholds is not None:
            try:
                t = np.asarray(fit_thresholds, dtype=float).flatten()
                if t.size > 0 and np.all(np.isfinite(t)):
                    thresholds_multiclass = t
            except Exception:
                pass

        fit_fidelity_multiclass = safe_float(fit_i.get("fidelity_multiclass", np.nan))
        if np.isfinite(fit_fidelity_multiclass):
            fidelity_multiclass = fit_fidelity_multiclass

        fit_weights = fit_i.get("weights", None)
        if fit_weights is not None:
            try:
                w = np.asarray(fit_weights, dtype=float).flatten()
                if w.size == n_est + 1 and np.all(np.isfinite(w)):
                    weights = w
            except Exception:
                pass

        prep_fidelity_any_ref = 1.0 - float(weights[0])

        thresholds_multiclass_list[ind] = thresholds_multiclass
        weights_list[ind] = weights
        fidelity_multiclass_arr[ind] = safe_float(fidelity_multiclass)
        prep_fidelity_any_ref_arr[ind] = prep_fidelity_any_ref

        ref_summary = summarize_ref_classification(
            ref_counts=ref_counts_list,
            threshold_any=threshold_any,
            thresholds_multiclass=thresholds_multiclass,
            n_nvs=n_est,
        )

        ref_p_any_minus_arr[ind] = ref_summary["p_any_minus"]
        ref_mean_num_minus_arr[ind] = ref_summary["mean_num_minus"]
        ref_prob_k_list[ind] = ref_summary["prob_k"]
        ref_k_est_list[ind] = ref_summary["k_est"]

        best_candidate_model = fit_i.get("best_candidate_model", model_list[ind])
        best_candidate_bic = safe_float(fit_i.get("best_candidate_bic", bic_arr[ind]))
        best_equal_bic = safe_float(fit_i.get("best_equal_bic", bic_arr[ind]))
        unequal_2nv_beats_equal = bool(fit_i.get("unequal_2nv_beats_equal", False))

        best_candidate_model_list[ind] = best_candidate_model
        best_candidate_bic_arr[ind] = best_candidate_bic
        best_equal_bic_arr[ind] = best_equal_bic
        unequal_2nv_beats_equal_arr[ind] = unequal_2nv_beats_equal

        strict_reason = (
            fit_i.get("strict_selected_reason", None)
            or fit_i.get("strict_reason", None)
            or fit_i.get("selected_reason", None)
        )
        strict_adjusted_bic = safe_float(fit_i.get("strict_adjusted_bic", np.nan))

        feedback_params[ind] = {
            "pillar_index": int(ind),
            "nv_name": getattr(nv_list[ind], "name", str(ind)),
            "n_nvs_est": int(n_est),
            "threshold_any": float(threshold_any),
            "fidelity_any": float(fidelity_any),
            "thresholds_multiclass": thresholds_multiclass.tolist(),
            "fidelity_multiclass": safe_float(fidelity_multiclass),
            "p_minus": float(p_minus),
            "bg": float(bg),
            "rate0": float(rate0),
            "delta": float(delta),
            "weights_k": weights.tolist(),
            "model": model_list[ind],
            "best_candidate_model": best_candidate_model,
            "best_candidate_bic": safe_float(best_candidate_bic),
            "best_equal_bic": safe_float(best_equal_bic),
            "unequal_2nv_beats_equal": unequal_2nv_beats_equal,
            "ref_p_any_minus": safe_float(ref_summary["p_any_minus"]),
            "ref_mean_num_minus": safe_float(ref_summary["mean_num_minus"]),
            "ref_prob_k": ref_summary["prob_k"].tolist(),
            "prep_fidelity_any_ref": float(prep_fidelity_any_ref),
            "red_chi_sq": float(red_chi_sq),
            "bic": safe_float(bic_arr[ind]),
            "nll": safe_float(nll_arr[ind]),
            "strict_reason": strict_reason,
            "strict_adjusted_bic": strict_adjusted_bic,
        }

        print(
            f"Pillar {ind}: "
            f"model={model_list[ind]}, "
            f"best_candidate={best_candidate_model}, "
            f"2nv_unequal_beats_equal={unequal_2nv_beats_equal}, "
            f"N_est={n_est}, "
            f"bg={bg:.3f}, "
            f"rate0={rate0:.3f}, "
            f"delta={delta:.3f}, "
            f"threshold_any={threshold_any:.3f}, "
            f"thresholds_multi={np.round(thresholds_multiclass, 3)}, "
            f"fid_any={fidelity_any:.3f}, "
            f"fid_multi={safe_float(fidelity_multiclass):.3f}, "
            f"ref P(any NV-)={ref_summary['p_any_minus']:.3f}, "
            f"ref mean k={ref_summary['mean_num_minus']:.3f}, "
            f"weights={np.round(weights, 3)}"
        )

        if do_plot_histograms:
            fig, ax = plot_one_pillar_hist_and_fit_from_values(
                sig_counts_list=sig_counts_list,
                ref_counts_list=ref_counts_list,
                prob_dist=prob_dist,
                n_est=n_est,
                threshold_any=threshold_any,
                thresholds_multiclass=thresholds_multiclass,
                weights=weights,
                p_minus=p_minus,
                bg=bg,
                rate0=rate0,
                delta=delta,
                fidelity_any=fidelity_any,
                fidelity_multi=safe_float(fidelity_multiclass),
                ref_p_any=ref_summary["p_any_minus"],
                ref_mean_k=ref_summary["mean_num_minus"],
                pillar_label=ind,
                density=True,
            )
            hist_figs[ind] = fig

            if save_hist_figs:
                timestamp = dm.get_time_stamp()
                nv_name = getattr(nv_list[ind], "name", f"pillar-{ind}")
                file_path = dm.get_file_path(
                    __file__,
                    timestamp,
                    f"{nv_name}-gpu-ref-only-multinv-charge-hist",
                )
                dm.save_figure(fig, file_path)

            kpl.show(block=True)

    analysis_dict = {
        "analysis_type": "gpu_backend_refined_multi_nv_binomial_ref_only",
        "note": (
            "All fit parameters, thresholds, estimated number of NVs per pillar, "
            "and feedback parameters are from the reference/no-ionization branch only. "
            "The signal/ionization branch is excluded from multi-NV analysis and is used only "
            "for visual comparison. This wrapper calls analysis.sc_gpu_bimodal_fitting directly."
        ),
        "used_gpu": True,
        "gpu_info": summarize_gpu(),
        "gpu_fit_debug": make_json_safe(gpu_debug),
        "gpu_fit_config": make_json_safe(getattr(gpu_fit_config, "__dict__", {})),
        "model_mode": str(model_mode),
        "prob_dist": prob_dist.name,
        "max_nvs_per_position": int(max_nvs_per_position),
        "force_nvs": None if force_nvs is None else int(force_nvs),
        "bic_extra_nv_penalty": float(bic_extra_nv_penalty),
        "strict_extra_nv_penalty": float(strict_extra_nv_penalty),
        "strict_bic_margin": float(strict_bic_margin),
        "strict_min_mode_weight": float(strict_min_mode_weight),
        "strict_min_mode_shots": int(strict_min_mode_shots),
        "strict_min_adjacent_dprime": float(strict_min_adjacent_dprime),
        "strict_require_all_modes": bool(strict_require_all_modes),
        "gpu_refined_n_starts": int(gpu_refined_n_starts),
        "gpu_refined_iters": int(gpu_refined_iters),
        "gpu_refined_step_shrink": float(gpu_refined_step_shrink),
        "gpu_refined_pdf_chunk_size": int(gpu_refined_pdf_chunk_size),
        "include_2nv_unequal": bool(include_2nv_unequal),
        "posthoc_physical_pruning": bool(posthoc_physical_pruning),
        "posthoc_require_empirical_peak_count": bool(posthoc_require_empirical_peak_count),
        "posthoc_min_peak_prominence_frac": float(posthoc_min_peak_prominence_frac),
        "posthoc_min_peak_distance_counts": int(posthoc_min_peak_distance_counts),
        "posthoc_min_mode_weight": float(posthoc_min_mode_weight),
        "posthoc_min_mode_shots": int(posthoc_min_mode_shots),
        "posthoc_min_adjacent_dprime": float(posthoc_min_adjacent_dprime),
        "posthoc_use_chi_square_gate": bool(posthoc_use_chi_square_gate),
        "posthoc_min_red_chi_sq_improvement": float(posthoc_min_red_chi_sq_improvement),
        "posthoc_min_expected_per_bin_for_chi": float(posthoc_min_expected_per_bin_for_chi),
        "posthoc_use_middle_mode_support_gate": bool(posthoc_use_middle_mode_support_gate),
        "posthoc_min_middle_obs_to_expected_ratio": float(posthoc_min_middle_obs_to_expected_ratio),
        "posthoc_middle_window_sigma": float(posthoc_middle_window_sigma),
        "posthoc_middle_window_max_frac_delta": float(posthoc_middle_window_max_frac_delta),
        "posthoc_min_middle_observed_fraction": float(posthoc_min_middle_observed_fraction),
        "posthoc_pruning_debug": make_json_safe(posthoc_pruning_debug),
        "n_jobs": None if n_jobs is None else int(n_jobs),
        "ok": ok_arr,
        "n_nvs_est": n_nvs_est_arr,
        "threshold_any": threshold_any_arr,
        "thresholds_multiclass": thresholds_multiclass_list,
        "readout_fidelity_any": fidelity_any_arr,
        "fidelity_multiclass": fidelity_multiclass_arr,
        "prep_fidelity_any_ref": prep_fidelity_any_ref_arr,
        "p_minus": p_minus_arr,
        "bg": bg_arr,
        "rate0": rate0_arr,
        "delta": delta_arr,
        "weights_k": weights_list,
        "red_chi_sq": red_chi_sq_arr,
        "bic": bic_arr,
        "nll": nll_arr,
        "fit_params_arr": fit_params_arr,
        "model": model_list,
        "best_candidate_model": best_candidate_model_list,
        "best_candidate_bic": best_candidate_bic_arr,
        "best_equal_bic": best_equal_bic_arr,
        "unequal_2nv_beats_equal": unequal_2nv_beats_equal_arr,
        "candidate_results": candidate_results_list,
        "ref_p_any_minus": ref_p_any_minus_arr,
        "ref_mean_num_minus": ref_mean_num_minus_arr,
        "ref_prob_k": ref_prob_k_list,
        "ref_k_est": ref_k_est_list,
        "feedback_params": feedback_params,
    }

    raw_data["charge_hist_multinv_binomial"] = analysis_dict

    print("\n=== GPU multi-NV reference-only histogram summary ===")
    print("Good fits:", int(np.sum(ok_arr)), "/", num_positions)
    for n in range(1, max_nvs_per_position + 1):
        num_n = int(np.sum(ok_arr & (np.rint(n_nvs_est_arr).astype(int) == n)))
        print(f"Estimated pillars with {n} NV(s): {num_n}")
    print("Median threshold_any:", np.nanmedian(threshold_any_arr))
    print("Median fidelity_any:", np.nanmedian(fidelity_any_arr))
    print("Median fidelity_multiclass:", np.nanmedian(fidelity_multiclass_arr))
    print("Median ref mean k:", np.nanmedian(ref_mean_num_minus_arr))
    print("Median ref P(any NV-):", np.nanmedian(ref_p_any_minus_arr))

    if np.any(unequal_2nv_beats_equal_arr):
        print(
            "Pillars where 2NV unequal diagnostic beats equal model:",
            int(np.sum(unequal_2nv_beats_equal_arr)),
        )

    good = ok_arr & np.isfinite(fidelity_any_arr) & np.isfinite(fidelity_multiclass_arr)
    if np.any(good):
        fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
        kpl.plot_points(ax, fidelity_any_arr[good], fidelity_multiclass_arr[good])
        ax.set_xlabel("Binary any-NV$^{-}$ fidelity")
        ax.set_ylabel("Multi-class fidelity")
        ax.set_title("GPU charge-readout fidelity by pillar")

    if save_analysis:
        timestamp = dm.get_time_stamp()
        try:
            repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
            repr_nv_name = repr_nv_sig.name
        except Exception:
            repr_nv_name = "gpu-multinv-charge-analysis"

        analysis_dict_for_save = dict(analysis_dict)
        analysis_dict_for_save["ref_k_est"] = None

        nv_names = [getattr(nv, "name", str(ind)) for ind, nv in enumerate(nv_list)]

        analysis_raw_data = {
            "timestamp": timestamp,
            "source_timestamp": raw_data.get("timestamp", None),
            "source_file_id": raw_data.get("file_id", None),
            "nv_names": nv_names,
            "charge_hist_multinv_binomial": make_json_safe(analysis_dict_for_save),
        }

        file_path = dm.get_file_path(
            __file__,
            timestamp,
            f"{repr_nv_name}-gpu-backend-refined-multinv-charge-analysis",
        )

        dm.save_raw_data(analysis_raw_data, file_path, keys_to_compress=[])
        print("Saved GPU reference-only multi-NV charge analysis:", file_path)

    return hist_figs




# =============================================================================
# Analysis accessors
# =============================================================================


def get_metric_label(key):
    return METRIC_INFO.get(key, (key, ""))[0]


def get_metric_note(key):
    return METRIC_INFO.get(key, (key, ""))[1]


def add_metric_note(ax, key, fontsize=7):
    note = get_metric_note(key)
    if note:
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )


def print_metric_definitions():
    print("\n=== Metric definitions ===")
    for key, (label, note) in METRIC_INFO.items():
        print(f"{key:25s} : {label} — {note}")


def get_charge_analysis(raw_data):
    if "charge_hist_multinv_binomial" not in raw_data:
        raise KeyError(
            "raw_data does not contain 'charge_hist_multinv_binomial'. "
            "Run process_and_plot(...) first or load the saved analysis file."
        )
    return raw_data["charge_hist_multinv_binomial"]


def arr_from_analysis(analysis, key):
    return np.asarray(analysis[key], dtype=float)


def get_good_mask(analysis):
    return np.asarray(analysis["ok"], dtype=bool)


def extract_multiclass_threshold_array(analysis, max_nvs_per_position=None):
    thresholds_list = analysis["thresholds_multiclass"]
    num_pillars = len(thresholds_list)

    if max_nvs_per_position is None:
        max_nvs_per_position = int(analysis.get("max_nvs_per_position", 3))

    threshold_mat = np.full((num_pillars, max_nvs_per_position), np.nan)

    for ind, thresholds in enumerate(thresholds_list):
        if thresholds is None:
            continue
        thresholds = np.asarray(thresholds, dtype=float).flatten()
        num_t = min(len(thresholds), max_nvs_per_position)
        threshold_mat[ind, :num_t] = thresholds[:num_t]

    return threshold_mat


def extract_weights_array(analysis, max_nvs_per_position=None):
    weights_list = analysis["weights_k"]
    num_pillars = len(weights_list)

    if max_nvs_per_position is None:
        max_nvs_per_position = int(analysis.get("max_nvs_per_position", 3))

    weights_mat = np.full((num_pillars, max_nvs_per_position + 1), np.nan)

    for ind, weights in enumerate(weights_list):
        if weights is None:
            continue
        weights = np.asarray(weights, dtype=float).flatten()
        num_w = min(len(weights), max_nvs_per_position + 1)
        weights_mat[ind, :num_w] = weights[:num_w]

    return weights_mat


def get_xy_from_nv_list(nv_list, coords_key=None):
    if coords_key is None:
        return None

    xy = []
    for nv in nv_list:
        coord = None
        try:
            coord = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
        except Exception:
            pass
        if coord is None:
            try:
                coord = getattr(nv, coords_key)
            except Exception:
                pass
        if coord is None:
            try:
                coord = nv[coords_key]
            except Exception:
                pass
        if coord is None:
            xy.append([np.nan, np.nan])
            continue
        coord = np.asarray(coord, dtype=float).flatten()
        xy.append([coord[0], coord[1]] if len(coord) >= 2 else [np.nan, np.nan])
    return np.asarray(xy, dtype=float)


# =============================================================================
# Compact scatter plots / summaries
# =============================================================================


def scatter_metric_vs_index(raw_data, key, ylabel=None, title=None, good_only=True, add_note=True):
    analysis = get_charge_analysis(raw_data)
    vals = arr_from_analysis(analysis, key)
    inds = np.arange(len(vals))
    good = np.isfinite(vals)
    if good_only:
        good &= get_good_mask(analysis)

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    ax.scatter(inds[good], vals[good], s=POINT_SIZE, alpha=POINT_ALPHA)
    label = ylabel if ylabel is not None else get_metric_label(key)
    ax.set_xlabel("Pillar index")
    ax.set_ylabel(label)
    ax.set_title(title if title is not None else label, fontsize=15)
    if add_note:
        add_metric_note(ax, key)
    return fig, ax


def scatter_metric_spatial(
    raw_data,
    key,
    coords_key,
    cbar_label=None,
    title=None,
    good_only=True,
    marker_size=9,
    add_note=True,
):
    analysis = get_charge_analysis(raw_data)
    nv_list = raw_data["nv_list"]
    vals = arr_from_analysis(analysis, key)
    xy = get_xy_from_nv_list(nv_list, coords_key=coords_key)
    if xy is None:
        raise ValueError("Could not get xy coordinates. Provide a valid coords_key.")

    good = np.isfinite(vals) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    if good_only:
        good &= get_good_mask(analysis)

    fig, ax = plt.subplots(figsize=SPATIAL_FIGSIZE, constrained_layout=False)
    try:
        fig.set_layout_engine(None)
    except Exception:
        pass

    sc = ax.scatter(xy[good, 0], xy[good, 1], c=vals[good], s=marker_size, alpha=POINT_ALPHA)
    label = cbar_label if cbar_label is not None else get_metric_label(key)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel(f"{coords_key} x")
    ax.set_ylabel(f"{coords_key} y")
    ax.set_title(title if title is not None else label, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    if add_note:
        add_metric_note(ax, key)
    try:
        fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.88)
    except Exception:
        pass
    return fig, ax


def scatter_two_metrics(raw_data, x_key, y_key, xlabel=None, ylabel=None, title=None, good_only=True):
    analysis = get_charge_analysis(raw_data)
    x = arr_from_analysis(analysis, x_key)
    y = arr_from_analysis(analysis, y_key)
    good = np.isfinite(x) & np.isfinite(y)
    if good_only:
        good &= get_good_mask(analysis)

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    ax.scatter(x[good], y[good], s=POINT_SIZE, alpha=POINT_ALPHA)
    xlabel = xlabel if xlabel is not None else get_metric_label(x_key)
    ylabel = ylabel if ylabel is not None else get_metric_label(y_key)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else f"{ylabel} vs {xlabel}", fontsize=15)
    return fig, ax


def plot_n_nv_count_summary(raw_data):
    analysis = get_charge_analysis(raw_data)
    ok = get_good_mask(analysis)
    n_nvs = arr_from_analysis(analysis, "n_nvs_est")
    max_n = int(np.nanmax(n_nvs[ok])) if np.any(ok) else 0
    ns = np.arange(1, max_n + 1)
    counts = np.array([np.sum(ok & (n_nvs == n)) for n in ns], dtype=int)

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    ax.bar(ns, counts)
    ax.set_xlabel("Estimated NVs per pillar")
    ax.set_ylabel("Number of pillars")
    ax.set_title("GPU multi-NV occupancy", fontsize=15)
    for n, c in zip(ns, counts):
        ax.text(n, c, str(c), ha="center", va="bottom", fontsize=8)
    return fig, ax


def plot_thresholds_vs_index(raw_data):
    analysis = get_charge_analysis(raw_data)
    good = get_good_mask(analysis)
    threshold_any = arr_from_analysis(analysis, "threshold_any")
    threshold_mat = extract_multiclass_threshold_array(analysis)
    inds = np.arange(len(threshold_any))

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    mask = good & np.isfinite(threshold_any)
    ax.scatter(inds[mask], threshold_any[mask], s=POINT_SIZE, alpha=POINT_ALPHA, label="any NV$^{-}$")

    for t_ind in range(threshold_mat.shape[1]):
        vals = threshold_mat[:, t_ind]
        mask = good & np.isfinite(vals)
        if not np.any(mask):
            continue
        ax.scatter(
            inds[mask],
            vals[mask],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            label=f"k={t_ind}|{t_ind + 1}",
        )

    ax.set_xlabel("Pillar index")
    ax.set_ylabel("Integrated counts")
    ax.set_title("GPU feedback thresholds", fontsize=15)
    ax.legend(fontsize=7)
    return fig, ax


def plot_weights_vs_index(raw_data):
    analysis = get_charge_analysis(raw_data)
    good = get_good_mask(analysis)
    weights_mat = extract_weights_array(analysis)
    inds = np.arange(weights_mat.shape[0])

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    for k in range(weights_mat.shape[1]):
        vals = weights_mat[:, k]
        mask = good & np.isfinite(vals)
        if not np.any(mask):
            continue
        ax.scatter(inds[mask], vals[mask], s=POINT_SIZE, alpha=POINT_ALPHA, label=f"P(k={k})")

    ax.set_xlabel("Pillar index")
    ax.set_ylabel("Fitted probability")
    ax.set_title("GPU reference charge-state weights", fontsize=15)
    ax.legend(fontsize=7)
    return fig, ax


def plot_fidelity_any_vs_multiclass(raw_data):
    analysis = get_charge_analysis(raw_data)
    ok = get_good_mask(analysis)
    fidelity_any = arr_from_analysis(analysis, "readout_fidelity_any")
    fidelity_multiclass = arr_from_analysis(analysis, "fidelity_multiclass")
    good = ok & np.isfinite(fidelity_any) & np.isfinite(fidelity_multiclass)

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)
    ax.scatter(fidelity_any[good], fidelity_multiclass[good], s=POINT_SIZE, alpha=POINT_ALPHA)
    ax.set_xlabel("Binary any-NV$^{-}$ fidelity")
    ax.set_ylabel("Multi-class fidelity")
    ax.set_title("GPU readout fidelity by pillar", fontsize=15)
    return fig, ax


# =============================================================================
# Histogram + fit plotting
# =============================================================================


def plot_one_pillar_hist_and_fit_from_values(
    sig_counts_list,
    ref_counts_list,
    prob_dist,
    n_est,
    threshold_any,
    thresholds_multiclass,
    weights,
    p_minus,
    bg,
    rate0,
    delta,
    fidelity_any,
    fidelity_multi,
    ref_p_any,
    ref_mean_k,
    pillar_label,
    density=True,
):
    fig = plot_histograms(sig_counts_list, ref_counts_list, density=density)
    ax = fig.gca()

    x_max = max(np.nanmax(sig_counts_list), np.nanmax(ref_counts_list))
    x_vals = np.linspace(0, x_max, 1000)
    single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)

    base = bg + n_est * rate0
    combined = np.zeros_like(x_vals, dtype=float)

    for k in range(n_est + 1):
        lam_k = base + k * delta
        comp = float(weights[k]) * single_pdf(x_vals, lam_k)
        combined += comp
        kpl.plot_line(ax, x_vals, comp, label=f"fit k={k}")

    kpl.plot_line(ax, x_vals, combined, color=kpl.KplColors.BLUE, label="combined ref fit")

    for t in np.asarray(thresholds_multiclass, dtype=float):
        if np.isfinite(t):
            ax.axvline(t, color=kpl.KplColors.GRAY, ls="dashed", lw=1)

    if np.isfinite(threshold_any):
        ax.axvline(threshold_any, color="black", ls="dashed", lw=2, label="any NV- threshold")

    txt = (
        f"Pillar/NV {pillar_label}\n"
        f"N_est = {n_est}\n"
        f"fid_any = {fidelity_any:.3f}\n"
        f"fid_multi = {fidelity_multi:.3f}\n"
        f"P(any NV-) = {ref_p_any:.3f}\n"
        f"mean k = {ref_mean_k:.2f}\n"
        f"p_minus = {p_minus:.3f}\n"
        f"rate0 = {rate0:.2f}\n"
        f"delta = {delta:.2f}"
    )

    try:
        kpl.anchored_text(ax, txt, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)
    except Exception:
        ax.text(
            0.98,
            0.50,
            txt,
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=7)
    return fig, ax


def plot_one_pillar_hist_and_fit(raw_data, pillar_ind, density=True):
    """
    Plot one selected pillar using the saved GPU analysis.

    Red histogram   = signal branch, visual only
    Green histogram = reference branch, fitted
    Fit components  = reference-only multi-NV model
    Dashed lines    = thresholds
    """
    analysis = get_charge_analysis(raw_data)
    if "counts" not in raw_data:
        raise KeyError("raw_data does not contain 'counts'. Load original raw data and attach analysis.")

    counts = np.asarray(raw_data["counts"])
    nv_list = raw_data["nv_list"]

    sig_counts = counts[0, pillar_ind].flatten()
    ref_counts = counts[1, pillar_ind].flatten()

    ok = bool(np.asarray(analysis["ok"])[pillar_ind])
    if not ok:
        raise ValueError(f"Pillar {pillar_ind} does not have a good fit.")

    n_est = int(np.asarray(analysis["n_nvs_est"])[pillar_ind])
    threshold_any = float(np.asarray(analysis["threshold_any"])[pillar_ind])
    thresholds = np.asarray(analysis["thresholds_multiclass"][pillar_ind], dtype=float)
    weights = np.asarray(analysis["weights_k"][pillar_ind], dtype=float)

    p_minus = float(np.asarray(analysis["p_minus"])[pillar_ind])
    bg = float(np.asarray(analysis["bg"])[pillar_ind])
    rate0 = float(np.asarray(analysis["rate0"])[pillar_ind])
    delta = float(np.asarray(analysis["delta"])[pillar_ind])

    fidelity_any = float(np.asarray(analysis["readout_fidelity_any"])[pillar_ind])
    fidelity_multi = float(np.asarray(analysis["fidelity_multiclass"])[pillar_ind])
    ref_p_any = float(np.asarray(analysis["ref_p_any_minus"])[pillar_ind])
    ref_mean_k = float(np.asarray(analysis["ref_mean_num_minus"])[pillar_ind])

    prob_dist_name = analysis.get("prob_dist", "COMPOUND_POISSON")
    prob_dist = ProbDist[prob_dist_name]

    try:
        nv_num = widefield.get_nv_num(nv_list[pillar_ind])
    except Exception:
        nv_num = pillar_ind

    return plot_one_pillar_hist_and_fit_from_values(
        sig_counts_list=sig_counts,
        ref_counts_list=ref_counts,
        prob_dist=prob_dist,
        n_est=n_est,
        threshold_any=threshold_any,
        thresholds_multiclass=thresholds,
        weights=weights,
        p_minus=p_minus,
        bg=bg,
        rate0=rate0,
        delta=delta,
        fidelity_any=fidelity_any,
        fidelity_multi=fidelity_multi,
        ref_p_any=ref_p_any,
        ref_mean_k=ref_mean_k,
        pillar_label=nv_num,
        density=density,
    )


# =============================================================================
# All summaries / selected examples
# =============================================================================


def plot_all_charge_multinv_summaries(
    raw_data,
    coords_key=None,
    save_figs=False,
    plot_set="minimal",
):
    """
    Summary plots after charge-state analysis.

    plot_set="minimal" keeps only the useful diagnostics:
        1. Occupancy count summary.
        2. Any-NV- threshold vs pillar index.
        3. Binary readout fidelity vs pillar index.
        4. Delta/separation vs pillar index.
        5. Spatial occupancy map, if coords_key is provided.
        6. Spatial threshold map, if coords_key is provided.

    plot_set="full" restores the older behavior with many scatter plots.
    """
    figs = []
    plot_set = str(plot_set).lower()

    if plot_set not in {"minimal", "compact", "full"}:
        raise ValueError("plot_set must be 'minimal', 'compact', or 'full'.")

    # Always useful.
    for plot_fn, name in [
        (plot_n_nv_count_summary, "NV occupancy summary"),
        (plot_thresholds_vs_index, "thresholds"),
    ]:
        try:
            fig, ax = plot_fn(raw_data)
            figs.append(fig)
        except Exception:
            print(f"Could not plot {name}")
            print(traceback.format_exc())

    # Minimal scalar QC plots.
    keys_to_plot = [
        ("readout_fidelity_any", get_metric_label("readout_fidelity_any")),
        ("delta", get_metric_label("delta")),
    ]

    if plot_set in {"compact", "full"}:
        keys_to_plot.extend(
            [
                ("n_nvs_est", get_metric_label("n_nvs_est")),
                ("threshold_any", get_metric_label("threshold_any")),
                ("ref_mean_num_minus", get_metric_label("ref_mean_num_minus")),
                ("p_minus", get_metric_label("p_minus")),
                ("rate0", get_metric_label("rate0")),
                ("bic", get_metric_label("bic")),
            ]
        )

    if plot_set == "full":
        keys_to_plot.extend(
            [
                ("fidelity_multiclass", get_metric_label("fidelity_multiclass")),
                ("prep_fidelity_any_ref", get_metric_label("prep_fidelity_any_ref")),
                ("ref_p_any_minus", get_metric_label("ref_p_any_minus")),
                ("red_chi_sq", get_metric_label("red_chi_sq")),
            ]
        )

    seen = set()
    for key, label in keys_to_plot:
        if key in seen:
            continue
        seen.add(key)
        try:
            fig, ax = scatter_metric_vs_index(raw_data, key, ylabel=label, title=label)
            figs.append(fig)
        except Exception:
            print(f"Could not plot {key}")
            print(traceback.format_exc())

    # Only in full mode: noisy pairwise scatter plots.
    if plot_set == "full":
        for plot_fn, name in [
            (plot_weights_vs_index, "weights"),
            (plot_fidelity_any_vs_multiclass, "fidelity_any vs fidelity_multiclass"),
        ]:
            try:
                fig, ax = plot_fn(raw_data)
                figs.append(fig)
            except Exception:
                print(f"Could not plot {name}")
                print(traceback.format_exc())

        pair_plots = [
            ("delta", "readout_fidelity_any", "delta", "Binary fidelity"),
            ("delta", "fidelity_multiclass", "delta", "Multi-class fidelity"),
            ("p_minus", "fidelity_multiclass", "p_minus", "Multi-class fidelity"),
            ("threshold_any", "delta", "Any-NV$^{-}$ threshold", "delta"),
            ("n_nvs_est", "threshold_any", "Estimated NVs", "Any-NV$^{-}$ threshold"),
            ("n_nvs_est", "ref_mean_num_minus", "Estimated NVs", "Mean k"),
        ]
        for x_key, y_key, xlabel, ylabel in pair_plots:
            try:
                fig, ax = scatter_two_metrics(raw_data, x_key, y_key, xlabel=xlabel, ylabel=ylabel)
                figs.append(fig)
            except Exception:
                print(f"Could not plot {y_key} vs {x_key}")
                print(traceback.format_exc())

    # Spatial maps: keep only two most useful by default.
    if coords_key is not None:
        if plot_set == "full":
            spatial_keys = [
                ("n_nvs_est", get_metric_label("n_nvs_est")),
                ("threshold_any", get_metric_label("threshold_any")),
                ("readout_fidelity_any", get_metric_label("readout_fidelity_any")),
                ("fidelity_multiclass", get_metric_label("fidelity_multiclass")),
                ("ref_mean_num_minus", get_metric_label("ref_mean_num_minus")),
                ("p_minus", get_metric_label("p_minus")),
                ("rate0", get_metric_label("rate0")),
                ("delta", get_metric_label("delta")),
                ("red_chi_sq", get_metric_label("red_chi_sq")),
                ("bic", get_metric_label("bic")),
            ]
        else:
            spatial_keys = [
                ("n_nvs_est", get_metric_label("n_nvs_est")),
                ("threshold_any", get_metric_label("threshold_any")),
            ]

        for key, label in spatial_keys:
            try:
                fig, ax = scatter_metric_spatial(
                    raw_data,
                    key,
                    coords_key=coords_key,
                    cbar_label=label,
                    title=f"{label} map",
                )
                figs.append(fig)
            except Exception:
                print(f"Could not make spatial map for {key}")
                print(traceback.format_exc())

    if save_figs:
        timestamp = dm.get_time_stamp()
        for ind, fig in enumerate(figs):
            file_path = dm.get_file_path(
                __file__,
                timestamp,
                f"gpu-charge-multinv-{plot_set}-summary-{ind:02d}",
            )
            dm.save_figure(fig, file_path)

    return figs



def find_n_nv_pillars(analysis, n):
    ok = np.asarray(analysis["ok"], dtype=bool)
    n_nvs_est = np.asarray(analysis["n_nvs_est"], dtype=float)
    return np.where(ok & (np.rint(n_nvs_est).astype(int) == int(n)))[0]


def print_best_n_nv_examples(analysis, n, num_examples=10, sort_key="fidelity_multiclass"):
    inds = find_n_nv_pillars(analysis, n)
    if inds.size == 0:
        print(f"\nNo {n}-NV pillars found.")
        return np.asarray([], dtype=int)

    metric = np.asarray(analysis[sort_key], dtype=float)
    ordered = inds[np.argsort(metric[inds])[::-1]][:num_examples]

    fidelity_multi = np.asarray(analysis["fidelity_multiclass"], dtype=float)
    fidelity_any = np.asarray(analysis["readout_fidelity_any"], dtype=float)
    threshold_any = np.asarray(analysis["threshold_any"], dtype=float)
    ref_mean_k = np.asarray(analysis["ref_mean_num_minus"], dtype=float)

    print(f"\nBest {n}-NV examples by {sort_key}:")
    for ind in ordered:
        print(
            f"pillar {ind}: "
            f"fid_multi={fidelity_multi[ind]:.3f}, "
            f"fid_any={fidelity_any[ind]:.3f}, "
            f"threshold_any={threshold_any[ind]:.1f}, "
            f"mean_k={ref_mean_k[ind]:.2f}"
        )
    return ordered


def save_selected_pillar_histograms(raw_data, pillar_inds, label, density=True, close_figs=True):
    timestamp = dm.get_time_stamp()
    saved_paths = []
    analysis = raw_data["charge_hist_multinv_binomial"]
    n_nvs_est = np.asarray(analysis["n_nvs_est"], dtype=float)
    fidelity_multi = np.asarray(analysis["fidelity_multiclass"], dtype=float)
    fidelity_any = np.asarray(analysis["readout_fidelity_any"], dtype=float)

    for pillar_ind in pillar_inds:
        pillar_ind = int(pillar_ind)
        fig, ax = plot_one_pillar_hist_and_fit(raw_data, pillar_ind=pillar_ind, density=density)
        n_est = int(n_nvs_est[pillar_ind])
        fid_multi = fidelity_multi[pillar_ind]
        fid_any = fidelity_any[pillar_ind]
        file_label = f"{label}-pillar-{pillar_ind:04d}-N{n_est}-fidmulti-{fid_multi:.3f}-fidany-{fid_any:.3f}"
        file_path = dm.get_file_path(__file__, timestamp, file_label)
        dm.save_figure(fig, file_path)
        saved_paths.append(file_path)
        print("Saved:", file_path)
        if close_figs:
            plt.close(fig)
    return saved_paths


def print_three_nv_statistical_counts(analysis, pillar_ind, num_shots=None):
    n_est = int(np.asarray(analysis["n_nvs_est"])[pillar_ind])
    if n_est != 3:
        print(f"Warning: pillar {pillar_ind} has n_est={n_est}, not 3.")

    p_minus = float(np.asarray(analysis["p_minus"])[pillar_ind])
    bg = float(np.asarray(analysis["bg"])[pillar_ind])
    rate0 = float(np.asarray(analysis["rate0"])[pillar_ind])
    delta = float(np.asarray(analysis["delta"])[pillar_ind])
    if num_shots is None:
        num_shots = 1.0

    print(f"\nPillar {pillar_ind}")
    print(f"N_est = {n_est}")
    print(f"rate0 = {rate0:.3f}")
    print(f"delta = {delta:.3f}")
    print(f"p_minus = {p_minus:.3f}\n")
    print("Expected 3-NV Poisson components:")
    print("k   mean lambda_k   sigma=sqrt(lambda)   P(k)      expected shots")

    for k in range(4):
        lam_k = bg + 3 * rate0 + k * delta
        sigma_k = np.sqrt(lam_k)
        prob_k = math.comb(3, k) * (p_minus**k) * ((1 - p_minus) ** (3 - k))
        shots_k = num_shots * prob_k
        print(f"{k}   {lam_k:10.2f}   {sigma_k:10.2f}        {prob_k:7.3f}   {shots_k:10.1f}")

    print("\nPeak separability:")
    for k in range(3):
        lam0 = bg + 3 * rate0 + k * delta
        lam1 = bg + 3 * rate0 + (k + 1) * delta
        dprime = delta / np.sqrt(lam0 + lam1)
        print(f"k={k} to k={k+1}: separation={delta:.2f} counts, d'={dprime:.2f}")


if __name__ == "__main__":
    kpl.init_kplotlib()

    raw_data = dm.get_raw_data(
        file_stem="2026_07_02-14_45_59-qnami-nv0_2026_02_20",
        load_npz=True,
    )

    gpu_fit_config = GpuMultimodeFitConfig(
        max_nvs=3,

        num_p=9,
        num_bg=3,
        num_rate0=10,
        num_delta=10,

        fit_chunk_size=128,
        candidate_chunk_size=128,
        refine_fit_chunk_size=32,

        use_refinement=True,
        refine_iters=3,

        broadened_sigma0=4.0,
        broadened_fano=0.0,
        broadened_sigma_frac=0.0,

        hierarchical_n1_red_chi_sq_stop=1.35,
        hierarchical_n2_red_chi_sq_stop=1.50,
        hierarchical_min_bic_improvement=25.0,
        hierarchical_min_red_chi_sq_improvement=0.08,
    )

    process_and_plot(
        raw_data,
        do_plot_histograms=False,
        prob_dist=ProbDist.BROADENED_COMPOUND_POISSON,
        max_nvs_per_position=3,
        force_nvs=None,
        bic_extra_nv_penalty=2.0,
        save_analysis=True,
        save_hist_figs=False,
        n_jobs=12,  # kept so old call signature still works

        model_mode="hierarchical",
        gpu_fit_config=gpu_fit_config,

        strict_extra_nv_penalty=80.0,
        strict_bic_margin=20.0,
        strict_min_mode_weight=0.04,
        strict_min_mode_shots=50,
        strict_min_adjacent_dprime=1.35,
        strict_require_all_modes=True,

        gpu_refined_n_starts=6,
        gpu_refined_iters=8,
        gpu_refined_pdf_chunk_size=2048,

        # Posthoc pruning
        posthoc_physical_pruning=True,
        posthoc_require_empirical_peak_count=True,
        posthoc_min_peak_prominence_frac=0.025,
        posthoc_min_peak_distance_counts=8,
        posthoc_min_mode_weight=0.08,
        posthoc_min_mode_shots=100,
        posthoc_min_adjacent_dprime=1.8,

        # New chi-square / middle-mode gates
        posthoc_use_chi_square_gate=True,
        posthoc_min_red_chi_sq_improvement=0.08,
        posthoc_min_expected_per_bin_for_chi=2.0,
        posthoc_use_middle_mode_support_gate=True,
        posthoc_min_middle_obs_to_expected_ratio=0.35,
        posthoc_middle_window_sigma=1.25,
        posthoc_middle_window_max_frac_delta=0.35,
        posthoc_min_middle_observed_fraction=0.01,

        include_2nv_unequal=False,
    )

    raw_data = dm.get_raw_data(
        file_stem="2026_07_02-18_43_33-qnami-nv0_2026_02_20-gpu-backend-refined-multinv-charge-analysis",
        load_npz=True,
    )


    analysis = raw_data["charge_hist_multinv_binomial"]

    figs = plot_all_charge_multinv_summaries(
        raw_data,
        coords_key="pixel",
        save_figs=True,
        plot_set="minimal",
    )

    # -------------------------------------------------------------------------
    # Find and print best examples for 1-NV, 2-NV, and 3-NV pillars
    # -------------------------------------------------------------------------
    one_nv_inds = find_n_nv_pillars(analysis, 1)
    two_nv_inds = find_n_nv_pillars(analysis, 2)
    three_nv_inds = find_n_nv_pillars(analysis, 3)

    print("\nNumber of 1-NV pillars:", len(one_nv_inds))
    print("Number of 2-NV pillars:", len(two_nv_inds))
    print("Number of 3-NV pillars:", len(three_nv_inds))

    one_nv_best = print_best_n_nv_examples(
        analysis,
        n=1,
        num_examples=10,
        sort_key="fidelity_multiclass",
    )

    two_nv_best = print_best_n_nv_examples(
        analysis,
        n=2,
        num_examples=10,
        sort_key="fidelity_multiclass",
    )

    three_nv_best = print_best_n_nv_examples(
        analysis,
        n=3,
        num_examples=10,
        sort_key="fidelity_multiclass",
    )

    # -------------------------------------------------------------------------
    # Plot best 1-NV, 2-NV, and 3-NV examples
    # -------------------------------------------------------------------------
    for pillar_ind in one_nv_best:
        plot_one_pillar_hist_and_fit(
            raw_data,
            pillar_ind=int(pillar_ind),
            density=True,
        )

    for pillar_ind in two_nv_best:
        plot_one_pillar_hist_and_fit(
            raw_data,
            pillar_ind=int(pillar_ind),
            density=True,
        )

    for pillar_ind in three_nv_best:
        plot_one_pillar_hist_and_fit(
            raw_data,
            pillar_ind=int(pillar_ind),
            density=True,
        )

    kpl.show(block=True)