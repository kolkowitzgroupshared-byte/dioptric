# -*- coding: utf-8 -*-
"""
Clean analysis for ONE very large particle-memory NV dataset.

Designed for the 11-GB / ~4000-run file:

    2026_08_20-16_37_57-qnami-nv0_2026_02_20-
    particle-memory-source_off_wait_0s-wait-0s

The script intentionally does NOT contain the older wait sweeps, duplicate
helpers, or unrelated single-run experiments.

Important large-file behavior
-----------------------------
* counts are loaded selectively from the NPZ;
* img_arrays PIXEL DATA IS NOT READ in the default counts-only mode;
* only rep 11 / rep 12 count vectors plus small metadata are loaded;
* drift / brightness / image-correlation / raw-image diagnostics are skipped;
* every figure explicitly calls fig.tight_layout().

Saved-array conventions
-----------------------
counts[exp, nv, run, step, rep]
img_arrays[exp, run, step, rep, y, x]

rep 11 : immediate charge check after initialization
rep 12 : charge check after the dark wait
"""

from __future__ import annotations

import gc
import os
import time
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import betabinom, binom, nbinom, norm, poisson

from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Analyze 0-s, 30-s, and 60-s source-off measurements.
#
# "npz_path_override" should normally remain None. Set it only if automatic
# Dioptric NAS/search-index resolution cannot locate that dataset.
DATASETS = [
    {
        "label": "dark_wait_0s",
        "file_stem": (
            "2026_08_20-16_37_57-qnami-nv0_2026_02_20-"
            "particle-memory-source_off_wait_0s-wait-0s"
        ),
        "npz_path_override": None,
        # For a fair comparison with the shorter wait-time datasets, analyze
        # only the first 2000 shots/runs from the 0-s acquisition.
        "max_runs": 3300,
    },
    {
        
        "label": "dark_wait_30s",
        "file_stem": (
            "2026_08_26-19_38_50-qnami-nv0_2026_02_20-"
            "particle-memory-source_off_wait_30s-wait-30s"
        ),
        "npz_path_override": None,
    },
    {
        "label": "dark_wait_60s",
        "file_stem": (
            "2026_08_23-16_00_17-qnami-nv0_2026_02_20-"
            "particle-memory-source_off_wait_60s-wait-60s"
        ),
        "npz_path_override": None,
    },

]

# These measurements are at DIFFERENT wait times under source-off
# conditions: 0 s, 30 s, and 60 s. Analyze each file separately and make
# direct comparison plots across all three wait times.
CALCULATE_POOLED_SOURCE_OFF = True
MAKE_COMPARISON_PLOTS = True
ALSO_RUN_APPENDED_ANALYSIS = False  # MUST remain False: 0 s / 30 s / 60 s are different conditions

# Do not make the old curve_fit Poisson histogram the primary model. The
# reference Poisson estimate is lambda = mean(K), exactly as in charge_monitor.py.
SHOW_CURVE_FIT_POISSON_CROSSCHECK = False

REP_INITIAL = 11
REP_FINAL = 12

# Camera-pixel calibration used by the previous analysis.
UM_PER_PIXEL = 0.43

# Charge classification margins. A positive value creates an explicit
# unclassified band around each NV's saved threshold.
INITIAL_MARGIN_COUNTS = 0.0
FINAL_MARGIN_COUNTS = 0.0

# Reject whole-run acquisition/readout collapses.  These are runs where the
# TOTAL raw signal across the full NV array drops far below its normal value.
# They are retained in the raw data but excluded from anomaly statistics.
REJECT_GLOBAL_DROP_RUNS = True

# A run is bad if either the total array signal collapses OR most individual NVs
# collapse together.  The second criterion catches cases where camera/background
# counts keep the total from reaching zero even though the NV signal disappears.
MIN_RUN_TOTAL_FRACTION = 0.50
PER_NV_COLLAPSE_FRACTION = 0.25
MAX_COLLAPSED_NV_FRACTION = 0.80

# -------------------------------------------------------------------------
# FAST COUNTS-ONLY MODE
# -------------------------------------------------------------------------
# IMPORTANT: keep False for the large 0-s / 60-s comparison.
#
# When False, the script NEVER streams/decompresses img_arrays pixel data.
# It uses only:
#   * counts rep 11
#   * counts rep 12
#   * thresholds
#   * NV metadata / coordinates
#
# This skips drift, brightness, image-correlation, and raw candidate-image
# diagnostics, which are the expensive part of the ~11-GB NPZ analysis.
CALCULATE_DRIFT = False

BRIGHT_MARGIN_COUNTS = 5.0
DRIFT_ROI_RADIUS_PX = 5
MAX_DRIFT_NVS = 30

# Streaming/reporting.
PROGRESS_EVERY = 50

# Dataset validation. The original single-file script required >=1000 runs
# because it was written only for one ~4000-run file. Multi-dataset analysis
# should accept shorter control measurements too.
MIN_VALID_RUNS = 1

# Spatial screening metric for each run.
CALCULATE_SPATIAL = False  # counts-only fast comparison; enable later if needed
SHORT_RANGE_UM = 30.0
PAIR_CHUNK_SIZE = 2000

# Plot/printing controls.
TOP_N = 12
TOP_EVENT_MAPS = 6
MAX_RAW_OUTLIER_IMAGES = 20
TRANSITION_HIST_BINS = 60

# Outlier thresholds.
#
# 3 sigma is useful as a screening level, but with ~4000 runs it is not rare:
# an ideal one-sided Gaussian tail predicts ~5.4 events above 3 sigma by chance.
# 4 sigma is therefore the default "strong outlier" screening threshold here.
SIGMA_THRESHOLDS = (3.0, 4.0, 5.0)
PRIMARY_OUTLIER_SIGMA = 4.0

# Poisson background model.
#
# The Poisson model is fit to INTEGER NV- -> NV0 losses, not to the percentage
# histogram.  To keep large positive outliers from inflating the fitted
# background rate, estimate p0 from runs with |robust loss z| below this value.
POISSON_BASELINE_MAX_ABS_ROBUST_Z = 3.0

# Add Bonferroni/look-elsewhere correction across all valid runs.
CALCULATE_TRIAL_CORRECTED_POISSON_SIGMA = True

# Exact reference-style Poisson/coincidence diagnostic from charge_monitor.py.
# Each NV timeline is circularly shifted by SCRAMBLE_SHIFT_PER_NV * nv_ind
# for the scrambled control, just as in the reference code.
CALCULATE_REFERENCE_POISSON = True
SCRAMBLE_SHIFT_PER_NV = 10

SCRIPT_VERSION = "BIGFILE_CLEAN_STREAM_V19H_0S_FIRST2000_OBSERVED_SEPARATE_2026-08-27"


# =============================================================================
# Generic helpers
# =============================================================================


def _format_bytes(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _print_memory(label: str) -> None:
    """Print process RSS when psutil is installed; otherwise stay silent."""
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss
        print(f"[memory] {label}: RSS={_format_bytes(rss)}", flush=True)
    except Exception:
        pass


def _robust_zscore(values):
    values = np.asarray(values, dtype=float)
    z = np.full(values.shape, np.nan, dtype=float)

    finite = np.isfinite(values)
    vals = values[finite]
    if vals.size == 0:
        return z, np.nan, np.nan

    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    sigma = 1.4826 * mad

    # A zero MAD can occur for discrete pair statistics.  Use ordinary sigma
    # only as a fallback so the diagnostic does not disappear entirely.
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan

    if np.isfinite(sigma) and sigma > 0:
        z[finite] = (values[finite] - med) / sigma

    return z, med, sigma


def _empirical_upper_tail_p(values):
    """One-sided empirical p for each value: fraction >= that observation."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite_inds = np.where(np.isfinite(values))[0]
    if finite_inds.size == 0:
        return out

    finite_vals = values[finite_inds]
    n = finite_vals.size

    # O(n log n), including ties correctly.
    sorted_vals = np.sort(finite_vals)
    for global_ind in finite_inds:
        v = values[global_ind]
        first_ge = np.searchsorted(sorted_vals, v, side="left")
        num_ge = n - first_ge
        out[global_ind] = (1.0 + num_ge) / (n + 1.0)

    return out


def _safe_divide(num, den):
    """Elementwise division with broadcasting and NaN for invalid denominators."""
    num, den = np.broadcast_arrays(
        np.asarray(num, dtype=float),
        np.asarray(den, dtype=float),
    )
    out = np.full(num.shape, np.nan, dtype=float)
    good = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[good] = num[good] / den[good]
    return out


def _pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 3:
        return np.nan
    xg = x[good]
    yg = y[good]
    if np.nanstd(xg) == 0 or np.nanstd(yg) == 0:
        return np.nan
    return float(np.corrcoef(xg, yg)[0, 1])


def _gaussian_one_sided_expected_count(num_trials, z_threshold):
    """Expected number above a one-sided Gaussian z threshold."""
    return float(num_trials) * float(norm.sf(float(z_threshold)))


def _summarize_z_outliers(
    z_values,
    good_run_mask=None,
    thresholds=SIGMA_THRESHOLDS,
):
    """
    Quantify positive-tail outliers for a z-like screening statistic.

    Returns per-threshold masks/indices/counts and the ideal-Gaussian expected
    false-positive count.  The Gaussian expectation is only a reference; robust
    z and spatial z are empirical screening scores, not guaranteed N(0,1).
    """
    z_values = np.asarray(z_values, dtype=float)
    valid = np.isfinite(z_values)

    if good_run_mask is not None:
        good_run_mask = np.asarray(good_run_mask, dtype=bool)
        if good_run_mask.shape != z_values.shape:
            raise ValueError(
                f"good_run_mask shape {good_run_mask.shape} does not match "
                f"z_values shape {z_values.shape}."
            )
        valid &= good_run_mask

    num_valid = int(np.sum(valid))
    summary = {
        "num_valid": num_valid,
        "thresholds": tuple(float(v) for v in thresholds),
        "by_threshold": {},
    }

    for threshold in thresholds:
        threshold = float(threshold)
        mask = valid & (z_values >= threshold)
        summary["by_threshold"][threshold] = {
            "mask": mask,
            "indices": np.where(mask)[0],
            "count": int(np.sum(mask)),
            "expected_gaussian_one_sided": _gaussian_one_sided_expected_count(
                num_valid, threshold
            ),
        }

    return summary


def _sigma_from_one_sided_p(p_values):
    """
    Convert one-sided p-values to equivalent Gaussian sigma.

    Very small p-values are clipped only to avoid numerical infinities.
    """
    p_values = np.asarray(p_values, dtype=float)
    out = np.full(p_values.shape, np.nan, dtype=float)

    good = np.isfinite(p_values) & (p_values >= 0.0) & (p_values <= 1.0)
    if not np.any(good):
        return out

    tiny = np.finfo(float).tiny
    upper = 1.0 - np.finfo(float).eps
    p_clip = np.clip(p_values[good], tiny, upper)
    out[good] = norm.isf(p_clip)
    return out


def _poisson_hist_fit_fn(k, amplitude, mean_count):
    """
    Poisson histogram model used with scipy.optimize.curve_fit.

    This mirrors the explicit fit-function -> curve_fit -> covariance ->
    chi-square workflow used in the reference conditional-charge script.

    amplitude is the fitted total normalization of the histogram and
    mean_count is the fitted Poisson mean.
    """
    k = np.asarray(k, dtype=float)
    return float(amplitude) * poisson.pmf(k, float(mean_count))


def _fit_poisson_histogram(
    lost,
    good_run_mask,
    loss_z,
    baseline_max_abs_robust_z=3.0,
):
    """
    Fit the CENTRAL loss-count histogram to a Poisson distribution.

    Important:
      * This is a histogram/visual null-model fit.
      * The per-run significance calculation below remains exposure-corrected
        with lambda_r = p0 * N_r because N_r can vary between runs.
      * Bins belonging to the large positive robust-z tail are not used to
        determine the background fit.

    Returns curve_fit parameters, 1-sigma parameter uncertainties, chi-square,
    reduced chi-square, and the histogram arrays used for the fit.
    """
    lost = np.asarray(lost, dtype=float)
    good_run_mask = np.asarray(good_run_mask, dtype=bool)
    loss_z = np.asarray(loss_z, dtype=float)

    valid = (
        good_run_mask
        & np.isfinite(lost)
        & np.isfinite(loss_z)
        & (lost >= 0)
    )
    baseline = valid & (
        np.abs(loss_z) < float(baseline_max_abs_robust_z)
    )

    # Fall back to all valid runs only if central selection becomes too small.
    if np.sum(baseline) < max(20, int(0.25 * np.sum(valid))):
        baseline = valid.copy()

    baseline_counts = np.rint(lost[baseline]).astype(int)
    all_good_counts = np.rint(lost[valid]).astype(int)

    if baseline_counts.size < 3:
        return {
            "success": False,
            "baseline_mask": baseline,
            "valid_mask": valid,
        }

    k_max = int(max(np.max(all_good_counts), np.max(baseline_counts)))
    k = np.arange(k_max + 1, dtype=int)

    hist_baseline = np.bincount(
        baseline_counts,
        minlength=k_max + 1,
    ).astype(float)
    hist_all_good = np.bincount(
        all_good_counts,
        minlength=k_max + 1,
    ).astype(float)

    # Fit only the range populated by the central/background sample.
    populated = np.where(hist_baseline > 0)[0]
    if populated.size < 3:
        return {
            "success": False,
            "baseline_mask": baseline,
            "valid_mask": valid,
        }

    fit_min = int(populated[0])
    fit_max = int(populated[-1])
    fit_mask = (k >= fit_min) & (k <= fit_max)

    x_fit = k[fit_mask].astype(float)
    y_fit = hist_baseline[fit_mask]

    # Poisson histogram counting uncertainty. Empty bins still carry useful
    # information; sigma=1 avoids division by zero and extreme overweighting.
    y_sigma = np.sqrt(np.maximum(y_fit, 1.0))

    p0_amp = float(max(np.sum(hist_baseline), 1.0))
    p0_mean = float(max(np.mean(baseline_counts), 1e-6))

    try:
        popt, pcov = curve_fit(
            _poisson_hist_fit_fn,
            x_fit,
            y_fit,
            p0=(p0_amp, p0_mean),
            sigma=y_sigma,
            absolute_sigma=True,
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            maxfev=20000,
        )

        perr = np.sqrt(np.diag(pcov))
        fit_y = _poisson_hist_fit_fn(x_fit, *popt)
        residuals = y_fit - fit_y
        chi_sq = float(np.sum((residuals / y_sigma) ** 2))
        dof = max(1, int(len(x_fit) - len(popt)))
        red_chi_sq = chi_sq / dof

        # Model curve over the entire displayed count range.
        model_curve = _poisson_hist_fit_fn(k, *popt)

        return {
            "success": True,
            "baseline_mask": baseline,
            "valid_mask": valid,
            "k": k,
            "hist_baseline": hist_baseline,
            "hist_all_good": hist_all_good,
            "fit_mask": fit_mask,
            "fit_amplitude": float(popt[0]),
            "fit_mean_count": float(popt[1]),
            "fit_amplitude_ste": float(perr[0]),
            "fit_mean_count_ste": float(perr[1]),
            "chi_sq": chi_sq,
            "dof": int(dof),
            "red_chi_sq": float(red_chi_sq),
            "model_curve": model_curve,
            "num_baseline_runs": int(np.sum(baseline)),
            "num_valid_runs": int(np.sum(valid)),
        }

    except Exception as exc:
        return {
            "success": False,
            "baseline_mask": baseline,
            "valid_mask": valid,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _observed_threshold_rarity(
    z_values,
    good_run_mask,
    thresholds=SIGMA_THRESHOLDS,
):
    """
    Quantify how often events exceed each positive z threshold.

    For each threshold returns:
      count                  observed number of events
      percent                observed percentage of valid runs
      fraction               observed fraction
      one_in                 empirical frequency: ~1 event per one_in runs
      empirical_resolution   1/N, the smallest nonzero empirical fraction
      gaussian_tail_fraction ideal one-sided Gaussian tail probability
      gaussian_expected_count expected count among N valid trials
      gaussian_one_in        ideal Gaussian "1 in N" frequency

    These are screening summaries. Robust-z and spatial-z need not be exactly
    standard normal, so the Gaussian columns are reference expectations.
    """
    z_values = np.asarray(z_values, dtype=float)
    good_run_mask = np.asarray(good_run_mask, dtype=bool)

    valid = good_run_mask & np.isfinite(z_values)
    n = int(np.sum(valid))

    out = {
        "num_valid": n,
        "empirical_resolution_fraction": (1.0 / n) if n > 0 else np.nan,
        "empirical_resolution_percent": (100.0 / n) if n > 0 else np.nan,
        "by_threshold": {},
    }

    for z_thr in thresholds:
        z_thr = float(z_thr)
        mask = valid & (z_values >= z_thr)
        count = int(np.sum(mask))
        fraction = (count / n) if n > 0 else np.nan
        percent = 100.0 * fraction if np.isfinite(fraction) else np.nan
        one_in = (n / count) if count > 0 else np.inf

        gauss_frac = float(norm.sf(z_thr))
        gauss_count = n * gauss_frac

        out["by_threshold"][z_thr] = {
            "mask": mask,
            "indices": np.where(mask)[0],
            "count": count,
            "fraction": float(fraction),
            "percent": float(percent),
            "one_in": float(one_in),
            "gaussian_tail_fraction": gauss_frac,
            "gaussian_tail_percent": 100.0 * gauss_frac,
            "gaussian_expected_count": float(gauss_count),
            "gaussian_one_in": float(1.0 / gauss_frac),
        }

    return out


def _poisson_model_threshold_rarity(
    lambda_by_run,
    valid_mask,
    thresholds=SIGMA_THRESHOLDS,
):
    """
    Exact discrete Poisson expectation for local-sigma thresholds.

    For each run and each target Gaussian-equivalent z threshold, find the
    smallest integer loss count k_thr satisfying

        P[X >= k_thr | lambda_r] <= norm.sf(z_thr)

    and sum the actual Poisson probability of crossing that discrete threshold
    across all valid runs.

    This is more accurate than simply using N * norm.sf(z) because a Poisson
    count is discrete and lambda_r can vary from run to run.
    """
    lambda_by_run = np.asarray(lambda_by_run, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid = valid_mask & np.isfinite(lambda_by_run) & (lambda_by_run >= 0)

    lambdas = lambda_by_run[valid]
    n = int(lambdas.size)

    out = {"num_valid": n, "by_threshold": {}}

    for z_thr in thresholds:
        z_thr = float(z_thr)
        p_target = float(norm.sf(z_thr))

        if n == 0:
            out["by_threshold"][z_thr] = {
                "expected_count": np.nan,
                "expected_fraction": np.nan,
                "expected_percent": np.nan,
                "expected_one_in": np.nan,
            }
            continue

        crossing_probs = np.empty(n, dtype=float)
        k_thresholds = np.empty(n, dtype=int)

        for i, lam in enumerate(lambdas):
            # isf returns x such that sf(x) <= q; we need P[X >= k] = sf(k-1).
            # Therefore k = isf(q, lambda) + 1, followed by a small correction
            # for floating/discrete edge cases.
            k_thr = int(poisson.isf(p_target, lam)) + 1
            k_thr = max(k_thr, 0)

            while k_thr > 0 and poisson.sf(k_thr - 2, lam) <= p_target:
                k_thr -= 1
            while poisson.sf(k_thr - 1, lam) > p_target:
                k_thr += 1

            k_thresholds[i] = k_thr
            crossing_probs[i] = float(poisson.sf(k_thr - 1, lam))

        expected_count = float(np.sum(crossing_probs))
        expected_fraction = expected_count / n
        expected_percent = 100.0 * expected_fraction
        expected_one_in = (
            1.0 / expected_fraction
            if expected_fraction > 0
            else np.inf
        )

        out["by_threshold"][z_thr] = {
            "target_gaussian_tail_p": p_target,
            "k_thresholds": k_thresholds,
            "per_run_crossing_probability": crossing_probs,
            "expected_count": expected_count,
            "expected_fraction": expected_fraction,
            "expected_percent": expected_percent,
            "expected_one_in": float(expected_one_in),
        }

    return out



def _poisson_sigma_count_threshold(lam, z_threshold):
    """
    Smallest integer k such that P[X >= k | Poisson(lam)] is at or below the
    one-sided Gaussian tail corresponding to z_threshold.
    """
    lam = float(lam)
    target_p = float(norm.sf(float(z_threshold)))

    # poisson.isf(q) returns x with sf(x) <= q. We need sf(k-1) <= q.
    k = int(poisson.isf(target_p, lam)) + 1
    k = max(k, 0)

    # Correct any discrete/floating boundary ambiguity.
    while k > 0 and poisson.sf(k - 2, lam) <= target_p:
        k -= 1
    while poisson.sf(k - 1, lam) > target_p:
        k += 1

    return int(k)


def _reference_poisson_distribution(
    switch_mask,
    good_run_mask,
    sigma_thresholds=SIGMA_THRESHOLDS,
    scramble_shift_per_nv=10,
    run_group_ids=None,
):
    """
    Reproduce the supplied charge_monitor.py Poisson coincidence analysis.

    Real / unscrambled:
        coincidences[run] = number of NV- -> NV0 transitions in that run
        lambda = mean(coincidences)
        expected = num_runs * poisson.pmf(k, lambda)

    Scrambled:
        independently circularly shift each NV's run timeline by
        scramble_shift_per_nv * nv_ind and recompute the coincidence histogram.

    The scrambling preserves the number of transitions for each NV but destroys
    true same-run coincidences between different NVs.

    Also quantifies the right-tail rarity at the 3, 4 and 5 sigma-equivalent
    Poisson thresholds:
      * integer K threshold
      * observed number and percent of real runs
      * observed ~1-in-N frequency
      * Poisson expected percent / expected runs
      * scrambled-control percent
      * observed/Poisson and observed/scrambled rate ratios
    """
    switch_mask = np.asarray(switch_mask, dtype=bool)
    good_run_mask = np.asarray(good_run_mask, dtype=bool)

    if switch_mask.ndim != 2:
        raise ValueError("switch_mask must have shape [nv, run].")
    if good_run_mask.shape != (switch_mask.shape[1],):
        raise ValueError("good_run_mask has wrong shape.")

    good_inds = np.where(good_run_mask)[0]
    states_by_nv = switch_mask[:, good_inds].copy()

    num_nvs, num_shots = states_by_nv.shape
    if num_shots == 0:
        return {"success": False}

    if run_group_ids is None:
        good_group_ids = np.zeros(num_shots, dtype=int)
    else:
        run_group_ids = np.asarray(run_group_ids)
        if run_group_ids.shape != good_run_mask.shape:
            raise ValueError(
                f"run_group_ids shape {run_group_ids.shape} does not match "
                f"good_run_mask shape {good_run_mask.shape}."
            )
        good_group_ids = run_group_ids[good_inds]

    # ------------------------------------------------------------------
    # UNSCRAMBLED coincidence distribution
    # ------------------------------------------------------------------
    coincidences = np.sum(states_by_nv, axis=0).astype(int)
    lam = float(np.mean(coincidences))

    # ------------------------------------------------------------------
    # SCRAMBLED control -- same idea as supplied reference code, but each
    # acquisition is rolled internally so no NV timeline crosses a dataset
    # boundary when the two 0-s files are appended.
    # ------------------------------------------------------------------
    scrambled_by_nv = states_by_nv.copy()
    unique_groups = np.unique(good_group_ids)

    for group_id in unique_groups:
        cols = np.where(good_group_ids == group_id)[0]
        if cols.size == 0:
            continue

        for nv_ind in range(num_nvs):
            shift = (
                int(scramble_shift_per_nv) * int(nv_ind)
            ) % int(cols.size)
            scrambled_by_nv[nv_ind, cols] = np.roll(
                states_by_nv[nv_ind, cols],
                shift,
            )

    scrambled = np.sum(scrambled_by_nv, axis=0).astype(int)
    scrambled_lam = float(np.mean(scrambled))

    k_max = int(
        max(
            np.max(coincidences),
            np.max(scrambled),
            np.ceil(lam + 8.0 * np.sqrt(max(lam, 1.0))),
            np.ceil(scrambled_lam + 8.0 * np.sqrt(max(scrambled_lam, 1.0))),
        )
    )
    x_vals = np.arange(k_max + 1, dtype=int)

    observed_hist = np.bincount(
        coincidences,
        minlength=k_max + 1,
    ).astype(float)
    scrambled_hist = np.bincount(
        scrambled,
        minlength=k_max + 1,
    ).astype(float)

    expected_dist = num_shots * poisson.pmf(x_vals, lam)
    scrambled_expected_dist = num_shots * poisson.pmf(
        x_vals,
        scrambled_lam,
    )

    # Simple Poisson dispersion checks.
    var_real = float(np.var(coincidences, ddof=1)) if num_shots > 1 else np.nan
    var_scrambled = (
        float(np.var(scrambled, ddof=1))
        if num_shots > 1
        else np.nan
    )
    dispersion_real = var_real / lam if lam > 0 else np.nan
    dispersion_scrambled = (
        var_scrambled / scrambled_lam
        if scrambled_lam > 0
        else np.nan
    )

    # If the real count distribution is broader than Poisson, show a
    # moment-matched negative-binomial curve as a diagnostic. This does NOT
    # replace Poisson for the requested reference comparison; it quantifies why
    # a single Poisson PMF may look poor.
    nbinom_expected_dist = None
    nbinom_n = np.nan
    nbinom_p = np.nan
    if (
        np.isfinite(var_real)
        and np.isfinite(lam)
        and lam > 0
        and var_real > lam
    ):
        nbinom_p = lam / var_real
        nbinom_n = lam * lam / (var_real - lam)
        nbinom_expected_dist = num_shots * nbinom.pmf(
            x_vals,
            nbinom_n,
            nbinom_p,
        )

    scrambled_nbinom_expected_dist = None
    scrambled_nbinom_n = np.nan
    scrambled_nbinom_p = np.nan
    if (
        np.isfinite(var_scrambled)
        and np.isfinite(scrambled_lam)
        and scrambled_lam > 0
        and var_scrambled > scrambled_lam
    ):
        scrambled_nbinom_p = scrambled_lam / var_scrambled
        scrambled_nbinom_n = (
            scrambled_lam * scrambled_lam
            / (var_scrambled - scrambled_lam)
        )
        scrambled_nbinom_expected_dist = num_shots * nbinom.pmf(
            x_vals,
            scrambled_nbinom_n,
            scrambled_nbinom_p,
        )

    threshold_summary = {}
    for z_thr in sigma_thresholds:
        z_thr = float(z_thr)
        k_thr = _poisson_sigma_count_threshold(lam, z_thr)

        obs_count = int(np.sum(coincidences >= k_thr))
        obs_fraction = obs_count / num_shots
        obs_percent = 100.0 * obs_fraction
        obs_one_in = num_shots / obs_count if obs_count > 0 else np.inf

        poisson_tail = float(poisson.sf(k_thr - 1, lam))
        poisson_percent = 100.0 * poisson_tail
        poisson_expected_count = num_shots * poisson_tail
        poisson_one_in = 1.0 / poisson_tail if poisson_tail > 0 else np.inf

        scr_count = int(np.sum(scrambled >= k_thr))
        scr_fraction = scr_count / num_shots
        scr_percent = 100.0 * scr_fraction
        scr_one_in = num_shots / scr_count if scr_count > 0 else np.inf

        obs_to_poisson = (
            obs_fraction / poisson_tail
            if poisson_tail > 0
            else np.inf
        )
        obs_to_scrambled = (
            obs_fraction / scr_fraction
            if scr_fraction > 0
            else np.inf
        )

        threshold_summary[z_thr] = {
            "k_threshold": int(k_thr),
            "observed_count": obs_count,
            "observed_fraction": float(obs_fraction),
            "observed_percent": float(obs_percent),
            "observed_one_in": float(obs_one_in),
            "poisson_tail_probability": poisson_tail,
            "poisson_expected_percent": float(poisson_percent),
            "poisson_expected_count": float(poisson_expected_count),
            "poisson_one_in": float(poisson_one_in),
            "scrambled_count": scr_count,
            "scrambled_fraction": float(scr_fraction),
            "scrambled_percent": float(scr_percent),
            "scrambled_one_in": float(scr_one_in),
            "observed_to_poisson_rate_ratio": float(obs_to_poisson),
            "observed_to_scrambled_rate_ratio": float(obs_to_scrambled),
        }

    # Empirical one-sided rank probability of each REAL run within real data.
    empirical_p = np.full(num_shots, np.nan, dtype=float)
    sorted_real = np.sort(coincidences)
    for i, val in enumerate(coincidences):
        first_ge = np.searchsorted(sorted_real, val, side="left")
        empirical_p[i] = (num_shots - first_ge) / num_shots

    # Empirical probability of each REAL run relative to the scrambled control.
    scrambled_empirical_p = np.full(num_shots, np.nan, dtype=float)
    sorted_scrambled = np.sort(scrambled)
    for i, val in enumerate(coincidences):
        first_ge = np.searchsorted(sorted_scrambled, val, side="left")
        num_ge = num_shots - first_ge
        # +1 correction because real event is tested against a separate null sample.
        scrambled_empirical_p[i] = (1.0 + num_ge) / (num_shots + 1.0)

    return {
        "success": True,
        "good_run_indices": good_inds,
        "num_nvs": int(num_nvs),
        "num_shots": int(num_shots),
        "coincidences": coincidences,
        "scrambled_coincidences": scrambled,
        "lambda": lam,
        "scrambled_lambda": scrambled_lam,
        "x_vals": x_vals,
        "observed_hist": observed_hist,
        "scrambled_hist": scrambled_hist,
        "expected_dist": expected_dist,
        "scrambled_expected_dist": scrambled_expected_dist,
        "variance": var_real,
        "scrambled_variance": var_scrambled,
        "dispersion": float(dispersion_real),
        "scrambled_dispersion": float(dispersion_scrambled),
        "nbinom_expected_dist": nbinom_expected_dist,
        "nbinom_n": float(nbinom_n),
        "nbinom_p": float(nbinom_p),
        "scrambled_nbinom_expected_dist": scrambled_nbinom_expected_dist,
        "scrambled_nbinom_n": float(scrambled_nbinom_n),
        "scrambled_nbinom_p": float(scrambled_nbinom_p),
        "threshold_summary": threshold_summary,
        "empirical_p": empirical_p,
        "scrambled_empirical_p": scrambled_empirical_p,
    }


def _analyze_poisson_loss_outliers(
    lost,
    evaluable_eligible_count,
    loss_z,
    good_run_mask,
    baseline_max_abs_robust_z=3.0,
):
    """
    Exposure-corrected Poisson model for integer NV- -> NV0 losses.

    Model
    -----
        K_r ~ Poisson(lambda_r)
        lambda_r = p0 * N_r

    where K_r is the number of losses in run r and N_r is the number of
    evaluable initially-NV- sites in that run.

    p0 is estimated from central/background runs satisfying
        |robust loss z| < baseline_max_abs_robust_z
    so that a few large positive-tail candidates do not inflate the baseline.

    The returned poisson_local_sigma is the one-sided Gaussian-equivalent sigma
    corresponding to P[X >= K_r | lambda_r].

    A Bonferroni-corrected p/sigma is also returned to account approximately for
    looking across all valid runs.
    """
    lost = np.asarray(lost, dtype=float)
    n_eval = np.asarray(evaluable_eligible_count, dtype=float)
    loss_z = np.asarray(loss_z, dtype=float)
    good_run_mask = np.asarray(good_run_mask, dtype=bool)

    if not (
        lost.shape == n_eval.shape == loss_z.shape == good_run_mask.shape
    ):
        raise ValueError("Poisson input arrays must have matching run shapes.")

    valid = (
        good_run_mask
        & np.isfinite(lost)
        & np.isfinite(n_eval)
        & (n_eval > 0)
        & np.isfinite(loss_z)
    )

    baseline = (
        valid
        & (np.abs(loss_z) < float(baseline_max_abs_robust_z))
    )

    # If a very small dataset leaves too few central runs, fall back to all
    # otherwise-valid runs rather than failing.
    if np.sum(baseline) < max(10, int(0.25 * np.sum(valid))):
        baseline = valid.copy()

    total_trials = float(np.sum(n_eval[baseline]))
    total_events = float(np.sum(lost[baseline]))

    if not np.isfinite(total_trials) or total_trials <= 0:
        raise ValueError("No valid evaluable NV trials for Poisson baseline.")

    p0 = total_events / total_trials
    lambda_by_run = p0 * n_eval

    tail_p = np.full(lost.shape, np.nan, dtype=float)
    pearson_residual = np.full(lost.shape, np.nan, dtype=float)

    valid_lambda = valid & np.isfinite(lambda_by_run) & (lambda_by_run >= 0.0)
    k_int = np.rint(lost[valid_lambda]).astype(int)
    lam = lambda_by_run[valid_lambda]

    # Survival function gives P[X >= observed] as sf(k-1).
    tail_p[valid_lambda] = poisson.sf(k_int - 1, lam)

    positive_lambda = valid_lambda & (lambda_by_run > 0)
    pearson_residual[positive_lambda] = (
        lost[positive_lambda] - lambda_by_run[positive_lambda]
    ) / np.sqrt(lambda_by_run[positive_lambda])

    local_sigma = _sigma_from_one_sided_p(tail_p)

    num_valid = int(np.sum(valid_lambda))
    bonferroni_p = np.full(lost.shape, np.nan, dtype=float)
    trial_sigma = np.full(lost.shape, np.nan, dtype=float)

    if CALCULATE_TRIAL_CORRECTED_POISSON_SIGMA and num_valid > 0:
        bonferroni_p[valid_lambda] = np.minimum(
            1.0,
            tail_p[valid_lambda] * float(num_valid),
        )
        trial_sigma = _sigma_from_one_sided_p(bonferroni_p)

    # Model dispersion diagnostic. Values >1 indicate broader-than-Poisson
    # fluctuations; if strongly >1, Poisson tail significances are optimistic.
    baseline_lambda = lambda_by_run[baseline]
    baseline_lost = lost[baseline]
    good_disp = np.isfinite(baseline_lambda) & (baseline_lambda > 0)
    if np.sum(good_disp) > 1:
        pearson_chi2 = float(
            np.sum(
                (baseline_lost[good_disp] - baseline_lambda[good_disp]) ** 2
                / baseline_lambda[good_disp]
            )
        )
        dispersion_dof = max(1, int(np.sum(good_disp)) - 1)
        pearson_dispersion = pearson_chi2 / dispersion_dof
    else:
        pearson_chi2 = np.nan
        dispersion_dof = 0
        pearson_dispersion = np.nan

    # Discrete expected count histogram. Because N_r varies slightly by run, the
    # correct visual null is a mixture of Poisson(lambda_r), not one fixed PMF.
    valid_lost_int = np.rint(lost[valid_lambda]).astype(int)
    if valid_lost_int.size > 0:
        k_max_observed = int(np.max(valid_lost_int))
        lambda_max = float(np.nanmax(lambda_by_run[valid_lambda]))
        k_max_model = int(np.ceil(lambda_max + 8.0 * np.sqrt(max(lambda_max, 1.0))))
        k_max = max(k_max_observed, k_max_model)
        k_values = np.arange(k_max + 1, dtype=int)

        observed_hist = np.bincount(
            valid_lost_int,
            minlength=k_max + 1,
        ).astype(float)

        expected_hist = np.sum(
            poisson.pmf(
                k_values[:, None],
                lambda_by_run[valid_lambda][None, :],
            ),
            axis=1,
        )
    else:
        k_values = np.array([], dtype=int)
        observed_hist = np.array([], dtype=float)
        expected_hist = np.array([], dtype=float)

    primary_mask = valid_lambda & (local_sigma >= float(PRIMARY_OUTLIER_SIGMA))

    # Explicit curve_fit Poisson histogram fit, following the fitting workflow
    # used in the reference conditional-charge analysis.
    histogram_fit = _fit_poisson_histogram(
        lost=lost,
        good_run_mask=good_run_mask,
        loss_z=loss_z,
        baseline_max_abs_robust_z=baseline_max_abs_robust_z,
    )

    poisson_threshold_model = _poisson_model_threshold_rarity(
        lambda_by_run=lambda_by_run,
        valid_mask=valid_lambda,
        thresholds=SIGMA_THRESHOLDS,
    )

    return {
        "baseline_mask": baseline,
        "valid_mask": valid_lambda,
        "num_baseline_runs": int(np.sum(baseline)),
        "num_valid_runs": num_valid,
        "baseline_transition_probability": float(p0),
        "lambda_by_run": lambda_by_run,
        "poisson_tail_p": tail_p,
        "poisson_local_sigma": local_sigma,
        "poisson_bonferroni_p": bonferroni_p,
        "poisson_trial_sigma": trial_sigma,
        "poisson_pearson_residual": pearson_residual,
        "pearson_chi2": pearson_chi2,
        "pearson_dispersion_dof": int(dispersion_dof),
        "pearson_dispersion": float(pearson_dispersion),
        "hist_k": k_values,
        "hist_observed": observed_hist,
        "hist_expected": expected_hist,
        "primary_outlier_mask": primary_mask,
        "primary_outlier_indices": np.where(primary_mask)[0],
        "histogram_fit": histogram_fit,
        "threshold_model_rarity": poisson_threshold_model,
    }


# =============================================================================
# NV camera coordinates
# =============================================================================


def _try_get_nv_img_xy(nv) -> Optional[Tuple[float, float]]:
    """Read camera coordinates from common attributes or nv.coords[PIXEL]."""
    for attr in (
        "pixel_coords",
        "img_coords",
        "image_coords",
        "camera_coords",
    ):
        value = getattr(nv, attr, None)
        if value is not None:
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                return float(arr[0]), float(arr[1])

    coords = getattr(nv, "coords", None)
    if isinstance(coords, dict):
        # First support enum keys such as CoordsKey.PIXEL.
        for key, value in coords.items():
            key_name = getattr(key, "name", None)
            key_text = str(key).upper()
            if (
                key_name == "PIXEL"
                or key_text == "PIXEL"
                or key_text.endswith(".PIXEL")
            ):
                arr = np.asarray(value, dtype=float).ravel()
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    return float(arr[0]), float(arr[1])

        # String-key fallback.
        for key in (
            "pixel",
            "pixels",
            "pixel_coords",
            "img",
            "image",
            "camera",
            "camera_coords",
        ):
            if key in coords:
                arr = np.asarray(coords[key], dtype=float).ravel()
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    return float(arr[0]), float(arr[1])

    return None


def _coerce_img_coords(nv_list) -> np.ndarray:
    coords = []
    for nv_ind, nv in enumerate(nv_list):
        xy = _try_get_nv_img_xy(nv)
        if xy is None:
            raise ValueError(
                f"Could not obtain camera PIXEL coordinates for NV {nv_ind}."
            )
        coords.append(xy)
    return np.asarray(coords, dtype=float)


# =============================================================================
# Large-NPZ discovery and header validation
# =============================================================================


class _CapturedDatasetPath(RuntimeError):
    def __init__(self, path):
        self.path = Path(path)
        super().__init__(str(path))


def _read_npy_header_from_stream(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        shape, fortran_order, dtype = np.lib.format._read_array_header(
            stream, version
        )
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def _find_npz_member_name(archive: zipfile.ZipFile, key: str) -> Optional[str]:
    target = f"{key}.npy"
    names = archive.namelist()
    if target in names:
        return target
    for name in names:
        if name.endswith("/" + target):
            return name
    return None


def _inspect_npz(npz_path):
    """Read only ZIP/NPY headers; never materialize array payloads."""
    path = Path(npz_path)
    info = {
        "path": path,
        "valid": False,
        "counts_shape": None,
        "counts_dtype": None,
        "images_shape": None,
        "images_dtype": None,
        "has_images": False,
    }

    try:
        if not path.exists() or not path.is_file():
            return info

        with zipfile.ZipFile(path, "r") as archive:
            counts_member = _find_npz_member_name(archive, "counts")
            image_member = _find_npz_member_name(archive, "img_arrays")

            if counts_member is None:
                return info

            with archive.open(counts_member, "r") as stream:
                shape, _, dtype = _read_npy_header_from_stream(stream)
                info["counts_shape"] = shape
                info["counts_dtype"] = dtype

            if image_member is not None:
                info["has_images"] = True
                with archive.open(image_member, "r") as stream:
                    shape, _, dtype = _read_npy_header_from_stream(stream)
                    info["images_shape"] = shape
                    info["images_dtype"] = dtype

        # Accept any nonempty particle-memory dataset with the expected array
        # dimensions. Exact count/image run agreement is checked later.
        counts_ok = (
            info["counts_shape"] is not None
            and len(info["counts_shape"]) == 5
            and int(info["counts_shape"][2]) >= int(MIN_VALID_RUNS)
        )
        images_ok = (
            (not info["has_images"])
            or (
                info["images_shape"] is not None
                and len(info["images_shape"]) == 6
                and int(info["images_shape"][1]) >= int(MIN_VALID_RUNS)
            )
        )
        info["valid"] = bool(counts_ok and images_ok)

    except Exception:
        return info

    return info


def _iter_path_like_values(obj, depth=0, max_depth=6):
    if depth > max_depth:
        return
    if isinstance(obj, (str, os.PathLike)):
        yield os.fspath(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_path_like_values(value, depth + 1, max_depth)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_path_like_values(value, depth + 1, max_depth)


def _try_metadata_without_npz(file_stem):
    """Load only the small JSON/TXT metadata. Never request the linked NPZ."""
    print("[large-file] loading metadata only (load_npz=False)...", flush=True)
    try:
        metadata = dm.get_raw_data(
            file_stem=file_stem,
            load_npz=False,
        )
        if isinstance(metadata, dict):
            print(
                f"[large-file] metadata loaded: {len(metadata)} top-level keys",
                flush=True,
            )
        else:
            print(
                f"[large-file] metadata loaded: {type(metadata).__name__}",
                flush=True,
            )
        return metadata
    except TypeError as exc:
        print(
            f"[large-file] metadata-only lookup TypeError: {exc}",
            flush=True,
        )
        return None
    except Exception as exc:
        print(
            f"[large-file] metadata-only lookup did not succeed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _discover_npz_path(file_stem, npz_path_override=None, metadata=None):
    """
    Locate the large particle-memory NPZ WITHOUT ever calling
    dm.get_raw_data(..., load_npz=True).

    This is important for the Dioptric NAS data manager: load_npz=True first
    downloads the complete NPZ into a bytes object, which is unsafe for an
    ~11-GB archive.

    Preferred resolution order:
      1. Explicit NPZ_PATH_OVERRIDE.
      2. Dioptric search index -> NAS parent -> <file_stem>.npz.
      3. Existing absolute/local NPZ paths found in the small metadata.
      4. Exact-name search in the current repo tree as a final fallback.
    """

    def validate(path, source):
        path = Path(path).expanduser()
        text_lower = str(path).lower()

        if "dmdsuite" in text_lower or "calibration" in text_lower:
            return None

        print(
            f"[large-file] checking NPZ ({source}): {path}",
            flush=True,
        )

        info = _inspect_npz(path)
        if info["valid"]:
            print(
                f"[large-file] VALIDATED NPZ ({source}): {path}\n"
                f"             counts={info['counts_shape']} "
                f"{info['counts_dtype']}\n"
                f"             img_arrays={info['images_shape']} "
                f"{info['images_dtype']}",
                flush=True,
            )
            return path.resolve()

        if path.exists():
            print(
                f"[large-file] candidate exists but failed dataset validation: "
                f"{path}",
                flush=True,
            )
        return None

    # ------------------------------------------------------------------
    # 1. Explicit override
    # ------------------------------------------------------------------
    if npz_path_override:
        valid = validate(npz_path_override, "override")
        if valid is None:
            raise ValueError(
                "NPZ_PATH_OVERRIDE does not point to the expected large "
                "particle-memory dataset."
            )
        return valid

    # ------------------------------------------------------------------
    # 2. Native Dioptric NAS path resolution
    # ------------------------------------------------------------------
    #
    # data_manager imports get_file_parent from utils.search_index, so in the
    # user's current Dioptric version it is normally available as
    # dm.get_file_parent.  This returns the mounted NAS directory containing
    # the file.  We can then open the NPZ DIRECTLY as a file instead of asking
    # data_manager to read all 11 GB into memory.
    # ------------------------------------------------------------------
    print(
        "[large-file] resolving NPZ directly through the Dioptric search index...",
        flush=True,
    )

    try:
        get_parent = getattr(dm, "get_file_parent", None)
        if get_parent is None:
            from utils.search_index import get_file_parent as get_parent

        file_parent = Path(get_parent(file_stem))
        candidate = file_parent / f"{file_stem}.npz"

        print(
            f"[large-file] search-index parent: {file_parent}",
            flush=True,
        )

        valid = validate(candidate, "Dioptric NAS/search-index")
        if valid is not None:
            return valid

    except Exception as exc:
        print(
            f"[large-file] direct NAS/search-index resolution failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 3. Paths embedded in the small metadata
    # ------------------------------------------------------------------
    if metadata is not None:
        candidates = []

        for value in _iter_path_like_values(metadata):
            try:
                p = Path(value).expanduser()
            except Exception:
                continue

            if p.suffix.lower() == ".npz":
                candidates.append(p)

            try:
                if p.exists() and p.is_dir():
                    candidates.append(p / f"{file_stem}.npz")
            except Exception:
                pass

        candidates = sorted(
            set(candidates),
            key=lambda p: (file_stem not in p.name, len(str(p))),
        )

        for candidate in candidates:
            valid = validate(candidate, "metadata")
            if valid is not None:
                return valid

    # ------------------------------------------------------------------
    # 4. Small exact-name local fallback
    # ------------------------------------------------------------------
    exact_name = f"{file_stem}.npz"
    searched_roots = []

    for root in (Path.cwd(), Path(__file__).resolve().parent):
        try:
            root = root.resolve()
        except Exception:
            pass

        if root in searched_roots:
            continue
        searched_roots.append(root)

        print(
            f"[large-file] local exact-name fallback under: {root}",
            flush=True,
        )

        try:
            hits = list(root.rglob(exact_name))
        except Exception:
            hits = []

        for hit in hits:
            valid = validate(hit, "local search")
            if valid is not None:
                return valid

    raise FileNotFoundError(
        "Could not safely locate the 11-GB particle-memory NPZ.\n"
        "The script deliberately does NOT call "
        "dm.get_raw_data(..., load_npz=True), because your Dioptric "
        "data_manager reads the complete NPZ into memory.\n\n"
        "Expected NAS file name:\n"
        f"    {file_stem}.npz\n\n"
        "If the search index cannot resolve the mounted NAS path, set "
        "NPZ_PATH_OVERRIDE near the top of this script to the full mounted "
        "NAS path of that NPZ."
    )


# =============================================================================
# Selective loading of small members
# =============================================================================


def _load_count_reps_streaming(npz_path, rep_initial, rep_final):
    """Stream only two count reps from counts.npy with bounded RAM.

    counts has shape [exp, nv, run, step, rep].  For normal C-order np.savez
    output, one NV block for exp=0 is contiguous, so we read one NV block at a
    time and keep only step=0 / rep_initial / rep_final.
    """
    with zipfile.ZipFile(npz_path, "r") as archive:
        member = _find_npz_member_name(archive, "counts")
        if member is None:
            raise KeyError("The dataset NPZ has no 'counts' member.")

        with archive.open(member, "r") as stream:
            shape, fortran_order, dtype = _read_npy_header_from_stream(stream)
            if len(shape) != 5:
                raise ValueError(
                    "Expected counts[exp,nv,run,step,rep], "
                    f"got {shape}"
                )
            if fortran_order:
                raise ValueError(
                    "counts is Fortran-order; the bounded-RAM streaming path "
                    "expects normal C-order np.savez output."
                )
            if dtype.hasobject:
                raise TypeError("counts must be a numeric ndarray, not object dtype.")

            num_exp, num_nvs, num_runs, num_steps, num_reps = map(int, shape)
            if num_exp < 1 or num_steps < 1:
                raise ValueError(f"Unexpected counts shape {shape}")
            for rep in (rep_initial, rep_final):
                if not 0 <= int(rep) < num_reps:
                    raise IndexError(f"rep {rep} outside 0..{num_reps-1}")

            c_initial = np.empty((num_nvs, num_runs), dtype=np.float32)
            c_final = np.empty((num_nvs, num_runs), dtype=np.float32)

            values_per_nv = num_runs * num_steps * num_reps
            bytes_per_nv = values_per_nv * int(dtype.itemsize)

            for nv_ind in range(num_nvs):
                payload = stream.read(bytes_per_nv)
                if len(payload) != bytes_per_nv:
                    raise EOFError(
                        f"Unexpected end of counts.npy while reading NV {nv_ind}."
                    )
                block = np.frombuffer(
                    payload,
                    dtype=dtype,
                    count=values_per_nv,
                ).reshape(num_runs, num_steps, num_reps)

                c_initial[nv_ind, :] = block[:, 0, int(rep_initial)]
                c_final[nv_ind, :] = block[:, 0, int(rep_final)]

    return c_initial, c_final


def _load_small_dataset_members(npz_path, metadata=None):
    """Load counts/thresholds/nv_list but NEVER access img_arrays here."""
    print("[large-file] loading counts + small metadata only...", flush=True)

    c11, c12 = _load_count_reps_streaming(
        npz_path,
        rep_initial=REP_INITIAL,
        rep_final=REP_FINAL,
    )
    num_nvs, num_runs = c11.shape

    # Load only the small metadata members through NumPy.  Do not access
    # archive["counts"] or archive["img_arrays"] here.
    with np.load(npz_path, allow_pickle=True) as archive:
        keys = set(archive.files)

        thresholds = None
        for key in ("analysis_thresholds", "thresholds"):
            if key in keys:
                thresholds = np.asarray(archive[key], dtype=np.float32).ravel()
                break

        nv_list = None
        if "nv_list" in keys:
            value = archive["nv_list"]
            if isinstance(value, np.ndarray) and value.dtype == object:
                nv_list = value.tolist()
            else:
                nv_list = value

        dark_wait_s = None
        if "dark_wait_s" in keys:
            value = archive["dark_wait_s"]
            try:
                dark_wait_s = float(np.asarray(value).item())
            except Exception:
                pass

    gc.collect()

    if thresholds is None and isinstance(metadata, dict):
        for key in ("analysis_thresholds", "thresholds"):
            if key in metadata:
                thresholds = np.asarray(metadata[key], dtype=np.float32).ravel()
                break

    if nv_list is None and isinstance(metadata, dict):
        nv_list = metadata.get("nv_list", None)

    if thresholds is None:
        raise ValueError("Could not find analysis_thresholds or thresholds.")
    if thresholds.shape != (num_nvs,):
        raise ValueError(
            f"Threshold shape {thresholds.shape} does not match {num_nvs} NVs."
        )
    if nv_list is None:
        raise ValueError("Could not load nv_list needed for camera coordinates.")

    if dark_wait_s is None and isinstance(metadata, dict):
        try:
            dark_wait_s = float(metadata.get("dark_wait_s", np.nan))
        except Exception:
            dark_wait_s = np.nan

    print(
        f"[large-file] counts ready: NVs={num_nvs}, runs={num_runs}, "
        f"dark_wait_s={dark_wait_s}",
        flush=True,
    )
    _print_memory("after selective counts load")

    return {
        "c11": c11,
        "c12": c12,
        "thresholds": thresholds,
        "nv_list": nv_list,
        "dark_wait_s": dark_wait_s,
    }


# =============================================================================
# Run-quality screening
# =============================================================================


def _detect_global_drop_runs(
    c11,
    c12,
    min_total_fraction=0.50,
    per_nv_collapse_fraction=0.25,
    max_collapsed_nv_fraction=0.80,
):
    """
    Detect whole-run acquisition/readout collapses from raw counts.

    Two independent tests are used for rep 11 and rep 12:

    1) total-signal test
       total(run) / median_total < min_total_fraction

    2) population-collapse test
       For every NV, first estimate its normal count from its median over runs.
       In each run, determine what fraction of NVs fell below
       per_nv_collapse_fraction * that NV's own median.
       If that fraction exceeds max_collapsed_nv_fraction, reject the run.

    This catches obvious zero/near-zero frames even when a camera background
    offset prevents the summed counts from becoming numerically close to zero.
    """
    c11 = np.asarray(c11, dtype=float)
    c12 = np.asarray(c12, dtype=float)

    if c11.shape != c12.shape or c11.ndim != 2:
        raise ValueError(
            f"Expected matching [nv,run] arrays, got {c11.shape} and {c12.shape}."
        )

    # ---- total-signal criterion ------------------------------------------------
    total11 = np.nansum(c11, axis=0)
    total12 = np.nansum(c12, axis=0)

    finite11 = np.isfinite(total11) & (total11 > 0)
    finite12 = np.isfinite(total12) & (total12 > 0)

    med11 = float(np.nanmedian(total11[finite11])) if np.any(finite11) else np.nan
    med12 = float(np.nanmedian(total12[finite12])) if np.any(finite12) else np.nan

    ratio11 = (
        total11 / med11
        if np.isfinite(med11) and med11 > 0
        else np.full(total11.shape, np.nan)
    )
    ratio12 = (
        total12 / med12
        if np.isfinite(med12) and med12 > 0
        else np.full(total12.shape, np.nan)
    )

    total_bad11 = (~np.isfinite(ratio11)) | (
        ratio11 < float(min_total_fraction)
    )
    total_bad12 = (~np.isfinite(ratio12)) | (
        ratio12 < float(min_total_fraction)
    )

    # ---- fraction-of-NVs collapse criterion -----------------------------------
    # Per-NV baselines prevent naturally dim NVs from being mistaken for failures.
    nv_med11 = np.nanmedian(c11, axis=1)
    nv_med12 = np.nanmedian(c12, axis=1)

    valid_nv11 = np.isfinite(nv_med11) & (nv_med11 > 0)
    valid_nv12 = np.isfinite(nv_med12) & (nv_med12 > 0)

    collapse11 = np.zeros(c11.shape, dtype=bool)
    collapse12 = np.zeros(c12.shape, dtype=bool)

    if np.any(valid_nv11):
        collapse11[valid_nv11, :] = (
            c11[valid_nv11, :]
            <= float(per_nv_collapse_fraction) * nv_med11[valid_nv11, None]
        )
        collapsed_fraction11 = np.mean(collapse11[valid_nv11, :], axis=0)
    else:
        collapsed_fraction11 = np.full(c11.shape[1], np.nan)

    if np.any(valid_nv12):
        collapse12[valid_nv12, :] = (
            c12[valid_nv12, :]
            <= float(per_nv_collapse_fraction) * nv_med12[valid_nv12, None]
        )
        collapsed_fraction12 = np.mean(collapse12[valid_nv12, :], axis=0)
    else:
        collapsed_fraction12 = np.full(c12.shape[1], np.nan)

    population_bad11 = np.isfinite(collapsed_fraction11) & (
        collapsed_fraction11 >= float(max_collapsed_nv_fraction)
    )
    population_bad12 = np.isfinite(collapsed_fraction12) & (
        collapsed_fraction12 >= float(max_collapsed_nv_fraction)
    )

    bad = total_bad11 | total_bad12 | population_bad11 | population_bad12

    bad_inds = np.where(bad)[0]
    print(
        f"[quality] global-drop rejection: {bad_inds.size}/{bad.size} runs rejected",
        flush=True,
    )
    print(
        f"[quality] criteria: total < {100*float(min_total_fraction):.0f}% median "
        f"OR >= {100*float(max_collapsed_nv_fraction):.0f}% of NVs below "
        f"{100*float(per_nv_collapse_fraction):.0f}% of their own median",
        flush=True,
    )

    if bad_inds.size:
        print(
            "[quality] rejected ORIGINAL run indices: "
            + ", ".join(str(int(v)) for v in bad_inds),
            flush=True,
        )
        for ind in bad_inds:
            reasons = []
            if total_bad11[ind]:
                reasons.append("rep11-total")
            if total_bad12[ind]:
                reasons.append("rep12-total")
            if population_bad11[ind]:
                reasons.append("rep11-population")
            if population_bad12[ind]:
                reasons.append("rep12-population")

            print(
                f"[quality]   R{ind}: "
                f"total11={ratio11[ind]:.3f}x, total12={ratio12[ind]:.3f}x, "
                f"collapsed11={100*collapsed_fraction11[ind]:.1f}%, "
                f"collapsed12={100*collapsed_fraction12[ind]:.1f}% "
                f"[{', '.join(reasons)}]",
                flush=True,
            )

    return {
        "bad_run_mask": bad,
        "good_run_mask": ~bad,
        "run_total11": total11,
        "run_total12": total12,
        "run_total_ratio11": ratio11,
        "run_total_ratio12": ratio12,
        "collapsed_nv_fraction11": collapsed_fraction11,
        "collapsed_nv_fraction12": collapsed_fraction12,
        "median_run_total11": med11,
        "median_run_total12": med12,
    }


# =============================================================================
# Charge-state analysis
# =============================================================================


def _analyze_charge_states(c11, c12, thresholds, good_run_mask=None):
    """Classify charge transitions without treating NaNs/gray-zone values as NV-."""
    c11 = np.asarray(c11, dtype=float)
    c12 = np.asarray(c12, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim == 1:
        if thresholds.shape[0] != c11.shape[0]:
            raise ValueError(
                f"Threshold length {thresholds.shape[0]} does not match "
                f"{c11.shape[0]} NVs."
            )
        threshold_col = thresholds[:, None]
    elif thresholds.ndim == 2:
        if thresholds.shape != c11.shape:
            raise ValueError(
                f"Per-run threshold shape {thresholds.shape} does not match "
                f"count shape {c11.shape}."
            )
        threshold_col = thresholds
    else:
        raise ValueError(
            "thresholds must have shape [nv] or [nv,run], "
            f"got {thresholds.shape}."
        )

    finite11 = np.isfinite(c11) & np.isfinite(threshold_col)
    finite12 = np.isfinite(c12) & np.isfinite(threshold_col)

    # A positive margin creates an explicit unclassified band around threshold.
    initial_nvm = finite11 & (c11 > (threshold_col + INITIAL_MARGIN_COUNTS))
    initial_nv0 = finite11 & (c11 <= (threshold_col - INITIAL_MARGIN_COUNTS))
    final_nvm = finite12 & (c12 > (threshold_col + FINAL_MARGIN_COUNTS))
    final_nv0 = finite12 & (c12 <= (threshold_col - FINAL_MARGIN_COUNTS))

    lost_mask = initial_nvm & final_nv0
    retained_mask = initial_nvm & final_nvm
    gained_mask = initial_nv0 & final_nvm

    # Keep the original meaning of eligible_count (confidently NV- at rep 11),
    # but use only classifiable rep-12 outcomes in transition probabilities.
    eligible = initial_nvm
    eligible_count = np.sum(eligible, axis=0).astype(int)
    evaluable_eligible_count = np.sum(lost_mask | retained_mask, axis=0).astype(int)
    initial_nv0_count = np.sum(initial_nv0, axis=0).astype(int)
    final_count = np.sum(final_nvm, axis=0).astype(int)
    lost = np.sum(lost_mask, axis=0).astype(int)
    retained = np.sum(retained_mask, axis=0).astype(int)
    gained = np.sum(gained_mask, axis=0).astype(int)

    loss_fraction = _safe_divide(lost, evaluable_eligible_count)
    retention = _safe_divide(retained, evaluable_eligible_count)
    net_change = final_count - eligible_count

    # Invalid acquisition runs must not define the baseline distribution and
    # must never become top anomaly candidates.
    if good_run_mask is not None:
        good_run_mask = np.asarray(good_run_mask, dtype=bool)
        if good_run_mask.shape != loss_fraction.shape:
            raise ValueError(
                f"good_run_mask shape {good_run_mask.shape} does not match "
                f"{loss_fraction.shape}"
            )
        loss_fraction = loss_fraction.copy()
        retention = retention.copy()
        loss_fraction[~good_run_mask] = np.nan
        retention[~good_run_mask] = np.nan

    loss_z, loss_med, loss_sigma = _robust_zscore(loss_fraction)
    loss_empirical_p = _empirical_upper_tail_p(loss_fraction)

    return {
        "eligible_mask": eligible,
        "initial_nv0_mask": initial_nv0,
        "final_nvm_mask": final_nvm,
        "final_nv0_mask": final_nv0,
        "switch_mask": lost_mask,
        "eligible_count": eligible_count,
        "evaluable_eligible_count": evaluable_eligible_count,
        "initial_nv0_count": initial_nv0_count,
        "final_count": final_count,
        "lost": lost,
        # Per-run retained COUNT. No retained_mask is exported/required by the
        # counts-only comparison code.
        "retained": retained,
        "gained": gained,
        "loss_fraction": loss_fraction,
        "retention": retention,
        "net_change": net_change,
        "loss_z": loss_z,
        "loss_median": loss_med,
        "loss_sigma": loss_sigma,
        "loss_empirical_p": loss_empirical_p,
    }


# =============================================================================
# Efficient short-range spatial screening for ALL 4000 runs
# =============================================================================


def _analyze_short_range_spatial(
    coords_xy,
    eligible_mask,
    switch_mask,
    um_per_pixel=0.43,
    short_range_um=30.0,
    pair_chunk_size=2000,
    good_run_mask=None,
):
    """
    Fast deterministic same-K spatial screening.

    For every run, count switched-NV pairs separated by <= short_range_um.
    The same-K expectation assumes the observed K switches are uniformly drawn
    from that run's eligible NVs:

        E[pairs] = M_eligible_close * K(K-1) / [N(N-1)]

    This is intentionally a screening statistic for the 4000-run file, not a
    per-run 2000-permutation Monte-Carlo test.
    """
    coords_um = np.asarray(coords_xy, dtype=float) * float(um_per_pixel)
    num_nvs, num_runs = switch_mask.shape

    tri_i, tri_j = np.triu_indices(num_nvs, k=1)
    dx = coords_um[tri_i, 0] - coords_um[tri_j, 0]
    dy = coords_um[tri_i, 1] - coords_um[tri_j, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    close = dist <= float(short_range_um)
    pair_i = tri_i[close]
    pair_j = tri_j[close]

    print(
        f"[spatial] close-pair geometry: {len(pair_i)} NV pairs within "
        f"{short_range_um:g} um",
        flush=True,
    )

    observed = np.zeros(num_runs, dtype=np.int64)
    eligible_close_pairs = np.zeros(num_runs, dtype=np.int64)

    chunk = max(1, int(pair_chunk_size))
    for start in range(0, len(pair_i), chunk):
        stop = min(start + chunk, len(pair_i))
        a = pair_i[start:stop]
        b = pair_j[start:stop]

        observed += np.sum(
            switch_mask[a, :] & switch_mask[b, :],
            axis=0,
            dtype=np.int64,
        )
        eligible_close_pairs += np.sum(
            eligible_mask[a, :] & eligible_mask[b, :],
            axis=0,
            dtype=np.int64,
        )

    n = np.sum(eligible_mask, axis=0).astype(float)
    k = np.sum(switch_mask, axis=0).astype(float)

    same_k_pair_probability = np.zeros(num_runs, dtype=float)
    good_n = n >= 2
    same_k_pair_probability[good_n] = (
        k[good_n]
        * (k[good_n] - 1.0)
        / (n[good_n] * (n[good_n] - 1.0))
    )

    expected = eligible_close_pairs * same_k_pair_probability
    enrichment = _safe_divide(observed, expected)
    pair_excess = observed.astype(float) - expected

    # Rejected acquisition runs must not define the spatial baseline.
    pair_excess_for_stats = pair_excess.astype(float).copy()
    if good_run_mask is not None:
        good_run_mask = np.asarray(good_run_mask, dtype=bool)
        if good_run_mask.shape != pair_excess_for_stats.shape:
            raise ValueError(
                f"good_run_mask shape {good_run_mask.shape} does not match "
                f"spatial run shape {pair_excess_for_stats.shape}."
            )
        pair_excess_for_stats[~good_run_mask] = np.nan

    spatial_z, spatial_med, spatial_sigma = _robust_zscore(pair_excess_for_stats)
    spatial_empirical_p = _empirical_upper_tail_p(pair_excess_for_stats)

    return {
        "short_range_um": float(short_range_um),
        "num_close_geometry_pairs": int(len(pair_i)),
        "observed_close_pairs": observed,
        "eligible_close_pairs": eligible_close_pairs,
        "expected_close_pairs_same_k": expected,
        "spatial_enrichment": enrichment,
        "pair_excess": pair_excess,
        "spatial_z": spatial_z,
        "spatial_median": spatial_med,
        "spatial_sigma": spatial_sigma,
        "spatial_empirical_p": spatial_empirical_p,
    }


# =============================================================================
# Raw-image drift and readout diagnostics
# =============================================================================


def _local_nv_centroid(img, x0, y0, radius_px=5):
    """Centroid only a small ROI; never cast the whole camera frame."""
    x0 = float(x0)
    y0 = float(y0)
    r = int(round(radius_px))
    xc = int(round(x0))
    yc = int(round(y0))

    x_min = max(0, xc - r)
    x_max = min(img.shape[1], xc + r + 1)
    y_min = max(0, yc - r)
    y_max = min(img.shape[0], yc + r + 1)
    if x_min >= x_max or y_min >= y_max:
        return None

    patch = np.asarray(img[y_min:y_max, x_min:x_max], dtype=np.float32)
    bg = float(np.nanpercentile(patch, 30))
    weights = patch - bg
    weights[(~np.isfinite(weights)) | (weights < 0)] = 0.0
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0:
        return None

    yy, xx = np.indices(patch.shape, dtype=np.float32)
    xx += x_min
    yy += y_min
    cx = float(np.sum(xx * weights, dtype=np.float64) / total)
    cy = float(np.sum(yy * weights, dtype=np.float64) / total)
    return cx, cy, total


def _integrate_single_nv(
    img,
    x0,
    y0,
    radius_px=3.0,
    bg_inner_px=5.0,
    bg_outer_px=8.0,
):
    """Local aperture integration with a small float32 patch only."""
    x0 = float(x0)
    y0 = float(y0)
    bound = int(np.ceil(bg_outer_px))
    xc = int(round(x0))
    yc = int(round(y0))

    x_min = max(0, xc - bound)
    x_max = min(img.shape[1], xc + bound + 1)
    y_min = max(0, yc - bound)
    y_max = min(img.shape[0], yc + bound + 1)
    if x_min >= x_max or y_min >= y_max:
        return np.nan, np.nan

    patch = np.asarray(img[y_min:y_max, x_min:x_max], dtype=np.float32)
    yy, xx = np.indices(patch.shape, dtype=np.float32)
    xx += x_min
    yy += y_min
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    signal_mask = rr <= radius_px
    bg_mask = (rr >= bg_inner_px) & (rr <= bg_outer_px)
    if np.sum(signal_mask) == 0 or np.sum(bg_mask) == 0:
        return np.nan, np.nan

    bg = float(np.nanmedian(patch[bg_mask]))
    signal = float(
        np.nansum(patch[signal_mask], dtype=np.float64)
        - bg * np.sum(signal_mask)
    )
    return signal, bg


def _estimate_run_drift_from_bright_nvs(
    img11,
    img12,
    coords_xy,
    counts11,
    counts12,
    thresholds,
    bright_margin_counts=5.0,
    roi_radius_px=5,
    max_reference_nvs=30,
):
    """Robust rep11->rep12 translation using references chosen from rep 11 only.

    Selecting references with a rep-12 brightness requirement would condition the
    artifact diagnostic on the outcome being tested and can hide genuine rep-12
    dimming/drift.  Rep 12 is therefore used only to measure the centroid shift.
    """
    ref_mask = (
        (counts11 > thresholds + bright_margin_counts)
        & np.isfinite(counts11)
        & np.isfinite(thresholds)
    )
    ref_inds = np.where(ref_mask)[0]

    if ref_inds.size == 0:
        return {
            "dx_px": np.nan,
            "dy_px": np.nan,
            "magnitude_px": np.nan,
            "scatter_px": np.nan,
            "num_reference_nvs": 0,
            "reference_nv_inds": np.array([], dtype=int),
        }

    brightness_score = counts11[ref_inds] - thresholds[ref_inds]
    order = np.argsort(brightness_score)[::-1]
    ref_inds = ref_inds[order[: int(max_reference_nvs)]]

    dx_list = []
    dy_list = []
    used = []

    for nv_ind in ref_inds:
        x0, y0 = coords_xy[nv_ind]
        cent11 = _local_nv_centroid(img11, x0, y0, radius_px=roi_radius_px)
        cent12 = _local_nv_centroid(img12, x0, y0, radius_px=roi_radius_px)
        if cent11 is None or cent12 is None:
            continue
        x11, y11, s11 = cent11
        x12, y12, s12 = cent12
        if s11 <= 0 or s12 <= 0:
            continue
        dx_list.append(x12 - x11)
        dy_list.append(y12 - y11)
        used.append(int(nv_ind))

    dx = np.asarray(dx_list, dtype=float)
    dy = np.asarray(dy_list, dtype=float)
    used = np.asarray(used, dtype=int)

    if dx.size < 3:
        return {
            "dx_px": np.nan,
            "dy_px": np.nan,
            "magnitude_px": np.nan,
            "scatter_px": np.nan,
            "num_reference_nvs": int(dx.size),
            "reference_nv_inds": used,
        }

    med_dx = np.nanmedian(dx)
    med_dy = np.nanmedian(dy)
    residual = np.sqrt((dx - med_dx) ** 2 + (dy - med_dy) ** 2)
    med_res = np.nanmedian(residual)
    mad_res = np.nanmedian(np.abs(residual - med_res))
    robust_sigma = 1.4826 * mad_res

    if np.isfinite(robust_sigma) and robust_sigma > 0:
        good = residual <= med_res + 4.0 * robust_sigma
    else:
        good = np.ones(dx.size, dtype=bool)

    dx_final = float(np.nanmedian(dx[good]))
    dy_final = float(np.nanmedian(dy[good]))
    mag = float(np.hypot(dx_final, dy_final))
    final_residual = np.sqrt(
        (dx[good] - dx_final) ** 2 + (dy[good] - dy_final) ** 2
    )
    scatter = float(np.nanmedian(final_residual))

    return {
        "dx_px": dx_final,
        "dy_px": dy_final,
        "magnitude_px": mag,
        "scatter_px": scatter,
        "num_reference_nvs": int(np.sum(good)),
        "reference_nv_inds": used[good],
    }


def _raw_image_ratio_for_run(
    img11,
    img12,
    coords_xy,
    reference_inds,
    drift_dx=0.0,
    drift_dy=0.0,
):
    ratios = []
    bg_changes = []

    for nv_ind in reference_inds:
        x0, y0 = coords_xy[nv_ind]
        sig11, bg11 = _integrate_single_nv(img11, x0, y0)
        sig12, bg12 = _integrate_single_nv(
            img12,
            x0 + drift_dx,
            y0 + drift_dy,
        )
        if np.isfinite(sig11) and np.isfinite(sig12) and sig11 > 0:
            ratios.append(sig12 / sig11)
        if np.isfinite(bg11) and np.isfinite(bg12):
            bg_changes.append(bg12 - bg11)

    ratio = float(np.nanmedian(ratios)) if ratios else np.nan
    bg_change = float(np.nanmedian(bg_changes)) if bg_changes else np.nan
    return ratio, bg_change


def _downsampled_image_correlation(img11, img12, stride=8):
    """Fast global similarity diagnostic using sparse pixels only."""
    a = np.asarray(img11[::stride, ::stride], dtype=np.float32).ravel()
    b = np.asarray(img12[::stride, ::stride], dtype=np.float32).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    if np.sum(good) < 10:
        return np.nan
    a = a[good]
    b = b[good]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _discard_stream_bytes(stream, num_bytes, chunk_bytes=8 * 1024 * 1024):
    remaining = int(num_bytes)
    while remaining > 0:
        data = stream.read(min(remaining, int(chunk_bytes)))
        if not data:
            raise EOFError("Unexpected end of compressed img_arrays stream.")
        remaining -= len(data)


def _read_frame_from_stream(stream, dtype, ny, nx):
    frame_bytes = int(ny) * int(nx) * int(dtype.itemsize)
    payload = stream.read(frame_bytes)
    if len(payload) != frame_bytes:
        raise EOFError("Unexpected end of img_arrays while reading a frame.")
    return np.frombuffer(payload, dtype=dtype, count=ny * nx).reshape(ny, nx)


def _iter_npz_image_pairs_stream(
    npz_path,
    rep_initial,
    rep_final,
    progress_every=50,
):
    """Sequentially yield (run_ind, rep11_image, rep12_image) with bounded RAM."""
    with zipfile.ZipFile(npz_path, "r") as archive:
        member = _find_npz_member_name(archive, "img_arrays")
        if member is None:
            raise KeyError("The NPZ does not contain img_arrays.npy.")
        info = archive.getinfo(member)

        with archive.open(member, "r") as stream:
            shape, fortran_order, dtype = _read_npy_header_from_stream(stream)
            if len(shape) != 6:
                raise ValueError(
                    "Expected img_arrays[exp,run,step,rep,y,x], "
                    f"got {shape}"
                )
            if fortran_order:
                raise ValueError(
                    "img_arrays is Fortran-order; this clean streaming path "
                    "expects normal C-order np.savez output."
                )

            num_exp, num_runs, num_steps, num_reps, ny, nx = map(int, shape)
            if num_exp < 1 or num_steps < 1:
                raise ValueError(f"Unexpected img_arrays shape {shape}")
            for rep in (rep_initial, rep_final):
                if not 0 <= int(rep) < num_reps:
                    raise IndexError(f"rep {rep} outside 0..{num_reps-1}")

            frame_bytes = ny * nx * int(dtype.itemsize)
            print(
                f"[large-file] streaming img_arrays: shape={shape}, "
                f"dtype={dtype}\n"
                f"             compressed={_format_bytes(info.compress_size)}, "
                f"uncompressed={_format_bytes(info.file_size)}",
                flush=True,
            )

            start_time = time.perf_counter()

            # This analysis uses exp=0 and step=0.  Group skipped frames into
            # large byte ranges so an 11-GB compressed member is decompressed
            # sequentially with far fewer Python-level read calls.
            target_reps = sorted({int(rep_initial), int(rep_final)})

            for run_ind in range(num_runs):
                frames = {}
                next_rep = 0

                for rep_ind in target_reps:
                    skip_frames = rep_ind - next_rep
                    if skip_frames > 0:
                        _discard_stream_bytes(
                            stream,
                            skip_frames * frame_bytes,
                        )

                    frames[rep_ind] = _read_frame_from_stream(
                        stream, dtype, ny, nx
                    )
                    next_rep = rep_ind + 1

                # Discard the remainder of step 0 and every later step for this
                # run in one grouped operation.
                trailing_frames = (num_reps - next_rep) + (
                    max(0, num_steps - 1) * num_reps
                )
                if trailing_frames > 0:
                    _discard_stream_bytes(
                        stream,
                        trailing_frames * frame_bytes,
                    )

                img11 = frames[int(rep_initial)]
                img12 = frames[int(rep_final)]

                yield run_ind, img11, img12

                if (
                    run_ind == 0
                    or (run_ind + 1) % max(1, int(progress_every)) == 0
                    or run_ind == num_runs - 1
                ):
                    elapsed = time.perf_counter() - start_time
                    rate = (run_ind + 1) / elapsed if elapsed > 0 else np.nan
                    print(
                        f"[large-file] image stream: run "
                        f"{run_ind + 1}/{num_runs} ({rate:.2f} runs/s)",
                        flush=True,
                    )


def _analyze_images_streaming(
    npz_path,
    coords_xy,
    c11,
    c12,
    thresholds,
    top_run_inds,
):
    num_runs = c11.shape[1]

    drift_dx = np.full(num_runs, np.nan, dtype=float)
    drift_dy = np.full(num_runs, np.nan, dtype=float)
    drift_mag = np.full(num_runs, np.nan, dtype=float)
    drift_scatter = np.full(num_runs, np.nan, dtype=float)
    drift_nrefs = np.zeros(num_runs, dtype=int)
    brightness_ratio = np.full(num_runs, np.nan, dtype=float)
    background_change = np.full(num_runs, np.nan, dtype=float)
    image_correlation = np.full(num_runs, np.nan, dtype=float)

    # Keep only a few candidate image pairs for final diagnostic plots.
    keep_images_for = set(int(v) for v in top_run_inds)
    candidate_images = {}

    seen_runs = 0
    for run_ind, img11, img12 in _iter_npz_image_pairs_stream(
        npz_path,
        REP_INITIAL,
        REP_FINAL,
        progress_every=PROGRESS_EVERY,
    ):
        seen_runs += 1
        if run_ind >= num_runs:
            raise ValueError(
                f"Image stream has more runs than counts ({num_runs})."
            )

        drift = _estimate_run_drift_from_bright_nvs(
            img11=img11,
            img12=img12,
            coords_xy=coords_xy,
            counts11=c11[:, run_ind],
            counts12=c12[:, run_ind],
            thresholds=thresholds,
            bright_margin_counts=BRIGHT_MARGIN_COUNTS,
            roi_radius_px=DRIFT_ROI_RADIUS_PX,
            max_reference_nvs=MAX_DRIFT_NVS,
        )

        drift_dx[run_ind] = drift["dx_px"]
        drift_dy[run_ind] = drift["dy_px"]
        drift_mag[run_ind] = drift["magnitude_px"]
        drift_scatter[run_ind] = drift["scatter_px"]
        drift_nrefs[run_ind] = drift["num_reference_nvs"]

        refs = drift["reference_nv_inds"]
        if refs.size > 0:
            ratio, bg = _raw_image_ratio_for_run(
                img11,
                img12,
                coords_xy,
                refs,
                drift_dx=drift["dx_px"] if np.isfinite(drift["dx_px"]) else 0.0,
                drift_dy=drift["dy_px"] if np.isfinite(drift["dy_px"]) else 0.0,
            )
            brightness_ratio[run_ind] = ratio
            background_change[run_ind] = bg

        image_correlation[run_ind] = _downsampled_image_correlation(
            img11, img12
        )

        if run_ind in keep_images_for:
            candidate_images[run_ind] = (
                np.asarray(img11).copy(),
                np.asarray(img12).copy(),
            )

    if seen_runs != num_runs:
        raise ValueError(
            f"Image/count run mismatch: streamed {seen_runs} image runs but "
            f"counts contains {num_runs} runs."
        )

    _print_memory("after streaming image diagnostics")

    return {
        "drift_dx": drift_dx,
        "drift_dy": drift_dy,
        "drift_mag": drift_mag,
        "drift_scatter": drift_scatter,
        "drift_nrefs": drift_nrefs,
        "brightness_ratio": brightness_ratio,
        "background_change": background_change,
        "image_correlation": image_correlation,
        "candidate_images": candidate_images,
    }


# =============================================================================
# Plotting
# =============================================================================


def _annotate_top(ax, x, y, top_inds, prefix="R"):
    for ind in top_inds:
        if np.isfinite(x[ind]) and np.isfinite(y[ind]):
            ax.annotate(
                f"{prefix}{ind}",
                (x[ind], y[ind]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
            )


def _make_figures(result):
    figures = {}
    runs = result["run"]
    top_inds = result["top_inds"]
    bad_run_mask = np.asarray(
        result.get("bad_run_mask", np.zeros(len(runs), dtype=bool)),
        dtype=bool,
    )

    def plot_good(values):
        """Return a float copy with rejected runs represented by NaN gaps."""
        arr = np.asarray(values, dtype=float).copy()
        if arr.ndim == 1 and arr.shape[0] == len(runs):
            arr[bad_run_mask] = np.nan
        return arr

    # -------------------------------------------------------------------------
    # FIGURE 0: acquisition quality / rejected runs
    # -------------------------------------------------------------------------
    if "run_total_ratio11" in result and "run_total_ratio12" in result:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(
            runs,
            result["run_total_ratio11"],
            linewidth=0.8,
            label=f"rep {REP_INITIAL}",
        )
        axes[0].plot(
            runs,
            result["run_total_ratio12"],
            linewidth=0.8,
            label=f"rep {REP_FINAL}",
        )
        axes[0].axhline(
            MIN_RUN_TOTAL_FRACTION,
            linestyle="--",
            linewidth=1.0,
            label="total-signal rejection threshold",
        )
        bad_inds = np.where(bad_run_mask)[0]
        if bad_inds.size:
            axes[0].scatter(
                runs[bad_inds],
                np.minimum(
                    result["run_total_ratio11"][bad_inds],
                    result["run_total_ratio12"][bad_inds],
                ),
                s=28,
                marker="x",
                label="rejected run",
            )
        axes[0].set_ylabel("Total raw signal / median")
        axes[0].set_title("Whole-run acquisition quality")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

        axes[1].plot(
            runs,
            100.0 * result["collapsed_nv_fraction11"],
            linewidth=0.8,
            label=f"rep {REP_INITIAL}",
        )
        axes[1].plot(
            runs,
            100.0 * result["collapsed_nv_fraction12"],
            linewidth=0.8,
            label=f"rep {REP_FINAL}",
        )
        axes[1].axhline(
            100.0 * MAX_COLLAPSED_NV_FRACTION,
            linestyle="--",
            linewidth=1.0,
            label="population-collapse threshold",
        )
        axes[1].set_xlabel("Original run index")
        axes[1].set_ylabel("Collapsed NVs (%)")
        axes[1].set_title(
            f"Fraction of NVs below {100*PER_NV_COLLAPSE_FRACTION:.0f}% "
            "of their own median"
        )
        axes[1].grid(alpha=0.2)
        axes[1].legend()

        fig.tight_layout()
        figures["run_quality_global_drop_rejection"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 1: populations and transition probability
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(runs, plot_good(result["eligible_count"]), linewidth=0.8, label="rep 11 NV-")
    axes[0].plot(runs, plot_good(result["final_count"]), linewidth=0.8, label="rep 12 NV-")
    axes[0].set_ylabel("NV- count")
    axes[0].set_title("Initial and final NV- population")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(runs, plot_good(result["lost"]), linewidth=0.8, label="NV- -> NV0")
    axes[1].plot(runs, plot_good(result["gained"]), linewidth=0.8, label="NV0 -> NV-")
    axes[1].set_ylabel("NV transitions")
    axes[1].set_title("True charge-state transitions")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    axes[2].plot(runs, 100.0 * plot_good(result["loss_fraction"]), linewidth=0.8)
    axes[2].axhline(
        100.0 * result["loss_median"],
        linestyle="--",
        linewidth=1.0,
        label=f"median = {100*result['loss_median']:.2f}%",
    )
    _annotate_top(
        axes[2], runs.astype(float), 100.0 * result["loss_fraction"], top_inds
    )
    axes[2].set_xlabel("Run index")
    axes[2].set_ylabel("P(NV- -> NV0) (%)")
    axes[2].set_title("Transition probability for all runs")
    axes[2].grid(alpha=0.2)
    axes[2].legend()

    fig.tight_layout()
    figures["charge_transitions_all_runs"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 2: histogram of per-run transition fraction
    # -------------------------------------------------------------------------
    loss_pct = 100.0 * np.asarray(result["loss_fraction"], dtype=float)
    good_loss = loss_pct[np.isfinite(loss_pct)]

    if good_loss.size > 0:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 6.5))

        ax.hist(
            good_loss,
            bins=int(TRANSITION_HIST_BINS),
            edgecolor="black",
            linewidth=0.5,
        )

        median_pct = float(np.nanmedian(good_loss))
        ax.axvline(
            median_pct,
            linestyle="--",
            linewidth=1.2,
            label=f"median = {median_pct:.2f}%",
        )

        # Translate robust-z thresholds back into transition-fraction units.
        if np.isfinite(result["loss_sigma"]) and result["loss_sigma"] > 0:
            for z_thr in SIGMA_THRESHOLDS:
                fraction_thr = result["loss_median"] + z_thr * result["loss_sigma"]
                ax.axvline(
                    100.0 * fraction_thr,
                    linestyle=":",
                    linewidth=1.0,
                    label=f"robust {z_thr:g} sigma = {100*fraction_thr:.2f}%",
                )

        num_rejected = int(np.sum(bad_run_mask))
        ax.set_xlabel("NV- -> NV0 transition fraction per run (%)")
        ax.set_ylabel("Number of runs")
        # ax.set_yscale("log")
        ax.set_title(
            "Distribution of per-run charge transition fraction\n"
            f"dark_wait_60s"
        )
        ax.grid(alpha=0.2)
        ax.legend()

        fig.tight_layout()
        figures["transition_fraction_histogram"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 3: robust anomaly score and empirical tail probability
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    loss_z_plot = plot_good(result["loss_z"])
    axes[0].plot(runs, loss_z_plot, linewidth=0.8)
    for z_thr in SIGMA_THRESHOLDS:
        axes[0].axhline(
            z_thr,
            linestyle="--",
            linewidth=1.0,
            label=f"robust z = {z_thr:g}",
        )

    robust_primary = np.where(result["primary_robust_outlier_mask"])[0]
    if robust_primary.size:
        axes[0].scatter(
            runs[robust_primary],
            result["loss_z"][robust_primary],
            s=28,
            marker="o",
            label=f">= {PRIMARY_OUTLIER_SIGMA:g} sigma outlier",
        )

    _annotate_top(axes[0], runs.astype(float), result["loss_z"], top_inds)
    axes[0].set_ylabel("Robust loss z")
    axes[0].set_title("Charge-loss robust-z screening")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2)

    axes[1].plot(runs, plot_good(result["loss_empirical_p"]), linewidth=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Run index")
    axes[1].set_ylabel("Empirical upper-tail p")
    axes[1].set_title("Empirical rarity of each run")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    figures["loss_anomaly_all_runs"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 4: reference-style Poisson coincidence distribution
    #
    # TOP ROW    = linear y scale: best for judging the peak and width.
    # BOTTOM ROW = log y scale: best for judging the rare upper tail.
    # LEFT       = real / unscrambled appended data.
    # RIGHT      = scrambled control.
    # -------------------------------------------------------------------------
    reference_poisson = result.get("reference_poisson")

    if reference_poisson is not None and reference_poisson.get("success", False):
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 11), sharex="col")

        x_vals = np.asarray(reference_poisson["x_vals"], dtype=int)

        def _plot_poisson_panel(
            ax,
            observed_hist,
            poisson_expected,
            nbinom_expected,
            title,
            log_scale=False,
            mark_thresholds=False,
            data_label="Data",
        ):
            ax.bar(
                x_vals,
                observed_hist,
                width=0.82,
                alpha=0.60,
                label=data_label,
            )

            ax.plot(
                x_vals,
                poisson_expected,
                marker="o",
                markersize=3,
                linewidth=1.4,
                label="Poisson pmf",
            )

            if nbinom_expected is not None:
                ax.plot(
                    x_vals,
                    nbinom_expected,
                    linestyle="-.",
                    linewidth=1.3,
                    label="Negative-binomial",
                )

            if mark_thresholds:
                for z_thr in SIGMA_THRESHOLDS:
                    summary = reference_poisson["threshold_summary"][float(z_thr)]
                    ax.axvline(
                        summary["k_threshold"],
                        linestyle="--",
                        linewidth=1.0,
                        label=(
                            f"{z_thr:g} sigma: "
                            f"K>={summary['k_threshold']}"
                        ),
                    )

            if log_scale:
                ax.set_yscale("log")
                ax.set_title(title + "\nlog y scale")
            else:
                ax.set_title(title + "\nlinear y scale")

            ax.set_xlabel("Number of NV- -> NV0 transitions")
            ax.set_ylabel("Number of occurrences")
            ax.grid(alpha=0.2)

            handles, labels = ax.get_legend_handles_labels()
            # Deduplicate legend entries while preserving order.
            dedup = dict(zip(labels, handles))
            ax.legend(
                dedup.values(),
                dedup.keys(),
                fontsize=8,
            )

        real_title = (
            "Unscrambled appended data"
            f"\nlambda=<K>={reference_poisson['lambda']:.3f}, "
            f"Var/lambda={reference_poisson['dispersion']:.2f}"
        )

        scrambled_title = (
            "Scrambled control"
            f"\nlambda=<K>={reference_poisson['scrambled_lambda']:.3f}, "
            f"Var/lambda={reference_poisson['scrambled_dispersion']:.2f}"
        )

        # Linear scale: exposes the central shape / goodness of fit.
        _plot_poisson_panel(
            axes[0, 0],
            reference_poisson["observed_hist"],
            reference_poisson["expected_dist"],
            reference_poisson.get("nbinom_expected_dist"),
            real_title,
            log_scale=False,
            mark_thresholds=True,
            data_label=f"Data ({reference_poisson['num_nvs']} NVs)",
        )
        _plot_poisson_panel(
            axes[0, 1],
            reference_poisson["scrambled_hist"],
            reference_poisson["scrambled_expected_dist"],
            reference_poisson.get("scrambled_nbinom_expected_dist"),
            scrambled_title,
            log_scale=False,
            mark_thresholds=False,
            data_label="Scrambled data",
        )

        # Log scale: exposes rare high-K events.
        _plot_poisson_panel(
            axes[1, 0],
            reference_poisson["observed_hist"],
            reference_poisson["expected_dist"],
            reference_poisson.get("nbinom_expected_dist"),
            real_title,
            log_scale=True,
            mark_thresholds=True,
            data_label=f"Data ({reference_poisson['num_nvs']} NVs)",
        )
        _plot_poisson_panel(
            axes[1, 1],
            reference_poisson["scrambled_hist"],
            reference_poisson["scrambled_expected_dist"],
            reference_poisson.get("scrambled_nbinom_expected_dist"),
            scrambled_title,
            log_scale=True,
            mark_thresholds=False,
            data_label="Scrambled data",
        )

        fig.suptitle(
            "Poisson coincidence distribution: "
            "linear core + logarithmic tail",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        figures["reference_poisson_linear_and_log"] = fig

        # ---------------------------------------------------------------------
        # FIGURE 4b: rarity of Poisson-threshold events
        # ---------------------------------------------------------------------
        thresholds = np.asarray(SIGMA_THRESHOLDS, dtype=float)
        observed_pct = []
        poisson_pct = []
        scrambled_pct = []

        for z_thr in thresholds:
            s = reference_poisson["threshold_summary"][float(z_thr)]
            observed_pct.append(s["observed_percent"])
            # poisson_pct.append(s["poisson_expected_percent"])
            # scrambled_pct.append(s["scrambled_percent"])

        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        x = np.arange(len(thresholds), dtype=float)
        width = 0.25

        ax.bar(
            x - width,
            observed_pct,
            width=width,
            label="Observed real runs",
        )
        # ax.bar(
        #     x,
        #     poisson_pct,
        #     width=width,
        #     label="Poisson expectation",
        # )
        # ax.bar(
        #     x + width,
        #     scrambled_pct,
        #     width=width,
        #     label="Scrambled control",
        # )

        # Percent spans orders of magnitude; use log if every displayed value is
        # positive. Zero observed bins remain visible through the text table.
        # all_positive = np.asarray(
        #     observed_pct + poisson_pct + scrambled_pct,
        #     dtype=float,
        # )
        all_positive = np.asarray(
                    observed_pct,
                    dtype=float,
                )
        if np.all(all_positive > 0):
            ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels([f">= {z:g} sigma" for z in thresholds])
        ax.set_ylabel("Runs in upper tail (%)")
        ax.set_title("How rare are high-coincidence events?")
        ax.grid(alpha=0.2, axis="y")
        ax.legend()

        for i, z_thr in enumerate(thresholds):
            s = reference_poisson["threshold_summary"][float(z_thr)]
            text_y = max(
                s["observed_percent"],
                s["poisson_expected_percent"],
                s["scrambled_percent"],
            )
            if text_y > 0:
                label = (
                    f"K>={s['k_threshold']}\\n"
                    f"obs {s['observed_count']}/{reference_poisson['num_shots']}"
                )
                ax.annotate(
                    label,
                    (i, text_y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

        fig.tight_layout()
        figures["reference_poisson_threshold_rarity"] = fig

    # Keep the more detailed exposure-corrected Poisson fit from V9 as a
    # complementary diagnostic rather than the primary Poisson visualization.
    poisson_result = result.get("poisson")
    hist_fit = (
        poisson_result.get("histogram_fit")
        if poisson_result is not None
        else None
    )

    if (
        SHOW_CURVE_FIT_POISSON_CROSSCHECK
        and hist_fit is not None
        and hist_fit.get("success", False)
    ):
        k = np.asarray(hist_fit["k"], dtype=int)
        hist_all = np.asarray(hist_fit["hist_all_good"], dtype=float)
        hist_base = np.asarray(hist_fit["hist_baseline"], dtype=float)
        fit_curve = np.asarray(hist_fit["model_curve"], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        axes[0].bar(
            k,
            hist_all,
            width=0.82,
            alpha=0.5,
            label="all good runs",
        )
        axes[0].plot(
            k,
            fit_curve,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="central-background Poisson fit",
        )
        axes[0].set_xlabel("NV- -> NV0 losses in one run")
        axes[0].set_ylabel("Number of runs")
        axes[0].set_title(
            f"Exposure-corrected cross-check\\n"
            f"fit mean={hist_fit['fit_mean_count']:.3f} +/- "
            f"{hist_fit['fit_mean_count_ste']:.3f}"
        )
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=8)

        positive = (hist_all > 0) | (fit_curve > 0)
        axes[1].bar(
            k[positive],
            hist_all[positive],
            width=0.82,
            alpha=0.5,
            label="observed",
        )
        axes[1].plot(
            k[positive],
            fit_curve[positive],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="Poisson fit",
        )
        axes[1].set_yscale("log")
        axes[1].set_xlabel("NV- -> NV0 losses in one run")
        axes[1].set_ylabel("Number of runs (log)")
        axes[1].set_title(
            f"Tail cross-check; chi2/dof={hist_fit['red_chi_sq']:.2f}"
        )
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        figures["poisson_exposure_corrected_crosscheck"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 5: Poisson local significance and trial-corrected significance
    # -------------------------------------------------------------------------
    if poisson_result is not None:
        local_sigma = plot_good(poisson_result["poisson_local_sigma"])
        trial_sigma = plot_good(poisson_result["poisson_trial_sigma"])

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(runs, local_sigma, linewidth=0.8)
        for z_thr in SIGMA_THRESHOLDS:
            axes[0].axhline(
                z_thr,
                linestyle="--",
                linewidth=1.0,
                label=f"{z_thr:g} sigma",
            )

        poisson_primary = np.where(result["primary_poisson_outlier_mask"])[0]
        if poisson_primary.size:
            axes[0].scatter(
                runs[poisson_primary],
                poisson_result["poisson_local_sigma"][poisson_primary],
                s=28,
                marker="o",
                label=f">= {PRIMARY_OUTLIER_SIGMA:g} sigma",
            )
            _annotate_top(
                axes[0],
                runs.astype(float),
                poisson_result["poisson_local_sigma"],
                poisson_primary,
            )

        axes[0].set_ylabel("Local Poisson sigma")
        axes[0].set_title(
            "Poisson upper-tail significance per run "
            "(before look-elsewhere correction)"
        )
        axes[0].grid(alpha=0.2)
        axes[0].legend(ncol=2)

        axes[1].plot(runs, trial_sigma, linewidth=0.8)
        for z_thr in SIGMA_THRESHOLDS:
            axes[1].axhline(z_thr, linestyle="--", linewidth=1.0)

        axes[1].set_xlabel("Run index")
        axes[1].set_ylabel("Trial-corrected sigma")
        axes[1].set_title(
            f"Bonferroni correction across "
            f"{poisson_result['num_valid_runs']} valid runs"
        )
        axes[1].grid(alpha=0.2)

        fig.tight_layout()
        figures["poisson_sigma_outliers_all_runs"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 6: threshold outlier counts + observed event percentages
    # -------------------------------------------------------------------------
    thresholds = np.asarray(SIGMA_THRESHOLDS, dtype=float)

    robust_counts = np.asarray([
        result["robust_outliers"]["by_threshold"][float(z)]["count"]
        for z in thresholds
    ], dtype=float)
    robust_pct = np.asarray([
        result["robust_outliers"]["by_threshold"][float(z)]["percent"]
        for z in thresholds
    ], dtype=float)

    poisson_counts = np.asarray([
        result["poisson_outliers"]["by_threshold"][float(z)]["count"]
        for z in thresholds
    ], dtype=float)
    poisson_pct = np.asarray([
        result["poisson_outliers"]["by_threshold"][float(z)]["percent"]
        for z in thresholds
    ], dtype=float)

    if result.get("spatial_outliers") is not None:
        spatial_counts = np.asarray([
            result["spatial_outliers"]["by_threshold"][float(z)]["count"]
            for z in thresholds
        ], dtype=float)
        spatial_pct = np.asarray([
            result["spatial_outliers"]["by_threshold"][float(z)]["percent"]
            for z in thresholds
        ], dtype=float)
    else:
        spatial_counts = np.zeros_like(thresholds)
        spatial_pct = np.zeros_like(thresholds)

    gaussian_pct = 100.0 * norm.sf(thresholds)

    poisson_model_pct = np.asarray([
        result["poisson"]["threshold_model_rarity"]["by_threshold"][float(z)][
            "expected_percent"
        ]
        for z in thresholds
    ], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    x = np.arange(len(thresholds), dtype=float)
    width = 0.25

    # Counts.
    axes[0].bar(x - width, robust_counts, width=width, label="robust loss z")
    axes[0].bar(x, poisson_counts, width=width, label="Poisson local sigma")
    axes[0].bar(x + width, spatial_counts, width=width, label="spatial z")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f">= {z:g} sigma" for z in thresholds])
    axes[0].set_ylabel("Observed outlier runs")
    axes[0].set_title("Observed outlier counts")
    axes[0].grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=8)

    # Percent / rarity. Use log y because 3, 4, 5 sigma span many decades.
    def positive_or_nan(arr):
        arr = np.asarray(arr, dtype=float).copy()
        arr[arr <= 0] = np.nan
        return arr

    axes[1].plot(
        x,
        positive_or_nan(robust_pct),
        marker="o",
        label="observed robust loss",
    )
    axes[1].plot(
        x,
        positive_or_nan(poisson_pct),
        marker="o",
        label="observed Poisson-local",
    )
    axes[1].plot(
        x,
        positive_or_nan(spatial_pct),
        marker="o",
        label="observed spatial",
    )
    axes[1].plot(
        x,
        gaussian_pct,
        marker="o",
        linestyle="--",
        label="ideal Gaussian tail",
    )
    axes[1].plot(
        x,
        poisson_model_pct,
        marker="o",
        linestyle=":",
        label="exact Poisson null",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f">= {z:g} sigma" for z in thresholds])
    axes[1].set_ylabel("Events (%)")
    axes[1].set_title("How rare are events at each threshold?")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    figures["sigma_threshold_outlier_counts_and_rarity"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 7: robust-loss and spatial-z distributions
    # -------------------------------------------------------------------------
    if result.get("spatial") is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        loss_z_vals = np.asarray(result["loss_z"], dtype=float)
        good_loss_z = loss_z_vals[
            np.isfinite(loss_z_vals) & (~bad_run_mask)
        ]
        axes[0].hist(
            good_loss_z,
            bins=60,
            edgecolor="black",
            linewidth=0.4,
        )
        for z_thr in SIGMA_THRESHOLDS:
            axes[0].axvline(
                z_thr,
                linestyle="--",
                linewidth=1.0,
                label=f"{z_thr:g} sigma",
            )
        axes[0].set_xlabel("Robust loss z")
        axes[0].set_ylabel("Number of runs")
        axes[0].set_title("Distribution of robust loss scores")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

        spatial_z_vals = np.asarray(result["spatial"]["spatial_z"], dtype=float)
        good_spatial_z = spatial_z_vals[
            np.isfinite(spatial_z_vals) & (~bad_run_mask)
        ]
        axes[1].hist(
            good_spatial_z,
            bins=60,
            edgecolor="black",
            linewidth=0.4,
        )
        for z_thr in SIGMA_THRESHOLDS:
            axes[1].axvline(
                z_thr,
                linestyle="--",
                linewidth=1.0,
                label=f"{z_thr:g} sigma",
            )
        axes[1].set_xlabel("Spatial z")
        axes[1].set_ylabel("Number of runs")
        axes[1].set_title("Distribution of spatial clustering scores")
        axes[1].grid(alpha=0.2)
        axes[1].legend()

        fig.tight_layout()
        figures["robust_and_spatial_z_histograms"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 8: spatial screening
    # -------------------------------------------------------------------------
    if result.get("spatial") is not None:
        spatial = result["spatial"]
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(runs, plot_good(spatial["observed_close_pairs"]), linewidth=0.8, label="observed")
        axes[0].plot(runs, plot_good(spatial["expected_close_pairs_same_k"]), linewidth=0.8, label="same-K expected")
        axes[0].set_ylabel("Close switched pairs")
        axes[0].set_title(
            f"Switched-NV pairs within {spatial['short_range_um']:g} um"
        )
        axes[0].grid(alpha=0.2)
        axes[0].legend()

        axes[1].plot(runs, plot_good(spatial["spatial_enrichment"]), linewidth=0.8)
        axes[1].axhline(1.0, linestyle="--", linewidth=1.0)
        axes[1].set_ylabel("Observed / expected")
        axes[1].set_title("Short-range spatial enrichment")
        axes[1].grid(alpha=0.2)

        axes[2].plot(runs, plot_good(spatial["spatial_z"]), linewidth=0.8)
        for z_thr in SIGMA_THRESHOLDS:
            axes[2].axhline(
                z_thr,
                linestyle="--",
                linewidth=1.0,
                label=f"z = {z_thr:g}",
            )

        spatial_primary = np.where(result["primary_spatial_outlier_mask"])[0]
        if spatial_primary.size:
            axes[2].scatter(
                runs[spatial_primary],
                spatial["spatial_z"][spatial_primary],
                s=28,
                marker="o",
                label=f">= {PRIMARY_OUTLIER_SIGMA:g} sigma outlier",
            )
            _annotate_top(
                axes[2],
                runs.astype(float),
                spatial["spatial_z"],
                spatial_primary,
            )

        axes[2].set_xlabel("Run index")
        axes[2].set_ylabel("Spatial z")
        axes[2].set_title("Per-run spatial clustering screening score")
        axes[2].grid(alpha=0.2)
        axes[2].legend(ncol=2)

        fig.tight_layout()
        figures["spatial_correlation_all_runs"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 5: drift over all runs
    # -------------------------------------------------------------------------
    if np.any(np.isfinite(result["drift_mag"])):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(runs, plot_good(result["drift_dx"]), linewidth=0.8, label="dx")
        axes[0].plot(runs, plot_good(result["drift_dy"]), linewidth=0.8, label="dy")
        axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
        axes[0].set_ylabel("Shift (px)")
        axes[0].set_title("Rep11 -> rep12 drift")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

        axes[1].plot(runs, plot_good(result["drift_mag"]), linewidth=0.8)
        axes[1].set_ylabel("|drift| (px)")
        axes[1].set_title("Drift magnitude")
        axes[1].grid(alpha=0.2)

        axes[2].plot(runs, plot_good(result["drift_scatter"]), linewidth=0.8, label="centroid scatter")
        axes[2].set_xlabel("Run index")
        axes[2].set_ylabel("Scatter (px)")
        axes[2].set_title("Reference-NV drift consistency")
        axes[2].grid(alpha=0.2)

        fig.tight_layout()
        figures["drift_all_runs"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 6: readout diagnostics
    # -------------------------------------------------------------------------
    if np.any(np.isfinite(result["brightness_ratio"])):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(runs, plot_good(result["brightness_ratio"]), linewidth=0.8)
        axes[0].axhline(1.0, linestyle="--", linewidth=1.0)
        axes[0].set_ylabel("rep12 / rep11")
        axes[0].set_title("Stable-NV raw brightness ratio (drift corrected)")
        axes[0].grid(alpha=0.2)

        axes[1].plot(runs, plot_good(result["background_change"]), linewidth=0.8)
        axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
        axes[1].set_ylabel("Background change")
        axes[1].set_title("Local camera background: rep12 - rep11")
        axes[1].grid(alpha=0.2)

        axes[2].plot(runs, plot_good(result["image_correlation"]), linewidth=0.8)
        axes[2].set_xlabel("Run index")
        axes[2].set_ylabel("Image Pearson r")
        axes[2].set_title("Downsampled rep11 / rep12 image similarity")
        axes[2].grid(alpha=0.2)

        fig.tight_layout()
        figures["readout_diagnostics_all_runs"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 7: transition probability versus drift -- EVERY RUN
    # -------------------------------------------------------------------------
    good = np.isfinite(result["drift_mag"]) & np.isfinite(result["loss_fraction"])
    if np.sum(good) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        x = result["drift_mag"]
        y = 100.0 * result["loss_fraction"]
        r_raw = _pearson_r(x, y)

        axes[0].scatter(x[good], y[good], s=18, alpha=0.55)
        _annotate_top(axes[0], x, y, top_inds)
        axes[0].set_xlabel("Drift magnitude (camera px)")
        axes[0].set_ylabel("P(NV- -> NV0) (%)")
        axes[0].set_title(f"Transition probability vs drift, all runs\nPearson r={r_raw:.3f}")
        axes[0].grid(alpha=0.2)

        # Since this dataset has one dark wait (0 s), centering by the global
        # median is the appropriate same-condition excess.
        excess = 100.0 * (result["loss_fraction"] - result["loss_median"])
        r_excess = _pearson_r(x, excess)
        axes[1].scatter(x[good], excess[good], s=18, alpha=0.55)
        axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
        _annotate_top(axes[1], x, excess, top_inds)
        axes[1].set_xlabel("Drift magnitude (camera px)")
        axes[1].set_ylabel("Transition excess above median (percentage points)")
        axes[1].set_title(f"Transition excess vs drift\nPearson r={r_excess:.3f}")
        axes[1].grid(alpha=0.2)

        fig.tight_layout()
        figures["transition_probability_vs_drift_all_runs"] = fig

        result["transition_drift_pearson_r"] = r_raw
        result["transition_excess_drift_pearson_r"] = r_excess

    # -------------------------------------------------------------------------
    # FIGURE 8: artifact/correlation cross-checks
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = result["drift_mag"]
    y = result["loss_z"]
    good = np.isfinite(x) & np.isfinite(y)
    axes[0, 0].scatter(x[good], y[good], s=18, alpha=0.55)
    axes[0, 0].set_xlabel("Drift magnitude (px)")
    axes[0, 0].set_ylabel("Loss z")
    axes[0, 0].set_title(f"Loss anomaly vs drift, r={_pearson_r(x, y):.3f}")
    axes[0, 0].grid(alpha=0.2)

    x = result["brightness_ratio"]
    y = result["loss_z"]
    good = np.isfinite(x) & np.isfinite(y)
    axes[0, 1].scatter(x[good], y[good], s=18, alpha=0.55)
    axes[0, 1].axvline(1.0, linestyle="--", linewidth=1.0)
    axes[0, 1].set_xlabel("Stable-NV rep12 / rep11 brightness")
    axes[0, 1].set_ylabel("Loss z")
    axes[0, 1].set_title(f"Loss anomaly vs brightness, r={_pearson_r(x, y):.3f}")
    axes[0, 1].grid(alpha=0.2)

    if result.get("spatial") is not None:
        x = result["loss_z"]
        y = result["spatial"]["spatial_z"]
        good = np.isfinite(x) & np.isfinite(y)
        axes[1, 0].scatter(x[good], y[good], s=18, alpha=0.55)
        axes[1, 0].axhline(
            PRIMARY_OUTLIER_SIGMA,
            linestyle="--",
            linewidth=1.0,
            label=f"spatial z = {PRIMARY_OUTLIER_SIGMA:g}",
        )
        axes[1, 0].axvline(
            PRIMARY_OUTLIER_SIGMA,
            linestyle="--",
            linewidth=1.0,
            label=f"loss z = {PRIMARY_OUTLIER_SIGMA:g}",
        )

        joint_inds = np.where(result["joint_loss_spatial_mask"])[0]
        if joint_inds.size:
            axes[1, 0].scatter(
                x[joint_inds],
                y[joint_inds],
                s=42,
                marker="o",
                label="joint outlier",
            )
            _annotate_top(axes[1, 0], x, y, joint_inds)

        axes[1, 0].set_xlabel("Robust loss z")
        axes[1, 0].set_ylabel("Spatial z")
        axes[1, 0].set_title(
            "Large-loss AND spatially clustered candidates"
        )
        axes[1, 0].grid(alpha=0.2)
        axes[1, 0].legend(fontsize=8)

        x = result["drift_mag"]
        y = result["spatial"]["spatial_z"]
        good = np.isfinite(x) & np.isfinite(y)
        axes[1, 1].scatter(x[good], y[good], s=18, alpha=0.55)
        axes[1, 1].set_xlabel("Drift magnitude (px)")
        axes[1, 1].set_ylabel("Spatial z")
        axes[1, 1].set_title(f"Spatial clustering vs drift, r={_pearson_r(x, y):.3f}")
        axes[1, 1].grid(alpha=0.2)
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")

    fig.tight_layout()
    figures["artifact_correlation_checks"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 9: top-event maps in physical coordinates
    # -------------------------------------------------------------------------
    coords_um = result["coords_xy"] * UM_PER_PIXEL
    n_maps = min(int(TOP_EVENT_MAPS), len(top_inds))
    if n_maps > 0:
        ncols = min(3, n_maps)
        nrows = int(np.ceil(n_maps / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), squeeze=False)

        for ax in axes.ravel():
            ax.axis("off")

        for plot_ind, run_ind in enumerate(top_inds[:n_maps]):
            ax = axes.ravel()[plot_ind]
            ax.axis("on")
            switched = result["switch_mask"][:, run_ind]
            ax.scatter(coords_um[:, 0], coords_um[:, 1], s=8, alpha=0.25, label="all NVs")
            ax.scatter(coords_um[switched, 0], coords_um[switched, 1], s=20, alpha=0.8, label="NV- -> NV0")
            spatial_z = (
                result["spatial"]["spatial_z"][run_ind]
                if result.get("spatial") is not None
                else np.nan
            )
            ax.set_title(
                f"R{run_ind}: loss={100*result['loss_fraction'][run_ind]:.2f}%\n"
                f"z_loss={result['loss_z'][run_ind]:.2f}, z_spatial={spatial_z:.2f}"
            )
            ax.set_xlabel("x (um)")
            ax.set_ylabel("y (um)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=7)

        fig.tight_layout()
        figures["top_event_maps"] = fig

    # -------------------------------------------------------------------------
    # FIGURE 10+: raw candidate image triptychs retained during streaming
    # -------------------------------------------------------------------------
    for run_ind, images in result.get("candidate_images", {}).items():
        img11, img12 = images
        diff = np.asarray(img12, dtype=np.float32) - np.asarray(img11, dtype=np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        axes[0].imshow(img11)
        axes[0].set_title(f"R{run_ind} rep {REP_INITIAL}")
        axes[1].imshow(img12)
        axes[1].set_title(f"R{run_ind} rep {REP_FINAL}")
        axes[2].imshow(diff)
        axes[2].set_title("rep12 - rep11")
        for ax in axes:
            ax.set_xlabel("camera x (px)")
            ax.set_ylabel("camera y (px)")
        fig.suptitle(
            f"Candidate R{run_ind}: loss={100*result['loss_fraction'][run_ind]:.2f}%, "
            f"loss z={result['loss_z'][run_ind]:.2f}, "
            f"drift={result['drift_mag'][run_ind]:.3f} px"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        figures[f"candidate_R{run_ind}_raw_images"] = fig

    # Counts-only cleanup: do not keep figures whose inputs are entirely
    # unavailable because image streaming/spatial analysis is disabled.
    if not CALCULATE_DRIFT:
        figures.pop("artifact_correlation_checks", None)
    if not CALCULATE_SPATIAL:
        figures.pop("top_event_maps", None)

    return figures


# =============================================================================
# Append-first combined 0-s analysis
# =============================================================================


def analyze_appended_particle_memory_files(datasets):
    """
    Append all runs from same-condition datasets FIRST, then perform ONE analysis.

    Important details
    -----------------
    * c11/c12 are concatenated along the run axis.
    * each dataset keeps its own saved charge thresholds; the combined threshold
      object is therefore [nv, global_run].
    * whole-run quality rejection is computed from the appended ensemble.
    * robust loss z, spatial z, Poisson tails, and event rarity are all computed
      ONCE from the appended ensemble.
    * reference-style scrambling is constrained within each source dataset.
    * raw-image drift streaming is intentionally skipped here; the counts-based
      combined analysis remains bounded-RAM and avoids decompressing both giant
      image members just to establish the statistical null.
    """
    print("\n" + "=" * 132)
    print("APPEND-FIRST SOURCE-OFF 0-s ANALYSIS")
    print("=" * 132)

    parts = []
    first_num_nvs = None
    reference_coords = None
    reference_nv_list = None

    global_start = 0

    for dataset_id, dataset in enumerate(datasets):
        label = str(dataset["label"])
        file_stem = str(dataset["file_stem"])
        override = dataset.get("npz_path_override")

        print(f"\n[append] loading dataset {dataset_id}: {label}", flush=True)
        metadata = _try_metadata_without_npz(file_stem)
        npz_path = _discover_npz_path(
            file_stem=file_stem,
            npz_path_override=override,
            metadata=metadata,
        )
        signature = _inspect_npz(npz_path)
        if not signature["valid"]:
            raise ValueError(
                f"Invalid particle-memory NPZ for {label}: {npz_path}"
            )

        small = _load_small_dataset_members(npz_path, metadata=metadata)
        c11 = np.asarray(small["c11"], dtype=np.float32)
        c12 = np.asarray(small["c12"], dtype=np.float32)
        thresholds = np.asarray(small["thresholds"], dtype=np.float32).ravel()
        nv_list = small["nv_list"]
        coords_xy = _coerce_img_coords(nv_list)

        num_nvs, num_runs_part = c11.shape

        if first_num_nvs is None:
            first_num_nvs = num_nvs
            reference_coords = coords_xy.copy()
            reference_nv_list = nv_list
        else:
            if num_nvs != first_num_nvs:
                raise ValueError(
                    f"Cannot append {label}: {num_nvs} NVs versus "
                    f"{first_num_nvs} in first dataset."
                )

            # Pair geometry should be the same. Remove an overall camera
            # translation before assessing coordinate consistency.
            delta = coords_xy - reference_coords
            global_shift = np.nanmedian(delta, axis=0)
            residual = delta - global_shift[None, :]
            residual_rms = float(
                np.sqrt(np.nanmean(np.sum(residual * residual, axis=1)))
            )
            print(
                f"[append] coordinate consistency after translation: "
                f"shift=({global_shift[0]:.3f}, {global_shift[1]:.3f}) px, "
                f"RMS residual={residual_rms:.3f} px",
                flush=True,
            )
            if residual_rms > 2.0:
                raise ValueError(
                    f"NV ordering/coordinates for {label} differ too much "
                    f"from first dataset (RMS residual={residual_rms:.3f} px)."
                )

        if thresholds.shape != (num_nvs,):
            raise ValueError(
                f"{label}: threshold shape {thresholds.shape} != ({num_nvs},)"
            )

        threshold_matrix = np.repeat(
            thresholds[:, None],
            num_runs_part,
            axis=1,
        )

        global_stop = global_start + num_runs_part
        parts.append(
            {
                "dataset_id": int(dataset_id),
                "label": label,
                "file_stem": file_stem,
                "npz_path": str(npz_path),
                "dark_wait_s": small["dark_wait_s"],
                "c11": c11,
                "c12": c12,
                "thresholds": threshold_matrix,
                "num_runs": int(num_runs_part),
                "global_start": int(global_start),
                "global_stop": int(global_stop),
            }
        )
        global_start = global_stop

    if not parts:
        raise ValueError("No datasets supplied.")

    # ------------------------------------------------------------------
    # APPEND FIRST
    # ------------------------------------------------------------------
    c11 = np.concatenate([p["c11"] for p in parts], axis=1)
    c12 = np.concatenate([p["c12"] for p in parts], axis=1)
    thresholds_by_run = np.concatenate(
        [p["thresholds"] for p in parts],
        axis=1,
    )

    num_nvs, num_runs = c11.shape
    runs = np.arange(num_runs, dtype=int)

    dataset_id_by_run = np.empty(num_runs, dtype=int)
    local_run_by_global = np.empty(num_runs, dtype=int)

    for p in parts:
        sl = slice(p["global_start"], p["global_stop"])
        dataset_id_by_run[sl] = p["dataset_id"]
        local_run_by_global[sl] = np.arange(p["num_runs"], dtype=int)

    print(
        f"\n[append] combined shape: NVs={num_nvs}, runs={num_runs} "
        f"from {len(parts)} datasets",
        flush=True,
    )
    for p in parts:
        print(
            f"[append]   global R{p['global_start']}..R{p['global_stop']-1} "
            f"= {p['label']} ({p['num_runs']} runs)",
            flush=True,
        )

    # ------------------------------------------------------------------
    # ONE quality analysis on appended counts
    # ------------------------------------------------------------------
    if REJECT_GLOBAL_DROP_RUNS:
        quality = _detect_global_drop_runs(
            c11,
            c12,
            min_total_fraction=MIN_RUN_TOTAL_FRACTION,
            per_nv_collapse_fraction=PER_NV_COLLAPSE_FRACTION,
            max_collapsed_nv_fraction=MAX_COLLAPSED_NV_FRACTION,
        )
    else:
        quality = {
            "bad_run_mask": np.zeros(num_runs, dtype=bool),
            "good_run_mask": np.ones(num_runs, dtype=bool),
            "run_total11": np.nansum(c11, axis=0),
            "run_total12": np.nansum(c12, axis=0),
            "run_total_ratio11": np.full(num_runs, np.nan),
            "run_total_ratio12": np.full(num_runs, np.nan),
            "collapsed_nv_fraction11": np.full(num_runs, np.nan),
            "collapsed_nv_fraction12": np.full(num_runs, np.nan),
            "median_run_total11": np.nan,
            "median_run_total12": np.nan,
        }

    # ------------------------------------------------------------------
    # ONE charge-state analysis on appended runs.
    # thresholds_by_run preserves the threshold saved with each measurement.
    # ------------------------------------------------------------------
    charge = _analyze_charge_states(
        c11,
        c12,
        thresholds_by_run,
        good_run_mask=quality["good_run_mask"],
    )

    finite_loss = np.where(np.isfinite(charge["loss_fraction"]))[0]
    top_inds = finite_loss[
        np.argsort(charge["loss_fraction"][finite_loss])[::-1]
    ][: min(TOP_N, len(finite_loss))]

    # ------------------------------------------------------------------
    # ONE spatial analysis on appended event masks.
    # ------------------------------------------------------------------
    spatial = None
    if CALCULATE_SPATIAL:
        spatial_eligible = charge["eligible_mask"].copy()
        spatial_switch = charge["switch_mask"].copy()
        spatial_eligible[:, quality["bad_run_mask"]] = False
        spatial_switch[:, quality["bad_run_mask"]] = False

        spatial = _analyze_short_range_spatial(
            coords_xy=reference_coords,
            eligible_mask=spatial_eligible,
            switch_mask=spatial_switch,
            um_per_pixel=UM_PER_PIXEL,
            short_range_um=SHORT_RANGE_UM,
            pair_chunk_size=PAIR_CHUNK_SIZE,
            good_run_mask=quality["good_run_mask"],
        )

        for key in (
            "observed_close_pairs",
            "eligible_close_pairs",
            "expected_close_pairs_same_k",
            "spatial_enrichment",
            "pair_excess",
            "spatial_z",
            "spatial_empirical_p",
        ):
            arr = np.asarray(spatial[key], dtype=float).copy()
            arr[quality["bad_run_mask"]] = np.nan
            spatial[key] = arr

    # ------------------------------------------------------------------
    # ONE Poisson/outlier analysis on appended ensemble.
    # ------------------------------------------------------------------
    poisson_result = _analyze_poisson_loss_outliers(
        lost=charge["lost"],
        evaluable_eligible_count=charge["evaluable_eligible_count"],
        loss_z=charge["loss_z"],
        good_run_mask=quality["good_run_mask"],
        baseline_max_abs_robust_z=POISSON_BASELINE_MAX_ABS_ROBUST_Z,
    )

    # Suppress old curve_fit figure; the requested reference PMF is primary.
    if (
        poisson_result.get("histogram_fit") is not None
        and not SHOW_CURVE_FIT_POISSON_CROSSCHECK
    ):
        poisson_result["histogram_fit"]["success"] = False

    reference_poisson = None
    if CALCULATE_REFERENCE_POISSON:
        reference_poisson = _reference_poisson_distribution(
            switch_mask=charge["switch_mask"],
            good_run_mask=quality["good_run_mask"],
            sigma_thresholds=SIGMA_THRESHOLDS,
            scramble_shift_per_nv=SCRAMBLE_SHIFT_PER_NV,
            run_group_ids=dataset_id_by_run,
        )

    robust_outliers = _observed_threshold_rarity(
        charge["loss_z"],
        good_run_mask=quality["good_run_mask"],
        thresholds=SIGMA_THRESHOLDS,
    )

    poisson_outliers = _observed_threshold_rarity(
        poisson_result["poisson_local_sigma"],
        good_run_mask=quality["good_run_mask"],
        thresholds=SIGMA_THRESHOLDS,
    )

    spatial_outliers = (
        _observed_threshold_rarity(
            spatial["spatial_z"],
            good_run_mask=quality["good_run_mask"],
            thresholds=SIGMA_THRESHOLDS,
        )
        if spatial is not None
        else None
    )

    primary_robust_mask = (
        quality["good_run_mask"]
        & np.isfinite(charge["loss_z"])
        & (charge["loss_z"] >= float(PRIMARY_OUTLIER_SIGMA))
    )
    primary_poisson_mask = (
        quality["good_run_mask"]
        & np.isfinite(poisson_result["poisson_local_sigma"])
        & (
            poisson_result["poisson_local_sigma"]
            >= float(PRIMARY_OUTLIER_SIGMA)
        )
    )
    if spatial is not None:
        primary_spatial_mask = (
            quality["good_run_mask"]
            & np.isfinite(spatial["spatial_z"])
            & (spatial["spatial_z"] >= float(PRIMARY_OUTLIER_SIGMA))
        )
    else:
        primary_spatial_mask = np.zeros(num_runs, dtype=bool)

    joint_loss_spatial_mask = primary_robust_mask & primary_spatial_mask
    joint_poisson_spatial_mask = primary_poisson_mask & primary_spatial_mask

    # No giant image streaming in append-first mode. Keep compatible fields.
    image_results = {
        "drift_dx": np.full(num_runs, np.nan),
        "drift_dy": np.full(num_runs, np.nan),
        "drift_mag": np.full(num_runs, np.nan),
        "drift_scatter": np.full(num_runs, np.nan),
        "drift_nrefs": np.full(num_runs, np.nan),
        "brightness_ratio": np.full(num_runs, np.nan),
        "background_change": np.full(num_runs, np.nan),
        "image_correlation": np.full(num_runs, np.nan),
        "candidate_images": {},
    }

    result = {
        "file_stem": "APPENDED-source_off_wait_0s",
        "dataset_label": "APPENDED source-off 0 s",
        "npz_path": [p["npz_path"] for p in parts],
        "source_parts": parts,
        "dark_wait_s": 0.0,
        "run": runs,
        "dataset_id_by_run": dataset_id_by_run,
        "local_run_by_global": local_run_by_global,
        "good_run_indices": runs[quality["good_run_mask"]],
        "rejected_run_indices": runs[quality["bad_run_mask"]],
        "coords_xy": reference_coords,
        "c11": c11,
        "c12": c12,
        "thresholds_by_run": thresholds_by_run,
        **quality,
        **charge,
        "spatial": spatial,
        "poisson": poisson_result,
        "reference_poisson": reference_poisson,
        "robust_outliers": robust_outliers,
        "spatial_outliers": spatial_outliers,
        "poisson_outliers": poisson_outliers,
        "primary_robust_outlier_mask": primary_robust_mask,
        "primary_poisson_outlier_mask": primary_poisson_mask,
        "primary_spatial_outlier_mask": primary_spatial_mask,
        "joint_loss_spatial_mask": joint_loss_spatial_mask,
        "joint_poisson_spatial_mask": joint_poisson_spatial_mask,
        "image_candidate_inds": np.array([], dtype=int),
        **image_results,
        "top_inds": top_inds,
    }

    figures = _make_figures(result)

    # Add dataset-boundary marks to run-index figures where practical.
    boundaries = [p["global_stop"] for p in parts[:-1]]
    for key in (
        "run_quality_global_drop_rejection",
        "charge_transitions_all_runs",
        "loss_anomaly_all_runs",
        "poisson_sigma_outliers_all_runs",
        "spatial_correlation_all_runs",
    ):
        fig = figures.get(key)
        if fig is None:
            continue
        for ax in fig.axes:
            for boundary in boundaries:
                ax.axvline(
                    boundary - 0.5,
                    linestyle=":",
                    linewidth=1.0,
                )

    # ------------------------------------------------------------------
    # Combined console summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 132)
    print("APPENDED DATASET SUMMARY")
    print("=" * 132)
    print(
        f"Total runs={num_runs}; accepted={int(np.sum(quality['good_run_mask']))}; "
        f"rejected={int(np.sum(quality['bad_run_mask']))}"
    )
    print(
        f"Empirical single-event resolution among accepted runs: "
        f"{100.0/max(1, np.sum(quality['good_run_mask'])):.6f}% "
        f"(~1 in {int(np.sum(quality['good_run_mask']))})"
    )

    if reference_poisson is not None and reference_poisson.get("success", False):
        print("\nREFERENCE POISSON AFTER APPENDING BOTH 0-s FILES")
        print("-" * 132)
        print(
            f"lambda=<K>={reference_poisson['lambda']:.4f}; "
            f"variance={reference_poisson['variance']:.4f}; "
            f"Var/lambda={reference_poisson['dispersion']:.3f}"
        )
        if reference_poisson["dispersion"] > 1.10:
            print(
                "Poisson is narrower than the measured distribution "
                "(overdispersion present)."
            )
            if reference_poisson.get("nbinom_expected_dist") is not None:
                print(
                    f"Negative-binomial diagnostic: "
                    f"n={reference_poisson['nbinom_n']:.3f}, "
                    f"p={reference_poisson['nbinom_p']:.5f}"
                )
        elif reference_poisson["dispersion"] < 0.90:
            print("Measured distribution is narrower than Poisson.")
        else:
            print("Variance is close to the Poisson expectation.")

        print(
            f"Scrambled: lambda={reference_poisson['scrambled_lambda']:.4f}; "
            f"Var/lambda={reference_poisson['scrambled_dispersion']:.3f}"
        )

        print("\nThreshold  Kcut   Real events                 Poisson null"
              "                  Scrambled control")
        print("-" * 132)
        for z_thr in SIGMA_THRESHOLDS:
            s = reference_poisson["threshold_summary"][float(z_thr)]
            obs_rarity = (
                f"1 in {s['observed_one_in']:.1f}"
                if np.isfinite(s["observed_one_in"])
                else "none observed"
            )
            pois_rarity = (
                f"1 in {s['poisson_one_in']:.0f}"
                if np.isfinite(s["poisson_one_in"])
                else "effectively zero"
            )
            scr_rarity = (
                f"1 in {s['scrambled_one_in']:.1f}"
                if np.isfinite(s["scrambled_one_in"])
                else "none observed"
            )
            print(
                f">={z_thr:.0f}σ       {s['k_threshold']:3d}   "
                f"{s['observed_count']:4d}/{reference_poisson['num_shots']} "
                f"= {s['observed_percent']:.6f}% ({obs_rarity:>14s})   "
                f"{s['poisson_expected_percent']:.6f}% "
                f"(~{s['poisson_expected_count']:.4g} runs; {pois_rarity:>12s})   "
                f"{s['scrambled_percent']:.6f}% ({scr_rarity:>14s})"
            )

    print("\nTOP APPENDED EVENTS WITH ORIGINAL FILE/RUN ID")
    print("-" * 132)
    print(
        "GlobalR  Dataset                       LocalR  Lost  Loss%  "
        "RobustZ  PoisSig  SpatialZ"
    )
    for ind in top_inds:
        did = int(dataset_id_by_run[ind])
        local = int(local_run_by_global[ind])
        spatial_z = (
            spatial["spatial_z"][ind] if spatial is not None else np.nan
        )
        print(
            f"{ind:7d}  {parts[did]['label']:<28s}  {local:6d}  "
            f"{charge['lost'][ind]:4d}  "
            f"{100*charge['loss_fraction'][ind]:6.2f}  "
            f"{charge['loss_z'][ind]:7.2f}  "
            f"{poisson_result['poisson_local_sigma'][ind]:7.2f}  "
            f"{spatial_z:8.2f}"
        )

    return result, figures


# =============================================================================
# Multi-dataset / pooled source-off analysis
# =============================================================================


def _concat_good(result, key):
    """Concatenate one per-run quantity from accepted runs in one result."""
    values = np.asarray(result[key])
    good = np.asarray(result["good_run_mask"], dtype=bool)
    if values.ndim != 1 or values.shape[0] != good.shape[0]:
        raise ValueError(
            f"Expected per-run 1D result[{key!r}], got {values.shape} "
            f"for {good.shape[0]} runs."
        )
    return values[good]


def _build_pooled_reference_poisson(results):
    """
    Pool the reference-style Poisson coincidence distributions from multiple
    source-off measurements.

    Real coincidence counts are concatenated across accepted runs.

    Scrambled controls are also concatenated, but each dataset is scrambled
    internally before pooling. This avoids circularly rolling an NV timeline
    across the boundary between two independent acquisitions.
    """
    real_parts = []
    scrambled_parts = []
    source_labels = []
    source_num_runs = []

    for result in results:
        ref = result.get("reference_poisson")
        if ref is None or not ref.get("success", False):
            continue

        real = np.asarray(ref["coincidences"], dtype=int)
        scrambled = np.asarray(ref["scrambled_coincidences"], dtype=int)
        if real.size == 0:
            continue

        real_parts.append(real)
        scrambled_parts.append(scrambled)
        source_labels.append(str(result.get("dataset_label", result["file_stem"])))
        source_num_runs.append(int(real.size))

    if not real_parts:
        return {"success": False}

    real = np.concatenate(real_parts)
    scrambled = np.concatenate(scrambled_parts)

    num_shots = int(real.size)
    lam = float(np.mean(real))
    scrambled_lam = float(np.mean(scrambled))

    k_max = int(
        max(
            np.max(real),
            np.max(scrambled),
            np.ceil(lam + 8.0 * np.sqrt(max(lam, 1.0))),
            np.ceil(scrambled_lam + 8.0 * np.sqrt(max(scrambled_lam, 1.0))),
        )
    )
    x_vals = np.arange(k_max + 1, dtype=int)

    observed_hist = np.bincount(real, minlength=k_max + 1).astype(float)
    scrambled_hist = np.bincount(scrambled, minlength=k_max + 1).astype(float)

    expected_dist = num_shots * poisson.pmf(x_vals, lam)
    scrambled_expected_dist = num_shots * poisson.pmf(
        x_vals,
        scrambled_lam,
    )

    var_real = float(np.var(real, ddof=1)) if num_shots > 1 else np.nan
    var_scrambled = (
        float(np.var(scrambled, ddof=1))
        if num_shots > 1
        else np.nan
    )

    dispersion_real = var_real / lam if lam > 0 else np.nan
    dispersion_scrambled = (
        var_scrambled / scrambled_lam
        if scrambled_lam > 0
        else np.nan
    )

    threshold_summary = {}
    for z_thr in SIGMA_THRESHOLDS:
        z_thr = float(z_thr)
        k_thr = _poisson_sigma_count_threshold(lam, z_thr)

        obs_count = int(np.sum(real >= k_thr))
        obs_fraction = obs_count / num_shots
        obs_percent = 100.0 * obs_fraction
        obs_one_in = num_shots / obs_count if obs_count > 0 else np.inf

        poisson_tail = float(poisson.sf(k_thr - 1, lam))
        poisson_percent = 100.0 * poisson_tail
        poisson_expected_count = num_shots * poisson_tail
        poisson_one_in = 1.0 / poisson_tail if poisson_tail > 0 else np.inf

        scr_count = int(np.sum(scrambled >= k_thr))
        scr_fraction = scr_count / num_shots
        scr_percent = 100.0 * scr_fraction
        scr_one_in = num_shots / scr_count if scr_count > 0 else np.inf

        threshold_summary[z_thr] = {
            "k_threshold": int(k_thr),
            "observed_count": obs_count,
            "observed_fraction": float(obs_fraction),
            "observed_percent": float(obs_percent),
            "observed_one_in": float(obs_one_in),
            "poisson_tail_probability": poisson_tail,
            "poisson_expected_percent": float(poisson_percent),
            "poisson_expected_count": float(poisson_expected_count),
            "poisson_one_in": float(poisson_one_in),
            "scrambled_count": scr_count,
            "scrambled_fraction": float(scr_fraction),
            "scrambled_percent": float(scr_percent),
            "scrambled_one_in": float(scr_one_in),
            "observed_to_poisson_rate_ratio": (
                float(obs_fraction / poisson_tail)
                if poisson_tail > 0
                else np.inf
            ),
            "observed_to_scrambled_rate_ratio": (
                float(obs_fraction / scr_fraction)
                if scr_fraction > 0
                else np.inf
            ),
        }

    return {
        "success": True,
        "source_labels": source_labels,
        "source_num_runs": source_num_runs,
        "num_datasets": len(real_parts),
        "num_shots": num_shots,
        "coincidences": real,
        "scrambled_coincidences": scrambled,
        "lambda": lam,
        "scrambled_lambda": scrambled_lam,
        "x_vals": x_vals,
        "observed_hist": observed_hist,
        "scrambled_hist": scrambled_hist,
        "expected_dist": expected_dist,
        "scrambled_expected_dist": scrambled_expected_dist,
        "variance": var_real,
        "scrambled_variance": var_scrambled,
        "dispersion": float(dispersion_real),
        "scrambled_dispersion": float(dispersion_scrambled),
        "threshold_summary": threshold_summary,
        "empirical_resolution_percent": 100.0 / num_shots,
    }


def _make_pooled_source_off_summary(results):
    """
    Combine accepted runs from all source-off datasets.

    The pooled robust-loss and spatial-z distributions are recalibrated from the
    concatenated raw per-run quantities rather than concatenating already
    normalized z scores.
    """
    if not results:
        return None, {}

    # ------------------------------ loss -------------------------------------
    pooled_loss_fraction = np.concatenate(
        [_concat_good(result, "loss_fraction") for result in results]
    ).astype(float)
    pooled_lost = np.concatenate(
        [_concat_good(result, "lost") for result in results]
    ).astype(float)
    pooled_eval_n = np.concatenate(
        [_concat_good(result, "evaluable_eligible_count") for result in results]
    ).astype(float)

    pooled_loss_z, pooled_loss_med, pooled_loss_sigma = _robust_zscore(
        pooled_loss_fraction
    )
    pooled_loss_rarity = _observed_threshold_rarity(
        pooled_loss_z,
        good_run_mask=np.ones(pooled_loss_z.shape, dtype=bool),
        thresholds=SIGMA_THRESHOLDS,
    )

    # ----------------------------- spatial -----------------------------------
    pair_excess_parts = []
    for result in results:
        spatial = result.get("spatial")
        if spatial is None:
            continue
        good = np.asarray(result["good_run_mask"], dtype=bool)
        pair_excess = np.asarray(spatial["pair_excess"], dtype=float)
        pair_excess_parts.append(pair_excess[good])

    if pair_excess_parts:
        pooled_pair_excess = np.concatenate(pair_excess_parts)
        pooled_spatial_z, pooled_spatial_med, pooled_spatial_sigma = _robust_zscore(
            pooled_pair_excess
        )
        pooled_spatial_rarity = _observed_threshold_rarity(
            pooled_spatial_z,
            good_run_mask=np.ones(pooled_spatial_z.shape, dtype=bool),
            thresholds=SIGMA_THRESHOLDS,
        )
    else:
        pooled_pair_excess = np.array([], dtype=float)
        pooled_spatial_z = np.array([], dtype=float)
        pooled_spatial_med = np.nan
        pooled_spatial_sigma = np.nan
        pooled_spatial_rarity = None

    # ---------------------- reference Poisson pooling ------------------------
    pooled_reference_poisson = _build_pooled_reference_poisson(results)

    # Exposure-corrected pooled Poisson tail, using the central pooled robust-z
    # background to estimate p0.
    valid_all = np.ones(pooled_lost.shape, dtype=bool)
    pooled_poisson = _analyze_poisson_loss_outliers(
        lost=pooled_lost,
        evaluable_eligible_count=pooled_eval_n,
        loss_z=pooled_loss_z,
        good_run_mask=valid_all,
        baseline_max_abs_robust_z=POISSON_BASELINE_MAX_ABS_ROBUST_Z,
    )
    pooled_poisson_rarity = _observed_threshold_rarity(
        pooled_poisson["poisson_local_sigma"],
        good_run_mask=valid_all,
        thresholds=SIGMA_THRESHOLDS,
    )

    pooled = {
        "num_datasets": len(results),
        "dataset_labels": [
            str(result.get("dataset_label", result["file_stem"]))
            for result in results
        ],
        "num_good_runs": int(pooled_loss_fraction.size),
        "loss_fraction": pooled_loss_fraction,
        "lost": pooled_lost,
        "evaluable_eligible_count": pooled_eval_n,
        "loss_z": pooled_loss_z,
        "loss_median": pooled_loss_med,
        "loss_sigma": pooled_loss_sigma,
        "loss_rarity": pooled_loss_rarity,
        "pair_excess": pooled_pair_excess,
        "spatial_z": pooled_spatial_z,
        "spatial_median": pooled_spatial_med,
        "spatial_sigma": pooled_spatial_sigma,
        "spatial_rarity": pooled_spatial_rarity,
        "poisson": pooled_poisson,
        "poisson_rarity": pooled_poisson_rarity,
        "reference_poisson": pooled_reference_poisson,
    }

    figures = {}

    # Pooled transition-fraction histogram.
    fig, ax = plt.subplots(figsize=(10, 6.5))
    loss_pct = 100.0 * pooled_loss_fraction
    ax.hist(
        loss_pct[np.isfinite(loss_pct)],
        bins=int(TRANSITION_HIST_BINS),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(
        100.0 * pooled_loss_med,
        linestyle="--",
        linewidth=1.2,
        label=f"pooled median = {100*pooled_loss_med:.3f}%",
    )
    if np.isfinite(pooled_loss_sigma) and pooled_loss_sigma > 0:
        for z_thr in SIGMA_THRESHOLDS:
            threshold = pooled_loss_med + float(z_thr) * pooled_loss_sigma
            ax.axvline(
                100.0 * threshold,
                linestyle=":",
                linewidth=1.0,
                label=f"{z_thr:g} sigma = {100*threshold:.3f}%",
            )
    ax.set_xlabel("NV- -> NV0 transition fraction per run (%)")
    ax.set_ylabel("Number of runs")
    ax.set_title(
        f"Pooled source-off 0-s background: {len(results)} datasets, "
        f"{pooled_loss_fraction.size} accepted runs"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["pooled_transition_fraction_histogram"] = fig

    # Pooled reference-style Poisson histogram.
    ref = pooled_reference_poisson
    if ref.get("success", False):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
        x_vals = np.asarray(ref["x_vals"], dtype=int)

        axes[0].bar(
            x_vals,
            ref["observed_hist"],
            width=0.82,
            alpha=0.6,
            label="pooled real runs",
        )
        axes[0].plot(
            x_vals,
            ref["expected_dist"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="Poisson pmf",
        )
        for z_thr in SIGMA_THRESHOLDS:
            s = ref["threshold_summary"][float(z_thr)]
            axes[0].axvline(
                s["k_threshold"],
                linestyle="--",
                linewidth=1.0,
                label=f"{z_thr:g} sigma: K>={s['k_threshold']}",
            )
        axes[0].set_yscale("log")
        axes[0].set_xlabel("NV- -> NV0 transitions in one run")
        axes[0].set_ylabel("Number of occurrences")
        axes[0].set_title(
            f"Pooled unscrambled\\n"
            f"lambda={ref['lambda']:.3f}, Var/lambda={ref['dispersion']:.2f}"
        )
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=8)

        axes[1].bar(
            x_vals,
            ref["scrambled_hist"],
            width=0.82,
            alpha=0.6,
            label="pooled scrambled control",
        )
        axes[1].plot(
            x_vals,
            ref["scrambled_expected_dist"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label="Poisson pmf",
        )
        axes[1].set_yscale("log")
        axes[1].set_xlabel("NV- -> NV0 transitions in pseudo-run")
        axes[1].set_title(
            f"Pooled scrambled\\n"
            f"lambda={ref['scrambled_lambda']:.3f}, "
            f"Var/lambda={ref['scrambled_dispersion']:.2f}"
        )
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        figures["pooled_reference_poisson_unscrambled_scrambled"] = fig

        # Pooled rarity comparison.
        fig, ax = plt.subplots(figsize=(10, 6.2))
        z_vals = np.asarray(SIGMA_THRESHOLDS, dtype=float)
        x = np.arange(len(z_vals), dtype=float)
        width = 0.25

        obs_pct = []
        pois_pct = []
        scr_pct = []
        for z_thr in z_vals:
            s = ref["threshold_summary"][float(z_thr)]
            obs_pct.append(s["observed_percent"])
            pois_pct.append(s["poisson_expected_percent"])
            scr_pct.append(s["scrambled_percent"])

        ax.bar(x - width, obs_pct, width=width, label="observed real")
        ax.bar(x, pois_pct, width=width, label="Poisson expectation")
        ax.bar(x + width, scr_pct, width=width, label="scrambled control")

        positive = np.asarray(obs_pct + pois_pct + scr_pct, dtype=float)
        if np.all(positive > 0):
            ax.set_yscale("log")

        ax.set_xticks(x)
        ax.set_xticklabels([f">= {z:g} sigma" for z in z_vals])
        ax.set_ylabel("Upper-tail runs (%)")
        ax.set_title(
            f"Pooled source-off event rarity ({ref['num_shots']} accepted runs)"
        )
        ax.grid(alpha=0.2, axis="y")
        ax.legend()
        fig.tight_layout()
        figures["pooled_reference_poisson_rarity"] = fig

    # Console summary.
    print("\n" + "=" * 132)
    print("POOLED SOURCE-OFF 0-s BACKGROUND")
    print("=" * 132)
    print(
        f"Datasets pooled: {len(results)} | accepted runs: "
        f"{pooled['num_good_runs']} | empirical single-event resolution: "
        f"{100.0/pooled['num_good_runs']:.6f}% "
        f"(~1 in {pooled['num_good_runs']})"
    )
    for result in results:
        print(
            f"  {result.get('dataset_label', result['file_stem'])}: "
            f"{int(np.sum(result['good_run_mask']))}/"
            f"{len(result['good_run_mask'])} accepted runs"
        )

    print("\nPOOLED ROBUST / POISSON / SPATIAL OUTLIER RATES")
    print("-" * 132)
    print(
        "Threshold   Metric          Events/valid      Percent       "
        "Empirical rarity"
    )
    print("-" * 132)

    for z_thr in SIGMA_THRESHOLDS:
        z_thr = float(z_thr)

        rr = pooled_loss_rarity["by_threshold"][z_thr]
        rr_rarity = (
            f"~1 in {rr['one_in']:.1f}"
            if np.isfinite(rr["one_in"])
            else f"none in {pooled_loss_rarity['num_valid']}"
        )
        print(
            f">={z_thr:1.0f} sigma    robust loss     "
            f"{rr['count']:5d}/{pooled_loss_rarity['num_valid']:<5d}   "
            f"{rr['percent']:10.6f}%   {rr_rarity}"
        )

        pr = pooled_poisson_rarity["by_threshold"][z_thr]
        pr_rarity = (
            f"~1 in {pr['one_in']:.1f}"
            if np.isfinite(pr["one_in"])
            else f"none in {pooled_poisson_rarity['num_valid']}"
        )
        print(
            f"             Poisson local   "
            f"{pr['count']:5d}/{pooled_poisson_rarity['num_valid']:<5d}   "
            f"{pr['percent']:10.6f}%   {pr_rarity}"
        )

        if pooled_spatial_rarity is not None:
            sr = pooled_spatial_rarity["by_threshold"][z_thr]
            sr_rarity = (
                f"~1 in {sr['one_in']:.1f}"
                if np.isfinite(sr["one_in"])
                else f"none in {pooled_spatial_rarity['num_valid']}"
            )
            print(
                f"             spatial         "
                f"{sr['count']:5d}/{pooled_spatial_rarity['num_valid']:<5d}   "
                f"{sr['percent']:10.6f}%   {sr_rarity}"
            )

    if pooled_reference_poisson.get("success", False):
        print("\nPOOLED REFERENCE-STYLE POISSON RARITY")
        print("-" * 132)
        print(
            f"lambda=<K>={pooled_reference_poisson['lambda']:.4f}, "
            f"variance={pooled_reference_poisson['variance']:.4f}, "
            f"Var/lambda={pooled_reference_poisson['dispersion']:.3f}"
        )
        print(
            f"scrambled lambda={pooled_reference_poisson['scrambled_lambda']:.4f}, "
            f"scrambled Var/lambda="
            f"{pooled_reference_poisson['scrambled_dispersion']:.3f}"
        )
        print(
            "Threshold  K cut   Observed real                 "
            "Poisson expectation              Scrambled"
        )
        for z_thr in SIGMA_THRESHOLDS:
            s = pooled_reference_poisson["threshold_summary"][float(z_thr)]
            obs_rarity = (
                f"~1 in {s['observed_one_in']:.1f}"
                if np.isfinite(s["observed_one_in"])
                else f"none in {pooled_reference_poisson['num_shots']}"
            )
            pois_rarity = (
                f"~1 in {s['poisson_one_in']:.0f}"
                if np.isfinite(s["poisson_one_in"])
                else "effectively zero"
            )
            scr_rarity = (
                f"~1 in {s['scrambled_one_in']:.1f}"
                if np.isfinite(s["scrambled_one_in"])
                else f"none in {pooled_reference_poisson['num_shots']}"
            )
            print(
                f">={z_thr:1.0f} sigma   K>={s['k_threshold']:3d}   "
                f"{s['observed_count']:4d}/{pooled_reference_poisson['num_shots']} "
                f"= {s['observed_percent']:.6f}% ({obs_rarity})   "
                f"{s['poisson_expected_percent']:.6f}% "
                f"(~{s['poisson_expected_count']:.4g} runs; {pois_rarity})   "
                f"{s['scrambled_percent']:.6f}% ({scr_rarity})"
            )

    return pooled, figures


# =============================================================================
# Main analysis
# =============================================================================


def analyze_big_particle_memory_file(
    file_stem,
    npz_path_override=None,
    dataset_label=None,
    max_runs=None,
):
    print("=" * 94, flush=True)
    print(f"[script] version: {SCRIPT_VERSION}", flush=True)
    print(f"[script] file stem: {file_stem}", flush=True)
    if dataset_label is not None:
        print(f"[script] dataset label: {dataset_label}", flush=True)
    print("=" * 94, flush=True)

    metadata = _try_metadata_without_npz(file_stem)
    npz_path = _discover_npz_path(
        file_stem=file_stem,
        npz_path_override=npz_path_override,
        metadata=metadata,
    )
    print(f"[large-file] using NPZ: {npz_path}", flush=True)

    signature = _inspect_npz(npz_path)
    if not signature["valid"]:
        raise ValueError(
            f"Refusing unvalidated NPZ: {npz_path}\n"
            f"counts={signature['counts_shape']}, "
            f"img_arrays={signature['images_shape']}"
        )

    small = _load_small_dataset_members(npz_path, metadata=metadata)
    c11 = small["c11"]
    c12 = small["c12"]
    thresholds = small["thresholds"]
    nv_list = small["nv_list"]

    # Optional run truncation. This is used for the 0-s acquisition so the
    # comparison can use the same number of shots as the shorter datasets.
    # Run indices remain the original first-run indices: 0 ... max_runs-1.
    source_num_runs = int(c11.shape[1])
    if max_runs is not None:
        max_runs = int(max_runs)
        if max_runs <= 0:
            raise ValueError("max_runs must be a positive integer or None.")
        use_num_runs = min(max_runs, source_num_runs)
        if use_num_runs < source_num_runs:
            print(
                f"[run selection] using FIRST {use_num_runs}/{source_num_runs} runs "
                f"for dataset {dataset_label or file_stem}",
                flush=True,
            )
            c11 = c11[:, :use_num_runs].copy()
            c12 = c12[:, :use_num_runs].copy()
        else:
            print(
                f"[run selection] requested max_runs={max_runs}; dataset has only "
                f"{source_num_runs}, so all runs are used.",
                flush=True,
            )

    coords_xy = _coerce_img_coords(nv_list)
    if coords_xy.shape != (len(thresholds), 2):
        raise ValueError(
            f"Coordinate shape {coords_xy.shape} does not match "
            f"{len(thresholds)} NVs."
        )

    if REJECT_GLOBAL_DROP_RUNS:
        quality = _detect_global_drop_runs(
            c11,
            c12,
            min_total_fraction=MIN_RUN_TOTAL_FRACTION,
            per_nv_collapse_fraction=PER_NV_COLLAPSE_FRACTION,
            max_collapsed_nv_fraction=MAX_COLLAPSED_NV_FRACTION,
        )
    else:
        num_runs_tmp = c11.shape[1]
        quality = {
            "bad_run_mask": np.zeros(num_runs_tmp, dtype=bool),
            "good_run_mask": np.ones(num_runs_tmp, dtype=bool),
            "run_total11": np.nansum(c11, axis=0),
            "run_total12": np.nansum(c12, axis=0),
            "run_total_ratio11": np.full(num_runs_tmp, np.nan),
            "run_total_ratio12": np.full(num_runs_tmp, np.nan),
            "collapsed_nv_fraction11": np.full(num_runs_tmp, np.nan),
            "collapsed_nv_fraction12": np.full(num_runs_tmp, np.nan),
            "median_run_total11": np.nan,
            "median_run_total12": np.nan,
        }

    charge = _analyze_charge_states(
        c11,
        c12,
        thresholds,
        good_run_mask=quality["good_run_mask"],
    )
    num_runs = c11.shape[1]

    if signature["has_images"] and CALCULATE_DRIFT:
        image_num_runs = int(signature["images_shape"][1])
        if max_runs is not None and num_runs < image_num_runs:
            raise ValueError(
                "Run truncation is currently supported in counts-only mode. "
                "Keep CALCULATE_DRIFT=False when max_runs is used."
            )
        if image_num_runs != num_runs:
            raise ValueError(
                f"Run-count mismatch before image streaming: counts has {num_runs} "
                f"runs but img_arrays has {image_num_runs}."
            )

    runs = np.arange(num_runs, dtype=int)

    finite_loss = np.where(np.isfinite(charge["loss_fraction"]))[0]
    top_inds = finite_loss[
        np.argsort(charge["loss_fraction"][finite_loss])[::-1]
    ][: min(TOP_N, len(finite_loss))]

    spatial = None
    if CALCULATE_SPATIAL:
        print("[spatial] calculating all-run short-range screening...", flush=True)
        spatial_eligible = charge["eligible_mask"].copy()
        spatial_switch = charge["switch_mask"].copy()
        spatial_eligible[:, quality["bad_run_mask"]] = False
        spatial_switch[:, quality["bad_run_mask"]] = False

        spatial = _analyze_short_range_spatial(
            coords_xy=coords_xy,
            eligible_mask=spatial_eligible,
            switch_mask=spatial_switch,
            um_per_pixel=UM_PER_PIXEL,
            short_range_um=SHORT_RANGE_UM,
            pair_chunk_size=PAIR_CHUNK_SIZE,
            good_run_mask=quality["good_run_mask"],
        )

        # Remove rejected acquisition runs from EVERY per-run spatial output.
        # Convert integer arrays to float where needed so NaN can represent a gap.
        for key in (
            "observed_close_pairs",
            "eligible_close_pairs",
            "expected_close_pairs_same_k",
            "spatial_enrichment",
            "pair_excess",
            "spatial_z",
            "spatial_empirical_p",
        ):
            arr = np.asarray(spatial[key], dtype=float).copy()
            arr[quality["bad_run_mask"]] = np.nan
            spatial[key] = arr

        _print_memory("after spatial screening")

    # -------------------------------------------------------------------------
    # Model-based + threshold-based outlier quantification
    # -------------------------------------------------------------------------
    poisson_result = _analyze_poisson_loss_outliers(
        lost=charge["lost"],
        evaluable_eligible_count=charge["evaluable_eligible_count"],
        loss_z=charge["loss_z"],
        good_run_mask=quality["good_run_mask"],
        baseline_max_abs_robust_z=POISSON_BASELINE_MAX_ABS_ROBUST_Z,
    )

    reference_poisson = None
    if CALCULATE_REFERENCE_POISSON:
        reference_poisson = _reference_poisson_distribution(
            switch_mask=charge["switch_mask"],
            good_run_mask=quality["good_run_mask"],
            sigma_thresholds=SIGMA_THRESHOLDS,
            scramble_shift_per_nv=SCRAMBLE_SHIFT_PER_NV,
        )

    # Rich threshold summaries: counts + observed percentages + "1 in N"
    # rarity + Gaussian reference expectation.
    robust_outliers = _observed_threshold_rarity(
        charge["loss_z"],
        good_run_mask=quality["good_run_mask"],
        thresholds=SIGMA_THRESHOLDS,
    )

    if spatial is not None:
        spatial_outliers = _observed_threshold_rarity(
            spatial["spatial_z"],
            good_run_mask=quality["good_run_mask"],
            thresholds=SIGMA_THRESHOLDS,
        )
    else:
        spatial_outliers = None

    poisson_outliers = _observed_threshold_rarity(
        poisson_result["poisson_local_sigma"],
        good_run_mask=quality["good_run_mask"],
        thresholds=SIGMA_THRESHOLDS,
    )

    primary_robust_mask = (
        quality["good_run_mask"]
        & np.isfinite(charge["loss_z"])
        & (charge["loss_z"] >= float(PRIMARY_OUTLIER_SIGMA))
    )
    primary_poisson_mask = (
        quality["good_run_mask"]
        & np.isfinite(poisson_result["poisson_local_sigma"])
        & (
            poisson_result["poisson_local_sigma"]
            >= float(PRIMARY_OUTLIER_SIGMA)
        )
    )

    if spatial is not None:
        primary_spatial_mask = (
            quality["good_run_mask"]
            & np.isfinite(spatial["spatial_z"])
            & (spatial["spatial_z"] >= float(PRIMARY_OUTLIER_SIGMA))
        )
    else:
        primary_spatial_mask = np.zeros(num_runs, dtype=bool)

    joint_loss_spatial_mask = primary_robust_mask & primary_spatial_mask
    joint_poisson_spatial_mask = primary_poisson_mask & primary_spatial_mask

    # Retain raw image pairs for the strongest statistically interesting runs,
    # not only the largest raw transition fractions.
    image_candidate_inds = np.unique(
        np.concatenate(
            [
                np.asarray(top_inds[:TOP_EVENT_MAPS], dtype=int),
                np.where(primary_robust_mask)[0],
                np.where(primary_poisson_mask)[0],
                np.where(primary_spatial_mask)[0],
                np.where(joint_loss_spatial_mask)[0],
                np.where(joint_poisson_spatial_mask)[0],
            ]
        )
    )

    if image_candidate_inds.size > int(MAX_RAW_OUTLIER_IMAGES):
        # Rank the union primarily by Poisson local sigma, then retain a bounded
        # number of raw image pairs so memory remains controlled.
        score = np.asarray(
            poisson_result["poisson_local_sigma"][image_candidate_inds],
            dtype=float,
        )
        score[~np.isfinite(score)] = -np.inf
        order = np.argsort(score)[::-1]
        image_candidate_inds = image_candidate_inds[
            order[: int(MAX_RAW_OUTLIER_IMAGES)]
        ]

    image_results = {
        "drift_dx": np.full(num_runs, np.nan),
        "drift_dy": np.full(num_runs, np.nan),
        "drift_mag": np.full(num_runs, np.nan),
        "drift_scatter": np.full(num_runs, np.nan),
        "drift_nrefs": np.zeros(num_runs, dtype=int),
        "brightness_ratio": np.full(num_runs, np.nan),
        "background_change": np.full(num_runs, np.nan),
        "image_correlation": np.full(num_runs, np.nan),
        "candidate_images": {},
    }

    if not CALCULATE_DRIFT:
        print(
            "[counts-only] img_arrays pixel data will NOT be read; "
            "skipping drift/image diagnostics.",
            flush=True,
        )

    if CALCULATE_DRIFT:
        if not signature["has_images"]:
            print("[large-file] no img_arrays found; skipping drift.", flush=True)
        else:
            print(
                f"[large-file] starting streaming image analysis for "
                f"{num_runs} runs; max drift references/run={MAX_DRIFT_NVS}",
                flush=True,
            )
            image_results = _analyze_images_streaming(
                npz_path=npz_path,
                coords_xy=coords_xy,
                c11=c11,
                c12=c12,
                thresholds=thresholds,
                top_run_inds=image_candidate_inds,
            )

            # Completely remove rejected runs from downstream image diagnostics.
            for key in (
                "drift_dx",
                "drift_dy",
                "drift_mag",
                "drift_scatter",
                "brightness_ratio",
                "background_change",
                "image_correlation",
            ):
                arr = np.asarray(image_results[key], dtype=float).copy()
                arr[quality["bad_run_mask"]] = np.nan
                image_results[key] = arr

            nrefs = np.asarray(image_results["drift_nrefs"], dtype=float).copy()
            nrefs[quality["bad_run_mask"]] = np.nan
            image_results["drift_nrefs"] = nrefs

    result = {
        "file_stem": file_stem,
        "dataset_label": (
            str(dataset_label) if dataset_label is not None else str(file_stem)
        ),
        "npz_path": str(npz_path),
        "dark_wait_s": small["dark_wait_s"],
        "source_num_runs": source_num_runs,
        "analyzed_num_runs": num_runs,
        "max_runs_requested": max_runs,
        "run": runs,
        "good_run_indices": runs[quality["good_run_mask"]],
        "rejected_run_indices": runs[quality["bad_run_mask"]],
        "coords_xy": coords_xy,
        "c11": c11,
        "c12": c12,
        **quality,
        **charge,
        "spatial": spatial,
        "poisson": poisson_result,
        "reference_poisson": reference_poisson,
        "robust_outliers": robust_outliers,
        "spatial_outliers": spatial_outliers,
        "poisson_outliers": poisson_outliers,
        "primary_robust_outlier_mask": primary_robust_mask,
        "primary_poisson_outlier_mask": primary_poisson_mask,
        "primary_spatial_outlier_mask": primary_spatial_mask,
        "joint_loss_spatial_mask": joint_loss_spatial_mask,
        "joint_poisson_spatial_mask": joint_poisson_spatial_mask,
        "image_candidate_inds": image_candidate_inds,
        **image_results,
        "top_inds": top_inds,
    }

    figures = _make_figures(result)

    bad_inds = np.where(result["bad_run_mask"])[0]
    print("\n" + "=" * 110)
    print("REJECTED GLOBAL-DROP RUNS")
    print("=" * 110)
    if bad_inds.size == 0:
        print("None")
    else:
        for ind in bad_inds:
            print(
                f"R{ind}: "
                f"rep11_total={result['run_total_ratio11'][ind]:.3f}x, "
                f"rep12_total={result['run_total_ratio12'][ind]:.3f}x, "
                f"collapsed11={100*result['collapsed_nv_fraction11'][ind]:.1f}%, "
                f"collapsed12={100*result['collapsed_nv_fraction12'][ind]:.1f}%"
            )

    reference_poisson = result.get("reference_poisson")
    if reference_poisson is not None and reference_poisson.get("success", False):
        print("\n" + "=" * 128)
        print("REFERENCE-STYLE POISSON COINCIDENCE DISTRIBUTION")
        print("=" * 128)
        print(
            f"Unscrambled: N={reference_poisson['num_shots']} good runs, "
            f"lambda=<K>={reference_poisson['lambda']:.4f} transitions/run, "
            f"variance={reference_poisson['variance']:.4f}, "
            f"Var/lambda={reference_poisson['dispersion']:.3f}"
        )
        print(
            f"Scrambled:   lambda=<K>={reference_poisson['scrambled_lambda']:.4f}, "
            f"variance={reference_poisson['scrambled_variance']:.4f}, "
            f"Var/lambda={reference_poisson['scrambled_dispersion']:.3f}"
        )

        print("\nPOISSON UPPER-TAIL EVENT RARITY")
        print("-" * 128)
        print(
            "Threshold   K cut   Observed real             Poisson expectation"
            "             Scrambled control        Excess(real/Poisson)"
        )
        print("-" * 128)

        for z_thr in SIGMA_THRESHOLDS:
            s = reference_poisson["threshold_summary"][float(z_thr)]

            obs_rarity = (
                f"~1 in {s['observed_one_in']:.1f}"
                if np.isfinite(s["observed_one_in"])
                else f"none in {reference_poisson['num_shots']}"
            )
            pois_rarity = (
                f"~1 in {s['poisson_one_in']:.0f}"
                if np.isfinite(s["poisson_one_in"])
                else "effectively zero"
            )
            scr_rarity = (
                f"~1 in {s['scrambled_one_in']:.1f}"
                if np.isfinite(s["scrambled_one_in"])
                else f"none in {reference_poisson['num_shots']}"
            )

            excess = s["observed_to_poisson_rate_ratio"]
            excess_text = (
                f"{excess:.2f}x"
                if np.isfinite(excess)
                else "inf"
            )

            print(
                f">={z_thr:1.0f} sigma    "
                f"K>={s['k_threshold']:3d}   "
                f"{s['observed_count']:4d}/{reference_poisson['num_shots']} "
                f"= {s['observed_percent']:.6f}% ({obs_rarity:>13s})   "
                f"{s['poisson_expected_percent']:.6f}% "
                f"(~{s['poisson_expected_count']:.4g} runs; {pois_rarity:>14s})   "
                f"{s['scrambled_percent']:.6f}% ({scr_rarity:>13s})   "
                f"{excess_text}"
            )

        print(
            "\nInterpretation: if the real upper tail substantially exceeds both "
            "the Poisson PMF and the scrambled control, the data contain more "
            "same-run multi-NV coincidences than expected from independent "
            "single-NV switching."
        )

    print("\n" + "=" * 128)
    print("OUTLIER QUANTIFICATION / EVENT RARITY")
    print("=" * 128)

    n_valid = result["robust_outliers"]["num_valid"]
    empirical_resolution_pct = result["robust_outliers"][
        "empirical_resolution_percent"
    ]

    print(
        f"Valid runs: {n_valid}  |  "
        f"empirical single-event resolution = "
        f"{empirical_resolution_pct:.5f}% (~1 in {n_valid})  |  "
        f"primary threshold = {PRIMARY_OUTLIER_SIGMA:g} sigma"
    )

    print(
        "Exposure-corrected Poisson baseline: "
        f"p0={100*result['poisson']['baseline_transition_probability']:.4f}%  |  "
        f"baseline runs={result['poisson']['num_baseline_runs']}  |  "
        f"Pearson dispersion={result['poisson']['pearson_dispersion']:.3f}"
    )

    hist_fit = result["poisson"].get("histogram_fit", {})
    if hist_fit.get("success", False):
        print(
            "Poisson histogram curve_fit: "
            f"mean={hist_fit['fit_mean_count']:.4f} +/- "
            f"{hist_fit['fit_mean_count_ste']:.4f} losses/run, "
            f"amplitude={hist_fit['fit_amplitude']:.1f} +/- "
            f"{hist_fit['fit_amplitude_ste']:.1f}, "
            f"chi2/dof={hist_fit['chi_sq']:.2f}/{hist_fit['dof']} "
            f"= {hist_fit['red_chi_sq']:.3f}"
        )
    else:
        print(
            "Poisson histogram curve_fit: FAILED/insufficient populated bins; "
            "exposure-corrected Poisson significance is still available."
        )
        if hist_fit.get("error"):
            print("  fit error:", hist_fit["error"])

    if np.isfinite(result["poisson"]["pearson_dispersion"]):
        if result["poisson"]["pearson_dispersion"] > 1.5:
            print(
                "WARNING: loss-count variance is substantially broader than "
                "Poisson; Poisson sigma may overstate tail significance."
            )
        elif result["poisson"]["pearson_dispersion"] < 0.67:
            print(
                "NOTE: loss-count fluctuations are narrower than a Poisson model."
            )

    print("\nOBSERVED EVENT RATE AT EACH SIGMA THRESHOLD")
    print("-" * 128)
    print(
        "Threshold  Metric           Events/valid     Percent        Empirical rarity"
        "          Null expectation"
    )
    print("-" * 128)

    for z_thr in SIGMA_THRESHOLDS:
        z_thr = float(z_thr)

        # Robust loss.
        rr = result["robust_outliers"]["by_threshold"][z_thr]
        rarity = (
            f"~1 in {rr['one_in']:.1f}"
            if np.isfinite(rr["one_in"])
            else f"none in {n_valid}"
        )
        print(
            f">={z_thr:1.0f}sigma    "
            f"{'robust loss':15s}  "
            f"{rr['count']:5d}/{result['robust_outliers']['num_valid']:<5d}  "
            f"{rr['percent']:10.5f}%   "
            f"{rarity:22s}  "
            f"Gaussian {rr['gaussian_tail_percent']:.6f}% "
            f"(~{rr['gaussian_expected_count']:.3g} runs)"
        )

        # Poisson-local sigma.
        pr = result["poisson_outliers"]["by_threshold"][z_thr]
        pm = result["poisson"]["threshold_model_rarity"]["by_threshold"][z_thr]
        rarity = (
            f"~1 in {pr['one_in']:.1f}"
            if np.isfinite(pr["one_in"])
            else f"none in {result['poisson_outliers']['num_valid']}"
        )
        print(
            f"           "
            f"{'Poisson local':15s}  "
            f"{pr['count']:5d}/{result['poisson_outliers']['num_valid']:<5d}  "
            f"{pr['percent']:10.5f}%   "
            f"{rarity:22s}  "
            f"Poisson {pm['expected_percent']:.6f}% "
            f"(~{pm['expected_count']:.3g} runs; "
            f"~1 in {pm['expected_one_in']:.1f})"
        )

        # Spatial.
        if result.get("spatial_outliers") is not None:
            sr = result["spatial_outliers"]["by_threshold"][z_thr]
            rarity = (
                f"~1 in {sr['one_in']:.1f}"
                if np.isfinite(sr["one_in"])
                else f"none in {result['spatial_outliers']['num_valid']}"
            )
            print(
                f"           "
                f"{'spatial':15s}  "
                f"{sr['count']:5d}/{result['spatial_outliers']['num_valid']:<5d}  "
                f"{sr['percent']:10.5f}%   "
                f"{rarity:22s}  "
                f"Gaussian {sr['gaussian_tail_percent']:.6f}% "
                f"(~{sr['gaussian_expected_count']:.3g} runs)"
            )

        print()

    joint_loss_spatial_inds = np.where(result["joint_loss_spatial_mask"])[0]
    joint_poisson_spatial_inds = np.where(result["joint_poisson_spatial_mask"])[0]

    if n_valid > 0:
        joint_loss_pct = 100.0 * joint_loss_spatial_inds.size / n_valid
        joint_poisson_pct = 100.0 * joint_poisson_spatial_inds.size / n_valid
    else:
        joint_loss_pct = np.nan
        joint_poisson_pct = np.nan

    print(
        f"Joint robust-loss >= {PRIMARY_OUTLIER_SIGMA:g} sigma AND "
        f"spatial >= {PRIMARY_OUTLIER_SIGMA:g} sigma: "
        f"{joint_loss_spatial_inds.size}/{n_valid} "
        f"({joint_loss_pct:.6f}%), runs={joint_loss_spatial_inds.tolist()}"
    )
    print(
        f"Joint Poisson-local >= {PRIMARY_OUTLIER_SIGMA:g} sigma AND "
        f"spatial >= {PRIMARY_OUTLIER_SIGMA:g} sigma: "
        f"{joint_poisson_spatial_inds.size}/{n_valid} "
        f"({joint_poisson_pct:.6f}%), runs={joint_poisson_spatial_inds.tolist()}"
    )

    poisson_rankable = np.where(
        np.isfinite(result["poisson"]["poisson_local_sigma"])
    )[0]
    poisson_top = poisson_rankable[
        np.argsort(
            result["poisson"]["poisson_local_sigma"][poisson_rankable]
        )[::-1]
    ][: min(TOP_N, len(poisson_rankable))]

    print("\nTOP POISSON-TAIL OUTLIERS")
    print("-" * 110)
    print(
        "Run   Lost   EvalN   Loss%  RobustZ  PoisSigma  PoisP       "
        "TrialSigma  SpatialZ"
    )
    for ind in poisson_top:
        spatial_z = (
            result["spatial"]["spatial_z"][ind]
            if result.get("spatial") is not None
            else np.nan
        )
        print(
            f"{ind:4d}  "
            f"{result['lost'][ind]:5d}  "
            f"{result['evaluable_eligible_count'][ind]:5d}  "
            f"{100*result['loss_fraction'][ind]:6.2f}  "
            f"{result['loss_z'][ind]:7.2f}  "
            f"{result['poisson']['poisson_local_sigma'][ind]:9.2f}  "
            f"{result['poisson']['poisson_tail_p'][ind]:.3e}  "
            f"{result['poisson']['poisson_trial_sigma'][ind]:10.2f}  "
            f"{spatial_z:8.2f}"
        )

    print("\n" + "=" * 110)
    print("TOP CHARGE-LOSS RUNS")
    print("=" * 110)
    print(
        "Run   Init   Final   Lost   Gain   Loss%    z_loss   p_emp    "
        "PoisSig  SpatialZ   Drift(px)  Raw12/11  ImgCorr"
    )
    print("-" * 110)

    for ind in top_inds:
        spatial_z = (
            spatial["spatial_z"][ind] if spatial is not None else np.nan
        )
        print(
            f"{ind:4d}  "
            f"{result['eligible_count'][ind]:5d}  "
            f"{result['final_count'][ind]:5d}  "
            f"{result['lost'][ind]:5d}  "
            f"{result['gained'][ind]:5d}  "
            f"{100*result['loss_fraction'][ind]:6.2f}  "
            f"{result['loss_z'][ind]:8.2f}  "
            f"{result['loss_empirical_p'][ind]:7.4f}  "
            f"{result['poisson']['poisson_local_sigma'][ind]:7.2f}  "
            f"{spatial_z:8.2f}  "
            f"{result['drift_mag'][ind]:9.3f}  "
            f"{result['brightness_ratio'][ind]:8.3f}  "
            f"{result['image_correlation'][ind]:7.3f}"
        )

    print("\nGENERATED FIGURES")
    print("-" * 110)
    for fig_ind, key in enumerate(figures, start=1):
        print(f"{fig_ind:2d}. {key}")
    print(f"Total figures: {len(figures)}")

    if np.any(np.isfinite(result["drift_mag"])):
        print("\nALL-RUN TRANSITION/DRIFT CORRELATION")
        print("-" * 110)
        print(
            "Pearson r [P(NV- -> NV0) vs drift] = "
            f"{_pearson_r(result['drift_mag'], result['loss_fraction']):.4f}"
        )

    return result, figures


# =============================================================================
# Run
# =============================================================================



# =============================================================================
# Characteristic distribution fits
# =============================================================================


def _logit(p):
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = float(x)
    if x >= 0:
        e = np.exp(-x)
        return 1.0 / (1.0 + e)
    e = np.exp(x)
    return e / (1.0 + e)


def _fit_beta_binomial_losses(lost, n_eval, valid_mask=None):
    """
    Fit a beta-binomial model to per-run NV- -> NV0 losses.

    For run r:
        K_r ~ BetaBinomial(N_r, alpha, beta)

    Reparameterize as:
        p = alpha / (alpha + beta)
        kappa = alpha + beta
        rho = 1 / (kappa + 1)

    p is the mean per-NV switching probability.
    rho quantifies extra-binomial / correlated run-to-run variation.

    rho -> 0 is the independent-binomial limit.
    """
    lost = np.asarray(lost, dtype=float)
    n_eval = np.asarray(n_eval, dtype=float)

    valid = (
        np.isfinite(lost)
        & np.isfinite(n_eval)
        & (n_eval > 0)
        & (lost >= 0)
        & (lost <= n_eval)
    )
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)

    k = np.rint(lost[valid]).astype(int)
    n = np.rint(n_eval[valid]).astype(int)

    if k.size < 10:
        return {"success": False, "num_runs": int(k.size)}

    # Independent-binomial MLE is a good initialization.
    p_binom = float(np.sum(k) / np.sum(n))
    p_binom = float(np.clip(p_binom, 1e-8, 1.0 - 1e-8))

    # Method-of-moments starting point for rho using the observed fraction.
    frac = k / n
    frac_var = float(np.var(frac, ddof=1)) if frac.size > 1 else 0.0
    mean_n = float(np.mean(n))
    binom_frac_var = p_binom * (1.0 - p_binom) / max(mean_n, 1.0)

    if frac_var > binom_frac_var and mean_n > 1:
        rho0 = (
            frac_var / max(binom_frac_var, 1e-15) - 1.0
        ) / (mean_n - 1.0)
        rho0 = float(np.clip(rho0, 1e-6, 0.2))
    else:
        rho0 = 1e-4

    kappa0 = max(1.0 / rho0 - 1.0, 1e-3)

    # Optimize in unconstrained coordinates:
    # theta[0] = logit(p)
    # theta[1] = log(kappa)
    x0 = np.array([_logit(p_binom), np.log(kappa0)], dtype=float)

    def objective(theta):
        p = _sigmoid(theta[0])
        kappa = float(np.exp(np.clip(theta[1], -20.0, 30.0)))

        alpha = p * kappa
        beta = (1.0 - p) * kappa

        ll = betabinom.logpmf(k, n, alpha, beta)
        if not np.all(np.isfinite(ll)):
            return 1e300
        return -float(np.sum(ll))

    opt = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 20000,
            "xatol": 1e-10,
            "fatol": 1e-8,
        },
    )

    theta = np.asarray(opt.x, dtype=float)
    p = _sigmoid(theta[0])
    kappa = float(np.exp(np.clip(theta[1], -20.0, 30.0)))
    alpha = p * kappa
    beta = (1.0 - p) * kappa
    rho = 1.0 / (kappa + 1.0)

    ll_beta_binom = -float(objective(theta))

    # Independent-binomial comparison.
    ll_binom = float(
        np.sum(
            binom.logpmf(
                k,
                n,
                p_binom,
            )
        )
    )

    # AIC: beta-binomial has 2 parameters; binomial has 1.
    aic_beta_binom = 2 * 2 - 2.0 * ll_beta_binom
    aic_binom = 2 * 1 - 2.0 * ll_binom
    delta_aic_binom_minus_beta = aic_binom - aic_beta_binom

    # Variance-inflation factor at a representative run size.
    n_mean = float(np.mean(n))
    variance_inflation = 1.0 + (n_mean - 1.0) * rho

    # Approximate standard errors from a numerical Hessian in transformed
    # coordinates. Failure here should not invalidate the fit itself.
    p_se = np.nan
    rho_se = np.nan
    try:
        eps0 = 1e-4
        eps1 = 1e-4
        f00 = objective(theta)

        def f(a, b):
            return objective(theta + np.array([a, b], dtype=float))

        h00 = (f(eps0, 0) - 2*f00 + f(-eps0, 0)) / eps0**2
        h11 = (f(0, eps1) - 2*f00 + f(0, -eps1)) / eps1**2
        h01 = (
            f(eps0, eps1)
            - f(eps0, -eps1)
            - f(-eps0, eps1)
            + f(-eps0, -eps1)
        ) / (4.0 * eps0 * eps1)

        hess = np.array([[h00, h01], [h01, h11]], dtype=float)
        cov_theta = np.linalg.inv(hess)

        dp_dtheta0 = p * (1.0 - p)
        # rho = 1/(exp(theta1)+1)
        drho_dtheta1 = -rho * (1.0 - rho)

        p_var = (dp_dtheta0**2) * cov_theta[0, 0]
        rho_var = (drho_dtheta1**2) * cov_theta[1, 1]

        if p_var >= 0:
            p_se = float(np.sqrt(p_var))
        if rho_var >= 0:
            rho_se = float(np.sqrt(rho_var))
    except Exception:
        pass

    return {
        "success": bool(opt.success) or np.isfinite(ll_beta_binom),
        "optimizer_success": bool(opt.success),
        "optimizer_message": str(opt.message),
        "num_runs": int(k.size),
        "mean_n": n_mean,
        "p": float(p),
        "p_se": float(p_se),
        "kappa": float(kappa),
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "rho_se": float(rho_se),
        "variance_inflation": float(variance_inflation),
        "loglike_beta_binom": float(ll_beta_binom),
        "aic_beta_binom": float(aic_beta_binom),
        "p_binom": float(p_binom),
        "loglike_binom": float(ll_binom),
        "aic_binom": float(aic_binom),
        "delta_aic_binom_minus_beta": float(delta_aic_binom_minus_beta),
        "k": k,
        "n": n,
    }


def _beta_binomial_expected_hist(fit, k_max=None):
    """
    Expected count histogram for a fitted beta-binomial, properly mixed over
    the actual per-run N_r values.
    """
    if fit is None or not fit.get("success", False):
        return np.array([], dtype=int), np.array([], dtype=float)

    k_obs = np.asarray(fit["k"], dtype=int)
    n = np.asarray(fit["n"], dtype=int)

    if k_max is None:
        k_max = int(np.max(k_obs))
    k_vals = np.arange(int(k_max) + 1, dtype=int)

    expected = np.zeros(k_vals.size, dtype=float)
    alpha = fit["alpha"]
    beta = fit["beta"]

    for n_r in n:
        valid_k = k_vals <= n_r
        expected[valid_k] += betabinom.pmf(
            k_vals[valid_k],
            int(n_r),
            alpha,
            beta,
        )

    return k_vals, expected


def _effective_dark_loss_rate(p0, p_t, wait_seconds):
    """
    Effective extra dark-loss rate using the 0-s measurement as baseline.

    Model:
        1 - p(t) = [1 - p(0)] exp(-Gamma_dark * t)

    so
        Gamma_dark = -ln[(1-p_t)/(1-p0)] / t

    With only 0 s and 60 s this is an effective two-point rate, not a validated
    full kinetic model.
    """
    p0 = float(p0)
    p_t = float(p_t)
    t = float(wait_seconds)

    if (
        not np.isfinite(p0)
        or not np.isfinite(p_t)
        or t <= 0
        or p0 >= 1
        or p_t >= 1
        or p_t <= p0
    ):
        return {
            "success": False,
            "additional_probability": np.nan,
            "rate_per_s": np.nan,
            "lifetime_s": np.nan,
        }

    q_extra = 1.0 - (1.0 - p_t) / (1.0 - p0)
    gamma = -np.log((1.0 - p_t) / (1.0 - p0)) / t
    lifetime = 1.0 / gamma if gamma > 0 else np.inf

    return {
        "success": True,
        "additional_probability": float(q_extra),
        "rate_per_s": float(gamma),
        "lifetime_s": float(lifetime),
    }


# =============================================================================
# Cross-dataset comparison plots
# =============================================================================


def _finite_1d(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def _mean_and_se(x):
    x = _finite_1d(x)
    if x.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(x))
    if x.size < 2:
        return mean, np.nan
    se = float(np.std(x, ddof=1) / np.sqrt(x.size))
    return mean, se


def _good_values(result, key):
    arr = np.asarray(result[key], dtype=float)
    good = np.asarray(result["good_run_mask"], dtype=bool)
    return arr[good]


def _comparison_summary_and_figures(results):
    """
    Compare source-off measurements acquired with 0-s and 60-s dark waits.

    Produces:
      * overlaid transition-fraction histograms and tail curves
      * overlaid loss-count / coincidence distributions (linear + log)
      * observed-real high-coincidence rarity comparison with side-by-side 0 s and 60 s bars at 3/4/5 sigma
      * quantitative summary bars with uncertainties
      * threshold-rate comparison plots for robust / Poisson / spatial outliers
    """
    if len(results) < 2:
        return None, {}

    labels = [str(r.get("dataset_label", r["file_stem"])) for r in results]
    figures = {}

    # ------------------------------------------------------------------
    # Collect quantitative summaries.
    # ------------------------------------------------------------------
    summary_rows = []
    characteristic_fits = []

    for result in results:
        label = str(result.get("dataset_label", result["file_stem"]))
        good = np.asarray(result["good_run_mask"], dtype=bool)

        beta_binom_fit = _fit_beta_binomial_losses(
            lost=result["lost"],
            n_eval=result["evaluable_eligible_count"],
            valid_mask=good,
        )
        characteristic_fits.append(beta_binom_fit)

        loss_fraction = np.asarray(result["loss_fraction"], dtype=float)[good]
        lost = np.asarray(result["lost"], dtype=float)[good]
        # Physical NV- populations for the comparison.
        #
        # eligible_count = all confidently NV- sites at rep 11.
        # final_count    = all confidently NV- sites at rep 12.
        #
        # evaluable_eligible_count is still the correct denominator for the
        # NV- -> NV0 transition probability, but it is not the quantity we want
        # to label as "mean initial NV-" in the summary plot.
        initial_nvm_count = np.asarray(
            result["eligible_count"],
            dtype=float,
        )[good]
        final_nvm_count = np.asarray(
            result["final_count"],
            dtype=float,
        )[good]

        mean_loss_frac, se_loss_frac = _mean_and_se(100.0 * loss_fraction)
        mean_lost, se_lost = _mean_and_se(lost)
        mean_init, se_init = _mean_and_se(initial_nvm_count)
        mean_final, se_final = _mean_and_se(final_nvm_count)

        ref = result.get("reference_poisson")
        if ref is not None and ref.get("success", False):
            lam = float(ref["lambda"])
            scr_lam = float(ref["scrambled_lambda"])
            fano = float(ref["dispersion"])
            scr_fano = float(ref["scrambled_dispersion"])
        else:
            lam = np.nan
            scr_lam = np.nan
            fano = np.nan
            scr_fano = np.nan

        row = {
            "label": label,
            "n_good": int(np.sum(good)),
            "mean_transition_frac_pct": mean_loss_frac,
            "se_transition_frac_pct": se_loss_frac,
            "mean_lost": mean_lost,
            "se_lost": se_lost,
            "mean_initial_nvminus": mean_init,
            "se_initial_nvminus": se_init,
            "mean_final_nvminus": mean_final,
            "se_final_nvminus": se_final,
            "poisson_lambda": lam,
            "scrambled_lambda": scr_lam,
            "fano_real": fano,
            "fano_scrambled": scr_fano,
            "beta_binom_p": (
                float(beta_binom_fit["p"])
                if beta_binom_fit.get("success", False)
                else np.nan
            ),
            "beta_binom_p_se": (
                float(beta_binom_fit["p_se"])
                if beta_binom_fit.get("success", False)
                else np.nan
            ),
            "beta_binom_rho": (
                float(beta_binom_fit["rho"])
                if beta_binom_fit.get("success", False)
                else np.nan
            ),
            "beta_binom_rho_se": (
                float(beta_binom_fit["rho_se"])
                if beta_binom_fit.get("success", False)
                else np.nan
            ),
            "beta_binom_variance_inflation": (
                float(beta_binom_fit["variance_inflation"])
                if beta_binom_fit.get("success", False)
                else np.nan
            ),
            "delta_aic_binom_minus_beta": (
                float(beta_binom_fit["delta_aic_binom_minus_beta"])
                if beta_binom_fit.get("success", False)
                else np.nan
            ),
        }

        # Threshold rarity summaries
        for z_thr in SIGMA_THRESHOLDS:
            z_thr = float(z_thr)
            rr = result["robust_outliers"]["by_threshold"][z_thr]
            pr = result["poisson_outliers"]["by_threshold"][z_thr]
            row[f"robust_ge_{int(z_thr)}sigma_count"] = int(rr["count"])
            row[f"robust_ge_{int(z_thr)}sigma_percent"] = float(rr["percent"])
            row[f"poisson_ge_{int(z_thr)}sigma_count"] = int(pr["count"])
            row[f"poisson_ge_{int(z_thr)}sigma_percent"] = float(pr["percent"])

            if result.get("spatial_outliers") is not None:
                sr = result["spatial_outliers"]["by_threshold"][z_thr]
                row[f"spatial_ge_{int(z_thr)}sigma_count"] = int(sr["count"])
                row[f"spatial_ge_{int(z_thr)}sigma_percent"] = float(sr["percent"])
            else:
                row[f"spatial_ge_{int(z_thr)}sigma_count"] = 0
                row[f"spatial_ge_{int(z_thr)}sigma_percent"] = np.nan

        summary_rows.append(row)

    # ------------------------------------------------------------------
    # Print comparison summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 140)
    print("COMPARISON OF SOURCE-OFF MEASUREMENTS: 0-s WAIT vs 60-s WAIT")
    print("=" * 140)
    print(
        "Dataset                           GoodRuns   Mean lost   Mean loss%   "
        "Mean init NV-   Mean final NV-   Poisson λ   Fano(real)   Fano(scr)"
    )
    print("-" * 140)
    for row in summary_rows:
        print(
            f"{row['label']:<32s}  "
            f"{row['n_good']:7d}   "
            f"{row['mean_lost']:8.3f}   "
            f"{row['mean_transition_frac_pct']:9.4f}%   "
            f"{row['mean_initial_nvminus']:12.3f}   "
            f"{row['mean_final_nvminus']:13.3f}   "
            f"{row['poisson_lambda']:9.3f}   "
            f"{row['fano_real']:10.3f}   "
            f"{row['fano_scrambled']:9.3f}"
        )

    print("\nCHARACTERISTIC BETA-BINOMIAL FIT")
    print("-" * 140)
    print(
        "Dataset                           p(loss)         rho(overdisp.)   "
        "variance inflation   ΔAIC[binom-beta]"
    )
    print("-" * 140)
    for row in summary_rows:
        p_pct = 100.0 * row["beta_binom_p"]
        p_se_pct = 100.0 * row["beta_binom_p_se"]
        print(
            f"{row['label']:<32s}  "
            f"{p_pct:8.4f}% +/- {p_se_pct:7.4f}%   "
            f"{row['beta_binom_rho']:.6f} +/- "
            f"{row['beta_binom_rho_se']:.6f}   "
            f"{row['beta_binom_variance_inflation']:10.3f}   "
            f"{row['delta_aic_binom_minus_beta']:12.2f}"
        )

    # Two-point effective dark-loss rate from the fitted p values.
    effective_dark_rate = None
    if len(summary_rows) >= 2:
        # Use explicit wait metadata when possible.
        wait_vals = []
        for result in results:
            wait_vals.append(float(result.get("dark_wait_s", np.nan)))

        # Find 0-s and longest positive-wait conditions.
        zero_inds = [
            i for i, t in enumerate(wait_vals)
            if np.isfinite(t) and abs(t) < 1e-12
        ]
        positive_inds = [
            i for i, t in enumerate(wait_vals)
            if np.isfinite(t) and t > 0
        ]

        if zero_inds and positive_inds:
            i0 = zero_inds[0]
            it = max(positive_inds, key=lambda i: wait_vals[i])

            effective_dark_rate = _effective_dark_loss_rate(
                summary_rows[i0]["beta_binom_p"],
                summary_rows[it]["beta_binom_p"],
                wait_vals[it],
            )

            if effective_dark_rate.get("success", False):
                print("\nEFFECTIVE EXTRA DARK-LOSS PARAMETER")
                print("-" * 140)
                print(
                    f"Using {wait_vals[i0]:g} s as baseline and "
                    f"{wait_vals[it]:g} s as delayed point:"
                )
                print(
                    f"  additional dark-loss probability over "
                    f"{wait_vals[it]:g} s = "
                    f"{100*effective_dark_rate['additional_probability']:.4f}%"
                )
                print(
                    f"  Gamma_dark = "
                    f"{effective_dark_rate['rate_per_s']:.6e} s^-1"
                )
                print(
                    f"  effective lifetime 1/Gamma_dark = "
                    f"{effective_dark_rate['lifetime_s']:.1f} s "
                    f"= {effective_dark_rate['lifetime_s']/60.0:.2f} min"
                )
                print(
                    "  NOTE: this is a two-point effective rate; more wait times "
                    "are needed to establish an exponential kinetic model."
                )

    print("\nEVENT-RATE COMPARISON")
    print("-" * 140)
    for z_thr in SIGMA_THRESHOLDS:
        z_thr = float(z_thr)
        print(f">= {z_thr:.0f} sigma threshold")
        for row in summary_rows:
            print(
                f"  {row['label']:<30s}  "
                f"robust: {row[f'robust_ge_{int(z_thr)}sigma_count']:4d} "
                f"({row[f'robust_ge_{int(z_thr)}sigma_percent']:.6f}%)   "
                f"poisson: {row[f'poisson_ge_{int(z_thr)}sigma_count']:4d} "
                f"({row[f'poisson_ge_{int(z_thr)}sigma_percent']:.6f}%)   "
                f"spatial: {row[f'spatial_ge_{int(z_thr)}sigma_count']:4d} "
                f"({row[f'spatial_ge_{int(z_thr)}sigma_percent']:.6f}%)"
            )

    # ------------------------------------------------------------------
    # Figure 1: transition-fraction histograms + upper tails.
    # ------------------------------------------------------------------
    all_loss_pct = []
    for result in results:
        vals = 100.0 * _good_values(result, "loss_fraction")
        all_loss_pct.append(vals)

    finite_all = np.concatenate([_finite_1d(v) for v in all_loss_pct])
    if finite_all.size > 0:
        lo = float(np.nanmin(finite_all))
        hi = float(np.nanmax(finite_all))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            bins = 30
        else:
            bins = np.linspace(lo, hi, int(TRANSITION_HIST_BINS) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

        for label, vals in zip(labels, all_loss_pct):
            vals = _finite_1d(vals)
            if vals.size == 0:
                continue

            axes[0].hist(
                vals,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                density=True,
                label=label,
            )

            vals_sorted = np.sort(vals)
            n = vals_sorted.size
            tail = np.arange(n, 0, -1, dtype=float) / n
            axes[1].plot(
                vals_sorted,
                100.0 * tail,
                linewidth=1.5,
                label=label,
            )

        axes[0].set_xlabel("NV- -> NV0 transition fraction per run (%)")
        axes[0].set_ylabel("Probability density")
        axes[0].set_title("Distribution of transition fraction")
        axes[0].grid(alpha=0.2)
        axes[0].legend(fontsize=8)

        axes[1].set_xlabel("NV- -> NV0 transition fraction per run (%)")
        axes[1].set_ylabel("Runs with >= this transition fraction (%)")
        axes[1].set_yscale("log")
        axes[1].set_title("Upper-tail comparison")
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        figures["comparison_transition_fraction_distribution"] = fig

    # ------------------------------------------------------------------
    # Figure 2: same-run transition-count / coincidence distributions.
    # Plot in probability mass form so the two measurements can be compared
    # directly even if accepted run counts differ.
    # ------------------------------------------------------------------
    # Build a common x range from all datasets.
    ref_max = 0
    for result in results:
        ref = result.get("reference_poisson")
        if ref is not None and ref.get("success", False):
            ref_max = max(ref_max, int(np.max(ref["x_vals"])))

    if ref_max > 0:
        x = np.arange(ref_max + 1, dtype=int)
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 11), sharex=True)

        for result, label in zip(results, labels):
            ref = result.get("reference_poisson")
            if ref is None or not ref.get("success", False):
                continue

            obs_hist = np.zeros_like(x, dtype=float)
            scr_hist = np.zeros_like(x, dtype=float)
            exp = np.zeros_like(x, dtype=float)
            exp_scr = np.zeros_like(x, dtype=float)

            xr = np.asarray(ref["x_vals"], dtype=int)
            obs_hist[xr] = np.asarray(ref["observed_hist"], dtype=float)
            scr_hist[xr] = np.asarray(ref["scrambled_hist"], dtype=float)
            exp[xr] = np.asarray(ref["expected_dist"], dtype=float)
            exp_scr[xr] = np.asarray(ref["scrambled_expected_dist"], dtype=float)

            n_real = max(1, int(ref["num_shots"]))
            prob_obs = obs_hist / n_real
            prob_scr = scr_hist / n_real
            prob_exp = exp / n_real
            prob_exp_scr = exp_scr / n_real

            axes[0, 0].step(
                x,
                prob_obs,
                where="mid",
                linewidth=1.5,
                label=f"{label} data",
            )
            axes[0, 0].plot(
                x,
                prob_exp,
                linestyle="--",
                linewidth=1.2,
                label=f"{label} Poisson",
            )

            axes[0, 1].step(
                x,
                prob_scr,
                where="mid",
                linewidth=1.5,
                label=f"{label} scrambled",
            )
            axes[0, 1].plot(
                x,
                prob_exp_scr,
                linestyle="--",
                linewidth=1.2,
                label=f"{label} Poisson",
            )

            axes[1, 0].step(
                x,
                np.where(prob_obs > 0, prob_obs, np.nan),
                where="mid",
                linewidth=1.5,
                label=f"{label} data",
            )
            axes[1, 0].plot(
                x,
                np.where(prob_exp > 0, prob_exp, np.nan),
                linestyle="--",
                linewidth=1.2,
                label=f"{label} Poisson",
            )

            axes[1, 1].step(
                x,
                np.where(prob_scr > 0, prob_scr, np.nan),
                where="mid",
                linewidth=1.5,
                label=f"{label} scrambled",
            )
            axes[1, 1].plot(
                x,
                np.where(prob_exp_scr > 0, prob_exp_scr, np.nan),
                linestyle="--",
                linewidth=1.2,
                label=f"{label} Poisson",
            )

        axes[0, 0].set_title("Real data — linear scale")
        axes[0, 1].set_title("Scrambled control — linear scale")
        axes[1, 0].set_title("Real data — log scale")
        axes[1, 1].set_title("Scrambled control — log scale")

        for ax in axes[0, :]:
            ax.set_ylabel("Probability mass per run")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        for ax in axes[1, :]:
            ax.set_ylabel("Probability mass per run")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        for ax in axes[1, :]:
            ax.set_xlabel("Number of NV- -> NV0 transitions")

        fig.suptitle(
            "Comparison of coincidence / transition-count distributions",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        figures["comparison_transition_count_poisson"] = fig

    # ------------------------------------------------------------------
    # NEW V19F COMPARISON:
    # "How rare are high-coincidence events?"
    #
    # Compare ALL configured wait times (0 s / 30 s / 60 s here).
    # Keep all three rarity references:
    #   * observed real
    #   * Poisson expectation
    #   * scrambled control
    #
    # Left panel: grouped bars for every wait time and rarity reference.
    # Right panel: lines connecting the observed-real 3σ / 4σ / 5σ values
    # for every wait time.
    # ------------------------------------------------------------------
    threshold_z = np.asarray(SIGMA_THRESHOLDS, dtype=float)

    comparison_rows = []

    for result in results:
        ref = result.get("reference_poisson")
        if ref is None or not ref.get("success", False):
            continue

        dark_wait_s = float(result.get("dark_wait_s", np.nan))
        if np.isfinite(dark_wait_s):
            if abs(dark_wait_s - round(dark_wait_s)) < 1e-9:
                wait_key = int(round(dark_wait_s))
                condition_label = f"{wait_key} s"
            else:
                wait_key = dark_wait_s
                condition_label = f"{dark_wait_s:g} s"
        else:
            wait_key = np.nan
            condition_label = str(
                result.get("dataset_label", result.get("file_stem", "dataset"))
            )

        observed_percent = np.asarray(
            [
                ref["threshold_summary"][float(z)]["observed_percent"]
                for z in threshold_z
            ],
            dtype=float,
        )
        poisson_percent = np.asarray(
            [
                ref["threshold_summary"][float(z)]["poisson_expected_percent"]
                for z in threshold_z
            ],
            dtype=float,
        )
        scrambled_percent = np.asarray(
            [
                ref["threshold_summary"][float(z)]["scrambled_percent"]
                for z in threshold_z
            ],
            dtype=float,
        )
        observed_count = np.asarray(
            [
                ref["threshold_summary"][float(z)]["observed_count"]
                for z in threshold_z
            ],
            dtype=int,
        )
        k_threshold = np.asarray(
            [
                ref["threshold_summary"][float(z)]["k_threshold"]
                for z in threshold_z
            ],
            dtype=int,
        )

        comparison_rows.append(
            {
                "wait_key": wait_key,
                "label": condition_label,
                "observed_percent": observed_percent,
                "poisson_percent": poisson_percent,
                "scrambled_percent": scrambled_percent,
                "observed_count": observed_count,
                "k_threshold": k_threshold,
                "num_runs": int(ref["num_shots"]),
            }
        )

    def _sort_key(row):
        wk = row["wait_key"]
        if np.isfinite(wk):
            return float(wk)
        return np.inf

    comparison_rows = sorted(comparison_rows, key=_sort_key)

    # ------------------------------------------------------------------
    # NEW V19G COMPARISON FIGURES
    # 1) Observed-only figure (no scrambled / no Poisson bars mixed in)
    # 2) Separate reference figure for Poisson expectation and scrambled
    # ------------------------------------------------------------------
    x = np.arange(len(threshold_z), dtype=float)

    # ===== Figure 2A: OBSERVED ONLY =====
    fig_obs, axes_obs = plt.subplots(1, 2, figsize=(15.0, 6.0), sharey=True)
    ax_obs_bar, ax_obs_line = axes_obs

    if len(comparison_rows) == 0:
        for ax in axes_obs:
            ax.text(
                0.5, 0.5,
                "No reference-Poisson rarity data available",
                transform=ax.transAxes, ha="center", va="center"
            )
        ax_obs_bar.set_title("Observed high-coincidence rarity")
        ax_obs_line.set_title("Observed-real trend vs threshold")
    else:
        n_datasets = len(comparison_rows)
        group_width = 0.72
        bar_width = group_width / max(1, n_datasets)
        offsets = (
            -group_width / 2.0
            + bar_width / 2.0
            + np.arange(n_datasets, dtype=float) * bar_width
        )

        any_positive_obs = False

        for ds_ind, row in enumerate(comparison_rows):
            observed_vals = np.asarray(row["observed_percent"], dtype=float)
            positive_obs = np.isfinite(observed_vals) & (observed_vals > 0)
            any_positive_obs |= bool(np.any(positive_obs))
            vals_plot = np.where(positive_obs, observed_vals, np.nan)

            offset = offsets[ds_ind]
            ax_obs_bar.bar(
                x + offset,
                vals_plot,
                width=bar_width,
                label=f"{row['label']}",
                align="center",
            )

            for x0, y0, n_obs, k_cut in zip(
                x + offset,
                row["observed_percent"],
                row["observed_count"],
                row["k_threshold"],
            ):
                if not np.isfinite(y0) or y0 <= 0:
                    continue
                ax_obs_bar.annotate(
                    f"{n_obs}/{row['num_runs']}\nK≥{k_cut}",
                    xy=(x0, y0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

            ax_obs_line.plot(
                x,
                vals_plot,
                marker="o",
                linewidth=2.0,
                label=f"Dark wait: {row['label']}",
            )
            for x0, y0, n_obs in zip(x, row["observed_percent"], row["observed_count"]):
                if not np.isfinite(y0) or y0 <= 0:
                    continue
                ax_obs_line.annotate(
                    f"{n_obs}",
                    xy=(x0, y0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )

        for ax in axes_obs:
            ax.set_xticks(x)
            ax.set_xticklabels([f"≥ {z:g}σ" for z in threshold_z])
            ax.set_xlabel("High-coincidence threshold")
            ax.grid(alpha=0.22, axis="y")
            if any_positive_obs:
                ax.set_yscale("log")

        ax_obs_bar.set_ylabel("Upper-tail runs (%)")
        ax_obs_bar.set_title("Observed high-coincidence rarity only")
        ax_obs_bar.legend(fontsize=9, title="Dark wait")

        ax_obs_line.set_title("Observed-real rarity across 3σ / 4σ / 5σ")
        ax_obs_line.legend(fontsize=9, title="Dark wait")

    fig_obs.tight_layout()
    figures["comparison_high_coincidence_observed_only"] = fig_obs
    # Keep old key pointed to the observed-only figure for compatibility.
    figures["comparison_high_coincidence_observed_real_side_by_side"] = fig_obs

    # ===== Figure 2B: REFERENCE CONTROLS (Poisson + scrambled) =====
    fig_ref, axes_ref = plt.subplots(1, 2, figsize=(15.6, 6.0), sharey=True)
    ax_ref_bar, ax_ref_line = axes_ref

    if len(comparison_rows) == 0:
        for ax in axes_ref:
            ax.text(
                0.5, 0.5,
                "No reference-Poisson rarity data available",
                transform=ax.transAxes, ha="center", va="center"
            )
        ax_ref_bar.set_title("Reference rarity channels")
        ax_ref_line.set_title("Reference-channel trends")
    else:
        ref_series_info = [
            ("poisson_percent", "Poisson expectation", 0.80),
            ("scrambled_percent", "scrambled control", 0.50),
        ]
        n_datasets = len(comparison_rows)
        n_series = len(ref_series_info)
        total_bars_per_group = max(1, n_datasets * n_series)
        group_width = 0.82
        bar_width = group_width / total_bars_per_group
        offsets = (
            -group_width / 2.0
            + bar_width / 2.0
            + np.arange(total_bars_per_group, dtype=float) * bar_width
        )

        any_positive_ref = False

        for ds_ind, row in enumerate(comparison_rows):
            for series_ind, (key, series_label, alpha) in enumerate(ref_series_info):
                vals = np.asarray(row[key], dtype=float)
                positive = np.isfinite(vals) & (vals > 0)
                any_positive_ref |= bool(np.any(positive))
                vals_plot = np.where(positive, vals, np.nan)

                offset = offsets[ds_ind * n_series + series_ind]
                ax_ref_bar.bar(
                    x + offset,
                    vals_plot,
                    width=bar_width,
                    label=f"{row['label']} — {series_label}",
                    alpha=alpha,
                    align="center",
                )

                ax_ref_line.plot(
                    x,
                    vals_plot,
                    marker="o",
                    linewidth=1.8,
                    alpha=alpha,
                    label=f"{row['label']} — {series_label}",
                )

        for ax in axes_ref:
            ax.set_xticks(x)
            ax.set_xticklabels([f"≥ {z:g}σ" for z in threshold_z])
            ax.set_xlabel("High-coincidence threshold")
            ax.grid(alpha=0.22, axis="y")
            if any_positive_ref:
                ax.set_yscale("log")

        ax_ref_bar.set_ylabel("Upper-tail runs (%)")
        ax_ref_bar.set_title("Reference rarity channels only")
        ax_ref_bar.legend(fontsize=8, ncol=2, title="Wait / reference type")

        ax_ref_line.set_title("Poisson + scrambled trends across 3σ / 4σ / 5σ")
        ax_ref_line.legend(fontsize=8, ncol=2, title="Wait / reference type")

    fig_ref.tight_layout()
    figures["comparison_high_coincidence_reference_controls"] = fig_ref

    # ------------------------------------------------------------------
    # Figure 3: quantitative summary metrics.
    # ------------------------------------------------------------------
    x = np.arange(len(summary_rows), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mean_loss = [row["mean_transition_frac_pct"] for row in summary_rows]
    se_loss = [row["se_transition_frac_pct"] for row in summary_rows]
    axes[0, 0].bar(x, mean_loss, yerr=se_loss)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0, 0].set_ylabel("Mean transition fraction (%)")
    axes[0, 0].set_title("Mean transition fraction")
    axes[0, 0].grid(alpha=0.2, axis="y")

    mean_lost = [row["mean_lost"] for row in summary_rows]
    se_lost = [row["se_lost"] for row in summary_rows]
    axes[0, 1].bar(x, mean_lost, yerr=se_lost)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Mean NV- -> NV0 transitions / run")
    axes[0, 1].set_title("Mean transition count")
    axes[0, 1].grid(alpha=0.2, axis="y")

    width = 0.35
    lam = np.asarray([row["poisson_lambda"] for row in summary_rows], dtype=float)
    lam_scr = np.asarray(
        [row["scrambled_lambda"] for row in summary_rows],
        dtype=float,
    )
    axes[1, 0].bar(x - width / 2, lam, width=width, label="real")
    axes[1, 0].bar(x + width / 2, lam_scr, width=width, label="scrambled")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Poisson mean λ")
    axes[1, 0].set_title("Real vs scrambled Poisson mean")
    axes[1, 0].grid(alpha=0.2, axis="y")
    axes[1, 0].legend()

    fano = np.asarray([row["fano_real"] for row in summary_rows], dtype=float)
    fano_scr = np.asarray([row["fano_scrambled"] for row in summary_rows], dtype=float)
    axes[1, 1].bar(x - width / 2, fano, width=width, label="real")
    axes[1, 1].bar(x + width / 2, fano_scr, width=width, label="scrambled")
    axes[1, 1].axhline(1.0, linestyle="--", linewidth=1.0, label="Poisson")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1, 1].set_ylabel("Variance / mean")
    axes[1, 1].set_title("Fano factor (overdispersion)")
    axes[1, 1].grid(alpha=0.2, axis="y")
    axes[1, 1].legend()

    fig.tight_layout()
    figures["comparison_summary_metrics"] = fig

    # ------------------------------------------------------------------
    # Figure 4: characteristic beta-binomial distribution fits.
    # Each condition is shown separately but with identical axes/model form.
    # ------------------------------------------------------------------
    successful_fit_inds = [
        i
        for i, fit in enumerate(characteristic_fits)
        if fit is not None and fit.get("success", False)
    ]

    if successful_fit_inds:
        # Common K range across conditions.
        k_max = max(
            int(np.max(characteristic_fits[i]["k"]))
            for i in successful_fit_inds
        )
        xk = np.arange(k_max + 1, dtype=int)

        fig, axes = plt.subplots(
            2,
            len(successful_fit_inds),
            figsize=(7.0 * len(successful_fit_inds), 10.5),
            squeeze=False,
            sharex="col",
        )

        for col, i in enumerate(successful_fit_inds):
            fit = characteristic_fits[i]
            result = results[i]
            label = labels[i]

            good = np.asarray(result["good_run_mask"], dtype=bool)
            lost_good = np.rint(
                np.asarray(result["lost"], dtype=float)[good]
            ).astype(int)

            obs = np.bincount(
                lost_good,
                minlength=k_max + 1,
            ).astype(float)
            obs_prob = obs / max(1, lost_good.size)

            # Beta-binomial mixture over actual N_r values.
            kb, beta_expected_counts = _beta_binomial_expected_hist(
                fit,
                k_max=k_max,
            )
            beta_prob = beta_expected_counts / max(1, fit["num_runs"])

            # Exposure-corrected Poisson mixture already computed by the
            # per-dataset analysis. Extend to common x range.
            pois_prob = np.zeros(k_max + 1, dtype=float)
            pk = np.asarray(result["poisson"]["hist_k"], dtype=int)
            pe = np.asarray(result["poisson"]["hist_expected"], dtype=float)
            valid_pk = pk <= k_max
            pois_prob[pk[valid_pk]] = (
                pe[valid_pk] / max(1, result["poisson"]["num_valid_runs"])
            )

            title = (
                f"{label}\\n"
                f"p={100*fit['p']:.3f}%, "
                f"rho={fit['rho']:.5f}, "
                f"ΔAIC={fit['delta_aic_binom_minus_beta']:.1f}"
            )

            for row_ind, log_scale in enumerate((False, True)):
                ax = axes[row_ind, col]

                ax.step(
                    xk,
                    obs_prob,
                    where="mid",
                    linewidth=1.5,
                    label="Observed",
                )
                ax.plot(
                    xk,
                    pois_prob,
                    linestyle="--",
                    linewidth=1.3,
                    label="Poisson",
                )
                ax.plot(
                    kb,
                    beta_prob,
                    linestyle="-.",
                    linewidth=1.5,
                    label="Beta-binomial",
                )

                if log_scale:
                    ax.set_yscale("log")
                    ax.set_title(title + "\\nlog y")
                else:
                    ax.set_title(title + "\\nlinear y")

                ax.set_xlabel("NV- -> NV0 transitions per run")
                ax.set_ylabel("Probability mass")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)

        fig.suptitle(
            "Characteristic distribution fits: Poisson vs beta-binomial",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        figures["comparison_characteristic_distribution_fits"] = fig

        # ------------------------------------------------------------------
        # Figure 5: characteristic fitted parameters.
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))
        x = np.arange(len(summary_rows), dtype=float)

        p_vals = 100.0 * np.asarray(
            [row["beta_binom_p"] for row in summary_rows],
            dtype=float,
        )
        p_err = 100.0 * np.asarray(
            [row["beta_binom_p_se"] for row in summary_rows],
            dtype=float,
        )
        axes[0].bar(x, p_vals, yerr=p_err)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=20, ha="right")
        axes[0].set_ylabel("Per-NV loss probability (%)")
        axes[0].set_title("Characteristic mean loss probability")
        axes[0].grid(alpha=0.2, axis="y")

        rho_vals = np.asarray(
            [row["beta_binom_rho"] for row in summary_rows],
            dtype=float,
        )
        rho_err = np.asarray(
            [row["beta_binom_rho_se"] for row in summary_rows],
            dtype=float,
        )
        axes[1].bar(x, rho_vals, yerr=rho_err)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=20, ha="right")
        axes[1].set_ylabel("Beta-binomial rho")
        axes[1].set_title("Correlated / run-to-run overdispersion")
        axes[1].grid(alpha=0.2, axis="y")

        infl = np.asarray(
            [row["beta_binom_variance_inflation"] for row in summary_rows],
            dtype=float,
        )
        axes[2].bar(x, infl)
        axes[2].axhline(
            1.0,
            linestyle="--",
            linewidth=1.0,
            label="Independent-binomial limit",
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=20, ha="right")
        axes[2].set_ylabel("Variance inflation factor")
        axes[2].set_title("Extra variance beyond independent switching")
        axes[2].grid(alpha=0.2, axis="y")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        figures["comparison_characteristic_parameters"] = fig

    # ------------------------------------------------------------------
    # Figure 4: outlier-rate comparison at 3/4/5 sigma.
    #
    # In counts-only mode CALCULATE_SPATIAL=False, so spatial percentages are
    # NaN. Do NOT create/log-scale an empty spatial panel.
    # ------------------------------------------------------------------
    zvals = np.asarray(SIGMA_THRESHOLDS, dtype=float)

    spatial_available = any(
        result.get("spatial_outliers") is not None
        for result in results
    )

    metric_specs = [
        ("robust", "Robust-loss outlier rate"),
        ("poisson", "Poisson-local outlier rate"),
    ]
    if spatial_available:
        metric_specs.append(("spatial", "Spatial outlier rate"))

    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=(5.6 * len(metric_specs), 5.5),
        squeeze=False,
    )
    axes = axes.ravel()

    for row in summary_rows:
        for ax, (metric, title) in zip(axes, metric_specs):
            y = np.asarray(
                [
                    row[f"{metric}_ge_{int(z)}sigma_percent"]
                    for z in zvals
                ],
                dtype=float,
            )

            finite_positive = np.isfinite(y) & (y > 0)

            # Plot finite values. Zero values are not drawable on a log y-axis,
            # so leave them absent rather than creating an invalid axis.
            y_plot = np.where(finite_positive, y, np.nan)

            if np.any(finite_positive):
                ax.plot(
                    zvals,
                    y_plot,
                    marker="o",
                    linewidth=1.5,
                    label=row["label"],
                )

                # Annotate literal zero-rate thresholds so their meaning is not
                # lost when using log scale.
                zero_mask = np.isfinite(y) & (y == 0)
                for x0 in zvals[zero_mask]:
                    ax.annotate(
                        "0 observed",
                        xy=(x0, np.nanmin(y[finite_positive])),
                        xytext=(0, -14),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        fontsize=7,
                    )

            ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("Threshold (sigma)")
        ax.set_ylabel("Runs beyond threshold (%)")
        ax.set_xticks(zvals)
        ax.grid(alpha=0.2)

        # Only use log scale if this panel contains at least one positive value.
        panel_has_positive = False
        for line in ax.get_lines():
            yy = np.asarray(line.get_ydata(), dtype=float)
            if np.any(np.isfinite(yy) & (yy > 0)):
                panel_has_positive = True
                break

        if panel_has_positive:
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                "No positive values to plot",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    fig.tight_layout()
    figures["comparison_outlier_rates"] = fig

    comparison = {
        "summary_rows": summary_rows,
        "labels": labels,
        "characteristic_fits": characteristic_fits,
        "effective_dark_rate": effective_dark_rate,
    }
    return comparison, figures


if __name__ == "__main__":
    kpl.init_kplotlib()

    individual_analyses = []
    individual_figures = {}

    # 1) Analyze each measurement separately.
    for dataset_ind, dataset in enumerate(DATASETS, start=1):
        print("\n" + "#" * 140)
        print(
            f"INDIVIDUAL DATASET {dataset_ind}/{len(DATASETS)}: "
            f"{dataset['label']}"
        )
        print("#" * 140)

        analysis_i, figures_i = analyze_big_particle_memory_file(
            file_stem=dataset["file_stem"],
            npz_path_override=dataset.get("npz_path_override"),
            dataset_label=dataset["label"],
            max_runs=dataset.get("max_runs"),
        )
        individual_analyses.append(analysis_i)
        individual_figures[dataset["label"]] = figures_i

    # 2) Build direct comparison plots between the two acquisition times.
    comparison = None
    comparison_figures = {}
    if MAKE_COMPARISON_PLOTS and len(individual_analyses) >= 2:
        comparison, comparison_figures = _comparison_summary_and_figures(
            individual_analyses
        )

    # 3) Optional appended analysis remains available in the file, but it is
    # not recommended for 0-s versus 60-s because these are different wait conditions.
    appended_analysis = None
    appended_figures = {}
    if ALSO_RUN_APPENDED_ANALYSIS and len(DATASETS) >= 2:
        print("\n" + "#" * 140)
        print("OPTIONAL APPENDED ANALYSIS (DISABLED BY DEFAULT FOR 0-s vs 60-s)")
        print("#" * 140)
        appended_analysis, appended_figures = analyze_appended_particle_memory_files(
            DATASETS
        )

    # Convenient interactive handles.
    analyses = individual_analyses
    analysis = individual_analyses[-1] if individual_analyses else None
    figures = comparison_figures if comparison_figures else individual_figures

    kpl.show(block=True)
