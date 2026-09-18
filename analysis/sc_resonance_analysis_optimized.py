# -*- coding: utf-8 -*-
"""
Clean wide-field ESR analysis.

Main features
-------------
- Preserves the historical file_ids block from the original script.
- Loads/combines multiple resonance files efficiently.
- Converts count arrays to float32 before reductions to avoid float16 overflow.
- Fits every NV to the same two-Voigt model using conservative multi-start fitting.
- Saves ALL requested plots as PDF.
- Produces:
    1. all-NV spectra + fits in one overlay plot
    2. resonance heatmap
    3. one combined fit-parameter scatter dashboard
    4. optional individual parameter-scatter PDFs
    5. frequency/splitting histograms
    6. SNR overview
    7. optional all-NV fit pages in ONE multipage PDF
- Fits all NVs first, then filters by user-specified resonance centers/cutoff.
- Saves all and filtered fitted parameters to CSV.
- Saves regular/scatter figures as high-resolution PNG.
- Saves only list-style multipage outputs as high-quality PDF.

Author: Saroj Chand
Cleaned/extended with ChatGPT
"""

from __future__ import annotations

import csv
import hashlib
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares

from majorroutines.pulsed_resonance import norm_voigt
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield


# ============================================================================
# USER SETTINGS
# ============================================================================

N_JOBS = -1
FIT_VERBOSE = 5
FIT_CACHE = True

# Bump this whenever the fitter changes so stale cached fits are not reused.
FIT_CACHE_VERSION = "multistart_free_voigt_v1"

# Conservative multi-start settings. Frequencies below are specified in MHz
# and automatically converted when the dataset frequency axis is in GHz.
FIT_WIDTH_STARTS_MHZ = (2.0, 5.0, 10.0)
FIT_MIN_WIDTH_MHZ = 0.2
FIT_MAX_WIDTH_MHZ = 30.0
FIT_SMOOTH_POINTS = 3

# Dense-curve guard: a candidate that shoots far above/below the measured
# population receives a large penalty instead of winning on chi-square alone.
FIT_SPIKE_ALLOWANCE = 1.0

# Plot switches
SAVE_PNG = True
SAVE_LIST_PLOTS_AS_PDF = True
SHOW_PLOTS = True

# Output quality
PNG_DPI = 300
PDF_DPI = 300

PLOT_ALL_NV_OVERLAY = True
PLOT_HEATMAP = True
PLOT_PARAMETER_DASHBOARD = True
PLOT_INDIVIDUAL_PARAMETER_PDFS = True
PLOT_HISTOGRAMS = True
PLOT_SNR_OVERVIEW = True

# Optional detailed per-NV fits.
# This puts the selected individual fits into ONE multipage PDF.
PLOT_INDIVIDUAL_FITS_PDF = True
FITS_PER_PAGE = 12
FIT_GRID_COLS = 3

# ---------------------------------------------------------------------------
# RESONANCE-PEAK FILTER
# ---------------------------------------------------------------------------
# Peaks identified from the SNR overview (GHz).
RESONANCE_TARGETS_GHZ = (
    2.7539,
    2.7773,
    2.8212,
    2.8421,
    2.9195,
    2.9370,
    2.9758,
    2.9906,
)

# Main knob to edit:
# An individual fitted center is considered a match when its nearest target
# is within this distance.
RESONANCE_FILTER_CUTOFF_MHZ = 8.0

# "both"   -> BOTH fitted centers f1 and f2 must match one of the target peaks.
# "either" -> at least ONE fitted center must match.
RESONANCE_FILTER_MODE = "both"

# Usually desirable for a genuine two-peak ESR fit: do not allow f1 and f2
# to map to the exact same target resonance.
REQUIRE_DISTINCT_TARGETS = True

# Downstream PDFs are made from the filtered NVs.
# The comparison overlay/diagnostic plots still retain the full population
# for context.
USE_RESONANCE_FILTER_FOR_PLOTS = True

# Optional legacy splitting classification retained for comparison.
SPLIT_TARGETS_GHZ = (0.135, 0.212)
SPLIT_TOL_GHZ = 0.015

# Robust plotting
SCATTER_ALPHA = 0.65
SCATTER_SIZE = 12
HIST_BINS = 35


# NV selection for plots/PDFs:
#   None -> include ALL NVs
#   list -> include only those original NV indices
MANUAL_FILTERED_INDICES = [
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92,
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
    106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
    142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
    155, 156, 157, 158, 159, 161, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 194, 195, 197, 198, 199, 200, 201, 202, 203, 204, 205,
    206, 207, 208, 209, 210, 211,
]

FILTERED_INDICES = None  # None = all NVs
# FILTERED_INDICES = MANUAL_FILTERED_INDICES  # use manual subset instead


def resolve_filtered_indices(num_nvs):
    """Return valid original NV indices selected for plotting/output."""
    if FILTERED_INDICES is None:
        return np.arange(num_nvs, dtype=int)

    inds = np.asarray(FILTERED_INDICES, dtype=int)
    inds = inds[(inds >= 0) & (inds < num_nvs)]
    return np.unique(inds)
# ============================================================================
# FIT MODEL
# ============================================================================

def two_voigt(freq, amp1, amp2, center1, center2, width, bg_offset):
    """Two equal-width normalized Voigt peaks plus a constant background."""
    freq = np.asarray(freq, dtype=float)
    return (
        amp1 * norm_voigt(freq, width, width, center1)
        + amp2 * norm_voigt(freq, width, width, center2)
        + bg_offset
    )


def _safe_sigma(ste):
    """Replace zero/non-finite STE values so weighted residuals stay finite."""
    ste = np.asarray(ste, dtype=float)
    finite_positive = ste[np.isfinite(ste) & (ste > 0)]
    fallback = np.nanmedian(finite_positive) if finite_positive.size else 1.0
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0
    return np.where(np.isfinite(ste) & (ste > 0), ste, fallback)


def residuals_fn(params, freq, y, yste):
    return (np.asarray(y, float) - two_voigt(freq, *params)) / _safe_sigma(yste)


def _frequency_units_and_width_guess(freqs):
    """
    Infer whether the frequency axis is GHz-like or MHz-like.
    Return a sensible initial 5-MHz width in native units.
    """
    med = float(np.nanmedian(np.abs(freqs)))
    if med < 100.0:       # ~2.87 -> GHz
        return "GHz", 0.005
    return "MHz", 5.0     # ~2870 -> MHz


def fit_one_nv(nv_idx, freqs, avg_counts, avg_counts_ste, freqs_dense):
    """
    Conservative multi-start two-Voigt fit.

    This intentionally stays close to the original fitting strategy that works
    reasonably well for these wide-field ESR data:

      * one initial resonance from each half of the scan
      * the same free two-Voigt model used previously
      * ordinary weighted least squares (no robust-loss reshaping)
      * several linewidth starts (2, 5, 10 MHz by default)
      * raw and lightly smoothed peak-position starts
      * only mild linewidth bounds (0.2--30 MHz by default)
      * choose the best solution by weighted chi-square, with a penalty for
        unphysical narrow/tall dense-curve spikes between sampled points

    The fit centers are NOT forced to the known orientation frequencies.
    Orientation classification remains a downstream analysis step.
    """

    freq_arr = np.asarray(freqs, dtype=float)
    y_full = np.asarray(avg_counts[nv_idx], dtype=float)
    sigma_full = _safe_sigma(avg_counts_ste[nv_idx])

    finite = (
        np.isfinite(freq_arr)
        & np.isfinite(y_full)
        & np.isfinite(sigma_full)
    )

    if np.count_nonzero(finite) < 8:
        return {
            "nv_idx": nv_idx,
            "success": False,
            "message": "too few finite points",
            "popt": np.full(6, np.nan),
            "fit_dense": np.full_like(freqs_dense, np.nan, dtype=float),
            "fit_at_data": np.full_like(freq_arr, np.nan, dtype=float),
            "chi2": np.nan,
            "red_chi2": np.nan,
            "rms_resid": np.nan,
            "nfev": 0,
        }

    # Fit only finite samples, but evaluate/store the model on the full axis.
    f = freq_arr[finite]
    y = y_full[finite]
    sigma = sigma_full[finite]

    fmin = float(np.nanmin(f))
    fmax = float(np.nanmax(f))
    fmid = 0.5 * (fmin + fmax)

    finite_y = y[np.isfinite(y)]
    ymin = float(np.nanmin(finite_y))
    ymax = float(np.nanmax(finite_y))
    yspan = max(ymax - ymin, 1e-6)

    # Infer GHz-like versus MHz-like axis and convert user settings.
    units, _ = _frequency_units_and_width_guess(f)
    if units == "GHz":
        width_starts = [v / 1000.0 for v in FIT_WIDTH_STARTS_MHZ]
        min_width = FIT_MIN_WIDTH_MHZ / 1000.0
        max_width = FIT_MAX_WIDTH_MHZ / 1000.0
    else:
        width_starts = list(FIT_WIDTH_STARTS_MHZ)
        min_width = float(FIT_MIN_WIDTH_MHZ)
        max_width = float(FIT_MAX_WIDTH_MHZ)

    scan_span = fmax - fmin
    max_width = min(max_width, max(scan_span / 2.0, min_width * 1.01))

    # ---------------------------------------------------------------------
    # Center starts: original raw maxima + lightly smoothed maxima.
    # ---------------------------------------------------------------------
    mid_ind = max(1, len(f) // 2)

    try:
        raw_low_ind = int(np.nanargmax(y[:mid_ind]))
        raw_high_ind = int(np.nanargmax(y[mid_ind:])) + mid_ind
    except ValueError:
        raw_low_ind = 0
        raw_high_ind = len(f) - 1

    center_starts = [
        (float(f[raw_low_ind]), float(f[raw_high_ind]))
    ]

    # Light smoothing is ONLY for initialization; fitting still uses raw data.
    smooth_points = int(max(1, FIT_SMOOTH_POINTS))
    if smooth_points > 1 and len(y) >= smooth_points:
        kernel = np.ones(smooth_points, dtype=float) / smooth_points
        pad_left = smooth_points // 2
        pad_right = smooth_points - 1 - pad_left
        y_pad = np.pad(y, (pad_left, pad_right), mode="edge")
        y_smooth = np.convolve(y_pad, kernel, mode="valid")

        try:
            sm_low_ind = int(np.nanargmax(y_smooth[:mid_ind]))
            sm_high_ind = int(np.nanargmax(y_smooth[mid_ind:])) + mid_ind
            center_starts.append(
                (float(f[sm_low_ind]), float(f[sm_high_ind]))
            )
        except ValueError:
            pass

    # Remove duplicate start pairs while preserving order.
    unique_center_starts = []
    for pair in center_starts:
        if pair not in unique_center_starts:
            unique_center_starts.append(pair)

    # A low percentile is less sensitive to one downward outlier than min(y).
    baseline_guess = float(np.nanpercentile(y, 20.0))

    # Keep one resonance in each half of the scan. This matches the acquisition
    # geometry and prevents both free peaks from collapsing onto the same noise
    # feature, while still allowing each center to move freely within its half.
    margin = max(scan_span * 1e-10, 1e-12)
    lower = np.array(
        [
            0.0,
            0.0,
            fmin,
            fmid + margin,
            min_width,
            ymin - yspan,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            np.inf,
            np.inf,
            fmid - margin,
            fmax,
            max_width,
            ymax + yspan,
        ],
        dtype=float,
    )

    # In the unlikely case that the finite-point axis is too short/odd for the
    # strict half split, fall back to full-scan center bounds.
    if lower[2] >= upper[2] or lower[3] >= upper[3]:
        lower[2] = fmin
        lower[3] = fmin
        upper[2] = fmax
        upper[3] = fmax

    best = None
    best_rank_score = np.inf
    best_chi2 = np.inf
    best_clean = False

    # ---------------------------------------------------------------------
    # Multi-start over center and linewidth guesses.
    # ---------------------------------------------------------------------
    for low_guess, high_guess in unique_center_starts:
        for width_guess in width_starts:
            width_guess = float(np.clip(width_guess, min_width, max_width))

            # norm_voigt is area-normalized. Convert a desired measured peak
            # height to an area-like coefficient for a sensible p0.
            desired_height = max(0.35 * yspan, 1e-5)

            try:
                n1 = float(
                    norm_voigt(
                        np.array([low_guess]),
                        width_guess,
                        width_guess,
                        low_guess,
                    )[0]
                )
                n2 = float(
                    norm_voigt(
                        np.array([high_guess]),
                        width_guess,
                        width_guess,
                        high_guess,
                    )[0]
                )
            except Exception:
                n1 = np.nan
                n2 = np.nan

            amp1_guess = (
                desired_height / n1
                if np.isfinite(n1) and n1 > 0
                else desired_height * max(width_guess, 1e-12)
            )
            amp2_guess = (
                desired_height / n2
                if np.isfinite(n2) and n2 > 0
                else desired_height * max(width_guess, 1e-12)
            )

            p0 = np.array(
                [
                    amp1_guess,
                    amp2_guess,
                    low_guess,
                    high_guess,
                    width_guess,
                    baseline_guess,
                ],
                dtype=float,
            )

            # Ensure the starting point is strictly within finite bounds.
            p0 = np.maximum(p0, lower + 1e-12)
            finite_upper = np.isfinite(upper)
            p0[finite_upper] = np.minimum(
                p0[finite_upper], upper[finite_upper] - 1e-12
            )

            try:
                result = least_squares(
                    residuals_fn,
                    p0,
                    args=(f, y, sigma),
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=20000,
                )
            except Exception:
                continue

            if not np.all(np.isfinite(result.x)):
                continue

            popt_test = np.asarray(result.x, dtype=float)
            fit_test = two_voigt(f, *popt_test)
            dense_test = two_voigt(freqs_dense, *popt_test)

            if not (
                np.all(np.isfinite(fit_test))
                and np.all(np.isfinite(dense_test))
            ):
                continue

            weighted = (y - fit_test) / sigma
            chi2 = float(np.nansum(weighted**2))

            # -------------------------------------------------------------
            # Dense-curve spike penalty.
            # -------------------------------------------------------------
            # A very narrow normalized Voigt can hide a huge peak between the
            # sampled frequency points. Rather than imposing a large minimum
            # linewidth, penalize only the actual unmeasured overshoot.
            dense_min = float(np.nanmin(dense_test))
            dense_max = float(np.nanmax(dense_test))

            allowed_low = ymin - FIT_SPIKE_ALLOWANCE * yspan
            allowed_high = ymax + FIT_SPIKE_ALLOWANCE * yspan

            excess_high = max(0.0, dense_max - allowed_high) / yspan
            excess_low = max(0.0, allowed_low - dense_min) / yspan
            spike_penalty = 1.0e4 * (excess_high**2 + excess_low**2)

            rank_score = chi2 + spike_penalty
            clean = spike_penalty == 0.0

            if rank_score < best_rank_score:
                best_rank_score = rank_score
                best_chi2 = chi2
                best = result
                best_clean = clean

    if best is None:
        return {
            "nv_idx": nv_idx,
            "success": False,
            "message": "all multistart candidates failed",
            "popt": np.full(6, np.nan),
            "fit_dense": np.full_like(freqs_dense, np.nan, dtype=float),
            "fit_at_data": np.full_like(freq_arr, np.nan, dtype=float),
            "chi2": np.nan,
            "red_chi2": np.nan,
            "rms_resid": np.nan,
            "nfev": 0,
        }

    popt = np.asarray(best.x, dtype=float)

    # Always store f1 <= f2 for cleaner downstream plots/CSV output.
    if popt[2] > popt[3]:
        popt[[0, 1]] = popt[[1, 0]]
        popt[[2, 3]] = popt[[3, 2]]

    fit_at_data = two_voigt(freq_arr, *popt)
    fit_dense = two_voigt(freqs_dense, *popt)

    residual = y_full - fit_at_data
    weighted_full = residual[finite] / sigma_full[finite]
    chi2 = float(np.nansum(weighted_full**2))
    dof = max(1, np.count_nonzero(finite) - len(popt))
    red_chi2 = chi2 / dof
    rms_resid = float(np.sqrt(np.nanmean(residual[finite] ** 2)))

    width = float(popt[4])
    width_mhz = width * 1000.0 if units == "GHz" else width

    message = (
        f"{best.message}; multistart; width={width_mhz:.3f} MHz; "
        f"dense_guard={'clean' if best_clean else 'penalized'}"
    )

    return {
        "nv_idx": nv_idx,
        "success": bool(best.success),
        "message": message,
        "popt": popt,
        "fit_dense": fit_dense,
        "fit_at_data": fit_at_data,
        "chi2": chi2,
        "red_chi2": red_chi2,
        "rms_resid": rms_resid,
        "nfev": int(best.nfev),
    }


# ============================================================================
# CACHE
# ============================================================================

def _cache_key(file_id, num_nvs, freqs):
    payload = (
        FIT_CACHE_VERSION
        + "|"
        + str(file_id)
        + f"|N={num_nvs}|"
        + f"|width_starts_mhz={FIT_WIDTH_STARTS_MHZ}"
        + f"|width_bounds_mhz={FIT_MIN_WIDTH_MHZ},{FIT_MAX_WIDTH_MHZ}"
        + f"|spike_allowance={FIT_SPIKE_ALLOWANCE}"
        + np.array2string(np.asarray(freqs, dtype=float), precision=10)
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(file_id, num_nvs, freqs):
    cache_dir = Path(__file__).resolve().parent / ".resonance_fit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"resonance_fit_{_cache_key(file_id, num_nvs, freqs)}.npz"


def save_fit_cache(path, fit_results):
    params = np.vstack([r["popt"] for r in fit_results])
    fit_dense = np.vstack([r["fit_dense"] for r in fit_results])
    fit_at_data = np.vstack([r["fit_at_data"] for r in fit_results])
    success = np.asarray([r["success"] for r in fit_results], dtype=bool)
    chi2 = np.asarray([r["chi2"] for r in fit_results], dtype=float)
    red_chi2 = np.asarray([r["red_chi2"] for r in fit_results], dtype=float)
    rms_resid = np.asarray([r["rms_resid"] for r in fit_results], dtype=float)
    nfev = np.asarray([r["nfev"] for r in fit_results], dtype=int)

    np.savez_compressed(
        path,
        params=params,
        fit_dense=fit_dense,
        fit_at_data=fit_at_data,
        success=success,
        chi2=chi2,
        red_chi2=red_chi2,
        rms_resid=rms_resid,
        nfev=nfev,
    )


def load_fit_cache(path):
    data = np.load(path, allow_pickle=False)
    n = len(data["success"])
    out = []
    for i in range(n):
        out.append(
            {
                "nv_idx": i,
                "success": bool(data["success"][i]),
                "message": "loaded from cache",
                "popt": data["params"][i],
                "fit_dense": data["fit_dense"][i],
                "fit_at_data": data["fit_at_data"][i],
                "chi2": float(data["chi2"][i]),
                "red_chi2": float(data["red_chi2"][i]),
                "rms_resid": float(data["rms_resid"][i]),
                "nfev": int(data["nfev"][i]),
            }
        )
    return out


# ============================================================================
# DATA LOADING
# ============================================================================

def split_blocks(counts, num_steps):
    """
    Original acquisition layout:
        [sig0 | sig1 | ref0 | ref1]
    Combine sig0/sig1 along repetition axis and interleave ref0/ref1.
    """
    counts = np.asarray(counts)
    adj_num_steps = num_steps // 4

    sig0 = counts[:, :, 0:adj_num_steps, :]
    sig1 = counts[:, :, adj_num_steps : 2 * adj_num_steps, :]
    ref0 = counts[:, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
    ref1 = counts[:, :, 3 * adj_num_steps :, :]

    sig = np.concatenate((sig0, sig1), axis=3)

    ref = np.empty(
        (
            counts.shape[0],
            counts.shape[1],
            adj_num_steps,
            2 * counts.shape[3],
        ),
        dtype=counts.dtype,
    )
    ref[:, :, :, 0::2] = ref0
    ref[:, :, :, 1::2] = ref1

    return sig, ref


def load_and_combine(file_ids):
    sig_chunks = []
    ref_chunks = []
    combined_data = None
    nv_list = None
    freqs = None
    expected_num_steps = None

    for ind, file_id in enumerate(file_ids):
        print(f"\nLoading {ind + 1}/{len(file_ids)}: {file_id}")

        kwargs = dict(load_npz=True, use_cache=True)
        if isinstance(file_id, str):
            data = dm.get_raw_data(file_stem=file_id, **kwargs)
        else:
            data = dm.get_raw_data(file_id=file_id, **kwargs)

        if not data:
            print(f"  WARNING: no data found for {file_id}; skipping")
            continue

        this_nv_list = data["nv_list"]
        this_freqs = np.asarray(data["freqs"], dtype=float)
        this_num_steps = int(data["num_steps"])

        if combined_data is None:
            combined_data = data
            nv_list = this_nv_list
            freqs = this_freqs
            expected_num_steps = this_num_steps
        else:
            if len(this_nv_list) != len(nv_list):
                raise ValueError(
                    f"NV-count mismatch for {file_id}: "
                    f"{len(this_nv_list)} vs {len(nv_list)}"
                )
            if this_num_steps != expected_num_steps:
                raise ValueError(
                    f"num_steps mismatch for {file_id}: "
                    f"{this_num_steps} vs {expected_num_steps}"
                )
            if not np.allclose(this_freqs, freqs, rtol=0, atol=1e-12):
                raise ValueError(f"Frequency-axis mismatch for {file_id}")

        counts = np.asarray(data["counts"])[0]
        sig, ref = split_blocks(counts, this_num_steps)

        print(f"  sig={sig.shape} {sig.dtype}, ref={ref.shape} {ref.dtype}")
        sig_chunks.append(sig)
        ref_chunks.append(ref)

    if not sig_chunks:
        raise RuntimeError("No valid datasets were loaded.")

    sig_counts = np.concatenate(sig_chunks, axis=1)
    ref_counts = np.concatenate(ref_chunks, axis=1)

    # Important: reductions on float16 can overflow.
    sig_counts = np.asarray(sig_counts, dtype=np.float32)
    ref_counts = np.asarray(ref_counts, dtype=np.float32)

    print(
        f"\nCombined: {len(sig_chunks)} file(s), "
        f"sig={sig_counts.shape} {sig_counts.dtype}, "
        f"ref={ref_counts.shape} {ref_counts.dtype}"
    )

    return combined_data, nv_list, freqs, sig_counts, ref_counts


# ============================================================================
# CLASSIFICATION / PARAMETER EXTRACTION
# ============================================================================

def _to_ghz(values):
    """Return a float array in GHz, accepting either GHz-like or MHz-like input."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmedian(np.abs(finite)) > 100.0:
        return arr / 1000.0
    return arr


def _targets_in_native_units(values, targets_ghz=RESONANCE_TARGETS_GHZ):
    """
    Convert target GHz values into the same units as `values`
    so target markers work for GHz- or MHz-style datasets.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmedian(np.abs(finite)) > 100.0:
        return np.asarray(targets_ghz, dtype=float) * 1000.0
    return np.asarray(targets_ghz, dtype=float)


def build_resonance_filter(
    f1,
    f2,
    success,
    targets_ghz=RESONANCE_TARGETS_GHZ,
    cutoff_mhz=RESONANCE_FILTER_CUTOFF_MHZ,
    mode=RESONANCE_FILTER_MODE,
    require_distinct=REQUIRE_DISTINCT_TARGETS,
):
    """
    Match each fitted center to its nearest user-specified resonance.

    Returns
    -------
    dict containing:
        selected
        target1_ghz, target2_ghz
        error1_mhz, error2_mhz
        matched1, matched2
        target_index1, target_index2

    Notes
    -----
    - Matching is done in GHz internally.
    - 'both' is the recommended mode for this two-Voigt model.
    """
    f1g = _to_ghz(f1)
    f2g = _to_ghz(f2)
    targets = np.asarray(targets_ghz, dtype=float)
    cutoff_ghz = float(cutoff_mhz) / 1000.0

    d1 = np.abs(f1g[:, None] - targets[None, :])
    d2 = np.abs(f2g[:, None] - targets[None, :])

    # Make all-NaN / failed fits safe.
    d1[~np.isfinite(d1)] = np.inf
    d2[~np.isfinite(d2)] = np.inf

    idx1 = np.argmin(d1, axis=1)
    idx2 = np.argmin(d2, axis=1)

    err1_ghz = d1[np.arange(len(f1g)), idx1]
    err2_ghz = d2[np.arange(len(f2g)), idx2]

    matched1 = err1_ghz <= cutoff_ghz
    matched2 = err2_ghz <= cutoff_ghz

    if mode.lower() == "both":
        selected = matched1 & matched2
    elif mode.lower() == "either":
        selected = matched1 | matched2
    else:
        raise ValueError("RESONANCE_FILTER_MODE must be 'both' or 'either'.")

    if require_distinct:
        selected &= idx1 != idx2

    selected &= np.asarray(success, dtype=bool)

    target1 = targets[idx1]
    target2 = targets[idx2]

    # Failed fits should not appear to have meaningful assignments.
    bad1 = ~np.isfinite(f1g)
    bad2 = ~np.isfinite(f2g)
    target1 = target1.astype(float)
    target2 = target2.astype(float)
    target1[bad1] = np.nan
    target2[bad2] = np.nan

    return {
        "selected": selected,
        "target1_ghz": target1,
        "target2_ghz": target2,
        "error1_mhz": err1_ghz * 1000.0,
        "error2_mhz": err2_ghz * 1000.0,
        "matched1": matched1,
        "matched2": matched2,
        "target_index1": idx1,
        "target_index2": idx2,
    }


def extract_parameters(fit_results, avg_snr):
    params = np.vstack([r["popt"] for r in fit_results])

    amp1 = params[:, 0]
    amp2 = params[:, 1]
    f1 = params[:, 2]
    f2 = params[:, 3]
    width = params[:, 4]
    bg = params[:, 5]

    splitting = np.abs(f2 - f1)
    contrast = 0.5 * (amp1 + amp2)
    center = 0.5 * (f1 + f2)

    success = np.asarray([r["success"] for r in fit_results], dtype=bool)
    red_chi2 = np.asarray([r["red_chi2"] for r in fit_results], dtype=float)
    rms_resid = np.asarray([r["rms_resid"] for r in fit_results], dtype=float)
    nfev = np.asarray([r["nfev"] for r in fit_results], dtype=int)

    snr_arr = np.asarray(avg_snr, dtype=float)
    if snr_arr.ndim == 1:
        snr_median = snr_arr.copy()
        snr_peak = np.abs(snr_arr)
    else:
        reduce_axes = tuple(range(1, snr_arr.ndim))
        snr_median = np.nanmedian(snr_arr, axis=reduce_axes)
        snr_peak = np.nanmax(np.abs(snr_arr), axis=reduce_axes)

    filt = build_resonance_filter(f1, f2, success)

    return {
        "nv_index": np.arange(len(f1), dtype=int),
        "amp1": amp1,
        "amp2": amp2,
        "f1": f1,
        "f2": f2,
        "center": center,
        "splitting": splitting,
        "width": width,
        "background": bg,
        "contrast": contrast,
        "success": success,
        "red_chi2": red_chi2,
        "rms_resid": rms_resid,
        "nfev": nfev,
        "snr_median": snr_median,
        "snr_peak": snr_peak,
        "filter_selected": filt["selected"],
        "nearest_target_f1_ghz": filt["target1_ghz"],
        "nearest_target_f2_ghz": filt["target2_ghz"],
        "f1_target_error_mhz": filt["error1_mhz"],
        "f2_target_error_mhz": filt["error2_mhz"],
        "f1_target_match": filt["matched1"],
        "f2_target_match": filt["matched2"],
    }


def subset_parameters(p, indices):
    """Subset array-like parameter entries while leaving scalar metadata alone."""
    indices = np.asarray(indices, dtype=int)
    n = len(p["nv_index"])
    out = {}

    for key, val in p.items():
        if isinstance(val, np.ndarray) and val.shape[:1] == (n,):
            out[key] = val[indices]
        else:
            out[key] = val

    return out


# ============================================================================
# OUTPUT DIRECTORY / SAVING
# ============================================================================

def make_output_dir(file_id):
    """
    Use Dioptric's data-manager path logic only to obtain the output folder.
    We then save actual .pdf/.csv files ourselves instead of allowing a
    default '.txt' suffix to be appended.
    """
    probe = Path(dm.get_file_path(__file__, "resonance_analysis_pdf", "probe"))
    out_dir = probe.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = str(file_id).replace("\\", "_").replace("/", "_").replace(":", "_")
    run_dir = out_dir / f"{timestamp}_{safe_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPDF/CSV output directory:\n  {run_dir}")
    return run_dir


def save_png(fig, path):
    if SAVE_PNG:
        fig.savefig(path, format="png", dpi=PNG_DPI, bbox_inches="tight")
        print(f"Saved: {path}")


def save_pdf_list(fig, path):
    """
    Use PDF only for list-style / collection-style outputs when requested.
    """
    if SAVE_LIST_PLOTS_AS_PDF:
        fig.savefig(path, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
        print(f"Saved: {path}")


# ============================================================================
# PLOTS
# ============================================================================

def plot_all_nv_overlay(
    freqs,
    avg_counts,
    freqs_dense,
    fit_results,
    p,
    out_dir,
):
    """
    One figure containing all NVs, with selected NVs highlighted.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    selected = np.asarray(p["filter_selected"], dtype=bool)

    # Full population as faint context.
    for nv_idx in range(len(avg_counts)):
        ax.plot(
            freqs,
            np.asarray(avg_counts[nv_idx], dtype=float),
            linewidth=0.35,
            alpha=0.08,
        )

    # Selected NVs highlighted.
    for nv_idx in np.flatnonzero(selected):
        ax.plot(
            freqs,
            np.asarray(avg_counts[nv_idx], dtype=float),
            linewidth=0.65,
            alpha=0.30,
        )

    # Selected-population median.
    if np.any(selected):
        # med = np.nanmedian(
        #     np.asarray(avg_counts, dtype=float)[selected],
        #     axis=0,
        # )
        # ax.plot(
        #     freqs,
        #     med,
        #     linewidth=2.5,
        #     label=f"Median selected (N={selected.sum()})",
        # )

        selected_fit_curves = np.asarray(
            [
                fit_results[i]["fit_dense"]
                for i in np.flatnonzero(selected)
                if fit_results[i]["success"]
            ],
            dtype=float,
        )
        # if selected_fit_curves.size:
        #     ax.plot(
        #         freqs_dense,
        #         np.nanmedian(selected_fit_curves, axis=0),
        #         linewidth=2.0,
        #         linestyle="--",
        #         label="Median selected fit",
        #     )

    # Mark the eight target resonances.
    target_native = _targets_in_native_units(freqs)
    for target in target_native:
        ax.axvline(target, linestyle=":", linewidth=0.9, alpha=0.7)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized NV population")
    ax.set_title(
        "All NV resonance traces with frequency-filtered NVs highlighted\n"
        f"cutoff={RESONANCE_FILTER_CUTOFF_MHZ:.1f} MHz, "
        f"mode={RESONANCE_FILTER_MODE}"
    )
    ax.grid(True, alpha=0.22)
    ax.legend()

    save_png(fig, out_dir / "01_all_nv_overlay_with_filtered_highlight.png")
    return fig


def plot_filtered_overlay(freqs, avg_counts, freqs_dense, fit_results, p, out_dir):
    selected_idx = np.flatnonzero(p["filter_selected"])

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    if len(selected_idx) == 0:
        ax.text(
            0.5,
            0.5,
            "No NVs passed the resonance filter",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        for nv_idx in selected_idx:
            ax.plot(
                freqs,
                np.asarray(avg_counts[nv_idx], dtype=float),
                linewidth=0.55,
                alpha=0.25,
            )

        # med = np.nanmedian(
        #     np.asarray(avg_counts, dtype=float)[selected_idx],
        #     axis=0,
        # )
        # ax.plot(freqs, med, linewidth=2.5, label="Median selected data")

        fit_curves = np.asarray(
            [fit_results[i]["fit_dense"] for i in selected_idx],
            dtype=float,
        )
        # ax.plot(
        #     freqs_dense,
        #     np.nanmedian(fit_curves, axis=0),
        #     linewidth=2.0,
        #     linestyle="--",
        #     label="Median selected fit",
        # )

    for target in _targets_in_native_units(freqs):
        ax.axvline(target, linestyle=":", linewidth=0.9, alpha=0.7)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized NV population")
    ax.set_title(
        f"Filtered NV resonance traces — N={len(selected_idx)}\n"
        f"target cutoff={RESONANCE_FILTER_CUTOFF_MHZ:.1f} MHz"
    )
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best")

    save_png(fig, out_dir / "02_filtered_nv_resonance_overlay.png")
    return fig


def plot_heatmap(freqs, avg_counts, p, out_dir):
    selected_idx = np.flatnonzero(p["filter_selected"])

    if USE_RESONANCE_FILTER_FOR_PLOTS:
        inds = selected_idx
        suffix = "filtered"
    else:
        inds = np.arange(len(avg_counts))
        suffix = "all"

    fig, ax = plt.subplots(figsize=(9.5, 8.0))

    if len(inds) == 0:
        ax.text(
            0.5,
            0.5,
            "No NVs passed the resonance filter",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        # Sort selected NVs by fitted f1.
        local_order = np.argsort(
            np.where(np.isfinite(p["f1"][inds]), p["f1"][inds], np.inf)
        )
        sorted_inds = inds[local_order]
        arr_sorted = np.asarray(avg_counts, dtype=float)[sorted_inds]

        im = ax.imshow(
            arr_sorted,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=[float(freqs[0]), float(freqs[-1]), 0, len(arr_sorted)],
        )
        fig.colorbar(im, ax=ax, label="Normalized NV population")

        for target in _targets_in_native_units(freqs):
            ax.axvline(target, linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Filtered NVs sorted by fitted f1")
    ax.set_title(f"ESR population heatmap ({suffix}, N={len(inds)})")

    save_png(fig, out_dir / "03_resonance_heatmap_filtered.png")
    return fig


def _scatter(ax, x, y, xlabel, ylabel, title=None):
    finite = np.isfinite(x) & np.isfinite(y)
    ax.scatter(
        np.asarray(x)[finite],
        np.asarray(y)[finite],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_parameter_dashboard(p, out_dir):
    """
    Parameter scatter dashboard for the FILTERED population.
    Uses original NV indices on index-based panels.
    """
    idx = np.asarray(p["nv_index"])

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    _scatter(axes[0], idx, p["f1"], "Original NV index", "f1", "Lower resonance")
    _scatter(axes[1], idx, p["f2"], "Original NV index", "f2", "Upper resonance")
    _scatter(axes[2], idx, p["splitting"], "Original NV index", "|f2-f1|", "Frequency splitting")

    _scatter(axes[3], idx, p["width"], "Original NV index", "Width", "Voigt width")
    _scatter(axes[4], idx, p["contrast"], "Original NV index", "Mean peak amplitude", "Contrast/amplitude")
    _scatter(axes[5], idx, p["red_chi2"], "Original NV index", "Reduced chi²", "Fit quality")

    _scatter(
        axes[6],
        p["splitting"],
        p["width"],
        "|f2-f1|",
        "Width",
        "Splitting vs width",
    )
    _scatter(
        axes[7],
        p["width"],
        p["contrast"],
        "Width",
        "Mean peak amplitude",
        "Width vs contrast",
    )
    _scatter(
        axes[8],
        p["snr_peak"],
        p["contrast"],
        "Peak |SNR|",
        "Mean peak amplitude",
        "SNR vs contrast",
    )

    fig.suptitle(
        f"Filtered ESR fit-parameter overview — N={len(idx)} "
        f"(cutoff {RESONANCE_FILTER_CUTOFF_MHZ:.1f} MHz)",
        y=0.995,
    )

    save_png(fig, out_dir / "04_filtered_parameter_scatter_dashboard.png")
    return fig


def plot_individual_parameter_pdfs(p, out_dir):
    idx = np.asarray(p["nv_index"])

    specs = [
        ("f1", "Filtered fitted lower resonance", "Frequency"),
        ("f2", "Filtered fitted upper resonance", "Frequency"),
        ("splitting", "Filtered resonance splitting", "|f2-f1|"),
        ("width", "Filtered Voigt width", "Width"),
        ("contrast", "Filtered mean fitted peak amplitude", "Amplitude"),
        ("red_chi2", "Filtered reduced chi-square", "Reduced chi²"),
        ("rms_resid", "Filtered fit RMS residual", "RMS residual"),
        ("snr_peak", "Filtered peak absolute SNR", "Peak |SNR|"),
    ]

    figs = []
    for num, (key, title, ylabel) in enumerate(specs, start=1):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        _scatter(ax, idx, p[key], "Original NV index", ylabel, title)
        save_png(fig, out_dir / f"05_{num:02d}_filtered_{key}_vs_nv_index.png")
        figs.append(fig)

    return figs


def plot_histograms(p, out_dir, freqs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for arr, ax, title, xlabel in [
        (p["f1"], axes[0, 0], "Filtered lower resonance distribution", "f1"),
        (p["f2"], axes[0, 1], "Filtered upper resonance distribution", "f2"),
        (p["splitting"], axes[1, 0], "Filtered splitting distribution", "|f2-f1|"),
        (p["width"], axes[1, 1], "Filtered linewidth distribution", "Width"),
    ]:
        vals = np.asarray(arr, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=HIST_BINS)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NV count")
        ax.grid(True, alpha=0.25)

    # Mark target resonances on f1/f2 histograms.
    target_native = _targets_in_native_units(freqs)
    for target in target_native:
        axes[0, 0].axvline(target, linestyle=":", linewidth=0.8, alpha=0.7)
        axes[0, 1].axvline(target, linestyle=":", linewidth=0.8, alpha=0.7)

    for target in SPLIT_TARGETS_GHZ:
        native_target = target if np.nanmedian(np.abs(freqs)) < 100 else target * 1000.0
        axes[1, 0].axvline(native_target, linestyle="--", linewidth=1)

    save_png(fig, out_dir / "06_filtered_frequency_and_width_histograms.png")
    return fig


def plot_filter_diagnostics(p_all, out_dir):
    """
    Helps choose the cutoff rather than guessing it blindly.
    """
    f1g = _to_ghz(p_all["f1"])
    f2g = _to_ghz(p_all["f2"])
    err1 = np.asarray(p_all["f1_target_error_mhz"], dtype=float)
    err2 = np.asarray(p_all["f2_target_error_mhz"], dtype=float)
    success = np.asarray(p_all["success"], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) all fitted centers with target lines
    finite1 = success & np.isfinite(f1g)
    finite2 = success & np.isfinite(f2g)
    axes[0, 0].hist(f1g[finite1], bins=HIST_BINS, alpha=0.65, label="f1")
    axes[0, 0].hist(f2g[finite2], bins=HIST_BINS, alpha=0.65, label="f2")
    for target in RESONANCE_TARGETS_GHZ:
        axes[0, 0].axvline(target, linestyle=":", linewidth=1)
    axes[0, 0].set_xlabel("Fitted center (GHz)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("All fitted resonance centers")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.25)

    # (b) nearest-target error distribution
    vals1 = err1[success & np.isfinite(err1)]
    vals2 = err2[success & np.isfinite(err2)]
    axes[0, 1].hist(vals1, bins=HIST_BINS, alpha=0.65, label="f1 error")
    axes[0, 1].hist(vals2, bins=HIST_BINS, alpha=0.65, label="f2 error")
    axes[0, 1].axvline(
        RESONANCE_FILTER_CUTOFF_MHZ,
        linestyle="--",
        linewidth=2,
        label=f"cutoff={RESONANCE_FILTER_CUTOFF_MHZ:.1f} MHz",
    )
    axes[0, 1].set_xlabel("Distance to nearest target (MHz)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Nearest-target fit errors")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)

    # (c) f1 vs f2, selected highlighted
    selected = np.asarray(p_all["filter_selected"], dtype=bool)
    axes[1, 0].scatter(
        f1g[success & ~selected],
        f2g[success & ~selected],
        s=10,
        alpha=0.20,
        label="Rejected",
    )
    axes[1, 0].scatter(
        f1g[selected],
        f2g[selected],
        s=18,
        alpha=0.70,
        label="Selected",
    )
    for target in RESONANCE_TARGETS_GHZ:
        axes[1, 0].axvline(target, linestyle=":", linewidth=0.6, alpha=0.5)
        axes[1, 0].axhline(target, linestyle=":", linewidth=0.6, alpha=0.5)
    axes[1, 0].set_xlabel("f1 (GHz)")
    axes[1, 0].set_ylabel("f2 (GHz)")
    axes[1, 0].set_title("Two-peak fit classification")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)

    # (d) retained population vs cutoff
    cutoffs = np.linspace(0.5, 25.0, 100)
    counts = []
    for cutoff in cutoffs:
        tmp = build_resonance_filter(
            p_all["f1"],
            p_all["f2"],
            p_all["success"],
            cutoff_mhz=cutoff,
            mode=RESONANCE_FILTER_MODE,
            require_distinct=REQUIRE_DISTINCT_TARGETS,
        )
        counts.append(int(np.sum(tmp["selected"])))

    axes[1, 1].plot(cutoffs, counts, linewidth=2)
    axes[1, 1].axvline(
        RESONANCE_FILTER_CUTOFF_MHZ,
        linestyle="--",
        linewidth=1.5,
    )
    axes[1, 1].axhline(
        int(np.sum(selected)),
        linestyle=":",
        linewidth=1.0,
    )
    axes[1, 1].set_xlabel("Frequency cutoff (MHz)")
    axes[1, 1].set_ylabel("NVs retained")
    axes[1, 1].set_title("Retained population vs cutoff")
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle(
        "Resonance-frequency filter diagnostics\n"
        f"mode={RESONANCE_FILTER_MODE}, "
        f"distinct targets={REQUIRE_DISTINCT_TARGETS}",
        y=0.995,
    )

    save_png(fig, out_dir / "07_resonance_filter_diagnostics.png")
    return fig


def plot_snr_overview(freqs, avg_snr, p_all, out_dir):
    snr = np.asarray(avg_snr, dtype=float)
    selected = np.asarray(p_all["filter_selected"], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if snr.ndim >= 2:
        snr2 = snr.reshape(snr.shape[0], -1)

        if snr2.shape[1] == len(freqs):
            # All traces faint
            for row in snr2:
                axes[0].plot(freqs, row, linewidth=0.35, alpha=0.08)

            # Filtered traces stronger
            for row in snr2[selected]:
                axes[0].plot(freqs, row, linewidth=0.55, alpha=0.25)

            # if np.any(selected):
            #     axes[0].plot(
            #         freqs,
            #         np.nanmedian(snr2[selected], axis=0),
            #         linewidth=2.2,
            #         label="Median filtered SNR",
            #     )

            for target in _targets_in_native_units(freqs):
                axes[0].axvline(target, linestyle=":", linewidth=0.9, alpha=0.7)

            axes[0].set_xlabel("Frequency")
        else:
            # axes[0].plot(np.nanmedian(snr2, axis=0), linewidth=2.0)
            axes[0].set_xlabel("SNR sample index")

    axes[0].set_ylabel("SNR")
    axes[0].set_title("SNR traces with target frequencies")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    idx = np.arange(len(p_all["snr_peak"]))
    axes[1].scatter(
        idx[~selected],
        p_all["snr_peak"][~selected],
        s=10,
        alpha=0.20,
        label="Rejected",
    )
    axes[1].scatter(
        idx[selected],
        p_all["snr_peak"][selected],
        s=18,
        alpha=0.70,
        label="Selected",
    )
    axes[1].set_xlabel("Original NV index")
    axes[1].set_ylabel("Peak |SNR|")
    axes[1].set_title("Per-NV peak |SNR|")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    save_png(fig, out_dir / "08_snr_overview_with_filter.png")
    return fig


def plot_individual_fits_multipage(
    freqs,
    avg_counts,
    avg_counts_ste,
    freqs_dense,
    fit_results,
    p_all,
    out_dir,
):
    """
    Save the selected NVs in one multipage vector PDF.
    FILTERED_INDICES=None includes all NVs; otherwise only the listed indices.
    """
    pdf_path = out_dir / "09_selected_individual_nv_fits.pdf"

    selected_idx = resolve_filtered_indices(len(avg_counts))

    rows = int(np.ceil(FITS_PER_PAGE / FIT_GRID_COLS))

    print(
        f"Saving {len(selected_idx)} selected NV fits to PDF "
        f"({FIT_GRID_COLS} columns x {rows} rows/page)..."
    )

    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(selected_idx), FITS_PER_PAGE):
            page_inds = selected_idx[start : start + FITS_PER_PAGE]

            fig, axes = plt.subplots(
                rows,
                FIT_GRID_COLS,
                figsize=(5.0 * FIT_GRID_COLS, 3.6 * rows),
                squeeze=False,
            )
            axes = axes.ravel()

            for slot, nv_idx in enumerate(page_inds):
                ax = axes[slot]
                y = np.asarray(avg_counts[nv_idx], dtype=float)
                yerr = np.abs(np.asarray(avg_counts_ste[nv_idx], dtype=float))
                fit = fit_results[nv_idx]

                ax.errorbar(
                    freqs,
                    y,
                    yerr=yerr,
                    fmt="o",
                    markersize=2.5,
                    linewidth=0.7,
                    capsize=1.5,
                )

                if np.any(np.isfinite(fit["fit_dense"])):
                    ax.plot(
                        freqs_dense,
                        fit["fit_dense"],
                        linewidth=1.2,
                    )

                for target in _targets_in_native_units(freqs):
                    ax.axvline(
                        target,
                        linestyle=":",
                        linewidth=0.45,
                        alpha=0.4,
                    )

                f1 = p_all["f1"][nv_idx]
                f2 = p_all["f2"][nv_idx]
                e1 = p_all["f1_target_error_mhz"][nv_idx]
                e2 = p_all["f2_target_error_mhz"][nv_idx]

                ax.set_title(
                    f"NV {nv_idx} | {f1:.4f}/{f2:.4f} GHz\n"
                    f"target err={e1:.1f}/{e2:.1f} MHz",
                    fontsize=8.5,
                )
                ax.set_xlabel("Frequency", fontsize=8)
                ax.set_ylabel("Normalized NV population", fontsize=8)
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=7)

            # Hide unused panels on the final page.
            for slot in range(len(page_inds), len(axes)):
                axes[slot].axis("off")

            if len(page_inds):
                fig.suptitle(
                    f"Selected ESR fits: {start + 1}–"
                    f"{start + len(page_inds)} of {len(selected_idx)}",
                    fontsize=14,
                    y=0.995,
                )

            fig.tight_layout(rect=[0, 0, 1, 0.975])

            # PdfPages preserves Matplotlib lines/text as vector graphics.
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved selected-NV vector PDF:\n  {pdf_path}")


# ============================================================================
# CSV EXPORT
# ============================================================================

def _write_parameter_csv(p, path):
    fieldnames = [
        "nv_index",
        "success",
        "filter_selected",
        "f1",
        "f2",
        "nearest_target_f1_ghz",
        "nearest_target_f2_ghz",
        "f1_target_error_mhz",
        "f2_target_error_mhz",
        "f1_target_match",
        "f2_target_match",
        "center",
        "splitting",
        "width",
        "amp1",
        "amp2",
        "contrast",
        "background",
        "reduced_chi2",
        "rms_residual",
        "nfev",
        "snr_median",
        "snr_peak",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(p["nv_index"])):
            writer.writerow(
                {
                    "nv_index": int(p["nv_index"][i]),
                    "success": bool(p["success"][i]),
                    "filter_selected": bool(p["filter_selected"][i]),
                    "f1": p["f1"][i],
                    "f2": p["f2"][i],
                    "nearest_target_f1_ghz": p["nearest_target_f1_ghz"][i],
                    "nearest_target_f2_ghz": p["nearest_target_f2_ghz"][i],
                    "f1_target_error_mhz": p["f1_target_error_mhz"][i],
                    "f2_target_error_mhz": p["f2_target_error_mhz"][i],
                    "f1_target_match": bool(p["f1_target_match"][i]),
                    "f2_target_match": bool(p["f2_target_match"][i]),
                    "center": p["center"][i],
                    "splitting": p["splitting"][i],
                    "width": p["width"][i],
                    "amp1": p["amp1"][i],
                    "amp2": p["amp2"][i],
                    "contrast": p["contrast"][i],
                    "background": p["background"][i],
                    "reduced_chi2": p["red_chi2"][i],
                    "rms_residual": p["rms_resid"][i],
                    "nfev": p["nfev"][i],
                    "snr_median": p["snr_median"][i],
                    "snr_peak": p["snr_peak"][i],
                }
            )

    print(f"Saved: {path}")


def save_parameter_csvs(p_all, p_filtered, out_dir):
    _write_parameter_csv(p_all, out_dir / "fit_parameters_all.csv")
    _write_parameter_csv(p_filtered, out_dir / "fit_parameters_filtered.csv")

    # Easy-to-copy text list of original selected NV indices.
    selected_txt = out_dir / "filtered_nv_indices.txt"
    selected_indices = [int(v) for v in p_filtered["nv_index"]]
    selected_txt.write_text(
        "Filtered NV indices:\n"
        + repr(selected_indices)
        + "\n\n"
        + f"Targets (GHz): {RESONANCE_TARGETS_GHZ}\n"
        + f"Cutoff (MHz): {RESONANCE_FILTER_CUTOFF_MHZ}\n"
        + f"Mode: {RESONANCE_FILTER_MODE}\n"
        + f"Require distinct targets: {REQUIRE_DISTINCT_TARGETS}\n",
        encoding="utf-8",
    )
    print(f"Saved: {selected_txt}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze(file_ids):
    t0 = time.time()

    combined_id = "_".join(map(str, file_ids))
    _, nv_list, freqs, sig_counts, ref_counts = load_and_combine(file_ids)

    print("\nProcessing thresholded resonance counts...")
    avg_counts, avg_counts_ste = widefield.process_counts(
        nv_list,
        sig_counts,
        ref_counts,
        threshold=True,
    )

    print("Calculating SNR...")
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    avg_counts = np.asarray(avg_counts, dtype=float)
    avg_counts_ste = np.asarray(avg_counts_ste, dtype=float)
    avg_snr = np.asarray(avg_snr, dtype=float)

    num_nvs = len(nv_list)
    freqs_dense = np.linspace(np.nanmin(freqs), np.nanmax(freqs), 45)

    cache_file = _cache_path(combined_id, num_nvs, freqs)

    if FIT_CACHE and cache_file.exists():
        print(f"[fit-cache] loading {cache_file}")
        fit_results = load_fit_cache(cache_file)

        if len(fit_results) != num_nvs:
            print("[fit-cache] size mismatch; refitting")
            fit_results = None
    else:
        fit_results = None

    if fit_results is None:
        print(f"Fitting {num_nvs} NVs with n_jobs={N_JOBS}...")

        # Avoid CPU oversubscription from BLAS/OpenMP inside each joblib worker.
        with parallel_backend("loky", inner_max_num_threads=1):
            fit_results = Parallel(
                n_jobs=N_JOBS,
                verbose=FIT_VERBOSE,
                batch_size="auto",
            )(
                delayed(fit_one_nv)(
                    nv_idx,
                    freqs,
                    avg_counts,
                    avg_counts_ste,
                    freqs_dense,
                )
                for nv_idx in range(num_nvs)
            )

        if FIT_CACHE:
            save_fit_cache(cache_file, fit_results)
            print(f"[fit-cache] saved {cache_file}")

    success = np.asarray([r["success"] for r in fit_results], dtype=bool)
    failed = np.flatnonzero(~success)

    valid_nfev = [r["nfev"] for r in fit_results if r["nfev"] > 0]
    median_nfev = np.median(valid_nfev) if valid_nfev else np.nan

    print(
        f"Fit success: {success.sum()}/{num_nvs}; "
        f"failed={len(failed)}; median nfev={median_nfev:.0f}"
    )
    if len(failed):
        print("Failed NV indices:", failed.tolist())

    # p_all = extract_parameters(fit_results, avg_snr)
    # selected_idx = np.flatnonzero(p_all["filter_selected"])
    # p_filtered = subset_parameters(p_all, selected_idx)

    p_all = extract_parameters(fit_results, avg_snr)

    # Use all NVs when FILTERED_INDICES=None, otherwise use the manual subset.
    selected_idx = resolve_filtered_indices(num_nvs)

    manual_mask = np.zeros(num_nvs, dtype=bool)
    manual_mask[selected_idx] = True
    p_all["filter_selected"] = manual_mask

    p_filtered = subset_parameters(p_all, selected_idx)

    print("Filtered idx:", selected_idx.tolist())

    print("\n=== Resonance-frequency filter ===")
    print("Targets (GHz):", RESONANCE_TARGETS_GHZ)
    print(f"Cutoff: {RESONANCE_FILTER_CUTOFF_MHZ:.2f} MHz")
    print("Mode:", RESONANCE_FILTER_MODE)
    print("Require distinct targets:", REQUIRE_DISTINCT_TARGETS)
    print(
        f"Selected: {len(selected_idx)}/{num_nvs} "
        f"({100.0 * len(selected_idx) / num_nvs:.1f}%)"
    )
    print("Selected NV indices:", selected_idx.tolist())

    # Per-target match counts for the two fitted centers.
    print("\nNearest-target counts among selected NVs:")
    for target in RESONANCE_TARGETS_GHZ:
        c1 = np.sum(
            np.isclose(
                p_all["nearest_target_f1_ghz"][selected_idx],
                target,
                atol=1e-9,
            )
        )
        c2 = np.sum(
            np.isclose(
                p_all["nearest_target_f2_ghz"][selected_idx],
                target,
                atol=1e-9,
            )
        )
        print(f"  {target:.4f} GHz: f1={int(c1):3d}, f2={int(c2):3d}")

    print("\n=== Splitting classification within filtered NVs ===")
    for target in SPLIT_TARGETS_GHZ:
        mask = (
            np.isfinite(p_filtered["splitting"])
            & (np.abs(_to_ghz(p_filtered["splitting"]) - target) <= SPLIT_TOL_GHZ)
        )
        inds = np.flatnonzero(mask)
        if len(inds):
            print(
                f"split ~{target:.3f}: N={len(inds)}, "
                f"median f1={np.nanmedian(p_filtered['f1'][inds]):.6f}, "
                f"median f2={np.nanmedian(p_filtered['f2'][inds]):.6f}, "
                f"median split={np.nanmedian(p_filtered['splitting'][inds]):.6f}"
            )
        else:
            print(f"split ~{target:.3f}: N=0")

    out_dir = make_output_dir(combined_id)
    save_parameter_csvs(p_all, p_filtered, out_dir)

    open_figs = []

    if PLOT_ALL_NV_OVERLAY:
        open_figs.append(
            plot_all_nv_overlay(
                freqs,
                avg_counts,
                freqs_dense,
                fit_results,
                p_all,
                out_dir,
            )
        )
        open_figs.append(
            plot_filtered_overlay(
                freqs,
                avg_counts,
                freqs_dense,
                fit_results,
                p_all,
                out_dir,
            )
        )

    if PLOT_HEATMAP:
        open_figs.append(plot_heatmap(freqs, avg_counts, p_all, out_dir))

    if PLOT_PARAMETER_DASHBOARD:
        open_figs.append(plot_parameter_dashboard(p_filtered, out_dir))

    if PLOT_INDIVIDUAL_PARAMETER_PDFS:
        open_figs.extend(plot_individual_parameter_pdfs(p_filtered, out_dir))

    if PLOT_HISTOGRAMS:
        open_figs.append(plot_histograms(p_filtered, out_dir, freqs))

    open_figs.append(plot_filter_diagnostics(p_all, out_dir))

    if PLOT_SNR_OVERVIEW:
        open_figs.append(plot_snr_overview(freqs, avg_snr, p_all, out_dir))

    if PLOT_INDIVIDUAL_FITS_PDF:
        plot_individual_fits_multipage(
            freqs,
            avg_counts,
            avg_counts_ste,
            freqs_dense,
            fit_results,
            p_all,
            out_dir,
        )

    print(f"\nAnalysis complete in {time.time() - t0:.1f} s.")
    print(f"Outputs saved in:\n  {out_dir}")

    if SHOW_PLOTS:
        plt.show(block=True)
    else:
        for fig in open_figs:
            plt.close(fig)


# ============================================================================
# HISTORICAL DATASET SELECTION
# ============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()
    # rubin 140NVs
    file_ids = [1795016507394]
    file_ids = [1796261430133]
    # rubin 107NVs
    file_ids = [1801725762770, 1801943804484]
    # after remoutnig the sample
    # rubin 304NVs
    file_ids = [1803870882950]
    # rubin 154NVs
    file_ids = [1806862148858]
    # rubib 81
    file_ids = [1809016009780]
    # rubib 75
    file_ids = [1810826711017]
    # rubib 75 after change magnet position
    file_ids = [1826522639984]
    # rubib 154 after change magnet position
    file_ids = [1827020564514]
    # # rubib 75 after change magnet position (new position)
    # file_ids = [1829782989309]
    # file_ids = [1830447165544]
    # file_ids = [1831411242534]
    file_ids = [1832069584608]
    file_ids = [1836425531438]

    # file_ids = [
    #     "2025_09_24-09_33_36-rubin-nv0_2025_09_08",
    # ]
    # file_ids = [
    #     "2025_10_03-06_59_37-rubin-nv0_2025_09_08",
    # ]
    ### 308NVs
    # file_ids = [
    #     "2025_10_04-23_59_18-rubin-nv0_2025_09_08",
    # ]

    # ### 254NVs
    file_ids = [
        "2025_10_07-07_19_37-rubin-nv0_2025_09_08",
    ]
    ## 136
    file_ids = [
        "2025_10_09-09_29_58-rubin-nv0_2025_09_08",
    ]

    ## 118 nVs
    # file_ids = [
    #     "2025_10_17-23_28_58-rubin-nv0_2025_09_08",
    # ]

    # ====================johnson sample mounts=========================
    ## 312 nVs
    # file_ids = [
    #     "2025_10_23-08_33_06-johnson-nv0_2025_10_21",
    # ]
    ## 312 nVs
    # file_ids = [
    #     "2025_10_24-09_48_53-johnson-nv0_2025_10_21",
    # ]
    ## 312 nVs
    # file_ids = [
    #     "2025_10_25-12_06_28-johnson-nv0_2025_10_21",
    # ]
    ## 230 nVs
    # file_ids = [
    #     "2025_10_27-11_35_46-johnson-nv0_2025_10_21",
    # ]
    ## 223 nVs
    # file_ids = [
    #     "2025_10_28-03_08_17-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    # file_ids = [
    #     "2025_11_01-07_35_08-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    ## current: I_y (ch1) = 0.73, I_z(ch2)=1.54
    # file_ids = [
    #     "2025_11_04-03_46_51-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    # ## current: I_y (ch1) = 0, I_z(ch2)=1.0
    # file_ids = [
    #     "2025_11_05-02_06_38-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    ## current: I_y (ch1) = 1.0, I_z(ch2)=0
    # file_ids = [
    #     "2025_11_05-22_51_27-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    # # current: I_y (ch1) = 1.0, I_z(ch2)=1.0
    # file_ids = [
    #     "2025_11_06-07_31_12-johnson-nv0_2025_10_21",
    # ]
    ## 312 nVs
    ##  current: I_y (ch1) = 0.0, I_z(ch2)=1.0
    # file_ids = [
    #     "2025_11_07-02_08_20-johnson-nv0_2025_10_21",
    # ]
    ## 312 nVs
    ###  current: I_y (ch1) = 1.0, I_z(ch2)=0.0
    # file_ids = [
    #     "2025_11_07-18_12_34-johnson-nv0_2025_10_21",
    # ]
    ## 312 nVs
    ###  current: I_y (ch1) = 1.0, I_z(ch2)=1.0
    # file_ids = [
    #     "2025_11_08-03_22_11-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    ###  current: I_y (ch1) = 0.0, I_z(ch2)=0.0
    # file_ids = [
    #     "2025_11_09-10_40_49-johnson-nv0_2025_10_21",
    # ]

    ####### Iy=3A, IZ = -3A
    # ## 312 nVs
    # file_ids = [
    #     "2025_11_20-09_14_44-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    # file_ids = [
    #     "2025_11_21-06_06_26-johnson-nv0_2025_10_21",
    # ]

    ####### Iy=3A, IZ =3A
    ## 312 nVs
    # file_ids = [
    #     "2025_11_27-11_18_26-johnson-nv0_2025_10_21",
    # ]

    ####### Iy=3A, IZ =0
    ## 312 nVs
    # file_ids = [
    #     "2025_11_28-01_53_35-johnson-nv0_2025_10_21",
    # ]
    ## 204 nVs
    # file_ids = [
    #     "2025_11_29-04_02_02-johnson-nv0_2025_10_21",
    # ]

    ####### Iy=0, IZ =-3A
    ## 312 nVs
    # file_ids = [
    #     "2025_12_10-10_28_25-johnson-nv0_2025_10_21",
    # ]
    ## 312 nVs
    # file_ids = [
    #     "2025_12_20-06_01_33-johnson-nv0_2025_10_21",
    # ]

    ####### Iy=0, IZ =0
    ##204 NVs
    # file_ids = [
    #     "2026_01_14-08_06_55-johnson-nv0_2025_10_21",
    # ]

    ##312 NVs
    # file_ids = [
    #     "2026_01_19-10_59_31-johnson-nv0_2025_10_21",
    # ]
    # file_ids = [
    #     "2026_02_01-20_03_21-johnson-nv0_2025_10_21",
    # ]

    ####### Iy=0, IZ =0
    ### new set ov nvs with flexible antenna
    # file_ids = [
    #     "2026_02_02-03_48_49-johnson-nv0_2025_10_21",
    # ]

    # file_ids = [
    #     "2026_02_07-04_26_34-johnson-nv0_2025_10_21",
    # ]

    # file_ids = [
    #     "2026_02_09-05_33_54-johnson-nv0_2025_10_21",
    # ]
    ########## Rubin with beads
    # file_ids = [
    #     "2026_02_16-08_20_17-rubin-nv0_2026_02_15",
    # ]

    ##########
    # file_ids = [
    #     "2026_02_19-15_37_59-rubin-nv0_2026_02_15",
    # ]
    # file_ids = [
    #     "2026_02_19-19_16_49-rubin-nv0_2026_02_15",
    #     "2026_02_19-22_39_18-rubin-nv0_2026_02_15",
    # ]
    # ####QNami
    # file_ids = [
    #     "2026_03_04-14_45_10-qnami-nv0_2026_02_20",
    # ]

    # file_ids = [
    #     "2026_03_21-16_56_11-qnami-nv0_2026_02_20",
    #     "2026_03_21-22_26_49-qnami-nv0_2026_02_20"
    # ]
    # file_ids = [
    #     "2026_03_27-13_38_36-qnami-nv0_2026_02_20",
    #     "2026_03_27-18_02_25-qnami-nv0_2026_02_20"
    # ]
    # file_ids = [
    #     "2026_03_28-07_17_36-qnami-nv0_2026_02_20"
    # ]
    # file_ids = [
    #     "2026_03_28-21_55_48-qnami-nv0_2026_02_20"
    # ]
    
    # file_ids = [
    #     "2026_06_16-02_26_28-qnami-nv0_2026_02_20"
    # ]
    file_ids = [
        "2026_09_05-08_06_34-qnami-nv0_2026_02_20"
    ]
    # file_ids = [
    #     "2026_09_11-07_20_45-qnami-nv0_2026_02_20"
    # ]
    # file_ids = [
    #     "2026_09_14-00_27_42-qnami-nv0_2026_02_20"
    # ]
    # file_ids = [
    #     "2026_09_17-11_59_59-qnami-nv0_2026_02_20"
    # ]
    # Run analysis using the final active file_ids assignment above.
    analyze(file_ids)
