# -*- coding: utf-8 -*-
"""
Robust SCC amplitude/duration optimization.

Designed for Dioptric wide-field NV data.

Key changes:
- cast counts to float32 before SNR calculation
- amplitude scans use a local weighted quadratic near each measured SNR maximum
- duration scans use slope*(t-delay)*exp[-(t-delay)/decay]
- duration optimum is obtained analytically as delay + decay
- never extrapolate outside the measured scan
- reject non-concave/poor fits and fall back to the measured maximum
- duration optima can be quantized to 4 ns
- return useful results for both amplitude and duration modes
- limit individual plots instead of opening one blocking window per NV

@author: Saroj Chand
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield


# =============================================================================
# USER SETTINGS
# =============================================================================

# FILE_STEM = "2026_09_15-04_51_20-qnami-nv0_2026_02_20"
# FILE_STEM = "2026_09_17-03_23_08-qnami-nv0_2026_02_20"
# FILE_STEM = "2026_10_14-16_05_00-johnson-nv0_2026_10_02"
FILE_STEM = "2026_10_20-22_45_11-johnson-nv0_2026_10_02"

# "amplitude" or "duration"
MODE = "duration"

AMPLITUDE_VALID_RANGE = (0.4, 1.6)
DURATION_VALID_RANGE_NS = (0.0, 400.0)
DURATION_QUANTUM_NS = 4.0

LOCAL_FIT_POINTS = 18
MAX_VERTEX_UNCERTAINTY_FRACTION = 0.30

# Duration physical-model fit settings
DURATION_MIN_FIT_POINTS = 4
DURATION_MAX_OPTIMUM_UNCERTAINTY_FRACTION = 0.35

SHOW_SUMMARY_PLOTS = True
SHOW_INDIVIDUAL_PLOTS = False
PLOT_INDIVIDUAL_FITS = True
MAX_INDIVIDUAL_PLOTS = 212

# PDF output
SAVE_SUMMARY_PDF = True
SAVE_INDIVIDUAL_PDF = True
PDF_COLS = 3
PDF_ROWS = 4

SAVE_RESULTS = True
SAVE_BASENAME = "optimal_scc_parameters_robust"

SAVE_SUMMARY_PNG = True
PNG_DPI = 300

def make_pdf_path(label):
    """Create a valid PDF path using Dioptric's data-manager output folder."""
    timestamp = dm.get_time_stamp()
    base = Path(dm.get_file_path(__file__, timestamp, label))
    pdf_path = base.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    return pdf_path


def save_figures_to_pdf(figures, pdf_path):
    """Save a sequence of Matplotlib figures to one multipage vector PDF."""
    if not figures:
        return

    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    print(f"Saved summary PDF:\n  {pdf_path}")

def save_figures_to_png(figures, label, dpi=PNG_DPI):
    """Save each summary figure as a separate high-resolution PNG."""

    if not figures:
        return

    timestamp = dm.get_time_stamp()

    base = Path(
        dm.get_file_path(
            __file__,
            timestamp,
            label,
        )
    ).with_suffix("")

    base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffixes = [
        "ensemble",
        "optima_distribution",
        "optimum_vs_snr",
    ]

    for ind, fig in enumerate(figures):

        if ind < len(suffixes):
            suffix = suffixes[ind]
        else:
            suffix = f"figure_{ind + 1}"

        png_path = (
            base.parent
            / f"{base.name}_{suffix}.png"
        )

        fig.savefig(
            png_path,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved summary PNG:\n  {png_path}")

@dataclass
class PeakEstimate:
    x_opt: float
    y_opt: float
    x_unc: float
    method: str
    status: str
    raw_x_max: float
    raw_y_max: float
    fit_x: np.ndarray | None = None
    fit_y: np.ndarray | None = None
    delay: float = np.nan
    slope: float = np.nan
    decay: float = np.nan
    continuous_x_opt: float = np.nan

def safe_sigma(sigma):
    """Replace invalid/zero uncertainties with a robust positive value."""
    sigma = np.asarray(sigma, dtype=float).copy()

    good = np.isfinite(sigma) & (sigma > 0)

    if np.any(good):
        fallback = float(np.nanmedian(sigma[good]))
    else:
        fallback = 1.0

    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    sigma[~good] = fallback

    # Stop one accidentally tiny error bar from dominating.
    good_vals = sigma[np.isfinite(sigma) & (sigma > 0)]
    if good_vals.size:
        floor = max(
            0.25 * np.nanpercentile(good_vals, 10),
            np.finfo(float).eps,
        )
        sigma = np.maximum(sigma, floor)

    return sigma


def intersect_valid_range(requested_range, x):
    """Never optimize outside the actual measured scan."""
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        raise ValueError("No finite scan values.")

    lo = max(float(requested_range[0]), float(np.min(finite)))
    hi = min(float(requested_range[1]), float(np.max(finite)))

    if not lo < hi:
        raise ValueError(
            f"Requested range {requested_range} does not overlap "
            f"measured range ({np.min(finite)}, {np.max(finite)})."
        )

    return lo, hi


def quantize(value, quantum):
    if quantum is None:
        return float(value)
    return float(round(value / quantum) * quantum)


def quadratic_value(coeff, x, x0):
    a, b, c = coeff
    dx = np.asarray(x, dtype=float) - x0
    return a * dx**2 + b * dx + c


def estimate_amplitude_peak(
    x,
    y,
    yerr,
    valid_range,
    local_fit_points=LOCAL_FIT_POINTS,
):
    """
    Robust optimum estimate for a single-peaked scan.

    Start from the best measured point, then fit only its local neighborhood.
    If the local quadratic is not a trustworthy downward parabola, keep the
    measured maximum instead.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    yerr = np.asarray(yerr, dtype=float).ravel()

    if not (x.size == y.size == yerr.size):
        raise ValueError("x, y and yerr must have equal length.")

    lo, hi = valid_range

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= lo)
        & (x <= hi)
    )

    if np.sum(mask) == 0:
        return PeakEstimate(
            np.nan, np.nan, np.nan,
            "none", "no_valid_points",
            np.nan, np.nan,
        )

    xx = x[mask]
    yy = y[mask]
    ss = safe_sigma(yerr[mask])

    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    ss = ss[order]

    i_max = int(np.nanargmax(yy))
    raw_x = float(xx[i_max])
    raw_y = float(yy[i_max])

    if xx.size < 3:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "too_few_points",
            raw_x, raw_y,
        )

    nfit = int(np.clip(local_fit_points, 3, xx.size))

    # nearest scan points to the measured maximum
    inds = np.argsort(np.abs(xx - raw_x))[:nfit]
    inds = np.sort(inds)

    xf = xx[inds]
    yf = yy[inds]
    sf = ss[inds]

    if np.unique(xf).size < 3:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "not_enough_distinct_x",
            raw_x, raw_y,
        )

    x0 = raw_x
    dx = xf - x0

    A = np.column_stack([dx**2, dx, np.ones_like(dx)])
    Aw = A / sf[:, None]
    yw = yf / sf

    try:
        coeff, _, rank, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "lstsq_failed",
            raw_x, raw_y,
        )

    if rank < 3:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "rank_deficient",
            raw_x, raw_y,
        )

    a, b, _ = coeff

    # Need a true local maximum.
    if not np.isfinite(a) or a >= 0:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "non_concave_fit",
            raw_x, raw_y,
        )

    vertex = x0 - b / (2.0 * a)

    # Do not extrapolate, even within the global valid range.
    local_lo = float(np.min(xf))
    local_hi = float(np.max(xf))

    if not (
        local_lo <= vertex <= local_hi
        and lo <= vertex <= hi
    ):
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "vertex_outside_local_window",
            raw_x, raw_y,
        )

    # Propagate coefficient covariance to vertex uncertainty.
    try:
        covariance = np.linalg.pinv(Aw.T @ Aw)

        grad = np.array([
            b / (2.0 * a**2),
            -1.0 / (2.0 * a),
            0.0,
        ])

        var_vertex = float(grad @ covariance @ grad)
        x_unc = np.sqrt(max(var_vertex, 0.0))
    except Exception:
        x_unc = np.nan

    span = hi - lo

    if (
        np.isfinite(x_unc)
        and span > 0
        and x_unc > MAX_VERTEX_UNCERTAINTY_FRACTION * span
    ):
        return PeakEstimate(
            raw_x, raw_y, x_unc,
            "measured_max", "vertex_poorly_constrained",
            raw_x, raw_y,
        )

    y_vertex = float(
        quadratic_value(coeff, np.array([vertex]), x0)[0]
    )

    fit_x = np.linspace(local_lo, local_hi, 300)
    fit_y = quadratic_value(coeff, fit_x, x0)

    boundary_tol = 1e-12 * max(1.0, abs(lo), abs(hi))
    at_boundary = (
        abs(raw_x - lo) <= boundary_tol
        or abs(raw_x - hi) <= boundary_tol
    )

    status = "ok_boundary_raw_max" if at_boundary else "ok"

    return PeakEstimate(
        float(vertex),
        y_vertex,
        float(x_unc) if np.isfinite(x_unc) else np.nan,
        "local_weighted_quadratic",
        status,
        raw_x,
        raw_y,
        fit_x,
        fit_y,
    )



# =============================================================================
# DURATION MODEL: PHYSICAL RISE-DECAY FIT
# =============================================================================


def duration_fit_fn(tau, delay, slope, dec):
    """
    SCC duration fit used in the original optimizer:

        SNR(tau) = slope * (tau - delay) * exp[-(tau - delay) / dec]

    For this model, the analytic maximum occurs at:

        tau_opt = delay + dec
    """
    tau = np.asarray(tau, dtype=float) - delay
    return slope * tau * np.exp(-tau / dec)


def estimate_duration_peak(
    x,
    y,
    yerr,
    valid_range,
    quantum=DURATION_QUANTUM_NS,
):
    """
    Fit a single SCC-duration sweep with the physical rise-decay model.

    If the fit fails, is poorly constrained, or predicts an optimum outside
    the measured scan, fall back to the best measured duration.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    yerr = np.asarray(yerr, dtype=float).ravel()

    if not (x.size == y.size == yerr.size):
        raise ValueError("x, y and yerr must have equal length.")

    lo, hi = valid_range

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= lo)
        & (x <= hi)
    )

    if np.sum(mask) == 0:
        return PeakEstimate(
            np.nan, np.nan, np.nan,
            "none", "no_valid_points",
            np.nan, np.nan,
        )

    xx = x[mask]
    yy = y[mask]
    ss = safe_sigma(yerr[mask])

    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    ss = ss[order]

    i_max = int(np.nanargmax(yy))
    raw_x = float(xx[i_max])
    raw_y = float(yy[i_max])

    def fallback(status, x_unc=np.nan, delay=np.nan, slope=np.nan, decay=np.nan):
        reported_x = quantize(raw_x, quantum)
        reported_x = float(np.clip(reported_x, lo, hi))
        return PeakEstimate(
            x_opt=reported_x,
            y_opt=raw_y,
            x_unc=x_unc,
            method="measured_max",
            status=status,
            raw_x_max=raw_x,
            raw_y_max=raw_y,
            delay=delay,
            slope=slope,
            decay=decay,
            continuous_x_opt=raw_x,
        )

    if xx.size < DURATION_MIN_FIT_POINTS:
        return fallback("too_few_points")

    span = hi - lo
    if span <= 0:
        return fallback("zero_scan_span")

    # Since tau_opt = delay + dec, seed the fit from the best measured point.
    # A small positive delay is usually more stable than starting exactly at 0.
    delay_guess = max(
        lo,
        min(
            0.10 * raw_x,
            raw_x - max(0.05 * span, 1e-6),
        ),
    )

    dec_guess = max(
        raw_x - delay_guess,
        0.10 * span,
        1e-6,
    )

    # At the model maximum, y_max = slope * dec / e.
    slope_guess = raw_y * np.e / dec_guess
    if not np.isfinite(slope_guess) or slope_guess <= 0:
        slope_guess = max(np.nanmax(np.abs(yy)), 1e-6) / max(dec_guess, 1e-6)

    p0 = [
        float(delay_guess),
        float(slope_guess),
        float(dec_guess),
    ]

    # Keep delay below the observed maximum. This strongly suppresses
    # unphysical solutions where the fit manufactures a peak before the data.
    delay_upper = min(raw_x, hi)

    lower_bounds = [
        lo,
        0.0,
        1e-9,
    ]

    upper_bounds = [
        delay_upper,
        np.inf,
        max(10.0 * span, 1.0),
    ]

    if not lower_bounds[0] < upper_bounds[0]:
        return fallback("delay_bounds_invalid")

    try:
        popt, pcov = curve_fit(
            duration_fit_fn,
            xx,
            yy,
            p0=p0,
            sigma=ss,
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000,
        )
    except Exception as exc:
        return fallback(f"fit_failed_{type(exc).__name__}")

    delay, slope, dec = [float(v) for v in popt]
    continuous_opt = delay + dec

    # Never extrapolate beyond the actual measured scan.
    if not (lo <= continuous_opt <= hi):
        return fallback(
            "fit_optimum_outside_measured_range",
            delay=delay,
            slope=slope,
            decay=dec,
        )

    # Propagate covariance for tau_opt = delay + dec.
    try:
        var_opt = (
            float(pcov[0, 0])
            + float(pcov[2, 2])
            + 2.0 * float(pcov[0, 2])
        )
        x_unc = np.sqrt(max(var_opt, 0.0))
    except Exception:
        x_unc = np.nan

    if (
        np.isfinite(x_unc)
        and span > 0
        and x_unc > DURATION_MAX_OPTIMUM_UNCERTAINTY_FRACTION * span
    ):
        return fallback(
            "fit_optimum_poorly_constrained",
            x_unc=x_unc,
            delay=delay,
            slope=slope,
            decay=dec,
        )

    reported_opt = quantize(continuous_opt, quantum)
    reported_opt = float(np.clip(reported_opt, lo, hi))
    y_opt = float(duration_fit_fn(reported_opt, *popt))

    fit_x = np.linspace(lo, hi, 500)
    fit_y = duration_fit_fn(fit_x, *popt)

    boundary_tol = 1e-12 * max(1.0, abs(lo), abs(hi))
    at_boundary = (
        abs(raw_x - lo) <= boundary_tol
        or abs(raw_x - hi) <= boundary_tol
    )
    status = "ok_boundary_raw_max" if at_boundary else "ok"

    return PeakEstimate(
        x_opt=reported_opt,
        y_opt=y_opt,
        x_unc=float(x_unc) if np.isfinite(x_unc) else np.nan,
        method="duration_rise_decay",
        status=status,
        raw_x_max=raw_x,
        raw_y_max=raw_y,
        fit_x=fit_x,
        fit_y=fit_y,
        delay=delay,
        slope=slope,
        decay=dec,
        continuous_x_opt=float(continuous_opt),
    )

def analyze_scan(
    data,
    parameter_name,
    requested_range,
    quantum=None,
    fit_mode="amplitude",
):
    """Common engine for amplitude and duration optimization."""
    nv_list = data["nv_list"]
    x = np.asarray(data["taus"], dtype=float).ravel()

    # Important for old float16 datasets.
    counts = np.asarray(data["counts"])
    sig_counts = np.asarray(counts[0], dtype=np.float32)
    ref_counts = np.asarray(counts[1], dtype=np.float32)

    avg_snr, avg_snr_ste = widefield.calc_snr(
        sig_counts,
        ref_counts,
    )

    avg_snr = np.asarray(avg_snr, dtype=float)
    avg_snr_ste = np.asarray(avg_snr_ste, dtype=float)

    if avg_snr.ndim != 2:
        raise ValueError(
            f"Expected avg_snr to be 2D; got {avg_snr.shape}"
        )

    if avg_snr.shape != avg_snr_ste.shape:
        raise ValueError(
            "avg_snr and avg_snr_ste shape mismatch: "
            f"{avg_snr.shape} vs {avg_snr_ste.shape}"
        )

    if avg_snr.shape[0] != len(nv_list):
        raise ValueError(
            f"Expected {len(nv_list)} NV rows; "
            f"got {avg_snr.shape[0]}"
        )

    if avg_snr.shape[1] != x.size:
        raise ValueError(
            f"Expected {x.size} scan columns; "
            f"got {avg_snr.shape[1]}"
        )

    valid_range = intersect_valid_range(
        requested_range,
        x,
    )

    # Ensemble median curve.
    ensemble_snr = np.nanmedian(avg_snr, axis=0)

    n_eff = np.sum(np.isfinite(avg_snr), axis=0)
    ensemble_ste = (
        np.nanmedian(avg_snr_ste, axis=0)
        / np.sqrt(np.maximum(n_eff, 1))
    )
    ensemble_ste = safe_sigma(ensemble_ste)

    if fit_mode == "duration":
        ensemble_peak = estimate_duration_peak(
            x,
            ensemble_snr,
            ensemble_ste,
            valid_range,
            quantum=quantum,
        )
    elif fit_mode == "amplitude":
        ensemble_peak = estimate_amplitude_peak(
            x,
            ensemble_snr,
            ensemble_ste,
            valid_range,
        )
    else:
        raise ValueError("fit_mode must be 'amplitude' or 'duration'.")

    estimates = []

    for nv_ind in range(len(nv_list)):
        if fit_mode == "duration":
            est = estimate_duration_peak(
                x,
                avg_snr[nv_ind],
                avg_snr_ste[nv_ind],
                valid_range,
                quantum=quantum,
            )
        else:
            est = estimate_amplitude_peak(
                x,
                avg_snr[nv_ind],
                avg_snr_ste[nv_ind],
                valid_range,
            )

        # Only truly unusable curves fall back to ensemble.
        if not np.isfinite(est.x_opt):
            est = PeakEstimate(
                ensemble_peak.x_opt,
                np.nan,
                np.nan,
                "ensemble_fallback",
                "ensemble_fallback",
                np.nan,
                np.nan,
            )

        # Duration is already quantized in estimate_duration_peak(), but keeping
        # this here makes the common pipeline explicit and safe.
        est.x_opt = quantize(est.x_opt, quantum)
        est.x_opt = float(np.clip(est.x_opt, *valid_range))

        estimates.append(est)

    opt_values = np.array(
        [est.x_opt for est in estimates],
        dtype=float,
    )

    opt_snrs = np.array(
        [est.y_opt for est in estimates],
        dtype=float,
    )

    opt_unc = np.array(
        [est.x_unc for est in estimates],
        dtype=float,
    )

    median_individual = float(np.nanmedian(opt_values))
    median_individual = quantize(median_individual, quantum)
    median_individual = float(
        np.clip(median_individual, *valid_range)
    )

    results = {
        "parameter_name": parameter_name,
        "fit_mode": fit_mode,
        "requested_valid_range": list(requested_range),
        "actual_valid_range": list(valid_range),
        "measured_scan_values": x.tolist(),
        "ensemble_optimum": float(ensemble_peak.x_opt),
        "ensemble_continuous_optimum": (
            float(ensemble_peak.continuous_x_opt)
            if np.isfinite(ensemble_peak.continuous_x_opt)
            else None
        ),
        "ensemble_optimum_uncertainty": (
            float(ensemble_peak.x_unc)
            if np.isfinite(ensemble_peak.x_unc)
            else None
        ),
        "ensemble_status": ensemble_peak.status,
        "ensemble_method": ensemble_peak.method,
        "median_individual_optimum": median_individual,
        "optimal_values": {
            int(i): float(opt_values[i])
            for i in range(len(nv_list))
        },
        "optimal_snrs": {
            int(i): (
                float(opt_snrs[i])
                if np.isfinite(opt_snrs[i])
                else None
            )
            for i in range(len(nv_list))
        },
        "optimal_value_uncertainties": {
            int(i): (
                float(opt_unc[i])
                if np.isfinite(opt_unc[i])
                else None
            )
            for i in range(len(nv_list))
        },
        "methods": {
            int(i): estimates[i].method
            for i in range(len(nv_list))
        },
        "statuses": {
            int(i): estimates[i].status
            for i in range(len(nv_list))
        },
    }

    if fit_mode == "duration":
        results["ensemble_duration_fit_parameters"] = {
            "delay": (
                float(ensemble_peak.delay)
                if np.isfinite(ensemble_peak.delay)
                else None
            ),
            "slope": (
                float(ensemble_peak.slope)
                if np.isfinite(ensemble_peak.slope)
                else None
            ),
            "decay": (
                float(ensemble_peak.decay)
                if np.isfinite(ensemble_peak.decay)
                else None
            ),
        }

        results["duration_fit_parameters"] = {
            int(i): {
                "delay": (
                    float(estimates[i].delay)
                    if np.isfinite(estimates[i].delay)
                    else None
                ),
                "slope": (
                    float(estimates[i].slope)
                    if np.isfinite(estimates[i].slope)
                    else None
                ),
                "decay": (
                    float(estimates[i].decay)
                    if np.isfinite(estimates[i].decay)
                    else None
                ),
                "continuous_optimum": (
                    float(estimates[i].continuous_x_opt)
                    if np.isfinite(estimates[i].continuous_x_opt)
                    else None
                ),
            }
            for i in range(len(nv_list))
        }

    aux = {
        "x": x,
        "avg_snr": avg_snr,
        "avg_snr_ste": avg_snr_ste,
        "ensemble_snr": ensemble_snr,
        "ensemble_ste": ensemble_ste,
        "ensemble_peak": ensemble_peak,
        "estimates": estimates,
        "opt_values": opt_values,
        "fit_mode": fit_mode,
    }

    return results, aux


def plot_summary(aux, parameter_label, unit=""):
    """Create and return the three compact summary figures."""
    x = aux["x"]
    avg_snr = aux["avg_snr"]
    ensemble_snr = aux["ensemble_snr"]
    ensemble_ste = aux["ensemble_ste"]
    ensemble_peak = aux["ensemble_peak"]
    opt_values = aux["opt_values"]
    fit_mode = aux.get("fit_mode", "amplitude")

    suffix = f" ({unit})" if unit else ""
    figures = []

    # Ensemble curve.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        x,
        ensemble_snr,
        yerr=ensemble_ste,
        fmt="o",
        capsize=3,
        label="Median SNR across NVs",
    )

    if ensemble_peak.fit_x is not None:
        ax.plot(
            ensemble_peak.fit_x,
            ensemble_peak.fit_y,
            label=("Rise-decay fit" if fit_mode == "duration" else "Local weighted quadratic"),
        )

    ax.axvline(
        ensemble_peak.x_opt,
        linestyle="--",
        label=f"Ensemble optimum = {ensemble_peak.x_opt:.4g}",
    )
    ax.set_xlabel(f"{parameter_label}{suffix}")
    ax.set_ylabel("SCC SNR")
    ax.set_title("Ensemble SCC optimization")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figures.append(fig)

    # Distribution of individual optima.
    fig, ax = plt.subplots(figsize=(7, 5))
    finite = opt_values[np.isfinite(opt_values)]

    if finite.size:
        bins = min(
            30,
            max(8, int(np.sqrt(finite.size))),
        )
        ax.hist(finite, bins=bins)
        med = np.nanmedian(finite)
        ax.axvline(
            med,
            linestyle="--",
            label=f"Median = {med:.4g}",
        )

    ax.set_xlabel(f"Optimal {parameter_label.lower()}{suffix}")
    ax.set_ylabel("NV count")
    ax.set_title("Per-NV SCC optima")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    figures.append(fig)

    # Optimum vs best directly measured SNR.
    raw_best_snr = np.nanmax(avg_snr, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        opt_values,
        raw_best_snr,
        alpha=0.65,
    )
    ax.set_xlabel(f"Optimal {parameter_label.lower()}{suffix}")
    ax.set_ylabel("Best measured SCC SNR")
    ax.set_title("Optimum vs measured peak SNR")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figures.append(fig)

    return figures


def plot_individual_examples(
    aux,
    parameter_label,
    unit="",
    max_plots=MAX_INDIVIDUAL_PLOTS,
    save_pdf=SAVE_INDIVIDUAL_PDF,
    show_plots=SHOW_INDIVIDUAL_PLOTS,
    pdf_label="scc-individual-fits",
):
    """
    Plot per-NV SCC optimization diagnostics.

    When save_pdf=True, save the selected NV plots into one multipage
    vector PDF using PDF_COLS x PDF_ROWS panels per page.
    """
    x = aux["x"]
    avg_snr = aux["avg_snr"]
    avg_snr_ste = aux["avg_snr_ste"]
    estimates = aux["estimates"]
    fit_mode = aux.get("fit_mode", "amplitude")

    suffix = f" ({unit})" if unit else ""

    n_total = len(estimates)
    n_plot = min(n_total, int(max_plots)) if max_plots is not None else n_total
    inds = np.arange(n_plot, dtype=int)

    if save_pdf:
        pdf_path = make_pdf_path(pdf_label)
        plots_per_page = PDF_COLS * PDF_ROWS

        with PdfPages(pdf_path) as pdf:
            for start in range(0, len(inds), plots_per_page):
                page_inds = inds[start : start + plots_per_page]

                fig, axes = plt.subplots(
                    PDF_ROWS,
                    PDF_COLS,
                    figsize=(5.0 * PDF_COLS, 3.6 * PDF_ROWS),
                    squeeze=False,
                )
                axes = axes.ravel()

                for slot, nv_ind in enumerate(page_inds):
                    ax = axes[slot]
                    est = estimates[nv_ind]

                    ax.errorbar(
                        x,
                        avg_snr[nv_ind],
                        yerr=avg_snr_ste[nv_ind],
                        fmt="o",
                        markersize=3.0,
                        linewidth=0.7,
                        capsize=1.5,
                        label="Data",
                    )

                    if est.fit_x is not None:
                        ax.plot(
                            est.fit_x,
                            est.fit_y,
                            linewidth=1.2,
                            label=("Rise-decay fit" if fit_mode == "duration" else "Local quadratic fit"),
                        )

                    ax.axvline(
                        est.x_opt,
                        linestyle="--",
                        linewidth=1.0,
                        label=f"Opt = {est.x_opt:.4g}",
                    )

                    ax.set_xlabel(f"{parameter_label}{suffix}", fontsize=8)
                    ax.set_ylabel("SCC SNR", fontsize=8)
                    ax.set_title(
                        f"NV {nv_ind}: {est.status}",
                        fontsize=9,
                    )
                    ax.tick_params(labelsize=7)
                    ax.grid(alpha=0.25)
                    ax.legend(fontsize=6)

                for slot in range(len(page_inds), len(axes)):
                    axes[slot].axis("off")

                if len(page_inds):
                    fig.suptitle(
                        f"{parameter_label} optimization — "
                        f"NV {page_inds[0]} to {page_inds[-1]}",
                        fontsize=14,
                        y=0.995,
                    )

                fig.tight_layout(rect=[0, 0, 1, 0.975])
                pdf.savefig(fig, bbox_inches="tight")

                if show_plots:
                    plt.show(block=False)

                plt.close(fig)

        print(f"Saved individual-fit PDF:\n  {pdf_path}")

    elif show_plots:
        # Optional one-window-per-NV display without PDF saving.
        for nv_ind in inds:
            est = estimates[nv_ind]
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.errorbar(
                x,
                avg_snr[nv_ind],
                yerr=avg_snr_ste[nv_ind],
                fmt="o",
                capsize=2,
                label="Data",
            )

            if est.fit_x is not None:
                ax.plot(est.fit_x, est.fit_y, label=("Rise-decay fit" if fit_mode == "duration" else "Local quadratic fit"))

            ax.axvline(
                est.x_opt,
                linestyle="--",
                label=f"Optimum = {est.x_opt:.4g}",
            )
            ax.set_xlabel(f"{parameter_label}{suffix}")
            ax.set_ylabel("SCC SNR")
            ax.set_title(f"NV {nv_ind}: {est.status}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            plt.show(block=False)


def print_status_summary(results):
    statuses = list(results["statuses"].values())
    names, counts = np.unique(
        statuses,
        return_counts=True,
    )

    print("Fit status counts:")
    for name, count in zip(names, counts):
        print(f"  {name}: {count}")


def process_and_plot_amplitudes(data):

    results, aux = analyze_scan(
        data,
        parameter_name="SCC amplitude",
        requested_range=AMPLITUDE_VALID_RANGE,
        quantum=None,
        fit_mode="amplitude",
    )

    print("\n========== SCC amplitude optimization ==========")
    print(
        "Actual valid range:",
        tuple(results["actual_valid_range"]),
    )
    print(
        "Ensemble optimum:",
        results["ensemble_optimum"],
    )
    print(
        "Median individual optimum:",
        results["median_individual_optimum"],
    )

    print_status_summary(results)

    # -------------------------------------------------------------------------
    # Summary plots
    # -------------------------------------------------------------------------

    if (
        SHOW_SUMMARY_PLOTS
        or SAVE_SUMMARY_PDF
        or SAVE_SUMMARY_PNG
    ):

        summary_figs = plot_summary(
            aux,
            parameter_label="SCC amplitude",
        )

        if SAVE_SUMMARY_PDF:

            save_figures_to_pdf(
                summary_figs,
                make_pdf_path(
                    f"{FILE_STEM}-scc-amplitude-summary"
                ),
            )

        if SAVE_SUMMARY_PNG:

            save_figures_to_png(
                summary_figs,
                f"{FILE_STEM}-scc-amplitude-summary",
            )

        if not SHOW_SUMMARY_PLOTS:
            for fig in summary_figs:
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Individual NV fits
    # -------------------------------------------------------------------------

    if PLOT_INDIVIDUAL_FITS:

        plot_individual_examples(
            aux,
            parameter_label="SCC amplitude",
            save_pdf=SAVE_INDIVIDUAL_PDF,
            show_plots=SHOW_INDIVIDUAL_PLOTS,
            pdf_label=(
                f"{FILE_STEM}-"
                "scc-amplitude-individual-fits"
            ),
        )

    return results

def process_and_plot_durations(data):

    results, aux = analyze_scan(
        data,
        parameter_name="SCC duration",
        requested_range=DURATION_VALID_RANGE_NS,
        quantum=DURATION_QUANTUM_NS,
        fit_mode="duration",
    )

    print("\n========== SCC duration optimization ==========")
    print(
        "Actual valid range:",
        tuple(results["actual_valid_range"]),
        "ns",
    )
    print(
        "Ensemble optimum:",
        results["ensemble_optimum"],
        "ns",
    )
    print(
        "Median individual optimum:",
        results["median_individual_optimum"],
        "ns",
    )

    fit_pars = results.get("ensemble_duration_fit_parameters", {})
    print("Ensemble fit method:", results.get("ensemble_method"))
    print("Ensemble fit status:", results.get("ensemble_status"))
    print("Fit delay:", fit_pars.get("delay"), "ns")
    print("Fit decay:", fit_pars.get("decay"), "ns")
    print("Fit slope:", fit_pars.get("slope"))
    print(
        "Continuous ensemble optimum:",
        results.get("ensemble_continuous_optimum"),
        "ns",
    )
    print(
        "Ensemble optimum uncertainty:",
        results.get("ensemble_optimum_uncertainty"),
        "ns",
    )

    print_status_summary(results)

    # -------------------------------------------------------------------------
    # Summary plots
    # -------------------------------------------------------------------------

    if (
        SHOW_SUMMARY_PLOTS
        or SAVE_SUMMARY_PDF
        or SAVE_SUMMARY_PNG
    ):

        summary_figs = plot_summary(
            aux,
            parameter_label="SCC duration",
            unit="ns",
        )

        if SAVE_SUMMARY_PDF:

            save_figures_to_pdf(
                summary_figs,
                make_pdf_path(
                    f"{FILE_STEM}-scc-duration-summary"
                ),
            )

        if SAVE_SUMMARY_PNG:

            save_figures_to_png(
                summary_figs,
                f"{FILE_STEM}-scc-duration-summary",
            )

        if not SHOW_SUMMARY_PLOTS:
            for fig in summary_figs:
                plt.close(fig)

    # -------------------------------------------------------------------------
    # Individual NV fits
    # -------------------------------------------------------------------------

    if PLOT_INDIVIDUAL_FITS:

        plot_individual_examples(
            aux,
            parameter_label="SCC duration",
            unit="ns",
            save_pdf=SAVE_INDIVIDUAL_PDF,
            show_plots=SHOW_INDIVIDUAL_PLOTS,
            pdf_label=(
                f"{FILE_STEM}-"
                "scc-duration-individual-fits"
            ),
        )

    return results

if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(
        file_stem=FILE_STEM,
        load_npz=True,
    )

    if MODE.lower() == "amplitude":
        results = process_and_plot_amplitudes(data)

    elif MODE.lower() == "duration":
        results = process_and_plot_durations(data)

    else:
        raise ValueError(
            "MODE must be 'amplitude' or 'duration'."
        )

    if SAVE_RESULTS:
        timestamp = dm.get_time_stamp()
        file_path = dm.get_file_path(
            __file__,
            timestamp,
            SAVE_BASENAME,
        )
        dm.save_raw_data(results, file_path)
        print("Saved:", file_path)


    print("\nResults summary")
    print(
        "ensemble optimum =",
        results["ensemble_optimum"],
    )
    print(
        "median individual optimum =",
        results["median_individual_optimum"],
    )

    kpl.show(block=True)
