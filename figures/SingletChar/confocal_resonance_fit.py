# -*- coding: utf-8 -*-
"""
Fitting program for Single-NV / single-pixel ESR (resonance) sweep produced by
do_resonance in control_panel_cryo.py (which calls confocal_resonance.main).

Fits the NV triplet ms = +1 / -1 doublet with a symmetric pair of Lorentzians
(shared linewidth and contrast, mirrored around the central D frequency).
An optional independent-dip mode is also provided as a fallback.

Creator: chemistatcode
Created on: April 9th, 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from pathlib import Path


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def lorentzian_dip(f, f0, fwhm, contrast):
    """Single Lorentzian dip (positive `contrast` = depth below baseline)."""
    return contrast * (fwhm / 2) ** 2 / ((f - f0) ** 2 + (fwhm / 2) ** 2)


def nv_doublet(f, offset, center, splitting, fwhm, contrast):
    """
    Symmetric NV ms=+/-1 doublet.

    Parameters
    ----------
    offset    : baseline (normalized signal away from resonance)
    center    : zero-field-splitting-like center frequency D (GHz)
    splitting : full peak-to-peak splitting between ms=-1 and ms=+1 (GHz)
    fwhm      : shared FWHM of each Lorentzian dip (GHz)
    contrast  : shared depth of each dip below the baseline
    """
    f_minus = center - splitting / 2.0
    f_plus = center + splitting / 2.0
    return (
        offset
        - lorentzian_dip(f, f_minus, fwhm, contrast)
        - lorentzian_dip(f, f_plus, fwhm, contrast)
    )


def nv_doublet_independent(f, offset, f1, fwhm1, c1, f2, fwhm2, c2):
    """Two Lorentzians with independent widths and contrasts (fallback model)."""
    return (
        offset
        - lorentzian_dip(f, f1, fwhm1, c1)
        - lorentzian_dip(f, f2, fwhm2, c2)
    )


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data(data_dir, base_name):
    """Load resonance data from the .txt JSON file written by save_raw_data."""
    data_dir = Path(data_dir)

    raw_txt = data_dir / f"{base_name}.txt"
    with open(raw_txt, "r") as f:
        raw = json.load(f)
    print(f"Loaded {raw_txt.name}")

    freqs_ghz = np.array(raw["freqs_ghz"], dtype=float)

    # Prefer the pre-computed run-averaged norm if it's there and finite.
    if "norm_mean" in raw:
        norm = np.array(raw["norm_mean"], dtype=float)
        if np.any(np.isfinite(norm)):
            ste = np.array(
                raw.get("norm_ste", np.full_like(norm, np.nan)), dtype=float
            )
            return freqs_ghz, norm, ste

    # Fall back to recomputing from raw counts (drops all-NaN runs).
    sig = np.array(raw["sig_counts"], dtype=float)
    ref = np.array(raw["ref_counts"], dtype=float)
    valid = ~np.all(np.isnan(sig), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_runs = sig[valid] / ref[valid]
    norm = np.nanmean(norm_runs, axis=0)
    n_valid = np.sum(np.isfinite(norm_runs), axis=0)
    ste = np.nanstd(norm_runs, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    print(f"  Recomputed norm from raw counts ({valid.sum()} valid runs)")
    return freqs_ghz, norm, ste


# ----------------------------------------------------------------------
# Initial-guess helpers
# ----------------------------------------------------------------------
def estimate_baseline(norm):
    """Use the upper percentile of the trace as the off-resonance baseline."""
    return float(np.nanpercentile(norm, 85))


def split_minima(freqs, norm):
    """
    Split the sweep at its midpoint and find the minimum in each half.
    Smooth lightly first so that single-point noise doesn't dominate.
    """
    n = len(norm)
    win = max(3, n // 25)
    smoothed = uniform_filter1d(norm, size=win, mode="nearest")
    half = n // 2
    i_left = int(np.nanargmin(smoothed[:half]))
    i_right = int(np.nanargmin(smoothed[half:])) + half
    return freqs[i_left], freqs[i_right]


def estimate_fwhm(freqs, norm, baseline, dip_freq):
    """
    Estimate the FWHM of the dip nearest `dip_freq` from the data itself:
    width at half-depth.
    """
    depth = baseline - np.nanmin(norm)
    if not np.isfinite(depth) or depth <= 0:
        return max((freqs[-1] - freqs[0]) / 20.0, 1e-3)
    half_level = baseline - depth / 2.0
    below = norm < half_level
    if not np.any(below):
        return max((freqs[-1] - freqs[0]) / 20.0, 1e-3)
    # Pick the contiguous run of points below half-level closest to dip_freq.
    idxs = np.where(below)[0]
    near = idxs[np.argmin(np.abs(freqs[idxs] - dip_freq))]
    lo = near
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = near
    while hi < len(below) - 1 and below[hi + 1]:
        hi += 1
    width = freqs[hi] - freqs[lo]
    return max(width, freqs[1] - freqs[0])


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def fit_doublet(freqs, norm, sigma=None):
    """Symmetric ms=+/-1 doublet fit (5 parameters)."""
    fmin, fmax = freqs[0], freqs[-1]
    span = fmax - fmin

    baseline = estimate_baseline(norm)
    f_left, f_right = split_minima(freqs, norm)
    center0 = 0.5 * (f_left + f_right)
    splitting0 = max(f_right - f_left, 2 * (freqs[1] - freqs[0]))
    contrast0 = max(baseline - np.nanmin(norm), 1e-4)
    fwhm0 = 0.5 * (
        estimate_fwhm(freqs, norm, baseline, f_left)
        + estimate_fwhm(freqs, norm, baseline, f_right)
    )

    p0 = [baseline, center0, splitting0, fwhm0, contrast0]
    bounds = (
        [0.0, fmin, 0.0, freqs[1] - freqs[0], 0.0],
        [np.inf, fmax, span, span, 1.0],
    )

    print(
        f"  Initial guess: baseline={baseline:.4f}, center={center0:.4f} GHz, "
        f"splitting={splitting0 * 1e3:.1f} MHz, FWHM={fwhm0 * 1e3:.1f} MHz, "
        f"contrast={contrast0:.4f}"
    )

    popt, pcov = curve_fit(
        nv_doublet, freqs, norm, p0=p0, bounds=bounds,
        sigma=sigma, absolute_sigma=sigma is not None, maxfev=50000,
    )
    return popt, pcov


def fit_doublet_independent(freqs, norm, sigma=None):
    """Two-Lorentzian fit with independent widths and contrasts (7 params)."""
    fmin, fmax = freqs[0], freqs[-1]
    span = fmax - fmin

    baseline = estimate_baseline(norm)
    f_left, f_right = split_minima(freqs, norm)
    contrast0 = max(baseline - np.nanmin(norm), 1e-4)
    fwhm_l = estimate_fwhm(freqs, norm, baseline, f_left)
    fwhm_r = estimate_fwhm(freqs, norm, baseline, f_right)

    p0 = [baseline, f_left, fwhm_l, contrast0, f_right, fwhm_r, contrast0]
    bounds = (
        [0.0, fmin, freqs[1] - freqs[0], 0.0, fmin, freqs[1] - freqs[0], 0.0],
        [np.inf, fmax, span, 1.0, fmax, span, 1.0],
    )

    popt, pcov = curve_fit(
        nv_doublet_independent, freqs, norm, p0=p0, bounds=bounds,
        sigma=sigma, absolute_sigma=sigma is not None, maxfev=50000,
    )
    return popt, pcov


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # ---- USER SETTINGS ----------------------------------------------------
    data_dir = r"G:\nvdata\pc_cryo\branch_master\confocal_resonance\2026_04"
    base_name = "2026_04_09-02_39_31-(lovelace)"
    independent_dips = False  # set True to drop the symmetric constraint
    # -----------------------------------------------------------------------

    freqs_ghz, norm, ste = load_data(data_dir, base_name)

    # Drop any NaN/Inf points.
    valid = np.isfinite(freqs_ghz) & np.isfinite(norm)
    freqs_ghz = freqs_ghz[valid]
    norm = norm[valid]
    if ste is not None:
        ste = ste[valid]
        bad = ~np.isfinite(ste) | (ste <= 0)
        if np.all(bad):
            ste = None
        else:
            ste = ste.copy()
            ste[bad] = np.nanmean(ste[~bad])

    if len(norm) == 0:
        print("ERROR: No valid data points found. Check the data file.")
        return

    print(f"Fitting {len(norm)} data points")

    fit_success = True
    try:
        if independent_dips:
            popt, pcov = fit_doublet_independent(freqs_ghz, norm, sigma=ste)
            fit_func = nv_doublet_independent
            model_label = "Independent double Lorentzian"
        else:
            popt, pcov = fit_doublet(freqs_ghz, norm, sigma=ste)
            fit_func = nv_doublet
            model_label = "NV ms=+/-1 doublet"
        fit_y = fit_func(freqs_ghz, *popt)
        ss_res = np.sum((norm - fit_y) ** 2)
        ss_tot = np.sum((norm - np.mean(norm)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot)
        dof = max(len(norm) - len(popt), 1)
        red_chi_sq = ss_res / dof
        residuals = norm - fit_y
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Fit failed: {e}")
        fit_success = False
        popt = None
        perr = None
        residuals = np.zeros_like(norm)
        r_squared = None
        red_chi_sq = None
        fit_func = nv_doublet_independent if independent_dips else nv_doublet
        model_label = "Independent double Lorentzian" if independent_dips else "NV ms=+/-1 doublet"

    # ----- Terminal output -----
    print("\n" + "=" * 60)
    print(f"CONFOCAL ESR FIT RESULTS  ({model_label})")
    print("=" * 60)
    if fit_success:
        if independent_dips:
            offset, f1, g1, c1, f2, g2, c2 = popt
            of_e, f1_e, g1_e, c1_e, f2_e, g2_e, c2_e = perr
            if f1 > f2:
                f1, f2 = f2, f1
                g1, g2 = g2, g1
                c1, c2 = c2, c1
                f1_e, f2_e = f2_e, f1_e
                g1_e, g2_e = g2_e, g1_e
                c1_e, c2_e = c2_e, c1_e
            center = 0.5 * (f1 + f2)
            splitting = f2 - f1
            print(f"  Baseline offset:   {offset:.6f} +/- {of_e:.6f}")
            print(f"  ms=-1 center:      {f1:.6f} GHz +/- {f1_e * 1e3:.3f} MHz")
            print(f"  ms=-1 FWHM:        {g1 * 1e3:.3f} MHz +/- {g1_e * 1e3:.3f} MHz")
            print(f"  ms=-1 contrast:    {c1:.4f} +/- {c1_e:.4f}")
            print(f"  ms=+1 center:      {f2:.6f} GHz +/- {f2_e * 1e3:.3f} MHz")
            print(f"  ms=+1 FWHM:        {g2 * 1e3:.3f} MHz +/- {g2_e * 1e3:.3f} MHz")
            print(f"  ms=+1 contrast:    {c2:.4f} +/- {c2_e:.4f}")
            print("-" * 60)
            print(f"  D (center):        {center:.6f} GHz")
            print(f"  Splitting:         {splitting * 1e3:.3f} MHz")
        else:
            offset, center, splitting, fwhm, contrast = popt
            of_e, ce_e, sp_e, fw_e, co_e = perr
            f_minus = center - splitting / 2.0
            f_plus = center + splitting / 2.0
            print(f"  Baseline offset:   {offset:.6f} +/- {of_e:.6f}")
            print(f"  D (center):        {center:.6f} GHz +/- {ce_e * 1e3:.3f} MHz")
            print(f"  Splitting:         {splitting * 1e3:.3f} MHz +/- {sp_e * 1e3:.3f} MHz")
            print(f"  FWHM (shared):     {fwhm * 1e3:.3f} MHz +/- {fw_e * 1e3:.3f} MHz")
            print(f"  Contrast (shared): {contrast:.4f} +/- {co_e:.4f}")
            print("-" * 60)
            print(f"  ms = -1:           {f_minus:.6f} GHz")
            print(f"  ms = +1:           {f_plus:.6f} GHz")
        print("-" * 60)
        print(f"  R-squared:           {r_squared:.6f}")
        print(f"  Reduced chi-squared: {red_chi_sq:.2e}")
    else:
        print("  Fit did not converge -- showing data only.")
    print("=" * 60 + "\n")

    # ----- Plotting: two-panel figure -----
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # Top panel: data + fit
    if ste is not None:
        ax_main.errorbar(
            freqs_ghz, norm, yerr=ste, fmt="o", markersize=4,
            color="navy", alpha=0.7, label="Data", capsize=2,
        )
    else:
        ax_main.plot(
            freqs_ghz, norm, "o", markersize=4, color="navy", alpha=0.7, label="Data",
        )

    if fit_success:
        f_smooth = np.linspace(freqs_ghz[0], freqs_ghz[-1], 1000)
        ax_main.plot(
            f_smooth, fit_func(f_smooth, *popt), "-",
            label=model_label, linewidth=2, color="darkorange",
        )
        # Mark the two dip centers.
        if independent_dips:
            f_minus, f_plus = sorted([popt[1], popt[4]])
        else:
            f_minus = popt[1] - popt[2] / 2.0
            f_plus = popt[1] + popt[2] / 2.0
        for fc, lbl in [(f_minus, "ms=-1"), (f_plus, "ms=+1")]:
            ax_main.axvline(fc, color="gray", linestyle=":", linewidth=1)
            ax_main.text(
                fc, ax_main.get_ylim()[1], f" {lbl}",
                fontsize=8, color="gray", verticalalignment="top",
            )

    ax_main.set_ylabel("Normalized Signal")
    ax_main.set_title("Confocal ESR  --  NV ms = +/-1 doublet fit")
    ax_main.legend(loc="lower right")
    ax_main.grid(True, linestyle="--", alpha=0.5)

    if fit_success:
        if independent_dips:
            offset, f1, g1, c1, f2, g2, c2 = popt
            if f1 > f2:
                f1, f2 = f2, f1
                g1, g2 = g2, g1
                c1, c2 = c2, c1
            annotation = (
                f"$f_{{-1}}$ = {f1:.4f} GHz\n"
                f"$f_{{+1}}$ = {f2:.4f} GHz\n"
                f"$\\Delta f$ = {(f2 - f1) * 1e3:.2f} MHz\n"
                f"FWHM$_{{-1}}$ = {g1 * 1e3:.2f} MHz\n"
                f"FWHM$_{{+1}}$ = {g2 * 1e3:.2f} MHz\n"
                f"$R^2$ = {r_squared:.4f}"
            )
        else:
            offset, center, splitting, fwhm, contrast = popt
            annotation = (
                f"$D$ = {center:.4f} GHz\n"
                f"$\\Delta f$ = {splitting * 1e3:.2f} MHz\n"
                f"FWHM = {fwhm * 1e3:.2f} MHz\n"
                f"contrast = {contrast:.3f}\n"
                f"$R^2$ = {r_squared:.4f}"
            )
        ax_main.text(
            0.02, 0.02, annotation, transform=ax_main.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
        )

    # Bottom panel: residuals
    ax_res.plot(
        freqs_ghz, residuals, "o", markersize=3, color="steelblue", alpha=0.7
    )
    ax_res.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_res.set_xlabel("Microwave Frequency (GHz)")
    ax_res.set_ylabel("Residuals")
    ax_res.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
