# -*- coding: utf-8 -*-
"""
Fitting program for Single-NV / single-pixel ESR (resonance) sweep produced by
do_resonance in control_panel_cryo.py (which calls confocal_resonance.main).

Fits a double-Lorentzian dip model to the normalized signal vs. frequency.

Creator: chemistatcode
Created on: April 9th, 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from pathlib import Path


def lorentzian(f, f0, gamma, contrast):
    """Single Lorentzian dip (positive contrast = depth below baseline)."""
    return contrast * (gamma / 2) ** 2 / ((f - f0) ** 2 + (gamma / 2) ** 2)


def double_lorentzian(f, offset, f1, gamma1, c1, f2, gamma2, c2):
    """Sum of two Lorentzian dips on a flat baseline."""
    return offset - lorentzian(f, f1, gamma1, c1) - lorentzian(f, f2, gamma2, c2)


def single_lorentzian(f, offset, f0, gamma, contrast):
    """Single Lorentzian dip on a flat baseline."""
    return offset - lorentzian(f, f0, gamma, contrast)


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
            ste = np.array(raw.get("norm_ste", np.full_like(norm, np.nan)), dtype=float)
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


def guess_dip_centers(freqs, norm, num_dips):
    """Pick `num_dips` candidate dip centers via peak-finding on the inverted trace."""
    inverted = np.nanmax(norm) - norm
    span = freqs[-1] - freqs[0]
    min_distance = max(1, int(len(freqs) * 0.03))  # ~3% of span
    prominence = 0.5 * np.nanstd(inverted)
    peaks, props = find_peaks(inverted, distance=min_distance, prominence=prominence)
    if len(peaks) == 0:
        # Fallback: pick the absolute minimum / split-quartile minima.
        if num_dips == 1:
            return [freqs[np.nanargmin(norm)]]
        q1 = np.nanargmin(norm[: len(norm) // 2])
        q2 = np.nanargmin(norm[len(norm) // 2 :]) + len(norm) // 2
        return [freqs[q1], freqs[q2]]
    # Sort by prominence, take the top `num_dips`.
    order = np.argsort(props["prominences"])[::-1]
    chosen = peaks[order[:num_dips]]
    chosen = np.sort(chosen)
    if len(chosen) < num_dips:
        # Pad with the global minimum if find_peaks didn't return enough.
        extras = [freqs[np.nanargmin(norm)]] * (num_dips - len(chosen))
        return list(freqs[chosen]) + extras
    return list(freqs[chosen])


def fit_double(freqs, norm, sigma=None):
    offset_guess = np.nanmedian(norm)
    contrast_guess = max(offset_guess - np.nanmin(norm), 1e-4)
    gamma_guess = 0.005  # 5 MHz default linewidth
    centers = guess_dip_centers(freqs, norm, 2)
    p0 = [
        offset_guess,
        centers[0], gamma_guess, contrast_guess,
        centers[1], gamma_guess, contrast_guess,
    ]
    fmin, fmax = freqs[0], freqs[-1]
    bounds = (
        [0.0, fmin, 1e-5, 0.0, fmin, 1e-5, 0.0],
        [np.inf, fmax, fmax - fmin, 1.0, fmax, fmax - fmin, 1.0],
    )
    popt, pcov = curve_fit(
        double_lorentzian, freqs, norm, p0=p0, bounds=bounds,
        sigma=sigma, absolute_sigma=sigma is not None, maxfev=20000,
    )
    return popt, pcov


def fit_single(freqs, norm, sigma=None):
    offset_guess = np.nanmedian(norm)
    contrast_guess = max(offset_guess - np.nanmin(norm), 1e-4)
    gamma_guess = 0.005
    center_guess = guess_dip_centers(freqs, norm, 1)[0]
    p0 = [offset_guess, center_guess, gamma_guess, contrast_guess]
    fmin, fmax = freqs[0], freqs[-1]
    bounds = (
        [0.0, fmin, 1e-5, 0.0],
        [np.inf, fmax, fmax - fmin, 1.0],
    )
    popt, pcov = curve_fit(
        single_lorentzian, freqs, norm, p0=p0, bounds=bounds,
        sigma=sigma, absolute_sigma=sigma is not None, maxfev=20000,
    )
    return popt, pcov


def main():
    # ---- USER SETTINGS ----------------------------------------------------
    data_dir = r"G:\nvdata\pc_cryo\branch_master\confocal_resonance\2026_04"
    base_name = "2026_04_09-02_39_31-(lovelace)"
    num_dips = 2  # set to 1 for a single resonance, 2 for the NV doublet
    # -----------------------------------------------------------------------

    freqs_ghz, norm, ste = load_data(data_dir, base_name)

    # Drop any NaN/Inf points
    valid = np.isfinite(freqs_ghz) & np.isfinite(norm)
    freqs_ghz = freqs_ghz[valid]
    norm = norm[valid]
    ste = ste[valid] if ste is not None else None
    if ste is not None:
        # curve_fit needs strictly positive sigma; replace zeros/NaNs.
        bad = ~np.isfinite(ste) | (ste <= 0)
        if np.all(bad):
            ste = None
        else:
            ste = ste.copy()
            ste[bad] = np.nanmean(ste[~bad])

    if len(norm) == 0:
        print("ERROR: No valid data points found. Check the data file.")
        return

    print(f"Fitting {len(norm)} data points with {num_dips}-Lorentzian model")

    fit_success = True
    try:
        if num_dips == 2:
            popt, pcov = fit_double(freqs_ghz, norm, sigma=ste)
            fit_func = double_lorentzian
        else:
            popt, pcov = fit_single(freqs_ghz, norm, sigma=ste)
            fit_func = single_lorentzian
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
        fit_func = double_lorentzian if num_dips == 2 else single_lorentzian

    # Terminal output
    print("\n" + "=" * 60)
    print("CONFOCAL ESR FIT RESULTS")
    print("=" * 60)
    if fit_success:
        if num_dips == 2:
            offset, f1, g1, c1, f2, g2, c2 = popt
            of_e, f1_e, g1_e, c1_e, f2_e, g2_e, c2_e = perr
            # Order so f1 < f2 for readability
            if f1 > f2:
                f1, f2 = f2, f1
                g1, g2 = g2, g1
                c1, c2 = c2, c1
                f1_e, f2_e = f2_e, f1_e
                g1_e, g2_e = g2_e, g1_e
                c1_e, c2_e = c2_e, c1_e
            splitting_mhz = (f2 - f1) * 1e3
            print(f"  Baseline offset: {offset:.6f} ± {of_e:.6f}")
            print(f"  Dip 1 center:    {f1:.6f} GHz ± {f1_e * 1e3:.3f} MHz")
            print(f"  Dip 1 FWHM:      {g1 * 1e3:.3f} MHz ± {g1_e * 1e3:.3f} MHz")
            print(f"  Dip 1 contrast:  {c1:.4f} ± {c1_e:.4f}")
            print(f"  Dip 2 center:    {f2:.6f} GHz ± {f2_e * 1e3:.3f} MHz")
            print(f"  Dip 2 FWHM:      {g2 * 1e3:.3f} MHz ± {g2_e * 1e3:.3f} MHz")
            print(f"  Dip 2 contrast:  {c2:.4f} ± {c2_e:.4f}")
            print("-" * 60)
            print(f"  Zero-field splitting (D): {(f1 + f2) / 2:.6f} GHz")
            print(f"  Splitting (2E or Zeeman): {splitting_mhz:.3f} MHz")
        else:
            offset, f0, g0, c0 = popt
            of_e, f0_e, g0_e, c0_e = perr
            print(f"  Baseline offset: {offset:.6f} ± {of_e:.6f}")
            print(f"  Center:          {f0:.6f} GHz ± {f0_e * 1e3:.3f} MHz")
            print(f"  FWHM:            {g0 * 1e3:.3f} MHz ± {g0_e * 1e3:.3f} MHz")
            print(f"  Contrast:        {c0:.4f} ± {c0_e:.4f}")
        print("-" * 60)
        print(f"  R-squared:           {r_squared:.6f}")
        print(f"  Reduced chi-squared: {red_chi_sq:.2e}")
    else:
        print("  Fit did not converge — showing data only.")
    print("=" * 60 + "\n")

    # Plotting: two-panel figure
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
            label=f"{num_dips}-Lorentzian Fit", linewidth=2, color="darkorange",
        )

    ax_main.set_ylabel("Normalized Signal")
    ax_main.set_title("Confocal ESR")
    ax_main.legend(loc="lower right")
    ax_main.grid(True, linestyle="--", alpha=0.5)

    if fit_success:
        if num_dips == 2:
            annotation = (
                f"$f_1$ = {f1:.4f} GHz\n"
                f"$f_2$ = {f2:.4f} GHz\n"
                f"$\\Delta f$ = {(f2 - f1) * 1e3:.2f} MHz\n"
                f"FWHM$_1$ = {g1 * 1e3:.2f} MHz\n"
                f"FWHM$_2$ = {g2 * 1e3:.2f} MHz\n"
                f"$R^2$ = {r_squared:.4f}"
            )
        else:
            annotation = (
                f"$f_0$ = {f0:.4f} GHz\n"
                f"FWHM = {g0 * 1e3:.2f} MHz\n"
                f"contrast = {c0:.3f}\n"
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
