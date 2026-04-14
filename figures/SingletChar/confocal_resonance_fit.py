# -*- coding: utf-8 -*-
"""
Fitting program for the Zeeman-split m_s = +/-1 lines of the NV ground-state
spin triplet, as measured by do_resonance in control_panel_cryo.py
(which calls confocal_resonance.main).

The two dips correspond to the m_s = -1 and m_s = +1 spin sublevels of the
NV triplet ground state, whose degeneracy is lifted by an applied magnetic
field. The two lines are fit independently (each with its own center, FWHM,
and contrast) so that an asymmetric pair -- e.g. one sharp line and one
broad line -- is captured correctly.

Creator: chemistatcode
Created on: April 9th, 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from pathlib import Path


# Electron gyromagnetic ratio for the NV center, used to convert the Zeeman
# splitting to the magnetic field projection along the NV axis.
GAMMA_E_GHZ_PER_T = 28.024  # GHz/T


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def lorentzian_dip(f, f0, fwhm, contrast):
    """Single Lorentzian dip (positive `contrast` = depth below baseline)."""
    return contrast * (fwhm / 2) ** 2 / ((f - f0) ** 2 + (fwhm / 2) ** 2)


def two_lorentzian_dip(
    f, offset, f_minus, fwhm_minus, c_minus, f_plus, fwhm_plus, c_plus
):
    """
    Sum of two independent Lorentzian dips on a flat baseline.

    Models the Zeeman-split m_s = +/-1 NV lines. Each line is fully
    independent in frequency, FWHM, and contrast so that asymmetric pairs
    (one sharp line + one broad line) are captured correctly.
    """
    return (
        offset
        - lorentzian_dip(f, f_minus, fwhm_minus, c_minus)
        - lorentzian_dip(f, f_plus, fwhm_plus, c_plus)
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


def seed_two_lines(freqs, norm):
    """
    Build initial-guess parameters for the two-line dip fit.

    Strategy:
      1. Baseline = 85th percentile.
      2. Sharp-line seed = global maximum of (baseline - norm), no smoothing.
         This guarantees that a single anomalously-low point is picked.
      3. Broad-line seed = minimum of a lightly smoothed trace, with a
         window of +/- 10% of the sweep around the sharp-line seed masked
         out so the same dip isn't picked twice.
      4. FWHM seeds: sharp = one sweep step (the narrowest meaningful
         width given the sampling); broad = data-driven half-depth width.
      5. Returned in (f_minus, f_plus) order with f_minus < f_plus.
    """
    n = len(norm)
    step = freqs[1] - freqs[0]
    baseline = estimate_baseline(norm)

    depth = baseline - norm
    i_sharp = int(np.nanargmax(depth))
    f_sharp = freqs[i_sharp]
    c_sharp = max(float(depth[i_sharp]), 1e-4)
    # A single-point-wide line is undersampled; the smallest meaningful FWHM
    # given the sweep is one step. Don't seed below that.
    fwhm_sharp = step

    # Mask +/- 10% of the sweep around the sharp line so we don't pick the
    # same point as the broad line.
    mask_half = max(int(round(0.10 * n)), 2)
    mask_lo = max(i_sharp - mask_half, 0)
    mask_hi = min(i_sharp + mask_half + 1, n)

    smooth_win = max(3, n // 25)
    smoothed = uniform_filter1d(norm, size=smooth_win, mode="nearest")
    masked = smoothed.copy()
    masked[mask_lo:mask_hi] = np.inf
    if np.all(~np.isfinite(masked)):
        # Degenerate case: only one obvious dip. Place the second seed at
        # the opposite end of the sweep.
        i_broad = 0 if i_sharp > n // 2 else n - 1
    else:
        i_broad = int(np.nanargmin(masked))
    f_broad = freqs[i_broad]
    c_broad = max(baseline - float(smoothed[i_broad]), 1e-4)
    fwhm_broad = estimate_fwhm(freqs, norm, baseline, f_broad)

    # Order so f_minus < f_plus.
    if f_sharp < f_broad:
        f_minus, fwhm_minus, c_minus = f_sharp, fwhm_sharp, c_sharp
        f_plus, fwhm_plus, c_plus = f_broad, fwhm_broad, c_broad
        sharp_label = "m_s=-1 (sharp)"
        broad_label = "m_s=+1 (broad)"
    else:
        f_minus, fwhm_minus, c_minus = f_broad, fwhm_broad, c_broad
        f_plus, fwhm_plus, c_plus = f_sharp, fwhm_sharp, c_sharp
        sharp_label = "m_s=+1 (sharp)"
        broad_label = "m_s=-1 (broad)"

    p0 = [baseline, f_minus, fwhm_minus, c_minus, f_plus, fwhm_plus, c_plus]
    print("  Initial seeds:")
    print(f"    baseline       : {baseline:.6f}")
    print(
        f"    sharp seed     : f = {f_sharp:.6f} GHz, "
        f"data value = {norm[i_sharp]:.6f}, depth = {c_sharp:.4f}, "
        f"FWHM = {fwhm_sharp * 1e3:.3f} MHz  [{sharp_label}]"
    )
    _print_local_window(freqs, norm, i_sharp)
    print(
        f"    broad seed     : f = {f_broad:.6f} GHz, "
        f"data value = {norm[i_broad]:.6f}, depth = {c_broad:.4f}, "
        f"FWHM = {fwhm_broad * 1e3:.3f} MHz  [{broad_label}]"
    )
    _print_local_window(freqs, norm, i_broad)
    return p0


def _print_local_window(freqs, norm, idx, halfwidth=2):
    """Print a small window of data points centered on `idx` for debugging."""
    lo = max(idx - halfwidth, 0)
    hi = min(idx + halfwidth + 1, len(norm))
    print("      local data:")
    for i in range(lo, hi):
        marker = " <-- seed" if i == idx else ""
        print(f"        f = {freqs[i]:.6f} GHz   norm = {norm[i]:.6f}{marker}")


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
def fit_two_lines(freqs, norm):
    """Independent two-Lorentzian fit (7 parameters)."""
    fmin, fmax = freqs[0], freqs[-1]
    span = fmax - fmin
    step = freqs[1] - freqs[0]

    p0 = seed_two_lines(freqs, norm)
    _, f_minus_seed, _, _, f_plus_seed, _, _ = p0

    # Cage each center within +/- 3 sweep steps of its seed. The seed
    # routine reliably finds the right peaks; the danger is curve_fit
    # drifting away from a shallow dip into a flat valley of the cost
    # surface and producing a degenerate fit. Width capped at 4 * step:
    # both Zeeman lines here are at most a couple of points wide, so
    # there's no reason to let the fitter make a 100-MHz Lorentzian.
    center_window = 3 * step
    fwhm_max = 4 * step
    bounds = (
        [
            0.0,
            max(f_minus_seed - center_window, fmin),
            step / 2,
            0.0,
            max(f_plus_seed - center_window, fmin),
            step / 2,
            0.0,
        ],
        [
            np.inf,
            min(f_minus_seed + center_window, fmax),
            fwhm_max,
            1.0,
            min(f_plus_seed + center_window, fmax),
            fwhm_max,
            1.0,
        ],
    )

    # Note: deliberately not passing `sigma` even when it's available.
    # With absolute_sigma + small ste, individual low-noise points get
    # over-weighted and the fit becomes unstable on shallow features.
    # Uniform weights give a more robust answer for low-SNR ODMR dips.
    popt, pcov = curve_fit(
        two_lorentzian_dip, freqs, norm, p0=p0, bounds=bounds,
        maxfev=50000,
    )

    # Enforce f_minus < f_plus on the fitted parameters.
    offset, f1, w1, c1, f2, w2, c2 = popt
    if f1 > f2:
        popt = np.array([offset, f2, w2, c2, f1, w1, c1])
        # Permute the covariance matrix to match the new ordering.
        perm = [0, 4, 5, 6, 1, 2, 3]
        pcov = pcov[np.ix_(perm, perm)]
    return popt, pcov


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # ---- USER SETTINGS ----------------------------------------------------
    data_dir = r"G:\nvdata\pc_cryo\branch_master\confocal_resonance\2026_04"
    base_name = "2026_04_10-16_00_01-(lovelace)"
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
        popt, pcov = fit_two_lines(freqs_ghz, norm)
        fit_y = two_lorentzian_dip(freqs_ghz, *popt)
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

    # ----- Terminal output -----
    print("\n" + "=" * 64)
    print("CONFOCAL ESR FIT RESULTS -- Zeeman-split m_s = +/-1 lines")
    print("=" * 64)
    if fit_success:
        offset, f_minus, fwhm_minus, c_minus, f_plus, fwhm_plus, c_plus = popt
        of_e, fm_e, wm_e, cm_e, fp_e, wp_e, cp_e = perr
        center = 0.5 * (f_minus + f_plus)
        splitting_ghz = f_plus - f_minus
        splitting_mhz = splitting_ghz * 1e3
        b_par_mt = splitting_ghz / (2 * GAMMA_E_GHZ_PER_T) * 1e3  # in mT

        print(f"  Baseline offset:   {offset:.6f} +/- {of_e:.6f}")
        print()
        print(f"  m_s = -1 line:")
        print(f"    center           : {f_minus:.6f} GHz +/- {fm_e * 1e3:.3f} MHz")
        print(f"    FWHM             : {fwhm_minus * 1e3:.3f} MHz +/- {wm_e * 1e3:.3f} MHz")
        print(f"    contrast         : {c_minus:.4f} +/- {cm_e:.4f}")
        print()
        print(f"  m_s = +1 line:")
        print(f"    center           : {f_plus:.6f} GHz +/- {fp_e * 1e3:.3f} MHz")
        print(f"    FWHM             : {fwhm_plus * 1e3:.3f} MHz +/- {wp_e * 1e3:.3f} MHz")
        print(f"    contrast         : {c_plus:.4f} +/- {cp_e:.4f}")
        print("-" * 64)
        print(f"  Center D = (f_-1 + f_+1)/2: {center:.6f} GHz")
        print(f"  Zeeman splitting Df       : {splitting_mhz:.3f} MHz")
        print(f"  B|| (along NV axis)       : {b_par_mt:.3f} mT")
        print("-" * 64)
        print(f"  R-squared:           {r_squared:.6f}")
        print(f"  Reduced chi-squared: {red_chi_sq:.2e}")
    else:
        print("  Fit did not converge -- showing data only.")
    print("=" * 64 + "\n")

    # ----- Plotting: two-panel figure -----
    # Use constrained_layout instead of tight_layout: the dip-center labels
    # are placed with a blended (data, axes-fraction) transform, which
    # tight_layout cannot compute a bounding box for.
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        constrained_layout=True,
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
        f_smooth = np.linspace(freqs_ghz[0], freqs_ghz[-1], 2000)
        ax_main.plot(
            f_smooth, two_lorentzian_dip(f_smooth, *popt), "-",
            label="Two-Lorentzian fit (independent widths)",
            linewidth=2, color="darkorange",
        )
        # Mark the two dip centers. Use a blended transform (x in data
        # coordinates, y in axes fraction) so the labels stay pinned to the
        # top of the panel and don't confuse tight_layout.
        label_transform = ax_main.get_xaxis_transform()
        for fc, lbl in [(popt[1], "m_s=-1"), (popt[4], "m_s=+1")]:
            ax_main.axvline(fc, color="gray", linestyle=":", linewidth=1)
            ax_main.text(
                fc, 0.97, f" {lbl}",
                transform=label_transform,
                fontsize=8, color="gray", verticalalignment="top",
            )

    ax_main.set_ylabel("Normalized Signal")
    ax_main.set_title("Confocal ESR -- NV triplet, Zeeman-split m_s = +/-1 lines")
    ax_main.legend(loc="lower right")
    ax_main.grid(True, linestyle="--", alpha=0.5)

    if fit_success:
        offset, f_minus, fwhm_minus, c_minus, f_plus, fwhm_plus, c_plus = popt
        splitting_mhz = (f_plus - f_minus) * 1e3
        b_par_mt = (f_plus - f_minus) / (2 * GAMMA_E_GHZ_PER_T) * 1e3
        annotation = (
            f"$f_{{-1}}$ = {f_minus:.4f} GHz\n"
            f"$f_{{+1}}$ = {f_plus:.4f} GHz\n"
            f"$\\Delta f$ = {splitting_mhz:.2f} MHz\n"
            f"$B_{{\\parallel}}$ = {b_par_mt:.2f} mT\n"
            f"FWHM$_{{-1}}$ = {fwhm_minus * 1e3:.2f} MHz\n"
            f"FWHM$_{{+1}}$ = {fwhm_plus * 1e3:.2f} MHz\n"
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

    plt.show()


if __name__ == "__main__":
    main()
