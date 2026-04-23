# -*- coding: utf-8 -*-
"""
Fitting program for Single-NV / single-pixel Rabi sweep without base routine.

Creator: chemistatcode
Created on: March 31st, 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


def rabi_oscillation(t, amp, freq, phase, decay, offset):
    """Damped cosine function for Rabi oscillations."""
    return amp * np.exp(-t / decay) * np.cos(2 * np.pi * freq * t + phase) + offset


def load_data(data_dir, base_name):
    """Load Rabi data, trying analysis npz first, then proc JSON, then raw JSON."""
    data_dir = Path(data_dir)

    # Option 1: analysis npz (saved by updated confocal_rabi.py)
    analysis_npz = data_dir / f"{base_name}-analysis.npz"
    if analysis_npz.exists():
        npz = np.load(analysis_npz)
        print(f"Loaded from {analysis_npz.name}")
        return npz["taus_ns"], npz["norm"]

    # Option 2: proc JSON (saved by updated confocal_rabi.py)
    proc_txt = data_dir / f"{base_name}-proc.txt"
    if proc_txt.exists():
        with open(proc_txt, "r") as f:
            proc = json.load(f)
        print(f"Loaded from {proc_txt.name}")
        return np.array(proc["tau_ns_list"]), np.array(proc["norm"])

    # Option 3: raw JSON with inline sig_counts/ref_counts
    raw_txt = data_dir / f"{base_name}.txt"
    with open(raw_txt, "r") as f:
        raw = json.load(f)
    taus = np.array(raw.get("tau_ns_list", raw.get("taus_ns")), dtype=float)
    # Try pre-computed norm first
    if "norm" in raw:
        norm = np.array(raw["norm"], dtype=float)
        if np.any(np.isfinite(norm)):
            print(f"Loaded norm from {raw_txt.name}")
            return taus, norm
    # Fall back to raw counts
    sig = np.array(raw["sig_counts"], dtype=float)
    ref = np.array(raw["ref_counts"], dtype=float)
    # Drop incomplete runs (all-NaN rows)
    valid = ~np.all(np.isnan(sig), axis=1)
    sig_avg = np.nanmean(sig[valid], axis=0)
    ref_avg = np.nanmean(ref[valid], axis=0)
    norm = sig_avg / ref_avg
    print(f"Loaded raw counts from {raw_txt.name}, {valid.sum()} valid runs")
    return taus, norm

#region File Input
def main():
    data_dir = r"G:\nvdata\pc_cryo\branch_master\confocal_rabi\2026_04"
    base_name = "2026_04_23-10_44_14-(Wu)"

    taus_ns, counts = load_data(data_dir, base_name)

    # Drop any NaN/Inf points
    valid = np.isfinite(counts)
    taus_ns = taus_ns[valid]
    counts = counts[valid]

    if len(counts) == 0:
        print("ERROR: No valid data points found. Check the data file.")
        return

    print(f"Fitting {len(counts)} data points")

    # Initial parameter guesses
    offset_guess = np.mean(counts)
    amp_guess = (np.max(counts) - np.min(counts)) / 2.0

    counts_zero_mean = counts - offset_guess
    fft_vals = np.fft.rfft(counts_zero_mean)
    fft_freqs = np.fft.rfftfreq(len(taus_ns), d=(taus_ns[1] - taus_ns[0]))
    freq_guess = fft_freqs[np.argmax(np.abs(fft_vals[1:])) + 1]

    if freq_guess == 0:
        freq_guess = 1.0 / (taus_ns[-1] - taus_ns[0])

    decay_guess = taus_ns[-1] / 2.0
    phase_guess = 0.0

    p0 = [amp_guess, freq_guess, phase_guess, decay_guess, offset_guess]
    bounds = (
        [0, 0, -2 * np.pi, 0, -np.inf],
        [np.inf, np.inf, 2 * np.pi, np.inf, np.inf],
    )

    # Fit
    try:
        popt, pcov = curve_fit(rabi_oscillation, taus_ns, counts, p0=p0, bounds=bounds)
        fit_y = rabi_oscillation(taus_ns, *popt)

        # Goodness-of-fit
        ss_res = np.sum((counts - fit_y) ** 2)
        ss_tot = np.sum((counts - np.mean(counts)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot)
        dof = len(counts) - len(popt)
        red_chi_sq = ss_res / dof
        residuals = counts - fit_y
        fit_success = True

    except Exception as e:
        print(f"Fit failed: {e}")
        popt = p0
        fit_y = rabi_oscillation(taus_ns, *popt)
        residuals = counts - fit_y
        r_squared = None
        red_chi_sq = None
        fit_success = False

    # Terminal output
    amp_fit, freq_fit, phase_fit, decay_fit, offset_fit = popt
    rabi_period = 1.0 / freq_fit
    pi_pulse = rabi_period / 2.0

    print("\n" + "=" * 50)
    print("RABI OSCILLATION FIT RESULTS")
    print("=" * 50)
    print(f"  Amplitude:       {amp_fit:.6f}")
    print(f"  Frequency:       {freq_fit:.6f} GHz  ({freq_fit * 1000:.2f} MHz)")
    print(f"  Phase:           {phase_fit:.4f} rad")
    print(f"  Decay (T2*):     {decay_fit:.2f} ns")
    print(f"  Offset:          {offset_fit:.6f}")
    print("-" * 50)
    print(f"  Rabi Period (1st oscillation): {rabi_period:.2f} ns")
    print(f"  Pi-pulse time:                 {pi_pulse:.2f} ns")
    if fit_success:
        print(f"  R-squared:                     {r_squared:.6f}")
        print(f"  Reduced chi-squared:           {red_chi_sq:.2e}")
    print("=" * 50 + "\n")

    # Plotting: two-panel figure
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # Top panel: data + fit
    ax_main.plot(
        taus_ns, counts, "o", markersize=4, label="Data", color="navy", alpha=0.7
    )
    t_smooth = np.linspace(taus_ns[0], taus_ns[-1], 500)
    ax_main.plot(
        t_smooth, rabi_oscillation(t_smooth, *popt), "-",
        label="Damped Rabi Fit", linewidth=2, color="darkorange",
    )
    ax_main.set_ylabel("Normalized Contrast")
    ax_main.set_title("Confocal Rabi Oscillation")
    ax_main.legend(loc="upper right")
    ax_main.grid(True, linestyle="--", alpha=0.5)

    if fit_success:
        annotation = (
            f"$f$ = {freq_fit * 1000:.1f} MHz\n"
            f"$T_{{\\mathrm{{Rabi}}}}$ = {rabi_period:.1f} ns\n"
            f"$\\pi$-pulse = {pi_pulse:.1f} ns\n"
            f"$\\tau_{{decay}}$ = {decay_fit:.0f} ns\n"
            f"$R^2$ = {r_squared:.4f}"
        )
        ax_main.text(
            0.02, 0.02, annotation, transform=ax_main.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
        )

    # Bottom panel: residuals
    ax_res.plot(
        taus_ns, residuals, "o", markersize=3, color="steelblue", alpha=0.7
    )
    ax_res.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax_res.set_xlabel("Microwave Pulse Duration (ns)")
    ax_res.set_ylabel("Residuals")
    ax_res.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
