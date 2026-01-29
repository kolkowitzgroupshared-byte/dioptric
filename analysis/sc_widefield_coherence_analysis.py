# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: sbchand
"""

import sys
import time
import traceback
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste):
    num_nvs = len(nv_list)

    def cos_func(phi, amp, phase_offset):
        return 0.5 * amp * np.cos(phi - phase_offset) + 0.5

    fit_fns = []
    popts = []

    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        guess_params = [1.0, 0.0]

        try:
            popt, _ = curve_fit(
                cos_func,
                phis,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
        except Exception:
            popt = None

        fit_fns.append(cos_func if popt is not None else None)
        popts.append(popt)
        # Create new figure for this NV
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot data points
        ax.errorbar(
            phis,
            nv_counts,
            yerr=abs(nv_counts_ste),
            fmt="o",
            label=f"NV {nv_ind}",
            capsize=3,
        )

        # Plot fit if successful
        if popt is not None:
            phi_fit = np.linspace(min(phis), max(phis), 200)
            fit_vals = cos_func(phi_fit, *popt)
            ax.plot(phi_fit, fit_vals, "-", label="Fit")
            residuals = cos_func(phis, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
            print(f"NV {nv_ind} - Reduced chi²: {red_chi_sq:.3f}")

        ax.set_xlabel("Phase (rad)")
        ax.set_ylabel("Normalized Counts")
        ax.set_title(f"Cosine Fit for NV {nv_ind}")
        ax.legend()
        ax.grid(True)

        # Beautify
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.show(block=True)


# Helper functions for rotations
def R_x(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def R_y(theta):
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def R_z(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


# Simulate spin echo sequence
def spin_echo_signal(phase_array_deg):
    signal = []
    for phi_deg in phase_array_deg:
        phi_rad = np.deg2rad(phi_deg)

        # Initial state: spin up along Z
        bloch_vector = np.array([0, 0, 1])

        # π/2 pulse along X → brings spin to Y
        bloch_vector = R_x(-np.pi / 2) @ bloch_vector

        # Free evolution → identity for ideal case (skip)

        # π pulse along X → refocus
        bloch_vector = R_x(-np.pi) @ bloch_vector

        # Free evolution again → identity

        # Final π/2 pulse along axis with variable phase
        # This is equivalent to a π/2 pulse around an axis in XY plane
        final_rotation = R_z(phi_rad) @ R_x(-np.pi / 2) @ R_z(-phi_rad)
        bloch_vector = final_rotation @ bloch_vector

        # Project onto Z (measurement axis)
        signal.append(bloch_vector[2])  # This is what you measure

    return np.array(signal)


# Simulate and plot
def simulate_plot():
    phase_deg = np.linspace(0, 360, 200)
    signal = spin_echo_signal(phase_deg)

    plt.plot(phase_deg, signal, label="Spin Echo Signal")
    plt.xlabel("Final π/2 Phase (degrees)", fontsize=15)
    plt.ylabel("Z Projection (~ 1 / Signal)", fontsize=15)
    plt.title("Phase-Sensitive Spin Echo Simulation", fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.show()


# Bloch vector rotation functions
def R_axis(theta, n):
    """Return a rotation matrix for angle theta around axis n (unit vector)."""
    n = n / np.linalg.norm(n)
    nx, ny, nz = n
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [
                ct + nx**2 * (1 - ct),
                nx * ny * (1 - ct) - nz * st,
                nx * nz * (1 - ct) + ny * st,
            ],
            [
                ny * nx * (1 - ct) + nz * st,
                ct + ny**2 * (1 - ct),
                ny * nz * (1 - ct) - nx * st,
            ],
            [
                nz * nx * (1 - ct) - ny * st,
                nz * ny * (1 - ct) + nx * st,
                ct + nz**2 * (1 - ct),
            ],
        ]
    )


def apply_sequence(num_pulses, overrotation=0.0, axis_error_deg=0.0):
    """Simulate an XY sequence with pulse imperfections."""
    bloch = np.array([0, 0, 1])  # Start in |0> (Z+)

    # Initial π/2 pulse around X
    bloch = R_axis(-np.pi / 2, np.array([1, 0, 0])) @ bloch

    # Imperfect π pulses
    for i in range(num_pulses):
        # Alternate between X and Y pulses (XY sequence)
        ideal_axis = np.array([1, 0, 0]) if i % 2 == 0 else np.array([0, 1, 0])

        # Apply small axis error
        angle_offset = np.deg2rad(axis_error_deg)
        axis = ideal_axis + angle_offset * np.random.randn(3)

        # π pulse with small overrotation
        pulse_angle = np.pi * (1 + overrotation)
        bloch = R_axis(-pulse_angle, axis) @ bloch

    # Final π/2 pulse (back to Z)
    bloch = R_axis(-np.pi / 2, np.array([1, 0, 0])) @ bloch

    return bloch[2]  # Z projection (fluorescence ~ ms=0)


def simulate_pulse_errors():
    # Sweep over number of pulses
    pulse_counts = [0, 2, 4, 8, 16, 32]
    overrotation = 0.01  # 1% overrotation error
    axis_error_deg = 2.0  # 2° axis error
    signals = []
    for n in pulse_counts:
        signal = apply_sequence(
            n, overrotation=overrotation, axis_error_deg=axis_error_deg
        )
        signals.append(signal)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(pulse_counts, signals, "o-", label="With pulse errors")
    plt.xlabel("Number of π pulses")
    plt.ylabel("Signal (Z projection)")
    plt.title("Decay of coherence due to pulse errors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def cos_func(phi_deg, amp, phase_offset_deg, offset):
    return amp * np.cos(np.radians(phi_deg) - np.radians(phase_offset_deg)) + offset


def fit_phase_fringe(phis_deg, norm_counts, norm_counts_ste):
    """
    Fit all NV phase fringes to cos_func and return a DataFrame with per-NV params.
    Returns:
      fit_df: DataFrame with columns [amplitude, phase_offset, offset, chi_sq, contrast]
      median_fit: dict with popt, phis_fit, vals_fit for the median NV trace
      med_counts, med_counts_ste: median normalized counts (+ STE) over NVs
    """
    num_nvs = norm_counts.shape[0]
    results = {
        "nv_index": [],
        "amplitude": [],
        "phase_offset": [],
        "offset": [],
        "chi_sq": [],
    }

    for i in range(num_nvs):
        y = norm_counts[i]
        yerr = np.asarray(norm_counts_ste[i])
        # Safe uncertainties (avoid zeros)
        yerr = np.where(
            np.asarray(yerr) <= 0,
            np.nanmedian(np.abs(y - np.nanmedian(y))) or 1.0,
            yerr,
        )

        try:
            popt, pcov = curve_fit(
                cos_func,
                phis_deg,
                y,
                p0=[0.25, 0.0, 0.5],
                sigma=yerr,
                absolute_sigma=True,
                maxfev=20000,
            )
            residuals = cos_func(phis_deg, *popt) - y
            dof = max(len(y) - len(popt), 1)
            chi_sq_red = np.nansum((residuals / yerr) ** 2) / dof
        except Exception:
            popt = [np.nan, np.nan, np.nan]
            chi_sq_red = np.nan

        results["amplitude"].append(popt[0])
        results["phase_offset"].append(popt[1])
        results["offset"].append(popt[2])
        results["chi_sq"].append(chi_sq_red)
        results["nv_index"].append(i)

    fit_df = pd.DataFrame(results)
    fit_df["contrast"] = 2.0 * fit_df["amplitude"]  # peak-to-trough

    # Median trace across NVs
    med_counts = np.nanmedian(norm_counts, axis=0)
    med_counts_ste = np.nanmedian(norm_counts_ste, axis=0)

    try:
        popt_med, _ = curve_fit(
            cos_func,
            phis_deg,
            med_counts,
            p0=[0.25, 0.0, 0.5],
            sigma=np.where(
                med_counts_ste <= 0,
                np.nanmedian(np.abs(med_counts - np.nanmedian(med_counts))) or 1.0,
                med_counts_ste,
            ),
            absolute_sigma=True,
            maxfev=20000,
        )
        phi_fit = np.linspace(np.min(phis_deg), np.max(phis_deg), 300)
        vals_fit = cos_func(phi_fit, *popt_med)
        median_fit = {"popt": popt_med, "phi_fit": phi_fit, "vals_fit": vals_fit}
    except Exception:
        median_fit = {
            "popt": [np.nan, np.nan, np.nan],
            "phi_fit": None,
            "vals_fit": None,
        }

    return fit_df, median_fit, med_counts, med_counts_ste


def load_and_process(file_stem):
    """
    Uses your dm.get_raw_data + widefield.process_counts to produce phis (deg),
    norm_counts [n_nv x n_phi], and norm_counts_ste with thresholding.
    """
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True, use_cache=True)
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    phis = np.array(data["phis"])
    # Your fitting uses degrees:
    phis_deg = phis  # if already in degrees; otherwise: np.degrees(phis)
    return phis_deg, np.array(norm_counts), np.array(norm_counts_ste)


def compare_two_runs(file_stem_A, label_A, file_stem_B, label_B, show_individual=False):
    # Load
    phis_A, norm_A, ste_A = load_and_process(file_stem_A)
    phis_B, norm_B, ste_B = load_and_process(file_stem_B)

    # Fit
    fit_A, med_A, medc_A, medste_A = fit_phase_fringe(phis_A, norm_A, ste_A)
    fit_B, med_B, medc_B, medste_B = fit_phase_fringe(phis_B, norm_B, ste_B)

    # --- Print summary stats ---
    def stats(name, df):
        c = df["contrast"].dropna()
        if len(c) == 0:
            print(f"{name}: no valid fits")
            return
        print(f"{name}  (N={len(c)} NVs)")
        print(f"  median contrast: {np.nanmedian(c):.3f}")
        print(f"  mean ± std:      {np.nanmean(c):.3f} ± {np.nanstd(c):.3f}")
        q10, q90 = np.nanpercentile(c, [10, 90])
        print(f"  10–90% range:    [{q10:.3f}, {q90:.3f}]")
        print()

    print("\n=== Per-NV fringe contrast (peak–to–trough = 2*amp) ===")
    stats(label_A, fit_A)
    stats(label_B, fit_B)

    # Median contrast from fitted medians (2*amp_med)
    def median_contrast_from_fit(med):
        amp_med = med["popt"][0] if med["popt"] is not None else np.nan
        return 2.0 * amp_med

    Cmed_A = median_contrast_from_fit(med_A)
    Cmed_B = median_contrast_from_fit(med_B)
    print(f"Median-fit contrast: {label_A}: {Cmed_A:.3f} | {label_B}: {Cmed_B:.3f}")
    print(f"Δ contrast ({label_A} − {label_B}): {Cmed_A - Cmed_B:.3f}\n")

    # --- Plots ---
    # (1) Overlay median fringes
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.errorbar(
        phis_A, medc_A, yerr=medste_A, fmt="o", alpha=0.6, label=f"{label_A} median"
    )
    if med_A["phi_fit"] is not None:
        ax1.plot(med_A["phi_fit"], med_A["vals_fit"], "--", alpha=0.9)

    ax1.errorbar(
        phis_B, medc_B, yerr=medste_B, fmt="s", alpha=0.6, label=f"{label_B} median"
    )
    if med_B["phi_fit"] is not None:
        ax1.plot(med_B["phi_fit"], med_B["vals_fit"], "-.", alpha=0.9)

    ax1.set_xlabel("Phase, φ (degrees)")
    ax1.set_ylabel("Median normalized counts")
    ax1.set_title("Median fringes: overlay")
    ax1.grid(True)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.legend()

    # (2) Side-by-side contrast distributions
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    data_to_plot = [
        fit_A["contrast"].dropna().values,
        fit_B["contrast"].dropna().values,
    ]
    ax2.violinplot(data_to_plot, showmeans=True, showextrema=True)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([label_A, label_B])
    ax2.set_ylabel("Per-NV fringe contrast (peak–to–trough)")
    ax2.set_title("Per-NV contrast distributions")
    ax2.grid(True)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # (3) Optional: plot individual NV fits for each run
    if show_individual:

        def plot_individual(phis, norm, ste, label):
            fig, ax = plt.subplots(figsize=(6, 5))
            for i in range(norm.shape[0]):
                ax.errorbar(phis, norm[i], yerr=np.abs(ste[i]), fmt="o", alpha=0.25)
            ax.set_xlabel("Phase (deg)")
            ax.set_ylabel("Normalized counts")
            ax.set_title(f"All NVs: {label}")
            ax.grid(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        plot_individual(phis_A, norm_A, ste_A, label_A)
        plot_individual(phis_B, norm_B, ste_B, label_B)

    plt.show()

    return fit_A, fit_B, Cmed_A, Cmed_B


def plot_contrast_scatter(fit_A, label_A, fit_B=None, label_B=None):
    """
    Scatter plot of per-NV contrast magnitudes (|2*amp|).
    If fit_B is provided, overlays both runs.
    """
    C_A = np.abs(2 * np.array(fit_A["amplitude"], dtype=float))
    x_A = np.arange(len(C_A))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x_A, C_A, marker="o", alpha=0.8, label=label_A)

    if fit_B is not None:
        C_B = np.abs(2 * np.array(fit_B["amplitude"], dtype=float))
        # Align x by index; if lengths differ, trim to min length
        n = min(len(C_A), len(C_B))
        x_B = np.arange(n) + 0.15  # tiny offset so points don’t fully overlap
        ax.scatter(x_B, C_B[:n], marker="s", alpha=0.8, label=label_B)

    ax.set_xlabel("NV index")
    ax.set_ylabel("Per-NV contrast")
    ax.set_title("Per-NV Fringe Contrast")
    ax.grid(True)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------- NEW HELPERS (append to your file) ----------
def process_one_file_degrees(file_stem):
    """
    Uses your load_and_process(...) + fit_phase_fringe(...) to get per-NV contrast,
    and attaches the evolution time from raw_data.
    Returns (evol_time_ns, fit_df) where fit_df has columns from fit_phase_fringe
    plus 'evol_time_ns' and 'file_stem'.
    """
    # Load raw to get evol_time
    raw = dm.get_raw_data(file_stem=file_stem, load_npz=True, use_cache=True)
    evol_time_ns = int(raw.get("evol_time", -1))

    # Reuse your pipeline for phis/normalized counts
    phis_deg, norm_counts, norm_counts_ste = load_and_process(file_stem)
    fit_df, _, _, _ = fit_phase_fringe(phis_deg, norm_counts, norm_counts_ste)

    fit_df["evol_time_ns"] = evol_time_ns
    fit_df["file_stem"] = file_stem
    # Ensure contrast is positive peak-to-trough
    fit_df["contrast"] = 2.0 * np.abs(fit_df["amplitude"])
    return evol_time_ns, fit_df


def gather_contrast_across_files(file_stems):
    """
    Loops over file stems and concatenates per-NV results into one tidy DataFrame.
    """
    all_dfs = []
    for fs in file_stems:
        _, df = process_one_file_degrees(fs)
        all_dfs.append(df)
    big = pd.concat(all_dfs, ignore_index=True)
    return big.sort_values("evol_time_ns")


def plot_per_nv_contrast_vs_tau(
    df_all, nv_indices=None, title="Per-NV Contrast vs Evolution Time"
):
    """
    Light spaghetti plot: contrast vs τ for each NV (or a chosen subset).
    """
    taus = np.sort(df_all["evol_time_ns"].unique())
    if nv_indices is None:
        nv_indices = sorted(df_all["nv_index"].unique())

    plt.figure(figsize=(7, 5))
    for nv in nv_indices:
        sub = df_all[df_all["nv_index"] == nv].set_index("evol_time_ns").reindex(taus)
        plt.plot(taus, sub["contrast"].values, "-o", alpha=0.35, linewidth=1)

    plt.xlabel("Evolution time τ (ns)")
    plt.ylabel("Per-NV contrast (peak-to-trough)")
    plt.title(title)
    plt.grid(True)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_median_contrast_vs_tau(
    df_all, revival_tau_us=19.6, title="Median NV Contrast vs Evolution Time"
):
    # Clean bad values
    df = df_all.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["contrast", "evol_time_ns"]
    )

    taus_ns = np.sort(df_all["evol_time_ns"].unique())
    taus_us = taus_ns / 1000.0

    med, q1, q3 = [], [], []
    for t_ns in taus_ns:
        vals = df.loc[df["evol_time_ns"] == t_ns, "contrast"].to_numpy()
        vals = vals[np.isfinite(vals)]
        med.append(np.median(vals) if len(vals) else np.nan)
        q1.append(np.percentile(vals, 25) if len(vals) else np.nan)
        q3.append(np.percentile(vals, 75) if len(vals) else np.nan)

    med = np.array(med)
    q1 = np.array(q1)
    q3 = np.array(q3)

    plt.figure(figsize=(7, 5))
    plt.plot(taus_us, med, "-o", linewidth=2, label="Median")
    plt.fill_between(taus_us, q1, q3, alpha=0.25, label="IQR (25–75%)")

    # Mark the revival (input given in µs); pick the closest τ
    if len(taus_us):
        idx = int(np.argmin(np.abs(taus_us - revival_tau_us)))
        plt.scatter(
            [taus_us[idx]],
            [med[idx]],
            color="red",
            s=20,
            zorder=5,
            label=f"Revival ≈ {taus_us[idx]:.3g} µs",
        )

    plt.xlabel("Evolution time τ (µs)")
    plt.ylabel("Contrast (peak-to-trough)")
    plt.xscale("log")
    plt.title(title)
    plt.grid(True)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_contrast_vs_nv_selected_taus(
    df_all, taus_to_plot_ns, title="Per-NV Contrast vs NV Index"
):
    """
    Scatter per-NV contrast vs NV index, but only for selected evolution times.
    Use circle marker for first tau, square for second tau.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Clean data
    df = df_all.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["contrast", "evol_time_ns", "nv_index"]
    )

    taus_ns = [t for t in taus_to_plot_ns if t in df["evol_time_ns"].unique()]
    markers = ["o", "s", "^", "D", "x"]  # extend if more taus

    plt.figure(figsize=(7, 5))
    for i, t_ns in enumerate(taus_ns):
        sub = df[df["evol_time_ns"] == t_ns].sort_values("nv_index")
        if not sub.empty:
            plt.scatter(
                sub["nv_index"],
                sub["contrast"],
                alpha=0.7,
                s=35,
                marker=markers[i % len(markers)],
                label=f"τ={t_ns/1000:g} µs",
            )  # ns→µs

    plt.xlabel("NV index")
    plt.ylabel("Contrast (peak-to-trough)")
    plt.title(title)
    plt.grid(True)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.show()


def contrast_ratio_rev_vs_min(df_all, revival_tau_ns=19600, base_tau_ns=16):
    """
    Compute ratio and difference of median contrast:
      contrast(revival_tau) vs contrast(base_tau == 16 ns).
    """
    df = df_all.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["contrast", "evol_time_ns"]
    )
    taus = df["evol_time_ns"].unique()
    if revival_tau_ns not in taus:
        print(f"Revival τ={revival_tau_ns} ns not found in data.")
        return None
    if base_tau_ns not in taus:
        print(f"Base τ=16 ns not found in data.")
        return None

    med_rev = np.median(
        df.loc[df["evol_time_ns"] == revival_tau_ns, "contrast"].to_numpy()
    )
    med_base = np.median(
        df.loc[df["evol_time_ns"] == base_tau_ns, "contrast"].to_numpy()
    )

    ratio = med_rev / med_base if med_base != 0 else np.nan
    diff = med_rev - med_base

    print(f"Median contrast @ revival (τ={revival_tau_ns} ns): {med_rev:.4f}")
    print(f"Median contrast @ 16 ns: {med_base:.4f}")
    print(f"Contrast ratio (revival / 16 ns): {ratio:.3f}")
    print(f"Contrast difference (revival − 16 ns): {diff:.4f}")

    return {
        "med_rev": med_rev,
        "med_base": med_base,
        "ratio": ratio,
        "difference": diff,
    }


# # ------------------ EXAMPLE CALL ------------------
# if __name__ == "__main__":
#     kpl.init_kplotlib()

#     # fit_spin, fit_xy8, Cmed_spin, Cmed_xy8 = compare_two_runs(
#     #     file_stem_A="2025_10_11-00_03_47-rubin-nv0_2025_09_08",  # spin echo
#     #     label_A="Spin Echo",
#     #     file_stem_B="2025_10_13-14_00_31-rubin-nv0_2025_09_08",   # XY8
#     #     label_B="XY8",
#     #     show_individual=False,  # set True if you want the per-NV scatter panels
#     # )
#     # plot_contrast_scatter(fit_spin, "Spin Echo", fit_xy8, "XY8")

#     file_list = [
#     "2025_10_14-02_12_16-rubin-nv0_2025_09_08",
#     "2025_10_14-03_18_45-rubin-nv0_2025_09_08",
#     "2025_10_14-04_23_11-rubin-nv0_2025_09_08",
#     "2025_10_14-05_27_14-rubin-nv0_2025_09_08",
#     "2025_10_14-06_31_04-rubin-nv0_2025_09_08",
#     "2025_10_14-07_35_17-rubin-nv0_2025_09_08",
#     "2025_10_14-08_40_25-rubin-nv0_2025_09_08",
#     "2025_10_14-09_44_20-rubin-nv0_2025_09_08",
#     "2025_10_14-10_48_49-rubin-nv0_2025_09_08",
#     "2025_10_14-11_55_38-rubin-nv0_2025_09_08",
#     "2025_10_14-13_02_58-rubin-nv0_2025_09_08",
#     "2025_10_14-14_10_50-rubin-nv0_2025_09_08",
#     "2025_10_14-15_20_59-rubin-nv0_2025_09_08",
#     ]

#     file_list = [
#     "2026_01_05-21_44_23-johnson-nv0_2025_10_21",
#     "2026_01_05-22_23_17-johnson-nv0_2025_10_21",
#     "2026_01_05-22_23_18-johnson-nv0_2025_10_21",
#     "2026_01_06-01_24_06-johnson-nv0_2025_10_21",
#     "2026_01_06-01_24_06-johnson-nv0_2025_10_21",
#     "2026_01_06-03_42_20-johnson-nv0_2025_10_21",
#     "2026_01_06-06_01_17-johnson-nv0_2025_10_21",
#     "2026_01_06-08_19_20-johnson-nv0_2025_10_21",
#     "2026_01_06-13_03_27-johnson-nv0_2025_10_21",
#     "2026_01_06-10_43_38-johnson-nv0_2025_10_21",
#     ]

#     df_all = gather_contrast_across_files(file_list)

#     # Median envelope + IQR band:
#     plot_median_contrast_vs_tau(df_all)
#     # plot_contrast_vs_nv_for_all_taus(df_all)
#     # Example call:
#     plot_contrast_vs_nv_selected_taus(df_all, [16, 19600])
#     # Suppose your revival is at τ = 19600 ns
#     res = contrast_ratio_rev_vs_min(df_all, revival_tau_ns=19600, base_tau_ns=16)

#     # Per-NV curves (all NVs lightly; or pass a subset like [0,1,2,3]):
#     # plot_per_nv_contrast_vs_tau(df_all)  # or plot_per_nv_contrast_vs_tau(df_all, nv_indices=[0,1,2,3])

#     kpl.show(block=True)
#     # If you want CSVs of per-NV contrasts:
#     # fit_spin.to_csv("spin_echo_perNV_contrast.csv", index=False)
#     # fit_xy8.to_csv("xy8_perNV_contrast.csv", index=False)


# -----------------------------
# Core fit: y = c + A cosφ + B sinφ
# -----------------------------
def fit_fringe_wls(phis_deg, y, yerr=None):
    """
    Weighted linear least squares fringe fit:
      y(φ) = c + A cosφ + B sinφ
    Returns:
      dict with A,B,c, amp=R, phase0_deg, contrast_ptp=2R, chi2_red
    """
    ph = np.deg2rad(np.asarray(phis_deg, float))
    y = np.asarray(y, float)

    X = np.column_stack([np.cos(ph), np.sin(ph), np.ones_like(ph)])  # [cos, sin, 1]

    if yerr is None:
        w = np.ones_like(y)
    else:
        yerr = np.asarray(yerr, float)
        # avoid zero/neg uncertainties
        yerr = np.where(
            ~np.isfinite(yerr) | (yerr <= 0),
            np.nanmedian(np.abs(y - np.nanmedian(y))) + 1e-12,
            yerr,
        )
        w = 1.0 / (yerr**2)

    # Weighted least squares via sqrt(W)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    # Solve
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    A, B, c = beta

    amp = float(np.sqrt(A * A + B * B))  # R
    phase0 = float(np.arctan2(B, A))  # radians, for cos(φ - phase0)
    phase0_deg = float(np.rad2deg(phase0) % 360.0)

    # Goodness (reduced chi^2)
    yfit = X @ beta
    resid = y - yfit
    dof = max(len(y) - 3, 1)
    chi2_red = float(np.nansum(w * resid**2) / dof)

    return dict(
        A=float(A),
        B=float(B),
        c=float(c),
        amp=amp,
        phase0_deg=phase0_deg,
        contrast_ptp=float(2.0 * amp),
        chi2_red=chi2_red,
    )


# -----------------------------
# Load + compute normalized counts + fit per NV
# -----------------------------
def analyze_fringe_file(file_stem, threshold=False):
    """
    Returns:
      per_nv_df: rows = NVs with contrast + fit params
      meta: dict with seq_type, evol_time_ns, phis_deg, file_stem
      med: dict with median fringe + its fit (contrast_medfit)
    """
    raw = dm.get_raw_data(file_stem=file_stem, load_npz=True, use_cache=True)

    nv_list = raw["nv_list"]
    counts = np.asarray(raw["counts"])  # (2, nv, ..., step, ...)
    sig_counts = counts[0]
    ref_counts = counts[1]

    # normalized counts vs phi (your standard pipeline)
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=threshold
    )
    norm_counts = np.asarray(norm_counts, float)  # (M, nphi)
    norm_counts_ste = np.asarray(norm_counts_ste, float)  # (M, nphi)

    phis = np.asarray(raw["phis"])
    phi_units = raw.get("phi-units", "deg")
    if phi_units.lower().startswith("rad"):
        phis_deg = np.rad2deg(phis)
    else:
        phis_deg = phis.astype(float)

    seq_type = raw.get("seq_type", "unknown")
    evol_time_ns = int(raw.get("evol_time", -1))

    # per-NV fits
    rows = []
    for i in range(norm_counts.shape[0]):
        fit = fit_fringe_wls(phis_deg, norm_counts[i], yerr=norm_counts_ste[i])
        rows.append(
            {
                "file_stem": file_stem,
                "seq_type": seq_type,
                "evol_time_ns": evol_time_ns,
                "nv_index": i,
                **fit,
            }
        )
    per_nv_df = pd.DataFrame(rows)

    # median fringe across NVs + its fit
    med_y = np.nanmedian(norm_counts, axis=0)
    med_e = np.nanmedian(norm_counts_ste, axis=0)
    med_fit = fit_fringe_wls(phis_deg, med_y, yerr=med_e)

    med = dict(
        phis_deg=phis_deg,
        median_fringe=med_y,
        median_fringe_ste=med_e,
        contrast_medfit=med_fit["contrast_ptp"],
        medfit=med_fit,
    )

    meta = dict(
        file_stem=file_stem,
        seq_type=seq_type,
        evol_time_ns=evol_time_ns,
        phis_deg=phis_deg,
    )

    return per_nv_df, meta, med


def analyze_many_files(file_stems, threshold=True):
    """
    Returns:
      per_nv_all: per-NV fits for all files
      run_summary: per-file summary (median across NVs + medfit)
    """
    file_stems = list(dict.fromkeys(file_stems))  # de-dup, preserve order
    per_nv_list = []
    run_rows = []

    for fs in file_stems:
        per_nv_df, meta, med = analyze_fringe_file(fs, threshold=threshold)
        per_nv_list.append(per_nv_df)

        # run-level summaries
        c = per_nv_df["contrast_ptp"].to_numpy()
        run_rows.append(
            {
                "file_stem": fs,
                "seq_type": meta["seq_type"],
                "evol_time_ns": meta["evol_time_ns"],
                "evol_time_us": meta["evol_time_ns"] / 1000.0,
                "median_contrast_perNV": float(np.nanmedian(c)),
                "q25_contrast_perNV": float(np.nanpercentile(c, 25)),
                "q75_contrast_perNV": float(np.nanpercentile(c, 75)),
                "contrast_medfit": float(med["contrast_medfit"]),
            }
        )

    per_nv_all = pd.concat(per_nv_list, ignore_index=True)
    run_summary = pd.DataFrame(run_rows).sort_values(["seq_type", "evol_time_ns"])
    return per_nv_all, run_summary


# -----------------------------
# Plots you asked for
# -----------------------------
def plot_median_contrast_vs_time(run_summary, use="median_contrast_perNV", logx=True):
    """
    One curve per seq_type: median contrast vs evol_time.
    use:
      - "median_contrast_perNV" : median of per-NV contrasts
      - "contrast_medfit"       : contrast from fitting the median fringe
    """
    plt.figure(figsize=(7.5, 5))
    for seq_type, sub in run_summary.groupby("seq_type"):
        sub = sub.sort_values("evol_time_us")
        x = sub["evol_time_us"].to_numpy()
        y = sub[use].to_numpy()
        plt.plot(x, y, "-o", label=f"{seq_type} ({use})")

        # IQR band only makes sense for perNV median
        if use == "median_contrast_perNV":
            q25 = sub["q25_contrast_perNV"].to_numpy()
            q75 = sub["q75_contrast_perNV"].to_numpy()
            plt.fill_between(x, q25, q75, alpha=0.2)

    plt.xlabel("evolution time (µs)")
    plt.ylabel("contrast (peak-to-trough)")
    plt.title("Median contrast vs evolution time by sequence")
    if logx:
        plt.xscale("log")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_contrast_hist(per_nv_all, seq_type, evol_time_ns, bins=30):
    sub = per_nv_all[
        (per_nv_all["seq_type"] == seq_type)
        & (per_nv_all["evol_time_ns"] == evol_time_ns)
    ]
    c = sub["contrast_ptp"].dropna().to_numpy()

    plt.figure(figsize=(6.5, 4.5))
    plt.hist(c, bins=bins)
    plt.xlabel("contrast (peak-to-trough)")
    plt.ylabel("count")
    plt.title(
        f"Per-NV contrast histogram: {seq_type}, τ={evol_time_ns/1000:g} µs (N={len(c)})"
    )
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


def plot_contrast_vs_nv(per_nv_all, seq_type, evol_time_ns):
    sub = per_nv_all[
        (per_nv_all["seq_type"] == seq_type)
        & (per_nv_all["evol_time_ns"] == evol_time_ns)
    ]
    sub = sub.sort_values("nv_index")

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(sub["nv_index"], sub["contrast_ptp"], "o", ms=3)
    plt.xlabel("NV index")
    plt.ylabel("contrast (peak-to-trough)")
    plt.title(f"Per-NV contrast vs NV index: {seq_type}, τ={evol_time_ns/1000:g} µs")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


def plot_median_fringe_overlay(file_stems, threshold=True):
    """
    Overlay median fringes (median across NVs) vs phase for several files.
    Useful sanity check that the phase scan is behaving.
    """
    plt.figure(figsize=(7.5, 5))
    for fs in file_stems:
        _, meta, med = analyze_fringe_file(fs, threshold=threshold)
        ph = med["phis_deg"]
        y = med["median_fringe"]
        plt.plot(
            ph,
            y,
            "-o",
            ms=3,
            label=f"{meta['seq_type']} τ={meta['evol_time_ns']/1000:g}µs",
        )

    plt.xlabel("phase φ (deg)")
    plt.ylabel("median normalized counts")
    plt.title("Median fringe overlay")
    plt.grid(True, ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     kpl.init_kplotlib()
#     file_list = [
#     "2026_01_05-21_44_23-johnson-nv0_2025_10_21",
#     "2026_01_05-22_23_17-johnson-nv0_2025_10_21",
#     "2026_01_05-22_23_18-johnson-nv0_2025_10_21",
#     "2026_01_06-01_24_06-johnson-nv0_2025_10_21",
#     "2026_01_06-01_24_06-johnson-nv0_2025_10_21",
#     "2026_01_06-03_42_20-johnson-nv0_2025_10_21",
#     "2026_01_06-06_01_17-johnson-nv0_2025_10_21",
#     "2026_01_06-08_19_20-johnson-nv0_2025_10_21",
#     "2026_01_06-13_03_27-johnson-nv0_2025_10_21",
#     "2026_01_06-10_43_38-johnson-nv0_2025_10_21",
#     ]
#     per_nv_all, run_summary = analyze_many_files(file_list, threshold=True)

#     print(run_summary[["seq_type","evol_time_ns","median_contrast_perNV","contrast_medfit"]])

#     # 1) Median contrast vs evolution time, separated by seq_type
#     plot_median_contrast_vs_time(run_summary, use="median_contrast_perNV", logx=True)
#     plot_median_contrast_vs_time(run_summary, use="contrast_medfit", logx=True)

#     # 2) Histogram / NV scatter for specific conditions
#     plot_contrast_hist(per_nv_all, seq_type="xy8", evol_time_ns=15000)
#     plot_contrast_vs_nv(per_nv_all, seq_type="xy8", evol_time_ns=15000)

#     # 3) Overlay median fringes (sanity check)
#     plot_median_fringe_overlay(file_list[:6], threshold=True)


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils import widefield
from utils import data_manager as dm


# ----------------------------
# Helpers
# ----------------------------
def _phi_to_degrees(phis):
    """If phis look like radians (<= ~2π), convert to degrees; else assume degrees."""
    phis = np.asarray(phis, float)
    mx = np.nanmax(phis)
    if mx <= 2 * np.pi + 1e-6:
        return np.degrees(phis)
    return phis


def _cos_func_deg(phi_deg, amp, phase_offset_deg, offset):
    return amp * np.cos(np.radians(phi_deg - phase_offset_deg)) + offset


def _fit_cosine_deg(phi_deg, y, yerr):
    """Weighted cosine fit. Returns (amp, phase_deg, offset, red_chi2, success)."""
    phi_deg = np.asarray(phi_deg, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)

    m = np.isfinite(phi_deg) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if np.count_nonzero(m) < 6:
        return np.nan, np.nan, np.nan, np.nan, False

    x = phi_deg[m]
    yy = y[m]
    ee = yerr[m]

    # decent initial guess
    off0 = float(np.nanmedian(yy))
    amp0 = float(0.5 * (np.nanpercentile(yy, 95) - np.nanpercentile(yy, 5)))
    phase0 = float(x[np.nanargmax(yy)]) if np.all(np.isfinite(x)) else 0.0
    p0 = [
        amp0 if np.isfinite(amp0) else 0.1,
        phase0,
        off0 if np.isfinite(off0) else 0.5,
    ]

    try:
        popt, _ = curve_fit(
            _cos_func_deg,
            x,
            yy,
            p0=p0,
            sigma=ee,
            absolute_sigma=True,
            maxfev=20000,
        )
        amp, ph, off = map(float, popt)

        # normalize sign convention: keep amp >= 0
        if np.isfinite(amp) and amp < 0:
            amp = -amp
            ph = (ph + 180.0) % 360.0
        if np.isfinite(ph):
            ph = ph % 360.0

        resid = _cos_func_deg(x, amp, ph, off) - yy
        dof = max(1, len(yy) - 3)
        red_chi2 = float(np.nansum((resid / ee) ** 2) / dof)
        return amp, ph, off, red_chi2, True
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, False


def _parse_N_from_seq_type(seq_type, default=8):
    """Try to infer N from seq_type string like 'xy8', 'xy16', 'hahn'."""
    s = (seq_type or "").lower()
    if "hahn" in s or "echo" in s:
        return 1
    for n in (2, 4, 8, 16):
        if f"xy{n}" in s:
            return n
    return default


# ----------------------------
# Load + extract contrast for one file
# ----------------------------
def extract_phase_scan_contrast(file_stem):
    data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
        use_cache=True,
    )

    nv_list = data["nv_list"]
    phis_deg = _phi_to_degrees(data["phis"])

    counts = np.asarray(data["counts"], float)
    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    norm_counts_ste = np.abs(np.asarray(norm_counts_ste, float))
    # guard against zeros
    good = np.isfinite(norm_counts_ste) & (norm_counts_ste > 0)
    if np.any(good):
        norm_counts_ste[~good] = np.nanmedian(norm_counts_ste[good])

    # evol_time conversion (use unit key if present)
    evol = float(data.get("evol_time", np.nan))
    evol_unit = (
        data.get("evol_time-unit") or data.get("evol_time_unit") or "ns"
    ).lower()
    if evol_unit in ("ns", "nanosecond", "nanoseconds"):
        evol_us = evol / 1e3
    elif evol_unit in ("us", "µs", "microsecond", "microseconds"):
        evol_us = evol
    elif evol_unit in ("s", "sec", "seconds"):
        evol_us = evol * 1e6
    else:
        # assume ns if unknown
        evol_us = evol / 1e3

    seq_type = data.get("seq_type", "")
    N = _parse_N_from_seq_type(seq_type, default=8)

    # total evolution time for XY_N style: t_total = 2 N tau
    # here evol_time is tau in your QUA (it sets tau_ticks), so:
    t_total_us = 2.0 * N * evol_us

    # Fit per-NV amplitudes
    num_nvs = len(nv_list)
    amps = np.full(num_nvs, np.nan)
    offs = np.full(num_nvs, np.nan)
    chis = np.full(num_nvs, np.nan)

    for nv_ind in range(num_nvs):
        amp, ph, off, chi, ok = _fit_cosine_deg(
            phis_deg, norm_counts[nv_ind], norm_counts_ste[nv_ind]
        )
        amps[nv_ind] = amp
        offs[nv_ind] = off
        chis[nv_ind] = chi

    # Robust summary across NVs
    amp_med = float(np.nanmedian(amps))
    amp_p16, amp_p84 = [float(x) for x in np.nanpercentile(amps, [16, 84])]

    # Also fit the MEDIAN fringe (median across NVs at each phi)
    med_fringe = np.nanmedian(norm_counts, axis=0)
    med_fringe_ste = np.nanmedian(norm_counts_ste, axis=0)
    amp_medfit, ph_medfit, off_medfit, chi_medfit, ok_medfit = _fit_cosine_deg(
        phis_deg, med_fringe, med_fringe_ste
    )

    # A useful "contrast" metric from sinusoid A cos + C:
    # contrast ≈ A / C  (equivalently (max-min)/(max+min))
    contrast_nv_med = float(np.nanmedian(amps / offs))
    contrast_medfit = (
        float(amp_medfit / off_medfit)
        if ok_medfit and np.isfinite(off_medfit)
        else np.nan
    )

    return dict(
        file_stem=file_stem,
        seq_type=seq_type,
        N=N,
        evol_us=evol_us,  # this is tau in your sequence
        t_total_us=t_total_us,  # 2*N*tau (useful to compare to your other plots)
        amps=amps,
        offs=offs,
        chis=chis,
        amp_med=amp_med,
        amp_p16=amp_p16,
        amp_p84=amp_p84,
        amp_medfit=amp_medfit,
        off_medfit=off_medfit,
        contrast_nv_med=contrast_nv_med,
        contrast_medfit=contrast_medfit,
    )


# ----------------------------
# Plot contrast decay across multiple files (different evol_time)
# ----------------------------
def plot_contrast_decay(file_stems, x_axis="t_total", use_logx=False):
    """
    x_axis: "t_total" (recommended; matches your t=2N tau plots) or "tau" (raw evol_time).
    """
    rows = [extract_phase_scan_contrast(st) for st in file_stems]
    rows.sort(key=lambda r: r["t_total_us"] if x_axis == "t_total" else r["evol_us"])

    x = np.array(
        [r["t_total_us"] if x_axis == "t_total" else r["evol_us"] for r in rows], float
    )

    # option 1: median amplitude across NVs (with 16–84% band)
    y = np.array([r["amp_med"] for r in rows], float)
    ylo = np.array([r["amp_med"] - r["amp_p16"] for r in rows], float)
    yhi = np.array([r["amp_p84"] - r["amp_med"] for r in rows], float)

    # option 2: amplitude from fit to median fringe
    y_medfit = np.array([r["amp_medfit"] for r in rows], float)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    ax.errorbar(
        x,
        y,
        yerr=np.vstack([ylo, yhi]),
        fmt="o",
        markersize=4,
        capsize=2,
        label="Median(A) across NVs (16–84%)",
    )
    ax.plot(
        x,
        y_medfit,
        "o",
        markersize=4,
        label="A from fit to median fringe",
    )

    if use_logx:
        ax.set_xscale("log")

    if x_axis == "t_total":
        ax.set_xlabel(r"Total evolution time $t = 2N\tau$ (µs)")
    else:
        ax.set_xlabel(r"Interpulse spacing $\tau$ (µs)")

    ax.set_ylabel("Fringe amplitude A (contrast proxy)")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend(frameon=False)

    # tighten x-lims exactly to data (prevents axis starting earlier than data)
    finite = np.isfinite(x) & (x > 0)
    if np.any(finite):
        ax.set_xlim(np.min(x[finite]), np.max(x[finite]))

    fig.tight_layout()
    return fig, ax, rows


# ----------------------------
# Example usage
# ----------------------------
PHASE_SCAN_FILES = [
    "2026_01_28-20_06_27-johnson-nv0_2025_10_21",
    "2026_01_28-22_36_03-johnson-nv0_2025_10_21",
    "2026_01_29-01_18_18-johnson-nv0_2025_10_21",
]

# Plot decay vs TOTAL time (recommended; comparable to your other t=2N tau plots)
fig, ax, rows = plot_contrast_decay(PHASE_SCAN_FILES, x_axis="t_total", use_logx=False)
plt.show()

# If you instead want it vs tau (raw evol_time from file)
# fig, ax, rows = plot_contrast_decay(PHASE_SCAN_FILES, x_axis="tau", use_logx=False)
# plt.show()
