# -*- coding: utf-8 -*-
"""
Created on Fall, 2024
@author: Saroj Chand
"""

import os
import sys
import time
import traceback
from datetime import datetime
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from scipy.interpolate import Rbf
from scipy.optimize import curve_fit, least_squares
from sklearn.cluster import KMeans

from majorroutines.pulsed_resonance import fit_resonance, norm_voigt, voigt, voigt_split
from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, NVSpinState


def voigt(
    freq,
    amp1,
    amp2,
    center1,
    center2,
    width,
    bg_offset,
    # bg_slope,
):
    """Voigt profile for two peaks with a linear background."""
    freq = np.array(freq)
    return (
        amp1 * norm_voigt(freq, width, width, center1)
        + amp2 * norm_voigt(freq, width, width, center2)
        + bg_offset
        # + bg_slope * freq
    )


def residuals_fn(params, freq, nv_counts, nv_counts_ste):
    """Compute residuals for least_squares optimization."""
    fit_vals = voigt(freq, *params)
    return (nv_counts - fit_vals) / nv_counts_ste  # Weighted residuals


def plot_nv_resonance(
    nv_list,
    freqs,
    sig_counts,
    ref_counts,
    file_id,
    num_cols=6,
):
    """
    Plot NV resonance data with fitted Voigt profiles (including background), with residuals and contrast values.
    """

    avg_counts, avg_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    num_nvs = len(nv_list)
    freqs_dense = np.linspace(min(freqs), max(freqs), 60)

    def process_nv(nv_idx):
        nv_counts = avg_counts[nv_idx]
        nv_counts_ste = avg_counts_ste[nv_idx]

        low_freq_guess = freqs[np.argmax(nv_counts[: len(freqs) // 2])]
        high_freq_guess = freqs[
            np.argmax(nv_counts[len(freqs) // 2 :]) + len(freqs) // 2
        ]
        max_amp = np.max(nv_counts)

        guess_params = [
            max_amp,
            max_amp,
            low_freq_guess,
            high_freq_guess,
            5,  # Width guess
            np.min(nv_counts),
        ]
        bounds = (
            [0, 0, min(freqs), min(freqs), 0, -np.inf],  # Lower bounds
            [np.inf, np.inf, max(freqs), max(freqs), np.inf, np.inf],  # Upper bounds
        )

        result = least_squares(
            residuals_fn,
            guess_params,
            args=(freqs, nv_counts, nv_counts_ste),
            bounds=bounds,
            max_nfev=20000,
        )
        popt = result.x
        fit_fns = voigt(freqs_dense, *popt)
        # Compute fit
        fit_curve = voigt(freqs, *popt)
        residuals = nv_counts - fit_curve
        chi_squared = np.sum((residuals / nv_counts_ste) ** 2)

        # Extract parameters
        amp1, amp2, f1, f2, width, bg_offset = popt
        contrast = (amp1 + amp2) / 2
        freq_diff = abs(f2 - f1)

        # SNR Calculation
        return (
            fit_fns,
            fit_curve,
            chi_squared,
            (f1, f2),
            freq_diff,
            width,
            contrast,
        )

    # Run in parallel
    fit_results = Parallel(n_jobs=-1)(
        delayed(process_nv)(nv_idx) for nv_idx in range(num_nvs)
    )
    # Unpacking results correctly
    (
        fit_fns,
        fit_data,
        chi_squared_list,
        center_freqs,
        center_freq_differences,
        avg_peak_widths,
        avg_peak_amplitudes,
    ) = zip(*fit_results)

    # Convert results to lists (since zip returns tuples)
    fit_fns = list(fit_fns)
    chi_squared_list = list(chi_squared_list)
    center_freqs = list(center_freqs)
    center_freq_differences = list(center_freq_differences)
    avg_peak_widths = list(avg_peak_widths)
    avg_peak_amplitudes = list(avg_peak_amplitudes)
    # snrs = list(snrs)
    # Plot SNR vs frequency for all NVs
    fig_snr, ax_snr = plt.subplots(figsize=(7, 5))
    for nv_idx in range(num_nvs):
        snrs = np.reshape(avg_snr[nv_idx], len(freqs))
        # Replace NaNs/infs with 0 so the plot still shows
        snrs = np.nan_to_num(snrs, nan=0.0, posinf=0.0, neginf=0.0)

        ax_snr.plot(freqs, snrs, linewidth=1, label=f"NV{nv_idx}")

    ax_snr.set_xlabel("Microwave Frequency (MHz)", fontsize=15)
    ax_snr.set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=15)
    ax_snr.set_title(f"SNR vs Frequency for {num_nvs} NVs", fontsize=15)

    ax_snr.grid(True, linestyle="--", alpha=0.6)
    ax_snr.tick_params(axis="both", labelsize=14)
    plt.show(block=True)
    # # # Set plot style
    # for nv_ind in range(num_nvs):
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     # Data points with error bars
    #     ax.errorbar(
    #         freqs,
    #         avg_counts[nv_ind],
    #         yerr=avg_counts_ste[nv_ind],
    #         fmt="o",
    #         color="steelblue",
    #         ecolor="gray",
    #         elinewidth=1,
    #         capsize=3,
    #         markersize=5,
    #         label="Data"
    #     )
    #     # Fit curve
    #     ax.plot(freqs_dense, fit_fns[nv_ind], "-", color="red", label="Fit")
    #     # Labels and style
    #     ax.set_xlabel("Frequency (GHz)")
    #     ax.set_ylabel("Normalized NV Population")
    #     ax.set_title(f"NV Index: {nv_ind}")
    #     ax.grid(True, linestyle="--", alpha=0.6)
    #     ax.legend()
    #     plt.show(block=True)

    # ----------------- Example of use in your pipeline -----------------
    # center_freqs is your list of (f1, f2) from the fit_results
    # If you can also return (amp1, amp2) per NV from the fit, pass as peak_amps=...
    # targets = (2.766, 2.786, 2.82, 2.840)  # GHz
    targets = (2.787527, 2.840802)  # GHz
    # targets = (2.7230, 2.7470, 2.8290, 2.8467)  # GHz
    out = classify_nv_by_ms_minus_targets(center_freqs, targets_ghz=targets, tol_mhz=60.0)
    # return

    # Access results:
    orientation_bins = out['bins']          # dict: {2.76: [nv_idx,...], 2.78: [...], ...}
    no_match = out['no_match']              # NVs with neither peak near any target
    multi_match = out['multi_match']        # ambiguous
    print("Bin counts:", {k: len(v) for k, v in orientation_bins.items()})
    print("No match:", no_match, "Multi-match:", multi_match)
    # Print the NV indices per orientation bin
    for t, idx_list in out['bins'].items():
        print(f"Target {t:.3f} GHz -> NV indices {idx_list}")
    # return
    ### snrs
    median_snr = np.median(snrs)
    print(f"median snr:{median_snr:.2f}")
    # Remove outliers from a data array using the IQR method.
    Q1 = np.percentile(snrs, 25)
    Q3 = np.percentile(snrs, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    print(f"lower bound:{lower_bound}")
    print(f"upper_bound:{upper_bound}")
    # Identify and filter out outliers
    cleaned_nv_indices = [
        i for i, val in enumerate(snrs) if lower_bound <= val <= upper_bound
    ]
    outlier_indices = [
        i for i, val in enumerate(snrs) if val < lower_bound or val > upper_bound
    ]

    # cleaned_snrs = [val for val in snrs if val in cleaned_nv_indices]
    # Remove index 144 from cleaned_nv_indices first
    # indices_to_remove = [4, 144]
    # cleaned_nv_indices = [
    #     idx for idx in cleaned_nv_indices if idx not in indices_to_remove
    # ]

    cleaned_snrs = [snrs[idx] for idx in cleaned_nv_indices]
    median_snr_cleaned = np.median(cleaned_snrs)
    print(f"snrs IQR: {IQR}")
    print(f"median snr cleaned:{median_snr_cleaned}")
    print(f"Number of nvs removed : {len(outlier_indices)}")
    print(f" removed nvs : {outlier_indices}")
    print(f"Number of nvs after filtering : {len(cleaned_nv_indices)}")
    print(f"cleaned indices: {cleaned_nv_indices}")
    # return
    filter_nvs = True
    # filter_nvs = False
    if filter_nvs:
        # target_peak_values = [0.113, 0.217]
        target_peak_values = [0.077, 0.176]
        # target_peak_values = [0.113, 0.264]
        # target_peak_values = [0.134, 0.326]
        tolerance = 0.01
        # Filter indices based on proximity to target peak differences
        filtered_indices = [
            idx
            for idx, freq_diff in enumerate(center_freq_differences)
            if any(
                target - tolerance <= freq_diff <= target + tolerance
                for target in target_peak_values
            )
        ]
        # Find non-matching indices
        non_matching_indices = [
            idx
            for idx in range(len(center_freq_differences))
            if idx not in filtered_indices
        ]
        if non_matching_indices:
            print(f"Non-matching NVs: {non_matching_indices}")

        # Initialize a dictionary to store indices for each orientation
        orientation_indices = {value: [] for value in target_peak_values}

        for idx, freq_diff in enumerate(center_freq_differences):
            for target in target_peak_values:
                if target - tolerance <= freq_diff <= target + tolerance:
                    orientation_indices[target].append(idx)

        avg_center_freqs = {}
        for orientation, indices in orientation_indices.items():
            if indices:
                freqs_selected_by_orienation = [center_freqs[idx] for idx in indices]
                avg_peak1 = np.median(
                    [freq[0] for freq in freqs_selected_by_orienation]
                )
                avg_peak2 = np.median(
                    [freq[1] for freq in freqs_selected_by_orienation]
                )
                avg_center_freqs[orientation] = (avg_peak1, avg_peak2)
        # Print the results
        for orientation, indices in orientation_indices.items():
            print(f"Orientation centered around {orientation} GHz:")
            print(f"NV Indices: {indices}")
            print(f"Number of NVs: {len(indices)}\n")
        print(f"All SNRs: {snrs}")
        print(f"Median SNR: {np.median(snrs)}")

        for orientation, avg_freqs in avg_center_freqs.items():
            print(f"Orientation {orientation} GHZ")
            print(f"Avearge Peak 1: {avg_freqs[0]:.6f} GHz")
            print(f"Avearge Peak 1: {avg_freqs[1]:.6f} GHz")

    else:
        filtered_indices = list(range(num_nvs))
    # filtered_indices =  [0, 1, 2, 4, 9, 10, 12, 14, 15, 17, 18, 20, 22, 23, 24, 27, 28, 29, 30, 33, 34, 37, 40, 41, 42, 45, 46, 49, 50, 51, 54, 55, 57, 58, 59, 60, 62, 65, 67, 68, 69, 72, 73, 75, 78, 79, 80, 81, 82, 85, 86, 87, 88, 90, 94, 95, 98, 99, 101, 102, 104, 106, 107, 111, 113, 114, 116, 117, 119, 122, 123, 125, 126, 127, 128, 130, 131, 133, 134, 135, 136, 137, 142, 143, 144, 145, 146, 148, 149, 151, 153, 155, 158, 161, 163, 164, 165, 166, 167, 170, 172, 173, 174, 175, 178, 181, 183, 185, 186, 187, 191, 192, 193, 195, 196, 197, 199, 200, 201, 203, 205, 207, 210, 211, 212, 214, 216, 218, 220, 221, 223, 225, 226, 227, 228, 229, 230, 233, 235, 237, 238, 239, 242, 244, 245, 246, 247, 249, 250, 252, 253]

    # mannual removal of indices
    # indices_to_remove_manually = []
    # filtered_indices = [
    #     filtered_indices[idx]
    #     for idx in range(num_nvs)
    #     if idx not in indices_to_remove_manually
    # ]
    return
    filtered_nv_list = [nv_list[idx] for idx in filtered_indices]
    filtered_avg_counts = [avg_counts[idx] for idx in filtered_indices]
    filtered_avg_counts_ste = [avg_counts_ste[idx] for idx in filtered_indices]
    filtered_center_freqs = [center_freqs[idx] for idx in filtered_indices]
    filtered_freq_differences = [
        center_freq_differences[idx] for idx in filtered_indices
    ]
    filtered_avg_peak_widths = [avg_peak_widths[idx] for idx in filtered_indices]
    filtered_avg_peak_amplitudes = [
        avg_peak_amplitudes[idx] for idx in filtered_indices
    ]
    # filtered_contrast_list = [contrast_list[idx] for idx in filtered_indices]
    filtered_chi_squared_list = [chi_squared_list[idx] for idx in filtered_indices]
    # filtered_fitted_data = [fit_data[idx] for idx in filtered_indices]
    filtered_fitted_data = [fit_fns[idx] for idx in filtered_indices]
    filtered_avg_snr = [avg_snr[idx] for idx in filtered_indices]
    
    fig_snr, ax_snr = plt.subplots(figsize=(7, 5))
    for nv_idx in range(num_nvs):
        snrs = np.reshape(filtered_avg_snr[nv_idx], len(freqs))
        # Replace NaNs/infs with 0 so the plot still shows
        snrs = np.nan_to_num(snrs, nan=0.0, posinf=0.0, neginf=0.0)

        ax_snr.plot(freqs, snrs, linewidth=1, label=f"NV{nv_idx}")

    ax_snr.set_xlabel("Microwave Frequency (MHz)", fontsize=15)
    ax_snr.set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=15)
    ax_snr.set_title(f"SNR vs Frequency for {num_nvs} NVs", fontsize=15)

    ax_snr.grid(True, linestyle="--", alpha=0.6)
    ax_snr.tick_params(axis="both", labelsize=14)

    # Plot histograms and scatter plots
    plots_data = [
        (
            "Histogram of Frequency Splitting",
            filtered_freq_differences,
            None,
            "Freq Splitting (GHz)",
            "Count",
            "teal",
            "MHz",
        ),
        (
            "ESR Freq Splitting vs Average Peak Width",
            filtered_freq_differences,
            filtered_avg_peak_widths,
            "Freq Splitting (GHz)",
            "Average Peak Width",
            "orange",
            "MHz",
        ),
        (
            "ESR Peak Freqs vs Avg Peak Amps",
            filtered_freq_differences,
            filtered_avg_peak_amplitudes,
            "Freq Splitting (GHz)",
            "Avg Peak Amp",
            "green",
            "arb. unit",
        ),
        (
            "NV Index vs Freq. Splitting.",
            list(range(len(filtered_freq_differences))),  # NV indices
            filtered_freq_differences,
            "NV Index",
            "Freq. Splitting (GHz)",
            "blue",
            "GHz",
        ),
    ]

    for title, x_data, y_data, xlabel, ylabel, color, unit in plots_data:
        kpl.init_kplotlib()
        plt.figure()
        if isinstance(y_data, list):
            plt.scatter(x_data, y_data, color=color, alpha=0.7, edgecolors="k")
        else:
            plt.hist(x_data, bins=9, color=color, alpha=0.7, edgecolor="black")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        kpl.show()
        # dm.save_figure(fig, file_path)
        # plt.close(fig)

    return
    # Plot filtered resonance fits
    sns.set(style="whitegrid", palette="muted")
    num_filtered_nvs = len(filtered_nv_list)
    colors = sns.color_palette("deep", num_filtered_nvs)
    num_rows = int(np.ceil(num_filtered_nvs / num_cols))
    fig_fitting, axes_fitting = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 2, num_rows * 1),
        sharex=True,
        sharey=False,
        # constrained_layout=True,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0},
    )
    axes_fitting = axes_fitting.flatten()

    for nv_idx, ax in enumerate(axes_fitting):
        if nv_idx < num_filtered_nvs:
            sns.lineplot(
                x=freqs,
                y=filtered_avg_counts[nv_idx],
                ax=ax,
                color=colors[nv_idx % len(colors)],
                lw=0,
                marker="o",
                markersize=2,
                label=f"{filtered_indices[nv_idx]}",
            )
            ax.legend(fontsize="xx-small")
            ax.errorbar(
                freqs,
                filtered_avg_counts[nv_idx],
                # yerr=filtered_avg_counts_ste[nv_idx],
                yerr=np.abs(filtered_avg_counts_ste[nv_idx]),
                fmt="none",
                ecolor="gray",
                alpha=0.6,
            )
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            # Plot fitted data on the same subplot
            ax.plot(
                freqs_dense,
                filtered_fitted_data[nv_idx],
                "-",
                color=colors[nv_idx % len(colors)],
                label="Fit",
                lw=1,
            )
            # Y-tick labels for the leftmost column
            # if nv_idx % num_cols == 0:
            #     ax.set_yticks(ax.get_yticks())
            # else:
            #     ax.set_yticklabels([])
            ax.set_yticklabels([])
            # Set custom tick locations in x axis
            if nv_idx >= (num_rows - 1) * num_cols:  # Bottom row
                ax.set_xlabel("Frequency (GHz)")
                ax.set_xticks(np.linspace(min(freqs), max(freqs), 5))
            for col in range(num_cols):
                bottom_row_idx = num_rows * num_cols - num_cols + col
                if bottom_row_idx < len(axes_fitting):
                    ax = axes_fitting[bottom_row_idx]
                    tick_positions = np.linspace(min(freqs), max(freqs), 5)
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(
                        [f"{tick:.2f}" for tick in tick_positions],
                        rotation=45,
                        fontsize=9,
                    )
                    ax.set_xlabel("Frequency (GHz)")
                else:
                    ax.set_xticklabels([])

            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            ax.axis("off")
    
    fig_fitting.tight_layout(pad=0.1, w_pad=0.0, h_pad=0.0)
    fig_fitting.text(
        -0.005,
        0.5,
        "NV$^{-}$ Population",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    # Optional outer labels/title (won't add gaps between subplots)
    # file_name = dm.get_file_name(file_id=file_id)
    plt.subplots_adjust(top=0.98, wspace=0.0, hspace=0.0)
    fig_fitting.suptitle(f"ESR {file_id}", y=0.995,fontsize=11)
    # plt.tight_layout()
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")
    # # file_path = dm.get_file_path(__file__, file_name, f"{file_id}_{date_time_str}")
    # plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.0, wspace=0.0)
    plt.show()
    # dm.save_figure(fig_fitting, file_path)
    # plt.close(fig_fitting)
    # return

def classify_nv_by_ms_minus_targets(center_freqs,
                                    targets_ghz=(2.76, 2.78, 2.82, 2.84),
                                    tol_mhz=6.0,
                                    peak_amps=None):
    """
    Classify NV indices into 4 orientation bins based on which ms=-1 target line
    (2.76, 2.78, 2.82, 2.84 GHz) their *clear* peak is closest to.

    Args
    ----
    center_freqs : list of (f1, f2) per NV (floats). Can be in MHz (e.g., 2760) or GHz (2.76).
    targets_ghz : iterable of target ms=-1 frequencies in GHz.
    tol_mhz     : matching tolerance in MHz.
    peak_amps   : optional list of (amp1, amp2) per NV (higher is "clearer").
                  If None, proximity decides; if provided, proximity is tie-broken by amplitude.

    Returns
    -------
    result : dict with keys:
        - 'bins': {target_ghz: [nv_idx, ...]}
        - 'assignments': array of length N with the assigned target (GHz) or np.nan
        - 'which_peak': array of 'f1'/'f2'/None showing which peak matched
        - 'no_match': [nv_idx, ...]   # neither peak within tolerance
        - 'multi_match': [nv_idx, ...] # both peaks matched different targets (rare)
        - 'units': 'GHz' or 'MHz' indicating normalization used internally
    """
    # Normalize units: infer MHz vs GHz
    cf = np.array(center_freqs, dtype=float)
    # guard shape
    if cf.ndim != 2 or cf.shape[1] != 2:
        raise ValueError("center_freqs must be list/array of (f1, f2) pairs")

    # If values are like 2700–2900 => MHz; if ~2.7–2.9 => GHz
    units = "GHz"
    cf_max = np.nanmax(cf)
    if cf_max > 100:  # likely MHz
        cf = cf / 1000.0
        units = "MHz->GHz"

    targets = np.array(targets_ghz, dtype=float)
    tol_ghz = tol_mhz / 1000.0

    N = cf.shape[0]
    assignments = np.full(N, np.nan, dtype=float)  # target GHz or nan
    which_peak  = np.array([None]*N, dtype=object)

    # If amplitudes provided, use them to prefer the "clearer" peak on ties
    has_amps = peak_amps is not None
    if has_amps:
        pa = np.array(peak_amps, dtype=float)
        if pa.shape != cf.shape:
            raise ValueError("peak_amps must be same shape as center_freqs (N x 2)")
    else:
        pa = np.zeros_like(cf)

    bins = {float(t): [] for t in targets}
    no_match = []
    multi_match = []

    for i in range(N):
        f1, f2 = cf[i, 0], cf[i, 1]
        if not (np.isfinite(f1) and np.isfinite(f2)):
            no_match.append(i)
            continue

        # distances to targets
        d1 = np.abs(targets - f1)
        d2 = np.abs(targets - f2)
        # candidates within tolerance
        cand1 = np.where(d1 <= tol_ghz)[0]
        cand2 = np.where(d2 <= tol_ghz)[0]

        # Decide which peak is the ms=-1 representative for this NV
        picked = None
        picked_peak = None

        if len(cand1) == 0 and len(cand2) == 0:
            # neither peak close to any target
            no_match.append(i)
        elif len(cand1) > 0 and len(cand2) == 0:
            # only f1 matches: choose closest target
            j = cand1[np.argmin(d1[cand1])]
            picked = targets[j]
            picked_peak = 'f1'
        elif len(cand2) > 0 and len(cand1) == 0:
            # only f2 matches
            j = cand2[np.argmin(d2[cand2])]
            picked = targets[j]
            picked_peak = 'f2'
        else:
            # both peaks have matches (rare). Prefer the closest-in-frequency;
            # on tie, prefer higher amplitude if provided.
            j1 = cand1[np.argmin(d1[cand1])]
            j2 = cand2[np.argmin(d2[cand2])]
            if d1[j1] < d2[j2]:
                picked, picked_peak = targets[j1], 'f1'
            elif d2[j2] < d1[j1]:
                picked, picked_peak = targets[j2], 'f2'
            else:
                # tie in proximity; use amplitude if available
                if has_amps and pa[i, 0] != pa[i, 1]:
                    if pa[i, 0] > pa[i, 1]:
                        picked, picked_peak = targets[j1], 'f1'
                    else:
                        picked, picked_peak = targets[j2], 'f2'
                else:
                    # still tied → mark multi and pick arbitrarily the lower-frequency peak
                    multi_match.append(i)
                    if f1 <= f2:
                        picked, picked_peak = targets[j1], 'f1'
                    else:
                        picked, picked_peak = targets[j2], 'f2'

        if picked is not None:
            assignments[i] = float(picked)
            which_peak[i] = picked_peak
            bins[float(picked)].append(i)

    return {
        'bins': bins,
        'assignments': assignments,   # in GHz
        'which_peak': which_peak,     # 'f1'/'f2'/None
        'no_match': no_match,
        'multi_match': multi_match,
        'units': units,
    }


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
    
    #====================johnson sample mounts=========================
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
 
    
    # Load the first dataset as a base
    combined_data = dm.get_raw_data(
        file_stem=file_ids[0], load_npz=True, use_cache=True
    )
    combined_sig_counts = None
    combined_ref_counts = None

    if combined_data:
        nv_list = combined_data["nv_list"]
        freqs = combined_data["freqs"]
        print(len(freqs))
        num_steps = combined_data["num_steps"]
        num_reps = combined_data["num_reps"]
        num_runs = combined_data["num_runs"]
        counts = np.array(combined_data["counts"])[0]
        adj_num_steps = num_steps // 4
        sig_counts_0 = counts[:, :, 0:adj_num_steps, :]
        sig_counts_1 = counts[:, :, adj_num_steps : 2 * adj_num_steps, :]
        combined_sig_counts = np.append(sig_counts_0, sig_counts_1, axis=3)

        ref_counts_0 = counts[:, :, 2 * adj_num_steps : 3 * adj_num_steps, :]
        ref_counts_1 = counts[:, :, 3 * adj_num_steps :, :]
        combined_ref_counts = np.empty(
            (len(nv_list), num_runs, adj_num_steps, 2 * num_reps)
        )
        combined_ref_counts[:, :, :, 0::2] = ref_counts_0
        combined_ref_counts[:, :, :, 1::2] = ref_counts_1

        # Process remaining files
        for file_id in file_ids[1:]:
            try:
                print(f"Processing file: {file_id}")
                new_data = dm.get_raw_data(
                    file_id=file_id, load_npz=False, use_cache=True
                )
                if not new_data:
                    print(f"Skipping file {file_id}: Data not found.")
                    continue

                new_counts = np.array(new_data["counts"])[0]

                new_sig_counts_0 = new_counts[:, :, 0:adj_num_steps, :]
                new_sig_counts_1 = new_counts[
                    :, :, adj_num_steps : 2 * adj_num_steps, :
                ]
                new_sig_counts = np.append(new_sig_counts_0, new_sig_counts_1, axis=3)

                new_ref_counts_0 = new_counts[
                    :, :, 2 * adj_num_steps : 3 * adj_num_steps, :
                ]
                new_ref_counts_1 = new_counts[:, :, 3 * adj_num_steps :, :]
                new_ref_counts = np.empty(
                    (len(nv_list), num_runs, adj_num_steps, 2 * num_reps)
                )
                new_ref_counts[:, :, :, 0::2] = new_ref_counts_0
                new_ref_counts[:, :, :, 1::2] = new_ref_counts_1

                # Append new data 
                combined_sig_counts = np.append(
                    combined_sig_counts, new_sig_counts, axis=1
                )
                combined_ref_counts = np.append(
                    combined_ref_counts, new_ref_counts, axis=1
                )
                combined_data["num_runs"] += new_data["num_runs"]

            except Exception as e:
                print(f"Error processing file {file_id}: {e}")

        # Generate unique filename
        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        combined_file_id = "_".join(map(str, file_ids))
        file_name = f"combined_{combined_file_id}_{date_time_str}.png"
        file_path = dm.get_file_path(__file__, "combined_plot", file_name)
        print(f"Combined plot saved to {file_path}")
        # Plot combined data
        plot_nv_resonance(
            nv_list,
            freqs,
            combined_sig_counts,
            combined_ref_counts,
            file_id=combined_file_id,
            num_cols=8,
        )
    else:
        print("No valid data available for plotting.")
 
    
    kpl.show(block=True)
