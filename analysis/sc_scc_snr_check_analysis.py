# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR

Created on Fall, 2024

@author: Saroj Chand
"""

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.constants import (
    CollectionMode,
    CoordsKey,
    NVSig,
    PosControlMode,
    VirtualLaserKey,
)
# import seaborn as sns
# import pandas as pd
# def process_and_plot(data, error_threshold=0.2):
#     threshold = True
#     nv_list = data["nv_list"]
#     counts = np.array(data["counts"])
#     sig_counts = counts[0]
#     ref_counts = counts[1]
#     # Apply threshold if needed
#     if threshold:
#         thresh_method = "otsu"
#         sig_counts, ref_counts = widefield.threshold_counts(
#             nv_list, sig_counts, ref_counts, method=thresh_method
#         )
#     # Report the results and return
#     avg_sig_counts, avg_sig_counts_ste, norms = widefield.average_counts(
#         sig_counts, ref_counts
#     )
#     norms_ms0_newaxis = norms[0][:, np.newaxis]
#     norms_ms1_newaxis = norms[1][:, np.newaxis]
#     contrast = norms_ms1_newaxis - norms_ms0_newaxis
#     norm_counts = (avg_sig_counts - norms_ms0_newaxis) / contrast
#     norm_counts_ste = avg_sig_counts_ste / contrast
#     # Ensure no negative yerr values
#     norm_counts_ste = np.abs(norm_counts_ste)
#     # Constrain norm_counts to be within [0, 1]
#     norm_counts_clipped = np.clip(norm_counts, 0, 1)
#     ### Plot 1: All Data
#     # Prepare data for seaborn plotting (with all data points)
#     all_nv_nums = [widefield.get_nv_num(nv) for nv in nv_list]
#     all_plot_data = pd.DataFrame(
#         {
#             "NV": all_nv_nums,
#             "Contrast": norm_counts_clipped.flatten(),
#             "Error": norm_counts_ste.flatten(),
#         }
#     )
#     # Set up the first plot with all data points
#     plt.figure(figsize=(15, 8))  # Adjust size for large numbers of NVs
#     sns.set(style="whitegrid")
#     ax_all = sns.barplot(x="NV", y="Contrast", data=all_plot_data, ci=None)
#     # Add error bars manually
#     for i, row in all_plot_data.iterrows():
#         ax_all.errorbar(
#             row["NV"], row["Contrast"], yerr=row["Error"], fmt="none", c="black"
#         )
#     # Customize plot
#     ax_all.set_xlabel("NV Index")
#     ax_all.set_ylabel("Normalized Contrast (0 to 1)")
#     ax_all.set_title("All NV Data")
#     plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
#     plt.tight_layout()
#     ### Plot 2: Filtered Good Data (Contrast between 0-1 and small error bars)
#     filtered_nv_list = []
#     filtered_norm_counts = []
#     filtered_norm_counts_ste = []
#     for i in range(len(norm_counts)):
#         if 0 <= norm_counts[i] <= 1 and norm_counts_ste[i] < error_threshold:
#             filtered_nv_list.append(widefield.get_nv_num(nv_list[i]))
#             filtered_norm_counts.append(norm_counts[i])
#             filtered_norm_counts_ste.append(norm_counts_ste[i])
#     # Prepare data for seaborn plotting (good data points only)
#     good_plot_data = pd.DataFrame(
#         {
#             "NV": filtered_nv_list,
#             "Contrast": np.array(filtered_norm_counts).flatten(),
#             "Error": np.array(filtered_norm_counts_ste).flatten(),
#         }
#     )
#     # Set up the second plot with good data points
#     plt.figure(figsize=(15, 8))  # Adjust size for large numbers of NVs
#     sns.set(style="whitegrid")
#     ax_good = sns.barplot(x="NV", y="Contrast", data=good_plot_data, ci=None)
#     # Add error bars manually for good data points
#     for i, row in good_plot_data.iterrows():
#         ax_good.errorbar(
#             row["NV"], row["Contrast"], yerr=row["Error"], fmt="none", c="black"
#         )
#     # Customize plot
#     ax_good.set_xlabel("NV Index")
#     ax_good.set_ylabel("Normalized Contrast (0 to 1)")
#     ax_good.set_title(f"Good NV Data (Error < {error_threshold})")
#     plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
#     plt.tight_layout()
#     # Show both plots
#     plt.show()
#     print(f"Mean normalized contrast (all data): {np.mean(norm_counts)}")
#     print(f"Mean normalized contrast (good data): {np.mean(filtered_norm_counts)}")
#     return
import seaborn as sns
from matplotlib import pyplot as plt

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield

# def process_and_plot(data):
#     threshold = True
#     nv_list = data["nv_list"]
#     num_nvs = len(nv_list)
#     counts = np.array(data["counts"])
#     sig_counts = counts[0]
#     ref_counts = counts[1]

#     # Apply thresholds
#     if threshold:
#         sig_counts, ref_counts = widefield.threshold_counts(
#             nv_list, sig_counts, ref_counts, dynamic_thresh=False
#         )

#     # Calculate metrics
#     avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
#     avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)
#     avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
#     avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)

#     # Extract single step values
#     step_ind = 0
#     avg_sig_counts = avg_sig_counts[:, step_ind]
#     avg_ref_counts = avg_ref_counts[:, step_ind]
#     avg_snr = avg_snr[:, step_ind]
#     avg_contrast = avg_contrast[:, step_ind]

#     # Compute distances
#     coords_key = "laser_COBO_638_aod"
#     distances = []
#     for nv in nv_list:
#         coords = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
#         dist = np.sqrt((90 - coords[0]) ** 2 + (90 - coords[1]) ** 2)
#         distances.append(dist)

#     # Prepare DataFrame for analysis
#     df = pd.DataFrame(
#         {
#             "NV Index": range(num_nvs),
#             "Signal Counts": avg_sig_counts,
#             "Reference Counts": avg_ref_counts,
#             "SNR": avg_snr,
#             "Contrast": avg_contrast,
#             "Distance": distances,
#         }
#     )

#     # Plot: Signal vs. Reference Counts
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(
#         data=df,
#         x="Reference Counts",
#         y="Signal Counts",
#         hue="SNR",
#         size="Distance",
#         sizes=(50, 200),
#     )
#     plt.title("Signal vs. Reference Counts")
#     plt.xlabel("Reference Counts")
#     plt.ylabel("Signal Counts")
#     plt.legend(title="SNR (Color) & Distance (Size)")
#     plt.grid(True)
#     plt.show()

#     # Plot: SNR Distribution
#     plt.figure(figsize=(8, 6))
#     sns.histplot(df["SNR"], kde=True, bins=15, color="blue", edgecolor="black")
#     plt.title("SNR Distribution")
#     plt.xlabel("SNR")
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.show()

#     # Plot: SNR vs. Distance
#     plt.figure(figsize=(8, 6))
#     sns.regplot(
#         data=df,
#         x="Distance",
#         y="SNR",
#         scatter_kws={"s": 100, "alpha": 0.7},
#         line_kws={"color": "red"},
#     )
#     plt.title("SNR vs. Distance")
#     plt.xlabel("Distance from Center (MHz)")
#     plt.ylabel("SNR")
#     plt.grid(True)
#     plt.show()

#     # Heatmap: SNR by NV Index and Distance
#     pivot_table = df.pivot_table(
#         values="SNR", index="NV Index", columns="Distance", aggfunc="mean"
#     )
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "SNR"}
#     )
#     plt.title("SNR Heatmap (NV Index vs Distance)")
#     plt.xlabel("Distance (MHz)")
#     plt.ylabel("NV Index")
#     plt.show()

#     return df


def process_and_plot(data):
    threshold = True
    print(data.keys())
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    # Apply thresholds
    if threshold:
        sig_counts, ref_counts = widefield.threshold_counts(
            nv_list, sig_counts, ref_counts, dynamic_thresh=False
        )

    # Calculate metrics
    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)

    # Extract single step values
    step_ind = 0
    avg_sig_counts = avg_sig_counts[:, step_ind]
    avg_sig_counts_ste = avg_sig_counts_ste[:, step_ind]
    avg_ref_counts = avg_ref_counts[:, step_ind]
    avg_ref_counts_ste = avg_ref_counts_ste[:, step_ind]
    avg_snr = avg_snr[:, step_ind]
    avg_snr_ste = avg_snr_ste[:, step_ind]
    avg_contrast = avg_contrast[:, step_ind]
    avg_contrast_ste = avg_contrast_ste[:, step_ind]

    # Get NV coordinates and Compute distances
    nv_coords = []
    distances = []
    distances = []
    scc_durations = []
    for nv in nv_list:
        coords = pos.get_nv_coords(nv, coords_key= CoordsKey.PIXEL, drift_adjust=False)
        nv_coords.append(coords)
        dist = round(np.sqrt((125 - coords[0]) ** 2 + (125 - coords[1]) ** 2), 3)
        distances.append(dist)
        # try enum key, then string fallback
        scc_dur = pos.get_nv_pulse_duration(nv, VirtualLaserKey.SCC)
        scc_durations.append(scc_dur)

    yellow_charge_readout_amp = data["opx_config"]["waveforms"]["yellow_charge_readout"]["sample"]
    yellow_spin_pol_amp = data["opx_config"]["waveforms"]["yellow_spin_pol"]["sample"]
    a, b, c = 1.5133e04, 2.6976, -38.63
    yellow_charge_readout_amp = int(a * (yellow_charge_readout_amp**b) + c)
    yellow_spin_pol_amp = int(a * (yellow_spin_pol_amp**b) + c)
    # Prepare DataFrame for analysis
    df = pd.DataFrame(
        {
            "NV Index": range(num_nvs),
            "Signal Counts": avg_sig_counts,
            "Signal STE": avg_sig_counts_ste,
            "Reference Counts": avg_ref_counts,
            "Reference STE": avg_ref_counts_ste,
            "SNR": avg_snr,
            "SNR STE": avg_snr_ste,
            "Contrast": avg_contrast,
            "Contrast STE": avg_contrast_ste,
            "Distance": distances,
            "scc_durations": scc_durations,
            "Y Coord": [coord[0] for coord in nv_coords],
            "X Coord": [coord[1] for coord in nv_coords],
        }
    )

    # # Plot: SNR and Contrast in NV Coordinate Space
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # # SNR plot
    # scatter = axes[0].scatter(
    #     df["X Coord"],
    #     df["Y Coord"],
    #     c=df["SNR"],
    #     cmap="coolwarm",
    #     s=20,
    #     edgecolor="black",
    # )
    # axes[0].set_title("SNR in RED AOD Coordinate Space")
    # axes[0].set_xlabel("X Coord")
    # axes[0].set_ylabel("Y Coord")
    # plt.colorbar(scatter, ax=axes[0], label="SNR")

    # # Contrast plot
    # scatter = axes[1].scatter(
    #     df["X Coord"],
    #     df["Y Coord"],
    #     c=df["Contrast"],
    #     cmap="coolwarm",
    #     s=20,
    #     edgecolor="black",
    # )
    # axes[1].set_title("Contrast in RED AOD Coordinate Space")
    # axes[1].set_xlabel("X Coord")
    # axes[1].set_ylabel("Y Coord")
    # plt.colorbar(scatter, ax=axes[1], label="Contrast")

    # # plt.tight_layout()
    # plt.show()

    # # Plot: Signal vs. Reference Counts with error bars
    # plt.figure(figsize=(8, 6))
    # plt.errorbar(
    #     df["Reference Counts"],
    #     df["Signal Counts"],
    #     xerr=df["Reference STE"],
    #     yerr=df["Signal STE"],
    #     fmt="o",
    #     ecolor="gray",
    #     capsize=3,
    #     label="NV Data",
    # )
    # plt.title("Signal vs. Reference Counts with Error Bars")
    # plt.xlabel("Reference Counts")
    # plt.ylabel("Signal Counts")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # # Plot: SNR Distribution
    # plt.figure(figsize=(6, 5))
    # sns.histplot(df["SNR"], kde=True, bins=15, color="blue", edgecolor="black")
    # plt.title("SNR Distribution")
    # plt.xlabel("SNR")
    # plt.ylabel("Frequency")
    # plt.grid(True)
    # plt.show()

    # Plot: SNR vs. Distance with error bars
    distance = df["Distance"]
    scc_durations = df["scc_durations"]
    snr = df["SNR"]
    yerr = df["SNR STE"]
    # indices_to_remove = [ind for ind in range(len(snr)) if snr[ind] < 0.05]
    indices_to_remove = []
    print(indices_to_remove)
    selected_indices = [ind for ind in range(num_nvs) if ind not in indices_to_remove]
    distance = [distance[ind] for ind in selected_indices]
    scc_durations = [scc_durations[ind] for ind in selected_indices]
    snr = [round(snr[ind], 3) for ind in selected_indices]
    yerr = [yerr[ind] for ind in selected_indices]
    median = round(np.median(snr), 3)
    print(f"scc_snrs:{snr}")
    plt.figure(figsize=(6, 5))
    plt.errorbar(
        scc_durations,
        snr,
        yerr,
        fmt="o",
        ecolor="gray",
        capsize=3,
        label=f"SNR (Median: {median})",
    )
    plt.title(f"SNRs of {num_nvs}NVs(readout amp:{yellow_charge_readout_amp}uW, spin pol amp:{yellow_spin_pol_amp}uW)", fontsize=13)
    plt.xlabel("SCC Durations (ns)", fontsize=15)
    plt.ylabel("SNR", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=11)
    plt.show()

    return df


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(
        file_stem="2026_01_15-19_26_11-johnson-nv0_2025_10_21", load_npz=True
    )
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}")
    # Process and visualize
    df = process_and_plot(data)
    # Save DataFrame if needed
    # df.to_csv("processed_nv_data.csv", index=False)
    kpl.show(block=True)
