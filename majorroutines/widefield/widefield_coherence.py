# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
@author: sbchand
"""

import sys
import time
import traceback

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
    # Define cosine fitting function
    # def cos_func(phi, amp, phase_offset, offset):
    #     return amp * np.cos(phi - phase_offset) + offset

    def cos_func(phi_deg, amp, phase_offset_deg, offset):
        return amp * np.cos(np.radians(phi_deg) - np.radians(phase_offset_deg)) + offset

    num_nvs = len(nv_list)
    fit_results = {
        "amplitude": [],
        "phase_offset": [],
        "offset": [],
        "chi_sq": [],
    }

    phi_degrees = []

    fig_all, ax_all = plt.subplots(figsize=(6, 5))

    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]
        guess_params = [1.0, 0.0, 0.5]  # amp, phase_offset, offset

        try:
            popt, pcov = curve_fit(
                cos_func,
                phis,
                nv_counts,
                p0=guess_params,
                sigma=nv_counts_ste,
                absolute_sigma=True,
            )
            fit_success = True
        except Exception:
            popt = [np.nan, np.nan, np.nan]
            fit_success = False

        if fit_success:
            residuals = cos_func(phis, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            fit_results["chi_sq"].append(chi_sq / (len(nv_counts) - len(popt)))
        else:
            fit_results["chi_sq"].append(np.nan)

        fit_results["amplitude"].append(popt[0])
        fit_results["phase_offset"].append(popt[1])
        fit_results["offset"].append(popt[2])
        phi_degrees.append(popt[1])

        # Plot individual fits
        phi_fit = np.linspace(min(phis), max(phis), 200)
        fit_vals = cos_func(phi_fit, *popt)
        ax_all.errorbar(
            phis,
            nv_counts,
            yerr=abs(nv_counts_ste),
            fmt="o",
            alpha=0.3,
            label=f"NV {nv_ind}",
        )
        if fit_success:
            ax_all.plot(phi_fit, fit_vals, "--", alpha=0.5)

    ax_all.set_xlabel("Phase (degrees)")
    ax_all.set_ylabel("Normalized Counts")
    ax_all.set_title(r"Cosine Fits ($\frac{\pi}{2}_x$ - xy8 – $\frac{\pi}{2}_\phi$)")
    # ax_all.set_title(r"Cosine Fits")
    ax_all.grid(True)
    ax_all.spines["right"].set_visible(False)
    ax_all.spines["top"].set_visible(False)

    # Plot and fit median across all NVs
    median_counts = np.median(norm_counts, axis=0)
    median_ste = np.median(norm_counts_ste, axis=0)

    try:
        popt_median, _ = curve_fit(
            cos_func,
            phis,
            median_counts,
            p0=[1.0, 0.0, 0.5],
            sigma=median_ste,
            absolute_sigma=True,
        )
        fit_median = True
    except Exception:
        popt_median = [0, 0, 0]
        fit_median = False

    fig_median, ax_median = plt.subplots(figsize=(6, 5))
    ax_median.errorbar(phis, median_counts, yerr=median_ste, fmt="o", label="Median NV")
    if fit_median:
        phi_fit = np.linspace(min(phis), max(phis), 200)
        ax_median.plot(phi_fit, cos_func(phi_fit, *popt_median), label="Fit")
        # ax_median.set_title(f"Median Fit  : phase offset ≈ {popt_median[1]:.1f}°")
        # ax_median.set_title(r"Median Fringe: $\pi/2_x$ – $\pi_x$ – $\pi/2_\phi$")
        ax_median.set_title(r"Median Fringe: $\pi/2_x$ – XY8 – $\pi/2_\phi$")

    ax_median.set_xlabel("Phase, $\phi$ (degrees)")
    ax_median.set_ylabel("Median Normalized Counts")
    ax_median.legend()
    ax_median.grid(True)
    ax_median.spines["right"].set_visible(False)
    ax_median.spines["top"].set_visible(False)

    # Scatter plots of fitted parameters
    fit_df = pd.DataFrame(fit_results)

    fig_params, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    axs[0].scatter(range(num_nvs), fit_df["amplitude"])
    axs[0].set_ylabel("Amplitude")

    axs[1].scatter(range(num_nvs), fit_df["phase_offset"])
    axs[1].set_ylabel("Phase Offset (deg)")

    axs[2].scatter(range(num_nvs), fit_df["offset"])
    axs[2].set_ylabel("Offset")
    axs[2].set_xlabel("NV Index")

    for ax in axs:
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    # fig_params.suptitle(
    #     r"Fitted Parameters: $\frac{\pi}{2}_x$ - xy8 – $\frac{\pi}{2}_\phi$"
    # )
    plt.tight_layout()
    plt.show()


# def create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste):
#     # fit function
#     def cos_func(phi, amp, phase_offset, offset):
#         return amp * np.cos(phi - phase_offset) + offset

#     num_nvs = len(nv_list)
#     fit_fns = []
#     popts = []
#     phi_degrees = []
#     for nv_ind in range(num_nvs):
#         nv_counts = norm_counts[nv_ind]
#         nv_counts_ste = norm_counts_ste[nv_ind]
#         guess_params = [1.0, 0.0, 0.5]  # amp, phase_offset, baseline offset

#         try:
#             popt, _ = curve_fit(
#                 cos_func,
#                 phis,
#                 nv_counts,
#                 p0=guess_params,
#                 sigma=nv_counts_ste,
#                 absolute_sigma=True,
#             )
#         except Exception:
#             popt = None

#         fit_fns.append(cos_func if popt is not None else None)
#         popts.append(popt)

#         # Create new figure for this NV
#         fig, ax = plt.subplots(figsize=(6, 5))

#         # Plot data points
#         ax.errorbar(
#             phis,
#             nv_counts,
#             yerr=abs(nv_counts_ste),
#             fmt="o",
#             label=f"NV {nv_ind}",
#             capsize=3,
#         )

#         # Plot fit if successful
#         if popt is not None:
#             phi_fit = np.linspace(min(phis), max(phis), 200)
#             fit_vals = cos_func(phi_fit, *popt)
#             ax.plot(phi_fit, fit_vals, "-", label="Fit")
#             residuals = cos_func(phis, *popt) - nv_counts
#             chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
#             red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
#             # print(f"NV {nv_ind} - Reduced chi²: {red_chi_sq:.3f}")
#             peak = popt[0]  # phase_offset
#             peak_phi = popt[1]  # phase_offset
#             offset = popt[2]  # cpitms
#             # phi_degree = np.degrees(peak_phi)
#             phi_degree = peak_phi
#             phi_degrees.append(phi_degree)
#             print(
#                 f"Peak Amp = {peak:.2f} Peak occurs at φ ≈ {peak_phi:.2f} rad ≈ {np.degrees(peak_phi):.1f}°, offset = {offset:2f}"
#             )
#     # phi_degrees = np.degrees(phi_degrees)
#     mean_phase_offset = np.mean(phi_degrees)
#     # Suggest correction
#     correction_angle = -mean_phase_offset

#     print(f"\nAverage Phase Offset = {mean_phase_offset:.2f}°")
#     print(f"Suggested Phase Correction = {correction_angle:.2f}°")

#     good_offsets = [phi for phi in phi_degrees if abs(phi) < 0.3]
#     avg_phase_offset = np.mean(good_offsets)
#     print(f"Suggested global IQ phase correction: {np.degrees(avg_phase_offset):.1f}°")
#     ax.set_xlabel("Phase (rad)")
#     ax.set_ylabel("Normalized Counts")
#     ax.set_title(f"Cosine Fit for NV {nv_ind}")
#     # plt.title(f"Fit: A={A_fit:.2f}, δ={np.rad2deg(delta_fit):.1f}°, C={C_fit:.2f}")
#     ax.legend()
#     ax.grid(True)
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     plt.tight_layout()
#     plt.show(block=True)
#     # A_fit, delta_fit, C_fit = popt

#     # plt.plot(phis, signal, "o", label="Data")
#     # plt.plot(phis, cos_func(phis, *popt), "-", label="Fit")
#     # plt.xlabel("Phase φ (deg)")
#     # plt.ylabel("Signal")
#     # plt.title(f"Fit: A={A_fit:.2f}, δ={np.rad2deg(delta_fit):.1f}°, C={C_fit:.2f}")
#     # plt.legend()
#     # plt.show()


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    phi_list,
    evol_time,
    seq_type,
    uwave_ind_list,
):
    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "widefield_coherence.py"

    # ---- minimal safety: map 360 -> 0, keep ints ----
    phi_list = [int(ph) % 360 for ph in phi_list]
    num_steps = len(phi_list)

    def run_fn(shuffled_step_inds):
        step_vals = [phi_list[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            step_vals,
            evol_time,
            seq_type,
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
        load_iq=True,
    )

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "phis": phi_list,
        "phi-units": "deg",  # <-- FIX
        "evol_time": evol_time,
        "evol_time-unit": "ns",
        "seq_type": seq_type,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_sig.name)
    dm.save_raw_data(raw_data, file_path)

    tb.reset_cfm()
    kpl.show()

    # if raw_fig is not None:
    #     dm.save_figure(raw_fig, file_path)
    # if fit_fig is not None:
    #     file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
    #     dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1817334208399
    file_id = 1825020210830  #
    # file_id = 1825070485845
    file_id = 1837498410890  # file_name = "2025_04_04-17_38_13-rubin-nv0_2025_02_26"

    # data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=True)
    data = dm.get_raw_data(
        # file_stem="2025_04_04-17_38_13-rubin-nv0_2025_02_26", #spin echo old
        # file_stem="2025_04_29-21_46_40-rubin-nv0_2025_02_26",  # spin echo
        # file_stem="2025_04_30-00_36_54-rubin-nv0_2025_02_26",  # ramsey
        # file_stem="2025_04_30-12_43_15-rubin-nv0_2025_02_26",  # xy8
        # file_stem="2025_10_10-21_51_14-rubin-nv0_2025_09_08",  # xy8
        # file_stem="2025_10_11-00_03_47-rubin-nv0_2025_09_08",  # spin echo
        # file_stem="2025_10_13-14_00_31-rubin-nv0_2025_09_08",  # xy8
        # file_stem="2026_01_05-18_44_14-johnson-nv0_2025_10_21",  # xy8
        file_stem= "2026_01_05-21_44_23-johnson-nv0_2025_10_21",
        load_npz=True,
        use_cache=True,
    )
    print(data.keys())
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    phis = data["phis_rad"]
    phis = np.degrees(phis)

    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]
    
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )
    num_nvs = len(nv_list)
    phi_step = phis[1] - phis[0]
    num_steps = len(phis)
    fit_fig = create_fit_figure(nv_list, phis, norm_counts, norm_counts_ste)

    kpl.show(block=True)
