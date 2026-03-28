# -*- coding: utf-8 -*-
"""
Widefield charge-state histogram acquisition + fidelity extraction.

Illuminates an area and records camera counts for many NVs while interleaving:
  (1) signal: with ionization pulse
  (2) reference: without ionization pulse
Then (optionally) plots per-NV histograms, fits the reference distribution with a
bimodal model (NV⁰ / NV⁻), extracts an optimal threshold, and reports readout &
preparation fidelities (plus ionization probability).

Created: Fall 2023 (M. Cambria)
Updated: Fall 2025 (Saroj Chand)
"""


import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
    analyze_charge_histogram_multinv_binomial,
)
from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey


# region Process and plotting functions
def plot_histograms(
    sig_counts_list, ref_counts_list, no_title=True, ax=None, density=False
):

    ### Histograms
    num_reps = len(ref_counts_list)
    labels = ["With ionization pulse", "Without ionization pulse"]
    colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
    counts_lists = [sig_counts_list, ref_counts_list]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    if not no_title:
        ax.set_title(f"Charge prep hist, {num_reps} reps")
    ax.set_xlabel("Integrated counts")
    if density:
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("Number of occurrences")

    for ind in range(2):
        if counts_lists is None or len(counts_lists) == 0:
            continue
        counts_list = counts_lists[ind]
        label = labels[ind]
        color = colors[ind]
        # kpl.histogram(ax, counts_list, label=label, color=color, density=density)  # MCC
        kpl.histogram(ax, counts_list, color=color, density=density)

    # ax.legend() # MCC
    # ax.tick_params(axis="y", rotation=90)
    ax.set_xlim(-0.5, None)

    if fig is not None:
        return fig

def process_and_plot(
    raw_data,
    do_plot_histograms=False,
    # prob_dist: ProbDist = ProbDist.COMPOUND_POISSON_WITH_IONIZATION,
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
):
    ### Setup

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])

    sig_counts_lists = [np.asarray(counts[0, nv_ind].flatten()) for nv_ind in range(num_nvs)]
    ref_counts_lists = [np.asarray(counts[1, nv_ind].flatten()) for nv_ind in range(num_nvs)]

    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    # Robust stem for analyzed output file name
    base_file_stem = raw_data.get("file_stem") or raw_data.get("file_name") or "raw_data"
    if isinstance(base_file_stem, (list, tuple)):
        base_file_stem = "_".join(map(str, base_file_stem))
    base_file_stem = str(base_file_stem).replace(" ", "_")

    ### Histograms and thresholding

    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []
    ion_prob_list = []
    red_chi_sq_list = []

    # Save-all containers for later histogram-only plotting from analyzed file
    fit_params_list = []
    fit_success_list = []
    nv_num_list = []

    hist_figs = []

    for ind in range(num_nvs):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Only use ref counts for threshold determination
        popt, _, red_chi_sq = fit_bimodal_histogram(
            ref_counts_list, prob_dist, no_print=False
        )

        if popt is not None:
            threshold, readout_fidelity = determine_threshold(
                popt, prob_dist, dark_mode_weight=0.5, do_print=True, ret_fidelity=True
            )
            prep_fidelity = 1 - popt[0]
            ion_prob = popt[-1]
            fit_success = True
            fit_params_to_save = np.asarray(popt, dtype=float)
        else:
            threshold = np.nan
            readout_fidelity = np.nan
            prep_fidelity = np.nan
            ion_prob = np.nan
            fit_success = False
            fit_params_to_save = np.array([], dtype=float)

        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        prep_fidelity_list.append(prep_fidelity)
        red_chi_sq_list.append(red_chi_sq)
        ion_prob_list.append(ion_prob)
        fit_params_list.append(fit_params_to_save)
        fit_success_list.append(fit_success)

        nv_num = widefield.get_nv_num(nv_list[ind])
        nv_num_list.append(nv_num)

        # Plot histograms with NV index and fidelity included
        if do_plot_histograms:
            fig = plot_histograms(sig_counts_list, ref_counts_list, density=True)
            ax = fig.gca()

            # Ref-count fit line
            if popt is not None:
                x_vals = np.linspace(0, np.max(ref_counts_list), 1000)

                single_mode_num_params = bimodal_histogram.get_single_mode_num_params(
                    prob_dist
                )
                single_mode_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)

                dark_mode_line = popt[0] * single_mode_pdf(
                    x_vals, *popt[1 : 1 + single_mode_num_params]
                )
                bright_mode_line = (1 - popt[0]) * single_mode_pdf(
                    x_vals, *popt[1 + single_mode_num_params :]
                )

                bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist)
                bimodal_line = bimodal_pdf(x_vals, *popt)

                kpl.plot_line(
                    ax,
                    x_vals,
                    dark_mode_line,
                    color=kpl.KplColors.RED,
                    label="NV$^{0}$ mode",
                )
                kpl.plot_line(
                    ax,
                    x_vals,
                    bright_mode_line,
                    color=kpl.KplColors.GREEN,
                    label="NV$^{-}$ mode",
                )
                kpl.plot_line(
                    ax,
                    x_vals,
                    bimodal_line,
                    color=kpl.KplColors.BLUE,
                    label="Combined",
                )
                ax.legend(loc=kpl.Loc.UPPER_RIGHT)

            # Threshold line
            if np.isfinite(threshold):
                ax.axvline(threshold, color=kpl.KplColors.GRAY, ls="dashed")

            info_str = (
                f"NV{nv_num}\n"
                f"Readout fidelity: {round(readout_fidelity, 3) if np.isfinite(readout_fidelity) else 'nan'}\n"
                f"Charge prep. fidelity: {round(prep_fidelity, 3) if np.isfinite(prep_fidelity) else 'nan'}"
            )
            kpl.anchored_text(ax, info_str, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)

            kpl.show(block=True)
            fig = None

            if fig is not None:
                hist_figs.append(fig)

    print(f"readout_fidelity_list: {readout_fidelity_list}")
    print(f"prep_fidelity_list: {prep_fidelity_list}")
    print(f"red_chi_sq_list: {red_chi_sq_list}")
    print(f"thresholds: {threshold_list}")

    # Convert to arrays
    # threshold_list = np.asarray(threshold_list, dtype=float)
    # readout_fidelity_list = np.asarray(readout_fidelity_list, dtype=float)
    # prep_fidelity_list = np.asarray(prep_fidelity_list, dtype=float)
    # red_chi_sq_list = np.asarray(red_chi_sq_list, dtype=float)
    # ion_prob_list = np.asarray(ion_prob_list, dtype=float)
    # fit_success_list = np.asarray(fit_success_list, dtype=bool)
    # nv_num_list = np.asarray(nv_num_list)

    # Store counts in object arrays so each NV's histogram data is preserved exactly
    # sig_counts_array = np.asarray(sig_counts_lists, dtype=object)
    # ref_counts_array = np.asarray(ref_counts_lists, dtype=object)
    # fit_params_array = np.asarray(fit_params_list, dtype=object)

    # Scatter readout vs prep fidelity
    fig, ax = plt.subplots()
    kpl.plot_points(ax, readout_fidelity_list, prep_fidelity_list)
    ax.set_xlabel("Readout fidelity")
    ax.set_ylabel("NV- preparation fidelity")

    # Report averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_list)
    std_readout_fidelity = np.nanstd(readout_fidelity_list)
    avg_prep_fidelity = np.nanmean(prep_fidelity_list)
    std_prep_fidelity = np.nanstd(prep_fidelity_list)

    str_readout_fidelity = tb.round_for_print(
        avg_readout_fidelity, std_readout_fidelity
    )
    print(f"Average readout fidelity: {str_readout_fidelity}")

    str_prep_fidelity = tb.round_for_print(avg_prep_fidelity, std_prep_fidelity)
    print(f"Average NV- preparation fidelity: {str_prep_fidelity}")

    avg_ion_prob = np.nanmean(ion_prob_list)
    print(f"Average ionization during readout probability: {round(avg_ion_prob, 6)}")

    var_ion_prob = np.nanvar(ion_prob_list)
    print(f"Variance ionization during readout probability: {round(var_ion_prob, 6)}")

    results = {
        # identity / indexing
        "nv_inds": np.arange(num_nvs),
        "nv_nums": nv_num_list,
        "num_nvs": num_nvs,

        # raw histogram data needed for later replotting
        "sig_counts_list": sig_counts_list,
        "ref_counts_list": ref_counts_list,

        # fit outputs
        "fit_params_list": fit_params_list,
        "fit_success_list": fit_success_list,
        "threshold_list": threshold_list,
        "prep_fidelity_list": prep_fidelity_list,
        "readout_fidelity_list": readout_fidelity_list,
        "red_chi_sq_list": red_chi_sq_list,
        "ion_prob_list": ion_prob_list,

        # acquisition metadata
        "num_reps": num_reps,
        "num_runs": num_runs,
        "num_shots": num_shots,
        "prob_dist_name": prob_dist.name if hasattr(prob_dist, "name") else str(prob_dist),

        # summaries
        "avg_readout_fidelity": avg_readout_fidelity,
        "std_readout_fidelity": std_readout_fidelity,
        "avg_prep_fidelity": avg_prep_fidelity,
        "std_prep_fidelity": std_prep_fidelity,
        "avg_ion_prob": avg_ion_prob,
        "var_ion_prob": var_ion_prob,
    }

    timestamp = dm.get_time_stamp()
    file_name = f"charge_state_analysis_hist_data_{base_file_stem}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)
    dm.save_raw_data(results, file_path)

def plot_histograms_from_analysis(analyzed_data, nv_indices=None, density=True):
    sig_counts_list_all = analyzed_data["sig_counts_list"]
    ref_counts_list_all = analyzed_data["ref_counts_list"]

    if nv_indices is None:
        nv_indices = range(len(sig_counts_list_all))

    figs = []

    prob_dist_name = analyzed_data.get("prob_dist_name", "COMPOUND_POISSON")
    prob_dist_local = getattr(ProbDist, prob_dist_name)

    single_mode_num_params = bimodal_histogram.get_single_mode_num_params(prob_dist_local)
    single_mode_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist_local)
    bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist_local)

    for nv_index in nv_indices:
        sig_counts = np.asarray(sig_counts_list_all[nv_index])
        ref_counts = np.asarray(ref_counts_list_all[nv_index])

        fig = plot_histograms(sig_counts, ref_counts, density=density)
        ax = fig.gca()

        fit_success = analyzed_data["fit_success_list"][nv_index]
        threshold = analyzed_data["threshold_list"][nv_index]
        nv_num = analyzed_data["nv_nums"][nv_index]
        readout_fidelity = analyzed_data["readout_fidelity_list"][nv_index]
        prep_fidelity = analyzed_data["prep_fidelity_list"][nv_index]

        if fit_success:
            popt = np.asarray(analyzed_data["fit_params_list"][nv_index], dtype=float)

            x_vals = np.linspace(0, np.max(ref_counts), 1000)

            dark_mode_line = popt[0] * single_mode_pdf(
                x_vals, *popt[1 : 1 + single_mode_num_params]
            )
            bright_mode_line = (1 - popt[0]) * single_mode_pdf(
                x_vals, *popt[1 + single_mode_num_params :]
            )
            bimodal_line = bimodal_pdf(x_vals, *popt)

            kpl.plot_line(
                ax, x_vals, dark_mode_line,
                color=kpl.KplColors.RED, label="NV$^{0}$ mode"
            )
            kpl.plot_line(
                ax, x_vals, bright_mode_line,
                color=kpl.KplColors.GREEN, label="NV$^{-}$ mode"
            )
            kpl.plot_line(
                ax, x_vals, bimodal_line,
                color=kpl.KplColors.BLUE, label="Combined"
            )
            ax.legend(loc=kpl.Loc.UPPER_RIGHT)

        if np.isfinite(threshold):
            ax.axvline(threshold, color=kpl.KplColors.GRAY, ls="dashed")

        info_str = (
            f"NV{nv_num}\n"
            f"Readout fidelity: {round(readout_fidelity, 3) if np.isfinite(readout_fidelity) else 'nan'}\n"
            f"Charge prep. fidelity: {round(prep_fidelity, 3) if np.isfinite(prep_fidelity) else 'nan'}"
        )
        kpl.anchored_text(ax, info_str, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)

        figs.append(fig)

    return figs

def plot_avg_images(raw_data):
    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = laser_dict["physical_name"]
    readout = laser_dict["duration"]
    readout_ms = readout / 10**6

    img_arrays = raw_data["img_arrays"]
    mean_img_arrays = np.mean(img_arrays, axis=(1, 2, 3))
    sig_img_array = mean_img_arrays[0]
    ref_img_array = mean_img_arrays[1]
    diff_img_array = sig_img_array - ref_img_array
    img_arrays_to_save = [sig_img_array, ref_img_array, diff_img_array]
    title_suffixes = ["sig", "ref", "diff"]

    img_figs = []

    for ind in range(3):
        img_array = img_arrays_to_save[ind]
        title_suffix = title_suffixes[ind]
        fig, ax = plt.subplots()
        title = f"{readout_laser}, {readout_ms} ms, {title_suffix}"
        kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
        img_figs.append(fig)

    return img_arrays_to_save, img_figs

if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_stem="2026_03_17-20_16_39-qnami-nv0_2026_02_20", load_npz=True,
    file_stem="2026_03_27-23_20_43-qnami-nv0_2026_02_20"
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True)
    process_and_plot(data, do_plot_histograms=True)
    # analyzed_data = dm.get_raw_data(
    # file_stem="2026_03_25-16_28_08-charge_state_analysis_hist_data_raw_data",
    # load_npz=True)
    
    # figs = plot_histograms_from_analysis(analyzed_data, density=True)
    kpl.show(block=True)
