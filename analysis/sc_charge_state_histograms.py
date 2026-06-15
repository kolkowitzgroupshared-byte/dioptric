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
from joblib import Parallel, delayed

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
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
):
    ### Setup
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])

    sig_counts_lists = [
        np.asarray(counts[0, nv_ind].flatten()) for nv_ind in range(num_nvs)
    ]
    ref_counts_lists = [
        np.asarray(counts[1, nv_ind].flatten()) for nv_ind in range(num_nvs)
    ]

    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    base_file_stem = (
        raw_data.get("file_stem") or raw_data.get("file_name") or "raw_data"
    )
    if isinstance(base_file_stem, (list, tuple)):
        base_file_stem = "_".join(map(str, base_file_stem))
    base_file_stem = str(base_file_stem).replace(" ", "_")

    def process_one_nv(ind):
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        try:
            popt, _, red_chi_sq = fit_bimodal_histogram(
                ref_counts_list, prob_dist, no_print=False
            )

            if popt is not None:
                threshold, readout_fidelity = determine_threshold(
                    popt,
                    prob_dist,
                    dark_mode_weight=0.5,
                    do_print=True,
                    ret_fidelity=True,
                )
                prep_fidelity = 1 - popt[0]
                ion_prob = popt[-1]
                fit_success = True
                fit_params_to_save = np.asarray(popt, dtype=float).tolist()
            else:
                threshold = np.nan
                readout_fidelity = np.nan
                prep_fidelity = np.nan
                ion_prob = np.nan
                red_chi_sq = np.nan
                fit_success = False
                fit_params_to_save = None

        except Exception as e:
            print(f"Error processing NV {ind}: {e}")
            threshold = np.nan
            readout_fidelity = np.nan
            prep_fidelity = np.nan
            ion_prob = np.nan
            red_chi_sq = np.nan
            fit_success = False
            fit_params_to_save = None

        nv_num = widefield.get_nv_num(nv_list[ind])

        return {
            "nv_ind": ind,
            "nv_num": nv_num,
            "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
            "readout_fidelity": (
                float(readout_fidelity) if np.isfinite(readout_fidelity) else np.nan
            ),
            "prep_fidelity": (
                float(prep_fidelity) if np.isfinite(prep_fidelity) else np.nan
            ),
            "ion_prob": float(ion_prob) if np.isfinite(ion_prob) else np.nan,
            "red_chi_sq": float(red_chi_sq) if np.isfinite(red_chi_sq) else np.nan,
            "fit_success": bool(fit_success),
            "fit_params": fit_params_to_save,
        }

    # -------- parallel processing over NVs --------
    nv_results = Parallel(n_jobs=-1)(
        delayed(process_one_nv)(ind) for ind in range(num_nvs)
    )

    # -------- unpack results --------
    threshold_list = [res["threshold"] for res in nv_results]
    readout_fidelity_list = [res["readout_fidelity"] for res in nv_results]
    prep_fidelity_list = [res["prep_fidelity"] for res in nv_results]
    ion_prob_list = [res["ion_prob"] for res in nv_results]
    red_chi_sq_list = [res["red_chi_sq"] for res in nv_results]
    fit_success_list = [res["fit_success"] for res in nv_results]
    fit_params_list = [res["fit_params"] for res in nv_results]
    nv_num_list = [res["nv_num"] for res in nv_results]

    print(f"readout_fidelity_list: {readout_fidelity_list}")
    print(f"prep_fidelity_list: {prep_fidelity_list}")
    print(f"red_chi_sq_list: {red_chi_sq_list}")
    print(f"thresholds: {threshold_list}")

    # Scatter readout vs prep fidelity
    fig, ax = plt.subplots()
    kpl.plot_points(ax, readout_fidelity_list, prep_fidelity_list)
    ax.set_xlabel("Readout fidelity")
    ax.set_ylabel("NV- preparation fidelity")

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
        "nv_inds": list(range(num_nvs)),
        "nv_nums": nv_num_list,
        "num_nvs": int(num_nvs),
        # save all histogram data
        "sig_counts_list": [np.asarray(x).ravel().tolist() for x in sig_counts_lists],
        "ref_counts_list": [np.asarray(x).ravel().tolist() for x in ref_counts_lists],
        # fit outputs
        "fit_params_list": fit_params_list,
        "fit_success_list": fit_success_list,
        "threshold_list": threshold_list,
        "prep_fidelity_list": prep_fidelity_list,
        "readout_fidelity_list": readout_fidelity_list,
        "red_chi_sq_list": red_chi_sq_list,
        "ion_prob_list": ion_prob_list,
        # metadata
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "num_shots": int(num_shots),
        "prob_dist_name": (
            prob_dist.name if hasattr(prob_dist, "name") else str(prob_dist)
        ),
        # summaries
        "avg_readout_fidelity": float(avg_readout_fidelity),
        "std_readout_fidelity": float(std_readout_fidelity),
        "avg_prep_fidelity": float(avg_prep_fidelity),
        "std_prep_fidelity": float(std_prep_fidelity),
        "avg_ion_prob": float(avg_ion_prob),
        "var_ion_prob": float(var_ion_prob),
    }

    timestamp = dm.get_time_stamp()
    file_name = f"charge_state_analysis_hist_data_{base_file_stem}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)
    dm.save_raw_data(results, file_path)

    # -------- optional plotting after parallel analysis --------
    if do_plot_histograms:
        prob_dist_name = results.get("prob_dist_name", "COMPOUND_POISSON")
        prob_dist_local = getattr(ProbDist, prob_dist_name)

        single_mode_num_params = bimodal_histogram.get_single_mode_num_params(
            prob_dist_local
        )
        single_mode_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist_local)
        bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist_local)

        for nv_index in range(num_nvs):
            sig_counts = np.asarray(results["sig_counts_list"][nv_index], dtype=float)
            ref_counts = np.asarray(results["ref_counts_list"][nv_index], dtype=float)

            fig = plot_histograms(sig_counts, ref_counts, density=True)
            ax = fig.gca()

            fit_success = results["fit_success_list"][nv_index]
            threshold = results["threshold_list"][nv_index]
            nv_num = results["nv_nums"][nv_index]
            readout_fidelity = results["readout_fidelity_list"][nv_index]
            prep_fidelity = results["prep_fidelity_list"][nv_index]

            if fit_success and results["fit_params_list"][nv_index] is not None:
                popt = np.asarray(results["fit_params_list"][nv_index], dtype=float)
                x_vals = np.linspace(0, np.max(ref_counts), 1000)

                dark_mode_line = popt[0] * single_mode_pdf(
                    x_vals, *popt[1 : 1 + single_mode_num_params]
                )
                bright_mode_line = (1 - popt[0]) * single_mode_pdf(
                    x_vals, *popt[1 + single_mode_num_params :]
                )
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
                    ax, x_vals, bimodal_line, color=kpl.KplColors.BLUE, label="Combined"
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

            kpl.show(block=True)

    return results


def plot_histograms_from_analysis(analyzed_data, nv_indices=None, density=True):
    sig_counts_list_all = analyzed_data["sig_counts_list"]
    ref_counts_list_all = analyzed_data["ref_counts_list"]

    if nv_indices is None:
        nv_indices = range(len(sig_counts_list_all))

    figs = []

    prob_dist_name = analyzed_data.get("prob_dist_name", "COMPOUND_POISSON")
    prob_dist_local = getattr(ProbDist, prob_dist_name)

    single_mode_num_params = bimodal_histogram.get_single_mode_num_params(
        prob_dist_local
    )
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
                ax, x_vals, bimodal_line, color=kpl.KplColors.BLUE, label="Combined"
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
    file_stem = "2026_04_02-19_01_32-qnami-nv0_2026_02_20"
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True)
    process_and_plot(data, do_plot_histograms=True)
    # analyzed_data = dm.get_raw_data(
    # file_stem="2026_03_25-16_28_08-charge_state_analysis_hist_data_raw_data",
    # load_npz=True)

    # figs = plot_histograms_from_analysis(analyzed_data, density=True)
    kpl.show(block=True)
