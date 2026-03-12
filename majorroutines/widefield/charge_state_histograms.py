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
    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    # laser_dict = tb.get_virtual_laser_dict(laser_key)
    # readout = laser_dict["duration"]
    # readout_ms = int(readout / 1e6)
    # readout_s = readout / 1e9

    ### Histograms
    num_reps = len(ref_counts_list)
    labels = ["With ionization pulse", "Without ionization pulse"]
    # colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
    colors = [kpl.KplColors.RED, kpl.KplColors.BLUE]  # MCC
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
        # if ind == 0:
        #     continue
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
    # ax.set_yticks([0, 0.04, 0.08])

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
    weak_esr = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    counts = np.array(raw_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    ### Histograms and thresholding

    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []
    ion_prob_list = []
    red_chi_sq_list = []
    hist_figs = []

    for ind in range(num_nvs):
        if ind in weak_esr:
            continue
        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        # Only use ref counts for threshold determination
        popt, _, red_chi_sq = fit_bimodal_histogram(
            ref_counts_list, prob_dist, no_print=False
        )
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, do_print=True, ret_fidelity=True
        )
        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        if popt is not None:
            prep_fidelity = 1 - popt[0]
            ion_prob = popt[-1]
        else:
            prep_fidelity = np.nan
            ion_prob = np.nan
        # prep_fidelity = (
        #     np.count_nonzero(np.array(ref_counts_list) > threshold) / num_shots
        # )  # MCC
        prep_fidelity_list.append(prep_fidelity)
        red_chi_sq_list.append(red_chi_sq)
        ion_prob_list.append(ion_prob)

        # Plot histograms with NV index and SNR included
        nv_num = widefield.get_nv_num(nv_list[ind])
        # if do_plot_histograms and nv_num in [37, 71, 74]:
        if do_plot_histograms:
            # if False:
            fig = plot_histograms(sig_counts_list, ref_counts_list, density=True)
            ax = fig.gca()

            # Ref counts fit line
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

                # # MCC hack for including ionization
                # dark_mode_pdf = bimodal_histogram.get_single_mode_pdf(
                #     ProbDist.COMPOUND_POISSON
                # )
                # bright_mode_pdf = bimodal_histogram.get_single_mode_pdf(
                #     ProbDist.COMPOUND_POISSON
                # )
                # dark_mode_line = popt[0] * dark_mode_pdf(x_vals, popt[1])
                # bright_mode_line = (1 - popt[0]) * bright_mode_pdf(x_vals, *popt[1:])

                bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist)
                bimodal_line = bimodal_pdf(x_vals, *popt)

                kpl.plot_line(
                    ax,
                    x_vals,
                    dark_mode_line,
                    color=kpl.KplColors.RED,
                    label="NV$^{0}$ mode",  # MCC
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

            # Threshold line
            if threshold is not None:
                ax.axvline(threshold, color=kpl.KplColors.GRAY, ls="dashed")

            # Add text of the fidelities
            snr_str = (
                f"NV{nv_num}\n"
                f"Readout fidelity: {round(readout_fidelity, 3)}\n"
                f"Charge prep. fidelity {round(prep_fidelity, 3)}"
            )
            kpl.anchored_text(ax, snr_str, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)

            # if readout_fidelity < 0.8:
            #     kpl.show(block=True)
            # plt.close(fig)
            kpl.show(block=True)
            fig = None

            if fig is not None:
                hist_figs.append(fig)

    print(f"readout_fidelity_list: {readout_fidelity_list}")
    print(f"prep_fidelity_list: {prep_fidelity_list}")
    print(f"red_chi_sq_list: {red_chi_sq_list}")
    print(f"thresholds: {threshold_list}")

    # Report out the results
    threshold_list = np.array(threshold_list)
    readout_fidelity_list = np.array(readout_fidelity_list)
    prep_fidelity_list = np.array(prep_fidelity_list)

    # Scatter readout vs prep fidelity
    fig, ax = plt.subplots()
    kpl.plot_points(ax, readout_fidelity_list, prep_fidelity_list)
    ax.set_xlabel("Readout fidelity")
    ax.set_ylabel("NV- preparation fidelity")

    # Plot prep fidelity vs distance from center
    # coords_key = "laser_INTE_520_aod"
    # distances = []
    # for nv in nv_list:
    #     coords = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
    #     dist = np.sqrt((110 - coords[0]) ** 2 + (110 - coords[1]) ** 2)
    #     distances.append(dist)
    # fig, ax = plt.subplots()
    # kpl.plot_points(ax, distances, prep_fidelity_list)
    # ax.set_xlabel("Distance from center frequencies (MHz)")
    # ax.set_ylabel("NV- preparation fidelity")

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
    avg_ion_prob = round(np.mean(ion_prob_list), 6)
    print(f"Average ionization during readout probability: {avg_ion_prob}")
    var_ion_prob = round(np.var(ion_prob_list), 6)
    print(f"Variance ionization during readout probability: {var_ion_prob}")

    return hist_figs


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


def main(
    nv_list,
    num_reps,
    num_runs,
    ion_do_target_inds=None,
    verify_charge_states=False,
    do_plot_histograms=False,
):
    ### Initial setup
    seq_file = "charge_state_histograms.py"
    num_steps = 1

    # Turn the list of NV indices to ionize into a list of True/False for
    # each NV according to whether it should be targeted
    if ion_do_target_inds is None:
        ion_do_target_list = None
    else:
        num_nvs = len(nv_list)
        ion_do_target_list = [ind in ion_do_target_inds for ind in range(num_nvs)]

    if verify_charge_states:
        charge_prep_fn = base_routine.charge_prep_loop
    else:
        charge_prep_fn = None

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(nv_list, VirtualLaserKey.CHARGE_POL)
        )
        ion_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.ION)
        seq_args = [
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            ion_coords_list,
            ion_do_target_list,
            verify_charge_states,
        ]
        # print("seq_args:", seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=True,
        save_images_avg_reps=False,
        charge_prep_fn=charge_prep_fn,
    )

    ### Processing

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    # Images
    try:
        imgs, img_figs = plot_avg_images(raw_data)
        # Save
        sig_img_array, ref_img_array, diff_img_array = imgs
        keys_to_compress = ["sig_img_array", "ref_img_array", "diff_img_array"]
        title_suffixes = ["sig", "ref", "diff"]
        num_figs = len(img_figs)
        for ind in range(num_figs):
            fig = img_figs[ind]
            title = title_suffixes[ind]
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{title}")
            dm.save_figure(fig, file_path)
    except Exception:
        print(traceback.format_exc())
        sig_img_array = None
        ref_img_array = None
        diff_img_array = None
        keys_to_compress = None

    # Histograms
    # try:
    #     hist_figs = process_and_plot(raw_data, do_plot_histograms=do_plot_histograms)
    #     # Save
    #     if hist_figs is not None:
    #         num_nvs = len(nv_list)
    #         for nv_ind in range(num_nvs):
    #             fig = hist_figs[nv_ind]
    #             nv_sig = nv_list[nv_ind]
    #             nv_name = nv_sig.name
    #             file_path = dm.get_file_path(__file__, timestamp, nv_name)
    #             dm.save_figure(fig, file_path)
    # except Exception:
    #     print(traceback.format_exc())

    # try:
    #     del raw_data["img_arrays"]
    # except Exception:
    #     pass

    ### Save raw data

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    raw_data |= {
        "timestamp": timestamp,
        "sig_img_array": sig_img_array,
        "ref_img_array": ref_img_array,
        "diff_img_array": diff_img_array,
        "img_array-units": "photons",
    }
    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    tb.reset_cfm()

    return raw_data


import numpy as np
import matplotlib.pyplot as plt

from analysis.bimodal_histogram import (
    ProbDist,
    fit_bimodal_histogram,
    determine_threshold,
    analyze_charge_histogram_multinv_binomial,
)

def find_nv_index_by_nvnum(nv_list, nv_num_target=122):
    # matches common dict keys; fallback treats nv_num_target as index
    keys = ["nv_num", "nv_number", "nv_id", "id", "num"]
    for i, nv in enumerate(nv_list):
        if isinstance(nv, dict):
            for k in keys:
                if k in nv:
                    try:
                        if int(nv[k]) == int(nv_num_target):
                            return i
                    except Exception:
                        pass
    if 0 <= int(nv_num_target) < len(nv_list):
        return int(nv_num_target)
    raise ValueError(f"Could not find NV {nv_num_target} in nv_list")

def analyze_nv122_ref_only(raw_data, nv_num=122, ref_axis=0, prob_dist=ProbDist.COMPOUND_POISSON, max_nvs=3):
    nv_list = raw_data["nv_list"]
    nv_ind = find_nv_index_by_nvnum(nv_list, nv_num)

    # your 5D array: (exp, nv, rep, step, shot)
    counts = np.asarray(raw_data["counts"], dtype=float)  # or raw_data["counts"] if that's the key
    ref_counts = counts[ref_axis, nv_ind, :, :, :].reshape(-1)
    ref_counts = ref_counts[np.isfinite(ref_counts)]

    print(f"NV {nv_num} -> nv_ind={nv_ind}, #REF shots={ref_counts.size}")

    # --- legacy bimodal fit ---
    popt, _, red_chi_sq = fit_bimodal_histogram(ref_counts, prob_dist)
    if popt is None:
        print("[bimodal] fit failed")
        thr_bi, fid_bi = np.nan, np.nan
    else:
        thr_bi, fid_bi = determine_threshold(popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True)
        print(f"[bimodal] thr={thr_bi:.2f}, fid={fid_bi:.3f}, red_chi_sq={red_chi_sq}")

    # --- binomial mult-NV structured model (tries N=1..max_nvs) ---
    fitN = analyze_charge_histogram_multinv_binomial(
        ref_counts,
        prob_dist=prob_dist,
        max_nvs=max_nvs,
        force_nvs=None,
        bic_extra_nv_penalty=2.0,
        seed=0,
    )
    if not fitN.get("ok", False):
        print("[binomial mult-NV] fit failed")
    else:
        N = fitN["n_nvs"]
        p_minus, rate0, delta = [float(v) for v in fitN["popt"]]
        print(
            f"[binomial mult-NV] N={N}, p_minus={p_minus:.3f}, rate0={rate0:.2f}, delta={delta:.2f}\n"
            f"  fidelity_any={fitN['fidelity_any']:.3f}, fidelity_multiclass={fitN['fidelity_multiclass']:.3f}\n"
            f"  threshold_any={fitN['threshold_any']:.2f}"
        )

    # plot histogram
    plt.figure(figsize=(6, 4))
    bins = int(min(250, max(40, np.sqrt(ref_counts.size))))
    plt.hist(ref_counts, bins=bins, density=True, alpha=0.7, label="REF counts (50 ms)")
    if np.isfinite(thr_bi):
        plt.axvline(thr_bi, ls="--", label=f"bimodal thr {thr_bi:.1f}")
    if fitN.get("ok", False):
        plt.axvline(fitN["threshold_any"], ls="--", label=f"any thr {fitN['threshold_any']:.1f}")
    plt.xlabel("counts")
    plt.ylabel("density")
    plt.title(f"NV {nv_num} (REF axis {ref_axis})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"nv_ind": nv_ind, "ref_counts": ref_counts, "bimodal": (popt, red_chi_sq), "binomial": fitN}

def compare_readout_sets(file_stems, readout_ms, nv_num=122, ref_axis=0):
    results = []

    plt.figure(figsize=(7, 5))

    for stem, ms in zip(file_stems, readout_ms):
        data = dm.get_raw_data(file_stem=stem, load_npz=True)  # or file_id
        out = analyze_nv122_ref_only(data, nv_num=nv_num, ref_axis=ref_axis)
        ref_counts = out["ref_counts"]

        bins = int(min(220, max(40, np.sqrt(ref_counts.size))))
        plt.hist(ref_counts, bins=bins, density=True, alpha=0.25, label=f"{ms} ms")

        results.append((ms, out))

    plt.title(f"NV {nv_num} REF hist overlay")
    plt.xlabel("counts")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print key metrics vs duration
    for ms, out in results:
        b = out["binomial"]
        if b.get("ok", False):
            print(f"{ms} ms: N={b['n_nvs']}, fid_any={b['fidelity_any']:.3f}, fid_multi={b['fidelity_multiclass']:.3f}, p_minus={b['popt'][0]:.3f}")
        else:
            print(f"{ms} ms: binomial fit failed")

    return results

if __name__ == "__main__":
    kpl.init_kplotlib()
    # data = dm.get_raw_data(file_id=1733583334808, load_npz=False)
    # data = dm.get_raw_data(file_id=1766803842180, load_npz=False)  # 50ms readout
    # data = dm.get_raw_data(file_id=1766834596476, load_npz=False)  # 100ms readout
    # data = dm.get_raw_data(file_id=1769860748790, load_npz=False)  # 24ms readout

    # rubin
    # data = dm.get_raw_data(file_id=1798326378514, load_npz=False)  # 60ms readout
    # data = dm.get_raw_data(file_id=1798728176452, load_npz=False)
    # data = dm.get_raw_data(file_id=1798741474977, load_npz=False)
    # data = dm.get_raw_data(file_id=1798741474977, load_npz=False)

    # data = dm.get_raw_data(file_id=1800336690568, load_npz=False)
    # 300 NVs
    # data = dm.get_raw_data(file_id=1802802596578, load_npz=False)
    # 200 NVs
    # data = dm.get_raw_data(file_id=1804953793322, load_npz=False)
    # 154 NVs
    # data = dm.get_raw_data(file_id=1806214782623, load_npz=False)
    # data = dm.get_raw_data(file_id=1806222218365, load_npz=False)
    # data = dm.get_raw_data(file_id=1806227898070, load_npz=False)
    # data = dm.get_raw_data(file_id=1806410973406, load_npz=False)
    data = dm.get_raw_data(
        # file_stem="2026_03_11-01_36_16-qnami-nv0_2026_02_20", load_npz=True
        file_stem="2026_03_11-20_46_02-qnami-nv0_2026_02_20", load_npz=True
    )
    # process_and_plot(data, do_plot_histograms=True)
    # data = dm.get_raw_data(
    #     file_stem="pti_x, opti_y", load_npz=True
    # )
    # out = analyze_nv122_ref_only(data, nv_num=122, ref_axis=1)
    # compare_readout_sets(
    #     file_stems=["2026_03_02-17_30_11-qnami-nv0_2026_02_20"],
    #     readout_ms=[50],          # <-- must match length
    #     nv_num=122,-
    #     ref_axis=1,
    # )
    process_and_plot(data, do_plot_histograms=True)
    kpl.show(block=True)
