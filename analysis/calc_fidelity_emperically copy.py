#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:10:35 2022
file to help find fidelity emperically, outside of having to use determine_charge_readout_params file

@author: carterfox
"""

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import majorroutines.charge_majorroutines.photonstatistics as model
import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tool_belt


def calculate_threshold_no_model(
    readout_time,
    nv0_hist,
    nvm_hist,
    mu_0,
    mu_m,
    x_vals_0,
    x_vals_m,
    power,
    nd_filter=None,
):
    (
        thresh,
        fid,
        threshold_list,
        fidelity_list,
    ) = model.calculate_threshold_from_experiment(
        x_vals_0, x_vals_m, mu_0, mu_m, nv0_hist, nvm_hist
    )
    # print(fid)

    fig3, ax = plt.subplots(1, 1)
    ax.plot(x_vals_0, nv0_hist, "r-o", label=r"$m_s$=0")
    ax.plot(x_vals_m, nvm_hist, "g-o", label=r"$m_s$=-1")
    ax.set_xlabel("Counts")
    ax.set_ylabel("Occur.")
    plt.axvline(x=thresh, color="blue", linestyle="--")
    ax.legend(title="Prep State")
    textstr = "\n".join(
        (
            r"$\mu_0=%.2f$" % (mu_0),
            r"$\mu_{-1}=%.2f$" % (mu_m),
            r"$threshold = %.1f$" % (thresh),
            r"$fidelity = %.3f$" % (fid),
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.72,
        0.58,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )
    if nd_filter:
        title_text = "{} us readout, {} V, {}".format(
            int(readout_time / 1e3), power, nd_filter
        )
    else:
        title_text = "{} us readout, {} V".format(int(readout_time / 1e3), power)
    ax.set_title(title_text)
    return thresh, fid, fig3, threshold_list, fidelity_list


def calc_histogram(nv0, nvm, dur, bins=None):
    # Counts are in us, readout is in ns
    dur_us = dur / 1e3
    # print(nv0)
    nv0_counts = nv0.tolist()
    nvm_counts = nvm.tolist()

    max_0 = int(max(nv0_counts))
    max_m = int(max(nvm_counts))
    if bins == None:
        occur_0, bin_edges_0 = np.histogram(
            nv0_counts,
            np.linspace(0, max_0, max_0 + 1),  # 200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts,
            np.linspace(0, max_m, max_m + 1),  # 200)  #
        )
    elif bins != None:
        occur_0, bin_edges_0 = np.histogram(
            nv0_counts,
            bins,  # np.linspace(0, max_0, max_0 + 1) #200)  #
        )
        occur_m, bin_edge_m = np.histogram(
            nvm_counts,
            bins,  # np.linspace(0, max_m,  max_m + 1) #200)  #
        )
    # norm_occur_m = occur_m/sum(occur_m)
    # norm_occur_0 = occur_0/sum(occur_0)

    # Histogram returns bin edges. A bin is defined with the first point
    # inclusive and the last exclusive - eg a count a 2 will fall into
    # bin [2,3) - so just drop the last bin edge for our x vals
    x_vals_0 = bin_edges_0[:-1]
    x_vals_m = bin_edge_m[:-1]

    return occur_0, x_vals_0, occur_m, x_vals_m


def calc_fidelity(nv0_counts, nvm_counts, readout_dur, readout_power):
    # readout_dur = 10e9 # just make it super big because we just want to use what was measured

    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(
        nv0_counts, nvm_counts, readout_dur, bins=None
    )
    # print(occur_m)
    max_x_val = max(list(x_vals_0) + list(x_vals_m)) + 10

    num_reps = len(nv0_counts)
    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    mean_m = sum(occur_m * x_vals_m) / num_reps

    (
        threshold,
        fidelity,
        fig,
        threshold_list,
        fidelity_list,
    ) = calculate_threshold_no_model(
        readout_dur,
        occur_0,
        occur_m,
        mean_0,
        mean_m,
        x_vals_0,
        x_vals_m,
        readout_power,
        None,
    )
    print(fidelity)

    return threshold, fidelity, threshold_list, fidelity_list


if __name__ == "__main__":
    filename = "2022_12_10-08_56_16-johnson-search-ion_pulse_dur"

    data = tool_belt.get_raw_data(filename)
    nv_sig = data["nv_sig"]
    readout_dur = nv_sig["charge_readout_dur"]
    readout_power = nv_sig["charge_readout_laser_power"]
    nvm_counts = np.array(data["sig_counts_eachshot_array"])[0]
    nv0_counts = np.array(data["ref_counts_eachshot_array"])[0]

    # plt.figure()
    # plt.hist(nvm_counts,histtype='step',bins=range(int(min(nvm_counts)), int(max(nvm_counts)) + 1, 1))
    # plt.hist(nv0_counts,histtype='step',bins=range(int(min(nv0_counts)), int(max(nv0_counts)) + 1, 1))
    # plt.show()

    mean_s = np.mean(nvm_counts)
    mean_r = np.mean(nv0_counts)
    std_s = np.std(nvm_counts)
    std_r = np.std(nv0_counts)

    # threshold, fidelity, threshold_list, fidelity_list  =calc_fidelity(nv0_counts, nvm_counts,readout_dur,readout_power)
    # print(threshold, fidelity)
    # print(threshold_list)
    # print(fidelity_list)

    occur_0, x_vals_0, occur_m, x_vals_m = calc_histogram(
        nv0_counts, nvm_counts, readout_dur, bins=None
    )
    # print(occur_m)
    max_x_val = max(list(x_vals_0) + list(x_vals_m)) + 10
    num_reps = len(nv0_counts)
    mean_0 = sum(occur_0 * x_vals_0) / num_reps
    mean_m = sum(occur_m * x_vals_m) / num_reps
    thred = 3
    area_below_0, area_above_0 = model.get_area(x_vals_0, occur_0.tolist(), thred)
    full_area_0 = area_below_0 + area_above_0

    area_below_m, area_above_m = model.get_area(x_vals_m, occur_m.tolist(), thred)
    full_area_m = area_below_m + area_above_m
