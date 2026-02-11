# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference

Created on Fall 2023
@author: mccambria
Updated on Fall 2024
@author: schand
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

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


def process_and_plot_mcc(raw_data):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    # num_nvs = 10
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]
    counts = np.array(raw_data["counts"])
    # [nv_ind, run_ind, steq_ind, rep_ind]
    ref_exp_ind = 1
    condensed_counts = [
        [
            counts[ref_exp_ind, nv_ind, :, step_ind, :].flatten()
            for step_ind in range(num_steps)
        ]
        for nv_ind in range(num_nvs)
    ]
    condensed_counts = np.array(condensed_counts)

    prob_dist = ProbDist.COMPOUND_POISSON

    readout_fidelity_arr = np.empty((num_nvs, num_steps))
    prep_fidelity_arr = np.empty((num_nvs, num_steps))
    red_chi_sq_arr = np.empty((num_nvs, num_steps))
    for nv_ind in range(num_nvs):
        for step_ind in range(num_steps):
            print(step_vals[step_ind])
            popt, _, red_chi_sq = fit_bimodal_histogram(
                condensed_counts[nv_ind, step_ind], prob_dist, no_plot=True
            )
            if popt is None:
                readout_fidelity = np.nan
                prep_fidelity = np.nan
            else:
                threshold, readout_fidelity = determine_threshold(
                    popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
                )
                prep_fidelity = 1 - popt[0]
            readout_fidelity_arr[nv_ind, step_ind] = readout_fidelity
            prep_fidelity_arr[nv_ind, step_ind] = prep_fidelity
            red_chi_sq_arr[nv_ind, step_ind] = red_chi_sq

    ### Plotting

    x_vals = step_vals
    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            x_vals /= 1e3
            x_label = "Polarization duration (us)"
        else:
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            x_vals /= 1e6
            x_label = "Readout duration (ms)"
        else:
            x_label = "Readout amplitude"

    # Three-line plot showing fidelities and red. chi sq.

    fig, ax0 = plt.subplots()
    ax0.set_xlabel(x_label)
    ax0.set_ylabel("Fidelity")
    kpl.plot_line(
        ax0,
        x_vals,
        np.nanmedian(readout_fidelity_arr, axis=0),
        label="Readout",
    )
    kpl.plot_line(
        ax0,
        x_vals,
        np.nanmedian(prep_fidelity_arr, axis=0),
        label="Charge prep.",
        color=kpl.KplColors.GREEN,
    )
    ax0.legend(loc=kpl.Loc.UPPER_LEFT)
    ax1 = ax0.twinx()
    color = kpl.KplColors.RED
    kpl.plot_line(ax1, x_vals, np.nanmedian(red_chi_sq_arr, axis=0), color=color)
    ax1.set_ylabel(r"$\chi^{2}_{\nu}$", color=color)
    ax1.tick_params(axis="y", color=color, labelcolor=color)
    ax1.xaxis.label.set_color(color)
    ax1.spines["right"].set_color(color)

    # print("Plotting results for first 10 NVs")

    # for nv_ind in range(num_nvs):
    #     arrs = [readout_fidelity_arr, prep_fidelity_arr, red_chi_sq_arr]
    #     ylabels = ["Readout fidelity", "Charge pol. fidelity", "Reduced chi squared"]
    #     arr = arrs[0]
    #     ylabel = ylabels[0]
    #     fig, ax0 = plt.subplots()
    #     ax0.set_title(f"NV{nv_ind}")
    #     ax0.set_xlabel(x_label)
    #     ax0.set_ylabel(ylabel)
    #     kpl.plot_points(ax0, step_vals, arr[nv_ind, :])
    #     ax1 = ax0.twinx()
    #     arr = arrs[1]
    #     ylabel = ylabels[1]
    #     ax1.set_ylabel(ylabel)
    #     kpl.plot_points(ax1, step_vals, arr[nv_ind, :])
    #     arr = arrs[2]
    #     ylabel = ylabels[2]
    #     kpl.plot_points(ax1, step_vals, arr[nv_ind, :])
    #     kpl.show(block=True)

    ### Extract optimal point for each NV

    opti_vals = []

    # Readout amp - fidelity increases with monotonically with amp, so find where
    # fit breaks down and stop
    if not optimize_pol_or_readout and not optimize_duration_or_amp:
        fidelity_arr = readout_fidelity_arr
        for nv_ind in range(num_nvs):
            nv_red_chi_sq_arr = red_chi_sq_arr[nv_ind, :]
            good_fits = nv_red_chi_sq_arr < 1.1
            num_errors = []
            for ind in range(num_steps):
                good_fits_model = [jnd <= ind for jnd in range(num_steps)]
                num_errors.append(np.sum(np.logical_xor(good_fits_model, good_fits)))
            opti_val = step_vals[np.argmin(num_errors)]
            opti_vals.append(opti_val)

            # Plot
            fig, ax = plt.subplots()
            kpl.plot_points(ax, step_vals, nv_red_chi_sq_arr)
            ax.axvline(opti_val)
            ax.set_title(nv_ind)
            kpl.show(block=True)
    # Polarization amp - fit a quadratic, exclude bad fits (red. chi sq. > 1.1)
    elif not optimize_pol_or_readout and not optimize_duration_or_amp:
        fidelity_arr = prep_fidelity_arr

        def quadratic(x, x0, y0, a):
            return y0 + a * (x - x0) ** 2

        for nv_ind in range(num_nvs):
            nv_fidelity_arr = fidelity_arr[nv_ind, :]
            nv_red_chi_sq_arr = red_chi_sq_arr[nv_ind, :]
            nv_fidelity_arr[nv_red_chi_sq_arr > 1.1] = np.nan
            x0_guess = nv_fidelity_arr[np.nanargmax(nv_fidelity_arr)]
            y0_guess = np.nanmax(nv_fidelity_arr)
            a_guess = -(max_step_val - min_step_val) / 10
            guess_params = [x0_guess, y0_guess, a_guess]
            popt, pcov = curve_fit(
                quadratic, step_vals, nv_fidelity_arr, guess_params, nan_policy="omit"
            )
            opti_vals.append(popt[0])

            # Plot
            fig, ax = plt.subplots()
            kpl.plot_points(ax, step_vals, nv_fidelity_arr)
            smooth_step_vals = np.linspace(min_step_val, max_step_val, 1000)
            kpl.plot_line(ax, smooth_step_vals, quadratic(smooth_step_vals, *popt))
            ax.set_title(nv_ind)
            kpl.show(block=True)

    print(opti_vals)


def find_optimal_combined_value(
    step_vals, readout_fidelity, goodness_of_fit, weight=0.5
):
    # Normalize both metrics to ensure equal weighting
    norm_fidelity = (readout_fidelity - np.nanmin(readout_fidelity)) / (
        np.nanmax(readout_fidelity) - np.nanmin(readout_fidelity)
    )
    norm_goodness = (goodness_of_fit - np.nanmin(goodness_of_fit)) / (
        np.nanmax(goodness_of_fit) - np.nanmin(goodness_of_fit)
    )

    # Combined score: weighted sum of normalized metrics
    combined_score = weight * norm_fidelity + (1 - weight) * norm_goodness

    # Find the step value corresponding to the maximum combined score
    max_index = np.nanargmax(combined_score)
    optimal_step_val = step_vals[max_index]
    max_combined_score = combined_score[max_index]

    return optimal_step_val, max_combined_score


def find_optimal_value_geom_mean(
    step_vals, prep_fidelity, readout_fidelity, goodness_of_fit, weights=(1, 1, 1)
):
    """
    Finds the optimal step value using a weighted geometric mean of fidelities and goodness of fit.

    Parameters:
    ----------
    step_vals : np.ndarray
        Array of step values.
    prep_fidelity : np.ndarray
        Array of preparation fidelities.
    readout_fidelity : np.ndarray
        Array of readout fidelities.
    goodness_of_fit : np.ndarray
        Array of goodness of fit (chi-squared values).
    weights : tuple(float, float, float)
        Weights for readout fidelity, prep fidelity, and goodness of fit, respectively.

    Returns:
    -------
    optimal_step_val : float
        The step value corresponding to the optimal combined score.
    """
    w1, w2, w3 = weights
    # Normalize metrics
    norm_prep_fidelity = (prep_fidelity - np.nanmin(prep_fidelity)) / (
        np.nanmax(prep_fidelity) - np.nanmin(prep_fidelity)
    )
    norm_readout_fidelity = (readout_fidelity - np.nanmin(readout_fidelity)) / (
        np.nanmax(readout_fidelity) - np.nanmin(readout_fidelity)
    )
    norm_goodness = (goodness_of_fit - np.nanmin(goodness_of_fit)) / (
        np.nanmax(goodness_of_fit) - np.nanmin(goodness_of_fit)
    )
    inverted_goodness = 1 - norm_goodness  # Minimize goodness of fit

    # Compute weighted geometric mean
    combined_score = (
        (norm_readout_fidelity**w1) * (norm_prep_fidelity**w2) * (inverted_goodness**w3)
    ) ** (1 / (w1 + w2 + w3))

    # Find the step value corresponding to the maximum combined score
    max_index = np.nanargmax(combined_score)
    optimal_step_val = step_vals[max_index]

    return optimal_step_val


def process_and_plot(raw_data):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps) * 0.39
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]

    counts = np.array(raw_data["counts"])
    # [nv_ind, run_ind, steq_ind, rep_ind]
    ref_exp_ind = 1
    condensed_counts = [
        [
            counts[ref_exp_ind, nv_ind, :, step_ind, :].flatten()
            for step_ind in range(num_steps)
        ]
        for nv_ind in range(num_nvs)
    ]
    condensed_counts = np.array(condensed_counts)

    prob_dist = ProbDist.COMPOUND_POISSON

    # Function to process a single NV and step
    def process_nv_step(nv_ind, step_ind):
        counts_data = condensed_counts[nv_ind, step_ind]
        popt, norm_res = fit_bimodal_histogram(counts_data, prob_dist)

        if popt is None:
            return np.nan, np.nan, np.nan, np.nan
        # Threshold, prep and readout fidelity
        threshold, readout_fidelity = determine_threshold(
            popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
        )
        prep_fidelity = 1 - popt[0]  # Population weight of dark state

        return readout_fidelity, prep_fidelity, norm_res

    # Parallel processing --> n_jobs=-1:using all available cores.
    results = Parallel(n_jobs=-1)(
        delayed(process_nv_step)(nv_ind, step_ind)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    )

    # Reshape results into arrays
    results = np.array(results).reshape(num_nvs, num_steps, 3)
    readout_fidelity_arr = results[:, :, 0]
    prep_fidelity_arr = results[:, :, 1]
    goodness_of_fit_arr = results[:, :, 2]

    ### Plotting
    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            x_label = "Polarization duration"
        else:
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            x_label = "Readout duration"
        else:
            x_label = "Readout amplitude"
            # x_label = "Pol. amplitude"

    ### Calculate Averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0)
    avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0)
    avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0)

    # Calculate the optimal step value
    optimal_step_val = find_optimal_value_geom_mean(
        step_vals,
        avg_readout_fidelity,
        avg_prep_fidelity,
        avg_goodness_of_fit,
    )

    # Plot average readout and prep fidelity
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        step_vals,
        avg_readout_fidelity,
        label="Avg. Readout Fidelity",
        color="blue",
    )
    ax1.plot(
        step_vals,
        avg_prep_fidelity,
        label="Avg. Prep Fidelity",
        color="orange",
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Fidelity")
    ax1.tick_params(axis="y")

    # Plot average goodness of fit (R²)
    ax2 = ax1.twinx()
    ax2.plot(
        step_vals,
        avg_goodness_of_fit,
        color="green",
        linestyle="--",
        label="Avg. Goodness of Fit (chi-squared)",
    )
    ax2.set_ylabel("Goodness of Fit (chi-squared)", color="green")

    ax2.tick_params(axis="y", labelcolor="green")
    ax2.axvline(
        optimal_step_val,
        color="red",
        linestyle="--",
        label=f"Optimal Step Val: {optimal_step_val:.2f}",
    )
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=9)

    # Title and layout
    ax1.set_title("Average Metrics Across All NVs", fontsize=12)
    fig.tight_layout()
    plt.show()


# endregion

def optimize_readout_duration(
    nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
):
    return _main(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration, True
    )


def optimize_readout_amp(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp):
    return _main(nv_list, num_steps, num_reps, num_runs, min_amp, max_amp, False)


def _main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_step_val,
    max_step_val,
    optimize_duration_or_amp,
    uwave_ind_list=[0, 1],
):
    ### Initial setup
    seq_file = "optimize_scc_readout.py"
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)
    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        # NumPy indexing allows list-based indexing
        shuffled_step_vals = [step_vals[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_step_vals,
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
    )

    ### Processing

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    raw_data |= {
        "timestamp": timestamp,
        "min_step_val": min_step_val,
        "max_step_val": max_step_val,
        "optimize_duration_or_amp": optimize_duration_or_amp,
    }

    try:
        process_and_plot(raw_data)

    except Exception:
        print(traceback.format_exc())

    try:
        del raw_data["img_arrays"]
    except Exception:
        pass

    ### Save and clean up

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()
    # raw_data = dm.get_raw_data(file_id=1714802805037, load_npz=False)  # Messed up duration variation
    # raw_data = dm.get_raw_data(file_id=1720970373150, load_npz=False)
    # process_and_plot_mcc(raw_data)
    # sys.exit()
    # # raw_data = dm.get_raw_data(file_id=1709868774004, load_npz=False) #yellow ampl var
    # raw_data = dm.get_raw_data(file_id=1710843759806, load_npz=False)  # yellow amp var
    # raw_data = dm.get_raw_data(file_id=1711618252292, load_npz=False) #green ampl var
    data = dm.get_raw_data(
        file_stem="2025_09_02-20_28_40-cannon-nv0_2025_08_31", load_npz=True
    )
    process_and_plot(data)
    kpl.show(block=True)
