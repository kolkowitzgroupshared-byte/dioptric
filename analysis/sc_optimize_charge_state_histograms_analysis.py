# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference
Created on Fall 2024
@author: saroj chand
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from utils import data_manager as dm
from utils import kplotlib as kpl


def find_optimal_value_geom_mean(
    step_vals, prep_fidelity, readout_fidelity, goodness_of_fit, weights=(1, 1, 1)
):
    """
    Finds the optimal step value using a weighted geometric mean of fidelities and goodness of fit.

    """
    w1, w2, w3 = weights

    # Remove the first entry from each list
    step_vals = step_vals[2:]
    prep_fidelity = prep_fidelity[2:]
    readout_fidelity = readout_fidelity[2:]
    goodness_of_fit = goodness_of_fit[2:]
    # Normalize metrics (avoid division by zero)
    norm_prep_fidelity = (prep_fidelity - np.nanmin(prep_fidelity)) / (
        np.nanmax(prep_fidelity) - np.nanmin(prep_fidelity) + 1e-12
    )
    norm_readout_fidelity = (readout_fidelity - np.nanmin(readout_fidelity)) / (
        np.nanmax(readout_fidelity) - np.nanmin(readout_fidelity) + 1e-12
    )
    norm_goodness = (goodness_of_fit - np.nanmin(goodness_of_fit)) / (
        np.nanmax(goodness_of_fit) - np.nanmin(goodness_of_fit) + 1e-12
    )
    inverted_goodness = 1 - norm_goodness  # Minimize goodness of fit

    # Compute weighted geometric mean
    # combined_score = (
    #     (norm_readout_fidelity**w1) * (norm_prep_fidelity**w2) * (inverted_goodness**w3)
    # ) ** (1 / (w1 + w2 + w3))
    combined_score = (
        w1 * norm_prep_fidelity + w2 * norm_readout_fidelity + w3 * inverted_goodness
    )
    # Find the step value corresponding to the maximum combined score
    max_index = np.nanargmax(combined_score)
    max_combined_score = combined_score[max_index]
    optimal_step_val = step_vals[max_index]
    optimal_prep_fidelity = prep_fidelity[max_index]
    optimal_readout_fidelity = readout_fidelity[max_index]

    return (
        optimal_step_val,
        optimal_prep_fidelity,
        optimal_readout_fidelity,
        max_combined_score,
    )


def fit_fn(tau, delay, slope, decay):
    """
    Fit function modeling the preparation fidelity as a function of polarization duration.
    """
    tau = np.array(tau) - delay
    return slope * tau * np.exp(-tau / decay)


def process_and_plot(raw_data, do_plot=False):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]
    # a, b, c = 3.7e5, 6.97, 8e-14 # old para
    # a, b, c = 161266.751, 6.617, -19.492  # new para
    # a, b, c = 1.16306103e04, 2.81008145e00, -2.50774288e01  # UPDATED 2025-09-14
    a, b, c = 1.5133e04, 2.6976, -38.63
    # get yellow amplitude
    yellow_charge_readout_amp = raw_data["opx_config"]["waveforms"][
        "yellow_charge_readout"
    ]["sample"]
    green_aod_cw_charge_pol_amp = raw_data["opx_config"]["waveforms"][
        "green_aod_cw-charge_pol"
    ]["sample"]

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
    # prob_dist = ProbDist.COMPOUND_POISSON_WITH_IONIZATION

    # Function to process a single NV and step
    def process_nv_step(nv_ind, step_ind):
        counts_data = condensed_counts[nv_ind, step_ind]

        try:
            # Fit the bimodal histogram and calculate metrics
            popt, pcov, chi_squared = fit_bimodal_histogram(counts_data, prob_dist)

            if popt is None:
                return np.nan, np.nan, np.nan

            # Calculate threshold and fidelities

            threshold, readout_fidelity = determine_threshold(
                popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
            )
            prep_fidelity = 1 - popt[0]  # Population weight of dark state

            return readout_fidelity, prep_fidelity, chi_squared
        except Exception as e:
            print(f"Error processing NV {nv_ind}, step {step_ind}: {e}")
            return np.nan, np.nan, np.nan

    # Parallel processing using all available cores
    results = Parallel(n_jobs=-1)(
        delayed(process_nv_step)(nv_ind, step_ind)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    )

    # Reshape results into a 3D array: (num_nvs, num_steps, 3)
    try:
        results = np.array(results, dtype=float).reshape(num_nvs, num_steps, 3)
    except ValueError as e:
        print(f"Error reshaping results: {e}")
        # Debugging: check the length and structure of `results`
        print(f"Length of results: {len(results)}")
        for i, res in enumerate(results[:10]):  # Print first 10 entries
            print(f"Result {i}: {res}")
        raise

    # Reshape results into arrays
    results = np.array(results).reshape(num_nvs, num_steps, 3)
    readout_fidelity_arr = results[:, :, 0]
    prep_fidelity_arr = results[:, :, 1]
    goodness_of_fit_arr = results[:, :, 2]

    ### Plotting
    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            # step_vals *= 1e-3
            step_vals = step_vals
            x_label = "Polarization duration (ns)"
        else:
            step_vals *= green_aod_cw_charge_pol_amp
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            step_vals *= 1e-6
            x_label = "Readout duration (ms)"
        else:
            step_vals *= yellow_charge_readout_amp
            # x_label = "Readout amplitude"
            step_vals = a * (step_vals**b) + c
            x_label = "Readout amplitude (uW)"
    print(step_vals)
    # return
    # print(step_vals)
    optimal_values = []
    optimal_step_vals = []
    nv_indces = []

    # # Find index of manually set step value
    # manual_step_val = 226.98624137
    # manual_index = np.where(np.isclose(step_vals, manual_step_val))[0][0]

    for nv_ind in range(num_nvs):
        try:
            # Calculate the optimal step value
            (
                optimal_step_val,
                optimal_readout_fidality,
                optimal_prep_fidality,
                max_combined_score,
            ) = find_optimal_value_geom_mean(
                step_vals,
                readout_fidelity_arr[nv_ind],
                prep_fidelity_arr[nv_ind],
                goodness_of_fit_arr[nv_ind],
                weights=(1.0, 1.0, 1.5),
            )
            # Manually override for the first NV
            # if nv_ind == 0:
            #     optimal_step_val = manual_step_val
            #     optimal_prep_fidality = prep_fidelity_arr[nv_ind][manual_index]
            #     optimal_readout_fidality = readout_fidelity_arr[nv_ind][manual_index]
            #     max_combined_score = (
            #         1.5 * optimal_prep_fidality
            #         + 1 * optimal_readout_fidality
            #         + 1 * (1 - goodness_of_fit_arr[nv_ind][manual_index])
            #     )
            optimal_step_vals.append(optimal_step_val)
            nv_indces.append(nv_ind)
            optimal_values.append(
                (
                    nv_ind,
                    optimal_step_val,
                    optimal_prep_fidality,
                    optimal_readout_fidality,
                    max_combined_score,
                )
            )

        except Exception as e:
            print(f"Failed to process NV{nv_ind}: {e}")
            optimal_values.append((nv_ind, np.nan, np.nan))
            continue

        if do_plot:
            # # Plotting
            fig, ax1 = plt.subplots(figsize=(7, 5))
            # Plot readout fidelity
            ax1.plot(
                step_vals,
                readout_fidelity_arr[nv_ind],
                label="Readout Fidelity",
                color="orange",
            )
            ax1.plot(
                step_vals,
                prep_fidelity_arr[nv_ind],
                label="Prep Fidelity",
                linestyle="--",
                color="green",
            )
            ax1.set_xlabel(x_label)
            ax1.set_ylabel("Fidelity")
            ax1.tick_params(axis="y", labelcolor="blue")
            ax1.grid(True, linestyle="--", alpha=0.6)

            # Plot Goodness of Fit ()
            ax2 = ax1.twinx()
            ax2.plot(
                step_vals,
                goodness_of_fit_arr[nv_ind],
                color="gray",
                linestyle="--",
                label=r"Goodness of Fit ($\chi^2_{\text{reduced}}$)",
                alpha=0.7,
            )
            ax2.set_ylabel(r"Goodness of Fit ($\chi^2_{\text{reduced}}$)", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

            # Highlight optimal step value
            ax1.axvline(
                optimal_step_val,
                color="red",
                linestyle="--",
                label=f"Optimal Step Val: {optimal_step_val:.3f}",
            )
            ax2.axvline(
                optimal_step_val,
                color="red",
                linestyle="--",
            )

            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=11)
            ax1.set_title(f"NV{nv_ind} - Optimal Step Val: {optimal_step_val:.3f}")
            plt.tight_layout()
            plt.show(block=True)

    # save opimal step values
    total_power = np.sum(optimal_step_vals) / len(optimal_step_vals)
    aom_voltage = ((total_power - c) / a) ** (1 / b)
    # Compute total power and AOM voltage
    valid_step_vals = [val for val in optimal_step_vals if not np.isnan(val)]
    if not valid_step_vals:
        raise ValueError("No valid step values found.")
    total_power = np.sum(valid_step_vals) / len(valid_step_vals)
    optimal_weigths = valid_step_vals / total_power
    aom_voltage = ((total_power - c) / a) ** (1 / b)
    print(len(optimal_weigths))
    print(list(optimal_weigths))
    results = {
        "optimal_weigths": optimal_weigths,
        "total_power": total_power,
        "aom_voltage": aom_voltage,
    }
    # results = {"nv_indices": nv_indces, "optimal step values": optimal_step_vals}
    timestamp = dm.get_time_stamp()
    file_name = f"optimal_values_{file_id}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)
    # dm.save_raw_data(results, file_path)
    print(results)
    # Get NV indices where readout fidelity is greater than 0.5
    count_above_threshold = sum(
        1 for val in optimal_values if val[2] > 0.4 and val[3] > 0.8
    )
    nv_indices_above_threshold = [
        val[0] for val in optimal_values if val[2] > 0.4 and val[3] > 0.8
    ]

    print(f"Number of optimal values with pre fidelity > 0.5: {count_above_threshold}")
    print("NV indices with prep fidelity > 0.5:", nv_indices_above_threshold)
    print(f"Processed data saved to '{file_path}'.")
    # return
    ### Calculate Averages
    avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0)
    avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0)
    avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0)
    # Calculate the optimal step value
    (
        optimal_step_val,
        optimal_readout_fidelity,
        optimal_prep_fidelity,
        max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        avg_readout_fidelity,
        avg_prep_fidelity,
        avg_goodness_of_fit,
        weights=(1, 1, 1),
    )
    # Plot average readout and prep fidelity
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        step_vals,
        avg_readout_fidelity,
        label="Avg. Readout Fidelity",
        color="orange",
    )
    ax1.plot(
        step_vals,
        avg_prep_fidelity,
        label="Avg. Prep Fidelity",
        color="green",
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Fidelity")
    ax1.tick_params(axis="y")

    # Plot average Goodness of Fit (reduced chi-squared))
    ax2 = ax1.twinx()
    ax2.plot(
        step_vals,
        avg_goodness_of_fit,
        color="gray",
        linestyle="--",
        label=r"Goodness of Fit ($\chi^2_{\text{reduced}}$)",
    )
    ax2.set_ylabel(r"Goodness of Fit ($\chi^2_{\text{reduced}}$)", color="gray")

    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.axvline(
        optimal_step_val,
        color="red",
        linestyle="--",
    )
    optimal_text = (
        f"Optimal {x_label}: {optimal_step_val:.3f}\n"
        f"Optimal Prep Fidelity: {optimal_prep_fidelity:.3f}\n"
        f"Optimal Readout Fidelity: {optimal_readout_fidelity:.3f}"
    )
    ax1.text(
        0.01,
        0.99,
        optimal_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.1", alpha=0.5, edgecolor="gray", facecolor="white"
        ),
    )
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)

    # Title and layout
    ax1.set_title(f"Average Metrics Across All NVs ({file_id})", fontsize=16)
    fig.tight_layout()
    plt.show()
    ### Calculate Medians
    median_readout_fidelity = np.nanmedian(readout_fidelity_arr, axis=0)
    median_prep_fidelity = np.nanmedian(prep_fidelity_arr, axis=0)
    median_goodness_of_fit = np.nanmedian(goodness_of_fit_arr, axis=0)

    (
        optimal_step_val,
        optimal_readout_fidelity,
        optimal_prep_fidelity,
        max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        median_readout_fidelity,
        median_prep_fidelity,
        median_goodness_of_fit,
        weights=(1, 1, 2),
    )

    # Plot average and median readout and prep fidelity
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        step_vals,
        median_readout_fidelity,
        label="Median Readout Fidelity",
        color="orange",
    )
    ax1.plot(
        step_vals,
        median_prep_fidelity,
        label="Median Prep Fidelity",
        color="green",
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Fidelity")
    ax1.tick_params(axis="y")

    # Plot average and median Goodness of Fit (reduced chi-squared))
    ax2 = ax1.twinx()
    ax2.plot(
        step_vals,
        median_goodness_of_fit,
        color="gray",
        linestyle="--",
        label=r"Goodness of Fit ($\chi^2_{\text{reduced}}$)",
    )
    ax2.set_ylabel(r"Goodness of Fit ($\chi^2_{\text{reduced}}$)", color="gray")

    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.axvline(
        optimal_step_val,
        color="red",
        linestyle="--",
    )
    # Add a box with optimal values on the left
    optimal_text = (
        f"Optimal {x_label}: {optimal_step_val:.3f}\n"
        f"Optimal Prep Fidelity: {optimal_prep_fidelity:.3f}\n"
        f"Optimal Readout Fidelity: {optimal_readout_fidelity:.3f}"
    )
    ax1.text(
        0.01,
        0.99,
        optimal_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.1", alpha=0.5, edgecolor="black", facecolor="white"
        ),
    )
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)

    # Title and layout
    ax1.set_title(f"Median Metrics Across All NVs ({file_id})", fontsize=16)
    fig.tight_layout()
    plt.show(block=False)


def fit_fn(tau, delay, slope, decay, transition):
    """
    Fit function modeling the preparation fidelity as a function of polarization duration.
    Smoothly transitions from an initial linear increase to an exponential decay.
    """
    tau = np.array(tau) - delay

    # Sigmoid-like transition function (soft transition)
    smooth_transition = 1 / (1 + np.exp(-(tau - transition) / (0.1 * transition)))

    # Linear rise component
    linear_part = slope * tau

    # Exponential decay component
    exp_part = slope * transition * np.exp(-(tau - transition) / decay)

    # Combine both with a smooth transition
    return (1 - smooth_transition) * linear_part + smooth_transition * exp_part


def fit_fn(tau, delay, slope, decay, transition):
    """
    Fit function modeling the preparation fidelity as a function of polarization duration.
    Ensures an initial steep increase followed by an exponential decay.
    """
    tau = np.array(tau) - delay
    tau = np.maximum(tau, 0)  # Ensure no negative time values

    # Enforce a minimum transition duration (e.g., 50 ns)
    transition = max(transition, 90)

    # Smooth transition function using tanh
    smooth_transition = 0.5 * (1 + np.tanh((tau - transition) / (0.05 * transition)))

    # Enforce an initial steep rise
    linear_part = slope * tau

    # Exponential decay component
    exp_part = slope * transition * np.exp(-(tau - transition) / decay)

    # Combine both components smoothly
    return (1 - smooth_transition) * linear_part + smooth_transition * exp_part


def fit_fn(tau, delay, slope, decay, transition):
    """
    Fit function modeling the preparation fidelity as a function of polarization duration.
    Ensures an initial steep increase followed by an exponential decay.
    """
    tau = np.array(tau) - delay
    tau = np.maximum(tau, 0)  # Ensure no negative time values

    # Enforce a minimum transition duration (e.g., 90 ns)
    # transition = max(transition, 48)
    transition = 400

    # Smooth transition function using tanh
    smooth_transition = 0.5 * (1 + np.tanh((tau - transition) / (0.6 * transition)))

    # Enforce an initial steep rise
    linear_part = slope * tau

    # Exponential decay component
    # exp_part = slope * transition * np.exp(-(tau - transition) / decay)
    exp_part = slope * transition * np.exp(-(tau - transition) / (2 * decay))

    # Combine both components smoothly
    return (1 - smooth_transition) * linear_part + smooth_transition * exp_part


def process_nv_step(nv_ind, step_ind, condensed_counts):
    counts_data = condensed_counts[nv_ind, step_ind]
    try:
        popt, pcov, chi_squared = fit_bimodal_histogram(
            counts_data, ProbDist.COMPOUND_POISSON
        )
        if popt is None:
            return np.nan, np.nan, np.nan
        threshold, readout_fidelity = determine_threshold(
            popt, ProbDist.COMPOUND_POISSON, dark_mode_weight=0.5, ret_fidelity=True
        )
        prep_fidelity = 1 - popt[0]  # Population weight of dark state
        return readout_fidelity, prep_fidelity, chi_squared
    except Exception as e:
        print(f"Error processing NV {nv_ind}, step {step_ind}: {e}")
        return np.nan, np.nan, np.nan


def process_and_plot_charge(raw_data, do_plot=False):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals = np.linspace(min_step_val, max_step_val, num_steps)

    counts = np.array(raw_data["counts"])
    ref_exp_ind = 1
    condensed_counts = np.array(
        [
            [
                counts[ref_exp_ind, nv_ind, :, step_ind, :].flatten()
                for step_ind in range(num_steps)
            ]
            for nv_ind in range(num_nvs)
        ]
    )

    # Process each NV-step pair in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_nv_step)(nv_ind, step_ind, condensed_counts)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    )

    try:
        results = np.array(results, dtype=float).reshape(num_nvs, num_steps, 3)
    except ValueError as e:
        print(f"Error reshaping results: {e}")
        return

    prep_fidelity = results[:, :, 1]
    readout_fidelity = results[:, :, 0]
    ### **Perform Fitting**
    opti_durs, opti_fidelities = [], []

    # --- Saturation models (with offset) ---
    def sat_decay_fit_fn(t, F0, A, t0, tau_r, tau_d):
        t = np.asarray(t, dtype=float)
        x = np.maximum(t - t0, 0.0)  # gate before t0
        tau_r = np.maximum(tau_r, 1e-12)
        tau_d = np.maximum(tau_d, 1e-12)
        return F0 + 2 * A * (1.0 - np.exp(-x / tau_r)) * np.exp(-x / tau_d)
        # return A * (1.0 - np.exp(-x / tau_r)) * np.exp(-x / tau_d)

    def sat_decay_x_peak(tau_r, tau_d):
        tau_r = max(float(tau_r), 1e-12)
        tau_d = max(float(tau_d), 1e-12)
        return tau_r * np.log(1.0 + tau_d / tau_r)

    # --- Robust initial guesses + bounds ---
    def sat_decay_initial_guess(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        span = float(x[-1] - x[0]) if x[-1] > x[0] else 1.0
        dt = np.median(np.diff(x))

        # Baseline & amplitude
        F0_0 = float(np.nanpercentile(y, 5))
        ymax = float(np.nanpercentile(y, 95))
        A_0 = float(max(1e-4, ymax - F0_0))  # if fidelity, cap elsewhere if you want

        # Onset t0 near strongest rise
        dy = np.diff(y, prepend=y[0])
        i_rise = int(np.clip(np.argmax(dy), 0, len(x) - 1))
        t0_0 = float(max(x[0], x[i_rise] - 0.5 * dt))

        # Time constants: start with τd >> τr so peak isn't too early
        tau_r0 = max(dt, 0.15 * span)
        tau_d0 = max(5 * tau_r0, 1.0 * span)

        p0 = [F0_0, A_0, t0_0, tau_r0, tau_d0]

        # Bounds (adjust if your y is guaranteed in [0,1])
        F0_lo, F0_hi = min(y) - 0.2 * abs(y).max(), max(y) + 0.2 * abs(y).max()
        A_lo, A_hi = 0.0, max(1.5 * (ymax - F0_0), 1e-3)
        t0_lo, t0_hi = x[0] - 2 * span, x[-1] + 2 * span
        tr_lo, tr_hi = dt / 10, 2 * span
        td_lo, td_hi = dt / 10, 10 * span

        lo = [F0_lo, A_lo, t0_lo, tr_lo, td_lo]
        hi = [F0_hi, A_hi, t0_hi, tr_hi, td_hi]
        return p0, (lo, hi)

    for nv_ind in range(num_nvs):
        r = readout_fidelity[nv_ind].astype(float)
        y = prep_fidelity[nv_ind].astype(float)
        x = step_vals.astype(float)
        # Clean
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x_f, y_f = x[m], y[m]
        if x_f.size < 4:
            raise RuntimeError("Not enough points for fit.")

        p0, bounds = sat_decay_initial_guess(x_f, y_f)
        popt, pcov = curve_fit(
            sat_decay_fit_fn, x_f, y_f, p0=p0, bounds=bounds, maxfev=200000
        )
        F0, A, t0, tau_r, tau_d = popt

        # analytic peak
        x_pk_rel = sat_decay_x_peak(tau_r, tau_d)
        t_peak = float(t0 + x_pk_rel)
        y_peak = float(sat_decay_fit_fn(t_peak, *popt))

        results = {"params": popt, "cov": pcov, "t_peak": t_peak, "y_peak": y_peak}

        return_curve = True
        grid = None
        if return_curve:
            if grid is None:
                grid = np.linspace(x_f.min(), x_f.max(), 1000)
            y_model = sat_decay_fit_fn(grid, *popt)
            results.update({"grid_t": grid, "grid_y": y_model})

        F0, A, t0, tau_r, tau_d = results["params"]
        opti_dur = float(np.clip(results["t_peak"], min_step_val, max_step_val))
        opti_fid = float(results["y_peak"])
        opti_durs.append(round(opti_dur / 4) * 4)
        opti_fidelities.append(round(opti_fid, 3))

        # Snap to hardware grid
        opti_durs.append(round(opti_dur / 4) * 4)
        opti_fidelities.append(round(opti_fid, 3))

        if do_plot:
            # --- Plot ---
            plt.figure(figsize=(6, 5))
            plt.scatter(x_f, y_f, label="Measured")
            plt.plot(results["grid_t"], results["grid_y"], label="Sat-Decay Fit")
            plt.axvline(
                opti_dur,
                color="green",
                linestyle="--",
                label=f"Peak ≈ {opti_dur:.0f} ns",
            )
            plt.scatter([opti_dur], [opti_fid], color="green", zorder=5)
            plt.xlabel("Duration (ns)")
            plt.ylabel("Fidelity")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show(block=True)

    if opti_durs:
        print("Optimal Polarization Durations:", opti_durs)

        # Filter out None values to compute median
        numeric_durations = [d for d in opti_durs if d is not None]
        median_duration = int(np.nanmedian(numeric_durations))
        # Replace None or out-of-range values with median
        opti_durs = [
            (
                median_duration
                if (d is None or (100 <= d <= 200) or (1930 <= d <= 2000))
                else d
            )
            for d in opti_durs
        ]

        print("Updated Optimal Durations:", opti_durs)
        # print("Optimal Preparation Fidelities:", opti_fidelities)
        print(f"Median Optimal Duration: {np.median(opti_durs)} ns")
        print(f"Median Optimal Fidelity: {np.median(opti_fidelities)}")
        print(f"Max Optimal Duration: {np.max(opti_durs)} ns")
        print(f"Min Optimal Duration: {np.min(opti_durs)} ns")
        ###
        plt.figure()
        plt.scatter(opti_durs, opti_fidelities)
        plt.xlabel("Polarization Duration (ns)")
        plt.ylabel("Preparation Fidelity")
        plt.title(f"NV Num: {nv_ind}")
        plt.legend()
        plt.show(block=True)

    return


if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_id = 1710843759806
    # file_id = 1712782503640  # yellow ampl var 50ms 160NVs
    # file_id = 1723230400688  # yellow ampl var 30ms
    # file_id = 1717056176426  # yellow duration var
    # file_id = 1711618252292  # green ampl var
    # file_id = 1712421496166  # green ampl var
    # file_id = 1720970373150  # yellow ampl var iter_1
    # file_id = 1731890617395  # green ampl var after new calinration
    # file_id = 1732177472880  # yellow ampl var 50ms 117NVs
    # file_id = 1751274459725  # yellow ampl var 50ms 117NVs afdter birge
    # file_id = 1751404919855  # yellow ampl var 50ms 117NVs afdter birge
    # file_id = 1752870018575  # yellow ampl var 50ms 117NVs afdter birge
    # file_id = 1752968732835  # green ampl var
    # file_id = 1770658719969  # readout yellow ampl var 50ms shallow nvs
    # file_id = 1776463838159  # yellow ampl var 50ms shallow nvs
    # file_id = 1766460747869  # yellow ampl var 50ms shallow nvs
    # file_id = 1780914190838  # yellow ampl var 60ms shallow nvs
    # file_id = 1782586909215  # yellow ampl 60ms shallow nvs (selected two orieantation)
    # file_id = 1766460747869  # yellow ampl var 50ms shallow 148nvs
    # file_id = 1780914190838  # yellow ampl var 60ms shallow 148nvs
    # file_id = 1782586909215  # yellow ampl var 60ms shallow 69nvs

    # file_id = 1767789140438  # pol dur var 200ns to 2us
    # file_id = 1768024979194  # pol dur var 100ns to 1us
    # file_id = 1769942144688  # pol dur var 100ns to 1us dataset
    # file_id = 1767789140438  # pol dur var 200ns to 2us
    # file_id = 1768024979194  # pol dur var 100ns to 1us
    # file_id = 1769942144688  # pol dur var 100ns to 1us dataset
    # file_id = 1770306530123  # pol dur var 16ns to 1028ns dataset 148NVs
    # file_id = 1770306530123  # pol dur var 16ns to 1028ns dataset 148NVs
    # file_id = 1778627435145 # pol dur var 16ns to 1028ns dataset 148NVs
    # file_id = 1778426682976  # green ampl var 60ms shallow nvs
    # file_id = 1778524699003  # green ampl var 60ms shallow nvs (large range)
    # file_id = 1780495024088  # green ampl var 60ms shallow nvs (large range
    # raw_data = dm.get_raw_data(file_id=1709868774004, load_npz=False) #yellow ampl var

    # rubin sample
    # file_id = 1791404857391  # yellow ampl 50ms
    # file_id = 1792169039756  # yellow ampl 50ms
    # file_id = 1793934866370  # yellow ampl 60ms 140NVs
    # file_id = 1794442033227  # yellow ampl 60ms 140NVs
    # file_id = 1800416464270  # yellow ampl 60ms 107NVs

    # file_id = 1802933502486  # yellow ampl 60ms 300NVs
    # file_id = 1804148043654  # yellow ampl 60ms 300NVs
    # file_id = 1804934228627 # yellow ampl 60ms 200NVs
    # file_id = 1805383839845  # yellow ampl 60ms 240NVs
    # file_id = 1806165457282  # yellow ampl 60ms 154NVs
    # file_id = 1807233914030  # yellow ampl 60ms 81NVs
    # file_id = 1807632138996  # yellow ampl 60ms 81NVs
    # file_id = 1808503038483  # yellow ampl 60ms 81NVs
    # file_id = 1809414309242  # yellow ampl 60ms 81NVs
    # file_id = 1834021972039  # yellow ampl 60ms 75NVs

    # file_id = "2025_03_05-05_04_57-rubin-nv0_2025_02_26"  # 1794442033227  # yellow ampl 60ms 140NVs
    # file_id = 1793116636570  # yellow ampl 24ms
    # file_id = 1792980892323  # yellow ampl 80ms
    # file_id = 1791756537192  # green durations
    # file_id = 1794216207756  # green durations 60ms 140NVs
    # file_id = 1791914648483  # green amps

    # file_id = 1800302862093  # green amps 107 NVs
    # file_id = 1801385197244  # green durations 60ms
    # file_id = 1803091794064  # green durations 303NVs
    # file_id = 1805189336738  # green durations 203NVs
    # file_id = 1805991515134  # green durations 240NVs
    # file_id = 1806362913488  # green durations 154NVs
    # file_id = 1807384237764  # green durations 81NVs
    # file_id = 1810477160439  # green durations 75NVs
    # file_id = 1810477160439  # green durations 75NVs
    # file_id = 1833010688783  # green durations 75NVs (4/13/2025)
    # file_id = 1834390490156  # green durations 75NVs (4/14/2025)
    # file_id = 1836625491633  # green durations 75NVs (4/16/2025)
    # file_id = (
    #     "2025_04_29-00_04_37-rubin-nv0_2025_02_26"  # green amplitude 75NVs (4/16/2025)
    # )
    # file_id = (
    #     "2025_05_12-21_22_21-rubin-nv0_2025_02_26"  # green amplitude 75NVs (4/16/2025)
    # )
    ### readout amp
    # file_stem = "2025_09_11-01_45_11-rubin-nv0_2025_09_08"  #
    # file_stem = "2025_09_11-23_23_30-rubin-nv0_2025_09_08"
    # file_id = "2025_09_13-20_27_20-rubin-nv0_2025_09_08"
    # file_id = "2025_09_119-06_48_20-rubin-nv0_2025_09_08"
    # file_id = "2025_09_19-22_37_11-rubin-nv0_2025_09_08"
    # file_id = "2025_09_28-04_31_09-rubin-nv0_2025_09_08"
    # file_id = "2025_09_28-22_59_27-rubin-nv0_2025_09_08"
    # file_id = "2025_10_01-17_07_50-rubin-nv0_2025_09_08"
    # file_id = "2025_10_08-19_20_50-rubin-nv0_2025_09_08"
    # file_id = "2025_10_08-19_20_50-rubin-nv0_2025_09_08"
    # file_id = "2025_10_22-07_09_20-rubin-nv0_2025_09_08"
    # file_id = "2025_10_22-09_46_22-johnson-nv0_2025_10_21"
    # file_id = "2025_10_22-13_38_22-johnson-nv0_2025_10_21"
    # file_id = "2025_10_23-02_24_51-johnson-nv0_2025_10_21"
    # file_id = "2025_10_26-16_36_03-johnson-nv0_2025_10_21"
    # file_id = "2025_10_30-06_21_14-johnson-nv0_2025_10_21"
    # file_id = "2026_02_01-00_09_17-johnson-nv0_2025_10_21"
    # file_id = "2026_02_07-16_42_58-johnson-nv0_2025_10_21"
    # file_id = "2026_02_09-21_28_39-johnson-nv0_2025_10_21"
    # file_id = "2026_02_16-21_30_52-rubin-nv0_2026_02_15"
    # file_id = "2026_03_03-02_22_47-qnami-nv0_2026_02_20"
    # file_id = "2026_03_18-03_53_55-qnami-nv0_2026_02_20"
    # file_id = "2026_03_18-11_46_55-qnami-nv0_2026_02_20"
    file_id = "2026_03_20-18_58_49-qnami-nv0_2026_02_20"
    

    ### pol amp var
    # file_id = "2025_09_12-16_53_34-rubin-nv0_2025_09_08"
    # file_id = "2025_09_12-18_30_09-rubin-nv0_2025_09_08"
    # file_id = "2025_09_12-20_43_54-rubin-nv0_2025_09_08"
    # file_id = "2025_09_19-03_41_13-rubin-nv0_2025_09_08"
    # file_id = "2025_09_20-03_38_25-rubin-nv0_2025_09_08"  # 10us
    # file_id = "2025_09_20-14_18_23-rubin-nv0_2025_09_08"  # 1us
    # file_id = "2025_09_28-20_18_06-rubin-nv0_2025_09_08"  # 1us
    # file_id = "2025_10_22-18_10_57-johnson-nv0_2025_10_21"
    # file_id = "2026_02_10-00_47_07-johnson-nv0_2025_10_21"

    ### pol dur var
    # file_id = "2025_09_12-04_47_45-rubin-nv0_2025_09_08"
    # file_id = "2025_09_13-00_31_26-rubin-nv0_2025_09_08"
    # file_id = "2025_09_13-02_58_29-rubin-nv0_2025_09_08"
    # file_id = "2025_09_14-21_59_00-rubin-nv0_2025_09_08"
    # file_id = "2025_09_18-12_06_11-rubin-nv0_2025_09_08"
    # file_id = "2025_09_18-16_19_05-rubin-nv0_2025_09_08"
    # file_id = "2025_09_19-11_56_40-rubin-nv0_2025_09_08"
    # file_id = "2025_09_21-22_20_08-rubin-nv0_2025_09_08"
    # file_id = "2025_09_23-19_06_00-rubin-nv0_2025_09_08"
    # file_id = "2025_09_28-00_14_24-rubin-nv0_2025_09_08"
    # file_id = "2025_09_28-22_59_27-rubin-nv0_2025_09_08"
    # file_id = "2025_10_23-15_48_46-johnson-nv0_2025_10_21"
    # file_id = "2025_10_26-20_37_42-johnson-nv0_2025_10_21"
    # file_id = "2025_10_30-18_37_28-johnson-nv0_2025_10_21"
    # file_id = "2025_11_01-19_02_31-johnson-nv0_2025_10_21"
    # file_id = "2025_11_22-19_58_10-johnson-nv0_2025_10_21"
    # file_id = "2026_02_07-20_07_20-johnson-nv0_2025_10_21"
    # file_id = "2026_02_10-03_18_17-johnson-nv0_2025_10_21"
    # file_id = "2026_03_02-20_10_14-qnami-nv0_2026_02_20"
    # file_id = "2026_03_17-06_00_50-qnami-nv0_2026_02_20"

    # dm.USE_NEW_CLOUD = False
    raw_data = dm.get_raw_data(file_stem=file_id, load_npz=True)
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}")
    process_and_plot(raw_data, do_plot=False)
    # process_and_plot_charge(raw_data, do_plot=True)
    # print(dm.get_file_name(1717056176426))
    plt.show(block=True)