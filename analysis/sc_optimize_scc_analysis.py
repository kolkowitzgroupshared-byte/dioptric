# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

@author: saroj chand
"""


# def fit_snr(taus, avg_snr_nv, avg_snr_ste_nv):
#     """Fit SNR data to a custom function."""

#     def fit_fn(tau, delay, slope, decay):
#         tau = np.array(tau) - delay
#         return slope * tau * np.exp(-tau / decay)

#     guess_params = [taus[0], np.max(avg_snr_nv), taus[-1]]
#     try:
#         popt, _ = curve_fit(
#             fit_fn,
#             taus,
#             avg_snr_nv,
#             p0=guess_params,
#             sigma=avg_snr_ste_nv,
#             absolute_sigma=True,
#             maxfev=10000,  # Increase the max number of iterations
#         )
#     except Exception as e:
#         print(f"Fitting failed for this NV: {e}")
#         popt = [taus[0], 0, taus[-1]]  # Default fallback parameters
#     return popt, fit_fn


# def plot_individual_nv_fits(nv_list, taus, avg_snr, avg_snr_ste):
#     """Create separate figures for individual NV SNR fits."""
#     figs = []  # Store all figures for later reference
#     optimal_durations = {}  # Store all opitmal
#     valid_range = (100, 240)
#     optimal_taus = []
#     for nv_ind in range(len(nv_list)):
#         fig, ax = plt.subplots(figsize=(8, 6))  # Create a new figure for each NV
#         popt, fit_fn = fit_snr(taus, avg_snr[nv_ind], avg_snr_ste[nv_ind])
#         tau_linspace = np.linspace(min(taus), max(taus), 1000)
#         # Find the tau corresponding to the max SNR
#         snr_values = fit_fn(tau_linspace, *popt)
#         optimal_tau = tau_linspace[np.argmax(snr_values)]
#         # Ensure optimal_tau is within the valid range
#         if not valid_range[0] <= optimal_tau <= valid_range[1]:
#             optimal_tau = np.nan  # Mark invalid values for later adjustment
#         # Round optimal_tau to the nearest number divisible by 4
#         if not np.isnan(optimal_tau):
#             optimal_tau = round(optimal_tau / 4) * 4
#         # Add to the list for median calculation
#         optimal_taus.append(optimal_tau)
#         # Add the NV-specific duration to the dictionary
#         nv_num = widefield.get_nv_num(nv_list[nv_ind])
#         optimal_durations[nv_num] = optimal_tau
#         # Plot the fit curve
#         sns.lineplot(
#             x=tau_linspace,
#             y=fit_fn(tau_linspace, *popt),
#             label="Fit",
#             ax=ax,
#         )
#         # Plot the data points
#         sns.scatterplot(
#             x=taus,
#             y=avg_snr[nv_ind],
#             ax=ax,
#             label="Data",
#             s=60,
#         )
#         # Customize the plot
#         nv_num = widefield.get_nv_num(nv_list[nv_ind])
#         ax.set_title(f"NV {nv_num} SNR Fit")
#         ax.set_xlabel("SCC Pulse Duration (ns)")
#         ax.set_ylabel("SNR")
#         ax.legend()
#         ax.grid(True)
#         figs.append(fig)  # Append the created figure to the li
#     # Calculate the median of all valid optimal taus
#     valid_taus = [tau for tau in optimal_taus if not np.isnan(tau)]
#     median_tau = np.median(valid_taus)
#     # Replace any invalid optimal_tau values with the median
#     for nv_num, tau in optimal_durations.items():
#         if np.isnan(tau) or not valid_range[0] <= tau <= valid_range[1]:
#             optimal_durations[nv_num] = 240
#     return figs, optimal_durations


# def process_and_plot(nv_list, taus, sig_counts, ref_counts, duration_or_amp):
#     """Process and plot data for signal, reference, and SNR."""
#     # Filter NVs by selected orientations
#     num_nvs = len(nv_list)
#     orientation_data = dm.get_raw_data(file_id=1723161184641)
#     orientation_indices = orientation_data["orientation_indices"]
#     selected_orientations = ["0.041", "0.147"]
#     selected_indices = []
#     for orientation in selected_orientations:
#         if str(orientation) in orientation_indices:
#             selected_indices.extend(orientation_indices[str(orientation)]["nv_indices"])
#     selected_indices = list(set(selected_indices))  # Remove duplicates
#     # Filter counts and NV list
#     nv_list = [nv_list[i] for i in selected_indices]
#     sig_counts = sig_counts[selected_indices, :, :, :]
#     ref_counts = ref_counts[selected_indices, :, :, :]
#     # Average counts and calculate metrics
#     # avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
#     # avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)
#     avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
#     # Average and Median SNR
#     avg_snr_all = np.mean(avg_snr, axis=0)
#     median_snr_all = np.median(avg_snr, axis=0)
#     avg_snr_ste_all = np.mean(avg_snr_ste, axis=0)
#     ### plot region
#     if duration_or_amp:
#         x_label = "duration(ns)"
#         y_label = "snr"
#     else:
#         x_label = "amplitude"
#         y_label = "snr"
#     fig_avg_snr, ax_avg_snr = plt.subplots()
#     sns.lineplot(x=taus, y=avg_snr_all, ax=ax_avg_snr, label="Average SNR")
#     sns.lineplot(
#         x=taus, y=median_snr_all, ax=ax_avg_snr, label="Median SNR", linestyle="--"
#     )
#     ax_avg_snr.fill_between(
#         taus,
#         avg_snr_all - avg_snr_ste_all,
#         avg_snr_all + avg_snr_ste_all,
#         alpha=0.2,
#         label="Error Bounds",
#     )
#     ax_avg_snr.set_xlabel(x_label)
#     ax_avg_snr.set_ylabel(y_label)
#     ax_avg_snr.legend()
#     ax_avg_snr.grid(True)
#     plt.title("Avg and Median SNR across NVs")
#     fig_snr_fits, optimal_durations = plot_individual_nv_fits(
#         nv_list, taus, avg_snr, avg_snr_ste
#     )
#     print(f"optimal_durations =  {optimal_durations}")
#     return fig_avg_snr, fig_snr_fits, optimal_durations

# if __name__ == "__main__":
#     kpl.init_kplotlib()
#     # Load data
#     data = dm.get_raw_data(file_id=1722305531191)  # duration
#     data = dm.get_raw_data(file_id=1724491290147)  # amplitide
#     nv_list = data["nv_list"]
#     taus = data["taus"]
#     counts = np.array(data["counts"])
#     sig_counts = counts[0]
#     ref_counts = counts[1]
#     # Process and plot
#     figs = process_and_plot(nv_list, taus, sig_counts, ref_counts, duration_or_amp=True)
#     # Show plots
#     plt.show(block=True)
import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield
from utils import widefield as widefield


def fit_duration(taus, avg_snr_nv, avg_snr_ste_nv):
    """Fit SNR data to a custom function."""

    def fit_fn(tau, delay, slope, decay):
        tau = np.array(tau) - delay
        return slope * tau * np.exp(-tau / decay)

    guess_params = [taus[0], np.max(avg_snr_nv), taus[-1]]
    try:
        popt, _ = curve_fit(
            fit_fn,
            taus,
            avg_snr_nv,
            p0=guess_params,
            sigma=avg_snr_ste_nv,
            absolute_sigma=True,
            maxfev=20000,  # Increase the max number of iterations
        )
    except Exception as e:
        print(f"Fitting failed for this NV: {e}")
        popt = [taus[0], 0, taus[-1]]  # Default fallback parameters
    return popt, fit_fn


def fit_amplitude(taus, snr_data, snr_ste):
    """Fit SNR data to a generalized logistic function."""

    def fit_fn(P, F_min, F_max, P_mid, steepness, asymmetry):
        """Generalized logistic function for asymmetric fitting."""
        P = np.maximum(P, 1e-10)
        return F_min + (F_max - F_min) / (
            1 + asymmetry * np.exp(-steepness * (P - P_mid))
        )

    # Initial guess for the parameters
    guess_params = [
        np.min(snr_data),  # F_min
        np.max(snr_data),  # F_max
        np.median(taus),  # P_mid (point of maximum growth)
        1.0,  # steepness
        1.0,  # asymmetry
    ]

    # Set bounds to prevent saturation
    bounds = (
        [np.min(snr_data), np.min(snr_data), 1.4, 0.1, 0.0],  # Lower bounds
        [np.max(snr_data), np.max(snr_data), 1.8, 5.0, 2.0],  # Upper bounds
    )

    try:
        # Perform the fit
        popt, _ = curve_fit(
            fit_fn,
            taus,
            snr_data,
            p0=guess_params,
            sigma=snr_ste,
            bounds=bounds,
            absolute_sigma=True,
            maxfev=10000,
        )
    except Exception as e:
        print(f"Fitting failed for amplitude data: {e}")
        popt = guess_params  # Use fallback parameters

    return popt, fit_fn


def process_and_plot(nv_list, duration_file_id, amp_file_id):
    """Process NV data for duration and amplitude optimization."""
    total_nvs = len(nv_list)
    optimal_durations = {nv: None for nv in range(total_nvs)}
    optimal_amplitudes = {nv: None for nv in range(total_nvs)}

    # Common: Filter NVs by selected orientations
    # orientation_data = dm.get_raw_data(file_id=1723161184641)
    # orientation_indices = orientation_data["orientation_indices"]
    # selected_orientations = ["0.041", "0.147"]
    # selected_indices = [
    #     idx
    #     for orientation in selected_orientations
    #     if orientation in orientation_indices
    #     for idx in orientation_indices[orientation]["nv_indices"]
    # ]
    # selected_indices = list(set(idx for idx in selected_indices if idx < total_nvs))
    selected_indices = range(total_nvs)

    def optimize_step_vals(file_id, fit_function, valid_range, duration_or_amp=False):
        data = dm.get_raw_data(file_id=file_id)
        taus, counts = data["taus"], np.array(data["counts"])
        sig_counts, ref_counts = (
            counts[0][selected_indices],
            counts[1][selected_indices],
        )
        avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

        optimal_values = {}
        for i, nv_ind in enumerate(selected_indices):
            try:
                popt, fit_fn = fit_function(taus, avg_snr[i], avg_snr_ste[i])
                tau_linspace = np.linspace(min(taus), max(taus), 1000)
                snr_values = fit_fn(tau_linspace, *popt)
                optimal_value = tau_linspace[np.argmax(snr_values)]

                # Apply constraints based on whether it's duration or amplitude
                if duration_or_amp:
                    # Round to nearest multiple of 4
                    optimal_value = max(
                        valid_range[0],
                        min(valid_range[1], round(optimal_value / 4) * 4),
                    )
                else:
                    # Keep as floating point within the valid range
                    optimal_value = max(
                        valid_range[0], min(valid_range[1], optimal_value)
                    )

                optimal_values[nv_ind] = optimal_value
            except Exception as e:
                print(f"Fitting failed for NV index {nv_ind}: {e}")
                optimal_values[nv_ind] = None  # Mark as unprocessed

        return optimal_values, avg_snr, avg_snr_ste, taus

    # Optimize durations
    duration_valid_range = (60, 240)
    optimal_durations, avg_snr, avg_snr_ste, taus = optimize_step_vals(
        duration_file_id, fit_duration, duration_valid_range, duration_or_amp=True
    )

    # Optimize amplitudes
    amp_valid_range = (0.8, 1.2)
    # amp_valid_range = (np.min(taus), np.max(taus))
    optimal_amplitudes, avg_snr, avg_snr_ste, taus = optimize_step_vals(
        amp_file_id, fit_duration, amp_valid_range, duration_or_amp=False
    )

    # Replace unprocessed NVs with medians
    valid_durations = [
        v
        for k, v in optimal_durations.items()
        if k in selected_indices and v is not None
    ]
    median_duration = np.median(valid_durations) if valid_durations else 0
    for nv_index in range(total_nvs):
        if optimal_durations.get(nv_index) is None:
            optimal_durations[nv_index] = median_duration

    valid_amplitudes = [
        v
        for k, v in optimal_amplitudes.items()
        if k in selected_indices and v is not None
    ]
    median_amplitude = np.median(valid_amplitudes) if valid_amplitudes else 0
    print(median_amplitude)
    for nv_index in range(total_nvs):
        if optimal_amplitudes.get(nv_index) is None:
            optimal_amplitudes[nv_index] = median_amplitude
    # Sort optimal_durations by index (key)
    sorted_optimal_durations = dict(sorted(optimal_durations.items()))
    sorted_optimal_amplitudes = dict(sorted(optimal_amplitudes.items()))

    # Update results
    results = {
        "optimal_durations": sorted_optimal_durations,
        "optimal_amplitudes": sorted_optimal_amplitudes,
    }

    timestamp = dm.get_time_stamp()
    file_name = "optimal_durations"
    file_path = dm.get_file_path(__file__, timestamp, file_name)
    # dm.save_raw_data(results, file_path)

    # Plot medians and means
    avg_snr_all = np.mean(avg_snr, axis=0)
    median_snr_all = np.median(avg_snr, axis=0)
    avg_snr_ste_all = np.mean(avg_snr_ste, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(x=taus, y=avg_snr_all, ax=ax, label="Average SNR")
    sns.lineplot(x=taus, y=median_snr_all, ax=ax, label="Median SNR", linestyle="--")
    ax.fill_between(
        taus,
        avg_snr_all - avg_snr_ste_all,
        avg_snr_all + avg_snr_ste_all,
        alpha=0.2,
        label="Error Bounds",
    )
    ax.set_xlabel("step vals")
    ax.set_ylabel("SNR")
    ax.legend()
    ax.grid(True)
    plt.title("Median and Average SNR across NVs")
    plt.show()

    return results


# def process_and_plot_amplitudes(nv_list, amp_file_id):
#     """Process NV data for amplitude optimization."""
#     total_nvs = len(nv_list)
#     optimal_amplitudes = {nv: None for nv in range(total_nvs)}
#     optimal_snrs = {nv: None for nv in range(total_nvs)}

#     selected_indices = range(total_nvs)

#     def optimize_amplitudes(file_id, fit_function, valid_range):
#         data = dm.get_raw_data(file_id=file_id)
#         taus, counts = data["taus"], np.array(data["counts"])
#         sig_counts, ref_counts = (
#             counts[0][selected_indices],
#             counts[1][selected_indices],
#         )
#         avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

#         optimal_values = {}
#         snr_values = {}
#         for i, nv_ind in enumerate(selected_indices):
#             try:
#                 popt, fit_fn = fit_function(taus, avg_snr[i], avg_snr_ste[i])
#                 tau_linspace = np.linspace(min(taus), max(taus), 1000)
#                 snr_values_curve = fit_fn(tau_linspace, *popt)
#                 optimal_value = tau_linspace[np.argmax(snr_values_curve)]

#                 # Keep the value within the valid range
#                 optimal_value = max(valid_range[0], min(valid_range[1], optimal_value))

#                 optimal_values[nv_ind] = optimal_value
#                 snr_values[nv_ind] = max(snr_values_curve)  # Optimal SNR
#             except Exception as e:
#                 print(f"Fitting failed for NV index {nv_ind}: {e}")
#                 optimal_values[nv_ind] = None  # Mark as unprocessed
#                 snr_values[nv_ind] = None

#         return optimal_values, snr_values, avg_snr, avg_snr_ste, taus

#     # Optimize amplitudes
#     amp_valid_range = (0, 400)
#     optimal_amplitudes, optimal_snrs, avg_snr, avg_snr_ste, taus = optimize_amplitudes(
#         amp_file_id, fit_duration, amp_valid_range
#     )

#     # Replace unprocessed NVs with medians
#     valid_amplitudes = [
#         v
#         for k, v in optimal_amplitudes.items()
#         if k in selected_indices and v is not None
#     ]
#     median_amplitude = np.median(valid_amplitudes) if valid_amplitudes else 0
#     for nv_index in range(total_nvs):
#         if optimal_amplitudes.get(nv_index) is None:
#             optimal_amplitudes[nv_index] = median_amplitude

#     # Plot individual NV fits
#     for nv_index in selected_indices:
#         plt.figure(figsize=(6, 4))
#         plt.errorbar(
#             taus,
#             avg_snr[nv_index],
#             yerr=avg_snr_ste[nv_index],
#             fmt="o",
#             label="SNR Data",
#         )
#         if optimal_amplitudes[nv_index] is not None:
#             tau_linspace = np.linspace(min(taus), max(taus), 1000)
#             popt, fit_fn = fit_duration(taus, avg_snr[nv_index], avg_snr_ste[nv_index])
#             plt.plot(
#                 tau_linspace,
#                 fit_fn(tau_linspace, *popt),
#                 label="Fitted Curve",
#             )
#             plt.axvline(
#                 optimal_amplitudes[nv_index],
#                 color="r",
#                 linestyle="--",
#                 label=f"Optimal Amp: {optimal_amplitudes[nv_index]:.2f}",
#             )
#         plt.title(f"NV {nv_index} - Amplitude Optimization")
#         plt.xlabel("Amplitude")
#         plt.ylabel("SNR")
#         plt.legend()
#         plt.grid(alpha=0.3)
#         plt.show()

#     # Print lists of amplitudes and SNRs
#     print("Optimal Amplitudes:")
#     print([optimal_amplitudes[nv] for nv in selected_indices])

#     print("Optimal SNRs:")
#     print([optimal_snrs[nv] for nv in selected_indices])

# # Sort optimal_amplitudes by index (key)
# sorted_optimal_amplitudes = dict(sorted(optimal_amplitudes.items()))
# sorted_optimal_snrs = dict(sorted(optimal_snrs.items()))

# # Update results
# results = {
#     "optimal_amplitudes": sorted_optimal_amplitudes,
#     "optimal_snrs": sorted_optimal_snrs,
# }


def process_and_plot_amplitudes(data):
    """Process NV data for amplitude optimization."""
    nv_list = data["nv_list"]
    taus, counts = data["taus"], np.array(data["counts"])
    total_nvs = len(nv_list)
    optimal_amplitudes = {nv: None for nv in range(total_nvs)}
    optimal_snrs = {nv: None for nv in range(total_nvs)}

    selected_indices = range(total_nvs)

    def optimize_amplitudes(fit_function, valid_range):
        sig_counts, ref_counts = (
            counts[0][selected_indices],
            counts[1][selected_indices],
        )
        avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

        optimal_values = {}
        snr_values = {}
        for i, nv_ind in enumerate(selected_indices):
            try:
                popt, fit_fn = fit_function(taus, avg_snr[i], avg_snr_ste[i])
                tau_linspace = np.linspace(min(taus), max(taus), 1000)
                snr_values_curve = fit_fn(tau_linspace, *popt)
                optimal_value = tau_linspace[np.argmax(snr_values_curve)]

                # Keep the value within the valid range
                optimal_value = max(valid_range[0], min(valid_range[1], optimal_value))

                optimal_values[nv_ind] = optimal_value
                snr_values[nv_ind] = max(snr_values_curve)  # Optimal SNR
            except Exception as e:
                print(f"Fitting failed for NV index {nv_ind}: {e}")
                optimal_values[nv_ind] = None  # Mark as unprocessed
                snr_values[nv_ind] = None

        return optimal_values, snr_values, avg_snr, avg_snr_ste, taus

    # Optimize amplitudes
    amp_valid_range = (0.5, 1.5)
    optimal_amplitudes, optimal_snrs, avg_snr, avg_snr_ste, taus = optimize_amplitudes(
     fit_duration, amp_valid_range
    )

    # Replace unprocessed NVs with medians
    valid_amplitudes = [
        v
        for k, v in optimal_amplitudes.items()
        if k in selected_indices and v is not None
    ]
    median_amplitude = np.median(valid_amplitudes) if valid_amplitudes else 0
    for nv_index in range(total_nvs):
        if optimal_amplitudes.get(nv_index) is None:
            optimal_amplitudes[nv_index] = median_amplitude

    # Print lists of optimal amplitudes and SNRs
    print("Optimal Amplitudes:")
    print([optimal_amplitudes[nv] for nv in selected_indices])

    print("Optimal SNRs:")
    print([optimal_snrs[nv] for nv in selected_indices])

    # Scatter plot: SNR vs Optimal Amplitudes
    valid_snrs = [
        optimal_snrs[nv] for nv in selected_indices if optimal_snrs[nv] is not None
    ]
    valid_amplitudes = [
        optimal_amplitudes[nv]
        for nv in selected_indices
        if optimal_snrs[nv] is not None
    ]

    # Ensure yerr_snr is correctly shaped
    yerr = np.median(avg_snr_ste, axis=1).flatten()

    if valid_snrs:
        median_snr = np.median(valid_snrs)
    else:
        median_snr = 0  # Set a default value if there are no valid SNRs
    plt.figure(figsize=(6, 5))
    plt.errorbar(
        valid_amplitudes,
        valid_snrs,
        yerr=yerr,
        fmt="o",
        ecolor="gray",
        capsize=3,
        label=f"Median SCC SNR:{median_snr:.3f}",
    )
    plt.xlabel("Optimal SCC Amplitude")
    plt.ylabel("SCC SNR")
    plt.title("SCC SNR vs Optimal SCC Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Plot only the median amplitude fit curve
    plt.figure(figsize=(6, 5))
    plt.errorbar(
        taus,
        np.median(avg_snr, axis=0),
        yerr=np.min(avg_snr_ste, axis=0),
        color="blue",
        fmt="o",
        ecolor="gray",
        capsize=3,
        label="Med. SCC SNR Data",
    )
    tau_linspace = np.linspace(min(taus), max(taus), 1000)
    popt, fit_fn = fit_duration(
        taus, np.median(avg_snr, axis=0), np.median(avg_snr_ste, axis=0)
    )
    plt.plot(
        tau_linspace,
        fit_fn(tau_linspace, *popt),
        color="orange",
        label=f"Optimal SCC SNR: {median_snr:.3f}",
    )
    plt.axvline(
        median_amplitude,
        color="r",
        linestyle="--",
        label=f"Optimal SCC Amp: {median_amplitude:.3f}",
    )
    plt.xlabel("SCC Amplitude")
    plt.ylabel("SCC SNR")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Median NV SCC Amplitude Optimization")
    plt.show()


def process_and_plot_durations(data):
    """Process NV data for duration optimization."""
    nv_list = data["nv_list"]
    total_nvs = len(nv_list)
    optimal_durations = {nv: None for nv in range(total_nvs)}
    optimal_snrs = {nv: None for nv in range(total_nvs)}

    selected_indices = range(total_nvs)

    def optimize_durations(data, fit_function, valid_range):
        taus, counts = data["taus"], np.array(data["counts"])
        sig_counts, ref_counts = counts[0], counts[1]
        avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

        optimal_values = {}
        snr_values = {}
        for i, nv_ind in enumerate(selected_indices):
            try:
                popt, fit_fn = fit_function(taus, avg_snr[i], avg_snr_ste[i])
                tau_linspace = np.linspace(min(taus), max(taus), 1000)
                snr_values_curve = fit_fn(tau_linspace, *popt)
                optimal_value = tau_linspace[np.argmax(snr_values_curve)]

                # Keep the value within the valid range and round to nearest integer divisible by 4
                optimal_value = max(valid_range[0], min(valid_range[1], optimal_value))
                # optimal_value = int(round(optimal_value / 4.0) * 4)

                optimal_values[nv_ind] = optimal_value
                snr_values[nv_ind] = max(snr_values_curve)  # Optimal SNR
            except Exception as e:
                print(f"Fitting failed for NV index {nv_ind}: {e}")
                optimal_values[nv_ind] = None  # Mark as unprocessed
                snr_values[nv_ind] = None

        return optimal_values, snr_values, avg_snr, avg_snr_ste, taus

    # Optimize durations
    duration_valid_range = (0, 400)
    optimal_durations, optimal_snrs, avg_snr, avg_snr_ste, taus = optimize_durations(
        data, fit_duration, duration_valid_range
    )

    # Replace unprocessed NVs with medians
    valid_durations = [
        v
        for k, v in optimal_durations.items()
        if k in selected_indices and v is not None
    ]
    median_duration = np.median(valid_durations) if valid_durations else 0
    median_duration = int(round(median_duration / 4.0) * 4)  # Ensure divisibility by 4
    for nv_index in range(total_nvs):
        if optimal_durations.get(nv_index) is None:
            optimal_durations[nv_index] = median_duration

    # # Plot individual NV fits
    # for nv_index in selected_indices:
    #     plt.figure(figsize=(6, 4))
    #     plt.errorbar(
    #         taus,
    #         avg_snr[nv_index],
    #         yerr=avg_snr_ste[nv_index],
    #         fmt="o",
    #         label="SNR Data",
    #     )
    #     if optimal_durations[nv_index] is not None:
    #         tau_linspace = np.linspace(min(taus), max(taus), 1000)
    #         popt, fit_fn = fit_duration(taus, avg_snr[nv_index], avg_snr_ste[nv_index])
    #         plt.plot(
    #             tau_linspace,
    #             fit_fn(tau_linspace, *popt),
    #             label="Fitted Curve",
    #         )
    #         plt.axvline(
    #             optimal_durations[nv_index],
    #             color="r",
    #             linestyle="--",
    #             label=f"Optimal Duration: {optimal_durations[nv_index]}",
    #         )
    #     plt.title(f"NV {nv_index} - Duration Optimization")
    #     plt.xlabel("Duration")
    #     plt.ylabel("SNR")
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #     plt.show(block=True)

    # Print lists of durations and SNRs
    print("Optimal Durations:")

    optimal_durations = [int(round(optimal_durations[nv]/4)*4) for nv in selected_indices]
    print("Optimal SNRs:")
    print(optimal_durations)
    optimal_snrs = [optimal_snrs[nv] for nv in selected_indices]
    print(optimal_snrs)
    # median = np.median(optimal_durations)
    # optimal_durations = [int(median) if (val < 24 or val > 200) else val for val in optimal_durations]
    plt.figure(figsize=(6, 5))
    plt.scatter(
        optimal_durations,
        optimal_snrs,
        marker="o",
        color="blue",
        edgecolors="black",
        alpha=0.6,
    )
    plt.xlabel("Durations")
    plt.ylabel("SNR")
    plt.title(f"SNR Vs SCC Durations ({len(optimal_durations)}NVs)")
    return
    # Sort optimal_durations by index (key)
    sorted_optimal_durations = dict(sorted(optimal_durations.items()))
    sorted_optimal_snrs = dict(sorted(optimal_snrs.items()))

    # Update results
    results = {
        "optimal_durations": sorted_optimal_durations,
        "optimal_snrs": sorted_optimal_snrs,
    }
    return results


if __name__ == "__main__":
    # Initialize plot settings
    kpl.init_kplotlib()
    # duration_file_id = 1722305531191
    # amp_file_id = 1724491290147  # same scc duration 160
    # amp_file_id = 1725708405583  # optimized durations for each
    # amp_file_id = 1731980653795  # amp
    # amp_file_id = 1771055850280
    # duration_file_id = 1732098676751  # duration
    # duration_file_id = 1732098676751  # duration
    # duration_file_id = 1800578617426  # duration

    # amp_file_id = 1786527980407

    #duration
    data = dm.get_raw_data(
        file_stem="2026_01_21-05_05_21-johnson-nv0_2025_10_21", load_npz=True
    )

    results = process_and_plot_durations(data)

    # amp_file id
    # data = dm.get_raw_data(
    #     file_stem="2025_10_27-01_08_57-johnson-nv0_2025_10_21", load_npz=True
    # )
    # results = process_and_plot_amplitudes(data)

    # print("Results:", results)
    # print(f"{file_name}_{duration_file_id}")
    kpl.show(block=True)
