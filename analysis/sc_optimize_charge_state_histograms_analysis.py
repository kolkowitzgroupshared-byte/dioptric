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


def _to_python_scalar(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _fit_params_to_list(fit_params_arr, num_nvs, num_steps):
    return [
        [
            None
            if fit_params_arr[nv_ind, step_ind] is None
            else np.asarray(fit_params_arr[nv_ind, step_ind], dtype=float).ravel().tolist()
            for step_ind in range(num_steps)
        ]
        for nv_ind in range(num_nvs)
    ]


def _counts_to_list(condensed_counts, num_nvs, num_steps):
    return [
        [
            np.asarray(condensed_counts[nv_ind, step_ind]).ravel().tolist()
            for step_ind in range(num_steps)
        ]
        for nv_ind in range(num_nvs)
    ]

def process_and_plot(raw_data, do_plot=False):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)

    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = raw_data["num_steps"]
    step_vals_raw = np.linspace(min_step_val, max_step_val, num_steps)

    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]

    a, b, c = 1.5133e04, 2.6976, -38.63

    yellow_charge_readout_amp = raw_data["opx_config"]["waveforms"][
        "yellow_charge_readout"
    ]["sample"]
    green_aod_cw_charge_pol_amp = raw_data["opx_config"]["waveforms"][
        "green_aod_cw-charge_pol"
    ]["sample"]

    counts = np.array(raw_data["counts"])
    ref_exp_ind = 1

    # [nv_ind, step_ind, shot_ind]
    condensed_counts = np.array(
        [
            [
                np.asarray(counts[ref_exp_ind, nv_ind, :, step_ind, :]).flatten()
                for step_ind in range(num_steps)
            ]
            for nv_ind in range(num_nvs)
        ],
        dtype=object,
    )

    prob_dist = ProbDist.COMPOUND_POISSON

    # --- analysis x-axis used in plots/optimization ---
    step_vals = step_vals_raw.copy()
    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            x_label = "Polarization duration (ns)"
        else:
            step_vals = step_vals * green_aod_cw_charge_pol_amp
            x_label = "Polarization amplitude"
    else:
        if optimize_duration_or_amp:
            step_vals = step_vals * 1e-6
            x_label = "Readout duration (ms)"
        else:
            step_vals = step_vals * yellow_charge_readout_amp
            step_vals = a * (step_vals**b) + c
            x_label = "Readout amplitude (uW)"

    def process_nv_step(nv_ind, step_ind):
        counts_data = np.asarray(condensed_counts[nv_ind, step_ind])

        try:
            popt, pcov, chi_squared = fit_bimodal_histogram(counts_data, prob_dist)

            if popt is None:
                return {
                    "threshold": np.nan,
                    "readout_fidelity": np.nan,
                    "prep_fidelity": np.nan,
                    "goodness_of_fit": np.nan,
                    "fit_success": False,
                    "fit_params": None,
                }

            threshold, readout_fidelity = determine_threshold(
                popt, prob_dist, dark_mode_weight=0.5, ret_fidelity=True
            )
            prep_fidelity = 1 - popt[0]

            return {
                "threshold": threshold,
                "readout_fidelity": readout_fidelity,
                "prep_fidelity": prep_fidelity,
                "goodness_of_fit": chi_squared,
                "fit_success": True,
                "fit_params": np.asarray(popt, dtype=float),
            }

        except Exception as e:
            print(f"Error processing NV {nv_ind}, step {step_ind}: {e}")
            return {
                "threshold": np.nan,
                "readout_fidelity": np.nan,
                "prep_fidelity": np.nan,
                "goodness_of_fit": np.nan,
                "fit_success": False,
                "fit_params": None,
            }

    flat_results = Parallel(n_jobs=-1)(
        delayed(process_nv_step)(nv_ind, step_ind)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    )

    # --- unpack into arrays ---
    threshold_arr = np.full((num_nvs, num_steps), np.nan)
    readout_fidelity_arr = np.full((num_nvs, num_steps), np.nan)
    prep_fidelity_arr = np.full((num_nvs, num_steps), np.nan)
    goodness_of_fit_arr = np.full((num_nvs, num_steps), np.nan)
    fit_success_arr = np.zeros((num_nvs, num_steps), dtype=bool)
    fit_params_arr = np.empty((num_nvs, num_steps), dtype=object)

    for flat_ind, res in enumerate(flat_results):
        nv_ind = flat_ind // num_steps
        step_ind = flat_ind % num_steps

        threshold_arr[nv_ind, step_ind] = res["threshold"]
        readout_fidelity_arr[nv_ind, step_ind] = res["readout_fidelity"]
        prep_fidelity_arr[nv_ind, step_ind] = res["prep_fidelity"]
        goodness_of_fit_arr[nv_ind, step_ind] = res["goodness_of_fit"]
        fit_success_arr[nv_ind, step_ind] = res["fit_success"]
        fit_params_arr[nv_ind, step_ind] = res["fit_params"]

    optimal_values = []
    optimal_step_vals = []
    nv_indices = []

    for nv_ind in range(num_nvs):
        try:
            (
                optimal_step_val,
                optimal_prep_fidality,
                optimal_readout_fidality,
                max_combined_score,
            ) = find_optimal_value_geom_mean(
                step_vals,
                readout_fidelity_arr[nv_ind],
                prep_fidelity_arr[nv_ind],
                goodness_of_fit_arr[nv_ind],
                weights=(1.0, 1.0, 1.0),
            )

            optimal_step_vals.append(optimal_step_val)
            nv_indices.append(nv_ind)
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
            optimal_values.append((nv_ind, np.nan, np.nan, np.nan, np.nan))
            continue

        if do_plot:
            fig, ax1 = plt.subplots(figsize=(7, 5))
            ax1.plot(step_vals, readout_fidelity_arr[nv_ind], label="Readout Fidelity", color="orange")
            ax1.plot(step_vals, prep_fidelity_arr[nv_ind], label="Prep Fidelity", linestyle="--", color="green")
            ax1.set_xlabel(x_label)
            ax1.set_ylabel("Fidelity")
            ax1.grid(True, linestyle="--", alpha=0.6)

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

            ax1.axvline(optimal_step_val, color="red", linestyle="--",
                        label=f"Optimal Step Val: {optimal_step_val:.3f}")
            ax2.axvline(optimal_step_val, color="red", linestyle="--")

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=11)
            ax1.set_title(f"NV{nv_ind} - Optimal Step Val: {optimal_step_val:.3f}")
            plt.tight_layout()
            plt.show(block=True)

    valid_step_vals = np.asarray([val for val in optimal_step_vals if not np.isnan(val)], dtype=float)
    if len(valid_step_vals) == 0:
        raise ValueError("No valid step values found.")

    total_power = np.sum(valid_step_vals) / len(valid_step_vals)
    optimal_weights = valid_step_vals / total_power
    aom_voltage = ((total_power - c) / a) ** (1 / b)

    avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0)
    avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0)
    avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0)

    (
        avg_optimal_step_val,
        avg_optimal_readout_fidelity,
        avg_optimal_prep_fidelity,
        avg_max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        avg_readout_fidelity,
        avg_prep_fidelity,
        avg_goodness_of_fit,
        weights=(1, 1, 1),
    )

    median_readout_fidelity = np.nanmedian(readout_fidelity_arr, axis=0)
    median_prep_fidelity = np.nanmedian(prep_fidelity_arr, axis=0)
    median_goodness_of_fit = np.nanmedian(goodness_of_fit_arr, axis=0)

    (
        median_optimal_step_val,
        median_optimal_readout_fidelity,
        median_optimal_prep_fidelity,
        median_max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        median_readout_fidelity,
        median_prep_fidelity,
        median_goodness_of_fit,
        weights=(1, 1, 2),
    )

    base_file_stem = raw_data.get("file_stem") or raw_data.get("file_name") or "raw_data"
    if isinstance(base_file_stem, (list, tuple)):
        base_file_stem = "_".join(map(str, base_file_stem))
    base_file_stem = str(base_file_stem).replace(" ", "_")
    results = {
        # identity / metadata
        "file_stem_source": str(base_file_stem),
        "num_nvs": int(num_nvs),
        "num_steps": int(num_steps),
        "nv_indices": list(range(num_nvs)),
        "step_vals_raw": np.asarray(step_vals_raw, dtype=float).tolist(),
        "step_vals": np.asarray(step_vals, dtype=float).tolist(),
        "x_label": x_label,
        "prob_dist_name": prob_dist.name if hasattr(prob_dist, "name") else str(prob_dist),

        # optimization mode metadata
        "optimize_pol_or_readout": bool(optimize_pol_or_readout),
        "optimize_duration_or_amp": bool(optimize_duration_or_amp),
        "yellow_charge_readout_amp": float(yellow_charge_readout_amp),
        "green_aod_cw_charge_pol_amp": float(green_aod_cw_charge_pol_amp),
        "power_fit_a": float(a),
        "power_fit_b": float(b),
        "power_fit_c": float(c),

        # processed per-NV / per-step data
        "readout_fidelity_arr": np.asarray(readout_fidelity_arr, dtype=float).tolist(),
        "prep_fidelity_arr": np.asarray(prep_fidelity_arr, dtype=float).tolist(),
        "goodness_of_fit_arr": np.asarray(goodness_of_fit_arr, dtype=float).tolist(),
        "threshold_arr": np.asarray(threshold_arr, dtype=float).tolist(),
        "fit_success_arr": np.asarray(fit_success_arr, dtype=bool).tolist(),
        "fit_params_arr": _fit_params_to_list(fit_params_arr, num_nvs, num_steps),

        # raw processed counts so you can replot / refit later
        "condensed_counts": _counts_to_list(condensed_counts, num_nvs, num_steps),

        # optimal values per NV
        "optimal_values": [
            [
                int(v[0]),
                float(v[1]) if not np.isnan(v[1]) else None,
                float(v[2]) if not np.isnan(v[2]) else None,
                float(v[3]) if not np.isnan(v[3]) else None,
                float(v[4]) if not np.isnan(v[4]) else None,
            ]
            for v in optimal_values
        ],
        "optimal_step_vals": np.asarray(optimal_step_vals, dtype=float).tolist(),
        "valid_step_vals": np.asarray(valid_step_vals, dtype=float).tolist(),
        "optimal_weights": np.asarray(optimal_weights, dtype=float).tolist(),
        "total_power": float(total_power),
        "aom_voltage": float(aom_voltage),

        # aggregate curves
        "avg_readout_fidelity": np.asarray(avg_readout_fidelity, dtype=float).tolist(),
        "avg_prep_fidelity": np.asarray(avg_prep_fidelity, dtype=float).tolist(),
        "avg_goodness_of_fit": np.asarray(avg_goodness_of_fit, dtype=float).tolist(),
        "median_readout_fidelity": np.asarray(median_readout_fidelity, dtype=float).tolist(),
        "median_prep_fidelity": np.asarray(median_prep_fidelity, dtype=float).tolist(),
        "median_goodness_of_fit": np.asarray(median_goodness_of_fit, dtype=float).tolist(),

        # aggregate optima
        "avg_optimal_step_val": float(avg_optimal_step_val),
        "avg_optimal_readout_fidelity": float(avg_optimal_readout_fidelity),
        "avg_optimal_prep_fidelity": float(avg_optimal_prep_fidelity),
        "avg_max_combined_score": float(avg_max_combined_score),
        "median_optimal_step_val": float(median_optimal_step_val),
        "median_optimal_readout_fidelity": float(median_optimal_readout_fidelity),
        "median_optimal_prep_fidelity": float(median_optimal_prep_fidelity),
        "median_max_combined_score": float(median_max_combined_score),
    }

    timestamp = dm.get_time_stamp()
    file_name = f"optimization_processed_full_{base_file_stem}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)
    dm.save_raw_data(results, file_path)

    print(f"Processed data saved to '{file_path}'.")
    return results


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
    ### readout amp
    # file_id = "2026_03_22-21_49_52-qnami-nv0_2026_02_20"
    
    ### pol amp var
    # file_id = "2026_03_24-21_11_43-qnami-nv0_2026_02_20" ## 1460
    file_id = "2026_03_25-23_32_41-qnami-nv0_2026_02_20" ## 1306
    
    ### pol dur var
    # file_id = "2026_03_17-06_00_50-qnami-nv0_2026_02_20"

    raw_data = dm.get_raw_data(file_stem=file_id, load_npz=True)
    process_and_plot(raw_data, do_plot=False)
    # process_and_plot_charge(raw_data, do_plot=True)
    plt.show(block=True)