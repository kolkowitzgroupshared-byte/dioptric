# -*- coding: utf-8 -*-
"""
Charge/readout or polarization optimization analysis.

CPU:
    Parallel SciPy bimodal histogram fits over NV x step.

GPU:
    Optional CuPy acceleration for recomputing optimal step values from saved
    processed data. This does not GPU-accelerate scipy fitting.

Created Fall 2024
Updated June 2026
@author: Saroj Chand
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

# Compatibility patch for old labrad with newer NumPy
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from joblib import Parallel, delayed

try:
    import cupy as cp

    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# Helpers
# =============================================================================


def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    return obj


def get_prob_dist(prob_dist_name):
    try:
        return ProbDist[prob_dist_name]
    except Exception:
        return getattr(ProbDist, prob_dist_name)


def _fit_params_to_list(fit_params_arr, num_nvs, num_steps):
    return [
        [
            None
            if fit_params_arr[nv_ind, step_ind] is None
            else np.asarray(
                fit_params_arr[nv_ind, step_ind],
                dtype=float,
            )
            .ravel()
            .tolist()
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


def find_optimal_value_geom_mean(
    step_vals,
    prep_fidelity,
    readout_fidelity,
    goodness_of_fit,
    weights=(1, 1, 1),
    skip_first=2,
):
    """
    Choose optimal step using weighted normalized score.

    score =
        w1 * normalized prep fidelity
      + w2 * normalized readout fidelity
      + w3 * inverted normalized goodness_of_fit

    Larger score is better.
    """

    w1, w2, w3 = weights

    step_vals = np.asarray(step_vals, dtype=float)[skip_first:]
    prep_fidelity = np.asarray(prep_fidelity, dtype=float)[skip_first:]
    readout_fidelity = np.asarray(readout_fidelity, dtype=float)[skip_first:]
    goodness_of_fit = np.asarray(goodness_of_fit, dtype=float)[skip_first:]

    good = (
        np.isfinite(step_vals)
        & np.isfinite(prep_fidelity)
        & np.isfinite(readout_fidelity)
        & np.isfinite(goodness_of_fit)
    )

    if not np.any(good):
        raise ValueError("No finite values for optimization.")

    step_vals = step_vals[good]
    prep_fidelity = prep_fidelity[good]
    readout_fidelity = readout_fidelity[good]
    goodness_of_fit = goodness_of_fit[good]

    norm_prep = (prep_fidelity - np.nanmin(prep_fidelity)) / (
        np.nanmax(prep_fidelity) - np.nanmin(prep_fidelity) + 1e-12
    )

    norm_readout = (readout_fidelity - np.nanmin(readout_fidelity)) / (
        np.nanmax(readout_fidelity) - np.nanmin(readout_fidelity) + 1e-12
    )

    norm_gof = (goodness_of_fit - np.nanmin(goodness_of_fit)) / (
        np.nanmax(goodness_of_fit) - np.nanmin(goodness_of_fit) + 1e-12
    )

    inverted_gof = 1.0 - norm_gof

    combined_score = (
        w1 * norm_prep
        + w2 * norm_readout
        + w3 * inverted_gof
    )

    max_index = int(np.nanargmax(combined_score))

    return (
        float(step_vals[max_index]),
        float(prep_fidelity[max_index]),
        float(readout_fidelity[max_index]),
        float(combined_score[max_index]),
    )


# =============================================================================
# Parallel CPU fitting worker
# =============================================================================


def fit_bimodal_nv_step_job(
    nv_ind,
    step_ind,
    counts_data,
    prob_dist_name="COMPOUND_POISSON",
):
    """
    Top-level worker for Windows/joblib.

    One job = one NV, one step.
    """

    try:
        prob_dist = get_prob_dist(prob_dist_name)
        counts_data = np.asarray(counts_data, dtype=float).flatten()

        popt, pcov, chi_squared = fit_bimodal_histogram(
            counts_data,
            prob_dist,
        )

        if popt is None:
            return {
                "nv_ind": int(nv_ind),
                "step_ind": int(step_ind),
                "threshold": np.nan,
                "readout_fidelity": np.nan,
                "prep_fidelity": np.nan,
                "goodness_of_fit": np.nan,
                "fit_success": False,
                "fit_params": None,
                "error": None,
            }

        threshold, readout_fidelity = determine_threshold(
            popt,
            prob_dist,
            dark_mode_weight=0.5,
            ret_fidelity=True,
        )

        prep_fidelity = 1.0 - float(popt[0])

        return {
            "nv_ind": int(nv_ind),
            "step_ind": int(step_ind),
            "threshold": float(threshold),
            "readout_fidelity": float(readout_fidelity),
            "prep_fidelity": float(prep_fidelity),
            "goodness_of_fit": float(chi_squared),
            "fit_success": True,
            "fit_params": np.asarray(popt, dtype=float),
            "error": None,
        }

    except Exception:
        return {
            "nv_ind": int(nv_ind),
            "step_ind": int(step_ind),
            "threshold": np.nan,
            "readout_fidelity": np.nan,
            "prep_fidelity": np.nan,
            "goodness_of_fit": np.nan,
            "fit_success": False,
            "fit_params": None,
            "error": traceback.format_exc(),
        }


# =============================================================================
# Main CPU parallel processing
# =============================================================================


def process_and_plot(
    raw_data,
    do_plot=False,
    n_jobs=12,
    joblib_verbose=10,
    save_condensed_counts=True,
):
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

    counts = np.asarray(raw_data["counts"])
    ref_exp_ind = 1

    condensed_counts = np.empty((num_nvs, num_steps), dtype=object)

    for nv_ind in range(num_nvs):
        for step_ind in range(num_steps):
            condensed_counts[nv_ind, step_ind] = np.asarray(
                counts[ref_exp_ind, nv_ind, :, step_ind, :]
            ).flatten()

    prob_dist = ProbDist.COMPOUND_POISSON

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

    print("\n=== Starting parallel charge optimization fits ===")
    print(f"num_nvs: {num_nvs}")
    print(f"num_steps: {num_steps}")
    print(f"total fits: {num_nvs * num_steps}")
    print(f"n_jobs: {n_jobs}")
    print(f"prob_dist: {prob_dist.name}")

    tasks = [
        (
            nv_ind,
            step_ind,
            condensed_counts[nv_ind, step_ind],
            prob_dist.name,
        )
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    ]

    if n_jobs is None or int(n_jobs) == 1:
        flat_results = [fit_bimodal_nv_step_job(*task) for task in tasks]
    else:
        flat_results = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            verbose=joblib_verbose,
            batch_size="auto",
            # pre_dispatch="2*n_jobs"
        )(
            delayed(fit_bimodal_nv_step_job)(*task)
            for task in tasks
        )

    threshold_arr = np.full((num_nvs, num_steps), np.nan)
    readout_fidelity_arr = np.full((num_nvs, num_steps), np.nan)
    prep_fidelity_arr = np.full((num_nvs, num_steps), np.nan)
    goodness_of_fit_arr = np.full((num_nvs, num_steps), np.nan)
    fit_success_arr = np.zeros((num_nvs, num_steps), dtype=bool)
    fit_params_arr = np.empty((num_nvs, num_steps), dtype=object)

    num_errors = 0

    for res in flat_results:
        nv_ind = int(res["nv_ind"])
        step_ind = int(res["step_ind"])

        if res["error"] is not None:
            num_errors += 1
            print(f"\nFit failed for NV {nv_ind}, step {step_ind}")
            print(res["error"])

        threshold_arr[nv_ind, step_ind] = res["threshold"]
        readout_fidelity_arr[nv_ind, step_ind] = res["readout_fidelity"]
        prep_fidelity_arr[nv_ind, step_ind] = res["prep_fidelity"]
        goodness_of_fit_arr[nv_ind, step_ind] = res["goodness_of_fit"]
        fit_success_arr[nv_ind, step_ind] = res["fit_success"]
        fit_params_arr[nv_ind, step_ind] = res["fit_params"]

    print("\n=== Fit summary ===")
    print("Successful fits:", int(np.sum(fit_success_arr)), "/", num_nvs * num_steps)
    print("Errors:", int(num_errors))

    optimal_values = []
    optimal_step_vals = []

    for nv_ind in range(num_nvs):
        try:
            (
                optimal_step_val,
                optimal_prep_fidelity,
                optimal_readout_fidelity,
                max_combined_score,
            ) = find_optimal_value_geom_mean(
                step_vals,
                prep_fidelity_arr[nv_ind],
                readout_fidelity_arr[nv_ind],
                goodness_of_fit_arr[nv_ind],
                weights=(1, 1, 1),
            )

            optimal_step_vals.append(optimal_step_val)

            optimal_values.append(
                (
                    nv_ind,
                    optimal_step_val,
                    optimal_prep_fidelity,
                    optimal_readout_fidelity,
                    max_combined_score,
                )
            )

        except Exception as e:
            print(f"Failed to optimize NV {nv_ind}: {e}")
            optimal_step_vals.append(np.nan)
            optimal_values.append((nv_ind, np.nan, np.nan, np.nan, np.nan))
            continue

        if do_plot:
            fig = plot_processed_nv_metrics_from_arrays(
                step_vals=step_vals,
                x_label=x_label,
                readout=readout_fidelity_arr[nv_ind],
                prep=prep_fidelity_arr[nv_ind],
                gof=goodness_of_fit_arr[nv_ind],
                nv_ind=nv_ind,
                opt_step=optimal_step_val,
                opt_prep=optimal_prep_fidelity,
                opt_readout=optimal_readout_fidelity,
                opt_score=max_combined_score,
            )
            plt.show(block=True)

    optimal_step_vals = np.asarray(optimal_step_vals, dtype=float)
    valid_step_vals = optimal_step_vals[np.isfinite(optimal_step_vals)]

    if len(valid_step_vals) == 0:
        raise ValueError("No valid step values found.")

    total_power = float(np.nanmean(valid_step_vals))
    optimal_weights = valid_step_vals / total_power

    if x_label == "Readout amplitude (uW)" and total_power > c:
        aom_voltage = float(((total_power - c) / a) ** (1 / b))
    else:
        aom_voltage = np.nan

    avg_readout_fidelity = np.nanmean(readout_fidelity_arr, axis=0)
    avg_prep_fidelity = np.nanmean(prep_fidelity_arr, axis=0)
    avg_goodness_of_fit = np.nanmean(goodness_of_fit_arr, axis=0)

    (
        avg_optimal_step_val,
        avg_optimal_prep_fidelity,
        avg_optimal_readout_fidelity,
        avg_max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        avg_prep_fidelity,
        avg_readout_fidelity,
        avg_goodness_of_fit,
        weights=(1, 1, 1),
    )

    median_readout_fidelity = np.nanmedian(readout_fidelity_arr, axis=0)
    median_prep_fidelity = np.nanmedian(prep_fidelity_arr, axis=0)
    median_goodness_of_fit = np.nanmedian(goodness_of_fit_arr, axis=0)

    (
        median_optimal_step_val,
        median_optimal_prep_fidelity,
        median_optimal_readout_fidelity,
        median_max_combined_score,
    ) = find_optimal_value_geom_mean(
        step_vals,
        median_prep_fidelity,
        median_readout_fidelity,
        median_goodness_of_fit,
        weights=(1, 1, 2),
    )

    base_file_stem = (
        raw_data.get("file_stem")
        or raw_data.get("file_name")
        or raw_data.get("timestamp")
        or "raw_data"
    )

    if isinstance(base_file_stem, (list, tuple)):
        base_file_stem = "_".join(map(str, base_file_stem))

    base_file_stem = str(base_file_stem).replace(" ", "_")

    results = {
        "file_stem_source": str(base_file_stem),
        "num_nvs": int(num_nvs),
        "num_steps": int(num_steps),
        "nv_indices": list(range(num_nvs)),
        "step_vals_raw": np.asarray(step_vals_raw, dtype=float).tolist(),
        "step_vals": np.asarray(step_vals, dtype=float).tolist(),
        "x_label": x_label,
        "prob_dist_name": prob_dist.name,
        "n_jobs": None if n_jobs is None else int(n_jobs),
        "optimize_pol_or_readout": bool(optimize_pol_or_readout),
        "optimize_duration_or_amp": bool(optimize_duration_or_amp),
        "yellow_charge_readout_amp": float(yellow_charge_readout_amp),
        "green_aod_cw_charge_pol_amp": float(green_aod_cw_charge_pol_amp),
        "power_fit_a": float(a),
        "power_fit_b": float(b),
        "power_fit_c": float(c),
        "readout_fidelity_arr": readout_fidelity_arr.tolist(),
        "prep_fidelity_arr": prep_fidelity_arr.tolist(),
        "goodness_of_fit_arr": goodness_of_fit_arr.tolist(),
        "threshold_arr": threshold_arr.tolist(),
        "fit_success_arr": fit_success_arr.tolist(),
        "fit_params_arr": _fit_params_to_list(
            fit_params_arr,
            num_nvs,
            num_steps,
        ),
        "optimal_values": [
            [
                int(v[0]),
                float(v[1]) if np.isfinite(v[1]) else None,
                float(v[2]) if np.isfinite(v[2]) else None,
                float(v[3]) if np.isfinite(v[3]) else None,
                float(v[4]) if np.isfinite(v[4]) else None,
            ]
            for v in optimal_values
        ],
        "optimal_step_vals": optimal_step_vals.tolist(),
        "valid_step_vals": valid_step_vals.tolist(),
        "optimal_weights": optimal_weights.tolist(),
        "total_power": float(total_power),
        "aom_voltage": float(aom_voltage) if np.isfinite(aom_voltage) else None,
        "avg_readout_fidelity": avg_readout_fidelity.tolist(),
        "avg_prep_fidelity": avg_prep_fidelity.tolist(),
        "avg_goodness_of_fit": avg_goodness_of_fit.tolist(),
        "median_readout_fidelity": median_readout_fidelity.tolist(),
        "median_prep_fidelity": median_prep_fidelity.tolist(),
        "median_goodness_of_fit": median_goodness_of_fit.tolist(),
        "avg_optimal_step_val": float(avg_optimal_step_val),
        "avg_optimal_prep_fidelity": float(avg_optimal_prep_fidelity),
        "avg_optimal_readout_fidelity": float(avg_optimal_readout_fidelity),
        "avg_max_combined_score": float(avg_max_combined_score),
        "median_optimal_step_val": float(median_optimal_step_val),
        "median_optimal_prep_fidelity": float(median_optimal_prep_fidelity),
        "median_optimal_readout_fidelity": float(median_optimal_readout_fidelity),
        "median_max_combined_score": float(median_max_combined_score),
    }

    if save_condensed_counts:
        results["condensed_counts"] = _counts_to_list(
            condensed_counts,
            num_nvs,
            num_steps,
        )
    else:
        results["condensed_counts"] = None

    timestamp = dm.get_time_stamp()
    file_name = f"optimization_processed_full_{base_file_stem}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)

    dm.save_raw_data(
        make_json_safe(results),
        file_path,
    )

    print(f"Processed data saved to: {file_path}")

    return results


# =============================================================================
# CPU and GPU recompute from saved processed data
# =============================================================================


def recompute_optimal_from_processed(
    analyzed_data,
    nv_ind,
    weights=(1, 1, 1),
):
    step_vals = np.asarray(analyzed_data["step_vals"], dtype=float)
    prep = np.asarray(analyzed_data["prep_fidelity_arr"][nv_ind], dtype=float)
    readout = np.asarray(analyzed_data["readout_fidelity_arr"][nv_ind], dtype=float)
    gof = np.asarray(analyzed_data["goodness_of_fit_arr"][nv_ind], dtype=float)

    return find_optimal_value_geom_mean(
        step_vals,
        prep,
        readout,
        gof,
        weights=weights,
    )


def recompute_all_optimal_values_from_processed(
    analyzed_data,
    weights=(1, 1, 1),
    nv_indices=None,
):
    num_nvs = int(analyzed_data["num_nvs"])

    if nv_indices is None:
        nv_indices = range(num_nvs)

    optimal_values = []
    optimal_step_vals = []

    for nv_ind in nv_indices:
        try:
            opt_step, opt_prep, opt_readout, opt_score = recompute_optimal_from_processed(
                analyzed_data,
                nv_ind,
                weights=weights,
            )

            optimal_values.append(
                {
                    "nv_ind": int(nv_ind),
                    "optimal_step_val": float(opt_step),
                    "optimal_prep_fidelity": float(opt_prep),
                    "optimal_readout_fidelity": float(opt_readout),
                    "max_combined_score": float(opt_score),
                }
            )
            optimal_step_vals.append(opt_step)

        except Exception as e:
            print(f"Failed on NV {nv_ind}: {e}")
            optimal_values.append(
                {
                    "nv_ind": int(nv_ind),
                    "optimal_step_val": np.nan,
                    "optimal_prep_fidelity": np.nan,
                    "optimal_readout_fidelity": np.nan,
                    "max_combined_score": np.nan,
                }
            )
            optimal_step_vals.append(np.nan)

    optimal_step_vals = np.asarray(optimal_step_vals, dtype=float)
    valid_step_vals = optimal_step_vals[np.isfinite(optimal_step_vals)]

    summary = {
        "optimal_values": optimal_values,
        "optimal_step_vals": optimal_step_vals,
        "valid_step_vals": valid_step_vals,
        "weights_used": tuple(weights),
        "used_gpu": False,
    }

    x_label = analyzed_data.get("x_label", "")

    if len(valid_step_vals) > 0:
        total_power = float(np.nanmean(valid_step_vals))
        summary["total_power"] = total_power
        summary["optimal_weights"] = valid_step_vals / total_power

        if x_label == "Readout amplitude (uW)":
            a = float(analyzed_data["power_fit_a"])
            b = float(analyzed_data["power_fit_b"])
            c = float(analyzed_data["power_fit_c"])

            if total_power > c:
                summary["aom_voltage"] = float(((total_power - c) / a) ** (1 / b))
            else:
                summary["aom_voltage"] = np.nan

    return summary


def recompute_all_optimal_values_from_processed_gpu(
    analyzed_data,
    weights=(1, 1, 1),
    skip_first=2,
):
    """
    GPU-vectorized recompute from saved arrays.

    This is fast, but only accelerates recomputing optimal steps.
    It does not accelerate the original SciPy histogram fits.
    """

    if not GPU_AVAILABLE:
        print("CuPy/GPU not available. Falling back to CPU.")
        return recompute_all_optimal_values_from_processed(
            analyzed_data,
            weights=weights,
        )

    w1, w2, w3 = weights

    step_vals_cpu = np.asarray(analyzed_data["step_vals"], dtype=float)
    prep_cpu = np.asarray(analyzed_data["prep_fidelity_arr"], dtype=float)
    readout_cpu = np.asarray(analyzed_data["readout_fidelity_arr"], dtype=float)
    gof_cpu = np.asarray(analyzed_data["goodness_of_fit_arr"], dtype=float)

    step_vals = cp.asarray(step_vals_cpu[skip_first:], dtype=cp.float64)
    prep = cp.asarray(prep_cpu[:, skip_first:], dtype=cp.float64)
    readout = cp.asarray(readout_cpu[:, skip_first:], dtype=cp.float64)
    gof = cp.asarray(gof_cpu[:, skip_first:], dtype=cp.float64)

    num_nvs = prep.shape[0]

    prep_min = cp.nanmin(prep, axis=1, keepdims=True)
    prep_max = cp.nanmax(prep, axis=1, keepdims=True)

    readout_min = cp.nanmin(readout, axis=1, keepdims=True)
    readout_max = cp.nanmax(readout, axis=1, keepdims=True)

    gof_min = cp.nanmin(gof, axis=1, keepdims=True)
    gof_max = cp.nanmax(gof, axis=1, keepdims=True)

    norm_prep = (prep - prep_min) / (prep_max - prep_min + 1e-12)
    norm_readout = (readout - readout_min) / (readout_max - readout_min + 1e-12)
    norm_gof = (gof - gof_min) / (gof_max - gof_min + 1e-12)

    score = (
        w1 * norm_prep
        + w2 * norm_readout
        + w3 * (1.0 - norm_gof)
    )

    score_safe = cp.where(cp.isfinite(score), score, -cp.inf)

    best_step_ind = cp.argmax(score_safe, axis=1)
    best_score = score_safe[cp.arange(num_nvs), best_step_ind]

    optimal_step_vals = step_vals[best_step_ind]
    optimal_prep = prep[cp.arange(num_nvs), best_step_ind]
    optimal_readout = readout[cp.arange(num_nvs), best_step_ind]

    good_nv = cp.isfinite(best_score)

    optimal_step_vals = cp.where(good_nv, optimal_step_vals, cp.nan)
    optimal_prep = cp.where(good_nv, optimal_prep, cp.nan)
    optimal_readout = cp.where(good_nv, optimal_readout, cp.nan)
    best_score = cp.where(good_nv, best_score, cp.nan)

    optimal_step_vals_cpu = cp.asnumpy(optimal_step_vals)
    optimal_prep_cpu = cp.asnumpy(optimal_prep)
    optimal_readout_cpu = cp.asnumpy(optimal_readout)
    best_score_cpu = cp.asnumpy(best_score)

    valid_step_vals = optimal_step_vals_cpu[np.isfinite(optimal_step_vals_cpu)]

    optimal_values = []

    for nv_ind in range(num_nvs):
        optimal_values.append(
            {
                "nv_ind": int(nv_ind),
                "optimal_step_val": float(optimal_step_vals_cpu[nv_ind])
                if np.isfinite(optimal_step_vals_cpu[nv_ind])
                else np.nan,
                "optimal_prep_fidelity": float(optimal_prep_cpu[nv_ind])
                if np.isfinite(optimal_prep_cpu[nv_ind])
                else np.nan,
                "optimal_readout_fidelity": float(optimal_readout_cpu[nv_ind])
                if np.isfinite(optimal_readout_cpu[nv_ind])
                else np.nan,
                "max_combined_score": float(best_score_cpu[nv_ind])
                if np.isfinite(best_score_cpu[nv_ind])
                else np.nan,
            }
        )

    summary = {
        "optimal_values": optimal_values,
        "optimal_step_vals": optimal_step_vals_cpu,
        "valid_step_vals": valid_step_vals,
        "weights_used": tuple(weights),
        "used_gpu": True,
    }

    x_label = analyzed_data.get("x_label", "")

    if len(valid_step_vals) > 0:
        total_power = float(np.nanmean(valid_step_vals))
        summary["total_power"] = total_power
        summary["optimal_weights"] = valid_step_vals / total_power

        if x_label == "Readout amplitude (uW)":
            a = float(analyzed_data["power_fit_a"])
            b = float(analyzed_data["power_fit_b"])
            c = float(analyzed_data["power_fit_c"])

            if total_power > c:
                summary["aom_voltage"] = float(((total_power - c) / a) ** (1 / b))
            else:
                summary["aom_voltage"] = np.nan

    return summary


# =============================================================================
# Plotting from processed data
# =============================================================================


def plot_processed_nv_metrics_from_arrays(
    step_vals,
    x_label,
    readout,
    prep,
    gof,
    nv_ind,
    opt_step,
    opt_prep,
    opt_readout,
    opt_score,
):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(step_vals, readout, label="Readout fidelity", color="orange")
    ax1.plot(step_vals, prep, label="Prep fidelity", linestyle="--", color="green")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Fidelity")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        step_vals,
        gof,
        color="gray",
        linestyle="--",
        label=r"Goodness of fit",
        alpha=0.7,
    )
    ax2.set_ylabel("Goodness of fit", color="gray")

    if np.isfinite(opt_step):
        ax1.axvline(
            opt_step,
            color="red",
            linestyle="--",
            label=f"Optimal = {opt_step:.3f}",
        )
        ax2.axvline(opt_step, color="red", linestyle="--")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=10,
    )

    ax1.set_title(
        f"NV {nv_ind}: opt={opt_step:.3f}, "
        f"prep={opt_prep:.3f}, readout={opt_readout:.3f}, score={opt_score:.3f}"
    )

    plt.tight_layout()
    return fig


def plot_processed_nv_metrics(
    analyzed_data,
    nv_ind,
    weights=(1, 1, 1),
):
    step_vals = np.asarray(analyzed_data["step_vals"], dtype=float)
    x_label = analyzed_data["x_label"]

    readout = np.asarray(analyzed_data["readout_fidelity_arr"][nv_ind], dtype=float)
    prep = np.asarray(analyzed_data["prep_fidelity_arr"][nv_ind], dtype=float)
    gof = np.asarray(analyzed_data["goodness_of_fit_arr"][nv_ind], dtype=float)

    opt_step, opt_prep, opt_readout, opt_score = recompute_optimal_from_processed(
        analyzed_data,
        nv_ind,
        weights=weights,
    )

    return plot_processed_nv_metrics_from_arrays(
        step_vals,
        x_label,
        readout,
        prep,
        gof,
        nv_ind,
        opt_step,
        opt_prep,
        opt_readout,
        opt_score,
    )


def plot_ref_histogram_from_processed(
    analyzed_data,
    nv_ind,
    step_ind,
    density=True,
):
    if analyzed_data.get("condensed_counts", None) is None:
        raise KeyError(
            "This processed file does not contain condensed_counts. "
            "Run process_and_plot(..., save_condensed_counts=True)."
        )

    counts = np.asarray(
        analyzed_data["condensed_counts"][nv_ind][step_ind],
        dtype=float,
    )

    threshold = float(analyzed_data["threshold_arr"][nv_ind][step_ind])
    fit_success = bool(analyzed_data["fit_success_arr"][nv_ind][step_ind])
    fit_params = analyzed_data["fit_params_arr"][nv_ind][step_ind]

    step_vals = np.asarray(analyzed_data["step_vals"], dtype=float)
    x_label = analyzed_data["x_label"]

    fig, ax = plt.subplots(figsize=(6.5, 5))

    kpl.histogram(
        ax,
        counts,
        density=density,
    )

    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Probability" if density else "Occurrences")
    ax.set_title(
        f"NV {nv_ind}, step {step_ind}, "
        f"{x_label} = {step_vals[step_ind]:.3f}"
    )

    if np.isfinite(threshold):
        ax.axvline(
            threshold,
            color=kpl.KplColors.GRAY,
            ls="dashed",
            label="Threshold",
        )

    if fit_success and fit_params is not None:
        popt = np.asarray(fit_params, dtype=float)

        prob_dist_name = analyzed_data.get("prob_dist_name", "COMPOUND_POISSON")
        prob_dist_local = get_prob_dist(prob_dist_name)

        x_max = max(
            np.nanmax(counts),
            threshold if np.isfinite(threshold) else 0,
        )

        x_vals = np.linspace(0, x_max + 1, 1000)

        single_mode_num_params = bimodal_histogram.get_single_mode_num_params(
            prob_dist_local
        )
        single_mode_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist_local)
        bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist_local)

        dark_mode_line = popt[0] * single_mode_pdf(
            x_vals,
            *popt[1 : 1 + single_mode_num_params],
        )

        bright_mode_line = (1 - popt[0]) * single_mode_pdf(
            x_vals,
            *popt[1 + single_mode_num_params :],
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
            ax,
            x_vals,
            bimodal_line,
            color=kpl.KplColors.BLUE,
            label="Combined",
        )

    ax.legend(loc=kpl.Loc.UPPER_RIGHT)

    return fig

def recompute_optimal_step_index_from_processed(
    analyzed_data,
    nv_ind,
    weights=(1, 1, 1),
    skip_first=2,
):
    """
    Return the optimal step index for one NV using the same score rule as
    find_optimal_value_geom_mean().
    """

    w1, w2, w3 = weights

    step_vals = np.asarray(analyzed_data["step_vals"], dtype=float)
    prep = np.asarray(analyzed_data["prep_fidelity_arr"][nv_ind], dtype=float)
    readout = np.asarray(analyzed_data["readout_fidelity_arr"][nv_ind], dtype=float)
    gof = np.asarray(analyzed_data["goodness_of_fit_arr"][nv_ind], dtype=float)

    original_inds = np.arange(step_vals.size)

    step_vals_use = step_vals[skip_first:]
    prep_use = prep[skip_first:]
    readout_use = readout[skip_first:]
    gof_use = gof[skip_first:]
    inds_use = original_inds[skip_first:]

    good = (
        np.isfinite(step_vals_use)
        & np.isfinite(prep_use)
        & np.isfinite(readout_use)
        & np.isfinite(gof_use)
    )

    if not np.any(good):
        raise ValueError(f"No finite optimization values for NV {nv_ind}.")

    step_vals_use = step_vals_use[good]
    prep_use = prep_use[good]
    readout_use = readout_use[good]
    gof_use = gof_use[good]
    inds_use = inds_use[good]

    norm_prep = (prep_use - np.nanmin(prep_use)) / (
        np.nanmax(prep_use) - np.nanmin(prep_use) + 1e-12
    )

    norm_readout = (readout_use - np.nanmin(readout_use)) / (
        np.nanmax(readout_use) - np.nanmin(readout_use) + 1e-12
    )

    norm_gof = (gof_use - np.nanmin(gof_use)) / (
        np.nanmax(gof_use) - np.nanmin(gof_use) + 1e-12
    )

    score = w1 * norm_prep + w2 * norm_readout + w3 * (1.0 - norm_gof)

    best_local = int(np.nanargmax(score))
    best_step_ind = int(inds_use[best_local])

    return {
        "step_ind": best_step_ind,
        "step_val": float(step_vals[best_step_ind]),
        "prep_fidelity": float(prep[best_step_ind]),
        "readout_fidelity": float(readout[best_step_ind]),
        "goodness_of_fit": float(gof[best_step_ind]),
        "score": float(score[best_local]),
    }

# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    kpl.init_kplotlib()

    # -------------------------------------------------------------------------
    # Option A: process new raw data with CPU parallel fitting.
    # -------------------------------------------------------------------------
    run_new_processing = False

    file_id = "2026_06_24-23_33_18-qnami-nv0_2026_02_20" ## pol amp
    file_id = "2026_06_26-21_58_16-qnami-nv0_2026_02_20" ## readout amp
    file_id = "2026_07_08-22_48_57-qnami-nv0_2026_02_20" ## readout amp
    
    if run_new_processing:
        raw_data = dm.get_raw_data(
            file_stem=file_id,
            load_npz=True,
        )

        raw_data["file_stem"] = file_id

        results = process_and_plot(
            raw_data,
            do_plot=False,
            n_jobs=12,
            joblib_verbose=10,
            save_condensed_counts=True,
        )

        kpl.show(block=True)
        sys.exit()

    # -------------------------------------------------------------------------
    # Option B: load processed data and recompute optima.
    # -------------------------------------------------------------------------
    analyzed_file_id = "2026_06_26-01_41_43-optimization_processed_full_raw_data"
    analyzed_file_id = "2026_06_30-18_16_28-optimization_processed_full_2026_06_24-23_33_18-qnami-nv0_2026_02_20"
    analyzed_file_id = "2026_07_09-13_04_44-optimization_processed_full_2026_07_08-22_48_57-qnami-nv0_2026_02_20"
    
    analyzed = dm.get_raw_data(
        file_stem=analyzed_file_id,
        load_npz=True,
    )

    new_weights = (0, 1, 1)

    print("GPU available:", GPU_AVAILABLE)

    summary = recompute_all_optimal_values_from_processed_gpu(
        analyzed,
        weights=new_weights,
    )

    print("used_gpu:", summary.get("used_gpu", False))
    print("num valid NVs:", len(summary["valid_step_vals"]))
    print("mean optimal step:", np.nanmean(summary["optimal_step_vals"]))

    if "optimal_weights" in summary:
        print("total_power:", summary.get("total_power", None))
        print("aom_voltage:", summary.get("aom_voltage", None))

    summary["source_analyzed_file"] = analyzed_file_id
    summary["weights_used"] = new_weights

    timestamp = dm.get_time_stamp()
    weights_str = "_".join(str(w) for w in new_weights)
    file_name = f"recomputed_summary_w_{weights_str}_{analyzed_file_id}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)

    # dm.save_raw_data(
    #     make_json_safe(summary),
    #     file_path,
    # )

    print("Saved recomputed summary:", file_path)

    # -------------------------------------------------------------------------
    # Plot only selected NVs. Do not plot all 1176 with block=True.
    # -------------------------------------------------------------------------
    inspect_nv_inds = [0, 10, 50, 100, 500, 1100]

    for nv_ind in inspect_nv_inds:
        if nv_ind >= int(analyzed["num_nvs"]):
            continue

        opt_step, opt_prep, opt_readout, opt_score = recompute_optimal_from_processed(
            analyzed,
            nv_ind=nv_ind,
            weights=new_weights,
        )

        print(
            f"NV {nv_ind}: opt_step={opt_step:.3f}, "
            f"prep={opt_prep:.3f}, readout={opt_readout:.3f}, "
            f"score={opt_score:.3f}"
        )

        plot_processed_nv_metrics(
            analyzed,
            nv_ind=nv_ind,
            weights=new_weights,
        )

        try:
            opt_info = recompute_optimal_step_index_from_processed(
                analyzed,
                nv_ind=nv_ind,
                weights=new_weights,
                skip_first=2,
            )

            opt_step_ind = opt_info["step_ind"]

            print(
                f"Plotting optimal histogram for NV {nv_ind}: "
                f"step_ind={opt_step_ind}, "
                f"step_val={opt_info['step_val']:.3f}"
            )

            plot_ref_histogram_from_processed(
                analyzed,
                nv_ind=nv_ind,
                step_ind=opt_step_ind,
                density=True,
            )

        except Exception as e:
            print(f"Could not plot optimal histogram for NV {nv_ind}: {e}")

    kpl.show(block=True)