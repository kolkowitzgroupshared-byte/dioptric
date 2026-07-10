# -*- coding: utf-8 -*-
"""
Repeated charge-readout survival analysis.

Experiment order for repeated_readout=True:
    exp 0 = ionized branch, readout 1
    exp 1 = ionized branch, readout 2
    exp 2 = reference/no-ionization branch, readout 1
    exp 3 = reference/no-ionization branch, readout 2

Goal:
    Optimize readout amplitude/duration using both:
        1. Single-shot charge classification fidelity.
        2. Charge-state survival between readout 1 and readout 2 with no re-prep.

Physical interpretation:
    The ionized branch is used to provide the NV0-like population for fitting a
    bimodal threshold. The reference branch is used to test whether the readout
    itself changes the charge state.

Created July 2026
@author: Saroj Chand
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

# Compatibility patch for old labrad with newer NumPy
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# Basic helpers
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


def _base_file_stem(raw_data):
    stem = (
        raw_data.get("file_stem")
        or raw_data.get("file_name")
        or raw_data.get("timestamp")
        or "raw_data"
    )
    if isinstance(stem, (list, tuple)):
        stem = "_".join(map(str, stem))
    return str(stem).replace(" ", "_")


def _conditional_mean(values):
    values = np.asarray(values, dtype=bool)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _norm01(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out
    xmin = np.nanmin(x[finite])
    xmax = np.nanmax(x[finite])
    out[finite] = (x[finite] - xmin) / (xmax - xmin + 1e-12)
    out[~finite] = np.nan
    return out


def _weighted_nanmean(arrays, weights):
    stack = np.stack([np.asarray(arr, dtype=float) for arr in arrays], axis=0)
    weights = np.asarray(weights, dtype=float)

    if np.any(~np.isfinite(weights)) or np.sum(weights) <= 0:
        weights = np.ones_like(weights, dtype=float)

    weights = weights / np.sum(weights)
    reshape = (weights.size,) + (1,) * (stack.ndim - 1)
    weights = weights.reshape(reshape)

    good = np.isfinite(stack)
    numerator = np.nansum(stack * weights, axis=0)
    denominator = np.sum(good * weights, axis=0)

    out = np.full(stack.shape[1:], np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _get_readout_axis(raw_data):
    """
    Convert swept step values into the analysis x-axis.

    For readout amplitude sweeps, this uses your empirical yellow AOM
    voltage-to-power calibration:
        P(uW) = a * V**b + c
    """

    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = int(raw_data["num_steps"])

    step_vals_raw = np.linspace(min_step_val, max_step_val, num_steps)

    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]

    a, b, c = 1.5133e04, 2.6976, -38.63

    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            return {
                "step_vals_raw": step_vals_raw,
                "step_vals": step_vals_raw,
                "x_label": "Polarization duration (ns)",
                "power_fit_a": a,
                "power_fit_b": b,
                "power_fit_c": c,
                "yellow_charge_readout_amp": np.nan,
            }
        return {
            "step_vals_raw": step_vals_raw,
            "step_vals": step_vals_raw,
            "x_label": "Polarization amplitude",
            "power_fit_a": a,
            "power_fit_b": b,
            "power_fit_c": c,
            "yellow_charge_readout_amp": np.nan,
        }

    if optimize_duration_or_amp:
        return {
            "step_vals_raw": step_vals_raw,
            "step_vals": step_vals_raw * 1e-6,
            "x_label": "Readout duration (ms)",
            "power_fit_a": a,
            "power_fit_b": b,
            "power_fit_c": c,
            "yellow_charge_readout_amp": np.nan,
        }

    yellow_charge_readout_amp = raw_data["opx_config"]["waveforms"][
        "yellow_charge_readout"
    ]["sample"]

    aom_voltage = step_vals_raw * yellow_charge_readout_amp
    readout_power_uW = a * (aom_voltage**b) + c

    return {
        "step_vals_raw": step_vals_raw,
        "step_vals": readout_power_uW,
        "x_label": "Readout amplitude (uW)",
        "power_fit_a": a,
        "power_fit_b": b,
        "power_fit_c": c,
        "yellow_charge_readout_amp": float(yellow_charge_readout_amp),
    }


def _aom_voltage_from_step_value(step_val, axis_info):
    if axis_info["x_label"] != "Readout amplitude (uW)":
        return np.nan

    a = float(axis_info["power_fit_a"])
    b = float(axis_info["power_fit_b"])
    c = float(axis_info["power_fit_c"])

    if not np.isfinite(step_val) or step_val <= c:
        return np.nan

    return float(((step_val - c) / a) ** (1 / b))


# =============================================================================
# Fitting and per-NV/step metrics
# =============================================================================


def _fit_bimodal_threshold(counts_data, prob_dist=ProbDist.COMPOUND_POISSON):
    try:
        popt, pcov, red_chi_sq = fit_bimodal_histogram(
            counts_data,
            prob_dist,
            no_plot=True,
        )

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
            popt,
            prob_dist,
            dark_mode_weight=0.5,
            ret_fidelity=True,
        )

        return {
            "threshold": float(threshold),
            "readout_fidelity": float(readout_fidelity),
            "prep_fidelity": float(1.0 - popt[0]),
            "goodness_of_fit": float(red_chi_sq),
            "fit_success": True,
            "fit_params": np.asarray(popt, dtype=float),
        }

    except Exception:
        return {
            "threshold": np.nan,
            "readout_fidelity": np.nan,
            "prep_fidelity": np.nan,
            "goodness_of_fit": np.nan,
            "fit_success": False,
            "fit_params": None,
            "error": traceback.format_exc(),
        }


def _process_repeated_readout_nv_step(
    nv_ind,
    step_ind,
    counts,
    prob_dist_name="COMPOUND_POISSON",
):
    """
    Process one NV and one step.

    Threshold/fidelity:
        Fit ionized R1 + reference R1.

    Survival:
        Classify reference R1 and reference R2 using the R1 threshold.
        This directly tests whether readout 1 perturbs the charge state before
        readout 2.
    """

    prob_dist = ProbDist[prob_dist_name]

    ion_r1 = counts[0, nv_ind, :, step_ind, :].flatten()
    ion_r2 = counts[1, nv_ind, :, step_ind, :].flatten()
    ref_r1 = counts[2, nv_ind, :, step_ind, :].flatten()
    ref_r2 = counts[3, nv_ind, :, step_ind, :].flatten()

    r1_for_fit = np.concatenate([ion_r1, ref_r1])
    r2_for_fit = np.concatenate([ion_r2, ref_r2])

    fit_r1 = _fit_bimodal_threshold(r1_for_fit, prob_dist=prob_dist)
    fit_r2 = _fit_bimodal_threshold(r2_for_fit, prob_dist=prob_dist)

    threshold = fit_r1["threshold"]

    out = {
        "nv_ind": int(nv_ind),
        "step_ind": int(step_ind),
        "threshold": threshold,
        "readout1_fidelity": fit_r1["readout_fidelity"],
        "readout2_fidelity": fit_r2["readout_fidelity"],
        "prep1_fidelity": fit_r1["prep_fidelity"],
        "prep2_fidelity": fit_r2["prep_fidelity"],
        "goodness1_of_fit": fit_r1["goodness_of_fit"],
        "goodness2_of_fit": fit_r2["goodness_of_fit"],
        "fit1_success": fit_r1["fit_success"],
        "fit2_success": fit_r2["fit_success"],
        "fit1_params": fit_r1["fit_params"],
        "fit2_params": fit_r2["fit_params"],
        "ref_same_state_survival": np.nan,
        "ref_nvm_survival": np.nan,
        "ref_nv0_survival": np.nan,
        "ref_nvm_to_nv0_prob": np.nan,
        "ref_nv0_to_nvm_prob": np.nan,
        "ion_same_state_survival": np.nan,
        "ion_nv0_survival": np.nan,
        "ion_nv0_to_nvm_prob": np.nan,
        "mean_ion_r1": float(np.nanmean(ion_r1)),
        "mean_ion_r2": float(np.nanmean(ion_r2)),
        "mean_ref_r1": float(np.nanmean(ref_r1)),
        "mean_ref_r2": float(np.nanmean(ref_r2)),
    }

    if not np.isfinite(threshold):
        return out

    n_ref = min(ref_r1.size, ref_r2.size)
    ref_s1 = ref_r1[:n_ref] > threshold
    ref_s2 = ref_r2[:n_ref] > threshold

    out["ref_same_state_survival"] = float(np.mean(ref_s1 == ref_s2))

    if np.any(ref_s1):
        ref_nvm_survival = float(np.mean(ref_s2[ref_s1]))
        out["ref_nvm_survival"] = ref_nvm_survival
        out["ref_nvm_to_nv0_prob"] = 1.0 - ref_nvm_survival

    if np.any(~ref_s1):
        ref_nv0_survival = float(np.mean(~ref_s2[~ref_s1]))
        out["ref_nv0_survival"] = ref_nv0_survival
        out["ref_nv0_to_nvm_prob"] = 1.0 - ref_nv0_survival

    # Ionized branch is a diagnostic/check, not the main optimization target.
    n_ion = min(ion_r1.size, ion_r2.size)
    ion_s1 = ion_r1[:n_ion] > threshold
    ion_s2 = ion_r2[:n_ion] > threshold

    out["ion_same_state_survival"] = float(np.mean(ion_s1 == ion_s2))
    if np.any(~ion_s1):
        ion_nv0_survival = float(np.mean(~ion_s2[~ion_s1]))
        out["ion_nv0_survival"] = ion_nv0_survival
        out["ion_nv0_to_nvm_prob"] = 1.0 - ion_nv0_survival

    return out


def _allocate_metric_arrays(num_nvs, num_steps):
    arr = lambda: np.full((num_nvs, num_steps), np.nan, dtype=float)
    barr = lambda: np.zeros((num_nvs, num_steps), dtype=bool)
    oarr = lambda: np.empty((num_nvs, num_steps), dtype=object)

    return {
        "threshold_arr": arr(),
        "readout1_fidelity_arr": arr(),
        "readout2_fidelity_arr": arr(),
        "prep1_fidelity_arr": arr(),
        "prep2_fidelity_arr": arr(),
        "goodness1_of_fit_arr": arr(),
        "goodness2_of_fit_arr": arr(),
        "fit1_success_arr": barr(),
        "fit2_success_arr": barr(),
        "fit1_params_arr": oarr(),
        "fit2_params_arr": oarr(),
        "ref_same_state_survival_arr": arr(),
        "ref_nvm_survival_arr": arr(),
        "ref_nv0_survival_arr": arr(),
        "ref_nvm_to_nv0_prob_arr": arr(),
        "ref_nv0_to_nvm_prob_arr": arr(),
        "ion_same_state_survival_arr": arr(),
        "ion_nv0_survival_arr": arr(),
        "ion_nv0_to_nvm_prob_arr": arr(),
        "mean_ion_r1_arr": arr(),
        "mean_ion_r2_arr": arr(),
        "mean_ref_r1_arr": arr(),
        "mean_ref_r2_arr": arr(),
    }


def _fill_metric_arrays(flat_results, num_nvs, num_steps):
    arrays = _allocate_metric_arrays(num_nvs, num_steps)

    for res in flat_results:
        nv_ind = int(res["nv_ind"])
        step_ind = int(res["step_ind"])

        for key in arrays:
            if key.endswith("_arr"):
                base_key = key[:-4]
                if base_key in res:
                    arrays[key][nv_ind, step_ind] = res[base_key]

        arrays["fit1_success_arr"][nv_ind, step_ind] = bool(res["fit1_success"])
        arrays["fit2_success_arr"][nv_ind, step_ind] = bool(res["fit2_success"])
        arrays["fit1_params_arr"][nv_ind, step_ind] = res["fit1_params"]
        arrays["fit2_params_arr"][nv_ind, step_ind] = res["fit2_params"]

    return arrays


# =============================================================================
# Optimization logic
# =============================================================================


def compute_repeated_readout_score(
    readout1_fidelity,
    readout2_fidelity,
    ref_same_state_survival,
    ref_nvm_survival,
    goodness1_of_fit,
    goodness2_of_fit,
    step_vals,
    ref_nv0_survival=None,
    score_weights=(0.35, 0.35, 0.20, 0.10),
):
    """
    Non-destructive two-readout score.

    Components:
        two_readout_fidelity:
            min(R1 fidelity, R2 fidelity)
            This forces both readouts to work.

        nondestructive_survival:
            survival between R1 and R2.
            High value means R1 did not disturb the charge state.

        fit_quality:
            favors well-fit histograms.

        low_power:
            weak preference for lower readout amplitude/duration.

    score_weights:
        (w_fidelity, w_survival, w_fit_quality, w_low_power)
    """

    w_fid, w_survival, w_fit, w_low = score_weights

    readout1_fidelity = np.asarray(readout1_fidelity, dtype=float)
    readout2_fidelity = np.asarray(readout2_fidelity, dtype=float)
    ref_same_state_survival = np.asarray(ref_same_state_survival, dtype=float)
    ref_nvm_survival = np.asarray(ref_nvm_survival, dtype=float)
    goodness1_of_fit = np.asarray(goodness1_of_fit, dtype=float)
    goodness2_of_fit = np.asarray(goodness2_of_fit, dtype=float)
    step_vals = np.asarray(step_vals, dtype=float)

    # Both readouts must be good. If one is bad, this is bad.
    two_readout_fidelity = np.minimum(readout1_fidelity, readout2_fidelity)

    # Non-destructive survival. Main one is NV- survival, because you care
    # about preserving reference NV- after the first readout.
    if ref_nv0_survival is None:
        nondestructive_survival = 0.5 * (
            ref_same_state_survival + ref_nvm_survival
        )
    else:
        ref_nv0_survival = np.asarray(ref_nv0_survival, dtype=float)
        nondestructive_survival = (
            0.40 * ref_same_state_survival
            + 0.40 * ref_nvm_survival
            + 0.20 * ref_nv0_survival
        )

    # Lower chi-square is better.
    fit_quality = 0.5 * (1.0 - _norm01(goodness1_of_fit)) + 0.5 * (
        1.0 - _norm01(goodness2_of_fit)
    )

    low_power = 1.0 - _norm01(step_vals)

    score = (
        w_fid * _norm01(two_readout_fidelity)
        + w_survival * _norm01(nondestructive_survival)
        + w_fit * fit_quality
        + w_low * low_power
    )

    bad = (
        ~np.isfinite(readout1_fidelity)
        | ~np.isfinite(readout2_fidelity)
        | ~np.isfinite(ref_same_state_survival)
        | ~np.isfinite(ref_nvm_survival)
        | ~np.isfinite(goodness1_of_fit)
        | ~np.isfinite(goodness2_of_fit)
    )

    score = np.where(bad, np.nan, score)
    return score


def choose_lowest_step_passing_criteria(
    step_vals,
    readout1_fidelity,
    readout2_fidelity,
    ref_same_state_survival,
    ref_nvm_survival,
    ref_nv0_survival=None,
    min_readout1_fidelity=0.85,
    min_readout2_fidelity=0.85,
    min_ref_same_state_survival=0.95,
    min_ref_nvm_survival=0.95,
    min_ref_nv0_survival=None,
):
    good = (
        np.isfinite(readout1_fidelity)
        & np.isfinite(readout2_fidelity)
        & np.isfinite(ref_same_state_survival)
        & np.isfinite(ref_nvm_survival)
        & (readout1_fidelity >= min_readout1_fidelity)
        & (readout2_fidelity >= min_readout2_fidelity)
        & (ref_same_state_survival >= min_ref_same_state_survival)
        & (ref_nvm_survival >= min_ref_nvm_survival)
    )

    if min_ref_nv0_survival is not None and ref_nv0_survival is not None:
        good = (
            good
            & np.isfinite(ref_nv0_survival)
            & (ref_nv0_survival >= min_ref_nv0_survival)
        )

    if np.any(good):
        ind = int(np.where(good)[0][0])
        return ind, "lowest step satisfying thresholds"

    return None, "thresholds not all satisfied"


def choose_optimal_step(
    step_vals,
    readout1_fidelity,
    readout2_fidelity,
    ref_same_state_survival,
    ref_nvm_survival,
    ref_nv0_survival,
    goodness1_of_fit,
    goodness2_of_fit,
    criteria,
    score_weights,
):
    ind, reason = choose_lowest_step_passing_criteria(
        step_vals,
        readout1_fidelity,
        readout2_fidelity,
        ref_same_state_survival,
        ref_nvm_survival,
        ref_nv0_survival=ref_nv0_survival,
        **criteria,
    )

    score = compute_repeated_readout_score(
        readout1_fidelity,
        readout2_fidelity,
        ref_same_state_survival,
        ref_nvm_survival,
        goodness1_of_fit,
        goodness2_of_fit,
        step_vals,
        ref_nv0_survival=ref_nv0_survival,
        score_weights=score_weights,
    )

    if ind is not None:
        return ind, reason, score

    score_safe = np.where(np.isfinite(score), score, -np.inf)
    if np.all(~np.isfinite(score_safe)) or np.nanmax(score_safe) == -np.inf:
        return None, "no finite score", score

    return int(np.nanargmax(score_safe)), "fallback max score", score


def _metric_at(arr, nv_ind, step_ind):
    if step_ind is None:
        return np.nan
    return float(np.asarray(arr, dtype=float)[nv_ind, step_ind])


def _compute_per_nv_optima(arrays, step_vals, axis_info, criteria, score_weights):
    num_nvs, num_steps = arrays["readout1_fidelity_arr"].shape

    optimal_values = []
    optimal_step_vals = []
    optimal_step_inds = []

    score_arr = np.full((num_nvs, num_steps), np.nan, dtype=float)

    for nv_ind in range(num_nvs):
        step_ind, reason, score = choose_optimal_step(
            step_vals,
            arrays["readout1_fidelity_arr"][nv_ind],
            arrays["readout2_fidelity_arr"][nv_ind],
            arrays["ref_same_state_survival_arr"][nv_ind],
            arrays["ref_nvm_survival_arr"][nv_ind],
            arrays["ref_nv0_survival_arr"][nv_ind],
            arrays["goodness1_of_fit_arr"][nv_ind],
            arrays["goodness2_of_fit_arr"][nv_ind],
            criteria=criteria,
            score_weights=score_weights,
        )

        score_arr[nv_ind, :] = score

        if step_ind is None:
            optimal_step_inds.append(-1)
            optimal_step_vals.append(np.nan)
            optimal_values.append(
                {
                    "nv_ind": int(nv_ind),
                    "optimal_step_ind": None,
                    "optimal_step_val": np.nan,
                    "reason": reason,
                    "score": np.nan,
                    "readout1_fidelity": np.nan,
                    "readout2_fidelity": np.nan,
                    "ref_same_state_survival": np.nan,
                    "ref_nvm_survival": np.nan,
                    "ref_nv0_survival": np.nan,
                    "aom_voltage": np.nan,
                }
            )
            continue

        step_val = float(step_vals[step_ind])
        optimal_step_inds.append(int(step_ind))
        optimal_step_vals.append(step_val)

        optimal_values.append(
            {
                "nv_ind": int(nv_ind),
                "optimal_step_ind": int(step_ind),
                "optimal_step_val": step_val,
                "reason": reason,
                "score": _metric_at(score_arr, nv_ind, step_ind),
                "readout1_fidelity": _metric_at(
                    arrays["readout1_fidelity_arr"], nv_ind, step_ind
                ),
                "readout2_fidelity": _metric_at(
                    arrays["readout2_fidelity_arr"], nv_ind, step_ind
                ),
                "ref_same_state_survival": _metric_at(
                    arrays["ref_same_state_survival_arr"], nv_ind, step_ind
                ),
                "ref_nvm_survival": _metric_at(
                    arrays["ref_nvm_survival_arr"], nv_ind, step_ind
                ),
                "ref_nv0_survival": _metric_at(
                    arrays["ref_nv0_survival_arr"], nv_ind, step_ind
                ),
                "aom_voltage": _aom_voltage_from_step_value(step_val, axis_info),
            }
        )

    return {
        "optimal_values": optimal_values,
        "optimal_step_inds": np.asarray(optimal_step_inds, dtype=int),
        "optimal_step_vals": np.asarray(optimal_step_vals, dtype=float),
        "score_arr": score_arr,
    }


# =============================================================================
# Main analysis
# =============================================================================


def process_repeated_readout_survival(
    raw_data,
    do_plot=True,
    save_data=True,
    n_jobs=12,
    joblib_verbose=10,
    min_readout1_fidelity=0.85,
    min_readout2_fidelity=0.85,
    min_ref_same_state_survival=0.95,
    min_ref_nvm_survival=0.95,
    min_ref_nv0_survival=None,
    score_weights=(0.25, 0.20, 0.20, 0.30, 0.05),
):
    """
    Full repeated-readout analysis from already-collected raw data.

    Expected counts shape:
        counts[exp, nv, run, step, rep]

    Expected exp order:
        exp 0 = ionized branch, readout 1
        exp 1 = ionized branch, readout 2
        exp 2 = reference/no-ionization branch, readout 1
        exp 3 = reference/no-ionization branch, readout 2
    """

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = int(raw_data["num_steps"])
    counts = np.asarray(raw_data["counts"], dtype=float)

    if counts.shape[0] < 4:
        raise ValueError(
            f"Expected at least 4 experiments for repeated readout, got {counts.shape[0]}"
        )

    axis_info = _get_readout_axis(raw_data)
    step_vals = np.asarray(axis_info["step_vals"], dtype=float)
    prob_dist = ProbDist.COMPOUND_POISSON

    criteria = {
        "min_readout1_fidelity": float(min_readout1_fidelity),
        "min_readout2_fidelity": float(min_readout2_fidelity),
        "min_ref_same_state_survival": float(min_ref_same_state_survival),
        "min_ref_nvm_survival": float(min_ref_nvm_survival),
        "min_ref_nv0_survival": (
            None
            if min_ref_nv0_survival is None
            else float(min_ref_nv0_survival)
        ),
    }

    print("\n=== Starting repeated-readout survival analysis ===")
    print(f"num_nvs: {num_nvs}")
    print(f"num_steps: {num_steps}")
    print(f"total fits per readout: {num_nvs * num_steps}")
    print(f"n_jobs: {n_jobs}")
    print(f"x axis: {axis_info['x_label']}")
    print("exp order: 0=ion R1, 1=ion R2, 2=ref R1, 3=ref R2")

    tasks = [
        (nv_ind, step_ind, counts, prob_dist.name)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    ]

    if n_jobs is None or int(n_jobs) == 1:
        flat_results = [_process_repeated_readout_nv_step(*task) for task in tasks]
    else:
        flat_results = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            verbose=joblib_verbose,
            batch_size="auto",
            pre_dispatch="2*n_jobs",
        )(
            delayed(_process_repeated_readout_nv_step)(*task)
            for task in tasks
        )

    arrays = _fill_metric_arrays(flat_results, num_nvs, num_steps)

    median = {
        "readout1_fidelity": np.nanmedian(arrays["readout1_fidelity_arr"], axis=0),
        "readout2_fidelity": np.nanmedian(arrays["readout2_fidelity_arr"], axis=0),
        "prep1_fidelity": np.nanmedian(arrays["prep1_fidelity_arr"], axis=0),
        "prep2_fidelity": np.nanmedian(arrays["prep2_fidelity_arr"], axis=0),
        "goodness1_of_fit": np.nanmedian(arrays["goodness1_of_fit_arr"], axis=0),
        "goodness2_of_fit": np.nanmedian(arrays["goodness2_of_fit_arr"], axis=0),
        "ref_same_state_survival": np.nanmedian(
            arrays["ref_same_state_survival_arr"], axis=0
        ),
        "ref_nvm_survival": np.nanmedian(arrays["ref_nvm_survival_arr"], axis=0),
        "ref_nv0_survival": np.nanmedian(arrays["ref_nv0_survival_arr"], axis=0),
        "ref_nvm_to_nv0_prob": np.nanmedian(
            arrays["ref_nvm_to_nv0_prob_arr"], axis=0
        ),
        "mean_ion_r1": np.nanmedian(arrays["mean_ion_r1_arr"], axis=0),
        "mean_ion_r2": np.nanmedian(arrays["mean_ion_r2_arr"], axis=0),
        "mean_ref_r1": np.nanmedian(arrays["mean_ref_r1_arr"], axis=0),
        "mean_ref_r2": np.nanmedian(arrays["mean_ref_r2_arr"], axis=0),
    }

    avg = {
        "readout1_fidelity": np.nanmean(arrays["readout1_fidelity_arr"], axis=0),
        "readout2_fidelity": np.nanmean(arrays["readout2_fidelity_arr"], axis=0),
        "ref_same_state_survival": np.nanmean(
            arrays["ref_same_state_survival_arr"], axis=0
        ),
        "ref_nvm_survival": np.nanmean(arrays["ref_nvm_survival_arr"], axis=0),
        "ref_nv0_survival": np.nanmean(arrays["ref_nv0_survival_arr"], axis=0),
    }

    population_step_ind, population_reason, population_score = choose_optimal_step(
        step_vals,
        median["readout1_fidelity"],
        median["readout2_fidelity"],
        median["ref_same_state_survival"],
        median["ref_nvm_survival"],
        median["ref_nv0_survival"],
        median["goodness1_of_fit"],
        median["goodness2_of_fit"],
        criteria=criteria,
        score_weights=score_weights,
    )

    per_nv = _compute_per_nv_optima(
        arrays,
        step_vals,
        axis_info,
        criteria=criteria,
        score_weights=score_weights,
    )

    if population_step_ind is None:
        population_step_val = np.nan
        population_aom_voltage = np.nan
    else:
        population_step_val = float(step_vals[population_step_ind])
        population_aom_voltage = _aom_voltage_from_step_value(
            population_step_val,
            axis_info,
        )

    valid_step_vals = per_nv["optimal_step_vals"][
        np.isfinite(per_nv["optimal_step_vals"])
    ]

    if valid_step_vals.size > 0:
        total_power = float(np.nanmean(valid_step_vals))
        optimal_weights = valid_step_vals / total_power
    else:
        total_power = np.nan
        optimal_weights = np.asarray([], dtype=float)

    results = {
        "analysis_type": "repeated_readout_survival",
        "file_stem_source": _base_file_stem(raw_data),
        "num_nvs": int(num_nvs),
        "num_steps": int(num_steps),
        "step_vals_raw": axis_info["step_vals_raw"].tolist(),
        "step_vals": step_vals.tolist(),
        "x_label": axis_info["x_label"],
        "power_fit_a": float(axis_info["power_fit_a"]),
        "power_fit_b": float(axis_info["power_fit_b"]),
        "power_fit_c": float(axis_info["power_fit_c"]),
        "yellow_charge_readout_amp": float(axis_info["yellow_charge_readout_amp"])
        if np.isfinite(axis_info["yellow_charge_readout_amp"])
        else None,
        "exp_order": {
            "0": "ionized_readout_1",
            "1": "ionized_readout_2",
            "2": "reference_readout_1",
            "3": "reference_readout_2",
        },
        "criteria": criteria,
        "score_weights": tuple(float(v) for v in score_weights),
        "population_optimal_step_ind": None
        if population_step_ind is None
        else int(population_step_ind),
        "population_optimal_step_val": float(population_step_val)
        if np.isfinite(population_step_val)
        else None,
        "population_aom_voltage": float(population_aom_voltage)
        if np.isfinite(population_aom_voltage)
        else None,
        "population_optimal_reason": population_reason,
        "population_score": population_score.tolist(),
        "per_nv_optimal_values": make_json_safe(per_nv["optimal_values"]),
        "per_nv_optimal_step_inds": per_nv["optimal_step_inds"].tolist(),
        "per_nv_optimal_step_vals": per_nv["optimal_step_vals"].tolist(),
        "per_nv_score_arr": per_nv["score_arr"].tolist(),
        "valid_step_vals": valid_step_vals.tolist(),
        "total_power": float(total_power) if np.isfinite(total_power) else None,
        "optimal_weights": optimal_weights.tolist(),
        "median": {key: val.tolist() for key, val in median.items()},
        "avg": {key: val.tolist() for key, val in avg.items()},
    }

    for key, val in arrays.items():
        if val.dtype == object:
            results[key] = _fit_params_to_list_object(val, num_nvs, num_steps)
        else:
            results[key] = val.tolist()

    print("\n=== Repeated-readout survival optimum ===")
    print("Population optimal step index:", results["population_optimal_step_ind"])
    print(f"Population optimal {axis_info['x_label']}: {population_step_val:.4g}")
    print("Population AOM voltage:", results["population_aom_voltage"])
    print("Reason:", population_reason)

    if population_step_ind is not None:
        i = population_step_ind
        print(
            "At population optimum: "
            f"R1 fid={median['readout1_fidelity'][i]:.3f}, "
            f"R2 fid={median['readout2_fidelity'][i]:.3f}, "
            f"ref same={median['ref_same_state_survival'][i]:.3f}, "
            f"ref NV- survival={median['ref_nvm_survival'][i]:.3f}, "
            f"ref NV0 survival={median['ref_nv0_survival'][i]:.3f}"
        )

    if valid_step_vals.size > 0:
        print("Per-NV mean optimal step:", float(np.nanmean(valid_step_vals)))
        print("Per-NV median optimal step:", float(np.nanmedian(valid_step_vals)))
        print("Number valid NVs:", int(valid_step_vals.size))
        
    if "optimal_weights" in results and len(results["optimal_weights"]) > 0:
        weights = np.asarray(results["optimal_weights"], dtype=float)
        print("\n=== Per-NV optimal readout weights ===")
        print("mean weight:", float(np.nanmean(weights)))
        print("median weight:", float(np.nanmedian(weights)))
        print("min weight:", float(np.nanmin(weights)))
        print("max weight:", float(np.nanmax(weights)))
        print("std weight:", float(np.nanstd(weights)))

    if save_data:
        timestamp = dm.get_time_stamp()
        file_name = f"repeated_readout_survival_processed_{_base_file_stem(raw_data)}"
        file_path = dm.get_file_path(__file__, timestamp, file_name)
        dm.save_raw_data(make_json_safe(results), file_path)
        results["saved_file_path"] = str(file_path)
        print("Saved repeated-readout survival analysis:", file_path)

    if do_plot:
        plot_repeated_readout_survival_summary(results)
        plot_repeated_readout_all_nv_scatters(results)
        plot_per_nv_optimal_step_distribution(results)

    return results


def _fit_params_to_list_object(fit_params_arr, num_nvs, num_steps):
    return [
        [
            None
            if fit_params_arr[nv_ind, step_ind] is None
            else np.asarray(fit_params_arr[nv_ind, step_ind], dtype=float)
            .ravel()
            .tolist()
            for step_ind in range(num_steps)
        ]
        for nv_ind in range(num_nvs)
    ]


def _population_optimum(results):
    opt_ind = results.get("population_optimal_step_ind", None)
    opt_val = results.get("population_optimal_step_val", None)
    if opt_ind is None or opt_val is None:
        return None, np.nan
    return int(opt_ind), float(opt_val)


def plot_repeated_readout_survival_summary(results):
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    opt_ind, opt_val = _population_optimum(results)

    med = results["median"]

    r1 = np.asarray(med["readout1_fidelity"], dtype=float)
    r2 = np.asarray(med["readout2_fidelity"], dtype=float)
    same = np.asarray(med["ref_same_state_survival"], dtype=float)
    nvm = np.asarray(med["ref_nvm_survival"], dtype=float)
    nv0 = np.asarray(med["ref_nv0_survival"], dtype=float)
    ion = np.asarray(med["mean_ion_r1"], dtype=float)
    ref1 = np.asarray(med["mean_ref_r1"], dtype=float)
    ref2 = np.asarray(med["mean_ref_r2"], dtype=float)
    score = np.asarray(results["population_score"], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    axes[0].plot(step_vals, r1, "o-", label="Readout 1 fidelity")
    axes[0].plot(step_vals, r2, "o-", label="Readout 2 fidelity")
    axes[0].set_ylabel("Fidelity")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()

    axes[1].plot(step_vals, same, "o-", label="Ref same-state survival")
    axes[1].plot(step_vals, nvm, "o-", label="Ref NV- survival")
    axes[1].plot(step_vals, nv0, "o-", label="Ref NV0 survival")
    axes[1].set_ylabel("Survival")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    axes[2].plot(step_vals, ion, "o-", label="Median ion R1 counts")
    axes[2].plot(step_vals, ref1, "o-", label="Median ref R1 counts")
    axes[2].plot(step_vals, ref2, "o-", label="Median ref R2 counts")
    axes[2].set_ylabel("Counts")
    axes[2].legend()

    axes[3].plot(step_vals, score, "o-", label="Population score")
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel(x_label)
    axes[3].legend()

    for ax in axes:
        if np.isfinite(opt_val):
            ax.axvline(
                opt_val,
                color="red",
                linestyle="--",
                label=f"Optimal = {opt_val:.3g}",
            )
        ax.grid(True, linestyle="--", alpha=0.5)

    title = (
        "Repeated-readout optimization\n"
        "classification + no-reprep reference survival"
    )
    if opt_ind is not None:
        title += f"\nchosen index {opt_ind}, {x_label}={opt_val:.4g}"
    fig.suptitle(title, fontsize=15)
    return fig


def plot_repeated_readout_all_nv_scatters(results, alpha=0.28, size=12):
    """
    Meaningful scatter plots over all NVs and all steps.

    These show whether high readout fidelity is being bought at the price of
    readout-induced charge conversion.
    """

    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    opt_ind, opt_val = _population_optimum(results)

    r1 = np.asarray(results["readout1_fidelity_arr"], dtype=float)
    r2 = np.asarray(results["readout2_fidelity_arr"], dtype=float)
    same = np.asarray(results["ref_same_state_survival_arr"], dtype=float)
    nvm = np.asarray(results["ref_nvm_survival_arr"], dtype=float)
    nv0 = np.asarray(results["ref_nv0_survival_arr"], dtype=float)
    nvm_to_nv0 = np.asarray(results["ref_nvm_to_nv0_prob_arr"], dtype=float)
    score = np.asarray(results["per_nv_score_arr"], dtype=float)

    x = np.tile(step_vals, r1.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)
    axes = axes.ravel()

    panels = [
        (r1, "R1 fidelity", "Fidelity"),
        (r2, "R2 fidelity", "Fidelity"),
        (same, "Reference same-state survival", "Survival"),
        (nvm, "Reference NV- survival", "Survival"),
        (nvm_to_nv0, "Reference NV- -> NV0 probability", "Ionization probability"),
        (score, "Per-NV combined score", "Score"),
    ]

    for ax, (arr, title, ylabel) in zip(axes, panels):
        y = arr.reshape(-1)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=size, alpha=alpha)
        med = np.nanmedian(arr, axis=0)
        ax.plot(step_vals, med, color="black", lw=2, label="Median")
        if np.isfinite(opt_val):
            ax.axvline(opt_val, color="red", linestyle="--", label="Optimal")
        ax.set_title(title,fontsize=15)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle("All-NV repeated-readout scatter diagnostics", fontsize=15)
    return fig


def plot_per_nv_optimal_step_distribution(results):
    step_vals = np.asarray(results["per_nv_optimal_step_vals"], dtype=float)
    valid = step_vals[np.isfinite(step_vals)]
    x_label = results["x_label"]
    opt_ind, opt_val = _population_optimum(results)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid, bins=40, alpha=0.8, color="tab:blue")

    if valid.size > 0:
        mean_val = float(np.nanmean(valid))
        median_val = float(np.nanmedian(valid))
        ax.axvline(mean_val, color="black", linestyle=":", label=f"Mean = {mean_val:.3g}")
        ax.axvline(
            median_val,
            color="green",
            linestyle="-.",
            label=f"Median = {median_val:.3g}",
        )

    if np.isfinite(opt_val):
        ax.axvline(
            opt_val,
            color="red",
            linestyle="--",
            label=f"Population optimal = {opt_val:.3g}",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of NVs")
    ax.set_title("Distribution of per-NV optimal readout settings", fontsize=15)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def plot_optimum_metric_scatter(results):
    """
    Scatter each NV at its own selected optimum.
    """

    opt_vals = results["per_nv_optimal_values"]

    r1 = np.asarray([v["readout1_fidelity"] for v in opt_vals], dtype=float)
    r2 = np.asarray([v["readout2_fidelity"] for v in opt_vals], dtype=float)
    same = np.asarray([v["ref_same_state_survival"] for v in opt_vals], dtype=float)
    nvm = np.asarray([v["ref_nvm_survival"] for v in opt_vals], dtype=float)
    step = np.asarray([v["optimal_step_val"] for v in opt_vals], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    sc0 = axes[0].scatter(r1, nvm, c=step, s=18, alpha=0.75)
    axes[0].set_xlabel("R1 fidelity")
    axes[0].set_ylabel("Ref NV- survival")
    axes[0].set_title("Readout fidelity vs ionization survival",fontsize=15)
    plt.colorbar(sc0, ax=axes[0], label=results["x_label"])

    sc1 = axes[1].scatter(r2, nvm, c=step, s=18, alpha=0.75)
    axes[1].set_xlabel("R2 fidelity")
    axes[1].set_ylabel("Ref NV- survival")
    axes[1].set_title("R2 fidelity vs ionization survival",fontsize=15)
    plt.colorbar(sc1, ax=axes[1], label=results["x_label"])

    sc2 = axes[2].scatter(same, nvm, c=step, s=18, alpha=0.75)
    axes[2].set_xlabel("Ref same-state survival")
    axes[2].set_ylabel("Ref NV- survival")
    axes[2].set_title("Survival consistency", fontsize=15)
    plt.colorbar(sc2, ax=axes[2], label=results["x_label"])

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)

    fig.suptitle("All NVs at their own selected optimum")
    return fig


def pick_representative_nvs(results):
    opt_vals = results["per_nv_optimal_values"]

    score = np.asarray([v["score"] for v in opt_vals], dtype=float)
    nvm = np.asarray([v["ref_nvm_survival"] for v in opt_vals], dtype=float)
    r1 = np.asarray([v["readout1_fidelity"] for v in opt_vals], dtype=float)

    valid = np.isfinite(score)
    valid_inds = np.where(valid)[0]

    if valid_inds.size == 0:
        raise ValueError("No valid NVs for representative selection.")

    good_nv = int(valid_inds[np.nanargmax(score[valid_inds])])
    low_survival_nv = int(valid_inds[np.nanargmin(nvm[valid_inds])])
    low_fidelity_nv = int(valid_inds[np.nanargmin(r1[valid_inds])])

    median_score = np.nanmedian(score[valid_inds])
    median_nv = int(valid_inds[np.nanargmin(np.abs(score[valid_inds] - median_score))])

    return {
        "good_nv": good_nv,
        "low_nv_minus_survival_nv": low_survival_nv,
        "low_readout_fidelity_nv": low_fidelity_nv,
        "median_nv": median_nv,
    }


def print_nv_optimum_summary(results, nv_ind):
    opt = results["per_nv_optimal_values"][nv_ind]

    print("\n=== NV repeated-readout optimum ===")
    print("NV:", int(nv_ind))
    print("step index:", opt["optimal_step_ind"])
    print(f"{results['x_label']}:", opt["optimal_step_val"])
    print("reason:", opt["reason"])
    print("score:", opt["score"])
    print("R1 fidelity:", opt["readout1_fidelity"])
    print("R2 fidelity:", opt["readout2_fidelity"])
    print("ref same-state survival:", opt["ref_same_state_survival"])
    print("ref NV- survival:", opt["ref_nvm_survival"])
    print("ref NV0 survival:", opt["ref_nv0_survival"])
    print("aom voltage:", opt["aom_voltage"])


def _format_fit_params_for_display(popt, prob_dist_local, red_chi_sq=None):
    """
    Human-readable fit parameter string for display in text box.
    """
    if popt is None:
        return "fit params: None"

    popt = np.asarray(popt, dtype=float).ravel()
    num_single = bimodal_histogram.get_single_mode_num_params(prob_dist_local)

    try:
        dark_weight = float(popt[0])
        bright_weight = 1.0 - dark_weight

        dark_params = popt[1 : 1 + num_single]
        bright_params = popt[1 + num_single : 1 + 2 * num_single]

        lines = [
            f"w0 = {dark_weight:.3f}",
            f"w- = {bright_weight:.3f}",
        ]

        if num_single == 1:
            lines += [
                f"NV0 rate = {dark_params[0]:.2f}",
                f"NV- rate = {bright_params[0]:.2f}",
            ]
        else:
            lines += [
                "NV0: " + ", ".join(f"{v:.2f}" for v in dark_params),
                "NV-: " + ", ".join(f"{v:.2f}" for v in bright_params),
            ]

        if red_chi_sq is not None and np.isfinite(red_chi_sq):
            lines.append(f"red χ² = {red_chi_sq:.3f}")

        return "\n".join(lines)

    except Exception:
        return "fit params unreadable"


def plot_two_readout_histograms_at_optimum(
    raw_data,
    analyzed_data,
    nv_ind,
    step_ind=None,
    density=True,
):
    """
    kplotlib-style two-readout histogram plot.

    Left:
        ionized R1 + reference R1 + R1 fit

    Right:
        ionized R2 + reference R2 + R2 fit

    Colors:
        ionized/reference histogram and NV0 fit = red
        reference/NV- fit = green
        combined fit = blue
    """

    counts_all = np.asarray(raw_data["counts"], dtype=float)

    if step_ind is None:
        step_ind = analyzed_data["per_nv_optimal_values"][nv_ind]["optimal_step_ind"]

    if step_ind is None or int(step_ind) < 0:
        raise ValueError(f"NV {nv_ind} has no valid optimal step.")

    step_ind = int(step_ind)

    ion_r1 = counts_all[0, nv_ind, :, step_ind, :].flatten()
    ion_r2 = counts_all[1, nv_ind, :, step_ind, :].flatten()
    ref_r1 = counts_all[2, nv_ind, :, step_ind, :].flatten()
    ref_r2 = counts_all[3, nv_ind, :, step_ind, :].flatten()

    ion_r1 = ion_r1[np.isfinite(ion_r1)]
    ion_r2 = ion_r2[np.isfinite(ion_r2)]
    ref_r1 = ref_r1[np.isfinite(ref_r1)]
    ref_r2 = ref_r2[np.isfinite(ref_r2)]

    threshold = float(
        np.asarray(analyzed_data["threshold_arr"], dtype=float)[nv_ind, step_ind]
    )

    fit1_success = bool(
        np.asarray(analyzed_data["fit1_success_arr"])[nv_ind, step_ind]
    )
    fit2_success = bool(
        np.asarray(analyzed_data["fit2_success_arr"])[nv_ind, step_ind]
    )

    fit1_params = analyzed_data["fit1_params_arr"][nv_ind][step_ind]
    fit2_params = analyzed_data["fit2_params_arr"][nv_ind][step_ind]

    gof1 = float(
        np.asarray(analyzed_data["goodness1_of_fit_arr"], dtype=float)[
            nv_ind, step_ind
        ]
    )
    gof2 = float(
        np.asarray(analyzed_data["goodness2_of_fit_arr"], dtype=float)[
            nv_ind, step_ind
        ]
    )

    step_vals = np.asarray(analyzed_data["step_vals"], dtype=float)
    x_label = analyzed_data["x_label"]
    step_val = float(step_vals[step_ind])

    prob_dist_name = analyzed_data.get("prob_dist_name", "COMPOUND_POISSON")
    prob_dist_local = ProbDist[prob_dist_name]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    plot_items = [
        {
            "title": "Readout 1",
            "ax": axes[0],
            "ion_counts": ion_r1,
            "ref_counts": ref_r1,
            "fit_success": fit1_success,
            "fit_params": fit1_params,
            "red_chi_sq": gof1,
            "r_fid": analyzed_data["per_nv_optimal_values"][nv_ind][
                "readout1_fidelity"
            ],
        },
        {
            "title": "Readout 2",
            "ax": axes[1],
            "ion_counts": ion_r2,
            "ref_counts": ref_r2,
            "fit_success": fit2_success,
            "fit_params": fit2_params,
            "red_chi_sq": gof2,
            "r_fid": analyzed_data["per_nv_optimal_values"][nv_ind][
                "readout2_fidelity"
            ],
        },
    ]

    for item in plot_items:
        ax = item["ax"]
        ion_counts = item["ion_counts"]
        ref_counts = item["ref_counts"]

        kpl.histogram(
            ax,
            ion_counts,
            density=density,
            color=kpl.KplColors.RED,
            label="Ionized branch",
        )

        kpl.histogram(
            ax,
            ref_counts,
            density=density,
            color=kpl.KplColors.GREEN,
            label="Reference branch",
        )

        ax.set_xlabel("Integrated counts")
        ax.set_ylabel("Probability" if density else "Occurrences")
        ax.set_title(
            f"{item['title']}: fidelity={item['r_fid']:.3f}", fontsize=15
        )

        if np.isfinite(threshold):
            ax.axvline(
                threshold,
                color=kpl.KplColors.GRAY,
                ls="dashed",
                label=f"Threshold = {threshold:.1f}",
            )

        if item["fit_success"] and item["fit_params"] is not None:
            popt = np.asarray(item["fit_params"], dtype=float)

            combined_counts = np.concatenate([ion_counts, ref_counts])
            x_max = max(
                np.nanmax(combined_counts) if combined_counts.size > 0 else 0,
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

            bright_mode_line = (1.0 - popt[0]) * single_mode_pdf(
                x_vals,
                *popt[1 + single_mode_num_params :],
            )

            bimodal_line = bimodal_pdf(x_vals, *popt)

            kpl.plot_line(
                ax,
                x_vals,
                dark_mode_line,
                color=kpl.KplColors.RED,
                # label="NV$^{0}$ mode",
            )

            kpl.plot_line(
                ax,
                x_vals,
                bright_mode_line,
                color=kpl.KplColors.GREEN,
                # label="NV$^{-}$ mode",
            )

            kpl.plot_line(
                ax,
                x_vals,
                bimodal_line,
                color=kpl.KplColors.BLUE,
                # label="Combined fit",
            )

            param_text = _format_fit_params_for_display(
                popt,
                prob_dist_local,
                red_chi_sq=item["red_chi_sq"],
            )

            ax.text(
                0.98,
                0.7,
                param_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )

        ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=11)

    opt = analyzed_data["per_nv_optimal_values"][nv_ind]

    fig.suptitle(
        f"NV {nv_ind}, step {step_ind}, {x_label} = {step_val:.3f}\n"
        f"ref same = {opt['ref_same_state_survival']:.3f}, "
        f"ref NV- survival = {opt['ref_nvm_survival']:.3f}, "
        f"ref NV0 survival = {opt['ref_nv0_survival']:.3f}",
        fontsize=15
    )

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    kpl.init_kplotlib()

    run_new_analysis = False

    # file_id = "2026_06_26-21_58_16-qnami-nv0_2026_02_20"
    file_id = "2026_07_09-04_28_20-qnami-nv0_2026_02_20"
    raw_data = dm.get_raw_data(
            file_stem=file_id,
            load_npz=True,
        )
    raw_data["file_stem"] = file_id
    
    if run_new_analysis:
        results = process_repeated_readout_survival(
            raw_data,
            do_plot=True,
            save_data=True,
            n_jobs=12,
            joblib_verbose=10,

            # Both readouts must classify well.
            min_readout1_fidelity=0.88,
            min_readout2_fidelity=0.88,

            # Non-destructive condition.
            min_ref_same_state_survival=0.96,
            min_ref_nvm_survival=0.97,

            # Use this if NV0 survival is meaningful/stable.
            min_ref_nv0_survival=None,

            # New score weights:
            # fidelity, survival, fit quality, low power
            score_weights=(0.35, 0.40, 0.15, 0.10),
        )

        plot_optimum_metric_scatter(results)

        reps = pick_representative_nvs(results)
        for label, nv_ind in reps.items():
            print("\n", label)
            print_nv_optimum_summary(results, nv_ind)

        kpl.show(block=True)
        sys.exit()

    processed_file = "2026_07_09-15_04_15-repeated_readout_survival_processed_2026_07_09-04_28_20-qnami-nv0_2026_02_20"
    results = dm.get_raw_data(
        file_stem=processed_file,
        load_npz=True,
    )

    plot_repeated_readout_survival_summary(results)
    # plot_repeated_readout_all_nv_scatters(results)
    # plot_per_nv_optimal_step_distribution(results)
    plot_optimum_metric_scatter(results)

    reps = pick_representative_nvs(results)
    for label, nv_ind in reps.items():
        print("\n", label)
        print_nv_optimum_summary(results, nv_ind)

        plot_two_readout_histograms_at_optimum(
            raw_data,
            results,
            nv_ind=nv_ind,
            step_ind=None,
            density=True,
        )

    kpl.show(block=True)
