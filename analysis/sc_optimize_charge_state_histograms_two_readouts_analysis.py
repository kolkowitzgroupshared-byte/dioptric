# -*- coding: utf-8 -*-
"""
Repeated two-readout charge-readout analysis + SLM spot-weight extraction.

Experiment order for repeated_readout=True
------------------------------------------
    exp 0 = ionized branch, readout 1
    exp 1 = ionized branch, readout 2
    exp 2 = reference / no-ionization branch, readout 1
    exp 3 = reference / no-ionization branch, readout 2

Expected counts shape
---------------------
    counts[exp, nv, run, step, rep]

Main outputs
------------
    results["per_nv_optimal_step_vals"]
        Per-NV target readout power, if the sweep is readout amplitude.

    results["slm_mean_norm_intensity_weight"]
        Main SLM intensity weights. Mean over selected NVs is 1.

    results["slm_amplitude_weight"]
        Use this only if your SLM hologram code expects field amplitude weights.

    results["slm_effective_aom_voltage"]
        Global AOM voltage corresponding to the selected-NV mean target power.

    results["slm_effective_step_value"]
        If your OPX sweep variable is a scale factor multiplying the base
        yellow_charge_readout waveform sample, this is the scale factor to set.

Created for CPU/joblib analysis, July 2026.
"""

from __future__ import annotations

# Keep every worker single-threaded. This avoids CPU oversubscription when using joblib.
import os
import copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

# Compatibility patch for old labrad code with newer NumPy.
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
# Configuration
# =============================================================================


@dataclass
class OptimizationConfig:
    """
    Criteria and score weights for selecting the best readout step.

    selection_mode
    --------------
    "threshold_then_score"
        Choose the lowest allowed step satisfying every threshold. If no step
        passes, choose the maximum combined score.

    "lowest_passing"
        Choose the lowest allowed step satisfying every threshold. Return no
        optimum if no step passes.

    "max_score"
        Ignore the pass/fail thresholds when choosing the optimum and select
        the maximum combined score directly.

    skip_first_steps
    ----------------
    Exclude the first N sweep points from both threshold and score selection.
    This reproduces the useful behavior of the older optimization script.
    """

    min_readout1_fidelity: float = 0.88
    min_readout2_fidelity: float = 0.88
    min_ref_same_state_survival: float = 0.96
    min_ref_nvm_survival: float = 0.97
    min_ref_nv0_survival: Optional[float] = None

    # Score = w_fidelity*fidelity + w_survival*survival + w_fit*fit_quality
    #         + w_low_power*low_power
    # Must have exactly four entries.
    score_weights: Tuple[float, float, float, float] = (0.35, 0.40, 0.15, 0.10)

    selection_mode: str = "threshold_then_score"
    skip_first_steps: int = 0
    prob_dist_name: str = "COMPOUND_POISSON"


@dataclass
class SlmWeightConfig:
    """SLM weight extraction options."""

    slm_efficiency: float = 1.0
    invalid_fill: float = 0.0
    clip_min: Optional[float] = 0.25
    clip_max: Optional[float] = 1.75
    renormalize_after_clip: bool = True


# =============================================================================
# Basic helpers
# =============================================================================


def make_json_safe(obj: Any) -> Any:
    """Recursively convert NumPy-heavy objects into JSON-saveable objects."""

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return make_json_safe(obj.tolist())
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def base_file_stem(raw_data: Dict[str, Any]) -> str:
    stem = (
        raw_data.get("file_stem")
        or raw_data.get("file_name")
        or raw_data.get("timestamp")
        or "raw_data"
    )
    if isinstance(stem, (list, tuple)):
        stem = "_".join(map(str, stem))
    return str(stem).replace(" ", "_")


def finite_flatten(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def nanmedian_axis0(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmedian(np.asarray(x, dtype=float), axis=0)


def nanmean_axis0(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(np.asarray(x, dtype=float), axis=0)


def norm01(x: np.ndarray, constant_value: float = 0.0) -> np.ndarray:
    """Normalize finite values to [0, 1]. NaNs remain NaN."""

    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out

    xmin = np.nanmin(x[finite])
    xmax = np.nanmax(x[finite])
    if np.isclose(xmax, xmin):
        out[finite] = constant_value
    else:
        out[finite] = (x[finite] - xmin) / (xmax - xmin)
    return out


def validate_counts_shape(counts: np.ndarray, num_nvs: int, num_steps: int) -> None:
    if counts.ndim != 5:
        raise ValueError(
            "Expected counts shape counts[exp, nv, run, step, rep]. "
            f"Got shape {counts.shape}."
        )
    if counts.shape[0] < 4:
        raise ValueError(f"Expected at least 4 experiment branches. Got {counts.shape[0]}.")
    if counts.shape[1] != num_nvs:
        raise ValueError(
            f"len(nv_list)={num_nvs}, but counts.shape[1]={counts.shape[1]}."
        )
    if counts.shape[3] != num_steps:
        raise ValueError(
            f"raw_data['num_steps']={num_steps}, but counts.shape[3]={counts.shape[3]}."
        )


# =============================================================================
# Axis / calibration helpers
# =============================================================================


def get_readout_axis(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert swept step values into an analysis x-axis.

    For readout amplitude sweeps, this uses the empirical yellow-AOM calibration:
        P(uW) = a * V**b + c

    where V = step_value * yellow_charge_readout waveform sample.
    """

    min_step_val = float(raw_data["min_step_val"])
    max_step_val = float(raw_data["max_step_val"])
    num_steps = int(raw_data["num_steps"])
    step_vals_raw = np.linspace(min_step_val, max_step_val, num_steps)

    optimize_pol_or_readout = bool(raw_data["optimize_pol_or_readout"])
    optimize_duration_or_amp = bool(raw_data["optimize_duration_or_amp"])

    # Yellow AOM voltage-to-power calibration.
    a, b, c = 1.5133e04, 2.6976, -38.63

    if optimize_pol_or_readout:
        if optimize_duration_or_amp:
            x_label = "Polarization duration (ns)"
        else:
            x_label = "Polarization amplitude scale"

        return {
            "step_vals_raw": step_vals_raw,
            "step_vals": step_vals_raw,
            "x_label": x_label,
            "is_readout_power_sweep": False,
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
            "is_readout_power_sweep": False,
            "power_fit_a": a,
            "power_fit_b": b,
            "power_fit_c": c,
            "yellow_charge_readout_amp": np.nan,
        }

    yellow_charge_readout_amp = raw_data["opx_config"]["waveforms"][
        "yellow_charge_readout"
    ]["sample"]

    aom_voltage = step_vals_raw * float(yellow_charge_readout_amp)
    readout_power_uW = a * (aom_voltage**b) + c

    return {
        "step_vals_raw": step_vals_raw,
        "step_vals": readout_power_uW,
        "x_label": "Readout power per SLM spot (uW)",
        "is_readout_power_sweep": True,
        "power_fit_a": a,
        "power_fit_b": b,
        "power_fit_c": c,
        "yellow_charge_readout_amp": float(yellow_charge_readout_amp),
    }


def aom_voltage_from_power_uW(power_uW: float, axis_info: Dict[str, Any]) -> float:
    """Invert P(uW)=a*V**b+c."""

    a = float(axis_info["power_fit_a"])
    b = float(axis_info["power_fit_b"])
    c = float(axis_info["power_fit_c"])

    if not np.isfinite(power_uW) or power_uW <= c:
        return np.nan
    return float(((power_uW - c) / a) ** (1.0 / b))


def opx_step_from_power_uW(power_uW: float, axis_info: Dict[str, Any]) -> float:
    """
    Convert desired power to the OPX sweep scale factor if this was a readout
    amplitude sweep.
    """

    if not bool(axis_info.get("is_readout_power_sweep", False)):
        return np.nan
    base_amp = float(axis_info.get("yellow_charge_readout_amp", np.nan))
    if not np.isfinite(base_amp) or base_amp == 0:
        return np.nan
    return float(aom_voltage_from_power_uW(power_uW, axis_info) / base_amp)


# =============================================================================
# Fitting one NV / one step
# =============================================================================


def fit_bimodal_threshold(
    counts_data: np.ndarray,
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
) -> Dict[str, Any]:
    """Fit one bimodal histogram and return threshold/fidelity/fit params."""

    try:
        counts_data = finite_flatten(counts_data)
        if counts_data.size < 10:
            raise ValueError(f"Not enough counts for fit: {counts_data.size}")

        popt, pcov, red_chi_sq = fit_bimodal_histogram(
            counts_data,
            prob_dist,
            no_plot=True,
        )

        if popt is None:
            raise RuntimeError("fit_bimodal_histogram returned popt=None")

        threshold, readout_fidelity = determine_threshold(
            popt,
            prob_dist,
            dark_mode_weight=0.5,
            ret_fidelity=True,
        )

        return {
            "threshold": float(threshold),
            "readout_fidelity": float(readout_fidelity),
            # popt[0] is the fitted dark/NV0 weight in the mixed histogram.
            # 1-popt[0] is a useful empirical bright/NV- population diagnostic.
            "prep_fidelity": float(1.0 - popt[0]),
            "goodness_of_fit": float(red_chi_sq),
            "fit_success": True,
            "fit_params": np.asarray(popt, dtype=float),
            "error": None,
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


def conditional_mean(masked_values: np.ndarray) -> float:
    masked_values = np.asarray(masked_values, dtype=bool)
    if masked_values.size == 0:
        return np.nan
    return float(np.mean(masked_values))


def process_one_nv_step(
    nv_ind: int,
    step_ind: int,
    counts: np.ndarray,
    prob_dist_name: str,
) -> Dict[str, Any]:
    """
    Process one NV at one sweep step.

    Fitting:
        R1 fit uses ionized R1 + reference R1.
        R2 fit uses ionized R2 + reference R2.

    Survival:
        Reference R1 and R2 are both classified using the R1 threshold.
        This is intentional: it asks whether readout 1 changes the charge state
        before readout 2, without moving the classification boundary.
    """

    prob_dist = ProbDist[prob_dist_name]

    ion_r1 = finite_flatten(counts[0, nv_ind, :, step_ind, :])
    ion_r2 = finite_flatten(counts[1, nv_ind, :, step_ind, :])
    ref_r1 = finite_flatten(counts[2, nv_ind, :, step_ind, :])
    ref_r2 = finite_flatten(counts[3, nv_ind, :, step_ind, :])

    # fit_r1 = fit_bimodal_threshold(np.concatenate([ion_r1, ref_r1]), prob_dist)
    # fit_r2 = fit_bimodal_threshold(np.concatenate([ion_r2, ref_r2]), prob_dist)

    fit_r1 = fit_bimodal_threshold(ref_r1, prob_dist)
    fit_r2 = fit_bimodal_threshold(ref_r2, prob_dist)

    threshold_r1 = fit_r1["threshold"]
    threshold_r2 = fit_r2["threshold"]

    out: Dict[str, Any] = {
        "nv_ind": int(nv_ind),
        "step_ind": int(step_ind),
        "threshold_r1": threshold_r1,
        "threshold_r2": threshold_r2,
        # Backward-compatible name: use R1 threshold as the main threshold.
        "threshold": threshold_r1,
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
        "fit1_error": fit_r1["error"],
        "fit2_error": fit_r2["error"],
        "ref_same_state_survival": np.nan,
        "ref_nvm_survival": np.nan,
        "ref_nv0_survival": np.nan,
        "ref_nvm_to_nv0_prob": np.nan,
        "ref_nv0_to_nvm_prob": np.nan,
        "ion_same_state_survival": np.nan,
        "ion_nv0_survival": np.nan,
        "ion_nv0_to_nvm_prob": np.nan,
        "num_ref_trials": int(min(ref_r1.size, ref_r2.size)),
        "num_ref_r1_nvm": 0,
        "num_ref_r1_nv0": 0,
        "num_ion_trials": int(min(ion_r1.size, ion_r2.size)),
        "num_ion_r1_nv0": 0,
        "mean_ion_r1": float(np.nanmean(ion_r1)) if ion_r1.size else np.nan,
        "mean_ion_r2": float(np.nanmean(ion_r2)) if ion_r2.size else np.nan,
        "mean_ref_r1": float(np.nanmean(ref_r1)) if ref_r1.size else np.nan,
        "mean_ref_r2": float(np.nanmean(ref_r2)) if ref_r2.size else np.nan,
    }

    if not np.isfinite(threshold_r1):
        return out

    # Reference branch: main non-destructive readout test.
    n_ref = min(ref_r1.size, ref_r2.size)
    if n_ref > 0:
        ref_s1_is_nvm = ref_r1[:n_ref] > threshold_r1
        ref_s2_is_nvm = ref_r2[:n_ref] > threshold_r1

        out["ref_same_state_survival"] = float(np.mean(ref_s1_is_nvm == ref_s2_is_nvm))
        out["num_ref_r1_nvm"] = int(np.sum(ref_s1_is_nvm))
        out["num_ref_r1_nv0"] = int(np.sum(~ref_s1_is_nvm))

        if np.any(ref_s1_is_nvm):
            ref_nvm_survival = float(np.mean(ref_s2_is_nvm[ref_s1_is_nvm]))
            out["ref_nvm_survival"] = ref_nvm_survival
            out["ref_nvm_to_nv0_prob"] = 1.0 - ref_nvm_survival

        if np.any(~ref_s1_is_nvm):
            ref_nv0_survival = float(np.mean(~ref_s2_is_nvm[~ref_s1_is_nvm]))
            out["ref_nv0_survival"] = ref_nv0_survival
            out["ref_nv0_to_nvm_prob"] = 1.0 - ref_nv0_survival

    # Ionized branch: useful diagnostic, not the main optimization target.
    n_ion = min(ion_r1.size, ion_r2.size)
    if n_ion > 0:
        ion_s1_is_nvm = ion_r1[:n_ion] > threshold_r1
        ion_s2_is_nvm = ion_r2[:n_ion] > threshold_r1

        out["ion_same_state_survival"] = float(np.mean(ion_s1_is_nvm == ion_s2_is_nvm))
        out["num_ion_r1_nv0"] = int(np.sum(~ion_s1_is_nvm))

        if np.any(~ion_s1_is_nvm):
            ion_nv0_survival = float(np.mean(~ion_s2_is_nvm[~ion_s1_is_nvm]))
            out["ion_nv0_survival"] = ion_nv0_survival
            out["ion_nv0_to_nvm_prob"] = 1.0 - ion_nv0_survival

    return out


# =============================================================================
# Arrays and metrics
# =============================================================================


FLOAT_METRICS = [
    "threshold",
    "threshold_r1",
    "threshold_r2",
    "readout1_fidelity",
    "readout2_fidelity",
    "prep1_fidelity",
    "prep2_fidelity",
    "goodness1_of_fit",
    "goodness2_of_fit",
    "ref_same_state_survival",
    "ref_nvm_survival",
    "ref_nv0_survival",
    "ref_nvm_to_nv0_prob",
    "ref_nv0_to_nvm_prob",
    "ion_same_state_survival",
    "ion_nv0_survival",
    "ion_nv0_to_nvm_prob",
    "mean_ion_r1",
    "mean_ion_r2",
    "mean_ref_r1",
    "mean_ref_r2",
]

INT_METRICS = [
    "num_ref_trials",
    "num_ref_r1_nvm",
    "num_ref_r1_nv0",
    "num_ion_trials",
    "num_ion_r1_nv0",
]

BOOL_METRICS = ["fit1_success", "fit2_success"]

OBJECT_METRICS = ["fit1_params", "fit2_params", "fit1_error", "fit2_error"]


def allocate_metric_arrays(num_nvs: int, num_steps: int) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for key in FLOAT_METRICS:
        arrays[f"{key}_arr"] = np.full((num_nvs, num_steps), np.nan, dtype=float)
    for key in INT_METRICS:
        arrays[f"{key}_arr"] = np.zeros((num_nvs, num_steps), dtype=int)
    for key in BOOL_METRICS:
        arrays[f"{key}_arr"] = np.zeros((num_nvs, num_steps), dtype=bool)
    for key in OBJECT_METRICS:
        arrays[f"{key}_arr"] = np.full((num_nvs, num_steps), None, dtype=object)
    return arrays


def fill_metric_arrays(
    flat_results: Sequence[Dict[str, Any]],
    num_nvs: int,
    num_steps: int,
) -> Dict[str, np.ndarray]:
    arrays = allocate_metric_arrays(num_nvs, num_steps)

    for res in flat_results:
        nv_ind = int(res["nv_ind"])
        step_ind = int(res["step_ind"])

        for key in FLOAT_METRICS + INT_METRICS + BOOL_METRICS + OBJECT_METRICS:
            arr_key = f"{key}_arr"
            if key in res:
                arrays[arr_key][nv_ind, step_ind] = res[key]

    return arrays


def object_array_to_nested_lists(arr: np.ndarray) -> List[List[Any]]:
    out: List[List[Any]] = []
    for row in arr:
        row_out = []
        for val in row:
            if val is None:
                row_out.append(None)
            elif isinstance(val, np.ndarray):
                row_out.append(np.asarray(val, dtype=float).ravel().tolist())
            else:
                row_out.append(make_json_safe(val))
        out.append(row_out)
    return out


def metric_at(arr: np.ndarray, nv_ind: int, step_ind: Optional[int]) -> float:
    if step_ind is None or int(step_ind) < 0:
        return np.nan
    return float(np.asarray(arr, dtype=float)[nv_ind, int(step_ind)])


# =============================================================================
# Optimization logic
# =============================================================================


def compute_score(
    readout1_fidelity: np.ndarray,
    readout2_fidelity: np.ndarray,
    ref_same_state_survival: np.ndarray,
    ref_nvm_survival: np.ndarray,
    ref_nv0_survival: np.ndarray,
    goodness1_of_fit: np.ndarray,
    goodness2_of_fit: np.ndarray,
    step_vals: np.ndarray,
    score_weights: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Non-destructive two-readout score.

    Components:
        fidelity: min(R1 fidelity, R2 fidelity)
        survival: weighted reference survival between readout 1 and readout 2
        fit_quality: lower reduced chi-square is better
        low_power: weak preference for lower readout power/duration
    """

    if len(score_weights) != 4:
        raise ValueError(
            "score_weights must have exactly four values: "
            "(w_fidelity, w_survival, w_fit_quality, w_low_power). "
            f"Got {score_weights}."
        )

    w_fid, w_survival, w_fit, w_low = [float(v) for v in score_weights]
    weight_sum = w_fid + w_survival + w_fit + w_low
    if weight_sum <= 0:
        raise ValueError("score_weights must sum to a positive value.")
    w_fid, w_survival, w_fit, w_low = [v / weight_sum for v in (w_fid, w_survival, w_fit, w_low)]

    r1 = np.asarray(readout1_fidelity, dtype=float)
    r2 = np.asarray(readout2_fidelity, dtype=float)
    same = np.asarray(ref_same_state_survival, dtype=float)
    nvm = np.asarray(ref_nvm_survival, dtype=float)
    nv0 = np.asarray(ref_nv0_survival, dtype=float)
    gof1 = np.asarray(goodness1_of_fit, dtype=float)
    gof2 = np.asarray(goodness2_of_fit, dtype=float)
    step_vals = np.asarray(step_vals, dtype=float)

    two_readout_fidelity = np.minimum(r1, r2)

    # Main physics target: preserve NV- in the reference branch.
    # Same-state survival is included to catch charge scrambling both ways.
    survival = 0.50 * nvm + 0.35 * same
    if np.any(np.isfinite(nv0)):
        survival = survival + 0.15 * nv0
    else:
        survival = survival / 0.85

    fit_quality = 0.5 * (1.0 - norm01(gof1, constant_value=0.0)) + 0.5 * (
        1.0 - norm01(gof2, constant_value=0.0)
    )

    # Lower power/duration gets a small preference.
    low_power = 1.0 - norm01(step_vals, constant_value=0.0)

    score = (
        w_fid * norm01(two_readout_fidelity, constant_value=1.0)
        + w_survival * norm01(survival, constant_value=1.0)
        + w_fit * fit_quality
        + w_low * low_power
    )

    bad = (
        ~np.isfinite(r1)
        | ~np.isfinite(r2)
        | ~np.isfinite(same)
        | ~np.isfinite(nvm)
        | ~np.isfinite(gof1)
        | ~np.isfinite(gof2)
    )
    return np.where(bad, np.nan, score)


def _allowed_step_mask(num_steps: int, config: OptimizationConfig) -> np.ndarray:
    """Return the steps permitted by ``skip_first_steps``."""

    skip = max(0, int(config.skip_first_steps))
    allowed = np.ones(int(num_steps), dtype=bool)
    allowed[: min(skip, int(num_steps))] = False
    return allowed


def choose_lowest_step_passing_criteria(
    readout1_fidelity: np.ndarray,
    readout2_fidelity: np.ndarray,
    ref_same_state_survival: np.ndarray,
    ref_nvm_survival: np.ndarray,
    ref_nv0_survival: np.ndarray,
    config: OptimizationConfig,
) -> Tuple[Optional[int], str]:
    r1 = np.asarray(readout1_fidelity, dtype=float)
    r2 = np.asarray(readout2_fidelity, dtype=float)
    same = np.asarray(ref_same_state_survival, dtype=float)
    nvm = np.asarray(ref_nvm_survival, dtype=float)
    nv0 = np.asarray(ref_nv0_survival, dtype=float)

    if not (r1.shape == r2.shape == same.shape == nvm.shape == nv0.shape):
        raise ValueError("All optimization metric arrays must have the same shape.")

    good = (
        _allowed_step_mask(r1.size, config)
        & np.isfinite(r1)
        & np.isfinite(r2)
        & np.isfinite(same)
        & np.isfinite(nvm)
        & (r1 >= config.min_readout1_fidelity)
        & (r2 >= config.min_readout2_fidelity)
        & (same >= config.min_ref_same_state_survival)
        & (nvm >= config.min_ref_nvm_survival)
    )

    if config.min_ref_nv0_survival is not None:
        good = (
            good
            & np.isfinite(nv0)
            & (nv0 >= float(config.min_ref_nv0_survival))
        )

    if np.any(good):
        return int(np.flatnonzero(good)[0]), "lowest step satisfying thresholds"

    return None, "thresholds not all satisfied"


def choose_optimal_step(
    step_vals: np.ndarray,
    readout1_fidelity: np.ndarray,
    readout2_fidelity: np.ndarray,
    ref_same_state_survival: np.ndarray,
    ref_nvm_survival: np.ndarray,
    ref_nv0_survival: np.ndarray,
    goodness1_of_fit: np.ndarray,
    goodness2_of_fit: np.ndarray,
    config: OptimizationConfig,
) -> Tuple[Optional[int], str, np.ndarray]:
    """
    Choose one sweep step using the configured threshold/score strategy.
    """

    mode = str(config.selection_mode).strip().lower()
    valid_modes = {"threshold_then_score", "lowest_passing", "max_score"}
    if mode not in valid_modes:
        raise ValueError(
            f"selection_mode must be one of {sorted(valid_modes)}; got {mode!r}."
        )

    score = compute_score(
        readout1_fidelity,
        readout2_fidelity,
        ref_same_state_survival,
        ref_nvm_survival,
        ref_nv0_survival,
        goodness1_of_fit,
        goodness2_of_fit,
        step_vals,
        score_weights=config.score_weights,
    )

    # Excluded steps are never allowed to win the score.
    score = np.asarray(score, dtype=float)
    score[~_allowed_step_mask(score.size, config)] = np.nan

    passing_ind, passing_reason = choose_lowest_step_passing_criteria(
        readout1_fidelity,
        readout2_fidelity,
        ref_same_state_survival,
        ref_nvm_survival,
        ref_nv0_survival,
        config,
    )

    if mode in {"threshold_then_score", "lowest_passing"} and passing_ind is not None:
        return passing_ind, passing_reason, score

    if mode == "lowest_passing":
        return None, passing_reason, score

    if not np.any(np.isfinite(score)):
        return None, "no finite score", score

    best_ind = int(np.nanargmax(score))
    if mode == "max_score":
        return best_ind, "maximum combined score", score

    return best_ind, "fallback maximum combined score", score

def compute_per_nv_optima(
    arrays: Dict[str, np.ndarray],
    step_vals: np.ndarray,
    axis_info: Dict[str, Any],
    config: OptimizationConfig,
) -> Dict[str, Any]:
    num_nvs, num_steps = arrays["readout1_fidelity_arr"].shape
    score_arr = np.full((num_nvs, num_steps), np.nan, dtype=float)
    optimal_step_inds = np.full(num_nvs, -1, dtype=int)
    optimal_step_vals = np.full(num_nvs, np.nan, dtype=float)
    optimal_step_raw_vals = np.full(num_nvs, np.nan, dtype=float)
    optimal_values: List[Dict[str, Any]] = []

    step_vals_raw = np.asarray(axis_info["step_vals_raw"], dtype=float)

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
            config,
        )
        score_arr[nv_ind, :] = score

        if step_ind is None:
            optimal_values.append(
                {
                    "nv_ind": int(nv_ind),
                    "optimal_step_ind": None,
                    "optimal_step_val": np.nan,
                    "optimal_step_raw_val": np.nan,
                    "reason": reason,
                    "score": np.nan,
                    "readout1_fidelity": np.nan,
                    "readout2_fidelity": np.nan,
                    "ref_same_state_survival": np.nan,
                    "ref_nvm_survival": np.nan,
                    "ref_nv0_survival": np.nan,
                    "aom_voltage": np.nan,
                    "opx_step_value": np.nan,
                }
            )
            continue

        step_ind = int(step_ind)
        step_val = float(step_vals[step_ind])
        step_raw = float(step_vals_raw[step_ind])
        optimal_step_inds[nv_ind] = step_ind
        optimal_step_vals[nv_ind] = step_val
        optimal_step_raw_vals[nv_ind] = step_raw

        aom_voltage = (
            aom_voltage_from_power_uW(step_val, axis_info)
            if bool(axis_info.get("is_readout_power_sweep", False))
            else np.nan
        )
        opx_step_value = (
            opx_step_from_power_uW(step_val, axis_info)
            if bool(axis_info.get("is_readout_power_sweep", False))
            else np.nan
        )

        optimal_values.append(
            {
                "nv_ind": int(nv_ind),
                "optimal_step_ind": step_ind,
                "optimal_step_val": step_val,
                "optimal_step_raw_val": step_raw,
                "reason": reason,
                "score": metric_at(score_arr, nv_ind, step_ind),
                "readout1_fidelity": metric_at(arrays["readout1_fidelity_arr"], nv_ind, step_ind),
                "readout2_fidelity": metric_at(arrays["readout2_fidelity_arr"], nv_ind, step_ind),
                "ref_same_state_survival": metric_at(arrays["ref_same_state_survival_arr"], nv_ind, step_ind),
                "ref_nvm_survival": metric_at(arrays["ref_nvm_survival_arr"], nv_ind, step_ind),
                "ref_nv0_survival": metric_at(arrays["ref_nv0_survival_arr"], nv_ind, step_ind),
                "aom_voltage": aom_voltage,
                "opx_step_value": opx_step_value,
            }
        )

    return {
        "optimal_values": optimal_values,
        "optimal_step_inds": optimal_step_inds,
        "optimal_step_vals": optimal_step_vals,
        "optimal_step_raw_vals": optimal_step_raw_vals,
        "score_arr": score_arr,
    }



# =============================================================================
# Fast re-optimization from saved processed data
# =============================================================================


SLM_RESULT_KEYS = (
    "slm_config",
    "slm_selected_inds",
    "slm_num_selected",
    "slm_target_power_uW",
    "slm_total_target_power_uW",
    "slm_total_target_power_mW",
    "slm_mean_target_power_uW",
    "slm_median_target_power_uW",
    "slm_effective_aom_power_uW",
    "slm_effective_aom_voltage",
    "slm_effective_step_value",
    "slm_mean_norm_intensity_weight",
    "slm_mean_norm_intensity_weight_clipped",
    "slm_amplitude_weight",
    "slm_power_fraction",
    "optimal_weights_aligned",
    "optimal_weights_clipped_aligned",
    "aom_voltage_for_slm",
)


def _axis_info_from_processed(results: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct the axis/calibration dictionary from processed results."""

    yellow_amp = results.get("yellow_charge_readout_amp", np.nan)
    yellow_amp = np.nan if yellow_amp is None else float(yellow_amp)

    return {
        "step_vals_raw": np.asarray(results["step_vals_raw"], dtype=float),
        "step_vals": np.asarray(results["step_vals"], dtype=float),
        "x_label": str(results["x_label"]),
        "is_readout_power_sweep": bool(
            results.get("is_readout_power_sweep", False)
        ),
        "power_fit_a": float(results["power_fit_a"]),
        "power_fit_b": float(results["power_fit_b"]),
        "power_fit_c": float(results["power_fit_c"]),
        "yellow_charge_readout_amp": yellow_amp,
    }


def _metric_arrays_from_processed(
    results: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Load the metric arrays needed to recompute optima."""

    required = (
        "readout1_fidelity_arr",
        "readout2_fidelity_arr",
        "ref_same_state_survival_arr",
        "ref_nvm_survival_arr",
        "ref_nv0_survival_arr",
        "goodness1_of_fit_arr",
        "goodness2_of_fit_arr",
    )
    missing = [key for key in required if key not in results]
    if missing:
        raise KeyError(
            "Processed data are missing arrays required for re-optimization: "
            + ", ".join(missing)
        )

    arrays: Dict[str, np.ndarray] = {}
    for key in FLOAT_METRICS:
        arr_key = f"{key}_arr"
        if arr_key in results:
            arrays[arr_key] = np.asarray(results[arr_key], dtype=float)

    for key in INT_METRICS:
        arr_key = f"{key}_arr"
        if arr_key in results:
            arrays[arr_key] = np.asarray(results[arr_key], dtype=int)

    for key in BOOL_METRICS:
        arr_key = f"{key}_arr"
        if arr_key in results:
            arrays[arr_key] = np.asarray(results[arr_key], dtype=bool)

    # compute_per_nv_optima only requires the floating-point arrays listed
    # above, but retaining the others makes this helper useful elsewhere.
    return arrays


def _recompute_population_medians(
    arrays: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Recompute population medians from the saved per-NV metric arrays."""

    mapping = {
        "readout1_fidelity": "readout1_fidelity_arr",
        "readout2_fidelity": "readout2_fidelity_arr",
        "prep1_fidelity": "prep1_fidelity_arr",
        "prep2_fidelity": "prep2_fidelity_arr",
        "goodness1_of_fit": "goodness1_of_fit_arr",
        "goodness2_of_fit": "goodness2_of_fit_arr",
        "ref_same_state_survival": "ref_same_state_survival_arr",
        "ref_nvm_survival": "ref_nvm_survival_arr",
        "ref_nv0_survival": "ref_nv0_survival_arr",
        "ref_nvm_to_nv0_prob": "ref_nvm_to_nv0_prob_arr",
        "mean_ion_r1": "mean_ion_r1_arr",
        "mean_ion_r2": "mean_ion_r2_arr",
        "mean_ref_r1": "mean_ref_r1_arr",
        "mean_ref_r2": "mean_ref_r2_arr",
    }

    medians: Dict[str, np.ndarray] = {}
    for output_key, array_key in mapping.items():
        if array_key in arrays:
            medians[output_key] = nanmedian_axis0(arrays[array_key])
    return medians


def resolve_manual_step_index(
    results: Dict[str, Any],
    step_ind: Optional[int] = None,
    step_val: Optional[float] = None,
) -> int:
    """
    Resolve a manual choice to a valid step index.

    Supply exactly one of ``step_ind`` or ``step_val``. A physical step value
    is mapped to the closest sampled sweep point.
    """

    if (step_ind is None) == (step_val is None):
        raise ValueError("Supply exactly one of step_ind or step_val.")

    step_vals = np.asarray(results["step_vals"], dtype=float)
    if step_ind is not None:
        resolved = int(step_ind)
        if not 0 <= resolved < step_vals.size:
            raise IndexError(
                f"step_ind={resolved} is outside [0, {step_vals.size - 1}]."
            )
        return resolved

    value = float(step_val)
    finite = np.isfinite(step_vals)
    if not np.any(finite):
        raise ValueError("No finite processed step values are available.")
    finite_inds = np.flatnonzero(finite)
    return int(finite_inds[np.argmin(np.abs(step_vals[finite] - value))])


def _optimal_value_record(
    results: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    score_arr: np.ndarray,
    nv_ind: int,
    step_ind: int,
    reason: str,
) -> Dict[str, Any]:
    """Build one per-NV optimum record at an explicitly selected step."""

    axis_info = _axis_info_from_processed(results)
    step_vals = np.asarray(results["step_vals"], dtype=float)
    step_vals_raw = np.asarray(results["step_vals_raw"], dtype=float)
    step_val = float(step_vals[step_ind])

    aom_voltage = (
        aom_voltage_from_power_uW(step_val, axis_info)
        if bool(axis_info.get("is_readout_power_sweep", False))
        else np.nan
    )
    opx_step_value = (
        opx_step_from_power_uW(step_val, axis_info)
        if bool(axis_info.get("is_readout_power_sweep", False))
        else np.nan
    )

    return {
        "nv_ind": int(nv_ind),
        "optimal_step_ind": int(step_ind),
        "optimal_step_val": step_val,
        "optimal_step_raw_val": float(step_vals_raw[step_ind]),
        "reason": str(reason),
        "score": metric_at(score_arr, nv_ind, step_ind),
        "readout1_fidelity": metric_at(
            arrays["readout1_fidelity_arr"], nv_ind, step_ind
        ),
        "readout2_fidelity": metric_at(
            arrays["readout2_fidelity_arr"], nv_ind, step_ind
        ),
        "ref_same_state_survival": metric_at(
            arrays["ref_same_state_survival_arr"], nv_ind, step_ind
        ),
        "ref_nvm_survival": metric_at(
            arrays["ref_nvm_survival_arr"], nv_ind, step_ind
        ),
        "ref_nv0_survival": metric_at(
            arrays["ref_nv0_survival_arr"], nv_ind, step_ind
        ),
        "aom_voltage": aom_voltage,
        "opx_step_value": opx_step_value,
    }


def apply_manual_step_overrides(
    results: Dict[str, Any],
    manual_step_inds: Optional[Dict[int, int]] = None,
    manual_step_vals: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Override automatic per-NV optima before calculating SLM weights.

    Examples
    --------
    manual_step_inds = {8: 12, 303: 15}
        Use exact sweep indices.

    manual_step_vals = {8: 7.5, 303: 9.0}
        Use the nearest sampled physical step values in ``results["step_vals"]``.
    """

    manual_step_inds = {} if manual_step_inds is None else dict(manual_step_inds)
    manual_step_vals = {} if manual_step_vals is None else dict(manual_step_vals)

    overlap = set(manual_step_inds) & set(manual_step_vals)
    if overlap:
        raise ValueError(
            "The same NV cannot be specified in both manual dictionaries: "
            f"{sorted(overlap)}"
        )

    if not manual_step_inds and not manual_step_vals:
        results["manual_step_overrides"] = {}
        return results

    arrays = _metric_arrays_from_processed(results)
    score_arr = np.asarray(results["per_nv_score_arr"], dtype=float)
    step_inds = np.asarray(results["per_nv_optimal_step_inds"], dtype=int)
    step_vals = np.asarray(results["per_nv_optimal_step_vals"], dtype=float)
    step_raw_vals = np.asarray(
        results["per_nv_optimal_step_raw_vals"], dtype=float
    )
    records = list(results["per_nv_optimal_values"])
    num_nvs = int(results["num_nvs"])

    applied: Dict[str, Dict[str, Any]] = {}

    choices: Dict[int, Tuple[Optional[int], Optional[float]]] = {}
    choices.update({int(nv): (int(ind), None) for nv, ind in manual_step_inds.items()})
    choices.update({int(nv): (None, float(val)) for nv, val in manual_step_vals.items()})

    for nv_ind, (step_ind_input, step_val_input) in choices.items():
        if not 0 <= nv_ind < num_nvs:
            raise IndexError(f"Manual NV index {nv_ind} is outside [0, {num_nvs - 1}].")

        resolved = resolve_manual_step_index(
            results,
            step_ind=step_ind_input,
            step_val=step_val_input,
        )
        record = _optimal_value_record(
            results,
            arrays,
            score_arr,
            nv_ind=nv_ind,
            step_ind=resolved,
            reason="manual override",
        )

        records[nv_ind] = make_json_safe(record)
        step_inds[nv_ind] = resolved
        step_vals[nv_ind] = record["optimal_step_val"]
        step_raw_vals[nv_ind] = record["optimal_step_raw_val"]

        applied[str(nv_ind)] = {
            "requested_step_ind": step_ind_input,
            "requested_step_val": step_val_input,
            "resolved_step_ind": resolved,
            "resolved_step_val": record["optimal_step_val"],
        }

    results["per_nv_optimal_values"] = records
    results["per_nv_optimal_step_inds"] = step_inds.tolist()
    results["per_nv_optimal_step_vals"] = step_vals.tolist()
    results["per_nv_optimal_step_raw_vals"] = step_raw_vals.tolist()
    results["manual_step_overrides"] = applied
    return results


def reoptimize_processed_results(
    processed_results: Dict[str, Any],
    opt_config: OptimizationConfig,
    slm_config: Optional[SlmWeightConfig] = SlmWeightConfig(),
    slm_selected_inds: Optional[Sequence[int]] = None,
    manual_step_inds: Optional[Dict[int, int]] = None,
    manual_step_vals: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Recompute population/per-NV optima and SLM weights without refitting.

    This function only uses the metric arrays already saved in the processed
    file. It is therefore fast enough for interactively changing thresholds,
    score weights, selection mode, skipped sweep points, clipping, and manual
    per-NV overrides.
    """

    results = copy.deepcopy(processed_results)
    arrays = _metric_arrays_from_processed(results)
    axis_info = _axis_info_from_processed(results)
    step_vals = np.asarray(results["step_vals"], dtype=float)
    medians = _recompute_population_medians(arrays)

    needed_medians = (
        "readout1_fidelity",
        "readout2_fidelity",
        "ref_same_state_survival",
        "ref_nvm_survival",
        "ref_nv0_survival",
        "goodness1_of_fit",
        "goodness2_of_fit",
    )
    missing = [key for key in needed_medians if key not in medians]
    if missing:
        raise KeyError(
            "Cannot recompute the population optimum; missing median arrays: "
            + ", ".join(missing)
        )

    population_step_ind, population_reason, population_score = choose_optimal_step(
        step_vals,
        medians["readout1_fidelity"],
        medians["readout2_fidelity"],
        medians["ref_same_state_survival"],
        medians["ref_nvm_survival"],
        medians["ref_nv0_survival"],
        medians["goodness1_of_fit"],
        medians["goodness2_of_fit"],
        opt_config,
    )

    per_nv = compute_per_nv_optima(arrays, step_vals, axis_info, opt_config)

    if population_step_ind is None:
        population_step_val = np.nan
        population_step_raw_val = np.nan
        population_aom_voltage = np.nan
        population_opx_step = np.nan
    else:
        population_step_ind = int(population_step_ind)
        population_step_val = float(step_vals[population_step_ind])
        population_step_raw_val = float(
            np.asarray(results["step_vals_raw"], dtype=float)[population_step_ind]
        )
        population_aom_voltage = (
            aom_voltage_from_power_uW(population_step_val, axis_info)
            if bool(axis_info.get("is_readout_power_sweep", False))
            else np.nan
        )
        population_opx_step = (
            opx_step_from_power_uW(population_step_val, axis_info)
            if bool(axis_info.get("is_readout_power_sweep", False))
            else np.nan
        )

    results.update(
        {
            "analysis_type": "reoptimized_repeated_readout_slm",
            "optimization_config": asdict(opt_config),
            "population_optimal_step_ind": population_step_ind,
            "population_optimal_step_val": population_step_val,
            "population_optimal_step_raw_val": population_step_raw_val,
            "population_aom_voltage": population_aom_voltage,
            "population_opx_step_value": population_opx_step,
            "population_optimal_reason": population_reason,
            "population_score": population_score.tolist(),
            "per_nv_optimal_values": make_json_safe(per_nv["optimal_values"]),
            "per_nv_optimal_step_inds": per_nv["optimal_step_inds"].tolist(),
            "per_nv_optimal_step_vals": per_nv["optimal_step_vals"].tolist(),
            "per_nv_optimal_step_raw_vals": per_nv[
                "optimal_step_raw_vals"
            ].tolist(),
            "per_nv_score_arr": per_nv["score_arr"].tolist(),
            "median": {key: val.tolist() for key, val in medians.items()},
        }
    )

    # Apply explicit human choices after automatic optimization so the final
    # SLM weights reflect those manual choices.
    results = apply_manual_step_overrides(
        results,
        manual_step_inds=manual_step_inds,
        manual_step_vals=manual_step_vals,
    )

    # Never retain stale weights from the configuration stored in the file.
    for key in SLM_RESULT_KEYS:
        results.pop(key, None)

    if slm_config is not None and bool(results["is_readout_power_sweep"]):
        results = add_slm_weights(
            results,
            selected_inds=slm_selected_inds,
            config=slm_config,
        )

    print("\n=== Fast re-optimization from processed data ===")
    print("selection mode:", opt_config.selection_mode)
    print("skip first steps:", int(opt_config.skip_first_steps))
    print("score weights:", tuple(opt_config.score_weights))
    print("population optimum:", results["population_optimal_step_val"])
    print("population reason:", results["population_optimal_reason"])
    print("manual overrides:", len(results.get("manual_step_overrides", {})))
    return results


def compare_optimization_configs(
    processed_results: Dict[str, Any],
    named_configs: Dict[str, OptimizationConfig],
    slm_config: Optional[SlmWeightConfig] = SlmWeightConfig(),
    slm_selected_inds: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Re-optimize the same processed arrays under several named settings."""

    if not named_configs:
        raise ValueError("named_configs cannot be empty.")

    out: Dict[str, Dict[str, Any]] = {}
    for name, config in named_configs.items():
        print(f"\n### Re-optimizing configuration: {name} ###")
        out[str(name)] = reoptimize_processed_results(
            processed_results,
            opt_config=config,
            slm_config=slm_config,
            slm_selected_inds=slm_selected_inds,
        )
    return out



# =============================================================================
# SLM weight extraction
# =============================================================================


def add_slm_weights(
    results: Dict[str, Any],
    selected_inds: Optional[Sequence[int]] = None,
    config: SlmWeightConfig = SlmWeightConfig(),
) -> Dict[str, Any]:
    """
    Compute SLM spot weights from per-NV optimal powers.

    Correct logic for SLM-focused spots:
        P_i = per-NV target readout power from the sweep.
        P_mean = mean(P_i) over selected NVs.
        SLM intensity weight w_i = P_i / P_mean.
        Global AOM power should be P_mean / slm_efficiency, not sum(P_i).

    If the hologram code expects field amplitude weights, use sqrt(w_i).
    """

    if not bool(results.get("is_readout_power_sweep", False)):
        raise ValueError(
            "SLM power weights require a readout-amplitude sweep. "
            f"This result x-axis is {results.get('x_label')}"
        )

    power_uW = np.asarray(results["per_nv_optimal_step_vals"], dtype=float)
    num_nvs = power_uW.size

    if selected_inds is None:
        selected_inds_arr = np.where(np.isfinite(power_uW))[0]
    else:
        selected_inds_arr = np.asarray(selected_inds, dtype=int)
        selected_inds_arr = selected_inds_arr[(selected_inds_arr >= 0) & (selected_inds_arr < num_nvs)]
        selected_inds_arr = np.unique(selected_inds_arr)

    selected_power = power_uW[selected_inds_arr]
    valid = np.isfinite(selected_power) & (selected_power > 0)
    selected_inds_valid = selected_inds_arr[valid]
    selected_power = selected_power[valid]

    if selected_power.size == 0:
        raise ValueError("No valid selected NV powers for SLM weight computation.")

    total_target_power_uW = float(np.sum(selected_power))
    mean_target_power_uW = float(np.mean(selected_power))
    median_target_power_uW = float(np.median(selected_power))

    effective_aom_power_uW = mean_target_power_uW / float(config.slm_efficiency)

    axis_info = {
        "power_fit_a": results["power_fit_a"],
        "power_fit_b": results["power_fit_b"],
        "power_fit_c": results["power_fit_c"],
        "yellow_charge_readout_amp": results.get("yellow_charge_readout_amp", np.nan),
        "is_readout_power_sweep": results.get("is_readout_power_sweep", True),
    }
    effective_aom_voltage = aom_voltage_from_power_uW(effective_aom_power_uW, axis_info)
    effective_step_value = opx_step_from_power_uW(effective_aom_power_uW, axis_info)

    intensity_weight = np.full(num_nvs, float(config.invalid_fill), dtype=float)
    intensity_weight[selected_inds_valid] = selected_power / mean_target_power_uW

    clipped_weight = intensity_weight.copy()
    if config.clip_min is not None or config.clip_max is not None:
        lo = -np.inf if config.clip_min is None else float(config.clip_min)
        hi = np.inf if config.clip_max is None else float(config.clip_max)
        clipped_weight[selected_inds_valid] = np.clip(clipped_weight[selected_inds_valid], lo, hi)

    if config.renormalize_after_clip:
        clipped_mean = float(np.mean(clipped_weight[selected_inds_valid]))
        if np.isfinite(clipped_mean) and clipped_mean > 0:
            clipped_weight[selected_inds_valid] = clipped_weight[selected_inds_valid] / clipped_mean

    amplitude_weight = np.full(num_nvs, float(config.invalid_fill), dtype=float)
    good_amp = clipped_weight > 0
    amplitude_weight[good_amp] = np.sqrt(clipped_weight[good_amp])

    power_fraction = np.full(num_nvs, float(config.invalid_fill), dtype=float)
    power_fraction[selected_inds_valid] = selected_power / total_target_power_uW

    results.update(
        {
            "slm_config": asdict(config),
            "slm_selected_inds": selected_inds_valid.astype(int).tolist(),
            "slm_num_selected": int(selected_inds_valid.size),
            "slm_target_power_uW": _full_power_vector(power_uW, selected_inds_valid).tolist(),
            "slm_total_target_power_uW": total_target_power_uW,
            "slm_total_target_power_mW": total_target_power_uW / 1000.0,
            "slm_mean_target_power_uW": mean_target_power_uW,
            "slm_median_target_power_uW": median_target_power_uW,
            "slm_effective_aom_power_uW": effective_aom_power_uW,
            "slm_effective_aom_voltage": effective_aom_voltage,
            "slm_effective_step_value": effective_step_value,
            "slm_mean_norm_intensity_weight": intensity_weight.tolist(),
            "slm_mean_norm_intensity_weight_clipped": clipped_weight.tolist(),
            "slm_amplitude_weight": amplitude_weight.tolist(),
            "slm_power_fraction": power_fraction.tolist(),
            # Backward-compatible aliases.
            "optimal_weights_aligned": intensity_weight.tolist(),
            "optimal_weights_clipped_aligned": clipped_weight.tolist(),
            "aom_voltage_for_slm": effective_aom_voltage,
        }
    )

    print("\n=== SLM effective AOM setting and spot weights ===")
    print("selected NVs:", int(selected_inds_valid.size))
    print("sum target power (uW):", total_target_power_uW)
    print("mean target power per selected NV (uW):", mean_target_power_uW)
    print("median target power per selected NV (uW):", median_target_power_uW)
    print("SLM efficiency:", float(config.slm_efficiency))
    print("effective AOM power to set (uW):", effective_aom_power_uW)
    print("effective AOM voltage to set:", effective_aom_voltage)
    print("effective OPX step value:", effective_step_value)

    _print_weight_stats("unclipped intensity weight", intensity_weight[selected_inds_valid])
    _print_weight_stats("clipped intensity weight", clipped_weight[selected_inds_valid])
    _print_weight_stats("amplitude weight", amplitude_weight[selected_inds_valid])
    print("sum power fractions:", float(np.sum(power_fraction[selected_inds_valid])))

    return results


def _full_power_vector(power_uW: np.ndarray, selected_inds_valid: np.ndarray) -> np.ndarray:
    out = np.full(power_uW.shape, np.nan, dtype=float)
    out[selected_inds_valid] = power_uW[selected_inds_valid]
    return out


def _print_weight_stats(label: str, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    print(f"\n{label}:")
    print("  mean:", float(np.mean(x)))
    print("  median:", float(np.median(x)))
    print("  min:", float(np.min(x)))
    print("  max:", float(np.max(x)))
    print("  std:", float(np.std(x)))


# =============================================================================
# Main analysis entry point
# =============================================================================


def process_repeated_readout_slm(
    raw_data: Dict[str, Any],
    do_plot: bool = True,
    save_data: bool = True,
    n_jobs: int = 12,
    joblib_verbose: int = 10,
    opt_config: OptimizationConfig = OptimizationConfig(),
    slm_config: Optional[SlmWeightConfig] = SlmWeightConfig(),
    slm_selected_inds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Full CPU/joblib pipeline.

    Steps:
        1. Fit bimodal histograms for each NV, step, readout.
        2. Compute R1/R2 fidelities and no-reprep survival metrics.
        3. Choose population and per-NV optimal readout steps.
        4. Convert per-NV optimal powers into SLM weights, if applicable.
        5. Optionally save and plot summary diagnostics.
    """

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = int(raw_data["num_steps"])
    counts = np.asarray(raw_data["counts"], dtype=float)
    validate_counts_shape(counts, num_nvs, num_steps)

    axis_info = get_readout_axis(raw_data)
    step_vals = np.asarray(axis_info["step_vals"], dtype=float)

    print("\n=== Starting repeated-readout SLM CPU analysis ===")
    print("source:", base_file_stem(raw_data))
    print("counts shape:", counts.shape)
    print("num_nvs:", num_nvs)
    print("num_steps:", num_steps)
    print("fits per readout:", num_nvs * num_steps)
    print("total fits:", 2 * num_nvs * num_steps)
    print("n_jobs:", int(n_jobs))
    print("x axis:", axis_info["x_label"])
    print("exp order: 0=ion R1, 1=ion R2, 2=ref R1, 3=ref R2")

    tasks = [
        (nv_ind, step_ind, counts, opt_config.prob_dist_name)
        for nv_ind in range(num_nvs)
        for step_ind in range(num_steps)
    ]

    if n_jobs is None or int(n_jobs) == 1:
        flat_results = [process_one_nv_step(*task) for task in tasks]
    else:
        flat_results = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            verbose=int(joblib_verbose),
            batch_size="auto",
            pre_dispatch="2*n_jobs",
        )(
            delayed(process_one_nv_step)(*task)
            for task in tasks
        )

    arrays = fill_metric_arrays(flat_results, num_nvs, num_steps)

    median = {
        "readout1_fidelity": nanmedian_axis0(arrays["readout1_fidelity_arr"]),
        "readout2_fidelity": nanmedian_axis0(arrays["readout2_fidelity_arr"]),
        "prep1_fidelity": nanmedian_axis0(arrays["prep1_fidelity_arr"]),
        "prep2_fidelity": nanmedian_axis0(arrays["prep2_fidelity_arr"]),
        "goodness1_of_fit": nanmedian_axis0(arrays["goodness1_of_fit_arr"]),
        "goodness2_of_fit": nanmedian_axis0(arrays["goodness2_of_fit_arr"]),
        "ref_same_state_survival": nanmedian_axis0(arrays["ref_same_state_survival_arr"]),
        "ref_nvm_survival": nanmedian_axis0(arrays["ref_nvm_survival_arr"]),
        "ref_nv0_survival": nanmedian_axis0(arrays["ref_nv0_survival_arr"]),
        "ref_nvm_to_nv0_prob": nanmedian_axis0(arrays["ref_nvm_to_nv0_prob_arr"]),
        "mean_ion_r1": nanmedian_axis0(arrays["mean_ion_r1_arr"]),
        "mean_ion_r2": nanmedian_axis0(arrays["mean_ion_r2_arr"]),
        "mean_ref_r1": nanmedian_axis0(arrays["mean_ref_r1_arr"]),
        "mean_ref_r2": nanmedian_axis0(arrays["mean_ref_r2_arr"]),
    }

    avg = {
        "readout1_fidelity": nanmean_axis0(arrays["readout1_fidelity_arr"]),
        "readout2_fidelity": nanmean_axis0(arrays["readout2_fidelity_arr"]),
        "ref_same_state_survival": nanmean_axis0(arrays["ref_same_state_survival_arr"]),
        "ref_nvm_survival": nanmean_axis0(arrays["ref_nvm_survival_arr"]),
        "ref_nv0_survival": nanmean_axis0(arrays["ref_nv0_survival_arr"]),
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
        opt_config,
    )

    per_nv = compute_per_nv_optima(arrays, step_vals, axis_info, opt_config)

    if population_step_ind is None:
        population_step_val = np.nan
        population_step_raw_val = np.nan
        population_aom_voltage = np.nan
        population_opx_step = np.nan
    else:
        population_step_ind = int(population_step_ind)
        population_step_val = float(step_vals[population_step_ind])
        population_step_raw_val = float(axis_info["step_vals_raw"][population_step_ind])
        population_aom_voltage = (
            aom_voltage_from_power_uW(population_step_val, axis_info)
            if bool(axis_info.get("is_readout_power_sweep", False))
            else np.nan
        )
        population_opx_step = (
            opx_step_from_power_uW(population_step_val, axis_info)
            if bool(axis_info.get("is_readout_power_sweep", False))
            else np.nan
        )

    results: Dict[str, Any] = {
        "analysis_type": "repeated_readout_slm_cpu",
        "file_stem_source": base_file_stem(raw_data),
        "num_nvs": int(num_nvs),
        "num_steps": int(num_steps),
        "step_vals_raw": axis_info["step_vals_raw"].tolist(),
        "step_vals": step_vals.tolist(),
        "x_label": axis_info["x_label"],
        "is_readout_power_sweep": bool(axis_info.get("is_readout_power_sweep", False)),
        "power_fit_a": float(axis_info["power_fit_a"]),
        "power_fit_b": float(axis_info["power_fit_b"]),
        "power_fit_c": float(axis_info["power_fit_c"]),
        "yellow_charge_readout_amp": (
            None
            if not np.isfinite(axis_info["yellow_charge_readout_amp"])
            else float(axis_info["yellow_charge_readout_amp"])
        ),
        "prob_dist_name": opt_config.prob_dist_name,
        "exp_order": {
            "0": "ionized_readout_1",
            "1": "ionized_readout_2",
            "2": "reference_readout_1",
            "3": "reference_readout_2",
        },
        "optimization_config": asdict(opt_config),
        "population_optimal_step_ind": population_step_ind,
        "population_optimal_step_val": population_step_val,
        "population_optimal_step_raw_val": population_step_raw_val,
        "population_aom_voltage": population_aom_voltage,
        "population_opx_step_value": population_opx_step,
        "population_optimal_reason": population_reason,
        "population_score": population_score.tolist(),
        "per_nv_optimal_values": make_json_safe(per_nv["optimal_values"]),
        "per_nv_optimal_step_inds": per_nv["optimal_step_inds"].tolist(),
        "per_nv_optimal_step_vals": per_nv["optimal_step_vals"].tolist(),
        "per_nv_optimal_step_raw_vals": per_nv["optimal_step_raw_vals"].tolist(),
        "per_nv_score_arr": per_nv["score_arr"].tolist(),
        "median": {k: v.tolist() for k, v in median.items()},
        "avg": {k: v.tolist() for k, v in avg.items()},
    }

    for key, val in arrays.items():
        if val.dtype == object:
            results[key] = object_array_to_nested_lists(val)
        else:
            results[key] = val.tolist()

    if slm_config is not None and bool(results["is_readout_power_sweep"]):
        results = add_slm_weights(results, selected_inds=slm_selected_inds, config=slm_config)

    print_analysis_summary(results)

    if save_data:
        timestamp = dm.get_time_stamp()
        file_name = f"repeated_readout_slm_processed_{base_file_stem(raw_data)}"
        file_path = dm.get_file_path(__file__, timestamp, file_name)
        dm.save_raw_data(make_json_safe(results), file_path)
        results["saved_file_path"] = str(file_path)
        print("\nSaved processed analysis:", file_path)

    if do_plot:
        plot_summary(results)
        plot_all_nv_scatters(results)
        plot_per_nv_optimal_step_distribution(results)
        plot_optimum_metric_scatter(results)
        if "slm_mean_norm_intensity_weight_clipped" in results:
            plot_slm_weight_distribution(results)

    return results


def print_analysis_summary(results: Dict[str, Any]) -> None:
    print("\n=== Repeated-readout optimum ===")
    print("population optimal step index:", results["population_optimal_step_ind"])
    print(
        f"population optimal {results['x_label']}:",
        results["population_optimal_step_val"],
    )
    print("population raw step value:", results["population_optimal_step_raw_val"])
    print("population AOM voltage:", results["population_aom_voltage"])
    print("population OPX step value:", results["population_opx_step_value"])
    print("reason:", results["population_optimal_reason"])

    opt_ind = results["population_optimal_step_ind"]
    if opt_ind is not None and int(opt_ind) >= 0:
        i = int(opt_ind)
        med = results["median"]
        print(
            "at population optimum: "
            f"R1 fid={med['readout1_fidelity'][i]:.3f}, "
            f"R2 fid={med['readout2_fidelity'][i]:.3f}, "
            f"ref same={med['ref_same_state_survival'][i]:.3f}, "
            f"ref NV- survival={med['ref_nvm_survival'][i]:.3f}, "
            f"ref NV0 survival={med['ref_nv0_survival'][i]:.3f}"
        )

    per_nv_vals = np.asarray(results["per_nv_optimal_step_vals"], dtype=float)
    valid = per_nv_vals[np.isfinite(per_nv_vals)]
    print("valid per-NV optima:", int(valid.size), "/", int(per_nv_vals.size))
    if valid.size:
        print("per-NV mean optimal step:", float(np.mean(valid)))
        print("per-NV median optimal step:", float(np.median(valid)))
        print("per-NV min/max optimal step:", float(np.min(valid)), float(np.max(valid)))


# =============================================================================
# Plotting
# =============================================================================


def _kpl_histogram(
    ax: plt.Axes,
    data: np.ndarray,
    density: bool = True,
    color: Any = None,
    label: Optional[str] = None,
    bins: str | int | np.ndarray = "auto",
    alpha: float = 0.55,
):
    """Use lab kplotlib histogram styling, with a matplotlib fallback."""

    data = finite_flatten(data)
    if data.size == 0:
        return None

    try:
        return kpl.histogram(
            ax,
            data,
            density=density,
            color=color,
            label=label,
        )
    except TypeError:
        return ax.hist(
            data,
            bins=bins,
            density=density,
            color=color,
            alpha=alpha,
            label=label,
        )


def _kpl_plot_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: Any = None,
    label: Optional[str] = None,
    **kwargs,
):
    """Use kpl.plot_line where possible, with a matplotlib fallback."""

    try:
        return kpl.plot_line(ax, x, y, color=color, label=label, **kwargs)
    except TypeError:
        return ax.plot(x, y, color=color, label=label, **kwargs)


def _add_population_vline(ax: plt.Axes, opt_val: float, label: str = "Population optimum") -> None:
    if np.isfinite(opt_val):
        ax.axvline(
            opt_val,
            color=kpl.KplColors.RED,
            linestyle="--",
            linewidth=1.8,
            label=label,
        )


def _style_axis(ax: plt.Axes, legend: bool = True, legend_fontsize: int = 9) -> None:
    ax.grid(True, linestyle="--", alpha=0.35)
    if legend:
        try:
            ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=legend_fontsize)
        except Exception:
            ax.legend(fontsize=legend_fontsize)


def population_optimum(results: Dict[str, Any]) -> Tuple[Optional[int], float]:
    opt_ind = results.get("population_optimal_step_ind", None)
    opt_val = results.get("population_optimal_step_val", None)
    if opt_ind is None or opt_val is None or not np.isfinite(opt_val):
        return None, np.nan
    return int(opt_ind), float(opt_val)


def plot_summary(results: Dict[str, Any]) -> plt.Figure:
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    opt_ind, opt_val = population_optimum(results)
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

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)

    axes[0].plot(step_vals, r1, "o-", color=kpl.KplColors.BLUE, label="Readout 1 fidelity")
    axes[0].plot(step_vals, r2, "o-", color=kpl.KplColors.GREEN, label="Readout 2 fidelity")
    axes[0].set_ylabel("Fidelity")
    axes[0].set_ylim(0, 1.02)

    axes[1].plot(step_vals, same, "o-", color=kpl.KplColors.BLUE, label="Ref same-state survival")
    axes[1].plot(step_vals, nvm, "o-", color=kpl.KplColors.GREEN, label="Ref NV- survival")
    axes[1].plot(step_vals, nv0, "o-", color=kpl.KplColors.RED, label="Ref NV0 survival")
    axes[1].set_ylabel("Survival")
    axes[1].set_ylim(0, 1.02)

    axes[2].plot(step_vals, ion, "o-", color=kpl.KplColors.RED, label="Median ion R1 counts")
    axes[2].plot(step_vals, ref1, "o-", color=kpl.KplColors.GREEN, label="Median ref R1 counts")
    axes[2].plot(step_vals, ref2, "o-", color=kpl.KplColors.BLUE, label="Median ref R2 counts")
    axes[2].set_ylabel("Counts")

    axes[3].plot(step_vals, score, "o-", color=kpl.KplColors.BLUE, label="Population score")
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel(x_label)

    for ax in axes:
        _add_population_vline(ax, opt_val)
        _style_axis(ax, legend=True, legend_fontsize=9)

    title = "Repeated two-readout optimization: classification + no-reprep survival"
    if opt_ind is not None:
        title += f"\nchosen population index {opt_ind}, {x_label}={opt_val:.4g}"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_all_nv_scatters(
    results: Dict[str, Any],
    alpha: float = 0.28,
    size: float = 12,
) -> plt.Figure:
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    _, opt_val = population_optimum(results)

    r1 = np.asarray(results["readout1_fidelity_arr"], dtype=float)
    r2 = np.asarray(results["readout2_fidelity_arr"], dtype=float)
    same = np.asarray(results["ref_same_state_survival_arr"], dtype=float)
    nvm = np.asarray(results["ref_nvm_survival_arr"], dtype=float)
    nvm_to_nv0 = np.asarray(results["ref_nvm_to_nv0_prob_arr"], dtype=float)
    score = np.asarray(results["per_nv_score_arr"], dtype=float)

    x = np.tile(step_vals, r1.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    panels = [
        (r1, "R1 fidelity", "Fidelity", kpl.KplColors.BLUE),
        (r2, "R2 fidelity", "Fidelity", kpl.KplColors.GREEN),
        (same, "Reference same-state survival", "Survival", kpl.KplColors.BLUE),
        (nvm, "Reference NV- survival", "Survival", kpl.KplColors.GREEN),
        (nvm_to_nv0, "Reference NV- → NV0 probability", "Ionization probability", kpl.KplColors.RED),
        (score, "Per-NV combined score", "Score", kpl.KplColors.BLUE),
    ]

    for ax, (arr, title, ylabel, color) in zip(axes, panels):
        arr = np.asarray(arr, dtype=float)
        y = arr.reshape(-1)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=size, alpha=alpha, color=color)
        med = np.nanmedian(arr, axis=0)
        ax.plot(step_vals, med, color=kpl.KplColors.GRAY, lw=2, label="Median")
        _add_population_vline(ax, opt_val)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        _style_axis(ax, legend=True, legend_fontsize=8)

    fig.suptitle("All-NV repeated-readout scatter diagnostics", fontsize=14)
    fig.tight_layout()
    return fig


def plot_per_nv_optimal_step_distribution(results: Dict[str, Any]) -> plt.Figure:
    step_vals = np.asarray(results["per_nv_optimal_step_vals"], dtype=float)
    valid = step_vals[np.isfinite(step_vals)]
    x_label = results["x_label"]
    _, opt_val = population_optimum(results)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    if valid.size:
        _kpl_histogram(
            ax,
            valid,
            density=False,
            color=kpl.KplColors.BLUE,
            label="Per-NV optima",
            bins=min(40, max(5, int(np.sqrt(valid.size)))),
            alpha=0.8,
        )
        mean_val = float(np.mean(valid))
        median_val = float(np.median(valid))
        ax.axvline(mean_val, color=kpl.KplColors.GRAY, linestyle=":", label=f"Mean = {mean_val:.3g}")
        ax.axvline(median_val, color=kpl.KplColors.GREEN, linestyle="-.", label=f"Median = {median_val:.3g}")

    _add_population_vline(ax, opt_val, label=f"Population = {opt_val:.3g}" if np.isfinite(opt_val) else "Population")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of NVs")
    ax.set_title("Distribution of per-NV optimal readout settings", fontsize=13)
    _style_axis(ax, legend=True, legend_fontsize=9)
    fig.tight_layout()
    return fig


def plot_optimum_metric_scatter(results: Dict[str, Any]) -> plt.Figure:
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
    axes[0].set_title("R1 fidelity vs NV- survival", fontsize=12)
    plt.colorbar(sc0, ax=axes[0], label=results["x_label"])

    sc1 = axes[1].scatter(r2, nvm, c=step, s=18, alpha=0.75)
    axes[1].set_xlabel("R2 fidelity")
    axes[1].set_ylabel("Ref NV- survival")
    axes[1].set_title("R2 fidelity vs NV- survival", fontsize=12)
    plt.colorbar(sc1, ax=axes[1], label=results["x_label"])

    sc2 = axes[2].scatter(same, nvm, c=step, s=18, alpha=0.75)
    axes[2].set_xlabel("Ref same-state survival")
    axes[2].set_ylabel("Ref NV- survival")
    axes[2].set_title("Survival consistency", fontsize=12)
    plt.colorbar(sc2, ax=axes[2], label=results["x_label"])

    for ax in axes:
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        _style_axis(ax, legend=False)

    fig.suptitle("All NVs at their own selected optimum", fontsize=14)
    return fig


def plot_slm_weight_distribution(results: Dict[str, Any]) -> plt.Figure:
    selected = np.asarray(results["slm_selected_inds"], dtype=int)
    intensity = np.asarray(results["slm_mean_norm_intensity_weight"], dtype=float)[selected]
    clipped = np.asarray(results["slm_mean_norm_intensity_weight_clipped"], dtype=float)[selected]
    amp = np.asarray(results["slm_amplitude_weight"], dtype=float)[selected]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [
        (intensity, "Unclipped intensity weights", kpl.KplColors.BLUE),
        (clipped, "Clipped / renormalized intensity weights", kpl.KplColors.GREEN),
        (amp, "Amplitude weights", kpl.KplColors.RED),
    ]

    for ax, (x, title, color) in zip(axes, panels):
        x = x[np.isfinite(x)]
        if x.size:
            _kpl_histogram(
                ax,
                x,
                density=False,
                color=color,
                label=title,
                bins=min(40, max(5, int(np.sqrt(x.size)))),
                alpha=0.85,
            )
            ax.axvline(np.mean(x), color=kpl.KplColors.GRAY, linestyle=":", label=f"mean={np.mean(x):.3g}")
            ax.axvline(np.median(x), color=kpl.KplColors.GREEN, linestyle="-.", label=f"median={np.median(x):.3g}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Number of NVs")
        _style_axis(ax, legend=True, legend_fontsize=8)

    fig.suptitle("SLM spot-weight diagnostics", fontsize=14)
    fig.tight_layout()
    return fig


# =============================================================================
# Representative NV selection and histogram plotting
# =============================================================================


def pick_representative_nvs(results: Dict[str, Any]) -> Dict[str, int]:
    opt_vals = results["per_nv_optimal_values"]

    score = np.asarray([v["score"] for v in opt_vals], dtype=float)
    nvm = np.asarray([v["ref_nvm_survival"] for v in opt_vals], dtype=float)
    r1 = np.asarray([v["readout1_fidelity"] for v in opt_vals], dtype=float)

    valid = np.isfinite(score)
    valid_inds = np.where(valid)[0]
    if valid_inds.size == 0:
        raise ValueError("No valid NVs for representative selection.")

    best_nv = int(valid_inds[np.nanargmax(score[valid_inds])])
    low_survival_nv = int(valid_inds[np.nanargmin(nvm[valid_inds])])
    low_fidelity_nv = int(valid_inds[np.nanargmin(r1[valid_inds])])

    median_score = np.nanmedian(score[valid_inds])
    median_nv = int(valid_inds[np.nanargmin(np.abs(score[valid_inds] - median_score))])

    reps = {
        "best_score_nv": best_nv,
        "median_score_nv": median_nv,
        "lowest_nv_minus_survival_nv": low_survival_nv,
        "lowest_readout1_fidelity_nv": low_fidelity_nv,
    }

    # Avoid plotting the same NV many times if one NV is both best/worst in a small set.
    unique_reps: Dict[str, int] = {}
    used = set()
    for label, nv in reps.items():
        if nv not in used:
            unique_reps[label] = nv
            used.add(nv)
    return unique_reps


def print_nv_optimum_summary(results: Dict[str, Any], nv_ind: int) -> None:
    opt = results["per_nv_optimal_values"][int(nv_ind)]

    print("\n=== NV repeated-readout optimum ===")
    print("NV:", int(nv_ind))
    print("step index:", opt["optimal_step_ind"])
    print(f"{results['x_label']}:", opt["optimal_step_val"])
    print("raw step value:", opt["optimal_step_raw_val"])
    print("reason:", opt["reason"])
    print("score:", opt["score"])
    print("R1 fidelity:", opt["readout1_fidelity"])
    print("R2 fidelity:", opt["readout2_fidelity"])
    print("ref same-state survival:", opt["ref_same_state_survival"])
    print("ref NV- survival:", opt["ref_nvm_survival"])
    print("ref NV0 survival:", opt["ref_nv0_survival"])
    print("AOM voltage:", opt["aom_voltage"])
    print("OPX step value:", opt["opx_step_value"])


def fit_param_text(popt: Any, prob_dist: ProbDist, red_chi_sq: Optional[float] = None) -> str:
    if popt is None:
        return "fit params: None"

    try:
        popt = np.asarray(popt, dtype=float).ravel()
        n_single = bimodal_histogram.get_single_mode_num_params(prob_dist)

        dark_weight = float(popt[0])
        bright_weight = 1.0 - dark_weight
        dark_params = popt[1 : 1 + n_single]
        bright_params = popt[1 + n_single : 1 + 2 * n_single]

        lines = [f"w0={dark_weight:.3f}", f"w-={bright_weight:.3f}"]
        if n_single == 1:
            lines += [f"NV0 rate={dark_params[0]:.2f}", f"NV- rate={bright_params[0]:.2f}"]
        else:
            lines += [
                "NV0: " + ", ".join(f"{v:.2f}" for v in dark_params),
                "NV-: " + ", ".join(f"{v:.2f}" for v in bright_params),
            ]
        if red_chi_sq is not None and np.isfinite(red_chi_sq):
            lines.append(f"red chi2={red_chi_sq:.3g}")
        return "\n".join(lines)
    except Exception:
        return "fit params unreadable"


def plot_fit_components(
    ax: plt.Axes,
    counts: np.ndarray,
    popt: Any,
    prob_dist: ProbDist,
    density: bool = True,
) -> None:
    if popt is None:
        return

    popt = np.asarray(popt, dtype=float).ravel()
    if not np.all(np.isfinite(popt)):
        return

    x_max = float(np.nanmax(counts)) if counts.size else 1.0
    x_vals = np.linspace(0, max(1.0, x_max + 1.0), 1000)

    n_single = bimodal_histogram.get_single_mode_num_params(prob_dist)
    single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)
    bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist)

    dark = popt[0] * single_pdf(x_vals, *popt[1 : 1 + n_single])
    bright = (1.0 - popt[0]) * single_pdf(x_vals, *popt[1 + n_single :])
    both = bimodal_pdf(x_vals, *popt)

    # With density=True, histogram is normalized as a PDF, so these curves match directly.
    # With density=False, scale approximately by sample count.
    if not density:
        scale = counts.size
        dark = dark * scale
        bright = bright * scale
        both = both * scale

    _kpl_plot_line(ax, x_vals, dark, color=kpl.KplColors.RED, label="NV0 fit", lw=2)
    _kpl_plot_line(ax, x_vals, bright, color=kpl.KplColors.GREEN, label="NV- fit", lw=2)
    _kpl_plot_line(ax, x_vals, both, color=kpl.KplColors.BLUE, label="Combined fit", lw=2)


def plot_two_readout_histograms_at_optimum(
    raw_data: Dict[str, Any],
    results: Dict[str, Any],
    nv_ind: int,
    step_ind: Optional[int] = None,
    use_population_step: bool = False,
    density: bool = True,
    bins: str | int = "auto",
) -> plt.Figure:
    """
    kplotlib-style two-readout histogram diagnostic.

    Left panel:
        ionized R1 + reference R1 + R1 bimodal fit.

    Right panel:
        ionized R2 + reference R2 + R2 bimodal fit.

    The vertical threshold is the R1 threshold, because that is what is used for
    survival classification.
    """

    counts_all = np.asarray(raw_data["counts"], dtype=float)
    nv_ind = int(nv_ind)

    if use_population_step:
        step_ind = results["population_optimal_step_ind"]
    elif step_ind is None:
        step_ind = results["per_nv_optimal_values"][nv_ind]["optimal_step_ind"]

    if step_ind is None or int(step_ind) < 0:
        raise ValueError(f"NV {nv_ind} has no valid step for histogram plotting.")
    step_ind = int(step_ind)

    ion_r1 = finite_flatten(counts_all[0, nv_ind, :, step_ind, :])
    ion_r2 = finite_flatten(counts_all[1, nv_ind, :, step_ind, :])
    ref_r1 = finite_flatten(counts_all[2, nv_ind, :, step_ind, :])
    ref_r2 = finite_flatten(counts_all[3, nv_ind, :, step_ind, :])

    threshold = float(np.asarray(results["threshold_r1_arr"], dtype=float)[nv_ind, step_ind])
    fit1_params = results["fit1_params_arr"][nv_ind][step_ind]
    fit2_params = results["fit2_params_arr"][nv_ind][step_ind]
    gof1 = float(np.asarray(results["goodness1_of_fit_arr"], dtype=float)[nv_ind, step_ind])
    gof2 = float(np.asarray(results["goodness2_of_fit_arr"], dtype=float)[nv_ind, step_ind])

    prob_dist = ProbDist[results.get("prob_dist_name", "COMPOUND_POISSON")]
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    step_val = float(step_vals[step_ind])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    panels = [
        {
            "ax": axes[0],
            "title": "Readout 1",
            "ion": ion_r1,
            "ref": ref_r1,
            "fit_params": fit1_params,
            "gof": gof1,
            "fid_arr": "readout1_fidelity_arr",
        },
        {
            "ax": axes[1],
            "title": "Readout 2",
            "ion": ion_r2,
            "ref": ref_r2,
            "fit_params": fit2_params,
            "gof": gof2,
            "fid_arr": "readout2_fidelity_arr",
        },
    ]

    for panel in panels:
        ax = panel["ax"]
        ion = panel["ion"]
        ref = panel["ref"]
        combined = np.concatenate([ion, ref])
        fid = float(np.asarray(results[panel["fid_arr"]], dtype=float)[nv_ind, step_ind])

        hist_bins = np.histogram_bin_edges(combined, bins=bins) if combined.size else bins

        _kpl_histogram(
            ax,
            ion,
            density=density,
            color=kpl.KplColors.RED,
            label="Ionized branch",
            bins=hist_bins,
            alpha=0.45,
        )
        _kpl_histogram(
            ax,
            ref,
            density=density,
            color=kpl.KplColors.GREEN,
            label="Reference branch",
            bins=hist_bins,
            alpha=0.45,
        )

        if np.isfinite(threshold):
            ax.axvline(
                threshold,
                color=kpl.KplColors.GRAY,
                linestyle="--",
                lw=2,
                label=f"R1 threshold={threshold:.1f}",
            )

        plot_fit_components(ax, ref, panel["fit_params"], prob_dist, density=density)

        ax.text(
            0.98,
            0.72,
            fit_param_text(panel["fit_params"], prob_dist, red_chi_sq=panel["gof"]),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        ax.set_xlabel("Integrated counts")
        ax.set_ylabel("Probability density" if density else "Occurrences")
        ax.set_title(f"{panel['title']}: fidelity={fid:.3f}", fontsize=12)
        _style_axis(ax, legend=True, legend_fontsize=8)

    same = float(np.asarray(results["ref_same_state_survival_arr"], dtype=float)[nv_ind, step_ind])
    nvm = float(np.asarray(results["ref_nvm_survival_arr"], dtype=float)[nv_ind, step_ind])
    nv0 = float(np.asarray(results["ref_nv0_survival_arr"], dtype=float)[nv_ind, step_ind])

    mode = "population optimum" if use_population_step else "per-NV optimum"
    fig.suptitle(
        f"NV {nv_ind}, {mode}: step {step_ind}, {x_label}={step_val:.4g}\n"
        f"ref same={same:.3f}, ref NV- survival={nvm:.3f}, ref NV0 survival={nv0:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_representative_histograms(
    raw_data: Dict[str, Any],
    results: Dict[str, Any],
    density: bool = True,
    use_population_step: bool = False,
) -> List[plt.Figure]:
    reps = pick_representative_nvs(results)
    figs = []
    for label, nv_ind in reps.items():
        print("\nRepresentative:", label)
        print_nv_optimum_summary(results, nv_ind)
        figs.append(
            plot_two_readout_histograms_at_optimum(
                raw_data,
                results,
                nv_ind=nv_ind,
                density=density,
                use_population_step=use_population_step,
            )
        )
    return figs



def plot_optimization_config_comparison(
    config_results: Dict[str, Dict[str, Any]],
) -> plt.Figure:
    """
    Compare how optimization settings change selected powers and SLM weights.

    The top row summarizes each configuration. The lower row overlays the
    per-NV optimum and clipped SLM-weight distributions.
    """

    if not config_results:
        raise ValueError("config_results cannot be empty.")

    names = list(config_results)
    x_pos = np.arange(len(names), dtype=float)

    population_vals = []
    median_per_nv = []
    threshold_fraction = []
    valid_fraction = []

    for name in names:
        results = config_results[name]
        population_vals.append(
            float(results.get("population_optimal_step_val", np.nan))
        )

        per_nv_steps = np.asarray(
            results["per_nv_optimal_step_vals"], dtype=float
        )
        median_per_nv.append(
            float(np.nanmedian(per_nv_steps))
            if np.any(np.isfinite(per_nv_steps))
            else np.nan
        )
        valid_fraction.append(
            float(np.mean(np.isfinite(per_nv_steps)))
            if per_nv_steps.size
            else np.nan
        )

        reasons = [
            str(item.get("reason", ""))
            for item in results["per_nv_optimal_values"]
        ]
        threshold_fraction.append(
            float(
                np.mean(
                    [
                        reason == "lowest step satisfying thresholds"
                        for reason in reasons
                    ]
                )
            )
            if reasons
            else np.nan
        )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    axes[0, 0].plot(
        x_pos,
        population_vals,
        "o-",
        label="Population optimum",
    )
    axes[0, 0].plot(
        x_pos,
        median_per_nv,
        "s--",
        label="Median per-NV optimum",
    )
    axes[0, 0].set_xticks(x_pos, names, rotation=20, ha="right")
    axes[0, 0].set_ylabel(
        next(iter(config_results.values()))["x_label"]
    )
    axes[0, 0].set_title("Selected readout setting")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    width = 0.38
    axes[0, 1].bar(
        x_pos - width / 2,
        100.0 * np.asarray(threshold_fraction),
        width,
        label="Threshold-selected",
    )
    axes[0, 1].bar(
        x_pos + width / 2,
        100.0 * np.asarray(valid_fraction),
        width,
        label="Valid optimum",
    )
    axes[0, 1].set_xticks(x_pos, names, rotation=20, ha="right")
    axes[0, 1].set_ylabel("NVs (%)")
    axes[0, 1].set_ylim(0.0, 105.0)
    axes[0, 1].set_title("Selection coverage")
    axes[0, 1].grid(True, axis="y", alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    all_step_values = np.concatenate(
        [
            np.asarray(results["per_nv_optimal_step_vals"], dtype=float)
            for results in config_results.values()
        ]
    )
    finite_steps = all_step_values[np.isfinite(all_step_values)]
    bins_step = (
        np.histogram_bin_edges(
            finite_steps,
            bins=min(40, max(8, int(np.sqrt(finite_steps.size)))),
        )
        if finite_steps.size
        else 10
    )

    for name, results in config_results.items():
        vals = np.asarray(
            results["per_nv_optimal_step_vals"], dtype=float
        )
        vals = vals[np.isfinite(vals)]
        if vals.size:
            axes[1, 0].hist(
                vals,
                bins=bins_step,
                histtype="step",
                linewidth=2.0,
                label=name,
            )
    axes[1, 0].set_xlabel(
        next(iter(config_results.values()))["x_label"]
    )
    axes[1, 0].set_ylabel("Number of NVs")
    axes[1, 0].set_title("Per-NV optimal-setting distribution")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    weight_sets = []
    for results in config_results.values():
        if "slm_mean_norm_intensity_weight_clipped" not in results:
            continue
        selected = np.asarray(results["slm_selected_inds"], dtype=int)
        vals = np.asarray(
            results["slm_mean_norm_intensity_weight_clipped"],
            dtype=float,
        )[selected]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            weight_sets.append(vals)

    if weight_sets:
        all_weights = np.concatenate(weight_sets)
        bins_weight = np.histogram_bin_edges(
            all_weights,
            bins=min(40, max(8, int(np.sqrt(all_weights.size)))),
        )
        for name, results in config_results.items():
            if "slm_mean_norm_intensity_weight_clipped" not in results:
                continue
            selected = np.asarray(results["slm_selected_inds"], dtype=int)
            vals = np.asarray(
                results["slm_mean_norm_intensity_weight_clipped"],
                dtype=float,
            )[selected]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                axes[1, 1].hist(
                    vals,
                    bins=bins_weight,
                    histtype="step",
                    linewidth=2.0,
                    label=name,
                )
        axes[1, 1].axvline(
            1.0,
            linestyle="--",
            linewidth=1.5,
            label="Mean-normalized weight",
        )
        axes[1, 1].legend(fontsize=9)

    axes[1, 1].set_xlabel("Clipped SLM intensity weight")
    axes[1, 1].set_ylabel("Number of NVs")
    axes[1, 1].set_title("Final SLM-weight distribution")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Sensitivity of optimal SLM settings to optimization parameters",
        fontsize=15,
    )
    fig.tight_layout()
    return fig


def _plot_reference_fit_histogram_panel(
    ax: plt.Axes,
    raw_data: Dict[str, Any],
    results: Dict[str, Any],
    nv_ind: int,
    step_ind: int,
    readout_number: int,
    row_label: str,
    density: bool,
    bins: str | int,
) -> None:
    """
    Plot one R1/R2 reference histogram and its saved reference-only fit.

    The ionized branch is included as a lightly shaded diagnostic, but the
    fitted components are evaluated against the reference histogram because
    process_one_nv_step() fitted the reference branch alone.
    """

    if readout_number not in (1, 2):
        raise ValueError("readout_number must be 1 or 2.")

    counts_all = np.asarray(raw_data["counts"], dtype=float)
    ion_exp = 0 if readout_number == 1 else 1
    ref_exp = 2 if readout_number == 1 else 3
    fit_key = "fit1_params_arr" if readout_number == 1 else "fit2_params_arr"
    fidelity_key = (
        "readout1_fidelity_arr"
        if readout_number == 1
        else "readout2_fidelity_arr"
    )
    gof_key = (
        "goodness1_of_fit_arr"
        if readout_number == 1
        else "goodness2_of_fit_arr"
    )

    ion = finite_flatten(counts_all[ion_exp, nv_ind, :, step_ind, :])
    ref = finite_flatten(counts_all[ref_exp, nv_ind, :, step_ind, :])
    combined = np.concatenate([ion, ref])
    hist_bins = (
        np.histogram_bin_edges(combined, bins=bins)
        if combined.size
        else bins
    )

    _kpl_histogram(
        ax,
        ion,
        density=density,
        color=kpl.KplColors.RED,
        label="Ionized branch",
        bins=hist_bins,
        alpha=0.25,
    )
    _kpl_histogram(
        ax,
        ref,
        density=density,
        color=kpl.KplColors.GREEN,
        label="Reference branch",
        bins=hist_bins,
        alpha=0.55,
    )

    threshold = float(
        np.asarray(results["threshold_r1_arr"], dtype=float)[nv_ind, step_ind]
    )
    if np.isfinite(threshold):
        ax.axvline(
            threshold,
            color=kpl.KplColors.GRAY,
            linestyle="--",
            linewidth=2.0,
            label=f"R1 threshold={threshold:.1f}",
        )

    fit_params = results[fit_key][nv_ind][step_ind]
    prob_dist = ProbDist[
        results.get("prob_dist_name", "COMPOUND_POISSON")
    ]
    plot_fit_components(
        ax,
        ref,
        fit_params,
        prob_dist,
        density=density,
    )

    fidelity = float(
        np.asarray(results[fidelity_key], dtype=float)[nv_ind, step_ind]
    )
    gof = float(
        np.asarray(results[gof_key], dtype=float)[nv_ind, step_ind]
    )
    step_val = float(
        np.asarray(results["step_vals"], dtype=float)[step_ind]
    )

    ax.text(
        0.98,
        0.72,
        fit_param_text(fit_params, prob_dist, red_chi_sq=gof),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )
    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Probability density" if density else "Occurrences")
    ax.set_title(
        f"{row_label}, readout {readout_number}\n"
        f"step {step_ind}, {results['x_label']}={step_val:.4g}, "
        f"fidelity={fidelity:.3f}",
        fontsize=11,
    )
    _style_axis(ax, legend=True, legend_fontsize=7.5)


def plot_auto_vs_manual_optimal_histograms(
    raw_data: Dict[str, Any],
    automatic_results: Dict[str, Any],
    nv_ind: int,
    manual_step_ind: Optional[int] = None,
    manual_step_val: Optional[float] = None,
    density: bool = True,
    bins: str | int = "auto",
) -> plt.Figure:
    """
    Compare the automatically selected and manually selected histograms.

    Top row: automatic optimum for the chosen NV.
    Bottom row: manual step supplied by index or nearest physical value.
    Columns: readout 1 and readout 2.
    """

    nv_ind = int(nv_ind)
    automatic_step_ind = int(
        automatic_results["per_nv_optimal_step_inds"][nv_ind]
    )
    if automatic_step_ind < 0:
        raise ValueError(f"NV {nv_ind} has no automatic optimum.")

    manual_resolved_ind = resolve_manual_step_index(
        automatic_results,
        step_ind=manual_step_ind,
        step_val=manual_step_val,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.0),
        sharex="col",
        sharey="row",
    )

    for col, readout_number in enumerate((1, 2)):
        _plot_reference_fit_histogram_panel(
            axes[0, col],
            raw_data,
            automatic_results,
            nv_ind=nv_ind,
            step_ind=automatic_step_ind,
            readout_number=readout_number,
            row_label="Automatic optimum",
            density=density,
            bins=bins,
        )
        _plot_reference_fit_histogram_panel(
            axes[1, col],
            raw_data,
            automatic_results,
            nv_ind=nv_ind,
            step_ind=manual_resolved_ind,
            readout_number=readout_number,
            row_label="Manual choice",
            density=density,
            bins=bins,
        )

    step_vals = np.asarray(automatic_results["step_vals"], dtype=float)
    fig.suptitle(
        f"NV {nv_ind}: automatic versus manual histogram selection\n"
        f"automatic step {automatic_step_ind} "
        f"({step_vals[automatic_step_ind]:.4g}); "
        f"manual step {manual_resolved_ind} "
        f"({step_vals[manual_resolved_ind]:.4g})",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_manual_override_histograms(
    raw_data: Dict[str, Any],
    automatic_results: Dict[str, Any],
    manual_step_inds: Optional[Dict[int, int]] = None,
    manual_step_vals: Optional[Dict[int, float]] = None,
    density: bool = True,
) -> List[plt.Figure]:
    """Create automatic-versus-manual histogram figures for all overrides."""

    manual_step_inds = {} if manual_step_inds is None else dict(manual_step_inds)
    manual_step_vals = {} if manual_step_vals is None else dict(manual_step_vals)

    overlap = set(manual_step_inds) & set(manual_step_vals)
    if overlap:
        raise ValueError(
            "Duplicate manual NVs in index/value dictionaries: "
            f"{sorted(overlap)}"
        )

    figures: List[plt.Figure] = []
    for nv_ind, step_ind in manual_step_inds.items():
        figures.append(
            plot_auto_vs_manual_optimal_histograms(
                raw_data,
                automatic_results,
                nv_ind=int(nv_ind),
                manual_step_ind=int(step_ind),
                density=density,
            )
        )

    for nv_ind, step_val in manual_step_vals.items():
        figures.append(
            plot_auto_vs_manual_optimal_histograms(
                raw_data,
                automatic_results,
                nv_ind=int(nv_ind),
                manual_step_val=float(step_val),
                density=density,
            )
        )

    return figures



# =============================================================================
# Convenience loader / saver helpers
# =============================================================================


def load_raw(file_id: str) -> Dict[str, Any]:
    raw_data = dm.get_raw_data(file_stem=file_id, load_npz=True)
    raw_data["file_stem"] = file_id
    return raw_data


def load_processed(processed_file: str) -> Dict[str, Any]:
    return dm.get_raw_data(file_stem=processed_file, load_npz=True)


def save_results_again(results: Dict[str, Any], prefix: str = "repeated_readout_slm_processed") -> str:
    timestamp = dm.get_time_stamp()
    source = results.get("file_stem_source", "raw_data")
    file_name = f"{prefix}_{source}"
    file_path = dm.get_file_path(__file__, timestamp, file_name)
    dm.save_raw_data(make_json_safe(results), file_path)
    print("Saved:", file_path)
    return str(file_path)


# =============================================================================
# Example usage
# =============================================================================


if __name__ == "__main__":
    kpl.init_kplotlib()

    # False: load the saved processed fit arrays and re-optimize quickly.
    # True: rerun all CPU/joblib histogram fits from the raw data.
    RUN_NEW_ANALYSIS = True

    # Raw data are still loaded when plotting histograms.
    # file_id = "2026_07_11-04_37_50-qnami-nv0_2026_02_20"
    # file_id = "2026_07_14-20_28_11-qnami-nv0_2026_02_20"
    # file_id = "2026_07_22-04_39_34-qnami-nv0_2026_02_20"
    # file_id = "2026_08_05-04_43_20-qnami-nv0_2026_02_20"
    file_id = "2026_08_12-10_12_17-qnami-nv0_2026_02_20" ## 631NVs
     

    # processed_file = "2026_07_14-16_52_29-repeated_readout_slm_processed_2026_07_11-04_37_50-qnami-nv0_2026_02_20"
    processed_file = "2026_07_18-11_47_06-repeated_readout_slm_processed_2026_07_18-11_10_09-qnami-nv0_2026_02_20"
    raw_data = load_raw(file_id)

    # -------------------------------------------------------------------------
    # Try several optimization strategies on the same processed fit arrays.
    #
    # Important:
    #   In "threshold_then_score" mode, score weights only matter when no step
    #   passes every threshold. Use "max_score" to directly study sensitivity
    #   to score weights.
    # -------------------------------------------------------------------------
    OPTIMIZATION_CONFIGS = {
        "threshold_balanced": OptimizationConfig(
            min_readout1_fidelity=0.88,
            min_readout2_fidelity=0.88,
            min_ref_same_state_survival=0.96,
            min_ref_nvm_survival=0.97,
            min_ref_nv0_survival=None,
            score_weights=(0.35, 0.40, 0.15, 0.10),
            selection_mode="threshold_then_score",
            skip_first_steps=2,
            prob_dist_name="COMPOUND_POISSON",
        ),
        "score_survival_focused": OptimizationConfig(
            min_readout1_fidelity=0.88,
            min_readout2_fidelity=0.88,
            min_ref_same_state_survival=0.96,
            min_ref_nvm_survival=0.97,
            min_ref_nv0_survival=None,
            score_weights=(0.20, 0.60, 0.10, 0.10),
            selection_mode="max_score",
            skip_first_steps=2,
            prob_dist_name="COMPOUND_POISSON",
        ),
        "score_fidelity_focused": OptimizationConfig(
            min_readout1_fidelity=0.88,
            min_readout2_fidelity=0.88,
            min_ref_same_state_survival=0.96,
            min_ref_nvm_survival=0.97,
            min_ref_nv0_survival=None,
            score_weights=(0.60, 0.20, 0.10, 0.10),
            selection_mode="max_score",
            skip_first_steps=2,
            prob_dist_name="COMPOUND_POISSON",
        ),
    }

    ACTIVE_CONFIG_NAME = "score_survival_focused"

    # Final SLM-weight handling.
    SLM_CONFIG = SlmWeightConfig(
        slm_efficiency=1.0,
        invalid_fill=0.0,
        clip_min=0.0,
        clip_max=2.0,
        renormalize_after_clip=True,
    )

    # None means every NV with a finite positive selected target power.
    # Example: SLM_SELECTED_INDS = [0, 1, 2, 8, 10, 303]
    SLM_SELECTED_INDS = None

    # -------------------------------------------------------------------------
    # Optional manual per-NV optimum overrides.
    #
    # These choices replace the automatic optimum before final SLM weights are
    # calculated. Use exact sampled index OR nearest physical step value.
    #
    # Examples:
    # MANUAL_STEP_INDS = {8: 12, 303: 15}
    # MANUAL_STEP_VALS = {364: 7.5, 422: 9.0}
    # -------------------------------------------------------------------------
    MANUAL_STEP_INDS: Dict[int, int] = {}
    MANUAL_STEP_VALS: Dict[int, float] = {}

    # Plot automatic-versus-manual R1/R2 histograms for every override above.
    PLOT_MANUAL_HISTOGRAMS = True

    # Additional automatic histogram diagnostics.
    PLOT_REPRESENTATIVE_HISTOGRAMS = False

    # Save the re-optimized result dictionary as a new processed file.
    SAVE_REOPTIMIZED_RESULTS = True
    # =========================================================================
    # LOAD / PROCESS
    # =========================================================================

    raw_data = load_raw(file_id)
    active_config = OPTIMIZATION_CONFIGS[ACTIVE_CONFIG_NAME]

    if RUN_NEW_ANALYSIS:
        results = process_repeated_readout_slm(
            raw_data,
            do_plot=False,
            save_data=True,
            n_jobs=12,
            joblib_verbose=10,
            opt_config=active_config,
            slm_config=SLM_CONFIG,
            slm_selected_inds=SLM_SELECTED_INDS,
        )
        automatic_results = copy.deepcopy(results)
        config_results = {ACTIVE_CONFIG_NAME: results}

    else:
        processed_results = load_processed(processed_file)

        # Compare several parameter choices without repeating histogram fits.
        config_results = compare_optimization_configs(
            processed_results,
            named_configs=OPTIMIZATION_CONFIGS,
            slm_config=SLM_CONFIG,
            slm_selected_inds=SLM_SELECTED_INDS,
        )
        plot_optimization_config_comparison(config_results)

        # Keep the automatic active result for auto-versus-manual histograms.
        automatic_results = config_results[ACTIVE_CONFIG_NAME]

        # Recompute the active configuration and then apply any manual choices.
        results = reoptimize_processed_results(
            processed_results,
            opt_config=active_config,
            slm_config=SLM_CONFIG,
            slm_selected_inds=SLM_SELECTED_INDS,
            manual_step_inds=MANUAL_STEP_INDS,
            manual_step_vals=MANUAL_STEP_VALS,
        )

    # =========================================================================
    # DIAGNOSTIC PLOTS
    # =========================================================================
    plot_summary(results)
    plot_all_nv_scatters(results)
    plot_per_nv_optimal_step_distribution(results)
    plot_optimum_metric_scatter(results)

    if "slm_mean_norm_intensity_weight_clipped" in results:
        plot_slm_weight_distribution(results)

    if PLOT_MANUAL_HISTOGRAMS and (
        MANUAL_STEP_INDS or MANUAL_STEP_VALS
    ):
        plot_manual_override_histograms(
            raw_data,
            automatic_results,
            manual_step_inds=MANUAL_STEP_INDS,
            manual_step_vals=MANUAL_STEP_VALS,
            density=True,
        )

    if PLOT_REPRESENTATIVE_HISTOGRAMS:
        plot_representative_histograms(
            raw_data,
            results,
            density=True,
            use_population_step=False,
        )

    if SAVE_REOPTIMIZED_RESULTS:
        save_results_again(
            results,
            prefix=(
                "reoptimized_slm_"
                + ACTIVE_CONFIG_NAME
            ),
        )

    # Final values intended for the SLM/AOM implementation.
    if "slm_mean_norm_intensity_weight_clipped" in results:
        print("\n=== FINAL VALUES TO USE ===")
        print(
            "effective AOM power (uW):",
            results["slm_effective_aom_power_uW"],
        )
        print(
            "effective AOM voltage:",
            results["slm_effective_aom_voltage"],
        )
        print(
            "effective OPX step value:",
            results["slm_effective_step_value"],
        )
        print(
            "SLM intensity weights key:",
            "slm_mean_norm_intensity_weight_clipped",
        )
        print(
            "SLM field-amplitude weights key:",
            "slm_amplitude_weight",
        )

    plt.show(block=True)