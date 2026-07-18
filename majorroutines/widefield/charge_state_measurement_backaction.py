# -*- coding: utf-8 -*-
"""
Repeated NV charge-state measurement with NV- survival analysis after one adaptive initialization.

Experiment sequence for each run
--------------------------------
    1. rep 0:
       Ionize all NVs and read out.

    2. reps 1 .. num_init_reps - 1:
       Adaptively polarize only unconfirmed NVs into NV- and read out.
       Confirmed NV- sites are blocked by the DMD.

    3. rep num_init_reps:
       Immediate verification. DMD pass-all, no polarization, read out.

    4. Each later rep:
       Keep the OPX paused, block the optical path with the DMD, wait for
       readout_interval_s in Python, restore DMD pass-all, send an all-False
       target mask, and perform one readout-only repetition.

No reinitialization occurs after the immediate verification image.

Expected counts returned by base_routine
----------------------------------------
    counts[exp, nv, run, step, rep]

The default QUA sequence is the existing ``charge_state_particle_memory.py``.
It already pauses for ``_cache_target_list`` on every rep after rep 0, so an
all-False list gives a readout-only repetition.

Created July 2026.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from majorroutines.widefield.charge_state_particle_memory import (
    _append_to_file_path,
    _copy_nv_list_with_confirmation_margin,
    _dmd_block_all,
    _dmd_pass_all_block_none,
    _get_thresholds,
    _prepare_dmd_indices,
    _wait_with_progress,
    make_dmd_adaptive_charge_prep_fn,
)
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield
from utils.constants import VirtualLaserKey


# =============================================================================
# Repeated-readout callback
# =============================================================================


def make_measurement_backaction_charge_prep_fn(
    base_charge_prep_fn,
    num_nvs: int,
    use_dmd: bool,
    initial_check_rep_ind: int,
    delayed_readout_rep_inds: Sequence[int],
    readout_interval_s: float,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
    block_all_during_wait: bool = True,
    wait_status_interval_s: float = 60.0,
    verbose: bool = True,
    phase_records: Optional[List[Dict[str, Any]]] = None,
):
    """
    Add one immediate verification and repeated delayed readouts.

    This intentionally follows the working particle-memory pattern:

        - the QUA program is paused waiting for ``_cache_target_list``;
        - Python performs the wall-clock dark wait;
        - Python sends all False only after the wait is complete;
        - QUA then performs exactly one readout-only repetition.
    """

    pulse_gen = tb.get_server_pulse_gen()
    dmd = tb.get_server_dmd() if use_dmd else None

    all_false = np.zeros(num_nvs, dtype=bool).tolist()
    delayed_rep_set = {int(rep_ind) for rep_ind in delayed_readout_rep_inds}
    delayed_rep_to_measurement_ind = {
        int(rep_ind): measurement_ind
        for measurement_ind, rep_ind in enumerate(delayed_readout_rep_inds, start=1)
    }

    run_counter = {"run_ind": -1}

    def wrapped(rep_ind, nv_list, initial_states_list=None):
        rep_ind = int(rep_ind)

        if rep_ind == 0:
            run_counter["run_ind"] += 1

        run_ind = int(run_counter["run_ind"])

        # ------------------------------------------------------------------
        # Immediate verification after adaptive initialization
        # ------------------------------------------------------------------
        if rep_ind == int(initial_check_rep_ind):
            callback_t0 = time.perf_counter()

            if use_dmd:
                _dmd_pass_all_block_none(
                    dmd=dmd,
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    dmd_settle_s=dmd_settle_s,
                )

            # Release exactly one readout-only repetition.
            pulse_gen.insert_input_stream("_cache_target_list", all_false)

            callback_s = float(time.perf_counter() - callback_t0)

            if phase_records is not None:
                phase_records.append(
                    {
                        "run_ind": run_ind,
                        "rep_ind": rep_ind,
                        "measurement_ind": 0,
                        "phase": "initial_verification",
                        "requested_wait_s": 0.0,
                        "actual_wait_s": 0.0,
                        "callback_s": callback_s,
                    }
                )

            if verbose:
                print(
                    f"[measurement 0] run {run_ind}, rep {rep_ind}: "
                    "immediate verification; releasing readout"
                )

            return

        # ------------------------------------------------------------------
        # Delayed readout-only repetitions
        # ------------------------------------------------------------------
        if rep_ind in delayed_rep_set:
            measurement_ind = delayed_rep_to_measurement_ind[rep_ind]

            previous_state = (
                None
                if initial_states_list is None
                else np.asarray(initial_states_list, dtype=bool)
            )
            previous_nvm_count = (
                None if previous_state is None else int(np.sum(previous_state))
            )

            # IMPORTANT: do not insert the input stream yet. The QUA program
            # remains paused while Python performs the dark wait.
            if use_dmd and block_all_during_wait:
                _dmd_block_all(
                    dmd=dmd,
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    dmd_settle_s=dmd_settle_s,
                )

            if verbose:
                print(
                    f"[measurement {measurement_ind}] run {run_ind}, rep {rep_ind}: "
                    f"dark wait start; requested={float(readout_interval_s):.3f}s; "
                    f"previous NV-={previous_nvm_count}"
                )

            wait_t0 = time.perf_counter()
            actual_wait_s = _wait_with_progress(
                readout_interval_s,
                status_interval_s=wait_status_interval_s,
                verbose=verbose,
            )
            wait_callback_s = float(time.perf_counter() - wait_t0)

            # Restore the optical path before releasing QUA for readout.
            if use_dmd:
                _dmd_pass_all_block_none(
                    dmd=dmd,
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    dmd_settle_s=dmd_settle_s,
                )

            # Only now release exactly one readout-only repetition.
            pulse_gen.insert_input_stream("_cache_target_list", all_false)

            if phase_records is not None:
                phase_records.append(
                    {
                        "run_ind": run_ind,
                        "rep_ind": rep_ind,
                        "measurement_ind": measurement_ind,
                        "phase": "delayed_readout",
                        "requested_wait_s": float(readout_interval_s),
                        "actual_wait_s": float(actual_wait_s),
                        "wait_callback_s": wait_callback_s,
                        "previous_nvm_from_callback": previous_nvm_count,
                        "block_all_during_wait": bool(
                            use_dmd and block_all_during_wait
                        ),
                    }
                )

            if verbose:
                print(
                    f"[measurement {measurement_ind}] run {run_ind}, rep {rep_ind}: "
                    f"wait complete, actual={actual_wait_s:.3f}s; releasing readout"
                )

            return

        # Adaptive initialization reps use the original callback.
        return base_charge_prep_fn(rep_ind, nv_list, initial_states_list)

    return wrapped


# =============================================================================
# Analysis
# =============================================================================


def _build_actual_measurement_times_s(
    raw_data: Dict[str, Any],
    num_runs: int,
    num_measurements: int,
    nominal_measurement_times_s: np.ndarray,
):
    """
    Reconstruct cumulative dark time for each run from phase_records.

    The returned time excludes initialization and camera/readout overhead. It is
    the cumulative time intentionally spent in the dark before each measurement.
    Missing records fall back to the nominal requested interval.
    """

    nominal_measurement_times_s = np.asarray(
        nominal_measurement_times_s,
        dtype=float,
    )

    if nominal_measurement_times_s.shape != (num_measurements,):
        raise ValueError(
            "nominal_measurement_times_s has shape "
            f"{nominal_measurement_times_s.shape}; expected {(num_measurements,)}."
        )

    nominal_interval_waits_s = np.diff(
        nominal_measurement_times_s,
        prepend=nominal_measurement_times_s[0],
    )
    nominal_interval_waits_s[0] = 0.0

    interval_waits_s_by_run = np.full(
        (num_runs, num_measurements),
        np.nan,
        dtype=float,
    )
    interval_waits_s_by_run[:, 0] = 0.0

    for record in raw_data.get("phase_records", []):
        if record.get("phase") != "delayed_readout":
            continue

        try:
            run_ind = int(record["run_ind"])
            measurement_ind = int(record["measurement_ind"])
            actual_wait_s = float(
                record.get(
                    "actual_wait_s",
                    record.get("requested_wait_s", np.nan),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

        if (
            0 <= run_ind < num_runs
            and 1 <= measurement_ind < num_measurements
            and np.isfinite(actual_wait_s)
        ):
            interval_waits_s_by_run[run_ind, measurement_ind] = actual_wait_s

    # Use the requested timing only when an old/incomplete dataset does not
    # contain a valid timing record for a particular run and measurement.
    for measurement_ind in range(1, num_measurements):
        missing = ~np.isfinite(interval_waits_s_by_run[:, measurement_ind])
        interval_waits_s_by_run[missing, measurement_ind] = (
            nominal_interval_waits_s[measurement_ind]
        )

    actual_measurement_times_s_by_run = np.cumsum(
        interval_waits_s_by_run,
        axis=1,
    )

    return interval_waits_s_by_run, actual_measurement_times_s_by_run


def analyze_measurement_backaction(
    raw_data: Dict[str, Any],
    initial_margin_counts: float = 1.0,
    final_margin_counts: float = 1.0,
) -> Dict[str, Any]:
    """
    Analyze NV- survival versus measurement number and elapsed dark time.

    Survival is defined from the current charge-state measurement: an NV
    verified as NV- at measurement 0 is counted as surviving whenever it is
    classified as NV- at the present readout. A site may be NV0 at one readout
    and return to NV- later.

    The analysis reports the current NV- survival fraction, the current
    confident NV0 population, interval transition statistics, and elapsed dark
    times. Only the state measured at each readout is used; no irreversible-loss history is constructed.
    """

    counts_all = np.asarray(raw_data["counts"], dtype=float)
    if counts_all.ndim != 5:
        raise ValueError(
            "Expected counts[exp, nv, run, step, rep]; "
            f"got shape {counts_all.shape}."
        )

    counts = counts_all[0, :, :, 0, :]  # [nv, run, rep]
    num_nvs, num_runs, num_reps = counts.shape

    measurement_rep_inds = np.asarray(
        raw_data["measurement_rep_inds"],
        dtype=int,
    )
    nominal_measurement_times_s = np.asarray(
        raw_data["measurement_times_s"],
        dtype=float,
    )

    if np.any(measurement_rep_inds < 0) or np.any(
        measurement_rep_inds >= num_reps
    ):
        raise IndexError("measurement_rep_inds contains an invalid repetition index.")

    measurement_counts = counts[:, :, measurement_rep_inds]
    num_measurements = measurement_counts.shape[2]

    if nominal_measurement_times_s.shape != (num_measurements,):
        raise ValueError(
            "measurement_times_s and measurement_rep_inds must have equal length."
        )

    measurement_numbers = np.arange(num_measurements, dtype=int)
    interval_numbers = measurement_numbers[1:]

    (
        actual_interval_waits_s_by_run,
        actual_measurement_times_s_by_run,
    ) = _build_actual_measurement_times_s(
        raw_data,
        num_runs=num_runs,
        num_measurements=num_measurements,
        nominal_measurement_times_s=nominal_measurement_times_s,
    )

    mean_actual_measurement_times_s = np.nanmean(
        actual_measurement_times_s_by_run,
        axis=0,
    )
    std_actual_measurement_times_s = np.nanstd(
        actual_measurement_times_s_by_run,
        axis=0,
    )

    thresholds = np.asarray(raw_data["analysis_thresholds"], dtype=float)
    if thresholds.shape != (num_nvs,):
        raise ValueError(
            f"Threshold shape {thresholds.shape} does not match {(num_nvs,)}."
        )

    threshold_3d = thresholds[:, None, None]
    initial_counts = measurement_counts[:, :, 0]

    initial_verified_nvm = initial_counts > (
        thresholds[:, None] + float(initial_margin_counts)
    )
    nvm_mask = measurement_counts > threshold_3d
    confident_nv0_mask = measurement_counts <= (
        threshold_3d - float(final_margin_counts)
    )
    ambiguous_mask = (~nvm_mask) & (~confident_nv0_mask)

    # Current-state state at each measurement. No history is imposed.
    survivor_mask = initial_verified_nvm[:, :, None] & nvm_mask
    current_confident_nv0_mask = (
        initial_verified_nvm[:, :, None] & confident_nv0_mask
    )

    num_initial_by_run = np.sum(initial_verified_nvm, axis=0)
    num_surviving_by_run = np.sum(survivor_mask, axis=0)
    num_current_confident_nv0_by_run = np.sum(
        current_confident_nv0_mask,
        axis=0,
    )

    survival_by_run = np.full(
        (num_runs, num_measurements),
        np.nan,
        dtype=float,
    )
    current_confident_nv0_fraction_by_run = np.full(
        (num_runs, num_measurements),
        np.nan,
        dtype=float,
    )

    good_runs = num_initial_by_run > 0
    denominators = num_initial_by_run[good_runs, None]

    survival_by_run[good_runs] = (
        num_surviving_by_run[good_runs] / denominators
    )
    current_confident_nv0_fraction_by_run[good_runs] = (
        num_current_confident_nv0_by_run[good_runs] / denominators
    )

    # A transition requires NV- at the previous measurement and confidently
    # NV0 at the next measurement. A later recovery is allowed.
    previous_nvm = nvm_mask[:, :, :-1]
    next_confident_nv0 = confident_nv0_mask[:, :, 1:]
    transition_mask = (
        initial_verified_nvm[:, :, None]
        & previous_nvm
        & next_confident_nv0
    )
    eligible_interval_mask = initial_verified_nvm[:, :, None] & previous_nvm

    transitions_by_run = np.sum(transition_mask, axis=0)
    eligible_by_run = np.sum(eligible_interval_mask, axis=0)

    transition_probability_by_run = np.full(
        (num_runs, num_measurements - 1),
        np.nan,
        dtype=float,
    )
    good_intervals = eligible_by_run > 0
    transition_probability_by_run[good_intervals] = (
        transitions_by_run[good_intervals] / eligible_by_run[good_intervals]
    )

    total_transitions = int(np.sum(transition_mask))
    total_eligible_intervals = int(np.sum(eligible_interval_mask))
    aggregate_transition_probability = (
        float(total_transitions / total_eligible_intervals)
        if total_eligible_intervals > 0
        else np.nan
    )

    mean_survival = np.nanmean(
        survival_by_run,
        axis=0,
    )
    std_survival = np.nanstd(
        survival_by_run,
        axis=0,
    )

    summary = {
        "analysis_type": "repeated_readout_charge_survival",
        "survival_definition": "current_charge_state",
        "num_nvs": int(num_nvs),
        "num_runs": int(num_runs),
        "num_measurements": int(num_measurements),
        "measurement_numbers": measurement_numbers.tolist(),
        "interval_numbers": interval_numbers.tolist(),
        "measurement_rep_inds": measurement_rep_inds.tolist(),
        "measurement_times_s": nominal_measurement_times_s.tolist(),
        "nominal_measurement_times_s": nominal_measurement_times_s.tolist(),
        "actual_interval_waits_s_by_run": actual_interval_waits_s_by_run.tolist(),
        "actual_measurement_times_s_by_run": (
            actual_measurement_times_s_by_run.tolist()
        ),
        "mean_actual_measurement_times_s": (
            mean_actual_measurement_times_s.tolist()
        ),
        "std_actual_measurement_times_s": (
            std_actual_measurement_times_s.tolist()
        ),
        "initial_margin_counts": float(initial_margin_counts),
        "final_margin_counts": float(final_margin_counts),
        "initial_verified_nvm_mask": initial_verified_nvm.tolist(),
        "nvm_mask": nvm_mask.tolist(),
        "confident_nv0_mask": confident_nv0_mask.tolist(),
        "ambiguous_mask": ambiguous_mask.tolist(),
        "survivor_mask": survivor_mask.tolist(),
        "current_confident_nv0_mask": current_confident_nv0_mask.tolist(),
        "transition_nvm_to_nv0_mask": transition_mask.tolist(),
        "num_initial_nvm_by_run": num_initial_by_run.tolist(),
        "num_current-state_nvm_by_run": num_surviving_by_run.tolist(),
        "num_current_confident_nv0_by_run": (
            num_current_confident_nv0_by_run.tolist()
        ),
        "survival_by_run": survival_by_run.tolist(),
        "current_confident_nv0_fraction_by_run": (
            current_confident_nv0_fraction_by_run.tolist()
        ),
        "transitions_by_run": transitions_by_run.tolist(),
        "eligible_by_run": eligible_by_run.tolist(),
        "transition_probability_by_run": transition_probability_by_run.tolist(),
        "mean_initial_nvm": float(np.mean(num_initial_by_run)),
        "std_initial_nvm": float(np.std(num_initial_by_run)),
        "mean_initial_nvm_fraction": float(
            np.mean(num_initial_by_run) / num_nvs
        ),
        "std_initial_nvm_fraction": float(
            np.std(num_initial_by_run / float(num_nvs))
        ),
        "mean_survival": mean_survival.tolist(),
        "std_survival": std_survival.tolist(),
        "mean_current_confident_nv0": np.mean(
            num_current_confident_nv0_by_run,
            axis=0,
        ).tolist(),
        "std_current_confident_nv0": np.std(
            num_current_confident_nv0_by_run,
            axis=0,
        ).tolist(),
        "mean_current_confident_nv0_fraction": np.nanmean(
            current_confident_nv0_fraction_by_run,
            axis=0,
        ).tolist(),
        "mean_transitions_by_interval": np.mean(
            transitions_by_run,
            axis=0,
        ).tolist(),
        "std_transitions_by_interval": np.std(
            transitions_by_run,
            axis=0,
        ).tolist(),
        "mean_transition_probability_by_interval": np.nanmean(
            transition_probability_by_run,
            axis=0,
        ).tolist(),
        "std_transition_probability_by_interval": np.nanstd(
            transition_probability_by_run,
            axis=0,
        ).tolist(),
        "aggregate_transition_probability": aggregate_transition_probability,
        "total_transitions": total_transitions,
        "total_eligible_intervals": total_eligible_intervals,
    }

    try:
        summary["fit_results"] = fit_measurement_backaction_curves(summary)
    except Exception as exc:
        summary["fit_results"] = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    print("\n=== Measurement-backaction analysis ===")
    print("survival definition: current measured charge state")
    print(
        "mean initially verified NV-:",
        f"{summary['mean_initial_nvm']:.2f} +/- "
        f"{summary['std_initial_nvm']:.2f} out of {num_nvs} "
        f"({100.0 * summary['mean_initial_nvm_fraction']:.2f}%)",
    )
    print(
        "final NV- survival:",
        summary["mean_survival"][-1],
    )
    print(
        "final current confident NV0 count:",
        summary["mean_current_confident_nv0"][-1],
    )
    print(
        "aggregate NV- -> NV0 probability per interval:",
        aggregate_transition_probability,
    )
    print(
        "mean actual cumulative dark time (s):",
        summary["mean_actual_measurement_times_s"][-1],
    )

    fit_results = summary.get("fit_results", {})
    if fit_results.get("success", False):
        meas_fit = fit_results["survival_vs_measurement"]
        time_fit = fit_results["survival_vs_time"]

        print("\n--- NV- survival fit parameters ---")
        print(
            "NV- survival vs measurement number: "
            f"N_1/e={meas_fit['scale']:.4g} ± {meas_fit['scale_stderr']:.2g}, "
            f"residual={meas_fit['plateau']:.4g} ± "
            f"{meas_fit['plateau_stderr']:.2g}"
        )
        print(
            "effective NV- -> NV0 probability/readout: "
            f"{meas_fit['loss_probability_per_measurement']:.4g} ± "
            f"{meas_fit['loss_probability_per_measurement_stderr']:.2g}"
        )
        print(
            "effective NV0 -> NV- recovery probability/readout: "
            f"{meas_fit['recovery_probability_per_measurement']:.4g} ± "
            f"{meas_fit['recovery_probability_per_measurement_stderr']:.2g}"
        )
        print(
            "NV- survival vs elapsed dark time: "
            f"tau={time_fit['scale']:.4g} ± {time_fit['scale_stderr']:.2g} s, "
            f"residual={time_fit['plateau']:.4g} ± "
            f"{time_fit['plateau_stderr']:.2g}"
        )
    else:
        print("fit failed:", fit_results.get("error", "unknown error"))

    return summary

# =============================================================================
# Fitting
# =============================================================================


def _survival_with_plateau(x, plateau, scale):
    """S(x) = plateau + (1 - plateau) exp(-x / scale)."""

    x = np.asarray(x, dtype=float)
    return plateau + (1.0 - plateau) * np.exp(-x / scale)



def _fit_two_parameter_curve(
    x,
    y,
    model,
    plateau_bounds,
    scale_name: str,
):
    """Fit a two-parameter exponential model and return errors/diagnostics."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0)
    x = x[valid]
    y = y[valid]

    if x.size < 4 or np.unique(x).size < 4:
        raise ValueError("At least four finite, distinct points are required.")

    x_span = float(np.max(x) - np.min(x))
    positive_x = x[x > 0.0]
    min_positive_x = float(np.min(positive_x)) if positive_x.size else 1.0
    scale_guess = max(x_span / 2.0, min_positive_x)

    plateau_low, plateau_high = map(float, plateau_bounds)
    plateau_guess = float(np.clip(y[-1], plateau_low + 1e-6, plateau_high - 1e-6))

    popt, pcov = curve_fit(
        model,
        x,
        y,
        p0=[plateau_guess, scale_guess],
        bounds=(
            [plateau_low, max(1e-12, min_positive_x * 1e-6)],
            [plateau_high, np.inf],
        ),
        maxfev=50000,
    )

    plateau, scale = map(float, popt)
    stderr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    plateau_stderr, scale_stderr = map(float, stderr)

    fitted = model(x, plateau, scale)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    dof = max(0, int(x.size - 2))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    dense_x = np.linspace(float(np.min(x)), float(np.max(x)), 500)
    dense_y = model(dense_x, plateau, scale)

    result = {
        "success": True,
        "model": model.__name__,
        "plateau": plateau,
        "plateau_stderr": plateau_stderr,
        "scale": scale,
        "scale_stderr": scale_stderr,
        "scale_name": str(scale_name),
        "half_scale": float(np.log(2.0) * scale),
        "half_scale_stderr": float(np.log(2.0) * scale_stderr),
        "r_squared": r_squared,
        "rmse": rmse,
        "degrees_of_freedom": dof,
        "parameter_covariance": pcov.tolist(),
        "fit_x": dense_x.tolist(),
        "fit_y": dense_y.tolist(),
    }
    return result


def fit_measurement_backaction_curves(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fit NV- survival versus readout number and elapsed dark time.

    NV- survival models
    -----------------------------
        S(n) = S_inf + (1-S_inf) exp(-n/N_c)
        S(t) = S_inf + (1-S_inf) exp(-t/tau)

    The plateau S_inf permits a nonzero steady NV- population and is
    consistent with effective NV- -> NV0 loss and NV0 -> NV- recovery.
    """

    measurement_numbers = np.asarray(analysis["measurement_numbers"], dtype=float)
    times_s = np.asarray(analysis["mean_actual_measurement_times_s"], dtype=float)
    survival = np.asarray(
        analysis["mean_survival"],
        dtype=float,
    )

    survival_meas = _fit_two_parameter_curve(
        measurement_numbers,
        survival,
        _survival_with_plateau,
        plateau_bounds=(0.0, 0.999999),
        scale_name="characteristic_measurement_count",
    )
    survival_time = _fit_two_parameter_curve(
        times_s,
        survival,
        _survival_with_plateau,
        plateau_bounds=(0.0, 0.999999),
        scale_name="time_constant_s",
    )

    # Two-state interpretation of the measurement-domain relaxation.
    plateau = survival_meas["plateau"]
    n_scale = survival_meas["scale"]
    pcov_meas = np.asarray(survival_meas["parameter_covariance"], dtype=float)

    relaxation_probability = float(1.0 - np.exp(-1.0 / n_scale))
    d_relaxation_d_scale = float(-np.exp(-1.0 / n_scale) / n_scale**2)

    loss_probability = float((1.0 - plateau) * relaxation_probability)
    recovery_probability = float(plateau * relaxation_probability)

    loss_gradient = np.asarray(
        [
            -relaxation_probability,
            (1.0 - plateau) * d_relaxation_d_scale,
        ],
        dtype=float,
    )
    recovery_gradient = np.asarray(
        [
            relaxation_probability,
            plateau * d_relaxation_d_scale,
        ],
        dtype=float,
    )
    relaxation_gradient = np.asarray(
        [0.0, d_relaxation_d_scale],
        dtype=float,
    )

    def propagated_stderr(gradient, covariance):
        variance = float(gradient @ covariance @ gradient.T)
        return float(np.sqrt(max(0.0, variance)))

    survival_meas.update(
        {
            "residual_nvm_fraction": plateau,
            "residual_nvm_fraction_stderr": survival_meas["plateau_stderr"],
            "relaxation_probability_per_measurement": relaxation_probability,
            "relaxation_probability_per_measurement_stderr": propagated_stderr(
                relaxation_gradient,
                pcov_meas,
            ),
            "loss_probability_per_measurement": loss_probability,
            "loss_probability_per_measurement_stderr": propagated_stderr(
                loss_gradient,
                pcov_meas,
            ),
            "recovery_probability_per_measurement": recovery_probability,
            "recovery_probability_per_measurement_stderr": propagated_stderr(
                recovery_gradient,
                pcov_meas,
            ),
        }
    )

    # Two-state interpretation of the time-domain relaxation.
    plateau_t = survival_time["plateau"]
    tau_s = survival_time["scale"]
    pcov_time = np.asarray(survival_time["parameter_covariance"], dtype=float)
    total_rate_per_s = float(1.0 / tau_s)
    d_rate_d_tau = float(-1.0 / tau_s**2)

    loss_rate_per_s = float((1.0 - plateau_t) * total_rate_per_s)
    recovery_rate_per_s = float(plateau_t * total_rate_per_s)

    loss_rate_gradient = np.asarray(
        [-total_rate_per_s, (1.0 - plateau_t) * d_rate_d_tau],
        dtype=float,
    )
    recovery_rate_gradient = np.asarray(
        [total_rate_per_s, plateau_t * d_rate_d_tau],
        dtype=float,
    )
    total_rate_gradient = np.asarray([0.0, d_rate_d_tau], dtype=float)

    survival_time.update(
        {
            "residual_nvm_fraction": plateau_t,
            "residual_nvm_fraction_stderr": survival_time["plateau_stderr"],
            "total_relaxation_rate_per_s": total_rate_per_s,
            "total_relaxation_rate_per_s_stderr": propagated_stderr(
                total_rate_gradient,
                pcov_time,
            ),
            "loss_rate_per_s": loss_rate_per_s,
            "loss_rate_per_s_stderr": propagated_stderr(
                loss_rate_gradient,
                pcov_time,
            ),
            "recovery_rate_per_s": recovery_rate_per_s,
            "recovery_rate_per_s_stderr": propagated_stderr(
                recovery_rate_gradient,
                pcov_time,
            ),
            "loss_rate_per_min": float(60.0 * loss_rate_per_s),
            "recovery_rate_per_min": float(60.0 * recovery_rate_per_s),
            "time_constant_min": float(tau_s / 60.0),
            "time_constant_min_stderr": float(
                survival_time["scale_stderr"] / 60.0
            ),
            "time_constant_h": float(tau_s / 3600.0),
            "time_constant_h_stderr": float(
                survival_time["scale_stderr"] / 3600.0
            ),
        }
    )

    return {
        "success": True,
        "survival_definition": "current_charge_state",
        "survival_vs_measurement": survival_meas,
        "survival_vs_time": survival_time,
        "identifiability_note": (
            "For a single fixed-interval dataset, elapsed time and measurement "
            "number are collinear. Use datasets with different intervals for "
            "the joint dark-time/readout-number fit."
        ),
    }

def _joint_dark_readout_model(x_data, plateau, gamma_dark_per_s, mu_readout):
    """S(t,n)=S_inf+(1-S_inf)exp(-gamma_dark*t-mu_readout*n)."""

    times_s, measurement_numbers = x_data
    return plateau + (1.0 - plateau) * np.exp(
        -gamma_dark_per_s * times_s - mu_readout * measurement_numbers
    )


def fit_joint_dark_and_readout_rates(
    analyses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Jointly fit NV- survival versus elapsed dark time and readout number.

        S(t,n) = S_inf + (1-S_inf) exp(-Gamma_dark*t - mu_readout*n)

    At least two datasets with different readout intervals are required.
    """

    times_all = []
    measurements_all = []
    survival_all = []
    interval_estimates = []

    for analysis in analyses:
        times_s = np.asarray(
            analysis["mean_actual_measurement_times_s"],
            dtype=float,
        )
        meas = np.asarray(analysis["measurement_numbers"], dtype=float)
        survival = np.asarray(
            analysis["mean_survival"],
            dtype=float,
        )

        valid = np.isfinite(times_s) & np.isfinite(meas) & np.isfinite(survival)
        times_all.append(times_s[valid])
        measurements_all.append(meas[valid])
        survival_all.append(survival[valid])

        positive = valid & (meas > 0)
        if np.any(positive):
            interval_estimates.append(
                float(np.median(times_s[positive] / meas[positive]))
            )

    if len(interval_estimates) < 2 or np.unique(
        np.round(interval_estimates, 6)
    ).size < 2:
        raise ValueError(
            "At least two datasets with different readout intervals are required "
            "to separate dark-time evolution from readout backaction."
        )

    times_all = np.concatenate(times_all)
    measurements_all = np.concatenate(measurements_all)
    survival_all = np.concatenate(survival_all)

    max_time = max(float(np.max(times_all)), 1.0)
    max_meas = max(float(np.max(measurements_all)), 1.0)
    plateau_guess = float(np.clip(np.min(survival_all), 0.0, 0.95))

    popt, pcov = curve_fit(
        _joint_dark_readout_model,
        (times_all, measurements_all),
        survival_all,
        p0=[plateau_guess, 0.5 / max_time, 0.5 / max_meas],
        bounds=([0.0, 0.0, 0.0], [0.999999, np.inf, np.inf]),
        maxfev=100000,
    )

    plateau, gamma_dark, mu_readout = map(float, popt)
    errors = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    plateau_err, gamma_err, mu_err = map(float, errors)

    fitted = _joint_dark_readout_model(
        (times_all, measurements_all),
        plateau,
        gamma_dark,
        mu_readout,
    )
    ss_res = float(np.sum((survival_all - fitted) ** 2))
    ss_tot = float(np.sum((survival_all - np.mean(survival_all)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((survival_all - fitted) ** 2)))

    # Readout contribution: two-state relaxation toward the fitted plateau.
    readout_relaxation_probability = float(1.0 - np.exp(-mu_readout))
    readout_loss_probability = float(
        (1.0 - plateau) * readout_relaxation_probability
    )
    readout_recovery_probability = float(
        plateau * readout_relaxation_probability
    )

    exp_mu = float(np.exp(-mu_readout))
    d_relax_d_mu = exp_mu

    grad_relax = np.asarray([0.0, 0.0, d_relax_d_mu])
    grad_loss = np.asarray(
        [
            -readout_relaxation_probability,
            0.0,
            (1.0 - plateau) * d_relax_d_mu,
        ]
    )
    grad_recovery = np.asarray(
        [
            readout_relaxation_probability,
            0.0,
            plateau * d_relax_d_mu,
        ]
    )

    def joint_stderr(gradient):
        variance = float(gradient @ pcov @ gradient.T)
        return float(np.sqrt(max(0.0, variance)))

    # Dark contribution: continuous two-state rates with the same plateau.
    dark_loss_rate_per_s = float((1.0 - plateau) * gamma_dark)
    dark_recovery_rate_per_s = float(plateau * gamma_dark)

    grad_dark_loss = np.asarray(
        [-gamma_dark, 1.0 - plateau, 0.0]
    )
    grad_dark_recovery = np.asarray(
        [gamma_dark, plateau, 0.0]
    )

    tau_dark_s = float(1.0 / gamma_dark) if gamma_dark > 0 else np.inf
    tau_dark_err_s = (
        float(gamma_err / gamma_dark**2) if gamma_dark > 0 else np.inf
    )

    return {
        "success": True,
        "survival_definition": "current_charge_state",
        "model": "S_inf + (1-S_inf) exp(-Gamma_dark*t - mu_readout*n)",
        "plateau": plateau,
        "plateau_stderr": plateau_err,
        "residual_nvm_fraction": plateau,
        "residual_nvm_fraction_stderr": plateau_err,
        "gamma_dark_per_s": gamma_dark,
        "gamma_dark_per_s_stderr": gamma_err,
        "gamma_dark_per_min": float(60.0 * gamma_dark),
        "gamma_dark_per_min_stderr": float(60.0 * gamma_err),
        "dark_loss_rate_per_s": dark_loss_rate_per_s,
        "dark_loss_rate_per_s_stderr": joint_stderr(grad_dark_loss),
        "dark_loss_rate_per_min": float(60.0 * dark_loss_rate_per_s),
        "dark_recovery_rate_per_s": dark_recovery_rate_per_s,
        "dark_recovery_rate_per_s_stderr": joint_stderr(grad_dark_recovery),
        "dark_recovery_rate_per_min": float(60.0 * dark_recovery_rate_per_s),
        "tau_dark_s": tau_dark_s,
        "tau_dark_s_stderr": tau_dark_err_s,
        "tau_dark_min": float(tau_dark_s / 60.0),
        "tau_dark_min_stderr": float(tau_dark_err_s / 60.0),
        "mu_readout": mu_readout,
        "mu_readout_stderr": mu_err,
        "relaxation_probability_per_readout": readout_relaxation_probability,
        "relaxation_probability_per_readout_stderr": joint_stderr(grad_relax),
        "loss_probability_per_readout": readout_loss_probability,
        "loss_probability_per_readout_stderr": joint_stderr(grad_loss),
        "recovery_probability_per_readout": readout_recovery_probability,
        "recovery_probability_per_readout_stderr": joint_stderr(grad_recovery),
        "r_squared": r_squared,
        "rmse": rmse,
        "readout_intervals_s": interval_estimates,
        "num_fitted_points": int(survival_all.size),
    }

# =============================================================================
# Plot
# =============================================================================


def _choose_time_axis(times_s: np.ndarray):
    """Choose seconds, minutes, or hours for a readable time axis."""

    times_s = np.asarray(times_s, dtype=float)
    finite = times_s[np.isfinite(times_s)]
    max_time_s = float(np.max(finite)) if finite.size else 0.0

    if max_time_s >= 7200.0:
        return times_s / 3600.0, "Cumulative dark time (h)"
    if max_time_s >= 120.0:
        return times_s / 60.0, "Cumulative dark time (min)"
    return times_s, "Cumulative dark time (s)"


def plot_measurement_backaction_summary(
    raw_data: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
):
    """
    Plot NV- survival, current confident NV0 population, and transition probability.

    Left column: measurement number.
    Right column: cumulative dark time.
    """

    if analysis is None:
        analysis = raw_data["measurement_backaction_analysis"]

    measurement_numbers = np.asarray(
        analysis["measurement_numbers"],
        dtype=float,
    )
    interval_numbers = np.asarray(
        analysis["interval_numbers"],
        dtype=float,
    )

    mean_actual_times_s = np.asarray(
        analysis["mean_actual_measurement_times_s"],
        dtype=float,
    )
    times, time_label = _choose_time_axis(mean_actual_times_s)
    interval_times = times[1:]

    survival = np.asarray(
        analysis["mean_survival"],
        dtype=float,
    )
    survival_std = np.asarray(
        analysis["std_survival"],
        dtype=float,
    )

    current_nv0 = np.asarray(
        analysis["mean_current_confident_nv0"],
        dtype=float,
    )
    current_nv0_std = np.asarray(
        analysis["std_current_confident_nv0"],
        dtype=float,
    )
    mean_transition_probability = np.asarray(
        analysis["mean_transition_probability_by_interval"],
        dtype=float,
    )
    std_transition_probability = np.asarray(
        analysis["std_transition_probability_by_interval"],
        dtype=float,
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 10),
        sharex="col",
    )

    def plot_survival(ax, x_values):
        ax.plot(
            x_values,
            survival,
            "o-",
            color=kpl.KplColors.BLUE,
            label="NV$^-$ survival",
        )
        ax.fill_between(
            x_values,
            np.maximum(0.0, survival - survival_std),
            np.minimum(1.0, survival + survival_std),
            color=kpl.KplColors.BLUE,
            alpha=0.18,
        )
        ax.set_ylabel("NV$^-$ survival fraction")
        ax.set_ylim(0.0, 1.02)
        ax.legend(fontsize=8)

    def plot_current_nv0(ax, x_values):
        """Plot the current confident NV0 population."""

        ax.plot(
            x_values,
            current_nv0,
            "o-",
            color=kpl.KplColors.RED,
            label="Current confident NV$^0$",
        )
        ax.fill_between(
            x_values,
            np.maximum(0.0, current_nv0 - current_nv0_std),
            current_nv0 + current_nv0_std,
            color=kpl.KplColors.RED,
            alpha=0.16,
        )
        ax.set_ylabel("Current NV$^0$ count", fontsize=15)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8)

    def plot_probability(ax, x_values):
        ax.plot(
            x_values,
            mean_transition_probability,
            "o-",
            color=kpl.KplColors.RED,
        )
        ax.fill_between(
            x_values,
            np.maximum(
                0.0,
                mean_transition_probability - std_transition_probability,
            ),
            mean_transition_probability + std_transition_probability,
            color=kpl.KplColors.RED,
            alpha=0.16,
        )
        ax.set_ylabel("NV$^- \\rightarrow$ NV$^0$\nprobability per interval")
        ax.set_ylim(bottom=0.0)

    plot_survival(axes[0, 0], measurement_numbers)
    plot_survival(axes[0, 1], times)
    plot_current_nv0(axes[1, 0], measurement_numbers)
    plot_current_nv0(axes[1, 1], times)

    fit_results = analysis.get("fit_results", {})
    if fit_results.get("success", False):
        fit_meas = fit_results["survival_vs_measurement"]
        fit_time = fit_results["survival_vs_time"]
        axes[0, 0].plot(
            fit_meas["fit_x"],
            fit_meas["fit_y"],
            "k--",
            linewidth=2.0,
            label=(
                f"fit: N$_c$={fit_meas['scale']:.2g}, "
                f"S$_\\infty$={fit_meas['plateau']:.3f}"
            ),
        )
        axes[0, 0].legend(fontsize=8)

        fit_time_x_s = np.asarray(fit_time["fit_x"], dtype=float)
        fit_time_x, _ = _choose_time_axis(fit_time_x_s)
        axes[0, 1].plot(
            fit_time_x,
            fit_time["fit_y"],
            "k--",
            linewidth=2.0,
            label=(
                f"fit: tau={fit_time['time_constant_min']:.2g} min, "
                f"S$_\\infty$={fit_time['plateau']:.3f}"
            ),
        )
        axes[0, 1].legend(fontsize=8)


    plot_probability(axes[2, 0], interval_numbers)
    plot_probability(axes[2, 1], interval_times)

    axes[0, 0].set_title("NV$^-$ survival vs measurement number", fontsize=15)
    axes[0, 1].set_title("NV$^-$ survival vs cumulative dark time", fontsize=15)
    axes[1, 0].set_title("Current confident NV$^0$ population", fontsize=15)
    axes[1, 1].set_title("Current confident NV$^0$ population", fontsize=15)
    axes[2, 0].set_xlabel("Measurement number", fontsize=15)
    axes[2, 1].set_xlabel(time_label)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    interval_s = float(raw_data.get("readout_interval_s", np.nan))
    total_nvs = int(analysis["num_nvs"])
    num_runs = int(analysis["num_runs"])
    initial_by_run = np.asarray(
        analysis["num_initial_nvm_by_run"],
        dtype=float,
    )
    mean_initial = float(np.nanmean(initial_by_run))
    std_initial = float(np.nanstd(initial_by_run))
    initial_percent = (
        100.0 * mean_initial / total_nvs
        if total_nvs > 0
        else np.nan
    )

    fig.suptitle(
        "NV charge-state survival under repeated readout\n"
        f"interval = {interval_s:g} s | initially verified NV$^-$ = "
        f"{mean_initial:.1f} ± {std_initial:.1f} of {total_nvs} "
        f"({initial_percent:.1f}%) | runs = {num_runs}",
        fontsize=15,
    )

    model_note = (
        r"Survival = fraction of initially verified NV$^-$ sites "
        r"classified as NV$^-$ at the current readout. "
        r"Fit: $S(x)=S_\infty+(1-S_\infty)e^{-x/x_c}$; "
        r"$S_\infty$ is the residual NV$^-$ fraction and "
        r"$x_c=N_c$ or $\tau$ is the 1/e relaxation scale."
        "\n"
        r"Second row shows the current confident NV$^0$ population; "
        r"ambiguous sites are excluded from both populations."
    )
    fig.text(
        0.5,
        0.012,
        model_note,
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.075, 1, 0.99])
    return fig

# =============================================================================
# Main experiment


def _estimate_analysis_interval_s(analysis: Dict[str, Any]) -> float:
    """Estimate the readout interval from cumulative time / measurement index."""

    times_s = np.asarray(
        analysis["mean_actual_measurement_times_s"],
        dtype=float,
    )
    measurement_numbers = np.asarray(
        analysis["measurement_numbers"],
        dtype=float,
    )

    valid = (
        np.isfinite(times_s)
        & np.isfinite(measurement_numbers)
        & (measurement_numbers > 0)
    )

    if not np.any(valid):
        return np.nan

    return float(
        np.nanmedian(
            times_s[valid] / measurement_numbers[valid]
        )
    )


def _format_interval_label(interval_s: float) -> str:
    """Create a compact label for a readout interval."""

    if not np.isfinite(interval_s):
        return "unknown interval"
    if interval_s >= 3600.0:
        return f"{interval_s / 3600.0:g} h interval"
    if interval_s >= 60.0:
        return f"{interval_s / 60.0:g} min interval"
    return f"{interval_s:g} s interval"


def plot_survival_interval_comparison(
    analyses: Sequence[Dict[str, Any]],
    labels: Optional[Sequence[str]] = None,
    joint_fit: Optional[Dict[str, Any]] = None,
    show_individual_runs: bool = False,
    time_axis_scale: str = "symlog",
):
    """
    Compare repeated-readout survival datasets in a three-panel figure.

    Layout
    ------
    Upper-left panel
        Mean NV- survival versus measurement number, with one independent
        measurement-domain fit for each readout interval.

    Lower-left panel
        The same survival data versus cumulative dark time, with one independent
        time-domain fit for each readout interval.

    Full-height right panel
        Dataset populations, independent-fit parameters, key differences, model
        definitions, and the optional joint dark/readout decomposition.

    Notes
    -----
    For one fixed-interval dataset, elapsed time and measurement number satisfy
    approximately t = n * interval. Therefore N_c and tau describe the same
    individual decay curve in different units. Multiple readout intervals are
    required to estimate separate dark-time and readout-number contributions.
    """

    if not analyses:
        raise ValueError("At least one analysis is required.")

    num_datasets = len(analyses)

    if labels is None:
        labels = [
            _format_interval_label(_estimate_analysis_interval_s(analysis))
            for analysis in analyses
        ]
    else:
        labels = list(labels)

    if len(labels) != num_datasets:
        raise ValueError(
            f"Expected {num_datasets} labels, received {len(labels)}."
        )

    valid_time_scales = {"linear", "log", "symlog"}
    if time_axis_scale not in valid_time_scales:
        raise ValueError(
            f"time_axis_scale must be one of {sorted(valid_time_scales)}."
        )

    # ------------------------------------------------------------------
    # Select one common time unit for every dataset.
    # Do not call _choose_time_axis separately for each dataset, because that
    # could put seconds and hours on the same axis.
    # ------------------------------------------------------------------
    all_times_s = np.concatenate(
        [
            np.asarray(
                analysis["mean_actual_measurement_times_s"],
                dtype=float,
            )
            for analysis in analyses
        ]
    )
    finite_times_s = all_times_s[np.isfinite(all_times_s)]
    max_time_s = float(np.max(finite_times_s)) if finite_times_s.size else 0.0

    if max_time_s >= 7200.0:
        time_divisor = 3600.0
        time_label = "Cumulative dark time (h)"
    elif max_time_s >= 120.0:
        time_divisor = 60.0
        time_label = "Cumulative dark time (min)"
    else:
        time_divisor = 1.0
        time_label = "Cumulative dark time (s)"

    all_display_times = all_times_s / time_divisor
    positive_display_times = all_display_times[
        np.isfinite(all_display_times) & (all_display_times > 0.0)
    ]
    min_positive_display_time = (
        float(np.min(positive_display_times))
        if positive_display_times.size
        else 1.0
    )

    # ------------------------------------------------------------------
    # Figure layout: two plots on the left and one full-height summary panel.
    # ------------------------------------------------------------------
    fig = plt.figure(
        figsize=(10.0, 10.0),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(2.35, 1.20),
        height_ratios=(1.0, 1.0),
        wspace=0.08,
        hspace=0.10,
    )

    ax_measurement = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[:, 1])

    parameter_rows: List[str] = []
    comparison_values: List[Dict[str, float]] = []

    for analysis, label in zip(analyses, labels):
        measurement_numbers_full = np.asarray(
            analysis["measurement_numbers"],
            dtype=float,
        )
        times_s_full = np.asarray(
            analysis["mean_actual_measurement_times_s"],
            dtype=float,
        )
        display_times_full = times_s_full / time_divisor
        survival_full = np.asarray(
            analysis["mean_survival"],
            dtype=float,
        )
        survival_std_full = np.asarray(
            analysis["std_survival"],
            dtype=float,
        )
        survival_by_run = np.asarray(
            analysis["survival_by_run"],
            dtype=float,
        )
        times_s_by_run = np.asarray(
            analysis["actual_measurement_times_s_by_run"],
            dtype=float,
        )

        valid_common = (
            np.isfinite(measurement_numbers_full)
            & np.isfinite(times_s_full)
            & np.isfinite(display_times_full)
            & np.isfinite(survival_full)
            & np.isfinite(survival_std_full)
        )

        if not np.any(valid_common):
            parameter_rows.extend(
                [
                    label,
                    "  no finite survival points",
                    "",
                ]
            )
            continue

        measurement_numbers = measurement_numbers_full[valid_common]
        display_times = display_times_full[valid_common]
        survival = survival_full[valid_common]
        survival_std = survival_std_full[valid_common]

        # Create one color from the upper plot and reuse it everywhere.
        data_line = ax_measurement.plot(
            measurement_numbers,
            survival,
            "o",
            markersize=4.8,
            label=f"{label}: data",
            zorder=3,
        )[0]
        dataset_color = data_line.get_color()

        ax_measurement.fill_between(
            measurement_numbers,
            np.maximum(0.0, survival - survival_std),
            np.minimum(1.0, survival + survival_std),
            color=dataset_color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        # On a true logarithmic axis, t=0 cannot be shown. For symlog and
        # linear axes, retain the immediate-verification point.
        if time_axis_scale == "log":
            time_plot_mask = display_times > 0.0
        else:
            time_plot_mask = np.ones(display_times.shape, dtype=bool)

        ax_time.plot(
            display_times[time_plot_mask],
            survival[time_plot_mask],
            "o",
            markersize=4.8,
            color=dataset_color,
            label=f"{label}: data",
            zorder=3,
        )
        ax_time.fill_between(
            display_times[time_plot_mask],
            np.maximum(
                0.0,
                survival[time_plot_mask] - survival_std[time_plot_mask],
            ),
            np.minimum(
                1.0,
                survival[time_plot_mask] + survival_std[time_plot_mask],
            ),
            color=dataset_color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        if show_individual_runs:
            for run_ind, run_survival in enumerate(survival_by_run):
                if run_ind >= times_s_by_run.shape[0]:
                    break

                run_display_times = times_s_by_run[run_ind] / time_divisor
                run_valid = (
                    np.isfinite(measurement_numbers_full)
                    & np.isfinite(run_display_times)
                    & np.isfinite(run_survival)
                )

                if not np.any(run_valid):
                    continue

                ax_measurement.plot(
                    measurement_numbers_full[run_valid],
                    run_survival[run_valid],
                    "-",
                    linewidth=0.8,
                    alpha=0.14,
                    color=dataset_color,
                    zorder=0,
                )

                run_time_valid = run_valid.copy()
                if time_axis_scale == "log":
                    run_time_valid &= run_display_times > 0.0

                ax_time.plot(
                    run_display_times[run_time_valid],
                    run_survival[run_time_valid],
                    "-",
                    linewidth=0.8,
                    alpha=0.14,
                    color=dataset_color,
                    zorder=0,
                )

        fit_results = analysis.get("fit_results", {})
        if not fit_results.get("success", False):
            parameter_rows.extend(
                [
                    label,
                    (
                        "  fit unavailable: "
                        f"{fit_results.get('error', 'unknown error')}"
                    ),
                    "",
                ]
            )
            continue

        fit_meas = fit_results["survival_vs_measurement"]
        fit_time = fit_results["survival_vs_time"]

        fit_meas_x = np.asarray(fit_meas["fit_x"], dtype=float)
        fit_meas_y = np.asarray(fit_meas["fit_y"], dtype=float)
        fit_time_x = (
            np.asarray(fit_time["fit_x"], dtype=float) / time_divisor
        )
        fit_time_y = np.asarray(fit_time["fit_y"], dtype=float)

        ax_measurement.plot(
            fit_meas_x,
            fit_meas_y,
            "-",
            linewidth=2.4,
            color=dataset_color,
            label=f"{label}: separate fit",
            zorder=2,
        )

        if time_axis_scale == "log":
            fit_time_mask = fit_time_x > 0.0
        else:
            fit_time_mask = np.ones(fit_time_x.shape, dtype=bool)

        ax_time.plot(
            fit_time_x[fit_time_mask],
            fit_time_y[fit_time_mask],
            "-",
            linewidth=2.4,
            color=dataset_color,
            label=f"{label}: separate fit",
            zorder=2,
        )

        total_nvs = int(analysis["num_nvs"])
        num_runs = int(analysis["num_runs"])
        initial_by_run = np.asarray(
            analysis["num_initial_nvm_by_run"],
            dtype=float,
        )
        mean_initial = float(np.nanmean(initial_by_run))
        std_initial = float(np.nanstd(initial_by_run))
        initial_percent = (
            100.0 * mean_initial / total_nvs
            if total_nvs > 0
            else np.nan
        )

        interval_s = _estimate_analysis_interval_s(analysis)
        final_survival = float(survival[-1])
        direct_transition = float(
            analysis["aggregate_transition_probability"]
        )

        n_c = float(fit_meas["scale"])
        n_c_err = float(fit_meas["scale_stderr"])
        plateau = float(fit_meas["plateau"])
        plateau_err = float(fit_meas["plateau_stderr"])
        relaxation = float(
            fit_meas["relaxation_probability_per_measurement"]
        )
        relaxation_err = float(
            fit_meas["relaxation_probability_per_measurement_stderr"]
        )
        fit_meas_r2 = float(fit_meas["r_squared"])

        tau_s = float(fit_time["scale"])
        tau_s_err = float(fit_time["scale_stderr"])
        fit_time_r2 = float(fit_time["r_squared"])

        if tau_s >= 7200.0:
            tau_value = tau_s / 3600.0
            tau_error = tau_s_err / 3600.0
            tau_unit = "h"
        elif tau_s >= 120.0:
            tau_value = tau_s / 60.0
            tau_error = tau_s_err / 60.0
            tau_unit = "min"
        else:
            tau_value = tau_s
            tau_error = tau_s_err
            tau_unit = "s"

        parameter_rows.extend(
            [
                label,
                (
                    f"  initial NV-: {mean_initial:.1f} ± {std_initial:.1f} "
                    f"of {total_nvs} ({initial_percent:.1f}%)"
                ),
                f"  runs: {num_runs}; interval: {interval_s:g} s",
                f"  N_c: {n_c:.3g} ± {n_c_err:.2g} readouts",
                f"  tau: {tau_value:.3g} ± {tau_error:.2g} {tau_unit}",
                (
                    f"  residual S_inf: {plateau:.4f} ± "
                    f"{plateau_err:.2g}"
                ),
                (
                    f"  relaxation/readout: "
                    f"{100.0 * relaxation:.2f} ± "
                    f"{100.0 * relaxation_err:.2f}%"
                ),
                (
                    f"  counted NV-→NV0/interval: "
                    f"{100.0 * direct_transition:.2f}%"
                ),
                f"  final survival: {100.0 * final_survival:.2f}%",
                (
                    f"  fit R²: measurement={fit_meas_r2:.4f}, "
                    f"time={fit_time_r2:.4f}"
                ),
                "",
            ]
        )

        comparison_values.append(
            {
                "label": label,
                "n_c": n_c,
                "plateau": plateau,
                "relaxation": relaxation,
                "direct_transition": direct_transition,
                "tau_s": tau_s,
            }
        )

    # ------------------------------------------------------------------
    # Upper-left panel: comparison versus measurement number.
    # ------------------------------------------------------------------
    ax_measurement.set_title(
        "NV$^-$ survival versus measurement number",
        fontsize=16,
    )
    ax_measurement.set_xlabel("Measurement number", fontsize=14)
    ax_measurement.set_ylabel("NV$^-$ survival fraction", fontsize=14)
    ax_measurement.set_ylim(0.0, 1.02)
    ax_measurement.grid(True, alpha=0.3)
    ax_measurement.legend(
        fontsize=9,
        ncol=2,
        loc="upper right",
    )
    ax_measurement.tick_params(labelsize=11)
    ax_measurement.set_xscale("log")
    # ------------------------------------------------------------------
    # Lower-left panel: comparison versus cumulative dark time.
    # ------------------------------------------------------------------
    ax_time.set_title(
        "NV$^-$ survival versus cumulative dark time",
        fontsize=16,
    )
    ax_time.set_xlabel(time_label, fontsize=14)
    ax_time.set_ylabel("NV$^-$ survival fraction", fontsize=14)
    ax_time.set_ylim(0.0, 1.02)

    if time_axis_scale == "symlog":
        # Keep t=0 visible and use logarithmic spacing above the shortest
        # positive measured wait.
        ax_time.set_xscale(
            "symlog",
            linthresh=min_positive_display_time,
            linscale=1.0,
        )
    elif time_axis_scale == "log":
        ax_time.set_xscale("log")
        ax_time.set_xlim(
            left=0.0,
        )

    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend(
        fontsize=9,
        ncol=2,
        loc="upper right",
    )
    ax_time.tick_params(labelsize=11)

    # ------------------------------------------------------------------
    # Full-height right panel: fitting and dataset summary.
    # ------------------------------------------------------------------
    info_lines = [
        "SEPARATE DATASET FITS",
        "",
        *parameter_rows,
    ]

    # if len(comparison_values) == 2:
    #     first, second = comparison_values
    #     info_lines.extend(
    #         [
    #             "KEY DIFFERENCES",
    #             (
    #                 f"  ΔN_c ({second['label']} - {first['label']}): "
    #                 f"{second['n_c'] - first['n_c']:+.3g} readouts"
    #             ),
    #             (
    #                 f"  Δtau: "
    #                 f"{(second['tau_s'] - first['tau_s']) / 60.0:+.3g} min"
    #             ),
    #             (
    #                 f"  ΔS_inf: "
    #                 f"{second['plateau'] - first['plateau']:+.4f}"
    #             ),
    #             (
    #                 f"  Δrelaxation/readout: "
    #                 f"{100.0 * (second['relaxation'] - first['relaxation']):+.2f}%"
    #             ),
    #             (
    #                 f"  Δcounted NV-→NV0/interval: "
    #                 f"{100.0 * (second['direct_transition'] - first['direct_transition']):+.2f}%"
    #             ),
    #             "",
    #         ]
    #     )

    # if joint_fit is not None and joint_fit.get("success", False):
    #     info_lines.extend(
    #         [
    #             "JOINT DARK/READOUT DECOMPOSITION",
    #             "  supplementary shared-parameter fit",
    #             (
    #                 f"  tau_dark: {joint_fit['tau_dark_min']:.3g} ± "
    #                 f"{joint_fit['tau_dark_min_stderr']:.2g} min"
    #             ),
    #             (
    #                 f"  gamma_dark: "
    #                 f"{joint_fit['gamma_dark_per_min']:.4g} ± "
    #                 f"{joint_fit['gamma_dark_per_min_stderr']:.2g} min^-1"
    #             ),
    #             (
    #                 f"  mu_readout: {joint_fit['mu_readout']:.4g} ± "
    #                 f"{joint_fit['mu_readout_stderr']:.2g}"
    #             ),
    #             (
    #                 f"  relaxation/readout: "
    #                 f"{100.0 * joint_fit['relaxation_probability_per_readout']:.2f} ± "
    #                 f"{100.0 * joint_fit['relaxation_probability_per_readout_stderr']:.2f}%"
    #             ),
    #             (
    #                 f"  shared residual S_inf: "
    #                 f"{joint_fit['residual_nvm_fraction']:.4f} ± "
    #                 f"{joint_fit['residual_nvm_fraction_stderr']:.2g}"
    #             ),
    #             f"  joint R²: {joint_fit['r_squared']:.4f}",
    #             "",
    #         ]
    #     )

    info_lines.extend(
        [
            "MODELS",
            r"  $S(n)=S_\infty+(1-S_\infty)e^{-n/N_c}$",
            r"  $S(t)=S_\infty+(1-S_\infty)e^{-t/\tau}$",
            r"  $N_c$: 1/e readout-number scale",
            r"  $\tau$: 1/e elapsed-time scale",
            r"  $S_\infty$: residual measured NV$^-$ fraction",
            "",
            # "Survival uses the current measured charge state.",
            # "A site may return to NV- after an earlier NV0 result.",
        ]
    )

    ax_summary.set_facecolor("0.975")
    ax_summary.set_xticks([])
    ax_summary.set_yticks([])
    ax_summary.set_xlim(0.0, 1.0)
    ax_summary.set_ylim(0.0, 1.0)

    for spine in ax_summary.spines.values():
        spine.set_visible(True)
        spine.set_color("0.80")
        spine.set_linewidth(1.0)

    # ax_summary.set_title(
    #     "Fit and dataset summary",
    #     fontsize=16,
    #     loc="left",
    #     pad=12,
    # )
    ax_summary.text(
        0.035,
        0.98,
        "\n".join(info_lines),
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=9.25,
        linespacing=1.18,
    )

    unique_total_nvs = sorted(
        {int(analysis["num_nvs"]) for analysis in analyses}
    )
    if len(unique_total_nvs) == 1:
        tracked_text = f"{unique_total_nvs[0]} tracked NV sites"
    else:
        tracked_text = "tracked NV sites: " + ", ".join(
            str(value) for value in unique_total_nvs
        )

    fig.suptitle(
        "NV$^-$ charge-state survival across readout intervals\n"
        f"{tracked_text}; points are means across runs",
        fontsize=17,
    )

    return fig


# =============================================================================
# Main experiment
# =============================================================================


def main(
    nv_list,
    num_init_reps: int = 10,
    num_delayed_readouts: int = 100,
    readout_interval_s: float = 5.0,
    num_runs: int = 5,
    mode: str = "dmd_block_confirmed",
    dmd_indices: Optional[Sequence[int]] = None,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
    confirm_margin_counts: float = 1.0,
    initial_analysis_margin_counts: float = 1.0,
    final_analysis_margin_counts: float = 1.0,
    block_all_during_wait: bool = True,
    wait_status_interval_s: float = 60.0,
    save_images: bool = True,
    save_images_avg_reps: bool = False,
    save_data: bool = True,
    save_fig: bool = True,
    reset_dmd_on_exit: bool = True,
    verbose: bool = True,
    seq_file: str = "charge_state_particle_memory.py",
) -> Dict[str, Any]:
    """Initialize once, then repeatedly measure without reinitialization."""

    if mode not in ("old", "dmd_block_confirmed"):
        raise ValueError("mode must be 'old' or 'dmd_block_confirmed'.")
    if int(num_init_reps) < 2:
        raise ValueError("num_init_reps must be at least 2.")
    if int(num_delayed_readouts) < 1:
        raise ValueError("num_delayed_readouts must be positive.")
    if float(readout_interval_s) < 0:
        raise ValueError("readout_interval_s cannot be negative.")
    if int(num_runs) < 1:
        raise ValueError("num_runs must be positive.")

    num_init_reps = int(num_init_reps)
    num_delayed_readouts = int(num_delayed_readouts)
    readout_interval_s = float(readout_interval_s)
    num_runs = int(num_runs)

    num_steps = 1
    num_nvs = len(nv_list)

    analysis_thresholds = _get_thresholds(nv_list)
    nv_run_list = _copy_nv_list_with_confirmation_margin(
        nv_list,
        confirm_margin_counts=confirm_margin_counts,
    )
    feedback_thresholds = _get_thresholds(nv_run_list)
    dmd_indices_arr = _prepare_dmd_indices(num_nvs, dmd_indices)

    initial_check_rep_ind = num_init_reps
    delayed_readout_rep_inds = np.arange(
        initial_check_rep_ind + 1,
        initial_check_rep_ind + 1 + num_delayed_readouts,
        dtype=int,
    )
    measurement_rep_inds = np.concatenate(
        ([initial_check_rep_ind], delayed_readout_rep_inds)
    ).astype(int)
    measurement_times_s = (
        np.arange(num_delayed_readouts + 1, dtype=float) * readout_interval_s
    )
    num_reps_total = int(delayed_readout_rep_inds[-1] + 1)

    expected_dark_wait_s_per_run = float(num_delayed_readouts * readout_interval_s)
    expected_dark_wait_s_total = float(num_runs * expected_dark_wait_s_per_run)

    print("\n=== Repeated measurement-backaction experiment ===")
    print("mode:", mode)
    print("num NVs:", num_nvs)
    print("initialization reps:", num_init_reps)
    print("immediate verification rep:", initial_check_rep_ind)
    print("delayed readout reps:", delayed_readout_rep_inds.tolist())
    print("readout interval (s):", readout_interval_s)
    print("delayed readouts per run:", num_delayed_readouts)
    print("runs:", num_runs)
    print("expected dark wait per run (s):", expected_dark_wait_s_per_run)
    print("expected total dark wait (s):", expected_dark_wait_s_total)

    feedback_records: List[Dict[str, Any]] = []
    phase_records: List[Dict[str, Any]] = []

    if mode == "old":
        base_charge_prep_fn = base_routine.charge_prep_no_verification_skip_first_rep
    else:
        base_charge_prep_fn = make_dmd_adaptive_charge_prep_fn(
            num_nvs=num_nvs,
            dmd_indices=dmd_indices_arr,
            dmd_radius_px=dmd_radius_px,
            dmd_plane=dmd_plane,
            dmd_settle_s=dmd_settle_s,
            verbose=verbose,
            feedback_records=feedback_records,
        )

    charge_prep_fn = make_measurement_backaction_charge_prep_fn(
        base_charge_prep_fn=base_charge_prep_fn,
        num_nvs=num_nvs,
        use_dmd=mode.startswith("dmd"),
        initial_check_rep_ind=initial_check_rep_ind,
        delayed_readout_rep_inds=delayed_readout_rep_inds,
        readout_interval_s=readout_interval_s,
        dmd_radius_px=dmd_radius_px,
        dmd_plane=dmd_plane,
        dmd_settle_s=dmd_settle_s,
        block_all_during_wait=block_all_during_wait,
        wait_status_interval_s=wait_status_interval_s,
        verbose=verbose,
        phase_records=phase_records,
    )

    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        ion_coords_list = widefield.get_coords_list(
            nv_run_list,
            VirtualLaserKey.ION,
        )
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(
                nv_run_list,
                VirtualLaserKey.CHARGE_POL,
            )
        )

        # No wait duration is passed to QUA. The host callback controls all
        # waits while the sequence is paused on its input stream.
        seq_args = [
            ion_coords_list,
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
        ]
        pulse_gen.stream_load(
            seq_file,
            tb.encode_seq_args(seq_args),
            num_reps_total,
        )

    raw_data = None
    experiment_t0 = time.perf_counter()

    try:
        raw_data = base_routine.main(
            nv_run_list,
            num_steps,
            num_reps_total,
            num_runs,
            run_fn=run_fn,
            save_images=save_images,
            save_images_avg_reps=save_images_avg_reps,
            charge_prep_fn=charge_prep_fn,
            num_exps=1,
            uwave_ind_list=[],
        )
    finally:
        if reset_dmd_on_exit and mode.startswith("dmd"):
            try:
                _dmd_pass_all_block_none(
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    dmd_settle_s=dmd_settle_s,
                )
            except Exception:
                print("Could not reset DMD:")
                print(traceback.format_exc())

    if raw_data is None:
        raise RuntimeError("base_routine.main returned no data.")

    experiment_wall_s = float(time.perf_counter() - experiment_t0)

    # Fail loudly if the callback was skipped. This catches the exact failure
    # mode where 100 requested waits finish in only a few seconds.
    delayed_phase_records = [
        record for record in phase_records if record.get("phase") == "delayed_readout"
    ]
    expected_delayed_records = num_runs * num_delayed_readouts

    if len(delayed_phase_records) != expected_delayed_records:
        raise RuntimeError(
            "The delayed-readout callback did not run the expected number of times. "
            f"Expected {expected_delayed_records}, recorded {len(delayed_phase_records)}. "
            "Check that base_routine.main receives charge_prep_fn and that the loaded "
            "QUA sequence waits for _cache_target_list on every rep after rep 0."
        )

    recorded_wait_s = float(
        np.sum([record["actual_wait_s"] for record in delayed_phase_records])
    )

    # Allow small timer/OS differences, but not a missing wait.
    if expected_dark_wait_s_total > 0 and recorded_wait_s < 0.95 * expected_dark_wait_s_total:
        raise RuntimeError(
            "Recorded dark waits are shorter than requested. "
            f"Requested total={expected_dark_wait_s_total:.3f}s, "
            f"recorded total={recorded_wait_s:.3f}s."
        )

    timestamp = dm.get_time_stamp()

    raw_data.update(
        {
            "analysis_type": "measurement_backaction_raw",
            "timestamp": timestamp,
            "mode": mode,
            "num_init_reps": num_init_reps,
            "num_delayed_readouts": num_delayed_readouts,
            "num_reps_total": num_reps_total,
            "initial_check_rep_ind": initial_check_rep_ind,
            "delayed_readout_rep_inds": delayed_readout_rep_inds,
            "measurement_rep_inds": measurement_rep_inds,
            "measurement_times_s": measurement_times_s,
            "readout_interval_s": readout_interval_s,
            "expected_dark_wait_s_per_run": expected_dark_wait_s_per_run,
            "expected_dark_wait_s_total": expected_dark_wait_s_total,
            "recorded_dark_wait_s_total": recorded_wait_s,
            "analysis_thresholds": analysis_thresholds,
            "feedback_thresholds": feedback_thresholds,
            "confirm_margin_counts": float(confirm_margin_counts),
            "dmd_indices": dmd_indices_arr,
            "dmd_radius_px": int(dmd_radius_px),
            "dmd_plane": int(dmd_plane),
            "block_all_during_wait": bool(block_all_during_wait),
            "feedback_records": feedback_records,
            "phase_records": phase_records,
            "experiment_wall_s": experiment_wall_s,
            "img_array-units": "photons",
            "sequence_file": seq_file,
        }
    )

    analysis = analyze_measurement_backaction(
        raw_data,
        initial_margin_counts=initial_analysis_margin_counts,
        final_margin_counts=final_analysis_margin_counts,
    )
    raw_data["measurement_backaction_analysis"] = analysis

    repr_nv_sig = widefield.get_repr_nv_sig(nv_run_list)
    interval_tag = f"interval-{readout_interval_s:g}s".replace(".", "p")
    count_tag = f"readouts-{num_delayed_readouts}"
    file_path = dm.get_file_path(
        __file__,
        timestamp,
        f"{repr_nv_sig.name}-measurement-backaction-{interval_tag}-{count_tag}",
    )

    if save_data:
        keys_to_compress = [
            "counts",
            "analysis_thresholds",
            "feedback_thresholds",
            "dmd_indices",
            "measurement_rep_inds",
            "measurement_times_s",
        ]
        if save_images and "img_arrays" in raw_data:
            keys_to_compress.append("img_arrays")

        dm.save_raw_data(
            raw_data,
            file_path,
            keys_to_compress=keys_to_compress,
        )
        print("Saved measurement-backaction data:", file_path)

    if save_fig:
        try:
            fig = plot_measurement_backaction_summary(raw_data, analysis)
            dm.save_figure(
                fig,
                _append_to_file_path(file_path, "summary"),
            )
        except Exception:
            print("Could not save measurement-backaction summary:")
            print(traceback.format_exc())

    tb.reset_cfm()
    raw_data["saved_file_path"] = str(file_path)
    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()

    file_stem_1_s = (
        "2026_07_17-13_56_56-"
        "qnami-nv0_2026_02_20-"
        "measurement-backaction-interval-1s-readouts-100"
    )

    file_stem_5_min = (
        "2026_07_17-07_56_54-"
        "qnami-nv0_2026_02_20-"
        "measurement-backaction-interval-300s-readouts-100"
    )

    raw_data_1_s = dm.get_raw_data(
        file_stem_1_s,
        load_npz=True,
    )
    analysis_1_s = analyze_measurement_backaction(
        raw_data_1_s,
        initial_margin_counts=1.0,
        final_margin_counts=1.0,
    )

    raw_data_5_min = dm.get_raw_data(
        file_stem_5_min,
        load_npz=True,
    )
    analysis_5_min = analyze_measurement_backaction(
        raw_data_5_min,
        initial_margin_counts=1.0,
        final_margin_counts=1.0,
    )

    analyses = [
        analysis_1_s,
        analysis_5_min,
    ]
    labels = [
        "1 s interval",
        "5 min interval",
    ]

    # The joint fit is used only for the supplementary dark-time/readout
    # decomposition shown in the full-height summary panel. The curves drawn
    # in both left panels are the independent fits for each dataset.
    joint_fit = fit_joint_dark_and_readout_rates(
        analyses
    )

    # One figure:
    #   top-left    survival versus measurement number
    #   bottom-left survival versus cumulative dark time
    #   full-right  fitting and dataset summary
    fig_comparison = plot_survival_interval_comparison(
        analyses,
        labels=labels,
        joint_fit=joint_fit,
        show_individual_runs=False,
        time_axis_scale="log",
    )

    print("\n=== Joint dark/readout survival fit ===")
    print(
        "Intrinsic dark relaxation rate:",
        joint_fit["gamma_dark_per_min"],
        "+/-",
        joint_fit["gamma_dark_per_min_stderr"],
        "per minute",
    )
    print(
        "Intrinsic dark time constant:",
        joint_fit["tau_dark_min"],
        "+/-",
        joint_fit["tau_dark_min_stderr"],
        "minutes",
    )
    print(
        "Effective NV- -> NV0 probability per readout:",
        joint_fit["loss_probability_per_readout"],
        "+/-",
        joint_fit["loss_probability_per_readout_stderr"],
    )
    print(
        "Effective NV0 -> NV- recovery probability per readout:",
        joint_fit["recovery_probability_per_readout"],
        "+/-",
        joint_fit["recovery_probability_per_readout_stderr"],
    )
    print(
        "Shared residual NV- fraction:",
        joint_fit["residual_nvm_fraction"],
        "+/-",
        joint_fit["residual_nvm_fraction_stderr"],
    )
    print(
        "Measurement relaxation exponent:",
        joint_fit["mu_readout"],
        "+/-",
        joint_fit["mu_readout_stderr"],
    )
    print("Fit R squared:", joint_fit["r_squared"])

    kpl.show(block=True)