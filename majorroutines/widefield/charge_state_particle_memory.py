# -*- coding: utf-8 -*-
"""
Adaptive NV- initialization, dark exposure, and delayed charge-state readout.

Experiment sequence for each run
--------------------------------
    1. rep 0: ionize all NVs and read out.
    2. reps 1 .. num_init_reps-1: adaptively polarize only unconfirmed NVs.
       Confirmed NV- sites are blocked by the DMD.
    3. optional immediate verification rep: DMD pass-all, no polarization,
       charge-state readout of every NV.
    4. dark exposure: OPX remains paused, DMD can block all optical paths, and
       Python waits for ``dark_wait_s``. Optional source-control callbacks can
       be called at the beginning/end of this interval.
    5. delayed final rep: DMD pass-all, no polarization, charge-state readout.
    6. classify high-confidence NV- -> NV0 transitions and find spatial clusters.

Expected counts returned by base_routine
----------------------------------------
    counts[exp, nv, run, step, rep]

The QUA file used by default is ``charge_state_particle_memory.py``.
Place it in your QM OPX sequence-library folder beside the existing
``charge_state_conditional_init.py`` sequence.

Created July 2026.
"""

from __future__ import annotations
import sys
import copy
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield
from utils.constants import VirtualLaserKey

import csv
from scipy.optimize import curve_fit

# =============================================================================
# Basic helpers
# =============================================================================


def _get_thresholds(nv_list) -> np.ndarray:
    thresholds = []
    for nv_ind, nv in enumerate(nv_list):
        threshold = getattr(nv, "threshold", None)
        if threshold is None or not np.isfinite(threshold):
            raise ValueError(
                f"NV {nv_ind} has invalid threshold {threshold}. "
                "Assign nv.threshold before running."
            )
        thresholds.append(float(threshold))
    return np.asarray(thresholds, dtype=float)


def _copy_nv_list_with_confirmation_margin(
    nv_list,
    confirm_margin_counts: float,
):
    """Copy NV objects and raise only the adaptive-confirmation threshold."""

    nv_run_list = []
    for nv in nv_list:
        nv_copy = copy.copy(nv)
        nv_copy.threshold = float(nv.threshold) + float(confirm_margin_counts)
        nv_run_list.append(nv_copy)
    return nv_run_list


def _prepare_dmd_indices(
    num_nvs: int,
    dmd_indices: Optional[Sequence[int]],
) -> np.ndarray:
    if dmd_indices is None:
        out = np.arange(num_nvs, dtype=int)
    else:
        out = np.asarray(dmd_indices, dtype=int)

    if out.shape != (num_nvs,):
        raise ValueError(
            f"dmd_indices must have length {num_nvs}; got shape {out.shape}."
        )
    return out


def _append_to_file_path(file_path, suffix: str) -> Path:
    file_path = Path(file_path)
    return file_path.with_name(f"{file_path.name}-{suffix}")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(key): _json_safe(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(val) for val in obj]
    return obj


# =============================================================================
# DMD helpers
# =============================================================================


def _dmd_pass_all_block_none(
    dmd=None,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
) -> None:
    """White/pass background with no black blocked disks."""

    if dmd is None:
        dmd = tb.get_server_dmd()

    dmd.block_loaded_indices(
        json.dumps([]),
        int(dmd_radius_px),
        int(dmd_plane),
    )
    time.sleep(float(dmd_settle_s))


def _dmd_block_all(
    dmd=None,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
) -> None:
    """
    Black/block background with no white pass disks.

    This follows the server convention used by ``pass_loaded_indices``:
    an empty pass list leaves the entire DMD plane blocked.
    """

    if dmd is None:
        dmd = tb.get_server_dmd()

    dmd.pass_loaded_indices(
        json.dumps([]),
        int(dmd_radius_px),
        int(dmd_plane),
    )
    time.sleep(float(dmd_settle_s))


def _dmd_block_confirmed(
    confirmed_mask: np.ndarray,
    dmd_indices: np.ndarray,
    dmd=None,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
) -> List[int]:
    """Pass background and block disks at confirmed NV- sites."""

    if dmd is None:
        dmd = tb.get_server_dmd()

    confirmed_mask = np.asarray(confirmed_mask, dtype=bool)
    dmd_indices = np.asarray(dmd_indices, dtype=int)
    confirmed_dmd_inds = dmd_indices[confirmed_mask].astype(int).tolist()

    dmd.block_loaded_indices(
        json.dumps(confirmed_dmd_inds),
        int(dmd_radius_px),
        int(dmd_plane),
    )
    time.sleep(float(dmd_settle_s))
    return confirmed_dmd_inds


# =============================================================================
# Adaptive charge preparation
# =============================================================================


def make_dmd_adaptive_charge_prep_fn(
    num_nvs: int,
    dmd_indices: Optional[Sequence[int]] = None,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
    verbose: bool = True,
    feedback_records: Optional[List[Dict[str, Any]]] = None,
):
    """
    Persistently confirm NV- sites and remove them from subsequent attempts.

    ``initial_states_list`` is the thresholded result from the previous rep.
    A confirmed site remains confirmed for the rest of the run.
    """

    dmd_indices_arr = _prepare_dmd_indices(num_nvs, dmd_indices)
    pulse_gen = tb.get_server_pulse_gen()
    dmd = tb.get_server_dmd()

    confirmed_nvm = np.zeros(num_nvs, dtype=bool)
    last_confirmed = {"mask": None}
    run_counter = {"run_ind": -1}

    def update_dmd(confirmed_mask: np.ndarray, force: bool = False):
        confirmed_mask = np.asarray(confirmed_mask, dtype=bool)

        if (
            not force
            and last_confirmed["mask"] is not None
            and np.array_equal(confirmed_mask, last_confirmed["mask"])
        ):
            return 0.0, False

        t0 = time.perf_counter()
        _dmd_block_confirmed(
            confirmed_mask,
            dmd_indices_arr,
            dmd=dmd,
            dmd_radius_px=dmd_radius_px,
            dmd_plane=dmd_plane,
            dmd_settle_s=dmd_settle_s,
        )
        elapsed = time.perf_counter() - t0
        last_confirmed["mask"] = confirmed_mask.copy()
        return float(elapsed), True

    def charge_prep_fn(rep_ind, nv_list, initial_states_list=None):
        nonlocal confirmed_nvm

        t_total0 = time.perf_counter()

        if rep_ind == 0:
            run_counter["run_ind"] += 1
            confirmed_nvm[:] = False
            last_confirmed["mask"] = None
            dmd_s, dmd_changed = update_dmd(confirmed_nvm, force=True)

            if feedback_records is not None:
                feedback_records.append(
                    {
                        "run_ind": int(run_counter["run_ind"]),
                        "rep_ind": int(rep_ind),
                        "phase": "start",
                        "newly_confirmed": 0,
                        "confirmed": 0,
                        "active": int(num_nvs),
                        "dmd_s": dmd_s,
                        "dmd_changed": dmd_changed,
                        "total_callback_s": float(time.perf_counter() - t_total0),
                    }
                )

            if verbose:
                print(
                    f"[adaptive] run {run_counter['run_ind']}, rep 0: "
                    f"confirmed=0/{num_nvs}, dmd={dmd_s:.3f}s"
                )
            return

        if initial_states_list is None:
            previous_nvm = np.zeros(num_nvs, dtype=bool)
        else:
            previous_nvm = np.asarray(initial_states_list, dtype=bool)
            if previous_nvm.shape != (num_nvs,):
                raise ValueError(
                    "initial_states_list shape mismatch: "
                    f"expected {(num_nvs,)}, got {previous_nvm.shape}."
                )

        active_before = ~confirmed_nvm
        newly_confirmed = active_before & previous_nvm
        confirmed_nvm[newly_confirmed] = True
        active_mask = ~confirmed_nvm

        dmd_s, dmd_changed = update_dmd(confirmed_nvm, force=False)

        t_opx0 = time.perf_counter()
        pulse_gen.insert_input_stream(
            "_cache_target_list",
            active_mask.astype(bool).tolist(),
        )
        opx_s = float(time.perf_counter() - t_opx0)
        total_s = float(time.perf_counter() - t_total0)

        if feedback_records is not None:
            feedback_records.append(
                {
                    "run_ind": int(run_counter["run_ind"]),
                    "rep_ind": int(rep_ind),
                    "phase": "adaptive_feedback",
                    "newly_confirmed": int(np.sum(newly_confirmed)),
                    "confirmed": int(np.sum(confirmed_nvm)),
                    "active": int(np.sum(active_mask)),
                    "dmd_s": dmd_s,
                    "dmd_changed": dmd_changed,
                    "opx_s": opx_s,
                    "total_callback_s": total_s,
                }
            )

        if verbose:
            print(
                f"[adaptive] run {run_counter['run_ind']}, rep {rep_ind}: "
                f"new={np.sum(newly_confirmed)}, "
                f"confirmed={np.sum(confirmed_nvm)}/{num_nvs}, "
                f"active={np.sum(active_mask)}/{num_nvs}, "
                f"dmd={dmd_s:.3f}s, opx={opx_s:.3f}s"
            )

    return charge_prep_fn


def _wait_with_progress(
    wait_s: float,
    status_interval_s: float = 60.0,
    verbose: bool = True,
) -> float:
    """Interruptible wall-clock wait using a monotonic timer."""

    wait_s = max(0.0, float(wait_s))
    status_interval_s = max(0.1, float(status_interval_s))
    start = time.perf_counter()
    end = start + wait_s

    while True:
        remaining = end - time.perf_counter()
        if remaining <= 0:
            break

        sleep_s = min(remaining, status_interval_s)
        time.sleep(sleep_s)

        if verbose and remaining > status_interval_s:
            elapsed = time.perf_counter() - start
            remaining_now = max(0.0, end - time.perf_counter())
            print(
                f"[dark exposure] elapsed={elapsed:.1f}s, "
                f"remaining={remaining_now:.1f}s"
            )

    return float(time.perf_counter() - start)


def make_particle_memory_charge_prep_fn(
    base_charge_prep_fn,
    num_nvs: int,
    use_dmd: bool,
    initial_check_rep_ind: Optional[int],
    final_readout_rep_ind: int,
    dark_wait_s: float,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
    block_all_during_wait: bool = True,
    exposure_start_fn: Optional[Callable[[], None]] = None,
    exposure_stop_fn: Optional[Callable[[], None]] = None,
    wait_status_interval_s: float = 60.0,
    verbose: bool = True,
    phase_records: Optional[List[Dict[str, Any]]] = None,
):
    """
    Add immediate verification and delayed final readout around adaptive prep.

    The long wait occurs while the QUA program is paused between repetitions.
    """

    pulse_gen = tb.get_server_pulse_gen()
    dmd = tb.get_server_dmd() if use_dmd else None
    all_false = np.zeros(num_nvs, dtype=bool).tolist()
    run_counter = {"run_ind": -1}

    def wrapped(rep_ind, nv_list, initial_states_list=None):
        if rep_ind == 0:
            run_counter["run_ind"] += 1

        run_ind = int(run_counter["run_ind"])

        # Read every NV immediately after adaptive initialization.
        if initial_check_rep_ind is not None and rep_ind == initial_check_rep_ind:
            t0 = time.perf_counter()
            if use_dmd:
                _dmd_pass_all_block_none(
                    dmd=dmd,
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    dmd_settle_s=dmd_settle_s,
                )
            pulse_gen.insert_input_stream("_cache_target_list", all_false)
            elapsed = float(time.perf_counter() - t0)

            if phase_records is not None:
                phase_records.append(
                    {
                        "run_ind": run_ind,
                        "rep_ind": int(rep_ind),
                        "phase": "initial_verification",
                        "callback_s": elapsed,
                    }
                )

            if verbose:
                print(
                    f"[initial verification] run {run_ind}, rep {rep_ind}: "
                    "DMD pass-all, target list all false"
                )
            return

        # The callback for the final rep runs after the initial verification
        # count has been received. Keep the OPX paused during the exposure.
        if rep_ind == final_readout_rep_ind:
            initial_state_from_previous_rep = (
                None
                if initial_states_list is None
                else np.asarray(initial_states_list, dtype=bool)
            )

            if initial_state_from_previous_rep is not None:
                num_initial_nvm = int(np.sum(initial_state_from_previous_rep))
            else:
                num_initial_nvm = None

            # --------------------------------------------------------------
            # Keep QUA paused: do NOT insert the input stream yet.
            # --------------------------------------------------------------

            if use_dmd and block_all_during_wait:
                _dmd_block_all(
                    dmd=dmd,
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    dmd_settle_s=dmd_settle_s,
                )

            if verbose:
                print(
                    f"[dark exposure] run {run_ind}: start, "
                    f"requested={float(dark_wait_s):.3f}s, "
                    f"initial NV- from previous readout={num_initial_nvm}"
                )

            source_started = False
            wait_t0 = time.perf_counter()

            try:
                if exposure_start_fn is not None:
                    exposure_start_fn()
                    source_started = True

                actual_wait_s = _wait_with_progress(
                    dark_wait_s,
                    status_interval_s=wait_status_interval_s,
                    verbose=verbose,
                )

            finally:
                if exposure_stop_fn is not None and source_started:
                    exposure_stop_fn()

            wait_callback_s = float(time.perf_counter() - wait_t0)

            # --------------------------------------------------------------
            # Prepare optical path before releasing QUA.
            # --------------------------------------------------------------
            if use_dmd:
                _dmd_pass_all_block_none(
                    dmd=dmd,
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    dmd_settle_s=dmd_settle_s,
                )

            # Only now release QUA for the final readout-only repetition.
            pulse_gen.insert_input_stream(
                "_cache_target_list",
                all_false,
            )

            if phase_records is not None:
                phase_records.append(
                    {
                        "run_ind": run_ind,
                        "rep_ind": int(rep_ind),
                        "phase": "dark_exposure",
                        "requested_wait_s": float(dark_wait_s),
                        "actual_wait_s": float(actual_wait_s),
                        "wait_callback_s": wait_callback_s,
                        "num_initial_nvm_from_callback": num_initial_nvm,
                        "block_all_during_wait": bool(
                            use_dmd and block_all_during_wait
                        ),
                    }
                )

            if verbose:
                print(
                    f"[dark exposure] run {run_ind}: complete, "
                    f"actual={actual_wait_s:.3f}s; resuming final readout"
                )
            return

        return base_charge_prep_fn(rep_ind, nv_list, initial_states_list)

    return wrapped


# =============================================================================
# Event classification and clustering
# =============================================================================


def _try_get_nv_img_xy(nv) -> Optional[Tuple[float, float]]:
    for attr in ("pixel_coords", "img_coords", "image_coords", "camera_coords"):
        val = getattr(nv, attr, None)
        if val is not None:
            arr = np.asarray(val, dtype=float).ravel()
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                return float(arr[0]), float(arr[1])

    coords = getattr(nv, "coords", None)
    if isinstance(coords, dict):
        for key in (
            "pixel",
            "pixels",
            "pixel_coords",
            "img",
            "image",
            "camera",
            "camera_coords",
        ):
            if key in coords:
                arr = np.asarray(coords[key], dtype=float).ravel()
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    return float(arr[0]), float(arr[1])
    return None


def _coerce_img_coords(
    nv_list,
    img_coords: Optional[Sequence[Sequence[float]]] = None,
) -> Optional[np.ndarray]:
    if img_coords is not None:
        arr = np.asarray(img_coords, dtype=float)
        if arr.shape != (len(nv_list), 2):
            raise ValueError(
                f"img_coords must have shape {(len(nv_list), 2)}; got {arr.shape}."
            )
        return arr

    coords = []
    for nv in nv_list:
        xy = _try_get_nv_img_xy(nv)
        if xy is None:
            return None
        coords.append(xy)
    return np.asarray(coords, dtype=float)


def _connected_components_within_radius(
    candidate_inds: np.ndarray,
    coords_xy: np.ndarray,
    radius: float,
) -> List[List[int]]:
    """Connected components of candidate NVs using a distance threshold."""

    candidate_inds = np.asarray(candidate_inds, dtype=int)
    if candidate_inds.size == 0:
        return []

    candidate_coords = np.asarray(coords_xy[candidate_inds], dtype=float)
    distance_sq = np.sum(
        (candidate_coords[:, None, :] - candidate_coords[None, :, :]) ** 2,
        axis=2,
    )
    adjacency = distance_sq <= float(radius) ** 2

    visited = np.zeros(candidate_inds.size, dtype=bool)
    components: List[List[int]] = []

    for start in range(candidate_inds.size):
        if visited[start]:
            continue

        stack = [start]
        visited[start] = True
        component_local = []

        while stack:
            current = stack.pop()
            component_local.append(current)
            neighbors = np.where(adjacency[current] & (~visited))[0]
            for neighbor in neighbors:
                visited[neighbor] = True
                stack.append(int(neighbor))

        components.append(candidate_inds[component_local].astype(int).tolist())

    components.sort(key=len, reverse=True)
    return components


def analyze_particle_charge_memory(
    raw_data: Dict[str, Any],
    initial_margin_counts: float = 1.0,
    final_margin_counts: float = 1.0,
    cluster_radius_px: Optional[float] = None,
    min_cluster_size: int = 2,
    img_coords: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """
    Find high-confidence NV- -> NV0 transitions during the dark exposure.

    Candidate definition
    --------------------
        initial_count > threshold + initial_margin_counts
        final_count   <= threshold - final_margin_counts

    Counts close to the threshold are intentionally treated as ambiguous rather
    than particle candidates.
    """

    counts_all = np.asarray(raw_data["counts"], dtype=float)
    if counts_all.ndim != 5:
        raise ValueError(
            "Expected counts[exp, nv, run, step, rep]; "
            f"got shape {counts_all.shape}."
        )

    counts = counts_all[0, :, :, 0, :]  # [nv, run, rep]
    num_nvs, num_runs, num_reps = counts.shape

    initial_rep_ind = int(raw_data["initial_state_rep_ind"])
    final_rep_ind = int(raw_data["final_readout_rep_ind"])
    if not (0 <= initial_rep_ind < num_reps):
        raise IndexError(f"Invalid initial_state_rep_ind={initial_rep_ind}.")
    if not (0 <= final_rep_ind < num_reps):
        raise IndexError(f"Invalid final_readout_rep_ind={final_rep_ind}.")

    thresholds = np.asarray(raw_data["analysis_thresholds"], dtype=float)
    if thresholds.shape != (num_nvs,):
        raise ValueError(
            f"Threshold shape {thresholds.shape} does not match num_nvs={num_nvs}."
        )

    initial_counts = counts[:, :, initial_rep_ind]
    final_counts = counts[:, :, final_rep_ind]
    threshold_2d = thresholds[:, None]

    initial_nvm = initial_counts > (
        threshold_2d + float(initial_margin_counts)
    )
    final_nvm_standard = final_counts > threshold_2d
    final_nv0_confident = final_counts <= (
        threshold_2d - float(final_margin_counts)
    )
    final_ambiguous = (~final_nvm_standard) & (~final_nv0_confident)

    candidate_mask = initial_nvm & final_nv0_confident
    retained_mask = initial_nvm & final_nvm_standard
    eligible_mask = initial_nvm

    num_initial_nvm_by_run = np.sum(initial_nvm, axis=0)
    num_final_nvm_by_run = np.sum(final_nvm_standard, axis=0)
    num_candidates_by_run = np.sum(candidate_mask, axis=0)
    num_retained_by_run = np.sum(retained_mask, axis=0)
    num_ambiguous_by_run = np.sum(initial_nvm & final_ambiguous, axis=0)

    retention_by_run = np.full(num_runs, np.nan, dtype=float)
    event_fraction_by_run = np.full(num_runs, np.nan, dtype=float)
    good_runs = num_initial_nvm_by_run > 0
    retention_by_run[good_runs] = (
        num_retained_by_run[good_runs] / num_initial_nvm_by_run[good_runs]
    )
    event_fraction_by_run[good_runs] = (
        num_candidates_by_run[good_runs] / num_initial_nvm_by_run[good_runs]
    )

    eligible_trials_by_nv = np.sum(eligible_mask, axis=1)
    event_trials_by_nv = np.sum(candidate_mask, axis=1)
    event_probability_by_nv = np.full(num_nvs, np.nan, dtype=float)
    good_nv = eligible_trials_by_nv > 0
    event_probability_by_nv[good_nv] = (
        event_trials_by_nv[good_nv] / eligible_trials_by_nv[good_nv]
    )

    coords_xy = _coerce_img_coords(raw_data["nv_list"], img_coords=img_coords)
    cluster_components_by_run: List[List[List[int]]] = []
    max_cluster_size_by_run = np.zeros(num_runs, dtype=int)
    num_clusters_by_run = np.zeros(num_runs, dtype=int)
    num_large_clusters_by_run = np.zeros(num_runs, dtype=int)

    if coords_xy is not None and cluster_radius_px is not None:
        for run_ind in range(num_runs):
            candidate_inds = np.where(candidate_mask[:, run_ind])[0]
            components = _connected_components_within_radius(
                candidate_inds,
                coords_xy,
                radius=float(cluster_radius_px),
            )
            cluster_components_by_run.append(components)
            max_cluster_size_by_run[run_ind] = (
                max((len(comp) for comp in components), default=0)
            )
            num_clusters_by_run[run_ind] = len(components)
            num_large_clusters_by_run[run_ind] = sum(
                len(comp) >= int(min_cluster_size) for comp in components
            )
    else:
        cluster_components_by_run = [[] for _ in range(num_runs)]

    summary: Dict[str, Any] = {
        "analysis_type": "particle_charge_memory",
        "num_nvs": int(num_nvs),
        "num_runs": int(num_runs),
        "dark_wait_s": float(raw_data["dark_wait_s"]),
        "exposure_label": str(raw_data.get("exposure_label", "unspecified")),
        "initial_state_rep_ind": initial_rep_ind,
        "final_readout_rep_ind": final_rep_ind,
        "initial_margin_counts": float(initial_margin_counts),
        "final_margin_counts": float(final_margin_counts),
        "cluster_radius_px": (
            None if cluster_radius_px is None else float(cluster_radius_px)
        ),
        "min_cluster_size": int(min_cluster_size),
        "initial_counts": initial_counts.tolist(),
        "final_counts": final_counts.tolist(),
        "initial_nvm_mask": initial_nvm.tolist(),
        "final_nvm_mask": final_nvm_standard.tolist(),
        "final_nv0_confident_mask": final_nv0_confident.tolist(),
        "final_ambiguous_mask": final_ambiguous.tolist(),
        "candidate_nvm_to_nv0_mask": candidate_mask.tolist(),
        "retained_nvm_mask": retained_mask.tolist(),
        "num_initial_nvm_by_run": num_initial_nvm_by_run.tolist(),
        "num_final_nvm_by_run": num_final_nvm_by_run.tolist(),
        "num_candidates_by_run": num_candidates_by_run.tolist(),
        "num_retained_by_run": num_retained_by_run.tolist(),
        "num_ambiguous_by_run": num_ambiguous_by_run.tolist(),
        "retention_by_run": retention_by_run.tolist(),
        "event_fraction_by_run": event_fraction_by_run.tolist(),
        "eligible_trials_by_nv": eligible_trials_by_nv.tolist(),
        "event_trials_by_nv": event_trials_by_nv.tolist(),
        "event_probability_by_nv": event_probability_by_nv.tolist(),
        "mean_initial_nvm": float(np.mean(num_initial_nvm_by_run)),
        "mean_final_nvm": float(np.mean(num_final_nvm_by_run)),
        "mean_candidates_per_run": float(np.mean(num_candidates_by_run)),
        "median_candidates_per_run": float(np.median(num_candidates_by_run)),
        "mean_retention": float(np.nanmean(retention_by_run)),
        "mean_event_fraction": float(np.nanmean(event_fraction_by_run)),
        "cluster_components_by_run": cluster_components_by_run,
        "max_cluster_size_by_run": max_cluster_size_by_run.tolist(),
        "num_clusters_by_run": num_clusters_by_run.tolist(),
        "num_large_clusters_by_run": num_large_clusters_by_run.tolist(),
        "coords_xy": None if coords_xy is None else coords_xy.tolist(),
    }

    print("\n=== Particle charge-memory analysis ===")
    print("exposure label:", summary["exposure_label"])
    print("dark wait (s):", summary["dark_wait_s"])
    print("mean initial NV-:", summary["mean_initial_nvm"], "/", num_nvs)
    print("mean final NV-:", summary["mean_final_nvm"], "/", num_nvs)
    print("mean candidate NV- -> NV0 per run:", summary["mean_candidates_per_run"])
    print("mean retention:", summary["mean_retention"])
    if cluster_radius_px is not None:
        print("max cluster size by run:", max_cluster_size_by_run.tolist())

    return summary


# =============================================================================
# KPL-style plots
# =============================================================================


def plot_particle_summary(
    raw_data: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
):
    if analysis is None:
        analysis = raw_data["particle_analysis"]

    runs = np.arange(int(analysis["num_runs"]))
    initial = np.asarray(analysis["num_initial_nvm_by_run"], dtype=float)
    final = np.asarray(analysis["num_final_nvm_by_run"], dtype=float)
    candidates = np.asarray(analysis["num_candidates_by_run"], dtype=float)
    retention = np.asarray(analysis["retention_by_run"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(
        runs,
        initial,
        "o-",
        color=kpl.KplColors.GREEN,
        label="Initial verified NV$^-$",
    )
    axes[0].plot(
        runs,
        final,
        "o-",
        color=kpl.KplColors.BLUE,
        label="Final NV$^-$",
    )
    axes[0].set_ylabel("Number of NVs")
    axes[0].legend()

    axes[1].plot(
        runs,
        candidates,
        "o-",
        color=kpl.KplColors.RED,
        label="NV$^- \\rightarrow$ NV$^0$ candidates",
    )
    axes[1].set_ylabel("Candidates")
    axes[1].legend()

    axes[2].plot(
        runs,
        retention,
        "o-",
        color=kpl.KplColors.BLUE,
        label="NV$^-$ retention",
    )
    axes[2].set_xlabel("Run index")
    axes[2].set_ylabel("Retention")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Charge-memory exposure: {analysis['exposure_label']}\n"
        f"dark wait = {analysis['dark_wait_s']:.3g} s",
        fontsize=14,
    )
    return fig


def plot_particle_event_map(
    raw_data: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    run_ind: int = 0,
    clim: Optional[Tuple[float, float]] = None,
    marker_size: float = 28,
):
    """Plot immediate and delayed frames with eligible/candidate NV overlays."""

    if analysis is None:
        analysis = raw_data["particle_analysis"]

    coords_raw = analysis.get("coords_xy", None)
    if coords_raw is None:
        raise ValueError(
            "No image coordinates were found. Pass img_coords to main() or "
            "analyze_particle_charge_memory()."
        )
    coords = np.asarray(coords_raw, dtype=float)

    run_ind = int(run_ind)
    num_runs = int(analysis["num_runs"])
    if not (0 <= run_ind < num_runs):
        raise IndexError(f"run_ind must be in [0, {num_runs - 1}].")

    eligible = np.asarray(analysis["initial_nvm_mask"], dtype=bool)[:, run_ind]
    candidates = np.asarray(
        analysis["candidate_nvm_to_nv0_mask"], dtype=bool
    )[:, run_ind]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    if "img_arrays" in raw_data:
        imgs = np.asarray(raw_data["img_arrays"], dtype=float)
        initial_rep = int(analysis["initial_state_rep_ind"])
        final_rep = int(analysis["final_readout_rep_ind"])
        initial_img = imgs[0, run_ind, 0, initial_rep]
        final_img = imgs[0, run_ind, 0, final_rep]

        if clim is None:
            both = np.stack([initial_img, final_img], axis=0)
            clim_use = (
                float(np.nanpercentile(both, 50)),
                float(np.nanpercentile(both, 99.8)),
            )
        else:
            clim_use = clim

        im0 = axes[0].imshow(
            initial_img,
            origin="upper",
            vmin=clim_use[0],
            vmax=clim_use[1],
        )
        axes[1].imshow(
            final_img,
            origin="upper",
            vmin=clim_use[0],
            vmax=clim_use[1],
        )
        cbar = fig.colorbar(im0, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("photons")
    else:
        for ax in axes:
            ax.set_xlim(np.nanmin(coords[:, 0]) - 5, np.nanmax(coords[:, 0]) + 5)
            ax.set_ylim(np.nanmax(coords[:, 1]) + 5, np.nanmin(coords[:, 1]) - 5)

    axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        s=marker_size * 0.6,
        facecolors="none",
        edgecolors=kpl.KplColors.GRAY,
        linewidths=0.6,
    )
    axes[0].scatter(
        coords[eligible, 0],
        coords[eligible, 1],
        s=marker_size,
        facecolors="none",
        edgecolors=kpl.KplColors.GREEN,
        linewidths=1.0,
        label="initial verified NV$^-$",
    )

    axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        s=marker_size * 0.6,
        facecolors="none",
        edgecolors=kpl.KplColors.GRAY,
        linewidths=0.6,
    )
    axes[1].scatter(
        coords[candidates, 0],
        coords[candidates, 1],
        s=marker_size * 1.3,
        facecolors="none",
        edgecolors=kpl.KplColors.RED,
        linewidths=1.4,
        label="NV$^- \\rightarrow$ NV$^0$ candidate",
    )

    axes[0].set_title(
        f"Immediate verification\n{np.sum(eligible)} verified NV$^-$",
        fontsize=13,
    )
    axes[1].set_title(
        f"After {analysis['dark_wait_s']:.3g} s\n"
        f"{np.sum(candidates)} candidates",
        fontsize=13,
    )

    for ax in axes:
        ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=8)
        ax.set_axis_off()

    fig.suptitle(
        f"Run {run_ind}: {analysis['exposure_label']}",
        fontsize=14,
    )
    return fig


def plot_event_probability_by_nv(
    raw_data: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
):
    if analysis is None:
        analysis = raw_data["particle_analysis"]

    probability = np.asarray(analysis["event_probability_by_nv"], dtype=float)
    nv_inds = np.arange(probability.size)
    good = np.isfinite(probability)

    fig, ax = plt.subplots(figsize=kpl.figsize)
    ax.scatter(
        nv_inds[good],
        probability[good],
        s=12,
        alpha=0.7,
        color=kpl.KplColors.RED,
    )
    ax.set_xlabel("NV index")
    ax.set_ylabel("Candidate probability per eligible exposure")
    ax.set_ylim(-0.01, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-NV NV$^- \\rightarrow$ NV$^0$ candidate probability")
    return fig


# =============================================================================
# Main experiment
# =============================================================================


def main(
    nv_list,
    num_init_reps: int = 10,
    num_runs: int = 10,
    dark_wait_s: float = 300.0,
    mode: str = "dmd_block_confirmed",
    dmd_indices: Optional[Sequence[int]] = None,
    dmd_radius_px: int = 8,
    dmd_plane: int = 230,
    dmd_settle_s: float = 0.001,
    confirm_margin_counts: float = 1.0,
    take_initial_check: bool = True,
    block_all_during_wait: bool = True,
    exposure_label: str = "No soruce",
    exposure_start_fn: Optional[Callable[[], None]] = None,
    exposure_stop_fn: Optional[Callable[[], None]] = None,
    wait_status_interval_s: float = 60.0,
    initial_event_margin_counts: float = 1.0,
    final_event_margin_counts: float = 1.0,
    cluster_radius_px: Optional[float] = None,
    min_cluster_size: int = 2,
    img_coords: Optional[Sequence[Sequence[float]]] = None,
    save_images: bool = True,
    save_images_avg_reps: bool = False,
    save_data: bool = True,
    save_fig: bool = True,
    reset_dmd_on_exit: bool = True,
    verbose: bool = True,
    seq_file: str = "charge_state_particle_memory.py",
) -> Dict[str, Any]:
    """Run one adaptive charge-memory exposure dataset."""

    if mode not in ("old", "dmd_block_confirmed"):
        raise ValueError("mode must be 'old' or 'dmd_block_confirmed'.")
    if int(num_init_reps) < 2:
        raise ValueError("num_init_reps must be at least 2 (ionize + one adaptive rep).")
    if int(num_runs) < 1:
        raise ValueError("num_runs must be positive.")
    if float(dark_wait_s) < 0:
        raise ValueError("dark_wait_s cannot be negative.")

    num_init_reps = int(num_init_reps)
    num_runs = int(num_runs)
    num_steps = 1
    num_nvs = len(nv_list)

    # Original thresholds are reserved for physical event classification.
    analysis_thresholds = _get_thresholds(nv_list)

    # Raised thresholds are used only for adaptive confirmation.
    nv_run_list = _copy_nv_list_with_confirmation_margin(
        nv_list,
        confirm_margin_counts=confirm_margin_counts,
    )
    feedback_thresholds = _get_thresholds(nv_run_list)
    dmd_indices_arr = _prepare_dmd_indices(num_nvs, dmd_indices)

    if take_initial_check:
        initial_check_rep_ind: Optional[int] = num_init_reps
        initial_state_rep_ind = num_init_reps
        final_readout_rep_ind = num_init_reps + 1
    else:
        initial_check_rep_ind = None
        initial_state_rep_ind = num_init_reps - 1
        final_readout_rep_ind = num_init_reps

    num_reps_total = final_readout_rep_ind + 1

    print("\n=== Adaptive particle charge-memory experiment ===")
    print("mode:", mode)
    print("num NVs:", num_nvs)
    print("num initialization reps:", num_init_reps)
    print("num total reps:", num_reps_total)
    print("num runs:", num_runs)
    print("initial state rep:", initial_state_rep_ind)
    print("final readout rep:", final_readout_rep_ind)
    print("dark wait (s):", dark_wait_s)
    print("exposure label:", exposure_label)
    print("confirm margin (counts):", confirm_margin_counts)
    print(
        "analysis threshold range:",
        float(np.min(analysis_thresholds)),
        float(np.max(analysis_thresholds)),
    )

    feedback_records: List[Dict[str, Any]] = []
    phase_records: List[Dict[str, Any]] = []

    if mode == "old":
        base_charge_prep_fn = (
            base_routine.charge_prep_no_verification_skip_first_rep
        )
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

    charge_prep_fn = make_particle_memory_charge_prep_fn(
        base_charge_prep_fn=base_charge_prep_fn,
        num_nvs=num_nvs,
        use_dmd=mode.startswith("dmd"),
        initial_check_rep_ind=initial_check_rep_ind,
        final_readout_rep_ind=final_readout_rep_ind,
        dark_wait_s=dark_wait_s,
        dmd_radius_px=dmd_radius_px,
        dmd_plane=dmd_plane,
        dmd_settle_s=dmd_settle_s,
        block_all_during_wait=block_all_during_wait,
        exposure_start_fn=exposure_start_fn,
        exposure_stop_fn=exposure_stop_fn,
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
    timestamp = dm.get_time_stamp()

    raw_data.update(
        {
            "analysis_type": "adaptive_particle_charge_memory_raw",
            "mode": mode,
            "timestamp": timestamp,
            "num_init_reps": num_init_reps,
            "num_reps_total": num_reps_total,
            "take_initial_check": bool(take_initial_check),
            "initial_check_rep_ind": initial_check_rep_ind,
            "initial_state_rep_ind": int(initial_state_rep_ind),
            "final_readout_rep_ind": int(final_readout_rep_ind),
            "dark_wait_s": float(dark_wait_s),
            "exposure_label": str(exposure_label),
            "block_all_during_wait": bool(block_all_during_wait),
            "analysis_thresholds": analysis_thresholds,
            "feedback_thresholds": feedback_thresholds,
            "confirm_margin_counts": float(confirm_margin_counts),
            "dmd_indices": dmd_indices_arr,
            "dmd_radius_px": int(dmd_radius_px),
            "dmd_plane": int(dmd_plane),
            "feedback_records": feedback_records,
            "phase_records": phase_records,
            "experiment_wall_s": experiment_wall_s,
            "img_array-units": "photons",
            "sequence_file": seq_file,
        }
    )

    particle_analysis = analyze_particle_charge_memory(
        raw_data,
        initial_margin_counts=initial_event_margin_counts,
        final_margin_counts=final_event_margin_counts,
        cluster_radius_px=cluster_radius_px,
        min_cluster_size=min_cluster_size,
        img_coords=img_coords,
    )
    raw_data["particle_analysis"] = particle_analysis

    repr_nv_sig = widefield.get_repr_nv_sig(nv_run_list)
    wait_tag = f"wait-{float(dark_wait_s):g}s".replace(".", "p")
    label_tag = str(exposure_label).replace(" ", "_")
    file_path = dm.get_file_path(
        __file__,
        timestamp,
        f"{repr_nv_sig.name}-particle-memory-{label_tag}-{wait_tag}",
    )

    if save_data:
        keys_to_compress = [
            "counts",
            "analysis_thresholds",
            "feedback_thresholds",
            "dmd_indices",
        ]
        if save_images and "img_arrays" in raw_data:
            keys_to_compress.append("img_arrays")

        dm.save_raw_data(
            raw_data,
            file_path,
            keys_to_compress=keys_to_compress,
        )
        print("Saved particle-memory raw data:", file_path)

    if save_fig:
        try:
            fig_summary = plot_particle_summary(raw_data, particle_analysis)
            dm.save_figure(
                fig_summary,
                _append_to_file_path(file_path, "summary"),
            )
        except Exception:
            print("Could not save particle summary plot:")
            print(traceback.format_exc())

        try:
            fig_probability = plot_event_probability_by_nv(
                raw_data,
                particle_analysis,
            )
            dm.save_figure(
                fig_probability,
                _append_to_file_path(file_path, "event-probability"),
            )
        except Exception:
            print("Could not save event probability plot:")
            print(traceback.format_exc())

        if save_images and "img_arrays" in raw_data:
            for run_ind in range(num_runs):
                try:
                    fig_map = plot_particle_event_map(
                        raw_data,
                        particle_analysis,
                        run_ind=run_ind,
                    )
                    dm.save_figure(
                        fig_map,
                        _append_to_file_path(file_path, f"event-map-run-{run_ind}"),
                    )
                except Exception:
                    print(f"Could not save event map for run {run_ind}:")
                    print(traceback.format_exc())
                    break

    tb.reset_cfm()
    raw_data["saved_file_path"] = str(file_path)
    return raw_data


# =============================================================================
# Convenience runners
# =============================================================================


def run_dark_wait_sweep(
    nv_list,
    dark_wait_values_s: Sequence[float],
    **main_kwargs,
) -> List[Dict[str, Any]]:
    """Run independent source-off/source-on datasets at several dark waits."""

    outputs = []
    for wait_s in dark_wait_values_s:
        print("\n" + "=" * 78)
        print(f"Running dark wait {float(wait_s):g} s")
        outputs.append(
            main(
                nv_list,
                dark_wait_s=float(wait_s),
                **main_kwargs,
            )
        )
    return outputs


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg false-discovery-rate correction.
    """

    p_values = np.asarray(p_values, dtype=float)
    num_tests = p_values.size

    order = np.argsort(p_values)
    ranked_p = p_values[order]

    ranked_q = (
        ranked_p
        * num_tests
        / np.arange(1, num_tests + 1)
    )

    # Enforce monotonic corrected p-values.
    ranked_q = np.minimum.accumulate(
        ranked_q[::-1]
    )[::-1]

    ranked_q = np.clip(
        ranked_q,
        0.0,
        1.0,
    )

    q_values = np.empty_like(ranked_q)
    q_values[order] = ranked_q

    return q_values


def analyze_and_plot_spatial_correlations(
    raw_data: Dict[str, Any],
    analysis: Dict[str, Any],
    coords_xy: np.ndarray,
    cluster_radius_px: float,
    num_permutations: int = 2000,
    random_seed: int = 12345,
    significance_level: float = 0.05,
    min_pair_repeats: int = 2,
    max_pairs_to_plot: int = 30,
):
    """
    Analyze and visualize spatial correlations between NV- -> NV0 events.

    The null model preserves, for each run:

        - the number of eligible NVs;
        - the number of candidate events.

    Candidate identities are randomized among eligible NVs.

    Returns
    -------
    result : dict
        Numerical spatial-correlation analysis.

    figures : list
        Aggregated Matplotlib figures.
    """

    coords_xy = np.asarray(
        coords_xy,
        dtype=float,
    )

    candidate_mask = np.asarray(
        analysis["candidate_nvm_to_nv0_mask"],
        dtype=bool,
    )

    eligible_mask = np.asarray(
        analysis["initial_nvm_mask"],
        dtype=bool,
    )

    num_nvs, num_runs = candidate_mask.shape

    if coords_xy.shape != (num_nvs, 2):
        raise ValueError(
            f"coords_xy must have shape {(num_nvs, 2)}; "
            f"got {coords_xy.shape}."
        )

    cluster_radius_px = float(
        cluster_radius_px
    )

    num_permutations = int(
        num_permutations
    )

    rng = np.random.default_rng(
        random_seed
    )

    # ==============================================================
    # Fixed geometry
    # ==============================================================

    displacement = (
        coords_xy[:, None, :]
        - coords_xy[None, :, :]
    )

    distance_matrix = np.sqrt(
        np.sum(
            displacement**2,
            axis=2,
        )
    )

    neighbor_matrix = (
        distance_matrix
        <= cluster_radius_px
    )

    np.fill_diagonal(
        neighbor_matrix,
        False,
    )

    def count_close_pairs(nv_inds):
        nv_inds = np.asarray(
            nv_inds,
            dtype=int,
        )

        if nv_inds.size < 2:
            return 0

        local_neighbors = neighbor_matrix[
            np.ix_(nv_inds, nv_inds)
        ]

        return int(
            np.sum(
                np.triu(
                    local_neighbors,
                    k=1,
                )
            )
        )

    # ==============================================================
    # Output arrays
    # ==============================================================

    runs = np.arange(num_runs)

    num_candidates_by_run = np.sum(
        candidate_mask,
        axis=0,
    )

    num_eligible_by_run = np.sum(
        eligible_mask,
        axis=0,
    )

    observed_close_pairs = np.zeros(
        num_runs,
        dtype=int,
    )

    expected_close_pairs = np.full(
        num_runs,
        np.nan,
        dtype=float,
    )

    std_close_pairs = np.full(
        num_runs,
        np.nan,
        dtype=float,
    )

    pair_p_values = np.ones(
        num_runs,
        dtype=float,
    )

    observed_max_cluster = np.zeros(
        num_runs,
        dtype=int,
    )

    expected_max_cluster = np.full(
        num_runs,
        np.nan,
        dtype=float,
    )

    cluster_p_values = np.ones(
        num_runs,
        dtype=float,
    )

    pair_z_scores = np.full(
        num_runs,
        np.nan,
        dtype=float,
    )

    null_close_pairs_all = np.zeros(
        (num_runs, num_permutations),
        dtype=np.int16,
    )

    null_max_cluster_all = np.zeros(
        (num_runs, num_permutations),
        dtype=np.int16,
    )

    pair_event_count = np.zeros(
        (num_nvs, num_nvs),
        dtype=np.int16,
    )

    cluster_participation_count_by_nv = np.zeros(
        num_nvs,
        dtype=int,
    )

    # Null distribution of total close pairs across all runs.
    global_null_close_pairs = np.zeros(
        num_permutations,
        dtype=int,
    )

    global_observed_close_pairs = 0

    # ==============================================================
    # Per-run randomization tests
    # ==============================================================

    for run_ind in range(num_runs):
        candidate_inds = np.where(
            candidate_mask[:, run_ind]
        )[0].astype(int)

        eligible_inds = np.where(
            eligible_mask[:, run_ind]
        )[0].astype(int)

        num_candidates = len(
            candidate_inds
        )

        observed_close_pairs[
            run_ind
        ] = count_close_pairs(
            candidate_inds
        )

        global_observed_close_pairs += (
            observed_close_pairs[run_ind]
        )

        observed_components = (
            _connected_components_within_radius(
                candidate_inds,
                coords_xy,
                radius=cluster_radius_px,
            )
        )

        observed_max_cluster[
            run_ind
        ] = max(
            (
                len(component)
                for component
                in observed_components
            ),
            default=0,
        )

        # Record candidate NVs belonging to multi-NV components.
        for component in observed_components:
            if len(component) >= 2:
                cluster_participation_count_by_nv[
                    np.asarray(
                        component,
                        dtype=int,
                    )
                ] += 1

        # Record nearby candidate-pair recurrence.
        if num_candidates >= 2:
            local_neighbors = neighbor_matrix[
                np.ix_(
                    candidate_inds,
                    candidate_inds,
                )
            ]

            row_local, col_local = np.where(
                np.triu(
                    local_neighbors,
                    k=1,
                )
            )

            for local_ind_1, local_ind_2 in zip(
                row_local,
                col_local,
            ):
                nv_ind_1 = int(
                    candidate_inds[local_ind_1]
                )
                nv_ind_2 = int(
                    candidate_inds[local_ind_2]
                )

                pair_event_count[
                    nv_ind_1,
                    nv_ind_2,
                ] += 1

                pair_event_count[
                    nv_ind_2,
                    nv_ind_1,
                ] += 1

        if (
            num_candidates < 2
            or len(eligible_inds) < num_candidates
        ):
            continue

        for permutation_ind in range(
            num_permutations
        ):
            random_inds = rng.choice(
                eligible_inds,
                size=num_candidates,
                replace=False,
            )

            random_close_pairs = (
                count_close_pairs(
                    random_inds
                )
            )

            null_close_pairs_all[
                run_ind,
                permutation_ind,
            ] = random_close_pairs

            random_components = (
                _connected_components_within_radius(
                    random_inds,
                    coords_xy,
                    radius=cluster_radius_px,
                )
            )

            random_max_cluster = max(
                (
                    len(component)
                    for component
                    in random_components
                ),
                default=0,
            )

            null_max_cluster_all[
                run_ind,
                permutation_ind,
            ] = random_max_cluster

        null_close_pairs = (
            null_close_pairs_all[run_ind]
        )

        null_max_clusters = (
            null_max_cluster_all[run_ind]
        )

        global_null_close_pairs += (
            null_close_pairs.astype(int)
        )

        expected_close_pairs[
            run_ind
        ] = float(
            np.mean(null_close_pairs)
        )

        std_close_pairs[
            run_ind
        ] = float(
            np.std(null_close_pairs)
        )

        expected_max_cluster[
            run_ind
        ] = float(
            np.mean(null_max_clusters)
        )

        pair_p_values[
            run_ind
        ] = (
            1
            + np.sum(
                null_close_pairs
                >= observed_close_pairs[run_ind]
            )
        ) / (
            num_permutations + 1
        )

        cluster_p_values[
            run_ind
        ] = (
            1
            + np.sum(
                null_max_clusters
                >= observed_max_cluster[run_ind]
            )
        ) / (
            num_permutations + 1
        )

        if std_close_pairs[run_ind] > 0:
            pair_z_scores[
                run_ind
            ] = (
                observed_close_pairs[run_ind]
                - expected_close_pairs[run_ind]
            ) / std_close_pairs[run_ind]

    # ==============================================================
    # Multiple testing and global test
    # ==============================================================

    pair_q_values = _benjamini_hochberg(
        pair_p_values
    )

    cluster_q_values = _benjamini_hochberg(
        cluster_p_values
    )

    global_pair_p_value = (
        1
        + np.sum(
            global_null_close_pairs
            >= global_observed_close_pairs
        )
    ) / (
        num_permutations + 1
    )

    nominal_significant_runs = np.where(
        pair_p_values < significance_level
    )[0].astype(int)

    fdr_significant_runs = np.where(
        pair_q_values < significance_level
    )[0].astype(int)

    # ==============================================================
    # Per-NV spatial statistics
    # ==============================================================

    event_count_by_nv = np.sum(
        candidate_mask,
        axis=1,
    )

    eligible_count_by_nv = np.sum(
        eligible_mask,
        axis=1,
    )

    event_probability_by_nv = np.full(
        num_nvs,
        np.nan,
        dtype=float,
    )

    good_nv = eligible_count_by_nv > 0

    event_probability_by_nv[
        good_nv
    ] = (
        event_count_by_nv[good_nv]
        / eligible_count_by_nv[good_nv]
    )

    cluster_probability_by_nv = np.zeros(
        num_nvs,
        dtype=float,
    )

    cluster_probability_by_nv[
        good_nv
    ] = (
        cluster_participation_count_by_nv[good_nv]
        / eligible_count_by_nv[good_nv]
    )

    # ==============================================================
    # Ranked repeated nearby pairs
    # ==============================================================

    upper_pair_rows, upper_pair_cols = np.where(
        np.triu(
            pair_event_count,
            k=1,
        )
        >= int(min_pair_repeats)
    )

    repeated_pair_records = []

    for nv_ind_1, nv_ind_2 in zip(
        upper_pair_rows,
        upper_pair_cols,
    ):
        count = int(
            pair_event_count[
                nv_ind_1,
                nv_ind_2,
            ]
        )

        distance = float(
            distance_matrix[
                nv_ind_1,
                nv_ind_2,
            ]
        )

        repeated_pair_records.append(
            {
                "nv_ind_1": int(nv_ind_1),
                "nv_ind_2": int(nv_ind_2),
                "count": count,
                "distance_px": distance,
            }
        )

    repeated_pair_records.sort(
        key=lambda record: record["count"],
        reverse=True,
    )

    # ==============================================================
    # Print numerical summary
    # ==============================================================

    print("\n" + "=" * 78)
    print("AGGREGATED SPATIAL-CORRELATION ANALYSIS")
    print("=" * 78)

    print("number of runs:", num_runs)
    print("cluster radius (px):", cluster_radius_px)
    print(
        "observed total close pairs:",
        global_observed_close_pairs,
    )
    print(
        "random expected total close pairs:",
        f"{np.mean(global_null_close_pairs):.2f} "
        f"+/- {np.std(global_null_close_pairs):.2f}",
    )
    print(
        "global close-pair p-value:",
        f"{global_pair_p_value:.5f}",
    )

    print(
        "\nNominal p < 0.05 runs "
        f"({len(nominal_significant_runs)}):"
    )
    print(
        nominal_significant_runs.tolist()
    )

    print(
        "\nFDR-corrected q < 0.05 runs "
        f"({len(fdr_significant_runs)}):"
    )
    print(
        fdr_significant_runs.tolist()
    )

    print(
        "\nTop repeated nearby pairs:"
    )

    for record in repeated_pair_records[:20]:
        print(
            f"NVs ({record['nv_ind_1']}, "
            f"{record['nv_ind_2']}): "
            f"{record['count']}/{num_runs} runs, "
            f"distance={record['distance_px']:.2f} px"
        )

    # ==============================================================
    # Figure 1: run-level overview
    # ==============================================================

    fig_runs, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    axes[0].plot(
        runs,
        num_candidates_by_run,
        "o-",
        markersize=3,
        linewidth=1,
        color=kpl.KplColors.RED,
        label="candidate events",
    )

    axes[0].set_ylabel("Candidates")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        runs,
        observed_close_pairs,
        "o",
        markersize=4,
        color=kpl.KplColors.RED,
        label="observed close pairs",
    )

    axes[1].plot(
        runs,
        expected_close_pairs,
        "-",
        linewidth=1.5,
        color=kpl.KplColors.BLUE,
        label="random expectation",
    )

    axes[1].fill_between(
        runs,
        expected_close_pairs
        - std_close_pairs,
        expected_close_pairs
        + std_close_pairs,
        color=kpl.KplColors.BLUE,
        alpha=0.18,
        label="random $\\pm 1\\sigma$",
    )

    axes[1].scatter(
        nominal_significant_runs,
        observed_close_pairs[
            nominal_significant_runs
        ],
        s=55,
        facecolors="none",
        edgecolors="orange",
        linewidths=1.4,
        label="nominal $p<0.05$",
    )

    if fdr_significant_runs.size > 0:
        axes[1].scatter(
            fdr_significant_runs,
            observed_close_pairs[
                fdr_significant_runs
            ],
            s=75,
            marker="*",
            color=kpl.KplColors.RED,
            label="FDR $q<0.05$",
        )

    axes[1].set_ylabel("Nearby pairs")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].semilogy(
        runs,
        pair_p_values,
        "o",
        markersize=4,
        color=kpl.KplColors.BLUE,
        label="per-run p-value",
    )

    axes[2].semilogy(
        runs,
        pair_q_values,
        ".",
        markersize=4,
        color=kpl.KplColors.RED,
        alpha=0.65,
        label="FDR q-value",
    )

    axes[2].axhline(
        significance_level,
        linestyle="--",
        linewidth=1.2,
        color=kpl.KplColors.GRAY,
        label=f"{significance_level:g}",
    )

    axes[2].set_xlabel("Run index")
    axes[2].set_ylabel("p or q value")
    axes[2].set_ylim(
        max(
            1.0 / (num_permutations + 1),
            1e-4,
        ),
        1.05,
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig_runs.suptitle(
        "Run-by-run spatial-correlation analysis\n"
        f"global close-pair p = {global_pair_p_value:.4g}",
        fontsize=14,
    )

    fig_runs.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    # ==============================================================
    # Figure 2: spatial event-frequency map
    # ==============================================================

    fig_spatial, ax = plt.subplots(
        figsize=(8, 7),
    )

    sc = ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        c=event_probability_by_nv,
        s=42,
        cmap="magma",
        vmin=0.0,
        vmax=max(
            0.05,
            float(
                np.nanpercentile(
                    event_probability_by_nv,
                    98,
                )
            ),
        ),
        edgecolors="none",
    )

    cluster_nv_mask = (
        cluster_participation_count_by_nv > 0
    )

    ax.scatter(
        coords_xy[
            cluster_nv_mask,
            0,
        ],
        coords_xy[
            cluster_nv_mask,
            1,
        ],
        s=78,
        facecolors="none",
        edgecolors=kpl.KplColors.RED,
        linewidths=1.1,
        label="participated in nearby-pair event",
    )

    cbar = fig_spatial.colorbar(
        sc,
        ax=ax,
    )

    cbar.set_label(
        "Candidate probability per eligible run"
    )

    ax.set_xlabel("Camera x pixel")
    ax.set_ylabel("Camera y pixel")
    ax.set_title(
        "Spatial distribution of NV$^- \\rightarrow$ NV$^0$ events"
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(
        fontsize=8,
        loc="upper right",
    )
    ax.grid(True, alpha=0.15)

    # ==============================================================
    # Figure 3: repeated nearby-pair network
    # ==============================================================

    fig_pairs, ax = plt.subplots(
        figsize=(8, 7),
    )

    ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        s=15,
        color=kpl.KplColors.GRAY,
        alpha=0.35,
    )

    pair_records_to_plot = repeated_pair_records[
        :int(max_pairs_to_plot)
    ]

    if pair_records_to_plot:
        max_pair_count = max(
            record["count"]
            for record
            in pair_records_to_plot
        )

        for record in pair_records_to_plot:
            nv_ind_1 = record["nv_ind_1"]
            nv_ind_2 = record["nv_ind_2"]
            count = record["count"]

            xy_1 = coords_xy[nv_ind_1]
            xy_2 = coords_xy[nv_ind_2]

            linewidth = (
                0.7
                + 3.0
                * count
                / max_pair_count
            )

            ax.plot(
                [xy_1[0], xy_2[0]],
                [xy_1[1], xy_2[1]],
                color=kpl.KplColors.RED,
                alpha=0.35
                + 0.55
                * count
                / max_pair_count,
                linewidth=linewidth,
            )

        pair_nv_inds = sorted(
            {
                record["nv_ind_1"]
                for record
                in pair_records_to_plot
            }
            | {
                record["nv_ind_2"]
                for record
                in pair_records_to_plot
            }
        )

        pair_nv_inds = np.asarray(
            pair_nv_inds,
            dtype=int,
        )

        ax.scatter(
            coords_xy[pair_nv_inds, 0],
            coords_xy[pair_nv_inds, 1],
            s=55,
            facecolors="none",
            edgecolors=kpl.KplColors.RED,
            linewidths=1.2,
        )

    else:
        ax.text(
            0.5,
            0.5,
            f"No nearby pair repeated in "
            f"{min_pair_repeats} or more runs",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel("Camera x pixel")
    ax.set_ylabel("Camera y pixel")
    ax.set_title(
        f"Repeated nearby candidate pairs\n"
        f"minimum recurrence = {min_pair_repeats} runs"
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

    # ==============================================================
    # Figure 4: p-value diagnostics and strongest run
    # ==============================================================

    fig_pvalues, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
    )

    bins = np.linspace(
        0.0,
        1.0,
        11,
    )

    axes[0].hist(
        pair_p_values,
        bins=bins,
        alpha=0.45,
        edgecolor=kpl.KplColors.BLUE,
        color=kpl.KplColors.BLUE,
    )

    expected_per_bin = (
        num_runs
        / (len(bins) - 1)
    )

    axes[0].axhline(
        expected_per_bin,
        linestyle="--",
        color=kpl.KplColors.GRAY,
        linewidth=1.2,
        label="uniform-null expectation",
    )

    axes[0].set_xlabel("Per-run close-pair p-value")
    axes[0].set_ylabel("Number of runs")
    axes[0].set_title("P-value distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    strongest_run = int(
        np.argmin(pair_p_values)
    )

    strongest_null = (
        null_close_pairs_all[
            strongest_run
        ]
    )

    hist_min = int(
        min(
            np.min(strongest_null),
            observed_close_pairs[
                strongest_run
            ],
        )
    )

    hist_max = int(
        max(
            np.max(strongest_null),
            observed_close_pairs[
                strongest_run
            ],
        )
    )

    strongest_bins = np.arange(
        hist_min - 0.5,
        hist_max + 1.5,
        1.0,
    )

    axes[1].hist(
        strongest_null,
        bins=strongest_bins,
        alpha=0.45,
        color=kpl.KplColors.BLUE,
        edgecolor=kpl.KplColors.BLUE,
        label="randomized runs",
    )

    axes[1].axvline(
        observed_close_pairs[
            strongest_run
        ],
        color=kpl.KplColors.RED,
        linewidth=2,
        label=(
            f"observed = "
            f"{observed_close_pairs[strongest_run]}"
        ),
    )

    axes[1].set_xlabel("Number of nearby pairs")
    axes[1].set_ylabel("Randomizations")
    axes[1].set_title(
        f"Most anomalous run: {strongest_run}\n"
        f"p={pair_p_values[strongest_run]:.4g}, "
        f"q={pair_q_values[strongest_run]:.4g}"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    # ==============================================================
    # Return results
    # ==============================================================

    result = {
        "cluster_radius_px": (
            cluster_radius_px
        ),
        "num_permutations": (
            num_permutations
        ),
        "num_candidates_by_run": (
            num_candidates_by_run.tolist()
        ),
        "num_eligible_by_run": (
            num_eligible_by_run.tolist()
        ),
        "observed_close_pairs": (
            observed_close_pairs.tolist()
        ),
        "expected_close_pairs": (
            expected_close_pairs.tolist()
        ),
        "std_close_pairs": (
            std_close_pairs.tolist()
        ),
        "pair_p_values": (
            pair_p_values.tolist()
        ),
        "pair_q_values": (
            pair_q_values.tolist()
        ),
        "pair_z_scores": (
            pair_z_scores.tolist()
        ),
        "observed_max_cluster": (
            observed_max_cluster.tolist()
        ),
        "expected_max_cluster": (
            expected_max_cluster.tolist()
        ),
        "cluster_p_values": (
            cluster_p_values.tolist()
        ),
        "cluster_q_values": (
            cluster_q_values.tolist()
        ),
        "global_observed_close_pairs": int(
            global_observed_close_pairs
        ),
        "global_expected_close_pairs": float(
            np.mean(
                global_null_close_pairs
            )
        ),
        "global_std_close_pairs": float(
            np.std(
                global_null_close_pairs
            )
        ),
        "global_pair_p_value": float(
            global_pair_p_value
        ),
        "nominal_significant_runs": (
            nominal_significant_runs.tolist()
        ),
        "fdr_significant_runs": (
            fdr_significant_runs.tolist()
        ),
        "event_count_by_nv": (
            event_count_by_nv.tolist()
        ),
        "event_probability_by_nv": (
            event_probability_by_nv.tolist()
        ),
        "cluster_participation_count_by_nv": (
            cluster_participation_count_by_nv.tolist()
        ),
        "cluster_probability_by_nv": (
            cluster_probability_by_nv.tolist()
        ),
        "repeated_pair_records": (
            repeated_pair_records
        ),
        "strongest_run": strongest_run,
    }

    figures = [
        fig_runs,
        fig_spatial,
        fig_pairs,
        fig_pvalues,
    ]

    return result, figures

#########################
"""
Compare adaptive particle-memory datasets acquired at different dark wait times.

The zero-wait dataset is used as the measurement/readout baseline.  The code
reports both the raw NV- retention and the additional dark-time survival
relative to that baseline:

    dark_survival(t) = retention(t) / retention(0)

The normalized dark survival is fit to

    D(t) = D_inf + (1 - D_inf) exp(-t / tau_dark)

This separation assumes that the zero-wait loss and dark-time loss combine
multiplicatively.
"""

def _mean_and_sem(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return mean, sem


def _pooled_fraction(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    good = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    if not np.any(good):
        return np.nan
    total_denominator = float(np.sum(denominator[good]))
    if total_denominator <= 0:
        return np.nan
    return float(np.sum(numerator[good]) / total_denominator)


def _dark_survival_model(wait_s, plateau, tau_s):
    """D(t) = D_inf + (1-D_inf) exp(-t/tau)."""
    wait_s = np.asarray(wait_s, dtype=float)
    return plateau + (1.0 - plateau) * np.exp(-wait_s / tau_s)


def _fit_dark_survival(
    wait_s: np.ndarray,
    dark_survival: np.ndarray,
    dark_survival_sem: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    wait_s = np.asarray(wait_s, dtype=float)
    dark_survival = np.asarray(dark_survival, dtype=float)

    valid = np.isfinite(wait_s) & np.isfinite(dark_survival) & (wait_s >= 0)
    x = wait_s[valid]
    y = dark_survival[valid]

    if x.size < 4 or np.unique(x).size < 4:
        return {
            "success": False,
            "error": "At least four distinct wait times are required.",
        }

    sigma = None
    if dark_survival_sem is not None:
        sem = np.asarray(dark_survival_sem, dtype=float)[valid]
        positive = np.isfinite(sem) & (sem > 0)
        if np.any(positive):
            replacement = float(np.nanmedian(sem[positive]))
            sigma = np.where(positive, sem, replacement)

    positive_x = x[x > 0]
    tau_guess = float(np.median(positive_x)) if positive_x.size else 60.0
    plateau_guess = float(np.clip(np.nanmin(y), 0.0, 0.99))

    popt, pcov = curve_fit(
        _dark_survival_model,
        x,
        y,
        p0=(plateau_guess, tau_guess),
        bounds=([0.0, 1e-9], [1.0, np.inf]),
        sigma=sigma,
        absolute_sigma=sigma is not None,
        maxfev=50000,
    )

    plateau, tau_s = map(float, popt)
    stderr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    plateau_stderr, tau_s_stderr = map(float, stderr)

    fitted = _dark_survival_model(x, plateau, tau_s)
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    max_wait_s = float(np.max(x))
    min_positive_wait_s = float(np.min(x[x > 0]))

    dense_positive_x = np.geomspace(
        max(0.1, min_positive_wait_s / 20),
        max_wait_s,
        500,
    )

    dense_x = np.concatenate(
        [
            [0.0],
            dense_positive_x,
        ]
    )

    dense_y = _dark_survival_model(
        dense_x,
        plateau,
        tau_s,
    )

    return {
        "success": True,
        "model": "D_inf + (1-D_inf) exp(-t/tau_dark)",
        "plateau": plateau,
        "plateau_stderr": plateau_stderr,
        "tau_dark_s": tau_s,
        "tau_dark_s_stderr": tau_s_stderr,
        "tau_dark_min": tau_s / 60.0,
        "tau_dark_min_stderr": tau_s_stderr / 60.0,
        "r_squared": r_squared,
        "fit_x_s": dense_x.tolist(),
        "fit_y": dense_y.tolist(),
    }


def load_wait_sweep(
    file_stems: Sequence[str],
    recompute_analysis: bool = False,
    initial_margin_counts: float = 1.0,
    final_margin_counts: float = 1.0,
    cluster_radius_px: Optional[float] = None,
    min_cluster_size: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load each raw dataset and return raw data plus particle analyses."""

    raw_datasets: List[Dict[str, Any]] = []
    analyses: List[Dict[str, Any]] = []

    for file_stem in file_stems:
        print("\nLoading:", file_stem)
        raw_data = dm.get_raw_data(
            file_stem=file_stem,
            load_npz=True,
        )

        analysis = raw_data.get("particle_analysis")
        if analysis is None or recompute_analysis:
            analysis = analyze_particle_charge_memory(
                raw_data,
                initial_margin_counts=initial_margin_counts,
                final_margin_counts=final_margin_counts,
                cluster_radius_px=cluster_radius_px,
                min_cluster_size=min_cluster_size,
            )
            raw_data["particle_analysis"] = analysis

        raw_datasets.append(raw_data)
        analyses.append(analysis)

    order = np.argsort(
        [float(analysis["dark_wait_s"]) for analysis in analyses]
    )
    raw_datasets = [raw_datasets[ind] for ind in order]
    analyses = [analyses[ind] for ind in order]
    return raw_datasets, analyses


def summarize_wait_sweep(
    analyses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine run-level statistics across independently saved wait datasets."""

    if not analyses:
        raise ValueError("No analyses were supplied.")

    rows: List[Dict[str, Any]] = []

    for analysis in analyses:
        wait_s = float(analysis["dark_wait_s"])
        num_nvs = int(analysis["num_nvs"])

        initial = np.asarray(analysis["num_initial_nvm_by_run"], dtype=float)
        final = np.asarray(analysis["num_final_nvm_by_run"], dtype=float)
        retained = np.asarray(analysis["num_retained_by_run"], dtype=float)
        candidates = np.asarray(analysis["num_candidates_by_run"], dtype=float)
        ambiguous = np.asarray(analysis["num_ambiguous_by_run"], dtype=float)
        retention_by_run = np.asarray(analysis["retention_by_run"], dtype=float)
        event_by_run = np.asarray(analysis["event_fraction_by_run"], dtype=float)

        retention_mean, retention_sem = _mean_and_sem(retention_by_run)
        event_mean, event_sem = _mean_and_sem(event_by_run)

        ambiguous_by_run = np.full(initial.shape, np.nan, dtype=float)
        good = initial > 0
        ambiguous_by_run[good] = ambiguous[good] / initial[good]
        ambiguous_mean, ambiguous_sem = _mean_and_sem(ambiguous_by_run)

        rows.append(
            {
                "dark_wait_s": wait_s,
                "dark_wait_min": wait_s / 60.0,
                "num_nvs": num_nvs,
                "num_runs": int(analysis["num_runs"]),
                "mean_initial_nvm": float(np.mean(initial)),
                "mean_final_nvm": float(np.mean(final)),
                "initial_nvm_fraction": float(np.mean(initial) / num_nvs),
                "final_nvm_fraction": float(np.mean(final) / num_nvs),
                "retention_mean_by_run": retention_mean,
                "retention_sem_by_run": retention_sem,
                "retention_pooled": _pooled_fraction(retained, initial),
                "event_fraction_mean_by_run": event_mean,
                "event_fraction_sem_by_run": event_sem,
                "event_fraction_pooled": _pooled_fraction(candidates, initial),
                "ambiguous_fraction_mean_by_run": ambiguous_mean,
                "ambiguous_fraction_sem_by_run": ambiguous_sem,
                "ambiguous_fraction_pooled": _pooled_fraction(ambiguous, initial),
                "mean_candidates_per_run": float(np.mean(candidates)),
                "mean_ambiguous_per_run": float(np.mean(ambiguous)),
            }
        )

    rows.sort(key=lambda row: row["dark_wait_s"])

    wait_s = np.asarray([row["dark_wait_s"] for row in rows], dtype=float)
    retention = np.asarray([row["retention_pooled"] for row in rows], dtype=float)
    retention_sem = np.asarray(
        [row["retention_sem_by_run"] for row in rows],
        dtype=float,
    )

    zero_inds = np.where(np.isclose(wait_s, 0.0))[0]
    if zero_inds.size == 0:
        raise ValueError(
            "A zero-wait dataset is required to separate the readout baseline."
        )

    zero_ind = int(zero_inds[0])
    baseline_retention = float(retention[zero_ind])
    baseline_sem = float(retention_sem[zero_ind])

    if not np.isfinite(baseline_retention) or baseline_retention <= 0:
        raise ValueError("The zero-wait retention is invalid.")

    dark_survival = retention / baseline_retention
    additional_dark_loss = 1.0 - dark_survival

    # Approximate independent-error propagation for the normalized quantity.
    dark_survival_sem = np.full_like(dark_survival, np.nan)
    for ind, (value, value_sem) in enumerate(zip(retention, retention_sem)):
        if not np.isfinite(value) or value <= 0:
            continue
        relative_variance = 0.0
        if np.isfinite(value_sem):
            relative_variance += (value_sem / value) ** 2
        if np.isfinite(baseline_sem):
            relative_variance += (baseline_sem / baseline_retention) ** 2
        dark_survival_sem[ind] = dark_survival[ind] * np.sqrt(relative_variance)

    # The zero-wait ratio is exactly one by definition.
    dark_survival[zero_ind] = 1.0
    dark_survival_sem[zero_ind] = 0.0
    additional_dark_loss[zero_ind] = 0.0

    fit = _fit_dark_survival(
        wait_s,
        dark_survival,
        dark_survival_sem=dark_survival_sem,
    )

    for ind, row in enumerate(rows):
        row["dark_survival_relative_to_zero"] = float(dark_survival[ind])
        row["dark_survival_sem"] = float(dark_survival_sem[ind])
        row["additional_dark_loss"] = float(additional_dark_loss[ind])

    return {
        "analysis_type": "particle_memory_dark_wait_sweep",
        "rows": rows,
        "wait_s": wait_s.tolist(),
        "zero_wait_retention": baseline_retention,
        "zero_wait_retention_sem": baseline_sem,
        "dark_survival": dark_survival.tolist(),
        "dark_survival_sem": dark_survival_sem.tolist(),
        "additional_dark_loss": additional_dark_loss.tolist(),
        "fit": fit,
        "interpretation_note": (
            "The zero-wait retention contains immediate/readout-induced loss. "
            "Dark survival divides each retention value by the zero-wait "
            "retention and therefore estimates additional wait-dependent loss "
            "under a multiplicative-baseline assumption."
        ),
    }


def _set_wait_axis(
    ax,
    wait_s: np.ndarray,
    linthresh_s: float = 10.0,
):
    """
    Log-like time axis that retains the zero-wait point.
    """

    wait_s = np.asarray(wait_s, dtype=float)

    ax.set_xscale(
        "symlog",
        linthresh=linthresh_s,
        linscale=1.0,
        base=10,
    )

    ax.set_xticks(wait_s)

    tick_labels = []
    for value in wait_s:
        if value == 0:
            label = "0"
        elif value < 60:
            label = f"{value:g} s"
        else:
            label = f"{value / 60:g} min"

        tick_labels.append(label)

    ax.set_xticklabels(
        tick_labels,
        rotation=30,
        ha="right",
    )

    max_wait = float(np.nanmax(wait_s))

    ax.set_xlim(
        -0.5 * linthresh_s,
        max_wait * 1.15,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )


def _set_zoomed_fraction_ylim(
    ax,
    values: np.ndarray,
    errors: np.ndarray | None = None,
    lower_limit: float = 0.0,
    upper_limit: float = 1.02,
    minimum_span: float = 0.05,
    padding_fraction: float = 0.20,
):
    """
    Set a linear but zoomed y-range for fractions close to one.
    """

    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)

    if not np.any(good):
        ax.set_ylim(lower_limit, upper_limit)
        return

    if errors is None:
        low_values = values[good]
        high_values = values[good]
    else:
        errors = np.asarray(errors, dtype=float)

        if errors.shape != values.shape:
            raise ValueError(
                "errors and values must have matching shapes."
            )

        finite_errors = np.where(
            np.isfinite(errors),
            errors,
            0.0,
        )

        low_values = (
            values[good]
            - finite_errors[good]
        )
        high_values = (
            values[good]
            + finite_errors[good]
        )

    data_min = float(np.nanmin(low_values))
    data_max = float(np.nanmax(high_values))

    span = max(
        data_max - data_min,
        float(minimum_span),
    )

    padding = float(padding_fraction) * span

    y_min = max(
        float(lower_limit),
        data_min - padding,
    )
    y_max = min(
        float(upper_limit),
        data_max + padding,
    )

    if y_max - y_min < minimum_span:
        center = 0.5 * (y_min + y_max)
        y_min = max(
            lower_limit,
            center - minimum_span / 2,
        )
        y_max = min(
            upper_limit,
            center + minimum_span / 2,
        )

    ax.set_ylim(y_min, y_max)


def plot_wait_sweep_summary(
    summary: Dict[str, Any],
    zoom_retention_axes: bool = True,
):
    """
    Create the main four-panel dark-wait comparison figure.

    Axis choices
    ------------
    x-axis:
        Symmetric logarithmic scale. This preserves the 0 s point while
        spreading out 10, 30, 60, 180, 300, and 600 s.

    y-axis:
        Linear for all panels because retention and survival are bounded
        fractions and may include values near zero.
    """

    rows = summary["rows"]

    wait_s = np.asarray(
        summary["wait_s"],
        dtype=float,
    )

    retention = np.asarray(
        [
            row["retention_pooled"]
            for row in rows
        ],
        dtype=float,
    )

    retention_sem = np.asarray(
        [
            row["retention_sem_by_run"]
            for row in rows
        ],
        dtype=float,
    )

    dark_survival = np.asarray(
        summary["dark_survival"],
        dtype=float,
    )

    dark_survival_sem = np.asarray(
        summary["dark_survival_sem"],
        dtype=float,
    )

    event_fraction = np.asarray(
        [
            row["event_fraction_pooled"]
            for row in rows
        ],
        dtype=float,
    )

    event_sem = np.asarray(
        [
            row["event_fraction_sem_by_run"]
            for row in rows
        ],
        dtype=float,
    )

    ambiguous_fraction = np.asarray(
        [
            row["ambiguous_fraction_pooled"]
            for row in rows
        ],
        dtype=float,
    )

    ambiguous_sem = np.asarray(
        [
            row["ambiguous_fraction_sem_by_run"]
            for row in rows
        ],
        dtype=float,
    )

    initial_fraction = np.asarray(
        [
            row["initial_nvm_fraction"]
            for row in rows
        ],
        dtype=float,
    )

    final_fraction = np.asarray(
        [
            row["final_nvm_fraction"]
            for row in rows
        ],
        dtype=float,
    )

    # Sort everything by wait time for safe plotting.
    sort_inds = np.argsort(wait_s)

    wait_s = wait_s[sort_inds]
    retention = retention[sort_inds]
    retention_sem = retention_sem[sort_inds]
    dark_survival = dark_survival[sort_inds]
    dark_survival_sem = dark_survival_sem[sort_inds]
    event_fraction = event_fraction[sort_inds]
    event_sem = event_sem[sort_inds]
    ambiguous_fraction = ambiguous_fraction[sort_inds]
    ambiguous_sem = ambiguous_sem[sort_inds]
    initial_fraction = initial_fraction[sort_inds]
    final_fraction = final_fraction[sort_inds]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8.5),
    )

    # ==============================================================
    # Panel 1: measured retention
    # ==============================================================

    ax = axes[0, 0]

    ax.errorbar(
        wait_s,
        retention,
        yerr=retention_sem,
        fmt="o-",
        linewidth=1.7,
        markersize=6,
        capsize=3,
        color=kpl.KplColors.BLUE,
        label="Measured NV$^-$ retention",
    )

    zero_wait_retention = float(
        summary["zero_wait_retention"]
    )

    ax.axhline(
        zero_wait_retention,
        linestyle="--",
        linewidth=1.4,
        color=kpl.KplColors.GRAY,
        label=(
            f"0 s baseline = "
            f"{zero_wait_retention:.3f}"
        ),
    )

    ax.set_ylabel(
        "Retention after final readout"
    )

    ax.set_title(
        "Measured retention versus dark wait"
    )

    ax.legend(
        fontsize=8,
    )

    if zoom_retention_axes:
        _set_zoomed_fraction_ylim(
            ax,
            retention,
            retention_sem,
            lower_limit=0.0,
            upper_limit=1.02,
            minimum_span=0.05,
        )
    else:
        ax.set_ylim(
            0.0,
            1.02,
        )

    # ==============================================================
    # Panel 2: baseline-normalized dark survival
    # ==============================================================

    ax = axes[0, 1]

    ax.errorbar(
        wait_s,
        dark_survival,
        yerr=dark_survival_sem,
        fmt="o",
        markersize=6,
        capsize=3,
        color=kpl.KplColors.GREEN,
        label="Retention / 0 s retention",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        color=kpl.KplColors.GRAY,
        label="No additional dark loss",
    )

    fit = summary.get(
        "fit",
        {},
    )

    if fit.get("success", False):
        fit_x_s = np.asarray(
            fit["fit_x_s"],
            dtype=float,
        )

        fit_y = np.asarray(
            fit["fit_y"],
            dtype=float,
        )

        fit_sort_inds = np.argsort(
            fit_x_s
        )

        ax.plot(
            fit_x_s[fit_sort_inds],
            fit_y[fit_sort_inds],
            "-",
            linewidth=2,
            color=kpl.KplColors.GREEN,
            label=(
                f"$\\tau_{{dark}}$ = "
                f"{fit['tau_dark_min']:.2g} ± "
                f"{fit['tau_dark_min_stderr']:.1g} min\n"
                f"$D_\\infty$ = "
                f"{fit['plateau']:.3f}"
            ),
        )

    ax.set_ylabel(
        "Dark survival relative to 0 s"
    )
    

    ax.set_title(
        "Additional dark-time evolution"
    )

    ax.legend(
        fontsize=8,
    )

    if zoom_retention_axes:
        combined_values = dark_survival.copy()
        combined_errors = dark_survival_sem.copy()

        if fit.get("success", False):
            combined_values = np.concatenate(
                [
                    combined_values,
                    np.asarray(
                        fit["fit_y"],
                        dtype=float,
                    ),
                ]
            )

            combined_errors = np.concatenate(
                [
                    combined_errors,
                    np.zeros(
                        len(fit["fit_y"]),
                        dtype=float,
                    ),
                ]
            )

        _set_zoomed_fraction_ylim(
            ax,
            combined_values,
            combined_errors,
            lower_limit=0.0,
            upper_limit=1.08,
            minimum_span=0.05,
        )
    else:
        ax.set_ylim(
            0.0,
            1.08,
        )

    # ==============================================================
    # Panel 3: confident and ambiguous loss
    # ==============================================================

    ax = axes[1, 0]

    ax.errorbar(
        wait_s,
        event_fraction,
        yerr=event_sem,
        fmt="o-",
        linewidth=1.7,
        markersize=6,
        capsize=3,
        color=kpl.KplColors.RED,
        label=(
            "Confident NV$^- \\rightarrow$ NV$^0$"
        ),
    )

    ax.errorbar(
        wait_s,
        ambiguous_fraction,
        yerr=ambiguous_sem,
        fmt="o-",
        linewidth=1.7,
        markersize=6,
        capsize=3,
        color=kpl.KplColors.GRAY,
        label="Ambiguous final state",
    )

    ax.set_xlabel(
        "Dark wait (s)"
    )

    ax.set_ylabel(
        "Fraction of initially verified NV$^-$"
    )

    ax.set_ylim(
        bottom=0.0,
    )

    ax.set_title(
        "Loss classification versus wait"
    )

    ax.legend(
        fontsize=8,
    )

    # ==============================================================
    # Panel 4: preparation and final populations
    # ==============================================================

    ax = axes[1, 1]

    ax.plot(
        wait_s,
        initial_fraction,
        "o-",
        linewidth=1.7,
        markersize=6,
        color=kpl.KplColors.GREEN,
        label="Initial verified NV$^-$ / all NVs",
    )

    ax.plot(
        wait_s,
        final_fraction,
        "o-",
        linewidth=1.7,
        markersize=6,
        color=kpl.KplColors.BLUE,
        label="Final NV$^-$ / all NVs",
    )

    ax.set_xlabel(
        "Dark wait (s)"
    )

    ax.set_ylabel(
        "Fraction of tracked NVs"
    )

    ax.set_title(
        "Preparation and final populations"
    )

    ax.legend(
        fontsize=8,
    )

    if zoom_retention_axes:
        population_values = np.concatenate(
            [
                initial_fraction,
                final_fraction,
            ]
        )

        _set_zoomed_fraction_ylim(
            ax,
            population_values,
            errors=None,
            lower_limit=0.0,
            upper_limit=1.02,
            minimum_span=0.08,
        )
    else:
        ax.set_ylim(
            0.0,
            1.02,
        )

    # ==============================================================
    # Shared x-axis formatting
    # ==============================================================

    for ax in axes.flat:
        _set_wait_axis(
            ax,
            wait_s,
            linthresh_s=10.0,
        )

    fig.suptitle(
        "Adaptive charge memory versus dark wait\n"
        "The 0 s data provide the measurement-loss baseline",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94],
    )

    return fig

def print_wait_sweep_table(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print("DARK-WAIT SWEEP SUMMARY")
    print("=" * 96)
    print(
        f"{'wait(s)':>8}  {'runs':>5}  {'initial':>9}  {'retention':>10}  "
        f"{'event':>9}  {'ambig.':>9}  {'dark surv.':>11}  {'dark loss':>10}"
    )

    for row in summary["rows"]:
        print(
            f"{row['dark_wait_s']:8.0f}  "
            f"{row['num_runs']:5d}  "
            f"{row['initial_nvm_fraction']:9.4f}  "
            f"{row['retention_pooled']:10.4f}  "
            f"{row['event_fraction_pooled']:9.4f}  "
            f"{row['ambiguous_fraction_pooled']:9.4f}  "
            f"{row['dark_survival_relative_to_zero']:11.4f}  "
            f"{row['additional_dark_loss']:10.4f}"
        )

    fit = summary.get("fit", {})
    if fit.get("success", False):
        print("\nFit to normalized dark survival:")
        print(
            f"tau_dark = {fit['tau_dark_s']:.3g} ± "
            f"{fit['tau_dark_s_stderr']:.2g} s "
            f"({fit['tau_dark_min']:.3g} ± "
            f"{fit['tau_dark_min_stderr']:.2g} min)"
        )
        print(
            f"D_inf = {fit['plateau']:.4f} ± "
            f"{fit['plateau_stderr']:.2g}; R² = {fit['r_squared']:.4f}"
        )
    else:
        print("\nFit unavailable:", fit.get("error", "unknown error"))


def save_wait_sweep_csv(summary: Dict[str, Any], csv_path: Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = summary["rows"]
    if not rows:
        raise ValueError("No rows to save.")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved wait-sweep CSV:", csv_path)
    return csv_path


def run_particle_memory_dark_wait_comparison_analysis(
    file_stems: Sequence[str] = None,
    recompute_analysis: bool = False,
    save_fig: bool = True,
    save_csv: bool = True,
):
    raw_datasets, analyses = load_wait_sweep(
        file_stems,
        recompute_analysis=recompute_analysis,
    )
    summary = summarize_wait_sweep(analyses)
    print_wait_sweep_table(summary)
    # fig = plot_wait_sweep_summary(summary)
    fig = plot_wait_sweep_summary(summary, zoom_retention_axes=True,)


    timestamp = dm.get_time_stamp()
    output_path = dm.get_file_path(
        __file__,
        timestamp,
        "particle-memory-dark-wait-comparison",
    )

    if save_fig:
        dm.save_figure(fig, output_path)
        print("Saved wait-sweep figure:", output_path)

    if save_csv:
        # dm paths may have no extension; save a nearby explicit CSV.
        csv_path = Path(str(output_path) + ".csv")
        save_wait_sweep_csv(summary, csv_path)

    return {
        "raw_datasets": raw_datasets,
        "analyses": analyses,
        "summary": summary,
        "fig": fig,
    }

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


def plot_nv_loss_row_by_row(
    analyses: Sequence[Dict[str, Any]],
    selected_waits_s: Optional[Sequence[float]] = None,
    subtract_zero_wait: bool = False,
    show_percent: bool = True,
    percentile_limit: float = 99.0,
):
    """
    Plot per-NV charge-loss probability as a wait-time × NV-index heat map.

    Rows:
        Dark-wait datasets.

    Columns:
        NV indices.

    Color:
        Probability of a confident NV- -> NV0 transition.

    Parameters
    ----------
    analyses:
        Sequence of particle_analysis dictionaries.

    selected_waits_s:
        Optional wait times to include, such as:
            [0, 60, 300, 600, 1800]

        None includes all wait times.

    subtract_zero_wait:
        False:
            Plot the absolute loss probability.

        True:
            Plot excess loss relative to the 0 s dataset.

    show_percent:
        Show probability in percent instead of fractions.

    percentile_limit:
        Upper color limit percentile. This prevents a few unstable NVs
        from making the rest of the heat map appear dark.
    """

    if not analyses:
        raise ValueError("No analyses were supplied.")

    analyses = sorted(
        analyses,
        key=lambda analysis: float(analysis["dark_wait_s"]),
    )

    # --------------------------------------------------------------
    # Select requested wait times
    # --------------------------------------------------------------

    if selected_waits_s is not None:
        selected_analyses = []

        for requested_wait in selected_waits_s:
            closest_analysis = min(
                analyses,
                key=lambda analysis: abs(
                    float(analysis["dark_wait_s"])
                    - float(requested_wait)
                ),
            )

            if closest_analysis not in selected_analyses:
                selected_analyses.append(closest_analysis)

        analyses = sorted(
            selected_analyses,
            key=lambda analysis: float(analysis["dark_wait_s"]),
        )

    wait_s = np.asarray(
        [
            float(analysis["dark_wait_s"])
            for analysis in analyses
        ],
        dtype=float,
    )

    probability_rows = []
    eligible_rows = []
    event_rows = []

    reference_num_nvs = None

    # --------------------------------------------------------------
    # Calculate probability for each NV at each wait time
    # --------------------------------------------------------------

    for analysis in analyses:
        initial_nvm_mask = np.asarray(
            analysis["initial_nvm_mask"],
            dtype=bool,
        )

        event_mask = np.asarray(
            analysis["candidate_nvm_to_nv0_mask"],
            dtype=bool,
        )

        if initial_nvm_mask.shape != event_mask.shape:
            raise ValueError(
                "initial_nvm_mask and candidate mask shapes differ."
            )

        num_nvs = initial_nvm_mask.shape[0]

        if reference_num_nvs is None:
            reference_num_nvs = num_nvs
        elif num_nvs != reference_num_nvs:
            raise ValueError(
                "The datasets contain different numbers of NVs."
            )

        eligible_count = np.sum(
            initial_nvm_mask,
            axis=1,
        ).astype(float)

        event_count = np.sum(
            event_mask,
            axis=1,
        ).astype(float)

        probability = np.full(
            num_nvs,
            np.nan,
            dtype=float,
        )

        good = eligible_count > 0

        probability[good] = (
            event_count[good]
            / eligible_count[good]
        )

        probability_rows.append(probability)
        eligible_rows.append(eligible_count)
        event_rows.append(event_count)

    probability_matrix = np.asarray(
        probability_rows,
        dtype=float,
    )  # shape = [wait, NV]

    eligible_matrix = np.asarray(
        eligible_rows,
        dtype=float,
    )

    event_matrix = np.asarray(
        event_rows,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Optional zero-wait subtraction
    # --------------------------------------------------------------

    if subtract_zero_wait:
        zero_inds = np.where(
            np.isclose(wait_s, 0.0)
        )[0]

        if zero_inds.size == 0:
            raise ValueError(
                "A 0 s dataset is required for baseline subtraction."
            )

        zero_probability = probability_matrix[
            int(zero_inds[0])
        ]

        plot_matrix = (
            probability_matrix
            - zero_probability[None, :]
        )

        title = (
            "Per-NV excess charge loss above the 0 s baseline"
        )

        colorbar_label = (
            "Excess loss probability"
        )
    else:
        plot_matrix = probability_matrix.copy()

        title = (
            "Per-NV charge loss at different dark waits"
        )

        colorbar_label = (
            "NV$^- \\rightarrow$ NV$^0$ probability"
        )

    if show_percent:
        plot_matrix = 100.0 * plot_matrix
        colorbar_label += " (%)"

    # Mask NVs that were never initially verified.
    masked_matrix = np.ma.masked_invalid(
        plot_matrix
    )

    finite_values = plot_matrix[
        np.isfinite(plot_matrix)
    ]

    if finite_values.size == 0:
        raise ValueError(
            "No finite loss probabilities were found."
        )

    # --------------------------------------------------------------
    # Color normalization
    # --------------------------------------------------------------

    if subtract_zero_wait:
        color_limit = float(
            np.nanpercentile(
                np.abs(finite_values),
                percentile_limit,
            )
        )

        color_limit = max(
            color_limit,
            0.1 if show_percent else 0.001,
        )

        norm = TwoSlopeNorm(
            vmin=-color_limit,
            vcenter=0.0,
            vmax=color_limit,
        )

        cmap = plt.get_cmap(
            "coolwarm"
        ).copy()

    else:
        color_limit = float(
            np.nanpercentile(
                finite_values,
                percentile_limit,
            )
        )

        color_limit = max(
            color_limit,
            0.1 if show_percent else 0.001,
        )

        norm = Normalize(
            vmin=0.0,
            vmax=color_limit,
        )

        cmap = plt.get_cmap(
            "magma"
        ).copy()

    cmap.set_bad(
        "lightgray"
    )

    # --------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------

    num_waits, num_nvs = plot_matrix.shape

    fig_height = max(
        4.0,
        0.65 * num_waits + 2.0,
    )

    fig, ax = plt.subplots(
        figsize=(15, fig_height),
        constrained_layout=True,
    )

    image = ax.imshow(
        masked_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        origin="upper",
    )

    wait_labels = []

    for current_wait_s in wait_s:
        if np.isclose(current_wait_s, 0.0):
            label = "0 s"
        elif current_wait_s < 60:
            label = f"{current_wait_s:g} s"
        else:
            label = f"{current_wait_s / 60:g} min"

        wait_labels.append(label)

    ax.set_yticks(
        np.arange(num_waits)
    )

    ax.set_yticklabels(
        wait_labels
    )

    ax.set_xlabel(
        "NV index"
    )

    ax.set_ylabel(
        "Dark wait"
    )

    ax.set_title(
        title
    )

    # Horizontal lines make each time row easier to identify.
    for boundary in (
        np.arange(num_waits + 1) - 0.5
    ):
        ax.axhline(
            boundary,
            linewidth=0.5,
            color="white",
            alpha=0.35,
        )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.015,
    )

    colorbar.set_label(
        colorbar_label
    )

    return {
        "wait_s": wait_s,
        "probability_matrix": probability_matrix,
        "plot_matrix": plot_matrix,
        "eligible_matrix": eligible_matrix,
        "event_matrix": event_matrix,
    }, fig
    

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr


def _correlation_summary(x, y):
    """Return Pearson and Spearman correlations after removing invalid values."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)

    if np.sum(good) < 4:
        return {
            "num_points": int(np.sum(good)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    pearson_result = pearsonr(
        x[good],
        y[good],
    )

    spearman_result = spearmanr(
        x[good],
        y[good],
    )

    return {
        "num_points": int(np.sum(good)),
        "pearson_r": float(pearson_result.statistic),
        "pearson_p": float(pearson_result.pvalue),
        "spearman_rho": float(spearman_result.statistic),
        "spearman_p": float(spearman_result.pvalue),
    }


def _prepare_image_for_registration(image):
    """
    Remove slowly varying camera background while preserving the NV pattern.
    """

    image = np.asarray(
        image,
        dtype=float,
    )

    image = np.nan_to_num(
        image,
        nan=float(np.nanmedian(image)),
    )

    # Mildly smooth shot noise.
    smoothed = gaussian_filter(
        image,
        sigma=1.0,
    )

    # Remove broad illumination/background structure.
    background = gaussian_filter(
        image,
        sigma=12.0,
    )

    processed = (
        smoothed
        - background
    )

    processed -= np.mean(
        processed
    )

    # Reduce edge-related artifacts in Fourier registration.
    y_window = np.hanning(
        processed.shape[0]
    )

    x_window = np.hanning(
        processed.shape[1]
    )

    processed *= (
        y_window[:, None]
        * x_window[None, :]
    )

    return processed


def _register_image_pair(
    reference_image,
    moving_image,
    upsample_factor=20,
):
    """
    Determine the translation needed to align moving_image to reference_image.

    Returns
    -------
    dx_px, dy_px, error
    """

    reference = _prepare_image_for_registration(
        reference_image
    )

    moving = _prepare_image_for_registration(
        moving_image
    )

    try:
        from skimage.registration import phase_cross_correlation

        shift_yx, error, _ = phase_cross_correlation(
            reference,
            moving,
            upsample_factor=int(upsample_factor),
            normalization="phase",
        )

        dy_px = float(
            shift_yx[0]
        )

        dx_px = float(
            shift_yx[1]
        )

        return dx_px, dy_px, float(error)

    except ImportError:
        # Integer-pixel fallback if scikit-image is unavailable.
        reference_fft = np.fft.fft2(
            reference
        )

        moving_fft = np.fft.fft2(
            moving
        )

        cross_power = (
            reference_fft
            * np.conj(moving_fft)
        )

        cross_power /= np.maximum(
            np.abs(cross_power),
            1e-12,
        )

        correlation = np.fft.ifft2(
            cross_power
        )

        peak = np.unravel_index(
            np.argmax(
                np.abs(correlation)
            ),
            correlation.shape,
        )

        shift_yx = np.asarray(
            peak,
            dtype=float,
        )

        shape = np.asarray(
            reference.shape,
            dtype=float,
        )

        midpoint = np.floor(
            shape / 2
        )

        wrap = (
            shift_yx > midpoint
        )

        shift_yx[wrap] -= shape[wrap]

        dy_px = float(
            shift_yx[0]
        )

        dx_px = float(
            shift_yx[1]
        )

        return dx_px, dy_px, np.nan


def analyze_drift_state_correlation(
    raw_data: Dict[str, Any],
    register_images: bool = True,
    label_outliers: bool = True,
):
    """
    Compare drift estimates with NV charge-state changes.

    Tests
    -----
    1. Saved tracker-position change between consecutive runs.
    2. Initial-to-final image displacement within each run.
    """

    analysis = raw_data[
        "particle_analysis"
    ]

    event_fraction = np.asarray(
        analysis["event_fraction_by_run"],
        dtype=float,
    )

    candidate_count = np.asarray(
        analysis["num_candidates_by_run"],
        dtype=float,
    )

    retention = np.asarray(
        analysis["retention_by_run"],
        dtype=float,
    )

    num_runs = len(
        event_fraction
    )

    run_inds = np.arange(
        num_runs
    )

    # ==============================================================
    # Saved pixel tracker values
    # ==============================================================

    tracker_xy = np.asarray(
        raw_data["pixel_drifts"],
        dtype=float,
    )

    if tracker_xy.shape != (num_runs, 2):
        raise ValueError(
            "Expected pixel_drifts shape "
            f"{(num_runs, 2)}, received {tracker_xy.shape}."
        )

    tracker_relative_xy = (
        tracker_xy
        - tracker_xy[0]
    )

    tracker_step_xy = np.full_like(
        tracker_xy,
        np.nan,
    )

    tracker_step_xy[1:] = np.diff(
        tracker_xy,
        axis=0,
    )

    tracker_step_magnitude = np.sqrt(
        np.sum(
            tracker_step_xy**2,
            axis=1,
        )
    )

    tracker_correlation = _correlation_summary(
        tracker_step_magnitude,
        event_fraction,
    )

    # ==============================================================
    # Direct initial-to-final image registration
    # ==============================================================

    image_dx_px = np.full(
        num_runs,
        np.nan,
    )

    image_dy_px = np.full(
        num_runs,
        np.nan,
    )

    image_registration_error = np.full(
        num_runs,
        np.nan,
    )

    if register_images:
        if "img_arrays" not in raw_data:
            raise ValueError(
                "img_arrays are unavailable. Load the NPZ data using "
                "dm.get_raw_data(..., load_npz=True)."
            )

        images = np.asarray(
            raw_data["img_arrays"],
            dtype=float,
        )

        # Expected shape:
        # [experiment, run, step, repetition, y, x]
        if images.ndim != 6:
            raise ValueError(
                "Expected img_arrays with six dimensions "
                "[exp, run, step, rep, y, x]; "
                f"received shape {images.shape}."
            )

        initial_rep = int(
            analysis["initial_state_rep_ind"]
        )

        final_rep = int(
            analysis["final_readout_rep_ind"]
        )

        for run_ind in range(
            num_runs
        ):
            initial_image = images[
                0,
                run_ind,
                0,
                initial_rep,
            ]

            final_image = images[
                0,
                run_ind,
                0,
                final_rep,
            ]

            (
                image_dx_px[run_ind],
                image_dy_px[run_ind],
                image_registration_error[run_ind],
            ) = _register_image_pair(
                initial_image,
                final_image,
                upsample_factor=20,
            )

    image_shift_magnitude = np.sqrt(
        image_dx_px**2
        + image_dy_px**2
    )

    image_correlation = _correlation_summary(
        image_shift_magnitude,
        event_fraction,
    )

    # ==============================================================
    # Plot
    # ==============================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        constrained_layout=True,
    )

    # --------------------------------------------------------------
    # Saved tracker position
    # --------------------------------------------------------------

    ax = axes[0, 0]

    ax.plot(
        run_inds,
        tracker_relative_xy[:, 0],
        "o-",
        label="Relative tracker x",
    )

    ax.plot(
        run_inds,
        tracker_relative_xy[:, 1],
        "o-",
        label="Relative tracker y",
    )

    ax.set_xlabel(
        "Run index"
    )

    ax.set_ylabel(
        "Recorded position relative to run 0 (px)"
    )

    ax.set_title(
        "Saved pixel-tracking values"
    )

    ax.legend()
    ax.grid(True, alpha=0.25)

    # --------------------------------------------------------------
    # Tracker jump and event fraction
    # --------------------------------------------------------------

    ax = axes[0, 1]

    ax.plot(
        run_inds,
        tracker_step_magnitude,
        "o-",
        label="Tracker change",
    )

    ax.set_xlabel(
        "Run index"
    )

    ax.set_ylabel(
        "Change from previous run (px)"
    )

    ax_event = ax.twinx()

    ax_event.plot(
        run_inds,
        100.0 * event_fraction,
        "s--",
        label="Transition fraction",
    )

    ax_event.set_ylabel(
        "NV$^- \\rightarrow$ NV$^0$ fraction (%)"
    )

    ax.set_title(
        "Tracker changes and charge-state loss"
    )

    ax.grid(True, alpha=0.25)

    # --------------------------------------------------------------
    # Direct image-registered drift
    # --------------------------------------------------------------

    ax = axes[1, 0]

    ax.plot(
        run_inds,
        image_dx_px,
        "o-",
        label="Initial-to-final dx",
    )

    ax.plot(
        run_inds,
        image_dy_px,
        "o-",
        label="Initial-to-final dy",
    )

    ax.set_xlabel(
        "Run index"
    )

    ax.set_ylabel(
        "Image-registration shift (px)"
    )

    ax.set_title(
        "Drift during each dark-wait interval"
    )

    ax.legend()
    ax.grid(True, alpha=0.25)

    # --------------------------------------------------------------
    # Drift-event correlation
    # --------------------------------------------------------------

    ax = axes[1, 1]

    ax.scatter(
        image_shift_magnitude,
        100.0 * event_fraction,
        s=45,
    )

    if label_outliers:
        for run_ind in range(
            num_runs
        ):
            if (
                np.isfinite(image_shift_magnitude[run_ind])
                and (
                    event_fraction[run_ind]
                    > np.nanpercentile(event_fraction, 85)
                    or image_shift_magnitude[run_ind]
                    > np.nanpercentile(
                        image_shift_magnitude,
                        85,
                    )
                )
            ):
                ax.annotate(
                    str(run_ind),
                    (
                        image_shift_magnitude[run_ind],
                        100.0 * event_fraction[run_ind],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                )

    ax.set_xlabel(
        "Initial-to-final image shift magnitude (px)"
    )

    ax.set_ylabel(
        "NV$^- \\rightarrow$ NV$^0$ fraction (%)"
    )

    ax.set_title(
        "Does measured drift predict state changes?\n"
        f"Spearman ρ = "
        f"{image_correlation['spearman_rho']:.3f}, "
        f"p = {image_correlation['spearman_p']:.3g}"
    )

    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Drift versus apparent NV charge-state transitions",
        fontsize=15,
    )

    result = {
        "run_inds": run_inds,
        "event_fraction": event_fraction,
        "candidate_count": candidate_count,
        "retention": retention,

        "tracker_xy": tracker_xy,
        "tracker_relative_xy": tracker_relative_xy,
        "tracker_step_xy": tracker_step_xy,
        "tracker_step_magnitude_px": tracker_step_magnitude,
        "tracker_event_correlation": tracker_correlation,

        "image_dx_px": image_dx_px,
        "image_dy_px": image_dy_px,
        "image_shift_magnitude_px": image_shift_magnitude,
        "image_registration_error": image_registration_error,
        "image_event_correlation": image_correlation,
    }

    print("\nSaved tracker-change versus event fraction:")
    print(tracker_correlation)

    if register_images:
        print("\nInitial-to-final image drift versus event fraction:")
        print(image_correlation)

    worst_event_run = int(
        np.nanargmax(event_fraction)
    )

    print("\nLargest apparent-loss run:")
    print("run:", worst_event_run)
    print(
        "event fraction:",
        event_fraction[worst_event_run],
    )
    print(
        "candidate count:",
        candidate_count[worst_event_run],
    )
    print(
        "retention:",
        retention[worst_event_run],
    )
    print(
        "tracker step:",
        tracker_step_magnitude[worst_event_run],
        "px",
    )
    print(
        "initial-to-final image shift:",
        image_shift_magnitude[worst_event_run],
        "px",
    )

    return result, fig
    
if __name__ == "__main__":
    kpl.init_kplotlib()
    
    file_stem = "2026_07_26-18_11_11-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1800s-wait-1800s"
    raw_data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
    )

    drift_result, fig_drift = (
        analyze_drift_state_correlation(
            raw_data,
            register_images=True,
        )
    )

    kpl.show(block=True)
    sys.exit()
    
    # FILE_STEMS = [
    # "2026_07_23-01_05_24-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s",
    # "2026_07_23-01_48_56-qnami-nv0_2026_02_20-particle-memory-source_off_wait_10s-wait-10s",
    # # "2026_07_23-03_05_50-qnami-nv0_2026_02_20-particle-memory-source_off_wait_30s-wait-30s",
    # "2026_07_23-05_13_51-qnami-nv0_2026_02_20-particle-memory-source_off_wait_60s-wait-60s",
    # "2026_07_23-09_19_48-qnami-nv0_2026_02_20-particle-memory-source_off_wait_180s-wait-180s",
    # "2026_07_23-15_55_08-qnami-nv0_2026_02_20-particle-memory-source_off_wait_300s-wait-300s",
    # "2026_07_24-00_29_16-qnami-nv0_2026_02_20-particle-memory-source_off_wait_600s-wait-600s",
    # "2026_07_24-08_56_35-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1200s-wait-1200s",
    # ]
    
    FILE_STEMS = [
    "2026_07_24-21_43_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s",
    "2026_07_24-22_27_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_10s-wait-10s",
    "2026_07_25-01_51_32-qnami-nv0_2026_02_20-particle-memory-source_off_wait_60s-wait-60s",
    "2026_07_25-05_57_38-qnami-nv0_2026_02_20-particle-memory-source_off_wait_180s-wait-180s",
    "2026_07_25-12_33_01-qnami-nv0_2026_02_20-particle-memory-source_off_wait_300s-wait-300s",
    "2026_07_25-21_07_06-qnami-nv0_2026_02_20-particle-memory-source_off_wait_600s-wait-600s",
    "2026_07_26-05_34_29-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1200s-wait-1200s",
    "2026_07_26-18_11_11-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1800s-wait-1800s",
    "2026_07_27-19_18_59-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    ]
    
    output = run_particle_memory_dark_wait_comparison_analysis(
        file_stems=FILE_STEMS,
        recompute_analysis=False,
        save_fig=True,
        save_csv=True,
    )
    analyses = output["analyses"]
    row_result, fig_rows = plot_nv_loss_row_by_row(
    analyses,
    selected_waits_s=[
        0,
        10,
        30,
        60,
        180,
        300,
        600,
        1200,
        1800,
        3600,
    ],
    subtract_zero_wait=False,
    show_percent=True,
    )
    
    excess_row_result, fig_excess_rows = plot_nv_loss_row_by_row(
    analyses,
    selected_waits_s=[
        0,
        10,
        30,
        60,
        180,
        300,
        600,
        1200,
        1800,
        3600,
    ],
    subtract_zero_wait=True,
    show_percent=True,
    )

    kpl.show(block=True)
    sys.exit()
    # ------------------------------------------------------------------
    # Load saved dataset
    # ------------------------------------------------------------------
    save_fig = False
    
    file_stem = (
        "2026_07_16-13_18_15-"
        "qnami-nv0_2026_02_20-"
        "particle-memory-source_off-wait-300s"
    )
        
    file_stem = (
        # "2026_07_19-09_54_11-qnami-nv0_2026_02_20-particle-memory-source_off-wait-300s"
    "2026_07_19-09_54_11-qnami-nv0_2026_02_20-particle-memory-source_off-wait-300s"
    
    )


    raw_data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
    )

    print("\nLoaded dataset:")
    print(file_stem)
    print(
        "counts shape:",
        np.asarray(raw_data["counts"]).shape,
    )
    print(
        "dark wait:",
        raw_data.get("dark_wait_s"),
    )
    print(
        "exposure label:",
        raw_data.get("exposure_label"),
    )
    print(
        "initial-state rep:",
        raw_data.get("initial_state_rep_ind"),
    )
    print(
        "final-readout rep:",
        raw_data.get("final_readout_rep_ind"),
    )

    # ------------------------------------------------------------------
    # Recover saved analysis settings
    # ------------------------------------------------------------------
    saved_analysis = raw_data.get(
        "particle_analysis",
        {},
    )

    initial_margin_counts = float(
        saved_analysis.get(
            "initial_margin_counts",
            1.0,
        )
    )

    final_margin_counts = float(
        saved_analysis.get(
            "final_margin_counts",
            1.0,
        )
    )

    min_cluster_size = int(
        saved_analysis.get(
            "min_cluster_size",
            2,
        )
    )

    # ------------------------------------------------------------------
    # Recover image coordinates
    # ------------------------------------------------------------------
    saved_coords = saved_analysis.get(
        "coords_xy",
        None,
    )

    if saved_coords is not None:
        coords_xy = np.asarray(
            saved_coords,
            dtype=float,
        )
    else:
        coords_xy = _coerce_img_coords(
            raw_data["nv_list"],
            img_coords=None,
        )

    if coords_xy is None:
        raise ValueError(
            "Could not recover NV image coordinates."
        )

    print(
        "coordinates shape:",
        coords_xy.shape,
    )

    # ------------------------------------------------------------------
    # Choose cluster radius
    # ------------------------------------------------------------------
    saved_cluster_radius = saved_analysis.get(
        "cluster_radius_px",
        None,
    )

    if saved_cluster_radius is not None:
        cluster_radius_px = float(
            saved_cluster_radius
        )
    else:
        displacement = (
            coords_xy[:, None, :]
            - coords_xy[None, :, :]
        )

        distance_matrix = np.sqrt(
            np.sum(
                displacement**2,
                axis=2,
            )
        )

        np.fill_diagonal(
            distance_matrix,
            np.inf,
        )

        nearest_neighbor_distance = np.min(
            distance_matrix,
            axis=1,
        )

        median_nn_px = float(
            np.nanmedian(
                nearest_neighbor_distance
            )
        )

        # Nearest-neighbor clustering radius.
        cluster_radius_px = (
            1.25 * median_nn_px
        )

        print(
            "median nearest-neighbor spacing:",
            median_nn_px,
            "px",
        )

    print("\nReanalysis settings:")
    print(
        "initial margin:",
        initial_margin_counts,
    )
    print(
        "final margin:",
        final_margin_counts,
    )
    print(
        "cluster radius:",
        cluster_radius_px,
        "px",
    )
    print(
        "minimum cluster size:",
        min_cluster_size,
    )

    # ------------------------------------------------------------------
    # Re-run charge-memory classification
    # ------------------------------------------------------------------
    analysis = analyze_particle_charge_memory(
        raw_data,
        initial_margin_counts=initial_margin_counts,
        final_margin_counts=final_margin_counts,
        cluster_radius_px=cluster_radius_px,
        min_cluster_size=min_cluster_size,
        img_coords=coords_xy,
    )

    raw_data["particle_analysis"] = analysis
    fig_summary = plot_particle_summary(
        raw_data,
        analysis,
    )

    fig_probability = plot_event_probability_by_nv(
        raw_data,
        analysis,
    )
    
    for run_ind in [0, 56, 80, 97]:
        plot_particle_event_map(
            raw_data,
            analysis,
            run_ind=run_ind,
        )

    plt.show(block=True)
    # ------------------------------------------------------------------
    # Aggregated correlation analysis and visualization
    # ------------------------------------------------------------------
    spatial_result, spatial_figures = (
        analyze_and_plot_spatial_correlations(
            raw_data=raw_data,
            analysis=analysis,
            coords_xy=coords_xy,
            cluster_radius_px=cluster_radius_px,
            num_permutations=2000,
            random_seed=12345,
            significance_level=0.05,
            min_pair_repeats=2,
            max_pairs_to_plot=30,
        )
    )

    raw_data[
        "spatial_correlation_analysis"
    ] = spatial_result
    
    if save_fig: 
        timestamp = raw_data.get(
            "timestamp",
            dm.get_time_stamp(),
        )

        plot_names = [
            "spatial-run-summary",
            "spatial-event-frequency",
            "repeated-nearby-pairs",
            "spatial-pvalue-diagnostics",
        ]

        for fig, plot_name in zip(
            spatial_figures,
            plot_names,
        ):
            fig_path = dm.get_file_path(
                __file__,
                timestamp,
                plot_name,
            )

            dm.save_figure(
                fig,
                fig_path,
            )

            print(
                "Saved:",
                fig_path,
            )

    plt.show(block=True)