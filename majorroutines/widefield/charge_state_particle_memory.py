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
    fig.tight_layout()
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
    exposure_label: str = "source_off",
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


if __name__ == "__main__":
    kpl.init_kplotlib()

    print(
        "Import this module from your control panel and call main(nv_list, ...).\n"
        "Example:\n\n"
        "    raw_data = main(\n"
        "        nv_list,\n"
        "        num_init_reps=10,\n"
        "        num_runs=20,\n"
        "        dark_wait_s=300,\n"
        "        exposure_label='source_off',\n"
        "        take_initial_check=True,\n"
        "        confirm_margin_counts=1.0,\n"
        "        initial_event_margin_counts=2.0,\n"
        "        final_event_margin_counts=2.0,\n"
        "        cluster_radius_px=6.0,\n"
        "        save_images=True,\n"
        "    )"
    )
