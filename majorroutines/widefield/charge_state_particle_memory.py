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
import csv
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

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

def filter_analyses_for_wait_trend(
    analyses,
    exclude_nv_inds=None,
):
    """
    Create filtered particle-analysis dictionaries and recompute all
    run-level quantities used by summarize_wait_sweep().

    The original analysis dictionaries are not modified.
    """

    if exclude_nv_inds is None:
        exclude_nv_inds = []

    exclude_nv_inds = np.unique(
        np.asarray(
            exclude_nv_inds,
            dtype=int,
        )
    )

    filtered_analyses = []

    for analysis in analyses:
        out = dict(analysis)

        initial_mask = np.asarray(
            analysis["initial_nvm_mask"],
            dtype=bool,
        )

        final_mask = np.asarray(
            analysis["final_nvm_mask"],
            dtype=bool,
        )

        candidate_mask = np.asarray(
            analysis[
                "candidate_nvm_to_nv0_mask"
            ],
            dtype=bool,
        )

        retained_mask = np.asarray(
            analysis["retained_nvm_mask"],
            dtype=bool,
        )

        final_ambiguous_mask = np.asarray(
            analysis["final_ambiguous_mask"],
            dtype=bool,
        )

        num_nvs, num_runs = initial_mask.shape

        keep_mask = np.ones(
            num_nvs,
            dtype=bool,
        )

        valid_excluded = exclude_nv_inds[
            (exclude_nv_inds >= 0)
            & (exclude_nv_inds < num_nvs)
        ]

        keep_mask[
            valid_excluded
        ] = False

        original_nv_inds = np.where(
            keep_mask
        )[0]

        # ----------------------------------------------------------
        # Apply the same NV filter to every dataset
        # ----------------------------------------------------------

        initial_filtered = initial_mask[
            keep_mask
        ]

        final_filtered = final_mask[
            keep_mask
        ]

        candidate_filtered = candidate_mask[
            keep_mask
        ]

        retained_filtered = retained_mask[
            keep_mask
        ]

        ambiguous_filtered = (
            initial_filtered
            & final_ambiguous_mask[
                keep_mask
            ]
        )

        num_filtered_nvs = int(
            np.sum(keep_mask)
        )

        # ----------------------------------------------------------
        # Recompute per-run counts
        # ----------------------------------------------------------

        num_initial_by_run = np.sum(
            initial_filtered,
            axis=0,
        )

        num_final_by_run = np.sum(
            final_filtered,
            axis=0,
        )

        num_candidates_by_run = np.sum(
            candidate_filtered,
            axis=0,
        )

        num_retained_by_run = np.sum(
            retained_filtered,
            axis=0,
        )

        num_ambiguous_by_run = np.sum(
            ambiguous_filtered,
            axis=0,
        )

        retention_by_run = np.full(
            num_runs,
            np.nan,
            dtype=float,
        )

        event_fraction_by_run = np.full(
            num_runs,
            np.nan,
            dtype=float,
        )

        good_runs = (
            num_initial_by_run > 0
        )

        retention_by_run[
            good_runs
        ] = (
            num_retained_by_run[good_runs]
            / num_initial_by_run[good_runs]
        )

        event_fraction_by_run[
            good_runs
        ] = (
            num_candidates_by_run[good_runs]
            / num_initial_by_run[good_runs]
        )

        # ----------------------------------------------------------
        # Recompute per-NV probabilities
        # ----------------------------------------------------------

        eligible_trials_by_nv = np.sum(
            initial_filtered,
            axis=1,
        )

        event_trials_by_nv = np.sum(
            candidate_filtered,
            axis=1,
        )

        event_probability_by_nv = np.full(
            num_filtered_nvs,
            np.nan,
            dtype=float,
        )

        good_nv = (
            eligible_trials_by_nv > 0
        )

        event_probability_by_nv[
            good_nv
        ] = (
            event_trials_by_nv[good_nv]
            / eligible_trials_by_nv[good_nv]
        )

        # ----------------------------------------------------------
        # Update the copy
        # ----------------------------------------------------------

        out.update(
            {
                "num_nvs": num_filtered_nvs,
                "excluded_nv_inds": (
                    valid_excluded.tolist()
                ),
                "original_nv_inds": (
                    original_nv_inds.tolist()
                ),

                "initial_nvm_mask": (
                    initial_filtered.tolist()
                ),
                "final_nvm_mask": (
                    final_filtered.tolist()
                ),
                "candidate_nvm_to_nv0_mask": (
                    candidate_filtered.tolist()
                ),
                "retained_nvm_mask": (
                    retained_filtered.tolist()
                ),

                "num_initial_nvm_by_run": (
                    num_initial_by_run.tolist()
                ),
                "num_final_nvm_by_run": (
                    num_final_by_run.tolist()
                ),
                "num_candidates_by_run": (
                    num_candidates_by_run.tolist()
                ),
                "num_retained_by_run": (
                    num_retained_by_run.tolist()
                ),
                "num_ambiguous_by_run": (
                    num_ambiguous_by_run.tolist()
                ),

                "retention_by_run": (
                    retention_by_run.tolist()
                ),
                "event_fraction_by_run": (
                    event_fraction_by_run.tolist()
                ),

                "eligible_trials_by_nv": (
                    eligible_trials_by_nv.tolist()
                ),
                "event_trials_by_nv": (
                    event_trials_by_nv.tolist()
                ),
                "event_probability_by_nv": (
                    event_probability_by_nv.tolist()
                ),

                "mean_initial_nvm": float(
                    np.mean(num_initial_by_run)
                ),
                "mean_final_nvm": float(
                    np.mean(num_final_by_run)
                ),
                "mean_candidates_per_run": float(
                    np.mean(
                        num_candidates_by_run
                    )
                ),
                "mean_retention": float(
                    np.nanmean(
                        retention_by_run
                    )
                ),
                "mean_event_fraction": float(
                    np.nanmean(
                        event_fraction_by_run
                    )
                ),
            }
        )

        filtered_analyses.append(out)

    return filtered_analyses

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
    exclude_nv_inds=None,
):
    raw_datasets, analyses = load_wait_sweep(
        file_stems,
        recompute_analysis=recompute_analysis,
    )
    
    if exclude_nv_inds is not None:
        analyses_for_summary = (
            filter_analyses_for_wait_trend(
                analyses,
                exclude_nv_inds=exclude_nv_inds,
            )
        )
    else:
        analyses_for_summary = analyses

    summary = summarize_wait_sweep(
        analyses_for_summary
    )

    print_wait_sweep_table(
        summary
    )

    fig = plot_wait_sweep_summary(
        summary,
        zoom_retention_axes=True,
    )


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
        "filtered_analyses": analyses_for_summary,
        "excluded_nv_inds": (
            []
            if exclude_nv_inds is None
            else list(exclude_nv_inds)
        ),
        "summary": summary,
        "fig": fig,
    }


def plot_nv_loss_row_by_row(
    analyses: Sequence[Dict[str, Any]],
    selected_waits_s: Optional[Sequence[float]] = None,
    subtract_zero_wait: bool = False,
    show_percent: bool = True,
    percentile_limit: float = 99.0,
    exclude_nv_inds: Optional[Sequence[int]] = None,
    max_zero_wait_loss_probability: Optional[float] = None,
    min_zero_wait_eligible_runs: int = 5,
):
    """
    Plot per-NV charge-loss probability as a wait-time × NV-index heat map.

    Bad actors can be removed in two ways:

        1. Manual exclusion:
               exclude_nv_inds=[12, 57, 143]

        2. Automatic exclusion using the 0 s control:
               max_zero_wait_loss_probability=0.30
               min_zero_wait_eligible_runs=10

    An automatically excluded NV must:

        - be initially verified in at least
          ``min_zero_wait_eligible_runs`` runs; and
        - show an NV- -> NV0 event probability at 0 s greater than or
          equal to ``max_zero_wait_loss_probability``.

    Excluded NVs remain at their original heat-map positions and appear gray.

    The returned ``filtered_analyses`` contains recomputed run-level statistics
    and can be passed to:

        summarize_wait_sweep(...)
        plot_wait_sweep_summary(...)

    Parameters
    ----------
    analyses:
        Sequence of particle-analysis dictionaries.

    selected_waits_s:
        Wait times to include in the heat map. For each requested value,
        the closest available dataset is selected.

    subtract_zero_wait:
        If True, subtract each NV's measured loss probability at 0 s.

    show_percent:
        If True, display probabilities in percent.

    percentile_limit:
        Percentile used for the color scale. This affects only the displayed
        color range and does not exclude NVs.

    exclude_nv_inds:
        Original NV indices to exclude manually.

    max_zero_wait_loss_probability:
        Automatic bad-actor threshold based on the 0 s control.
        Use None to disable automatic filtering.

    min_zero_wait_eligible_runs:
        Minimum number of eligible 0 s trials required before an NV can be
        excluded automatically.

    Returns
    -------
    result, fig

    result contains:
        wait_s
        probability_matrix
        filtered_probability_matrix
        plot_matrix
        eligible_matrix
        event_matrix
        excluded_nv_inds
        automatically_excluded_nv_inds
        manually_excluded_nv_inds
        kept_nv_inds
        filtered_analyses
    """

    # ==============================================================
    # Input checks
    # ==============================================================

    if not analyses:
        raise ValueError(
            "No analyses were supplied."
        )

    percentile_limit = float(
        percentile_limit
    )

    if not (
        0.0 < percentile_limit <= 100.0
    ):
        raise ValueError(
            "percentile_limit must lie in (0, 100]."
        )

    min_zero_wait_eligible_runs = int(
        min_zero_wait_eligible_runs
    )

    if min_zero_wait_eligible_runs < 1:
        raise ValueError(
            "min_zero_wait_eligible_runs must be at least 1."
        )

    if max_zero_wait_loss_probability is not None:
        max_zero_wait_loss_probability = float(
            max_zero_wait_loss_probability
        )

        if not (
            0.0
            <= max_zero_wait_loss_probability
            <= 1.0
        ):
            raise ValueError(
                "max_zero_wait_loss_probability must lie in [0, 1]."
            )

    all_analyses = sorted(
        list(analyses),
        key=lambda analysis: float(
            analysis["dark_wait_s"]
        ),
    )

    # ==============================================================
    # Validate mask shapes
    # ==============================================================

    required_mask_keys = (
        "initial_nvm_mask",
        "final_nvm_mask",
        "final_ambiguous_mask",
        "candidate_nvm_to_nv0_mask",
        "retained_nvm_mask",
    )

    reference_num_nvs = None

    for analysis in all_analyses:
        mask_arrays = {}

        for key in required_mask_keys:
            if key not in analysis:
                raise KeyError(
                    f"Analysis at dark wait "
                    f"{analysis.get('dark_wait_s')} s "
                    f"is missing {key!r}."
                )

            mask_arrays[key] = np.asarray(
                analysis[key],
                dtype=bool,
            )

        reference_shape = mask_arrays[
            "initial_nvm_mask"
        ].shape

        if len(reference_shape) != 2:
            raise ValueError(
                "Expected masks with shape [nv, run]; "
                f"got {reference_shape}."
            )

        for key, mask in mask_arrays.items():
            if mask.shape != reference_shape:
                raise ValueError(
                    f"{key} has shape {mask.shape}, "
                    f"but initial_nvm_mask has shape "
                    f"{reference_shape}."
                )

        current_num_nvs = int(
            reference_shape[0]
        )

        if reference_num_nvs is None:
            reference_num_nvs = current_num_nvs

        elif current_num_nvs != reference_num_nvs:
            raise ValueError(
                "The datasets contain different numbers of NVs."
            )

    num_nvs = int(
        reference_num_nvs
    )

    # ==============================================================
    # Helper: calculate per-NV event probability
    # ==============================================================

    def calculate_nv_probability(
        analysis: Dict[str, Any],
    ):
        initial_mask = np.asarray(
            analysis["initial_nvm_mask"],
            dtype=bool,
        )

        event_mask = np.asarray(
            analysis[
                "candidate_nvm_to_nv0_mask"
            ],
            dtype=bool,
        )

        eligible_count = np.sum(
            initial_mask,
            axis=1,
        ).astype(float)

        event_count = np.sum(
            event_mask,
            axis=1,
        ).astype(float)

        probability = np.full(
            initial_mask.shape[0],
            np.nan,
            dtype=float,
        )

        good = eligible_count > 0

        probability[good] = (
            event_count[good]
            / eligible_count[good]
        )

        return (
            probability,
            eligible_count,
            event_count,
        )

    # ==============================================================
    # Find the 0 s control
    # ==============================================================

    zero_analyses = [
        analysis
        for analysis in all_analyses
        if np.isclose(
            float(analysis["dark_wait_s"]),
            0.0,
        )
    ]

    zero_analysis = (
        zero_analyses[0]
        if zero_analyses
        else None
    )

    zero_required = (
        subtract_zero_wait
        or max_zero_wait_loss_probability is not None
    )

    if zero_required and zero_analysis is None:
        raise ValueError(
            "A 0 s dataset is required for zero-wait subtraction "
            "or automatic bad-actor filtering."
        )

    if zero_analysis is not None:
        (
            zero_probability,
            zero_eligible_count,
            zero_event_count,
        ) = calculate_nv_probability(
            zero_analysis
        )

    else:
        zero_probability = np.full(
            num_nvs,
            np.nan,
            dtype=float,
        )

        zero_eligible_count = np.zeros(
            num_nvs,
            dtype=float,
        )

        zero_event_count = np.zeros(
            num_nvs,
            dtype=float,
        )

    # ==============================================================
    # Manual filtering
    # ==============================================================

    manual_excluded_mask = np.zeros(
        num_nvs,
        dtype=bool,
    )

    if exclude_nv_inds is not None:
        manual_inds = np.unique(
            np.asarray(
                exclude_nv_inds,
                dtype=int,
            )
        )

        invalid_inds = manual_inds[
            (manual_inds < 0)
            | (manual_inds >= num_nvs)
        ]

        if invalid_inds.size > 0:
            raise IndexError(
                "exclude_nv_inds contains invalid indices: "
                f"{invalid_inds.tolist()}"
            )

        manual_excluded_mask[
            manual_inds
        ] = True

    # ==============================================================
    # Automatic filtering from the 0 s control
    # ==============================================================

    automatic_excluded_mask = np.zeros(
        num_nvs,
        dtype=bool,
    )

    if max_zero_wait_loss_probability is not None:
        automatic_excluded_mask = (
            np.isfinite(
                zero_probability
            )
            & (
                zero_eligible_count
                >= min_zero_wait_eligible_runs
            )
            & (
                zero_probability
                >= max_zero_wait_loss_probability
            )
        )

    # Combine manual and automatic filters.
    excluded_mask = (
        manual_excluded_mask
        | automatic_excluded_mask
    )

    keep_mask = ~excluded_mask

    excluded_nv_inds = np.where(
        excluded_mask
    )[0].astype(int)

    manually_excluded_nv_inds = np.where(
        manual_excluded_mask
    )[0].astype(int)

    automatically_excluded_nv_inds = np.where(
        automatic_excluded_mask
    )[0].astype(int)

    kept_nv_inds = np.where(
        keep_mask
    )[0].astype(int)

    if kept_nv_inds.size == 0:
        raise ValueError(
            "The requested filter excludes every NV."
        )

    print(
        "\n"
        + "=" * 78
    )
    print(
        "PER-NV WAIT-SWEEP FILTERING"
    )
    print(
        "=" * 78
    )

    print(
        "Total NVs:",
        num_nvs,
    )

    print(
        "Manually excluded:",
        manually_excluded_nv_inds.tolist(),
    )

    print(
        "Automatically excluded from 0 s control:",
        automatically_excluded_nv_inds.tolist(),
    )

    print(
        "Combined excluded:",
        excluded_nv_inds.tolist(),
    )

    print(
        "NVs retained:",
        int(kept_nv_inds.size),
        "/",
        num_nvs,
    )

    # ==============================================================
    # Create filtered analyses for the wait-time trend
    # ==============================================================

    filtered_analyses = []

    for analysis in all_analyses:
        filtered_analysis = dict(
            analysis
        )

        initial_mask = np.asarray(
            analysis["initial_nvm_mask"],
            dtype=bool,
        )[keep_mask]

        final_mask = np.asarray(
            analysis["final_nvm_mask"],
            dtype=bool,
        )[keep_mask]

        final_ambiguous_mask = np.asarray(
            analysis["final_ambiguous_mask"],
            dtype=bool,
        )[keep_mask]

        candidate_mask = np.asarray(
            analysis[
                "candidate_nvm_to_nv0_mask"
            ],
            dtype=bool,
        )[keep_mask]

        retained_mask = np.asarray(
            analysis["retained_nvm_mask"],
            dtype=bool,
        )[keep_mask]

        (
            num_filtered_nvs,
            num_runs,
        ) = initial_mask.shape

        # ----------------------------------------------------------
        # Recalculate run-level counts
        # ----------------------------------------------------------

        num_initial_by_run = np.sum(
            initial_mask,
            axis=0,
        )

        num_final_by_run = np.sum(
            final_mask,
            axis=0,
        )

        num_candidates_by_run = np.sum(
            candidate_mask,
            axis=0,
        )

        num_retained_by_run = np.sum(
            retained_mask,
            axis=0,
        )

        num_ambiguous_by_run = np.sum(
            initial_mask
            & final_ambiguous_mask,
            axis=0,
        )

        # ----------------------------------------------------------
        # Recalculate run-level fractions
        # ----------------------------------------------------------

        retention_by_run = np.full(
            num_runs,
            np.nan,
            dtype=float,
        )

        event_fraction_by_run = np.full(
            num_runs,
            np.nan,
            dtype=float,
        )

        good_runs = (
            num_initial_by_run > 0
        )

        retention_by_run[
            good_runs
        ] = (
            num_retained_by_run[
                good_runs
            ]
            / num_initial_by_run[
                good_runs
            ]
        )

        event_fraction_by_run[
            good_runs
        ] = (
            num_candidates_by_run[
                good_runs
            ]
            / num_initial_by_run[
                good_runs
            ]
        )

        # ----------------------------------------------------------
        # Recalculate per-NV event probabilities
        # ----------------------------------------------------------

        eligible_trials_by_nv = np.sum(
            initial_mask,
            axis=1,
        )

        event_trials_by_nv = np.sum(
            candidate_mask,
            axis=1,
        )

        event_probability_by_nv = np.full(
            num_filtered_nvs,
            np.nan,
            dtype=float,
        )

        good_nv = (
            eligible_trials_by_nv > 0
        )

        event_probability_by_nv[
            good_nv
        ] = (
            event_trials_by_nv[
                good_nv
            ]
            / eligible_trials_by_nv[
                good_nv
            ]
        )

        if np.any(
            np.isfinite(
                retention_by_run
            )
        ):
            mean_retention = float(
                np.nanmean(
                    retention_by_run
                )
            )
        else:
            mean_retention = np.nan

        if np.any(
            np.isfinite(
                event_fraction_by_run
            )
        ):
            mean_event_fraction = float(
                np.nanmean(
                    event_fraction_by_run
                )
            )
        else:
            mean_event_fraction = np.nan

        filtered_analysis.update(
            {
                "num_nvs": int(
                    num_filtered_nvs
                ),
                "num_runs": int(
                    num_runs
                ),
                "original_nv_inds": (
                    kept_nv_inds.tolist()
                ),
                "excluded_nv_inds": (
                    excluded_nv_inds.tolist()
                ),
                "initial_nvm_mask": (
                    initial_mask.tolist()
                ),
                "final_nvm_mask": (
                    final_mask.tolist()
                ),
                "final_ambiguous_mask": (
                    final_ambiguous_mask.tolist()
                ),
                "candidate_nvm_to_nv0_mask": (
                    candidate_mask.tolist()
                ),
                "retained_nvm_mask": (
                    retained_mask.tolist()
                ),
                "num_initial_nvm_by_run": (
                    num_initial_by_run.tolist()
                ),
                "num_final_nvm_by_run": (
                    num_final_by_run.tolist()
                ),
                "num_candidates_by_run": (
                    num_candidates_by_run.tolist()
                ),
                "num_retained_by_run": (
                    num_retained_by_run.tolist()
                ),
                "num_ambiguous_by_run": (
                    num_ambiguous_by_run.tolist()
                ),
                "retention_by_run": (
                    retention_by_run.tolist()
                ),
                "event_fraction_by_run": (
                    event_fraction_by_run.tolist()
                ),
                "eligible_trials_by_nv": (
                    eligible_trials_by_nv.tolist()
                ),
                "event_trials_by_nv": (
                    event_trials_by_nv.tolist()
                ),
                "event_probability_by_nv": (
                    event_probability_by_nv.tolist()
                ),
                "mean_initial_nvm": float(
                    np.mean(
                        num_initial_by_run
                    )
                ),
                "mean_final_nvm": float(
                    np.mean(
                        num_final_by_run
                    )
                ),
                "mean_candidates_per_run": float(
                    np.mean(
                        num_candidates_by_run
                    )
                ),
                "median_candidates_per_run": float(
                    np.median(
                        num_candidates_by_run
                    )
                ),
                "mean_retention": (
                    mean_retention
                ),
                "mean_event_fraction": (
                    mean_event_fraction
                ),
                "filter_description": (
                    "Bad-actor NVs removed before "
                    "wait-sweep trend calculation."
                ),
            }
        )

        # ----------------------------------------------------------
        # Filter other fields whose first dimension is NV index
        # ----------------------------------------------------------

        per_nv_keys = (
            "initial_counts",
            "final_counts",
            "final_nv0_confident_mask",
            "coords_xy",
        )

        for key in per_nv_keys:
            value = analysis.get(
                key,
                None,
            )

            if value is None:
                continue

            array = np.asarray(
                value
            )

            if (
                array.ndim >= 1
                and array.shape[0] == num_nvs
            ):
                filtered_analysis[
                    key
                ] = array[
                    keep_mask
                ].tolist()

        # Cluster indices refer to the full original NV list.
        # Remove them instead of returning stale cluster information.
        for key in (
            "cluster_components_by_run",
            "max_cluster_size_by_run",
            "num_clusters_by_run",
            "num_large_clusters_by_run",
        ):
            filtered_analysis.pop(
                key,
                None,
            )

        filtered_analyses.append(
            filtered_analysis
        )

    # ==============================================================
    # Select wait times for the heat map
    # ==============================================================

    selected_analyses = all_analyses

    if selected_waits_s is not None:
        selected_analyses = []

        for requested_wait in selected_waits_s:
            closest_analysis = min(
                all_analyses,
                key=lambda analysis: abs(
                    float(
                        analysis["dark_wait_s"]
                    )
                    - float(
                        requested_wait
                    )
                ),
            )

            if (
                closest_analysis
                not in selected_analyses
            ):
                selected_analyses.append(
                    closest_analysis
                )

        selected_analyses = sorted(
            selected_analyses,
            key=lambda analysis: float(
                analysis["dark_wait_s"]
            ),
        )

    wait_s = np.asarray(
        [
            float(
                analysis["dark_wait_s"]
            )
            for analysis in selected_analyses
        ],
        dtype=float,
    )

    # ==============================================================
    # Calculate each heat-map row
    # ==============================================================

    probability_rows = []
    eligible_rows = []
    event_rows = []

    for analysis in selected_analyses:
        (
            probability,
            eligible_count,
            event_count,
        ) = calculate_nv_probability(
            analysis
        )

        probability_rows.append(
            probability
        )

        eligible_rows.append(
            eligible_count
        )

        event_rows.append(
            event_count
        )

    probability_matrix = np.asarray(
        probability_rows,
        dtype=float,
    )

    eligible_matrix = np.asarray(
        eligible_rows,
        dtype=float,
    )

    event_matrix = np.asarray(
        event_rows,
        dtype=float,
    )

   # ==============================================================
    # Physically remove excluded NVs from all displayed matrices
    # ==============================================================

    filtered_probability_matrix = probability_matrix[
        :,
        keep_mask,
    ]

    filtered_eligible_matrix = eligible_matrix[
        :,
        keep_mask,
    ]

    filtered_event_matrix = event_matrix[
        :,
        keep_mask,
    ]

    filtered_zero_probability = zero_probability[
        keep_mask
    ]


    # Defensive shape checks
    if (
        filtered_probability_matrix.shape[1]
        != filtered_zero_probability.size
    ):
        raise RuntimeError(
            "Filtered probability matrix and zero-wait vector "
            "have inconsistent NV dimensions: "
            f"{filtered_probability_matrix.shape} versus "
            f"{filtered_zero_probability.shape}."
        )

    if (
        filtered_eligible_matrix.shape
        != filtered_probability_matrix.shape
    ):
        raise RuntimeError(
            "Filtered eligible and probability matrices "
            "have inconsistent shapes."
        )

    if (
        filtered_event_matrix.shape
        != filtered_probability_matrix.shape
    ):
        raise RuntimeError(
            "Filtered event and probability matrices "
            "have inconsistent shapes."
        )


    # ==============================================================
    # Optional zero-wait subtraction
    # ==============================================================

    if subtract_zero_wait:
        plot_matrix = (
            filtered_probability_matrix
            - filtered_zero_probability[
                None,
                :,
            ]
        )

        title = (
            "Per-NV excess charge loss "
            "above the 0 s baseline"
        )

        colorbar_label = (
            "Excess loss probability"
        )

    else:
        plot_matrix = (
            filtered_probability_matrix.copy()
        )

        title = (
            "Per-NV charge loss at "
            "different dark waits"
        )

        colorbar_label = (
            "NV$^- \\rightarrow$ NV$^0$ probability"
    )

    if show_percent:
        plot_matrix = (
            100.0 * plot_matrix
        )

        colorbar_label += " (%)"

    # Excluded and unavailable entries appear gray.
    masked_matrix = np.ma.masked_invalid(
        plot_matrix
    )

    finite_values = plot_matrix[
        np.isfinite(
            plot_matrix
        )
    ]

    if finite_values.size == 0:
        raise ValueError(
            "No finite loss probabilities remain after filtering."
        )

    # ==============================================================
    # Color normalization
    # ==============================================================

    if subtract_zero_wait:
        color_limit = float(
            np.nanpercentile(
                np.abs(
                    finite_values
                ),
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

    # ==============================================================
    # Plot
    # ==============================================================

    num_waits = int(
        plot_matrix.shape[0]
    )

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
        if np.isclose(
            current_wait_s,
            0.0,
        ):
            label = "0 s"

        elif current_wait_s < 60:
            label = (
                f"{current_wait_s:g} s"
            )

        else:
            label = (
                f"{current_wait_s / 60:g} min"
            )

        wait_labels.append(
            label
        )

    ax.set_yticks(
        np.arange(
            num_waits
        )
    )

    ax.set_yticklabels(
        wait_labels
    )

    ax.set_xlabel(
        "Original NV index"
    )

    ax.set_ylabel(
        "Dark wait"
    )

    ax.set_title(
        title
    )

    # Horizontal boundaries separate the wait-time rows.
    for boundary in (
        np.arange(
            num_waits + 1
        )
        - 0.5
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

    # ==============================================================
    # Return numerical results
    # ==============================================================

    result = {
        "wait_s": wait_s,

        # Complete original matrices: wait × 631 NVs
        "probability_matrix": (
            probability_matrix
        ),
        "eligible_matrix": (
            eligible_matrix
        ),
        "event_matrix": (
            event_matrix
        ),
        "zero_wait_probability": (
            zero_probability
        ),

        # Compact filtered matrices: wait × 602 NVs in your current run
        "filtered_probability_matrix": (
            filtered_probability_matrix
        ),
        "filtered_eligible_matrix": (
            filtered_eligible_matrix
        ),
        "filtered_event_matrix": (
            filtered_event_matrix
        ),
        "filtered_zero_wait_probability": (
            filtered_zero_probability
        ),

        "plot_matrix": (
            plot_matrix
        ),

        "excluded_nv_mask": (
            excluded_mask
        ),
        "excluded_nv_inds": (
            excluded_nv_inds.tolist()
        ),
        "manually_excluded_nv_inds": (
            manually_excluded_nv_inds.tolist()
        ),
        "automatically_excluded_nv_inds": (
            automatically_excluded_nv_inds.tolist()
        ),
        "kept_nv_inds": (
            kept_nv_inds.tolist()
        ),

        "num_original_nvs": int(
            num_nvs
        ),
        "num_retained_nvs": int(
            kept_nv_inds.size
        ),

        "max_zero_wait_loss_probability": (
            max_zero_wait_loss_probability
        ),
        "min_zero_wait_eligible_runs": (
            min_zero_wait_eligible_runs
        ),

        "zero_wait_eligible_count": (
            zero_eligible_count
        ),
        "zero_wait_event_count": (
            zero_event_count
        ),

        "filtered_analyses": (
            filtered_analyses
        ),
    }

    return result, fig

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


def find_persistent_bad_nvs(
    analyses,
    selected_waits_s=None,
    high_probability_threshold=0.10,
    min_fraction_high=0.70,
    min_high_waits=6,
    min_valid_waits=7,
    min_eligible_per_wait=10,
    min_median_probability=0.10,
    min_pooled_probability=0.10,
    require_short_and_long_waits=True,
    short_wait_max_s=60.0,
    long_wait_min_s=300.0,
    min_short_high_waits=2,
    min_long_high_waits=2,
    verbose=True,
):
    """
    Identify NVs with consistently high NV- -> NV0 probability throughout
    the dark-wait sweep.

    The function operates entirely on the cached ``analyses`` dictionaries.
    It does not reload raw data or rerun charge-state classification.
    """

    if not analyses:
        raise ValueError(
            "No analyses were supplied."
        )

    if not 0.0 <= high_probability_threshold <= 1.0:
        raise ValueError(
            "high_probability_threshold must lie in [0, 1]."
        )

    if not 0.0 <= min_fraction_high <= 1.0:
        raise ValueError(
            "min_fraction_high must lie in [0, 1]."
        )

    # --------------------------------------------------------------
    # Sort datasets by wait time
    # --------------------------------------------------------------

    all_analyses = sorted(
        list(analyses),
        key=lambda analysis: float(
            analysis["dark_wait_s"]
        ),
    )

    available_waits_s = np.asarray(
        [
            float(analysis["dark_wait_s"])
            for analysis in all_analyses
        ],
        dtype=float,
    )

    # --------------------------------------------------------------
    # Select the requested wait times
    # --------------------------------------------------------------

    if selected_waits_s is None:
        selected_indices = list(
            range(len(all_analyses))
        )

    else:
        selected_indices = []

        for requested_wait_s in selected_waits_s:
            closest_ind = int(
                np.argmin(
                    np.abs(
                        available_waits_s
                        - float(requested_wait_s)
                    )
                )
            )

            if closest_ind not in selected_indices:
                selected_indices.append(
                    closest_ind
                )

        selected_indices.sort(
            key=lambda ind: available_waits_s[ind]
        )

    selected_analyses = [
        all_analyses[ind]
        for ind in selected_indices
    ]

    wait_s = available_waits_s[
        selected_indices
    ]

    # --------------------------------------------------------------
    # Build wait × NV matrices
    # --------------------------------------------------------------

    probability_rows = []
    eligible_rows = []
    event_rows = []

    num_nvs_reference = None

    for analysis in selected_analyses:
        initial_mask = np.asarray(
            analysis["initial_nvm_mask"],
            dtype=bool,
        )

        event_mask = np.asarray(
            analysis[
                "candidate_nvm_to_nv0_mask"
            ],
            dtype=bool,
        )

        if initial_mask.shape != event_mask.shape:
            raise ValueError(
                "Initial and event masks have different shapes."
            )

        if initial_mask.ndim != 2:
            raise ValueError(
                "Expected masks with shape [nv, run]."
            )

        num_nvs = initial_mask.shape[0]

        if num_nvs_reference is None:
            num_nvs_reference = num_nvs

        elif num_nvs != num_nvs_reference:
            raise ValueError(
                "Different wait datasets contain different NV counts."
            )

        eligible_count = np.sum(
            initial_mask,
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

        probability_rows.append(
            probability
        )

        eligible_rows.append(
            eligible_count
        )

        event_rows.append(
            event_count
        )

    probability_matrix = np.asarray(
        probability_rows,
        dtype=float,
    )

    eligible_matrix = np.asarray(
        eligible_rows,
        dtype=float,
    )

    event_matrix = np.asarray(
        event_rows,
        dtype=float,
    )

    num_waits, num_nvs = (
        probability_matrix.shape
    )

    # --------------------------------------------------------------
    # Determine valid and high-probability measurements
    # --------------------------------------------------------------

    valid_matrix = (
        np.isfinite(
            probability_matrix
        )
        & (
            eligible_matrix
            >= int(min_eligible_per_wait)
        )
    )

    high_matrix = (
        valid_matrix
        & (
            probability_matrix
            >= float(
                high_probability_threshold
            )
        )
    )

    num_valid_waits_by_nv = np.sum(
        valid_matrix,
        axis=0,
    )

    num_high_waits_by_nv = np.sum(
        high_matrix,
        axis=0,
    )

    fraction_high_by_nv = np.full(
        num_nvs,
        np.nan,
        dtype=float,
    )

    has_valid_waits = (
        num_valid_waits_by_nv > 0
    )

    fraction_high_by_nv[
        has_valid_waits
    ] = (
        num_high_waits_by_nv[
            has_valid_waits
        ]
        / num_valid_waits_by_nv[
            has_valid_waits
        ]
    )

    # --------------------------------------------------------------
    # Median probability across valid waits
    # --------------------------------------------------------------

    median_probability_by_nv = np.full(
        num_nvs,
        np.nan,
        dtype=float,
    )

    for nv_ind in range(num_nvs):
        values = probability_matrix[
            valid_matrix[:, nv_ind],
            nv_ind,
        ]

        if values.size > 0:
            median_probability_by_nv[
                nv_ind
            ] = float(
                np.median(values)
            )

    # --------------------------------------------------------------
    # Pooled probability across all valid waits
    # --------------------------------------------------------------

    pooled_probability_by_nv = np.full(
        num_nvs,
        np.nan,
        dtype=float,
    )

    for nv_ind in range(num_nvs):
        valid_wait_mask = valid_matrix[
            :,
            nv_ind,
        ]

        total_eligible = np.sum(
            eligible_matrix[
                valid_wait_mask,
                nv_ind,
            ]
        )

        total_events = np.sum(
            event_matrix[
                valid_wait_mask,
                nv_ind,
            ]
        )

        if total_eligible > 0:
            pooled_probability_by_nv[
                nv_ind
            ] = (
                total_events
                / total_eligible
            )

    # --------------------------------------------------------------
    # Require persistence at both short and long waits
    # --------------------------------------------------------------

    short_wait_rows = (
        wait_s
        <= float(short_wait_max_s)
    )

    long_wait_rows = (
        wait_s
        >= float(long_wait_min_s)
    )

    num_short_high_by_nv = np.sum(
        high_matrix[
            short_wait_rows
        ],
        axis=0,
    )

    num_long_high_by_nv = np.sum(
        high_matrix[
            long_wait_rows
        ],
        axis=0,
    )

    # --------------------------------------------------------------
    # Final persistent-bad-actor criterion
    # --------------------------------------------------------------

    bad_mask = (
        (
            num_valid_waits_by_nv
            >= int(min_valid_waits)
        )
        & (
            num_high_waits_by_nv
            >= int(min_high_waits)
        )
        & (
            fraction_high_by_nv
            >= float(min_fraction_high)
        )
        & (
            median_probability_by_nv
            >= float(
                min_median_probability
            )
        )
        & (
            pooled_probability_by_nv
            >= float(
                min_pooled_probability
            )
        )
    )

    if require_short_and_long_waits:
        bad_mask &= (
            (
                num_short_high_by_nv
                >= int(min_short_high_waits)
            )
            & (
                num_long_high_by_nv
                >= int(min_long_high_waits)
            )
        )

    bad_nv_inds = np.where(
        bad_mask
    )[0].astype(int)

    kept_nv_inds = np.where(
        ~bad_mask
    )[0].astype(int)

    # --------------------------------------------------------------
    # Diagnostic output
    # --------------------------------------------------------------

    if verbose:
        print(
            "\n"
            + "=" * 78
        )
        print(
            "PERSISTENT FULL-SWEEP BAD-ACTOR FILTER"
        )
        print(
            "=" * 78
        )

        print(
            "Wait times:",
            wait_s.tolist(),
        )

        print(
            "Total NVs:",
            num_nvs,
        )

        print(
            "High probability threshold:",
            high_probability_threshold,
        )

        print(
            "Minimum high waits:",
            min_high_waits,
            "/",
            num_waits,
        )

        print(
            "Minimum fraction high:",
            min_fraction_high,
        )

        print(
            "Minimum valid waits:",
            min_valid_waits,
        )

        print(
            "Minimum median probability:",
            min_median_probability,
        )

        print(
            "Minimum pooled probability:",
            min_pooled_probability,
        )

        print(
            "Persistent bad NV count:",
            bad_nv_inds.size,
        )

        print(
            "Persistent bad NV indices:",
            bad_nv_inds.tolist(),
        )

        print(
            "NVs retained:",
            kept_nv_inds.size,
            "/",
            num_nvs,
        )

        if bad_nv_inds.size > 0:
            print(
                "\nRemoved NV details:"
            )

            for nv_ind in bad_nv_inds:
                values_percent = []

                for value in probability_matrix[
                    :,
                    nv_ind,
                ]:
                    if np.isfinite(value):
                        values_percent.append(
                            f"{100.0 * value:.1f}"
                        )
                    else:
                        values_percent.append(
                            "nan"
                        )

                print(
                    f"NV {nv_ind:3d}: "
                    f"high={num_high_waits_by_nv[nv_ind]}/"
                    f"{num_valid_waits_by_nv[nv_ind]}, "
                    f"short-high="
                    f"{num_short_high_by_nv[nv_ind]}, "
                    f"long-high="
                    f"{num_long_high_by_nv[nv_ind]}, "
                    f"median="
                    f"{100 * median_probability_by_nv[nv_ind]:.1f}%, "
                    f"pooled="
                    f"{100 * pooled_probability_by_nv[nv_ind]:.1f}%, "
                    f"probabilities="
                    f"[{', '.join(values_percent)}]%"
                )

    result = {
        "wait_s": wait_s,
        "probability_matrix": probability_matrix,
        "eligible_matrix": eligible_matrix,
        "event_matrix": event_matrix,
        "valid_matrix": valid_matrix,
        "high_matrix": high_matrix,
        "num_valid_waits_by_nv": (
            num_valid_waits_by_nv
        ),
        "num_high_waits_by_nv": (
            num_high_waits_by_nv
        ),
        "fraction_high_by_nv": (
            fraction_high_by_nv
        ),
        "median_probability_by_nv": (
            median_probability_by_nv
        ),
        "pooled_probability_by_nv": (
            pooled_probability_by_nv
        ),
        "num_short_high_by_nv": (
            num_short_high_by_nv
        ),
        "num_long_high_by_nv": (
            num_long_high_by_nv
        ),
        "bad_nv_mask": bad_mask,
        "bad_nv_inds": (
            bad_nv_inds.tolist()
        ),
        "kept_nv_inds": (
            kept_nv_inds.tolist()
        ),
    }

    return bad_nv_inds.tolist(), result

def fit_individual_nv_dark_survival(
    analyses,
    min_eligible_per_wait=10,
    min_valid_waits=6,
    min_fit_amplitude=0.005,
    min_r_squared=0.0,
    max_relative_tau_error=3.0,
    max_tau_factor=100.0,
    verbose=True,
):
    """
    Fit every NV independently to

        D_i(t) = D_inf,i + (1 - D_inf,i) exp(-t / tau_i)

    where D_i(t) is that NV's retention normalized to its measured
    zero-wait retention.

    Parameters
    ----------
    analyses:
        Preferably the filtered analyses returned by
        plot_nv_loss_row_by_row()["filtered_analyses"].

    min_eligible_per_wait:
        Minimum number of initial NV- trials required at a wait time.

    min_valid_waits:
        Minimum number of usable wait-time measurements needed for fitting.

    min_fit_amplitude:
        Minimum fitted loss amplitude, 1 - D_inf, required for a
        meaningful decay fit.

    min_r_squared:
        Minimum R-squared for inclusion in the lifetime distribution.

    max_relative_tau_error:
        Maximum allowed tau_stderr / tau.

    max_tau_factor:
        Reject fits with tau greater than this factor times the longest
        measured wait time.
    """

    if not analyses:
        raise ValueError(
            "No analyses were supplied."
        )

    sorted_analyses = sorted(
        list(analyses),
        key=lambda analysis: float(
            analysis["dark_wait_s"]
        ),
    )

    wait_s = np.asarray(
        [
            float(analysis["dark_wait_s"])
            for analysis in sorted_analyses
        ],
        dtype=float,
    )

    zero_rows = np.where(
        np.isclose(
            wait_s,
            0.0,
        )
    )[0]

    if zero_rows.size == 0:
        raise ValueError(
            "A 0 s dataset is required for individual-NV normalization."
        )

    zero_row_ind = int(
        zero_rows[0]
    )

    probability_rows = []
    probability_sem_rows = []
    eligible_rows = []
    retained_rows = []

    reference_num_nvs = None

    for analysis in sorted_analyses:
        initial_mask = np.asarray(
            analysis["initial_nvm_mask"],
            dtype=bool,
        )

        retained_mask = np.asarray(
            analysis["retained_nvm_mask"],
            dtype=bool,
        )

        if initial_mask.shape != retained_mask.shape:
            raise ValueError(
                "Initial and retained masks have different shapes."
            )

        if initial_mask.ndim != 2:
            raise ValueError(
                "Expected masks with shape [nv, run]."
            )

        num_nvs = initial_mask.shape[0]

        if reference_num_nvs is None:
            reference_num_nvs = num_nvs

        elif num_nvs != reference_num_nvs:
            raise ValueError(
                "The wait datasets contain different numbers of NVs."
            )

        eligible_count = np.sum(
            initial_mask,
            axis=1,
        ).astype(float)

        retained_count = np.sum(
            retained_mask,
            axis=1,
        ).astype(float)

        retention_probability = np.full(
            num_nvs,
            np.nan,
            dtype=float,
        )

        good = eligible_count > 0

        retention_probability[good] = (
            retained_count[good]
            / eligible_count[good]
        )

        # Adjusted binomial estimate prevents zero uncertainty when
        # all or none of the trials are retained.
        adjusted_probability = np.full(
            num_nvs,
            np.nan,
            dtype=float,
        )

        adjusted_probability[good] = (
            retained_count[good] + 0.5
        ) / (
            eligible_count[good] + 1.0
        )

        retention_sem = np.full(
            num_nvs,
            np.nan,
            dtype=float,
        )

        retention_sem[good] = np.sqrt(
            adjusted_probability[good]
            * (
                1.0
                - adjusted_probability[good]
            )
            / (
                eligible_count[good] + 1.0
            )
        )

        probability_rows.append(
            retention_probability
        )

        probability_sem_rows.append(
            retention_sem
        )

        eligible_rows.append(
            eligible_count
        )

        retained_rows.append(
            retained_count
        )

    retention_matrix = np.asarray(
        probability_rows,
        dtype=float,
    )

    retention_sem_matrix = np.asarray(
        probability_sem_rows,
        dtype=float,
    )

    eligible_matrix = np.asarray(
        eligible_rows,
        dtype=float,
    )

    retained_matrix = np.asarray(
        retained_rows,
        dtype=float,
    )

    num_waits, num_nvs = (
        retention_matrix.shape
    )

    # filtered_analyses preserves the original NV indices.
    original_nv_inds_raw = sorted_analyses[0].get(
        "original_nv_inds"
    )

    if original_nv_inds_raw is None:
        original_nv_inds = np.arange(
            num_nvs,
            dtype=int,
        )

    else:
        original_nv_inds = np.asarray(
            original_nv_inds_raw,
            dtype=int,
        )

        if original_nv_inds.size != num_nvs:
            raise ValueError(
                "original_nv_inds has the wrong length."
            )

    max_wait_s = float(
        np.max(wait_s)
    )

    maximum_allowed_tau_s = (
        max_tau_factor
        * max_wait_s
    )

    fit_results = []

    for compact_nv_ind in range(num_nvs):
        original_nv_ind = int(
            original_nv_inds[
                compact_nv_ind
            ]
        )

        valid_mask = (
            np.isfinite(
                retention_matrix[
                    :,
                    compact_nv_ind
                ]
            )
            & (
                eligible_matrix[
                    :,
                    compact_nv_ind
                ]
                >= int(min_eligible_per_wait)
            )
        )

        num_valid_waits = int(
            np.sum(valid_mask)
        )

        result = {
            "compact_nv_ind": int(
                compact_nv_ind
            ),
            "original_nv_ind": (
                original_nv_ind
            ),
            "success": False,
            "quality_pass": False,
            "num_valid_waits": (
                num_valid_waits
            ),
        }

        if num_valid_waits < min_valid_waits:
            result["error"] = (
                "Too few valid wait times."
            )

            fit_results.append(
                result
            )

            continue

        if not valid_mask[zero_row_ind]:
            result["error"] = (
                "Insufficient zero-wait trials."
            )

            fit_results.append(
                result
            )

            continue

        zero_retention = float(
            retention_matrix[
                zero_row_ind,
                compact_nv_ind,
            ]
        )

        zero_retention_sem = float(
            retention_sem_matrix[
                zero_row_ind,
                compact_nv_ind,
            ]
        )

        if (
            not np.isfinite(zero_retention)
            or zero_retention <= 0
        ):
            result["error"] = (
                "Invalid zero-wait retention."
            )

            fit_results.append(
                result
            )

            continue

        retention = retention_matrix[
            :,
            compact_nv_ind
        ]

        retention_sem = (
            retention_sem_matrix[
                :,
                compact_nv_ind
            ]
        )

        dark_survival = (
            retention
            / zero_retention
        )

        # Approximate propagated uncertainty for the normalized ratio.
        dark_survival_sem = np.sqrt(
            (
                retention_sem
                / zero_retention
            ) ** 2
            + (
                retention
                * zero_retention_sem
                / zero_retention**2
            ) ** 2
        )

        dark_survival[
            zero_row_ind
        ] = 1.0

        valid_sem = dark_survival_sem[
            valid_mask
        ]

        positive_sem = valid_sem[
            np.isfinite(valid_sem)
            & (valid_sem > 0)
        ]

        sem_floor = (
            float(
                np.nanmedian(
                    positive_sem
                )
            )
            if positive_sem.size > 0
            else 0.01
        )

        dark_survival_sem[
            zero_row_ind
        ] = max(
            sem_floor,
            1e-3,
        )

        try:
            fit = _fit_dark_survival(
                wait_s[
                    valid_mask
                ],
                dark_survival[
                    valid_mask
                ],
                dark_survival_sem[
                    valid_mask
                ],
            )

        except Exception as exc:
            result["error"] = str(exc)

            fit_results.append(
                result
            )

            continue

        if not fit.get(
            "success",
            False,
        ):
            result["error"] = fit.get(
                "error",
                "Fit failed.",
            )

            fit_results.append(
                result
            )

            continue

        plateau = float(
            fit["plateau"]
        )

        tau_s = float(
            fit["tau_dark_s"]
        )

        tau_stderr_s = float(
            fit["tau_dark_s_stderr"]
        )

        r_squared = float(
            fit["r_squared"]
        )

        fit_amplitude = (
            1.0 - plateau
        )

        relative_tau_error = (
            tau_stderr_s / tau_s
            if (
                np.isfinite(tau_stderr_s)
                and tau_s > 0
            )
            else np.inf
        )

        predicted_survival_at_max_wait = float(
            _dark_survival_model(
                max_wait_s,
                plateau,
                tau_s,
            )
        )

        predicted_loss_at_max_wait = (
            1.0
            - predicted_survival_at_max_wait
        )

        quality_pass = bool(
            np.isfinite(tau_s)
            and tau_s > 0
            and tau_s <= maximum_allowed_tau_s
            and fit_amplitude
            >= float(min_fit_amplitude)
            and np.isfinite(r_squared)
            and r_squared
            >= float(min_r_squared)
            and relative_tau_error
            <= float(max_relative_tau_error)
        )

        result.update(
            {
                "success": True,
                "quality_pass": (
                    quality_pass
                ),
                "plateau": plateau,
                "plateau_stderr": float(
                    fit[
                        "plateau_stderr"
                    ]
                ),
                "fit_amplitude": (
                    fit_amplitude
                ),
                "tau_s": tau_s,
                "tau_s_stderr": (
                    tau_stderr_s
                ),
                "tau_min": (
                    tau_s / 60.0
                ),
                "tau_min_stderr": (
                    tau_stderr_s / 60.0
                ),
                "relative_tau_error": (
                    relative_tau_error
                ),
                "r_squared": (
                    r_squared
                ),
                "predicted_loss_at_max_wait": (
                    predicted_loss_at_max_wait
                ),
                "wait_s": (
                    wait_s.tolist()
                ),
                "valid_mask": (
                    valid_mask.tolist()
                ),
                "dark_survival": (
                    dark_survival.tolist()
                ),
                "dark_survival_sem": (
                    dark_survival_sem.tolist()
                ),
                "fit_x_s": (
                    fit["fit_x_s"]
                ),
                "fit_y": (
                    fit["fit_y"]
                ),
            }
        )

        fit_results.append(
            result
        )

    successful_fits = [
        fit
        for fit in fit_results
        if fit.get(
            "success",
            False,
        )
    ]

    quality_fits = [
        fit
        for fit in successful_fits
        if fit.get(
            "quality_pass",
            False,
        )
    ]

    if verbose:
        print(
            "\n"
            + "=" * 78
        )

        print(
            "INDIVIDUAL-NV DARK-SURVIVAL FITS"
        )

        print(
            "=" * 78
        )

        print(
            "NVs analyzed:",
            num_nvs,
        )

        print(
            "Successful fits:",
            len(successful_fits),
        )

        print(
            "Fits passing quality cuts:",
            len(quality_fits),
        )

        if quality_fits:
            tau_values_min = np.asarray(
                [
                    fit["tau_min"]
                    for fit in quality_fits
                ],
                dtype=float,
            )

            print(
                "Median tau:",
                f"{np.median(tau_values_min):.2f} min",
            )

            print(
                "Central 68% tau interval:",
                f"{np.percentile(tau_values_min, 16):.2f}",
                "to",
                f"{np.percentile(tau_values_min, 84):.2f}",
                "min",
            )

    return {
        "wait_s": wait_s,
        "original_nv_inds": (
            original_nv_inds
        ),
        "retention_matrix": (
            retention_matrix
        ),
        "retention_sem_matrix": (
            retention_sem_matrix
        ),
        "eligible_matrix": (
            eligible_matrix
        ),
        "retained_matrix": (
            retained_matrix
        ),
        "fit_results": (
            fit_results
        ),
        "successful_fits": (
            successful_fits
        ),
        "quality_fits": (
            quality_fits
        ),
        "maximum_allowed_tau_s": (
            maximum_allowed_tau_s
        ),
    }
    
def plot_individual_nv_lifetime_histogram(
    individual_fit_result,
    quality_only=True,
    num_bins=20,
):
    """
    Plot the distribution of independently fitted NV charge lifetimes.
    """

    fits = (
        individual_fit_result[
            "quality_fits"
        ]
        if quality_only
        else individual_fit_result[
            "successful_fits"
        ]
    )

    tau_min = np.asarray(
        [
            fit["tau_min"]
            for fit in fits
            if (
                np.isfinite(
                    fit["tau_min"]
                )
                and fit["tau_min"] > 0
            )
        ],
        dtype=float,
    )

    if tau_min.size == 0:
        raise ValueError(
            "No valid individual-NV lifetimes are available."
        )

    lower = float(
        np.min(tau_min)
    )

    upper = float(
        np.max(tau_min)
    )

    if np.isclose(
        lower,
        upper,
    ):
        lower *= 0.8
        upper *= 1.2

    bins = np.logspace(
        np.log10(lower),
        np.log10(upper),
        int(num_bins) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    ax.hist(
        tau_min,
        bins=bins,
        edgecolor="black",
        alpha=0.8,
    )

    median_tau_min = float(
        np.median(tau_min)
    )

    ax.axvline(
        median_tau_min,
        linestyle="--",
        linewidth=2,
        label=(
            f"Median = "
            f"{median_tau_min:.2f} min"
        ),
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Individual-NV fitted charge lifetime, "
        r"$\tau_i$ (min)"
    )

    ax.set_ylabel(
        "Number of NVs"
    )

    ax.set_title(
        "Distribution of individual-NV "
        "dark charge lifetimes"
    )

    ax.legend()

    fig.tight_layout()

    return fig


def plot_fastest_individual_nv_curves(
    individual_fit_result,
    top_n=5,
    quality_only=True,
):
    """
    Plot the NVs with the largest fitted loss at the longest measured
    dark wait.
    """

    fits = (
        individual_fit_result[
            "quality_fits"
        ]
        if quality_only
        else individual_fit_result[
            "successful_fits"
        ]
    )

    if not fits:
        raise ValueError(
            "No individual-NV fits are available."
        )

    ranked_fits = sorted(
        fits,
        key=lambda fit: float(
            fit[
                "predicted_loss_at_max_wait"
            ]
        ),
        reverse=True,
    )

    selected_fits = ranked_fits[
        : int(top_n)
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 6.0)
    )

    selected_original_nv_inds = []

    for fit in selected_fits:
        original_nv_ind = int(
            fit["original_nv_ind"]
        )

        selected_original_nv_inds.append(
            original_nv_ind
        )

        wait_s = np.asarray(
            fit["wait_s"],
            dtype=float,
        )

        dark_survival = np.asarray(
            fit["dark_survival"],
            dtype=float,
        )

        dark_survival_sem = np.asarray(
            fit["dark_survival_sem"],
            dtype=float,
        )

        valid_mask = np.asarray(
            fit["valid_mask"],
            dtype=bool,
        )

        fit_x_s = np.asarray(
            fit["fit_x_s"],
            dtype=float,
        )

        fit_y = np.asarray(
            fit["fit_y"],
            dtype=float,
        )

        label = (
            f"NV {original_nv_ind}: "
            f"$\\tau$={fit['tau_min']:.2f} min, "
            f"loss@max="
            f"{100 * fit['predicted_loss_at_max_wait']:.1f}%"
        )

        ax.errorbar(
            wait_s[
                valid_mask
            ],
            dark_survival[
                valid_mask
            ],
            yerr=dark_survival_sem[
                valid_mask
            ],
            marker="o",
            linestyle="none",
            capsize=2,
        )

        ax.plot(
            fit_x_s,
            fit_y,
            linewidth=2,
            label=label,
        )

    ax.set_xscale(
        "symlog",
        linthresh=10.0,
    )

    ax.set_xlabel(
        "Dark wait time (s)"
    )

    ax.set_ylabel(
        "Dark survival relative to 0 s"
    )

    ax.set_title(
        f"Top {len(selected_fits)} fastest-decaying NVs"
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    print(
        "\nNVs most likely to switch at long times:",
        selected_original_nv_inds,
    )

    return (
        fig,
        selected_original_nv_inds,
    )
def remove_individual_nv_fit_outliers(
    individual_fit_result,
    mad_z_threshold=3.5,
    min_r_squared=0.0,
    max_relative_tau_error=1.5,
    max_tau_factor=10.0,
    min_tau_s=1.0,
    verbose=True,
):
    """
    Remove pathological individual-NV lifetime fits.

    Filtering occurs in two stages:

    1. Remove poorly constrained fits:
       - nonfinite or nonpositive tau
       - low R-squared
       - excessive relative tau uncertainty
       - tau far beyond the experimental time window

    2. Remove statistical outliers using a robust modified z-score
       applied to log10(tau).

    Returns a copy of individual_fit_result whose ``quality_fits`` entry
    contains only the cleaned fits. The original result is not modified.
    """

    fits = list(
        individual_fit_result[
            "quality_fits"
        ]
    )

    wait_s = np.asarray(
        individual_fit_result["wait_s"],
        dtype=float,
    )

    max_measured_wait_s = float(
        np.nanmax(wait_s)
    )

    max_allowed_tau_s = (
        float(max_tau_factor)
        * max_measured_wait_s
    )

    preliminarily_kept = []
    quality_rejected = []

    # --------------------------------------------------------------
    # Remove poorly constrained fits
    # --------------------------------------------------------------

    for fit in fits:
        tau_s = float(
            fit.get("tau_s", np.nan)
        )

        tau_stderr_s = float(
            fit.get(
                "tau_s_stderr",
                np.nan,
            )
        )

        r_squared = float(
            fit.get(
                "r_squared",
                np.nan,
            )
        )

        relative_tau_error = (
            tau_stderr_s / tau_s
            if (
                np.isfinite(tau_s)
                and tau_s > 0
                and np.isfinite(tau_stderr_s)
            )
            else np.inf
        )

        rejection_reasons = []

        if (
            not np.isfinite(tau_s)
            or tau_s <= float(min_tau_s)
        ):
            rejection_reasons.append(
                "invalid or too-small tau"
            )

        if (
            np.isfinite(tau_s)
            and tau_s > max_allowed_tau_s
        ):
            rejection_reasons.append(
                "tau exceeds experimental range"
            )

        if (
            not np.isfinite(r_squared)
            or r_squared
            < float(min_r_squared)
        ):
            rejection_reasons.append(
                "poor R-squared"
            )

        if (
            not np.isfinite(
                relative_tau_error
            )
            or relative_tau_error
            > float(max_relative_tau_error)
        ):
            rejection_reasons.append(
                "large tau uncertainty"
            )

        fit_copy = dict(fit)

        fit_copy[
            "relative_tau_error"
        ] = relative_tau_error

        if rejection_reasons:
            fit_copy[
                "outlier_reason"
            ] = "; ".join(
                rejection_reasons
            )

            quality_rejected.append(
                fit_copy
            )

        else:
            preliminarily_kept.append(
                fit_copy
            )

    if len(preliminarily_kept) < 3:
        raise ValueError(
            "Fewer than three individual-NV fits remain "
            "after fit-quality filtering."
        )

    # --------------------------------------------------------------
    # Robust outlier removal in log10(tau)
    # --------------------------------------------------------------

    tau_s_values = np.asarray(
        [
            fit["tau_s"]
            for fit in preliminarily_kept
        ],
        dtype=float,
    )

    log_tau = np.log10(
        tau_s_values
    )

    median_log_tau = float(
        np.median(log_tau)
    )

    absolute_deviation = np.abs(
        log_tau - median_log_tau
    )

    mad_log_tau = float(
        np.median(
            absolute_deviation
        )
    )

    if (
        not np.isfinite(mad_log_tau)
        or mad_log_tau <= 0
    ):
        # No measurable spread, so do not classify distribution outliers.
        modified_z_score = np.zeros(
            log_tau.size,
            dtype=float,
        )

    else:
        modified_z_score = (
            0.67448975
            * (
                log_tau
                - median_log_tau
            )
            / mad_log_tau
        )

    distribution_outlier_mask = (
        np.abs(
            modified_z_score
        )
        > float(mad_z_threshold)
    )

    cleaned_fits = []
    distribution_outliers = []

    for fit_ind, fit in enumerate(
        preliminarily_kept
    ):
        fit_copy = dict(fit)

        fit_copy[
            "log_tau_modified_z"
        ] = float(
            modified_z_score[
                fit_ind
            ]
        )

        if distribution_outlier_mask[
            fit_ind
        ]:
            fit_copy[
                "outlier_reason"
            ] = (
                "extreme lifetime in "
                "log10(tau) distribution"
            )

            distribution_outliers.append(
                fit_copy
            )

        else:
            cleaned_fits.append(
                fit_copy
            )

    all_removed_fits = (
        quality_rejected
        + distribution_outliers
    )

    # --------------------------------------------------------------
    # Return a compatible result for the existing plotting functions
    # --------------------------------------------------------------

    cleaned_result = dict(
        individual_fit_result
    )

    # Existing plotting functions use quality_fits.
    cleaned_result[
        "quality_fits"
    ] = cleaned_fits

    cleaned_result[
        "outlier_removed_fits"
    ] = all_removed_fits

    cleaned_result[
        "fit_quality_rejected"
    ] = quality_rejected

    cleaned_result[
        "distribution_outliers"
    ] = distribution_outliers

    cleaned_result[
        "outlier_filter_settings"
    ] = {
        "mad_z_threshold": float(
            mad_z_threshold
        ),
        "min_r_squared": float(
            min_r_squared
        ),
        "max_relative_tau_error": float(
            max_relative_tau_error
        ),
        "max_tau_factor": float(
            max_tau_factor
        ),
        "min_tau_s": float(
            min_tau_s
        ),
        "max_allowed_tau_s": float(
            max_allowed_tau_s
        ),
    }

    if verbose:
        print(
            "\n"
            + "=" * 78
        )

        print(
            "INDIVIDUAL-NV FIT OUTLIER REMOVAL"
        )

        print(
            "=" * 78
        )

        print(
            "Initial quality fits:",
            len(fits),
        )

        print(
            "Rejected by fit quality:",
            len(quality_rejected),
        )

        print(
            "Rejected by log-tau MAD:",
            len(distribution_outliers),
        )

        print(
            "Final retained fits:",
            len(cleaned_fits),
        )

        if all_removed_fits:
            print(
                "\nRemoved fits:"
            )

            for fit in all_removed_fits:
                original_nv_ind = fit[
                    "original_nv_ind"
                ]

                tau_min = float(
                    fit.get(
                        "tau_min",
                        np.nan,
                    )
                )

                r_squared = float(
                    fit.get(
                        "r_squared",
                        np.nan,
                    )
                )

                relative_error = float(
                    fit.get(
                        "relative_tau_error",
                        np.nan,
                    )
                )

                reason = fit.get(
                    "outlier_reason",
                    "unknown",
                )

                print(
                    f"NV {original_nv_ind:3d}: "
                    f"tau={tau_min:.3g} min, "
                    f"R²={r_squared:.3f}, "
                    f"relative error={relative_error:.2f}, "
                    f"reason={reason}"
                )

    return cleaned_result


def _find_analysis_for_wait(
    analyses,
    wait_s,
    atol=1e-9,
):
    """Return the analysis dict matching a requested dark wait."""
    for analysis in analyses:
        if np.isclose(float(analysis["dark_wait_s"]), float(wait_s), atol=atol):
            return analysis
    return None


def _detect_run_outliers(
    values,
    method="mad",
    threshold=3.5,
):
    """
    Detect outlier runs from a 1D array.

    method = "mad"
        robust median absolute deviation method

    method = "zscore"
        mean/std based method
    """
    values = np.asarray(values, dtype=float)

    outlier_mask = np.zeros(values.shape, dtype=bool)
    valid = np.isfinite(values)

    if np.sum(valid) < 4:
        return outlier_mask

    x = values[valid]

    if method.lower() == "zscore":
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1))
        if std <= 0:
            return outlier_mask
        z = np.abs((x - mean) / std)
        outlier_mask[valid] = z > threshold
        return outlier_mask

    # Default: robust MAD
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))

    if mad <= 0:
        # fallback to std if MAD collapses
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1))
        if std <= 0:
            return outlier_mask
        z = np.abs((x - mean) / std)
        outlier_mask[valid] = z > 2.5
        return outlier_mask

    robust_z = 0.6745 * (x - median) / mad
    outlier_mask[valid] = np.abs(robust_z) > threshold
    return outlier_mask


def plot_wait_sweep_run_bars(
    analyses,
    selected_waits_s=None,
    metric="num_candidates_by_run",
    outlier_method="mad",
    outlier_threshold=3.5,
    ncols=3,
    sharey=True,
    show_mean=True,
    show_median=True,
    sort_runs=False,
):
    """
    Plot per-run bar charts for each dark-wait dataset.

    Parameters
    ----------
    analyses : list of analysis dicts
        Already loaded particle-memory analyses.

    selected_waits_s : list or None
        Which wait times to show. If None, all available waits are used.

    metric : str
        One of:
            "num_candidates_by_run"
            "retention_by_run"
            "num_ambiguous_by_run"
            "num_initial_nvm_by_run"
            "num_final_nvm_by_run"
            "num_retained_by_run"

    outlier_method : str
        "mad" or "zscore"

    outlier_threshold : float
        Threshold for outlier detection.

    sort_runs : bool
        If True, sort bars within each panel by value.
    """

    metric_label_map = {
        "num_candidates_by_run": "Confident NV$^- \\rightarrow$ NV$^0$ count / run",
        "retention_by_run": "Retention fraction / run",
        "num_ambiguous_by_run": "Ambiguous count / run",
        "num_initial_nvm_by_run": "Initial NV$^-$ count / run",
        "num_final_nvm_by_run": "Final NV$^-$ count / run",
        "num_retained_by_run": "Retained NV$^-$ count / run",
    }

    if metric not in metric_label_map:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Choose from {list(metric_label_map.keys())}."
        )

    if selected_waits_s is None:
        selected_waits_s = sorted(
            [float(analysis["dark_wait_s"]) for analysis in analyses]
        )

    selected_analyses = []
    for wait_s in selected_waits_s:
        analysis = _find_analysis_for_wait(analyses, wait_s)
        if analysis is not None:
            selected_analyses.append(analysis)

    if len(selected_analyses) == 0:
        raise ValueError("No matching analyses found for selected_waits_s.")

    num_panels = len(selected_analyses)
    ncols = min(ncols, num_panels)
    nrows = int(np.ceil(num_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.8 * nrows),
        sharey=sharey,
    )

    axes = np.atleast_1d(axes).ravel()

    results = {
        "metric": metric,
        "waits_s": [],
        "per_wait": [],
    }

    for panel_ind, analysis in enumerate(selected_analyses):
        ax = axes[panel_ind]

        wait_s = float(analysis["dark_wait_s"])
        values = np.asarray(analysis[metric], dtype=float)
        run_inds = np.arange(len(values), dtype=int)

        if sort_runs:
            sort_inds = np.argsort(values)
            values = values[sort_inds]
            run_inds = run_inds[sort_inds]

        outlier_mask = _detect_run_outliers(
            values,
            method=outlier_method,
            threshold=outlier_threshold,
        )

        colors = np.array(
            [kpl.KplColors.BLUE] * len(values),
            dtype=object,
        )
        colors[outlier_mask] = kpl.KplColors.RED

        ax.bar(
            np.arange(len(values)),
            values,
            color=colors,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
        )

        if show_mean:
            mean_val = float(np.nanmean(values))
            ax.axhline(
                mean_val,
                color=kpl.KplColors.GREEN,
                linestyle="--",
                linewidth=1.5,
                label=f"mean = {mean_val:.3g}",
            )
        else:
            mean_val = np.nan

        if show_median:
            median_val = float(np.nanmedian(values))
            ax.axhline(
                median_val,
                color=kpl.KplColors.GRAY,
                linestyle=":",
                linewidth=1.5,
                label=f"median = {median_val:.3g}",
            )
        else:
            median_val = np.nan

        ax.set_title(
            f"wait = {wait_s:g} s\n"
            f"runs = {len(values)}, outliers = {int(np.sum(outlier_mask))}",
            fontsize=11,
        )
        ax.set_xlabel("Run index" if not sort_runs else "Sorted run index")
        ax.set_ylabel(metric_label_map[metric])
        ax.grid(True, axis="y", alpha=0.3)

        # Show original run numbers if sorted=False; otherwise show dense index
        if not sort_runs:
            x_positions = np.arange(len(values))
            ax.set_xticks(x_positions)
            if len(values) <= 20:
                ax.set_xticklabels([str(ind) for ind in run_inds], rotation=0)
            else:
                step = max(1, len(values) // 10)
                shown = np.arange(0, len(values), step)
                ax.set_xticks(shown)
                ax.set_xticklabels([str(run_inds[i]) for i in shown], rotation=0)

        ax.legend(fontsize=8, loc="best")

        results["waits_s"].append(wait_s)
        results["per_wait"].append(
            {
                "dark_wait_s": wait_s,
                "values": values.tolist(),
                "mean": mean_val,
                "median": median_val,
                "outlier_run_inds": run_inds[outlier_mask].astype(int).tolist(),
                "num_outliers": int(np.sum(outlier_mask)),
            }
        )

    # turn off unused axes
    for ax in axes[num_panels:]:
        ax.axis("off")

    fig.suptitle(
        f"Per-run bar plots: {metric_label_map[metric]}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return results, fig


import numpy as np
import matplotlib.pyplot as plt

from utils import data_manager as dm
from utils import kplotlib as kpl


def plot_nv_minus_by_run_separate_reps(
    file_stems,
    selected_waits_s=None,
    rep_inds=(1, 11, 12),
    rep_labels=None,
    exclude_nv_inds=None,
    show_fraction=False,
    ncols=3,
    verbose=True,
):
    """
    Make 3 separate figures (or as many reps as requested), one figure per rep.

    Each figure contains one subplot per dark-wait dataset, and each subplot
    shows the run-by-run number (or fraction) of NVs classified as NV-.

    Parameters
    ----------
    file_stems : sequence of str
        Raw particle-memory dataset file stems.

    selected_waits_s : sequence or None
        Only include these dark-wait times.

    rep_inds : tuple/list of int
        Rep indices to inspect, e.g. (1, 11, 12).

    rep_labels : dict or None
        Optional labels for rep indices, e.g.
        {1: "rep 1: early init", 11: "rep 11: initial check", 12: "rep 12: after wait"}

    exclude_nv_inds : sequence or None
        Original NV indices to exclude.

    show_fraction : bool
        False -> plot number of NV-.
        True  -> plot fraction of kept NVs classified NV-.

    ncols : int
        Number of subplot columns per figure.

    Returns
    -------
    dataset_results : list of dict
        Per-dataset computed run-by-run values.

    figures : dict
        figures[rep_ind] = matplotlib figure
    """

    if exclude_nv_inds is None:
        exclude_nv_inds = np.array([], dtype=int)
    else:
        exclude_nv_inds = np.unique(np.asarray(exclude_nv_inds, dtype=int))

    if rep_labels is None:
        rep_labels = {
            rep_inds[0]: f"rep {rep_inds[0]}",
            rep_inds[1]: f"rep {rep_inds[1]}",
            rep_inds[2]: f"rep {rep_inds[2]}",
        }

    # --------------------------------------------------------------
    # Load all requested datasets
    # --------------------------------------------------------------
    dataset_results = []

    for file_stem in file_stems:
        raw_data = dm.get_raw_data(
            file_stem=file_stem,
            load_npz=True,
        )

        wait_s = float(raw_data["dark_wait_s"])

        if selected_waits_s is not None:
            selected_waits_s = np.asarray(selected_waits_s, dtype=float)
            if not np.any(np.isclose(wait_s, selected_waits_s)):
                continue

        counts_all = np.asarray(raw_data["counts"], dtype=float)

        if counts_all.ndim != 5:
            raise ValueError(
                f"Expected counts[exp, nv, run, step, rep], got {counts_all.shape}"
            )

        # counts shape -> [nv, run, rep]
        counts = counts_all[0, :, :, 0, :]
        num_nvs, num_runs, num_reps = counts.shape

        if "analysis_thresholds" in raw_data:
            thresholds = np.asarray(raw_data["analysis_thresholds"], dtype=float)
        elif "thresholds" in raw_data:
            thresholds = np.asarray(raw_data["thresholds"], dtype=float)
        else:
            raise ValueError(f"No thresholds found in {file_stem}")

        if thresholds.shape != (num_nvs,):
            raise ValueError(
                f"Threshold shape mismatch: {thresholds.shape} vs {(num_nvs,)}"
            )

        # Apply optional NV exclusion
        keep_mask = np.ones(num_nvs, dtype=bool)
        valid_excluded = exclude_nv_inds[
            (exclude_nv_inds >= 0) & (exclude_nv_inds < num_nvs)
        ]
        keep_mask[valid_excluded] = False

        counts = counts[keep_mask, :, :]
        thresholds = thresholds[keep_mask]
        num_kept_nvs = int(np.sum(keep_mask))

        # classify all reps with the same threshold
        nvm_mask = counts > thresholds[:, None, None]

        rep_counts = {}
        for rep_ind in rep_inds:
            if rep_ind < 0 or rep_ind >= num_reps:
                raise ValueError(
                    f"Requested rep {rep_ind} is outside range [0, {num_reps - 1}] "
                    f"for file {file_stem}"
                )

            values = np.sum(nvm_mask[:, :, rep_ind], axis=0).astype(float)

            if show_fraction:
                values = values / num_kept_nvs

            rep_counts[rep_ind] = values

        dataset_results.append(
            {
                "file_stem": file_stem,
                "dark_wait_s": wait_s,
                "num_runs": num_runs,
                "num_reps": num_reps,
                "num_kept_nvs": num_kept_nvs,
                "rep_counts": rep_counts,
            }
        )

    dataset_results.sort(key=lambda d: d["dark_wait_s"])

    if not dataset_results:
        raise ValueError("No matching datasets were loaded.")

    # --------------------------------------------------------------
    # Make one figure per rep
    # --------------------------------------------------------------
    figures = {}

    num_panels = len(dataset_results)
    ncols = min(int(ncols), num_panels)
    nrows = int(np.ceil(num_panels / ncols))

    for rep_ind in rep_inds:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.2 * ncols, 3.8 * nrows),
            sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()

        for panel_ind, dataset in enumerate(dataset_results):
            ax = axes[panel_ind]

            yvals = np.asarray(dataset["rep_counts"][rep_ind], dtype=float)
            num_runs = int(dataset["num_runs"])
            run_inds = np.arange(num_runs)

            ax.bar(
                run_inds,
                yvals,
                alpha=0.75,
            )

            mean_val = float(np.mean(yvals))
            std_val = float(np.std(yvals, ddof=1)) if len(yvals) > 1 else 0.0

            ax.axhline(
                mean_val,
                linestyle="--",
                linewidth=1.4,
                color="k",
                alpha=0.7,
            )

            ax.set_title(
                f"wait = {dataset['dark_wait_s']:g} s\n"
                f"mean = {mean_val:.2f}, std = {std_val:.2f}",
                fontsize=10,
            )
            ax.set_xlabel("Run index")

            if show_fraction:
                ax.set_ylabel("NV$^-$ fraction")
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylabel("Number of NV$^-$")

            ax.grid(True, axis="y", alpha=0.25)

            tick_step = max(1, int(np.ceil(num_runs / 10)))
            ticks = np.arange(0, num_runs, tick_step)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])

        for ax in axes[num_panels:]:
            ax.axis("off")

        fig.suptitle(
            f"Run-by-run NV$^-$ population: {rep_labels.get(rep_ind, f'rep {rep_ind}')}",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        figures[rep_ind] = fig

    # --------------------------------------------------------------
    # Optional text summary
    # --------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 90)
        print("RUN-BY-RUN NV- POPULATION SUMMARY")
        print("=" * 90)

        for rep_ind in rep_inds:
            print(f"\n--- {rep_labels.get(rep_ind, f'rep {rep_ind}')} ---")
            for dataset in dataset_results:
                values = np.asarray(dataset["rep_counts"][rep_ind], dtype=float)
                print(
                    f"wait={dataset['dark_wait_s']:>6g} s | "
                    f"mean={np.mean(values):.3f}, "
                    f"std={np.std(values, ddof=1) if len(values) > 1 else 0.0:.3f}, "
                    f"min={np.min(values):.3f}, "
                    f"max={np.max(values):.3f}"
                )

    return dataset_results, figures

if __name__ == "__main__":
    kpl.init_kplotlib()
    # file_stem = "2026_07_26-18_11_11-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1800s-wait-1800s"
    # raw_data = dm.get_raw_data(
    #     file_stem=file_stem,
    #     load_npz=True,
    # )

    # drift_result, fig_drift = (
    #     analyze_drift_state_correlation(
    #         raw_data,
    #         register_images=True,
    #     )
    # )

    # kpl.show(block=True)
    # sys.exit()
    
    # FILE_STEMS = [
    # "2026_07_23-01_05_24-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s",
    # "2026_07_23-01_48_56-qnami-nv0_2026_02_20-particle-memory-source_off_wait_10s-wait-10s",
    # "2026_07_23-03_05_50-qnami-nv0_2026_02_20-particle-memory-source_off_wait_30s-wait-30s",
    # "2026_07_23-05_13_51-qnami-nv0_2026_02_20-particle-memory-source_off_wait_60s-wait-60s",
    # "2026_07_23-09_19_48-qnami-nv0_2026_02_20-particle-memory-source_off_wait_180s-wait-180s",
    # "2026_07_23-15_55_08-qnami-nv0_2026_02_20-particle-memory-source_off_wait_300s-wait-300s",
    # "2026_07_24-00_29_16-qnami-nv0_2026_02_20-particle-memory-source_off_wait_600s-wait-600s",
    # "2026_07_24-08_56_35-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1200s-wait-1200s",
    # ]
    
    # FILE_STEMS = [
    # "2026_07_24-21_43_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s",
    # "2026_07_24-22_27_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_10s-wait-10s",
    # "2026_07_24-23_44_20-qnami-nv0_2026_02_20-particle-memory-source_off_wait_30s-wait-30s",
    # "2026_07_25-01_51_32-qnami-nv0_2026_02_20-particle-memory-source_off_wait_60s-wait-60s",
    # "2026_07_25-05_57_38-qnami-nv0_2026_02_20-particle-memory-source_off_wait_180s-wait-180s",
    # "2026_07_25-12_33_01-qnami-nv0_2026_02_20-particle-memory-source_off_wait_300s-wait-300s",
    # "2026_07_25-21_07_06-qnami-nv0_2026_02_20-particle-memory-source_off_wait_600s-wait-600s",
    # "2026_07_26-05_34_29-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1200s-wait-1200s",
    # "2026_07_26-18_11_11-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1800s-wait-1800s",
    # "2026_07_27-19_18_59-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    # ]
    
    FILE_STEMS = [
    "2026_08_08-23_11_09-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s",
    "2026_08_08-23_19_25-qnami-nv0_2026_02_20-particle-memory-source_off_wait_10s-wait-10s",
    "2026_08_08-23_34_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_30s-wait-30s",
    "2026_08_08-23_59_13-qnami-nv0_2026_02_20-particle-memory-source_off_wait_60s-wait-60s",
    "2026_08_09-01_04_06-qnami-nv0_2026_02_20-particle-memory-source_off_wait_180s-wait-180s",
    "2026_08_09-02_49_00-qnami-nv0_2026_02_20-particle-memory-source_off_wait_300s-wait-300s",
    "2026_08_09-06_13_54-qnami-nv0_2026_02_20-particle-memory-source_off_wait_600s-wait-600s",
    "2026_08_09-12_58_47-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1200s-wait-1200s",
    "2026_08_09-23_03_43-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1800s-wait-1800s",
    ]
    ####
    FILE_STEMS = [
    "2026_08_13-11_33_24-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    ]
    selected_waits_s = [
        # 0,
        # 10,
        # 30,
        # 60,
        # 180,
        # 300,
        # 600,
        # 1200,
        # 1800,
        3600,
    ]

    rep_stats, rep_figs = plot_nv_minus_by_run_separate_reps(
        FILE_STEMS,
        selected_waits_s=selected_waits_s,
        rep_inds=(1, 11, 12),
        rep_labels={
            1: "rep 1: early initialization",
            11: "rep 11: immediate final check",
            12: "rep 12: after dark wait",
        },
        exclude_nv_inds=None,   # or BAD_NV_INDS
        show_fraction=False,    # True if you want fraction instead of count
        ncols=3,
        verbose=True,
    )

    kpl.show(block=True)
    
    # # ------------------------------------------------------------------
    # # Load datasets only once
    # # ------------------------------------------------------------------
    # output = run_particle_memory_dark_wait_comparison_analysis(
    #     file_stems=FILE_STEMS,
    #     recompute_analysis=False,
    #     save_fig=True,
    #     save_csv=False,
    # )

    # analyses = output["analyses"]

    # # # ------------------------------------------------------------------
    # # # Save lightweight wait-sweep analysis cache
    # # # ------------------------------------------------------------------

    # analysis_cache_timestamp = dm.get_time_stamp()

    # analysis_cache_file_path = dm.get_file_path(
    #     __file__,
    #     analysis_cache_timestamp,
    #     "particle-memory-dark-wait-analysis-cache",
    # )

    # analysis_cache = {
    #     "analysis_type": "particle_memory_dark_wait_analysis_cache",
    #     "timestamp": analysis_cache_timestamp,
    #     "file_stems": list(FILE_STEMS),
    #     "analyses": _json_safe(analyses),
    #     "summary": _json_safe(output.get("summary", {})),
    # }

    # dm.save_raw_data(
    #     analysis_cache,
    #     analysis_cache_file_path,
    # )

    # print(
    #     "Saved wait-sweep analysis cache:",
    #     analysis_cache_file_path,
    # )
    
    # ------------------------------------------------------------------
    # Load previously saved lightweight analysis cache
    # ------------------------------------------------------------------

    # analysis_cache_file_stem = (
    #     "2026_08_05-18_19_19-"
    #     "particle-memory-dark-wait-analysis-cache"
    # )
    
    analysis_cache_file_stem = (
        "2026_08_10-10_41_25-particle-memory-dark-wait-analysis-cache"
    )

    analysis_cache = dm.get_raw_data(
        file_stem=analysis_cache_file_stem,
        load_npz=True,
    )

    analyses = analysis_cache[
        "analyses"
    ]

    FILE_STEMS = analysis_cache[
        "file_stems"
    ]

    print(
        "Loaded cached analyses:",
        len(analyses),
    )

    print(
        "Wait times:",
        [
            analysis["dark_wait_s"]
            for analysis in analyses
        ],
    )



    # ------------------------------------------------------------------
    # Select wait-time datasets
    # ------------------------------------------------------------------

    selected_waits_s = [
        0,
        10,
        30,
        60,
        180,
        300,
        600,
        1200,
        1800,
    ]



    run_bar_result, fig_run_bars = plot_wait_sweep_run_bars(
        analyses,
        selected_waits_s=selected_waits_s,
        metric="num_candidates_by_run",
        outlier_method="mad",
        outlier_threshold=2.0,
        ncols=3,
        sharey=True,
        show_mean=True,
        show_median=True,
        sort_runs=False,
    )
    
    
    run_retention_result, fig_retention_bars = plot_wait_sweep_run_bars(
        analyses,
        selected_waits_s=selected_waits_s,
        metric="retention_by_run",
        outlier_method="mad",
        outlier_threshold=2.0,
        ncols=3,
        sharey=True,
        show_mean=True,
        show_median=True,
        sort_runs=False,
    )
    
    run_bar_result_sorted, fig_run_bars_sorted = plot_wait_sweep_run_bars(
        analyses,
        selected_waits_s=selected_waits_s,
        metric="num_candidates_by_run",
        outlier_method="mad",
        outlier_threshold=2.0,
        ncols=3,
        sharey=True,
        sort_runs=True,
    )
    kpl.show(block=True)
    # ------------------------------------------------------------------
    # Identify NVs that are persistently unstable across the wait sweep
    # ------------------------------------------------------------------

    BAD_NV_INDS, bad_actor_result = find_persistent_bad_nvs(
        analyses,
        selected_waits_s=selected_waits_s,

        # A wait-time probability of at least 10% is high.
        high_probability_threshold=0.05,

        # High in at least 70% of valid wait datasets.
        min_fraction_high=0.60,

        # Explicitly require at least 6 high datasets.
        min_high_waits=2,

        # Require usable data in at least 7 datasets.
        min_valid_waits=2,

        min_eligible_per_wait=10,

        # Require consistently high central probability.
        min_median_probability=0.10,

        # Require high probability when all eligible trials are pooled.
        min_pooled_probability=0.10,

        # Require high behavior at both ends of the sweep.
        require_short_and_long_waits=True,
        short_wait_max_s=60.0,
        long_wait_min_s=300.0,
        min_short_high_waits=2,
        min_long_high_waits=2,

        verbose=True,
    )

    print(
        "\nBad NVs used for all plots:",
        BAD_NV_INDS,
    )


    # ------------------------------------------------------------------
    # Absolute heat map after removing persistent bad actors
    # ------------------------------------------------------------------

    row_result, fig_rows = plot_nv_loss_row_by_row(
        analyses,
        selected_waits_s=selected_waits_s,
        subtract_zero_wait=False,
        show_percent=True,
        percentile_limit=99.0,

        # Apply the persistent bad-actor list.
        exclude_nv_inds=BAD_NV_INDS,

        # Disable the old zero-wait-only automatic filter.
        max_zero_wait_loss_probability=None,
    )


    # ------------------------------------------------------------------
    # Get analyses with the same NVs removed
    # ------------------------------------------------------------------

    filtered_analyses = row_result[
        "filtered_analyses"
    ]


    # ------------------------------------------------------------------
    # Recompute the wait-time summary and lifetime fit
    # ------------------------------------------------------------------

    filtered_summary = summarize_wait_sweep(
        filtered_analyses
    )

    print_wait_sweep_table(
        filtered_summary
    )

    fig_filtered_trend = plot_wait_sweep_summary(
        filtered_summary,
        zoom_retention_axes=True,
    )


    # ------------------------------------------------------------------
    # Baseline-subtracted heat map using the same NV population
    # ------------------------------------------------------------------

    excess_row_result, fig_excess_rows = (
        plot_nv_loss_row_by_row(
            analyses,
            selected_waits_s=selected_waits_s,

            # This must be True for the baseline-subtracted plot.
            subtract_zero_wait=False,

            show_percent=True,
            percentile_limit=99.0,
            exclude_nv_inds=BAD_NV_INDS,
            max_zero_wait_loss_probability=None,
        )
    )


    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------

    print(
        "\nOriginal NV count:",
        row_result["num_original_nvs"],
    )

    print(
        "Excluded persistent bad actors:",
        len(BAD_NV_INDS),
    )

    print(
        "Retained NV count:",
        row_result["num_retained_nvs"],
    )

    print(
        "Excluded indices:",
        BAD_NV_INDS,
    )



    # ------------------------------------------------------------------
    # Fit every retained NV
    # ------------------------------------------------------------------

    individual_fit_result = (
        fit_individual_nv_dark_survival(
            filtered_analyses,
            min_eligible_per_wait=10,
            min_valid_waits=6,
            min_fit_amplitude=0.005,
            min_r_squared=0.0,
            max_relative_tau_error=6.0,
            verbose=True,
        )
    )


    # ------------------------------------------------------------------
    # Remove poor fits and extreme lifetime outliers
    # ------------------------------------------------------------------

    clean_individual_fit_result = (
        remove_individual_nv_fit_outliers(
            individual_fit_result,

            # Robust outlier threshold in log10(tau).
            mad_z_threshold=3.5,

            # Fit-quality cuts.
            min_r_squared=0.0,
            max_relative_tau_error=1.5,

            # Do not trust tau values more than 10 times
            # the longest measured dark time.
            max_tau_factor=10.0,

            min_tau_s=1.0,
            verbose=True,
        )
    )


    # ------------------------------------------------------------------
    # Histogram after removing outliers
    # ------------------------------------------------------------------

    fig_nv_tau_histogram = (
        plot_individual_nv_lifetime_histogram(
            clean_individual_fit_result,
            quality_only=True,
            num_bins=20,
        )
    )


    # ------------------------------------------------------------------
    # Plot the five fastest NVs remaining after outlier removal
    # ------------------------------------------------------------------

    fig_fast_nvs, fast_nv_inds = (
        plot_fastest_individual_nv_curves(
            clean_individual_fit_result,
            top_n=5,
            quality_only=True,
        )
    )

    print(
        "Fastest retained NV indices:",
        fast_nv_inds,
    )

    print(
        "Removed fit outlier indices:",
        [
            fit["original_nv_ind"]
            for fit in clean_individual_fit_result[
                "outlier_removed_fits"
            ]
        ],
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