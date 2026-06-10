# -*- coding: utf-8 -*-
"""
DMD turn-off crosstalk and extinction-ratio experiment for widefield NVs.

Experiment name
---------------
    dmd_turnoff_crosstalk_extinction_matrix

Purpose
-------
This is the reverse of the usual "single NV ON" DMD crosstalk experiment.

Instead of:
    only NV i ON -> measure response at all NVs j

this experiment does:
    all selected NV apertures ON
    then turn OFF / remove one NV aperture i at a time
    measure response change at all NVs j

Definitions
-----------
all_on_counts[j]:
    Counts at measured NV j when all selected DMD apertures are ON.

off_counts_matrix[j, i]:
    Counts at measured NV j when source NV i is turned OFF.

turnoff_delta_matrix[j, i]:
    all_on_counts[j] - off_counts_matrix[j, i]

This is the signal removed from measured NV j when source NV i is turned OFF.

turnoff_crosstalk[j, i]:
    turnoff_delta_matrix[j, i] / turnoff_delta_matrix[i, i]

This tells how much turning OFF source NV i also changes another NV j,
relative to the intended removed signal at NV i.

Extinction
----------
For each source NV i:

    extinction_ratio[i] = all_on_counts[i] / off_counts_matrix[i, i]

    extinction_db[i] = 10 * log10(extinction_ratio[i])

    leakage_fraction[i] = off_counts_matrix[i, i] / all_on_counts[i]

Important DMD mask convention
-----------------------------
Recommended mode here is NOT white-background block_single.

Instead, this script uses:
    all ON:       pass_loaded_indices(all source indices)
    source i OFF: pass_loaded_indices(all source indices except i)

So the DMD passes only the selected NV apertures, not the whole chip.

Expected sequence file
----------------------
    servers/timing/sequencelibrary/QM_opx/camera/dmd_crosstalk_readout.py
"""

import json
import sys
import time
import traceback

import cv2
import labrad
import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


# =============================================================================
# Constants
# =============================================================================

DMD_CHAIN_PATH = "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1271.npz"

EXPERIMENT_NAME = "dmd_turnoff_crosstalk_extinction_matrix"


# =============================================================================
# Utility helpers
# =============================================================================

def _resolve_repo_path(path_like):
    """Resolve a repo-relative path."""
    from pathlib import Path

    p = Path(path_like)

    if p.is_absolute():
        return p

    return common.get_repo_path() / p


def unique_keep_order(vals):
    """Return unique values while preserving order."""
    out = []
    seen = set()

    for val in vals:
        val = int(val)

        if val not in seen:
            out.append(val)
            seen.add(val)

    return out


def subset_nv_list(nv_list_all, global_inds):
    """
    Convenience helper when nv_list_all order matches the global DMD/NV order.
    """
    return [nv_list_all[int(ind)] for ind in global_inds]


def load_dmd_chain_points(chain_path=DMD_CHAIN_PATH):
    """
    Load DMD points and inside-DMD indices from the saved DMD chain file.
    """
    path = _resolve_repo_path(chain_path)

    with np.load(path, allow_pickle=False) as npz:
        dmd_points = np.asarray(npz["dmd_points"], dtype=np.float32)

        if "inside_dmd_indices" in npz.files:
            inside_dmd_indices = np.asarray(
                npz["inside_dmd_indices"],
                dtype=np.int32,
            )
        else:
            inside = (
                (dmd_points[:, 0] >= 0)
                & (dmd_points[:, 0] < 1920)
                & (dmd_points[:, 1] >= 0)
                & (dmd_points[:, 1] < 1080)
            )
            inside_dmd_indices = np.where(inside)[0].astype(np.int32)

    return dmd_points, inside_dmd_indices


def choose_center_dmd_indices_from_chain(
    nv_list_all,
    chain_path=DMD_CHAIN_PATH,
    num_sources=200,
    min_source_pitch_px=None,
):
    """
    Choose source NV indices from the saved DMD chain.

    Parameters
    ----------
    min_source_pitch_px : None, float, or int
        If None, choose dense center NVs.
        If 25 or 30, choose separated NVs for debugging crosstalk.
    """
    dmd_points, inside_dmd_indices = load_dmd_chain_points(chain_path)

    valid_inds = [
        int(ind)
        for ind in inside_dmd_indices
        if int(ind) < len(nv_list_all)
    ]

    center = np.array([960, 540], dtype=np.float32)
    dist = np.linalg.norm(dmd_points[valid_inds] - center[None, :], axis=1)

    sorted_inds = [
        valid_inds[i]
        for i in np.argsort(dist)
    ]

    chosen = []

    for ind in sorted_inds:
        p = dmd_points[ind]

        if min_source_pitch_px is not None and len(chosen) > 0:
            chosen_pts = dmd_points[chosen]
            d = np.linalg.norm(chosen_pts - p[None, :], axis=1)

            if np.min(d) < min_source_pitch_px:
                continue

        chosen.append(int(ind))

        if len(chosen) >= num_sources:
            break

    return chosen, dmd_points


def get_center_dmd_indices(n=10, chain_path=None, include_only_inside=True):
    """
    Return n global NV indices closest to DMD chip center.
    """
    config = common.get_config_dict()

    if chain_path is None:
        spatial = config.get("SpatialCalibrations", {})
        chain_path = spatial.get(
            "dmd_chain_calib_path",
            DMD_CHAIN_PATH,
        )

    path = _resolve_repo_path(chain_path)
    data = np.load(path, allow_pickle=True)
    dmd_pts = np.asarray(data["dmd_points"], dtype=np.float32)

    if include_only_inside:
        inside = (
            (dmd_pts[:, 0] >= 0)
            & (dmd_pts[:, 0] < 1920)
            & (dmd_pts[:, 1] >= 0)
            & (dmd_pts[:, 1] < 1080)
        )
        candidate_inds = np.where(inside)[0]
    else:
        candidate_inds = np.arange(len(dmd_pts))

    center = np.array([960, 540], dtype=np.float32)
    dist = np.linalg.norm(dmd_pts[candidate_inds] - center, axis=1)
    selected = candidate_inds[np.argsort(dist)[:n]]

    return selected.astype(int).tolist()


def print_dmd_pitch_stats(source_inds, dmd_points):
    """
    Print nearest-neighbor pitch statistics for chosen DMD source points.
    """
    pts = np.asarray(dmd_points[source_inds], dtype=np.float32)

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)

    nn = np.min(d, axis=1)

    print("\n=== DMD source pitch stats ===")
    print("num sources:", len(source_inds))
    print("nearest pitch min:", float(np.min(nn)))
    print("nearest pitch 5%:", float(np.percentile(nn, 5)))
    print("nearest pitch median:", float(np.median(nn)))
    print("nearest pitch max:", float(np.max(nn)))

    return nn


def ensure_representative_nv(nv_list, expected_counts=1500.0):
    """
    Ensure nv_list has a representative NV.

    base_routine.main() needs widefield.get_repr_nv_sig(nv_list) to return
    a valid NVSig. When you create a subset, the original representative NV
    may not be included.
    """
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)

    if repr_nv_sig is None:
        for nv in nv_list:
            nv.representative = False

        nv_list[0].representative = True
        repr_nv_sig = nv_list[0]

        if getattr(repr_nv_sig, "expected_counts", None) is None:
            repr_nv_sig.expected_counts = expected_counts

        print(
            "No representative NV found in subset. "
            f"Set {repr_nv_sig.name} as representative."
        )

    return repr_nv_sig


def _mean_counts_per_nv(raw_data, exp_ind=0):
    """
    Convert raw_data["counts"] into mean count and STE per NV.

    Typical shape from base_routine:
        counts[exp_ind, nv_ind, run_ind, step_ind, rep_ind]

    This function averages all axes after the NV axis.
    """
    counts = np.asarray(raw_data["counts"])
    arr = np.asarray(counts[exp_ind], dtype=np.float32)

    if arr.ndim < 2:
        raise ValueError(f"Unexpected counts[exp_ind] shape: {arr.shape}")

    num_nvs = arr.shape[0]
    arr_flat = arr.reshape(num_nvs, -1)

    mean_counts = np.mean(arr_flat, axis=1)

    if arr_flat.shape[1] > 1:
        ste_counts = np.std(arr_flat, axis=1, ddof=1) / np.sqrt(arr_flat.shape[1])
    else:
        ste_counts = np.zeros(num_nvs, dtype=np.float32)

    return mean_counts.astype(np.float32), ste_counts.astype(np.float32)


def _mean_image(raw_data, exp_ind=0):
    """
    Return mean image for an acquisition if raw_data contains img_arrays.

    Typical shape:
        img_arrays[exp_ind, run_ind, step_ind, rep_ind, y, x]
    """
    if "img_arrays" not in raw_data:
        return None

    img_arrays = np.asarray(raw_data["img_arrays"])

    if img_arrays.size == 0:
        return None

    arr = np.asarray(img_arrays[exp_ind], dtype=np.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim < 2:
        return None

    axes = tuple(range(arr.ndim - 2))

    return np.mean(arr, axis=axes).astype(np.float32)


def _plot_matrix(
    matrix,
    title,
    xlabel,
    ylabel,
    cbar_label,
    xlabels=None,
    ylabels=None,
    vmin=None,
    vmax=None,
):
    """
    Plot matrix without tight_layout/subplots_adjust.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlabels is not None and len(xlabels) <= 25:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels([str(x) for x in xlabels], rotation=90)

    if ylabels is not None and len(ylabels) <= 25:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels([str(y) for y in ylabels])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    return fig


# =============================================================================
# Turn-off crosstalk and extinction analysis
# =============================================================================

def _compute_turnoff_crosstalk_extinction_products(
    off_counts_matrix,
    all_on_counts,
    measured_global_inds,
    source_global_inds,
    background_counts=None,
    eps=1e-9,
):
    """
    Compute turn-off crosstalk and extinction products.

    Parameters
    ----------
    off_counts_matrix : array, shape (num_measured, num_sources)
        off_counts_matrix[row, col] is the measured count at row NV when
        source_global_inds[col] is turned OFF.

    all_on_counts : array, shape (num_measured,)
        Counts when all selected source apertures are ON.

    background_counts : None or array, shape (num_measured,)
        Optional block_all background.

    Returns
    -------
    dict containing:
        all_on_counts_bg_sub
        off_counts_bg_sub
        turnoff_delta_matrix
        turnoff_crosstalk
        turnoff_abs_crosstalk
        turnoff_intended_drop
        turnoff_intended_all_on
        turnoff_intended_off
        turnoff_leakage_fraction
        turnoff_extinction_ratio
        turnoff_extinction_db
        turnoff_worst_abs_frac
        turnoff_worst_signed_frac
        turnoff_worst_target
        turnoff_worst_delta
    """
    off_counts_matrix = np.asarray(off_counts_matrix, dtype=np.float32)
    all_on_counts = np.asarray(all_on_counts, dtype=np.float32)

    measured_global_inds = [int(x) for x in measured_global_inds]
    source_global_inds = [int(x) for x in source_global_inds]

    if background_counts is not None:
        background_counts = np.asarray(background_counts, dtype=np.float32)
        all_on_sub = all_on_counts - background_counts
        off_sub = off_counts_matrix - background_counts[:, None]
    else:
        all_on_sub = all_on_counts.copy()
        off_sub = off_counts_matrix.copy()

    # Signal removed when source i is turned off.
    delta_matrix = all_on_sub[:, None] - off_sub

    row_lookup = {
        int(gind): row
        for row, gind in enumerate(measured_global_inds)
    }

    num_sources = len(source_global_inds)

    intended_drop = np.full(num_sources, np.nan, dtype=np.float32)
    intended_all_on = np.full(num_sources, np.nan, dtype=np.float32)
    intended_off = np.full(num_sources, np.nan, dtype=np.float32)

    for col, source_gind in enumerate(source_global_inds):
        row = row_lookup.get(int(source_gind), None)

        if row is None:
            continue

        intended_drop[col] = delta_matrix[row, col]
        intended_all_on[col] = all_on_sub[row]
        intended_off[col] = off_sub[row, col]

    # Signed crosstalk:
    # positive means measured NV decreases when source is turned off.
    # negative means measured NV increases when source is turned off.
    turnoff_crosstalk = np.full_like(delta_matrix, np.nan, dtype=np.float32)
    turnoff_abs_crosstalk = np.full_like(delta_matrix, np.nan, dtype=np.float32)

    for col in range(num_sources):
        denom = intended_drop[col]

        if np.isfinite(denom) and abs(denom) > eps:
            turnoff_crosstalk[:, col] = delta_matrix[:, col] / denom
            turnoff_abs_crosstalk[:, col] = np.abs(delta_matrix[:, col]) / abs(denom)

    # Extinction at the intended row.
    intended_all_on_safe = np.maximum(intended_all_on, eps)
    intended_off_safe = np.maximum(intended_off, eps)

    leakage_fraction = intended_off_safe / intended_all_on_safe
    extinction_ratio = intended_all_on_safe / intended_off_safe
    extinction_db = 10.0 * np.log10(extinction_ratio)

    removed_fraction = intended_drop / intended_all_on_safe

    worst_abs_frac = np.full(num_sources, np.nan, dtype=np.float32)
    worst_signed_frac = np.full(num_sources, np.nan, dtype=np.float32)
    worst_target = [None] * num_sources
    worst_delta = np.full(num_sources, np.nan, dtype=np.float32)

    for col, source_gind in enumerate(source_global_inds):
        source_gind = int(source_gind)

        vals_abs = turnoff_abs_crosstalk[:, col].copy()
        vals_signed = turnoff_crosstalk[:, col].copy()

        for row, measured_gind in enumerate(measured_global_inds):
            if int(measured_gind) == source_gind:
                vals_abs[row] = np.nan
                vals_signed[row] = np.nan

        if np.all(np.isnan(vals_abs)):
            continue

        worst_row = int(np.nanargmax(vals_abs))

        worst_abs_frac[col] = vals_abs[worst_row]
        worst_signed_frac[col] = vals_signed[worst_row]
        worst_target[col] = measured_global_inds[worst_row]
        worst_delta[col] = delta_matrix[worst_row, col]

    return {
        "all_on_counts_bg_sub": all_on_sub.astype(np.float32),
        "off_counts_bg_sub": off_sub.astype(np.float32),
        "turnoff_delta_matrix": delta_matrix.astype(np.float32),
        "turnoff_crosstalk": turnoff_crosstalk.astype(np.float32),
        "turnoff_abs_crosstalk": turnoff_abs_crosstalk.astype(np.float32),
        "turnoff_intended_drop": intended_drop.astype(np.float32),
        "turnoff_intended_all_on": intended_all_on.astype(np.float32),
        "turnoff_intended_off": intended_off.astype(np.float32),
        "turnoff_removed_fraction": removed_fraction.astype(np.float32),
        "turnoff_leakage_fraction": leakage_fraction.astype(np.float32),
        "turnoff_extinction_ratio": extinction_ratio.astype(np.float32),
        "turnoff_extinction_db": extinction_db.astype(np.float32),
        "turnoff_worst_abs_frac": worst_abs_frac.astype(np.float32),
        "turnoff_worst_signed_frac": worst_signed_frac.astype(np.float32),
        "turnoff_worst_target": worst_target,
        "turnoff_worst_delta": worst_delta.astype(np.float32),
    }


def process_and_plot_turnoff(raw_data, vmax_abs_crosstalk=0.3):
    """
    Plot turn-off crosstalk and extinction diagnostics.
    """
    off_counts_matrix = np.asarray(raw_data["off_counts_matrix"], dtype=np.float32)
    all_on_counts = np.asarray(raw_data["all_on_counts"], dtype=np.float32)

    measured_global_inds = raw_data.get(
        "measured_global_inds",
        list(range(off_counts_matrix.shape[0])),
    )
    source_global_inds = raw_data.get(
        "source_global_inds",
        list(range(off_counts_matrix.shape[1])),
    )

    if "turnoff_delta_matrix" not in raw_data:
        products = _compute_turnoff_crosstalk_extinction_products(
            off_counts_matrix,
            all_on_counts,
            measured_global_inds,
            source_global_inds,
            background_counts=raw_data.get("background_counts", None),
        )
        raw_data |= products

    delta_matrix = np.asarray(raw_data["turnoff_delta_matrix"], dtype=np.float32)
    turnoff_crosstalk = np.asarray(raw_data["turnoff_crosstalk"], dtype=np.float32)
    turnoff_abs_crosstalk = np.asarray(
        raw_data["turnoff_abs_crosstalk"],
        dtype=np.float32,
    )

    extinction_db = np.asarray(raw_data["turnoff_extinction_db"], dtype=float)
    extinction_ratio = np.asarray(raw_data["turnoff_extinction_ratio"], dtype=float)
    leakage_fraction = np.asarray(raw_data["turnoff_leakage_fraction"], dtype=float)
    intended_drop = np.asarray(raw_data["turnoff_intended_drop"], dtype=float)
    worst_abs_frac = np.asarray(raw_data["turnoff_worst_abs_frac"], dtype=float)

    figs = []

    figs.append(
        _plot_matrix(
            off_counts_matrix,
            title="DMD turn-off: one-off counts matrix",
            xlabel="blocked / removed source global NV index",
            ylabel="measured global NV index",
            cbar_label="mean counts",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
        )
    )

    figs.append(
        _plot_matrix(
            delta_matrix,
            title="DMD turn-off: removed-signal matrix",
            xlabel="blocked / removed source global NV index",
            ylabel="measured global NV index",
            cbar_label="all_on - one_off counts",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
        )
    )

    figs.append(
        _plot_matrix(
            turnoff_crosstalk,
            title="DMD turn-off: signed normalized crosstalk",
            xlabel="blocked / removed source global NV index",
            ylabel="measured global NV index",
            cbar_label="Delta[j,i] / Delta[i,i]",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
            vmin=-vmax_abs_crosstalk,
            vmax=vmax_abs_crosstalk,
        )
    )

    figs.append(
        _plot_matrix(
            turnoff_abs_crosstalk,
            title="DMD turn-off: absolute normalized crosstalk",
            xlabel="blocked / removed source global NV index",
            ylabel="measured global NV index",
            cbar_label="abs(Delta[j,i]) / abs(Delta[i,i])",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
            vmin=0,
            vmax=vmax_abs_crosstalk,
        )
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(all_on_counts, "o-")
    ax.set_xlabel("measured NV row")
    ax.set_ylabel("all-on counts")
    ax.set_title("All-selected-apertures ON reference")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(intended_drop, "o-")
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("source column")
    ax.set_ylabel("intended removed signal")
    ax.set_title("Intended drop when each source NV is turned OFF")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(extinction_ratio, "o-")
    ax.set_xlabel("source column")
    ax.set_ylabel("extinction ratio = ON / OFF")
    ax.set_title("DMD extinction ratio per source NV")
    ax.set_yscale("log")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(extinction_db, "o-")
    ax.set_xlabel("source column")
    ax.set_ylabel("extinction [dB]")
    ax.set_title("DMD extinction in dB per source NV")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(extinction_db, worst_abs_frac, alpha=0.8)
    ax.set_xlabel("extinction [dB]")
    ax.set_ylabel("worst off-target / intended drop")
    ax.set_title("Turn-off crosstalk vs extinction")
    ax.set_yscale("log")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(intended_drop, worst_abs_frac, alpha=0.8)
    ax.set_xlabel("intended removed signal")
    ax.set_ylabel("worst off-target / intended drop")
    ax.set_title("Turn-off crosstalk vs intended signal")
    ax.set_yscale("log")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    finite = worst_abs_frac[np.isfinite(worst_abs_frac)]
    ax.hist(finite, bins=30)
    ax.set_xlabel("worst off-target / intended drop")
    ax.set_ylabel("number of source NVs")
    ax.set_title("Distribution of turn-off crosstalk")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    finite = extinction_db[np.isfinite(extinction_db)]
    ax.hist(finite, bins=30)
    ax.set_xlabel("extinction [dB]")
    ax.set_ylabel("number of source NVs")
    ax.set_title("Distribution of DMD extinction")
    figs.append(fig)

    print("\n=== DMD turn-off crosstalk and extinction summary ===")
    print("experiment:", raw_data.get("experiment", None))
    print("num measured NVs:", raw_data.get("num_nvs", None))
    print("num source NVs:", raw_data.get("num_sources", None))

    print("\nExtinction:")
    print("median extinction ratio:", float(np.nanmedian(extinction_ratio)))
    print("median extinction dB:", float(np.nanmedian(extinction_db)))
    print("10% extinction dB:", float(np.nanpercentile(extinction_db, 10)))
    print("median leakage fraction OFF/ON:", float(np.nanmedian(leakage_fraction)))
    print("90% leakage fraction OFF/ON:", float(np.nanpercentile(leakage_fraction, 90)))

    print("\nTurn-off crosstalk:")
    print("median worst abs crosstalk:", float(np.nanmedian(worst_abs_frac)))
    print("90% worst abs crosstalk:", float(np.nanpercentile(worst_abs_frac, 90)))
    print("max worst abs crosstalk:", float(np.nanmax(worst_abs_frac)))

    print("\nWeak / bad examples:")
    source_global_inds_arr = np.asarray(source_global_inds, dtype=int)

    bad_order = np.argsort(worst_abs_frac)[::-1]

    for k in bad_order[:10]:
        if not np.isfinite(worst_abs_frac[k]):
            continue

        print(
            f"source {int(source_global_inds_arr[k])}: "
            f"ext_db={extinction_db[k]:.2f}, "
            f"ext_ratio={extinction_ratio[k]:.3g}, "
            f"intended_drop={intended_drop[k]:.3f}, "
            f"worst_xtalk={worst_abs_frac[k]:.3f}, "
            f"worst_target={raw_data['turnoff_worst_target'][k]}"
        )

    return figs


# =============================================================================
# Optional movie helper
# =============================================================================

def save_dmd_turnoff_image_movie(
    raw_data,
    image_key="mean_images_by_source",
    fps=5,
    subtract_background=False,
    normalize="global",
    output_label=None,
    cmap=None,
    output_scale=3,
    target_width=None,
    resize_interp=cv2.INTER_CUBIC,
):
    """
    Make a high-resolution mp4 movie from saved one-off images.

    Expected image stack:
        raw_data["mean_images_by_source"] shape = (num_sources, height, width)
    """
    if image_key not in raw_data:
        raise KeyError(
            f"raw_data does not contain '{image_key}'. "
            "Rerun with save_images=True so mean_images_by_source is saved."
        )

    frames = np.asarray(raw_data[image_key], dtype=np.float32)

    if frames.ndim != 3:
        raise ValueError(
            f"Expected {image_key} shape (num_frames, y, x), got {frames.shape}"
        )

    num_frames, height, width = frames.shape

    if subtract_background and "background_img" in raw_data:
        bg = np.asarray(raw_data["background_img"], dtype=np.float32)
        frames = frames - bg[None, :, :]

    frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

    if cmap is None:
        cmap = plt.get_cmap(plt.rcParams.get("image.cmap", "viridis"))
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if normalize == "global":
        vmin, vmax = np.percentile(frames, [0, 99.99])
    else:
        vmin, vmax = None, None

    if output_label is None:
        radius = raw_data.get("dmd_radius_px", "unknown")
        num_sources_saved = raw_data.get("num_sources", num_frames)
        output_label = (
            f"dmd-turnoff-crosstalk-extinction-movie-"
            f"{num_sources_saved}src-r{radius}-hires"
        )

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, output_label)
    mp4_path = str(file_path) + ".mp4"

    if target_width is not None:
        out_width = int(target_width)
        out_height = int(round(height * out_width / width))
    else:
        out_width = int(round(width * output_scale))
        out_height = int(round(height * output_scale))

    out_width = 2 * (out_width // 2)
    out_height = 2 * (out_height // 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        mp4_path,
        fourcc,
        float(fps),
        (out_width, out_height),
        True,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")

    source_inds = raw_data.get("source_global_inds", list(range(num_frames)))

    scale_factor = out_width / width
    font_scale = max(0.7, 0.8 * scale_factor)
    thickness = max(2, int(round(2 * scale_factor)))
    text_x = max(20, int(round(20 * scale_factor)))
    text_y = max(35, int(round(35 * scale_factor)))

    for ind in range(num_frames):
        img = frames[ind]

        if normalize == "per_frame":
            lo, hi = np.percentile(img, [1, 99.8])
        else:
            lo, hi = vmin, vmax

        if hi <= lo:
            hi = lo + 1.0

        img_norm = np.clip((img - lo) / (hi - lo), 0, 1)

        rgba = cmap(img_norm, bytes=True)
        rgb = rgba[:, :, :3]
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        frame_bgr = cv2.resize(
            frame_bgr,
            (out_width, out_height),
            interpolation=resize_interp,
        )

        frame_bgr = np.ascontiguousarray(frame_bgr)

        source_ind = int(source_inds[ind]) if ind < len(source_inds) else ind
        text = f"{ind + 1}/{num_frames}   turned OFF NV {source_ind}"

        cv2.putText(
            frame_bgr,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame_bgr,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        writer.write(frame_bgr)

    writer.release()

    movie_meta = {
        "timestamp": timestamp,
        "experiment": "dmd_turnoff_crosstalk_extinction_movie",
        "movie_path": mp4_path,
        "source_file_timestamp": raw_data.get("timestamp", None),
        "image_key": image_key,
        "num_frames": int(num_frames),
        "input_height": int(height),
        "input_width": int(width),
        "output_height": int(out_height),
        "output_width": int(out_width),
        "fps": float(fps),
        "subtract_background": bool(subtract_background),
        "normalize": normalize,
        "output_scale": float(output_scale),
        "target_width": None if target_width is None else int(target_width),
        "dmd_radius_px": raw_data.get("dmd_radius_px", None),
        "num_sources": raw_data.get("num_sources", num_frames),
        "source_global_inds": raw_data.get("source_global_inds", None),
        "cmap": str(cmap.name) if hasattr(cmap, "name") else str(cmap),
    }

    dm.save_raw_data(movie_meta, file_path, keys_to_compress=[])

    print("\nSaved high-resolution DMD turn-off movie:")
    print(mp4_path)
    print("movie size:", out_width, "x", out_height)

    return mp4_path


# =============================================================================
# Main experiment
# =============================================================================

def main(
    nv_list,
    num_reps,
    num_runs,
    source_global_inds=None,
    measured_global_inds=None,
    dmd_radius_px=20,
    do_polarize=True,
    targeted_polarization=False,
    take_background=True,
    save_images=True,
    save_raw_counts_by_source=True,
    dmd_settle_s=0.10,
    dmd_plane=230,
):
    """
    Run DMD turn-off crosstalk and extinction experiment.

    This experiment does:

        1. Optional block_all background.
        2. All selected DMD apertures ON.
        3. For each source NV i:
               pass all selected apertures except i
               acquire image / counts
        4. Compute:
               turnoff_delta_matrix = all_on - one_off
               turnoff_crosstalk = Delta[j,i] / Delta[i,i]
               extinction_ratio = all_on[i] / off[i,i]

    Parameters
    ----------
    nv_list : list[NVSig]
        NVs measured by base_routine. Rows of the matrix follow this order.

    source_global_inds : list[int]
        Global DMD/NV indices to turn OFF one at a time.
        Columns follow this order.

    measured_global_inds : list[int]
        Global indices corresponding to nv_list rows.

    dmd_radius_px : int
        Radius of each passed DMD disk.

    dmd_plane : int
        DMD plane argument passed to DMD server.
    """
    seq_file = "dmd_crosstalk_readout.py"

    num_steps = 1
    num_exps = 1

    if source_global_inds is None:
        source_global_inds = list(range(len(nv_list)))

    source_global_inds = [int(ind) for ind in source_global_inds]

    if measured_global_inds is None:
        measured_global_inds = list(range(len(nv_list)))

    measured_global_inds = [int(ind) for ind in measured_global_inds]

    if len(measured_global_inds) != len(nv_list):
        raise ValueError("measured_global_inds must have same length as nv_list.")

    num_measured = len(nv_list)
    num_sources = len(source_global_inds)

    repr_nv_sig = ensure_representative_nv(nv_list)

    pulse_gen = tb.get_server_pulse_gen()

    cxn = labrad.connect(username="", password="")
    dmd = cxn.dmd_dlp6500

    print("DMD state before initialize:")
    print(dmd.get_state())

    print("DMD initialize_pass_state:")
    print(dmd.initialize_pass_state())

    def run_readout_once():
        """
        Run one widefield acquisition with the current static DMD mask.
        """
        def run_fn(shuffled_step_inds):
            pol_coords_list, pol_duration_list, pol_amp_list = (
                widefield.get_pulse_parameter_lists(
                    nv_list,
                    VirtualLaserKey.CHARGE_POL,
                )
            )

            seq_args = [
                pol_coords_list,
                pol_duration_list,
                pol_amp_list,
                bool(do_polarize),
                bool(targeted_polarization),
            ]

            seq_args_string = tb.encode_seq_args(seq_args)
            pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

        data = base_routine.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            run_fn=run_fn,
            save_images=save_images,
            save_images_avg_reps=False,
            charge_prep_fn=None,
            num_exps=num_exps,
        )

        return data

    # -------------------------------------------------------------------------
    # 1. Optional block_all background.
    # -------------------------------------------------------------------------
    background_counts = None
    background_counts_ste = None
    background_img = None
    background_raw_counts = None

    if take_background:
        print("\nTaking DMD block_all background...")

        dmd.block_all()
        time.sleep(dmd_settle_s)

        bg_data = run_readout_once()

        background_counts, background_counts_ste = _mean_counts_per_nv(bg_data)
        background_img = _mean_image(bg_data)

        if save_raw_counts_by_source:
            background_raw_counts = np.asarray(bg_data["counts"])

        print("background_counts:")
        print(background_counts)

    # -------------------------------------------------------------------------
    # 2. All selected source apertures ON reference.
    # -------------------------------------------------------------------------
    print("\nTaking DMD all-selected-apertures ON reference...")

    dmd.pass_loaded_indices(
        json.dumps([int(ind) for ind in source_global_inds]),
        int(dmd_radius_px),
        int(dmd_plane),
    )

    time.sleep(dmd_settle_s)

    all_on_data = run_readout_once()

    all_on_counts, all_on_counts_ste = _mean_counts_per_nv(all_on_data)
    all_on_img = _mean_image(all_on_data)

    if save_raw_counts_by_source:
        all_on_raw_counts = np.asarray(all_on_data["counts"])
    else:
        all_on_raw_counts = None

    print("all_on_counts:")
    print(all_on_counts)

    # -------------------------------------------------------------------------
    # 3. Turn OFF one selected NV aperture at a time.
    # -------------------------------------------------------------------------
    off_counts_matrix = np.zeros((num_measured, num_sources), dtype=np.float32)
    off_counts_matrix_ste = np.zeros((num_measured, num_sources), dtype=np.float32)

    raw_counts_by_source = []
    mean_images_by_source = []

    for col, source_ind in enumerate(source_global_inds):
        print(
            f"\n=== DMD turn-off source {col + 1}/{num_sources}: "
            f"global index {source_ind} ==="
        )

        passed_inds = [
            int(ind)
            for ind in source_global_inds
            if int(ind) != int(source_ind)
        ]

        dmd.pass_loaded_indices(
            json.dumps(passed_inds),
            int(dmd_radius_px),
            int(dmd_plane),
        )

        time.sleep(dmd_settle_s)

        step_data = run_readout_once()

        mean_counts, ste_counts = _mean_counts_per_nv(step_data)

        off_counts_matrix[:, col] = mean_counts
        off_counts_matrix_ste[:, col] = ste_counts

        if save_raw_counts_by_source:
            raw_counts_by_source.append(np.asarray(step_data["counts"]))

        mean_img = _mean_image(step_data)

        if mean_img is not None:
            mean_images_by_source.append(mean_img)

        print("one-off mean_counts:")
        print(mean_counts)

    # -------------------------------------------------------------------------
    # 4. Restore DMD state.
    # -------------------------------------------------------------------------
    try:
        dmd.zero_block_on()
    except Exception:
        print("Could not restore zero_block_on:")
        print(traceback.format_exc())

    # -------------------------------------------------------------------------
    # 5. Compute turn-off crosstalk and extinction products.
    # -------------------------------------------------------------------------
    products = _compute_turnoff_crosstalk_extinction_products(
        off_counts_matrix,
        all_on_counts,
        measured_global_inds,
        source_global_inds,
        background_counts=background_counts,
    )

    timestamp = dm.get_time_stamp()

    repr_nv_sig = ensure_representative_nv(nv_list)
    repr_nv_name = repr_nv_sig.name

    raw_data = {
        "timestamp": timestamp,
        "experiment": EXPERIMENT_NAME,
        "nv_list": nv_list,
        "num_nvs": num_measured,
        "num_sources": num_sources,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "num_steps": num_steps,
        "num_exps": num_exps,
        "seq_file": seq_file,
        "source_global_inds": source_global_inds,
        "measured_global_inds": measured_global_inds,
        "dmd_radius_px": int(dmd_radius_px),
        "dmd_plane": int(dmd_plane),
        "do_polarize": bool(do_polarize),
        "targeted_polarization": bool(targeted_polarization),
        "take_background": bool(take_background),
        "save_images": bool(save_images),
        "save_raw_counts_by_source": bool(save_raw_counts_by_source),
        "dmd_settle_s": float(dmd_settle_s),
        "mask_convention": "all_selected_apertures_on_then_pass_all_except_one",
        "background_counts": background_counts,
        "background_counts_ste": background_counts_ste,
        "background_img": background_img,
        "background_raw_counts": background_raw_counts,
        "all_on_counts": all_on_counts,
        "all_on_counts_ste": all_on_counts_ste,
        "all_on_img": all_on_img,
        "all_on_raw_counts": all_on_raw_counts,
        "off_counts_matrix": off_counts_matrix,
        "off_counts_matrix_ste": off_counts_matrix_ste,
        # Compatibility alias:
        "counts_matrix": off_counts_matrix,
        "counts_matrix_ste": off_counts_matrix_ste,
        "raw_counts_by_source": raw_counts_by_source,
        "img_array-units": "photons",
    }

    raw_data |= products

    if len(mean_images_by_source) > 0:
        try:
            raw_data["mean_images_by_source"] = np.stack(
                mean_images_by_source,
                axis=0,
            )
        except Exception:
            print("Skipping mean_images_by_source stack because shapes are inconsistent.")
            raw_data["mean_images_by_source"] = np.asarray(
                mean_images_by_source,
                dtype=object,
            )

    # -------------------------------------------------------------------------
    # 6. Plot.
    # -------------------------------------------------------------------------
    try:
        figs = process_and_plot_turnoff(raw_data)
    except Exception:
        print("process_and_plot_turnoff failed, but raw data will still be saved.")
        print(traceback.format_exc())
        figs = []

    # -------------------------------------------------------------------------
    # 7. Save.
    # -------------------------------------------------------------------------
    file_path = dm.get_file_path(
        __file__,
        timestamp,
        (
            f"{repr_nv_name}-dmd-turnoff-crosstalk-extinction-"
            f"{num_sources}src-r{dmd_radius_px}"
        ),
    )

    keys_to_compress = [
        "background_counts",
        "background_counts_ste",
        "background_img",
        "background_raw_counts",
        "all_on_counts",
        "all_on_counts_ste",
        "all_on_img",
        "all_on_raw_counts",
        "off_counts_matrix",
        "off_counts_matrix_ste",
        "counts_matrix",
        "counts_matrix_ste",
        "all_on_counts_bg_sub",
        "off_counts_bg_sub",
        "turnoff_delta_matrix",
        "turnoff_crosstalk",
        "turnoff_abs_crosstalk",
        "turnoff_intended_drop",
        "turnoff_intended_all_on",
        "turnoff_intended_off",
        "turnoff_removed_fraction",
        "turnoff_leakage_fraction",
        "turnoff_extinction_ratio",
        "turnoff_extinction_db",
        "turnoff_worst_abs_frac",
        "turnoff_worst_signed_frac",
        "turnoff_worst_delta",
        "mean_images_by_source",
    ]

    if save_raw_counts_by_source:
        keys_to_compress += [
            "raw_counts_by_source",
        ]

    keys_to_compress = [
        key
        for key in keys_to_compress
        if key in raw_data and raw_data[key] is not None
    ]

    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    for ind, fig in enumerate(figs):
        fig_path = dm.get_file_path(
            __file__,
            timestamp,
            (
                f"{repr_nv_name}-dmd-turnoff-crosstalk-extinction-"
                f"{num_sources}src-r{dmd_radius_px}-{ind}"
            ),
        )
        dm.save_figure(fig, fig_path)

    tb.reset_cfm()

    print("\nSaved DMD turn-off crosstalk/extinction data:")
    print(file_path)

    return raw_data


# =============================================================================
# Saved-data analysis helper
# =============================================================================

def analyze_saved_turnoff_file(
    file_stem,
    load_npz=True,
    allow_pickle=True,
    make_movie=False,
    movie_fps=5,
):
    """
    Load saved turn-off data, replot, and optionally make movie.
    """
    kpl.init_kplotlib()

    raw_data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=load_npz,
        allow_pickle=allow_pickle,
    )

    figs = process_and_plot_turnoff(raw_data)

    movie_path = None

    if make_movie:
        movie_path = save_dmd_turnoff_image_movie(
            raw_data,
            image_key="mean_images_by_source",
            fps=movie_fps,
            subtract_background=False,
            normalize="global",
            cmap=None,
            output_scale=4,
        )

        print("movie_path:")
        print(movie_path)

    kpl.show(block=True)

    return raw_data, figs, movie_path


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()
    # -------------------------------------------------------------------------
    # Option A: analyze existing saved data.
    # -------------------------------------------------------------------------
    raw_data, figs, movie_path = analyze_saved_turnoff_file(
        file_stem="2026_06_10-00_20_38-qnami-nv0_2026_02_20-dmd-turnoff-crosstalk-extinction-200src-r6",
        make_movie=True,
        movie_fps=5,
    )
    
    
    # sys.exit()

    # -------------------------------------------------------------------------
    # Option B: run a new experiment.
    #
    # This part assumes nv_list_all already exists in your interactive namespace.
    # If this file is run directly, you should load nv_list_all the same way
    # you normally load your full NV list.
    # -------------------------------------------------------------------------

    # print(
    #     "\nThis script is ready for the DMD turn-off crosstalk/extinction experiment.\n"
    #     "To run a new experiment, import this file in your normal experiment session "
    #     "where nv_list_all is already loaded, then call main(...).\n"
    # )