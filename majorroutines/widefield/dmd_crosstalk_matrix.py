# -*- coding: utf-8 -*-
"""
DMD optical crosstalk matrix experiment for widefield NVs.

Purpose
-------
For each source NV index i, set a DMD mask, acquire a simple widefield
charge-readout image/counts measurement, and record the response at every
measured NV j.

The output matrix is:

    C[j, i] = response at measured NV j when DMD source NV i is selected

This first version is intentionally slow and safe:
    - DMD masks are changed in Python between acquisitions.
    - The OPX/QM sequence only performs optional charge polarization and
      charge readout.
    - Good for validating DMD mapping, optical leakage, and aperture radius.

Sequence file expected at:
    servers/timing/sequencelibrary/QM_opx/camera/dmd_crosstalk_readout.py

DMD index convention
--------------------
The DMD server's loaded NV points are indexed by the global NV order in:
    slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz

Therefore:
    source_global_inds must be global DMD/NV indices.
    measured_global_inds should describe how nv_list rows map to global indices.

Clean first test:
    inds = get_center_dmd_indices(10)
    measured_inds = [0] + inds       # include global NV0 for representative/positioning
    nv_sub = subset_nv_list(nv_list_all, measured_inds)

    main(
        nv_sub,
        num_reps=5,
        num_runs=3,
        source_global_inds=inds,
        measured_global_inds=measured_inds,
        dmd_radius_px=20,
        dmd_mode="pass_single",
    )
"""

import json
import sys
import time
import traceback

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


def get_center_dmd_indices(n=10, chain_path=None, include_only_inside=True):
    """
    Return n global NV indices closest to DMD chip center.

    This is useful for first tests because center NVs are usually less affected
    by extrapolation and edge clipping.
    """
    config = common.get_config_dict()

    if chain_path is None:
        spatial = config.get("SpatialCalibrations", {})
        chain_path = spatial.get(
            "dmd_chain_calib_path",
            "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1277.npz",
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


def subset_nv_list(nv_list_all, global_inds):
    """
    Convenience helper when nv_list_all order matches the global DMD order.
    """
    return [nv_list_all[int(ind)] for ind in global_inds]


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

    This averages all axes except the final two image axes.
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

def _normalize_crosstalk_matrix(
    matrix,
    measured_global_inds,
    source_global_inds,
    mode="source_diag",
    diag_floor=1e-6,
):
    """
    Normalize DMD crosstalk matrix using different conventions.

    matrix[row, col] = response at measured NV row when source NV col is passed.

    Modes
    -----
    source_diag:
        N[j,i] = C[j,i] / C[i,i]
        Best for: leakage fraction relative to intended source.

    global_mean_diag:
        N[j,i] = C[j,i] / mean(C[i,i])
        Best for: avoiding huge ratios from weak diagonal sources.

    symmetric_diag:
        N[j,i] = C[j,i] / sqrt(C[i,i] * C[j,j])
        Best for: pairwise symmetric comparison, only meaningful when
        measured/source sets overlap.

    column_sum:
        N[j,i] = C[j,i] / sum_j C[j,i]
        Best for: fraction of total transmitted light going to each measured NV.
    """
    mat = np.asarray(matrix, dtype=np.float32)
    measured_global_inds = [int(x) for x in measured_global_inds]
    source_global_inds = [int(x) for x in source_global_inds]

    row_lookup = {gind: row for row, gind in enumerate(measured_global_inds)}
    source_lookup = {gind: col for col, gind in enumerate(source_global_inds)}

    norm = np.full_like(mat, np.nan, dtype=np.float32)
    diag_values = np.full(len(source_global_inds), np.nan, dtype=np.float32)

    # Get intended diagonal response for each source column.
    for col, source_gind in enumerate(source_global_inds):
        row = row_lookup.get(source_gind, None)
        if row is not None:
            diag_values[col] = mat[row, col]

    finite_diag = diag_values[np.isfinite(diag_values)]
    finite_diag = finite_diag[np.abs(finite_diag) > diag_floor]

    if len(finite_diag) > 0:
        mean_diag = float(np.nanmean(finite_diag))
    else:
        mean_diag = np.nan

    if mode == "source_diag":
        for col, source_gind in enumerate(source_global_inds):
            denom = diag_values[col]
            if np.isfinite(denom) and abs(denom) > diag_floor:
                norm[:, col] = mat[:, col] / denom

    elif mode == "global_mean_diag":
        if np.isfinite(mean_diag) and abs(mean_diag) > diag_floor:
            norm = mat / mean_diag

    elif mode == "symmetric_diag":
        for col, source_gind in enumerate(source_global_inds):
            source_diag = diag_values[col]

            if not np.isfinite(source_diag) or abs(source_diag) <= diag_floor:
                continue

            for row, measured_gind in enumerate(measured_global_inds):
                measured_col = source_lookup.get(measured_gind, None)

                if measured_col is None:
                    continue

                measured_diag = diag_values[measured_col]

                if not np.isfinite(measured_diag) or abs(measured_diag) <= diag_floor:
                    continue

                denom = np.sqrt(abs(source_diag * measured_diag))

                if denom > diag_floor:
                    norm[row, col] = mat[row, col] / denom

    elif mode == "column_sum":
        col_sum = np.nansum(mat, axis=0)

        for col in range(mat.shape[1]):
            denom = col_sum[col]
            if np.isfinite(denom) and abs(denom) > diag_floor:
                norm[:, col] = mat[:, col] / denom

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return norm.astype(np.float32), diag_values.astype(np.float32)

def _compute_all_normalizations(
    matrix,
    measured_global_inds,
    source_global_inds,
):
    """
    Compute several useful DMD crosstalk normalizations.

    source_diag is the main one:
        C[j,i] / C[i,i]

    global_mean_diag is useful when weak diagonals make source_diag blow up.

    symmetric_diag is useful for pairwise comparison when measured/source sets match.

    column_sum tells what fraction of total transmitted signal goes to each row.
    """
    norm_source_diag, diag_values = _normalize_crosstalk_matrix(
        matrix,
        measured_global_inds,
        source_global_inds,
        mode="source_diag",
    )

    norm_global_mean_diag, _ = _normalize_crosstalk_matrix(
        matrix,
        measured_global_inds,
        source_global_inds,
        mode="global_mean_diag",
    )

    norm_symmetric_diag, _ = _normalize_crosstalk_matrix(
        matrix,
        measured_global_inds,
        source_global_inds,
        mode="symmetric_diag",
    )

    norm_column_sum, _ = _normalize_crosstalk_matrix(
        matrix,
        measured_global_inds,
        source_global_inds,
        mode="column_sum",
    )

    return {
        "normalized_crosstalk": norm_source_diag,
        "normalized_crosstalk_global_mean_diag": norm_global_mean_diag,
        "normalized_crosstalk_symmetric_diag": norm_symmetric_diag,
        "normalized_crosstalk_column_sum": norm_column_sum,
        "diag_values": diag_values,
    }

def _compute_crosstalk_products(
    counts_matrix,
    measured_global_inds,
    source_global_inds,
    background_counts=None,
):
    """
    Compute background-subtracted matrix, normalized matrices, diagonal values,
    and worst off-target fraction.

    The main normalized matrix is still:
        normalized_crosstalk = C[j,i] / C[i,i]

    Additional normalizations are returned for diagnostics.
    """
    counts_matrix = np.asarray(counts_matrix, dtype=np.float32)

    if background_counts is not None:
        background_counts = np.asarray(background_counts, dtype=np.float32)
        counts_bg_sub = counts_matrix - background_counts[:, None]
    else:
        counts_bg_sub = counts_matrix.copy()

    norms = _compute_all_normalizations(
        counts_bg_sub,
        measured_global_inds,
        source_global_inds,
    )

    normalized_crosstalk = norms["normalized_crosstalk"]
    diag_values = norms["diag_values"]

    max_off_frac = []
    for col, source_gind in enumerate(source_global_inds):
        source_gind = int(source_gind)
        off_vals = normalized_crosstalk[:, col].copy()

        for row, measured_gind in enumerate(measured_global_inds):
            if int(measured_gind) == source_gind:
                off_vals[row] = np.nan

        if np.all(np.isnan(off_vals)):
            max_off_frac.append(np.nan)
        else:
            max_off_frac.append(np.nanmax(off_vals))

    return {
        "counts_bg_sub": counts_bg_sub.astype(np.float32),
        "normalized_crosstalk": normalized_crosstalk.astype(np.float32),
        "normalized_crosstalk_global_mean_diag": norms[
            "normalized_crosstalk_global_mean_diag"
        ].astype(np.float32),
        "normalized_crosstalk_symmetric_diag": norms[
            "normalized_crosstalk_symmetric_diag"
        ].astype(np.float32),
        "normalized_crosstalk_column_sum": norms[
            "normalized_crosstalk_column_sum"
        ].astype(np.float32),
        "diag_values": np.asarray(diag_values, dtype=np.float32),
        "max_off_frac": np.asarray(max_off_frac, dtype=np.float32),
    }

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
    Plot matrix without calling tight_layout/subplots_adjust.

    This avoids the Matplotlib layout-engine/colorbar crash:
        RuntimeError: Colorbar layout of new layout engine not compatible...
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
# Analysis / plotting
# =============================================================================


def process_and_plot(raw_data):
    """
    Plot DMD crosstalk matrices and summary diagnostics.

    Plots:
        1. Raw counts matrix
        2. Analysis counts matrix: counts_bg_sub if available, else raw
        3. Normalized by intended NV: C[j,i] / C[i,i]
        4. Normalized by mean diagonal: C[j,i] / mean(C[i,i])
        5. Column-sum normalized matrix
        6. Diagonal signal per source
        7. Worst off-target / intended response per source
    """
    counts_matrix = np.asarray(raw_data["counts_matrix"], dtype=np.float32)

    measured_global_inds = raw_data.get(
        "measured_global_inds", list(range(counts_matrix.shape[0]))
    )
    source_global_inds = raw_data.get(
        "source_global_inds", list(range(counts_matrix.shape[1]))
    )

    # Use the same matrix for all derived analysis plots.
    # if "counts_bg_sub" in raw_data and raw_data["counts_bg_sub"] is not None:
    #     counts_for_analysis = np.asarray(raw_data["counts_bg_sub"], dtype=np.float32)
    #     counts_label = "background-subtracted"
    # else:
    #     counts_for_analysis = counts_matrix.copy()
    #     counts_label = "raw"
    
    counts_for_analysis = counts_matrix.copy()
    counts_label = "raw"

    norms = _compute_all_normalizations(
        counts_for_analysis,
        measured_global_inds,
        source_global_inds,
    )

    normalized_crosstalk = norms["normalized_crosstalk"]
    normalized_global_mean = norms["normalized_crosstalk_global_mean_diag"]
    normalized_symmetric = norms["normalized_crosstalk_symmetric_diag"]
    normalized_column_sum = norms["normalized_crosstalk_column_sum"]
    diag_values = norms["diag_values"]

    # Recompute worst off-target ratio from the same analysis matrix.
    products = _compute_crosstalk_products(
        counts_for_analysis,
        measured_global_inds,
        source_global_inds,
        background_counts=None,
    )
    max_off_frac = products["max_off_frac"]

    figs = []

    # 1. Raw counts matrix
    figs.append(
        _plot_matrix(
            counts_matrix,
            title="DMD crosstalk: raw counts",
            xlabel="source global NV index",
            ylabel="measured global NV index",
            cbar_label="mean counts",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
        )
    )

    # 2. Analysis counts matrix
    figs.append(
        _plot_matrix(
            counts_for_analysis,
            title=f"DMD crosstalk: {counts_label} counts",
            xlabel="source global NV index",
            ylabel="measured global NV index",
            cbar_label="mean counts",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
        )
    )

    # 3. Main crosstalk normalization
    figs.append(
        _plot_matrix(
            normalized_crosstalk,
            title="DMD crosstalk: normalized by intended NV",
            xlabel="source global NV index",
            ylabel="measured global NV index",
            cbar_label="C[j,i] / C[i,i]",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
            vmin=0,
            vmax=1,
        )
    )

    # 4. Mean-diagonal normalization
    figs.append(
        _plot_matrix(
            normalized_global_mean,
            title="DMD crosstalk: normalized by mean diagonal",
            xlabel="source global NV index",
            ylabel="measured global NV index",
            cbar_label="C[j,i] / mean(C[i,i])",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
            vmin=0,
            vmax=1,
        )
    )
    # symmetric diagonal normalized 
    figs.append(
    _plot_matrix(
        normalized_symmetric,
        title= "DMD crosstalk: symmetric diagonal normalized",
        xlabel="source global NV index",
        ylabel="measured global NV index",
        cbar_label="C[j,i] / sqrt(C[i,i] C[j,j])",
        xlabels=source_global_inds,
        ylabels=measured_global_inds,
        vmin=0,
        vmax=1,
    )
)

    # 5. Column-sum normalization
    figs.append(
        _plot_matrix(
            normalized_column_sum,
            title="DMD crosstalk: column-sum normalized",
            xlabel="source global NV index",
            ylabel="measured global NV index",
            cbar_label="C[j,i] / sum_j C[j,i]",
            xlabels=source_global_inds,
            ylabels=measured_global_inds,
            vmin=0,
            vmax=1,
        )
    )
    

    # 6. Diagonal signal per source
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(diag_values, "o-")
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("source column")
    ax.set_ylabel("intended response C[i,i]")
    ax.set_title(f"Diagonal signal per DMD source ({counts_label})")
    figs.append(fig)

    # 7. Worst off-target fraction per source
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(max_off_frac, "o-")
    ax.axhline(0.3, linestyle="--", label="target = 0.3")
    ax.axhline(1.0, linestyle="--", label="bad = 1.0")
    ax.set_xlabel("source column")
    ax.set_ylabel("max off-target / intended response")
    ax.set_title("Worst optical crosstalk per DMD source")
    ax.legend(fontsize=8)
    figs.append(fig)

    return figs
   
def analyze_crosstalk_metrics(
    raw_data,
    counts_key="counts_bg_sub",
    use_source_only=True,
    diag_threshold=2.0,
    ratio_threshold=0.3,
    bad_ratio_threshold=1.0,
    vmax_norm=0.3,
    do_plot=True,
):
    """
    Analyze DMD crosstalk matrix with scatter plots, histograms, and outlier detection.

    Parameters
    ----------
    raw_data : dict
        Raw data from dmd_crosstalk_matrix.main.

    counts_key : str
        Matrix key to analyze.
        Recommended:
            "counts_bg_sub" if background subtraction is reliable.
            "counts_matrix" if using raw counts.

    use_source_only : bool
        If True, removes the extra reference-NV row.
        This is important when:
            measured_inds = [0] + source_inds

    diag_threshold : float
        Minimum diagonal signal for a source to be considered strong.

    ratio_threshold : float
        Target/acceptable worst-crosstalk ratio.

    bad_ratio_threshold : float
        Ratio where off-target signal is comparable to or larger than target signal.

    vmax_norm : float
        Color scale maximum for normalized matrix plot.

    do_plot : bool
        If True, generate plots.

    Returns
    -------
    metrics : dict
        Contains matrices, summary numbers, source classifications, and figures.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------
    # Choose matrix.
    # ---------------------------------------------------------------------
    if counts_key in raw_data:
        C_full = np.asarray(raw_data[counts_key], dtype=float)
    elif "counts_bg_sub" in raw_data:
        C_full = np.asarray(raw_data["counts_bg_sub"], dtype=float)
        counts_key = "counts_bg_sub"
    else:
        C_full = np.asarray(raw_data["counts_matrix"], dtype=float)
        counts_key = "counts_matrix"

    measured_inds = [int(x) for x in raw_data["measured_global_inds"]]
    source_inds = [int(x) for x in raw_data["source_global_inds"]]

    # ---------------------------------------------------------------------
    # Remove reference row if measured_inds = [0] + source_inds.
    # ---------------------------------------------------------------------
    if use_source_only:
        source_rows = [measured_inds.index(src) for src in source_inds]
        C = C_full[source_rows, :]
        row_inds = source_inds
    else:
        C = C_full
        row_inds = measured_inds

    num_sources = len(source_inds)

    # ---------------------------------------------------------------------
    # Normalize each column by intended diagonal response.
    # ---------------------------------------------------------------------
    norms = _compute_all_normalizations(
        C,
        row_inds,
        source_inds,
    )

    Cnorm = norms["normalized_crosstalk"]
    Cnorm_global = norms["normalized_crosstalk_global_mean_diag"]
    Cnorm_symmetric = norms["normalized_crosstalk_symmetric_diag"]
    Cnorm_column_sum = norms["normalized_crosstalk_column_sum"]
    diag_vals = norms["diag_values"].astype(float)

    # ---------------------------------------------------------------------
    # Compute worst off-diagonal response for each source.
    # ---------------------------------------------------------------------
    worst_ratio = np.full(num_sources, np.nan, dtype=float)
    worst_off_signal = np.full(num_sources, np.nan, dtype=float)
    worst_target = [None] * num_sources
    off_vals = []

    for col, src in enumerate(source_inds):
        if src not in row_inds:
            continue

        intended_row = row_inds.index(src)
        intended = C[intended_row, col]

        col_vals = C[:, col].copy()
        col_vals[intended_row] = np.nan

        if np.all(np.isnan(col_vals)):
            continue

        worst_row = int(np.nanargmax(col_vals))
        max_off = float(col_vals[worst_row])

        worst_off_signal[col] = max_off
        worst_target[col] = row_inds[worst_row]

        if np.isfinite(intended) and abs(intended) > 1e-12:
            worst_ratio[col] = max_off / intended

        for r in range(C.shape[0]):
            if r != intended_row:
                off_vals.append(C[r, col])

    off_vals = np.asarray(off_vals, dtype=float)

    # ---------------------------------------------------------------------
    # Classify sources.
    # ---------------------------------------------------------------------
    good = (diag_vals > diag_threshold) & (worst_ratio < ratio_threshold)
    usable = (diag_vals > diag_threshold) & (worst_ratio < 0.5)
    weak = diag_vals <= diag_threshold
    bad_xtalk = worst_ratio >= bad_ratio_threshold
    high_xtalk = (worst_ratio >= ratio_threshold) & (worst_ratio < bad_ratio_threshold)

    good_sources = np.asarray(source_inds)[good]
    usable_sources = np.asarray(source_inds)[usable]
    weak_sources = np.asarray(source_inds)[weak]
    bad_xtalk_sources = np.asarray(source_inds)[bad_xtalk]
    high_xtalk_sources = np.asarray(source_inds)[high_xtalk]

    # ---------------------------------------------------------------------
    # Summary numbers.
    # ---------------------------------------------------------------------
    mean_diag = np.nanmean(diag_vals)
    median_diag = np.nanmedian(diag_vals)
    mean_off = np.nanmean(off_vals)
    median_off = np.nanmedian(off_vals)
    max_off = np.nanmax(off_vals)

    mean_worst_ratio = np.nanmean(worst_ratio)
    median_worst_ratio = np.nanmedian(worst_ratio)
    max_worst_ratio = np.nanmax(worst_ratio)

    mean_off_over_mean_diag = mean_off / mean_diag
    max_off_over_mean_diag = max_off / mean_diag

    print("\nDMD crosstalk summary")
    print("---------------------")
    print(f"counts_key: {counts_key}")
    print(f"matrix shape used: {C.shape}")
    print(f"number of sources: {num_sources}")
    print(f"mean diagonal signal: {mean_diag:.4f}")
    print(f"median diagonal signal: {median_diag:.4f}")
    print(f"mean off-diagonal signal: {mean_off:.4f}")
    print(f"median off-diagonal signal: {median_off:.4f}")
    print(f"max off-diagonal signal: {max_off:.4f}")
    print(f"mean off / mean diag: {mean_off_over_mean_diag:.4f}")
    print(f"max off / mean diag: {max_off_over_mean_diag:.4f}")
    print(f"mean worst ratio: {mean_worst_ratio:.4f}")
    print(f"median worst ratio: {median_worst_ratio:.4f}")
    print(f"max worst ratio: {max_worst_ratio:.4f}")

    print("\nClassification")
    print("--------------")
    print(
        f"good sources, diag > {diag_threshold}, "
        f"ratio < {ratio_threshold}: {len(good_sources)}"
    )
    print(
        f"usable sources, diag > {diag_threshold}, "
        f"ratio < 0.5: {len(usable_sources)}"
    )
    print(
        f"weak diagonal sources, diag <= {diag_threshold}: "
        f"{len(weak_sources)}"
    )
    print(
        f"high crosstalk sources, {ratio_threshold} <= ratio < "
        f"{bad_ratio_threshold}: {len(high_xtalk_sources)}"
    )
    print(
        f"bad crosstalk sources, ratio >= {bad_ratio_threshold}: "
        f"{len(bad_xtalk_sources)}"
    )

    print("\nBad crosstalk sources:")
    for col, src in enumerate(source_inds):
        if worst_ratio[col] >= bad_ratio_threshold:
            print(
                f"source {src}: "
                f"diag={diag_vals[col]:.3f}, "
                f"worst target={worst_target[col]}, "
                f"worst off={worst_off_signal[col]:.3f}, "
                f"ratio={worst_ratio[col]:.3f}"
            )

    # ---------------------------------------------------------------------
    # Plots.
    # ---------------------------------------------------------------------
    figs = []

    if do_plot:
        # -------------------------------------------------------------
        # 1. Normalized matrix.
        # -------------------------------------------------------------
        fig, ax = plt.subplots()
        im = ax.imshow(Cnorm, aspect="auto", vmin=0, vmax=vmax_norm)
        ax.set_xlabel("DMD source NV")
        ax.set_ylabel("Measured NV")
        ax.set_title("Normalized DMD crosstalk matrix")
        fig.colorbar(im, ax=ax, label="C[j,i] / C[i,i]")
        figs.append(fig)

        # -------------------------------------------------------------
        # 2. Diagonal signal vs worst ratio.
        # -------------------------------------------------------------
        fig, ax = plt.subplots()

        ax.scatter(
            diag_vals[good],
            worst_ratio[good],
            label="Good",
            alpha=0.8,
        )

        ax.scatter(
            diag_vals[weak],
            worst_ratio[weak],
            label="Weak diagonal",
            alpha=0.8,
        )

        ax.scatter(
            diag_vals[high_xtalk],
            worst_ratio[high_xtalk],
            label="High crosstalk",
            alpha=0.8,
        )

        ax.scatter(
            diag_vals[bad_xtalk],
            worst_ratio[bad_xtalk],
            label="Bad crosstalk",
            alpha=0.8,
        )

        ax.axhline(
            ratio_threshold,
            linestyle="--",
            label=f"Target ratio = {ratio_threshold}",
        )

        ax.axhline(
            bad_ratio_threshold,
            linestyle="--",
            label=f"Bad ratio = {bad_ratio_threshold}",
        )

        ax.axvline(
            diag_threshold,
            linestyle="--",
            label=f"Weak diag cutoff = {diag_threshold}",
        )

        ax.set_xlabel("Diagonal signal")
        ax.set_ylabel("Worst off-diagonal / diagonal")
        ax.set_title("DMD crosstalk outliers")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        figs.append(fig)

        # -------------------------------------------------------------
        # 3. Diagonal signal vs absolute leakage.
        # -------------------------------------------------------------
        fig, ax = plt.subplots()

        ax.scatter(
            diag_vals,
            worst_off_signal,
            alpha=0.8,
            label="Sources",
        )

        x_max = np.nanmax(diag_vals)
        xline = np.linspace(0, x_max, 200)

        ax.plot(
            xline,
            ratio_threshold * xline,
            linestyle="--",
            label=f"{ratio_threshold} × diagonal",
        )

        ax.plot(
            xline,
            bad_ratio_threshold * xline,
            linestyle="--",
            label=f"{bad_ratio_threshold} × diagonal",
        )

        ax.axvline(
            diag_threshold,
            linestyle="--",
            label=f"Weak diag cutoff = {diag_threshold}",
        )

        ax.set_xlabel("Diagonal signal")
        ax.set_ylabel("Worst off-diagonal signal")
        ax.set_title("Absolute leakage vs intended signal")
        ax.legend(fontsize=8)
        figs.append(fig)

        # -------------------------------------------------------------
        # 4. Histogram of worst ratios.
        # -------------------------------------------------------------
        fig, ax = plt.subplots()

        finite_ratio = worst_ratio[np.isfinite(worst_ratio)]
        ax.hist(finite_ratio, bins=30)

        ax.axvline(
            ratio_threshold,
            linestyle="--",
            label=f"Target = {ratio_threshold}",
        )

        ax.axvline(
            bad_ratio_threshold,
            linestyle="--",
            label=f"Bad = {bad_ratio_threshold}",
        )

        ax.set_xlabel("Worst off-diagonal / diagonal")
        ax.set_ylabel("Number of source NVs")
        ax.set_title("Distribution of worst DMD crosstalk")
        ax.legend(fontsize=8)
        figs.append(fig)

        # -------------------------------------------------------------
        # 5. Worst ratio per source.
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            worst_ratio,
            "o-",
            label="Worst crosstalk ratio",
        )

        ax.axhline(
            ratio_threshold,
            linestyle="--",
            label=f"Target = {ratio_threshold}",
        )

        ax.axhline(
            bad_ratio_threshold,
            linestyle="--",
            label=f"Bad = {bad_ratio_threshold}",
        )

        ax.set_xlabel("Source column")
        ax.set_ylabel("Worst off-diagonal / diagonal")
        ax.set_title("Worst optical crosstalk per source")
        ax.legend(fontsize=8)
        figs.append(fig)

    metrics = {
        "counts_key": counts_key,
        "counts_matrix_used": C,
        "normalized_crosstalk": Cnorm,
        "row_global_inds": row_inds,
        "source_global_inds": source_inds,
        "diag_vals": diag_vals,
        "off_vals": off_vals,
        "worst_ratio": worst_ratio,
        "worst_off_signal": worst_off_signal,
        "worst_target": worst_target,
        "good_mask": good,
        "usable_mask": usable,
        "weak_mask": weak,
        "high_xtalk_mask": high_xtalk,
        "bad_xtalk_mask": bad_xtalk,
        "good_sources": good_sources,
        "usable_sources": usable_sources,
        "weak_sources": weak_sources,
        "high_xtalk_sources": high_xtalk_sources,
        "bad_xtalk_sources": bad_xtalk_sources,
        "mean_diag": mean_diag,
        "median_diag": median_diag,
        "mean_off": mean_off,
        "median_off": median_off,
        "max_off": max_off,
        "mean_off_over_mean_diag": mean_off_over_mean_diag,
        "max_off_over_mean_diag": max_off_over_mean_diag,
        "mean_worst_ratio": mean_worst_ratio,
        "median_worst_ratio": median_worst_ratio,
        "max_worst_ratio": max_worst_ratio,
        "figs": figs,
        "normalized_crosstalk_global_mean_diag": Cnorm_global,
        "normalized_crosstalk_symmetric_diag": Cnorm_symmetric,
        "normalized_crosstalk_column_sum": Cnorm_column_sum,
    }

    return metrics
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
    dmd_mode="pass_single",
    do_polarize=True,
    targeted_polarization=False,
    take_background=True,
    save_images=True,
    save_raw_counts_by_source=True,
    dmd_settle_s=0.10,
    dmd_plane=230,
):
    """
    Run DMD crosstalk matrix experiment.

    Parameters
    ----------
    nv_list : list[NVSig]
        NVs measured by base_routine. Rows of the crosstalk matrix follow this order.

    source_global_inds : list[int]
        Global DMD/NV indices to select one at a time. Columns follow this order.

    measured_global_inds : list[int] or None
        Global indices corresponding to nv_list rows.
        If None, assumes measured_global_inds = [0, 1, ..., len(nv_list)-1].

    dmd_mode : str
        "pass_single":
            black/block background, white/pass disk at selected source NV.
        "block_single":
            white/pass background, black/block disk at selected source NV.
        "pass_all":
            no source-dependent DMD mask; control mode.

    do_polarize : bool
        If True, sequence applies charge polarization before readout.

    targeted_polarization : bool
        Passed into macro_polarize. Usually False for first optical crosstalk tests.

    take_background : bool
        If True, take one DMD block_all acquisition first.

    save_raw_counts_by_source : bool
        If True, save full raw counts for each DMD source. This is useful for
        debugging but can become large for many sources.
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

    # Ensure base_routine has a valid positioning/reference NV.
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

    counts_matrix = np.zeros((num_measured, num_sources), dtype=np.float32)
    counts_matrix_ste = np.zeros((num_measured, num_sources), dtype=np.float32)
    raw_counts_by_source = []
    mean_images_by_source = []

    for col, source_ind in enumerate(source_global_inds):
        print(f"\n=== DMD source {col + 1}/{num_sources}: global index {source_ind} ===")

        if dmd_mode == "pass_single":
            dmd.pass_loaded_indices(
                json.dumps([int(source_ind)]),
                int(dmd_radius_px),
                int(dmd_plane),
            )

        elif dmd_mode == "block_single":
            dmd.block_loaded_indices(
                json.dumps([int(source_ind)]),
                int(dmd_radius_px),
                int(dmd_plane),
            )

        elif dmd_mode == "pass_all":
            dmd.pass_all(True)

        else:
            raise ValueError(
                "dmd_mode must be 'pass_single', 'block_single', or 'pass_all'."
            )

        time.sleep(dmd_settle_s)

        step_data = run_readout_once()
        mean_counts, ste_counts = _mean_counts_per_nv(step_data)

        counts_matrix[:, col] = mean_counts
        counts_matrix_ste[:, col] = ste_counts

        if save_raw_counts_by_source:
            raw_counts_by_source.append(np.asarray(step_data["counts"]))

        mean_img = _mean_image(step_data)
        if mean_img is not None:
            mean_images_by_source.append(mean_img)

        print("mean_counts:")
        print(mean_counts)

    # Return to normal pass state with zero blocked.
    try:
        dmd.zero_block_on()
    except Exception:
        print("Could not restore zero_block_on:")
        print(traceback.format_exc())

    # Compute derived arrays before plotting/saving so they are always present.
    products = _compute_crosstalk_products(
        counts_matrix,
        measured_global_inds,
        source_global_inds,
        background_counts,
    )

    counts_bg_sub = products["counts_bg_sub"]
    normalized_crosstalk = products["normalized_crosstalk"]
    normalized_crosstalk_global_mean_diag = products[
        "normalized_crosstalk_global_mean_diag"
    ]
    normalized_crosstalk_symmetric_diag = products[
        "normalized_crosstalk_symmetric_diag"
    ]
    normalized_crosstalk_column_sum = products[
        "normalized_crosstalk_column_sum"
    ]
    diag_values = products["diag_values"]
    max_off_frac = products["max_off_frac"]
    
    timestamp = dm.get_time_stamp()
    repr_nv_sig = ensure_representative_nv(nv_list)
    repr_nv_name = repr_nv_sig.name

    raw_data = {
        "timestamp": timestamp,
        "experiment": "dmd_crosstalk_matrix",
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
        "dmd_mode": dmd_mode,
        "dmd_plane": int(dmd_plane),
        "do_polarize": bool(do_polarize),
        "targeted_polarization": bool(targeted_polarization),
        "take_background": bool(take_background),
        "save_images": bool(save_images),
        "save_raw_counts_by_source": bool(save_raw_counts_by_source),
        "dmd_settle_s": float(dmd_settle_s),
        "counts_matrix": counts_matrix,
        "counts_matrix_ste": counts_matrix_ste,
        "background_counts": background_counts,
        "background_counts_ste": background_counts_ste,
        "counts_bg_sub": counts_bg_sub,
        "normalized_crosstalk": normalized_crosstalk,
        "diag_values": diag_values,
        "max_off_frac": max_off_frac,
        "background_raw_counts": background_raw_counts,
        "img_array-units": "photons",
    }
    raw_data |= {
    "counts_bg_sub": counts_bg_sub,
    "normalized_crosstalk": normalized_crosstalk,
    "normalized_crosstalk_global_mean_diag": normalized_crosstalk_global_mean_diag,
    "normalized_crosstalk_symmetric_diag": normalized_crosstalk_symmetric_diag,
    "normalized_crosstalk_column_sum": normalized_crosstalk_column_sum,
    "diag_values": diag_values,
    "max_off_frac": max_off_frac,
    }

    if len(mean_images_by_source) > 0:
        try:
            raw_data["mean_images_by_source"] = np.stack(mean_images_by_source, axis=0)
        except Exception:
            print("Skipping mean_images_by_source because shapes are inconsistent.")
        
    
    if background_img is not None:
            raw_data["background_img"] = background_img

    if len(mean_images_by_source) > 0:
        try:
            raw_data["mean_images_by_source"] = np.stack(mean_images_by_source, axis=0)
        except Exception:
            raw_data["mean_images_by_source"] = np.asarray(mean_images_by_source, dtype=object)

    try:
        figs = process_and_plot(raw_data)
    except Exception:
        print("process_and_plot failed, but raw data will still be saved.")
        print(traceback.format_exc())
        figs = []

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        f"{repr_nv_name}-dmd-crosstalk-{num_sources}src-r{dmd_radius_px}",
    )

    keys_to_compress = [
        "counts_matrix",
        "counts_matrix_ste",
        "counts_bg_sub",
        "normalized_crosstalk",
        "normalized_crosstalk_global_mean_diag",
        "normalized_crosstalk_symmetric_diag",
        "normalized_crosstalk_column_sum",
        "diag_values",
        "max_off_frac",
        "background_counts",
        "background_counts_ste",
        "background_img",
    ]

    if save_raw_counts_by_source:
        keys_to_compress += [
            "raw_counts_by_source",
            "background_raw_counts",
            "mean_images_by_source",
        ]

    keys_to_compress = [key for key in keys_to_compress if key in raw_data]

    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    for ind, fig in enumerate(figs):
        fig_path = dm.get_file_path(
            __file__,
            timestamp,
            f"{repr_nv_name}-dmd-crosstalk-{num_sources}src-r{dmd_radius_px}-{ind}",
        )
        dm.save_figure(fig, fig_path)

    tb.reset_cfm()

    return raw_data



if __name__ == "__main__":
    kpl.init_kplotlib()

    raw_data = dm.get_raw_data(
        file_stem=(
            "2026_05_11-02_24_11-qnami-nv0_2026_02_20-dmd-crosstalk-200src-r20"
        ),
        load_npz=True,
        allow_pickle=True,
    )

    process_and_plot(raw_data)

    metrics = analyze_crosstalk_metrics(
        raw_data,
        counts_key="counts_bg_sub",
        use_source_only=True,
        diag_threshold=2.0,
        ratio_threshold=0.3,
        bad_ratio_threshold=1.0,
        vmax_norm=0.3,
        do_plot=True,
    )

    print("\nGood source indices:")
    print(metrics["good_sources"].tolist())

    print("\nUsable source indices:")
    print(metrics["usable_sources"].tolist())

    print("\nWeak diagonal source indices:")
    print(metrics["weak_sources"].tolist())

    print("\nBad crosstalk source indices:")
    print(metrics["bad_xtalk_sources"].tolist())

    kpl.show(block=True)
    # sys.exit()