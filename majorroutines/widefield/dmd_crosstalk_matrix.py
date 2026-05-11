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


def _normalize_by_matching_diagonal(matrix, measured_global_inds, source_global_inds):
    """
    Normalize each source column by the measured row corresponding to the same
    global NV index.

    This works even when matrix is rectangular or rows/columns are not in the
    same order.
    """
    mat = np.asarray(matrix, dtype=np.float32)
    measured_global_inds = [int(x) for x in measured_global_inds]
    source_global_inds = [int(x) for x in source_global_inds]

    row_lookup = {gind: row for row, gind in enumerate(measured_global_inds)}

    norm = np.full_like(mat, np.nan, dtype=np.float32)
    diag_values = np.full(len(source_global_inds), np.nan, dtype=np.float32)

    for col, source_gind in enumerate(source_global_inds):
        row = row_lookup.get(source_gind, None)
        if row is None:
            continue

        denom = mat[row, col]
        diag_values[col] = denom

        if np.isfinite(denom) and abs(denom) > 1e-9:
            norm[:, col] = mat[:, col] / denom

    return norm, diag_values


def _compute_crosstalk_products(
    counts_matrix,
    measured_global_inds,
    source_global_inds,
    background_counts=None,
):
    """
    Compute background-subtracted matrix, normalized matrix, diagonal values,
    and worst off-target fraction.

    This is called before plotting, so these arrays are always available for saving
    even if plotting fails.
    """
    counts_matrix = np.asarray(counts_matrix, dtype=np.float32)

    if background_counts is not None:
        background_counts = np.asarray(background_counts, dtype=np.float32)
        counts_bg_sub = counts_matrix - background_counts[:, None]
    else:
        counts_bg_sub = counts_matrix.copy()

    normalized_crosstalk, diag_values = _normalize_by_matching_diagonal(
        counts_bg_sub,
        measured_global_inds,
        source_global_inds,
    )

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

    return (
        counts_bg_sub.astype(np.float32),
        normalized_crosstalk.astype(np.float32),
        np.asarray(diag_values, dtype=np.float32),
        np.asarray(max_off_frac, dtype=np.float32),
    )


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
    Plot raw, background-subtracted, and normalized DMD crosstalk matrices.

    This function is robust: it uses precomputed counts_bg_sub and
    normalized_crosstalk when available, otherwise computes them.
    """
    counts_matrix = np.asarray(raw_data["counts_matrix"], dtype=np.float32)

    measured_global_inds = raw_data.get(
        "measured_global_inds", list(range(counts_matrix.shape[0]))
    )
    source_global_inds = raw_data.get(
        "source_global_inds", list(range(counts_matrix.shape[1]))
    )

    # For the first DMD crosstalk characterization, use raw counts.
    # Background subtraction can be re-enabled later after the DMD effect is large.
    counts_for_analysis = counts_matrix.copy()

    normalized_crosstalk, diag_values = _normalize_by_matching_diagonal(
        counts_for_analysis,
        measured_global_inds,
        source_global_inds,
    )

    figs = []

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

    if "max_off_frac" in raw_data and raw_data["max_off_frac"] is not None:
        max_off_frac = np.asarray(raw_data["max_off_frac"], dtype=np.float32)
    else:
        _, _, _, max_off_frac = _compute_crosstalk_products(
            counts_matrix,
            measured_global_inds,
            source_global_inds,
            raw_data.get("background_counts", None),
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(max_off_frac, "o-")
    ax.set_xlabel("source column")
    ax.set_ylabel("max off-target / intended response")
    ax.set_title("Worst optical crosstalk per DMD source")
    figs.append(fig)

    return figs


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
    (
        counts_bg_sub,
        normalized_crosstalk,
        diag_values,
        max_off_frac,
    ) = _compute_crosstalk_products(
        counts_matrix,
        measured_global_inds,
        source_global_inds,
        background_counts,
    )

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

    if save_raw_counts_by_source and len(raw_counts_by_source) > 0:
        try:
            raw_data["raw_counts_by_source"] = np.stack(raw_counts_by_source, axis=0)
        except Exception:
            raw_data["raw_counts_by_source"] = np.asarray(raw_counts_by_source, dtype=object)

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
        "diag_values",
        "max_off_frac",
        "raw_counts_by_source",
        "background_raw_counts",
        "background_img",
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

    # Reload / analysis example:
    raw_data = dm.get_raw_data(file_stem="2026_05_10-19_54_17-qnami-nv0_2026_02_20-dmd-crosstalk-10src-r30", load_npz=True)
    process_and_plot(raw_data)

    C = np.asarray(raw_data["counts_bg_sub"], dtype=float)
    Cnorm = np.asarray(raw_data["normalized_crosstalk"], dtype=float)

    source_inds = raw_data["source_global_inds"]
    measured_inds = raw_data["measured_global_inds"]

    print("counts_bg_sub:")
    print(np.round(C, 2))

    print("normalized_crosstalk:")
    print(np.round(Cnorm, 3))

    # Diagonal/off-diagonal metrics
    diag_vals = []
    off_vals = []

    for col, src in enumerate(source_inds):
        if src in measured_inds:
            row = measured_inds.index(src)
            diag_vals.append(C[row, col])

            for r in range(C.shape[0]):
                if r != row:
                    off_vals.append(C[r, col])

    diag_vals = np.asarray(diag_vals)
    off_vals = np.asarray(off_vals)

    print("Mean diagonal signal:", np.nanmean(diag_vals))
    print("Mean off-diagonal signal:", np.nanmean(off_vals))
    print("Max off-diagonal signal:", np.nanmax(off_vals))
    print("Max off/mean diag:", np.nanmax(off_vals) / np.nanmean(diag_vals))

    print("    )")
    kpl.show(block=True)
    # sys.exit()