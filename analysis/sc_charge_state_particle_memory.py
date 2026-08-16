# -*- coding: utf-8 -*-
"""
Back-to-back analysis for particle-memory NV charge-state datasets.

This is an analysis-only file. It does not run the experiment.

Primary workflow
----------------
1. Load sequential 3600 s particle-memory files.
2. Classify NV- using the saved per-NV thresholds.
3. Concatenate runs in acquisition order.
4. Plot rep 11 and rep 12 continuously across file boundaries.
5. Plot rep12 - rep11 and rep12 / rep11.
6. Optionally inspect one run at the raw-image level.
7. Optionally run the more detailed drift/artifact diagnostic.

Expected saved arrays
---------------------
counts[exp, nv, run, step, rep]
img_arrays[exp, run, step, rep, y, x]
"""

from __future__ import annotations
import sys
import re
from datetime import datetime
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import convolve, map_coordinates
from scipy.signal import fftconvolve

from utils import data_manager as dm
from utils import kplotlib as kpl


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


def estimate_drift_from_bright_nvs(
    img11,
    img12,
    coords_xy,
    counts11,
    thresholds,
    roi_radius=5,
    threshold_margin=5.0,
    min_spot_signal=5.0,
):
    """
    Estimate image drift using centroids of bright NVs only.

    Returns
    -------
    dx, dy : float
        Robust median drift from rep11 -> rep12, in camera pixels.

    details : dict
        Per-NV displacement information.
    """

    img11 = np.asarray(img11, dtype=float)
    img12 = np.asarray(img12, dtype=float)
    coords_xy = np.asarray(coords_xy, dtype=float)

    # ----------------------------------------------------------
    # Select clearly bright NVs in rep 11
    # ----------------------------------------------------------
    bright_mask = counts11 > (thresholds + threshold_margin)
    bright_inds = np.where(bright_mask)[0]

    dx_list = []
    dy_list = []
    used_inds = []

    def spot_centroid(img, x0, y0, r):
        x0 = int(round(x0))
        y0 = int(round(y0))

        y_min = max(0, y0 - r)
        y_max = min(img.shape[0], y0 + r + 1)
        x_min = max(0, x0 - r)
        x_max = min(img.shape[1], x0 + r + 1)

        patch = img[y_min:y_max, x_min:x_max].copy()

        if patch.size == 0:
            return None

        # Estimate local background from lower-intensity pixels
        background = np.percentile(patch, 30)
        weights = patch - background
        weights[weights < 0] = 0

        total = np.sum(weights)

        if total < min_spot_signal:
            return None

        yy, xx = np.indices(patch.shape)

        cx = np.sum(xx * weights) / total + x_min
        cy = np.sum(yy * weights) / total + y_min

        return cx, cy

    # ----------------------------------------------------------
    # Centroid each bright NV in rep 11 and rep 12
    # ----------------------------------------------------------
    for nv_ind in bright_inds:

        x, y = coords_xy[nv_ind]

        c11 = spot_centroid(
            img11,
            x,
            y,
            roi_radius,
        )

        c12 = spot_centroid(
            img12,
            x,
            y,
            roi_radius,
        )

        if c11 is None or c12 is None:
            continue

        x11, y11 = c11
        x12, y12 = c12

        dx_list.append(x12 - x11)
        dy_list.append(y12 - y11)
        used_inds.append(nv_ind)

    dx_arr = np.asarray(dx_list)
    dy_arr = np.asarray(dy_list)
    used_inds = np.asarray(used_inds)

    if len(dx_arr) < 3:
        return np.nan, np.nan, {
            "used_nv_inds": used_inds,
            "dx": dx_arr,
            "dy": dy_arr,
        }

    # ----------------------------------------------------------
    # Robust outlier rejection
    # ----------------------------------------------------------
    med_dx = np.median(dx_arr)
    med_dy = np.median(dy_arr)

    radial_residual = np.sqrt(
        (dx_arr - med_dx) ** 2
        + (dy_arr - med_dy) ** 2
    )

    med_res = np.median(radial_residual)
    mad_res = np.median(
        np.abs(radial_residual - med_res)
    )

    if mad_res > 0:
        good = radial_residual < (
            med_res + 4.0 * 1.4826 * mad_res
        )
    else:
        good = np.ones(len(dx_arr), dtype=bool)

    dx_final = np.median(dx_arr[good])
    dy_final = np.median(dy_arr[good])

    details = {
        "used_nv_inds": used_inds[good],
        "all_used_nv_inds": used_inds,
        "dx": dx_arr,
        "dy": dy_arr,
        "good_mask": good,
        "num_bright_rep11": len(bright_inds),
        "num_used": np.sum(good),
    }

    return float(dx_final), float(dy_final), details


def _parse_file_stem_timestamp(file_stem: str) -> Optional[datetime]:
    """Parse YYYY_MM_DD-HH_MM_SS from the beginning of a data file stem."""
    match = re.match(r"^(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})", str(file_stem))
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y_%m_%d-%H_%M_%S")
    except ValueError:
        return None


def _format_file_boundary_label(file_stem: str, file_ind: int) -> str:
    timestamp = _parse_file_stem_timestamp(file_stem)
    if timestamp is None:
        return f"file {file_ind + 1}"
    return timestamp.strftime("%b %d\n%H:%M")


def _draw_file_boundaries(
    ax,
    dataset_results,
    *,
    annotate=True,
    alpha=0.35,
):
    """Draw separators between sequential raw-data files."""
    for dataset_ind, dataset in enumerate(dataset_results):
        start = int(dataset["global_run_start"])
        if dataset_ind > 0:
            ax.axvline(
                start - 0.5,
                linestyle="--",
                linewidth=1.0,
                color="0.45",
                alpha=alpha,
            )

        if annotate:
            stop = int(dataset["global_run_stop"])
            center = 0.5 * (start + stop - 1)
            label = _format_file_boundary_label(
                dataset["file_stem"],
                dataset_ind,
            )
            ax.text(
                center,
                1.01,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="0.35",
            )


def plot_nv_minus_by_run_separate_reps(
    file_stems,
    selected_waits_s=None,
    rep_inds=(11, 12),
    rep_labels=None,
    exclude_nv_inds=None,
    show_fraction=False,
    ncols=3,
    verbose=True,
    back_to_back=True,
    mark_file_boundaries=True,
    show_rep_comparison=True,
    show_difference=True,
    show_retention=True,
):
    """
    Plot run-by-run NV- population for one or more rep indices.

    For sequential files acquired back-to-back, ``back_to_back=True`` preserves
    the supplied file order and concatenates all runs onto one global run axis.

    This keeps the original function name and return structure:
        dataset_results, figures

    Parameters
    ----------
    file_stems : sequence of str
        Raw particle-memory dataset file stems, in acquisition order.

    selected_waits_s : sequence or None
        Optional dark-wait filter. Duplicate wait times are retained, which is
        important for repeated 3600 s datasets acquired back-to-back.

    rep_inds : sequence of int
        Rep indices to inspect. For the present experiment use (11, 12).

    rep_labels : dict or None
        Human-readable labels keyed by rep index.

    exclude_nv_inds : sequence or None
        Original NV indices to exclude.

    show_fraction : bool
        False -> plot number of NV-.
        True  -> plot fraction of retained NVs classified as NV-.

    ncols : int
        Used only when ``back_to_back=False`` to reproduce the old subplot view.

    back_to_back : bool
        True -> concatenate files in the supplied order onto a global run axis.
        False -> retain the older one-subplot-per-file behavior.

    mark_file_boundaries : bool
        Draw dashed vertical lines and timestamp labels between files.

    show_rep_comparison : bool
        Add one overlay figure containing all requested reps.

    show_difference : bool
        For exactly two reps, add rep_final - rep_initial versus global run.

    show_retention : bool
        For exactly two reps, add rep_final / rep_initial versus global run.

    Returns
    -------
    dataset_results : list of dict
        Per-file values plus global run indices.

    figures : dict
        Integer keys contain one figure per rep.
        Additional keys may include "comparison", "difference", and "retention".
    """

    file_stems = list(file_stems)
    rep_inds = tuple(int(rep_ind) for rep_ind in rep_inds)

    if len(rep_inds) == 0:
        raise ValueError("rep_inds cannot be empty.")

    if rep_labels is None:
        rep_labels = {rep_ind: f"rep {rep_ind}" for rep_ind in rep_inds}
    else:
        rep_labels = {
            rep_ind: rep_labels.get(rep_ind, f"rep {rep_ind}")
            for rep_ind in rep_inds
        }

    if exclude_nv_inds is None:
        exclude_nv_inds = np.array([], dtype=int)
    else:
        exclude_nv_inds = np.unique(
            np.asarray(exclude_nv_inds, dtype=int)
        )

    selected_waits_arr = None
    if selected_waits_s is not None:
        selected_waits_arr = np.asarray(
            selected_waits_s,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Load files in the order supplied.
    # DO NOT sort by dark_wait_s: repeated 3600 s files are sequential data.
    # ------------------------------------------------------------------
    dataset_results = []
    global_run_offset = 0

    for file_ind, file_stem in enumerate(file_stems):
        raw_data = dm.get_raw_data(
            file_stem=file_stem,
            load_npz=True,
        )

        wait_s = float(raw_data["dark_wait_s"])

        if selected_waits_arr is not None:
            if not np.any(np.isclose(wait_s, selected_waits_arr)):
                continue

        counts_all = np.asarray(
            raw_data["counts"],
            dtype=float,
        )

        if counts_all.ndim != 5:
            raise ValueError(
                "Expected counts[exp, nv, run, step, rep], "
                f"got {counts_all.shape} for {file_stem}"
            )

        # [exp, nv, run, step, rep] -> [nv, run, rep]
        counts = counts_all[0, :, :, 0, :]
        num_nvs, num_runs, num_reps = counts.shape

        if "analysis_thresholds" in raw_data:
            thresholds = np.asarray(
                raw_data["analysis_thresholds"],
                dtype=float,
            )
        elif "thresholds" in raw_data:
            thresholds = np.asarray(
                raw_data["thresholds"],
                dtype=float,
            )
        else:
            raise ValueError(
                f"No thresholds found in {file_stem}."
            )

        if thresholds.shape != (num_nvs,):
            raise ValueError(
                f"Threshold shape mismatch in {file_stem}: "
                f"{thresholds.shape} vs {(num_nvs,)}"
            )

        keep_mask = np.ones(num_nvs, dtype=bool)
        valid_excluded = exclude_nv_inds[
            (exclude_nv_inds >= 0)
            & (exclude_nv_inds < num_nvs)
        ]
        keep_mask[valid_excluded] = False

        counts_kept = counts[keep_mask, :, :]
        thresholds_kept = thresholds[keep_mask]
        num_kept_nvs = int(np.sum(keep_mask))

        if num_kept_nvs == 0:
            raise ValueError(
                f"No NVs remain after exclusions for {file_stem}."
            )

        nvm_mask = (
            counts_kept
            > thresholds_kept[:, None, None]
        )

        rep_counts = {}
        for rep_ind in rep_inds:
            if not (0 <= rep_ind < num_reps):
                raise ValueError(
                    f"Requested rep {rep_ind} is outside "
                    f"[0, {num_reps - 1}] for {file_stem}."
                )

            values = np.sum(
                nvm_mask[:, :, rep_ind],
                axis=0,
            ).astype(float)

            if show_fraction:
                values = values / float(num_kept_nvs)

            rep_counts[rep_ind] = values

        global_run_inds = (
            global_run_offset
            + np.arange(num_runs, dtype=int)
        )

        file_timestamp = _parse_file_stem_timestamp(file_stem)

        dataset_results.append(
            {
                "file_ind": int(file_ind),
                "file_stem": file_stem,
                "file_timestamp": file_timestamp,
                "dark_wait_s": wait_s,
                "num_runs": int(num_runs),
                "num_reps": int(num_reps),
                "num_nvs": int(num_nvs),
                "num_kept_nvs": int(num_kept_nvs),
                "rep_counts": rep_counts,
                "global_run_inds": global_run_inds,
                "global_run_start": int(global_run_offset),
                "global_run_stop": int(global_run_offset + num_runs),
            }
        )

        global_run_offset += num_runs

    if not dataset_results:
        raise ValueError("No matching datasets were loaded.")

    # Warn if supplied filenames are not chronological.
    parsed_times = [
        dataset["file_timestamp"]
        for dataset in dataset_results
    ]
    finite_times = [
        timestamp
        for timestamp in parsed_times
        if timestamp is not None
    ]
    if (
        verbose
        and len(finite_times) == len(parsed_times)
        and any(
            parsed_times[ind] > parsed_times[ind + 1]
            for ind in range(len(parsed_times) - 1)
        )
    ):
        print(
            "Warning: FILE_STEMS are not in chronological order. "
            "The back-to-back plot preserves the supplied order."
        )

    figures = {}

    # ------------------------------------------------------------------
    # Back-to-back mode: one continuous figure per rep.
    # ------------------------------------------------------------------
    if back_to_back:
        all_global_runs = np.concatenate(
            [
                dataset["global_run_inds"]
                for dataset in dataset_results
            ]
        )

        all_rep_values = {
            rep_ind: np.concatenate(
                [
                    np.asarray(
                        dataset["rep_counts"][rep_ind],
                        dtype=float,
                    )
                    for dataset in dataset_results
                ]
            )
            for rep_ind in rep_inds
        }

        ylabel = (
            "NV$^-$ fraction"
            if show_fraction
            else "Number of NV$^-$"
        )

        for rep_ind in rep_inds:
            fig, ax = plt.subplots(
                figsize=(12.5, 4.8)
            )

            values = all_rep_values[rep_ind]

            ax.plot(
                all_global_runs,
                values,
                "o-",
                markersize=4,
                linewidth=1.2,
                label=rep_labels[rep_ind],
            )

            # Per-file mean segments make slow changes easier to see.
            for dataset in dataset_results:
                x = dataset["global_run_inds"]
                y = np.asarray(
                    dataset["rep_counts"][rep_ind],
                    dtype=float,
                )
                ax.hlines(
                    np.nanmean(y),
                    x[0] - 0.35,
                    x[-1] + 0.35,
                    linestyles="--",
                    linewidth=1.0,
                    color="0.25",
                    alpha=0.55,
                )

            if mark_file_boundaries:
                _draw_file_boundaries(
                    ax,
                    dataset_results,
                    annotate=True,
                )

            ax.set_xlabel("Global run index")
            ax.set_ylabel(ylabel)
            ax.set_title(
                "Back-to-back NV$^-$ population: "
                f"{rep_labels[rep_ind]}"
            )
            ax.grid(True, alpha=0.25)
            ax.legend()

            if show_fraction:
                ax.set_ylim(0.0, 1.02)

            fig.tight_layout()
            figures[rep_ind] = fig

        # --------------------------------------------------------------
        # Overlay rep comparison.
        # --------------------------------------------------------------
        if show_rep_comparison:
            fig, ax = plt.subplots(
                figsize=(12.5, 5.0)
            )

            for rep_ind in rep_inds:
                ax.plot(
                    all_global_runs,
                    all_rep_values[rep_ind],
                    "o-",
                    markersize=4,
                    linewidth=1.2,
                    label=rep_labels[rep_ind],
                )

            if mark_file_boundaries:
                _draw_file_boundaries(
                    ax,
                    dataset_results,
                    annotate=True,
                )

            ax.set_xlabel("Global run index")
            ax.set_ylabel(ylabel)
            ax.set_title(
                "Back-to-back immediate vs delayed NV$^-$ population"
            )
            ax.grid(True, alpha=0.25)
            ax.legend()

            if show_fraction:
                ax.set_ylim(0.0, 1.02)

            fig.tight_layout()
            figures["comparison"] = fig

        # --------------------------------------------------------------
        # Two-rep derived quantities.
        # --------------------------------------------------------------
        if len(rep_inds) == 2:
            rep_initial, rep_final = rep_inds
            initial = all_rep_values[rep_initial]
            final = all_rep_values[rep_final]

            difference = final - initial

            if show_difference:
                fig, ax = plt.subplots(
                    figsize=(12.5, 4.8)
                )
                ax.axhline(
                    0.0,
                    color="0.25",
                    linestyle="--",
                    linewidth=1.0,
                )
                ax.plot(
                    all_global_runs,
                    difference,
                    "o-",
                    markersize=4,
                    linewidth=1.2,
                )

                if mark_file_boundaries:
                    _draw_file_boundaries(
                        ax,
                        dataset_results,
                        annotate=True,
                    )

                ax.set_xlabel("Global run index")
                ax.set_ylabel(
                    "Δ NV$^-$ fraction"
                    if show_fraction
                    else "Δ number of NV$^-$"
                )
                ax.set_title(
                    f"{rep_labels[rep_final]} "
                    f"− {rep_labels[rep_initial]}"
                )
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                figures["difference"] = fig

            if show_retention:
                retention = np.full(
                    initial.shape,
                    np.nan,
                    dtype=float,
                )
                good = initial > 0
                retention[good] = (
                    final[good]
                    / initial[good]
                )

                fig, ax = plt.subplots(
                    figsize=(12.5, 4.8)
                )
                ax.axhline(
                    1.0,
                    color="0.25",
                    linestyle="--",
                    linewidth=1.0,
                )
                ax.plot(
                    all_global_runs,
                    retention,
                    "o-",
                    markersize=4,
                    linewidth=1.2,
                )

                if mark_file_boundaries:
                    _draw_file_boundaries(
                        ax,
                        dataset_results,
                        annotate=True,
                    )

                ax.set_xlabel("Global run index")
                ax.set_ylabel(
                    f"rep {rep_final} / rep {rep_initial}"
                )
                ax.set_title(
                    "Back-to-back NV$^-$ retention"
                )
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                figures["retention"] = fig

        # Save useful concatenated arrays in a summary entry.
        for dataset in dataset_results:
            dataset["back_to_back"] = True

    # ------------------------------------------------------------------
    # Legacy mode: one subplot per file, one figure per rep.
    # ------------------------------------------------------------------
    else:
        num_panels = len(dataset_results)
        ncols_use = min(
            max(1, int(ncols)),
            num_panels,
        )
        nrows = int(
            np.ceil(num_panels / ncols_use)
        )

        for rep_ind in rep_inds:
            fig, axes = plt.subplots(
                nrows,
                ncols_use,
                figsize=(
                    5.2 * ncols_use,
                    3.8 * nrows,
                ),
                sharey=True,
            )
            axes = np.atleast_1d(
                axes
            ).ravel()

            for panel_ind, dataset in enumerate(dataset_results):
                ax = axes[panel_ind]
                yvals = np.asarray(
                    dataset["rep_counts"][rep_ind],
                    dtype=float,
                )
                run_inds = np.arange(
                    dataset["num_runs"]
                )

                ax.bar(
                    run_inds,
                    yvals,
                    alpha=0.75,
                )

                mean_val = float(
                    np.nanmean(yvals)
                )
                std_val = (
                    float(np.nanstd(yvals, ddof=1))
                    if len(yvals) > 1
                    else 0.0
                )

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
                ax.set_ylabel(
                    "NV$^-$ fraction"
                    if show_fraction
                    else "Number of NV$^-$"
                )
                ax.grid(
                    True,
                    axis="y",
                    alpha=0.25,
                )

                if show_fraction:
                    ax.set_ylim(0.0, 1.02)

            for ax in axes[num_panels:]:
                ax.axis("off")

            fig.suptitle(
                "Run-by-run NV$^-$ population: "
                f"{rep_labels[rep_ind]}",
                fontsize=14,
            )
            fig.tight_layout(
                rect=[0, 0, 1, 0.94]
            )
            figures[rep_ind] = fig

    # ------------------------------------------------------------------
    # Text summary.
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 90)
        print("BACK-TO-BACK NV- POPULATION SUMMARY")
        print("=" * 90)

        for dataset_ind, dataset in enumerate(dataset_results):
            timestamp_label = _format_file_boundary_label(
                dataset["file_stem"],
                dataset_ind,
            ).replace("\n", " ")

            print(
                f"\nFile {dataset_ind + 1}: {timestamp_label} | "
                f"wait={dataset['dark_wait_s']:g} s | "
                f"global runs "
                f"{dataset['global_run_start']}..."
                f"{dataset['global_run_stop'] - 1}"
            )

            for rep_ind in rep_inds:
                values = np.asarray(
                    dataset["rep_counts"][rep_ind],
                    dtype=float,
                )
                print(
                    f"  {rep_labels[rep_ind]}: "
                    f"mean={np.nanmean(values):.3f}, "
                    f"std="
                    f"{np.nanstd(values, ddof=1) if len(values) > 1 else 0.0:.3f}, "
                    f"min={np.nanmin(values):.3f}, "
                    f"max={np.nanmax(values):.3f}"
                )

        if len(rep_inds) == 2:
            rep_initial, rep_final = rep_inds
            initial = np.concatenate(
                [
                    np.asarray(
                        dataset["rep_counts"][rep_initial],
                        dtype=float,
                    )
                    for dataset in dataset_results
                ]
            )
            final = np.concatenate(
                [
                    np.asarray(
                        dataset["rep_counts"][rep_final],
                        dtype=float,
                    )
                    for dataset in dataset_results
                ]
            )

            difference = final - initial
            good = initial > 0
            retention = np.full(
                initial.shape,
                np.nan,
                dtype=float,
            )
            retention[good] = final[good] / initial[good]

            print("\nCombined:")
            print(
                f"  total runs = {len(initial)}"
            )
            print(
                f"  mean Δ(rep {rep_final} - rep {rep_initial}) = "
                f"{np.nanmean(difference):.3f}"
            )
            print(
                f"  median retention rep {rep_final}/rep {rep_initial} = "
                f"{np.nanmedian(retention):.4f}"
            )

    return dataset_results, figures


def _estimate_integer_shift_2d(
    img_ref: np.ndarray,
    img_target: np.ndarray,
) -> Tuple[float, float]:
    """
    Estimate approximate global shift between two images.

    Returns
    -------
    dx, dy : float
        Shift that approximately maps img_ref -> img_target.
    """
    a = np.asarray(img_ref, dtype=float)
    b = np.asarray(img_target, dtype=float)

    a = np.nan_to_num(a - np.nanmean(a), nan=0.0)
    b = np.nan_to_num(b - np.nanmean(b), nan=0.0)

    corr = fftconvolve(
        b,
        a[::-1, ::-1],
        mode="same",
    )

    peak = np.unravel_index(np.argmax(corr), corr.shape)
    center = np.array(corr.shape) // 2
    dy, dx = np.array(peak) - center

    return float(dx), float(dy)


def _draw_nv_circles(
    ax,
    coords_xy: np.ndarray,
    nv_inds: Sequence[int],
    radius_px: float = 4.0,
    color: str = "w",
    linewidth: float = 0.8,
    linestyle: str = "-",
    alpha: float = 0.9,
):
    """Draw circles around specified NV indices."""
    for nv_ind in nv_inds:
        x, y = coords_xy[nv_ind]
        circ = Circle(
            (x, y),
            radius=radius_px,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.add_patch(circ)


def plot_rep11_rep12_images_for_run(
    file_stem: str,
    run_ind: int,
    rep_inds: Tuple[int, int] = (11, 12),
    img_coords=None,
    exclude_nv_inds: Optional[Sequence[int]] = None,
    selected_nv_inds: Optional[Sequence[int]] = None,
    show_all_nv_circles: bool = False,
    show_shifted_overlay: bool = True,
    circle_radius_px: float = 4.0,
    clim: Optional[Tuple[float, float]] = None,
    diff_clim: Optional[Tuple[float, float]] = None,
    verbose: bool = True,
):
    """
    Inspect one run by comparing rep 11 and rep 12 images.

    rep 11:
        image immediately after initialization

    rep 12:
        image after dark wait

    This is useful for checking whether an apparent transition may instead be
    explained by image drift or another technical artifact.
    """

    if exclude_nv_inds is None:
        exclude_nv_inds = np.array([], dtype=int)
    else:
        exclude_nv_inds = np.unique(np.asarray(exclude_nv_inds, dtype=int))

    if selected_nv_inds is not None:
        selected_nv_inds = np.unique(np.asarray(selected_nv_inds, dtype=int))

    rep_a, rep_b = map(int, rep_inds)

    raw_data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
    )

    if "img_arrays" not in raw_data:
        raise ValueError("raw_data does not contain img_arrays.")

    img_arrays = np.asarray(raw_data["img_arrays"], dtype=float)
    counts_all = np.asarray(raw_data["counts"], dtype=float)

    if img_arrays.ndim != 6:
        raise ValueError(
            f"Expected img_arrays[exp, run, step, rep, y, x], got {img_arrays.shape}"
        )

    if counts_all.ndim != 5:
        raise ValueError(
            f"Expected counts[exp, nv, run, step, rep], got {counts_all.shape}"
        )

    # counts -> [nv, run, rep]
    counts = counts_all[0, :, :, 0, :]
    num_nvs, num_runs, num_reps = counts.shape

    if not (0 <= run_ind < num_runs):
        raise ValueError(f"run_ind={run_ind} outside [0, {num_runs - 1}]")

    for rep_ind in (rep_a, rep_b):
        if not (0 <= rep_ind < num_reps):
            raise ValueError(
                f"rep_ind={rep_ind} outside [0, {num_reps - 1}]"
            )

    if "analysis_thresholds" in raw_data:
        thresholds = np.asarray(raw_data["analysis_thresholds"], dtype=float)
    elif "thresholds" in raw_data:
        thresholds = np.asarray(raw_data["thresholds"], dtype=float)
    else:
        raise ValueError("No thresholds found in raw_data.")

    if thresholds.shape != (num_nvs,):
        raise ValueError(
            f"Threshold shape mismatch: {thresholds.shape} vs {(num_nvs,)}"
        )

    nv_list = raw_data["nv_list"]

    # Use your existing helper if it already exists in this module.
    coords_xy = _coerce_img_coords(
        nv_list,
        img_coords=img_coords,
    )

    if coords_xy.shape[0] != num_nvs:
        raise ValueError(
            f"Coordinate count mismatch: {coords_xy.shape[0]} vs {num_nvs}"
        )

    # Build valid NV mask
    keep_mask = np.ones(num_nvs, dtype=bool)
    valid_excluded = exclude_nv_inds[
        (exclude_nv_inds >= 0) & (exclude_nv_inds < num_nvs)
    ]
    keep_mask[valid_excluded] = False

    kept_nv_inds = np.where(keep_mask)[0]

    # Images: [y, x]
    img_a = img_arrays[0, run_ind, 0, rep_a, :, :]
    img_b = img_arrays[0, run_ind, 0, rep_b, :, :]
    diff_img = img_b - img_a

    # Classification at rep 11 and rep 12
    counts_a = counts[:, run_ind, rep_a]
    counts_b = counts[:, run_ind, rep_b]

    state_a = counts_a > thresholds
    state_b = counts_b > thresholds

    state_a = state_a & keep_mask
    state_b = state_b & keep_mask

    lost_mask = state_a & (~state_b)
    gained_mask = (~state_a) & state_b
    retained_mask = state_a & state_b

    lost_inds = np.where(lost_mask)[0]
    gained_inds = np.where(gained_mask)[0]
    retained_inds = np.where(retained_mask)[0]

    # Restrict overlay to selected NVs if requested
    if selected_nv_inds is not None:
        selected_nv_inds = selected_nv_inds[
            (selected_nv_inds >= 0) & (selected_nv_inds < num_nvs)
        ]

        sel_mask = np.zeros(num_nvs, dtype=bool)
        sel_mask[selected_nv_inds] = True

        lost_inds_plot = np.where(lost_mask & sel_mask)[0]
        gained_inds_plot = np.where(gained_mask & sel_mask)[0]
        retained_inds_plot = np.where(retained_mask & sel_mask)[0]
        background_inds_plot = selected_nv_inds
    else:
        lost_inds_plot = lost_inds
        gained_inds_plot = gained_inds
        retained_inds_plot = retained_inds
        background_inds_plot = kept_nv_inds

    # Estimate simple global shift
    # dx, dy = _estimate_integer_shift_2d(img_a, img_b)
    dx, dy, drift_details = estimate_drift_from_bright_nvs(
        img11=img_a,
        img12=img_b,
        coords_xy=coords_xy,
        counts11=counts_a,
        thresholds=thresholds,
        roi_radius=5,
        threshold_margin=5.0,
    )

    if clim is None:
        combined = np.stack([img_a, img_b], axis=0)
        vmin = float(np.nanpercentile(combined, 50))
        vmax = float(np.nanpercentile(combined, 99.8))
        clim_use = (vmin, vmax)
    else:
        clim_use = clim

    if diff_clim is None:
        abs_lim = float(np.nanpercentile(np.abs(diff_img), 99.5))
        diff_clim_use = (-abs_lim, abs_lim)
    else:
        diff_clim_use = diff_clim

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.3),
    )

    ax_a, ax_b, ax_diff = axes

    im_a = ax_a.imshow(
        img_a,
        origin="upper",
        vmin=clim_use[0],
        vmax=clim_use[1],
    )
    im_b = ax_b.imshow(
        img_b,
        origin="upper",
        vmin=clim_use[0],
        vmax=clim_use[1],
    )
    im_diff = ax_diff.imshow(
        diff_img,
        origin="upper",
        cmap="RdBu_r",
        vmin=diff_clim_use[0],
        vmax=diff_clim_use[1],
    )

    # Optionally show all kept NVs in gray
    if show_all_nv_circles:
        _draw_nv_circles(
            ax_a,
            coords_xy,
            background_inds_plot,
            radius_px=circle_radius_px,
            color="0.7",
            linewidth=0.5,
            alpha=0.7,
        )
        _draw_nv_circles(
            ax_b,
            coords_xy,
            background_inds_plot,
            radius_px=circle_radius_px,
            color="0.7",
            linewidth=0.5,
            alpha=0.7,
        )

    # Overlay changed states
    _draw_nv_circles(
        ax_a,
        coords_xy,
        lost_inds_plot,
        radius_px=circle_radius_px,
        color="red",
        linewidth=1.2,
    )
    _draw_nv_circles(
        ax_b,
        coords_xy,
        lost_inds_plot,
        radius_px=circle_radius_px,
        color="red",
        linewidth=1.2,
    )

    _draw_nv_circles(
        ax_a,
        coords_xy,
        gained_inds_plot,
        radius_px=circle_radius_px,
        color="cyan",
        linewidth=1.2,
    )
    _draw_nv_circles(
        ax_b,
        coords_xy,
        gained_inds_plot,
        radius_px=circle_radius_px,
        color="cyan",
        linewidth=1.2,
    )

    # Optional shifted overlay on rep 12 panel for quick drift check
    if show_shifted_overlay:
        shifted_coords = coords_xy.copy()
        shifted_coords[:, 0] = shifted_coords[:, 0] + dx
        shifted_coords[:, 1] = shifted_coords[:, 1] + dy

        _draw_nv_circles(
            ax_b,
            shifted_coords,
            background_inds_plot if show_all_nv_circles else lost_inds_plot,
            radius_px=circle_radius_px,
            color="yellow",
            linewidth=0.9,
            linestyle="--",
            alpha=0.9,
        )

    ax_a.set_title(f"rep {rep_a} (run {run_ind})")
    ax_b.set_title(
        f"rep {rep_b} (run {run_ind})\n"
        f"estimated shift from rep {rep_a}: dx={dx:.1f}, dy={dy:.1f} px"
    )
    ax_diff.set_title(f"rep {rep_b} - rep {rep_a}")

    for ax in axes:
        ax.set_axis_off()

    cbar1 = fig.colorbar(
        im_a,
        ax=[ax_a, ax_b],
        fraction=0.030,
        pad=0.02,
    )
    cbar1.set_label("photons")

    cbar2 = fig.colorbar(
        im_diff,
        ax=ax_diff,
        fraction=0.046,
        pad=0.04,
    )
    cbar2.set_label("Δ photons")

    summary_text = (
        f"kept NVs = {np.sum(keep_mask)} / {num_nvs}\n"
        f"rep {rep_a} NV- = {int(np.sum(state_a))}\n"
        f"rep {rep_b} NV- = {int(np.sum(state_b))}\n"
        f"lost = {len(lost_inds)}\n"
        f"gained = {len(gained_inds)}\n"
        f"retained = {len(retained_inds)}"
    )

    fig.text(
        0.50,
        0.02,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.suptitle(
        f"Rep {rep_a} vs rep {rep_b} image comparison | wait = {raw_data['dark_wait_s']:g} s | run = {run_ind}",
        fontsize=14,
    )

    result = {
        "file_stem": file_stem,
        "dark_wait_s": float(raw_data["dark_wait_s"]),
        "run_ind": int(run_ind),
        "rep_inds": (rep_a, rep_b),
        "estimated_shift_dx_px": dx,
        "estimated_shift_dy_px": dy,
        "lost_inds": lost_inds.tolist(),
        "gained_inds": gained_inds.tolist(),
        "retained_inds": retained_inds.tolist(),
        "num_kept_nvs": int(np.sum(keep_mask)),
    }

    if verbose:
        print("\n" + "=" * 80)
        print("REP-TO-REP IMAGE COMPARISON")
        print("=" * 80)
        print("file:", file_stem)
        print("wait_s:", raw_data["dark_wait_s"])
        print("run:", run_ind)
        print(f"rep {rep_a} NV-:", int(np.sum(state_a)))
        print(f"rep {rep_b} NV-:", int(np.sum(state_b)))
        print("lost:", len(lost_inds), lost_inds.tolist())
        print("gained:", len(gained_inds), gained_inds.tolist())
        print("estimated shift (dx, dy):", (dx, dy))

    return result, fig


def _diagnostic_get_coords_xy(raw_data, img_coords=None):
    """
    Return NV camera coordinates as [nv, 2] with columns [x, y].

    If img_coords is not explicitly supplied, this uses your existing
    _coerce_img_coords(...) helper.
    """

    if img_coords is not None:
        coords_xy = np.asarray(img_coords, dtype=float)

    else:
        if "nv_list" not in raw_data:
            raise ValueError(
                "No nv_list found and img_coords was not provided."
            )

        try:
            coords_xy = _coerce_img_coords(
                raw_data["nv_list"],
                img_coords=None,
            )
        except Exception as exc:
            raise ValueError(
                "Could not extract NV camera coordinates. "
                "Pass img_coords explicitly or use your existing "
                "_coerce_img_coords() helper."
            ) from exc

        coords_xy = np.asarray(coords_xy, dtype=float)

    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError(
            f"Expected coords_xy shape [nv,2], got {coords_xy.shape}"
        )

    return coords_xy


def integrate_nv_counts_from_image(
    img,
    coords_xy,
    radius_px=3.0,
    bg_inner_px=5.0,
    bg_outer_px=8.0,
):
    """
    Integrate each NV directly from the raw camera image.

    Signal:
        circular aperture centered on known NV (x,y).

    Background:
        median pixel value in an annulus around the NV.

    Returns
    -------
    net_counts : [nv]
        Background-subtracted integrated counts.

    raw_counts : [nv]
        Raw aperture sums.

    backgrounds : [nv]
        Estimated local background per pixel.

    num_signal_pixels : [nv]
        Number of pixels in each signal aperture.
    """

    img = np.asarray(img, dtype=float)
    coords_xy = np.asarray(coords_xy, dtype=float)

    num_nvs = len(coords_xy)

    net_counts = np.full(num_nvs, np.nan)
    raw_counts = np.full(num_nvs, np.nan)
    backgrounds = np.full(num_nvs, np.nan)
    num_signal_pixels = np.zeros(num_nvs, dtype=int)

    bounding_radius = int(np.ceil(bg_outer_px))

    for nv_ind, (x0, y0) in enumerate(coords_xy):

        if not np.isfinite(x0) or not np.isfinite(y0):
            continue

        # Integer ROI boundaries only.
        xc = int(round(float(x0)))
        yc = int(round(float(y0)))

        x_min = max(0, xc - bounding_radius)
        x_max = min(
            int(img.shape[1]),
            xc + bounding_radius + 1,
        )

        y_min = max(0, yc - bounding_radius)
        y_max = min(
            int(img.shape[0]),
            yc + bounding_radius + 1,
        )

        if x_min >= x_max or y_min >= y_max:
            continue

        patch = img[
            y_min:y_max,
            x_min:x_max,
        ]

        yy, xx = np.indices(
            patch.shape,
            dtype=float,
        )

        xx_global = xx + x_min
        yy_global = yy + y_min

        # Distance from TRUE floating-point NV coordinate.
        rr = np.sqrt(
            (xx_global - float(x0)) ** 2
            + (yy_global - float(y0)) ** 2
        )

        signal_mask = (
            rr <= float(radius_px)
        )

        bg_mask = (
            (rr >= float(bg_inner_px))
            & (rr <= float(bg_outer_px))
        )

        n_signal = int(
            np.sum(signal_mask)
        )

        if (
            n_signal == 0
            or np.sum(bg_mask) == 0
        ):
            continue

        raw_signal = float(
            np.sum(
                patch[signal_mask]
            )
        )

        # Median is intentionally used rather than mean:
        # more robust if another NV contributes to the annulus.
        bg_per_pixel = float(
            np.median(
                patch[bg_mask]
            )
        )

        net_signal = (
            raw_signal
            - bg_per_pixel * n_signal
        )

        raw_counts[nv_ind] = raw_signal
        backgrounds[nv_ind] = bg_per_pixel
        net_counts[nv_ind] = net_signal
        num_signal_pixels[nv_ind] = n_signal

    return (
        net_counts,
        raw_counts,
        backgrounds,
        num_signal_pixels,
    )


def _build_aperture_response_map(
    img,
    radius_px=3.0,
    bg_inner_px=5.0,
    bg_outer_px=8.0,
):
    """
    Build a map where the value at pixel (x,y) is approximately the
    background-subtracted signal inside a circular aperture centered there.

    This makes the global drift scan fast.
    """

    img = np.asarray(
        img,
        dtype=float,
    )

    r = int(
        np.ceil(bg_outer_px)
    )

    yy, xx = np.mgrid[
        -r:r + 1,
        -r:r + 1,
    ]

    rr = np.sqrt(
        xx ** 2
        + yy ** 2
    )

    signal_kernel = (
        rr <= radius_px
    ).astype(float)

    bg_kernel = (
        (rr >= bg_inner_px)
        & (rr <= bg_outer_px)
    ).astype(float)

    ones = np.ones_like(
        img,
        dtype=float,
    )

    # Signal aperture sum.
    signal_sum = convolve(
        img,
        signal_kernel,
        mode="constant",
        cval=0.0,
    )

    signal_norm = convolve(
        ones,
        signal_kernel,
        mode="constant",
        cval=0.0,
    )

    # Background-annulus sum.
    bg_sum = convolve(
        img,
        bg_kernel,
        mode="constant",
        cval=0.0,
    )

    bg_norm = convolve(
        ones,
        bg_kernel,
        mode="constant",
        cval=0.0,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        bg_mean = (
            bg_sum
            / bg_norm
        )

    net_map = (
        signal_sum
        - bg_mean * signal_norm
    )

    return net_map


def _sample_map_at_xy(
    image,
    coords_xy,
):
    """
    Bilinear sample image at coordinates [x,y].
    """

    coords_xy = np.asarray(
        coords_xy,
        dtype=float,
    )

    xs = coords_xy[:, 0]
    ys = coords_xy[:, 1]

    values = map_coordinates(
        np.asarray(image, dtype=float),
        [ys, xs],
        order=1,
        mode="constant",
        cval=np.nan,
    )

    return values


def _scan_coordinate_shift(
    aperture_map,
    coords_xy,
    reference_inds,
    shift_range_px=2.0,
    shift_step_px=0.1,
):
    """
    Find global coordinate shift that maximizes aggregate NV signal.

    This is NOT whole-image correlation.

    It only evaluates signal at the known positions of selected bright NVs.
    """

    coords_xy = np.asarray(
        coords_xy,
        dtype=float,
    )

    reference_inds = np.asarray(
        reference_inds,
        dtype=int,
    )

    if len(reference_inds) < 3:
        return {
            "dx": np.nan,
            "dy": np.nan,
            "score": np.nan,
            "dx_values": np.array([]),
            "dy_values": np.array([]),
            "score_map": np.empty((0, 0)),
        }

    reference_coords = (
        coords_xy[reference_inds]
    )

    dx_values = np.arange(
        -shift_range_px,
        shift_range_px
        + 0.5 * shift_step_px,
        shift_step_px,
    )

    dy_values = np.arange(
        -shift_range_px,
        shift_range_px
        + 0.5 * shift_step_px,
        shift_step_px,
    )

    score_map = np.full(
        (
            len(dy_values),
            len(dx_values),
        ),
        np.nan,
    )

    for iy, dy in enumerate(
        dy_values
    ):

        for ix, dx in enumerate(
            dx_values
        ):

            shifted_coords = (
                reference_coords.copy()
            )

            shifted_coords[:, 0] += dx
            shifted_coords[:, 1] += dy

            vals = _sample_map_at_xy(
                aperture_map,
                shifted_coords,
            )

            vals = vals[
                np.isfinite(vals)
            ]

            if len(vals) < 3:
                continue

            # Sum makes use of the complete ensemble.
            #
            # Missing/dark NVs simply contribute less signal,
            # while the surviving bright NVs still determine alignment.
            score_map[iy, ix] = (
                np.sum(vals)
            )

    if not np.any(
        np.isfinite(score_map)
    ):
        return {
            "dx": np.nan,
            "dy": np.nan,
            "score": np.nan,
            "dx_values": dx_values,
            "dy_values": dy_values,
            "score_map": score_map,
        }

    best_flat = np.nanargmax(
        score_map
    )

    best_iy, best_ix = (
        np.unravel_index(
            best_flat,
            score_map.shape,
        )
    )

    return {
        "dx": float(
            dx_values[best_ix]
        ),
        "dy": float(
            dy_values[best_iy]
        ),
        "score": float(
            score_map[
                best_iy,
                best_ix,
            ]
        ),
        "dx_values": dx_values,
        "dy_values": dy_values,
        "score_map": score_map,
    }


def estimate_nv_coordinate_drift(
    img11,
    img12,
    coords_xy,
    reference_inds,
    radius_px=3.0,
    bg_inner_px=5.0,
    bg_outer_px=8.0,
    shift_range_px=2.0,
    shift_step_px=0.1,
):
    """
    Estimate drift using ONLY known bright-NV positions.

    Important:
    We independently find the best coordinate shift for rep11 and rep12.

    Therefore:

        relative drift = best_shift_rep12 - best_shift_rep11

    rather than assuming stored coordinates are perfectly centered in rep11.
    """

    map11 = _build_aperture_response_map(
        img11,
        radius_px=radius_px,
        bg_inner_px=bg_inner_px,
        bg_outer_px=bg_outer_px,
    )

    map12 = _build_aperture_response_map(
        img12,
        radius_px=radius_px,
        bg_inner_px=bg_inner_px,
        bg_outer_px=bg_outer_px,
    )

    fit11 = _scan_coordinate_shift(
        map11,
        coords_xy,
        reference_inds,
        shift_range_px=shift_range_px,
        shift_step_px=shift_step_px,
    )

    fit12 = _scan_coordinate_shift(
        map12,
        coords_xy,
        reference_inds,
        shift_range_px=shift_range_px,
        shift_step_px=shift_step_px,
    )

    dx = (
        fit12["dx"]
        - fit11["dx"]
    )

    dy = (
        fit12["dy"]
        - fit11["dy"]
    )

    return {
        "dx_px": float(dx),
        "dy_px": float(dy),
        "rep11_shift_dx_px": fit11["dx"],
        "rep11_shift_dy_px": fit11["dy"],
        "rep12_shift_dx_px": fit12["dx"],
        "rep12_shift_dy_px": fit12["dy"],
        "rep11_scan": fit11,
        "rep12_scan": fit12,
        "num_reference_nvs": int(
            len(reference_inds)
        ),
    }


def diagnose_run(
    file_stem,
    run_ind,
    rep_initial=11,
    rep_final=12,
    img_coords=None,
    exclude_nv_inds=None,

    # Classification
    bright_margin_counts=5.0,

    # Raw-image aperture
    aperture_radius_px=3.0,
    bg_inner_px=5.0,
    bg_outer_px=8.0,

    # Drift
    drift_range_px=2.0,
    drift_step_px=0.1,
    max_drift_nvs=250,

    show_images=True,
    verbose=True,
):
    """
    Comprehensive diagnostic for one suspicious particle-memory run.

    Tests:
        1. NV- -> NV0 loss fraction.
        2. Loss anomaly relative to all other runs.
        3. Rep11 vs rep12 threshold-centered distribution.
        4. Raw camera-image integration at every known NV coordinate.
        5. Rep12/rep11 raw-image intensity ratios.
        6. Global linear brightness scaling.
        7. Each NV compared with its own history.
        8. Whole-camera intensity behavior.
        9. Global mechanical/image drift using bright NV coordinates only.
       10. Spatial distribution of lost NVs.

    Returns
    -------
    result : dict
    figures : dict
    """

    # =====================================================================
    # Load raw dataset
    # =====================================================================

    raw_data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
    )

    counts_all = np.asarray(
        raw_data["counts"],
        dtype=float,
    )

    if counts_all.ndim != 5:
        raise ValueError(
            "Expected counts[exp,nv,run,step,rep], "
            f"got {counts_all.shape}"
        )

    counts = (
        counts_all[
            0,
            :,
            :,
            0,
            :,
        ]
    )

    num_nvs, num_runs, num_reps = (
        counts.shape
    )

    if not (
        0 <= run_ind < num_runs
    ):
        raise ValueError(
            f"run_ind={run_ind}; valid range "
            f"is 0...{num_runs - 1}"
        )

    if not (
        0 <= rep_initial < num_reps
    ):
        raise ValueError(
            f"rep_initial={rep_initial} invalid"
        )

    if not (
        0 <= rep_final < num_reps
    ):
        raise ValueError(
            f"rep_final={rep_final} invalid"
        )

    # =====================================================================
    # Analysis thresholds
    # =====================================================================

    if "analysis_thresholds" in raw_data:

        thresholds = np.asarray(
            raw_data[
                "analysis_thresholds"
            ],
            dtype=float,
        )

    elif "thresholds" in raw_data:

        thresholds = np.asarray(
            raw_data["thresholds"],
            dtype=float,
        )

    else:
        raise ValueError(
            "No thresholds found."
        )

    if thresholds.shape != (
        num_nvs,
    ):
        raise ValueError(
            f"Threshold shape={thresholds.shape}, "
            f"expected {(num_nvs,)}"
        )

    # =====================================================================
    # Camera coordinates
    # =====================================================================

    coords_xy_all = (
        _diagnostic_get_coords_xy(
            raw_data,
            img_coords=img_coords,
        )
    )

    if len(coords_xy_all) != num_nvs:
        raise ValueError(
            f"Found {len(coords_xy_all)} coordinates "
            f"but {num_nvs} NVs."
        )

    # =====================================================================
    # Exclusion mask
    # =====================================================================

    keep = np.ones(
        num_nvs,
        dtype=bool,
    )

    if exclude_nv_inds is not None:

        excluded = np.asarray(
            exclude_nv_inds,
            dtype=int,
        )

        excluded = excluded[
            (excluded >= 0)
            & (excluded < num_nvs)
        ]

        keep[excluded] = False

    original_nv_inds = (
        np.arange(num_nvs)[keep]
    )

    counts = counts[keep]
    thresholds = thresholds[keep]
    coords_xy = coords_xy_all[keep]

    num_kept = len(
        thresholds
    )

    # =====================================================================
    # Extracted count classification for all runs
    # =====================================================================

    c11_all = (
        counts[
            :,
            :,
            rep_initial,
        ]
    )

    c12_all = (
        counts[
            :,
            :,
            rep_final,
        ]
    )

    state11_all = (
        c11_all
        > thresholds[:, None]
    )

    state12_all = (
        c12_all
        > thresholds[:, None]
    )

    lost_all = (
        state11_all
        & (~state12_all)
    )

    gained_all = (
        (~state11_all)
        & state12_all
    )

    retained_all = (
        state11_all
        & state12_all
    )

    # =====================================================================
    # Run-by-run loss
    # =====================================================================

    eligible_by_run = np.sum(
        state11_all,
        axis=0,
    )

    lost_by_run = np.sum(
        lost_all,
        axis=0,
    )

    gained_by_run = np.sum(
        gained_all,
        axis=0,
    )

    loss_fraction_by_run = (
        lost_by_run
        / np.maximum(
            eligible_by_run,
            1,
        )
    )

    # =====================================================================
    # Target run
    # =====================================================================

    c11 = c11_all[:, run_ind]
    c12 = c12_all[:, run_ind]

    state11 = (
        state11_all[:, run_ind]
    )

    state12 = (
        state12_all[:, run_ind]
    )

    lost = (
        lost_all[:, run_ind]
    )

    gained = (
        gained_all[:, run_ind]
    )

    retained = (
        retained_all[:, run_ind]
    )

    lost_inds_local = np.where(
        lost
    )[0]

    gained_inds_local = np.where(
        gained
    )[0]

    lost_inds_original = (
        original_nv_inds[
            lost_inds_local
        ]
    )

    gained_inds_original = (
        original_nv_inds[
            gained_inds_local
        ]
    )

    # =====================================================================
    # Robust run-level anomaly significance
    # =====================================================================

    other_runs = (
        np.arange(num_runs)
        != run_ind
    )

    background_loss = (
        loss_fraction_by_run[
            other_runs
        ]
    )

    bg_loss_median = float(
        np.median(
            background_loss
        )
    )

    bg_loss_mad = float(
        np.median(
            np.abs(
                background_loss
                - bg_loss_median
            )
        )
    )

    bg_loss_sigma = (
        1.4826
        * bg_loss_mad
    )

    target_loss_fraction = float(
        loss_fraction_by_run[
            run_ind
        ]
    )

    if bg_loss_sigma > 0:

        loss_robust_z = (
            target_loss_fraction
            - bg_loss_median
        ) / bg_loss_sigma

    else:
        loss_robust_z = np.inf

    empirical_p = (
        1
        + np.sum(
            background_loss
            >= target_loss_fraction
        )
    ) / (
        len(background_loss)
        + 1
    )

    # =====================================================================
    # Threshold-centered counts
    # =====================================================================

    margin11 = (
        c11 - thresholds
    )

    margin12 = (
        c12 - thresholds
    )

    # =====================================================================
    # Historical behavior of EACH NV
    # =====================================================================

    delta_all = (
        c12_all
        - c11_all
    )

    delta_target = (
        delta_all[
            :,
            run_ind,
        ]
    )

    historical_delta = np.full(
        num_kept,
        np.nan,
    )

    historical_mad = np.full(
        num_kept,
        np.nan,
    )

    nv_historical_residual = np.full(
        num_kept,
        np.nan,
    )

    nv_historical_z = np.full(
        num_kept,
        np.nan,
    )

    for nv_ind in range(
        num_kept
    ):

        valid_runs = (
            other_runs
            & state11_all[
                nv_ind
            ]
        )

        values = (
            delta_all[
                nv_ind,
                valid_runs,
            ]
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) < 3:
            continue

        med = float(
            np.median(values)
        )

        mad = float(
            np.median(
                np.abs(
                    values - med
                )
            )
        )

        historical_delta[
            nv_ind
        ] = med

        historical_mad[
            nv_ind
        ] = mad

        residual = (
            delta_target[nv_ind]
            - med
        )

        nv_historical_residual[
            nv_ind
        ] = residual

        sigma = (
            1.4826 * mad
        )

        if sigma > 0:

            nv_historical_z[
                nv_ind
            ] = (
                residual
                / sigma
            )

    # =====================================================================
    # Raw camera images
    # =====================================================================

    if "img_arrays" not in raw_data:
        raise ValueError(
            "Raw dataset does not contain img_arrays."
        )

    img_arrays = np.asarray(
        raw_data["img_arrays"],
        dtype=float,
    )

    if img_arrays.ndim != 6:
        raise ValueError(
            "Expected img_arrays"
            "[exp,run,step,rep,y,x], "
            f"got {img_arrays.shape}"
        )

    img11 = (
        img_arrays[
            0,
            run_ind,
            0,
            rep_initial,
            :,
            :,
        ]
    )

    img12 = (
        img_arrays[
            0,
            run_ind,
            0,
            rep_final,
            :,
            :,
        ]
    )

    diff_img = (
        img12 - img11
    )

    # =====================================================================
    # RAW-IMAGE aperture integration at NV coordinates
    # =====================================================================

    (
        img_net11,
        img_raw11,
        img_bg11,
        img_npix11,
    ) = integrate_nv_counts_from_image(
        img11,
        coords_xy,
        radius_px=aperture_radius_px,
        bg_inner_px=bg_inner_px,
        bg_outer_px=bg_outer_px,
    )

    (
        img_net12,
        img_raw12,
        img_bg12,
        img_npix12,
    ) = integrate_nv_counts_from_image(
        img12,
        coords_xy,
        radius_px=aperture_radius_px,
        bg_inner_px=bg_inner_px,
        bg_outer_px=bg_outer_px,
    )

    # =====================================================================
    # Raw-image NV ratios
    # =====================================================================

    img_ratio = np.full(
        num_kept,
        np.nan,
    )

    valid_img_ratio = (
        np.isfinite(img_net11)
        & np.isfinite(img_net12)
        & (img_net11 > 0)
    )

    img_ratio[
        valid_img_ratio
    ] = (
        img_net12[
            valid_img_ratio
        ]
        / img_net11[
            valid_img_ratio
        ]
    )

    # NVs confidently bright at rep11.
    bright11 = (
        c11
        > thresholds
        + bright_margin_counts
    )

    valid_bright_img = (
        bright11
        & valid_img_ratio
    )

    if np.any(
        valid_bright_img
    ):

        median_img_ratio = float(
            np.nanmedian(
                img_ratio[
                    valid_bright_img
                ]
            )
        )

    else:
        median_img_ratio = np.nan

    # =====================================================================
    # Global linear scaling:
    #
    # raw rep12 integrated signal ~= slope * raw rep11 + intercept
    # =====================================================================

    global_slope = np.nan
    global_intercept = np.nan
    raw_correlation = np.nan

    fit_mask = (
        valid_bright_img
        & np.isfinite(img_net11)
        & np.isfinite(img_net12)
    )

    if np.sum(fit_mask) >= 3:

        xfit = img_net11[
            fit_mask
        ]

        yfit = img_net12[
            fit_mask
        ]

        global_slope, global_intercept = (
            np.polyfit(
                xfit,
                yfit,
                1,
            )
        )

        if (
            np.std(xfit) > 0
            and np.std(yfit) > 0
        ):

            raw_correlation = float(
                np.corrcoef(
                    xfit,
                    yfit,
                )[0, 1]
            )

    # =====================================================================
    # Drift reference NVs
    #
    # Select confidently NV- in rep11.
    # Then use brightest raw-image NVs from that group.
    # =====================================================================

    drift_candidates = np.where(
        bright11
        & np.isfinite(img_net11)
        & (img_net11 > 0)
    )[0]

    if len(drift_candidates) > 0:

        order = np.argsort(
            img_net11[
                drift_candidates
            ]
        )[::-1]

        drift_reference_inds = (
            drift_candidates[
                order[
                    :max_drift_nvs
                ]
            ]
        )

    else:
        drift_reference_inds = (
            np.array(
                [],
                dtype=int,
            )
        )

    drift_result = (
        estimate_nv_coordinate_drift(
            img11,
            img12,
            coords_xy,
            reference_inds=drift_reference_inds,
            radius_px=aperture_radius_px,
            bg_inner_px=bg_inner_px,
            bg_outer_px=bg_outer_px,
            shift_range_px=drift_range_px,
            shift_step_px=drift_step_px,
        )
    )

    drift_dx = (
        drift_result[
            "dx_px"
        ]
    )

    drift_dy = (
        drift_result[
            "dy_px"
        ]
    )

    drift_magnitude = float(
        np.sqrt(
            drift_dx ** 2
            + drift_dy ** 2
        )
    )

    # =====================================================================
    # Whole-camera common-mode intensity for ALL runs
    # =====================================================================

    imgs11_all = (
        img_arrays[
            0,
            :,
            0,
            rep_initial,
            :,
            :,
        ]
    )

    imgs12_all = (
        img_arrays[
            0,
            :,
            0,
            rep_final,
            :,
            :,
        ]
    )

    total_img11_by_run = np.sum(
        imgs11_all,
        axis=(-2, -1),
    )

    total_img12_by_run = np.sum(
        imgs12_all,
        axis=(-2, -1),
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        whole_image_ratio_by_run = (
            total_img12_by_run
            / total_img11_by_run
        )

    # Median camera value is useful as a crude background monitor.
    median_img11_by_run = np.median(
        imgs11_all,
        axis=(-2, -1),
    )

    median_img12_by_run = np.median(
        imgs12_all,
        axis=(-2, -1),
    )

    camera_background_change_by_run = (
        median_img12_by_run
        - median_img11_by_run
    )

    # =====================================================================
    # Extracted-count brightness ratio for all runs
    # =====================================================================

    median_count_ratio_by_run = np.full(
        num_runs,
        np.nan,
    )

    for r in range(
        num_runs
    ):

        bright_r = (
            c11_all[:, r]
            > thresholds
            + bright_margin_counts
        )

        good_r = (
            bright_r
            & np.isfinite(
                c11_all[:, r]
            )
            & np.isfinite(
                c12_all[:, r]
            )
            & (
                c11_all[:, r]
                > 0
            )
        )

        if np.sum(good_r) >= 3:

            median_count_ratio_by_run[
                r
            ] = float(
                np.median(
                    c12_all[
                        good_r,
                        r,
                    ]
                    / c11_all[
                        good_r,
                        r,
                    ]
                )
            )

    # =====================================================================
    # Figure 1: quantitative diagnostics
    # =====================================================================

    fig_diag, axes = plt.subplots(
        2,
        4,
        figsize=(19, 9.5),
    )

    axes = axes.ravel()

    run_inds = np.arange(
        num_runs
    )

    # ---------------------------------------------------------------------
    # Panel 1: loss fraction
    # ---------------------------------------------------------------------

    ax = axes[0]

    ax.bar(
        run_inds,
        100
        * loss_fraction_by_run,
        alpha=0.65,
    )

    ax.scatter(
        run_ind,
        100
        * target_loss_fraction,
        s=80,
        zorder=10,
    )

    ax.axhline(
        100
        * bg_loss_median,
        linestyle="--",
    )

    ax.set_xlabel(
        "Run index"
    )

    ax.set_ylabel(
        "NV$^-$ → NV$^0$ (%)"
    )

    ax.set_title(
        f"Charge-loss anomaly\n"
        f"run {run_ind}: "
        f"{100*target_loss_fraction:.1f}% | "
        f"z={loss_robust_z:.1f}"
    )

    # ---------------------------------------------------------------------
    # Panel 2: threshold-centered counts
    # ---------------------------------------------------------------------

    ax = axes[1]

    ax.scatter(
        margin11[retained],
        margin12[retained],
        s=10,
        alpha=0.4,
        label="NV- → NV-",
    )

    ax.scatter(
        margin11[lost],
        margin12[lost],
        s=18,
        alpha=0.75,
        label="NV- → NV0",
    )

    ax.scatter(
        margin11[gained],
        margin12[gained],
        s=18,
        alpha=0.75,
        label="NV0 → NV-",
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    finite_margin = np.concatenate(
        [
            margin11[
                np.isfinite(margin11)
            ],
            margin12[
                np.isfinite(margin12)
            ],
        ]
    )

    if len(finite_margin) > 0:

        lim = float(
            np.percentile(
                np.abs(
                    finite_margin
                ),
                99,
            )
        )

        if lim > 0:

            ax.plot(
                [-lim, lim],
                [-lim, lim],
                "--",
                alpha=0.5,
            )

            ax.set_xlim(
                -lim,
                lim,
            )

            ax.set_ylim(
                -lim,
                lim,
            )

    ax.set_xlabel(
        f"rep {rep_initial} − threshold"
    )

    ax.set_ylabel(
        f"rep {rep_final} − threshold"
    )

    ax.set_title(
        "Threshold-centered\nper-NV distribution"
    )

    ax.legend(
        fontsize=8
    )

    # ---------------------------------------------------------------------
    # Panel 3: RAW IMAGE integrated rep11 vs rep12
    # ---------------------------------------------------------------------

    ax = axes[2]

    good_scatter = (
        np.isfinite(img_net11)
        & np.isfinite(img_net12)
    )

    ax.scatter(
        img_net11[
            good_scatter
            & retained
        ],
        img_net12[
            good_scatter
            & retained
        ],
        s=10,
        alpha=0.4,
        label="retained",
    )

    ax.scatter(
        img_net11[
            good_scatter
            & lost
        ],
        img_net12[
            good_scatter
            & lost
        ],
        s=20,
        alpha=0.8,
        label="lost",
    )

    finite_values = np.concatenate(
        [
            img_net11[
                good_scatter
            ],
            img_net12[
                good_scatter
            ],
        ]
    )

    if len(finite_values) > 0:

        low = float(
            np.nanpercentile(
                finite_values,
                1,
            )
        )

        high = float(
            np.nanpercentile(
                finite_values,
                99,
            )
        )

        ax.plot(
            [low, high],
            [low, high],
            "--",
            alpha=0.5,
            label="rep12 = rep11",
        )

        if np.isfinite(
            global_slope
        ):

            xx = np.array(
                [low, high]
            )

            yy = (
                global_slope
                * xx
                + global_intercept
            )

            ax.plot(
                xx,
                yy,
                linewidth=1.5,
                label=(
                    f"fit: y={global_slope:.2f}x"
                    f"{global_intercept:+.1f}"
                ),
            )

    ax.set_xlabel(
        f"rep {rep_initial} aperture counts"
    )

    ax.set_ylabel(
        f"rep {rep_final} aperture counts"
    )

    ax.set_title(
        "RAW image integration\n"
        f"corr={raw_correlation:.3f}"
    )

    ax.legend(
        fontsize=7
    )

    # ---------------------------------------------------------------------
    # Panel 4: raw image ratio distributions
    # ---------------------------------------------------------------------

    ax = axes[3]

    ratio_retained = (
        img_ratio[
            retained
            & np.isfinite(img_ratio)
        ]
    )

    ratio_lost = (
        img_ratio[
            lost
            & np.isfinite(img_ratio)
        ]
    )

    if len(ratio_retained) > 0:

        ax.hist(
            ratio_retained,
            bins=50,
            range=(0, 1.5),
            alpha=0.55,
            density=True,
            label="retained",
        )

    if len(ratio_lost) > 0:

        ax.hist(
            ratio_lost,
            bins=50,
            range=(0, 1.5),
            alpha=0.55,
            density=True,
            label="lost",
        )

    ax.axvline(
        1,
        linestyle="--",
    )

    ax.set_xlabel(
        "raw aperture rep12 / rep11"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Per-NV brightness ratio\n"
        f"median bright={median_img_ratio:.3f}"
    )

    ax.legend()

    # ---------------------------------------------------------------------
    # Panel 5: each NV vs own history
    # ---------------------------------------------------------------------

    ax = axes[4]

    hist_good = (
        state11
        & np.isfinite(
            nv_historical_residual
        )
    )

    hist_lost = (
        lost
        & np.isfinite(
            nv_historical_residual
        )
    )

    if np.any(hist_good):

        ax.hist(
            nv_historical_residual[
                hist_good
            ],
            bins=50,
            alpha=0.55,
            label="all rep11 NV-",
        )

    if np.any(hist_lost):

        ax.hist(
            nv_historical_residual[
                hist_lost
            ],
            bins=50,
            alpha=0.65,
            label="lost",
        )

    ax.axvline(
        0,
        linestyle="--",
    )

    ax.set_xlabel(
        "Δcount(run) − median Δcount(other runs)"
    )

    ax.set_ylabel(
        "NV count"
    )

    ax.set_title(
        "Each NV vs its own history"
    )

    ax.legend()

    # ---------------------------------------------------------------------
    # Panel 6: extracted common-mode ratio across runs
    # ---------------------------------------------------------------------

    ax = axes[5]

    ax.plot(
        run_inds,
        median_count_ratio_by_run,
        ".-",
    )

    ax.scatter(
        run_ind,
        median_count_ratio_by_run[
            run_ind
        ],
        s=80,
        zorder=10,
    )

    ax.axhline(
        1,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel(
        "Run index"
    )

    ax.set_ylabel(
        "Median rep12 / rep11"
    )

    ax.set_title(
        "Common-mode NV brightness\n"
        "(extracted counts)"
    )

    # ---------------------------------------------------------------------
    # Panel 7: whole-camera ratio
    # ---------------------------------------------------------------------

    ax = axes[6]

    ax.plot(
        run_inds,
        whole_image_ratio_by_run,
        ".-",
    )

    ax.scatter(
        run_ind,
        whole_image_ratio_by_run[
            run_ind
        ],
        s=80,
        zorder=10,
    )

    ax.axhline(
        1,
        linestyle="--",
        alpha=0.5,
    )

    ax.set_xlabel(
        "Run index"
    )

    ax.set_ylabel(
        "Total image rep12 / rep11"
    )

    ax.set_title(
        "Whole-camera intensity"
    )

    # ---------------------------------------------------------------------
    # Panel 8: spatial map
    # ---------------------------------------------------------------------

    ax = axes[7]

    ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        s=8,
        alpha=0.2,
        label="all NVs",
    )

    if np.any(lost):

        sc = ax.scatter(
            coords_xy[lost, 0],
            coords_xy[lost, 1],
            c=img_ratio[lost],
            s=40,
            alpha=0.9,
            label="lost",
            vmin=0,
            vmax=1,
        )

        fig_diag.colorbar(
            sc,
            ax=ax,
            label="raw rep12 / rep11",
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Camera x"
    )

    ax.set_ylabel(
        "Camera y"
    )

    ax.set_title(
        "Spatial distribution\nof lost NVs"
    )

    ax.legend(
        fontsize=8
    )

    fig_diag.suptitle(
        f"Run {run_ind} diagnostic | "
        f"wait={raw_data.get('dark_wait_s', np.nan):g} s",
        fontsize=15,
    )

    # =====================================================================
    # Figure 2: raw images
    # =====================================================================

    figures = {
        "diagnostics": fig_diag,
    }

    if show_images:

        fig_img, image_axes = (
            plt.subplots(
                1,
                3,
                figsize=(16, 5),
            )
        )

        combined = np.concatenate(
            [
                img11.ravel(),
                img12.ravel(),
            ]
        )

        vmin = float(
            np.nanpercentile(
                combined,
                20,
            )
        )

        vmax = float(
            np.nanpercentile(
                combined,
                99.8,
            )
        )

        diff_lim = float(
            np.nanpercentile(
                np.abs(diff_img),
                99.5,
            )
        )

        image_axes[0].imshow(
            img11,
            vmin=vmin,
            vmax=vmax,
        )

        image_axes[1].imshow(
            img12,
            vmin=vmin,
            vmax=vmax,
        )

        image_axes[2].imshow(
            diff_img,
            cmap="RdBu_r",
            vmin=-diff_lim,
            vmax=diff_lim,
        )

        # Original NV positions.
        image_axes[0].scatter(
            coords_xy[:, 0],
            coords_xy[:, 1],
            s=10,
            facecolors="none",
            alpha=0.25,
        )

        image_axes[1].scatter(
            coords_xy[:, 0],
            coords_xy[:, 1],
            s=10,
            facecolors="none",
            alpha=0.25,
        )

        # Highlight the NVs classified as lost.
        for ax in image_axes[:2]:

            ax.scatter(
                coords_xy[lost, 0],
                coords_xy[lost, 1],
                s=45,
                facecolors="none",
                linewidths=0.8,
            )

        # Shifted coordinates based on measured drift.
        shifted_coords = (
            coords_xy.copy()
        )

        shifted_coords[:, 0] += (
            drift_dx
        )

        shifted_coords[:, 1] += (
            drift_dy
        )

        image_axes[1].scatter(
            shifted_coords[
                drift_reference_inds,
                0,
            ],
            shifted_coords[
                drift_reference_inds,
                1,
            ],
            s=16,
            facecolors="none",
            linewidths=0.5,
            alpha=0.4,
        )

        image_axes[0].set_title(
            f"rep {rep_initial}"
        )

        image_axes[1].set_title(
            f"rep {rep_final}\n"
            f"drift dx={drift_dx:.2f}, "
            f"dy={drift_dy:.2f} px"
        )

        image_axes[2].set_title(
            f"rep {rep_final} − rep {rep_initial}"
        )

        for ax in image_axes:
            ax.set_axis_off()

        fig_img.suptitle(
            f"Run {run_ind}: raw camera images"
        )

        fig_img.tight_layout(
            rect=[0, 0, 1, 0.94]
        )

        figures[
            "raw_images"
        ] = fig_img

    # =====================================================================
    # Figure 3: drift scans
    # =====================================================================

    scan11 = (
        drift_result[
            "rep11_scan"
        ]
    )

    scan12 = (
        drift_result[
            "rep12_scan"
        ]
    )

    if (
        scan11[
            "score_map"
        ].size > 0
        and
        scan12[
            "score_map"
        ].size > 0
    ):

        fig_drift, ax_drift = (
            plt.subplots(
                1,
                2,
                figsize=(11, 4.5),
            )
        )

        extent = [
            scan11[
                "dx_values"
            ][0],
            scan11[
                "dx_values"
            ][-1],
            scan11[
                "dy_values"
            ][0],
            scan11[
                "dy_values"
            ][-1],
        ]

        ax_drift[0].imshow(
            scan11[
                "score_map"
            ],
            origin="lower",
            extent=extent,
            aspect="auto",
        )

        ax_drift[0].scatter(
            scan11["dx"],
            scan11["dy"],
            marker="x",
            s=80,
        )

        ax_drift[0].set_title(
            f"rep {rep_initial} coordinate scan"
        )

        ax_drift[1].imshow(
            scan12[
                "score_map"
            ],
            origin="lower",
            extent=extent,
            aspect="auto",
        )

        ax_drift[1].scatter(
            scan12["dx"],
            scan12["dy"],
            marker="x",
            s=80,
        )

        ax_drift[1].set_title(
            f"rep {rep_final} coordinate scan"
        )

        for ax in ax_drift:

            ax.set_xlabel(
                "Δx (px)"
            )

            ax.set_ylabel(
                "Δy (px)"
            )

        fig_drift.suptitle(
            f"Bright-NV coordinate drift | "
            f"relative Δ = "
            f"({drift_dx:.2f}, "
            f"{drift_dy:.2f}) px"
        )

        fig_drift.tight_layout(
            rect=[0, 0, 1, 0.93]
        )

        figures[
            "drift_scan"
        ] = fig_drift

    # =====================================================================
    # Package results
    # =====================================================================

    result = {
        "file_stem":
            file_stem,

        "run_ind":
            int(run_ind),

        "dark_wait_s":
            float(
                raw_data.get(
                    "dark_wait_s",
                    np.nan,
                )
            ),

        "num_nvs":
            int(num_kept),

        # Charge classification
        "rep11_nvm":
            int(
                np.sum(state11)
            ),

        "rep12_nvm":
            int(
                np.sum(state12)
            ),

        "num_lost":
            int(
                np.sum(lost)
            ),

        "num_gained":
            int(
                np.sum(gained)
            ),

        "loss_fraction":
            float(
                target_loss_fraction
            ),

        "background_loss_fraction_median":
            float(
                bg_loss_median
            ),

        "loss_robust_z":
            float(
                loss_robust_z
            ),

        "loss_empirical_p":
            float(
                empirical_p
            ),

        # Raw image aperture analysis
        "img_net11":
            img_net11,

        "img_net12":
            img_net12,

        "img_raw11":
            img_raw11,

        "img_raw12":
            img_raw12,

        "img_bg11":
            img_bg11,

        "img_bg12":
            img_bg12,

        "img_ratio":
            img_ratio,

        "median_img_ratio_bright":
            float(
                median_img_ratio
            ),

        "global_raw_slope":
            float(
                global_slope
            ),

        "global_raw_intercept":
            float(
                global_intercept
            ),

        "raw_image_nv_correlation":
            float(
                raw_correlation
            ),

        # Historical NV behavior
        "nv_delta":
            delta_target,

        "nv_historical_delta":
            historical_delta,

        "nv_historical_residual":
            nv_historical_residual,

        "nv_historical_z":
            nv_historical_z,

        # Spatial
        "coords_xy":
            coords_xy,

        "lost_nv_inds":
            lost_inds_original.tolist(),

        "gained_nv_inds":
            gained_inds_original.tolist(),

        # Drift
        "drift_dx_px":
            float(
                drift_dx
            ),

        "drift_dy_px":
            float(
                drift_dy
            ),

        "drift_magnitude_px":
            float(
                drift_magnitude
            ),

        "drift_num_reference_nvs":
            int(
                len(
                    drift_reference_inds
                )
            ),

        "drift_result":
            drift_result,

        # Common mode
        "median_count_ratio_by_run":
            median_count_ratio_by_run,

        "whole_image_ratio_by_run":
            whole_image_ratio_by_run,

        "camera_background_change_by_run":
            camera_background_change_by_run,
    }

    # =====================================================================
    # Print concise report
    # =====================================================================

    if verbose:

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"RUN {run_ind} DIAGNOSTIC"
        )

        print(
            "=" * 90
        )

        print(
            f"wait: "
            f"{result['dark_wait_s']:g} s"
        )

        print()

        print(
            f"rep {rep_initial} NV-: "
            f"{result['rep11_nvm']}"
        )

        print(
            f"rep {rep_final} NV-: "
            f"{result['rep12_nvm']}"
        )

        print(
            f"NV- -> NV0 losses: "
            f"{result['num_lost']} "
            f"({100*result['loss_fraction']:.2f}%)"
        )

        print(
            f"other-run median loss: "
            f"{100*result['background_loss_fraction_median']:.2f}%"
        )

        print(
            f"robust loss z: "
            f"{result['loss_robust_z']:.2f}"
        )

        print(
            f"empirical p: "
            f"{result['loss_empirical_p']:.4g}"
        )

        print()

        print(
            "RAW CAMERA APERTURE ANALYSIS"
        )

        print(
            f"median rep12/rep11 ratio "
            f"(bright NVs): "
            f"{result['median_img_ratio_bright']:.4f}"
        )

        print(
            f"global raw scaling slope: "
            f"{result['global_raw_slope']:.4f}"
        )

        print(
            f"raw NV correlation: "
            f"{result['raw_image_nv_correlation']:.4f}"
        )

        print()

        print(
            "DRIFT FROM BRIGHT NV COORDINATES"
        )

        print(
            f"dx = "
            f"{result['drift_dx_px']:.3f} px"
        )

        print(
            f"dy = "
            f"{result['drift_dy_px']:.3f} px"
        )

        print(
            f"|drift| = "
            f"{result['drift_magnitude_px']:.3f} px"
        )

        print(
            f"reference NVs = "
            f"{result['drift_num_reference_nvs']}"
        )

        print()

        print(
            "COMMON-MODE CAMERA"
        )

        print(
            f"whole-image rep12/rep11 = "
            f"{whole_image_ratio_by_run[run_ind]:.4f}"
        )

        print(
            f"median other runs = "
            f"{np.nanmedian(whole_image_ratio_by_run[other_runs]):.4f}"
        )

        print(
            f"camera background change = "
            f"{camera_background_change_by_run[run_ind]:.4f}"
        )

        print()

        print(
            f"lost NV indices:\n"
            f"{lost_inds_original.tolist()}"
        )

    return result, figures


if __name__ == "__main__":
    kpl.init_kplotlib()

    # ==================================================================
    # Back-to-back 3600 s datasets
    # ==================================================================
    FILE_STEMS = [
        "2026_08_13-11_33_24-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
        "2026_08_14-02_16_30-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
        "2026_08_14-14_19_56-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
        "2026_08_15-02_23_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
        "2026_08_15-14_26_42-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    ]

    SELECTED_WAITS_S = [3600]

    REP_INDS = (11, 12)
    REP_LABELS = {
        11: "rep 11: immediate final check",
        12: "rep 12: after 3600 s dark wait",
    }

    # Set this to BAD_NV_INDS if you want the same exclusion list used
    # elsewhere in your analysis.
    EXCLUDE_NV_INDS = None

    rep_stats, rep_figs = plot_nv_minus_by_run_separate_reps(
        FILE_STEMS,
        selected_waits_s=SELECTED_WAITS_S,
        rep_inds=REP_INDS,
        rep_labels=REP_LABELS,
        exclude_nv_inds=EXCLUDE_NV_INDS,
        show_fraction=False,
        verbose=True,
        back_to_back=True,
        mark_file_boundaries=True,
        show_rep_comparison=True,
        show_difference=True,
        show_retention=True,
    )

    # ==================================================================
    # Optional raw-image inspection of one local run in one file
    # ==================================================================
    INSPECT_IMAGES = True
    INSPECT_FILE_IND = 0
    INSPECT_RUN_IND = 7

    if INSPECT_IMAGES:
        image_result, image_fig = plot_rep11_rep12_images_for_run(
            file_stem=FILE_STEMS[INSPECT_FILE_IND],
            run_ind=INSPECT_RUN_IND,
            rep_inds=REP_INDS,
            exclude_nv_inds=EXCLUDE_NV_INDS,
            show_all_nv_circles=False,
            circle_radius_px=2.0,
            verbose=True,
        )

    # ==================================================================
    # Optional deeper artifact/drift diagnostic
    # ==================================================================
    RUN_DIAGNOSTIC = True

    if RUN_DIAGNOSTIC:
        diagnostic_result, diagnostic_figs = diagnose_run(
            file_stem=FILE_STEMS[INSPECT_FILE_IND],
            run_ind=INSPECT_RUN_IND,
            rep_initial=REP_INDS[0],
            rep_final=REP_INDS[1],
            exclude_nv_inds=EXCLUDE_NV_INDS,
            bright_margin_counts=5.0,
            aperture_radius_px=2.5,
            bg_inner_px=4.0,
            bg_outer_px=6.0,
            drift_range_px=2.0,
            drift_step_px=0.01,
            max_drift_nvs=250,
            show_images=True,
            verbose=True,
        )

    kpl.show(block=True)


    sys.exit()
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

    FILE_STEMS = [
     "2026_07_27-19_18_59-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
        ]
    
    FILE_STEMS = [
    "2026_08_13-11_33_24-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    "2026_08_14-02_16_30-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    "2026_08_14-14_19_56-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    "2026_08_15-02_23_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    "2026_08_15-14_26_42-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s"
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
        # 3600,
        3600,
    ]


    rep_stats, rep_figs = plot_nv_minus_by_run_separate_reps(
        FILE_STEMS,
        selected_waits_s=[3600],
        rep_inds=(11, 12),
        rep_labels={
            11: "rep 11: immediate final check",
            12: "rep 12: after 3600 s dark wait",
        },
        exclude_nv_inds=None,
        show_fraction=False,
        verbose=True,
        back_to_back=True,
        mark_file_boundaries=True,
        show_rep_comparison=True,
        show_difference=True,
        show_retention=True,
    )
    # rep_stats, rep_figs = plot_nv_minus_by_run_separate_reps(
    #     FILE_STEMS,
    #     selected_waits_s=selected_waits_s,
    #     rep_inds=(11, 12),
    #     rep_labels={
    #         11: "rep 11: immediate final check",
    #         12: "rep 12: after dark wait",
    #     },
    #     exclude_nv_inds=None,   # or BAD_NV_INDS
    #     show_fraction=False,    # True if you want fraction instead of count
    #     ncols=3,
    #     verbose=True,
    # )


    # Inspect specific run
    # run_ind = 7

    # result, fig = plot_rep11_rep12_images_for_run(
    #     file_stem=FILE_STEMS[0],
    #     run_ind=run_ind,
    #     rep_inds=(11, 12),
    #     exclude_nv_inds=None,
    #     show_all_nv_circles=False,
    #     circle_radius_px=2.0,
    #     verbose=True,
    # )
    

    
    # diag, figs = diagnose_run(
    #     file_stem=FILE_STEMS[0],
    #     run_ind=7,
    #     rep_initial=11,
    #     rep_final=12,

    #     exclude_nv_inds=None,

    #     bright_margin_counts=5.0,

    #     # Raw-image integration
    #     aperture_radius_px=2.5,
    #     bg_inner_px=4.0,
    #     bg_outer_px=6.0,

    #     # Drift search
    #     drift_range_px=2.0,
    #     drift_step_px=0.01,
    #     max_drift_nvs=250,

    #     show_images=True,
    #     verbose=True,
    # )

    kpl.show(block=True)
    sys.exit()
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