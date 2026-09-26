# -*- coding: utf-8 -*-
"""
Spin Echo Analysis and Visualization

- Combine one or more spin-echo files.
- Process thresholded wide-field NV data.
- Save:
    1. summary PDF/PNG (median full trace + first-revival zoom)
    2. multipage PDF of all NV full traces
    3. multipage PDF of all NV first-revival zoom traces

Plots use total evolution time = 2*tau.

@author: Saroj Chand
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield


# =============================================================================
# USER SETTINGS
# =============================================================================

FILE_STEMS = [
    "2026_09_18-04_06_09-qnami-nv0_2026_02_20",
    "2026_09_18-11_34_02-qnami-nv0_2026_02_20",
    "2026_09_18-18_56_52-qnami-nv0_2026_02_20",
    "2026_09_19-04_08_37-qnami-nv0_2026_02_20",
    "2026_09_19-11_31_54-qnami-nv0_2026_02_20",
    "2026_09_19-18_54_03-qnami-nv0_2026_02_20",
    "2026_09_20-07_36_49-qnami-nv0_2026_02_20",
    "2026_09_20-17_46_31-qnami-nv0_2026_02_20",
    "2026_09_21-03_58_53-qnami-nv0_2026_02_20",
]

APPLY_THRESHOLD = True

# Acquisition used:
# revival_period = int(35.66e3 / 2)  # tau in ns
REVIVAL_PERIOD_TAU_NS = 35.66e3 / 2

# Dense first-revival window in tau:
# revival_period +/- 6e3 ns
FIRST_REVIVAL_HALF_WIDTH_TAU_NS = 6.0e3

# PDF layout: 12 NVs/page
PDF_COLS = 3
PDF_ROWS = 4

SAVE_SUMMARY_PDF = True
SAVE_FULL_TRACES_PDF = True
SAVE_FIRST_REVIVAL_PDF = True
SAVE_SUMMARY_PNG = True

SHOW_SUMMARY = True

OUTPUT_BASENAME = "spin_echo_analysis"

# Save one compact, analysis-ready combined dataset.
# This is the file the physics fitting pipeline should load.
SAVE_COMBINED_PROCESSED = True
COMBINED_SAVE_BASENAME = "qnami_spin_echo_combined_52G"

# None -> all NVs
# e.g. [0, 1, 2, 7, 15] -> selected NVs only
NV_INDICES = None


# =============================================================================
# HELPERS
# =============================================================================

def safe_ste(arr):
    arr = np.abs(np.asarray(arr, dtype=float))
    good = np.isfinite(arr) & (arr > 0)

    if np.any(good):
        fallback = float(np.nanmedian(arr[good]))
    else:
        fallback = 0.0

    return np.where(np.isfinite(arr), arr, fallback)


def resolve_nv_indices(num_nvs):
    if NV_INDICES is None:
        return np.arange(num_nvs, dtype=int)

    inds = np.asarray(NV_INDICES, dtype=int)
    inds = inds[(inds >= 0) & (inds < num_nvs)]
    return np.unique(inds)


def get_output_base():
    timestamp = dm.get_time_stamp()

    probe = Path(
        dm.get_file_path(
            __file__,
            timestamp,
            OUTPUT_BASENAME,
        )
    )

    probe.parent.mkdir(parents=True, exist_ok=True)
    return probe.with_suffix("")


def first_revival_mask(taus_ns):
    """
    Select exactly the dense first-revival acquisition region in tau.
    """
    taus_ns = np.asarray(taus_ns, dtype=float)

    lo = REVIVAL_PERIOD_TAU_NS - FIRST_REVIVAL_HALF_WIDTH_TAU_NS
    hi = REVIVAL_PERIOD_TAU_NS + FIRST_REVIVAL_HALF_WIDTH_TAU_NS

    # protects against the 4 ns quantization used in the experiment
    tol = 2.1

    return (
        np.isfinite(taus_ns)
        & (taus_ns >= lo - tol)
        & (taus_ns <= hi + tol)
    )


# =============================================================================
# DATA LOADING / PROCESSING
# =============================================================================

def group_file_stems_by_num_reps(file_stems):
    """Group files by rep-axis length while checking NV count and tau grid."""
    groups = {}
    reference_taus = None
    reference_num_nvs = None

    for stem in file_stems:
        raw = dm.get_raw_data(
            file_stem=stem,
            load_npz=True,
        )

        counts = np.asarray(raw["counts"])
        taus = np.asarray(raw["taus"], dtype=float).ravel()
        nv_list = raw["nv_list"]

        if counts.ndim != 5:
            raise ValueError(
                f"{stem}: expected counts shape "
                f"(sig/ref, NV, run, step, rep), got {counts.shape}"
            )

        if reference_taus is None:
            reference_taus = taus.copy()
            reference_num_nvs = len(nv_list)
        else:
            if len(nv_list) != reference_num_nvs:
                raise ValueError(
                    f"{stem}: NV count mismatch: "
                    f"{len(nv_list)} vs {reference_num_nvs}"
                )

            if (
                taus.shape != reference_taus.shape
                or not np.allclose(
                    taus,
                    reference_taus,
                    rtol=0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    f"{stem}: tau grid does not match the first file."
                )

        n_reps = int(counts.shape[-1])
        groups.setdefault(n_reps, []).append(stem)

        print(
            f"  {stem}: raw shape {counts.shape} "
            f"-> rep group {n_reps}"
        )

    return groups


def load_and_process(file_stems):
    """
    Preserve the OLD widefield normalization for mixed-num_reps data.

    widefield.process_counts() uses the original rep axis for reference
    normalization (even reps -> ms=0, odd reps -> ms=±1), so run and rep
    must NOT be flattened together before calling it.

    Files are grouped by num_reps, each group is processed with the old
    pipeline, then the normalized group traces are combined using the number
    of acquired shots per tau point as weights.
    """
    print("=" * 72)
    print("SPIN ECHO ANALYSIS")
    print("=" * 72)

    print(f"Combining {len(file_stems)} file(s):")
    for ind, stem in enumerate(file_stems, start=1):
        print(f"  {ind}: {stem}")

    print()
    print("Inspecting repetition structure...")
    groups = group_file_stems_by_num_reps(file_stems)

    print()
    print("Rep groups:")
    for n_reps, stems in sorted(groups.items()):
        print(f"  {n_reps} reps: {len(stems)} file(s)")

    weighted_norm_sum = None
    weighted_ste_sq_sum = None
    total_shots = 0

    base_data = None
    reference_taus = None
    reference_num_nvs = None

    for n_reps, stems in sorted(groups.items()):
        print()
        print("-" * 72)
        print(
            f"Processing {len(stems)} file(s) with "
            f"{n_reps} reps using OLD normalization"
        )
        print("-" * 72)

        # Safe because all files in this group have identical rep length.
        group_data = widefield.process_multiple_files(
            stems,
            load_npz=True,
        )

        nv_list = group_data["nv_list"]
        taus_ns = np.asarray(
            group_data["taus"],
            dtype=float,
        ).ravel()

        counts = np.asarray(group_data["counts"])

        if counts.ndim != 5:
            raise ValueError(
                f"Unexpected combined counts shape: {counts.shape}"
            )

        sig_counts = np.asarray(
            counts[0],
            dtype=np.float32,
        )
        ref_counts = np.asarray(
            counts[1],
            dtype=np.float32,
        )

        # OLD NORMALIZATION: keep [NV, run, step, rep] intact.
        group_norm_counts, group_norm_counts_ste = (
            widefield.process_counts(
                nv_list,
                sig_counts,
                ref_counts,
                threshold=APPLY_THRESHOLD,
            )
        )

        group_norm_counts = np.asarray(
            group_norm_counts,
            dtype=float,
        )
        group_norm_counts_ste = np.asarray(
            group_norm_counts_ste,
            dtype=float,
        )

        num_runs = int(counts.shape[2])
        num_reps_group = int(counts.shape[4])
        group_shots = num_runs * num_reps_group

        print(f"Group counts shape: {counts.shape}")
        print(
            f"Group weight: {num_runs} runs x "
            f"{num_reps_group} reps = {group_shots} shots/step"
        )

        if base_data is None:
            base_data = group_data
            reference_taus = taus_ns.copy()
            reference_num_nvs = len(nv_list)

            weighted_norm_sum = np.zeros_like(
                group_norm_counts,
                dtype=float,
            )
            weighted_ste_sq_sum = np.zeros_like(
                group_norm_counts_ste,
                dtype=float,
            )
        else:
            if len(nv_list) != reference_num_nvs:
                raise ValueError(
                    "NV count changed between rep groups."
                )

            if (
                taus_ns.shape != reference_taus.shape
                or not np.allclose(
                    taus_ns,
                    reference_taus,
                    rtol=0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    "Tau grid changed between rep groups."
                )

        weighted_norm_sum += (
            group_shots * group_norm_counts
        )

        weighted_ste_sq_sum += (
            group_shots * group_norm_counts_ste
        ) ** 2

        total_shots += group_shots

    if total_shots <= 0:
        raise ValueError("No valid shots were found.")

    norm_counts = weighted_norm_sum / total_shots

    norm_counts_ste = (
        np.sqrt(weighted_ste_sq_sum)
        / total_shots
    )
    norm_counts_ste = safe_ste(norm_counts_ste)

    data = base_data
    nv_list = data["nv_list"]
    taus_ns = reference_taus

    total_evolution_us = 2.0 * taus_ns / 1e3
    zoom_mask = first_revival_mask(taus_ns)

    print()
    print("=" * 72)
    print("COMBINED NORMALIZED DATA")
    print("=" * 72)
    print(f"NVs:                  {len(nv_list)}")
    print(f"Total tau points:     {len(taus_ns)}")
    print(f"Total shots/step:     {total_shots}")
    print(
        f"First-revival points: "
        f"{int(np.sum(zoom_mask))}"
    )
    print(
        "Expected first revival in total evolution: "
        f"{2 * REVIVAL_PERIOD_TAU_NS / 1e3:.3f} us"
    )

    if np.any(zoom_mask):
        print(
            "Dense revival window in total evolution: "
            f"{total_evolution_us[zoom_mask].min():.3f} to "
            f"{total_evolution_us[zoom_mask].max():.3f} us"
        )
    else:
        print(
            "WARNING: no points found in "
            "first-revival window."
        )

    # Store processing metadata in memory so the compact saved dataset has
    # full provenance without copying the huge raw counts arrays.
    data["source_file_stems"] = list(file_stems)
    data["total_shots_per_step"] = int(total_shots)
    data["normalization_method"] = (
        "old widefield.process_counts per num_reps group; "
        "shot-weighted combination across rep groups"
    )
    data["rep_group_summary"] = {
        str(int(n_reps)): {
            "num_files": int(len(stems)),
            "file_stems": list(stems),
        }
        for n_reps, stems in sorted(groups.items())
    }

    return (
        data,
        nv_list,
        taus_ns,
        total_evolution_us,
        norm_counts,
        norm_counts_ste,
        zoom_mask,
    )


def save_combined_processed_data(
    data,
    nv_list,
    taus_ns,
    total_evolution_us,
    norm_counts,
    norm_counts_ste,
):
    """
    Save a compact analysis-ready dataset.

    Deliberately does NOT save the giant combined raw-count array. The source
    file stems are preserved, so the raw combination can always be recreated.

    The physics fitting pipeline can load this file directly because it
    contains:
        nv_list
        taus
        total_evolution_times
        norm_counts
        norm_counts_ste
    """
    timestamp = dm.get_time_stamp()

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        COMBINED_SAVE_BASENAME,
    )

    processed = {
        "nv_list": nv_list,
        "taus": np.asarray(taus_ns, dtype=float),
        "total_evolution_times": np.asarray(
            total_evolution_us,
            dtype=float,
        ),
        "norm_counts": np.asarray(
            norm_counts,
            dtype=float,
        ),
        "norm_counts_ste": np.asarray(
            norm_counts_ste,
            dtype=float,
        ),
        "source_file_stems": list(
            data.get("source_file_stems", FILE_STEMS)
        ),
        "total_shots_per_step": int(
            data.get("total_shots_per_step", 0)
        ),
        "normalization_method": data.get(
            "normalization_method",
            "old widefield normalization",
        ),
        "rep_group_summary": data.get(
            "rep_group_summary",
            {},
        ),
        "revival_period_tau_ns": float(
            REVIVAL_PERIOD_TAU_NS
        ),
    }

    # Preserve orientation information if it exists in the source data.
    if "orientations" in data:
        processed["orientations"] = np.asarray(
            data["orientations"]
        )

    # Preserve a few useful experiment labels when available.
    for key in (
        "sample",
        "sample_name",
        "sequence",
        "sequence_name",
        "num_steps",
    ):
        if key in data:
            processed[key] = data[key]

    dm.save_raw_data(
        processed,
        file_path,
    )

    file_stem = Path(
        str(file_path)
    ).name

    print()
    print("=" * 72)
    print("SAVED COMBINED PROCESSED SPIN-ECHO DATA")
    print("=" * 72)
    print(f"Saved:     {file_path}")
    print(f"File stem: {file_stem}")
    print()
    print("Use this stem in the physics fitter:")
    print(f'FILE_STEM = "{file_stem}"')
    print("=" * 72)

    return file_path


# =============================================================================
# SUMMARY
# =============================================================================

def make_summary_figure(
    nv_list,
    total_evolution_us,
    norm_counts,
    norm_counts_ste,
    zoom_mask,
):
    median_counts = np.nanmedian(norm_counts, axis=0)

    # Representative per-NV STE for display, not uncertainty of the median.
    median_ste = np.nanmedian(norm_counts_ste, axis=0)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    ax = axes[0]
    ax.errorbar(
        total_evolution_us,
        median_counts,
        yerr=median_ste,
        fmt="o",
        markersize=3.5,
        capsize=1.5,
        linewidth=0.8,
    )
    ax.axvline(
        2 * REVIVAL_PERIOD_TAU_NS / 1e3,
        linestyle="--",
        linewidth=1.0,
        label="Expected first revival",
    )
    ax.set_xlabel("Total evolution time (us)")
    ax.set_ylabel(r"Normalized NV$^{-}$ population")
    ax.set_title(f"Median spin echo — {len(nv_list)} NVs")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]

    if np.any(zoom_mask):
        ax.errorbar(
            total_evolution_us[zoom_mask],
            median_counts[zoom_mask],
            yerr=median_ste[zoom_mask],
            fmt="o-",
            markersize=4,
            capsize=1.5,
            linewidth=0.9,
        )

    ax.axvline(
        2 * REVIVAL_PERIOD_TAU_NS / 1e3,
        linestyle="--",
        linewidth=1.0,
        label="Expected first revival",
    )
    ax.set_xlabel("Total evolution time (us)")
    ax.set_ylabel(r"Normalized NV$^{-}$ population")
    ax.set_title("Median — dense first-revival region")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(
        f"Spin echo | combined files = {len(FILE_STEMS)}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()

    return fig


# =============================================================================
# MULTIPAGE PDFs
# =============================================================================

def save_full_trace_pdf(
    pdf_path,
    nv_indices,
    total_evolution_us,
    norm_counts,
    norm_counts_ste,
):
    plots_per_page = PDF_COLS * PDF_ROWS

    print(
        f"\nSaving full-trace PDF: {len(nv_indices)} NVs "
        f"({PDF_COLS} x {PDF_ROWS}, {plots_per_page}/page)"
    )

    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(nv_indices), plots_per_page):
            page_inds = nv_indices[start : start + plots_per_page]

            fig, axes = plt.subplots(
                PDF_ROWS,
                PDF_COLS,
                figsize=(5.0 * PDF_COLS, 3.4 * PDF_ROWS),
                squeeze=False,
            )
            axes = axes.ravel()

            for slot, nv_ind in enumerate(page_inds):
                ax = axes[slot]

                ax.errorbar(
                    total_evolution_us,
                    norm_counts[nv_ind],
                    yerr=norm_counts_ste[nv_ind],
                    fmt="o",
                    markersize=2.7,
                    capsize=1.2,
                    linewidth=0.65,
                )

                ax.axvline(
                    2 * REVIVAL_PERIOD_TAU_NS / 1e3,
                    linestyle="--",
                    linewidth=0.6,
                    alpha=0.65,
                )

                ax.set_title(f"NV {nv_ind}", fontsize=9)
                ax.set_xlabel("2tau (us)", fontsize=8)
                ax.set_ylabel(r"Norm. NV$^{-}$ pop.", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.22)

            for slot in range(len(page_inds), len(axes)):
                axes[slot].axis("off")

            fig.suptitle(
                f"Spin-echo full traces — "
                f"{start + 1}–{start + len(page_inds)} of {len(nv_indices)}",
                fontsize=14,
                y=0.995,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.975])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {pdf_path}")


def save_first_revival_pdf(
    pdf_path,
    nv_indices,
    total_evolution_us,
    norm_counts,
    norm_counts_ste,
    zoom_mask,
):
    if not np.any(zoom_mask):
        print("No first-revival points found; zoom PDF skipped.")
        return

    x_zoom = total_evolution_us[zoom_mask]
    plots_per_page = PDF_COLS * PDF_ROWS

    print(
        f"\nSaving first-revival zoom PDF: {len(nv_indices)} NVs "
        f"({PDF_COLS} x {PDF_ROWS}, {plots_per_page}/page)"
    )

    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(nv_indices), plots_per_page):
            page_inds = nv_indices[start : start + plots_per_page]

            fig, axes = plt.subplots(
                PDF_ROWS,
                PDF_COLS,
                figsize=(5.0 * PDF_COLS, 3.4 * PDF_ROWS),
                squeeze=False,
            )
            axes = axes.ravel()

            for slot, nv_ind in enumerate(page_inds):
                ax = axes[slot]

                y_zoom = norm_counts[nv_ind, zoom_mask]
                yerr_zoom = norm_counts_ste[nv_ind, zoom_mask]

                # Connect the dense ordered points to make the revival shape
                # easier to inspect.
                ax.errorbar(
                    x_zoom,
                    y_zoom,
                    yerr=yerr_zoom,
                    fmt="o-",
                    markersize=2.8,
                    capsize=1.2,
                    linewidth=0.75,
                )

                ax.axvline(
                    2 * REVIVAL_PERIOD_TAU_NS / 1e3,
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.7,
                )

                ax.set_title(f"NV {nv_ind}", fontsize=9)
                ax.set_xlabel("2tau (us)", fontsize=8)
                ax.set_ylabel(r"Norm. NV$^{-}$ pop.", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.22)

            for slot in range(len(page_inds), len(axes)):
                axes[slot].axis("off")

            fig.suptitle(
                f"First-revival dense grid — "
                f"{start + 1}–{start + len(page_inds)} of {len(nv_indices)}",
                fontsize=14,
                y=0.995,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.975])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    kpl.init_kplotlib()

    (
        data,
        nv_list,
        taus_ns,
        total_evolution_us,
        norm_counts,
        norm_counts_ste,
        zoom_mask,
    ) = load_and_process(FILE_STEMS)

    if SAVE_COMBINED_PROCESSED:
        save_combined_processed_data(
            data,
            nv_list,
            taus_ns,
            total_evolution_us,
            norm_counts,
            norm_counts_ste,
        )

    nv_indices = resolve_nv_indices(len(nv_list))
    print(f"NVs selected for PDFs: {len(nv_indices)}")

    output_base = get_output_base()

    summary_fig = make_summary_figure(
        nv_list,
        total_evolution_us,
        norm_counts,
        norm_counts_ste,
        zoom_mask,
    )

    if SAVE_SUMMARY_PDF:
        summary_pdf = Path(str(output_base) + "_summary.pdf")
        summary_fig.savefig(
            summary_pdf,
            format="pdf",
            bbox_inches="tight",
        )
        print(f"Saved: {summary_pdf}")

    if SAVE_SUMMARY_PNG:
        summary_png = Path(str(output_base) + "_summary.png")
        summary_fig.savefig(
            summary_png,
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved: {summary_png}")

    if SAVE_FULL_TRACES_PDF:
        save_full_trace_pdf(
            Path(str(output_base) + "_all_nv_full_traces.pdf"),
            nv_indices,
            total_evolution_us,
            norm_counts,
            norm_counts_ste,
        )

    if SAVE_FIRST_REVIVAL_PDF:
        save_first_revival_pdf(
            Path(str(output_base) + "_all_nv_first_revival_zoom.pdf"),
            nv_indices,
            total_evolution_us,
            norm_counts,
            norm_counts_ste,
            zoom_mask,
        )

    if SHOW_SUMMARY:
        plt.show(block=True)
    else:
        plt.close(summary_fig)


if __name__ == "__main__":
    main()
