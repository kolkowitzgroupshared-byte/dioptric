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
    # "2026_09_18-YY_YY_YY-qnami-nv0_2026_02_20",
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

def load_and_process(file_stems):
    print("=" * 72)
    print("SPIN ECHO ANALYSIS")
    print("=" * 72)

    print(f"Combining {len(file_stems)} file(s):")
    for ind, stem in enumerate(file_stems, start=1):
        print(f"  {ind}: {stem}")

    data = widefield.process_multiple_files(
        file_stems,
        load_npz=True,
    )

    nv_list = data["nv_list"]
    taus_ns = np.asarray(data["taus"], dtype=float)

    # Spin-echo total evolution time = 2*tau
    total_evolution_us = 2.0 * taus_ns / 1e3

    counts = np.asarray(data["counts"])
    sig_counts = np.asarray(counts[0], dtype=np.float32)
    ref_counts = np.asarray(counts[1], dtype=np.float32)

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list,
        sig_counts,
        ref_counts,
        threshold=APPLY_THRESHOLD,
    )

    norm_counts = np.asarray(norm_counts, dtype=float)
    norm_counts_ste = safe_ste(norm_counts_ste)

    if norm_counts.shape != norm_counts_ste.shape:
        raise ValueError(
            f"norm_counts shape {norm_counts.shape} does not match "
            f"norm_counts_ste shape {norm_counts_ste.shape}"
        )

    if norm_counts.shape[0] != len(nv_list):
        raise ValueError(
            f"Expected {len(nv_list)} NV rows, got {norm_counts.shape[0]}"
        )

    if norm_counts.shape[1] != len(taus_ns):
        raise ValueError(
            f"Expected {len(taus_ns)} tau columns, got {norm_counts.shape[1]}"
        )

    zoom_mask = first_revival_mask(taus_ns)

    print()
    print(f"NVs:                  {len(nv_list)}")
    print(f"Total tau points:     {len(taus_ns)}")
    print(f"First-revival points: {int(np.sum(zoom_mask))}")
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
        print("WARNING: no points found in first-revival window.")

    return (
        data,
        nv_list,
        taus_ns,
        total_evolution_us,
        norm_counts,
        norm_counts_ste,
        zoom_mask,
    )


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
