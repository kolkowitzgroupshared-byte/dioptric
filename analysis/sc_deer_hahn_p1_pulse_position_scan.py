# -*- coding: utf-8 -*-
"""
Analysis for widefield DEER-Hahn RF-position sweep.

Uses the same 102 selected group-A NV indices used in the current
2.7773-GHz DEER-Hahn analysis.

Primary plotted quantity:
    raw ON-OFF DEER contrast

No per-NV median subtraction is applied.

@author: schand
"""

import traceback
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield


FILE_STEM = [
    "2026_09_29-16_06_48-qnami-nv0_2026_02_20",
    "2026_09_29-17_35_28-qnami-nv0_2026_02_20",
]

DYNAMIC_THRESHOLD = False

TARGET_GROUP = "group_A"
TARGET_ESR_GHZ = 2.7773
TARGET_ORIENTATION = "(1, 1, -1)"

TARGET_NV_INDICES = np.array(
    [
        0, 8, 9, 11, 12, 14, 15, 21, 22, 24, 26, 27, 29, 31, 32, 33,
        35, 39, 41, 44, 46, 48, 50, 53, 57, 61, 64, 66, 67, 70, 72, 73,
        75, 79, 81, 82, 83, 84, 86, 87, 88, 90, 91, 92, 93, 95, 97, 98,
        102, 106, 108, 110, 111, 114, 116, 117, 118, 119, 125, 126, 129,
        132, 133, 134, 136, 139, 140, 141, 142, 146, 151, 156, 157, 159,
        160, 162, 163, 166, 168, 170, 171, 173, 174, 175, 177, 178, 179,
        182, 184, 185, 192, 193, 196, 197, 198, 199, 200, 202, 204, 206,
        207, 208,
    ],
    dtype=int,
)


def main():
    if len(FILE_STEM) == 0:
        raise ValueError(
            "Set FILE_STEM to one or more RF-position sweep data files."
        )

    kpl.init_kplotlib()

    data = dm.get_raw_data(
        file_stem=FILE_STEM,
        load_npz=True,
        use_cache=True,
    )

    all_nv_list = data[
        "nv_list"
    ]

    counts = np.asarray(
        data[
            "counts"
        ]
    )

    offset_ns = np.asarray(
        data[
            "rf_center_offset_ns"
        ],
        dtype=float,
    )

    if counts.ndim == 5:
        counts_exp = counts[0]
    elif counts.ndim == 4:
        counts_exp = counts
    else:
        raise ValueError(
            f"Unexpected counts shape: {counts.shape}"
        )

    selected = TARGET_NV_INDICES

    if np.any(
        selected >= counts_exp.shape[0]
    ):
        raise ValueError(
            "Selected NV indices exceed loaded NV dimension."
        )

    nv_list = [
        all_nv_list[
            int(ind)
        ]
        for ind in selected
    ]

    counts_exp = np.ascontiguousarray(
        counts_exp[
            selected,
            ...,
        ]
    )

    num_steps_actual = (
        counts_exp.shape[2]
    )

    if num_steps_actual != 2 * len(
        offset_ns
    ):
        raise ValueError(
            "Expected two acquisition steps per timing offset: "
            f"counts steps={num_steps_actual}, "
            f"offsets={len(offset_ns)}"
        )

    on_inds = np.arange(
        0,
        num_steps_actual,
        2,
    )
    off_inds = np.arange(
        1,
        num_steps_actual,
        2,
    )

    sig_counts = counts_exp[
        :,
        :,
        on_inds,
        :,
    ]
    ref_counts = counts_exp[
        :,
        :,
        off_inds,
        :,
    ]

    if DYNAMIC_THRESHOLD:
        (
            sig_counts,
            ref_counts,
        ) = widefield.threshold_counts(
            nv_list,
            sig_counts,
            ref_counts,
            dynamic_thresh=True,
        )

    (
        avg_contrast,
        avg_contrast_ste,
    ) = widefield.calc_contrast(
        sig_counts,
        ref_counts,
    )

    p25 = np.nanpercentile(
        avg_contrast,
        25,
        axis=0,
    )
    p50 = np.nanpercentile(
        avg_contrast,
        50,
        axis=0,
    )
    p75 = np.nanpercentile(
        avg_contrast,
        75,
        axis=0,
    )

    x_us = offset_ns / 1000

    # ---------------------------------------------------------
    # Main figure
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8.5, 5.8)
    )

    for row in avg_contrast:
        ax.plot(
            x_us,
            row,
            linewidth=0.55,
            alpha=0.10,
        )

    ax.fill_between(
        x_us,
        p25,
        p75,
        alpha=0.22,
        label="IQR",
    )

    ax.plot(
        x_us,
        p50,
        "o-",
        markersize=3,
        linewidth=1.5,
        label="Median",
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="NV pi center",
    )

    rf_freq = data.get(
        "rf_freq_ghz",
        None,
    )
    rf_len = data.get(
        "rf_len_ns",
        None,
    )

    title = (
        "DEER-Hahn RF pulse-position sweep"
    )

    if rf_freq is not None:
        title += (
            f" | {1000*float(rf_freq):.1f} MHz"
        )

    if rf_len is not None:
        title += (
            f" | {float(rf_len):.0f} ns"
        )

    ax.set_title(
        title
    )
    ax.set_xlabel(
        "P1/RF pulse-center offset from NV pi (us)"
    )
    ax.set_ylabel(
        "Raw DEER contrast (ON - OFF)"
    )
    ax.grid(
        alpha=0.15
    )
    ax.legend()

    # ---------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------
    fig_h, ax_h = plt.subplots(
        figsize=(8.5, 7.0)
    )

    extent = [
        float(x_us[0]),
        float(x_us[-1]),
        avg_contrast.shape[0] - 0.5,
        -0.5,
    ]

    im = ax_h.imshow(
        avg_contrast,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    ax_h.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )

    ax_h.set_xlabel(
        "P1/RF pulse-center offset from NV pi (us)"
    )
    ax_h.set_ylabel(
        "Selected group-A NV row"
    )
    ax_h.set_title(
        "RF-position DEER contrast heatmap"
    )

    fig_h.colorbar(
        im,
        ax=ax_h,
        label="Raw DEER contrast (ON - OFF)",
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    timestamp = dm.get_time_stamp()

    repr_nv_sig = (
        widefield.get_repr_nv_sig(
            nv_list
        )
    )

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_sig.name
        + "-deer-hahn-rf-position-analysis",
    )

    analysis_data = {
        "timestamp": timestamp,
        "source_files": FILE_STEM,
        "target_group": TARGET_GROUP,
        "target_esr_ghz": TARGET_ESR_GHZ,
        "target_orientation": TARGET_ORIENTATION,
        "selected_nv_indices": np.ascontiguousarray(
            selected
        ),
        "rf_center_offset_ns": np.ascontiguousarray(
            offset_ns
        ),
        "rf_freq_ghz": data.get(
            "rf_freq_ghz",
            None,
        ),
        "rf_freq_off_ghz": data.get(
            "rf_freq_off_ghz",
            None,
        ),
        "rf_len_ns": data.get(
            "rf_len_ns",
            None,
        ),
        "tau_ns": data.get(
            "tau_ns",
            None,
        ),
        "nv_pi_ns": data.get(
            "nv_pi_ns",
            None,
        ),
        "avg_contrast": np.ascontiguousarray(
            avg_contrast
        ),
        "avg_contrast_ste": np.ascontiguousarray(
            avg_contrast_ste
        ),
        "p25_raw_contrast": np.ascontiguousarray(
            p25
        ),
        "median_raw_contrast": np.ascontiguousarray(
            p50
        ),
        "p75_raw_contrast": np.ascontiguousarray(
            p75
        ),
    }

    dm.save_raw_data(
        analysis_data,
        file_path,
        [
            "selected_nv_indices",
            "rf_center_offset_ns",
            "avg_contrast",
            "avg_contrast_ste",
            "p25_raw_contrast",
            "median_raw_contrast",
            "p75_raw_contrast",
        ],
    )

    dm.save_figure(
        fig,
        file_path,
    )

    heatmap_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_sig.name
        + "-deer-hahn-rf-position-analysis-heatmap",
    )

    dm.save_figure(
        fig_h,
        heatmap_path,
    )

    # ---------------------------------------------------------
    # Compact PDF
    # ---------------------------------------------------------
    pdf_path = Path(
        str(
            file_path
        )
    ).with_suffix(
        ".pdf"
    )

    with PdfPages(
        pdf_path
    ) as pdf:
        pdf.savefig(
            fig
        )
        pdf.savefig(
            fig_h
        )

        fig_s = plt.figure(
            figsize=(8.5, 11)
        )

        fig_s.suptitle(
            "DEER-Hahn RF Position Sweep",
            fontsize=16,
            y=0.97,
        )

        summary = [
            f"Selected NVs: {len(selected)} ({TARGET_GROUP})",
            f"NV branch: ~{TARGET_ESR_GHZ:.4f} GHz",
            f"P1 frequency: {1000*float(rf_freq):.3f} MHz"
            if rf_freq is not None
            else "P1 frequency: unavailable",
            f"P1 pulse: {float(rf_len):.0f} ns"
            if rf_len is not None
            else "P1 pulse: unavailable",
            f"Offset range: {x_us.min():.3f} to {x_us.max():.3f} us",
            "",
            "Timing convention:",
            "  offset < 0 : P1 pulse before NV pi",
            "  offset = 0 : pulse centers aligned",
            "  offset > 0 : P1 pulse after NV pi",
            "",
            "Primary plotted signal:",
            "  raw ON-OFF DEER contrast",
            "  no per-NV median subtraction",
        ]

        fig_s.text(
            0.08,
            0.90,
            "\n".join(
                summary
            ),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            linespacing=1.4,
        )

        pdf.savefig(
            fig_s
        )
        plt.close(
            fig_s
        )

    print(
        f"Saved PDF: {pdf_path}"
    )

    kpl.show(
        block=True
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            traceback.format_exc()
        )
