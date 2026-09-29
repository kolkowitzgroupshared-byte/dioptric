# -*- coding: utf-8 -*-
"""
Analysis for widefield DEER-Hahn Rabi.

Expected acquisition structure:
    pulse lengths : [L0, L0, L1, L1, ...]
    RF frequencies: [ON, OFF, ON, OFF, ...]

The analysis:
    - splits ON/OFF steps
    - optionally thresholds SCC counts
    - computes DEER contrast
    - shows raw ON-OFF DEER contrast for selected NVs, median/IQR, and heatmap
    - fits the median DEER-Rabi oscillation to a damped cosine
    - saves analyzed arrays + figures through utils.data_manager

@author: schand
"""

import traceback
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield



# FILE_STEM = [
#     "2026_09_27-05_59_19-qnami-nv0_2026_02_20",
#     "2026_09_27-14_40_02-qnami-nv0_2026_02_20",
# ]
FILE_STEM = [
    "2026_09_27-20_31_09-qnami-nv0_2026_02_20",
    "2026_09_28-02_06_10-qnami-nv0_2026_02_20",
    "2026_09_28-07_53_23-qnami-nv0_2026_02_20",          
    ]

# Keep False if you want to analyze the raw SCC count contrast exactly as
# in the current standalone DEER-Hahn analysis.
DYNAMIC_THRESHOLD = True

PDF_WIDTH = 8.5
TOP_N = 20
INDIVIDUALS_PER_PAGE = 12

# Same NV subset used in the DEER-Hahn 2.777 GHz group-A analysis.
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


def damped_cosine(
    t_ns,
    offset,
    amp,
    freq_per_ns,
    decay_ns,
    phase,
):
    return (
        offset
        + amp
        * np.exp(
            -t_ns / np.abs(decay_ns)
        )
        * np.cos(
            2
            * np.pi
            * freq_per_ns
            * t_ns
            + phase
        )
    )


def fit_median_rabi(
    rf_len_ns,
    median_curve,
):
    x = np.asarray(
        rf_len_ns,
        dtype=float,
    )
    y = np.asarray(
        median_curve,
        dtype=float,
    )

    good = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[good]
    y = y[good]

    if len(x) < 6:
        return None, None

    dx = np.median(
        np.diff(x)
    )

    y_centered = (
        y
        - np.nanmean(y)
    )

    transform = np.fft.rfft(
        y_centered
    )
    freqs = np.fft.rfftfreq(
        len(x),
        d=dx,
    )

    if len(freqs) < 2:
        return None, None

    fft_ind = (
        np.argmax(
            np.abs(
                transform[1:]
            )
        )
        + 1
    )

    freq_guess = freqs[
        fft_ind
    ]

    offset_guess = np.nanmedian(
        y
    )
    amp_guess = 0.5 * (
        np.nanmax(y)
        - np.nanmin(y)
    )

    if not np.isfinite(amp_guess) or amp_guess == 0:
        amp_guess = np.nanstd(y)

    decay_guess = max(
        float(np.ptp(x)),
        4.0 * dx,
    )

    phase_guess = np.angle(
        transform[fft_ind]
    )

    p0 = [
        offset_guess,
        amp_guess,
        freq_guess,
        decay_guess,
        phase_guess,
    ]

    lower = [
        -np.inf,
        -np.inf,
        0.0,
        max(dx, 1.0),
        -4 * np.pi,
    ]

    upper = [
        np.inf,
        np.inf,
        0.5 / dx,
        100 * max(
            np.ptp(x),
            dx,
        ),
        4 * np.pi,
    ]

    try:
        popt, pcov = curve_fit(
            damped_cosine,
            x,
            y,
            p0=p0,
            bounds=(
                lower,
                upper,
            ),
            maxfev=50_000,
        )
    except Exception:
        return None, None

    return popt, pcov



def get_nv_label(nv_list, nv_ind, original_nv_indices=None):
    """Return a compact label while preserving the original 0-211 NV index."""
    original_ind = (
        int(nv_ind)
        if original_nv_indices is None
        else int(original_nv_indices[nv_ind])
    )
    nv = nv_list[nv_ind]

    if hasattr(nv, "name") and nv.name is not None:
        return f"NV {original_ind} ({nv.name})"

    return f"NV {original_ind}"


def save_combined_pdf(
    pdf_path,
    data,
    nv_list,
    rf_len_ns,
    raw_contrast,
    p25,
    p50,
    p75,
    popt,
    rabi_period_ns,
    pi_time_ns,
    fit_decay_ns,
    fit_freq_per_ns,
    ranked_nv_indices,
    rabi_response_ptp,
    original_nv_indices,
):
    """
    Save one combined DEER-Hahn Rabi analysis PDF.

    Pages
    -----
    1. Measurement / fit summary
    2. Median + IQR + damped-cosine fit
    3. NV x RF-duration heatmap
    4. Top-response NV traces
    5+. Individual NV traces, ranked by peak-to-peak response
    """
    num_nvs = len(nv_list)
    num_runs = int(data.get("num_runs", 0))

    with PdfPages(pdf_path) as pdf:
        # ====================================================
        # Page 1: summary
        # ====================================================
        fig = plt.figure(
            figsize=(PDF_WIDTH, 11.0)
        )
        fig.suptitle(
            "DEER-Hahn Rabi Analysis",
            fontsize=16,
            y=0.97,
        )

        rf_freq = data.get(
            "rf_freq_ghz",
            None,
        )
        rf_off = data.get(
            "rf_freq_off_ghz",
            None,
        )
        ref_detuning = data.get(
            "ref_detuning_ghz",
            None,
        )
        tau_ns = data.get(
            "tau_ns",
            None,
        )
        nv_pi_ns = data.get(
            "nv_pi_ns",
            None,
        )

        lines = [
            "MEASUREMENT",
            f"NV selection: {TARGET_GROUP} {TARGET_ORIENTATION}",
            f"Target ODMR branch: ~{TARGET_ESR_GHZ:.4f} GHz",
            f"NVs analyzed: {num_nvs}",
            f"Combined runs: {num_runs}",
            f"Rabi points: {len(rf_len_ns)}",
            (
                "P1/RF pulse-length range: "
                f"{rf_len_ns.min():.0f}-{rf_len_ns.max():.0f} ns"
            ),
            "",
            "DEER / HAHN SETTINGS",
        ]

        if rf_freq is not None:
            lines.append(
                f"P1 ON frequency: {1000*float(rf_freq):.3f} MHz"
            )

        if rf_off is not None:
            lines.append(
                f"P1 OFF frequency: {1000*float(rf_off):.3f} MHz"
            )

        if ref_detuning is not None:
            lines.append(
                f"Reference detuning: {1000*float(ref_detuning):.1f} MHz"
            )

        if tau_ns is not None:
            lines.append(
                f"Hahn tau: {float(tau_ns)/1000:.3f} us"
            )
            lines.append(
                f"Total free evolution 2tau: {2*float(tau_ns)/1000:.3f} us"
            )

        if nv_pi_ns is not None:
            lines.append(
                f"NV pi pulse: {float(nv_pi_ns):.0f} ns"
            )

        lines.extend(
            [
                f"Dynamic thresholding: {DYNAMIC_THRESHOLD}",
                "Plotted contrast: raw ON-OFF (no per-NV baseline subtraction)",
                "",
                "MEDIAN DEER-RABI FIT",
                f"Rabi period T_R: {rabi_period_ns:.2f} ns",
                f"P1 pi time: {pi_time_ns:.2f} ns",
                f"Decay time: {fit_decay_ns:.2f} ns",
            ]
        )

        if np.isfinite(fit_freq_per_ns):
            lines.append(
                f"Rabi frequency: {1000*fit_freq_per_ns:.4f} MHz"
            )
        else:
            lines.append(
                "Rabi frequency: fit unavailable"
            )

        fig.text(
            0.08,
            0.90,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=10.5,
            family="monospace",
            linespacing=1.35,
        )

        fig.text(
            0.08,
            0.20,
            "Median fit model",
            fontsize=11,
            fontweight="bold",
        )

        fig.text(
            0.08,
            0.155,
            (
                r"$C(t)=C_0+A\,e^{-t/T_d}"
                r"\cos(2\pi f_R t+\phi)$"
            ),
            fontsize=13,
        )

        fig.text(
            0.08,
            0.11,
            (
                r"$T_R=1/f_R,\qquad "
                r"t_{\pi}=T_R/2$"
            ),
            fontsize=13,
        )

        fig.text(
            0.08,
            0.055,
            (
                "The median fit is an ensemble calibration. "
                "Individual NVs may have different amplitudes, phases, "
                "decay times, and apparent Rabi periods."
            ),
            fontsize=9,
            wrap=True,
        )

        pdf.savefig(fig)
        plt.close(fig)

        # ====================================================
        # Page 2: median/IQR + fit
        # ====================================================
        fig, ax = plt.subplots(
            figsize=(PDF_WIDTH, 6.5)
        )

        for row in raw_contrast:
            ax.plot(
                rf_len_ns,
                row,
                linewidth=0.55,
                alpha=0.10,
            )

        ax.fill_between(
            rf_len_ns,
            p25,
            p75,
            alpha=0.22,
            label="IQR",
        )

        ax.plot(
            rf_len_ns,
            p50,
            "o-",
            markersize=3,
            linewidth=1.4,
            label="Median",
        )

        if popt is not None:
            dense_t = np.linspace(
                rf_len_ns.min(),
                rf_len_ns.max(),
                1200,
            )

            ax.plot(
                dense_t,
                damped_cosine(
                    dense_t,
                    *popt,
                ),
                linewidth=1.7,
                label=(
                    f"Fit: T_R={rabi_period_ns:.1f} ns, "
                    f"t_pi={pi_time_ns:.1f} ns"
                ),
            )

        title = "Median DEER-Hahn Rabi"
        if rf_freq is not None:
            title += (
                f" | P1 = {1000*float(rf_freq):.1f} MHz"
            )

        ax.set_title(title)
        ax.set_xlabel(
            "P1 / RF pulse duration (ns)"
        )
        ax.set_ylabel(
            "Raw DEER contrast (ON - OFF)"
        )
        ax.legend()
        ax.grid(alpha=0.15)

        pdf.savefig(fig)
        plt.close(fig)

        # ====================================================
        # Page 3: heatmap
        # ====================================================
        fig, ax = plt.subplots(
            figsize=(PDF_WIDTH, 7.0)
        )

        extent = [
            float(rf_len_ns[0]),
            float(rf_len_ns[-1]),
            raw_contrast.shape[0] - 0.5,
            -0.5,
        ]

        im = ax.imshow(
            raw_contrast,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )

        ax.set_xlabel(
            "P1 / RF pulse duration (ns)"
        )
        ax.set_ylabel(
            "Selected NV row (group A)"
        )
        ax.set_title(
            "DEER-Hahn Rabi raw ON-OFF contrast heatmap"
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Raw DEER contrast (ON - OFF)",
        )

        pdf.savefig(fig)
        plt.close(fig)

        # ====================================================
        # Page 4: strongest-response NVs
        # ====================================================
        top_n = min(
            TOP_N,
            num_nvs,
        )

        fig, ax = plt.subplots(
            figsize=(PDF_WIDTH, 6.5)
        )

        for rank, nv_ind in enumerate(
            ranked_nv_indices[:top_n],
            start=1,
        ):
            nv_ind = int(nv_ind)

            ax.plot(
                rf_len_ns,
                raw_contrast[nv_ind],
                linewidth=1.0,
                alpha=0.75,
                label=(
                    get_nv_label(
                        nv_list,
                        nv_ind,
                        original_nv_indices,
                    )
                    if rank <= 10
                    else None
                ),
            )

        ax.set_xlabel(
            "P1 / RF pulse duration (ns)"
        )
        ax.set_ylabel(
            "Raw DEER contrast (ON - OFF)"
        )
        ax.set_title(
            f"Top {top_n} NVs by peak-to-peak DEER-Rabi response"
        )
        ax.grid(alpha=0.15)

        if top_n > 0:
            ax.legend(
                fontsize=7,
                ncol=2,
            )

        pdf.savefig(fig)
        plt.close(fig)

        # ====================================================
        # Page 5+: individual NVs
        # ====================================================
        ncols = 3
        nrows = 4
        per_page = ncols * nrows

        for start in range(
            0,
            num_nvs,
            per_page,
        ):
            chunk = ranked_nv_indices[
                start : start + per_page
            ]

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(PDF_WIDTH, 11.0),
                sharex=True,
            )

            axes = np.asarray(
                axes
            ).reshape(-1)

            for plot_ind, ax in enumerate(
                axes
            ):
                if plot_ind >= len(chunk):
                    ax.axis("off")
                    continue

                nv_ind = int(
                    chunk[plot_ind]
                )
                y = raw_contrast[
                    nv_ind
                ]

                ax.plot(
                    rf_len_ns,
                    y,
                    "o-",
                    markersize=2.2,
                    linewidth=0.8,
                )

                ax.set_title(
                    (
                        f"{get_nv_label(nv_list, nv_ind, original_nv_indices)}\n"
                        f"p-p={rabi_response_ptp[nv_ind]:.3g}"
                    ),
                    fontsize=8,
                )

                ax.grid(
                    alpha=0.12
                )

            fig.supxlabel(
                "P1 / RF pulse duration (ns)"
            )
            fig.supylabel(
                "Raw DEER contrast (ON - OFF)"
            )

            fig.suptitle(
                (
                    "Individual DEER-Rabi traces "
                    f"({start+1}-{start+len(chunk)} of {num_nvs})"
                ),
                fontsize=13,
                y=0.995,
            )

            pdf.savefig(fig)
            plt.close(fig)

        info = pdf.infodict()
        info["Title"] = (
            "DEER-Hahn Rabi Analysis"
        )
        info["Subject"] = (
            "Widefield fixed-frequency P1 DEER-Rabi analysis"
        )
        info["Author"] = "Saroj Chand"


def main():
    if len(FILE_STEM) == 0:
        raise ValueError(
            "Set FILE_STEM to one or more DEER-Rabi data files."
        )

    kpl.init_kplotlib()

    data = dm.get_raw_data(
        file_stem=FILE_STEM,
        load_npz=True,
        use_cache=True,
    )

    all_nv_list = data["nv_list"]
    num_nvs_full = len(all_nv_list)

    selected_nv_indices = np.asarray(
        TARGET_NV_INDICES,
        dtype=int,
    )

    if selected_nv_indices.size == 0:
        raise ValueError("TARGET_NV_INDICES is empty.")

    if np.any(selected_nv_indices < 0) or np.any(
        selected_nv_indices >= num_nvs_full
    ):
        raise ValueError(
            "TARGET_NV_INDICES contains indices outside the loaded NV list: "
            f"num_nvs_full={num_nvs_full}, "
            f"min={selected_nv_indices.min()}, "
            f"max={selected_nv_indices.max()}."
        )

    nv_list = [
        all_nv_list[int(ind)]
        for ind in selected_nv_indices
    ]

    print(
        "\nNV selection"
        f"\n  group        : {TARGET_GROUP}"
        f"\n  orientation  : {TARGET_ORIENTATION}"
        f"\n  ODMR branch  : ~{TARGET_ESR_GHZ:.4f} GHz"
        f"\n  selected NVs : {len(nv_list)} / {num_nvs_full}"
    )

    rf_len_ns = np.asarray(
        data["rf_len_ns"],
        dtype=float,
    )

    counts = np.asarray(
        data["counts"]
    )

    if counts.ndim == 5:
        counts_exp = counts[0]
    elif counts.ndim == 4:
        counts_exp = counts
    else:
        raise ValueError(
            "Unexpected counts shape: "
            f"{counts.shape}"
        )

    if counts_exp.shape[0] != num_nvs_full:
        raise ValueError(
            "Counts/NV-list mismatch: "
            f"counts has {counts_exp.shape[0]} NVs, "
            f"nv_list has {num_nvs_full}."
        )

    # Subset before thresholding, averaging, fitting, ranking, and plotting.
    counts_exp = np.ascontiguousarray(
        counts_exp[selected_nv_indices, ...]
    )

    num_steps_actual = (
        counts_exp.shape[2]
    )

    if num_steps_actual != 2 * len(rf_len_ns):
        raise ValueError(
            "Expected two acquisition steps per physical Rabi duration: "
            f"counts steps={num_steps_actual}, "
            f"physical lengths={len(rf_len_ns)}."
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
        sig_counts, ref_counts = (
            widefield.threshold_counts(
                nv_list,
                sig_counts,
                ref_counts,
                dynamic_thresh=True,
            )
        )

    (
        avg_sig,
        avg_sig_ste,
        _,
    ) = widefield.average_counts(
        sig_counts
    )

    (
        avg_ref,
        avg_ref_ste,
        _,
    ) = widefield.average_counts(
        ref_counts
    )

    (
        avg_contrast,
        avg_contrast_ste,
    ) = widefield.calc_contrast(
        sig_counts,
        ref_counts,
    )

    (
        avg_snr,
        avg_snr_ste,
    ) = widefield.calc_snr(
        sig_counts,
        ref_counts,
    )

    # Keep baseline-subtracted contrast only as a secondary diagnostic.
    # The primary fit and all report plots use RAW ON-OFF DEER contrast.
    baseline = np.nanmedian(
        avg_contrast,
        axis=1,
        keepdims=True,
    )
    contrast_centered = (
        avg_contrast
        - baseline
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

    popt, pcov = fit_median_rabi(
        rf_len_ns,
        p50,
    )

    if popt is not None:
        (
            fit_offset,
            fit_amp,
            fit_freq_per_ns,
            fit_decay_ns,
            fit_phase,
        ) = popt

        rabi_period_ns = (
            1.0
            / fit_freq_per_ns
        )
        pi_time_ns = (
            rabi_period_ns
            / 2.0
        )
    else:
        fit_offset = np.nan
        fit_amp = np.nan
        fit_freq_per_ns = np.nan
        fit_decay_ns = np.nan
        fit_phase = np.nan
        rabi_period_ns = np.nan
        pi_time_ns = np.nan

    print(
        "\nDEER-Rabi median fit"
        f"\n  Rabi period : {rabi_period_ns:.2f} ns"
        f"\n  pi time     : {pi_time_ns:.2f} ns"
        f"\n  decay       : {fit_decay_ns:.2f} ns"
    )

    timestamp = dm.get_time_stamp()

    repr_nv_sig = (
        widefield.get_repr_nv_sig(
            nv_list
        )
    )
    repr_nv_name = (
        repr_nv_sig.name
    )

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_name
        + "-deer-hahn-rabi-analysis",
    )

    rabi_response_ptp = np.ascontiguousarray(
        np.nanmax(
            avg_contrast,
            axis=1,
        )
        - np.nanmin(
            avg_contrast,
            axis=1,
        )
    )

    ranked_nv_indices = np.ascontiguousarray(
        np.argsort(
            rabi_response_ptp
        )[::-1]
    )

    ranked_original_nv_indices = np.ascontiguousarray(
        selected_nv_indices[ranked_nv_indices]
    )

    analysis_data = {
        "timestamp": timestamp,
        "source_files": FILE_STEM,
        "target_group": TARGET_GROUP,
        "target_esr_ghz": float(TARGET_ESR_GHZ),
        "target_orientation": TARGET_ORIENTATION,
        "num_nvs_full": int(num_nvs_full),
        "num_nvs_selected": int(len(nv_list)),
        "selected_nv_indices": np.ascontiguousarray(selected_nv_indices),
        "ranked_original_nv_indices": ranked_original_nv_indices,
        "rf_len_ns": np.ascontiguousarray(
            rf_len_ns
        ),
        "rf_freq_ghz": data.get(
            "rf_freq_ghz",
            None,
        ),
        "rf_freq_off_ghz": data.get(
            "rf_freq_off_ghz",
            None,
        ),
        "ref_detuning_ghz": data.get(
            "ref_detuning_ghz",
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
        "dynamic_threshold": bool(
            DYNAMIC_THRESHOLD
        ),
        "avg_sig": np.ascontiguousarray(
            avg_sig
        ),
        "avg_sig_ste": np.ascontiguousarray(
            avg_sig_ste
        ),
        "avg_ref": np.ascontiguousarray(
            avg_ref
        ),
        "avg_ref_ste": np.ascontiguousarray(
            avg_ref_ste
        ),
        "avg_contrast": np.ascontiguousarray(
            avg_contrast
        ),
        "avg_contrast_ste": np.ascontiguousarray(
            avg_contrast_ste
        ),
        "avg_snr": np.ascontiguousarray(
            avg_snr
        ),
        "avg_snr_ste": np.ascontiguousarray(
            avg_snr_ste
        ),
        "contrast_centered": np.ascontiguousarray(
            contrast_centered
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
        "fit_params": np.ascontiguousarray(
            np.asarray(
                [
                    fit_offset,
                    fit_amp,
                    fit_freq_per_ns,
                    fit_decay_ns,
                    fit_phase,
                ],
                dtype=float,
            )
        ),
        "rabi_period_ns": float(
            rabi_period_ns
        ),
        "pi_time_ns": float(
            pi_time_ns
        ),
        "rabi_response_ptp": rabi_response_ptp,
        "ranked_nv_indices": ranked_nv_indices,
    }

    dm.save_raw_data(
        analysis_data,
        file_path,
        [
            "rf_len_ns",
            "selected_nv_indices",
            "ranked_original_nv_indices",
            "avg_sig",
            "avg_sig_ste",
            "avg_ref",
            "avg_ref_ste",
            "avg_contrast",
            "avg_contrast_ste",
            "avg_snr",
            "avg_snr_ste",
            "contrast_centered",
            "p25_raw_contrast",
            "median_raw_contrast",
            "p75_raw_contrast",
            "fit_params",
            "rabi_response_ptp",
            "ranked_nv_indices",
        ],
    )

    # ---------------------------------------------------------
    # Figure 1: all NVs + median
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    for row in avg_contrast:
        ax.plot(
            rf_len_ns,
            row,
            linewidth=0.7,
            alpha=0.15,
        )

    ax.fill_between(
        rf_len_ns,
        p25,
        p75,
        alpha=0.22,
        label="IQR",
    )

    ax.plot(
        rf_len_ns,
        p50,
        "o-",
        markersize=3,
        linewidth=1.4,
        label="Median",
    )

    if popt is not None:
        dense_t = np.linspace(
            rf_len_ns.min(),
            rf_len_ns.max(),
            1000,
        )

        ax.plot(
            dense_t,
            damped_cosine(
                dense_t,
                *popt,
            ),
            linewidth=1.6,
            label=(
                f"Fit: T_R={rabi_period_ns:.1f} ns, "
                f"pi={pi_time_ns:.1f} ns"
            ),
        )

    ax.set_xlabel(
        "P1 / RF pulse duration (ns)"
    )
    ax.set_ylabel(
        "Raw DEER contrast (ON - OFF)"
    )

    rf_freq = data.get(
        "rf_freq_ghz",
        None,
    )

    title = f"DEER-Hahn Rabi | {TARGET_GROUP} | ~{TARGET_ESR_GHZ:.4f} GHz NV family"

    if rf_freq is not None:
        title += (
            f" | {1000*float(rf_freq):.1f} MHz"
        )

    ax.set_title(
        title
    )
    ax.legend()
    ax.grid(alpha=0.15)

    dm.save_figure(
        fig,
        file_path,
    )

    # ---------------------------------------------------------
    # Figure 2: heatmap
    # ---------------------------------------------------------
    heatmap_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_name
        + "-deer-hahn-rabi-analysis-heatmap",
    )

    fig_h, ax_h = plt.subplots(
        figsize=(8.5, 7.0)
    )

    extent = [
        float(rf_len_ns[0]),
        float(rf_len_ns[-1]),
        avg_contrast.shape[0] - 0.5,
        -0.5,
    ]

    im = ax_h.imshow(
        avg_contrast,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
    )

    ax_h.set_xlabel(
        "P1 / RF pulse duration (ns)"
    )
    ax_h.set_ylabel(
        "Selected NV row (group A)"
    )
    ax_h.set_title(
        "DEER-Hahn Rabi raw ON-OFF contrast"
    )

    fig_h.colorbar(
        im,
        ax=ax_h,
        label="Raw DEER contrast (ON - OFF)",
    )

    dm.save_figure(
        fig_h,
        heatmap_path,
    )

    # ---------------------------------------------------------
    # Combined PDF report beside the analyzed DM file
    # ---------------------------------------------------------
    pdf_path = Path(
        str(file_path)
    ).with_suffix(
        ".pdf"
    )

    save_combined_pdf(
        pdf_path,
        data,
        nv_list,
        rf_len_ns,
        avg_contrast,
        p25,
        p50,
        p75,
        popt,
        rabi_period_ns,
        pi_time_ns,
        fit_decay_ns,
        fit_freq_per_ns,
        ranked_nv_indices,
        rabi_response_ptp,
        selected_nv_indices,
    )

    print(
        f"Combined PDF saved: {pdf_path}"
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
