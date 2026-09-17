# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment - Enhanced

Created on Fall, 2024
@auhtor : Saroj Chand
"""

import sys
import traceback
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
from joblib import Parallel, delayed
from scipy.optimize import curve_fit, least_squares

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield


def fit_rabi_data(
    nv_list,
    taus,
    avg_counts,
    avg_counts_ste,
    epsilon=1e-10,
    n_freq_starts=4,
):
    """
    Robust Rabi fitting.

    Uses:
      - several FFT frequency candidates
      - several phase starts
      - bounded least_squares
      - soft_l1 robust loss
      - best-fit selection from all starts

    popt format:
        [amp, freq, decay, phase, baseline]

    Rabi period:
        1 / freq
    """

    taus = np.asarray(taus, dtype=float)
    avg_counts = np.asarray(avg_counts, dtype=float)
    avg_counts_ste = np.asarray(avg_counts_ste, dtype=float)

    num_nvs = len(nv_list)

    # -------------------------------------------------------------------------
    # Cleaner parameterization
    # -------------------------------------------------------------------------
    def cos_decay(tau, amp, freq, decay, phase, baseline):
        return (
            baseline
            + amp
            * np.exp(-tau / decay)
            * np.cos(2 * np.pi * freq * tau + phase)
        )

    # -------------------------------------------------------------------------
    # Fit one NV
    # -------------------------------------------------------------------------
    def fit_single_nv(nv_idx):

        print(f"Fitting NV {nv_idx}...")

        y = np.asarray(avg_counts[nv_idx], dtype=float)

        sigma = np.abs(
            np.asarray(
                avg_counts_ste[nv_idx],
                dtype=float,
            )
        )

        good = (
            np.isfinite(taus)
            & np.isfinite(y)
            & np.isfinite(sigma)
        )

        t = taus[good]
        y = y[good]
        sigma = sigma[good]

        if len(t) < 8:
            return None, [np.nan] * 5

        # ---------------------------------------------------------------------
        # Safe uncertainties
        # ---------------------------------------------------------------------
        positive_sigma = sigma[sigma > 0]

        if positive_sigma.size:
            sigma_floor = np.median(positive_sigma)
        else:
            sigma_floor = 1.0

        sigma = np.where(
            sigma > epsilon,
            sigma,
            sigma_floor,
        )

        # ---------------------------------------------------------------------
        # Frequency bounds
        # ---------------------------------------------------------------------
        dt = np.median(np.diff(t))
        t_span = np.ptp(t)

        nyquist = 0.5 / dt

        # Allow very slow oscillations but prevent zero-frequency nonsense.
        freq_min = max(
            1.0 / (20.0 * t_span),
            epsilon,
        )

        freq_max = 0.95 * nyquist

        # ---------------------------------------------------------------------
        # FFT frequency candidates
        # ---------------------------------------------------------------------
        y_centered = y - np.mean(y)

        transform = np.fft.rfft(
            y_centered
        )

        fft_freqs = np.fft.rfftfreq(
            len(y),
            d=dt,
        )

        fft_mag = np.abs(transform)

        valid_fft = (
            (fft_freqs >= freq_min)
            & (fft_freqs <= freq_max)
        )

        valid_inds = np.where(
            valid_fft
        )[0]

        if valid_inds.size == 0:
            return None, [np.nan] * 5

        # Sort FFT peaks from strongest to weakest.
        order = valid_inds[
            np.argsort(
                fft_mag[valid_inds]
            )[::-1]
        ]

        candidate_inds = order[
            :n_freq_starts
        ]

        candidate_freqs = fft_freqs[
            candidate_inds
        ]

        # ---------------------------------------------------------------------
        # Initial amplitude/baseline/decay
        # ---------------------------------------------------------------------
        y_min = np.min(y)
        y_max = np.max(y)
        y_span = max(
            y_max - y_min,
            1e-4,
        )

        amp_guess = 0.5 * y_span
        baseline_guess = np.mean(y)

        decay_guess = max(
            t_span,
            dt,
        )

        # ---------------------------------------------------------------------
        # Bounds
        # ---------------------------------------------------------------------
        lower = np.array(
            [
                0.0,                    # amplitude
                freq_min,               # frequency
                max(dt / 2, epsilon),   # decay
                -2 * np.pi,             # phase
                y_min - y_span,         # baseline
            ]
        )

        upper = np.array(
            [
                2.0 * y_span,
                freq_max,
                20.0 * t_span,
                2 * np.pi,
                y_max + y_span,
            ]
        )

        # ---------------------------------------------------------------------
        # Weighted residual
        # ---------------------------------------------------------------------
        def residual(params):

            model = cos_decay(
                t,
                *params,
            )

            return (
                model - y
            ) / sigma

        best_result = None
        best_score = np.inf

        # Different phase starts help prevent local minima.
        phase_starts = [
            0.0,
            np.pi / 2,
            np.pi,
            -np.pi / 2,
        ]

        # ---------------------------------------------------------------------
        # Multi-start fit
        # ---------------------------------------------------------------------
        for freq_guess in candidate_freqs:

            for phase_guess in phase_starts:

                p0 = np.array(
                    [
                        amp_guess,
                        freq_guess,
                        decay_guess,
                        phase_guess,
                        baseline_guess,
                    ]
                )

                # Ensure initial point is strictly inside bounds.
                p0 = np.maximum(
                    p0,
                    lower + 1e-12,
                )

                p0 = np.minimum(
                    p0,
                    upper - 1e-12,
                )

                try:

                    result = least_squares(
                        residual,
                        p0,
                        bounds=(lower, upper),
                        loss="soft_l1",
                        f_scale=1.0,
                        x_scale="jac",
                        max_nfev=20000,
                    )

                    if not result.success:
                        continue

                    # Compare solutions using ordinary weighted residual
                    # rather than robust cost.
                    res = residual(
                        result.x
                    )

                    score = np.mean(
                        res**2
                    )

                    if (
                        np.isfinite(score)
                        and score < best_score
                    ):
                        best_score = score
                        best_result = result

                except Exception:
                    continue

        # ---------------------------------------------------------------------
        # Failed completely
        # ---------------------------------------------------------------------
        if best_result is None:

            print(
                f"NV {nv_idx}: fit failed"
            )

            return None, [np.nan] * 5

        popt = best_result.x

        amp, freq, decay, phase, baseline = popt

        rabi_period = (
            1.0 / freq
            if freq > epsilon
            else np.nan
        )

        print(
            f"NV {nv_idx}: "
            f"period={rabi_period:.1f} ns, "
            f"freq={freq:.6f} 1/ns, "
            f"decay={decay:.1f} ns, "
            f"score={best_score:.3f}"
        )

        # Function used by your plotting code
        fit_fn = lambda tau: cos_decay(
            np.asarray(tau),
            *popt,
        )

        return fit_fn, popt

    # =========================================================================
    # Parallel fitting
    # =========================================================================

    results = Parallel(
        n_jobs=-1
    )(
        delayed(fit_single_nv)(
            nv_idx
        )
        for nv_idx in range(num_nvs)
    )

    fit_fns, popts = zip(
        *results
    )

    # =========================================================================
    # Median trace
    # =========================================================================

    median_trace = np.nanmedian(
        avg_counts,
        axis=0,
    )

    median_ste = (
        np.sqrt(
            np.nansum(
                avg_counts_ste**2,
                axis=0,
            )
        )
        / num_nvs
    )

    # Fit median using same machinery by temporarily making it one NV.
    def fit_median():

        y = median_trace

        sigma = np.abs(
            median_ste
        )

        sigma[
            ~np.isfinite(sigma)
            | (sigma <= epsilon)
        ] = np.nanmedian(
            sigma[
                np.isfinite(sigma)
                & (sigma > epsilon)
            ]
        )

        dt = np.median(
            np.diff(taus)
        )

        t_span = np.ptp(
            taus
        )

        transform = np.fft.rfft(
            y - np.mean(y)
        )

        fft_freqs = np.fft.rfftfreq(
            len(taus),
            d=dt,
        )

        max_ind = (
            np.argmax(
                np.abs(transform[1:])
            )
            + 1
        )

        freq_guess = fft_freqs[
            max_ind
        ]

        y_span = max(
            np.ptp(y),
            1e-4,
        )

        p0 = [
            0.5 * y_span,
            freq_guess,
            t_span,
            0.0,
            np.mean(y),
        ]

        lower = [
            0,
            epsilon,
            dt / 2,
            -2 * np.pi,
            np.min(y) - y_span,
        ]

        upper = [
            2 * y_span,
            0.95 * 0.5 / dt,
            20 * t_span,
            2 * np.pi,
            np.max(y) + y_span,
        ]

        def res(params):
            return (
                cos_decay(
                    taus,
                    *params,
                )
                - y
            ) / sigma

        try:

            result = least_squares(
                res,
                p0,
                bounds=(
                    lower,
                    upper,
                ),
                loss="soft_l1",
                x_scale="jac",
                max_nfev=20000,
            )

            median_popt = (
                result.x
            )

            median_fit_fn = (
                lambda tau:
                cos_decay(
                    np.asarray(tau),
                    *median_popt,
                )
            )

            print(
                "Median Rabi Period = "
                f"{1 / median_popt[1]:.1f} ns"
            )

            return (
                median_fit_fn,
                median_popt,
            )

        except Exception as exc:

            print(
                "Median fit failed:",
                exc,
            )

            return (
                None,
                [np.nan] * 5,
            )

    median_fit_fn, median_popt = (
        fit_median()
    )

    return (
        fit_fns,
        popts,
        median_fit_fn,
        median_popt,
    )

def plot_rabi_fits(
    nv_list,
    taus,
    avg_counts,
    avg_counts_ste,
    fit_fns,
    popts,
    median_fit_fn=None,
    median_popt=None,
    period_bin_width=8,
    period_round_to=4,
    period_keep_range=(0, 1000),
    save_individual_pdf=True,
    save_histogram_pdf=True,
    show_individual=False,
    show_histogram=True,
    output_label="rabi",
):
    """
    Plot/analyze Rabi fits.

    Outputs
    -------
    1. Rabi-period histogram
       - optionally shown
       - optionally saved as vector PDF

    2. Individual Rabi fits
       - each figure is 8 x 5 inches
       - optionally shown
       - optionally saved together in ONE multipage vector PDF

    Returns
    -------
    filtered_periods : np.ndarray
    filtered_indices : np.ndarray
    """

    taus = np.asarray(taus, dtype=float)
    avg_counts = np.asarray(avg_counts, dtype=float)
    avg_counts_ste = np.asarray(avg_counts_ste, dtype=float)

    num_nvs = len(nv_list)
    epsilon = 1e-10

    rabi_periods = []
    kept_indices = []

    # =========================================================================
    # Extract Rabi periods
    # =========================================================================

    for nv_ind in range(num_nvs):

        try:
            rabi_freq = abs(float(popts[nv_ind][1]))
        except (TypeError, ValueError, IndexError):
            continue

        if not np.isfinite(rabi_freq) or rabi_freq <= epsilon:
            continue

        period = 1.0 / rabi_freq

        if not np.isfinite(period) or period <= 0:
            continue

        if period_round_to is not None and period_round_to > 0:
            period = (
                int(np.round(period / period_round_to))
                * period_round_to
            )

        rabi_periods.append(period)
        kept_indices.append(nv_ind)

    print()
    print("Raw Rabi Periods (ns):")
    print(rabi_periods)

    # =========================================================================
    # Filter periods
    # =========================================================================

    p_f = np.array([], dtype=float)
    idx_f = np.array([], dtype=int)

    if len(rabi_periods) > 0:

        p = np.asarray(rabi_periods, dtype=float)
        idx = np.asarray(kept_indices, dtype=int)

        # IQR filtering
        q1, q3 = np.percentile(p, [25, 75])
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (p >= lower) & (p <= upper)

        # Optional absolute period range
        if period_keep_range is not None:
            lo, hi = period_keep_range
            mask &= (p >= lo) & (p <= hi)

        p_f = p[mask]
        idx_f = idx[mask]

        print()
        print("Filtered Rabi Periods (ns):")
        print(p_f.tolist())

        print()
        print("Filtered idx:")
        print(idx_f.tolist())

        print(
            f"\nNumber before/after filtering: "
            f"{len(p)}/{len(p_f)}"
        )

        if p_f.size > 0:
            print(
                f"Median Rabi Period: "
                f"{np.median(p_f):.3f} ns"
            )

    # =========================================================================
    # Rabi-period histogram
    # =========================================================================

    if p_f.size > 0:

        period_span = p_f.max() - p_f.min()

        if period_span == 0:
            nbins = 5
        else:
            nbins = max(
                5,
                int(
                    np.ceil(
                        period_span
                        / max(1, period_bin_width)
                    )
                ),
            )

        fig_h, ax_h = plt.subplots(
            figsize=(6, 5)
        )

        ax_h.hist(
            p_f,
            bins=nbins,
        )

        ax_h.axvline(
            np.median(p_f),
            linestyle="--",
            linewidth=1.5,
            label=f"Median = {np.median(p_f):.0f} ns",
        )

        ax_h.set_title(
            "Rabi Periods",
            fontsize=15,
        )

        ax_h.set_xlabel(
            "Rabi Period (ns)",
            fontsize=15,
        )

        ax_h.set_ylabel(
            "Number of Occurrences",
            fontsize=15,
        )

        ax_h.tick_params(
            axis="both",
            labelsize=12,
        )

        ax_h.grid(
            True,
            alpha=0.3,
        )

        ax_h.legend()

        # ---------------------------------------------------------------------
        # Save histogram as VECTOR PDF
        # ---------------------------------------------------------------------

        if save_histogram_pdf:

            timestamp = dm.get_time_stamp()

            hist_path = Path(
                dm.get_file_path(
                    __file__,
                    timestamp,
                    f"{output_label}-rabi-period-histogram",
                )
            )

            # dm.get_file_path may produce ".txt".
            # Replace suffix instead of appending ".pdf".
            hist_path = hist_path.with_suffix(".pdf")

            # Make sure directory exists.
            hist_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fig_h.savefig(
                hist_path,
                format="pdf",
                bbox_inches="tight",
            )

            print()
            print(
                "Saved Rabi-period histogram:"
            )
            print(hist_path)

        if show_histogram:
            plt.show(block=False)
        else:
            plt.close(fig_h)

    else:

        print(
            "\nNo valid Rabi periods remain; "
            "histogram skipped."
        )

    # =========================================================================
    # Prepare multipage individual-Rabi PDF
    # =========================================================================

    pdf = None
    pdf_path = None

    if save_individual_pdf:

        timestamp = dm.get_time_stamp()

        pdf_path = Path(
            dm.get_file_path(
                __file__,
                timestamp,
                f"{output_label}-individual-rabi-fits",
            )
        )

        # Replace .txt rather than creating .txt.pdf
        pdf_path = pdf_path.with_suffix(".pdf")

        pdf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf = PdfPages(
            pdf_path
        )

    # Dense time axis for fits
    tau_dense = np.linspace(
        float(np.min(taus)),
        float(np.max(taus)),
        300,
    )

    # =========================================================================
    # Individual Rabi fits
    # =========================================================================

    for nv_ind in range(num_nvs):

        # Same figure size for every Rabi plot
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        y = np.asarray(
            avg_counts[nv_ind],
            dtype=float,
        )

        yerr = np.abs(
            np.asarray(
                avg_counts_ste[nv_ind],
                dtype=float,
            )
        )

        # ---------------------------------------------------------------------
        # Data
        # ---------------------------------------------------------------------

        ax.errorbar(
            taus,
            y,
            yerr=yerr,
            fmt="o",
            markersize=4,
            capsize=2,
        )

        # ---------------------------------------------------------------------
        # Fit
        # ---------------------------------------------------------------------

        if fit_fns[nv_ind] is not None:

            try:

                ax.plot(
                    tau_dense,
                    fit_fns[nv_ind](
                        tau_dense
                    ),
                    "-",
                    linewidth=1.5,
                )

            except Exception:
                pass

        # ---------------------------------------------------------------------
        # Rabi period
        # ---------------------------------------------------------------------

        period_str = "N/A"

        try:

            rabi_freq = abs(
                float(
                    popts[nv_ind][1]
                )
            )

            if (
                np.isfinite(rabi_freq)
                and rabi_freq > epsilon
            ):

                period = (
                    1.0 / rabi_freq
                )

                if (
                    period_round_to
                    is not None
                    and period_round_to > 0
                ):

                    period = (
                        int(
                            np.round(
                                period
                                / period_round_to
                            )
                        )
                        * period_round_to
                    )

                period_str = (
                    f"{period:.0f} ns"
                )

        except Exception:
            pass

        # ---------------------------------------------------------------------
        # Formatting
        # ---------------------------------------------------------------------

        ax.set_title(
            f"NV {nv_ind} "
            f"(Rabi: {period_str})",
            fontsize=13,
        )

        ax.set_xlabel(
            "Pulse Duration (ns)",
            fontsize=12,
        )

        ax.set_ylabel(
            "Norm. NV$^{-}$ Population",
            fontsize=12,
        )

        ax.tick_params(
            axis="both",
            labelsize=11,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        fig.tight_layout()

        # ---------------------------------------------------------------------
        # Save page into combined VECTOR PDF
        # ---------------------------------------------------------------------

        if pdf is not None:

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

        # ---------------------------------------------------------------------
        # Optional display
        # ---------------------------------------------------------------------

        if show_individual:

            plt.show(
                block=True
            )

        else:

            plt.close(fig)

    # =========================================================================
    # Close multipage PDF
    # =========================================================================

    if pdf is not None:

        pdf.close()

        print()
        print(
            "Saved individual Rabi fits PDF:"
        )
        print(pdf_path)

    return p_f, idx_f



def plot_rabi_fits(
    nv_list,
    taus,
    avg_counts,
    avg_counts_ste,
    fit_fns,
    popts,
    median_fit_fn=None,
    median_popt=None,
    period_bin_width=8,
    period_round_to=4,
    period_keep_range=(0, 1000),

    # Saving
    save_individual_pdf=True,
    save_histogram_pdf=True,
    save_histogram_png=True,

    # Display
    show_individual=False,
    show_histogram=True,

    # Multipage PDF layout
    pdf_cols=3,
    pdf_rows=4,

    output_label="rabi",
):
    """
    Analyze and plot Rabi fits.

    Outputs
    -------
    - Rabi-period histogram
        * optional vector PDF
        * optional 300-dpi PNG

    - Individual Rabi fits
        * combined into ONE multipage vector PDF
        * default = 3 columns x 4 rows = 12 NVs/page

    Returns
    -------
    p_f   : filtered Rabi periods
    idx_f : corresponding original NV indices
    """

    taus = np.asarray(taus, dtype=float)
    avg_counts = np.asarray(avg_counts, dtype=float)
    avg_counts_ste = np.asarray(avg_counts_ste, dtype=float)

    num_nvs = len(nv_list)
    epsilon = 1e-10

    rabi_periods = []
    kept_indices = []

    # =========================================================================
    # Extract Rabi periods
    # =========================================================================

    for nv_ind in range(num_nvs):

        try:
            rabi_freq = abs(float(popts[nv_ind][1]))
        except (TypeError, ValueError, IndexError):
            continue

        if not np.isfinite(rabi_freq) or rabi_freq <= epsilon:
            continue

        period = 1.0 / rabi_freq

        if not np.isfinite(period) or period <= 0:
            continue

        if period_round_to is not None and period_round_to > 0:
            period = (
                int(np.round(period / period_round_to))
                * period_round_to
            )

        rabi_periods.append(period)
        kept_indices.append(nv_ind)

    print()
    print("Raw Rabi Periods (ns):")
    print(rabi_periods)

    # =========================================================================
    # Filter Rabi periods
    # =========================================================================

    p_f = np.array([], dtype=float)
    idx_f = np.array([], dtype=int)

    if len(rabi_periods) > 0:

        p = np.asarray(rabi_periods, dtype=float)
        idx = np.asarray(kept_indices, dtype=int)

        # IQR filter
        q1, q3 = np.percentile(p, [25, 75])
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (p >= lower) & (p <= upper)

        # Optional absolute range
        if period_keep_range is not None:
            lo, hi = period_keep_range
            mask &= (p >= lo) & (p <= hi)

        p_f = p[mask]
        idx_f = idx[mask]

        print()
        print("Filtered Rabi Periods (ns):")
        print(p_f.tolist())

        print()
        print("Filtered idx:")
        print(idx_f.tolist())

        print(
            f"\nNumber before/after filtering: "
            f"{len(p)}/{len(p_f)}"
        )

        if p_f.size > 0:
            print(
                f"Median Rabi Period: "
                f"{np.median(p_f):.3f} ns"
            )

    # =========================================================================
    # Histogram
    # =========================================================================

    if p_f.size > 0:

        period_span = p_f.max() - p_f.min()

        if period_span == 0:
            nbins = 5
        else:
            nbins = max(
                5,
                int(
                    np.ceil(
                        period_span
                        / max(1, period_bin_width)
                    )
                ),
            )

        fig_h, ax_h = plt.subplots(
            figsize=(6, 5)
        )

        ax_h.hist(
            p_f,
            bins=nbins,
        )

        median_period = np.median(p_f)

        ax_h.axvline(
            median_period,
            linestyle="--",
            linewidth=1.5,
            label=f"Median = {median_period:.0f} ns",
        )

        ax_h.set_title(
            "Rabi Periods",
            fontsize=15,
        )

        ax_h.set_xlabel(
            "Rabi Period (ns)",
            fontsize=15,
        )

        ax_h.set_ylabel(
            "Number of Occurrences",
            fontsize=15,
        )

        ax_h.tick_params(
            axis="both",
            labelsize=12,
        )

        ax_h.grid(
            True,
            alpha=0.3,
        )

        ax_h.legend()

        fig_h.tight_layout()

        # ---------------------------------------------------------------------
        # Save histogram PDF + PNG
        # ---------------------------------------------------------------------

        if save_histogram_pdf or save_histogram_png:

            timestamp = dm.get_time_stamp()

            base_path = Path(
                dm.get_file_path(
                    __file__,
                    timestamp,
                    f"{output_label}-rabi-period-histogram",
                )
            )

            base_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if save_histogram_pdf:

                hist_pdf_path = base_path.with_suffix(
                    ".pdf"
                )

                fig_h.savefig(
                    hist_pdf_path,
                    format="pdf",
                    bbox_inches="tight",
                )

                print()
                print(
                    "Saved histogram PDF:"
                )
                print(hist_pdf_path)

            if save_histogram_png:

                hist_png_path = base_path.with_suffix(
                    ".png"
                )

                fig_h.savefig(
                    hist_png_path,
                    format="png",
                    dpi=300,
                    bbox_inches="tight",
                )

                print(
                    "Saved histogram PNG:"
                )
                print(hist_png_path)

        if show_histogram:
            plt.show(block=False)
        else:
            plt.close(fig_h)

    else:

        print(
            "\nNo valid Rabi periods remain "
            "after filtering."
        )

    # =========================================================================
    # Multipage grid PDF of individual Rabi fits
    # =========================================================================

    if save_individual_pdf:

        timestamp = dm.get_time_stamp()

        pdf_path = Path(
            dm.get_file_path(
                __file__,
                timestamp,
                f"{output_label}-individual-rabi-fits",
            )
        ).with_suffix(".pdf")

        pdf_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plots_per_page = (
            pdf_cols * pdf_rows
        )

        tau_dense = np.linspace(
            float(np.min(taus)),
            float(np.max(taus)),
            300,
        )

        # Each subplot remains reasonably large.
        page_width = 5.0 * pdf_cols
        page_height = 3.4 * pdf_rows

        with PdfPages(pdf_path) as pdf:

            for start in range(
                0,
                num_nvs,
                plots_per_page,
            ):

                stop = min(
                    start + plots_per_page,
                    num_nvs,
                )

                page_inds = list(
                    range(start, stop)
                )

                fig, axes = plt.subplots(
                    pdf_rows,
                    pdf_cols,
                    figsize=(
                        page_width,
                        page_height,
                    ),
                    squeeze=False,
                )

                axes = axes.ravel()

                # -------------------------------------------------------------
                # Plot NVs on this page
                # -------------------------------------------------------------

                for slot, nv_ind in enumerate(
                    page_inds
                ):

                    ax = axes[slot]

                    y = np.asarray(
                        avg_counts[nv_ind],
                        dtype=float,
                    )

                    yerr = np.abs(
                        np.asarray(
                            avg_counts_ste[nv_ind],
                            dtype=float,
                        )
                    )

                    # Data
                    ax.errorbar(
                        taus,
                        y,
                        yerr=yerr,
                        fmt="o",
                        markersize=2.8,
                        capsize=1.5,
                        linewidth=0.7,
                    )

                    # Fit
                    if fit_fns[nv_ind] is not None:

                        try:

                            ax.plot(
                                tau_dense,
                                fit_fns[nv_ind](
                                    tau_dense
                                ),
                                "-",
                                linewidth=1.2,
                            )

                        except Exception:
                            pass

                    # ---------------------------------------------------------
                    # Rabi period
                    # ---------------------------------------------------------

                    period_str = "N/A"

                    try:

                        rabi_freq = abs(
                            float(
                                popts[nv_ind][1]
                            )
                        )

                        if (
                            np.isfinite(
                                rabi_freq
                            )
                            and rabi_freq
                            > epsilon
                        ):

                            period = (
                                1.0
                                / rabi_freq
                            )

                            if (
                                period_round_to
                                is not None
                                and period_round_to
                                > 0
                            ):

                                period = (
                                    int(
                                        np.round(
                                            period
                                            / period_round_to
                                        )
                                    )
                                    * period_round_to
                                )

                            period_str = (
                                f"{period:.0f} ns"
                            )

                    except Exception:
                        pass

                    # ---------------------------------------------------------
                    # Formatting
                    # ---------------------------------------------------------

                    ax.set_title(
                        f"NV {nv_ind}  |  "
                        f"Rabi {period_str}",
                        fontsize=9,
                    )

                    ax.set_xlabel(
                        "Pulse Duration (ns)",
                        fontsize=8,
                    )

                    ax.set_ylabel(
                        "Norm. NV$^{-}$ Pop.",
                        fontsize=8,
                    )

                    ax.tick_params(
                        axis="both",
                        labelsize=7,
                    )

                    ax.grid(
                        True,
                        alpha=0.25,
                    )

                # -------------------------------------------------------------
                # Hide empty slots on last page
                # -------------------------------------------------------------

                for slot in range(
                    len(page_inds),
                    len(axes),
                ):
                    axes[slot].axis("off")

                # -------------------------------------------------------------
                # Page title
                # -------------------------------------------------------------

                fig.suptitle(
                    f"Rabi Fits — "
                    f"NV {page_inds[0]} to "
                    f"{page_inds[-1]}",
                    fontsize=14,
                    y=0.995,
                )

                fig.tight_layout(
                    rect=[
                        0,
                        0,
                        1,
                        0.975,
                    ]
                )

                # Vector PDF page
                pdf.savefig(
                    fig,
                    bbox_inches="tight",
                )

                if show_individual:
                    plt.show(
                        block=False
                    )

                plt.close(fig)

        print()
        print(
            "Saved combined Rabi-fit PDF:"
        )
        print(pdf_path)

    # =========================================================================
    # Optional display without PDF saving
    # =========================================================================

    elif show_individual:

        tau_dense = np.linspace(
            float(np.min(taus)),
            float(np.max(taus)),
            300,
        )

        for nv_ind in range(num_nvs):

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            y = avg_counts[nv_ind]
            yerr = np.abs(
                avg_counts_ste[nv_ind]
            )

            ax.errorbar(
                taus,
                y,
                yerr=yerr,
                fmt="o",
            )

            if fit_fns[nv_ind] is not None:
                try:
                    ax.plot(
                        tau_dense,
                        fit_fns[nv_ind](
                            tau_dense
                        ),
                        "-",
                    )
                except Exception:
                    pass

            ax.set_title(
                f"NV {nv_ind}"
            )

            ax.set_xlabel(
                "Pulse Duration (ns)"
            )

            ax.set_ylabel(
                "Norm. NV$^{-}$ Population"
            )

            ax.grid(True)

            fig.tight_layout()

            plt.show(
                block=True
            )

    return p_f, idx_f

def remove_outliers(data):
    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"({Q1, Q3, IQR, lower_bound, upper_bound})")
    return (data >= lower_bound) & (data <= upper_bound)


if __name__ == "__main__":
    kpl.init_kplotlib()
    ### combine
    # # Combine, remove duplicates, sort
    # list1 = [0, 1, 2, 3, 10, 14, 17, 18, 19, 26, 31, 32, 36, 37, 38, 41, 43, 46, 47, 50, 53, 54, 55, 56, 59, 62, 63, 65, 72, 73, 76, 77, 78, 80, 81, 82, 83, 85, 89, 92, 94, 98, 101, 102, 104, 105, 106, 107, 108, 109, 110, 112, 113, 115, 116, 124, 125, 126, 128, 136, 137, 140, 142, 143, 145, 146, 147, 148, 149, 150, 152, 154, 155, 156, 157, 159, 164, 166, 170, 171, 177, 179, 180, 181, 182, 186, 188, 189, 191, 193, 194, 198, 200, 202, 203, 204, 205, 206, 207, 208, 212, 214, 216, 218, 219, 222, 223, 227, 228, 230, 231, 232, 233, 236, 237, 239, 241, 242, 243, 246, 248, 249, 250, 252, 253, 254, 255, 258, 261, 262, 264, 265, 266, 270, 274, 275, 276, 277, 280, 281, 283, 286, 288, 289, 291, 293, 294, 295, 296, 299, 301, 303, 305, 306, 307]
    # list2 = [2, 3, 4, 5, 6, 12, 15, 21, 23, 24, 28, 29, 31, 33, 34, 39, 40, 42, 43, 48, 49, 51, 52, 54, 57, 60, 63, 64, 66, 68, 69, 70, 72, 74, 79, 82, 84, 85, 87, 88, 89, 90, 94, 95, 96, 97, 100, 102, 103, 111, 117, 118, 121, 127, 129, 130, 131, 132, 133, 134, 135, 138, 139, 141, 151, 158, 160, 162, 163, 165, 167, 169, 173, 174, 180, 181, 184, 189, 190, 192, 201, 210, 211, 213, 215, 217, 220, 221, 223, 225, 226, 229, 231, 234, 235, 238, 240, 243, 244, 245, 251, 254, 256, 257, 259, 260, 261, 264, 265, 266, 267, 270, 271, 275, 282, 285, 287, 288, 290, 292, 296, 297, 298, 300, 302, 304, 306, 307]
    # avg_snr = ['0.018', '0.062', '0.057', '0.080', '0.107', '-0.001', '0.089', '0.106', '0.131', '0.076', '0.105', '0.089', '0.063', '0.000', '0.097', '0.075', '0.047', '0.058', '0.009', '0.055', '0.097', '0.011', '0.054', '0.081', '0.046', '0.139', '0.064', '0.015', '0.112', '0.052', '0.076', '0.090', '0.069', '0.077', '0.025', '0.015', '0.100', '0.024', '0.001', '0.066', '0.049', '0.061', '0.079', '0.035', '0.026', '0.094', '0.061', '0.100', '0.098', '0.069', '0.099', '0.137', '0.029', '0.036', '0.042', '0.063', '0.097', '0.068', '0.088', '0.022', '0.112', '0.075', '0.123', '0.098', '0.136', '0.061', '0.061', '0.034', '0.072', '0.094', '0.002', '0.052', '0.080', '0.077', '0.141', '0.092', '0.090', '0.031', '0.074', '0.062', '0.112', '0.083', '0.067', '0.048', '0.082', '0.062', '0.045', '0.030', '0.050', '0.093', '-0.004', '0.076', '0.123', '0.101', '0.075', '0.052', '0.105', '0.064', '0.093', '0.071', '0.082', '0.097', '0.025', '0.020', '0.028', '0.080', '0.080', '0.092', '0.063', '0.083', '0.065', '0.075', '0.147', '0.019', '0.030', '0.050', '0.006', '0.108', '0.095', '0.070', '0.036', '0.092', '0.150', '0.011', '0.105', '0.017', '0.058', '0.013', '0.096', '0.082', '0.101', '0.088', '0.056', '0.060', '0.099', '0.088', '0.020', '0.100', '0.077', '0.020', '0.109', '0.081', '0.092', '0.113', '0.064', '0.039', '0.041', '0.044', '0.110', '0.037', '0.143']
    # # Target 2.77 GHz -> NV indices
    # list1 = [4, 10, 12, 14, 15, 18, 20, 22, 23, 27, 28, 29, 34, 40, 45, 49, 50, 51, 54, 59, 60, 62, 68, 69, 72, 73, 79, 86, 94, 95, 99, 101, 102, 104, 106, 107, 111, 113, 117, 122, 123, 128, 130, 131, 133, 134, 136, 144, 151, 158, 167, 172, 174, 178, 183, 186, 191, 193, 197, 200, 207, 210, 220, 221, 229, 233, 235, 237, 238, 244, 246, 250, 252, 253]
    # # Target 2.82 GHz -> NV indices
    # list2 = [0, 1, 2, 9, 17, 24, 30, 33, 37, 41, 42, 46, 55, 57, 58, 65, 67, 75, 78, 80, 81, 82, 85, 87, 88, 90, 98, 114, 116, 119, 125, 126, 127, 135, 137, 142, 143, 145, 146, 148, 149, 153, 155, 161, 163, 164, 165, 166, 170, 173, 175, 181, 185, 187, 192, 195, 196, 199, 201, 203, 205, 211, 212, 214, 216, 218, 223, 225, 226, 227, 228, 230, 239, 242, 245, 247, 249]
    # combined_sorted = sorted(set(list1 + list2))
    # print(combined_sorted)
    # print(len(combined_sorted))
    # # Convert string list to floats
    # avg_snr = [float(x) for x in avg_snr]

    # # Compute averages for each target list
    # avg_list1 = np.mean([avg_snr[i] for i in list1 if i < len(avg_snr)])
    # avg_list2 = np.mean([avg_snr[i] for i in list2 if i < len(avg_snr)])

    # # Combined list (unique + sorted)
    # combined_sorted = sorted(set(list1 + list2))
    # avg_combined = np.mean([avg_snr[i] for i in combined_sorted if i < len(avg_snr)])

    # print("Target 2.77 GHz -> average SNR:", avg_list1)
    # print("Target 2.82 GHz -> average SNR:", avg_list2)
    # print("Combined -> average SNR:", avg_combined)
    # print("Total combined indices:", len(combined_sorted))
    # sys.exit()

    # file_id = 1772297872545  # two orientations with freqsa round 2.79
    # file_id = 1772755741220  # two orientations with freqs aroud 2.84
    # file_id = 1774582403511  # all four orientation measured with two frequency tone per sig gen
    # file_id = 1775776922337  # all four orientation measured with two frequency tone per sig gen with offset pulses both microwaave ()
    # rubin
    # file_id = 1775776922337  # all four orientation measured with two frequency tone per sig gen with offset pulses both microwaave ()
    # file_id = 1779670263899
    # rubin sample
    # file_id = 1795718888560
    # file_id = 1796958071866
    # file_id = 1795718888560
    # file_id = 1796958071866
    # 300 NVs
    # file_id = 1803593992080
    # file_id = 1804466558303
    # file_id = 1817818887926  # 75NVs iq modulation test 68Mhz orientation

    # After changing magnet postion
    # file_id = 1833635613442  # i channel
    # 75NVs iq modulation both degenerate orientation
    # file_id = 1832587019842  # q channel
    # file_id = 1842383067959  # i channel
    # file_stem = box_cloud.get_file_stem_from_file_id(file_id)
    # data = dm.get_raw_data(file_id=file_id, load_npz=False, use_cache=False)
    # file_stem = "2025_04_30-07_09_33-rubin-nv0_2025_02_26"
    # file_stem = "2025_09_21-04_35_06-rubin-nv0_2025_09_08"
    # file_stem = "2025_10_02-05_57_27-rubin-nv0_2025_09_08"
    # file_stem = ["2025_10_05-20_06_59-rubin-nv0_2025_09_08"]
    # file_stem = ["2025_10_06-03_26_08-rubin-nv0_2025_09_08"] ## 2.76, 2.84
    # file_stem = ["2025_10_06-21_18_40-rubin-nv0_2025_09_08"] ## 2.78, 2.82

    ##133 MHz deer
    # file_stem = ["2025_10_13-20_49_30-rubin-nv0_2025_09_08"] ## deer
    # indices_113_MHz = [0, 1, 3, 6, 10, 14, 16, 17, 19, 23, 24, 25, 26, 27, 32, 33, 34, 35, 37, 38, 41, 49, 50, 51, 53, 54, 55, 60, 62, 63, 64, 66, 67, 68, 70, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 86, 88, 90, 92, 93, 95, 96, 99, 100, 101, 102, 103, 105, 108, 109, 111, 113, 114]
    ### johnson
    # file_stem = ["2025_10_24-03_41_53-johnson-nv0_2025_10_21"] ## 2.78, 2.84
    # file_stem = ["2025_10_24-23_14_38-johnson-nv0_2025_10_21"] ## 2.78, 2.84
    # file_stem = ["2025_10_25-05_55_59-johnson-nv0_2025_10_21"] ## 2.78, 2.84
    # file_stem = ["2025_10_28-09_07_10-johnson-nv0_2025_10_21"]
    # file_stem = ["2025_11_13-05_41_07-johnson-nv0_2025_10_21"]  ## 2.91, 2.94

    # ##### Testing flexible loop
    # file_stem = ["2026_02_02-10_41_39-johnson-nv0_2025_10_21",
    #              "2026_02_02-13_49_00-johnson-nv0_2025_10_21",
    #              ]  ## 2.77
    # file_stem = ["2026_02_02-17_08_15-johnson-nv0_2025_10_21"]  ## 2.91, 2.94
    # file_stem = ["2026_02_02-21_30_27-johnson-nv0_2025_10_21"]
    # file_stem = ["2026_02_03-04_16_43-johnson-nv0_2025_10_21"]

    ##148 MHz deer
    # file_stem =  ["2026_02_04-14_02_35-johnson-nv0_2025_10_21"]
    file_stem = ["2026_02_02-21_30_27-johnson-nv0_2025_10_21"] 
    
    ##qnami array sample
    file_stem = ["2026_02_20-04_26_28-rubin-nv0_2026_02_15"]  ### loop 
    # file_stem = ["2026_02_22-00_11_44-rubin-nv0_2026_02_15"]  ### resonator 
    
    ##qnami array sample 1277
    file_stem = ["2026_03_28-07_59_17-qnami-nv0_2026_02_20"]  ### loop     
    
    ##qnami array sample 1277
    file_stem = ["2026_09_08-20_08_21-qnami-nv0_2026_02_20"]  ### loop     
         
    ##qnami array sample 631
    file_stem = ["2026_09_11-13_46_26-qnami-nv0_2026_02_20"]  ### loop     

    ##qnami array sample 631
    file_stem = ["2026_09_12-02_21_34-qnami-nv0_2026_02_20"]  ### loop
    
    ##qnami array sample 631
    file_stem = ["2026_09_16-08_26_27-qnami-nv0_2026_02_20"]  ### loop          

    data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
        use_cache=False,
    )

    nv_list = data["nv_list"]
    taus = data["taus"]

    counts = np.asarray(
        data["counts"]
    )

    sig_counts = counts[0]
    ref_counts = counts[1]

    avg_counts, avg_counts_ste = (
        widefield.process_counts(
            nv_list,
            sig_counts,
            ref_counts,
            threshold=True,
        )
    )

    # ============================================================================
    # Fit Rabi data
    # ============================================================================

    fit_fns, popts, median_fit_fn, median_popt = fit_rabi_data(
        nv_list,
        taus,
        avg_counts,
        avg_counts_ste,
    )

    # ============================================================================
    # Output label
    # ============================================================================

    if isinstance(
        file_stem,
        (list, tuple),
    ):
        output_label = file_stem[-1]
    else:
        output_label = file_stem

    # ============================================================================
    # Plot + save
    # ============================================================================

    rabi_periods, filtered_inds = plot_rabi_fits(
        nv_list,
        taus,
        avg_counts,
        avg_counts_ste,
        fit_fns,
        popts,
        median_fit_fn,
        median_popt,

        # Histogram
        period_bin_width=8,
        period_round_to=4,
        period_keep_range=(0, 1000),

        # Save
        save_individual_pdf=True,
        save_histogram_pdf=True,
        save_histogram_png=True,

        # Display
        show_individual=False,
        show_histogram=True,

        # PDF layout
        pdf_cols=3,
        pdf_rows=4,

        output_label=output_label,
    )

    print()
    print("Filtered idx:")
    print(filtered_inds.tolist())

kpl.show(block=True)
