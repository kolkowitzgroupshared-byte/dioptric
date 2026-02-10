# -*- coding: utf-8 -*-
"""
Optimize aod parameters
Created by @Saroj Chand on Jan 21st 2026
@author: sbchand
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt


def process_and_plot(
    nv_list,
    taus_ns,
    sig_counts,
    ref_counts,
    median_band="iqr",  # "iqr" or "mad_sem"
    max_nvs_to_plot=None,  # e.g. 60 to avoid clutter; None -> all
    alpha_per_nv=0.15,
    lw_per_nv=1.0,
    show_points=False,
):
    """
    Manual raw plots (no widefield.plot_raw_data). Converts tau from ns -> us.

    Returns:
        dict with figs/axes and computed arrays.
    """
    taus_ns = np.asarray(taus_ns, dtype=float)
    taus_us = taus_ns / 1e3  # ns -> us

    num_nvs = len(nv_list)
    plot_n = (
        num_nvs if (max_nvs_to_plot is None) else min(int(max_nvs_to_plot), num_nvs)
    )
    nv_inds = np.arange(plot_n, dtype=int)

    # --- compute per-NV means + STE and SNR via your pipeline ---
    avg_sig, ste_sig, _ = widefield.average_counts(sig_counts)  # (Nnv, Ntau)
    avg_ref, ste_ref, _ = widefield.average_counts(ref_counts)  # (Nnv, Ntau)
    avg_snr, ste_snr = widefield.calc_snr(sig_counts, ref_counts)  # (Nnv, Ntau)

    # basic sanity
    if avg_sig.shape[0] != num_nvs:
        raise ValueError(
            f"avg_sig has {avg_sig.shape[0]} NVs but nv_list has {num_nvs}."
        )
    if avg_sig.shape[1] != len(taus_us):
        raise ValueError("taus length doesn't match data's tau axis.")

    xlab = r"AOD access time $\tau$ (µs)"

    def _plot_bundle(ax, x, Y, title, ylab):
        # per-NV lines
        for i in nv_inds:
            ax.plot(x, Y[i], lw=lw_per_nv, alpha=alpha_per_nv)
            if show_points:
                ax.plot(x, Y[i], marker=".", ls="None", alpha=alpha_per_nv)

        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, ls="--", lw=0.5)

    # --- 1) Signal per NV ---
    fig_sig, ax_sig = plt.subplots(figsize=(6, 5))
    _plot_bundle(ax_sig, taus_us, avg_sig, "Signal counts (per NV)", "Signal counts")

    # --- 2) Reference per NV ---
    fig_ref, ax_ref = plt.subplots(figsize=(6, 5))
    _plot_bundle(
        ax_ref, taus_us, avg_ref, "Reference counts (per NV)", "Reference counts"
    )

    # --- 3) SNR per NV ---
    fig_snr, ax_snr = plt.subplots(figsize=(6, 5))
    _plot_bundle(ax_snr, taus_us, avg_snr, "SNR (per NV)", "SNR")

    # --- 4) Mean SNR across NVs ---
    mean_snr = np.mean(avg_snr, axis=0)
    fig_mean, ax_mean = plt.subplots(figsize=(6, 5))
    ax_mean.plot(taus_us, mean_snr, lw=2)
    if show_points:
        ax_mean.plot(taus_us, mean_snr, marker=".", ls="None")
    ax_mean.set_title("Mean SNR across NVs")
    ax_mean.set_xlabel(xlab)
    ax_mean.set_ylabel("Mean SNR")
    ax_mean.grid(True, ls="--", lw=0.5)

    # --- 5) Median SNR across NVs + robust band ---
    med_snr = np.median(avg_snr, axis=0)
    fig_med, ax_med = plt.subplots(figsize=(6, 5))
    ax_med.plot(taus_us, med_snr, lw=2, label="median")

    if median_band == "iqr":
        q25 = np.quantile(avg_snr, 0.25, axis=0)
        q75 = np.quantile(avg_snr, 0.75, axis=0)
        ax_med.fill_between(
            taus_us, q25, q75, alpha=0.25, linewidth=0, label="IQR (25–75%)"
        )
        ax_med.set_title("Median SNR across NVs (IQR band)")
    elif median_band == "mad_sem":
        mad = np.median(np.abs(avg_snr - med_snr[None, :]), axis=0)
        robust_sigma = 1.4826 * mad
        robust_sem = robust_sigma / np.sqrt(avg_snr.shape[0])
        ax_med.fill_between(
            taus_us,
            med_snr - robust_sem,
            med_snr + robust_sem,
            alpha=0.25,
            linewidth=0,
            label=r"MAD/$\sqrt{N}$ band",
        )
        ax_med.set_title(r"Median SNR across NVs (MAD/$\sqrt{N}$ band)")
    else:
        raise ValueError("median_band must be 'iqr' or 'mad_sem'")

    if show_points:
        ax_med.plot(taus_us, med_snr, marker=".", ls="None")
    ax_med.set_xlabel(xlab)
    ax_med.set_ylabel("Median SNR")
    ax_med.grid(True, ls="--", lw=0.5)
    ax_med.legend(loc="best", fontsize=9)

    return {
        "taus_us": taus_us,
        "avg_sig": avg_sig,
        "ste_sig": ste_sig,
        "avg_ref": avg_ref,
        "ste_ref": ste_ref,
        "avg_snr": avg_snr,
        "ste_snr": ste_snr,
        "fig_sig": fig_sig,
        "fig_ref": fig_ref,
        "fig_snr": fig_snr,
        "fig_mean": fig_mean,
        "fig_median": fig_med,
    }


# def optimize_scc_duration(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
#     return _main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau,)


def main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    ### Some initial setup
    uwave_ind_list = [0, 1]

    seq_file = "optimize_aod_access_time.py"

    taus = np.linspace(min_tau, max_tau, num_steps)

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        # print(f"seq_args before encoding: {seq_args}")
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
    )

    # save data
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": min_tau,
        "max_tau": max_tau,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    ### Process and plot
    counts = raw_data["counts"]
    sig_counts = counts[0]
    ref_counts = counts[1]

    ### Process and plot
    try:
        figs = process_and_plot(nv_list, taus, sig_counts, ref_counts)
    except Exception:
        print(traceback.format_exc())
        figs = None

    ### Clean up and return
    tb.reset_cfm()

    kpl.show()

    if figs is not None:
        for ind in range(len(figs)):
            fig = figs[ind]
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{ind}")
            dm.save_figure(fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # data = dm.get_raw_data(file_id=1564881159891)
    # data = dm.get_raw_data(file_id=1720799193270)
    data = dm.get_raw_data(
        file_stem="2026_01_21-02_29_31-johnson-nv0_2025_10_21", load_npz=True
    )

    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    # sig_counts, ref_counts = widefield.threshold_counts(nv_list, sig_counts, ref_counts)

    # process_and_plot(nv_list, taus, sig_counts, ref_counts)
    out = process_and_plot(
        nv_list,
        taus,  # ns
        sig_counts,
        ref_counts,
        median_band="iqr",
        # max_nvs_to_plot=80,  # avoid spaghetti
        alpha_per_nv=0.6,
    )

    plt.show(block=True)
