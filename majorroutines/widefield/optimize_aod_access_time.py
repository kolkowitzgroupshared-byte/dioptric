# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023

@author: mccambria
updated by @Saroj Chand on Marrch 21st 2025
@author: mccambria
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

def process_and_plot(nv_list, taus, sig_counts, ref_counts, median_band="iqr"):
    """
    Plots per-NV raw signal/ref counts, per-NV SNR, and summary across NVs:
      - mean SNR across NVs
      - median SNR across NVs (with a robust band)

    Args:
        nv_list: list of NV identifiers (whatever widefield expects)
        taus: 1D array-like (x-axis)
        sig_counts, ref_counts: arrays consumed by widefield.average_counts / widefield.calc_snr
        median_band: "iqr" (default) or "mad_sem" for median uncertainty

    Returns:
        (sig_fig, ref_fig, snr_fig, mean_snr_fig, median_snr_fig, fit_fig)
    """
    taus = np.asarray(taus, dtype=float)
    num_nvs = len(nv_list)

    # --- Averages + SNR (your existing pipeline) ---
    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    xlabel = "aod_access_time (us)"  # adjust if taus are ns

    # --- Signal plot ---
    sig_fig, sig_ax = plt.subplots()
    widefield.plot_raw_data(sig_ax, nv_list, taus, avg_sig_counts, avg_sig_counts_ste)
    sig_ax.set_xlabel(xlabel)
    sig_ax.set_ylabel("Signal counts")
    sig_ax.set_title("Signal (per NV)")

    # --- Reference plot ---
    ref_fig, ref_ax = plt.subplots()
    widefield.plot_raw_data(ref_ax, nv_list, taus, avg_ref_counts, avg_ref_counts_ste)
    ref_ax.set_xlabel(xlabel)
    ref_ax.set_ylabel("Reference counts")
    ref_ax.set_title("Reference (per NV)")

    # --- Per-NV SNR plot ---
    snr_fig, snr_ax = plt.subplots()
    widefield.plot_raw_data(snr_ax, nv_list, taus, avg_snr, avg_snr_ste)
    snr_ax.set_xlabel(xlabel)
    snr_ax.set_ylabel("SNR")
    snr_ax.set_title("SNR (per NV)")

    # --- Mean across NVs (your current "Average across NVs" but cleaned) ---
    mean_snr_fig, mean_snr_ax = plt.subplots()
    mean_snr = np.mean(avg_snr, axis=0)
    kpl.plot_points(mean_snr_ax, taus, mean_snr, yerr=None)
    mean_snr_ax.set_xlabel(xlabel)
    mean_snr_ax.set_ylabel("Mean SNR")
    mean_snr_ax.set_title("Mean SNR across NVs")

    # --- Median across NVs (requested) ---
    median_snr_fig, median_snr_ax = plt.subplots()
    med_snr = np.median(avg_snr, axis=0)

    # Default: IQR band (25–75%), robust and easy to interpret
    if median_band == "iqr":
        q25 = np.quantile(avg_snr, 0.25, axis=0)
        q75 = np.quantile(avg_snr, 0.75, axis=0)
        kpl.plot_points(median_snr_ax, taus, med_snr, yerr=None)
        median_snr_ax.fill_between(taus, q25, q75, alpha=0.25, linewidth=0)
        median_snr_ax.set_title("Median SNR across NVs (IQR band 25–75%)")

    # Optional: MAD-based SEM-ish errorbars for the median
    elif median_band == "mad_sem":
        mad = np.median(np.abs(avg_snr - med_snr[None, :]), axis=0)
        robust_sigma = 1.4826 * mad
        robust_sem = robust_sigma / np.sqrt(avg_snr.shape[0])
        kpl.plot_points(median_snr_ax, taus, med_snr, yerr=robust_sem)
        median_snr_ax.set_title("Median SNR across NVs (MAD/√N errorbars)")
    else:
        raise ValueError("median_band must be 'iqr' or 'mad_sem'")

    median_snr_ax.set_xlabel(xlabel)
    median_snr_ax.set_ylabel("Median SNR")

    return sig_fig, ref_fig, snr_fig, mean_snr_fig, median_snr_fig



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
        file_stem="2026_01_20-21_13_29-johnson-nv0_2025_10_21", load_npz=True
    )

    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    # sig_counts, ref_counts = widefield.threshold_counts(nv_list, sig_counts, ref_counts)

    process_and_plot(nv_list, taus, sig_counts, ref_counts)

    plt.show(block=True)
