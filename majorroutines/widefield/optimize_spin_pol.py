# -*- coding: utf-8 -*-
"""
Optimize SCC parameters

Created on December 6th, 2023
@author: mccambria
updated by @SarojChand on August 6th, 2025
@author: schand
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


def process_and_plot(nv_list, taus, sig_counts, ref_counts, duration_or_amp):
    num_nvs = len(nv_list)

    # sig_counts, ref_counts = determine_threshold(nv_list, sig_counts, ref_counts)

    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    # avg_snr_ste = None

    xlabel = (
        "SCC pulse duration (ns)" if duration_or_amp else "SCC relative AOD amplitude"
    )

    sig_fig, sig_ax = plt.subplots()
    widefield.plot_raw_data(sig_ax, nv_list, taus, avg_sig_counts, avg_sig_counts_ste)
    sig_ax.set_xlabel(xlabel)
    sig_ax.set_ylabel("Signal counts")

    ref_fig, ref_ax = plt.subplots()
    widefield.plot_raw_data(ref_ax, nv_list, taus, avg_ref_counts, avg_ref_counts_ste)
    ref_ax.set_xlabel(xlabel)
    ref_ax.set_ylabel("Reference counts")

    snr_fig, snr_ax = plt.subplots()
    widefield.plot_raw_data(snr_ax, nv_list, taus, avg_snr, avg_snr_ste)
    snr_ax.set_xlabel(xlabel)
    snr_ax.set_ylabel("SNR")

    for ind in range(num_nvs):
        fig, ax = plt.subplots()
        kpl.plot_points(ax, taus, avg_snr[ind], yerr=avg_snr_ste[ind])
        ax.set_title(ind)
        plt.show(block=True)

    # Average across NVs
    avg_snr_fig, avg_snr_ax = plt.subplots()
    # avg_avg_snr = np.quantile(avg_snr, 0.75, axis=0)
    # avg_avg_snr_ste = np.quantile(avg_snr_ste, 0.75, axis=0)
    avg_avg_snr = np.mean(avg_snr, axis=0)
    avg_avg_snr_ste = None
    kpl.plot_points(avg_snr_ax, taus, avg_avg_snr, yerr=avg_avg_snr_ste)
    avg_snr_ax.set_xlabel("Ionization pulse duration (ns)")
    avg_snr_ax.set_ylabel("Average SNR")

    # Fits and optimum values
    def fit_fn(tau, delay, slope, dec):
        tau = np.array(tau) - delay
        return slope * tau * np.exp(-tau / dec)

    fit_fig, fit_ax = plt.subplots()
    opti_snrs = []
    opti_durations = []
    for nv_ind in range(num_nvs):
        nv_sig = nv_list[nv_ind]
        opti_snr = np.max(avg_snr[nv_ind])
        opti_duration = taus[np.argmax(avg_snr[nv_ind])]
        guess_params = [20, opti_snr / opti_duration, 300]
        avg_snr_nv = avg_snr[nv_ind]
        avg_snr_ste_nv = avg_snr_ste[nv_ind]
        try:
            popt, pcov = curve_fit(
                fit_fn,
                taus,
                avg_snr_nv,
                p0=guess_params,
                sigma=avg_snr_ste_nv,
                absolute_sigma=True,
            )
        except Exception:
            popt = (20, 0, 300)
        opti_duration = popt[-1] + popt[0]
        opti_snr = fit_fn(opti_duration, *popt)
        dof = len(taus) - len(guess_params)
        red_chi_sq = (
            np.sum(((avg_snr_nv - fit_fn(taus, *popt)) / avg_snr_ste_nv) ** 2) / dof
        )
        print(red_chi_sq)
        tau_linspace = np.linspace(popt[0], np.max(taus), 1000)
        nv_num = widefield.get_nv_num(nv_sig)
        color = kpl.data_color_cycler[nv_num]
        kpl.plot_line(
            fit_ax,
            tau_linspace,
            fit_fn(tau_linspace, *popt),
            color=color,
            label=str(nv_num),
        )
        # kpl.plot_points(
        #     fit_ax,
        #     taus,
        #     avg_snr_nv,
        #     yerr=avg_snr_ste_nv,
        #     color=color,
        #     label=str(nv_num),
        # )
        # fit_ax.legend()
        # fit_ax.set_xlabel("SCC pulse duration (ns)")
        # fit_ax.set_ylabel("SNR")
        # fit_fig, fit_ax = plt.subplots()

        # ind = -3
        # opti_snr = round(avg_snr[nv_ind, ind], 3)
        # opti_duration = round(taus[ind])
        opti_snrs.append(opti_snr)
        opti_durations.append(opti_duration)
    print("Optimum SNRs")
    print([round(val, 3) for val in opti_snrs])
    print(f"Optimum {xlabel}")
    print([round(val) for val in opti_durations])
    fit_ax.legend(ncols=3)
    fit_ax.set_xlabel(xlabel)
    fit_ax.set_ylabel("SNR")

    return sig_fig, ref_fig, snr_fig, avg_snr_fig, fit_fig
    # return sig_fig, ref_fig, snr_fig, avg_snr_fig


def optimize_spin_pol_duration(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    uwave_ind_list,
):
    return _main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        min_tau,
        max_tau,
        True,
        uwave_ind_list,
    )


def optimize_spin_pol_amp(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_amp,
    max_amp,
    uwave_ind_list,
):
    return _main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        min_amp,
        max_amp,
        False,
        uwave_ind_list,
    )


def _main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    duration_or_amp,
    uwave_ind_list=[0, 1],
):
    ### Some initial setup

    seq_file = (
        "optimize_spin_pol_duration.py"
        if duration_or_amp
        else "optimize_spin_pol_amp.py"
    )

    taus = np.linspace(min_tau, max_tau, num_steps)

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        # print(f"DEBUG: seq_args before encoding: {seq_args}")
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,
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
        figs = process_and_plot(nv_list, taus, sig_counts, ref_counts, duration_or_amp)
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
    data = dm.get_raw_data(file_id=1720799193270)
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    # sig_counts, ref_counts = widefield.threshold_counts(nv_list, sig_counts, ref_counts)

    process_and_plot(nv_list, taus, sig_counts, ref_counts, False)

    plt.show(block=True)
