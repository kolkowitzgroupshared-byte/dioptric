# -*- coding: utf-8 -*-
"""
Optimize SCC parameters
Created on December 6th, 2023
@author: mccambria
Updated on Fall 2024
@author: schand
"""

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


def process_and_plot(nv_list, taus, sig_counts, ref_counts):
    avg_sig_counts, avg_sig_counts_ste = widefield.process_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste = widefield.process_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    # avg_snr_ste = None

    sig_fig, sig_ax = plt.subplots()
    widefield.plot_raw_data(sig_ax, nv_list, taus, avg_sig_counts, avg_sig_counts_ste)
    sig_ax.set_xlabel("Ionization pulse duration (ns)")
    sig_ax.set_ylabel("Signal counts")

    ref_fig, ref_ax = plt.subplots()
    widefield.plot_raw_data(ref_ax, nv_list, taus, avg_ref_counts, avg_ref_counts_ste)
    ref_ax.set_xlabel("Ionization pulse duration (ns)")
    ref_ax.set_ylabel("Reference counts")

    snr_fig, snr_ax = plt.subplots()
    widefield.plot_raw_data(snr_ax, nv_list, taus, avg_snr, avg_snr_ste)
    snr_ax.set_xlabel("Ionization pulse duration (ns)")
    snr_ax.set_ylabel("SNR")

    return sig_fig, ref_fig, snr_fig

    # Average across NVs
    # snr_fig, snr_ax = plt.subplots()
    # avg_avg_snr = np.quantile(avg_snr, 0.75, axis=0)
    # avg_avg_snr_ste = np.quantile(avg_snr_ste, 0.75, axis=0)
    # kpl.plot_points(snr_ax, taus, avg_avg_snr, yerr=avg_avg_snr_ste)
    # snr_ax.set_xlabel("Ionization pulse duration (ns)")
    # snr_ax.set_ylabel("Average SNR")

    # return sig_fig, ref_fig, snr_fig, avg_snr_fig


def main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau):
    ### Some initial setup

    seq_file = "optimize_scc.py"
    taus = np.linspace(min_tau, max_tau, num_steps)
    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def step_fn(tau_ind):
        tau = taus[tau_ind]
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(tau)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, step_fn, num_exps=2
    )
    sig_counts = counts[0]
    ref_counts = counts[1]

    ### Process and plot

    sig_fig, ref_fig, snr_fig, avg_snr_fig = process_and_plot(
        nv_list, taus, sig_counts, ref_counts
    )

    ### Clean up and return

    tb.reset_cfm()

    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": min_tau,
        "max_tau": max_tau,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig["name"]
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-sig")
    dm.save_figure(sig_fig, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-ref")
    dm.save_figure(ref_fig, file_path)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-snr")
    dm.save_figure(snr_fig, file_path)
    # file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-avg_snr")
    # dm.save_figure(avg_snr_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = "2023_11_27-19_31_32-johnson-nv0_2023_11_25"
    # data = dm.get_raw_data(file_name)
    # data = dm.get_raw_data(file_id=1394470302639)  # 120
    data = dm.get_raw_data(file_id=1394440880735)  # 180
    # data = dm.get_raw_data(file_id=1394488086583)  # 60

    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = data["counts"]
    sig_counts = counts[0]
    ref_counts = counts[1]

    process_and_plot(nv_list, taus, sig_counts, ref_counts)

    plt.show(block=True)
