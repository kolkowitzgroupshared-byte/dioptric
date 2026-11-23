# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from majorroutines.widefield.spin_echo import (
    create_fit_figure as spin_echo_create_fit_figure,
)
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


def create_raw_data_figure(nv_list, taus, counts, counts_ste):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, taus, counts, counts_ste)
    ax.set_xlabel("Tau (ns)")
    ax.set_ylabel("Counts")
    return fig


def create_raw_data_figure_sep(nv_list, taus, counts, counts_ste):
    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(nrows=6, sharex=True, figsize=[6.5, 6.0])
    widefield.plot_fit(axes_pack, nv_list, taus, counts, counts_ste)
    axes_pack[-1].set_xlabel("Tau (ns)")
    axes_pack[3].set_ylabel("Counts")

    yticks = [
        [34, 36],
        [40, 42],
        [37, 40],
        [30, 32],
        [34, 36],
        [28.5, 29],
    ]
    for ind in range(len(axes_pack)):
        ax = axes_pack[ind]
        ax.set_yticks(yticks[ind])

    return fig


def create_fit_figure(nv_list, taus, counts, counts_ste):
    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(nrows=6, sharex=True, figsize=[6.5, 6.0])

    tau_step = taus[1] - taus[0]
    freqs = np.fft.rfftfreq(len(taus), d=tau_step)
    freqs = [1000 * el for el in freqs[1:]]  # Convert to MHz

    nv_mags = []
    for nv_ind in range(len(nv_list)):
        nv_counts = counts[nv_ind]
        transform = np.fft.rfft(nv_counts)
        transform_mag = np.absolute(transform)
        nv_mags.append(transform_mag[1:])

    widefield.plot_fit(axes_pack, nv_list, freqs, nv_mags)

    axes_pack[-1].set_xlabel("Frequency (MHz)")
    axes_pack[3].set_ylabel("FFT magnitude")

    yticks = [
        [0, 5],
        [0, 10],
        [0, 10],
        [0, 5],
        [0, 10],
        [0, 3],
    ]
    ylims = [
        [0, 9],
        [0, 14],
        [0, 14],
        [0, 9],
        [0, 17],
        [0, 4],
    ]
    for ind in range(len(axes_pack)):
        ax = axes_pack[ind]
        ax.set_yticks(yticks[ind])
        ax.set_ylim(ylims[ind])

    return fig


def main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, detuning):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "ramsey.py"
    taus = np.linspace(min_tau, max_tau, num_steps)

    # #uwave_ind_list = [0,1]
    # uwave_freq_list = []
    # for ind in uwave_ind_list:
    #     uwave_dict = tb.get_uwave_dict(ind)
    #     uwave_freq = uwave_dict["frequency"]
    #     uwave_freq -= detuning / 1000
    #     uwave_freq_list.append(uwave_freq)
    
    uwave_ind_list = [1] 
    # base_routine looks frequencies up by generator number (uwave_freq_list[uwave_ind]), 
    # so keep one entry per generator up to the highest one used; only those in  
    # # uwave_ind_list are driven 
    uwave_freq_list = [tb.get_uwave_dict(ind)["frequency"] for ind in range(max(uwave_ind_list) + 1)]
    for ind in uwave_ind_list:
        uwave_freq_list[ind] -= detuning / 1000

    ### Collect the data

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        uwave_ind_list=uwave_ind_list,
        uwave_freq_list=uwave_freq_list,
        save_images=False,
    )

    ### Process and plot

    # try:
    #     raw_fig = create_raw_data_figure(raw_data)
    #     fit_fig = create_fit_figure(raw_data)
    # except Exception as exc:
    #     print(exc)
    #     raw_fig = None
    #     fit_fig = None
    
    raw_fig = None
    fit_fig = None

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
        "detuning": detuning,
        "detuning-units": "MHz",
        "uwave_ind_list": uwave_ind_list,
        "uwave_freq_list": uwave_freq_list,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1550116175341)

    spin_echo_create_fit_figure(data)

    plt.show(block=True)
