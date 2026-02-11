# -*- coding: utf-8 -*-
"""
Lighweight check of the SCC SNR

Created on December 6th, 2023

@author: mccambria
@author: schand
"""

import sys
import time
import traceback

import numpy as np
from matplotlib import pyplot as plt

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def process_and_plot(data):
    threshold = True
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    if threshold:
        sig_counts, ref_counts = widefield.threshold_counts(
            nv_list, sig_counts, ref_counts, dynamic_thresh=True
        )

    ### Report the results

    avg_sig_counts, avg_sig_counts_ste, _ = widefield.average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = widefield.average_counts(ref_counts)

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)

    # There's only one point, so only consider that
    step_ind = 0
    avg_sig_counts = avg_sig_counts[:, step_ind]
    avg_sig_counts_ste = avg_sig_counts_ste[:, step_ind]
    avg_ref_counts = avg_ref_counts[:, step_ind]
    avg_ref_counts_ste = avg_ref_counts_ste[:, step_ind]
    avg_snr = avg_snr[:, step_ind]
    avg_snr_ste = avg_snr_ste[:, step_ind]
    avg_contrast = avg_contrast[:, step_ind]
    avg_contrast_ste = avg_contrast_ste[:, step_ind]

    # Print
    avg_snr = avg_snr.tolist()
    print(f"avg_snr = {[f'{snr:.3f}' for snr in avg_snr]}")
    print(f"Median SNR: {np.median(avg_snr):.3f}")
    return avg_snr

    ### Plot

    # Normalized counts bar plots
    # fig, ax = plt.subplots()
    # for ind in range(num_nvs):
    #     nv_sig = nv_list[ind]
    #     nv_num = widefield.get_nv_num(nv_sig)
    #     kpl.plot_bars(ax, nv_num, norm_counts[ind], yerr=norm_counts_ste[ind])
    # ax.set_xlabel("NV index")
    # ax.set_ylabel("Contrast")

    # SNR bar plots
    # figsize = kpl.figsize
    # figsize[1] *= 1.5
    # counts_fig, axes_pack = plt.subplots(2, 1, sharex=True, figsize=figsize)
    # snr_fig, ax = plt.subplots()
    # for ind in range(len(nv_list)):
    #     nv_sig = nv_list[ind]
    #     nv_num = widefield.get_nv_num(nv_sig)
    #     kpl.plot_bars(
    #         axes_pack[0], nv_num, avg_ref_counts[ind], yerr=avg_ref_counts_ste[ind]
    #     )
    #     kpl.plot_bars(
    #         axes_pack[1], nv_num, avg_sig_counts[ind], yerr=avg_sig_counts_ste[ind]
    #     )
    #     kpl.plot_bars(ax, nv_num, avg_snr[ind], yerr=avg_snr_ste[ind])
    # axes_pack[0].set_xlabel("NV index")
    # ax.set_xlabel("NV index")
    # axes_pack[0].set_ylabel("NV- | prep in ms=0")
    # axes_pack[1].set_ylabel("NV- | prep in ms=1")
    # ax.set_ylabel("SNR")
    # return counts_fig, snr_fig

    # SNR histogram
    fig, ax = plt.subplots()
    kpl.histogram(ax, avg_snr, kpl.HistType.STEP, nbins=10)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Number of occurrences")

    # SNR vs red frequency
    coords_key = "laser_COBO_638_aod"
    distances = []
    for nv in nv_list:
        coords = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
        dist = np.sqrt((90 - coords[0]) ** 2 + (90 - coords[1]) ** 2)
        distances.append(dist)
    fig, ax = plt.subplots()
    kpl.plot_points(ax, distances, avg_snr)
    ax.set_xlabel("Distance from center frequencies (MHz)")
    ax.set_ylabel("SNR")


def main(nv_list, num_reps, num_runs, uwave_ind_list=[0, 1]):
    ### Some initial setup
    num_steps = 1

    seq_file = "scc_snr_check.py"
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
        ]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data
    data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
        charge_prep_fn=None,
        num_exps=2,
        # load_iq=True,
    )

    ### Report results and cleanup

    timestamp = dm.get_time_stamp()

    repr_nv_name = widefield.get_repr_nv_sig(nv_list).name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(data, file_path)

    try:
        figs = process_and_plot(data)
    except Exception:
        print(traceback.format_exc())
        figs = None
    if figs is not None:
        num_figs = len(figs)
        for ind in range(num_figs):
            file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + f"-{ind}")
            dm.save_figure(figs[ind], file_path)

    ### Clean up and return
    tb.reset_cfm()
    kpl.show()


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1732924888109)
    avg_snr = process_and_plot(data)
    print(f"Minimum SNR: {np.min(avg_snr)}")
    sys.exit()

    data = dm.get_raw_data(file_id=1731300731766)  # -8
    avg_snr_a = process_and_plot(data)
    data = dm.get_raw_data(file_id=1731322739905)  # +0
    avg_snr_b = process_and_plot(data)
    # fmt: off
    test = [0.207, 0.206, 0.211, 0.183, 0.08, 0.224, 0.095, 0.078, 0.136, 0.165, 0.13, 0.18, 0.153, 0.074, 0.08, 0.142, 0.188, 0.077, 0.121, 0.137, 0.085, 0.157, 0.135, 0.075, 0.168, 0.158, 0.12, 0.074, 0.167, 0.073, 0.149, 0.135, 0.119, 0.193, 0.104, 0.091, 0.127, 0.125, 0.105, 0.139, 0.151, 0.119, 0.134, 0.11, 0.105, 0.133, 0.149, 0.102, 0.083, 0.097, 0.175, 0.096, 0.161, 0.158, 0.1, 0.093, 0.132, 0.131, 0.083, 0.114, 0.144, 0.142, 0.116, 0.143, 0.121, 0.116, 0.102, 0.113, 0.087, 0.119, 0.119, 0.131, 0.144, 0.122, 0.087, 0.087, 0.089, 0.089, 0.131, 0.075, 0.09, 0.085, 0.099, 0.123, 0.133, 0.097, 0.083, 0.097, 0.148, 0.118, 0.078, 0.081, 0.112, 0.119, 0.137, 0.133, 0.074, 0.106, 0.165, 0.16, 0.132, 0.088, 0.081]
    # fmt: on
    test2 = np.array(test) - np.array(avg_snr_b)
    fig, ax = plt.subplots()
    kpl.plot_points(ax, range(len(test2)), test2)
    ax.set_xlabel("NV order index")
    ax.set_ylabel("SNR difference from predicted value")
    kpl.show(block=True)

    data = dm.get_raw_data(file_id=1731342940279)  # +8
    data = dm.get_raw_data(file_id=1731408940779)
    data = dm.get_raw_data(file_id=1732884193552)
    avg_snr_c = process_and_plot(data)
    nv_list = data["nv_list"]

    fig, ax = plt.subplots()
    nv_nums = [widefield.get_nv_num(nv) for nv in nv_list]
    kpl.plot_points(ax, nv_nums, avg_snr_a, label="Cross")
    kpl.plot_points(ax, nv_nums, avg_snr_b, label="Full")
    kpl.plot_points(ax, nv_nums, avg_snr_c, label="High power")
    ax.legend()
    kpl.show(block=True)
