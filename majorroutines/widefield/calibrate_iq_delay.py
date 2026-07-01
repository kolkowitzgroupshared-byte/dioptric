# -*- coding: utf-8 -*-
"""
Widefield IQ-delay calibration.

Sweeps an IQ timing offset (taus) using "calibrate_iq_delay.py", processes I/Q readout counts
(threshold + normalization), plots the median I and Q response across NVs vs delay, and saves
raw_data + metadata.

Created: 2023-11-29 (mccambria)
Updated: 2024-04-17 (sbchand)
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


# def create_raw_data_figure(raw_data):
#     nv_list = raw_data["nv_list"]
#     counts = raw_data["counts"]
#     i_counts, q_counts = counts[0], counts[1]
#     i_norm_counts, i_norm_counts_ste = widefield.process_counts(
#         nv_list, i_counts, threshold=True
#     )
#     q_norm_counts, q_norm_counts_ste = widefield.process_counts(
#         nv_list, q_counts, threshold=True
#     )
#     fig, ax = plt.subplots()
#     ax.set_xlabel("Delay (ns)")
#     ax.set_ylabel("Counts")
#     return fig

def create_median_data_figure(raw_data):
    nv_list = raw_data["nv_list"]
    delays = raw_data["taus"]  # assumed in ns
    counts = np.array(raw_data["counts"])
    iq_buffer = raw_data["config"]["CommonDurations"]["iq_buffer"]

    i_counts, q_counts = counts[0], counts[1]
    # Normalize and threshold counts
    i_norm_counts, i_norm_counts_ste = widefield.process_counts(
        nv_list, i_counts, threshold=True
    )
    q_norm_counts, q_norm_counts_ste = widefield.process_counts(
        nv_list, q_counts, threshold=True
    )

    # Compute median across NVs (axis 0: NV index)
    i_median = np.median(i_norm_counts, axis=0)
    q_median = np.median(q_norm_counts, axis=0)

    i_median_ste = np.median(i_norm_counts_ste, axis=0)
    q_median_ste = np.median(q_norm_counts_ste, axis=0)
    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(
        delays,
        i_median,
        yerr=i_median_ste,
        label="I Median",
        fmt="-o",
        capsize=3,
    )
    ax.errorbar(
        delays,
        q_median,
        yerr=q_median_ste,
        label="Q Median",
        fmt="-s",
        capsize=3,
    )
    ax.set_xlabel("Delay (ns)", fontsize=15)
    ax.set_ylabel("Normalized Counts", fontsize=15)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend()
    ax.set_title(
        f"Median NV Counts vs IQ Delay (iq_buffer ={iq_buffer}ns)", fontsize=15
    )

    return fig


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit


# def gaussian(x, a, x0, sigma, offset):
#     return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


# def create_median_data_figure(raw_data):
#     nv_list = raw_data["nv_list"]
#     delays = np.array(raw_data["taus"])  # assumed in ns
#     counts = np.array(raw_data["counts"])
#     i_counts, q_counts = counts[0], counts[1]

#     # Normalize and threshold counts
#     i_norm_counts, i_norm_counts_ste = widefield.process_counts(
#         nv_list, i_counts, threshold=True
#     )
#     q_norm_counts, q_norm_counts_ste = widefield.process_counts(
#         nv_list, q_counts, threshold=True
#     )

#     # Compute median and STE across NVs
#     i_median = np.median(i_norm_counts, axis=0)
#     q_median = np.median(q_norm_counts, axis=0)
#     i_median_ste = np.median(i_norm_counts_ste, axis=0)
#     q_median_ste = np.median(q_norm_counts_ste, axis=0)

#     # Fit Gaussian to both I and Q
#     try:
#         popt_i, _ = curve_fit(
#             gaussian, delays, i_median, p0=[-0.5, delays[np.argmin(i_median)], 100, 0.1]
#         )
#     except RuntimeError:
#         popt_i = None

#     try:
#         popt_q, _ = curve_fit(
#             gaussian, delays, q_median, p0=[-0.5, delays[np.argmin(q_median)], 100, 0.1]
#         )
#     except RuntimeError:
#         popt_q = None

#     # Plot
#     fig, ax = plt.subplots()
#     ax.errorbar(
#         delays, i_median, yerr=i_median_ste, label="I Median", fmt="-o", capsize=3
#     )
#     ax.errorbar(
#         delays, q_median, yerr=q_median_ste, label="Q Median", fmt="-s", capsize=3
#     )

#     # Plot fitted curves if fit succeeded
#     fit_x = np.linspace(delays[0], delays[-1], 300)
#     if popt_i is not None:
#         ax.plot(fit_x, gaussian(fit_x, *popt_i), label="I Fit", linestyle="--")
#     if popt_q is not None:
#         ax.plot(fit_x, gaussian(fit_x, *popt_q), label="Q Fit", linestyle="--")

#     ax.set_xlabel("Delay (ns)")
#     ax.set_ylabel("Normalized Counts")
#     ax.set_title("Median NV Counts vs IQ Delay")
#     ax.legend()

#     return fig


def main(nv_list, num_steps, num_reps, num_runs, taus, uwave_ind_list=[1]):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "calibrate_iq_delay.py"
    # taus = np.linspace(min_tau, max_tau, num_steps)

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]
        # print(seq_args)
        # sys.exit()
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Collect the data
    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,
    )

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
    }
    ### save the raw data
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    ### Clean up and return
    tb.reset_cfm()

    ### Process and plot
    create_median_data_figure(raw_data)
    # dm.save_figure(raw_fig, file_path)
    kpl.show()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_id = 1838919367119 # buffer 16ns
    # file_id = 1838833811044  # buffer 32ns
    file_id = 1838880724558
    # file_id = 1838788378840  # buffer 100
    file_stem = "2025_05_01-20_52_49-rubin-nv0_2025_02_26"
    data = dm.get_raw_data(file_stem=file_stem, load_npz=False, use_cache=False)
    # file_name = dm.get_file_name(file_id=file_id)
    # print(f"{file_name}_{file_id}")
    # nv_list = data["nv_list"]
    # num_nvs = len(nv_list)
    # num_steps = data["num_steps"]
    # num_runs = data["num_runs"]
    # taus = data["taus"]
    # counts = np.array(data["counts"])

    # avg_counts, avg_counts_ste = widefield.process_counts(counts)
    # raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    create_median_data_figure(data)

    plt.show(block=True)
