# -*- coding: utf-8 -*-
"""
Calibrate green/red timing: sweep green delay and measure SNR after ~1 µs spin polarization.

Runs a single-NV widefield sequence ("calibrate_green_red_delay.py") over a range of delays,
computes SNR from signal/reference counts, plots SNR vs delay, and saves raw_data + figure.

Created: 2024-03-31 (mccambria)
Updated: 2025-01-05 (Saroj Chand)
"""

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig
from utils.positioning import get_scan_1d as calculate_freqs


def create_raw_data_figure(delays, avg_snr, avg_snr_ste):
    fig, ax = plt.subplots()
    kpl.plot_points(ax, delays, avg_snr, yerr=avg_snr_ste)

    ax.set_xlabel("Green delay (ns)")
    ax.set_ylabel("SNR")

    return fig


def main(
    nv_sig: NVSig,
    num_steps,
    num_reps,
    num_runs,
    min_delay,
    max_delay,
    uwave_ind=0,
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    delays = np.linspace(min_delay, max_delay, num_steps)
    seq_file = "calibrate_green_red_delay.py"

    ### Collect the data

    nv_list = [nv_sig]

    def run_fn(step_ind_list):
        # Base seq args
        seq_args = widefield.get_base_scc_seq_args(nv_list)
        seq_args.append(uwave_ind)

        # Shuffled voltages
        delays_shuffle = [delays[ind] for ind in step_ind_list]
        seq_args.append(delays_shuffle)

        # Pass it over to the OPX
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    counts, raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, run_fn=run_fn, uwave_ind_list=uwave_ind
    )

    ### Process and plot

    # experiment, nv, run, step, rep
    sig_counts = counts[0]
    ref_counts = counts[1]

    # avg_sig_counts, avg_sig_counts_ste = widefield.average_counts(sig_counts)
    # avg_ref_counts, avg_ref_counts_ste = widefield.average_counts(ref_counts)
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    avg_snr = avg_snr[0]
    avg_snr_ste = avg_snr_ste[0]

    raw_fig = create_raw_data_figure(delays, avg_snr, avg_snr_ste)

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "nv_sig": nv_sig,
        "timestamp": timestamp,
        "aod_min_voltage": aod_min_voltage,
        "aod_max_voltage": aod_max_voltage,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(raw_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1470392816628, load_npz=True)

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    freqs = data["freqs"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    kpl.show(block=True)
