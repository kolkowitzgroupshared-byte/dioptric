# -*- coding: utf-8 -*-
"""
fill this out!

@author:alyssa-matthews
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb


def process_raw_buffer(
    new_tags,
    new_channels,
    current_tags,
    current_channels,
    gate_open_channel,
    gate_close_channel,
):
    """
    Process a single gate window for standard lifetime measurements.
    """
    current_tags.extend(new_tags)
    current_channels.extend(new_channels)
    current_channels_array = np.array(current_channels)

    result = np.nonzero(current_channels_array == gate_open_channel)
    gate_open_click_inds = result[0].tolist()

    result = np.nonzero(current_channels_array == gate_close_channel)
    gate_close_click_inds = result[0].tolist()

    new_processed_tags = []

    num_closed_samples = min(len(gate_open_click_inds), len(gate_close_click_inds))

    for list_ind in range(num_closed_samples):
        gate_open_click_ind = gate_open_click_inds[list_ind]
        gate_close_click_ind = gate_close_click_inds[list_ind]

        rep = current_tags[gate_open_click_ind + 1 : gate_close_click_ind]
        rep = np.array(rep, dtype=np.int64)
        rep -= current_tags[gate_open_click_ind]
        new_processed_tags.extend(rep.astype(int).tolist())

    if num_closed_samples > 0:
        leftover_start = gate_close_click_inds[num_closed_samples - 1]
        del current_tags[0 : leftover_start + 1]
        del current_channels[0 : leftover_start + 1]

    return new_processed_tags, num_closed_samples


def main(
    sample_sig,
    apd_indices,
    readout_times,
    filter_pos,
    num_reps,
    num_runs,
    num_bins,
    laser_power=None,
    seq_file="lifetime_caf_single_pulse.py",
):
    if len(apd_indices) > 1:
        msg = "Currently lifetime only supports single APDs!!"
        raise NotImplementedError(msg)

    tb.reset_cfm()
    kpl.init_kplotlib()
    repr_th_name = "lifetime"

    # --- Hardware Setup ---
    pulsegen_server = tb.get_server_pulse_streamer()
    counter_server = tb.get_server_counter()

    if len(filter_pos) != 0:
        slider_1 = tb.get_server_slider_1()
        slider_3 = tb.get_server_slider_3()

        slider_1_pos, slider_3_pos = filter_pos[0], filter_pos[1]
        slider_1.set_filter(slider_1_pos)
        slider_3.set_filter(slider_3_pos)
    else:
        slider_1_pos, slider_3_pos = None, None

    # Extract timings and handle the readout delay (expected by the sequence file)
    readout_delay = int(readout_times[0])
    pulse_time = int(readout_times[1])  # exc_ns
    readout_time = int(readout_times[2]) if len(readout_times) > 2 else 0

    # --- Sequence Loading ---
    laser_vkey = "SPIN_READOUT"

    # Matches: readout_delay_ns, exc_ns, detect_ns, laser_vkey_arg, laser_power
    seq_args = [
        readout_delay,
        pulse_time,
        readout_time,
        laser_vkey,
        laser_power,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)
    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time_m = (num_runs * (num_reps * seq_time_s + 1)) / 60  # m
    print(f" \nExpected run time: {expected_run_time_m:.2f} minutes. ")

    # --- Live Plot Setup ---
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax2 = ax.twinx()

    ax.set_title("Lifetime")
    ax.set_xlabel("Time after illumination (us)")
    ax.set_ylabel("kcps", color="r")
    ax2.set_ylabel("Total Raw Counts", color="k")

    (line_kcps,) = ax.plot([], [], "r-", label="kcps")
    (line_raw,) = ax2.plot([], [], "k-", alpha=0.7, label="Raw Counts")

    # --- Execution ---
    startFunctionTime = time.time()
    start_timestamp = dm.get_time_stamp()

    processed_tags = []
    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print(f" \nRun index: {run_ind + 1}/{num_runs}")

        if tb.safe_stop():
            break

        counter_server.start_tag_stream()
        pulsegen_server.stream_start(int(num_reps))

        channel_mapping = counter_server.get_channel_mapping()
        gate_open_channel = channel_mapping[1]
        gate_close_channel = channel_mapping[2]

        current_tags = []
        current_channels = []
        num_processed_reps = 0

        while num_processed_reps < num_reps:
            if tb.safe_stop():
                break

            new_tags, new_channels = counter_server.read_tag_stream()
            new_tags = np.array(new_tags, dtype=np.int64)

            new_processed_tags, num_new_processed_reps = process_raw_buffer(
                new_tags,
                new_channels,
                current_tags,
                current_channels,
                gate_open_channel,
                gate_close_channel,
            )

            if num_new_processed_reps > 750000:
                print(f"Processed {num_new_processed_reps} reps out of 10^6 max")
                print("Tell Matt that the time tagger is too slow!")

            num_processed_reps += num_new_processed_reps
            processed_tags.extend(new_processed_tags)

        counter_server.stop_tag_stream()

        # --- Live Plotting Update ---
        readout_time_ps = 1000 * readout_time
        binned_samples, _ = np.histogram(processed_tags, num_bins, (0, readout_time_ps))

        bin_size_ns = readout_time / num_bins
        bin_size_s = bin_size_ns / 1e9

        # Calculate kcps based on the total reps gathered so far
        total_reps_so_far = num_reps * (run_ind + 1)
        # total_reps_so_far = num_reps / num_runs
        binned_samples_kcps = binned_samples / bin_size_s / 1e3 / total_reps_so_far

        bin_center_offset = bin_size_ns / 2
        bin_centers_ns = (
            np.linspace(0, readout_time, num_bins, endpoint=False) + bin_center_offset
        )
        x_data_us = np.array(bin_centers_ns) / 1e3

        line_kcps.set_data(x_data_us, binned_samples_kcps)
        line_raw.set_data(x_data_us, binned_samples)

        ax.relim()
        ax.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

        # --- Incremental Data Saving ---
        raw_data = {
            "start_timestamp": start_timestamp,
            "sample_sig": getattr(sample_sig, "name", "sample"),
            "laser_power": laser_power,
            "slider_1_pos": slider_1_pos,
            "slider_3_pos": slider_3_pos,
            "readout_time": readout_time,
            "readout_time-units": "ns",
            "pulse_time": pulse_time,
            "pulse_time-units": "ns",
            "readout_delay": readout_delay,
            "readout_delay-units": "ns",
            "num_reps": num_reps,
            "num_runs": num_runs,
            "run_ind": run_ind,
            "num_bins": num_bins,
            "processed_tags": processed_tags,
            "processed_tags-units": "ps",
        }

        file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)
        dm.save_raw_data(raw_data, file_path)

    # --- Final Wrap-up ---
    tb.reset_cfm()

    endFunctionTime = time.time()
    time_elapsed = endFunctionTime - startFunctionTime

    raw_data.update(
        {
            "time_elapsed": time_elapsed,
            "binned_samples": binned_samples.tolist(),
            "bin_centers": bin_centers_ns.tolist(),
        }
    )

    print(f"Saved final data to: {file_path}")
    dm.save_figure(fig, file_path)
    dm.save_raw_data(raw_data, file_path)
    print("FIN --")

    plt.ioff()
    plt.show()

    return raw_data


if __name__ == "__main__":

    class Dummy:
        name = "caf_test"

    sample_sig = Dummy()

    main(
        sample_sig=sample_sig,
        apd_indices=[0],
        readout_times=[0, 200, 200],  # readout delay, excitation, readout time
        filter_pos=[2, 2],
        num_reps=100000,
        num_runs=1,
        num_bins=2000,
        laser_power=0.1e-3,
        seq_file="lifetime_caf_single_pulse.py",
    )
