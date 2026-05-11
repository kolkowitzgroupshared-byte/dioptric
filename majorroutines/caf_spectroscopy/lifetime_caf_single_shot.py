# -*- coding: utf-8 -*-
"""
performs single shot lifetime measurements for caf. can do either continuous or pulsed depending on sequence selected

@author:alyssa-matthews
"""

import csv
import json
import os
import time

import labrad
import matplotlib.pyplot as plt
import numpy
from scipy.optimize import curve_fit

import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt
from utils import common
from utils import data_manager as dm


def main(
    nv_sig,
    apd_indices,
    readout_times,
    filter_pos,
    num_reps,
    num_runs,
    num_bins,
    sequence_file,  # Moved up! Required positional argument
    laser_power=None,
):
    if len(apd_indices) > 1:
        msg = "Currently lifetime only supports single APDs!!"
        raise NotImplementedError(msg)

    tool_belt.reset_cfm()
    repr_th_name = "irr4"

    pulsegen_server = tool_belt.get_server_pulse_streamer()
    counter_server = tool_belt.get_server_counter()

    if len(filter_pos) != 0:
        slider_1 = tool_belt.get_server_slider_1()
        slider_3 = tool_belt.get_server_slider_3()

        slider_1_pos, slider_3_pos = filter_pos

        slider_1.set_filter(slider_1_pos)
        slider_3.set_filter(slider_3_pos)

    # Handle the readout_times list for both sequences
    # Expected format passed from wrapper: [delay_ns, exc_ns, detect_ns]
    if len(readout_times) >= 3:
        delay_ns = int(
            readout_times[0]
        )  # readout_delay OR recovery_delay depending on sequence
        pulse_time = int(readout_times[1])  # exc_ns
        readout_time = int(readout_times[2])  # detect_ns
    else:
        # Fallback if only 2 arguments are provided
        delay_ns = 0
        readout_time = int(readout_times[0])  # detect_ns
        pulse_time = int(readout_times[1])  # exc_ns

    calc_readout_time = readout_time

    # Set the virtual laser key
    laser_vkey = "SPIN_READOUT"

    print(f"Loading sequence: {sequence_file}")

    # Map variables to the exact format expected by BOTH sequence files
    # args = [delay_ns, exc_ns, detect_ns, laser_vkey, laser_power]
    seq_args = [
        delay_ns,
        pulse_time,
        readout_time,
        laser_vkey,
        laser_power,
    ]

    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(sequence_file, seq_args_string)  # LOAD
    seq_time = ret_vals[0]

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_runs * (num_reps * seq_time_s + 1)  # s
    expected_run_time_m = expected_run_time / 60  # m
    print(" \nExpected run time: {:.2f} minutes. ".format(expected_run_time_m))

    startFunctionTime = time.time()
    start_timestamp = dm.get_time_stamp()

    tool_belt.init_safe_stop()

    # 1. Figure out the hardware channel map
    counter_server.start_tag_stream()
    channel_mapping = counter_server.get_channel_mapping()
    counter_server.stop_tag_stream()

    apd_channel = channel_mapping[0]  # The actual APD click channel
    gate_open_channel = channel_mapping[1]  # The start trigger for our timer

    # 2. Calculate Histogram parameters
    readout_time_ps = int(1000 * calc_readout_time)
    bin_size_ps = int(readout_time_ps / num_bins)
    run_time_s = num_reps * seq_time_s  # Calculate exact time one run takes

    # Initialize a master array to hold our counts
    binned_samples = numpy.zeros(num_bins, dtype=numpy.int64)

    for run_ind in range(num_runs):
        print(f" \nRun index: {run_ind}")

        if tool_belt.safe_stop():
            break

        seq_args_string = tool_belt.encode_seq_args(seq_args)

        # -- THE NEW ACQUISITION FLOW --

        # A. Arm the hardware histogram
        counter_server.start_histogram(
            gate_open_channel, apd_channel, bin_size_ps, num_bins
        )

        # B. Fire the laser sequence
        pulsegen_server.stream_start(int(num_reps))

        # C. Wait for the sequence to physically finish playing
        time.sleep(run_time_s + 0.1)

        # D. Read the completely processed array
        run_binned_samples = counter_server.read_histogram()
        binned_samples += numpy.array(run_binned_samples, dtype=numpy.int64)

        # E. Clean up the hardware
        counter_server.stop_histogram()

        # Save the data incrementally
        raw_data = {
            "start_timestamp": start_timestamp,
            "sequence_file": sequence_file,
            "nv_sig": nv_sig,
            "laser_power": laser_power,
            "laser_vkey": laser_vkey,
            "slider_1_pos": filter_pos[0],
            "slider_3_pos": filter_pos[1],
            "delay_ns": delay_ns,
            "delay_ns-units": "ns",
            "readout_time": readout_time,
            "readout_time-units": "ns",
            "pulse_time": pulse_time,
            "pulse_time-units": "ns",
            "calc_readout_time": calc_readout_time,
            "calc_readout_time-units": "ns",
            "num_reps": num_reps,
            "num_runs": num_runs,
            "run_ind": run_ind,
            "num_bins": num_bins,
            "binned_samples": binned_samples.tolist(),
        }

        file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)
        dm.save_raw_data(raw_data, file_path)

    tool_belt.reset_cfm()

    # Calculate bin properties for plotting
    bin_size_ns = calc_readout_time / num_bins
    bin_size_s = bin_size_ns / 1e9
    binned_samples_kcps = binned_samples / bin_size_s / 1e3 / num_reps / num_runs
    bin_center_offset = bin_size_ns / 2
    bin_centers_ns = numpy.linspace(0, readout_time, num_bins) + bin_center_offset

    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax2 = ax.twinx()
    ax.plot(numpy.array(bin_centers_ns) / 10**3, binned_samples_kcps, "r-")
    ax2.plot(numpy.array(bin_centers_ns) / 10**3, binned_samples, "r-")

    ax.set_xlabel("X data")
    ax.set_ylabel("kcps", color="k")
    ax2.set_ylabel("Total Raw Counts", color="k")

    ax.set_title("Lifetime")
    ax.set_xlabel("Time after illumination (us)")

    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

    endFunctionTime = time.time()
    time_elapsed = endFunctionTime - startFunctionTime

    # Final save mapping
    raw_data = {
        "start_timestamp": start_timestamp,
        "time_elapsed": time_elapsed,
        "sequence_file": sequence_file,
        "nv_sig": nv_sig,
        "laser_power": laser_power,
        "laser_vkey": laser_vkey,
        "slider_1_pos": filter_pos[0],
        "slider_3_pos": filter_pos[1],
        "delay_ns": delay_ns,
        "delay_ns-units": "ns",
        "readout_time": readout_time,
        "readout_time-units": "ns",
        "pulse_time": pulse_time,
        "pulse_time-units": "ns",
        "calc_readout_time": calc_readout_time,
        "calc_readout_time-units": "ns",
        "num_bins": num_bins,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "binned_samples": binned_samples.tolist(),
        "bin_centers": bin_centers_ns.tolist(),
    }
    print(file_path)

    dm.save_figure(fig, file_path)
    file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)
    dm.save_raw_data(raw_data, file_path)
    folder_path, file_name = os.path.split(file_path)
    lifetime_json_to_csv(file_name, folder_path)
    print("FIN --")


def lifetime_json_to_csv(
    file, folder, nv_data_dir="E:/Shared drives/Kolkowitz Lab Group/nvdata"
):
    data = tool_belt.get_raw_data(file, folder)
    binned_samples = data["binned_samples"]
    bin_centers = data["bin_centers"]

    csv_data = []

    for bin_ind in range(len(bin_centers)):
        row = []
        row.append(bin_centers[bin_ind])
        row.append(binned_samples[bin_ind])
        csv_data.append(row)

    tool_belt.write_csv(csv_data, file, folder)


# import csv
# import json
# import os
# import time

# import labrad
# import matplotlib.pyplot as plt
# import numpy
# from scipy.optimize import curve_fit

# import majorroutines.targeting as targeting
# import utils.tool_belt as tool_belt
# from utils import common
# from utils import data_manager as dm


# def process_raw_buffer(
#     new_tags,
#     new_channels,
#     current_tags,
#     current_channels,
#     gate_open_channel,
#     gate_close_channel,
# ):

#     current_tags.extend(new_tags)
#     current_channels.extend(new_channels)
#     current_channels_array = numpy.array(current_channels)

#     result = numpy.nonzero(current_channels_array == gate_open_channel)
#     gate_open_click_inds = result[0].tolist()

#     result = numpy.nonzero(current_channels_array == gate_close_channel)
#     gate_close_click_inds = result[0].tolist()

#     new_processed_tags = []

#     num_closed_samples = len(gate_close_click_inds)
#     for list_ind in range(num_closed_samples):
#         gate_open_click_ind = gate_open_click_inds[list_ind]
#         gate_close_click_ind = gate_close_click_inds[list_ind]

#         rep = current_tags[gate_open_click_ind + 1 : gate_close_click_ind]
#         rep = numpy.array(rep, dtype=numpy.int64)
#         rep -= current_tags[gate_open_click_ind]
#         new_processed_tags.extend(rep.astype(int).tolist())

#     if len(gate_close_click_inds) > 0:
#         leftover_start = gate_close_click_inds[-1]
#         del current_tags[0 : leftover_start + 1]
#         del current_channels[0 : leftover_start + 1]

#     return new_processed_tags, num_closed_samples


# def main(
#     nv_sig,
#     apd_indices,
#     readout_times,
#     filter_pos,
#     num_reps,
#     num_runs,
#     num_bins,
#     sequence_file,  # Moved up! Required positional argument
#     laser_power=None,
# ):
#     if len(apd_indices) > 1:
#         msg = "Currently lifetime only supports single APDs!!"
#         raise NotImplementedError(msg)

#     tool_belt.reset_cfm()
#     repr_th_name = "irr4"

#     pulsegen_server = tool_belt.get_server_pulse_streamer()
#     counter_server = tool_belt.get_server_counter()

#     if len(filter_pos) != 0:
#         slider_1 = tool_belt.get_server_slider_1()
#         slider_3 = tool_belt.get_server_slider_3()

#         slider_1_pos, slider_3_pos = filter_pos
#         # print(slider_1_pos, slider_3_pos)

#         slider_1.set_filter(slider_1_pos)
#         slider_3.set_filter(slider_3_pos)
#         # print("main is started")

#     # print("main is started")

#     # Handle the readout_times list for both sequences
#     # Expected format passed from wrapper: [delay_ns, exc_ns, detect_ns]
#     if len(readout_times) >= 3:
#         delay_ns = int(
#             readout_times[0]
#         )  # readout_delay OR recovery_delay depending on sequence
#         pulse_time = int(readout_times[1])  # exc_ns
#         readout_time = int(readout_times[2])  # detect_ns
#     else:
#         # Fallback if only 2 arguments are provided
#         delay_ns = 0
#         readout_time = int(readout_times[0])  # detect_ns
#         pulse_time = int(readout_times[1])  # exc_ns

#     calc_readout_time = readout_time

#     # Set the virtual laser key
#     laser_vkey = "SPIN_READOUT"

#     print(f"Loading sequence: {sequence_file}")

#     # Map variables to the exact format expected by BOTH sequence files
#     # args = [delay_ns, exc_ns, detect_ns, laser_vkey, laser_power]
#     seq_args = [
#         delay_ns,
#         pulse_time,
#         readout_time,
#         laser_vkey,
#         laser_power,
#     ]

#     seq_args_string = tool_belt.encode_seq_args(seq_args)
#     ret_vals = pulsegen_server.stream_load(sequence_file, seq_args_string)  # LOAD
#     seq_time = ret_vals[0]

#     seq_time_s = seq_time / (10**9)  # s
#     expected_run_time = num_runs * (num_reps * seq_time_s + 1)  # s
#     expected_run_time_m = expected_run_time / 60  # m
#     print(" \nExpected run time: {:.2f} minutes. ".format(expected_run_time_m))

#     startFunctionTime = time.time()
#     start_timestamp = dm.get_time_stamp()

#     processed_tags = []

#     tool_belt.init_safe_stop()

#     for run_ind in range(num_runs):
#         print(" \nRun index: {}".format(run_ind))

#         if tool_belt.safe_stop():
#             break

#         seq_args_string = tool_belt.encode_seq_args(seq_args)

#         counter_server.start_tag_stream()
#         pulsegen_server.stream_start(int(num_reps))

#         channel_mapping = counter_server.get_channel_mapping()
#         gate_open_channel = channel_mapping[1]
#         gate_close_channel = channel_mapping[2]

#         current_tags = []
#         current_channels = []
#         num_processed_reps = 0

#         while num_processed_reps < num_reps:
#             if tool_belt.safe_stop():
#                 break

#             new_tags, new_channels = counter_server.read_tag_stream()
#             new_tags = numpy.array(new_tags, dtype=numpy.int64)

#             ret_vals = process_raw_buffer(
#                 new_tags,
#                 new_channels,
#                 current_tags,
#                 current_channels,
#                 gate_open_channel,
#                 gate_close_channel,
#             )

#             new_processed_tags, num_new_processed_reps = ret_vals
#             if num_new_processed_reps > 750000:
#                 print(
#                     "Processed {} reps out of 10^6 max".format(num_new_processed_reps)
#                 )
#                 print("Tell Matt that the time tagger is too slow!")

#             num_processed_reps += num_new_processed_reps

#             processed_tags.extend(new_processed_tags)

#         counter_server.stop_tag_stream()

#         # Save the data we have incrementally for long measurements
#         raw_data = {
#             "start_timestamp": start_timestamp,
#             "sequence_file": sequence_file,
#             "nv_sig": nv_sig,
#             "laser_power": laser_power,
#             "laser_vkey": laser_vkey,
#             "slider_1_pos": filter_pos[0],
#             "slider_3_pos": filter_pos[1],
#             "delay_ns": delay_ns,
#             "delay_ns-units": "ns",
#             "readout_time": readout_time,
#             "readout_time-units": "ns",
#             "pulse_time": pulse_time,
#             "pulse_time-units": "ns",
#             "calc_readout_time": calc_readout_time,
#             "calc_readout_time-units": "ns",
#             "num_reps": num_reps,
#             "num_runs": num_runs,
#             "run_ind": run_ind,
#             "num_bins": num_bins,
#             "processed_tags": processed_tags,
#             "processed_tags-units": "ps",
#         }

#         file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)
#         dm.save_raw_data(raw_data, file_path)

#     tool_belt.reset_cfm()

#     readout_time_ps = 1000 * calc_readout_time
#     binned_samples, bin_edges = numpy.histogram(
#         processed_tags, num_bins, (0, readout_time_ps)
#     )

#     bin_size_ns = calc_readout_time / num_bins
#     bin_size_s = bin_size_ns / 1e9
#     binned_samples_kcps = binned_samples / bin_size_s / 1e3 / num_reps / num_runs
#     bin_center_offset = bin_size_ns / 2
#     bin_centers_ns = numpy.linspace(0, readout_time, num_bins) + bin_center_offset

#     fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
#     ax2 = ax.twinx()
#     ax.plot(numpy.array(bin_centers_ns) / 10**3, binned_samples_kcps, "r-")
#     ax2.plot(numpy.array(bin_centers_ns) / 10**3, binned_samples, "r-")

#     ax.set_xlabel("X data")
#     ax.set_ylabel("kcps", color="k")
#     ax2.set_ylabel("Total Raw Counts", color="k")

#     ax.set_title("Lifetime")
#     ax.set_xlabel("Time after illumination (us)")

#     fig.canvas.draw()
#     fig.set_tight_layout(True)
#     fig.canvas.flush_events()

#     endFunctionTime = time.time()
#     time_elapsed = endFunctionTime - startFunctionTime

#     # Final save mapping
#     raw_data = {
#         "start_timestamp": start_timestamp,
#         "time_elapsed": time_elapsed,
#         "sequence_file": sequence_file,
#         "nv_sig": nv_sig,
#         "laser_power": laser_power,
#         "laser_vkey": laser_vkey,
#         "slider_1_pos": filter_pos[0],
#         "slider_3_pos": filter_pos[1],
#         "delay_ns": delay_ns,
#         "delay_ns-units": "ns",
#         "readout_time": readout_time,
#         "readout_time-units": "ns",
#         "pulse_time": pulse_time,
#         "pulse_time-units": "ns",
#         "calc_readout_time": calc_readout_time,
#         "calc_readout_time-units": "ns",
#         "num_bins": num_bins,
#         "num_reps": num_reps,
#         "num_runs": num_runs,
#         "binned_samples": binned_samples.tolist(),
#         "bin_centers": bin_centers_ns.tolist(),
#         "processed_tags": processed_tags,
#         "processed_tags-units": "ps",
#     }
#     print(file_path)

#     dm.save_figure(fig, file_path)
#     file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)
#     dm.save_raw_data(raw_data, file_path)
#     print("FIN --")


# def lifetime_json_to_csv(
#     file, folder, nv_data_dir="E:/Shared drives/Kolkowitz Lab Group/nvdata"
# ):
#     data = tool_belt.get_raw_data(file, folder)
#     binned_samples = data["binned_samples"]
#     bin_centers = data["bin_centers"]

#     csv_data = []

#     for bin_ind in range(len(bin_centers)):
#         row = []
#         row.append(bin_centers[bin_ind])
#         row.append(binned_samples[bin_ind])
#         csv_data.append(row)

#     tool_belt.write_csv(csv_data, file, folder)
