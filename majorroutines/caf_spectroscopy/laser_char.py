# -*- coding: utf-8 -*-
"""
This is a program to record the lifetime (right now, specifically of the Er
implanted materials fro mVictor brar's group).

It takes the same structure as a standard t1 measurement. We shine 532 nm
light, wait some time, and then read out the counts WITHOUT shining 532 nm
light.

Adding a variable 'filter' to pass into the function to signify what filter
was used to take the measurement (2/20/2020)

Created on Mon Nov 11 12:49:55 2019

@author: agardill
"""

# Imports

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


def process_raw_buffer(
    new_tags,
    new_channels,
    current_tags,
    current_channels,
    gate_open_channel,
    gate_close_channel,
):
    # The processing here will be bin_size agnostic

    # Tack the new data onto the leftover data (leftovers are necessary if
    # the last read contained a gate open without a matching gate close)
    current_tags.extend(new_tags)
    current_channels.extend(new_channels)
    current_channels_array = numpy.array(current_channels)

    # Find gate open clicks
    result = numpy.nonzero(current_channels_array == gate_open_channel)
    gate_open_click_inds = result[0].tolist()

    # Find gate close clicks
    result = numpy.nonzero(current_channels_array == gate_close_channel)
    gate_close_click_inds = result[0].tolist()

    new_processed_tags = []

    # Loop over the number of closes we have since there are guaranteed to
    # be opens
    num_closed_samples = len(gate_close_click_inds)
    for list_ind in range(num_closed_samples):
        gate_open_click_ind = gate_open_click_inds[list_ind]
        gate_close_click_ind = gate_close_click_inds[list_ind]

        # Extract all the counts between these two indices as a single sample
        rep = current_tags[gate_open_click_ind + 1 : gate_close_click_ind]
        rep = numpy.array(rep, dtype=numpy.int64)
        # Make relative to gate open
        rep -= current_tags[gate_open_click_ind]
        new_processed_tags.extend(rep.astype(int).tolist())

    # Clear processed tags
    if len(gate_close_click_inds) > 0:
        leftover_start = gate_close_click_inds[-1]
        del current_tags[0 : leftover_start + 1]
        del current_channels[0 : leftover_start + 1]

    return new_processed_tags, num_closed_samples


def main(
    nv_sig,
    apd_indices,
    readout_times,
    filter_pos,
    num_reps,
    num_runs,
    num_bins,
    use_wavegen=False,
    laser_power=None,
):
    if len(apd_indices) > 1:
        msg = "Currently lifetime only supports single APDs!!"
        raise NotImplementedError(msg)

    tool_belt.reset_cfm()
    repr_th_name = "irr4"
    # config =

    pulsegen_server = tool_belt.get_server_pulse_streamer()
    counter_server = tool_belt.get_server_counter()
    wavegen_server = tool_belt.get_server_wave_gen()

    # Initialize all the devices
    if len(filter_pos) != 0:
        slider_1 = tool_belt.get_server_slider_1()
        slider_3 = tool_belt.get_server_slider_3()

        # Set filter translators
        slider_1_pos, slider_3_pos = filter_pos
        slider_1.set_filter(slider_1_pos)
        slider_3.set_filter(slider_3_pos)

    # Define the times to be used in the sequence
    laser_name = "laser_COBO_515"
    laser_wavelength = 515

    # In ns
    readout_time = int(readout_times[0])
    pulse_time = int(readout_times[1])
    calc_readout_time = readout_time
    total_readout_time = ...

    # TODO!!
    # Set wavegen settings if triggering the laser
    if use_wavegen is True:
        if pulse_time * 2 < 1e3:
            if total_readout_time < 1e3:
                wavegen_server.set_TTL(
                    2,  # channel
                    pulse_time * 2,  # pulse_len
                    5,  # output_volt
                    2.5,  # threshold_volt
                )
            else:
                wavegen_server.set_TTL(
                    2,  # channel
                    pulse_time * 2,  # pulse_len
                    5,  # output_volt
                    2.5,  # threshold_volt
                )
        else:
            wavegen_server.set_TTL(
                2,  # channel
                pulse_time * 2,  # pulse_len
                5,  # output_volt
                2.5,  # threshold_volt
            )

    # Analyze the sequence
    # pulls the file of the sequence from serves/timing/sequencelibrary
    file_name = os.path.basename(__file__)
    print(file_name)
    seq_args = [
        readout_time,
        pulse_time,
        laser_name,
        laser_wavelength,
    ]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    ret_vals = pulsegen_server.stream_load(file_name, seq_args_string)  # LOAD
    seq_time = ret_vals[0]

    # Report the expected run time

    seq_time_s = seq_time / (10**9)  # s
    expected_run_time = num_runs * (num_reps * seq_time_s + 1)  # s
    expected_run_time_m = expected_run_time / 60  # m
    print(" \nExpected run time: {:.2f} minutes. ".format(expected_run_time_m))

    # Bit more setup

    # Record the start time
    startFunctionTime = time.time()
    # start_timestamp = tool_belt.get_time_stamp()
    # dm_folder = common.get_data_manager_folder()
    start_timestamp = dm.get_time_stamp()

    # opti_coords_list = []

    # Collect the data
    processed_tags = []

    # Start 'Press enter to stop...'
    tool_belt.init_safe_stop()

    for run_ind in range(num_runs):
        print(" \nRun index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tool_belt.safe_stop():
            break

        seq_args_string = tool_belt.encode_seq_args(seq_args)

        # Optimize
        # opti_coords = targeting.main_with_cxn(cxn, nv_sig, apd_indices)
        # opti_coords_list.append(opti_coords)

        # Expose the stream
        counter_server.start_tag_stream()
        # Stream the sequence
        # seq_args = [start_readout_time, end_readout_time, polarization_time,
        #         aom_delay_time, apd_indices[0]]
        # seq_args = [int(el) for el in seq_args]
        pulsegen_server.stream_start(int(num_reps))

        # Find the gate channel
        # The order of channel_mapping is APD, APD gate open, APD gate close
        channel_mapping = counter_server.get_channel_mapping()
        # print(channel_mapping)
        gate_open_channel = channel_mapping[1]
        gate_close_channel = channel_mapping[2]

        # Initialize state
        current_tags = []
        current_channels = []
        num_processed_reps = 0

        while num_processed_reps < num_reps:
            # Break out of the while if the user says stop
            if tool_belt.safe_stop():
                break

            new_tags, new_channels = counter_server.read_tag_stream()
            new_tags = numpy.array(new_tags, dtype=numpy.int64)
            # new = counter_server.read_counter_simple()  # N
            # print(new)
            # print(new_tags)
            # print(new_channels)

            ret_vals = process_raw_buffer(
                new_tags,
                new_channels,
                current_tags,
                current_channels,
                gate_open_channel,
                gate_close_channel,
            )
            # print(ret_vals)

            new_processed_tags, num_new_processed_reps = ret_vals
            # MCC test
            if num_new_processed_reps > 750000:
                print(
                    "Processed {} reps out of 10^6 max".format(num_new_processed_reps)
                )
                print("Tell Matt that the time tagger is too slow!")

            num_processed_reps += num_new_processed_reps

            processed_tags.extend(new_processed_tags)

        counter_server.stop_tag_stream()

        # Save the data we have incrementally for long measurements
        raw_data = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            # 'filter': filter,
            # 'reference_measurement?': reference,
            "laser_power": laser_power,
            "slider_1_pos": filter_pos[0],
            "slider_3_pos": filter_pos[1],
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
            "processed_tags": processed_tags,
            "processed_tags-units": "ps",
        }

        file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)
        dm.save_raw_data(raw_data, file_path)

        # # This will continuously be the same file path so we will overwrite
        # # the existing file with the latest version
        # file_path = tool_belt.get_file_path(
        #     __file__, start_timestamp, nv_sig["name"], "incremental"
        # )
        # tool_belt.save_raw_data(raw_data, file_path)

    # Hardware clean up
    tool_belt.reset_cfm()

    # Bin the data
    readout_time_ps = 1000 * calc_readout_time
    binned_samples, bin_edges = numpy.histogram(
        processed_tags, num_bins, (0, readout_time_ps)
    )

    # Compute the centers of the bins
    bin_size_ns = calc_readout_time / num_bins  # ns
    bin_size_s = bin_size_ns / 1e9
    binned_samples_kcps = binned_samples / bin_size_s / 1e3 / num_reps / num_runs
    bin_center_offset = bin_size_ns / 2
    bin_centers_ns = numpy.linspace(0, readout_time, num_bins) + bin_center_offset

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax2 = ax.twinx()
    ax.plot(numpy.array(bin_centers_ns) / 10**3, binned_samples_kcps, "r-")
    ax2.plot(numpy.array(bin_centers_ns) / 10**3, binned_samples, "r-")

    ax.set_xlabel("X data")
    ax.set_ylabel("kcps", color="k")
    ax2.set_ylabel("Total Raw Counts", color="k")

    ax.set_title("Lifetime")
    ax.set_xlabel("Time after illumination (us)")
    # ax.set_ylabel('kcps')

    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

    # Save the data
    endFunctionTime = time.time()
    time_elapsed = endFunctionTime - startFunctionTime

    raw_data = {
        "start_timestamp": start_timestamp,
        "time_elapsed": time_elapsed,
        "nv_sig": nv_sig,
        # "nv_sig-units": tool_belt.get_nv_sig_units(),
        # 'filter': filter,
        # 'reference_measurement?': reference,
        # 'voltage': voltage,
        "laser_power": laser_power,
        "slider_1_pos": filter_pos[0],
        "slider_3_pos": filter_pos[1],
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
        "processed_tags": processed_tags,
        "processed_tags-units": "ps",
    }
    print(file_path)

    # file_path = tool_belt.get_file_path(__file__, timestamp, nv_sig["name"])
    dm.save_figure(fig, file_path)
    # tool_belt.save_raw_data(raw_data, file_path)
    file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)
    dm.save_raw_data(raw_data, file_path)
    print("FIN --")


# # %%
# if __name__ == "__main__":
#     folder = "pc_rabi/branch_master/lifetime_v2/2022_09"
#     file = "2022_09_17-00_12_47-rubin-nv0_2022_09_16"

#     file_bckg = "2022_09_14-12_39_05-rubin-no_nv"

#     lifetime_json_to_csv(file, folder)

#     # data = tool_belt.get_raw_data(file_bckg, folder)
#     # bkgd= numpy.array(data['binned_samples'])
#     # decay_list = []

#     # file_list = tool_belt.get_file_list(
#     #     'pc_rabi/branch_master/lifetime_v2/2022_09/2022_09_18',
#     #     'txt',
#     # )
#     file_list = [
#         "2022_09_13-17_07_24-rubin-nv1_2022_08_10.txt",
#     ]

#     for file_name in file_list:
#         file = file_name[:-4]
#         lifetime_json_to_csv(file, folder)
