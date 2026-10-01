# -*- coding: utf-8 -*-
"""
performs single shot lifetime measurements for caf. can do either continuous or pulsed depending on sequence selected

faster variant of lifetime_caf_single_shot:
- one hardware histogram is armed for the whole measurement and read cumulatively
  (no re-creating / re-syncing the Histogram every run)
- waits for the pulse streamer to actually report finished (requires the
  has_finished setting on pulse_gen_SWAB_82), instead of a blind fixed sleep
- incremental raw data saves only every save_every runs

@author:alyssa-matthews
"""

import os
import time

import matplotlib.pyplot as plt
import numpy

import utils.tool_belt as tool_belt
from utils import common
from utils import data_manager as dm
from utils.constants import ModMode, VirtualLaserKey


def laser_on_constant(pulsegen_server, laser_vkey):
    """Hold the laser on via pulse streamer constant(), wired the same way
    tool_belt.process_laser_seq wires it inside the sequence.
    """
    laser_name = tool_belt.get_physical_laser_name(laser_vkey)
    config = common.get_config_dict()
    pulser_wiring = config["Wiring"]["PulseGen"]
    mod_mode = config["Optics"]["PhysicalLasers"][laser_name]["mod_mode"]

    if mod_mode is ModMode.DIGITAL:
        pulsegen_server.constant([pulser_wiring[f"do_{laser_name}_dm"]])
    elif mod_mode is ModMode.ANALOG:
        laser_power = tool_belt.get_virtual_laser_dict(laser_vkey).get("laser_power")
        if laser_power is None:
            raise ValueError(f"No laser_power set for {laser_vkey} (analog mod)")
        pulsegen_server.constant(
            [], [pulser_wiring[f"ao_{laser_name}_am"]], [float(laser_power)]
        )


def wait_for_sequence(pulsegen_server, run_time_s, timeout_s, settle_s=0.05):
    """Sleep for the expected run time, then poll the pulse streamer until it
    reports the stream is finished.
    """
    time.sleep(run_time_s)

    deadline = time.monotonic() + timeout_s
    while not pulsegen_server.has_finished():
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Pulse streamer did not finish within {timeout_s:.1f} s "
                f"of the expected run time ({run_time_s:.2f} s)."
            )
        time.sleep(0.01)

    # Give the time tagger a moment to process the last tags
    time.sleep(settle_s)


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
    save_every=10,  # incremental raw data save every N runs
    presat_time_ms=0,  # laser on for this long before each run to saturate; 0 = off
):
    if len(apd_indices) > 1:
        msg = "Currently lifetime only supports single APDs!!"
        raise NotImplementedError(msg)

    tool_belt.reset_cfm()
    repr_th_name = "irr4"

    pulsegen_server = tool_belt.get_server_pulse_streamer()
    counter_server = tool_belt.get_server_counter()

    # if not hasattr(pulsegen_server, "has_finished"):
    #     msg = (
    #         "Pulse streamer server has no has_finished setting. Restart the "
    #         "pulse_gen_SWAB_82 server, then restart this Python console so the "
    #         "LabRAD connection picks up the new setting."
    #     )
    #     raise RuntimeError(msg)

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
        readout_time = int(readout_times[2])  # detect_ns (decay bin after laser off)
    else:
        # Fallback if only 2 arguments are provided
        delay_ns = 0
        readout_time = int(readout_times[0])  # detect_ns
        pulse_time = int(readout_times[1])  # exc_ns

    calc_readout_time = (
        delay_ns + pulse_time + readout_time
    )  # gate is open for full exc + decay bin

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
    expected_run_time = num_runs * (
        num_reps * seq_time_s + presat_time_ms / 1e3 + 0.2
    )  # s
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
    # finish_timeout_s = max(1.0, 0.1 * run_time_s)

    # Running total of counts, read cumulatively from the hardware histogram
    binned_samples = numpy.zeros(num_bins, dtype=numpy.int64)

    file_path = dm.get_file_path(__file__, start_timestamp, repr_th_name)

    static_data = {
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
        "num_bins": num_bins,
        "presat_time_ms": presat_time_ms,
        "presat_time_ms-units": "ms",
    }

    # Arm the hardware histogram ONCE. It accumulates across all runs; the pulse
    # streamer is idle between runs so no start triggers arrive in the gaps.
    counter_server.start_histogram(
        gate_open_channel, apd_channel, bin_size_ps, num_bins
    )

    runs_completed = 0
    try:
        for run_ind in range(num_runs):
            run_start = time.perf_counter()
            print(f" \nRun index: {run_ind}")

            if tool_belt.safe_stop():
                break

            # Pre-saturate: hold the laser on, then hand straight over to the sequence
            if presat_time_ms > 0:
                laser_on_constant(pulsegen_server, VirtualLaserKey[laser_vkey])
                # busy-wait, time.sleep on Windows can overshoot by several ms
                t_end = time.perf_counter() + presat_time_ms / 1e3
                while time.perf_counter() < t_end:
                    pass
                # constant() replaces the uploaded sequence, so re-load it.
                # The laser stays on until stream_start takes over.
                pulsegen_server.stream_load(sequence_file, seq_args_string)

            # Fire the laser sequence and wait for the streamer to report done
            pulsegen_server.stream_start(int(num_reps))
            time.sleep(run_time_s + 0.1)

            after_start = time.perf_counter() - run_start
            print(f"Run time: {after_start:.3f} s")

            # Cumulative read: this is the total over all completed runs
            before_read = time.perf_counter() - run_start
            binned_samples = numpy.array(
                counter_server.read_histogram(), dtype=numpy.int64
            )
            runs_completed = run_ind + 1
            after_read = time.perf_counter() - run_start

            print(f"Run read time: {after_read - before_read:.3f} s")

            # Save the data incrementally, but not every run
            if runs_completed % save_every == 0:
                dm.save_raw_data(
                    {
                        **static_data,
                        "run_ind": run_ind,
                        "binned_samples": binned_samples.tolist(),
                    },
                    file_path,
                )

            overhead_s = time.perf_counter() - run_start - run_time_s
            print(f"Run overhead: {overhead_s:.3f} s")

    finally:
        counter_server.stop_histogram()

    tool_belt.reset_cfm()

    # Calculate bin properties for plotting
    bin_size_ns = calc_readout_time / num_bins
    bin_size_s = bin_size_ns / 1e9
    binned_samples_kcps = (
        binned_samples / bin_size_s / 1e3 / num_reps / max(runs_completed, 1)
    )
    bin_center_offset = bin_size_ns / 2
    bin_centers_ns = numpy.arange(num_bins) * bin_size_ns + bin_center_offset

    if calc_readout_time >= 1e9:
        time_divisor, time_unit = 1e9, "s"
    elif calc_readout_time >= 1e6:
        time_divisor, time_unit = 1e6, "ms"
    elif calc_readout_time >= 1e3:
        time_divisor, time_unit = 1e3, "us"
    else:
        time_divisor, time_unit = 1, "ns"

    fig, ax = plt.subplots(1, 1, figsize=(10, 8.5))
    ax2 = ax.twinx()
    ax.plot(numpy.array(bin_centers_ns) / time_divisor, binned_samples_kcps, "r-")
    ax2.plot(numpy.array(bin_centers_ns) / time_divisor, binned_samples, "r-")

    ax.set_ylabel("kcps", color="k")
    ax2.set_ylabel("Total Raw Counts", color="k")

    ax.set_title("Lifetime")
    ax.set_xlabel(f"Time after illumination ({time_unit})")

    fig.canvas.draw()
    fig.set_tight_layout(True)
    fig.canvas.flush_events()

    endFunctionTime = time.time()
    time_elapsed = endFunctionTime - startFunctionTime

    # Final save mapping
    raw_data = {
        **static_data,
        "time_elapsed": time_elapsed,
        "runs_completed": runs_completed,
        "save_every": save_every,
        "binned_samples": binned_samples.tolist(),
        "bin_centers": bin_centers_ns.tolist(),
    }
    print(file_path)

    dm.save_figure(fig, file_path)
    dm.save_raw_data(raw_data, file_path)
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
