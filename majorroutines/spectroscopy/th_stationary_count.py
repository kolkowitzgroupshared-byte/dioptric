# -*- coding: utf-8 -*-
"""
Sit on the passed coordinates and record counts.

Updated for NVSig dataclass and VirtualLaserKey-based optics config.
"""

import matplotlib.pyplot as plt
import numpy as np

import utils.kplotlib as kpl
import utils.positioning as pos
import utils.tool_belt as tool_belt
from utils import common
from utils.constants import CoordsKey, CountFormat, VirtualLaserKey


def main(
    nv_sig,
    run_time,
    disable_opt=None,
    nv_minus_init=False,
    nv_zero_init=False,
    background_subtraction=False,
):
    # -------------------- Initial setup --------------------
    if disable_opt is not None:
        nv_sig.disable_opt = disable_opt

    tool_belt.reset_cfm()

    # Imaging readout duration (ns): per-NV override, otherwise config default
    vld_img = tool_belt.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld_img["duration"]))
    )
    readout_sec = readout * 1e-9

    ## get servers
    pulsegen_server = tool_belt.get_server_pulse_streamer()
    counter_server = tool_belt.get_server_counter()

    # -------------------- Laser selection / power --------------------
    # Imaging laser (VirtualLaserKey.IMAGING)
    readout_laser = vld_img["physical_name"]
    tool_belt.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tool_belt.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # get argument and seq files
    delay = 0
    seq_args = [delay, readout, readout_laser, readout_power]
    seq_args_string = tool_belt.encode_seq_args(seq_args)
    seq_name = "simple_readout.py"

    # Program pulse generator
    period = pulsegen_server.stream_load(seq_name, seq_args_string)[0]  # ns
    total_num_samples = int(run_time / period)
    run_time_s = run_time * 1e-9

    # -------------------- Figure setup --------------------
    samples = np.full(total_num_samples, np.nan, dtype=float)  # NaNs don't get plotted
    write_pos = 0
    x_vals = (np.arange(total_num_samples) + 1) * (period * 1e-9)  # elapsed time in s
    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    kpl.plot_line(ax, x_vals, samples)
    ax.set_xlim(-0.05 * run_time_s, 1.05 * run_time_s)
    ax.set_xlabel("Time (s)")

    cfg = common.get_config_dict()
    count_fmt: CountFormat = cfg["count_format"]  # CountFormat.KCPS or CountFormat.RAW
    # count_fmt = CountFormat.RAW
    ax.set_ylabel("Raw counts")
    # ax.set_ylabel("Kcps" if count_fmt == CountFormat.KCPS is not None else "Counts")
    # ax.set_ylabel("Count rateS (kcps)")
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass

    # -------------------- Acquisition --------------------
    counter_server.start_tag_stream()
    # stream_start(-1): run until stopped
    pulsegen_server.stream_start(-1)
    tool_belt.init_safe_stop()

    leftover_sample = None
    snr = lambda nv, bg: (nv - bg) / np.sqrt(max(nv, 1))  # avoid /0

    def _ensure_1d_counts(arr_like):
        """Flattens list/np arrays of counts to 1D ints."""
        if arr_like is None:
            return np.array([], dtype=int)
        arr = np.array(arr_like)
        if arr.ndim == 0:
            return np.array([int(arr)], dtype=int)
        if arr.dtype != np.int64 and arr.dtype != np.int32:
            arr = arr.astype(int, copy=False)
        # If modulo-gates (N,2) during charge init, we'll diff later
        return arr

    while True:
        if tool_belt.safe_stop():
            break

        # Read new samples
        new = counter_server.read_counter_simple()  # N
        new = _ensure_1d_counts(new)

        # Background subtraction interleave handling
        if background_subtraction and new.size > 0:
            if leftover_sample is not None:
                new = np.insert(new, 0, leftover_sample)
                leftover_sample = None
            if new.size % 2 == 1:
                leftover_sample = int(new[-1])
                new = new[:-1]
            if new.size > 0:
                # pair (NV, BG) -> SNR
                paired = [
                    snr(int(new[2 * i]), int(new[2 * i + 1]))
                    for i in range(new.size // 2)
                ]
                new = np.array(paired, dtype=float)

        n_new = new.size
        if n_new == 0:
            continue

        # Write into circular-ish buffer area: if overflow, drop earliest
        num_written = int(np.count_nonzero(~np.isnan(samples)))
        overflow = (num_written + n_new) - total_num_samples
        if overflow > 0:
            # shift left and append
            keep = total_num_samples - n_new
            keep = max(keep, 0)
            samples[:keep] = samples[num_written - keep : num_written]
            samples[keep:] = new[-n_new:]
            write_pos = total_num_samples
        else:
            samples[write_pos : write_pos + n_new] = new
            write_pos += n_new

        # # Update plot in kcps
        # samples_kcps = samples / (1e3 * readout_sec)
        # kpl.plot_line_update(ax, x=x_vals, y=samples_kcps, relim_x=False)
        kpl.plot_line_update(ax, x=x_vals, y=samples, relim_x=False)

    # -------------------- Cleanup + stats --------------------
    tool_belt.reset_cfm()

    if write_pos > 0:
        avg = float(np.nanmean(samples[:write_pos])) / (1e3 * readout_sec)
        std = float(np.nanstd(samples[:write_pos])) / (1e3 * readout_sec)
    else:
        avg, std = 0.0, 0.0

    print(f"Average: {avg}")
    print(f"Standard deviation: {std}")
    return avg, std
