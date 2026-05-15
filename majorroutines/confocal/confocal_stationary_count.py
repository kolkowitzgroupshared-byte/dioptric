# -*- coding: utf-8 -*-
"""
Sit on the passed coordinates and record counts.

Updated for NVSig dataclass and VirtualLaserKey-based optics config.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import utils.kplotlib as kpl
import utils.positioning as pos
import utils.tool_belt as tool_belt
from utils import common
from utils import data_manager as dm
from utils.constants import CoordsKey, VirtualLaserKey, CountFormat

def main(
    nv_sig,
    run_time,
    disable_opt=None,
    nv_minus_init=False,
    nv_zero_init=False,
    background_subtraction=False,
    background_coords=None,  # SAMPLE-space [x, y] (or [x, y, z] — z ignored)
    save_data=True,
    save_file_path=None,
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

    charge_init = bool(nv_minus_init or nv_zero_init)

    pulsegen_server = (
        tool_belt.get_server_pulse_streamer()
    )  # or get_server_pulse_streamer() in your stack
    counter_server = tool_belt.get_server_counter()

    # -------------------- Background subtraction motion (optional) --------------------
    if background_subtraction:
        # Drift-adjust both NV and background target in SAMPLE XY
        nv_xy = pos.adjust_coords_for_drift(nv_sig=nv_sig, coords_key=CoordsKey.SAMPLE)
        nv_x, nv_y = nv_xy[:2] if hasattr(nv_xy, "__len__") else (nv_xy, None)

        if background_coords is None:
            raise ValueError(
                "background_subtraction=True but background_coords is None"
            )

        # Ensure bg is XY list; ignore/strip Z if present
        if hasattr(background_coords, "__len__"):
            bg_xy = list(background_coords[:2])
        else:
            raise ValueError(
                "background_coords must be a 2-element iterable [x, y] (SAMPLE space)"
            )

        bg_xy = pos.adjust_coords_for_drift(coords=bg_xy, coords_key=CoordsKey.SAMPLE)
        bg_x, bg_y = bg_xy[:2]

        x_voltages, y_voltages = pos.get_scan_two_point_2d(nv_x, nv_y, bg_x, bg_y)

        # SAMPLE positioner server (piezo)
        xy_server = pos.get_positioner_server(CoordsKey.SAMPLE)
        # Some drivers accept (x, y, loop=True); prefer 2-arg form and fall back to 3-arg if available
        try:
            xy_server.load_stream_xy(x_voltages, y_voltages)
        except TypeError:
            xy_server.load_stream_xy(x_voltages, y_voltages, True)

    # -------------------- Laser selection / power --------------------
    # Imaging laser (VirtualLaserKey.IMAGING)
    readout_laser = vld_img["physical_name"]
    tool_belt.set_filter(nv_sig, VirtualLaserKey.IMAGING)
    readout_power = tool_belt.set_laser_power(nv_sig, VirtualLaserKey.IMAGING)

    # Charge init selection
    if charge_init:
        if nv_minus_init:
            init_key = VirtualLaserKey.CHARGE_POL  # green, polarize to NV-
        elif nv_zero_init:
            init_key = VirtualLaserKey.ION  # red, ionize to NV0
        else:
            init_key = None

        vld_init = tool_belt.get_virtual_laser_dict(init_key)
        init = int(nv_sig.pulse_durations.get(init_key, int(vld_init["duration"])))
        init_laser = vld_init["physical_name"]

        tool_belt.set_filter(nv_sig, init_key)
        init_power = tool_belt.set_laser_power(nv_sig, init_key)

        seq_args = [init, readout, init_laser, init_power, readout_laser, readout_power]
        seq_args_string = tool_belt.encode_seq_args(seq_args)
        seq_name = "charge_init-simple_readout_background_subtraction.py"
    else:
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
    count_fmt: CountFormat = cfg["count_format"] # CountFormat.KCPS or CountFormat.RAW
    # count_fmt = CountFormat.RAW
    ax.set_ylabel("Raw counts")
    # ax.set_ylabel("Kcps" if count_fmt == CountFormat.KCPS is not None else "Counts")
    # ax.set_ylabel("Count rateS (kcps)")
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass

    # -------------------- Save-file setup (active streaming) --------------------
    save_fh = None
    if save_data:
        if save_file_path is None:
            timestamp = dm.get_time_stamp()
            name = getattr(nv_sig, "name", "nv")
            save_file_path = dm.get_file_path(__file__, timestamp, name)
        else:
            timestamp = dm.get_time_stamp()
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        save_fh = open(save_file_path, "w", encoding="utf-8", buffering=1)
        save_fh.write("# Stationary count data\n")
        save_fh.write(f"# timestamp: {timestamp}\n")
        save_fh.write(f"# nv_name: {getattr(nv_sig, 'name', 'nv')}\n")
        save_fh.write(f"# run_time_ns: {int(run_time)}\n")
        save_fh.write(f"# period_ns: {int(period)}\n")
        save_fh.write(f"# readout_ns: {int(readout)}\n")
        save_fh.write(f"# charge_init: {bool(charge_init)}\n")
        save_fh.write(f"# background_subtraction: {bool(background_subtraction)}\n")
        save_fh.write("# columns: sample_index\ttime_s\tcounts\n")
        save_fh.flush()
        os.fsync(save_fh.fileno())
        print(f"Stationary count data streaming to: {save_file_path}")

    cum_index = 0

    # -------------------- Acquisition --------------------
    try:
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
            if charge_init:
                new = counter_server.read_counter_modulo_gates(2)  # Nx2
            else:
                new = counter_server.read_counter_simple()  # N

            new = _ensure_1d_counts(new)

            # Convert modulo-gates to single count per gate
            if charge_init and new.size > 0:
                # Expect Nx2; if flattened, reshape safely
                new = np.array(new)
                new = new.reshape((-1, 2)) if new.ndim == 1 else new
                new = np.maximum(new[:, 0] - new[:, 1], 0)

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

            # Stream every sample to disk before any plotting / ring-buffer logic
            # so cancellations and buffer overwrites never lose data.
            if save_fh is not None:
                for v in new:
                    t_s = (cum_index + 1) * period * 1e-9
                    save_fh.write(f"{cum_index}\t{t_s:.9e}\t{float(v):.6f}\n")
                    cum_index += 1
                save_fh.flush()
                os.fsync(save_fh.fileno())
            else:
                cum_index += n_new

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
    finally:
        if save_fh is not None:
            try:
                save_fh.flush()
                os.fsync(save_fh.fileno())
            except Exception:
                pass
            save_fh.close()
            print(f"Stationary count data saved to: {save_file_path}")
