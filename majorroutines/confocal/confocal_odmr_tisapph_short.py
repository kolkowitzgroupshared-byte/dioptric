# -*- coding: utf-8 -*-
"""
Short ODMR with Ti:Sapph singlet shelving.

Sweeps MW frequency while using a fixed Ti:Sapph wavelength.
Each frequency step produces two APD-gated measurements:
    gate 0 = reference (MW pi, no Ti:Sapph)
    gate 1 = signal    (MW pi, Ti:Sapph ON)

On-resonance MW drives population to ms=±1, which Ti:Sapph shelves
into the dark singlet → signal drops relative to reference.
Off-resonance stays ms=0, no shelving → signal ≈ reference.

The contrast (sig-ref)/ref vs MW frequency reveals the ODMR dip
enhanced by singlet shelving.
"""

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
from utils import tool_belt as tb
from utils import positioning as pos
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode, CoordsKey


def main(
    nv_sig,
    freq_center_ghz,
    freq_span_mhz,
    num_steps,
    num_reps,
    num_runs,
    uwave_ind=0,
    uwave_power_dbm=None,
    probe_ns=None,
    readout_ns=None,
    pol_ns=None,
    laser_power=None,
    optimize_between_runs=False,
    do_plot=True,
    shuffle=False,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    # Laser config
    readout_vkey = VirtualLaserKey.SPIN_READOUT
    spin_pol_vkey = VirtualLaserKey.SPIN_POL

    vld_read = tb.get_virtual_laser_dict(readout_vkey)
    if readout_ns is None:
        readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, int(vld_read["duration"])))
    readout_ns = int(readout_ns)

    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    if pol_ns is None:
        pol_ns = int(nv_sig.pulse_durations.get(spin_pol_vkey, int(vld_pol["duration"])))
    pol_ns = int(pol_ns)

    # Ti:Sapph probe duration
    if probe_ns is None:
        vld_singlet = tb.get_virtual_laser_dict(VirtualLaserKey.SINGLET_DRIVE)
        probe_ns = int(vld_singlet["duration"])
    probe_ns = int(probe_ns)

    print(f"Readout duration (ns): {readout_ns}")
    print(f"Polarization duration (ns): {pol_ns}")
    print(f"Ti:sapph probe duration (ns): {probe_ns}")

    # Sig gen setup
    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))
    if uwave_power_dbm is None:
        uwave_power_dbm = vsg.get("uwave_power", None)

    # Frequency array
    span_ghz = freq_span_mhz * 1e-3
    freqs_ghz = np.linspace(
        freq_center_ghz - span_ghz / 2,
        freq_center_ghz + span_ghz / 2,
        num_steps,
    )

    # Sequence setup
    seq_file = "odmr_tisapph_short.py"
    seq_args = [
        int(pol_ns),
        int(probe_ns),
        int(readout_ns),
        int(uwave_ind),
        spin_pol_vkey,
        readout_vkey,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)

    # Data arrays
    ref_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    sig_counts = np.full((num_runs, num_steps), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()

    if do_plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("MW frequency (GHz)")
        ax.set_ylabel("Norm signal (sig/ref)")
        ax.set_title("Short ODMR with Ti:Sapph shelving")
        (line,) = ax.plot([], [], marker="o")
    else:
        fig = None

    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print(f"Run {run_ind + 1}/{num_runs}")

        if tb.safe_stop():
            break

        # Optimization between runs
        if optimize_between_runs:
            try:
                z_coords, z_counts = targeting.optimize(nv_sig, coords_key=CoordsKey.Z)
                print(f"  Optimized Z: {z_coords}, counts={z_counts}")
            except Exception as e:
                print(f"  Z optimization failed on run {run_ind}: {e}")
            try:
                galvo_key = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
                xy_coords, xy_counts = targeting.optimize(nv_sig, coords_key=galvo_key)
                print(f"  Optimized XY: {xy_coords}, counts={xy_counts}")
            except Exception as e:
                print(f"  XY optimization failed on run {run_ind}: {e}")
            if do_plot:
                for f_num in plt.get_fignums():
                    if plt.figure(f_num) is not fig:
                        plt.close(f_num)

        # Reload sequence and sig gen (optimization resets servers)
        pulsegen_server.stream_load(seq_file, seq_args_string)
        if uwave_power_dbm is not None:
            sig_gen.set_amp(float(uwave_power_dbm))
        sig_gen.uwave_on()

        sweep_order = np.arange(num_steps)
        if shuffle:
            np.random.shuffle(sweep_order)

        counter_server.start_tag_stream()
        try:
            for step_ind in sweep_order:
                if tb.safe_stop():
                    break

                f = float(freqs_ghz[step_ind])
                sig_gen.set_freq(f)

                counter_server.clear_buffer()
                pulsegen_server.stream_start(int(num_reps))

                new_counts = counter_server.read_counter_modulo_gates(2, int(num_reps))
                count_arr = np.array(new_counts, dtype=np.int64)

                ref_counts[run_ind, step_ind] = count_arr[:, 0].sum()
                sig_counts[run_ind, step_ind] = count_arr[:, 1].sum()

                ref_val = ref_counts[run_ind, step_ind]
                sig_val = sig_counts[run_ind, step_ind]
                norm_val = sig_val / ref_val if ref_val > 0 else float("nan")

                print(
                    f"  f={f:.6f} GHz | "
                    f"ref={int(ref_val)}, sig={int(sig_val)}, "
                    f"norm={norm_val:.4f}"
                )

        finally:
            try:
                counter_server.stop_tag_stream()
            except Exception:
                pass

        # Update plot
        if do_plot:
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_runs = sig_counts[: run_ind + 1] / ref_counts[: run_ind + 1]
            norm_mean = np.nanmean(norm_runs, axis=0)

            line.set_data(freqs_ghz, norm_mean)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

    # Final processing
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_runs = sig_counts / ref_counts
    norm_mean = np.nanmean(norm_runs, axis=0)
    norm_ste = np.nanstd(norm_runs, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(norm_runs), axis=0)
    )

    print(f"\nFinal norm: min={np.nanmin(norm_mean):.4f}, max={np.nanmax(norm_mean):.4f}")

    # Save data
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "freqs_ghz": freqs_ghz.tolist(),
        "ref_counts": ref_counts.tolist(),
        "sig_counts": sig_counts.tolist(),
        "norm_mean": norm_mean.tolist(),
        "norm_ste": norm_ste.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_power_dbm": uwave_power_dbm,
        "freq_center_ghz": freq_center_ghz,
        "freq_span_mhz": freq_span_mhz,
        "probe_ns": int(probe_ns),
        "readout_ns": int(readout_ns),
        "pol_ns": int(pol_ns),
    }

    nv_name = nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    print(f"Data saved to {file_path}")

    return raw_data
