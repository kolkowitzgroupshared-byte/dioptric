# -*- coding: utf-8 -*-
"""
Single resonance measurement without base routine.

Very close to old working ESR style:
- stream_load once
- set microwave frequency
- stream_start
- read_counter_modulo_gates(2, 1)

Returns:
    raw_data, proc_data
"""
import sys
import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
from utils import positioning as pos
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode
from utils.constants import CoordsKey


def main(
    nv_sig,
    freq_center_ghz,
    freq_span_mhz,
    num_steps,
    num_reps,
    num_runs,
    uwave_ind,
    uwave_power_dbm=None,
    laser_power=None,
    optimize_between_runs=False,
    do_plot=True,
    shuffle=False,
    norm_mode=NormMode.SINGLE_VALUED,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    readout_vkey = VirtualLaserKey.SPIN_READOUT
    vld = tb.get_virtual_laser_dict(readout_vkey)
    readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, int(vld["duration"])))
    print(f"Readout duration (ns): {readout_ns}")
    readout_ns = int(readout_ns)

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    pol_ns = int(nv_sig.pulse_durations.get(spin_pol_vkey, int(vld_pol["duration"])))
    print(f"Polarization duration (ns): {pol_ns}")

    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_power_dbm is None:
        uwave_power_dbm = vsg.get("uwave_power", None)
    if uwave_power_dbm is not None:
        sig_gen.set_amp(float(uwave_power_dbm))
    sig_gen.uwave_on()

    seq_file = "resonance.py"
    seq_args = [
        int(pol_ns),
        int(readout_ns),
        int(uwave_ind),
        spin_pol_vkey,
        readout_vkey,
        laser_power,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)
    pulsegen_server.stream_load(seq_file, seq_args_string)

    span_ghz = freq_span_mhz * 1e-3
    freqs_ghz = np.linspace(
        freq_center_ghz - span_ghz / 2,
        freq_center_ghz + span_ghz / 2,
        num_steps,
    )

    sig_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ref_counts = np.full((num_runs, num_steps), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()

    if do_plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Normalized signal")
        ax.set_title("Confocal ESR")
        (line,) = ax.plot([], [], "o-")
    else:
        fig = None
        ax = None
        line = None

    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print(f"Run {run_ind + 1}/{num_runs}")

        if tb.safe_stop():
            break

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

                # Each row is [ref, sig] for one repetition
                count_arr = np.array(new_counts, dtype=np.int64)
                print(f"  count_arr shape: {count_arr.shape}")  # should be (num_reps, 2)
                ref_counts[run_ind, step_ind] = count_arr[:, 0].sum()
                sig_counts[run_ind, step_ind] = count_arr[:, 1].sum()
    
                ref_val = ref_counts[run_ind, step_ind]
                sig_val = sig_counts[run_ind, step_ind]
                norm_val = sig_val / ref_val if ref_val > 0 else float("nan")

                print(
                    f"  f={f:.6f} GHz | "
                    f"ref={int(ref_val)}, sig={int(sig_val)}, norm={norm_val:.4f}"
                )

        finally:
            try:
                counter_server.stop_tag_stream()
            except Exception:
                pass

        if do_plot:
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_runs = sig_counts[: run_ind + 1] / ref_counts[: run_ind + 1]
            norm_mean = np.nanmean(norm_runs, axis=0)

            line.set_data(freqs_ghz, norm_mean)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        if optimize_between_runs:
            try:
                z_coords, z_counts = targeting.optimize(nv_sig, coords_key=CoordsKey.Z)
                galvo_key = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
                xy_coords, xy_counts = targeting.optimize(nv_sig, coords_key=galvo_key)
                print(f"  Optimized: Z={z_coords}, XY={xy_coords}, counts={xy_counts}")
            except Exception as e:
                print(f"  Optimization failed on run {run_ind}: {e}")
            for f_num in plt.get_fignums():
                if plt.figure(f_num) is not fig:
                    plt.close(f_num)
    
    print(f"run {run_ind}: norm_mean min={norm_mean.min():.6f} max={norm_mean.max():.6f}, runs averaged={run_ind+1}")

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_runs = sig_counts / ref_counts

    norm_mean = np.nanmean(norm_runs, axis=0)
    norm_ste = np.nanstd(norm_runs, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(norm_runs), axis=0)
    )

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "freqs_ghz": freqs_ghz.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_power_dbm": uwave_power_dbm,
        "readout_ns": int(readout_ns),
        "sig_counts": sig_counts.tolist(),
        "ref_counts": ref_counts.tolist(),
        "norm_mean": norm_mean.tolist(),
        "norm_ste": norm_ste.tolist(),
        "opti_coords_list": opti_coords_list,
    }

    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)
    print(f"Saved data to {file_path}")

    tb.reset_cfm()
    return raw_data


if __name__ == "__main__":
    # example:
    kpl.init_kplotlib()
    data = dm.get_raw_data(file_stem="2026_03_09-13_37_07-(lovelace)", load_npz=True)
    nv_sig = data["nv_sig"]
    sig_counts = np.asarray(data["sig_counts"]) 
    ref_counts = np.asarray(data["ref_counts"])   
    norm_mean = np.asarray(data["norm_mean"])
    norm_ste = np.asarray(data["norm_ste"])
    freqs_ghz = np.asarray(data["freqs_ghz"])

    plt.figure()
    plt.errorbar(freqs_ghz, norm_mean, yerr=norm_ste, fmt="o-")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Normalized signal")
    plt.title("Confocal ESR")
    plt.grid(True) 
    kpl.show(block =True)
    