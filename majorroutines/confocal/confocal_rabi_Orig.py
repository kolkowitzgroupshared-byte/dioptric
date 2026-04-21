# -*- coding: utf-8 -*-
"""
Single-NV / single-pixel Rabi sweep.

- set microwave amp/freq once
- for each run:
    - optional targeting
    - for each tau:
        - stream_load once for that tau
        - stream_start
        - read_counter_modulo_gates(2, 1)

Sequence convention:
    gate 0 = reference  (no MW pulse)
    gate 1 = signal     (MW pulse of duration tau_ns)

Returns:
    raw_data, proc_data

Created on March 17th, 2026

@author: sbchand
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
# from figures.zfs_vs_t.zfs_vs_t_main import fig
# from figures.zfs_vs_t.deconvolve_spectral_function import fig
# from majorroutines.calibration import optimize_xy
from utils import tool_belt as tb
from utils import kplotlib as kpl
import majorroutines.targeting as targeting
from utils import positioning as pos
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode, CoordsKey


def _build_tau_ns_list(
    uwave_dur_ns_list=None,
    uwave_dur_min_ns=None,
    uwave_dur_max_ns=None,
    num_steps=None,
):
    if uwave_dur_ns_list is not None:
        tau_ns_list = np.asarray(uwave_dur_ns_list, dtype=int).ravel()
    else:
        if uwave_dur_min_ns is None or uwave_dur_max_ns is None or num_steps is None:
            raise ValueError(
                "Provide either uwave_dur_ns_list OR "
                "(uwave_dur_min_ns, uwave_dur_max_ns, num_steps)."
            )
        tau_ns_list = np.linspace(
            int(uwave_dur_min_ns),
            int(uwave_dur_max_ns),
            int(num_steps),
        )
        tau_ns_list = np.rint(tau_ns_list).astype(int)

    tau_ns_list = np.unique(tau_ns_list)
    if len(tau_ns_list) == 0:
        raise ValueError("tau_ns_list is empty.")
    if np.any(tau_ns_list < 0):
        raise ValueError("All microwave pulse durations must be >= 0 ns.")
    return tau_ns_list


def _process_rabi_counts(sig_counts, ref_counts, num_reps, readout_ns, norm_mode):
    """
    sig_counts, ref_counts shape = (num_runs, num_steps)
    Returns per-step processed arrays.
    """
    sig_counts = np.asarray(sig_counts, dtype=float)
    ref_counts = np.asarray(ref_counts, dtype=float)

    num_steps = sig_counts.shape[1]

    sig_kcps = np.full(num_steps, np.nan, dtype=float)
    ref_kcps = np.full(num_steps, np.nan, dtype=float)
    norm = np.full(num_steps, np.nan, dtype=float)
    norm_ste = np.full(num_steps, np.nan, dtype=float)
    num_valid_runs = np.zeros(num_steps, dtype=int)

    for step_ind in range(num_steps):
        valid_mask = np.isfinite(sig_counts[:, step_ind]) & np.isfinite(
            ref_counts[:, step_ind]
        )
        num_valid_runs[step_ind] = int(np.sum(valid_mask))

        if not np.any(valid_mask):
            continue

        sig_col = sig_counts[valid_mask, step_ind].reshape(-1, 1)
        ref_col = ref_counts[valid_mask, step_ind].reshape(-1, 1)

        sig_kcps_i, ref_kcps_i, norm_i, norm_ste_i = tb.process_counts(
            sig_col,
            ref_col,
            int(num_reps),
            int(readout_ns),
            norm_mode=norm_mode,
        )

        sig_kcps[step_ind] = float(sig_kcps_i[0])
        ref_kcps[step_ind] = float(ref_kcps_i[0])
        norm[step_ind] = float(norm_i[0])
        norm_ste[step_ind] = float(norm_ste_i[0])

    contrast = 1.0 - norm

    return {
        "sig_kcps": sig_kcps,
        "ref_kcps": ref_kcps,
        "norm": norm,
        "norm_ste": norm_ste,
        "contrast": contrast,
        "num_valid_runs": num_valid_runs,
    }


def main(
    nv_sig,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    num_steps,
    uwave_ind=0,
    readout_ns=None,
    uwave_power_dbm=10,
    uwave_freq_ghz=2.8322,
    laser_power=None,
    optimize_between_runs=True,
    optimize_xy_kwargs=None,
    do_plot=True,
    do_save=True,
    norm_mode=NormMode.SINGLE_VALUED,
):
    tb.reset_cfm()
    kpl.init_kplotlib()
    
    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    tau_ns_list = np.linspace(min_tau, max_tau, num_steps)
    tau_ns_list = np.rint(tau_ns_list).astype(int)
    tau_ns_list = np.unique(tau_ns_list)

    readout_vkey = VirtualLaserKey.SPIN_READOUT
    pol_vkey = VirtualLaserKey.SPIN_POL

    pol_dict = tb.get_virtual_laser_dict(pol_vkey)
    polarization_ns = int(nv_sig.pulse_durations.get(pol_vkey, pol_dict["duration"]))

    if readout_ns is None:
        readout_dict = tb.get_virtual_laser_dict(readout_vkey)
        readout_ns = int(
            nv_sig.pulse_durations.get(readout_vkey, readout_dict["duration"])
        )
    readout_ns = int(readout_ns)

    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_power_dbm is None:
        uwave_power_dbm = vsg.get("uwave_power", None)

    if uwave_power_dbm is not None:
        sig_gen.set_amp(float(uwave_power_dbm))

    freq_ghz = uwave_freq_ghz if uwave_freq_ghz is not None else vsg["frequency"]
    sig_gen.set_freq(float(freq_ghz))
    sig_gen.uwave_on()

    seq_file = "rabi.py"

    ref_counts = np.full((num_runs, len(tau_ns_list)), np.nan)
    sig_counts = np.full((num_runs, len(tau_ns_list)), np.nan)

    timestamp = dm.get_time_stamp()

    if do_plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("MW pulse duration τ (ns)")
        ax.set_ylabel("Normalized signal")
        ax.set_title("Rabi")
        (line_norm,) = ax.plot([], [], marker="o")
    else:
        fig = None
        ax = None
        line_norm = None

    tb.init_safe_stop()
    for run_ind in range(num_runs):
        print(f"Run {run_ind + 1}/{num_runs}")

        if tb.safe_stop():
            break
        
        if optimize_between_runs:
            targeting.compensate_for_drift(nv_sig)
        
        
        
        # if optimize_between_runs:
        #     try:
        #         # 1D Z optimization
        #         z_coords, z_counts = targeting.optimize(nv_sig, coords_key=CoordsKey.Z)
        #         # 1D XY galvo optimization
        #         galvo_key = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
        #         xy_coords, xy_counts = targeting.optimize(nv_sig, coords_key=galvo_key)
        #         print(f"  Optimized: Z={z_coords}, XY={xy_coords}, counts={xy_counts}")
        #     except Exception as e:
        #         print(f"  Optimization failed on run {run_ind}: {e}")
        #     # Close optimize plots without closing the Rabi figure
        #     for f_num in plt.get_fignums():
        #         if plt.figure(f_num) is not fig:
        #             plt.close(f_num)

        # Re-enable sig gen after optimization (reset_cfm turns it off)
        if uwave_power_dbm is not None:
            sig_gen.set_amp(float(uwave_power_dbm))
        sig_gen.set_freq(float(freq_ghz))
        sig_gen.uwave_on()

        # Open stream ONCE per run, not per tau step
        counter_server.start_tag_stream()

        try:
            for step_ind, tau_ns in enumerate(tau_ns_list):
                if tb.safe_stop():
                    break

                seq_args = [
                    int(tau_ns),
                    int(polarization_ns),
                    int(readout_ns),
                    int(uwave_ind),
                    pol_vkey.name,
                    readout_vkey.name,
                    laser_power,
                ]
                
                if step_ind == 0 and run_ind == 0:
                    seq_args_string = tb.encode_seq_args(seq_args)
                    ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)
                    print(f"  Sequence period: {ret_vals} ns (new rabi.py loaded)")
                counter_server.clear_buffer()
                pulsegen_server.stream_start(int(num_reps))
                
                new_counts = counter_server.read_counter_modulo_gates(2, int(num_reps))

                # Sum across all reps (each entry is [ref, sig] for one rep)
                count_arr = np.array(new_counts, dtype=np.int64)
                ref_counts[run_ind, step_ind] = count_arr[:, 0].sum()
                sig_counts[run_ind, step_ind] = count_arr[:, 1].sum()

                ref_val = ref_counts[run_ind, step_ind]
                sig_val = sig_counts[run_ind, step_ind]
                norm_val = sig_val / ref_val if ref_val > 0 else float("nan")
                print(
                    f"tau={int(tau_ns):>4d} ns | "
                    f"ref={int(ref_val)}, sig={int(sig_val)}, "
                    f"norm={norm_val:.4f}"
                )

                # Update plot after each tau step for responsiveness
                if do_plot:
                    proc_partial = _process_rabi_counts(
                        sig_counts[: run_ind + 1, :],
                        ref_counts[: run_ind + 1, :],
                        int(num_reps),
                        int(readout_ns),
                        norm_mode,
                    )
                    line_norm.set_data(tau_ns_list, proc_partial["norm"])
                    ax.relim()
                    ax.autoscale_view()
                    plt.pause(0.01)
        finally:
            try:
                counter_server.stop_tag_stream()
            except Exception:
                pass


    proc_arrays = _process_rabi_counts(
        sig_counts,
        ref_counts,
        int(num_reps),
        int(readout_ns),
        norm_mode,
    )

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "freq_ghz": float(freq_ghz),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_power_dbm": uwave_power_dbm,
        "polarization_ns": int(polarization_ns),
        "readout_ns": int(readout_ns),
        "tau_ns_list": tau_ns_list.tolist(),
        #"opti_coords_list": opti_coords_list,
        "sig_counts": sig_counts.tolist(),
        "ref_counts": ref_counts.tolist(),
    }

    proc_data = {
        "freq_ghz": float(freq_ghz),
        "tau_ns_list": tau_ns_list.tolist(),
        "sig_kcps": proc_arrays["sig_kcps"].tolist(),
        "ref_kcps": proc_arrays["ref_kcps"].tolist(),
        "norm": proc_arrays["norm"].tolist(),
        "norm_ste": proc_arrays["norm_ste"].tolist(),
        "contrast": proc_arrays["contrast"].tolist(),
        "num_valid_runs": proc_arrays["num_valid_runs"].tolist(),
    }
    
    # Image file saving
    timestamp = dm.get_time_stamp()
    file_path  = dm.get_file_path(__file__, timestamp, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(fig, file_path)
    print(f"Saved data to {file_path}")

    tb.reset_cfm()
    return raw_data, proc_data


if __name__ == "__main__":
    # example:
    # raw, proc = main(
    #     nv_sig=nv_sig,
    #     freq_ghz=2.8786,
    #     num_reps=10000,
    #     num_runs=10,
    #     uwave_dur_min_ns=0,
    #     uwave_dur_max_ns=300,
    #     num_steps=31,
    #     uwave_ind=0,
    # )
    pass