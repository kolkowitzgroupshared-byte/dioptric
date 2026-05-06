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
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import NormMode, VirtualLaserKey


def main(
    nv_sig,
    freq_center_ghz=2.8786,
    freq_span_mhz=200.0,
    num_steps=51,
    num_reps=1,
    num_runs=40,
    uwave_ind=0,
    readout_ns=10e6,  # None, # if not NONE shows normalized plot
    uwave_power_dbm=None,
    laser_power=None,
    do_plot=True,
    do_save=True,
    shuffle=False,
    norm_mode=NormMode.SINGLE_VALUED,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    # readout_vkey=VirtualLaserKey.SPIN_READOUT
    # readout_vkey=VirtualLaserKey.if hasattr(nv_sig, "readout_vkey"):
    readout_vkey = VirtualLaserKey.SPIN_READOUT

    vld = tb.get_virtual_laser_dict(readout_vkey)
    if readout_ns is None:
        readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, int(vld["duration"])))
    readout_ns = int(readout_ns)

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    pol_ns = int(nv_sig.pulse_durations.get(spin_pol_vkey, int(vld_pol["duration"])))

    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_power_dbm is None:
        uwave_power_dbm = vsg.get("uwave_power", None)
    if uwave_power_dbm is not None:
        sig_gen.set_amp(float(uwave_power_dbm))
    sig_gen.uwave_on()

    # sequence is loaded once
    seq_file = "resonance_caf.py"
    seq_args = [
        # pol_ns,
        readout_ns,
        int(uwave_ind),
        readout_vkey.name if hasattr(readout_vkey, "name") else str(readout_vkey),
        laser_power,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)
    pulsegen_server.stream_load(seq_file, seq_args_string)

    # frequency axis
    span_ghz = freq_span_mhz * 1e-3
    freqs_ghz = np.linspace(
        freq_center_ghz - span_ghz / 2,
        freq_center_ghz + span_ghz / 2,
        num_steps,
    )

    sweep_order = np.arange(num_steps)
    if shuffle:
        np.random.shuffle(sweep_order)

    sig_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ref_counts = np.full((num_runs, num_steps), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()
    opti_coords_list = []

    if do_plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Normalized signal")
        (line,) = ax.plot([], [], "o-")
    else:
        fig = None

    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print(f"Run {run_ind + 1}/{num_runs}")

        if tb.safe_stop():
            break

        counter_server.start_tag_stream()

        try:
            for step_ind in sweep_order:
                if tb.safe_stop():
                    break

                f = float(freqs_ghz[step_ind])
                sig_gen.set_freq(f)

                counter_server.clear_buffer()
                pulsegen_server.stream_start(int(num_reps))

                new_counts = counter_server.read_counter_modulo_gates(2, 1)
                sample_counts = new_counts[0]
                # print("len(new_counts) =", len(new_counts))
                # print("first few =", new_counts[:5])

                # gate0 = ref, gate1 = sig
                ref_counts[run_ind, step_ind] = sample_counts[0]
                sig_counts[run_ind, step_ind] = sample_counts[1]

        finally:
            try:
                counter_server.stop_tag_stream()
            except Exception:
                pass

        if do_plot:
            valid_runs = np.isfinite(sig_counts[: run_ind + 1]) & np.isfinite(
                ref_counts[: run_ind + 1]
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                # norm_runs = sig_counts[: run_ind + 1] / ref_counts[: run_ind + 1]
                norm_runs = sig_counts[: run_ind + 1] / np.maximum(
                    ref_counts[: run_ind + 1], 1
                )

            norm_mean = np.nanmean(norm_runs, axis=0)

            line.set_data(freqs_ghz, norm_mean)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

    # process
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_runs = sig_counts / np.maximum(ref_counts, 1)

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

    # print("Resonance measurement complete.")
    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(fig, file_path)

    # file_path = dm.get_file_path(__file__, timestamp, nv_sig["name"])
    # if fig is not None:
    #     dm.save_figure(fig, file_path)
    #     dm.save_raw_data(raw_data, file_path)
    print(f"Saved data to {file_path}")

    # if do_save:
    #     file_path = dm.get_file_path(__file__, timestamp, nv_sig["name"])
    #     print('test')
    #     if fig is not None:
    #         dm.save_figure(fig, file_path)
    #     dm.save_raw_data(raw_data, file_path)
    #     print(f"Saved data to {file_path}")

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
    kpl.show(block=True)


# # -*- coding: utf-8 -*-
# """
# Confocal ESR experiment using base routine.
# Sweeps microwave frequency, reads signal and reference counts via APD tagger.

# Adapted to CaF2 code.

# Created on Mar 16, 2026
# @author: j-chen1
# """

# import numpy as np

# from majorroutines.caf_spectroscopy.sideillum_base_routine import main as base_routine
# from utils import data_manager as dm
# from utils import tool_belt as tb


# def main(
#     coords,
#     freqs,
#     num_reps,
#     num_runs,
#     apd_ch,
#     apd_time,
#     use_reference=True,
#     norm_style="contrast",
# ):
#     """
#     Parameters:
#         seq_file: seq program path
#         scan_coords: [x, y, z] center
#         freqs: array of frequencies in GHz
#         num_reps: repetitions per point
#         num_runs: number of full sweeps
#         apd_ch: APD channel
#         apd_time: APD collection time in seconds
#         run_nir_fn: function(bool) to toggle NIR (optional)
#         use_reference: whether to use signal/reference gates
#     """
#     pulse_streamer = tb.get_server_pulse_streamer()
#     tagger = tb.get_server_time_tagger()

#     seq_file = "resonance_caf.py"

#     def apd_read_fn(tagger, apd_ch, apd_time):
#         if use_reference:
#             counts = tb.read_apd_2gates(tagger, apd_ch, apd_time)
#         else:
#             counts = tb.read_apd_counts(tagger, apd_ch, apd_time)
#         return counts

#     # Call the base routine
#     raw_data = base_routine(
#         scan_coords=coords,
#         num_steps=len(freqs),
#         num_reps=num_reps,
#         num_runs=num_runs,
#         # seq_args_fn=seq_args_fn,
#         apd_read_fn=apd_read_fn,  # TODO: ??
#         tagger=tagger,
#         apd_ch=apd_ch,
#         apd_time=apd_time,
#     )

#     # Optional normalization
#     # if use_reference:
#     #     raw_counts = np.array(raw_data["counts"])
#     #     norm, ste = tb.process_counts_array(
#     #         raw_counts, gate_mode="2gate", norm_style=norm_style
#     #     )
#     #     raw_data["norm"] = norm.tolist()
#     #     raw_data["norm_ste"] = ste.tolist()
#     #     raw_data["norm_style"] = norm_style

#     raw_data["freqs"] = freqs.tolist()
#     raw_data["use_reference"] = use_reference
#     dm.save_raw_data(raw_data, dm.get_file_path(__file__, raw_data["timestamp"]))
#     return raw_data


# if __name__ == "__main__":
#     main()
