# -*- coding: utf-8 -*-
"""
Sweep the green-laser spin-readout pulse duration and measure shot-noise-
limited SNR per rep at each duration to find the readout time that
maximizes NV spin-readout SNR.

For each readout_ns in `readout_times_ns`:
    - override nv_sig.pulse_durations[VirtualLaserKey.SPIN_READOUT]
    - load `spin_contrast_simple.py` (MW-OFF ref gate, MW-ON sig gate)
    - run num_runs * num_reps alternating reps, drift-compensating between
      runs if requested
    - compute SNR per rep as in determine_standard_readout_params.py:
          snr_per_rep = (ref - sig) / sqrt(ref + sig) / sqrt(total_reps)

Save a summary, plot SNR vs readout duration, and report the best readout.

@author: chemistatcode
"""

import copy
import traceback

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tb
from utils import common
from utils import data_manager as dm
from utils.constants import VirtualLaserKey


SEQ_NAME = "spin_contrast_simple.py"


def _get_pulse_duration_ns(nv_sig, vkey, fallback_ns):
    pulse_durations = getattr(nv_sig, "pulse_durations", None)
    if pulse_durations is None:
        return int(fallback_ns)
    try:
        return int(pulse_durations.get(vkey, fallback_ns))
    except Exception:
        return int(fallback_ns)


def main(
    nv_sig,
    readout_times_ns,
    freq_center_ghz,
    num_reps,
    num_runs,
    uwave_ind,
    uwave_power_dbm=None,
    pi_pulse_ns=None,
    optimize_between_runs=True,
    do_plot=True,
):
    """Optimize the green spin-readout pulse duration via SNR per rep."""
    with common.labrad_connect() as cxn:
        return main_with_cxn(
            cxn,
            nv_sig,
            readout_times_ns,
            freq_center_ghz,
            num_reps,
            num_runs,
            uwave_ind,
            uwave_power_dbm,
            pi_pulse_ns,
            optimize_between_runs,
            do_plot,
        )


def main_with_cxn(
    cxn,
    nv_sig,
    readout_times_ns,
    freq_center_ghz,
    num_reps,
    num_runs,
    uwave_ind,
    uwave_power_dbm,
    pi_pulse_ns,
    optimize_between_runs,
    do_plot,
):
    # -------------------- Setup --------------------
    # tb.reset_cfm(cxn)
    kpl.init_kplotlib()

    pulsegen_server = tb.get_server_pulse_streamer()
    counter_server = tb.get_server_counter()
    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    # pi_pulse override: mutate live config dict; sequence picks it up via
    # get_virtual_sig_gen_dict on each stream_load. Restored in finally.
    orig_pi_pulse = vsg["pi_pulse"]
    if pi_pulse_ns is not None:
        vsg["pi_pulse"] = int(pi_pulse_ns)

    uwave_freq_ghz = float(freq_center_ghz)
    if uwave_power_dbm is None:
        uwave_power_dbm = float(vsg["uwave_power"])
    else:
        uwave_power_dbm = float(uwave_power_dbm)

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    readout_vkey = VirtualLaserKey.SPIN_READOUT
    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    pol_ns = _get_pulse_duration_ns(nv_sig, spin_pol_vkey, vld_pol["duration"])

    # -------------------- Normalize --------------------
    readout_times_ns = np.asarray(readout_times_ns, dtype=int).ravel()
    if readout_times_ns.size == 0:
        raise ValueError("readout_times_ns must contain at least one value")
    n = readout_times_ns.size

    ref_totals = np.full(n, np.nan, dtype=float)
    sig_totals = np.full(n, np.nan, dtype=float)
    contrasts = np.full(n, np.nan, dtype=float)
    snr_per_rep = np.full(n, np.nan, dtype=float)
    n_reps_used = np.zeros(n, dtype=int)

    # -------------------- Sweep --------------------
    tb.init_safe_stop()
    try:
        for i, readout_ns in enumerate(readout_times_ns):
            if tb.safe_stop():
                break

            readout_ns = int(readout_ns)
            print("\n" + "=" * 64)
            print(f"[{i + 1}/{n}] readout_ns = {readout_ns}")
            print("=" * 64)

            nv_sig_run = copy.deepcopy(nv_sig)
            nv_sig_run.pulse_durations[readout_vkey] = readout_ns

            seq_args = [
                int(pol_ns),
                int(readout_ns),
                int(uwave_ind),
                spin_pol_vkey,
                readout_vkey,
                None,
            ]
            seq_args_string = tb.encode_seq_args(seq_args)
            pulsegen_server.stream_load(SEQ_NAME, seq_args_string)

            ref_accum = 0
            sig_accum = 0
            n_reps_accum = 0

            for run_ind in range(int(num_runs)):
                if tb.safe_stop():
                    break

                if optimize_between_runs:
                    try:
                        targeting.compensate_for_drift(nv_sig_run, no_crash=True)
                    except Exception:
                        traceback.print_exc()
                    pulsegen_server.stream_load(SEQ_NAME, seq_args_string)

                sig_gen.set_amp(uwave_power_dbm)
                sig_gen.set_freq(uwave_freq_ghz)
                sig_gen.uwave_on()

                counter_server.start_tag_stream()
                try:
                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))
                    # new_counts = counter_server.read_counter_modulo_gates(
                    #     2, int(num_reps)
                    # )
                    new_counts = counter_server.read_counter_separate_gates(int(num_reps))

                finally:
                    try:
                        counter_server.stop_tag_stream()
                    except Exception:
                        pass

                arr = np.array(new_counts, dtype=np.int64)
                if arr.size == 0:
                    continue
                arr = arr.reshape(-1, 2)
                ref_accum += int(arr[:, 0].sum())
                sig_accum += int(arr[:, 1].sum())
                n_reps_accum += int(arr.shape[0])

                print(
                    f"  run {run_ind + 1}/{num_runs}: "
                    f"ref+={arr[:, 0].sum()}, sig+={arr[:, 1].sum()}, "
                    f"reps+={arr.shape[0]}"
                )

            if n_reps_accum == 0:
                print("  No reps collected for this readout; skipping.")
                continue

            ref_totals[i] = float(ref_accum)
            sig_totals[i] = float(sig_accum)
            n_reps_used[i] = n_reps_accum
            contrasts[i] = (
                (ref_accum - sig_accum) / ref_accum if ref_accum > 0 else np.nan
            )
            denom = np.sqrt(ref_accum**2 - sig_accum**2)  #Update to def of SNR
            snr_per_rep[i] = (
                (ref_accum - sig_accum) / denom 
                if denom > 0
                else np.nan
            )

            print(
                f"  totals: ref={ref_accum}, sig={sig_accum}, reps={n_reps_accum}"
            )
            print(
                f"  contrast={100 * contrasts[i]:.2f}%, "
                f"SNR per rep={snr_per_rep[i]:.4f}"
            )
    finally:
        try:
            sig_gen.uwave_off()
        except Exception:
            pass
        vsg["pi_pulse"] = orig_pi_pulse
        # tb.reset_cfm(cxn)

    # -------------------- Best & summary --------------------
    ok = np.isfinite(snr_per_rep)
    i_best = int(np.nanargmax(snr_per_rep)) if ok.any() else None

    # -------------------- Save --------------------
    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "readout_times_ns": readout_times_ns.tolist(),
        "ref_totals": ref_totals.tolist(),
        "sig_totals": sig_totals.tolist(),
        "contrasts": contrasts.tolist(),
        "snr_per_rep": snr_per_rep.tolist(),
        "n_reps_used": n_reps_used.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_freq_ghz": uwave_freq_ghz,
        "uwave_power_dbm": uwave_power_dbm,
        "pi_pulse_ns": int(vsg["pi_pulse"]) if pi_pulse_ns is None else int(pi_pulse_ns),
        "pol_ns": int(pol_ns),
        "sequence": SEQ_NAME,
        "optimize_between_runs": bool(optimize_between_runs),
    }

    # -------------------- Plot --------------------
    fig = None
    if do_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        if ok.any():
            ax.plot(
                readout_times_ns[ok],
                snr_per_rep[ok],
                "o-",
                color="darkorange",
                markersize=6,
                linewidth=1.5,
                label="SNR per rep",
            )
            if i_best is not None:
                ax.axvline(
                    readout_times_ns[i_best],
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    label=(
                        f"Best: {readout_times_ns[i_best]} ns  "
                        f"(SNR/rep={snr_per_rep[i_best]:.4f})"
                    ),
                )
        ax.set_xlabel("Green spin-readout pulse duration (ns)")
        ax.set_ylabel("SNR per rep")
        ax.set_title("Green readout optimization (SNR)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best")
        plt.tight_layout()

    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)
    print(f"\nSaved sweep to {file_path}")

    # -------------------- Summary table --------------------
    print("\n" + "=" * 78)
    print("GREEN READOUT SWEEP SUMMARY (SNR-based)")
    print("=" * 78)
    print(
        f"{'readout (ns)':>14} {'SNR/rep':>10} {'contrast':>10} "
        f"{'ref total':>12} {'sig total':>12}"
    )
    for t, s, c, rt, st in zip(
        readout_times_ns, snr_per_rep, contrasts, ref_totals, sig_totals
    ):
        if np.isfinite(s):
            print(
                f"{int(t):>14d} {s:>10.4f} {100 * c:>9.2f}% "
                f"{rt:>12.0f} {st:>12.0f}"
            )
        else:
            print(
                f"{int(t):>14d} {'--':>10} {'--':>10} {'--':>12} {'--':>12}"
            )
    if i_best is not None:
        print("-" * 78)
        print(
            f"Best readout: {int(readout_times_ns[i_best])} ns, "
            f"SNR/rep = {snr_per_rep[i_best]:.4f}, "
            f"contrast = {100 * contrasts[i_best]:.2f}%"
        )
    print("=" * 78)

    if do_plot:
        plt.show()
    return raw_data
