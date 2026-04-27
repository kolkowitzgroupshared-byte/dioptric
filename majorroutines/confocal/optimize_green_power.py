# -*- coding: utf-8 -*-
"""
Sweep green-laser power (and optionally spin-readout duration) and measure
spin-readout SNR at each (power, readout) point. Per-rep SNR is defined as

    SNR_per_rep = (ref - sig) / sqrt(ref + sig) / sqrt(num_reps)

where `ref` and `sig` are total counts summed across all reps in the MW-OFF
(ms=0) and MW-ON (ms=+/-1) gates of the `spin_contrast_simple.py` sequence.
This is the standard figure of merit for optimizing green power: raw counts
rise with power, but shot noise grows with them and charge-state mixing eats
into contrast at high power, so SNR peaks at an intermediate power.

User-facing power is in mW; conversion to W is done internally before calling
the laser server's set_power.

@author: chemistatcode 
Date: April 15th 2026
"""

import copy
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

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
    powers_mW,
    readout_times_ns=None,
    num_reps=1000,
    uwave_ind=0,
    uwave_freq_ghz=None,
    uwave_power_dbm=None,
    pi_pulse_ns=None,
    laser_name="laser_COBO_520",
    settle_time=0.2,
    do_plot=True,
):
    cxn = common.labrad_connect()
    return main_with_cxn(
        cxn,
        nv_sig,
        powers_mW,
        readout_times_ns,
        num_reps,
        uwave_ind,
        uwave_freq_ghz,
        uwave_power_dbm,
        pi_pulse_ns,
        laser_name,
        settle_time,
        do_plot,
    )


def main_with_cxn(
    cxn,
    nv_sig,
    powers_mW,
    readout_times_ns,
    num_reps,
    uwave_ind,
    uwave_freq_ghz,
    uwave_power_dbm,
    pi_pulse_ns,
    laser_name,
    settle_time,
    do_plot,
):
    # -------------------- Setup --------------------
    tb.reset_cfm()
    kpl.init_kplotlib()

    pulsegen_server = tb.get_server_pulse_streamer()
    counter_server = tb.get_server_counter()
    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    # Override pi_pulse by mutating the live config dict; the sequence fetches
    # pi_pulse via get_virtual_sig_gen_dict on each stream_load, so the
    # override propagates. Restored in the finally block below.
    orig_pi_pulse = vsg["pi_pulse"]
    if pi_pulse_ns is not None:
        vsg["pi_pulse"] = int(pi_pulse_ns)

    if uwave_freq_ghz is None:
        uwave_freq_ghz = float(vsg["frequency"])
    else:
        uwave_freq_ghz = float(uwave_freq_ghz)
    if uwave_power_dbm is None:
        uwave_power_dbm = float(vsg["uwave_power"])
    else:
        uwave_power_dbm = float(uwave_power_dbm)

    laser_server = getattr(cxn, laser_name)
    orig_power_w = None
    try:
        orig_power_w = float(laser_server.get_power())
    except Exception:
        pass

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    readout_vkey = VirtualLaserKey.SPIN_READOUT
    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    vld_readout = tb.get_virtual_laser_dict(readout_vkey)
    pol_ns = _get_pulse_duration_ns(nv_sig, spin_pol_vkey, vld_pol["duration"])

    # -------------------- Normalize inputs --------------------
    powers_mW = np.asarray(powers_mW, dtype=float).ravel()
    powers_W = powers_mW * 1e-3

    if readout_times_ns is None:
        default_readout = _get_pulse_duration_ns(
            nv_sig, readout_vkey, vld_readout["duration"]
        )
        readout_times_ns = [int(default_readout)]
    else:
        readout_times_ns = np.atleast_1d(readout_times_ns).astype(int).tolist()

    n_readouts = len(readout_times_ns)
    n_powers = powers_mW.size
    ref_totals = np.full((n_readouts, n_powers), np.nan, dtype=float)
    sig_totals = np.full((n_readouts, n_powers), np.nan, dtype=float)
    contrasts = np.full((n_readouts, n_powers), np.nan, dtype=float)
    snr_per_rep = np.full((n_readouts, n_powers), np.nan, dtype=float)

    # -------------------- Sweep --------------------
    tb.init_safe_stop()
    try:
        for ir, readout in enumerate(readout_times_ns):
            if tb.safe_stop():
                break

            seq_args = [
                int(pol_ns),
                int(readout),
                int(uwave_ind),
                spin_pol_vkey,
                readout_vkey,
                None,
            ]
            seq_args_string = tb.encode_seq_args(seq_args)
            pulsegen_server.stream_load(SEQ_NAME, seq_args_string)

            sig_gen.set_amp(uwave_power_dbm)
            sig_gen.set_freq(uwave_freq_ghz)
            sig_gen.uwave_on()

            for ip, p_w in enumerate(powers_W):
                if tb.safe_stop():
                    break

                laser_server.set_power(float(p_w))
                time.sleep(settle_time)

                counter_server.start_tag_stream()
                try:
                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))
                    new_counts = counter_server.read_counter_modulo_gates(
                        2, int(num_reps)
                    )
                finally:
                    try:
                        counter_server.stop_tag_stream()
                    except Exception:
                        pass

                arr = np.array(new_counts, dtype=np.int64)
                if arr.size == 0:
                    continue
                arr = arr.reshape(-1, 2)
                ref_tot = float(arr[:, 0].sum())
                sig_tot = float(arr[:, 1].sum())
                n_used = arr.shape[0]

                ref_totals[ir, ip] = ref_tot
                sig_totals[ir, ip] = sig_tot
                contrasts[ir, ip] = (
                    (ref_tot - sig_tot) / ref_tot if ref_tot > 0 else np.nan
                )
                denom = np.sqrt(ref_tot + sig_tot)
                snr_per_rep[ir, ip] = (
                    (ref_tot - sig_tot) / denom / np.sqrt(n_used)
                    if denom > 0
                    else np.nan
                )

                print(
                    f"readout={readout} ns, P={powers_mW[ip]:.3f} mW -> "
                    f"ref={ref_tot:.0f}, sig={sig_tot:.0f}, "
                    f"contrast={100 * contrasts[ir, ip]:.2f}%, "
                    f"SNR/rep={snr_per_rep[ir, ip]:.4f}"
                )
    finally:
        try:
            sig_gen.uwave_off()
        except Exception:
            pass
        try:
            restore = orig_power_w if orig_power_w is not None else 0.0
            laser_server.set_power(float(restore))
        except Exception:
            traceback.print_exc()
        vsg["pi_pulse"] = orig_pi_pulse
        tb.reset_cfm()

    # -------------------- Per-readout argmax (optimal power) --------------------
    optimal_powers_mW = [np.nan] * n_readouts
    optimal_snr = [np.nan] * n_readouts
    for ir in range(n_readouts):
        row = snr_per_rep[ir]
        ok = np.isfinite(row)
        if ok.any():
            idx = np.nanargmax(row)
            optimal_powers_mW[ir] = float(powers_mW[idx])
            optimal_snr[ir] = float(row[idx])
            print(
                f"Optimal @ readout={readout_times_ns[ir]} ns: "
                f"P={optimal_powers_mW[ir]:.3f} mW, SNR/rep={optimal_snr[ir]:.4f}"
            )

    # -------------------- Plot --------------------
    fig = None
    if do_plot and n_powers > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_readouts, 1)))

        for ir in range(n_readouts):
            row = snr_per_rep[ir]
            ok = np.isfinite(row)
            label = f"{readout_times_ns[ir]} ns"
            if np.isfinite(optimal_powers_mW[ir]):
                label += f"  (P_opt={optimal_powers_mW[ir]:.3f} mW)"
            ax.plot(
                powers_mW[ok],
                row[ok],
                "o-",
                color=colors[ir],
                label=label,
            )
            if np.isfinite(optimal_powers_mW[ir]):
                ax.axvline(
                    optimal_powers_mW[ir], ls=":", color=colors[ir], alpha=0.5
                )

        ax.set_xlabel("Green power (mW)")
        ax.set_ylabel("SNR per rep")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()

    # -------------------- Save --------------------
    ts = dm.get_time_stamp()
    nv_label = getattr(nv_sig, "name", "nv")
    file_path = dm.get_file_path(__file__, ts, nv_label)

    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "laser_name": laser_name,
        "uwave_ind": int(uwave_ind),
        "uwave_freq_ghz": uwave_freq_ghz,
        "uwave_power_dbm": uwave_power_dbm,
        "pi_pulse_ns": int(vsg["pi_pulse"]),
        "pol_ns": int(pol_ns),
        "powers_mW": powers_mW.tolist(),
        "readout_times_ns": [int(r) for r in readout_times_ns],
        "num_reps": int(num_reps),
        "ref_totals": ref_totals.tolist(),
        "sig_totals": sig_totals.tolist(),
        "contrasts": contrasts.tolist(),
        "snr_per_rep": snr_per_rep.tolist(),
        "optimal_powers_mW": optimal_powers_mW,
        "optimal_snr_per_rep": optimal_snr,
        "sequence": SEQ_NAME,
        "settle_time": settle_time,
    }
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    print(f"Saved to {file_path}")
    return raw_data


if __name__ == "__main__":
    # Example usage: load a recent nv_sig and run a small 2D sweep.
    recent_file = "<timestamp-nv_sig_file>"  # e.g. "2026_04_15-10_30_00-nv1"
    try:
        nv_sig = dm.get_raw_data(recent_file)["nv_sig"]
    except Exception:
        print("Update `recent_file` in __main__ to point at a saved nv_sig.")
        raise

    main(
        nv_sig,
        powers_mW=np.linspace(0.05, 5.0, 10),
        readout_times_ns=[610],
        num_reps=1000,
        uwave_ind=0,
        laser_name="laser_COBO_520",
    )
