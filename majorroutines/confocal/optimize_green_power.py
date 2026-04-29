# -*- coding: utf-8 -*-
"""
Sweep green-laser power (and optionally spin-readout duration) and measure
spin-readout SNR at each (power, readout) point. Per-rep SNR is defined as

    SNR = (ref - sig) / sqrt(ref + sig)

where `ref` and `sig` are total counts summed across all runs in the MW-OFF
(ms=0) and MW-ON (ms=+/-1) gates of the `spin_contrast_simple.py` sequence.
This is the standard figure of merit for optimizing green power: raw counts
rise with power, but shot noise grows with them and charge-state mixing eats
into contrast at high power, so SNR peaks at an intermediate power.

User-facing power is in mW; conversion to W is done internally before calling
the laser server's set_power.

@author: chemistatcode
Date: April 15th 2026
"""

import time
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
    powers_mW,
    readout_times_ns=None,
    num_reps=1000,
    num_runs=1,
    uwave_ind=0,
    uwave_freq_ghz=None,
    uwave_power_dbm=None,
    pi_pulse_ns=None,
    laser_name="laser_COBO_520",
    settle_time=1.0,
    settle_tol_frac=0.02,
    optimize_between_runs=False,
    optimize_every_n_powers=None,
    randomize_power_order=False,
    do_plot=True,
):
    cxn = common.labrad_connect()
    return main_with_cxn(
        cxn,
        nv_sig,
        powers_mW,
        readout_times_ns,
        num_reps,
        num_runs,
        uwave_ind,
        uwave_freq_ghz,
        uwave_power_dbm,
        pi_pulse_ns,
        laser_name,
        settle_time,
        settle_tol_frac,
        optimize_between_runs,
        optimize_every_n_powers,
        randomize_power_order,
        do_plot,
    )


def main_with_cxn(
    cxn,
    nv_sig,
    powers_mW,
    readout_times_ns,
    num_reps,
    num_runs,
    uwave_ind,
    uwave_freq_ghz,
    uwave_power_dbm,
    pi_pulse_ns,
    laser_name,
    settle_time,
    settle_tol_frac,
    optimize_between_runs,
    optimize_every_n_powers,
    randomize_power_order,
    do_plot,
):
    # -------------------- Setup --------------------
    # Don't reset_cfm at entry — match confocal_optimize_green_readout.py so the
    # green-laser feedthrough state is the same as the comparison routine.
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

    # Digital channel for the green-laser TTL gate. We hold this HIGH during
    # the settle window so get_actual_power() reads the actual emitted power;
    # otherwise the laser is dark between sequences and `pa?` returns ~0 W.
    cfg = common.get_config_dict()
    try:
        green_do_chan = int(cfg["Wiring"]["PulseGen"][f"do_{laser_name}_dm"])
    except Exception:
        green_do_chan = None

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    readout_vkey = VirtualLaserKey.SPIN_READOUT
    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    vld_readout = tb.get_virtual_laser_dict(readout_vkey)
    pol_ns = _get_pulse_duration_ns(nv_sig, spin_pol_vkey, vld_pol["duration"])

    # -------------------- Normalize inputs --------------------
    powers_mW = np.asarray(powers_mW, dtype=float).ravel()
    powers_W = powers_mW * 1e-3
    num_runs = int(num_runs)

    if readout_times_ns is None:
        default_readout = _get_pulse_duration_ns(
            nv_sig, readout_vkey, vld_readout["duration"]
        )
        readout_times_ns = [int(default_readout)]
    else:
        readout_times_ns = np.atleast_1d(readout_times_ns).astype(int).tolist()

    n_readouts = len(readout_times_ns)
    n_powers = powers_mW.size

    # Per-run storage — shape (num_runs, n_readouts, n_powers)
    ref_counts_all = np.full((num_runs, n_readouts, n_powers), np.nan, dtype=float)
    sig_counts_all = np.full((num_runs, n_readouts, n_powers), np.nan, dtype=float)
    actual_powers_w_all = np.full((num_runs, n_readouts, n_powers), np.nan, dtype=float)

    rng = np.random.default_rng()

    # Final summary arrays
    ref_totals = np.full((n_readouts, n_powers), np.nan, dtype=float)
    sig_totals = np.full((n_readouts, n_powers), np.nan, dtype=float)
    contrasts = np.full((n_readouts, n_powers), np.nan, dtype=float)
    snr_per_rep = np.full((n_readouts, n_powers), np.nan, dtype=float)

    # -------------------- Live figure --------------------
    if do_plot and n_powers > 0:
        fig, (ax_snr, ax_contrast) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        fig.suptitle("Green power optimization")
        ax_snr.set_ylabel("SNR")
        ax_snr.grid(True, linestyle="--", alpha=0.5)
        ax_contrast.set_xlabel("Green power (mW)")
        ax_contrast.set_ylabel("Contrast (%)")
        ax_contrast.grid(True, linestyle="--", alpha=0.5)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_readouts, 1)))
        lines_snr = []
        lines_contrast = []
        for ir in range(n_readouts):
            (l_snr,) = ax_snr.plot([], [], "o-", color=colors[ir],
                                   label=f"{readout_times_ns[ir]} ns")
            (l_con,) = ax_contrast.plot([], [], "o-", color=colors[ir],
                                        label=f"{readout_times_ns[ir]} ns")
            lines_snr.append(l_snr)
            lines_contrast.append(l_con)
        ax_snr.legend(loc="best")
        ax_contrast.legend(loc="best")
        plt.tight_layout()
    else:
        fig = None

    # -------------------- Sweep --------------------
    tb.init_safe_stop()
    try:
        for run_ind in range(num_runs):
            if tb.safe_stop():
                break

            print(f"\nRun {run_ind + 1}/{num_runs}")

            # Optimize once per run
            if optimize_between_runs:
                try:
                    targeting.compensate_for_drift(nv_sig, no_crash=True)
                except Exception:
                    traceback.print_exc()

            # Restore MW state after drift compensation
            sig_gen.set_amp(uwave_power_dbm)
            sig_gen.set_freq(uwave_freq_ghz)
            sig_gen.uwave_on()

            counter_server.start_tag_stream()
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

                    if randomize_power_order:
                        order = rng.permutation(n_powers)
                    else:
                        order = np.arange(n_powers)

                    for step, ip in enumerate(order):
                        if tb.safe_stop():
                            break
                        ip = int(ip)
                        p_w = float(powers_W[ip])

                        if (
                            optimize_between_runs
                            and optimize_every_n_powers is not None
                            and step > 0
                            and (step % int(optimize_every_n_powers) == 0)
                        ):
                            try:
                                counter_server.stop_tag_stream()
                            except Exception:
                                pass
                            try:
                                targeting.compensate_for_drift(nv_sig, no_crash=True)
                            except Exception:
                                traceback.print_exc()
                            sig_gen.set_amp(uwave_power_dbm)
                            sig_gen.set_freq(uwave_freq_ghz)
                            sig_gen.uwave_on()
                            counter_server.start_tag_stream()
                            pulsegen_server.stream_load(SEQ_NAME, seq_args_string)

                        laser_server.set_power(p_w)

                        # Hold the green TTL HIGH during settling so the diode
                        # is actually emitting and `get_actual_power` returns a
                        # meaningful reading. Without this the laser is gated
                        # off between sequences and `pa?` reads ~0 W.
                        if green_do_chan is not None:
                            try:
                                pulsegen_server.constant([green_do_chan])
                            except Exception:
                                pass

                        # Settle-and-verify: wait for the diode to actually reach
                        # the setpoint before measuring. The COBO 520 takes time
                        # to ramp, especially when stepping over a large range.
                        deadline = time.time() + max(float(settle_time), 0.5)
                        actual_w = np.nan
                        tol = float(settle_tol_frac) * max(p_w, 1e-4)
                        while time.time() < deadline:
                            try:
                                actual_w = float(laser_server.get_actual_power())
                                if abs(actual_w - p_w) <= tol:
                                    break
                            except Exception:
                                pass
                            time.sleep(0.05)
                        try:
                            actual_w = float(laser_server.get_actual_power())
                        except Exception:
                            pass

                        # Reload the sequence — `pulsegen_server.constant` above
                        # replaced the loaded waveform with a constant output, so
                        # we need to re-arm the sequence before stream_start.
                        if green_do_chan is not None:
                            pulsegen_server.stream_load(SEQ_NAME, seq_args_string)

                        counter_server.clear_buffer()
                        pulsegen_server.stream_start(int(num_reps))

                        new_counts = counter_server.read_counter_summed(int(num_reps))

                        ref_counts_all[run_ind, ir, ip] = int(new_counts[0])
                        sig_counts_all[run_ind, ir, ip] = int(new_counts[1])
                        actual_powers_w_all[run_ind, ir, ip] = actual_w

                        actual_mW = (
                            actual_w * 1e3 if np.isfinite(actual_w) else float("nan")
                        )
                        print(
                            f"  readout={readout} ns, P_set={powers_mW[ip]:.3f} mW, "
                            f"P_act={actual_mW:.3f} mW | "
                            f"ref={int(new_counts[0])}, sig={int(new_counts[1])}"
                        )
            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            # After each run, recompute totals and update live plot
            valid_runs = run_ind + 1
            ref_run_totals = np.nansum(ref_counts_all[:valid_runs], axis=0)
            sig_run_totals = np.nansum(sig_counts_all[:valid_runs], axis=0)

            with np.errstate(divide="ignore", invalid="ignore"):
                contrasts_live = np.where(
                    ref_run_totals > 0,
                    (ref_run_totals - sig_run_totals) / ref_run_totals,
                    np.nan,
                )
                denom_live = np.sqrt(ref_run_totals + sig_run_totals)
                snr_live = np.where(
                    denom_live > 0,
                    (ref_run_totals - sig_run_totals) / denom_live,
                    np.nan,
                )

            for ir in range(n_readouts):
                ok = np.isfinite(snr_live[ir])
                if ok.any():
                    best_idx = int(np.nanargmax(snr_live[ir]))
                    print(
                        f"  Run {valid_runs} totals (readout={readout_times_ns[ir]} ns): "
                        f"best SNR={snr_live[ir, best_idx]:.4f} at "
                        f"P={powers_mW[best_idx]:.3f} mW"
                    )

            if fig is not None:
                for ir in range(n_readouts):
                    lines_snr[ir].set_data(powers_mW, snr_live[ir])
                    lines_contrast[ir].set_data(powers_mW, contrasts_live[ir] * 100)
                for ax in (ax_snr, ax_contrast):
                    ax.relim()
                    ax.autoscale_view()
                plt.pause(0.01)

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

    # -------------------- Final computation --------------------
    ref_totals = np.nansum(ref_counts_all, axis=0)
    sig_totals = np.nansum(sig_counts_all, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        contrasts = np.where(
            ref_totals > 0,
            (ref_totals - sig_totals) / ref_totals,
            np.nan,
        )
        denom = np.sqrt(ref_totals + sig_totals)
        snr_per_rep = np.where(denom > 0, (ref_totals - sig_totals) / denom, np.nan)

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
        "num_runs": num_runs,
        "ref_counts_all": ref_counts_all.tolist(),
        "sig_counts_all": sig_counts_all.tolist(),
        "actual_powers_w_all": actual_powers_w_all.tolist(),
        "ref_totals": ref_totals.tolist(),
        "sig_totals": sig_totals.tolist(),
        "contrasts": contrasts.tolist(),
        "snr_per_rep": snr_per_rep.tolist(),
        "optimal_powers_mW": optimal_powers_mW,
        "optimal_snr_per_rep": optimal_snr,
        "sequence": SEQ_NAME,
        "settle_time": settle_time,
        "settle_tol_frac": float(settle_tol_frac),
        "optimize_between_runs": bool(optimize_between_runs),
        "optimize_every_n_powers": (
            None if optimize_every_n_powers is None else int(optimize_every_n_powers)
        ),
        "randomize_power_order": bool(randomize_power_order),
    }
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    # -------------------- Summary table --------------------
    print("\n" + "=" * 78)
    print("GREEN POWER SWEEP SUMMARY (SNR-based)")
    print("=" * 78)
    for ir in range(n_readouts):
        print(f"\nReadout: {readout_times_ns[ir]} ns")
        print(
            f"{'power (mW)':>14} {'SNR':>10} {'contrast':>10} "
            f"{'ref total':>12} {'sig total':>12}"
        )
        for ip in range(n_powers):
            s = snr_per_rep[ir, ip]
            c = contrasts[ir, ip]
            rt = ref_totals[ir, ip]
            st = sig_totals[ir, ip]
            if np.isfinite(s):
                print(
                    f"{powers_mW[ip]:>14.3f} {s:>10.4f} {100 * c:>9.2f}% "
                    f"{rt:>12.0f} {st:>12.0f}"
                )
            else:
                print(
                    f"{powers_mW[ip]:>14.3f} {'--':>10} {'--':>10} {'--':>12} {'--':>12}"
                )
        if np.isfinite(optimal_powers_mW[ir]):
            print("-" * 78)
            print(
                f"Best power: {optimal_powers_mW[ir]:.3f} mW, "
                f"SNR = {optimal_snr[ir]:.4f}, "
                f"contrast = {100 * contrasts[ir, int(np.nanargmax(snr_per_rep[ir]))]:.2f}%"
            )
    print("=" * 78)

    if fig is not None:
        plt.show(block=False)
        plt.pause(0.5)

    print(f"\nSaved to {file_path}")
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
        num_runs=3,
        uwave_ind=0,
        laser_name="laser_COBO_520",
    )
