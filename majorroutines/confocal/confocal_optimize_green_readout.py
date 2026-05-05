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

    # Per-run storage — shape (num_runs, n) — allows proper averaging across runs
    ref_counts_all = np.full((int(num_runs), n), np.nan, dtype=float)
    sig_counts_all = np.full((int(num_runs), n), np.nan, dtype=float)

    # Final summary arrays — computed after all runs
    ref_totals  = np.full(n, np.nan, dtype=float)
    sig_totals  = np.full(n, np.nan, dtype=float)
    contrasts   = np.full(n, np.nan, dtype=float)
    snr_per_rep = np.full(n, np.nan, dtype=float)
    n_reps_used = np.zeros(n, dtype=int)

    # -------------------- Live figure --------------------
    if do_plot:
        fig, (ax_snr, ax_contrast) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        fig.suptitle("Green readout optimization")
        ax_snr.set_ylabel("SNR")
        ax_snr.grid(True, linestyle="--", alpha=0.5)
        ax_contrast.set_xlabel("Green spin-readout pulse duration (ns)")
        ax_contrast.set_ylabel("Contrast (%)")
        ax_contrast.grid(True, linestyle="--", alpha=0.5)
        (line_snr,) = ax_snr.plot([], [], "o-", color="darkorange", label="SNR")
        (line_contrast,) = ax_contrast.plot([], [], "o-", color="steelblue", label="Contrast")
        ax_snr.legend(loc="best")
        ax_contrast.legend(loc="best")
        plt.tight_layout()
        vline_snr = None
        vline_contrast = None
    else:
        fig = None
        ax_snr = None
        ax_contrast = None
        line_snr = None
        line_contrast = None
        vline_snr = None
        vline_contrast = None

    # -------------------- Sweep --------------------
    tb.init_safe_stop()
    try:
        for run_ind in range(int(num_runs)):
            if tb.safe_stop():
                break

            print(f"\nRun {run_ind + 1}/{num_runs}")

            # Optimize once per run — before sweeping all taus.
            # This ensures all taus in this run see the same NV condition.
            if optimize_between_runs:
                try:
                    targeting.compensate_for_drift(nv_sig, no_crash=True)
                except Exception:
                    traceback.print_exc()

            # Restore MW state after drift compensation (reset_cfm turns it off)
            sig_gen.set_amp(uwave_power_dbm)
            sig_gen.set_freq(uwave_freq_ghz)
            sig_gen.uwave_on()

            counter_server.start_tag_stream()
            try:
                for i, readout_ns in enumerate(readout_times_ns):
                    if tb.safe_stop():
                        break

                    readout_ns = int(readout_ns)

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

                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))

                    new_counts = counter_server.read_counter_summed(int(num_reps))

                    ref_counts_all[run_ind, i] = int(new_counts[0])
                    sig_counts_all[run_ind, i] = int(new_counts[1])

                    print(
                        f"  tau={int(readout_ns)} ns | "
                        f"ref={int(new_counts[0])}, sig={int(new_counts[1])}"
                    )

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            # After each complete run, recompute totals and update plot
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

            print(
                f"  Run {valid_runs} totals: "
                f"best SNR={np.nanmax(snr_live):.4f} at "
                f"{readout_times_ns[np.nanargmax(snr_live)]} ns"
            )

            if do_plot:
                ok_live = np.isfinite(snr_live)
                i_best_live = int(np.nanargmax(snr_live)) if ok_live.any() else None
                line_snr.set_data(readout_times_ns, snr_live)
                line_contrast.set_data(readout_times_ns, contrasts_live * 100)
                if i_best_live is not None:
                    best_ns = readout_times_ns[i_best_live]
                    if vline_snr is not None:
                        vline_snr.remove()
                    if vline_contrast is not None:
                        vline_contrast.remove()
                    vline_snr = ax_snr.axvline(
                        best_ns, color="gray", linestyle=":", linewidth=1,
                        label=f"Best: {best_ns} ns",
                    )
                    vline_contrast = ax_contrast.axvline(
                        best_ns, color="gray", linestyle=":", linewidth=1,
                    )
                    ax_snr.legend(loc="best", fontsize=9)
                for ax in (ax_snr, ax_contrast):
                    ax.relim()
                    ax.autoscale_view()
                plt.pause(0.01)

    finally:
        try:
            sig_gen.uwave_off()
        except Exception:
            pass
        vsg["pi_pulse"] = orig_pi_pulse
        tb.reset_cfm()

    # -------------------- Final computation --------------------
    ref_totals = np.nansum(ref_counts_all, axis=0)
    sig_totals = np.nansum(sig_counts_all, axis=0)
    n_reps_used = np.sum(np.isfinite(ref_counts_all), axis=0) * int(num_reps)

    with np.errstate(divide="ignore", invalid="ignore"):
        contrasts = np.where(
            ref_totals > 0,
            (ref_totals - sig_totals) / ref_totals,
            np.nan,
        )
        denom = np.sqrt(ref_totals + sig_totals)
        snr_per_rep = np.where(denom > 0, (ref_totals - sig_totals) / denom, np.nan)

    ok = np.isfinite(snr_per_rep)
    i_best = int(np.nanargmax(snr_per_rep)) if ok.any() else None

    # -------------------- Save --------------------
    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "readout_times_ns": readout_times_ns.tolist(),
        "ref_counts_all": ref_counts_all.tolist(),
        "sig_counts_all": sig_counts_all.tolist(),
        "ref_totals": ref_totals.tolist(),
        "sig_totals": sig_totals.tolist(),
        "contrasts": contrasts.tolist(),
        "snr": snr_per_rep.tolist(),
        "n_reps_used": n_reps_used.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_freq_ghz": uwave_freq_ghz,
        "uwave_power_dbm": uwave_power_dbm,
        "pol_ns": int(pol_ns),
        "sequence": SEQ_NAME,
        "optimize_between_runs": bool(optimize_between_runs),
    }

    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)
    print(f"\nSaved sweep to {file_path}")

    # -------------------- Summary table --------------------
    print("\n" + "=" * 78)
    print("GREEN READOUT SWEEP SUMMARY (SNR-based)")
    print("=" * 78)
    print(
        f"{'readout (ns)':>14} {'SNR':>10} {'contrast':>10} "
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
            f"SNR = {snr_per_rep[i_best]:.4f}, "
            f"contrast = {100 * contrasts[i_best]:.2f}%"
        )
    print("=" * 78)

    if do_plot and fig is not None:
        plt.show(block=False)
        plt.pause(0.5)

    return raw_data
