# -*- coding: utf-8 -*-
"""
Sweep the dark gap between Ti:sapph turn-OFF and the readout window opening.
No microwaves are used. Two gates per rep:

    gate 0 = reference: Ti:sapph OFF (green readout only)
    gate 1 = signal:    Ti:sapph ON for tisapph_ns, then dark gap delay_ns, then readout

Per step:
    sig_per_rep = gate1_total / num_reps
    ref_per_rep = gate0_total / num_reps
    norm        = sig_per_rep / ref_per_rep      (1 means TiSapph had no effect)
    contrast    = sig_per_rep - ref_per_rep      (extra counts attributable to TiSapph)

Returns:
    raw_data, proc_data
"""

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey
import majorroutines.targeting as targeting


SEQ_NAME = "tisapph_readout_delay.py"


def _build_delay_list(delay_ns_list=None, min_delay_ns=None, max_delay_ns=None,
                     num_steps=None, log_spaced=False):
    if delay_ns_list is not None:
        arr = np.asarray(delay_ns_list, dtype=int).ravel()
    else:
        if min_delay_ns is None or max_delay_ns is None or num_steps is None:
            raise ValueError(
                "Provide delay_ns_list OR (min_delay_ns, max_delay_ns, num_steps)."
            )
        if log_spaced:
            lo = max(int(min_delay_ns), 1)
            arr = np.logspace(np.log10(lo), np.log10(int(max_delay_ns)),
                              int(num_steps))
        else:
            arr = np.linspace(int(min_delay_ns), int(max_delay_ns), int(num_steps))
        arr = np.rint(arr).astype(int)
    arr = np.unique(arr)
    if arr.size == 0:
        raise ValueError("delay list is empty.")
    if np.any(arr < 0):
        raise ValueError("All delays must be >= 0 ns.")
    return arr


def main(
    nv_sig,
    tisapph_ns,
    num_reps,
    num_runs,
    delay_ns_list=None,
    min_delay_ns=None,
    max_delay_ns=None,
    num_steps=None,
    log_spaced=False,
    readout_ns=None,
    pol_ns=None,
    optimize_between_runs=True,
    do_plot=True,
    do_save=True,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server  = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    # ---- resolve pulse durations from nv_sig / config ----
    readout_vkey = VirtualLaserKey.SPIN_READOUT
    spin_pol_vkey = VirtualLaserKey.SPIN_POL

    if readout_ns is None:
        vld_r = tb.get_virtual_laser_dict(readout_vkey)
        readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, int(vld_r["duration"])))
    if pol_ns is None:
        vld_p = tb.get_virtual_laser_dict(spin_pol_vkey)
        pol_ns = int(nv_sig.pulse_durations.get(spin_pol_vkey, int(vld_p["duration"])))
    readout_ns = int(readout_ns)
    pol_ns     = int(pol_ns)
    tisapph_ns = int(tisapph_ns)

    # ---- delay list ----
    delay_list = _build_delay_list(
        delay_ns_list=delay_ns_list,
        min_delay_ns=min_delay_ns,
        max_delay_ns=max_delay_ns,
        num_steps=num_steps,
        log_spaced=log_spaced,
    )
    num_steps_eff = len(delay_list)

    # ---- result buffers ----
    ref_counts = np.full((num_runs, num_steps_eff), np.nan)
    sig_counts = np.full((num_runs, num_steps_eff), np.nan)

    step_wall = np.full((num_runs, num_steps_eff), np.nan)
    run_wall  = np.full(num_runs, np.nan)

    # ---- live plot ----
    if do_plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].set_ylabel("Cumulative counts per run (avg over runs)")
        axes[1].set_ylabel("Signal / Reference")
        axes[1].set_xlabel("Dark gap delay (ns)")
        axes[0].set_title(
            f"Ti:sapph -> dark gap -> readout  tisapph_ns={tisapph_ns}"
        )
        (l_ref,) = axes[0].plot([], [], "o-", label="Reference (TiSapph OFF)")
        (l_sig,) = axes[0].plot([], [], "o-", label="Signal (TiSapph ON)")
        (l_norm,) = axes[1].plot([], [], "o-", color="black")
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()
        if log_spaced:
            axes[0].set_xscale("log"); axes[1].set_xscale("log")
    else:
        fig = None

    print(f"seq         = {SEQ_NAME}")
    print(f"pol_ns      = {pol_ns}")
    print(f"tisapph_ns  = {tisapph_ns}")
    print(f"readout_ns  = {readout_ns}")
    print(f"delays (ns) = {delay_list.tolist()}")
    print(f"reps / runs = {num_reps} / {num_runs}")

    timestamp = dm.get_time_stamp()
    tb.init_safe_stop()
    start_t = time.time()

    try:
        for run_ind in range(num_runs):
            print(f"\nRun {run_ind + 1}/{num_runs}")
            if tb.safe_stop():
                break

            if optimize_between_runs:
                try:
                    targeting.compensate_for_drift(nv_sig, no_crash=True)
                except Exception:
                    traceback.print_exc()

            counter_server.start_tag_stream()
            t_run = time.perf_counter()
            try:
                for step_ind, delay_ns in enumerate(delay_list):
                    if tb.safe_stop():
                        break

                    seq_args = [
                        int(pol_ns),
                        int(tisapph_ns),
                        int(delay_ns),
                        int(readout_ns),
                        spin_pol_vkey,
                        readout_vkey,
                    ]
                    seq_args_string = tb.encode_seq_args(seq_args)

                    t_step = time.perf_counter()
                    pulsegen_server.stream_load(SEQ_NAME, seq_args_string)
                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))

                    # 2-gate summed counts: [ref_total, sig_total]
                    counts = counter_server.read_counter_summed(int(num_reps))
                    step_wall[run_ind, step_ind] = time.perf_counter() - t_step

                    ref_counts[run_ind, step_ind] = int(counts[0])
                    sig_counts[run_ind, step_ind] = int(counts[1])

                    ref_total = int(counts[0])
                    sig_total = int(counts[1])
                    norm = (sig_total / ref_total) if ref_total > 0 else float("nan")
                    print(
                        f"  delay={int(delay_ns):>7d} ns | "
                        f"ref={ref_total}, sig={sig_total}, "
                        f"sig/ref={norm:.4f}  | {step_wall[run_ind, step_ind]:.2f} s"
                    )
            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            run_wall[run_ind] = time.perf_counter() - t_run

            # ---- update live plot once per run ----
            if do_plot:
                # For each delay step, average the per-run totals across all
                # runs completed so far. Each y-value is the mean of
                # "cumulative counts over all reps in one run" — units of
                # counts per run, not counts per rep, not summed across runs.
                ref_mean_per_run = np.nanmean(ref_counts[: run_ind + 1], axis=0)
                sig_mean_per_run = np.nanmean(sig_counts[: run_ind + 1], axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    norm_mean = np.where(
                        ref_mean_per_run > 0,
                        sig_mean_per_run / ref_mean_per_run,
                        np.nan,
                    )
                l_ref.set_data(delay_list, ref_mean_per_run)
                l_sig.set_data(delay_list, sig_mean_per_run)
                l_norm.set_data(delay_list, norm_mean)
                for ax in axes:
                    ax.relim(); ax.autoscale_view()
                plt.pause(0.01)
    finally:
        tb.reset_cfm()

    elapsed = time.time() - start_t

    # ---- per-rep normalization for return values ----
    # Cumulative totals across all runs (one number per delay).
    ref_cum_total = np.nansum(ref_counts, axis=0)
    sig_cum_total = np.nansum(sig_counts, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_cum = np.where(ref_cum_total > 0,
                            sig_cum_total / ref_cum_total, np.nan)
    contrast_cum = sig_cum_total - ref_cum_total

    # Per-rep means for downstream analyses that prefer rate units.
    nr_total = float(num_reps) * float(num_runs)
    ref_per_rep_mean = ref_cum_total / nr_total
    sig_per_rep_mean = sig_cum_total / nr_total

    raw_data = {
        "timestamp": timestamp,
        "elapsed_s": float(elapsed),
        "nv_sig": nv_sig,
        "sequence": SEQ_NAME,
        "delay_list_ns": delay_list.tolist(),
        "tisapph_ns": int(tisapph_ns),
        "pol_ns": int(pol_ns),
        "readout_ns": int(readout_ns),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "ref_counts": ref_counts.tolist(),
        "sig_counts": sig_counts.tolist(),
        "step_wall_s": step_wall.tolist(),
        "run_wall_s":  run_wall.tolist(),
    }

    proc_data = {
        "delay_list_ns": delay_list.tolist(),
        "ref_cumulative": ref_cum_total.tolist(),
        "sig_cumulative": sig_cum_total.tolist(),
        "norm_mean": norm_cum.tolist(),                 # sig/ref (totals -- equals per-rep ratio)
        "contrast_cumulative": contrast_cum.tolist(),   # sig_total - ref_total
        "ref_per_rep_mean": ref_per_rep_mean.tolist(),
        "sig_per_rep_mean": sig_per_rep_mean.tolist(),
    }

    if do_save:
        fp = dm.get_file_path(__file__, dm.get_time_stamp(),
                              getattr(nv_sig, "name", "nv"))
        dm.save_raw_data(raw_data, fp)
        if fig is not None:
            dm.save_figure(fig, fp)
        print(f"Saved data to {fp}")

    return raw_data, proc_data


if __name__ == "__main__":
    pass
