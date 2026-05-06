# -*- coding: utf-8 -*-
"""
Ti:sapph AOM delay calibration experiment.

Sweeps config["Optics"]["PhysicalLasers"]["laser_TISAPPH"]["delay"] from
delay_min_ns to delay_max_ns and measures normalized counts (sig/ref).

For each delay value the sequence file tisapph_delay_cal.py is reloaded with
the updated config — this is how the delay is swept without changing seq_args.

Expected result:
    - Large delay (>= real AOM delay): Ti:sapph light arrives during probe/readout
      window → NV fluorescence collected → sig counts HIGH → norm >> 1
    - Small delay (< real AOM delay): Ti:sapph light misses the readout gate
      → sig ≈ background → norm ≈ 0

The edge where norm rises = real Ti:sapph AOM hardware delay.
Update cryo.py with this value.

Created April 2026
@author: sbchand
"""

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
import utils.kplotlib as kpl
import utils.tool_belt as tb
from utils import common, data_manager as dm
from utils.constants import VirtualLaserKey

SEQ_FILE = "tisapph_delay_cal.py"


def main(
    nv_sig,
    delay_min_ns=0,
    delay_max_ns=1000,
    num_steps=21,
    num_reps=10000,
    num_runs=1,
    probe_ns=10000,
    optimize_between_runs=False,
    do_plot=True,
    do_save=True,
):
    """
    Sweep Ti:sapph AOM delay and find the real hardware delay.

    Parameters
    ----------
    nv_sig : NVSig
    delay_min_ns : int
        Start of delay sweep (ns). Default 0.
    delay_max_ns : int
        End of delay sweep (ns). Default 1000.
    num_steps : int
        Number of delay values. Default 21 → 50 ns steps.
    num_reps : int
        Repetitions per delay per run. Default 10,000.
    num_runs : int
        Number of runs to average. Default 1.
    probe_ns : int
        Ti:sapph probe pulse duration (ns). Should be long enough to
        ensure overlap when delay is correct. Default 10,000 ns.
    optimize_between_runs : bool
        Run drift compensation between runs. Default False.
    do_plot : bool
        Show live plot. Default True.
    do_save : bool
        Save raw data and figure. Default True.
    """
    kpl.init_kplotlib()
    tb.reset_cfm()

    pulsegen_server = tb.get_server_pulse_streamer()
    counter_server  = tb.get_server_counter()

    readout_vkey  = VirtualLaserKey.SPIN_READOUT
    spin_pol_vkey = VirtualLaserKey.SPIN_POL

    vld_read = tb.get_virtual_laser_dict(readout_vkey)
    vld_pol  = tb.get_virtual_laser_dict(spin_pol_vkey)

    readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, int(vld_read["duration"])))
    pol_ns     = int(nv_sig.pulse_durations.get(spin_pol_vkey, int(vld_pol["duration"])))
    probe_ns   = int(probe_ns)

    delay_ns_list = np.round(
        np.linspace(int(delay_min_ns), int(delay_max_ns), int(num_steps))
    ).astype(int)
    n = len(delay_ns_list)

    # Seq args — no uwave_ind since there is no MW in this sequence
    seq_args = [
        int(pol_ns),
        int(probe_ns),
        int(readout_ns),
        spin_pol_vkey,
        readout_vkey,
    ]

    # Storage
    ref_counts_all = np.full((int(num_runs), n), np.nan, dtype=float)
    sig_counts_all = np.full((int(num_runs), n), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()

    # -------------------- Live figure --------------------
    if do_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Ti:sapph AOM delay in config (ns)")
        ax.set_ylabel("Normalized counts (sig / ref)")
        ax.set_title("Ti:sapph AOM delay calibration")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="ref level")
        ax.grid(True, linestyle="--", alpha=0.5)
        (line,) = ax.plot([], [], "o-", color="steelblue", label="sig/ref")
        ax.legend(loc="best")
        plt.tight_layout()
    else:
        fig = None
        ax = None
        line = None

    # -------------------- Sweep --------------------
    tb.init_safe_stop()
    try:
        for run_ind in range(int(num_runs)):
            if tb.safe_stop():
                break

            print(f"\nRun {run_ind + 1}/{num_runs}")

            if optimize_between_runs:
                try:
                    targeting.compensate_for_drift(nv_sig, no_crash=True)
                except Exception:
                    import traceback
                    traceback.print_exc()

            counter_server.start_tag_stream()
            try:
                for i, delay_ns in enumerate(delay_ns_list):
                    if tb.safe_stop():
                        break

                    # Update the Ti:sapph delay in the config and reload sequence.
                    # This is how the delay is swept — the sequence reads it from config.
                    config = common.get_config_dict()
                    config["Optics"]["PhysicalLasers"]["laser_TISAPPH"]["delay"] = int(delay_ns)

                    seq_args_string = tb.encode_seq_args(seq_args)
                    pulsegen_server.stream_load(SEQ_FILE, seq_args_string)

                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))

                    new_counts = counter_server.read_counter_summed(int(num_reps))

                    ref_counts_all[run_ind, i] = int(new_counts[0])
                    sig_counts_all[run_ind, i] = int(new_counts[1])

                    ref_val = ref_counts_all[run_ind, i]
                    sig_val = sig_counts_all[run_ind, i]
                    norm_val = sig_val / ref_val if ref_val > 0 else float("nan")

                    print(
                        f"  delay={int(delay_ns):>5d} ns | "
                        f"ref={int(ref_val):>6d}, sig={int(sig_val):>6d}, "
                        f"norm={norm_val:.4f}"
                    )

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            # Update plot after each run
            if do_plot:
                ref_totals = np.nansum(ref_counts_all[: run_ind + 1], axis=0)
                sig_totals = np.nansum(sig_counts_all[: run_ind + 1], axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    norm_mean = np.where(ref_totals > 0, sig_totals / ref_totals, np.nan)
                line.set_data(delay_ns_list, norm_mean)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)

    finally:
        try:
            tb.reset_cfm()
        except Exception:
            pass

    # -------------------- Final results --------------------
    ref_totals = np.nansum(ref_counts_all, axis=0)
    sig_totals = np.nansum(sig_counts_all, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_mean = np.where(ref_totals > 0, sig_totals / ref_totals, np.nan)

    # Find the cutoff — last delay where norm is still above 50% of max.
    # Small delay: Ti:sapph overlaps with probe window → counts high.
    # Large delay: Ti:sapph fires too early, light misses probe window → counts low.
    # The falling edge = real Ti:sapph AOM hardware delay.
    ok = np.isfinite(norm_mean)
    edge_ns = None
    if ok.any():
        norm_max = np.nanmax(norm_mean)
        threshold = 0.5 * norm_max
        above = norm_mean >= threshold
        if np.any(above):
            # Last True index = falling edge
            edge_ns = int(delay_ns_list[int(np.where(above)[0][-1])])

    # -------------------- Summary --------------------
    print("\n" + "=" * 62)
    print("Ti:sapph AOM DELAY CALIBRATION SUMMARY")
    print("=" * 62)
    print(f"{'delay (ns)':>12}  {'ref':>8}  {'sig':>8}  {'norm':>8}")
    for d, r, s, nv in zip(delay_ns_list, ref_totals, sig_totals, norm_mean):
        flag = " ← edge" if edge_ns is not None and int(d) == edge_ns else ""
        nv_str = f"{nv:.4f}" if np.isfinite(nv) else "  --"
        print(f"{int(d):>12d}  {int(r):>8d}  {int(s):>8d}  {nv_str:>8}{flag}")
    print("=" * 62)
    if edge_ns is not None:
        print(f"\nEstimated Ti:sapph AOM hardware delay: {edge_ns} ns")
        print(f"(falling edge of sig/ref curve — set this as config_delay in cryo.py)")
        print(f"Update cryo.py:")
        print(
            f'  config["Optics"]["PhysicalLasers"]'
            f'["laser_TISAPPH"]["delay"] = {edge_ns}'
        )
    print("=" * 62)

    # -------------------- Save --------------------
    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "delay_ns_list": delay_ns_list.tolist(),
        "ref_counts_all": ref_counts_all.tolist(),
        "sig_counts_all": sig_counts_all.tolist(),
        "ref_totals": ref_totals.tolist(),
        "sig_totals": sig_totals.tolist(),
        "norm_mean": norm_mean.tolist(),
        "estimated_delay_ns": edge_ns,
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "pol_ns": int(pol_ns),
        "probe_ns": int(probe_ns),
        "readout_ns": int(readout_ns),
    }

    if do_save:
        ts = dm.get_time_stamp()
        file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
        dm.save_raw_data(raw_data, file_path)
        if fig is not None:
            dm.save_figure(fig, file_path)
        print(f"Saved to {file_path}")

    if do_plot and fig is not None:
        if edge_ns is not None:
            ax.axvline(
                edge_ns, color="red", linestyle=":",
                linewidth=1.5, label=f"Cutoff (real delay): {edge_ns} ns"
            )
            ax.legend(loc="best")
        plt.show(block=False)
        plt.pause(0.5)

    return raw_data
