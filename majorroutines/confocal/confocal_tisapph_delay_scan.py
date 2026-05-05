# -*- coding: utf-8 -*-
"""Sweep Ti:sapph TTL delay relative to the readout pulse and find the
overlap location at the sample plane.

Two interleaved gates per repetition (built into the sequence file):
    gate 0 = reference (Ti:sapph OFF)
    gate 1 = signal    (Ti:sapph pulse at swept delay)

The Ti:sapph TTL is placed in the sequence WITHOUT delay compensation; the
green readout train IS delay-compensated as usual. The peak of (sig - ref)
in delta_ns therefore directly gives the true Ti:sapph hardware delay.

@author: sarojchand
@author: chemistatcode
"""

import matplotlib.pyplot as plt
import numpy as np

from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey


def _bump_centroid(d_arr, c_arr, baseline):
    """Centroid of (counts - baseline) over points above baseline.

    Returns (delay_at_centroid, counts_at_centroid_interpolated) or None.
    """
    finite = np.isfinite(c_arr)
    d = np.asarray(d_arr, dtype=float)[finite]
    c = np.asarray(c_arr, dtype=float)[finite]
    if d.size < 3:
        return None
    excess = c - baseline
    excess = np.where(excess > 0, excess, 0.0)
    if excess.sum() <= 0:
        return None
    centroid = float(np.sum(d * excess) / np.sum(excess))
    # Interpolate counts at centroid for a clean marker
    counts_at = float(np.interp(centroid, d, c))
    return centroid, counts_at


def main(
    nv_sig,
    num_reps=int(2e4),
    num_runs=3,
    delta_min_ns=-2000,
    delta_max_ns=2000,
    num_steps=81,
    pol_ns=None,
    readout_ns=None,
    tisapph_pulse_ns=None,
    spin_pol_vkey=VirtualLaserKey.SPIN_POL,
    readout_vkey=VirtualLaserKey.SPIN_READOUT,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    # ------------------------------------------------------------------
    # Resolve durations
    # ------------------------------------------------------------------
    if pol_ns is None:
        pol_ns = int(tb.get_virtual_laser_dict(spin_pol_vkey)["duration"])
    if readout_ns is None:
        readout_ns = int(tb.get_virtual_laser_dict(readout_vkey)["duration"])
    if tisapph_pulse_ns is None:
        # Default: 2x readout so the bump has a flat top -> easier centroiding
        tisapph_pulse_ns = int(2 * readout_ns)
    pol_ns = int(pol_ns)
    readout_ns = int(readout_ns)
    tisapph_pulse_ns = int(tisapph_pulse_ns)

    if delta_max_ns < delta_min_ns:
        raise ValueError(
            f"delta_max_ns ({delta_max_ns}) < delta_min_ns ({delta_min_ns})"
        )

    deltas_ns = np.linspace(delta_min_ns, delta_max_ns, num_steps)
    deltas_ns = np.rint(deltas_ns).astype(int)
    # Snap to 8 ns granularity (Pulse Streamer)
    deltas_ns = (deltas_ns // 8) * 8

    sig_counts = np.full((num_runs, len(deltas_ns)), np.nan, dtype=float)
    ref_counts = np.full((num_runs, len(deltas_ns)), np.nan, dtype=float)

    # ------------------------------------------------------------------
    # Live plot: raw sig/ref on top, ratio on bottom
    # ------------------------------------------------------------------
    fig, (ax_raw, ax_ratio) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax_raw.set_ylabel("Counts / rep")
    ax_raw.set_title(
        f"Ti:sapph delay calibration "
        f"(pol={pol_ns} ns, TS pulse={tisapph_pulse_ns} ns, "
        f"readout={readout_ns} ns)"
    )
    (line_ref,) = ax_raw.plot(
        [], [], marker="o", linestyle="-", label="reference (TS off)"
    )
    (line_sig,) = ax_raw.plot(
        [], [], marker="o", linestyle="-", label="signal (TS on)"
    )
    ax_raw.legend(loc="best")

    ax_ratio.set_xlabel("delta_ns  (commanded TS TTL - readout TTL, ns)")
    ax_ratio.set_ylabel("sig / ref")
    (line_ratio,) = ax_ratio.plot(
        [], [], marker="o", linestyle="-", color="C2"
    )
    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    seq_file = "tisapph_delay_calib.py"
    tb.init_safe_stop()

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    for run_ind in range(num_runs):
        print(f"Run {run_ind + 1}/{num_runs}")
        if tb.safe_stop():
            break

        for ind, delta_ns in enumerate(deltas_ns):
            if tb.safe_stop():
                break

            seq_args = [
                int(pol_ns),
                int(tisapph_pulse_ns),
                int(readout_ns),
                int(delta_ns),
                spin_pol_vkey.name,
                readout_vkey.name,
            ]
            seq_args_string = tb.encode_seq_args(seq_args)

            counter_server.start_tag_stream()
            try:
                counter_server.clear_buffer()
                pulsegen_server.stream_immediate(
                    seq_file, seq_args_string, int(num_reps)
                )
                # 2 gates per rep (ref, sig)
                new_counts = counter_server.read_counter_modulo_gates(
                    2, int(num_reps)
                )
            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            new_counts = np.asarray(new_counts, dtype=float)
            if new_counts.ndim == 2:
                ref_total = float(new_counts[:, 0].sum())
                sig_total = float(new_counts[:, 1].sum())
            else:
                ref_total = float(new_counts[0])
                sig_total = float(new_counts[1])

            ref_counts[run_ind, ind] = ref_total / num_reps
            sig_counts[run_ind, ind] = sig_total / num_reps

            print(
                f"  delta={int(delta_ns):>+6d} ns | "
                f"ref={ref_counts[run_ind, ind]:.3f} | "
                f"sig={sig_counts[run_ind, ind]:.3f}"
            )

        # update plot after each run
        avg_ref = np.nanmean(ref_counts[: run_ind + 1], axis=0)
        avg_sig = np.nanmean(sig_counts[: run_ind + 1], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_ratio = np.where(avg_ref > 0, avg_sig / avg_ref, np.nan)

        line_ref.set_data(deltas_ns, avg_ref)
        line_sig.set_data(deltas_ns, avg_sig)
        line_ratio.set_data(deltas_ns, avg_ratio)
        for ax in (ax_raw, ax_ratio):
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.01)

    # ------------------------------------------------------------------
    # Final averaged arrays
    # ------------------------------------------------------------------
    avg_ref = np.nanmean(ref_counts, axis=0)
    avg_sig = np.nanmean(sig_counts, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_ratio = np.where(avg_ref > 0, avg_sig / avg_ref, np.nan)

    # ------------------------------------------------------------------
    # Find the bump: centroid of (ratio - baseline) above baseline
    # ------------------------------------------------------------------
    finite_mask = np.isfinite(avg_ratio)
    best_delta_ns = None
    best_ratio = None
    selection_mode = "no_bump_found"

    if finite_mask.sum() >= 3:
        baseline = float(np.nanmedian(avg_ratio))
        result = _bump_centroid(deltas_ns, avg_ratio, baseline)
        if result is not None:
            best_delta_ns, best_ratio = result
            selection_mode = "bump_centroid"
        else:
            # Fallback: max of ratio
            best_ind = int(np.nanargmax(avg_ratio))
            best_delta_ns = float(deltas_ns[best_ind])
            best_ratio = float(avg_ratio[best_ind])
            selection_mode = "max_ratio_fallback"

    if best_delta_ns is not None:
        print(f"\nBest delta_ns = {best_delta_ns:.1f} ns ({selection_mode})")
        print(f"Ratio at best = {best_ratio:.4f}")
        print(
            f"-> Set config Optics/PhysicalLasers/laser_TISAPPH/delay "
            f"to {-best_delta_ns:.0f} ns"
        )
        print(
            "   (sign convention: TS TTL fires `delay` ns before readout TTL "
            "to make the optical pulses coincide at the sample)"
        )
    else:
        print("No valid bump location found.")

    # ------------------------------------------------------------------
    # Mark the optimum on the plots
    # ------------------------------------------------------------------
    if best_delta_ns is not None:
        ax_ratio.plot(
            [best_delta_ns], [best_ratio],
            marker="o", markersize=10, color="red", zorder=5,
            label=f"centroid: {best_delta_ns:.0f} ns",
        )
        for ax in (ax_raw, ax_ratio):
            ax.axvline(best_delta_ns, color="red", linestyle=":", linewidth=1)
        ax_ratio.legend(loc="best")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    raw_data = {
        "timestamp": dm.get_time_stamp(),
        "nv_sig": nv_sig,
        "deltas_ns": deltas_ns.tolist(),
        "ref_counts_per_run": ref_counts.tolist(),
        "sig_counts_per_run": sig_counts.tolist(),
        "avg_ref": avg_ref.tolist(),
        "avg_sig": avg_sig.tolist(),
        "avg_ratio": avg_ratio.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "pol_ns": int(pol_ns),
        "readout_ns": int(readout_ns),
        "tisapph_pulse_ns": int(tisapph_pulse_ns),
        "spin_pol_vkey": spin_pol_vkey.name,
        "readout_vkey": readout_vkey.name,
        "best_delta_ns": (None if best_delta_ns is None else float(best_delta_ns)),
        "best_ratio": (None if best_ratio is None else float(best_ratio)),
        "selection_mode": selection_mode,
    }
    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(fig, file_path)
    print(f"Saved to {file_path}")

    plt.draw()
    tb.reset_cfm()

    return {
        "deltas_ns": deltas_ns,
        "ref_counts": ref_counts,
        "sig_counts": sig_counts,
        "avg_ref": avg_ref,
        "avg_sig": avg_sig,
        "avg_ratio": avg_ratio,
        "best_delta_ns": best_delta_ns,
        "best_ratio": best_ratio,
        "selection_mode": selection_mode,
        "pol_ns": pol_ns,
        "readout_ns": readout_ns,
        "tisapph_pulse_ns": tisapph_pulse_ns,
    }


if __name__ == "__main__":
    pass
