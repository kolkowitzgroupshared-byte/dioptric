# -*- coding: utf-8 -*-
"""Sweep APD gate delay after the readout laser and find where scaled
counts match nv_sig.expected_counts.

@author: sarojchand
@author: chemistatcode
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey


def _gaussian_plus_offset(x, amp, mu, sigma, offset):
    return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)) + offset


def _gaussian_crossing(popt, target):
    """Delay where the Gaussian descends to `target`, or None."""
    amp, mu, sigma, offset = popt
    if amp <= 0 or sigma <= 0:
        return None
    frac = (target - offset) / amp
    if not (0.0 < frac <= 1.0):
        return None
    return float(mu + sigma * np.sqrt(-2.0 * np.log(frac)))


def _linear_crossing(d_arr, c_arr, target):
    """First downward crossing of `target`, linearly interpolated."""
    finite = np.isfinite(c_arr)
    d = np.asarray(d_arr, dtype=float)[finite]
    c = np.asarray(c_arr, dtype=float)[finite]
    if d.size < 1:
        return None
    if c[0] <= target:
        return float(d[0]), float(c[0])
    for i in range(1, d.size):
        if c[i] <= target:
            y0, y1 = c[i - 1], c[i]
            x0, x1 = d[i - 1], d[i]
            if y0 == y1:
                return float(x0), float(target)
            frac = (y0 - target) / (y0 - y1)
            return float(x0 + frac * (x1 - x0)), float(target)
    return None


def main(
    nv_sig,
    num_reps=int(2e5),
    num_runs=3,
    delay_min_ns=0,
    delay_max_ns=1000,
    num_steps=41,
    laser_on_ns=None,
    gate_width_ns=300,
    laser_vkey=VirtualLaserKey.SPIN_READOUT,
    laser_fall_delay_ns=None,
    apd_gate_delay_ns=0,
    tolerance=0.10,
    expected_readout_vkey=VirtualLaserKey.IMAGING,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    if laser_on_ns is None:
        vld = tb.get_virtual_laser_dict(laser_vkey)
        laser_on_ns = int(nv_sig.pulse_durations.get(laser_vkey, int(vld["duration"])))
    laser_on_ns = int(laser_on_ns)

    if laser_fall_delay_ns is None:
        laser_name = tb.get_physical_laser_name(laser_vkey)
        laser_fall_delay_ns = int(tb.get_physical_laser_dict(laser_name)["delay"])
    laser_fall_delay_ns = int(laser_fall_delay_ns)
    apd_gate_delay_ns = int(apd_gate_delay_ns)

    # Safety: negative delays risk overlapping the APD gate with the laser.
    if delay_min_ns < 0:
        print(f"WARNING: clamping delay_min_ns={delay_min_ns} to 0 to protect APD.")
        delay_min_ns = 0
    if delay_max_ns < delay_min_ns:
        raise ValueError(
            f"delay_max_ns ({delay_max_ns}) < delay_min_ns ({delay_min_ns})"
        )

    delays_ns = np.linspace(delay_min_ns, delay_max_ns, num_steps)
    delays_ns = np.rint(delays_ns).astype(int)

    # Scale observed totals to match expected_counts, which is referenced
    # to the IMAGING readout duration (not num_reps * gate_width).
    expected = getattr(nv_sig, "expected_counts", None)
    tolerance = float(tolerance)
    imaging_readout_ns = int(
        tb.get_virtual_laser_dict(expected_readout_vkey)["duration"]
    )
    scan_total_gate_ns = float(num_reps) * float(gate_width_ns)
    scale_to_imaging = float(imaging_readout_ns) / scan_total_gate_ns
    if expected is not None:
        expected = float(expected)

    counts = np.full((num_runs, len(delays_ns)), np.nan, dtype=float)

    fig, ax = plt.subplots()
    ax.set_xlabel("Delay from laser-off command to APD gate (ns)")
    ax.set_ylabel(
        f"Counts (scaled to {expected_readout_vkey.name} = "
        f"{imaging_readout_ns * 1e-6:.3f} ms)"
    )
    ax.set_title("APD gate delay scan")
    (line_avg,) = ax.plot([], [], marker="o", linestyle="None", label="Counts")
    if expected is not None:
        ax.axhline(
            expected, color="gray", linestyle="--", linewidth=1,
            label=f"expected = {expected:g}",
        )
        ax.axhspan(
            expected * (1 - tolerance), expected * (1 + tolerance),
            color="gray", alpha=0.15,
            label=f"+/-{tolerance * 100:.0f}% band",
        )
    ax.legend(loc="best")

    seq_file = "confocal_apd_gate_overlap_scan.py"
    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print(f"Run {run_ind + 1}/{num_runs}")
        if tb.safe_stop():
            break

        for ind, delay_ns in enumerate(delays_ns):
            if tb.safe_stop():
                break

            seq_args = [
                int(laser_on_ns),
                int(gate_width_ns),
                int(delay_ns),
                laser_vkey.name,
                int(laser_fall_delay_ns),
                int(apd_gate_delay_ns),
            ]
            seq_args_string = tb.encode_seq_args(seq_args)

            counter_server.start_tag_stream()
            try:
                counter_server.clear_buffer()
                pulsegen_server.stream_immediate(
                    seq_file, seq_args_string, int(num_reps)
                )
                new_counts = counter_server.read_counter_simple(int(num_reps))
            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            total_counts = int(np.sum(new_counts)) if len(new_counts) > 0 else 0
            scaled_counts = total_counts * scale_to_imaging
            counts[run_ind, ind] = scaled_counts
            print(f"  delay={int(delay_ns):>5d} ns | counts={scaled_counts:.2f}")

        avg_counts_so_far = np.nanmean(counts[: run_ind + 1], axis=0)
        line_avg.set_data(delays_ns, avg_counts_so_far)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

    avg_counts = np.nanmean(counts, axis=0)

    # Gaussian fit on a flat baseline. Bounds allow mu before delays_ns[0]
    # so the visible "right half" of a partial Gaussian fits cleanly.
    gaussian_popt = None
    gaussian_perr = None
    finite_mask = np.isfinite(avg_counts)
    if finite_mask.sum() >= 4:
        d_fit = delays_ns[finite_mask].astype(float)
        c_fit = avg_counts[finite_mask].astype(float)
        span = float(d_fit[-1] - d_fit[0]) if d_fit.size > 1 else 1000.0
        offset_seed = float(np.nanmin(c_fit))
        amp_seed = max(float(np.nanmax(c_fit) - offset_seed), 1.0)
        mu_seed = float(d_fit[int(np.nanargmax(c_fit))])
        sigma_seed = max(span / 4.0, 1.0)
        p0 = [amp_seed, mu_seed, sigma_seed, offset_seed]
        bounds = (
            [0.0, d_fit[0] - span, 1.0, 0.0],
            [10.0 * amp_seed, d_fit[-1] + span, 10.0 * sigma_seed,
             max(float(np.nanmax(c_fit)), 1.0)],
        )
        try:
            gaussian_popt, pcov = curve_fit(
                _gaussian_plus_offset, d_fit, c_fit,
                p0=p0, bounds=bounds, maxfev=50000,
            )
            gaussian_perr = np.sqrt(np.diag(pcov)).tolist()
            gaussian_popt = gaussian_popt.tolist()
            amp_f, mu_f, sigma_f, offset_f = gaussian_popt
            print(
                f"Gaussian fit: amp={amp_f:.4g}, mu={mu_f:.1f} ns, "
                f"sigma={sigma_f:.1f} ns, offset={offset_f:.4g}"
            )
        except Exception as e:
            print(f"Gaussian fit failed: {e}")
            gaussian_popt = None

    # Optimum: Gaussian crossing -> linear crossing -> min-counts fallback.
    if expected is not None:
        crossing_from_fit = (
            _gaussian_crossing(gaussian_popt, expected)
            if gaussian_popt is not None else None
        )
        span_sel = float(delays_ns[-1] - delays_ns[0])
        in_range = (
            crossing_from_fit is not None
            and delays_ns[0] - span_sel <= crossing_from_fit <= delays_ns[-1] + span_sel
        )
        if in_range:
            best_delay_ns = float(crossing_from_fit)
            best_counts = float(expected)
            selection_mode = "expected_counts_gaussian"
        else:
            crossing = _linear_crossing(delays_ns, avg_counts, expected)
            if crossing is not None:
                best_delay_ns, best_counts = crossing
                selection_mode = "expected_counts_linear_fallback"
            else:
                print(f"WARNING: counts never reached expected={expected:g}; using min.")
                best_ind = int(np.nanargmin(avg_counts))
                best_delay_ns = float(delays_ns[best_ind])
                best_counts = float(avg_counts[best_ind])
                selection_mode = "min_counts_fallback"
    else:
        best_ind = int(np.nanargmax(avg_counts))
        best_delay_ns = float(delays_ns[best_ind])
        best_counts = float(avg_counts[best_ind])
        selection_mode = "max_counts_fallback"

    print(f"Optimal delay = {best_delay_ns:.1f} ns")
    print(f"Counts at optimum = {best_counts:.4g}")
    if expected is not None:
        print(f"Expected counts = {expected:g}")

    if gaussian_popt is not None:
        t_smooth = np.linspace(delays_ns[0], delays_ns[-1], 400)
        ax.plot(
            t_smooth, _gaussian_plus_offset(t_smooth, *gaussian_popt),
            linestyle="--", color="orange", linewidth=1.5,
            label="Gaussian fit",
        )
    ax.plot(
        [best_delay_ns], [best_counts],
        marker="o", markersize=10, color="red", zorder=5,
        label=f"optimum: {best_delay_ns:.1f} ns, counts={best_counts:.4g}",
    )
    ax.axvline(best_delay_ns, color="red", linestyle=":", linewidth=1)
    ax.legend(loc="best")

    raw_data = {
        "timestamp": dm.get_time_stamp(),
        "nv_sig": nv_sig,
        "delays_ns": delays_ns.tolist(),
        "counts_per_run": counts.tolist(),
        "avg_counts": avg_counts.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "laser_on_ns": int(laser_on_ns),
        "gate_width_ns": int(gate_width_ns),
        "laser_fall_delay_ns": int(laser_fall_delay_ns),
        "apd_gate_delay_ns": int(apd_gate_delay_ns),
        "expected_counts": (None if expected is None else float(expected)),
        "expected_readout_vkey": expected_readout_vkey.name,
        "imaging_readout_ns": int(imaging_readout_ns),
        "scale_to_imaging": float(scale_to_imaging),
        "tolerance": float(tolerance),
        "best_delay_ns": float(best_delay_ns),
        "best_counts": float(best_counts),
        "selection_mode": selection_mode,
        "gaussian_popt": gaussian_popt,
        "gaussian_perr": gaussian_perr,
    }
    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    dm.save_figure(fig, file_path)
    print(f"Saved to {file_path}")

    plt.draw()
    tb.reset_cfm()

    return {
        "delays_ns": delays_ns,
        "counts": counts,
        "avg_counts": avg_counts,
        "expected_counts": expected,
        "tolerance": tolerance,
        "best_delay_ns": best_delay_ns,
        "best_counts": best_counts,
        "selection_mode": selection_mode,
        "laser_fall_delay_ns": laser_fall_delay_ns,
        "apd_gate_delay_ns": apd_gate_delay_ns,
        "gaussian_popt": gaussian_popt,
        "gaussian_perr": gaussian_perr,
    }


if __name__ == "__main__":
    pass
