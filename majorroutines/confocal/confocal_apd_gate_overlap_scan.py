# -*- coding: utf-8 -*-
"""
Sweep the *physical* delay between the end of the readout laser pulse
and the opening of the APD gate. Positive delay = APD gate physically
opens after the laser physically turns off (guaranteed no overlap);
negative delay = APD gate opens while the laser is still on.

Two hardware-delay parameters convert between the command timeline
(what the pulser actually streams) and the physical timeline (what the
sample sees):

    laser_fall_delay_ns -- time between the laser command falling edge
        and the optical power actually dropping. Defaults to the laser's
        "delay" entry in the config, which typically characterizes the
        rising edge; override if the falling edge has been measured.
    apd_gate_delay_ns   -- propagation delay from APD-gate command HIGH
        to the APD actually enabling counting. Defaults to 0.

Both are forwarded to the sequence so `gate_delay_ns` is interpreted in
physical time.

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
    """Gaussian on a flat baseline. Models the counts-vs-delay curve as
    the descending tail of a peak centered near (or before) delay=0."""
    return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)) + offset


def _gaussian_crossing(popt, target):
    """Given fitted params (amp, mu, sigma, offset), return the delay
    where the Gaussian descends to `target`, or None if no valid
    crossing exists."""
    amp, mu, sigma, offset = popt
    if amp <= 0 or sigma <= 0:
        return None
    frac = (target - offset) / amp
    if not (0.0 < frac <= 1.0):
        return None  # target outside [offset, offset + amp]
    return float(mu + sigma * np.sqrt(-2.0 * np.log(frac)))


def main(
    nv_sig,
    num_reps=int(2e5),
    num_runs=3,
    delay_min_ns=-1000,
    delay_max_ns=1000,
    num_steps=81,
    laser_on_ns=None,
    gate_width_ns=300,
    laser_vkey=VirtualLaserKey.SPIN_READOUT,
    laser_fall_delay_ns=None,
    apd_gate_delay_ns=0,
    tolerance=0.10,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    if laser_on_ns is None:
        vld = tb.get_virtual_laser_dict(laser_vkey)
        laser_on_ns = int(nv_sig.pulse_durations.get(laser_vkey, int(vld["duration"])))
    laser_on_ns = int(laser_on_ns)

    # Default the laser falling-edge delay to the laser's config "delay"
    # (which is usually the rising-edge delay). This is a best-effort
    # placeholder until the falling edge is characterized separately.
    if laser_fall_delay_ns is None:
        laser_name = tb.get_physical_laser_name(laser_vkey)
        laser_fall_delay_ns = int(
            tb.get_physical_laser_dict(laser_name)["delay"]
        )
    laser_fall_delay_ns = int(laser_fall_delay_ns)
    apd_gate_delay_ns = int(apd_gate_delay_ns)

    print(
        f"laser_on_ns={laser_on_ns}, gate_width_ns={gate_width_ns}, "
        f"laser_fall_delay_ns={laser_fall_delay_ns}, "
        f"apd_gate_delay_ns={apd_gate_delay_ns}"
    )

    # Safety: a negative gate_delay_ns physically opens the APD gate while
    # the readout laser is still on, which risks damaging the detector via
    # stray/scattered green light. Clamp to zero and warn.
    if delay_min_ns < 0:
        print(
            f"WARNING: delay_min_ns={delay_min_ns} would overlap the APD "
            f"gate with the readout laser. Clamping to 0 ns to protect the "
            f"detector. Pass a negative value deliberately only with a "
            f"physical shutter / ND filter protecting the APD."
        )
        delay_min_ns = 0
    if delay_max_ns < delay_min_ns:
        raise ValueError(
            f"delay_max_ns ({delay_max_ns}) must be >= delay_min_ns "
            f"({delay_min_ns})."
        )

    delays_ns = np.linspace(delay_min_ns, delay_max_ns, num_steps)
    delays_ns = np.rint(delays_ns).astype(int)

    # Physical-timing summary: show the user exactly what each swept delay
    # means relative to the readout pulse. gate_delay_ns is already the
    # *physical* delay (laser-off -> gate-on), so status is just a reminder
    # of the margin each point has.
    print("\nPhysical timing table (gate opens relative to end of laser pulse):")
    print(f"{'delay (ns)':>12}  {'margin vs laser-off':>22}  status")
    for d in delays_ns:
        status = "SAFE" if d > 0 else ("AT EDGE" if d == 0 else "OVERLAP")
        print(f"{int(d):>12d}  {int(d):>19d} ns  {status}")
    print(
        "Note: 'SAFE' assumes laser_fall_delay_ns correctly characterizes "
        "the optical fall time. If the true fall tail is longer than the "
        "configured value, even SAFE points can briefly overlap."
    )
    print()

    # nv_sig.expected_counts is in the same scale as the total counts this
    # scan integrates per delay (summed over num_reps). The displayed
    # "counts" value at the optimal delay should match expected_counts
    # directly -- e.g. expected_counts=55 means the optimum is the first
    # delay where counts settle to 55 within the tolerance window.
    expected = getattr(nv_sig, "expected_counts", None)
    tolerance = float(tolerance)
    if expected is None:
        print(
            "NOTE: nv_sig.expected_counts is None. The optimum will fall "
            "back to the maximum-counts delay. For characterization, set "
            "nv_sig.expected_counts in control_panel_cryo.py."
        )
    else:
        expected = float(expected)
        pct = tolerance * 100.0
        print(
            f"Target: counts should drop to expected_counts = {expected:g} "
            f"(+/-{pct:.0f}%) at the optimum. Criterion: first delay with "
            f"counts <= {expected * (1 + tolerance):.2f}."
        )

    # counts[run, ind] stores total photons at that delay (summed over
    # num_reps reps), directly comparable to expected_counts.
    counts = np.full((num_runs, len(delays_ns)), np.nan, dtype=float)

    fig, ax = plt.subplots()
    ax.set_xlabel("Delay from laser-off command to APD gate (ns)")
    ax.set_ylabel("Counts")
    ax.set_title(
        "APD gate delay scan -- find first delay where counts = expected"
    )
    (line_avg,) = ax.plot([], [], marker="o", label="Counts")
    if expected is not None:
        ax.axhline(
            expected, color="gray", linestyle="--", linewidth=1,
            label=f"expected = {expected:g}",
        )
        # +/- tolerance band. Upper edge is the selection threshold (first
        # delay at or below it is the optimum). Lower edge is a sanity
        # guide -- if the plateau lives below it, expected_counts may be
        # stale or the NV drifted.
        ax.axhspan(
            expected * (1 - tolerance),
            expected * (1 + tolerance),
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
                # The sequence emits one sample-clock edge (= one APD gate
                # window) per repetition, so read all num_reps samples
                # and sum. Reading only 1 sample throws away all but the
                # first rep's photons and almost always returns 0.
                new_counts = counter_server.read_counter_simple(int(num_reps))
            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            total_counts = int(np.sum(new_counts)) if len(new_counts) > 0 else 0
            counts[run_ind, ind] = total_counts

            print(f"  delay={int(delay_ns):>5d} ns | counts={total_counts}")

        avg_counts_so_far = np.nanmean(counts[: run_ind + 1], axis=0)
        line_avg.set_data(delays_ns, avg_counts_so_far)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

    avg_counts = np.nanmean(counts, axis=0)

    # Linear-interpolation crossing helper. Kept as a fallback when the
    # Gaussian fit fails or doesn't intersect expected.
    def _crossing_delay(d_arr, c_arr, target):
        """Return (delay, counts) at the first downward crossing of
        target, or None if no crossing exists."""
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

    # Fit a Gaussian-on-baseline to the (delay, counts) curve. The sweep
    # usually shows only the descending half of the peak (the user's
    # "partial Gaussian" observation) so bounds allow mu to sit before
    # the first sampled delay.
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
                _gaussian_plus_offset, d_fit, c_fit, p0=p0, bounds=bounds,
                maxfev=50000,
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

    if expected is not None:
        crossing_from_fit = (
            _gaussian_crossing(gaussian_popt, expected)
            if gaussian_popt is not None
            else None
        )
        # Sanity-check: a Gaussian crossing that lands far outside the
        # swept range is probably a fit artifact; fall back to linear.
        span = float(delays_ns[-1] - delays_ns[0])
        in_range = (
            crossing_from_fit is not None
            and delays_ns[0] - span <= crossing_from_fit <= delays_ns[-1] + span
        )
        if in_range:
            best_delay_ns = float(crossing_from_fit)
            best_counts = float(expected)
            selection_mode = "expected_counts_gaussian"
        else:
            crossing = _crossing_delay(delays_ns, avg_counts, expected)
            if crossing is not None:
                best_delay_ns, best_counts = crossing
                selection_mode = "expected_counts_linear_fallback"
            else:
                print(
                    f"WARNING: counts never dropped to expected = "
                    f"{expected:g} (neither Gaussian fit nor raw samples "
                    "cross it). Reporting the lowest-counts delay instead."
                )
                best_ind = int(np.nanargmin(avg_counts))
                best_delay_ns = float(delays_ns[best_ind])
                best_counts = float(avg_counts[best_ind])
                selection_mode = "min_counts_fallback"
    else:
        best_ind = int(np.nanargmax(avg_counts))
        best_delay_ns = float(delays_ns[best_ind])
        best_counts = float(avg_counts[best_ind])
        selection_mode = "max_counts_fallback"

    print("\nDone")
    print(
        f"Optimal delay (laser-off cmd -> APD gate) = {best_delay_ns:.1f} ns"
    )
    print(f"Counts at optimum = {best_counts:.4g}")
    if expected is not None:
        print(f"Expected counts = {expected:g}")
        # Prefer the fitted baseline (offset) as the plateau indicator
        # when the fit succeeded -- it's less noisy than the raw min.
        if gaussian_popt is not None:
            plateau = float(gaussian_popt[3])  # offset
            plateau_label = "fitted offset"
        else:
            plateau = float(np.nanmin(avg_counts))
            plateau_label = "raw min counts"
        if plateau < expected * (1 - tolerance):
            print(
                f"NOTE: plateau counts ({plateau_label} = {plateau:.4g}) "
                f"are below expected * (1 - {tolerance:.2f}) = "
                f"{expected * (1 - tolerance):.4g}. nv_sig.expected_counts "
                "may be stale or the NV drifted -- re-measure."
            )
    print(f"Selection mode: {selection_mode}")

    # Overlay the Gaussian fit curve on the plot for visual inspection.
    if gaussian_popt is not None:
        t_smooth = np.linspace(delays_ns[0], delays_ns[-1], 400)
        ax.plot(
            t_smooth, _gaussian_plus_offset(t_smooth, *gaussian_popt),
            linestyle="--", color="orange", linewidth=1.5,
            label="Gaussian fit",
        )

    # Mark the (possibly interpolated) optimum on the curve.
    ax.plot(
        [best_delay_ns], [best_counts],
        marker="o", markersize=10, color="red", zorder=5,
        label=f"optimum: delay={best_delay_ns:.1f} ns, counts={best_counts:.4g}",
    )
    ax.axvline(best_delay_ns, color="red", linestyle=":", linewidth=1)
    ax.legend(loc="best")

    # Save raw data + figure for archive
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
    print(f"Saved data + figure to {file_path}")

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
