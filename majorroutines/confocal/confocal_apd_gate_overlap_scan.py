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
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils.constants import VirtualLaserKey


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

    # nv_sig.expected_counts is stored in photons-per-single-readout units
    # (same as what targeting / stationary_count compare against). We scale
    # it by num_reps here so every number reported by this routine is in
    # the same unit as the total counts actually read from the counter.
    expected_counts_per_rep = getattr(nv_sig, "expected_counts", None)
    tolerance = float(tolerance)
    if expected_counts_per_rep is None:
        expected_total = None
        print(
            "NOTE: nv_sig.expected_counts is None. The optimum will fall "
            "back to the maximum-counts delay. For characterization, set "
            "nv_sig.expected_counts in control_panel_cryo.py to the NV's "
            "photons-per-readout value."
        )
    else:
        expected_total = float(expected_counts_per_rep) * float(num_reps)
        pct = tolerance * 100.0
        print(
            f"Target: total counts should drop to expected_counts * num_reps "
            f"= {expected_counts_per_rep} * {int(num_reps)} = "
            f"{expected_total:.0f} (+/-{pct:.0f}%) at the optimum. "
            f"Criterion: first delay with counts <= "
            f"{expected_total * (1 + tolerance):.0f}."
        )

    counts = np.full((num_runs, len(delays_ns)), np.nan, dtype=float)

    fig, ax = plt.subplots()
    ax.set_xlabel("Delay from laser-off command to APD gate (ns)")
    ax.set_ylabel(f"Total counts (summed over {int(num_reps)} reps)")
    ax.set_title(
        "APD gate delay scan -- find first delay where counts = expected"
    )
    (line_avg,) = ax.plot([], [], marker="o", label="Counts")
    if expected_total is not None:
        ax.axhline(
            expected_total, color="gray", linestyle="--", linewidth=1,
            label=f"expected = {expected_total:.0f}",
        )
        # +/- tolerance band. Upper edge is the selection threshold (first
        # delay at or below it is the optimum). Lower edge is a sanity
        # guide -- if the plateau lives below it, expected_counts may be
        # stale or the NV drifted.
        ax.axhspan(
            expected_total * (1 - tolerance),
            expected_total * (1 + tolerance),
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

            if expected_total is None:
                print(f"  delay={int(delay_ns):>5d} ns | counts={total_counts}")
            else:
                ratio = total_counts / expected_total
                print(
                    f"  delay={int(delay_ns):>5d} ns | "
                    f"counts={total_counts} (ratio to expected = {ratio:.2f})"
                )

        avg_counts_so_far = np.nanmean(counts[: run_ind + 1], axis=0)
        line_avg.set_data(delays_ns, avg_counts_so_far)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

    avg_counts = np.nanmean(counts, axis=0)

    # Optimal delay selection -- all comparisons in total-counts units
    if expected_total is not None:
        upper = expected_total * (1 + tolerance)
        lower = expected_total * (1 - tolerance)
        below = np.where(np.isfinite(avg_counts) & (avg_counts <= upper))[0]
        if below.size > 0:
            best_ind = int(below[0])  # earliest delay that reaches plateau
            best_delay_ns = int(delays_ns[best_ind])
            selection_mode = "expected_counts"
        else:
            print(
                "WARNING: counts never dropped to "
                f"{upper:.0f} (= expected * (1 + {tolerance:.2f})). The "
                "scan range may not extend far enough past the laser fall. "
                "Reporting the lowest-counts delay instead."
            )
            best_ind = int(np.nanargmin(avg_counts))
            best_delay_ns = int(delays_ns[best_ind])
            selection_mode = "min_counts_fallback"
    else:
        best_ind = int(np.nanargmax(avg_counts))
        best_delay_ns = int(delays_ns[best_ind])
        selection_mode = "max_counts_fallback"

    print("\nDone")
    print(f"Optimal delay (laser-off cmd -> APD gate) = {best_delay_ns} ns")
    print(f"Counts at optimum = {avg_counts[best_ind]:.0f}")
    if expected_total is not None:
        ratio = avg_counts[best_ind] / expected_total
        print(
            f"Expected counts = {expected_total:.0f} "
            f"({expected_counts_per_rep} per rep * {int(num_reps)} reps) "
            f"(ratio = {ratio:.2f})"
        )
        if avg_counts[best_ind] < lower:
            print(
                f"NOTE: counts at the optimum ({avg_counts[best_ind]:.0f}) "
                f"are below expected * (1 - {tolerance:.2f}) = {lower:.0f}. "
                "Either nv_sig.expected_counts is stale / too high, or the "
                "NV has drifted -- re-measure expected_counts."
            )
    print(f"Selection mode: {selection_mode}")

    # Overlay optimum marker on the plot
    ax.axvline(
        best_delay_ns, color="red", linestyle=":", linewidth=1,
        label=f"optimum = {best_delay_ns} ns",
    )
    ax.legend(loc="best")
    plt.draw()

    tb.reset_cfm()

    return {
        "delays_ns": delays_ns,
        "counts": counts,
        "avg_counts": avg_counts,
        "expected_counts_per_rep": expected_counts_per_rep,
        "expected_total": expected_total,
        "tolerance": tolerance,
        "best_delay_ns": best_delay_ns,
        "selection_mode": selection_mode,
        "laser_fall_delay_ns": laser_fall_delay_ns,
        "apd_gate_delay_ns": apd_gate_delay_ns,
    }


if __name__ == "__main__":
    pass
