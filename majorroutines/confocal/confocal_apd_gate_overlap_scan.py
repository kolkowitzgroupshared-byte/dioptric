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

    delays_ns = np.linspace(delay_min_ns, delay_max_ns, num_steps)
    delays_ns = np.rint(delays_ns).astype(int)

    counts = np.full((num_runs, len(delays_ns)), np.nan, dtype=float)

    fig, ax = plt.subplots()
    ax.set_xlabel("Physical delay from end of readout pulse to APD gate (ns)")
    ax.set_ylabel("Total counts")
    ax.set_title("APD gate delay scan (physical time, no overlap for delay >= 0)")
    (line_avg,) = ax.plot([], [], marker="o")

    seq_file = "apd_gate_overlap_scan.py"

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
                new_counts = counter_server.read_counter_simple(1)
            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            total_counts = int(new_counts[0]) if len(new_counts) > 0 else 0
            counts[run_ind, ind] = total_counts

            print(f"  delay={int(delay_ns):>5d} ns | counts={total_counts}")

        avg_counts = np.nanmean(counts[: run_ind + 1], axis=0)
        line_avg.set_data(delays_ns, avg_counts)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

    avg_counts = np.nanmean(counts, axis=0)
    best_ind = int(np.nanargmax(avg_counts))
    best_delay_ns = int(delays_ns[best_ind])

    print("\nDone")
    print(f"Best physical delay (laser-off -> APD-on) = {best_delay_ns} ns")
    print(f"Max average counts = {avg_counts[best_ind]:.1f}")

    tb.reset_cfm()

    return {
        "delays_ns": delays_ns,
        "counts": counts,
        "avg_counts": avg_counts,
        "best_delay_ns": best_delay_ns,
        "laser_fall_delay_ns": laser_fall_delay_ns,
        "apd_gate_delay_ns": apd_gate_delay_ns,
    }


if __name__ == "__main__":
    pass
