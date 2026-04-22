# -*- coding: utf-8 -*-
"""
Single-NV / single-pixel Rabi sweep.

- set microwave amp/freq once
- for each run:
    - optional targeting
    - for each tau:
        - stream_load once for that tau
        - stream_start
        - read_counter_modulo_gates(2, num_reps)

Sequence convention:
    gate 0 = reference  (no MW pulse)
    gate 1 = signal     (MW pulse of duration tau_ns)

Returns:
    raw_data, proc_data

Created on March 17th, 2026

@author: sbchand
"""

import matplotlib.pyplot as plt
import numpy as np
import time

# BUG FIX 1: Removed three bad imports that crashed on load and shadowed 'fig':
#   - from figures.zfs_vs_t.zfs_vs_t_main import fig          (shadowed local fig)
#   - from figures.zfs_vs_t.deconvolve_spectral_function import fig  (shadowed local fig again)
#   - from majorroutines.calibration import optimize_xy        (unused, wrong namespace)
from utils import tool_belt as tb
from utils import kplotlib as kpl
import majorroutines.targeting as targeting
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode


def _build_tau_ns_list(
    uwave_dur_ns_list=None,
    uwave_dur_min_ns=None,
    uwave_dur_max_ns=None,
    num_steps=None,
):
    if uwave_dur_ns_list is not None:
        tau_ns_list = np.asarray(uwave_dur_ns_list, dtype=int).ravel()
    else:
        if uwave_dur_min_ns is None or uwave_dur_max_ns is None or num_steps is None:
            raise ValueError(
                "Provide either uwave_dur_ns_list OR "
                "(uwave_dur_min_ns, uwave_dur_max_ns, num_steps)."
            )
        tau_ns_list = np.linspace(
            int(uwave_dur_min_ns),
            int(uwave_dur_max_ns),
            int(num_steps),
        )
        tau_ns_list = np.rint(tau_ns_list).astype(int)

    tau_ns_list = np.unique(tau_ns_list)
    if len(tau_ns_list) == 0:
        raise ValueError("tau_ns_list is empty.")
    if np.any(tau_ns_list < 0):
        raise ValueError("All microwave pulse durations must be >= 0 ns.")
    return tau_ns_list


def _process_rabi_counts(sig_counts, ref_counts, num_reps, readout_ns, norm_mode):
    """
    sig_counts, ref_counts shape = (num_runs, num_steps)
    Returns per-step processed arrays.
    """
    sig_counts = np.asarray(sig_counts, dtype=float)
    ref_counts = np.asarray(ref_counts, dtype=float)

    num_steps = sig_counts.shape[1]

    sig_kcps = np.full(num_steps, np.nan, dtype=float)
    ref_kcps = np.full(num_steps, np.nan, dtype=float)
    norm = np.full(num_steps, np.nan, dtype=float)
    norm_ste = np.full(num_steps, np.nan, dtype=float)
    num_valid_runs = np.zeros(num_steps, dtype=int)

    for step_ind in range(num_steps):
        valid_mask = np.isfinite(sig_counts[:, step_ind]) & np.isfinite(
            ref_counts[:, step_ind]
        )
        num_valid_runs[step_ind] = int(np.sum(valid_mask))

        if not np.any(valid_mask):
            continue

        sig_col = sig_counts[valid_mask, step_ind].reshape(-1, 1)
        ref_col = ref_counts[valid_mask, step_ind].reshape(-1, 1)

        sig_kcps_i, ref_kcps_i, norm_i, norm_ste_i = tb.process_counts(
            sig_col,
            ref_col,
            int(num_reps),
            int(readout_ns),
            norm_mode=norm_mode,
        )

        sig_kcps[step_ind] = float(sig_kcps_i[0])
        ref_kcps[step_ind] = float(ref_kcps_i[0])
        norm[step_ind] = float(norm_i[0])
        norm_ste[step_ind] = float(norm_ste_i[0])

    contrast = 1.0 - norm

    return {
        "sig_kcps": sig_kcps,
        "ref_kcps": ref_kcps,
        "norm": norm,
        "norm_ste": norm_ste,
        "contrast": contrast,
        "num_valid_runs": num_valid_runs,
    }


def _expected_step_time_s(tau_ns, polarization_ns, readout_ns, num_reps,
                          uwave_delay_ns=151, laser_delay_ns=0,
                          meas_buffer_ns=1000, transient_ns=200):
    """
    Pure-Python mirror of rabi.py's period formula.

    period = front_buffer + 2*(pol + tau + 2*transient + readout + meas_buffer)

    Returns the expected wall-clock time for one step in seconds:
        period_ns * num_reps * 1e-9

    Overhead for stream_load + LabRAD round-trips is NOT included here;
    it shows up as the gap between measured and expected time.

    All inputs are cast to plain Python int/float before arithmetic to avoid
    numpy.int64 silent overflow (int64 max ~9.2e18, but intermediate products
    like period_ns * num_reps can exceed that with large num_reps).
    """
    front_buffer = max(int(uwave_delay_ns), int(laser_delay_ns))
    period_ns = (
        front_buffer
        + 2 * (int(polarization_ns) + int(tau_ns) + 2 * int(transient_ns)
               + int(readout_ns) + int(meas_buffer_ns))
    )
    return float(period_ns) * float(num_reps) * 1e-9


def main(
    nv_sig,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    num_steps,
    uwave_ind=0,
    readout_ns=None,
    uwave_power_dbm=10,
    uwave_freq_ghz=2.8322,
    laser_power=None,
    optimize_between_runs=True,
    optimize_xy_kwargs=None,
    do_plot=True,
    do_save=True,
    norm_mode=NormMode.SINGLE_VALUED,
):
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    tau_ns_list = np.linspace(min_tau, max_tau, num_steps)
    tau_ns_list = np.rint(tau_ns_list).astype(int)
    tau_ns_list = np.unique(tau_ns_list)

    readout_vkey = VirtualLaserKey.SPIN_READOUT
    pol_vkey = VirtualLaserKey.SPIN_POL

    pol_dict = tb.get_virtual_laser_dict(pol_vkey)
    polarization_ns = int(nv_sig.pulse_durations.get(pol_vkey, pol_dict["duration"]))

    if readout_ns is None:
        readout_dict = tb.get_virtual_laser_dict(readout_vkey)
        readout_ns = int(
            nv_sig.pulse_durations.get(readout_vkey, readout_dict["duration"])
        )
    readout_ns = int(readout_ns)

    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_power_dbm is None:
        uwave_power_dbm = vsg.get("uwave_power", None)

    if uwave_power_dbm is not None:
        sig_gen.set_amp(float(uwave_power_dbm))

    freq_ghz = uwave_freq_ghz if uwave_freq_ghz is not None else vsg["frequency"]
    sig_gen.set_freq(float(freq_ghz))
    sig_gen.uwave_on()

    seq_file = "rabi.py"

    # --- Timing diagnostics setup ---
    # uwave_delay comes from cryo config (sig_gen_STAN_sg394_3 delay = 151 ns).
    # laser_delay for the green laser (laser_COBO_520) is 0 ns.
    # These mirror the values used inside rabi.py so our expected times are exact.
    _uwave_delay_ns = 151
    _laser_delay_ns = 0
    # Per-step expected hardware time (pure sequence, no overhead)
    step_expected_s = np.array([
        _expected_step_time_s(
            int(tau), polarization_ns, readout_ns, num_reps,
            uwave_delay_ns=_uwave_delay_ns,
            laser_delay_ns=_laser_delay_ns,
        )
        for tau in tau_ns_list
    ])
    # Accumulators filled during the run
    step_wall_times = np.full((num_runs, len(tau_ns_list)), np.nan)  # actual s per step
    run_wall_times  = np.full(num_runs, np.nan)                      # actual s per run

    ref_counts = np.full((num_runs, len(tau_ns_list)), np.nan)
    sig_counts = np.full((num_runs, len(tau_ns_list)), np.nan)

    # Running average accumulators for O(1) live plot updates.
    # After run_ind runs, norm_running[step] = mean(sig/ref) across all completed runs.
    # On each new point we do one scalar update rather than reprocessing the full array.
    norm_running = np.full(len(tau_ns_list), np.nan)  # cross-run mean norm per step
    norm_sq_running = np.full(len(tau_ns_list), np.nan)  # cross-run mean of norm^2 (for STE)
    valid_run_count = np.zeros(len(tau_ns_list), dtype=int)  # how many runs contributed

    timestamp = dm.get_time_stamp()

    if do_plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("MW pulse duration τ (ns)")
        ax.set_ylabel("Normalized signal")
        ax.set_title("Rabi")
        (line_norm,) = ax.plot([], [], marker="o")
        (line_ste_hi,) = ax.plot([], [], color="gray", linewidth=0.7, alpha=0.5)
        (line_ste_lo,) = ax.plot([], [], color="gray", linewidth=0.7, alpha=0.5)
    else:
        fig = None
        ax = None
        line_norm = None
        line_ste_hi = None
        line_ste_lo = None

    tb.init_safe_stop()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    for run_ind in range(num_runs):
        print(f"Run {run_ind + 1}/{num_runs}")

        if tb.safe_stop():
            break
        print(time.strftime("%Y-%m-%d %H:%M:%S"))

        if optimize_between_runs:
            targeting.compensate_for_drift(nv_sig)
        print(time.strftime("%Y-%m-%d %H:%M:%S"))

        # Re-enable sig gen after optimization (reset_cfm turns it off)
        if uwave_power_dbm is not None:
            sig_gen.set_amp(float(uwave_power_dbm))
        sig_gen.set_freq(float(freq_ghz))
        sig_gen.uwave_on()

        # Open tag stream ONCE per run
        counter_server.start_tag_stream()
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        run_t0 = time.perf_counter()

        try:
            for step_ind, tau_ns in enumerate(tau_ns_list):
                if tb.safe_stop():
                    break

                seq_args = [
                    int(tau_ns),
                    int(polarization_ns),
                    int(readout_ns),
                    int(uwave_ind),
                    pol_vkey.name,
                    readout_vkey.name,
                    laser_power,
                ]

                seq_args_string = tb.encode_seq_args(seq_args)

                # Time each LabRAD call individually to find where overhead lives.
                # On first step of first run, all four timings are printed so you
                # can immediately see which call dominates.
                step_t0 = time.perf_counter()

                t0 = time.perf_counter()
                ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)
                t_stream_load = time.perf_counter() - t0

                if step_ind == 0 and run_ind == 0:
                    print(f"  Sequence period: {ret_vals} ns (rabi.py loaded)")

                t0 = time.perf_counter()
                counter_server.clear_buffer()
                t_clear = time.perf_counter() - t0

                t0 = time.perf_counter()
                pulsegen_server.stream_start(int(num_reps))
                t_stream_start = time.perf_counter() - t0

                # read_counter_summed returns just [ref_total, sig_total] — 2 integers.
                # Transfer cost is constant (16 bytes) regardless of num_reps,
                # vs num_reps*16 bytes for read_counter_separate_gates.
                # Requires counter.py setting 212 to be deployed on the tagger server.
                new_counts = counter_server.read_counter_summed(int(num_reps))

                step_wall = time.perf_counter() - step_t0
                t_read = step_wall - t_stream_load - t_clear - t_stream_start
                step_wall_times[run_ind, step_ind] = step_wall

                # Diagnostic on first step to verify the new method works
                if step_ind == 0 and run_ind == 0:
                    print(f"  [diag] read_counter_summed returned: {new_counts}", flush=True)

                # Unpack directly — no array needed
                ref_counts[run_ind, step_ind] = int(new_counts[0])
                sig_counts[run_ind, step_ind] = int(new_counts[1])

                ref_val = ref_counts[run_ind, step_ind]
                sig_val = sig_counts[run_ind, step_ind]
                norm_val = sig_val / ref_val if ref_val > 0 else float("nan")
                exp_s = step_expected_s[step_ind]
                overhead_ms = (step_wall - exp_s) * 1e3

                # Everything on one line — can't be split across a scroll boundary.
                # Detail breakdown prints every step of run 1, then every 10th step.
                _print_detail = (run_ind == 0) or (step_ind % 10 == 0)
                timing_str = (
                    f"  [load={t_stream_load*1e3:.0f} clr={t_clear*1e3:.0f} "
                    f"start={t_stream_start*1e3:.0f} read={t_read*1e3:.0f}ms]"
                    if _print_detail else ""
                )
                print(
                    f"tau={int(tau_ns):>4d} ns | "
                    f"ref={int(ref_val)}, sig={int(sig_val)}, norm={norm_val:.4f} | "
                    f"wall={step_wall:.2f}s exp={exp_s:.2f}s ovhd={overhead_ms:+.0f}ms"
                    + timing_str,
                    flush=True,
                )

                # Welford accumulation is still per-step (O(1), negligible cost)
                # so norm_running stays current throughout the run.
                # The actual draw + plt.pause is deferred to once per run below.
                if do_plot and np.isfinite(norm_val):
                    n = valid_run_count[step_ind]
                    if n == 0:
                        norm_running[step_ind] = norm_val
                        norm_sq_running[step_ind] = norm_val ** 2
                    else:
                        norm_running[step_ind] += (norm_val - norm_running[step_ind]) / (n + 1)
                        norm_sq_running[step_ind] += (norm_val ** 2 - norm_sq_running[step_ind]) / (n + 1)
                    valid_run_count[step_ind] += 1

        finally:
            try:
                counter_server.stop_tag_stream()
            except Exception:
                pass

        run_wall = time.perf_counter() - run_t0
        run_wall_times[run_ind] = run_wall
        run_expected_s = step_expected_s.sum()
        valid_steps = np.sum(np.isfinite(step_wall_times[run_ind]))
        avg_overhead_ms = (
            (np.nansum(step_wall_times[run_ind]) - run_expected_s) / valid_steps * 1e3
            if valid_steps > 0 else float("nan")
        )
        print(
            f"  Run {run_ind + 1} done | "
            f"wall={run_wall:.1f}s  exp={run_expected_s:.1f}s  "
            f"avg overhead/step={avg_overhead_ms:+.0f}ms"
        )

        # Draw once per run — avoids 30x plt.pause calls (each ~20-50ms) per run.
        # STE is computed from the final state of the accumulators for this run.
        if do_plot:
            n_valid_arr = np.where(valid_run_count > 1, valid_run_count, 0)
            var_arr = np.where(
                n_valid_arr > 1,
                norm_sq_running - norm_running ** 2,
                0.0,
            )
            ste_arr = np.where(
                n_valid_arr > 1,
                np.sqrt(np.maximum(var_arr, 0.0) / np.maximum(n_valid_arr, 1)),
                0.0,
            )
            line_norm.set_data(tau_ns_list, norm_running)
            line_ste_hi.set_data(tau_ns_list, norm_running + ste_arr)
            line_ste_lo.set_data(tau_ns_list, norm_running - ste_arr)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

    proc_arrays = _process_rabi_counts(
        sig_counts,
        ref_counts,
        int(num_reps),
        int(readout_ns),
        norm_mode,
    )

    # --- Full-experiment timing summary ---
    total_wall_s = np.nansum(run_wall_times)
    total_expected_s = step_expected_s.sum() * num_runs
    total_overhead_s = total_wall_s - total_expected_s
    per_step_overhead_ms = (
        total_overhead_s / (len(tau_ns_list) * num_runs) * 1e3
    )
    efficiency_pct = (
        total_expected_s / total_wall_s * 100 if total_wall_s > 0 else 0.0
    )
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print(f"  Total wall time      : {total_wall_s:.1f} s  ({total_wall_s / 60:.2f} min)")
    print(f"  Total expected (HW)  : {total_expected_s:.1f} s  ({total_expected_s / 60:.2f} min)")
    print(f"  Total overhead       : {total_overhead_s:.1f} s  ({total_overhead_s / 60:.2f} min)")
    print(f"  Avg overhead / step  : {per_step_overhead_ms:.0f} ms"
          f"  (stream_load + LabRAD + read_counter latency)")
    print(f"  HW efficiency        : {efficiency_pct:.1f}%  (ideal = 100%)")
    print("=" * 70 + "\n")

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "freq_ghz": float(freq_ghz),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_power_dbm": uwave_power_dbm,
        "polarization_ns": int(polarization_ns),
        "readout_ns": int(readout_ns),
        "tau_ns_list": tau_ns_list.tolist(),
        "sig_counts": sig_counts.tolist(),
        "ref_counts": ref_counts.tolist(),
        "step_wall_times_s": step_wall_times.tolist(),
        "step_expected_times_s": step_expected_s.tolist(),
        "run_wall_times_s": run_wall_times.tolist(),
    }

    proc_data = {
        "freq_ghz": float(freq_ghz),
        "tau_ns_list": tau_ns_list.tolist(),
        "sig_kcps": proc_arrays["sig_kcps"].tolist(),
        "ref_kcps": proc_arrays["ref_kcps"].tolist(),
        "norm": proc_arrays["norm"].tolist(),
        "norm_ste": proc_arrays["norm_ste"].tolist(),
        "contrast": proc_arrays["contrast"].tolist(),
        "num_valid_runs": proc_arrays["num_valid_runs"].tolist(),
    }

    # BUG FIX 5 & 6: Gate saving on do_save flag, and guard fig against None
    if do_save:
        save_timestamp = dm.get_time_stamp()
        file_path = dm.get_file_path(__file__, save_timestamp, getattr(nv_sig, "name", "nv"))
        dm.save_raw_data(raw_data, file_path)
        if fig is not None:
            dm.save_figure(fig, file_path)
        print(f"Saved data to {file_path}")

    tb.reset_cfm()
    return raw_data, proc_data


def _diagnose_read_counter(nv_sig, num_reps_list=(1000, 5000, 20000, 50000, 100000)):
    """
    Standalone timing diagnostic — does NOT need an NV.

    Loads a fixed rabi.py sequence once, then for each num_reps value:
      - stream_start
      - read_counter_modulo_gates(2, num_reps)
    and prints the wall time vs expected hardware time.

    If read_counter overhead is CONSTANT across num_reps  → tagger polling interval (Cause A).
    If read_counter overhead SCALES with num_reps         → pulser startup latency (Cause B).

    Call from control_panel_cryo.py:
        from majorroutines.confocal.confocal_rabi import _diagnose_read_counter
        _diagnose_read_counter(nv_sig)
    """
    from utils.constants import VirtualLaserKey

    tb.reset_cfm()
    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    # Fixed sequence parameters — tau=200 ns, same as a typical Rabi midpoint.
    tau_ns       = 200
    pol_vkey     = VirtualLaserKey.SPIN_POL
    readout_vkey = VirtualLaserKey.SPIN_READOUT
    pol_dict     = tb.get_virtual_laser_dict(pol_vkey)
    readout_dict = tb.get_virtual_laser_dict(readout_vkey)
    polarization_ns = int(nv_sig.pulse_durations.get(pol_vkey,  pol_dict["duration"]))
    readout_ns      = int(nv_sig.pulse_durations.get(readout_vkey, readout_dict["duration"]))

    seq_args = [tau_ns, polarization_ns, readout_ns, 0,
                pol_vkey.name, readout_vkey.name, None]
    seq_args_string = tb.encode_seq_args(seq_args)

    # Load the sequence once — period is fixed regardless of num_reps.
    ret_vals = pulsegen_server.stream_load("rabi.py", seq_args_string)
    period_ns = int(ret_vals[0]) if hasattr(ret_vals, '__len__') else int(ret_vals)
    print(f"\nSequence period: {period_ns} ns")
    print(f"{'num_reps':>10}  {'exp_s':>7}  {'read_s':>7}  {'overhead_ms':>12}  {'ratio':>6}")
    print("-" * 55)

    counter_server.start_tag_stream()
    try:
        for num_reps in num_reps_list:
            exp_s = _expected_step_time_s(
                tau_ns, polarization_ns, readout_ns, num_reps,
                uwave_delay_ns=151, laser_delay_ns=0,
            )

            counter_server.clear_buffer()

            t0 = time.perf_counter()
            pulsegen_server.stream_start(int(num_reps))
            new_counts = counter_server.read_counter_modulo_gates(2, int(num_reps))
            wall_s = time.perf_counter() - t0

            overhead_ms = (wall_s - exp_s) * 1e3
            ratio = wall_s / exp_s if exp_s > 0 else float("nan")
            count_arr = np.array(new_counts, dtype=np.int64)
            total_counts = int(count_arr.sum())

            print(
                f"{num_reps:>10}  {exp_s:>7.3f}s  {wall_s:>7.3f}s  "
                f"{overhead_ms:>+10.0f}ms  {ratio:>5.2f}x  "
                f"(total counts={total_counts})"
            )
    finally:
        counter_server.stop_tag_stream()

    tb.reset_cfm()
    print(
        "\nInterpretation:\n"
        "  overhead ~constant across num_reps → tagger polling interval (fix: reduce poll period in tagger server)\n"
        "  overhead scales with num_reps      → pulser startup latency  (fix: warm-start or pre-arm the pulser)\n"
    )


if __name__ == "__main__":
    pass
