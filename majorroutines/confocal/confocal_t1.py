# -*- coding: utf-8 -*-
"""
Single-NV T1 (longitudinal relaxation time) sweep.

Sequence convention (see t1.py):
    gate 0 = reference  (polarize → immediate readout — measures ms=0 fluorescence)
    gate 1 = signal     (polarize → dark wait τ → readout — measures spin relaxation)

norm(τ) = sig / ref decays as:
    norm(τ) = A * exp(-τ / T1) + C

where C ≈ 0.75 (thermal equilibrium mix) and A ≈ 0.25 for a single NV.

Fitting is performed at the end of the run and the T1 estimate is printed.

Created: 2026
@author: Yael
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import time

from utils import tool_belt as tb
from utils import kplotlib as kpl
import majorroutines.targeting as targeting
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tau_ns_list(
    tau_ns_list=None,
    tau_min_ns=None,
    tau_max_ns=None,
    num_steps=None,
    log_scale=True,
):
    """
    Build the array of dark-wait durations to sweep.

    Prefer log-spaced steps for T1 (dynamics span several decades).
    Pass tau_ns_list directly to override.
    """
    if tau_ns_list is not None:
        arr = np.asarray(tau_ns_list, dtype=int).ravel()
    else:
        if tau_min_ns is None or tau_max_ns is None or num_steps is None:
            raise ValueError(
                "Provide either tau_ns_list OR "
                "(tau_min_ns, tau_max_ns, num_steps)."
            )
        if log_scale:
            arr = np.logspace(
                np.log10(max(int(tau_min_ns), 1)),
                np.log10(int(tau_max_ns)),
                int(num_steps),
            )
        else:
            arr = np.linspace(int(tau_min_ns), int(tau_max_ns), int(num_steps))
        arr = np.rint(arr).astype(int)

    arr = np.unique(arr)
    if len(arr) == 0:
        raise ValueError("tau_ns_list is empty.")
    if np.any(arr < 0):
        raise ValueError("All tau values must be >= 0 ns.")
    return arr


def _t1_model(tau, amplitude, T1_ns, offset):
    """Exponential decay: norm(τ) = A * exp(-τ/T1) + C"""
    return amplitude * np.exp(-tau / T1_ns) + offset


def _fit_t1(tau_ns_list, norm, norm_ste=None):
    """
    Fit T1 decay curve.

    Returns dict with keys: T1_ns, T1_us, amplitude, offset, T1_ste_ns,
    success (bool), message (str).
    """
    valid = np.isfinite(norm)
    if np.sum(valid) < 4:
        return {"success": False, "message": "Not enough valid points to fit."}

    tau_fit = tau_ns_list[valid].astype(float)
    norm_fit = norm[valid]
    sigma = norm_ste[valid] if norm_ste is not None else None

    # Initial guess: A ≈ norm[0] - norm[-1], T1 ≈ span/3, C ≈ norm[-1]
    A0 = float(norm_fit[0] - norm_fit[-1])
    T1_0 = float(tau_fit[-1] - tau_fit[0]) / 3.0
    C0 = float(norm_fit[-1])
    p0 = [A0, max(T1_0, 1.0), C0]

    try:
        popt, pcov = curve_fit(
            _t1_model,
            tau_fit,
            norm_fit,
            p0=p0,
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            maxfev=10000,
            bounds=([-2, 1, -1], [2, 1e15, 2]),
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "success": True,
            "amplitude": float(popt[0]),
            "T1_ns": float(popt[1]),
            "T1_us": float(popt[1]) / 1e3,
            "T1_ms": float(popt[1]) / 1e6,
            "offset": float(popt[2]),
            "T1_ste_ns": float(perr[1]),
            "T1_ste_us": float(perr[1]) / 1e3,
            "message": "OK",
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def _expected_step_time_s(tau_ns, polarization_ns, readout_ns, num_reps,
                          laser_delay_ns=0,
                          meas_buffer_ns=1000, transient_ns=200):
    """
    Mirror of t1.py's period formula.

    period = front_buffer + 2 * (pol + transient + tau + transient + readout + meas_buffer)

    Returns expected hardware time (seconds) for one step at num_reps.
    """
    front_buffer = int(laser_delay_ns)
    sig_dark = 2 * int(transient_ns) + int(tau_ns)
    exp_duration = int(polarization_ns) + sig_dark + int(readout_ns) + int(meas_buffer_ns)
    period_ns = front_buffer + 2 * exp_duration
    return float(period_ns) * float(num_reps) * 1e-9


def _process_t1_counts(sig_counts, ref_counts, num_reps, readout_ns, norm_mode):
    """
    Compute normalized signal per step, averaged across valid runs.

    sig_counts, ref_counts: shape (num_runs, num_steps)
    Returns dict of per-step arrays.
    """
    sig_counts = np.asarray(sig_counts, dtype=float)
    ref_counts = np.asarray(ref_counts, dtype=float)
    num_steps = sig_counts.shape[1]

    sig_kcps   = np.full(num_steps, np.nan)
    ref_kcps   = np.full(num_steps, np.nan)
    norm       = np.full(num_steps, np.nan)
    norm_ste   = np.full(num_steps, np.nan)
    num_valid  = np.zeros(num_steps, dtype=int)

    for step_ind in range(num_steps):
        valid_mask = (
            np.isfinite(sig_counts[:, step_ind]) &
            np.isfinite(ref_counts[:, step_ind])
        )
        num_valid[step_ind] = int(np.sum(valid_mask))
        if not np.any(valid_mask):
            continue

        sig_col = sig_counts[valid_mask, step_ind].reshape(-1, 1)
        ref_col = ref_counts[valid_mask, step_ind].reshape(-1, 1)

        sig_k, ref_k, n, n_ste = tb.process_counts(
            sig_col, ref_col,
            int(num_reps), int(readout_ns),
            norm_mode=norm_mode,
        )
        sig_kcps[step_ind]  = float(sig_k[0])
        ref_kcps[step_ind]  = float(ref_k[0])
        norm[step_ind]      = float(n[0])
        norm_ste[step_ind]  = float(n_ste[0])

    return {
        "sig_kcps":      sig_kcps,
        "ref_kcps":      ref_kcps,
        "norm":          norm,
        "norm_ste":      norm_ste,
        "num_valid_runs": num_valid,
    }


# ---------------------------------------------------------------------------
# Main measurement routine
# ---------------------------------------------------------------------------

def main(
    nv_sig,
    num_reps,
    num_runs,
    tau_min_ns=100,
    tau_max_ns=10_000_000,   # 10 ms — well past typical NV T1
    num_steps=30,
    tau_ns_list=None,        # override log sweep if provided
    log_scale=True,
    readout_ns=None,
    laser_power=None,
    optimize_between_runs=True,
    do_plot=True,
    do_save=True,
    norm_mode=NormMode.SINGLE_VALUED,
):
    """
    Sweep dark-wait time τ and measure T1 decay of a single NV.

    Parameters
    ----------
    nv_sig : NvSig — NV site object (carries coords and pulse_durations)
    num_reps : int — repetitions per tau per run (sets shot noise floor)
    num_runs : int — averages to accumulate (drift-compensated between runs)
    tau_min_ns : int — shortest dark wait in ns
    tau_max_ns : int — longest dark wait in ns (should be ≥ 3×T1)
    num_steps  : int — number of τ points
    tau_ns_list : array-like or None — explicit τ list (overrides min/max/steps)
    log_scale  : bool — use log-spaced steps (recommended for T1)
    readout_ns : int or None — override per-NV readout duration
    laser_power : float or None — laser power (passed to pulse sequence)
    optimize_between_runs : bool — run targeting.compensate_for_drift between runs
    do_plot    : bool — live plot normalized signal vs τ
    do_save    : bool — save raw + processed data to disk
    norm_mode  : NormMode — normalization method

    Returns
    -------
    raw_data  : dict
    proc_data : dict   (includes T1 fit result)
    """
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server  = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    # Build tau list
    tau_ns_arr = _build_tau_ns_list(
        tau_ns_list=tau_ns_list,
        tau_min_ns=tau_min_ns,
        tau_max_ns=tau_max_ns,
        num_steps=num_steps,
        log_scale=log_scale,
    )
    num_steps = len(tau_ns_arr)  # may differ from requested if values collapsed

    # Virtual laser keys — same as Rabi
    readout_vkey = VirtualLaserKey.SPIN_READOUT
    pol_vkey     = VirtualLaserKey.SPIN_POL

    pol_dict     = tb.get_virtual_laser_dict(pol_vkey)
    readout_dict = tb.get_virtual_laser_dict(readout_vkey)

    polarization_ns = int(nv_sig.pulse_durations.get(pol_vkey, pol_dict["duration"]))

    if readout_ns is None:
        readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, readout_dict["duration"]))
    readout_ns = int(readout_ns)

    seq_file = "t1.py"

    # Timing diagnostics — green laser delay is 0 ns in cryo config
    _laser_delay_ns = 0
    step_expected_s = np.array([
        _expected_step_time_s(
            int(tau), polarization_ns, readout_ns, num_reps,
            laser_delay_ns=_laser_delay_ns,
        )
        for tau in tau_ns_arr
    ])

    # Accumulators
    step_wall_times = np.full((num_runs, num_steps), np.nan)
    run_wall_times  = np.full(num_runs, np.nan)
    ref_counts      = np.full((num_runs, num_steps), np.nan)
    sig_counts      = np.full((num_runs, num_steps), np.nan)

    # Running mean / STE for live plot (Welford update — O(1) per step)
    norm_running    = np.full(num_steps, np.nan)
    norm_sq_running = np.full(num_steps, np.nan)
    valid_run_count = np.zeros(num_steps, dtype=int)

    timestamp = dm.get_time_stamp()

    # Set up plot
    if do_plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Dark wait τ (ns)")
        ax.set_ylabel("Normalized signal  (sig / ref)")
        ax.set_title("T1 relaxation")
        ax.set_xscale("log" if log_scale else "linear")
        (line_norm,)   = ax.plot([], [], marker="o", label="data")
        (line_ste_hi,) = ax.plot([], [], color="gray", linewidth=0.7, alpha=0.5)
        (line_ste_lo,) = ax.plot([], [], color="gray", linewidth=0.7, alpha=0.5)
        ax.legend()
    else:
        fig = ax = line_norm = line_ste_hi = line_ste_lo = None

    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print(f"\nRun {run_ind + 1}/{num_runs}  {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if tb.safe_stop():
            print("Safe stop triggered — ending early.")
            break

        if optimize_between_runs:
            targeting.compensate_for_drift(nv_sig)

        # Open tag stream ONCE per run (avoid open/close overhead per step)
        counter_server.start_tag_stream()
        run_t0 = time.perf_counter()

        try:
            for step_ind, tau_ns in enumerate(tau_ns_arr):
                if tb.safe_stop():
                    break

                seq_args = [
                    int(tau_ns),
                    int(polarization_ns),
                    int(readout_ns),
                    pol_vkey.name,
                    readout_vkey.name,
                    laser_power,
                ]
                seq_args_string = tb.encode_seq_args(seq_args)

                step_t0 = time.perf_counter()

                t0 = time.perf_counter()
                ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)
                t_stream_load = time.perf_counter() - t0

                if step_ind == 0 and run_ind == 0:
                    print(f"  Sequence period: {ret_vals} ns  (t1.py loaded)")

                t0 = time.perf_counter()
                counter_server.clear_buffer()
                t_clear = time.perf_counter() - t0

                t0 = time.perf_counter()
                pulsegen_server.stream_start(int(num_reps))
                t_stream_start = time.perf_counter() - t0

                # read_counter_summed (counter.py setting 212):
                #   transfers only [ref_total, sig_total] — 2 ints regardless of num_reps.
                #   This is the key throughput optimisation vs read_counter_separate_gates.
                new_counts = counter_server.read_counter_summed(int(num_reps))

                step_wall = time.perf_counter() - step_t0
                t_read    = step_wall - t_stream_load - t_clear - t_stream_start
                step_wall_times[run_ind, step_ind] = step_wall

                if step_ind == 0 and run_ind == 0:
                    print(f"  [diag] read_counter_summed returned: {new_counts}")

                ref_counts[run_ind, step_ind] = int(new_counts[0])
                sig_counts[run_ind, step_ind] = int(new_counts[1])

                ref_val = ref_counts[run_ind, step_ind]
                sig_val = sig_counts[run_ind, step_ind]
                norm_val = sig_val / ref_val if ref_val > 0 else float("nan")
                exp_s       = step_expected_s[step_ind]
                overhead_ms = (step_wall - exp_s) * 1e3

                _print_detail = (run_ind == 0) or (step_ind % 10 == 0)
                timing_str = (
                    f"  [load={t_stream_load*1e3:.0f} clr={t_clear*1e3:.0f} "
                    f"start={t_stream_start*1e3:.0f} read={t_read*1e3:.0f}ms]"
                    if _print_detail else ""
                )
                print(
                    f"tau={int(tau_ns):>8d} ns | "
                    f"ref={int(ref_val)}, sig={int(sig_val)}, norm={norm_val:.4f} | "
                    f"wall={step_wall:.2f}s exp={exp_s:.2f}s ovhd={overhead_ms:+.0f}ms"
                    + timing_str,
                    flush=True,
                )

                # Welford running mean for live plot
                if do_plot and np.isfinite(norm_val):
                    n = valid_run_count[step_ind]
                    if n == 0:
                        norm_running[step_ind]    = norm_val
                        norm_sq_running[step_ind] = norm_val ** 2
                    else:
                        norm_running[step_ind]    += (norm_val - norm_running[step_ind]) / (n + 1)
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

        # Draw once per run — avoids num_steps × plt.pause (each ~20–50 ms)
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
            line_norm.set_data(tau_ns_arr, norm_running)
            line_ste_hi.set_data(tau_ns_arr, norm_running + ste_arr)
            line_ste_lo.set_data(tau_ns_arr, norm_running - ste_arr)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

    # --- Final processing ---
    proc_arrays = _process_t1_counts(
        sig_counts, ref_counts,
        int(num_reps), int(readout_ns), norm_mode,
    )

    # Fit T1
    fit_result = _fit_t1(tau_ns_arr, proc_arrays["norm"], proc_arrays["norm_ste"])
    if fit_result["success"]:
        T1_ms = fit_result["T1_ms"]
        T1_ste_us = fit_result.get("T1_ste_us", float("nan"))
        print(f"\nT1 = {T1_ms:.3f} ms  (±{T1_ste_us:.1f} µs  1-sigma)")
        if do_plot and fig is not None:
            tau_fine = np.logspace(
                np.log10(max(tau_ns_arr[0], 1)),
                np.log10(tau_ns_arr[-1]),
                300,
            )
            fit_curve = _t1_model(
                tau_fine,
                fit_result["amplitude"],
                fit_result["T1_ns"],
                fit_result["offset"],
            )
            ax.plot(tau_fine, fit_curve, "r--", label=f"T1 = {T1_ms:.3f} ms")
            ax.legend()
            plt.pause(0.1)
    else:
        print(f"\nT1 fit failed: {fit_result['message']}")

    # --- Timing summary ---
    total_wall_s     = np.nansum(run_wall_times)
    total_expected_s = step_expected_s.sum() * num_runs
    total_overhead_s = total_wall_s - total_expected_s
    per_step_ms      = (
        total_overhead_s / (num_steps * num_runs) * 1e3
        if num_steps * num_runs > 0 else float("nan")
    )
    efficiency_pct = (
        total_expected_s / total_wall_s * 100 if total_wall_s > 0 else 0.0
    )
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print(f"  Total wall time      : {total_wall_s:.1f} s  ({total_wall_s / 60:.2f} min)")
    print(f"  Total expected (HW)  : {total_expected_s:.1f} s  ({total_expected_s / 60:.2f} min)")
    print(f"  Total overhead       : {total_overhead_s:.1f} s  ({total_overhead_s / 60:.2f} min)")
    print(f"  Avg overhead / step  : {per_step_ms:.0f} ms")
    print(f"  HW efficiency        : {efficiency_pct:.1f}%")
    print("=" * 70 + "\n")

    # --- Package results ---
    raw_data = {
        "timestamp":             timestamp,
        "nv_sig":                nv_sig,
        "num_reps":              int(num_reps),
        "num_runs":              int(num_runs),
        "polarization_ns":       int(polarization_ns),
        "readout_ns":            int(readout_ns),
        "tau_ns_list":           tau_ns_arr.tolist(),
        "log_scale":             log_scale,
        "sig_counts":            sig_counts.tolist(),
        "ref_counts":            ref_counts.tolist(),
        "step_wall_times_s":     step_wall_times.tolist(),
        "step_expected_times_s": step_expected_s.tolist(),
        "run_wall_times_s":      run_wall_times.tolist(),
    }

    proc_data = {
        "tau_ns_list":   tau_ns_arr.tolist(),
        "sig_kcps":      proc_arrays["sig_kcps"].tolist(),
        "ref_kcps":      proc_arrays["ref_kcps"].tolist(),
        "norm":          proc_arrays["norm"].tolist(),
        "norm_ste":      proc_arrays["norm_ste"].tolist(),
        "num_valid_runs": proc_arrays["num_valid_runs"].tolist(),
        "t1_fit":        fit_result,
    }

    if do_save:
        save_timestamp = dm.get_time_stamp()
        file_path = dm.get_file_path(__file__, save_timestamp, getattr(nv_sig, "name", "nv"))
        dm.save_raw_data(raw_data, file_path)
        if fig is not None:
            dm.save_figure(fig, file_path)
        print(f"Saved data to {file_path}")

    tb.reset_cfm()
    return raw_data, proc_data


if __name__ == "__main__":
    pass
