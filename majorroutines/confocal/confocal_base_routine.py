# -*- coding: utf-8 -*-
"""
majorroutines/confocal/confocal_base_routine.py

Unified confocal “step-sweep” runner for SWAB pulse streamer + SWAB tagger counter.

Assumptions / contract:
- step_fn(step_ind) returns (seq_file, seq_args_string)
- Pulse streamer server supports: stream_load(seq_file, seq_args_string) -> [period_ns, ...]
                              and: stream_start(num_reps)
- Counter server supports:
    start_tag_stream(apd_indices?) / stop_tag_stream()
    clear_buffer()
    read_counter_modulo_gates(num_gates_per_rep, num_to_read)
  where read_counter_modulo_gates(num_exps, 1) returns ONE aggregated vector of length num_exps:
    [gate0_sum_over_all_reps, gate1_sum_over_all_reps, ...]
"""

import time
import traceback
from math import isclose
from random import shuffle

import numpy as np

from utils import positioning as pos
from utils import tool_belt as tb


# -------------------------
# basic type hygiene
# -------------------------
def _as_pos_int(name, val):
    try:
        iv = int(val)
    except Exception:
        raise TypeError(f"{name} must be an integer, got {type(val).__name__}: {val!r}")
    if iv < 0:
        raise ValueError(f"{name} must be >= 0, got {iv}")
    if isinstance(val, float) and not isclose(val, iv, rel_tol=0.0, abs_tol=1e-9):
        raise TypeError(f"{name} must be an integer, got non-integer float: {val!r}")
    return iv


def _to_int_list(name, x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_as_pos_int(name, v) for v in x]
    return [_as_pos_int(name, x)]


# -------------------------
# tagger compatibility
# -------------------------
def _start_tag_stream_compat(counter, apd_indices):
    # Some servers require apd_indices, some ignore it.
    try:
        counter.start_tag_stream(apd_indices)
    except TypeError:
        counter.start_tag_stream()


def _stop_tag_stream_compat(counter):
    try:
        counter.stop_tag_stream()
    except Exception:
        # Some variants use stop_stream / stop; keep broad to avoid masking root cause.
        try:
            counter.stop_stream()
        except Exception:
            pass


def _normalize_modulo_return(raw, num_exps: int) -> np.ndarray:
    """
    Expect ONE sample with num_exps counts.
    Typical shapes:
      - [[sig, ref]]
      - [sig, ref]
      - np.array variants
    Returns shape: (num_exps,)
    """
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    arr = np.array(raw)

    if arr.ndim == 2:
        # keep first row (we asked for num_to_read=1)
        arr = arr[0]
    elif arr.ndim != 1:
        raise RuntimeError(f"Unexpected modulo_gates return shape: {arr.shape}")

    if arr.size != num_exps:
        raise RuntimeError(f"Expected {num_exps} counts, got size={arr.size}, shape={arr.shape}")
    return arr.astype(np.int64, copy=False)


# -------------------------
# pulse streamer helper
# -------------------------
def _run_seq_blocking_swab(pulsegen, seq_file: str, seq_args_string: str, num_reps: int) -> int:
    """
    SWAB pattern: stream_load -> stream_start -> wait.
    Returns period_ns.
    """
    period_ns = int(pulsegen.stream_load(seq_file, seq_args_string)[0])
    pulsegen.stream_start(int(num_reps))

    # wait for completion so the tagger isn't mid-flight
    wait_s = (period_ns * 1e-9) * int(num_reps) + 0.05  # +margin
    # cap safety (prevents infinite sleeps if period is nonsense)
    wait_s = min(max(wait_s, 0.0), 60.0)
    time.sleep(wait_s)
    return period_ns


# -------------------------
# main routine
# -------------------------
def main(
    nv_sig,
    num_steps,
    num_reps,
    num_runs,
    *,
    step_fn,                   # (step_ind) -> (seq_file, seq_args_string)
    uwave_ind_list=(0,),
    uwave_freq_list=None,
    num_exps=2,
    apd_indices=(0,),
    load_iq=False,
    charge_prep_fn=None,
    shuffle_steps=True,
    per_step_pause_s=0.0,
    # allow future compatibility with older call sites:
    run_fn=None,               # optional: if you want to do something once per run
    stream_load_in_run_fn=False,
    **_ignored_kwargs,
) -> dict:
    """
    Returns dict with:
      counts shape: (num_exps, num_runs, num_steps) aggregated over num_reps
      step_ind_master_list: the (possibly shuffled) step order per run
    """

    num_steps = _as_pos_int("num_steps", num_steps)
    num_reps  = _as_pos_int("num_reps",  num_reps)
    num_runs  = _as_pos_int("num_runs",  num_runs)
    num_exps  = _as_pos_int("num_exps",  num_exps)

    uwave_ind_list = _to_int_list("uwave_ind", uwave_ind_list)
    apd_indices    = _to_int_list("apd_index", apd_indices)

    # ---------- init & positioning ----------
    tb.reset_cfm()
    pos.set_xyz_on_nv(nv_sig)

    pulsegen = tb.get_server_pulse_streamer()
    counter  = tb.get_server_counter()

    # ---------- microwave setup ----------
    sig_gens = []
    for uwave_ind in uwave_ind_list:
        vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
        uwave_power = vsg["uwave_power"]
        freq = uwave_freq_list[uwave_ind] if uwave_freq_list else vsg["frequency"]

        sg = tb.get_server_sig_gen(uwave_ind)
        if load_iq:
            sg.load_iq()
        sg.set_amp(uwave_power)
        sg.set_freq(freq)
        print(f"MW[{uwave_ind}]  freq: {freq} GHz,  power: {uwave_power} dBm")
        sig_gens.append(sg)

    # counts[exp, run, step] aggregated over num_reps
    counts = np.zeros((num_exps, num_runs, num_steps), dtype=np.int64)

    step_ind_master_list = [None] * num_runs
    crash_counter = [0] * num_runs
    crash_log = [[] for _ in range(num_runs)]  # list of (step_ind, error_str)

    step_ind_list = list(range(num_steps))

    tb.init_safe_stop()

    # ---------- run ----------
    try:
        for run_ind in range(num_runs):
            print(f"\n[Run {run_ind + 1}/{num_runs}]")
            if tb.safe_stop():
                break

            # optional: user charge prep
            if charge_prep_fn:
                try:
                    charge_prep_fn(nv_sig)
                except Exception:
                    crash_counter[run_ind] += 1
                    crash_log[run_ind].append(("charge_prep_fn", traceback.format_exc()))

            # optional: hook for older patterns (do nothing by default)
            if run_fn is not None:
                try:
                    run_fn(run_ind)
                except Exception:
                    crash_counter[run_ind] += 1
                    crash_log[run_ind].append(("run_fn", traceback.format_exc()))

            # turn microwaves on
            for sg in sig_gens:
                try:
                    sg.uwave_on()
                except Exception:
                    crash_counter[run_ind] += 1
                    crash_log[run_ind].append(("uwave_on", traceback.format_exc()))

            # choose step order
            if shuffle_steps:
                shuffle(step_ind_list)
            else:
                step_ind_list = list(range(num_steps))
            step_ind_master_list[run_ind] = step_ind_list.copy()

            # start tag stream once per run
            _start_tag_stream_compat(counter, apd_indices)

            try:
                for step_ind in step_ind_list:
                    if tb.safe_stop():
                        break

                    try:
                        seq_file, seq_args_string = step_fn(int(step_ind))
                        # print("SEQ ARGS STRING:", seq_args_string)

                        # IMPORTANT: clear buffer before each step
                        counter.clear_buffer()

                        if stream_load_in_run_fn:
                            # if some caller already did stream_load elsewhere, just start
                            pulsegen.stream_start(int(num_reps))
                            # still need a period estimate to sleep — safest: do a load anyway
                            # (but if you truly want to skip, you must provide your own wait)
                            period_ns = int(pulsegen.stream_load(seq_file, seq_args_string)[0])
                            time.sleep(min((period_ns * 1e-9) * int(num_reps) + 0.05, 60.0))
                        else:
                            _run_seq_blocking_swab(pulsegen, seq_file, seq_args_string, int(num_reps))

                        # Read ONE aggregated [gate0, gate1, ...] sample
                        raw = counter.read_counter_modulo_gates(int(num_exps), 1)
                        vec = _normalize_modulo_return(raw, int(num_exps))
                        counts[:, run_ind, step_ind] = vec

                        if per_step_pause_s > 0:
                            time.sleep(float(per_step_pause_s))

                    except Exception:
                        crash_counter[run_ind] += 1
                        crash_log[run_ind].append((int(step_ind), traceback.format_exc()))
                        # keep going to next step
                        continue

            finally:
                # always stop the tag stream for this run
                _stop_tag_stream_compat(counter)

            # microwaves off after each run
            for sg in sig_gens:
                try:
                    sg.uwave_off()
                except Exception:
                    crash_counter[run_ind] += 1
                    crash_log[run_ind].append(("uwave_off", traceback.format_exc()))

    except Exception:
        # catch unexpected outer failures
        print(traceback.format_exc())

    finally:
        # ensure MW is off even on crash
        for sg in sig_gens:
            try:
                sg.uwave_off()
            except Exception:
                pass
        try:
            _stop_tag_stream_compat(counter)
        except Exception:
            pass

    return {
        "nv_sig": nv_sig,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "num_exps": num_exps,
        "uwave_ind_list": uwave_ind_list,
        "uwave_freq_list": uwave_freq_list,
        "apd_indices": apd_indices,
        "counts": counts,  # (num_exps, num_runs, num_steps), aggregated over num_reps
        "counts-units": "photons",
        "step_ind_master_list": step_ind_master_list,
        "crash_counter": crash_counter,
        "crash_log": crash_log,
        "note": "counts are aggregated over num_reps via read_counter_modulo_gates(num_exps, 1)",
    }
