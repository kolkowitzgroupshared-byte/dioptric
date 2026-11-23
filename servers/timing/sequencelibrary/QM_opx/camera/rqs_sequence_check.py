# -*- coding: utf-8 -*-
"""
Checks for the RQS sequence (rqs.py). Run from the repo root, like the __main__
blocks of the sequence files.

Part 1, no OPX needed: reads the converter files and checks that the waits rebuild
    the converter's pulse times.
Part 2: simulates a Y row and an X row, each in both analysis phases, finds every
    microwave pulse in the simulated I/Q outputs, and compares its start time,
    length and IQ phase with the converter's values. With test-field tones, it also
    fits each tone's amplitude and phase at the sensing start and checks that the
    tones cover the whole sensing window. It then simulates one shot again, started
    1236 ns later: phase-locked (PHASE_LOCKED = True), the tone phases must not move;
    free-running, they must move with the recorded shot time, as a continuously
    running signal would.
Part 3, optional: compiles a block of rows to check that it fits on the OPX.

Created on November 3rd, 2026; tone checks added on November 5th, 2026

@author: schand
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx.camera import rqs

# Converter output inside this repo (<dioptric>/RQS_toolkit/RQS_toolkit/outputs); change it if yours is elsewhere
REPO_ROOT = Path(__file__).resolve().parents[5]
TIMING_DIR = REPO_ROOT / "RQS_toolkit" / "RQS_toolkit" / "outputs" / "rqs_16k7_phase_locked_opx_timing"
BASE_SCC_SEQ_ARGS = [
    [[109.062, 107.003], [110.183, 105.383]],
    [1000, 1000],
    [1.0, 1.0],
    [[73.477, 72.33], [74.328, 70.992]],
    [168, 184],
    [1.0, 1.0],
    [False, False, False],
    [1],
]
MW_ELEMENT = None  # IQ element with mixInputs, if your config has one (None = find it)
MW_I_PORT, MW_Q_PORT = None, None  # otherwise the analog ports of I and Q (None = find them in the simulation)
MW_CONTROLLER = "con1"
SWITCH_PORT = 10  # digital port of the microwave switch of the generator in use (None = skip the switch report)
STEPS_TO_SIMULATE = (0, 1, 2, 3)  # (0,) simulates a single shot (~6 min) instead of four
SIM_DURATION_US = 400  # one shot per simulation (each shot is followed by a ~50 ms readout)
COMPILE_BLOCK = 0  # Part 3: number of rows to compile, e.g. 1000 (0 = skip)
# Test-field tones to simulate, as passed to rqs.get_seq: [frequency_hz, amp_v, phase_deg] ([] = none)
TONES = [] #[[100e3, 0.1, 0.0]]
STEPS_TO_SIMULATE = (0,)
PHASE_LOCKED = True  # False checks the free-running mode
RESET_TEST_DELAY_CC = 309  # the delay test starts the shot this much later (1236 ns)


def expected_times(acquisition, row):
    """Converter's pi-pulse starts and analysis start, in ns after the first pi/2 starts."""
    starts = str(acquisition["pi_starts_ns_from_first_pi2"].iloc[row])
    pi_starts = [int(el) for el in starts.split(";")] if starts not in ("", "nan") else []
    return pi_starts, int(acquisition["final_pi2_start_ns_from_first_pi2"].iloc[row])


def converter_lengths(acquisition):
    """pi/2 and pi lengths (ns) that the converter assumed."""
    return int(acquisition["pi2_duration_ns"].iloc[0]), int(acquisition["pi_duration_ns"].iloc[0])


def check_converter_files(acquisition):
    """Part 1: rebuild every row's pulse times from the waits that rqs.py will play."""
    rows = list(range(len(acquisition)))
    waits, offsets, analysis_inds, prep_deg, pi_deg = rqs.load_rqs_rows(TIMING_DIR, rows)
    pi2_ns, pi_ns = converter_lengths(acquisition)
    worst = 0
    for j, row in enumerate(rows):
        row_waits = 4 * (np.array(waits[offsets[j] : offsets[j + 1]]) + rqs.WAIT_CORRECTION_CC)
        row_waits[-1] += 4 * rqs.ANALYSIS_CORRECTION_CC
        t, pi_starts = pi2_ns, []
        for wait in row_waits[:-1]:
            t += wait
            pi_starts.append(t)
            t += pi_ns
        expected_pi, expected_final = expected_times(acquisition, row)
        errors = np.abs(np.array(pi_starts) - expected_pi) if pi_starts else np.zeros(0)
        worst = max(worst, errors.max(initial=0), abs(t + row_waits[-1] - expected_final))
    by_basis = {}
    for basis, ind in zip(acquisition["basis"], analysis_inds):
        by_basis.setdefault(basis, set()).add(90 * ind)
    print("Part 1: converter files")
    print(f"  {len(rows)} rows, {len(waits)} waits, shortest wait {min(waits)} clock cycles; "
          f"pulse lengths assumed by the converter: pi/2 {pi2_ns} ns, pi {pi_ns} ns")
    print(f"  phases: preparation {prep_deg} deg, pi pulses {pi_deg} deg, analysis "
          + ", ".join(f"{basis} rows {sorted(phases)} deg" for basis, phases in sorted(by_basis.items())))
    print(f"  largest difference between the rebuilt and the converter's pulse times: {worst} ns")
    if TONES:
        total = sum(abs(amp) for _, amp, _ in TONES)
        print("  tones: " + ", ".join(f"{freq / 1e3:g} kHz at {amp:.3f} V, phase {phase:g} deg" for freq, amp, phase in TONES)
              + f"; together {total:.3f} V at their peak (OPX limit 0.5 V)")
        print(f"  each tone starts {rqs.TONE_LEAD_NS} ns before the preparation pi/2 and stops {rqs.TONE_TAIL_NS} ns "
              f"after the analysis pi/2; TONE_DELAY_NS = {rqs.TONE_DELAY_NS}")
    print(f"  mode: {'phase-locked (tones restart every shot)' if PHASE_LOCKED else 'free-running (shot times recorded)'}")
    print()
    return min(waits)


def circular_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def find_pulses(i_samples, q_samples, intermediate_frequency):
    """Every microwave pulse: start (ns), length (ns) and IQ phase (deg), from 1 ns samples."""
    i_samples, q_samples = np.asarray(i_samples, float), np.asarray(q_samples, float)
    z = (i_samples - np.median(i_samples)) + 1j * (q_samples - np.median(q_samples))  # without DC offsets
    z = z * np.exp(-2j * np.pi * intermediate_frequency * 1e-9 * np.arange(len(z)))
    envelope = np.abs(z)
    baseline = np.median(envelope)
    on = envelope > baseline + 0.5 * (envelope.max() - baseline)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    pulses = []
    for start, stop in zip(edges[::2], edges[1::2]):
        phase = np.degrees(np.angle(z[start:stop].mean())) % 360
        pulses.append(dict(start=int(start), length=int(stop - start), phase=phase))
    return pulses


def segments(values, digital=False):
    """(start, length) of every interval where a channel is on, from 1 ns samples."""
    values = np.asarray(values, float)
    if digital:
        on = values > 0.5
    else:
        deviation = np.abs(values - np.median(values))
        if deviation.max() < 1e-3:
            return []
        on = deviation > 0.5 * deviation.max()
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def print_switch_windows(shot, windows):
    """Each microwave pulse of a shot, with the switch window closest to it. In the simulated OPX
    outputs every window should start iq_delay after its I/Q pulse (that delay matches the
    generator's modulation delay physically) and last as long as the pulse."""
    pulses = ([("prep pi/2", shot["prep"])] + [(f"pi {k + 1}", p) for k, p in enumerate(shot["pis"])]
              + [("analysis pi/2", shot["analysis"])])
    print("    switch windows next to each I/Q pulse (simulated OPX outputs):")
    used = set()
    for name, p in pulses:
        candidates = [(abs(w[0] - p["start"]), k, w) for k, w in enumerate(windows)
                      if abs(w[0] - p["start"]) < 1000 and k not in used]
        if candidates:
            _, k, w = min(candidates)
            used.add(k)
            print(f"      {name:14s}: I/Q at {p['start']:>8d} ns for {p['length']:4d} ns; "
                  f"switch at {w[0] - p['start']:+5d} ns for {w[1]:4d} ns")
        else:
            print(f"      {name:14s}: I/Q at {p['start']:>8d} ns for {p['length']:4d} ns; NO switch window within 1 us")
    if len(windows) > len(used):
        print(f"      {len(windows) - len(used)} other switch window(s) in the simulated time (for example the reference shot)")


def microwave_pulses(analog, digital, i_port=None, q_port=None, intermediate_frequency=0, exclude=()):
    """Microwave pulses from the simulated outputs, and which channels they came from.

    I/Q: the two analog ports with the most 16-400 ns pulses (unless given), apart from the
    ports in exclude (the test-field tones). Without them, the digital channel with the most
    such pulses gives the timing, and the phases are not measured.
    """
    is_pulse = lambda length: 16 <= length <= 400  # microwave pulses; AOD tones and laser pulses fall outside
    if i_port is None:
        scores = {port: sum(is_pulse(length) for _, length in segments(values))
                  for port, values in analog.items() if port not in exclude}
        found = sorted(port for port, score in scores.items() if score)
        found = sorted(sorted(found, key=lambda port: -scores[port])[:2], key=lambda port: int(port))
        i_port, q_port = (found + [None, None])[:2]
    if i_port is not None:
        i_samples = analog[str(i_port)]
        q_samples = analog[str(q_port)] if q_port is not None else np.zeros_like(i_samples)
        pulses = find_pulses(i_samples, q_samples, intermediate_frequency)
        if q_port is None:
            for pulse in pulses:
                pulse["phase"] = None
            return pulses, f"analog port {i_port} only (phases not measured)"
        return pulses, f"analog ports {i_port} (I) and {q_port} (Q)"
    best, best_count = None, 0
    for port, values in digital.items():
        count = sum(is_pulse(length) for _, length in segments(values, True))
        if count > best_count:
            best, best_count = port, count
    if best is None:
        return [], "no channel with microwave-like pulses"
    pulses = [dict(start=a, length=b, phase=None) for a, b in segments(digital[best], True) if is_pulse(b)]
    return pulses, f"digital port {best} (phases not measured)"


def port_outputs(values_by_key):
    """Samples by port number. The simulator can list a port twice (e.g. "1-9" and "9"):
    keep the copy that carries a signal."""
    ports = {}
    for key, values in values_by_key.items():
        port, values = str(key).split("-")[-1], np.asarray(values, float)
        if port not in ports or np.ptp(values) > np.ptp(ports[port]):
            ports[port] = values
    return ports


def print_elements(opx_config):
    """Each element's output ports, to identify the microwave channels."""
    for name, el in opx_config["elements"].items():
        if "mixInputs" in el:
            analog = f"I {el['mixInputs']['I'][-1]}, Q {el['mixInputs']['Q'][-1]}"
        elif "singleInput" in el:
            analog = str(el["singleInput"]["port"][-1])
        else:
            analog = "-"
        digital = ", ".join(str(v["port"][-1]) for v in el.get("digitalInputs", {}).values()) or "-"
        print(f"    {name:40s} analog {analog:10s} digital {digital}")


def group_shots(pulses):
    """Split the pulses into shots: a pi/2, then pi pulses, then a pi/2."""
    pulses = [pulse for pulse in pulses if 16 <= pulse["length"] <= 400]
    short = min((pulse["length"] for pulse in pulses), default=0)
    shots, current = [], None
    for pulse in pulses:
        is_pi2 = pulse["length"] < 1.5 * short  # a pi pulse lasts about twice a pi/2
        if current is None:
            if is_pi2:
                current = dict(prep=pulse, pis=[])
        elif is_pi2:
            current["analysis"] = pulse
            shots.append(current)
            current = None
        else:
            current["pis"].append(pulse)
    return shots


def compare(shots, steps, rows, acquisition, min_wait_cc=None):
    """Part 2 report: timing, lengths and phases of each simulated shot, and what to change."""
    _, _, analysis_inds, prep_deg, pi_deg = rqs.load_rqs_rows(TIMING_DIR, rows)
    pi2_ns, pi_ns = converter_lengths(acquisition)
    print(f"Part 2: simulation ({sum(shot is not None for shot in shots)} of {len(steps)} shots found)")
    pi_errors, analysis_errors, pi2_lengths, pi_lengths, mirrored = [], [], [], [], None
    for shot, step in zip(shots, steps):
        row, minus = rows[step >> 1], step & 1
        label = f"  {acquisition['row_id'].iloc[row]:>8s} {'minus' if minus else 'plus':5s}: "
        if shot is None:
            print(label + "no complete shot in the simulated window")
            continue
        expected_pi, expected_final = expected_times(acquisition, row)
        if len(shot["pis"]) != len(expected_pi):
            print(label + f"{len(shot['pis'])} pi pulses found, {len(expected_pi)} expected")
            continue
        t0 = shot["prep"]["start"]
        errors = np.array([pulse["start"] - t0 for pulse in shot["pis"]], int) - np.array(expected_pi, int)
        analysis_error = shot["analysis"]["start"] - t0 - expected_final
        pi_errors += [(k + 1, e) for k, e in enumerate(errors)]
        analysis_errors.append((len(errors), analysis_error))
        pi2_lengths += [shot["prep"]["length"], shot["analysis"]["length"]]
        pi_lengths += [pulse["length"] for pulse in shot["pis"]]
        span = f"{errors.min():+d} to {errors.max():+d} ns" if errors.size else "no pi pulses"
        text = f"pi timing error {span}, analysis {analysis_error:+d} ns"
        if shot["prep"]["phase"] is None:
            text += "; phases not measured"
        else:
            pi_phase = np.degrees(np.angle(np.mean([np.exp(1j * np.radians(p["phase"])) for p in shot["pis"]]))) if shot["pis"] else 0
            prep_rel = (shot["prep"]["phase"] - pi_phase) % 360
            analysis_rel = (shot["analysis"]["phase"] - pi_phase) % 360
            if mirrored is None:
                mirrored = circular_difference(prep_rel, -(prep_deg - pi_deg)) < 10
            sign = -1 if mirrored else 1
            expected_analysis = (sign * (90 * analysis_inds[step >> 1] + 180 * minus - pi_deg)) % 360
            text += (f"; phase vs pi pulses: preparation {round(prep_rel) % 360} deg, "
                     f"analysis {round(analysis_rel) % 360} deg (expected {round(expected_analysis) % 360})")
        print(label + text)
        print("             pi errors (ns): " + " ".join(f"{e:+d}" for e in errors))
    if mirrored:
        print("  The IQ phases come out mirrored (the preparation reads 270 instead of 90 deg): the measured\n"
              "  sign convention is opposite to the nominal one. Calibration A sets the physical axes anyway.")
    if not pi_errors:
        return

    measured_pi2, measured_pi = int(np.median(pi2_lengths)), int(np.median(pi_lengths))
    k, e = np.array(pi_errors, float).T
    slope, offset = np.polyfit(k, e, 1) if len(set(k)) > 1 else (0.0, float(e.mean()))
    print(f"  Pulse lengths: pi/2 {measured_pi2} ns, pi {measured_pi} ns (the converter assumed {pi2_ns} and {pi_ns} ns)")
    print(f"  Timing drift: {slope:+.1f} ns per pi pulse (offset {offset:+.1f} ns)")
    if abs(measured_pi - pi_ns) > 1 or abs(measured_pi2 - pi2_ns) > 1:
        print(f"  -> Rerun the converter with pi_duration_ns={measured_pi}, pi2_duration_ns={measured_pi2}, copy the new\n"
              f"     timing folder over and run this check again. About {slope - (measured_pi - pi_ns):+.0f} ns per pulse\n"
              f"     would remain: that part is overhead on the OPX.")
        return
    if np.abs(e).max() <= 1 and max(abs(a) for _, a in analysis_errors) <= 1:
        print("  Timing matches the converter.")
        return
    wait_corr = int(round(slope / 4))
    analysis_corr = int(round(np.mean([a - 4 * wait_corr * (n + 1) for n, a in analysis_errors]) / 4))
    if min_wait_cc is not None and wait_corr > min_wait_cc - 4:
        need = measured_pi + 4 * wait_corr + 20
        print(f"  -> A {wait_corr}-cycle correction would push the shortest wait ({min_wait_cc} cycles) below the OPX minimum.\n"
              f"     Re-export the bundle with min_gap_s of at least {need} ns, rerun the converter, then set\n"
              f"     rqs.WAIT_CORRECTION_CC += {wait_corr} and rqs.ANALYSIS_CORRECTION_CC += {analysis_corr}.")
    else:
        changes = [f"rqs.{name} += {value}" for name, value in
                   (("WAIT_CORRECTION_CC", wait_corr), ("ANALYSIS_CORRECTION_CC", analysis_corr)) if value]
        print("  -> Set " + " and ".join(changes) + ", then run this check again." if changes else
              "  -> Remaining errors are below one clock cycle (4 ns); nothing to correct.")


def signed_difference(a, b):
    return (a - b + 180) % 360 - 180


def tone_fit(samples, start, stop, tones):
    """Amplitude (V) and phase (deg) of each tone between samples start and stop, in the convention
    amp * cos(2 pi f (t - start) - phase); a constant absorbs any DC offset."""
    y = np.asarray(samples, float)[start:stop]
    x = np.arange(len(y)) * 1e-9
    columns = [np.ones_like(x)]
    for freq, _, _ in tones:
        columns += [np.cos(2 * np.pi * freq * x), np.sin(2 * np.pi * freq * x)]
    coef = np.linalg.lstsq(np.array(columns).T, y, rcond=None)[0]
    return [(float(np.hypot(c, s)), float(np.degrees(np.arctan2(s, c)) % 360)) for c, s in zip(coef[1::2], coef[2::2])]


def analyze_tones(samples, shot, tones):
    """Tone amplitudes and phases at the sensing start (the end of the preparation pi/2), and how
    long before the sensing window the tones start and after the analysis pi/2 they stop."""
    if not tones or shot is None:
        return None
    if samples is None:
        return dict(fits=None, lead=None, tail=None)
    samples = np.asarray(samples, float)
    sensing_start = shot["prep"]["start"] + shot["prep"]["length"]
    sensing_stop = shot["analysis"]["start"]
    on = np.flatnonzero(np.abs(samples) > 1e-6)
    if on.size == 0:
        return dict(fits=None, lead=None, tail=None)
    lead = int(sensing_start - on[0])
    tail = int(on[-1] + 1 - (shot["analysis"]["start"] + shot["analysis"]["length"]))
    covers = on[0] < sensing_start and on[-1] >= sensing_stop
    fits = tone_fit(samples, sensing_start, sensing_stop, tones) if covers else None
    return dict(fits=fits, lead=lead, tail=tail)


def shot_times(sim):
    """Shot times (ns) that the free-running mode recorded in a simulated job, or None."""
    try:
        handles = sim.result_handles
        handles.wait_for_all_values()
        handle = handles.get(rqs.SHOT_TIME_STREAM)
        values = None if handle is None else np.asarray(handle.fetch_all())
    except Exception as exc:
        print(f"  could not read the shot times: {exc}")
        return None
    if values is None or values.size == 0:
        return None
    if values.dtype.names:
        values = values["value"] if "value" in values.dtype.names else values[values.dtype.names[0]]
    return np.asarray(values, dtype=np.int64).ravel()


def report_tones(results, steps, rows, acquisition, delay_test, times=None):
    """Part 2 report for the tones: amplitude, phase at the sensing start, coverage, and the
    delay test (phase-locked: phases must not move; free-running: they must follow the shot time)."""
    pi2_ns = converter_lengths(acquisition)[0]
    offset, delay = rqs.TONE_PHASE_OFFSET_DEG, rqs.TONE_DELAY_NS
    mode = "phase-locked" if PHASE_LOCKED else "free-running"
    print(f"\nPart 2b: test-field tones, {mode} (sensing start = end of the preparation pi/2)")
    if PHASE_LOCKED:
        expected = [(phase + offset - 360 * freq * delay * 1e-9) % 360 for freq, _, phase in TONES]
        # What the phases would be if frame rotations turned the other way on the OPX
        mirrored = [(-(phase + offset) + 360 * freq * delay * 1e-9 - 720 * freq * (rqs.TONE_LEAD_NS + pi2_ns) * 1e-9) % 360
                    for freq, _, phase in TONES]
    errors, mirror_errors, ratios, leads, tails = [], [], [], [], []
    for ind, (result, step) in enumerate(zip(results, steps)):
        row, minus = rows[step >> 1], step & 1
        label = f"  {acquisition['row_id'].iloc[row]:>8s} {'minus' if minus else 'plus':5s}: "
        if result is None:
            print(label + "no complete shot in the simulated window")
            continue
        if result["lead"] is None:
            print(label + "no tone on the tone port")
            continue
        leads.append(result["lead"])
        tails.append(result["tail"])
        coverage = f"on from {result['lead']} ns before the sensing window to {result['tail']} ns after the analysis pi/2"
        if result["fits"] is None:
            print(label + "the tones do not cover the sensing window: " + coverage)
            continue
        parts = []
        for k, ((freq, amp, _), (fit_amp, fit_phase)) in enumerate(zip(TONES, result["fits"])):
            ratios.append(fit_amp / amp if amp else np.nan)
            text = f"{freq / 1e3:g} kHz {fit_amp:.4f} V (set {amp:.4f}), phase {fit_phase:.1f} deg"
            if PHASE_LOCKED:
                errors.append(signed_difference(fit_phase, expected[k]))
                mirror_errors.append(signed_difference(fit_phase, mirrored[k]))
                text += f" (expected {expected[k]:.1f})"
            parts.append(text)
        print(label + "; ".join(parts))
        print(" " * 17 + coverage)
        if not PHASE_LOCKED:
            shot_time = None if times is None or times[ind] is None else int(times[ind][0])
            print(" " * 17 + (f"shot time recorded: tones start at {shot_time} ns, sensing starts at "
                              f"{shot_time + rqs.TONE_LEAD_NS + pi2_ns} ns" if shot_time is not None
                              else "no shot time recorded"))

    if all(result is None for result in results):
        print("  -> No complete shot to check the tones on: see Part 2 above.")
        return
    if not leads:
        print("  -> No tone found on the tone port: check the tone elements in the config.")
        return
    if not ratios:
        print("  -> The tones stopped before the sensing window ended. Something in the pulse macros probably waits\n"
              "     for every element (an align without arguments), so the microwave waited for the tones. The tones\n"
              "     then need to be played differently: send me this output.")
        return
    if min(tails) < 0:
        print(f"  -> The tones stop up to {-min(tails)} ns before the analysis pi/2 ends. Apply the timing corrections\n"
              f"     above first; if that is not enough, increase rqs.TONE_TAIL_NS.")
    if PHASE_LOCKED:
        if max(np.abs(errors)) <= 1:
            print("  Tone phases at the sensing start match the set phases.")
        elif max(np.abs(mirror_errors)) <= 1:
            print(f"  -> The frame rotation turns the other way on this OPX: set rqs.TONE_FRAME_SIGN = {-rqs.TONE_FRAME_SIGN}"
                  f" and run this check again.")
        elif np.ptp(errors) <= 1:
            print(f"  -> Every tone is off by the same {np.mean(errors):+.1f} deg: set rqs.TONE_PHASE_OFFSET_DEG = "
                  f"{offset - np.mean(errors):.1f} and run this check again.")
        else:
            print("  -> The tone phases are off by different amounts: send me this output.")
    elif times is not None and any(t is None for t in times):
        print("  -> Some shots have no recorded time: the free-running decoder needs every shot's time.")
    if np.nanmax(np.abs(np.array(ratios) - 1)) > 0.02:
        print(f"  Tone amplitudes are {np.nanmean(ratios):.3f} times the set values in the simulation.")
    if delay_test is None:
        return
    (fits, time), (delayed_fits, delayed_time) = delay_test
    delay_ns = 4 * RESET_TEST_DELAY_CC
    shifted = [signed_difference(b, a) for (_, a), (_, b) in zip(fits, delayed_fits)]
    if PHASE_LOCKED:
        if max(np.abs(shifted)) <= 1:
            print(f"  Phase lock: starting the shot {delay_ns} ns later left the tone phases unchanged.")
        else:
            print(f"  -> Starting the shot {delay_ns} ns later moved the tone phases by "
                  + ", ".join(f"{s:+.1f}" for s in shifted)
                  + " deg: the phase reset is not working, so the tones would not be phase-locked. Send me this output.")
        return
    # Free-running: a later shot sees the signal later, so each phase moves by -360 f delay
    moved_ns = None if time is None or delayed_time is None else int(delayed_time[0] - time[0])
    expected_shift = [signed_difference(-360 * freq * delay_ns * 1e-9, 0) for freq, _, _ in TONES]
    good_phase = max(abs(signed_difference(s, e)) for s, e in zip(shifted, expected_shift)) <= 1
    good_time = moved_ns == delay_ns
    detail = (f"phases moved by " + ", ".join(f"{s:+.1f}" for s in shifted) + " deg (expected "
              + ", ".join(f"{e:+.1f}" for e in expected_shift) + f"), recorded shot time moved by {moved_ns} ns")
    if good_phase and good_time:
        print(f"  Free-running: starting the shot {delay_ns} ns later {detail}. The tones follow the OPX clock\n"
              f"  and the recorded times track them, as the free-running decoder needs.")
    else:
        print(f"  -> Starting the shot {delay_ns} ns later: {detail}. "
              + ("The tones don't follow the OPX clock. " if not good_phase else "")
              + ("The recorded times don't match. " if not good_time else "")
              + "Send me this output.")


if __name__ == "__main__":
    if not (TIMING_DIR / "opx_ready_acquisition.csv").is_file():
        raise FileNotFoundError(
            f"No converter output in {TIMING_DIR}. Copy the bundle folder and its _opx_timing folder "
            "from the computer that ran the notebook, or set TIMING_DIR to where they are."
        )
    acquisition = pd.read_csv(TIMING_DIR / "opx_ready_acquisition.csv")
    min_wait_cc = check_converter_files(acquisition)

    started = time.time()
    step = lambda message: print(f"[{time.time() - started:6.1f} s] {message}", flush=True)
    step("Loading the OPX config")
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 2e3  # shorter simulation, as in spin_echo.py

    element = MW_ELEMENT
    if element is None:
        candidates = [name for name, el in opx_config["elements"].items() if "mixInputs" in el]
        element = candidates[0] if len(candidates) == 1 else None
    if element is not None:
        element_config = opx_config["elements"][element]
        i_spec, q_spec = element_config["mixInputs"]["I"], element_config["mixInputs"]["Q"]
        con, i_port, q_port = i_spec[0], i_spec[-1], q_spec[-1]
        intermediate_frequency = element_config.get("intermediate_frequency", 0) or 0
        step(f"Microwave element {element}: I on {con} port {i_port}, Q on port {q_port}, IF {intermediate_frequency / 1e6:g} MHz")
    else:
        con, i_port, q_port, intermediate_frequency = MW_CONTROLLER, MW_I_PORT, MW_Q_PORT, 0
        step("No IQ element with mixInputs; the microwave channels will be found in the simulated outputs")
        print_elements(opx_config)

    mw_elements = rqs.get_mw_elements(BASE_SCC_SEQ_ARGS[-1])
    step("The waits apply to these microwave elements: " + ", ".join(mw_elements))
    tone_port = None
    if TONES:
        tone_elements = [rqs.TONE_ELEMENT.format(k) for k in range(len(TONES))]
        missing = [name for name in tone_elements if name not in opx_config["elements"]]
        if missing:
            raise KeyError(f"No {', '.join(missing)} in the OPX config: add the RQS tone elements to the config file")
        tone_port = str(opx_config["elements"][tone_elements[0]]["singleInput"]["port"][-1])
        step(f"Tones on analog port {tone_port}: " + ", ".join(tone_elements))

    step("Connecting to the OPX")
    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        # Part 2: the first Y row and the first X row, each in both analysis phases.
        # Each step is simulated on its own, since every shot is followed by a long readout.
        first_x = int(np.flatnonzero(acquisition["basis"].to_numpy() == "X")[0])
        rows = [0, first_x]
        steps = list(STEPS_TO_SIMULATE)
        exclude = (tone_port,) if tone_port else ()

        def simulate_step(step_val):
            """Microwave shot, tone analysis and recorded shot times of one simulated step."""
            seq, seq_ret_vals = rqs.get_seq(BASE_SCC_SEQ_ARGS, [step_val], TIMING_DIR, rows, TONES, PHASE_LOCKED, 1)
            sim = opx.simulate(seq, SimulationConfig(duration=int(SIM_DURATION_US * 1e3 / 4)))
            samples = sim.get_simulated_samples()
            outputs = getattr(samples, con)
            analog, digital = port_outputs(outputs.analog), port_outputs(outputs.digital)
            pulses, source = microwave_pulses(analog, digital, i_port, q_port, intermediate_frequency, exclude)
            found = group_shots(pulses)
            step(f"  microwave pulses from {source}: {len(pulses)} pulses, {len(found)} complete shots")
            shot = found[0] if found else None
            if SWITCH_PORT is not None and shot is not None:
                if str(SWITCH_PORT) in digital:
                    print_switch_windows(shot, segments(digital[str(SWITCH_PORT)], True))
                else:
                    print(f"    no simulated output on digital port {SWITCH_PORT}: set SWITCH_PORT")
            tone_result = analyze_tones(analog.get(tone_port) if tone_port else None, shot, TONES)
            return samples, shot, tone_result, (None if PHASE_LOCKED else shot_times(sim))

        shots, tone_results, times, first_samples = [], [], [], None
        for step_val in steps:
            step(f"Simulating step {step_val} of {len(steps) - 1} ({SIM_DURATION_US} us)")
            samples, shot, tone_result, shot_time = simulate_step(step_val)
            if first_samples is None:
                first_samples = samples
            shots.append(shot)
            tone_results.append(tone_result)
            times.append(shot_time)
        if not any(shot is not None for shot in shots):
            print("  No complete shot: increase SIM_DURATION_US (the shot starts after the polarization pulses).")
        if not PHASE_LOCKED and not TONES:
            print("  Free-running shot times (ns): " + ", ".join(str(None if t is None else int(t[0])) for t in times))

        # Delay test: the same shot started later. Phase-locked, the tone phases must not move;
        # free-running, they must follow the shot time.
        delay_test = None
        if TONES and tone_results[0] is not None and tone_results[0]["fits"] is not None:
            step(f"Simulating step 0 again, started {4 * RESET_TEST_DELAY_CC} ns later (delay test)")
            rqs._DEBUG_START_DELAY_CC = RESET_TEST_DELAY_CC
            try:
                _, _, delayed, delayed_time = simulate_step(0)
            finally:
                rqs._DEBUG_START_DELAY_CC = 0
            if delayed is not None and delayed["fits"] is not None:
                delay_test = ((tone_results[0]["fits"], times[0]), (delayed["fits"], delayed_time))

        compare(shots, steps, rows, acquisition, min_wait_cc)
        if TONES:
            report_tones(tone_results, steps, rows, acquisition, delay_test, times)
        getattr(first_samples, con).plot()
        plt.show(block=True)

        # Part 3: does a block of rows fit on the OPX?
        if COMPILE_BLOCK:
            block = list(range(min(COMPILE_BLOCK, len(acquisition))))
            block_steps = [2 * j + phase for j, row in enumerate(block)
                           for _ in range(int(acquisition["shots"].iloc[row])) for phase in (0, 1)]
            start = time.time()
            seq, seq_ret_vals = rqs.get_seq(BASE_SCC_SEQ_ARGS, block_steps, TIMING_DIR, block, TONES, PHASE_LOCKED, 1)
            opx.compile(seq)
            print(f"\nPart 3: {len(block)} rows ({len(block_steps)} steps) compiled in {time.time() - start:.1f} s")

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
