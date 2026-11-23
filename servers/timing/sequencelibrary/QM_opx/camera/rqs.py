# -*- coding: utf-8 -*-
"""
Widefield randomized quantum sensing (RQS)

One step is one shot of one acquisition row of the RQS plan: the preparation pi/2,
the row's refocusing pi pulses at the times computed by the RQS toolkit, and the
analysis pi/2. With phase cycling, every row is also played with the analysis pi/2
shifted by 180 deg. All timings come from the files written by
rqs_opx_export_with_pi2 (the <bundle>_opx_timing folder).

Test field (optional): kHz tones on analog output 1, one config element per tone
(ao_rqs_tone_0, ao_rqs_tone_1, ...). The tone [frequency_hz, amp_v, phase_deg] is
amp_v * cos(2 pi f t - phase). In the RQS convention B(t) = p cos(2 pi f t) +
q sin(2 pi f t), p and q are then proportional to cos(phase) and sin(phase):
phase 0 is a pure cosine, 90 deg a pure sine. Two modes, chosen by phase_locked:
- Phase-locked: every shot restarts the tones, and t is measured from the sensing
  start (the end of the preparation pi/2), so every shot sees the same phase.
- Free-running: the tones are never restarted. They follow the OPX clock as if they
  had been running since the program started, and t is the OPX time since then.
  Each shot then catches the tones at a different phase. The OPX records when each
  shot's tones start (stream SHOT_TIME_STREAM, in ns); the sensing starts
  TONE_LEAD_NS plus the pi/2 length later, and that is the shot's start time for
  the free-running decoder. This also works with no tones, e.g. for an external
  signal, as long as its phase follows the OPX clock.

The waits between pulses apply to the microwave elements only, so that the tones
keep playing through the sequence; every other element waits at the final align.

Adapted from spin_echo.py on November 3rd, 2026; test-field tones added on
November 5th, 2026

@author: schand
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

# Rotation axis of each pulse, as the IQ phase passed to the seq_utils pulse macros
AXIS_DEG = {"+X": 0, "+Y": 90, "-X": 180, "-Y": 270}
# Subtracted from every wait if the simulation shows a fixed extra delay per pulse
WAIT_CORRECTION_CC = 10
# Subtracted from the last wait if the analysis pi/2 alone starts late (its phase is chosen in real time)
ANALYSIS_CORRECTION_CC = 4

# Test-field tones on analog output 1
TONE_ELEMENT = "ao_rqs_tone_{}"  # config element of tone k
TONE_LEAD_NS = 1000  # each tone starts this long before the preparation pi/2, so the low-pass filter has settled
TONE_TAIL_NS = 500  # and ends this long after the analysis pi/2
TONE_DELAY_NS = 0  # how much later the tones reach the NVs than the microwave does (filter, cables); from the phase check
TONE_PHASE_OFFSET_DEG = 0  # added to every tone's phase if rqs_sequence_check.py reports a constant phase error
TONE_FRAME_SIGN = 1  # -1 if rqs_sequence_check.py reports mirrored tone phases
MW_ELEMENTS = None  # elements the microwave macros play on (None: every config element named after the signal generator)
SHOT_TIME_STREAM = "rqs_shot_times"  # free-running: OPX time (ns since the program started) at which each shot's tones start
_DEBUG_START_DELAY_CC = 0  # set by rqs_sequence_check.py to test the phase reset


def load_rqs_rows(timing_dir, rows):
    """Waits (clock cycles) and pulse phases of the given acquisition rows.

    The waits of rows[j] are waits[offsets[j]:offsets[j + 1]]: one before each pi
    pulse, then one before the analysis pi/2. analysis_inds[j] is the analysis
    phase of rows[j] in units of 90 deg.
    """
    timing_dir = Path(timing_dir)
    acquisition = pd.read_csv(timing_dir / "opx_ready_acquisition.csv")
    gates = pd.read_csv(timing_dir / "opx_full_microwave_sequence.csv")
    axes = gates["nominal_rotation_axis"].astype(str).str.replace("−", "-").str.strip()
    operation = gates["operation"].astype(str)
    prep_axes = set(axes[operation.str.startswith("PREPARE")])
    pi_axes = set(axes[operation == "REFOCUS"])
    if len(prep_axes) != 1 or len(pi_axes) != 1:
        raise ValueError("Expected one preparation axis and one pi-pulse axis for all rows")
    is_analysis = operation.str.startswith("MEASURE")
    analysis_axis = dict(zip(gates["row_id"][is_analysis], axes[is_analysis]))

    waits, offsets, analysis_inds = [], [0], []
    for row in rows:
        entry = acquisition.iloc[row]
        row_waits = [int(el) - WAIT_CORRECTION_CC for el in str(entry["waits_cc"]).split(";")]
        row_waits[-1] -= ANALYSIS_CORRECTION_CC
        if len(row_waits) != int(entry["pulse_count"]) + 1:
            raise ValueError(f"Row {entry['row_id']}: expected one wait per pi pulse, plus one")
        waits.extend(row_waits)
        offsets.append(len(waits))
        analysis_inds.append(AXIS_DEG[analysis_axis[entry["row_id"]]] // 90)
    if np.any(np.less(waits, 4)):
        raise RuntimeError("Wait shorter than 4 clock cycles")
    prep_deg = AXIS_DEG[prep_axes.pop()]
    pi_deg = AXIS_DEG[pi_axes.pop()]
    return waits, offsets, analysis_inds, prep_deg, pi_deg


def load_sensing_timing(timing_dir):
    """Sensing time and pi/2 length (ns) that the converter used; the same for every row."""
    acquisition = pd.read_csv(Path(timing_dir) / "opx_ready_acquisition.csv")
    sensing = set(acquisition["sensing_duration_ns"].astype(int))
    if len(sensing) != 1:
        raise ValueError("Expected one sensing time for all rows")
    return sensing.pop(), int(acquisition["pi2_duration_ns"].iloc[0])


def get_mw_elements(uwave_ind_list):
    """Elements the microwave pulse macros play on: the switch and the I/Q elements of each signal generator."""
    if MW_ELEMENTS is not None:
        return list(MW_ELEMENTS)
    if not uwave_ind_list:
        raise ValueError("RQS needs a microwave signal generator (uwave_ind_list is empty)")
    config_module = common.get_config_module()
    sig_gens = config_module.config["Microwaves"]["VirtualSigGens"]
    names = [sig_gens[int(ind)]["physical_name"] for ind in uwave_ind_list]
    elements = [
        element
        for element in config_module.opx_config["elements"]
        if any(re.search(rf"{re.escape(name)}(_|$)", element) for name in names)
    ]
    if not elements:
        raise RuntimeError(f"No OPX elements found for {names}: set rqs.MW_ELEMENTS")
    return elements


def tone_settings(tones, pi2_ns, phase_locked=True):
    """(element, frequency in Hz, amplitude scale, frame in turns) of each tone [frequency_hz, amp_v, phase_deg].

    Phase-locked: the frame set at the start of each shot's tone. Free-running: the frame set
    once when the program starts.
    """
    if not tones:
        return []
    if TONE_LEAD_NS % 4 or TONE_TAIL_NS % 4:
        raise ValueError("TONE_LEAD_NS and TONE_TAIL_NS must be multiples of 4 ns")
    if abs(TONE_DELAY_NS) > TONE_LEAD_NS / 2:
        raise ValueError("TONE_DELAY_NS is too large for TONE_LEAD_NS: increase the lead")
    opx_config = common.get_config_module().opx_config
    cw_volts = opx_config["waveforms"]["cw"]["sample"]
    total = sum(abs(float(amp)) for _, amp, _ in tones)
    if total > 0.5:
        raise ValueError(f"The tones add up to {total:.3f} V at their peak; the OPX output allows 0.5 V")
    settings = []
    for k, (freq, amp, phase) in enumerate(tones):
        element = TONE_ELEMENT.format(k)
        if element not in opx_config["elements"]:
            raise KeyError(f"No element {element} in the OPX config: add the RQS tone elements to the config file")
        freq = int(round(float(freq)))
        # Phase of the OPX output at t = 0, so that the field reaches the NVs at the set phase
        turns = -(float(phase) + TONE_PHASE_OFFSET_DEG) / 360 + freq * TONE_DELAY_NS * 1e-9
        if phase_locked:
            # t = 0 is the sensing start, TONE_LEAD_NS plus the preparation pi/2 after the tone starts
            turns -= freq * (TONE_LEAD_NS + pi2_ns) * 1e-9
        settings.append((element, freq, float(amp) / cw_volts, (TONE_FRAME_SIGN * turns) % 1.0))
    return settings


def macro_analysis_pulse(uwave_ind_list, phase_ind):
    """Analysis pi/2 at phase 90 * phase_ind deg (phase_ind: QUA int from 0 to 3)."""
    with qua.if_(phase_ind == 0):
        seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
    with qua.elif_(phase_ind == 1):
        seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=90)
    with qua.elif_(phase_ind == 2):
        seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=180)
    with qua.else_():
        seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=270)


def get_seq(base_scc_seq_args, step_vals, timing_dir, rows, tones, phase_locked, num_reps=1):
    """tones: list of [frequency_hz, amp_v, phase_deg] on analog output 1; empty for no test field.
    phase_locked: True restarts the tones every shot; False lets them run freely on the OPX
    clock and records each shot's time (see the module docstring).

    amp_v is the amplitude into 50 ohm at the OPX output, before any attenuator.
    """
    buffer = seq_utils.get_widefield_operation_buffer()
    waits, offsets, analysis_inds, prep_deg, pi_deg = load_rqs_rows(timing_dir, rows)
    sensing_ns, pi2_ns = load_sensing_timing(timing_dir)
    phase_locked = bool(phase_locked)
    tone_list = tone_settings(tones, pi2_ns, phase_locked)
    clock_element = TONE_ELEMENT.format(0)  # free-running without tones: a silent 16 ns play marks the shot time
    if not phase_locked and not tone_list:
        if clock_element not in common.get_config_module().opx_config["elements"]:
            raise KeyError(f"No element {clock_element} in the OPX config: add the RQS tone elements to the config file")
    lead_cc = TONE_LEAD_NS // 4
    # Covers the preparation pi/2, the sensing window and the analysis pi/2, plus lead and tail
    tone_ns = TONE_LEAD_NS + 2 * pi2_ns + sensing_ns + TONE_TAIL_NS + max(-TONE_DELAY_NS, 0)
    tone_cc = int(np.ceil(tone_ns / 4))
    reset_phase = getattr(qua, "reset_if_phase", None) or qua.reset_phase

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)
        waits_qua = qua.declare(int, value=waits)
        offsets_qua = qua.declare(int, value=offsets)
        analysis_inds_qua = qua.declare(int, value=analysis_inds)
        row = qua.declare(int)
        wait_ind = qua.declare(int)
        last_ind = qua.declare(int)
        phase_ind = qua.declare(int)
        for element, freq, _, turns in tone_list:
            qua.update_frequency(element, freq)
            if not phase_locked:
                # Free-running: the phase is set once and then follows the OPX clock
                qua.reset_frame(element)
                qua.frame_rotation_2pi(turns, element)
        if _DEBUG_START_DELAY_CC:
            qua.wait(_DEBUG_START_DELAY_CC)

        def uwave_macro_sig(uwave_ind_list, step_val):
            mw_elements = get_mw_elements(uwave_ind_list)
            # step = 2 * (index in rows) + phase; phase 1 = analysis pi/2 shifted by 180 deg
            qua.assign(row, step_val >> 1)
            qua.assign(last_ind, offsets_qua[row + 1] - 1)
            qua.assign(phase_ind, (analysis_inds_qua[row] + (step_val & 1) * 2) & 3)
            qua.align()
            # Test field: phase-locked restarts every tone at the same phase; free-running keeps the
            # phase running and records when the shot's tones start
            for k, (element, _, scale, turns) in enumerate(tone_list):
                if phase_locked:
                    qua.reset_frame(element)
                    reset_phase(element)
                    qua.frame_rotation_2pi(turns, element)
                    qua.play("cw" * qua.amp(scale), element, duration=tone_cc)
                else:
                    stream = SHOT_TIME_STREAM if k == 0 else None
                    qua.play("cw" * qua.amp(scale), element, duration=tone_cc, timestamp_stream=stream)
            if not phase_locked and not tone_list:
                qua.play("off", clock_element, duration=4, timestamp_stream=SHOT_TIME_STREAM)
            qua.wait(lead_cc, *mw_elements)
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=prep_deg)
            with qua.for_(wait_ind, offsets_qua[row], wait_ind < last_ind, wait_ind + 1):
                qua.wait(waits_qua[wait_ind], *mw_elements)
                seq_utils.macro_pi_pulse(uwave_ind_list, phase=pi_deg)
            qua.wait(waits_qua[last_ind], *mw_elements)
            macro_analysis_pulse(uwave_ind_list, phase_ind)
            qua.align()  # every other element waits until the microwave and the tones are done
            qua.wait(buffer)

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                step_val,
                num_reps,
            )
    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    # Simulate row 0 in both phases, then row 1, with a 100 kHz test tone, and compare the
    # pi-pulse starts in the plot with the converter's times printed here
    repo_root = Path(__file__).resolve().parents[5]
    timing_dir = repo_root / "RQS_toolkit" / "RQS_toolkit" / "outputs" / "rqs_phase_locked_opx_timing"
    rows = [0, 1]
    tones = [[100e3, 0.1, 0.0]]
    phase_locked = True

    try:
        acquisition = pd.read_csv(Path(timing_dir) / "opx_ready_acquisition.csv")
        for row in rows:
            print(
                acquisition["row_id"][row],
                "pi starts (ns after the first pi/2 starts):",
                acquisition["pi_starts_ns_from_first_pi2"][row],
            )
        seq, seq_ret_vals = get_seq(
            [
                [[109.062, 107.003], [110.183, 105.383]],
                [1000, 1000],
                [1.0, 1.0],
                [[73.477, 72.33], [74.328, 70.992]],
                [168, 184],
                [1.0, 1.0],
                [False, False, False],
                [1],
            ],
            [0, 1, 2],
            timing_dir,
            rows,
            tones,
            phase_locked,
            1,
        )

        sim_config = SimulationConfig(duration=int(600e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
