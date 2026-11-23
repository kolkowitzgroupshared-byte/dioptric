# -*- coding: utf-8 -*-
"""
Widefield spin echo with a phase-locked test tone, to calibrate the AC field the loop
produces at each NV.

Each shot is a Hahn echo lasting one period T = 1/f of the tone: the preparation pi/2,
a pi pulse centered half a period later, and the analysis pi/2 one period after the
sensing start (the end of the preparation pi/2). The tone comes from OPX analog
output 1, with the same elements and phase convention as rqs.py, and is restarted
every shot. A tone amplitude * cos(2 pi f t - theta) gives the echo phase
    phase = gamma * B * sin(theta) * 2 T / pi,    gamma = 2 pi x 28.025 Hz/nT,
and the echo signal follows cos(phase).

At low bias field, choose f so that the echo (1/f) lands on a 13C revival: here ~60 us,
so f ~ 16.7 kHz (see the routine's header).

Each step is one (tone phase, tone amplitude) pair. The amplitudes and frame
rotations of all steps are precomputed and indexed by the step value.

Created on November 9th, 2026

@author: schand
"""

import numpy as np
from qm import qua

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence, rqs


def echo_timing(freq_hz, uwave_ind_list):
    """pi and pi/2 lengths (ns), the tone period (ns), and the wait (clock cycles) on each
    side of the pi pulse, so that the analysis pi/2 starts one period after the sensing start."""
    sig_gens = common.get_config_module().config["Microwaves"]["VirtualSigGens"]
    sig_gen = sig_gens[int(uwave_ind_list[0])]
    pi_ns = int(sig_gen["pi_pulse"])
    pi2_ns = int(sig_gen["pi_on_2_pulse"])
    period_ns = 1e9 / float(freq_hz)
    wait_cc = int(round((period_ns - pi_ns) / 2 / 4)) - rqs.WAIT_CORRECTION_CC
    if wait_cc < 4:
        raise ValueError(f"{freq_hz / 1e3:g} kHz is too fast for a {pi_ns} ns pi pulse")
    return pi_ns, pi2_ns, period_ns, wait_cc


def step_settings(freq_hz, phases_deg, amps_v, pi2_ns):
    """Tone element, frequency, and per-step amplitude scales and frame rotations.

    Steps run over all phases, and for each phase over all amplitudes:
    step = phase_index * len(amps_v) + amplitude_index.
    """
    peak = max(abs(float(amp)) for amp in amps_v)
    cw_volts = common.get_config_module().opx_config["waveforms"]["cw"]["sample"]
    scales, turns = [], []
    element = freq = None
    for phase in phases_deg:
        element, freq, _, phase_turns = rqs.tone_settings([[freq_hz, peak, phase]], pi2_ns, True)[0]
        for amp in amps_v:
            scales.append(float(amp) / cw_volts)
            turns.append(float(phase_turns))
    return element, freq, scales, turns


def get_seq(base_scc_seq_args, step_vals, freq_hz, phases_deg, amps_v, pulse_phases_deg=None, num_reps=1):
    """step_vals: step indices, see step_settings. amps_v: tone amplitudes (V into 50 ohm at
    the OPX output). phases_deg: tone phases at the sensing start, as in rqs.py.
    pulse_phases_deg: None for the plain echo (no IQ phases, as before), or the IQ phases of
    (first pi/2, pi, last pi/2) in degrees, e.g. (0, 90, 0) for x,y,x; needs IQ mode on."""
    buffer = seq_utils.get_widefield_operation_buffer()
    uwave_ind_list = base_scc_seq_args[-1]
    pi_ns, pi2_ns, period_ns, wait_cc = echo_timing(freq_hz, uwave_ind_list)
    element, freq, scales, turns = step_settings(freq_hz, phases_deg, amps_v, pi2_ns)
    lead_cc = rqs.TONE_LEAD_NS // 4
    tone_ns = rqs.TONE_LEAD_NS + 2 * pi2_ns + period_ns + rqs.TONE_TAIL_NS + max(-rqs.TONE_DELAY_NS, 0)
    tone_cc = int(np.ceil(tone_ns / 4))
    reset_phase = getattr(qua, "reset_if_phase", None) or qua.reset_phase

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)
        scales_qua = qua.declare(qua.fixed, value=scales)
        turns_qua = qua.declare(qua.fixed, value=turns)
        qua.update_frequency(element, freq)

        def uwave_macro_sig(uwave_ind_list, step_val):
            mw_elements = rqs.get_mw_elements(uwave_ind_list)
            qua.align()
            # Tone restarted at this step's phase, then playing through the echo
            qua.reset_frame(element)
            reset_phase(element)
            qua.frame_rotation_2pi(turns_qua[step_val], element)
            qua.play("cw" * qua.amp(scales_qua[step_val]), element, duration=tone_cc)
            # The waits apply to the microwave elements only, so the tone keeps playing
            qua.wait(lead_cc, *mw_elements)
            if pulse_phases_deg is None:
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list)
                qua.wait(wait_cc, *mw_elements)
                seq_utils.macro_pi_pulse(uwave_ind_list)
                qua.wait(wait_cc, *mw_elements)
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list)
            else:
                first, middle, last = (int(round(float(p))) % 360 for p in pulse_phases_deg)
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=first)
                qua.wait(wait_cc, *mw_elements)
                seq_utils.macro_pi_pulse(uwave_ind_list, phase=middle)
                qua.wait(wait_cc, *mw_elements)
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=last)
            qua.align()  # every other element waits until the echo and the tone are done
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
