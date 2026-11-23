# -*- coding: utf-8 -*-
"""
Widefield XY4-N dynamical decoupling with a phase-locked test tone, to calibrate the AC
field the loop produces at each NV directly at an in-band frequency (e.g. 100 kHz).

Same as spin_echo_tone.py, with the single pi pulse replaced by n_pi pi pulses in XY4
order (phases 0, 90, 0, 90, ... as in xy.py), their centers half a tone period apart:
  (pi/2)_0 - 1/(4f) - pi - 1/(2f) - pi - ... - pi - 1/(4f) - (pi/2)_0
measured between pulse centers, inside a sensing window of n_pi / (2f) that runs from the
end of the preparation pi/2 to the start of the analysis pi/2 (the same window
convention as spin_echo_tone.py). The sign the NV accumulates phase with is then a
square wave at the tone frequency, so for a tone amplitude * cos(2 pi f t - theta)
    phase = gamma * B * cos(theta) * (2/pi) * n_pi / (2f),
and static offsets cancel (the sequence is balanced). Choosing the window as a whole
number of 13C precession periods also cancels the 13C field: at ~33.3 kHz, 100 kHz with
n_pi = 12 gives 60 us, two 13C periods.

Pulse lengths come from the config (VirtualSigGens). Each step is one (tone phase, tone
amplitude) pair, as in spin_echo_tone.py.

Created on November 15th, 2026
"""

import numpy as np
from qm import qua

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence, rqs
from servers.timing.sequencelibrary.QM_opx.camera.spin_echo_tone import step_settings

XY4_PHASES_DEG = (0, 90, 0, 90)


def xy_phases(n_pi):
    """XY4 phases repeated over n_pi pulses."""
    return [XY4_PHASES_DEG[k % 4] for k in range(int(n_pi))]


def xy_timing(freq_hz, n_pi, uwave_ind_list):
    """pi and pi/2 lengths (ns), sensing window (ns), and the waits (clock cycles):
    first gap, gap between pi pulses, last gap. The pulse centers sit half a tone period
    apart, within one clock cycle (an odd pi length in clock cycles splits the 4 ns
    remainder between the first and last gaps)."""
    sig_gens = common.get_config_module().config["Microwaves"]["VirtualSigGens"]
    sig_gen = sig_gens[int(uwave_ind_list[0])]
    pi_ns = int(sig_gen["pi_pulse"])
    pi2_ns = int(sig_gen["pi_on_2_pulse"])
    n_pi = int(n_pi)
    if n_pi < 1:
        raise ValueError("n_pi must be at least 1")
    half_period_ns = 1e9 / (2 * float(freq_hz))
    window_ns = n_pi * half_period_ns
    free_cc = int(round((window_ns - n_pi * pi_ns) / 4))
    mid_cc = int(round((half_period_ns - pi_ns) / 4))
    edge_sum_cc = free_cc - (n_pi - 1) * mid_cc
    first_cc = edge_sum_cc // 2
    last_cc = edge_sum_cc - first_cc
    correction = rqs.WAIT_CORRECTION_CC
    waits = (first_cc - correction, mid_cc - correction, last_cc - correction)
    if min(waits) < 4:
        raise ValueError(f"{freq_hz / 1e3:g} kHz is too fast for a {pi_ns} ns pi pulse")
    return pi_ns, pi2_ns, window_ns, waits


def get_seq(base_scc_seq_args, step_vals, freq_hz, phases_deg, amps_v, n_pi, num_reps=1):
    """step_vals: step indices (step = phase_index * len(amps_v) + amplitude_index).
    amps_v: tone amplitudes (V into 50 ohm at the OPX output). phases_deg: tone phases at
    the sensing start, as in rqs.py. n_pi: number of pi pulses (XY4 order)."""
    buffer = seq_utils.get_widefield_operation_buffer()
    uwave_ind_list = base_scc_seq_args[-1]
    pi_ns, pi2_ns, window_ns, (first_cc, mid_cc, last_cc) = xy_timing(freq_hz, n_pi, uwave_ind_list)
    element, freq, scales, turns = step_settings(freq_hz, phases_deg, amps_v, pi2_ns)
    phases = xy_phases(n_pi)
    lead_cc = rqs.TONE_LEAD_NS // 4
    tone_ns = rqs.TONE_LEAD_NS + 2 * pi2_ns + window_ns + rqs.TONE_TAIL_NS + max(-rqs.TONE_DELAY_NS, 0)
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
            # Tone restarted at this step's phase, then playing through the sequence
            qua.reset_frame(element)
            reset_phase(element)
            qua.frame_rotation_2pi(turns_qua[step_val], element)
            qua.play("cw" * qua.amp(scales_qua[step_val]), element, duration=tone_cc)
            # The waits apply to the microwave elements only, so the tone keeps playing
            qua.wait(lead_cc, *mw_elements)
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(first_cc, *mw_elements)
            for k, phase in enumerate(phases):
                seq_utils.macro_pi_pulse(uwave_ind_list, phase=phase)
                qua.wait(mid_cc if k < len(phases) - 1 else last_cc, *mw_elements)
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.align()  # every other element waits until the sequence and the tone are done
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
