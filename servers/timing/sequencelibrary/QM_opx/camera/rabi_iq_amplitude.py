"""Purcell fixed-duration IQ amplitude Rabi (dual-rail SCC).

Commanded amplitudes are absolute OPX I/Q vector voltages 0..0.5 V,
not a QUA multiplier 0..0.5. config/purcell.py defines
i_phase=0.5*cos(phase), q_phase=0.5*sin(phase); QUA scale = volts / 0.5.

This intentionally bypasses seq_utils.macro_pi_pulse(...,amp=...),
whose amp keyword currently does not reach qua.play().
No OPX configuration, existing experiment, or hardware setting is altered.
"""
import math

from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

FULL_SCALE_V = 0.5


def validate_amplitudes(step_vals):
    amplitudes = [float(x) for x in step_vals]
    if len(amplitudes) < 2:
        raise ValueError("Provide at least two IQ amplitudes")
    if not all(math.isfinite(x) and 0 <= x <= FULL_SCALE_V for x in amplitudes):
        raise ValueError("IQ amplitude must be between 0 and 0.5 V")
    return amplitudes


def get_seq(base_scc_seq_args, step_vals, pulse_duration_ns,
            phase_deg=0, num_reps=2):
    """SCC signal + alternating dual-rail references at each amplitude."""
    amplitudes = validate_amplitudes(step_vals)
    if num_reps < 2 or num_reps % 2:
        raise ValueError("Dual-rail SCC requires an even number of reps >= 2")
    if pulse_duration_ns < 16 or int(pulse_duration_ns) != pulse_duration_ns or pulse_duration_ns % 4:
        raise ValueError("Fixed pulse duration must be an integer multiple of 4 ns >=16 ns")
    if int(phase_deg) != phase_deg or not 0 <= int(phase_deg) < 360:
        raise ValueError("IQ phase must be an integer 0..359 degrees")
    phase_deg = int(phase_deg)
    # Purcell IQ phase-grid waveforms are already defined for all integer phases.
    chain_list = base_scc_seq_args[-1]
    if isinstance(chain_list, int):
        chain_list = [chain_list]
    if len(chain_list) != 1:
        raise ValueError("Select exactly one microwave chain for IQ amplitude Rabi")
    duration_cc = int(pulse_duration_ns // 4)
    scales = [float(round(a / FULL_SCALE_V, 9)) for a in amplitudes]
    pulse = f"pi_pulse_{phase_deg}"

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        scale = qua.declare(qua.fixed)

        def uwave_macro_sig(uwave_ind_list, step_val):
            # Keep the generator TTL (unscaled) and scale BOTH DAC quadratures.
            # The zero-amplitude control skips the TTL drive to avoid LO leakage.
            for uwave_ind in uwave_ind_list:
                i_el = seq_utils.get_sig_gen_i_element(uwave_ind)
                q_el = seq_utils.get_sig_gen_q_element(uwave_ind)
                gate_el = seq_utils.get_sig_gen_element(uwave_ind)
                qua.align(i_el, q_el, gate_el)
                with qua.if_(step_val > 0):
                    qua.play(pulse * qua.amp(step_val), i_el,
                             duration=duration_cc)
                    qua.play(pulse * qua.amp(step_val), q_el,
                             duration=duration_cc)
                    qua.play("pi_pulse", gate_el, duration=duration_cc)
                with qua.else_():
                    qua.wait(duration_cc, i_el, q_el, gate_el)
                qua.align(i_el, q_el, gate_el)

        with qua.for_each_(scale, scales):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                scale,
                num_reps=num_reps,
                reference=True,
            )

    return seq, []
