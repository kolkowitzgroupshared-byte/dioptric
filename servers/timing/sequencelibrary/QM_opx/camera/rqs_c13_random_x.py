# -*- coding: utf-8 -*-
"""Random pi_X (and periodic XY8 control) for 13C with optional IQ projection.

The host specifies nine edge-to-edge gap vectors, one binary pulse-axis-mode
vector (1=all-X, 0=periodic XY8), and one readout-basis vector (0 or 90 deg).
All random interrogation pulses are strictly calibrated pi_X. QUA handles
SCC polarization/readout and alternating dual-rail reference as before.
"""
from qm import qua
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

AXES_XY8 = (0, 90, 0, 90, 90, 0, 90, 0)


def get_seq(base_scc_seq_args, waits_cc, pulse_modes, readout_phases, num_reps=1):
    if len(waits_cc) != 9:
        raise ValueError("Expected nine gap arrays")
    nsteps = len(pulse_modes)
    if nsteps == 0 or len(readout_phases) != nsteps:
        raise ValueError("Mismatched mode/readout arrays")
    if any(len(v) != nsteps for v in waits_cc):
        raise ValueError("All wait arrays must have nsteps entries")
    if any(type(v) is not int or v < 4 for row in waits_cc for v in row):
        raise ValueError("QUA gaps must be Python int >= 4 clock cycles")
    if any(v not in (0, 1) for v in pulse_modes):
        raise ValueError("Mode must be 0=XY8 or 1=pi_X")
    if any(v not in (0, 90) for v in readout_phases):
        raise ValueError("Readout phase must be 0 or 90 degrees")
    if not isinstance(num_reps, int) or num_reps < 1:
        raise ValueError("Invalid repetitions")

    buffer = seq_utils.get_widefield_operation_buffer()
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        gap_vars = tuple(qua.declare(int) for _ in range(9))
        mode = qua.declare(int)
        basis = qua.declare(int)
        vals = tuple(waits_cc) + (pulse_modes, readout_phases)
        uwave_ind_list = base_scc_seq_args[-1]

        def signal(inds, step_tuple):
            gaps = step_tuple[:9]
            axis_mode = step_tuple[9]
            readout = step_tuple[10]
            qua.align()
            seq_utils.macro_pi_on_2_pulse(inds, phase=0)
            qua.wait(gaps[0])
            for j, xy8_phase in enumerate(AXES_XY8):
                # At every step, the host selects all-X for random masks
                # or the standard XY8 phase cycle for periodic comparisons.
                with qua.if_(axis_mode == 1):
                    seq_utils.macro_pi_pulse(inds, phase=0)
                with qua.else_():
                    seq_utils.macro_pi_pulse(inds, phase=xy8_phase)
                qua.wait(gaps[j + 1])
            with qua.if_(readout == 0):
                seq_utils.macro_pi_on_2_pulse(inds, phase=0)
            with qua.else_():
                seq_utils.macro_pi_on_2_pulse(inds, phase=90)
            qua.wait(buffer)

        with qua.for_each_(gap_vars + (mode, basis), vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [signal],
                gap_vars + (mode, basis),
                num_reps,
                reference=True,
            )
    return seq, []
