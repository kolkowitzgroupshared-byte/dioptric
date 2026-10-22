# -*- coding: utf-8 -*-
"""Random-timing XY8-8 spectroscopy for 13C, with SCC dual-rail reference.

Each step contains nine *edge-to-edge* wait intervals in OPX clock cycles:
pi/2 - w0 - pi_X - w1 - pi_Y - ... - pi_X - w8 - pi/2.
All patterns have eight pi pulses and therefore the same nominal pulse-error
order. The pulse CENTER times are calculated and saved by the host routine.
This is randomized-filter (RQS-inspired) dephasing spectroscopy, NOT the
noncommuting pooled-spectroscopy algorithm from Kannan's presentation.

The host supplies the random schedules and interleaved XY8 comparison steps.
"""
from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


PHASES_XY8_DEGREES = (0, 90, 0, 90, 90, 0, 90, 0)
NUM_GAPS = 9


def get_seq(base_scc_seq_args, wait_tables_cc, num_reps=1):
    """Build a single QUA program containing the ordered host-provided steps.

    wait_tables_cc: nine same-length lists, column i = wait after pulse i.
    Every wait must be >= 4 clock cycles (16 ns); QUA waits are in 4-ns cc.
    """
    if len(wait_tables_cc) != NUM_GAPS:
        raise ValueError("Need nine wait vectors for eight XY8 pi pulses.")
    lengths = [len(v) for v in wait_tables_cc]
    if not lengths or lengths[0] == 0 or any(n != lengths[0] for n in lengths):
        raise ValueError("Wait vectors must be nonempty and equally long.")
    if any((type(w) is not int or w < 4) for vec in wait_tables_cc for w in vec):
        raise ValueError("Every wait must be a Python integer >= 4 cc.")
    if not isinstance(num_reps, int) or num_reps < 1:
        raise ValueError("num_reps must be a positive int.")

    buffer = seq_utils.get_widefield_operation_buffer()
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        waits = tuple(qua.declare(int) for _ in range(NUM_GAPS))
        uwave_ind_list = base_scc_seq_args[-1]

        def uwave_macro_sig(indices, step_waits):
            qua.align()
            seq_utils.macro_pi_on_2_pulse(indices, phase=0)
            qua.wait(step_waits[0])
            for j, phase in enumerate(PHASES_XY8_DEGREES):
                seq_utils.macro_pi_pulse(indices, phase=phase)
                qua.wait(step_waits[j + 1])
            seq_utils.macro_pi_on_2_pulse(indices, phase=0)
            qua.wait(buffer)

        with qua.for_each_(waits, tuple(wait_tables_cc)):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                waits,
                num_reps,
                reference=True,
            )

    return seq, []
