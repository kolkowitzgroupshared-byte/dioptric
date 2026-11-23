# -*- coding: utf-8 -*-
"""
Two-pulse population check: does a second pi pulse act on the spin?

Steps (step_val): 0 = no pulse, 1 = one pi pulse, 2 + k = two pi pulses with gaps_ns[k]
between them (end of the first to start of the second). This tests populations, not
coherence: the spin starts in m_s = 0, one pi pulse flips it, a second flips it back, and the
gap does not matter because T1 is far longer than any gap. If the second pulse never acts,
every pair reads like a single pi pulse.

Every step lasts the same time (the pulses are followed by a wait that fills up to the
longest pair), so the readout comes at the same moment after polarization in every step.

iq_phase: IQ phase (deg) of the pi pulses, which needs IQ mode on; None plays them in CW mode.

Created on November 18th, 2026
"""

from qm import qua

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence, rqs


def get_seq(base_scc_seq_args, step_vals, gaps_ns, iq_phase=None, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    uwave_ind_list = base_scc_seq_args[-1]
    sig_gen = common.get_config_module().config["Microwaves"]["VirtualSigGens"][uwave_ind_list[0]]
    pi_cc = seq_utils.convert_ns_to_cc(sig_gen["pi_pulse"])
    gaps_cc = [max(4, int(round(float(g) / 4))) for g in gaps_ns]
    total_cc = 2 * pi_cc + max(gaps_cc) + 4  # longest pair, plus a minimal tail
    # wait after the pulses, per step, so that every step lasts total_cc
    tails_cc = [total_cc, total_cc - pi_cc] + [total_cc - 2 * pi_cc - g for g in gaps_cc]

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)
        gaps_qua = qua.declare(int, value=gaps_cc)
        tails_qua = qua.declare(int, value=tails_cc)
        gap = qua.declare(int)
        tail = qua.declare(int)

        def pi_pulse(uwave_ind_list):
            if iq_phase is None:
                seq_utils.macro_pi_pulse(uwave_ind_list)
            else:
                seq_utils.macro_pi_pulse(uwave_ind_list, phase=int(round(float(iq_phase))) % 360)

        def uwave_macro_sig(uwave_ind_list, step_val):
            mw_elements = rqs.get_mw_elements(uwave_ind_list)
            qua.align()
            qua.assign(tail, tails_qua[step_val])
            with qua.if_(step_val == 1):
                pi_pulse(uwave_ind_list)
            with qua.elif_(step_val >= 2):
                qua.assign(gap, gaps_qua[step_val - 2])
                pi_pulse(uwave_ind_list)
                qua.wait(gap, *mw_elements)
                pi_pulse(uwave_ind_list)
            qua.wait(tail, *mw_elements)
            qua.align()
            qua.wait(buffer)

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(base_scc_seq_args, [uwave_macro_sig], step_val, num_reps)

    return seq, []
