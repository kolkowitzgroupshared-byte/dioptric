# -*- coding: utf-8 -*-
"""Fixed-duration multisine RQS QUA sequence with variable X pulse count.

The host gives padded arrays of free-evolution gaps in QUA clock cycles.
All pi rotations are about X. Zero-padded slots are ignored using n_pi.
The ordinary camera SCC readout includes alternating references.
"""
from qm import qua
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, gaps_cc, n_pi_list, basis_list, parity_list, num_reps=2):
    n=len(n_pi_list)
    if not n or len(basis_list)!=n or len(parity_list)!=n:
        raise ValueError("Step counts disagree")
    if len(gaps_cc)<2 or any(len(col)!=n for col in gaps_cc):
        raise ValueError("Incorrect padded gap vectors")
    limit=len(gaps_cc)-1
    if any(type(v) is not int or v<0 or v>limit for v in n_pi_list):
        raise ValueError("Pulse count out of range")
    if any(v not in (0,1) for v in parity_list):
        raise ValueError('Even flag must be 0/1')
    if any(v not in (0,1) for v in basis_list):
        raise ValueError("Basis must be X=0 or Y=1")
    if any(type(v) is not int or v<4 for col in gaps_cc for v in col):
        raise ValueError("Every wait needs at least four OPX cycles")
    if num_reps<2 or num_reps%2:
        raise ValueError("Even repetitions needed for dual-rail reference")
    buffer=seq_utils.get_widefield_operation_buffer()
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        gaps=tuple(qua.declare(int) for _ in range(limit+1))
        count=qua.declare(int)
        basis=qua.declare(int)
        even_flag=qua.declare(int)
        args=tuple(gaps_cc)+(n_pi_list,basis_list,parity_list)

        def sensing(inds, params):
            g=params[:limit+1]
            num=params[limit+1]
            final_basis=params[limit+2]
            is_even=params[limit+3]
            qua.align()
            seq_utils.macro_pi_on_2_pulse(inds,phase=0)
            qua.wait(g[0])
            for j in range(limit):
                with qua.if_(num>j):
                    seq_utils.macro_pi_pulse(inds,phase=0)
                    qua.wait(g[j+1])
            with qua.if_(final_basis==0):
                with qua.if_(is_even==1):
                    seq_utils.macro_pi_on_2_pulse(inds,phase=180)
                with qua.else_():
                    seq_utils.macro_pi_on_2_pulse(inds,phase=0)
            with qua.else_():
                with qua.if_(is_even==1):
                    seq_utils.macro_pi_on_2_pulse(inds,phase=270)
                with qua.else_():
                    seq_utils.macro_pi_on_2_pulse(inds,phase=90)
            qua.wait(buffer)

        with qua.for_each_(gaps+(count,basis,even_flag),args):
            base_scc_sequence.macro(
                base_scc_seq_args,[sensing],gaps+(count,basis,even_flag),
                num_reps,reference=True)
    return seq,[]

