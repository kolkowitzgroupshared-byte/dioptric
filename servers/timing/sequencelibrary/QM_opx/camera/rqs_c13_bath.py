# -*- coding: utf-8 -*-
"""SCC X-coherence for fixed-T random-multisine 13C bath sensing.

Variable-count pi_X pulses, periodic XY8-8 and Hahn echo controls.
Each row has physically timed, padded gap vectors plus number of pi pulses,
phase-mode (0=all-X, 1=XY8-8), and pulse-count parity.
The macro handles polarization, microwave sequence, SCC, dual-rail ref, camera.
"""
from qm import qua
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

XY8_PHASES = (0,90,0,90,90,0,90,0)
XY4_PHASES = (0,90,0,90)


def get_seq(base_scc_seq_args,gap_vectors,n_pi_list,
            phase_modes,even_flags,num_reps=2):
    n=len(n_pi_list)
    if n<1 or len(phase_modes)!=n or len(even_flags)!=n:
        raise ValueError("Incompatible pulse-count/phase vectors")
    if len(gap_vectors)<2 or any(len(column)!=n for column in gap_vectors):
        raise ValueError("Incorrect QUA padded gap arrays")
    maximum=len(gap_vectors)-1
    if any(type(v) is not int or not 0<=v<=maximum for v in n_pi_list):
        raise ValueError("Invalid variable pi count")
    if any(v not in (0,1,2) for v in phase_modes):
        raise ValueError("Phase mode must be 0=X, 1=XY8 or 2=XY4")
    if any(v not in (0,1) for v in even_flags):
        raise ValueError("Parity flag must be 0 or 1")
    if any(type(q) is not int or q<4 for column in gap_vectors for q in column):
        raise ValueError("QUA waits must be >=4 cycles")
    if num_reps<2 or num_reps%2:
        raise ValueError("Require positive even SCC repetitions")
    if any((m==1 and npi!=8) or (m==2 and npi!=4)
           for m,npi in zip(phase_modes,n_pi_list)):
        raise ValueError("XY8 requires 8 pulses; XY4 requires 4 pulses")
    buffer=seq_utils.get_widefield_operation_buffer()
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        gap_var=tuple(qua.declare(int) for _ in range(maximum+1))
        count=qua.declare(int)
        phase_mode=qua.declare(int)
        is_even=qua.declare(int)
        vals=tuple(gap_vectors)+(n_pi_list,phase_modes,even_flags)

        def sensing(inds,step):
            gaps=step[:maximum+1]
            npulse=step[maximum+1]
            mode=step[maximum+2]
            even=step[maximum+3]
            qua.align()
            seq_utils.macro_pi_on_2_pulse(inds,phase=0)
            qua.wait(gaps[0])
            for j in range(maximum):
                with qua.if_(npulse>j):
                    if j<8:
                        with qua.if_(mode==1):
                            seq_utils.macro_pi_pulse(inds,phase=XY8_PHASES[j])
                        with qua.else_():
                            if j<4:
                                with qua.if_(mode==2):
                                    seq_utils.macro_pi_pulse(inds,phase=XY4_PHASES[j])
                                with qua.else_():
                                    seq_utils.macro_pi_pulse(inds,phase=0)
                            else:
                                seq_utils.macro_pi_pulse(inds,phase=0)
                    else:
                        seq_utils.macro_pi_pulse(inds,phase=0)
                    qua.wait(gaps[j+1])
            # Physical pulse parity restores a common X readout convention:
            # initial pi/2_X and final pi/2_-X (even) or pi/2_X (odd).
            with qua.if_(even==1):
                seq_utils.macro_pi_on_2_pulse(inds,phase=180)
            with qua.else_():
                seq_utils.macro_pi_on_2_pulse(inds,phase=0)
            qua.wait(buffer)

        with qua.for_each_(gap_var+(count,phase_mode,is_even),vals):
            base_scc_sequence.macro(
                base_scc_seq_args,[sensing],
                gap_var+(count,phase_mode,is_even),
                num_reps,reference=True)
    return seq,[]
