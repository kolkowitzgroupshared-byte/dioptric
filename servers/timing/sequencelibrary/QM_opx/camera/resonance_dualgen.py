# -*- coding: utf-8 -*-
"""
Widefield dual generation resonance sequence
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

def get_seq(
    base_scc_seq_args,
    step_inds=None,
    num_reps=1,          # kept for backward compat; not used for loops below
    num_reps_sig=None,   # signal quarter reps (from host)
    num_reps_ref=None,   # reference quarter reps (from host)
):
    # fallbacks
    if num_reps_sig is None: num_reps_sig = num_reps
    if num_reps_ref is None: num_reps_ref = max(1, num_reps // 4)

    buffer = seq_utils.get_widefield_operation_buffer()

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        step_ind = qua.declare(int)
        rep = qua.declare(int)

        total_num_steps = len(step_inds)
        quarter = total_num_steps // 4
        half    = total_num_steps // 2
        three_q = 3 * quarter

        # Pulse only ONE source in signal blocks; none in reference blocks
        def uwave_macro(uwave_ind_list, step_ind):
            # Q1: pulse uwave_ind_list[0]
            with qua.if_(step_ind < quarter):
                seq_utils.macro_pi_pulse([uwave_ind_list[0]])
                qua.wait(buffer)
            # Q2: pulse uwave_ind_list[1]
            with qua.else_():
                with qua.if_(step_ind < half):
                    seq_utils.macro_pi_pulse([uwave_ind_list[1]])
                    qua.wait(buffer)
                # Q3/Q4: no ESR pulse (microwaves off/fixed handled routine-side)
                with qua.else_():
                    qua.nop()
                    qua.wait(buffer)

        with qua.for_each_(step_ind, step_inds):
            # SIGNAL: Q1 & Q2 (loop num_reps_sig times)
            with qua.if_(step_ind < half):
                qua.assign(rep, 0)
                with qua.while_(rep < num_reps_sig):
                    base_scc_sequence.macro(
                        base_scc_seq_args,
                        uwave_macro,
                        step_ind,
                        num_reps=1,
                        reference=False,
                    )
                    qua.assign(rep, rep + 1)

            # REFERENCE: Q3 & Q4 (loop fewer times: num_reps_ref)
            with qua.else_():
                qua.assign(rep, 0)
                with qua.while_(rep < num_reps_ref):
                    base_scc_sequence.macro(
                        base_scc_seq_args,
                        uwave_macro,
                        step_ind,
                        num_reps=1,
                        reference=False, 
                    )
                    qua.assign(rep, rep + 1)

    # No return values used
    seq_ret_vals = []

    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 10e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[108.477, 107.282], [109.356, 108.789]],
                [220, 220],
                [1.0, 1.0],
                [[73.558, 71.684], [74.227, 72.947]],
                [124, 124],
                [1.0, 1.0],
                [False, False],
                [0],
            ],
            [70, 219],
            1,
        )

        sim_config = SimulationConfig(duration=int(300e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
