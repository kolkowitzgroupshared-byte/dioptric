# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    # pi_pulse_duration = seq_utils.get_macro_pi_pulse_duration([1])

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        def uwave_macro_sig(uwave_ind_list, step_val):
            seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
            # seq_utils.macro_pi_pulse(uwave_ind_list)
            qua.wait(buffer)

        def uwave_macro_ref(uwave_ind_list, step_val):
            pass

        base_scc_sequence.macro(
            base_scc_seq_args,
            [uwave_macro_sig, uwave_macro_ref],
            num_reps=num_reps,
            reference=False,
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

    try:
        seq, seq_ret_vals = get_seq(
            [
                [
                    [108.61033817964635, 109.89718413914437],
                    [109.19233817964634, 110.44518413914437],
                    [108.63333817964634, 110.49318413914438],
                ],
                [
                    [73.37605409727466, 75.19065445569203],
                    [73.91305409727465, 75.64165445569202],
                    [73.42805409727465, 75.66565445569202],
                ],
                [1000, 160, 1000],
                [1.0, 0.1, 1.2],
                [],
                [0, 1],
            ],
            10,
        )

        sim_config = SimulationConfig(duration=int(400e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
