"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
@author: sbchand
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig
import utils.tool_belt as tb

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, step_vals, num_reps=1):
    step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals]

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)

        def uwave_macro_sig(uwave_ind_list, step_val):
            # seq_utils.macro_pi_pulse(uwave_ind_list, duration_cc=step_val, phase=0)
            seq_utils.macro_pi_pulse(uwave_ind_list, duration_cc=step_val)

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                step_val,
                num_reps,
            )
    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 10e3
    tb.set_delays_to_zero(opx_config)

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [
                    [107.247, 107.388],
                    [95.571, 94.737],
                ],
                [188, 108],
                [1.0, 1.0],
                [
                    [72.0, 72.997],
                    [62.207, 62.846],
                ],
                [104, 56],
                [1.0, 1.0],
                [False, False],
                [0,1],
            ],
            [
                112.0,
                184.0,
            ],
            1,
        )

        sim_config = SimulationConfig(duration=int(100e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(f"An error occurred: {exc}")
    finally:
        qmm.close_all_quantum_machines()
