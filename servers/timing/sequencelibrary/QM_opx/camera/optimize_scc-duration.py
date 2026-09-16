# -*- coding: utf-8 -*-
"""
Widefield ESR

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, scc_duration_steps, num_reps):
    duration_step_vals = [seq_utils.convert_ns_to_cc(el) for el in scc_duration_steps]
    with qua.program() as seq:
        ### init
        seq_utils.init()
        seq_utils.macro_run_aods()

        duration_override = qua.declare(int)

        def uwave_macro_sig(uwave_ind_list, step_val):
            seq_utils.macro_pi_pulse(uwave_ind_list)

        def uwave_macro_ref(uwave_ind_list, step_val):
            pass

        def one_step():
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig, uwave_macro_ref],
                num_reps=num_reps,
                scc_duration_override=duration_override,
                reference=False,
            )

        with qua.for_each_(duration_override, duration_step_vals):
            one_step()

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
                    [108.748, 106.905],
                    [115.626, 101.27],
                    [108.961, 115.735],
                    [98.369, 100.929],
                ],
                [1000, 1000, 1000, 1000],
                [1.0, 1.0, 1.0, 1.0],
                [
                    [73.525, 72.483],
                    [78.674, 67.831],
                    [73.831, 79.615],
                    [65.056, 67.6],
                ],
                [140, 140, 140, 140],
                [1.0, 1.0, 1.0, 1.0],
                [False, False, False, False],  # Spin flip do target list
                [0, 1],
            ],
            [1000, 1120, 1240],
            5,
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
