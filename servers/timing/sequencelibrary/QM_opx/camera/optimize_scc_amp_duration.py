# -*- coding: utf-8 -*-
"""
Widefield charge state readout after polarization/ionization varying both readout duration and amplitude, no spin manipulation
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, step_vals, num_reps):
    step_vals = np.array(step_vals)
    duration_step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals[:, 0]]
    amp_step_vals = step_vals[:, 1]

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        duration_override = qua.declare(int)
        amp_override = qua.declare(qua.fixed)

        def uwave_macro_sig(uwave_ind_list, step_val):
            # seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
            seq_utils.macro_pi_pulse(uwave_ind_list)

        def uwave_macro_ref(uwave_ind_list, step_val):
            pass

        def one_step():
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig, uwave_macro_ref],
                num_reps=num_reps,
                scc_duration_override=duration_override,
                scc_amp_override=amp_override,
                reference=False,
            )

        # with qua.for_each_(duration_override, duration_step_vals):
        #     with qua.for_each_(amp_override, amp_step_vals):
        #         one_step()
        # Outer loop for durations

        # Cartesian product of durations and amplitudes
        with qua.for_each_(
            (duration_override, amp_override), (duration_step_vals, amp_step_vals)
        ):
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
                [[108.826, 106.773], [110.002, 105.138], [110.183, 108.395]],
                [1000, 1000, 1000],
                [1.0, 1.0, 1.0],
                [[73.347, 72.222], [74.196, 70.89], [74.442, 73.533]],
                [140, 140, 140],
                [1.0, 1.0, 1.0],
                [False, False, False],
                [0, 1],
            ],
            [
                [100, 1.5],
                [196.0, 1.17],
                [228.0, 1.2],
                [164.0, 1.17],
                [196.0, 1.08],
                [196.0, 1.02],
                [100.0, 0.99],
                [100.0, 1.08],
            ],
            5,
        )

        sim_config = SimulationConfig(duration=int(250e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
