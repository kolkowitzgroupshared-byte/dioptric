# -*- coding: utf-8 -*-
"""
Widefield iQ delay calibration
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, step_vals, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    uwave_ind_list = base_scc_seq_args[-1]
    uwave_ind = uwave_ind_list[0]
    # Get elements
    i_el = seq_utils.get_sig_gen_i_element(uwave_ind)
    q_el = seq_utils.get_sig_gen_q_element(uwave_ind)
    sig_gen_el = seq_utils.get_sig_gen_element(uwave_ind)

    # Convert step values (ns) to clock cycles
    step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals]

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        delay = qua.declare(int)

        def uwave_macro_i(uwave_ind_list_inner, delay):
            qua.align()
            qua.play("pi_pulse", i_el)
            qua.wait(delay, sig_gen_el)
            qua.play("pi_pulse", sig_gen_el)
            qua.wait(buffer)

        def uwave_macro_q(uwave_ind_list_inner, delay):
            qua.align()
            qua.play("pi_pulse", q_el)
            qua.wait(delay, sig_gen_el)
            qua.play("pi_pulse", sig_gen_el)
            qua.wait(buffer)

        with qua.for_each_(delay, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_i, uwave_macro_q],
                step_val=delay,
                num_reps=num_reps,
                reference=False,
            )

    return seq, []


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 10e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[107.711, 107.741], [107.435, 105.997]],
                [132, 132],
                [1.0, 1.0],
                [[72.436, 73.249], [72.173, 71.834]],
                [40, 88],
                [1.0, 1.0],
                [False, False],
                [1],
            ],
            [
                340.0,
                20.0,
                200.0,
            ],
            2,
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
