# -*- coding: utf-8 -*-
"""
Widefield DQ relaxation interleave
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence
from utils import common
from utils import tool_belt as tb
from utils.constants import NVSpinState


def get_seq(
    base_scc_seq_args,
    step_vals,
    num_reps=1,
):
    step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals]

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        step_val = qua.declare(int)

        #
        def uwave_macro_ref(uwave_ind_list, step_val):
            qua.align()
            # seq_utils.macro_pi_pulse(uwave_ind_list[:1])
            seq_utils.macro_pi_pulse(uwave_ind_list[1:])
            qua.wait(step_val)
            seq_utils.macro_pi_pulse(uwave_ind_list[1:])
            # seq_utils.macro_pi_pulse(uwave_ind_list[:1])

        def uwave_macro_sig(uwave_ind_list, step_val):
            qua.align()
            seq_utils.macro_pi_pulse(uwave_ind_list[1:])
            # seq_utils.macro_pi_pulse(uwave_ind_list[:1])
            qua.wait(step_val)
            seq_utils.macro_pi_pulse(uwave_ind_list[:1])
            # seq_utils.macro_pi_pulse(uwave_ind_list[1:])

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig, uwave_macro_ref],
                step_val,
                num_reps=num_reps,
                reference=False,
            )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            [
                [112.03744137001495, 109.50814699059372],
                [112.22844137001495, 108.72114699059371],
                [111.05444137001496, 108.90314699059371],
            ],
            [
                [76.0990499534296, 75.08248628148773],
                [76.2550499534296, 74.39048628148774],
                [75.4050499534296, 74.55748628148774],
            ],
            1000.0,
            "NVSpinState.HIGH",
            "NVSpinState.LOW",
            "NVSpinState.ZERO",
            "NVSpinState.LOW",
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(400e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
