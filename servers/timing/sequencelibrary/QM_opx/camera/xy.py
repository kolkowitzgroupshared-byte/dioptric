# -*- coding: utf-8 -*-
"""
Widefield XY8 Coherence Sequence

Created on October 13th, 2023
Saroj Chand on March 22nd, 2025
"""

import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, step_vals, xy_seq, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    uwave_ind_list = base_scc_seq_args[-1]
    # macro_pi_pulse_duration = seq_utils.get_macro_pi_pulse_duration(uwave_ind_list)
    macro_pi_pulse_duration = seq_utils.convert_ns_to_cc(88)
    # Adjust step values to compensate for internal delays
    step_vals = [
        seq_utils.convert_ns_to_cc(el) - macro_pi_pulse_duration for el in step_vals
    ]
    # Choose pulse phase pattern
    phase_dict = {
        "hahn": [0],
        "xy2": [0, 90],
        "xy4": [0, 90, 0, 90],
        "xy8": [0, 90, 0, 90, 90, 0, 90, 0],
        "xy16": [0, 90, 0, 90, 90, 0, 90, 0, 180, 270, 180, 270, 270, 180, 270, 180],
    }

    # Parse xy_seq, e.g. "xy8-4" → base="xy8", reps=4
    match = re.match(r"([a-zA-Z]+\d*)(?:-(\d+))?", xy_seq.lower())
    base_seq = match.group(1)
    num_blocks = int(match.group(2)) if match.group(2) else 1
    # Fetch and repeat the base phase pattern
    base_phases = phase_dict.get(base_seq)
    xy_phases = base_phases * num_blocks

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)

        def uwave_macro_sig(uwave_ind_list, step_val):
            qua.align()
            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(step_val)
            for i, phase in enumerate(xy_phases):
                seq_utils.macro_pi_pulse(uwave_ind_list, phase=phase)
                if i < len(xy_phases) - 1:
                    qua.wait(2 * step_val)  # 2τ between pis
                else:
                    qua.wait(step_val)  # τ after last pi

            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(buffer)

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
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 2e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[107.721, 107.702], [107.439, 105.963]],
                [164, 144],
                [1.0, 1.0],
                [[72.443, 73.218], [72.175, 71.806]],
                [88, 80],
                [1.0, 1.0],
                [False, False],
                [0, 1],
            ],
            [
                200,
                18796,
                312752,
                42920,
                1000,
                147776,
            ],
            "xy8-1",
            1,
        )

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(f"An error occurred: {exc}")
    finally:
        qmm.close_all_quantum_machines()
