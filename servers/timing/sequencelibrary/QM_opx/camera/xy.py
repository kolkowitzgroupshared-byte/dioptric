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

    # Prefer using your library helper (not hard-coded 88 ns)
    t_pi_cc  = seq_utils.get_macro_pi_pulse_duration(uwave_ind_list[0])  # if available
    t_pi2_cc = t_pi_cc // 2  # if pi/2 is exactly half; otherwise get explicitly

    phase_dict = {
        "hahn": [0],
        "xy2": [0, 90],
        "xy4": [0, 90, 0, 90],
        "xy8": [0, 90, 0, 90, 90, 0, 90, 0],
        "xy16": [0, 90, 0, 90, 90, 0, 90, 0, 180, 270, 180, 270, 270, 180, 270, 180],
    }

    match = re.match(r"([a-zA-Z]+\d*)(?:-(\d+))?", xy_seq.lower())
    if not match:
        raise ValueError(f"Bad xy_seq: {xy_seq}")
    base_seq = match.group(1)
    num_blocks = int(match.group(2)) if match.group(2) else 1

    base_phases = phase_dict.get(base_seq)
    if base_phases is None:
        raise ValueError(f"Unknown base seq: {base_seq}")
    xy_phases = base_phases * num_blocks

    # Pre-filter taus that would produce negative waits (important for short taus)
    step_vals_cc = []
    for tau_ns in step_vals:
        tau_cc = seq_utils.convert_ns_to_cc(tau_ns)

        # center-to-center timing:
        w_edge = tau_cc - (t_pi2_cc // 2) - (t_pi_cc // 2)
        w_mid  = 2 * tau_cc - t_pi_cc

        if w_edge < 0 or w_mid < 0:
            continue
        step_vals_cc.append(tau_cc)

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        tau_cc = qua.declare(int)

        def uwave_macro_sig(uwave_ind_list, tau_cc):
            qua.align()

            w_edge = tau_cc - (t_pi2_cc // 2) - (t_pi_cc // 2)
            w_mid  = 2 * tau_cc - t_pi_cc

            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(w_edge)

            for i, phase in enumerate(xy_phases):
                seq_utils.macro_pi_pulse(uwave_ind_list, phase=phase)
                qua.wait(w_mid if i < len(xy_phases) - 1 else w_edge)

            seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            qua.wait(buffer)

        with qua.for_each_(tau_cc, step_vals_cc):
            base_scc_sequence.macro(base_scc_seq_args, [uwave_macro_sig], tau_cc, num_reps)

    return seq, []

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
                [0],
            ],
            [200, 18796],
            "xy16-1",
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
