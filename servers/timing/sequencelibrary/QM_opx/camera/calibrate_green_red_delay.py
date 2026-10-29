# -*- coding: utf-8 -*-
"""
Widefield green-red delay calibration
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence
from utils import common
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


def get_seq(period, num_reps=1):
    half_period_cc = seq_utils.convert_ns_to_cc(period / 2, allow_rounding=True)
    buffer = seq_utils.get_widefield_operation_buffer()

    green_laser = tb.get_physical_laser_name(VirtualLaserKey.POLARIZATION)
    red_laser = tb.get_physical_laser_name(VirtualLaserKey.IONIZATION)
    green_el = seq_utils.get_laser_mod_element(green_laser)
    red_el = seq_utils.get_laser_mod_element(red_laser)

    with qua.program() as seq:
        # seq_utils.turn_on_aods()

        ### Define one rep here
        def one_rep():
            # seq_utils.turn_on_aods()
            # qua.wait(buffer)
            qua.align()
            for el in [green_el, red_el]:
                qua.play("on", el, duration=half_period_cc)
                qua.play("off", el, duration=half_period_cc)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)

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
        seq, seq_ret_vals = get_seq(10000, -1)

        sim_config = SimulationConfig(duration=int(20e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
