#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constant sequence for the QM OPX camera. This sequence is used to generate a constant trigger signal for the camera, with a specified period and number of repetitions.
"""


import numpy
import qm
from qm import qua
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import SimulationConfig
from servers.timing.sequencelibrary.QM_opx import seq_utils
import utils.common as common
import matplotlib.pyplot as plt


def get_seq(args, num_reps=None):
    period_ms = args[0]
    if num_reps == None:
        num_reps = -1

    period = seq_utils.convert_ns_to_cc(period_ms * 10**6)
    default_duration = seq_utils.get_default_pulse_duration()
    wait_duration = period - 2 * default_duration
    camera_el = f"do_camera_trigger"
    with qua.program() as seq:
        ### Define one rep here
        def one_rep():
            qua.play("on", camera_el)
            qua.play("off", camera_el)
            qua.wait(wait_duration, camera_el)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        # one_rep()

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [5e-3]
        ret_vals = get_seq(args)
        seq, seq_ret_vals = ret_vals

        # Serialize to file
        # sourceFile = open('debug2.py', 'w')
        # print(generate_qua_script(seq, opx_config), file=sourceFile)
        # sourceFile.close()

        sim_config = SimulationConfig(duration=int(10e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
