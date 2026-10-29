#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constant sequence for the QM OPX camera. This sequence is used to generate a constant trigger signal for the camera, with a specified period and number of repetitions.
"""

import logging

import matplotlib.pyplot as plt
import numpy
import qm
from qm import QuantumMachinesManager, generate_qua_script, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(num_reps=None):
    buffer = 250
    num_reps = 10

    with qua.program() as seq:
        # reps_ind = qua.declare(int)
        # with qua.for_(reps_ind, 0, reps_ind < num_reps, reps_ind + 1):
        #     qua.play("spin_polarize", "ao_laser_OPTO_589_am")

        reps_ind = qua.declare(int, value=0)
        with qua.while_(reps_ind < num_reps):
            qua.play("spin_polarize", "ao_laser_OPTO_589_am")
            qua.assign(reps_ind, reps_ind + 1)

        qua.align()
        qua.wait(buffer)
        qua.pause()
        qua.play("spin_polarize", "ao_laser_OPTO_589_am")

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
        ret_vals = get_seq(10)
        seq, seq_ret_vals = ret_vals

        # Serialize to file
        # sourceFile = open('debug2.py', 'w')
        # print(generate_qua_script(seq, opx_config), file=sourceFile)
        # sourceFile.close()

        sim_config = SimulationConfig(duration=round(100e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    # except Exception as exc:
    # print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
