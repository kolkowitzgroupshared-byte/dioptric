# -*- coding: utf-8 -*-
"""
Widefield illumination and collection

Created on October 5th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy
from qm import QuantumMachinesManager, generate_qua_script, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils
from utils.constants import ModMode


def get_seq(readout_duration_ns, readout_laser, num_reps):
    if num_reps is None:
        num_reps = 1

    laser_element = seq_utils.get_laser_mod_element(readout_laser, sticky=True)
    camera_element = "do_camera_trigger"
    readout_duration = round(readout_duration_ns / 4)
    default_duration = seq_utils.get_default_pulse_duration()
    with qua.program() as seq:
        ### Define one rep here
        def one_rep(rep_ind):
            qua.play("on", laser_element)
            qua.play("on", camera_element)
            qua.wait(readout_duration - default_duration)
            qua.align()
            qua.ramp_to_zero(laser_element)
            qua.ramp_to_zero(camera_element)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps)

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
        args = [3000.0, "laser_OPTO_589"]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=round(10e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
