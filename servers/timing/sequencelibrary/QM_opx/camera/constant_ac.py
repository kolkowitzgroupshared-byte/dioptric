#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constant sequence for the QM OPX camera. This sequence is used to generate a constant trigger signal for the camera, with a specified period and number of repetitions.
"""

import matplotlib.pyplot as plt
import numpy
import qm
from qm import QuantumMachinesManager, generate_qua_script, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(
    digital_channels, analog_channels, analog_voltages, analog_freqs, num_reps=None
):
    # Validate analog_voltages
    for val in analog_voltages:
        if val > 0.5:
            raise RuntimeError("Analog voltages must be <= 0.5 V.")

    if num_reps is None:
        num_reps = -1

    analog_freqs = [int(el * 10**6) for el in analog_freqs]
    clock_cycles = 250  # * 4 ns / clock_cycle = 1 us
    with qua.program() as seq:
        ### Non-repeated stuff here
        num_analog_channels = len(analog_channels)
        amps = [None] * num_analog_channels
        for ind in range(len(analog_channels)):
            # Update freqs
            chan = analog_channels[ind]
            element = f"ao{chan}"
            freq = analog_freqs[ind]
            qua.update_frequency(element, freq)
            # Declare amplitudes. These just scale the voltage of the pulse we're running
            # ("cw"), which has an amplitude of 0.5 V, so double the passed value to
            # get a true voltage amplitude
            amp = 2 * analog_voltages[ind]
            amps[ind] = qua.declare(qua.fixed, value=amp)

        ### Define one rep here
        def one_rep():
            for chan in digital_channels:
                element = f"do{chan}"
                qua.play("on", element, duration=clock_cycles)
            for ind in range(len(analog_channels)):
                chan = analog_channels[ind]
                element = f"ao{chan}"
                amp = amps[ind]
                qua.play("cw" * qua.amp(amp), element, duration=clock_cycles)
            # qua.play("on", "ao_laser_OPTO_589_am", duration=clock_cycles)
            # qua.play("cw" * qua.amp(0), element, duration=clock_cycles)

        ### Handle the reps in the utils code
        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        # one_rep()

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
        ret_vals = get_seq([1], [2, 6], [0.19, 0.19], [75, 75])
        seq, seq_ret_vals = ret_vals

        # Serialize to file
        # sourceFile = open('debug2.py', 'w')
        # print(generate_qua_script(seq, opx_config), file=sourceFile)
        # sourceFile.close()

        sim_config = SimulationConfig(duration=10000 // 4)
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
