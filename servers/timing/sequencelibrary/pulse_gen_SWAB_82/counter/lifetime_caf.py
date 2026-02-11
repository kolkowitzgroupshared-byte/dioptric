# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:34:08 2019

2/24/2020 Setting the start of the readout_time at the beginning of the sequence.

@author: Aedan
"""

import numpy
from pulsestreamer import OutputState, Sequence

import utils.tool_belt as tool_belt
from utils import common

# from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # %% Parse wiring and args

    # The first 3 args are ns durations and we need them as int64s
    durations = []
    for ind in range(3):
        durations.append(numpy.int64(args[ind]))

    # Unpack the durations
    start_time, end_time, init_time = durations

    # Get the APD index
    init_laser_name = args[3]
    laser_wavelength = args[4]
    apd_index = args[5]

    init_laser_delay = config["Optics"]["PhysicalLasers"][init_laser_name]["delay"]

    pulser_wiring = config["Wiring"]["PulseGen"]
    pulser_do_apd_gate = pulser_wiring["do_apd_gate"]
    # pulser_do_aom = pulser_wiring['do_532_aom']
    pulser_do_laser = pulser_wiring[f"do_laser_COBO_{int(laser_wavelength)}_dm"]

    # %% Calclate total period. This is fixed for each tau index

    # The period is independent of the particular tau, but it must be long
    # enough to accomodate the longest tau
    period = init_laser_delay + end_time

    # %% Define the sequence

    seq = Sequence()

    # APD

    train = [(init_laser_delay + start_time, LOW), (end_time - start_time, HIGH)]
    seq.setDigital(pulser_do_apd_gate, train)

    # Pulse the laser with the AOM for polarization and readout
    train = [(init_time, HIGH), (end_time + init_laser_delay - init_time, LOW)]
    # tool_belt.process_laser_seq(
    #     pulse_streamer, seq, config, init_laser_name, init_laser_power, train
    # )
    seq.setDigital(pulser_do_laser, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    config = common.get_config_dict()
    tool_belt.set_delays_to_zero(config)

    seq_args = [0, 1000000, 60000, "laser_COBO_515", 515, 0]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
