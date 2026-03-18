# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:34:08 2019

2/24/2020 Setting the start of the readout_time at the beginning of the sequence.

@author: Aedan
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

import utils.tool_belt as tool_belt
from utils import common

# from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    #     # Wiring
    # pulse_gen_wiring = config["Wiring"]["PulseGen"]
    # do_daq_clock = pulse_gen_wiring["do_sample_clock"]
    # do_daq_gate = pulse_gen_wiring["do_apd_gate"]

    # # Cast to 64-bit ints
    # delay = np.int64(delay)
    # readout_time = np.int64(readout_time)

    # tail_pad = np.int64(300)
    # period = np.int64(delay + readout_time + tail_pad)

    # # Define the sequence
    # seq = Sequence()

    # # DAQ sample clock: 100 ns HIGH with 100 ns LOW buffers
    # clock_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    # seq.setDigital(do_daq_clock, clock_train)

    # # APD gate during readout
    # gate_train = [(delay, LOW), (readout_time, HIGH), (tail_pad, LOW)]
    # seq.setDigital(do_daq_gate, gate_train)

    # # No laser control

    # final = OutputState([], 0.0, 0.0)
    # return seq, final, [period]

    # Parse wiring and args

    # The first 3 args are ns durations and we need them as int64s

    # Unpack the durations
    readout_time, pulse_time, laser_name, laser_wavelength = (
        np.int64(args[0]),
        np.int64(args[1]),
        args[2],
        args[3],
    )

    pulse_gen_wiring = config["Wiring"]["PulseGen"]
    do_daq_clock = pulse_gen_wiring["do_sample_clock"]
    do_daq_laser = pulse_gen_wiring[f"do_laser_COBO_{int(laser_wavelength)}_dm"]
    do_daq_gate = pulse_gen_wiring["do_apd_gate"]
    laser_delay = config["Optics"]["PhysicalLasers"][laser_name]["delay"]
    tail_pad = np.int64(300)

    seq = Sequence()
    period = laser_delay + pulse_time + readout_time + tail_pad

    ## LASER
    laser_train = [
        (laser_delay, LOW),
        (pulse_time, HIGH),
        (readout_time + tail_pad, LOW),
    ]
    seq.setDigital(do_daq_laser, laser_train)

    ## APD
    apd_train = [
        (laser_delay + pulse_time, LOW),
        (readout_time, HIGH),
        (tail_pad, LOW),
    ]
    seq.setDigital(do_daq_gate, apd_train)

    ## CLOCK
    clock_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(do_daq_clock, clock_train)

    # apd_train = [
    #     (init_laser_delay + start_time, LOW),
    #     (end_time - start_time, HIGH),
    #     (tail_pad, LOW),
    # ]
    # seq.setDigital(do_daq_gate, apd_train)

    # laser_train = [(init_time, HIGH), (end_time + init_laser_delay - init_time, LOW)]
    # seq.setDigital(do_daq_laser, laser_train)

    # # DAQ sample clock: 100 ns HIGH with 100 ns LOW buffers
    # clock_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    # seq.setDigital(do_daq_clock, clock_train)

    # tool_belt.process_laser_seq(
    #     pulse_streamer, seq, config, init_laser_name, init_laser_power, train
    # )

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    config = common.get_config_dict()
    tool_belt.set_delays_to_zero(config)

    seq_args = [1000000, 60000, "laser_COBO_515", 515]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
