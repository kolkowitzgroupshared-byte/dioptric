# -*- coding: utf-8 -*-
"""
Created on Fri Mar 6 2026

@author: Jenny
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

import utils.tool_belt as tool_belt
from utils import common

# from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # The first 3 args are ns durations and we need them as int64s

    # Unpack the durations
    readout_time, pulse_time, laser_name, _ = (
        np.int64(args[0]),
        np.int64(args[1]),
        args[2],
        args[3],
    )

    pulse_gen_wiring = config["Wiring"]["PulseGen"]
    do_daq_clock = pulse_gen_wiring["do_sample_clock"]
    do_daq_wavegen = pulse_gen_wiring["do_wavegen"]
    do_daq_gate = pulse_gen_wiring["do_apd_gate"]
    laser_delay = config["Optics"]["PhysicalLasers"][laser_name]["delay"]
    tail_pad = np.int64(500)

    seq = Sequence()
    laser_delay = 1000
    period = laser_delay + pulse_time + readout_time + tail_pad

    # Waveform Generator
    wavegen_train = [(0, LOW), (0, HIGH), (0, LOW)]

    # APD
    apd_train = [
        (laser_delay, LOW),
        (readout_time, HIGH),
        (pulse_time + tail_pad, LOW),
    ]
    seq.setDigital(do_daq_gate, apd_train)

    # Clock
    clock_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(do_daq_clock, clock_train)

    ## LASER
    # laser_train = [
    #     (laser_delay, LOW),
    #     (pulse_time, HIGH),
    #     (readout_time + tail_pad, LOW),
    # ]
    laser_train = [
        (laser_delay, LOW),
        (pulse_time, HIGH),
        (readout_time + tail_pad, LOW),
    ]
    seq.setDigital(do_daq_wavegen, laser_train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    config = common.get_config_dict()
    tool_belt.set_delays_to_zero(config)

    seq_args = [1000000, 60000, "laser_COBO_515", 515]
    seq, final, ret_vals = get_seq(None, config, seq_args)
    seq.plot()
