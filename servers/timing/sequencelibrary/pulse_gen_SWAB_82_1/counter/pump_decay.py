# -*- coding: utf-8 -*-
"""
Created on Tues Feb 3, 2026

turn on the MW and pulse a square wave frequency modulation

@author: jchen-1
"""

import numpy
from pulsestreamer import OutputState, Sequence

import utils.tool_belt as tool_belt
from utils.tool_belt import States

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):

    # Unpack the args
    period, sig_gen_name = args

    period = numpy.int64(period)
    half_period = numpy.int64(period / 2)
    # state = States(state)

    pulser_wiring = config["Wiring"]["PulseStreamer"]
    sig_gen_gate_chan_name = "do_{}_gate".format(sig_gen_name)
    pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
    pulser_ao_fm = pulser_wiring["ao_fm"]

    # Define the sequence
    seq = Sequence()

    train = [(period, HIGH)]
    seq.setDigital(pulser_do_sig_gen_gate, train)

    train = [(half_period, HIGH), (half_period, LOW)]
    seq.setAnalog(pulser_ao_fm, train)

    final_digital = []
    final = OutputState(final_digital, 0.0, 0.0)

    return seq, final, [period]


if __name__ == "__main__":
    config = tool_belt.get_config_dict()
    args = [20 * 10**8, States.HIGH]
    #    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
