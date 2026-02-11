# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

"""

# import numpy
# from pulsestreamer import OutputState, Sequence

# from utils import common
# from utils import tool_belt as tb

# LOW = 0
# HIGH = 1


# def get_seq(pulse_streamer, config, args):
#     # Unpack the args
#     delay, readout_time, laser_name, laser_power = args

#     # Get what we need out of the wiring dictionary
#     pulse_gen_wiring = config["Wiring"]["PulseGen"]
#     pulse_gen_do_daq_clock = pulse_gen_wiring["do_sample_clock"]
#     pulse_gen_do_daq_gate = pulse_gen_wiring["do_apd_gate"]

#     # Convert the 32 bit ints into 64 bit ints
#     delay = numpy.int64(delay)
#     readout_time = numpy.int64(readout_time)

#     period = numpy.int64(delay + readout_time + 300)

#     # tb.check_laser_power(laser_name, laser_power)

#     # Define the sequence
#     seq = Sequence()

#     # The clock signal will be high for 100 ns with buffers of 100 ns on
#     # either side. During the buffers, everything should be low. The buffers
#     # account for any timing jitters/delays and ensure that everything we
#     # expect to be on one side of the clock signal is indeed on that side.
#     train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
#     seq.setDigital(pulse_gen_do_daq_clock, train)

#     train = [(delay, LOW), (readout_time, HIGH), (300, LOW)]
#     seq.setDigital(pulse_gen_do_daq_gate, train)

#     train = [(period, HIGH)]
#     tb.process_laser_seq(seq, laser_name, laser_power, train)

#     final_digital = []
#     final = OutputState(final_digital, 0.0, 0.0)

#     return seq, final, [period]

# if __name__ == "__main__":
#     config = common.get_config_dict()
#     args = [500000, 10000000.0, "laser_INTE_520", 1.0]
#     # args = [5000, 10000.0, 1, 'integrated_520',None]
#     #    seq_args_string = tool_belt.encode_seq_args(args)
#     seq, ret_vals, period = get_seq(None, config, args)
#     seq.plot()
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:36 2019

Laser-free variant: drives DAQ clock and APD gate only.
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):
    # Unpack the args (ignore laser params for now)
    delay, readout_time, *_ = (
        args  # accepts [delay, readout_time, laser_name, laser_power]
    )

    # Wiring
    pulse_gen_wiring = config["Wiring"]["PulseGen"]
    do_daq_clock = pulse_gen_wiring["do_sample_clock"]
    do_daq_gate = pulse_gen_wiring["do_apd_gate"]

    # Cast to 64-bit ints
    delay = np.int64(delay)
    readout_time = np.int64(readout_time)

    tail_pad = np.int64(300)
    period = np.int64(delay + readout_time + tail_pad)

    # Define the sequence
    seq = Sequence()

    # DAQ sample clock: 100 ns HIGH with 100 ns LOW buffers
    clock_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(do_daq_clock, clock_train)

    # APD gate during readout
    gate_train = [(delay, LOW), (readout_time, HIGH), (tail_pad, LOW)]
    seq.setDigital(do_daq_gate, gate_train)

    # No laser control

    final = OutputState([], 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    config = common.get_config_dict()
    # Keep 4-arg shape for compatibility; laser args are ignored
    args = [500_000, 10_000_000, "laser_INTE_520", 1.0]
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
