# -*- coding: utf-8 -*-
"""
Created on Tue Feb  11 21:24:36 2026

Simple readout: drives DAQ clock, APD gate, and imaging laser (continuous ON).
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common
from utils import tool_belt as tb 
from utils.constants import VirtualLaserKey, NormMode

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):
    # Unpack the args (ignore laser params for now)
    delay, readout_time, *_ = (
        args  # accepts [delay, readout_time, laser_name, laser_power]
    )
    readout_vkey = VirtualLaserKey.IMAGING

    # Wiring
    pulse_gen_wiring = config["Wiring"]["PulseGen"]
    do_daq_clock = pulse_gen_wiring["do_sample_clock"]
    do_daq_gate = pulse_gen_wiring["do_apd_gate"]
    do_tisapph_aom = pulse_gen_wiring["do_laser_TISAPPH_dm"]
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

    # Laser train: on continuously during both measurements
    # laser_train = [(int(period), HIGH)]
    # tb.process_laser_seq(seq, readout_vkey, laser_train)

    tisapph_train = [(int(period), HIGH)]
    seq.setDigital(do_tisapph_aom, tisapph_train)


    final = OutputState([], 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    config = common.get_config_dict()
    # Keep 4-arg shape for compatibility; laser args are ignored
    args = [500_000, 10_000_000, "laser_INTE_520", 1.0]
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()