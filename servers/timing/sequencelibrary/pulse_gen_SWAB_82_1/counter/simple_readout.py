# -*- coding: utf-8 -*-
"""
simple_readout_with_marker.py

Args:
  delay (ns)
  readout_time (ns)
  laser_name (str)
  laser_power (float)
  marker_width (ns)      # e.g. 100
  marker_at_readout (int)  # 1 => marker at readout start; 0 => marker at period start
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey

LOW = 0
HIGH = 1


def _to_virtual_key(x):
    # robust conversion for args coming in as Enum or string
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        s = x.strip()
        # common cases: "IMAGING", "VirtualLaserKey.IMAGING", "imaging"
        if "VirtualLaserKey." in s:
            s = s.split("VirtualLaserKey.", 1)[1]
        s_up = s.upper()
        if s_up in VirtualLaserKey.__members__:
            return VirtualLaserKey[s_up]
    # fallback: assume already valid
    return x


def get_seq(pulse_streamer, config, args):
    """
    Args:
      delay (ns)
      readout_time (ns)
      pulse_time (ns)
      virtual_laser_key (e.g. VirtualLaserKey.IMAGING or "IMAGING")
    """
    print(args)
    delay, readout_time, pulse_time, vkey = args

    delay = np.int64(delay)
    readout_time = np.int64(readout_time)
    pulse_time = np.int64(pulse_time)

    vkey = _to_virtual_key(vkey)

    w = config["Wiring"]["PulseGen"]
    # do_clk = w["do_sample_clock"]
    do_gate = w["do_apd_gate"]
    # do_mark = w.get("do_pixel_marker", None)

    tail = np.int64(300)
    period = np.int64(delay + readout_time + tail)

    seq = Sequence()

    # # sample clock: 100 ns HIGH near end of period (same as your template)
    # clk_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    # seq.setDigital(do_clk, clk_train)

    # APD gate high during readout
    gate_train = [(delay + pulse_time, LOW), (readout_time, HIGH), (tail, LOW)]
    seq.setDigital(do_gate, gate_train)

    # laser ON for entire period (or you can gate only during readout if you prefer)
    # laser_train = [(period, HIGH)]
    laser_train = [(delay, LOW), (pulse_time, HIGH), (readout_time + tail, LOW)]
    tb.process_laser_seq(seq, vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    config = common.get_config_dict()
    args = [5e5, 100e6, 10e6, "imaging"]
    #    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
