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
      virtual_laser_key (e.g. VirtualLaserKey.IMAGING or "IMAGING")
      marker_width_ns (ns)  # e.g. 100
      marker_at_readout (int)  # 1 => marker at readout start, 0 => marker at t=0
    """
    delay, readout_time, vkey, marker_width_ns, marker_at_readout = args

    delay = np.int64(delay)
    readout_time = np.int64(readout_time)
    marker_width_ns = np.int64(marker_width_ns)
    marker_at_readout = int(marker_at_readout)

    vkey = _to_virtual_key(vkey)

    w = config["Wiring"]["PulseGen"]
    do_clk = w["do_sample_clock"]
    do_gate = w["do_apd_gate"]
    do_mark = w.get("do_pixel_marker", None)

    tail = np.int64(300)
    period = np.int64(delay + readout_time + tail)

    seq = Sequence()

    # sample clock: 100 ns HIGH near end of period (same as your template)
    clk_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    seq.setDigital(do_clk, clk_train)

    # APD gate high during readout
    gate_train = [(delay, LOW), (readout_time, HIGH), (tail, LOW)]
    seq.setDigital(do_gate, gate_train)

    # marker pulse (scope/camera trigger)
    if do_mark is not None:
        t0 = delay if marker_at_readout else np.int64(0)
        t0 = np.int64(max(0, min(int(t0), int(period - marker_width_ns))))
        mark_train = [
            (t0, LOW),
            (marker_width_ns, HIGH),
            (period - t0 - marker_width_ns, LOW),
        ]
        seq.setDigital(do_mark, mark_train)

    # laser ON for entire period (or you can gate only during readout if you prefer)
    laser_train = [(period, HIGH)]
    tb.process_laser_seq(seq, vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    config = common.get_config_dict()
    args = [500000, 10000000.0, "laser_INTE_520", 1.0]
    # args = [5000, 10000.0, 1, 'integrated_520',None]
    #    seq_args_string = tool_belt.encode_seq_args(args)
    seq, ret_vals, period = get_seq(None, config, args)
    seq.plot()
