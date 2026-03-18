# -*- coding: utf-8 -*-
"""
Simple APD-gate / laser overlap scan using VirtualLaserKey.

Args:
    [laser_on_ns, gate_width_ns, gate_offset_ns, laser_vkey]

Convention:
    gate_offset_ns > 0  -> APD gate starts AFTER laser onset
    gate_offset_ns < 0  -> APD gate starts BEFORE laser onset
"""

from pulsestreamer import Sequence, OutputState
import numpy as np

from utils import tool_belt as tb
from utils.constants import Digital, VirtualLaserKey

LOW = Digital.LOW
HIGH = Digital.HIGH


def _as_int64(name, value):
    try:
        value = int(value)
    except Exception:
        raise TypeError(f"{name} must be int-like, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return np.int64(value)


def _vkey_from_arg(x):
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        name = x.split(".")[-1]
        return VirtualLaserKey[name]
    raise TypeError(f"Bad virtual laser key: {x!r}")


def get_seq(pulse_streamer, config, args):
    laser_on_ns, gate_width_ns, gate_offset_ns, laser_vkey_arg = args

    laser_on_ns = _as_int64("laser_on_ns", laser_on_ns)
    gate_width_ns = _as_int64("gate_width_ns", gate_width_ns)
    gate_offset_ns = int(gate_offset_ns)
    laser_vkey = _vkey_from_arg(laser_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]

    laser_name = tb.get_physical_laser_name(laser_vkey)
    laser_delay = int(config["Optics"]["PhysicalLasers"][laser_name]["delay"])

    base_buffer = max(200, laser_delay)

    if gate_offset_ns >= 0:
        laser_onset = base_buffer
        gate_onset = base_buffer + gate_offset_ns
    else:
        gate_onset = base_buffer
        laser_onset = base_buffer - gate_offset_ns

    laser_cmd_onset = laser_onset - laser_delay

    back_buffer = 500
    period = (
        max(
            int(laser_cmd_onset + laser_on_ns),
            int(gate_onset + gate_width_ns),
        )
        + back_buffer
    )

    seq = Sequence()

    # sample clock
    if period >= 300:
        clk_train = [(period - 200, LOW), (100, HIGH), (100, LOW)]
    else:
        clk_train = [(period, LOW)]
    seq.setDigital(do_sample_clock, clk_train)

    # APD gate
    apd_train = [
        (int(gate_onset), LOW),
        (int(gate_width_ns), HIGH),
        (int(period - gate_onset - gate_width_ns), LOW),
    ]
    seq.setDigital(do_apd_gate, apd_train)

    # Laser pulse
    laser_train = [
        (int(laser_cmd_onset), LOW),
        (int(laser_on_ns), HIGH),
        (int(period - laser_cmd_onset - laser_on_ns), LOW),
    ]
    tb.process_laser_seq(seq, laser_vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [period]


if __name__ == "__main__":
    from utils import common

    config = common.get_config_dict()
    args = [500, 300, 0, "SPIN_READOUT"]
    seq, final, ret = get_seq(None, config, args)
    print("Period (ns):", ret[0])
    seq.plot()
