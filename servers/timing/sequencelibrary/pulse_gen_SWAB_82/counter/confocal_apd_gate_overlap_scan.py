# -*- coding: utf-8 -*-
"""
APD-gate timing sweep referenced to the *physical* end of the readout
laser pulse.

Args:
    [laser_on_ns, gate_width_ns, gate_delay_ns, laser_vkey,
     laser_fall_delay_ns, apd_gate_delay_ns]

Timing definitions (all physical, i.e. at the device output, not at the
command pin):

    t_cmd_laser_on   -- command HIGH goes out to the laser.
    t_phys_laser_off = t_cmd_laser_on + laser_on_ns + laser_fall_delay_ns
    t_phys_gate_on   = t_cmd_gate_on  + apd_gate_delay_ns
    gate_delay_ns    = t_phys_gate_on - t_phys_laser_off  (swept)

So gate_delay_ns > 0 means the APD physically opens AFTER the laser has
physically turned off; gate_delay_ns = 0 means they coincide;
gate_delay_ns < 0 means the APD physically opens while the laser is
still on.

`laser_fall_delay_ns` defaults are the caller's responsibility -- the
majorroutine pulls the laser's rising-edge delay from the config and
passes it here; override in the caller if the falling edge has been
characterized separately.
"""

from pulsestreamer import Sequence, OutputState
import numpy as np

from utils import tool_belt as tb
from utils.constants import Digital, VirtualLaserKey

LOW = Digital.LOW
HIGH = Digital.HIGH


def _as_int64_nonneg(name, value):
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
    (
        laser_on_ns,
        gate_width_ns,
        gate_delay_ns,
        laser_vkey_arg,
        laser_fall_delay_ns,
        apd_gate_delay_ns,
    ) = args

    laser_on_ns = _as_int64_nonneg("laser_on_ns", laser_on_ns)
    gate_width_ns = _as_int64_nonneg("gate_width_ns", gate_width_ns)
    laser_fall_delay_ns = _as_int64_nonneg(
        "laser_fall_delay_ns", laser_fall_delay_ns
    )
    apd_gate_delay_ns = _as_int64_nonneg(
        "apd_gate_delay_ns", apd_gate_delay_ns
    )
    gate_delay_ns = int(gate_delay_ns)
    laser_vkey = _vkey_from_arg(laser_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]

    # Everything below is on the *command* timeline. The physical laser-off
    # edge lands laser_fall_delay_ns after the command falls; the physical
    # APD-gate-open edge lands apd_gate_delay_ns after the command rises.
    # The laser's rising-edge delay doesn't enter this calculation because
    # we only care about the fall-to-gate relationship.
    base_buffer = 200
    laser_cmd_onset = base_buffer

    # Solve t_phys_gate_on - t_phys_laser_off = gate_delay_ns:
    #   t_cmd_gate_on = laser_cmd_onset + laser_on_ns
    #                   + laser_fall_delay_ns + gate_delay_ns - apd_gate_delay_ns
    gate_cmd_onset = (
        int(laser_cmd_onset)
        + int(laser_on_ns)
        + int(laser_fall_delay_ns)
        + int(gate_delay_ns)
        - int(apd_gate_delay_ns)
    )

    # If gate_delay_ns is very negative the gate command could land
    # before t=0; shift the whole sequence forward.
    if gate_cmd_onset < 0:
        shift = -gate_cmd_onset
        laser_cmd_onset += shift
        gate_cmd_onset = 0

    back_buffer = 500
    period = (
        max(
            # Physical laser-off edge lives inside the sequence window.
            int(laser_cmd_onset + laser_on_ns + laser_fall_delay_ns),
            # Physical APD-gate-close edge lives inside the sequence window.
            int(gate_cmd_onset + gate_width_ns + apd_gate_delay_ns),
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

    # APD gate (command timeline; physical opening is shifted by
    # apd_gate_delay_ns, which the pulser can't see)
    apd_train = [
        (int(gate_cmd_onset), LOW),
        (int(gate_width_ns), HIGH),
        (int(period - gate_cmd_onset - gate_width_ns), LOW),
    ]
    seq.setDigital(do_apd_gate, apd_train)

    # Laser pulse (command timeline; physical onset is laser_rise_delay_ns
    # later, physical offset is laser_fall_delay_ns later)
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
    args = [500, 300, 0, "SPIN_READOUT", 960, 0]
    seq, final, ret = get_seq(None, config, args)
    print("Period (ns):", ret[0])
    seq.plot()
