# double_lifetime_recovery.py
#
# Double-pulse lifetime recovery sequence:
#
#   excitation pulse 1
#   optional readout delay
#   lifetime readout window 1, laser OFF, APD gate ON
#   dark recovery delay Δ, laser OFF, APD gate OFF
#   excitation pulse 2
#   optional readout delay
#   lifetime readout window 2, laser OFF, APD gate ON
#
# args = [
#     recovery_delay_ns,
#     exc_ns,
#     readout_delay_ns,
#     detect_ns,
#     laser_vkey,
#     laser_power,
# ]

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import tool_belt as tb
from utils.constants import Digital, VirtualLaserKey

LOW = Digital.LOW
HIGH = Digital.HIGH


def _as_int64(name, value):
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return np.int64(value)


def _vkey_from_arg(x):
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        return VirtualLaserKey[x.split(".")[-1]]
    raise TypeError(f"Bad virtual laser key: {x!r}")


def get_seq(pulse_streamer, config, args):
    (
        recovery_delay_ns,
        exc_ns,
        readout_delay_ns,
        detect_ns,
        laser_vkey_arg,
        laser_power,
    ) = args

    recovery_delay_ns = _as_int64("recovery_delay_ns", recovery_delay_ns)
    exc_ns = _as_int64("exc_ns", exc_ns)
    readout_delay_ns = _as_int64("readout_delay_ns", readout_delay_ns)
    detect_ns = _as_int64("detect_ns", detect_ns)
    laser_vkey = _vkey_from_arg(laser_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]

    laser_name = tb.get_physical_laser_name(laser_vkey)
    laser_delay = _as_int64(
        "laser_delay",
        config["Optics"]["PhysicalLasers"][laser_name]["delay"],
    )

    front_buffer = np.int64(laser_delay)
    meas_buffer = np.int64(1000)

    period = np.int64(
        front_buffer
        + exc_ns
        + readout_delay_ns
        + detect_ns
        + recovery_delay_ns
        + exc_ns
        + readout_delay_ns
        + detect_ns
        + meas_buffer
    )

    seq = Sequence()

    # Sample clock marker near the end of each repetition.
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # APD gate:
    # gate 0 -> readout 1
    # gate 1 -> readout 2
    apd_train = [
        (int(front_buffer), LOW),

        # excitation pulse 1
        (int(exc_ns), LOW),

        # optional wait after laser turns off
        (int(readout_delay_ns), LOW),

        # lifetime readout 1
        (int(detect_ns), HIGH),

        # dark recovery
        (int(recovery_delay_ns), LOW),

        # excitation pulse 2
        (int(exc_ns), LOW),

        # optional wait after laser turns off
        (int(readout_delay_ns), LOW),

        # lifetime readout 2
        (int(detect_ns), HIGH),

        # final buffer
        (int(meas_buffer), LOW),
    ]
    seq.setDigital(do_apd_gate, apd_train)

    # Laser is ON only during excitation pulses.
    laser_train = [
        (int(front_buffer), LOW),

        # excitation pulse 1
        (int(exc_ns), HIGH),

        # delay + readout 1, laser OFF
        (int(readout_delay_ns), LOW),
        (int(detect_ns), LOW),

        # dark recovery
        (int(recovery_delay_ns), LOW),

        # excitation pulse 2
        (int(exc_ns), HIGH),

        # delay + readout 2, laser OFF
        (int(readout_delay_ns), LOW),
        (int(detect_ns), LOW),

        # final buffer
        (int(meas_buffer), LOW),
    ]

    # This matches your current code style.
    # If your tool_belt supports passing laser_power directly, you can update this.
    tb.process_laser_seq(seq, laser_vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common

    cfg = common.get_config_dict()

    # args = [
    #     recovery_delay_ns,
    #     exc_ns,
    #     readout_delay_ns,
    #     detect_ns,
    #     laser_vkey,
    #     laser_power,
    # ]
    args = [5000, 50, 0, 500, "SPIN_READOUT", None]

    seq, final, ret = get_seq(None, cfg, args)
    print("Period ns:", ret[0])
    seq.plot()