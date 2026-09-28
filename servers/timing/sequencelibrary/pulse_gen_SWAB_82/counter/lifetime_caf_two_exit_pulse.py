""" """

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
    wait_ns, pol_ns, gap_ns, exc_ns, detect_ns, laser_vkey_arg, laser_power = args

    wait_ns = _as_int64("wait_ns", wait_ns)
    pol_ns = _as_int64("pol_ns", pol_ns)
    gap_ns = _as_int64("gap_ns", gap_ns)
    exc_ns = _as_int64("exc_ns", exc_ns)
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

    meas_buffer = np.int64(1000)
    front_buffer = np.int64(laser_delay)

    period = np.int64(front_buffer + wait_ns + pol_ns + detect_ns + meas_buffer)

    seq = Sequence()

    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # gate 0 -> readout 1
    # gate 1 -> readout 2
    apd_train = [
        (int(front_buffer), LOW),
        (int(wait_ns), LOW),
        (int(pol_ns), LOW),
        (int(detect_ns), HIGH),
        (int(meas_buffer), LOW),
    ]
    seq.setDigital(do_apd_gate, apd_train)

    laser_train = [
        (int(front_buffer), LOW),
        (int(wait_ns), LOW),
        (int(pol_ns), HIGH),
        (int(gap_ns), LOW),
        (int(exc_ns), HIGH),
        (int(detect_ns - gap_ns - exc_ns), LOW),
        (int(meas_buffer), LOW),
    ]
    tb.process_laser_seq(seq, laser_vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common

    cfg = common.get_config_dict()

    # args = [readout_delay_ns, exc_ns, detect_ns, laser_vkey, laser_power]
    args = [0, 1e3, 100, 100, 1e3, "SPIN_READOUT", None]
    # ^ first arg should be a variable
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()
