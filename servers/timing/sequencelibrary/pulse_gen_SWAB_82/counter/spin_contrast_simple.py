# -*- coding: utf-8 -*-
"""
Simple spin-contrast test sequence, updated to match resonance.py style.

Two APD-gated experiments per repetition:
    gate 0 = reference (MW OFF)
    gate 1 = signal    (MW ON)

Args:
    [pol_ns, readout_ns, uwave_ind, spin_pol_vkey, readout_vkey, laser_power]

Notes:
    - Uses resonance.py-style timing/alignment
    - Pulls uwave_ns = pi_pulse from virtual sig gen config
    - Uses the readout laser path in the same simple way as resonance.py
"""

from pulsestreamer import Sequence, OutputState
import numpy as np

from utils import tool_belt as tb
from utils.constants import Digital, VirtualLaserKey

LOW = Digital.LOW
HIGH = Digital.HIGH


def _as_int64(name, v):
    try:
        iv = int(v)
    except Exception:
        raise TypeError(f"{name} must be int-like, got {type(v).__name__}: {v!r}")
    if iv < 0:
        raise ValueError(f"{name} must be >= 0, got {iv}")
    return np.int64(iv)


def _vkey_from_arg(x):
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        name = x.split(".")[-1]
        try:
            return VirtualLaserKey[name]
        except Exception:
            return VirtualLaserKey(x)
    raise TypeError(f"Bad virtual laser key: {x!r}")


def get_seq(pulse_streamer, config, args, num_reps=1):
    # Same argument style as resonance.py
    pol_ns, readout_ns, uwave_ind, spin_pol_vkey, readout_vkey, laser_power = args

    pol_ns = _as_int64("pol_ns", pol_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)
    uwave_ind = int(uwave_ind)
    readout_vkey = _vkey_from_arg(readout_vkey)

    pulser_wiring = config["Wiring"]["PulseGen"]

    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]

    # MW source from virtual sig gen
    vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
    uwave_ns = _as_int64("uwave_ns", vsg["pi_pulse"])
    sig_gen_name = vsg["physical_name"]
    uwave_delay = _as_int64(
        "uwave_delay",
        config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"],
    )
    do_sig_gen_gate = pulser_wiring[f"do_{sig_gen_name}_dm"]

    # Laser timing: match resonance.py simplicity
    laser_name = tb.get_physical_laser_name(readout_vkey)
    laser_delay = _as_int64(
        "laser_delay",
        config["Optics"]["PhysicalLasers"][laser_name]["delay"],
    )

    meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
    transient = np.int64(1000)

    front_buffer = np.int64(max(uwave_delay, laser_delay))

    period = np.int64(
        front_buffer
        + 2 * (pol_ns + transient + uwave_ns + transient + readout_ns + meas_buffer)
    )

    seq = Sequence()

    # Sample clock
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # APD gate: both readouts open
    apd_train = [
        (int(front_buffer - uwave_delay), LOW),
        (int(pol_ns), LOW),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), HIGH),
        (int(meas_buffer), LOW),

        # signal block
        (int(pol_ns), LOW),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), HIGH),
        (int(meas_buffer + uwave_delay), LOW),
    ]
    seq.setDigital(do_apd_gate, apd_train)

    # MW gate: OFF in first block, ON in second
    mw_train = [
        (int(front_buffer - uwave_delay), LOW),
        (int(pol_ns), LOW),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), LOW),
        (int(meas_buffer), LOW),

        # signal block
        (int(pol_ns), LOW),
        (int(transient), LOW),
        (int(uwave_ns), HIGH),
        (int(transient), LOW),
        (int(readout_ns), LOW),
        (int(meas_buffer + uwave_delay), LOW),
    ]
    seq.setDigital(do_sig_gen_gate, mw_train)

    # Laser train: same style as resonance.py
    laser_train = [
        (int(front_buffer - uwave_delay), LOW),
        (int(pol_ns), HIGH),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), HIGH),
        (int(meas_buffer), LOW),

        # signal block
        (int(pol_ns), HIGH),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), HIGH),
        (int(meas_buffer + uwave_delay), LOW),
    ]
    tb.process_laser_seq(seq, readout_vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common

    cfg = common.get_config_dict()
    args = [2000, 440, 0, "SPIN_POL", "SPIN_READOUT", None]

    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()