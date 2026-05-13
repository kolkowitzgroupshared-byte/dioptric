# -*- coding: utf-8 -*-
"""
Ti:sapph -> dark gap -> readout sequence WITHOUT microwaves.

Two APD-gated experiments per repetition:
    gate 0 = reference: Ti:sapph OFF (green readout only)
    gate 1 = signal:    Ti:sapph ON for tisapph_ns, then dark gap delay_ns, then readout

Block layout (per gate):
    pol -> transient -> tisapph_ns -> delay_ns -> readout -> meas_buffer

    tisapph_ns : fixed Ti:sapph illumination duration
    delay_ns   : dark gap from TiSapph-off to APD-gate-open  <-- the swept variable

No microwave channel is driven.

Args:
    [pol_ns, tisapph_ns, delay_ns, readout_ns, spin_pol_vkey, readout_vkey]
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
    (pol_ns, tisapph_ns, delay_ns, readout_ns,
     spin_pol_vkey_arg, readout_vkey_arg) = args

    pol_ns     = _as_int64("pol_ns", pol_ns)
    tisapph_ns = _as_int64("tisapph_ns", tisapph_ns)
    delay_ns   = _as_int64("delay_ns", delay_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)

    spin_pol_vkey = _vkey_from_arg(spin_pol_vkey_arg)
    readout_vkey  = _vkey_from_arg(readout_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate     = pulser_wiring["do_apd_gate"]
    do_tisapph_aom  = pulser_wiring["do_laser_TISAPPH_dm"]

    # Laser delays / physical names
    spin_pol_laser_name = tb.get_physical_laser_name(spin_pol_vkey)
    readout_laser_name  = tb.get_physical_laser_name(readout_vkey)

    spin_pol_delay = _as_int64(
        "spin_pol_delay",
        config["Optics"]["PhysicalLasers"][spin_pol_laser_name]["delay"],
    )
    readout_delay = _as_int64(
        "readout_delay",
        config["Optics"]["PhysicalLasers"][readout_laser_name]["delay"],
    )
    tisapph_aom_delay = _as_int64(
        "tisapph_aom_delay",
        config["Optics"]["PhysicalLasers"]["laser_TISAPPH"]["delay"],
    )

    meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
    transient = np.int64(200)  # match working rabi.py

    front_buffer = np.int64(
        max(spin_pol_delay, readout_delay, tisapph_aom_delay)
    )

    # One block: pol | transient | tisapph_ns | delay_ns | readout_ns | meas_buffer
    block_ns = np.int64(
        pol_ns + transient
        + tisapph_ns + delay_ns + readout_ns
        + meas_buffer
    )
    period = np.int64(front_buffer + 2 * block_ns)

    seq = Sequence()

    # Sample clock: one short pulse at end of period
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    block_use_tisapph = [False, True]  # gate 0 = ref, gate 1 = signal

    # ---------------- APD gate ----------------
    apd_train = [(int(front_buffer), LOW)]
    for _ in range(2):
        apd_train.extend([
            (int(pol_ns),     LOW),
            (int(transient),  LOW),
            (int(tisapph_ns), LOW),
            (int(delay_ns),   LOW),
            (int(readout_ns), HIGH),
            (int(meas_buffer), LOW),
        ])
    seq.setDigital(do_apd_gate, apd_train)

    # ---------------- Ti:sapph AOM gate ----------------
    tisapph_train = [(int(front_buffer - tisapph_aom_delay), LOW)]
    for use_tisapph in block_use_tisapph:
        tisapph_train.extend([
            (int(pol_ns),     LOW),
            (int(transient),  LOW),
            (int(tisapph_ns), HIGH if use_tisapph else LOW),
            (int(delay_ns),   LOW),
            (int(readout_ns), LOW),
            (int(meas_buffer), LOW),
        ])
    tisapph_train.append((int(tisapph_aom_delay), LOW))
    seq.setDigital(do_tisapph_aom, tisapph_train)

    # ---------------- Pol + readout laser ----------------
    if spin_pol_laser_name == readout_laser_name:
        shared_delay = max(spin_pol_delay, readout_delay)
        combined = [(int(front_buffer - shared_delay), LOW)]
        for _ in range(2):
            combined.extend([
                (int(pol_ns),     HIGH),
                (int(transient),  LOW),
                (int(tisapph_ns), LOW),
                (int(delay_ns),   LOW),
                (int(readout_ns), HIGH),
                (int(meas_buffer), LOW),
            ])
        combined.append((int(shared_delay), LOW))
        tb.process_laser_seq(seq, readout_vkey, combined)
    else:
        spin_pol_train = [(int(front_buffer - spin_pol_delay), LOW)]
        for _ in range(2):
            spin_pol_train.extend([
                (int(pol_ns),     HIGH),
                (int(transient),  LOW),
                (int(tisapph_ns), LOW),
                (int(delay_ns),   LOW),
                (int(readout_ns), LOW),
                (int(meas_buffer), LOW),
            ])
        spin_pol_train.append((int(spin_pol_delay), LOW))
        tb.process_laser_seq(seq, spin_pol_vkey, spin_pol_train)

        readout_train = [(int(front_buffer - readout_delay), LOW)]
        for _ in range(2):
            readout_train.extend([
                (int(pol_ns),     LOW),
                (int(transient),  LOW),
                (int(tisapph_ns), LOW),
                (int(delay_ns),   LOW),
                (int(readout_ns), HIGH),
                (int(meas_buffer), LOW),
            ])
        readout_train.append((int(readout_delay), LOW))
        tb.process_laser_seq(seq, readout_vkey, readout_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common
    cfg = common.get_config_dict()
    # Example: pol=2us, tisapph=20us, delay=1us, readout=440ns
    args = [2000, 20000, 1000, 440, "SPIN_POL", "SPIN_READOUT"]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()
