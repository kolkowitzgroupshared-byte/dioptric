# -*- coding: utf-8 -*-
"""
Ti:sapph AOM delay calibration sequence.

Identical to resonance_tisapph_singlet_scan.py structure but:
  - No MW (no pi pulse, no uwave gate)
  - 2 blocks instead of 4

Two APD-gated blocks per repetition:
    block 0 (ref) : green pol → transient → probe_ns (Ti:sapph OFF) → readout (APD HIGH)
    block 1 (sig) : green pol → transient → probe_ns (Ti:sapph ON)  → readout (APD HIGH)

The Ti:sapph AOM is offset by tisapph_aom_delay (from config) — same as singlet scan.

Small config_delay → AOM fires close to probe window → light arrives during probe → HIGH counts
Large config_delay → AOM fires too early → light misses probe window → LOW counts
Falling edge = real Ti:sapph AOM hardware delay

Args: [pol_ns, probe_ns, readout_ns, spin_pol_vkey, readout_vkey]

Created April 2026
@author: sbchand
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

import utils.tool_belt as tb
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
    pol_ns, probe_ns, readout_ns, spin_pol_vkey_arg, readout_vkey_arg = args

    pol_ns     = _as_int64("pol_ns",     pol_ns)
    probe_ns   = _as_int64("probe_ns",   probe_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)

    spin_pol_vkey = _vkey_from_arg(spin_pol_vkey_arg)
    readout_vkey  = _vkey_from_arg(readout_vkey_arg)

    pulser_wiring   = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate     = pulser_wiring["do_apd_gate"]
    do_tisapph_aom  = pulser_wiring["do_laser_TISAPPH_dm"]

    # Laser delays
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

    # Ti:sapph AOM delay — swept via cryo.py
    tisapph_aom_delay = _as_int64(
        "tisapph_aom_delay",
        config["Optics"]["PhysicalLasers"]["laser_TISAPPH"]["delay"],
    )

    meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
    transient   = np.int64(1000)

    # front_buffer — no MW so only laser and Ti:sapph delays matter
    front_buffer = np.int64(
        max(spin_pol_delay, readout_delay, tisapph_aom_delay)
    )

    # Block: pol → transient → probe → readout → meas_buffer (no MW)
    block_ns = np.int64(pol_ns + transient + probe_ns + readout_ns + meas_buffer)

    # 2 blocks: ref + sig
    period = np.int64(front_buffer + 2 * block_ns)

    print(
        f"[tisapph_delay_cal] tisapph_aom_delay={tisapph_aom_delay} ns  "
        f"pol={pol_ns} ns  probe={probe_ns} ns  "
        f"readout={readout_ns} ns  period={period} ns"
    )

    seq = Sequence()

    # -------------------- Sample clock --------------------
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # -------------------- APD gate --------------------
    apd_train = [(int(front_buffer), LOW)]
    for _ in range(2):
        apd_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(probe_ns), LOW),
            (int(readout_ns), HIGH),
            (int(meas_buffer), LOW),
        ])
    seq.setDigital(do_apd_gate, apd_train)

    # -------------------- Ti:sapph AOM --------------------
    # block 0 (ref): OFF
    # block 1 (sig): ON during probe_ns
    # Offset by tisapph_aom_delay — identical to singlet scan pattern (no MW terms)
    tisapph_train = [(int(front_buffer - tisapph_aom_delay), LOW)]
    for use_tisapph in [False, True]:
        tisapph_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(probe_ns), HIGH if use_tisapph else LOW),
            (int(readout_ns), LOW),
            (int(meas_buffer), LOW),
        ])
    tisapph_train.append((int(tisapph_aom_delay), LOW))
    seq.setDigital(do_tisapph_aom, tisapph_train)

    # -------------------- Green laser --------------------
    if spin_pol_laser_name == readout_laser_name:
        shared_delay = max(spin_pol_delay, readout_delay)
        laser_train = [(int(front_buffer - shared_delay), LOW)]
        for _ in range(2):
            laser_train.extend([
                (int(pol_ns), HIGH),
                (int(transient), LOW),
                (int(probe_ns), LOW),
                (int(readout_ns), HIGH),
                (int(meas_buffer), LOW),
            ])
        laser_train.append((int(shared_delay), LOW))
        tb.process_laser_seq(seq, readout_vkey, laser_train)
    else:
        spin_pol_train = [(int(front_buffer - spin_pol_delay), LOW)]
        for _ in range(2):
            spin_pol_train.extend([
                (int(pol_ns), HIGH),
                (int(transient), LOW),
                (int(probe_ns), LOW),
                (int(readout_ns), LOW),
                (int(meas_buffer), LOW),
            ])
        spin_pol_train.append((int(spin_pol_delay), LOW))
        tb.process_laser_seq(seq, spin_pol_vkey, spin_pol_train)

        readout_train = [(int(front_buffer - readout_delay), LOW)]
        for _ in range(2):
            readout_train.extend([
                (int(pol_ns), LOW),
                (int(transient), LOW),
                (int(probe_ns), LOW),
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
    # Args: [pol_ns, probe_ns, readout_ns, spin_pol_vkey, readout_vkey]
    args = [2000, 10000, 650, "SPIN_POL", "SPIN_READOUT"]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()
