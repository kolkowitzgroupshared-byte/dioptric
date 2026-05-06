# -*- coding: utf-8 -*-
"""
Short ODMR sequence with Ti:Sapph singlet shelving.

Two blocks per repetition:
    gate 0 = reference: Green Pol → MW pi → (no Ti:Sapph) → Green Readout
    gate 1 = signal:    Green Pol → MW pi → Ti:Sapph     → Green Readout

The MW frequency is swept by the main routine between stream_start calls.
The sequence itself is loaded once and reused for all frequencies.

Args:
    [pol_ns, probe_ns, readout_ns, uwave_ind, spin_pol_vkey, readout_vkey]
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
        try:
            return VirtualLaserKey[name]
        except Exception:
            return VirtualLaserKey(x)
    raise TypeError(f"Bad virtual laser key: {x!r}")


def get_seq(pulse_streamer, config, args, num_reps=1):
    pol_ns, probe_ns, readout_ns, uwave_ind, spin_pol_vkey_arg, readout_vkey_arg = args

    pol_ns = _as_int64("pol_ns", pol_ns)
    probe_ns = _as_int64("probe_ns", probe_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)
    uwave_ind = int(uwave_ind)

    spin_pol_vkey = _vkey_from_arg(spin_pol_vkey_arg)
    readout_vkey = _vkey_from_arg(readout_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]

    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]
    do_tisapph_aom = pulser_wiring["do_laser_TISAPPH_dm"]

    # MW source from virtual sig gen
    vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
    uwave_ns = _as_int64("uwave_ns", vsg["pi_pulse"])
    sig_gen_name = vsg["physical_name"]
    uwave_delay = _as_int64(
        "uwave_delay",
        config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"],
    )
    do_sig_gen_gate = pulser_wiring[f"do_{sig_gen_name}_dm"]

    # Laser delays / physical names
    spin_pol_laser_name = tb.get_physical_laser_name(spin_pol_vkey)
    readout_laser_name = tb.get_physical_laser_name(readout_vkey)

    spin_pol_delay = _as_int64(
        "spin_pol_delay",
        config["Optics"]["PhysicalLasers"][spin_pol_laser_name]["delay"],
    )
    readout_delay = _as_int64(
        "readout_delay",
        config["Optics"]["PhysicalLasers"][readout_laser_name]["delay"],
    )

    # Ti:sapph AOM timing offset
    tisapph_aom_delay = _as_int64(
        "tisapph_aom_delay",
        pulser_wiring.get("do_tisapph_aom_delay", 0),
    )

    meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
    transient = np.int64(1000)
    # Buffer after Ti:Sapph probe to let AOM fully close before readout
    probe_buffer = np.int64(5000)  # 5 μs — adjust if Ti:Sapph AOM is slower

    front_buffer = np.int64(
        max(uwave_delay, spin_pol_delay, readout_delay, tisapph_aom_delay)
    )

    # One block: pol → transient → MW pi → transient → probe → probe_buffer → readout → meas_buffer
    block_ns = np.int64(
        pol_ns + transient + uwave_ns + transient + probe_ns + probe_buffer + readout_ns + meas_buffer
    )

    period = np.int64(front_buffer + 2 * block_ns)

    seq = Sequence()

    # ---- Sample clock ----
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # ---- Block logic ----
    # Block 0 = reference (no Ti:Sapph), Block 1 = signal (Ti:Sapph ON)
    block_use_tisapph = [False, True]

    # ---- APD gate ----
    apd_train = [(int(front_buffer), LOW)]
    for _ in range(2):
        apd_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(uwave_ns), LOW),
            (int(transient), LOW),
            (int(probe_ns), LOW),
            (int(probe_buffer), LOW),
            (int(readout_ns), HIGH),
            (int(meas_buffer), LOW),
        ])
    seq.setDigital(do_apd_gate, apd_train)

    # ---- MW gate (ON in both blocks — pi pulse) ----
    mw_train = [(int(front_buffer - uwave_delay), LOW)]
    for _ in range(2):
        mw_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(uwave_ns), HIGH),
            (int(transient), LOW),
            (int(probe_ns), LOW),
            (int(probe_buffer), LOW),
            (int(readout_ns), LOW),
            (int(meas_buffer), LOW),
        ])
    mw_train.append((int(uwave_delay), LOW))
    seq.setDigital(do_sig_gen_gate, mw_train)

    # ---- Ti:Sapph AOM gate (OFF in ref, ON in signal) ----
    tisapph_train = [(int(front_buffer - tisapph_aom_delay), LOW)]
    for use_tisapph in block_use_tisapph:
        tisapph_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(uwave_ns), LOW),
            (int(transient), LOW),
            (int(probe_ns), HIGH if use_tisapph else LOW),
            (int(probe_buffer), LOW),
            (int(readout_ns), LOW),
            (int(meas_buffer), LOW),
        ])
    tisapph_train.append((int(tisapph_aom_delay), LOW))
    seq.setDigital(do_tisapph_aom, tisapph_train)

    # ---- Green laser (pol + readout in both blocks) ----
    if spin_pol_laser_name == readout_laser_name:
        shared_delay = max(spin_pol_delay, readout_delay)
        laser_train = [(int(front_buffer - shared_delay), LOW)]
        for _ in range(2):
            laser_train.extend([
                (int(pol_ns), HIGH),
                (int(transient), LOW),
                (int(uwave_ns), LOW),
                (int(transient), LOW),
                (int(probe_ns), LOW),
                (int(probe_buffer), LOW),
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
                (int(uwave_ns), LOW),
                (int(transient), LOW),
                (int(probe_ns), LOW),
                (int(probe_buffer), LOW),
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
                (int(uwave_ns), LOW),
                (int(transient), LOW),
                (int(probe_ns), LOW),
                (int(probe_buffer), LOW),
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
    args = [2000, 100000, 440, 0, "SPIN_POL", "SPIN_READOUT"]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()
