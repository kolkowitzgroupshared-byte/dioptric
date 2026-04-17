# -*- coding: utf-8 -*-
"""
Spin-contrast sequence with decoupled APD gate width and laser readout
duration.  Based on spin_contrast_simple.py but the APD gate can be
shorter than (and delayed relative to) the laser readout pulse.

Two APD-gated experiments per repetition:
    gate 0 = reference (MW OFF)
    gate 1 = signal    (MW ON)

Args:
    [pol_ns, laser_on_ns, gate_width_ns, gate_delay_ns,
     uwave_ind, spin_pol_vkey, readout_vkey, laser_power]

gate_delay_ns: offset from the readout-laser rising edge to APD gate open.
gate_width_ns: how long the APD gate stays HIGH.
Constraint: gate_delay_ns + gate_width_ns <= laser_on_ns.
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
    (
        pol_ns,
        laser_on_ns,
        gate_width_ns,
        gate_delay_ns,
        uwave_ind,
        spin_pol_vkey,
        readout_vkey,
        laser_power,
    ) = args

    pol_ns = _as_int64("pol_ns", pol_ns)
    laser_on_ns = _as_int64("laser_on_ns", laser_on_ns)
    gate_width_ns = _as_int64("gate_width_ns", gate_width_ns)
    gate_delay_ns = _as_int64("gate_delay_ns", gate_delay_ns)
    uwave_ind = int(uwave_ind)
    readout_vkey = _vkey_from_arg(readout_vkey)

    if gate_delay_ns + gate_width_ns > laser_on_ns:
        raise ValueError(
            f"gate_delay_ns ({gate_delay_ns}) + gate_width_ns ({gate_width_ns}) "
            f"> laser_on_ns ({laser_on_ns})"
        )

    gate_tail_ns = np.int64(laser_on_ns - gate_delay_ns - gate_width_ns)

    pulser_wiring = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]

    vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
    uwave_ns = _as_int64("uwave_ns", vsg["pi_pulse"])
    sig_gen_name = vsg["physical_name"]
    uwave_delay = _as_int64(
        "uwave_delay",
        config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"],
    )
    do_sig_gen_gate = pulser_wiring[f"do_{sig_gen_name}_dm"]

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
        + 2 * (pol_ns + transient + uwave_ns + transient + laser_on_ns + meas_buffer)
    )

    seq = Sequence()

    # Sample clock
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # APD gate: decoupled from laser — gate opens at gate_delay_ns into
    # each readout block and stays HIGH for gate_width_ns.
    def _apd_readout_block():
        segs = []
        if gate_delay_ns > 0:
            segs.append((int(gate_delay_ns), LOW))
        segs.append((int(gate_width_ns), HIGH))
        if gate_tail_ns > 0:
            segs.append((int(gate_tail_ns), LOW))
        return segs

    apd_train = [
        (int(front_buffer - uwave_delay), LOW),
        (int(pol_ns), LOW),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        *_apd_readout_block(),
        (int(meas_buffer), LOW),

        # signal block
        (int(pol_ns), LOW),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        *_apd_readout_block(),
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
        (int(laser_on_ns), LOW),
        (int(meas_buffer), LOW),

        # signal block
        (int(pol_ns), LOW),
        (int(transient), LOW),
        (int(uwave_ns), HIGH),
        (int(transient), LOW),
        (int(laser_on_ns), LOW),
        (int(meas_buffer + uwave_delay), LOW),
    ]
    seq.setDigital(do_sig_gen_gate, mw_train)

    # Laser train: on for laser_on_ns (not gate_width_ns)
    laser_train = [
        (int(front_buffer - uwave_delay), LOW),
        (int(pol_ns), HIGH),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        (int(laser_on_ns), HIGH),
        (int(meas_buffer), LOW),

        # signal block
        (int(pol_ns), HIGH),
        (int(transient), LOW),
        (int(uwave_ns), LOW),
        (int(transient), LOW),
        (int(laser_on_ns), HIGH),
        (int(meas_buffer + uwave_delay), LOW),
    ]
    tb.process_laser_seq(seq, readout_vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common

    cfg = common.get_config_dict()
    # laser_on=1000, gate_width=300, gate_delay=50
    args = [2000, 1000, 300, 50, 0, "SPIN_POL", "SPIN_READOUT", None]

    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()
