# -*- coding: utf-8 -*-
"""
Confocal ESR pulse-streamer sequence.

Experiment 0 (signal):
    pol -> MW ON -> readout

Experiment 1 (reference):
    pol -> MW OFF -> readout

This sequence produces EXACTLY 2 APD gates per repetition:
    gate0 = signal
    gate1 = reference
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import tool_belt as tb
from utils.constants import Digital, ModMode, VirtualLaserKey

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


def _set_laser_train(seq, cfg, readout_vkey, train, readout_power=None):
    wiring = cfg["Wiring"]["PulseGen"]

    vld = cfg["Optics"]["VirtualLasers"][readout_vkey]
    laser_name = vld["physical_name"]
    pld = cfg["Optics"]["PhysicalLasers"][laser_name]
    mod_mode = pld["mod_mode"]

    if mod_mode is ModMode.DIGITAL:
        do_chan = wiring[f"do_{laser_name}_dm"]
        seq.setDigital(do_chan, train)
        return

    if mod_mode is ModMode.ANALOG:
        ao_chan = wiring[f"ao_{laser_name}_am"]
        if readout_power is None:
            if "laser_power" not in vld:
                raise ValueError(
                    f"{readout_vkey} is ANALOG but no readout_power provided and no "
                    f"'laser_power' in config."
                )
            readout_power = vld["laser_power"]

        power = float(readout_power)
        power_dict = {LOW: 0.0, HIGH: power}
        processed = [(int(d), power_dict[val]) for (d, val) in train]
        seq.setAnalog(ao_chan, processed)
        return

    raise ValueError(f"Unknown mod_mode for {laser_name}: {mod_mode!r}")


def get_seq(pulse_streamer, config, args):
    if len(args) == 3 and isinstance(args[0], (list, tuple)):
        base_args, mw_dur_ns, _num_reps_ignored = args
    elif len(args) == 2 and isinstance(args[0], (list, tuple)):
        base_args, mw_dur_ns = args
    else:
        raise ValueError(
            "Expected args as [base_args, mw_dur_ns] or [base_args, mw_dur_ns, num_reps]. "
            f"Got: {args!r}"
        )

    # base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_mw_dur_ns]
    pol_ns, readout_ns, uwave_ind_list, readout_vkey_arg, readout_power, max_mw_dur_ns = base_args

    pol_ns = _as_int64("pol_ns", pol_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)
    mw_dur_ns = _as_int64("mw_dur_ns", mw_dur_ns)
    max_mw_dur_ns = _as_int64("max_mw_dur_ns", max_mw_dur_ns)

    if mw_dur_ns > max_mw_dur_ns:
        max_mw_dur_ns = mw_dur_ns
    pad_ns = _as_int64("pad_ns", max_mw_dur_ns - mw_dur_ns)

    if isinstance(uwave_ind_list, (int, np.integer)):
        uwave_ind_list = [int(uwave_ind_list)]
    else:
        uwave_ind_list = [int(x) for x in uwave_ind_list]

    readout_vkey = _vkey_from_arg(readout_vkey_arg)

    wiring = config["Wiring"]["PulseGen"]
    do_apd_gate = wiring["do_apd_gate"]
    do_sample_clock = wiring["do_sample_clock"]

    laser_name = config["Optics"]["VirtualLasers"][readout_vkey]["physical_name"]
    laser_delay = _as_int64(
        "laser_delay",
        config["Optics"]["PhysicalLasers"][laser_name]["delay"],
    )

    uwave_delays = []
    do_sig_gen_dm_list = []
    for uwave_ind in uwave_ind_list:
        vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
        sig_gen_name = vsg["physical_name"]
        do_sig_gen_dm_list.append(wiring[f"do_{sig_gen_name}_dm"])
        uwave_delays.append(config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"])

    uwave_delay = _as_int64("uwave_delay", max(uwave_delays) if uwave_delays else 0)
    common_delay = np.int64(max(laser_delay, uwave_delay))
    uwave_buffer = _as_int64("uwave_buffer", config["CommonDurations"]["uwave_buffer"])

    # gate0 = signal, gate1 = reference
    apd_train = [
        (common_delay, LOW),

        # signal experiment
        (pol_ns, LOW),
        (uwave_buffer, LOW),
        (max_mw_dur_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),

        # reference experiment
        (uwave_buffer, LOW),
        (pol_ns, LOW),
        (uwave_buffer, LOW),
        (max_mw_dur_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),

        (uwave_buffer, LOW),
    ]

    laser_train = [
        (common_delay - laser_delay, LOW),

        # signal experiment
        (pol_ns, HIGH),
        (uwave_buffer, LOW),
        (max_mw_dur_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),

        # reference experiment
        (uwave_buffer, LOW),
        (pol_ns, HIGH),
        (uwave_buffer, LOW),
        (max_mw_dur_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),

        (uwave_buffer + laser_delay, LOW),
    ]

    mw_train = [
        (common_delay - uwave_delay, LOW),

        # signal experiment
        (pol_ns, LOW),
        (uwave_buffer, LOW),
        (mw_dur_ns, HIGH),
        (pad_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, LOW),

        # reference experiment
        (uwave_buffer, LOW),
        (pol_ns, LOW),
        (uwave_buffer, LOW),
        (max_mw_dur_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, LOW),

        (uwave_buffer + uwave_delay, LOW),
    ]

    seq = Sequence()
    seq.setDigital(do_apd_gate, apd_train)
    _set_laser_train(seq, config, readout_vkey, laser_train, readout_power=readout_power)

    for do_sig_gen_dm in do_sig_gen_dm_list:
        seq.setDigital(do_sig_gen_dm, mw_train)

    period_ns = np.int64(sum(int(d) for d, _ in apd_train))

    if period_ns >= 300:
        clk_train = [(int(period_ns - 200), LOW), (100, HIGH), (100, LOW)]
    else:
        clk_train = [(int(period_ns), LOW)]
    seq.setDigital(do_sample_clock, clk_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period_ns)]


if __name__ == "__main__":
    from utils import common

    cfg = common.get_config_dict()

    pol_ns = 10_000
    readout_ns = 300
    uwave_ind_list = [0]
    readout_vkey = "SPIN_READOUT"
    readout_power = None
    max_mw_dur_ns = 2000
    mw_dur_ns = 2000

    base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_mw_dur_ns]
    seq, final, ret = get_seq(None, cfg, [base_args, mw_dur_ns])

    print("Period (ns):", int(ret[0]))
    seq.plot()