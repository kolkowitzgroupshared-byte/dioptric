# # -*- coding: utf-8 -*-
# """
# Ti:sapph singlet scan sequence with Ti:sapph overlapping the readout.

# Four APD-gated experiments per repetition:
#     gate 0 = ms0, Ti:sapph OFF
#     gate 1 = ms0, Ti:sapph ON
#     gate 2 = ms1, Ti:sapph OFF
#     gate 3 = ms1, Ti:sapph ON

# Args:
#     [pol_ns, probe_ns, readout_ns, uwave_ind, spin_pol_vkey, readout_vkey]

# Behavior:
#     - probe_ns = 0   -> Ti:sapph overlaps exactly with readout
#     - probe_ns > 0   -> Ti:sapph turns on probe_ns earlier and stays on through readout
# """

# from pulsestreamer import Sequence, OutputState
# import numpy as np

# from utils import tool_belt as tb
# from utils.constants import Digital, VirtualLaserKey

# LOW = Digital.LOW
# HIGH = Digital.HIGH


# def _as_int64(name, v):
#     try:
#         iv = int(v)
#     except Exception:
#         raise TypeError(f"{name} must be int-like, got {type(v).__name__}: {v!r}")
#     if iv < 0:
#         raise ValueError(f"{name} must be >= 0, got {iv}")
#     return np.int64(iv)


# def _vkey_from_arg(x):
#     if isinstance(x, VirtualLaserKey):
#         return x
#     if isinstance(x, str):
#         name = x.split(".")[-1]
#         try:
#             return VirtualLaserKey[name]
#         except Exception:
#             return VirtualLaserKey(x)
#     raise TypeError(f"Bad virtual laser key: {x!r}")


# def get_seq(pulse_streamer, config, args, num_reps=1):
#     pol_ns, probe_ns, readout_ns, uwave_ind, spin_pol_vkey_arg, readout_vkey_arg = args

#     pol_ns = _as_int64("pol_ns", pol_ns)
#     probe_ns = _as_int64("probe_ns", probe_ns)
#     readout_ns = _as_int64("readout_ns", readout_ns)
#     uwave_ind = int(uwave_ind)

#     spin_pol_vkey = _vkey_from_arg(spin_pol_vkey_arg)
#     readout_vkey = _vkey_from_arg(readout_vkey_arg)

#     pulser_wiring = config["Wiring"]["PulseGen"]

#     do_sample_clock = pulser_wiring["do_sample_clock"]
#     do_apd_gate = pulser_wiring["do_apd_gate"]
#     do_tisapph_aom = pulser_wiring["do_laser_TISAPPH_dm"]

#     # MW source from virtual sig gen
#     vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
#     uwave_ns = _as_int64("uwave_ns", vsg["pi_pulse"])
#     sig_gen_name = vsg["physical_name"]
#     uwave_delay = _as_int64(
#         "uwave_delay",
#         config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"],
#     )
#     do_sig_gen_gate = pulser_wiring[f"do_{sig_gen_name}_dm"]

#     # Laser delays / physical names
#     spin_pol_laser_name = tb.get_physical_laser_name(spin_pol_vkey)
#     readout_laser_name = tb.get_physical_laser_name(readout_vkey)

#     spin_pol_delay = _as_int64(
#         "spin_pol_delay",
#         config["Optics"]["PhysicalLasers"][spin_pol_laser_name]["delay"],
#     )
#     readout_delay = _as_int64(
#         "readout_delay",
#         config["Optics"]["PhysicalLasers"][readout_laser_name]["delay"],
#     )

#     # Ti:sapph AOM timing offset
#     tisapph_aom_delay = _as_int64(
#         "tisapph_aom_delay",
#         config["Optics"]["PhysicalLasers"]["laser_TISAPPH"]["delay"],
#     )

#     meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
#     transient = np.int64(1000)

#     front_buffer = np.int64(
#         max(uwave_delay, spin_pol_delay, readout_delay, tisapph_aom_delay)
#     )

#     # one block = pol -> optional MW -> optional Ti:sapph probe(+overlap) -> readout
#     block_ns = np.int64(
#         pol_ns + transient +
#         uwave_ns + transient +
#         probe_ns + readout_ns + meas_buffer
#     )

#     period = np.int64(front_buffer + 4 * block_ns)

#     seq = Sequence()

#     # sample clock
#     clk_train = (
#         [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
#         if period >= 300
#         else [(int(period), LOW)]
#     )
#     seq.setDigital(do_sample_clock, clk_train)

#     # block logic
#     block_use_mw = [False, False, True, True]
#     block_use_tisapph = [False, True, False, True]

#     # ---------------- APD gate ----------------
#     apd_train = [(int(front_buffer), LOW)]
#     for _ in range(4):
#         apd_train.extend([
#             (int(pol_ns), LOW),
#             (int(transient), LOW),
#             (int(uwave_ns), LOW),
#             (int(transient), LOW),
#             (int(probe_ns), LOW),
#             (int(readout_ns), HIGH),
#             (int(meas_buffer), LOW),
#         ])
#     seq.setDigital(do_apd_gate, apd_train)

#     # ---------------- MW gate ----------------
#     mw_train = [(int(front_buffer - uwave_delay), LOW)]
#     for use_mw in block_use_mw:
#         mw_train.extend([
#             (int(pol_ns), LOW),
#             (int(transient), LOW),
#             (int(uwave_ns), HIGH if use_mw else LOW),
#             (int(transient), LOW),
#             (int(probe_ns), LOW),
#             (int(readout_ns), LOW),
#             (int(meas_buffer), LOW),
#         ])
#     mw_train.append((int(uwave_delay), LOW))
#     seq.setDigital(do_sig_gen_gate, mw_train)

#     # ---------------- Ti:sapph AOM gate ----------------
#     # Overlap version:
#     # OFF blocks: LOW during both probe and readout
#     # ON blocks : HIGH during both probe and readout
#     tisapph_train = [(int(front_buffer - tisapph_aom_delay), LOW)]
#     for use_tisapph in block_use_tisapph:
#         tisapph_train.extend([
#             (int(pol_ns), LOW),
#             (int(transient), LOW),
#             (int(uwave_ns), LOW),
#             (int(transient), LOW),
#             (int(probe_ns), HIGH if use_tisapph else LOW),
#             (int(readout_ns), HIGH if use_tisapph else LOW),
#             (int(meas_buffer), LOW),
#         ])
#     tisapph_train.append((int(tisapph_aom_delay), LOW))
#     seq.setDigital(do_tisapph_aom, tisapph_train)

#     # ---------------- Spin/readout laser(s) ----------------
#     if spin_pol_laser_name == readout_laser_name:
#         shared_delay = max(spin_pol_delay, readout_delay)

#         combined_laser_train = [(int(front_buffer - shared_delay), LOW)]
#         for _ in range(4):
#             combined_laser_train.extend([
#                 (int(pol_ns), HIGH),   # polarization pulse
#                 (int(transient), LOW),
#                 (int(uwave_ns), LOW),
#                 (int(transient), LOW),
#                 (int(probe_ns), LOW),
#                 (int(readout_ns), HIGH),  # readout pulse
#                 (int(meas_buffer), LOW),
#             ])
#         combined_laser_train.append((int(shared_delay), LOW))

#         tb.process_laser_seq(seq, readout_vkey, combined_laser_train)

#     else:
#         spin_pol_train = [(int(front_buffer - spin_pol_delay), LOW)]
#         for _ in range(4):
#             spin_pol_train.extend([
#                 (int(pol_ns), HIGH),
#                 (int(transient), LOW),
#                 (int(uwave_ns), LOW),
#                 (int(transient), LOW),
#                 (int(probe_ns), LOW),
#                 (int(readout_ns), LOW),
#                 (int(meas_buffer), LOW),
#             ])
#         spin_pol_train.append((int(spin_pol_delay), LOW))
#         tb.process_laser_seq(seq, spin_pol_vkey, spin_pol_train)

#         readout_train = [(int(front_buffer - readout_delay), LOW)]
#         for _ in range(4):
#             readout_train.extend([
#                 (int(pol_ns), LOW),
#                 (int(transient), LOW),
#                 (int(uwave_ns), LOW),
#                 (int(transient), LOW),
#                 (int(probe_ns), LOW),
#                 (int(readout_ns), HIGH),
#                 (int(meas_buffer), LOW),
#             ])
#         readout_train.append((int(readout_delay), LOW))
#         tb.process_laser_seq(seq, readout_vkey, readout_train)

#     final = OutputState([], 0.0, 0.0)
#     return seq, final, [int(period)]


# if __name__ == "__main__":
#     from utils import common

#     cfg = common.get_config_dict()

#     # probe_ns = 0 -> exact overlap of Ti:sapph and readout
#     args = [2000, 0, 440, 0, "SPIN_POL", "SPIN_READOUT"]

#     seq, final, ret = get_seq(None, cfg, args)
#     print("Period (ns):", ret[0])
#     seq.plot()

# -*- coding: utf-8 -*-
"""
Ti:sapph singlet scan sequence.

Four APD-gated experiments per repetition:
    gate 0 = ms0, Ti:sapph OFF
    gate 1 = ms0, Ti:sapph ON
    gate 2 = ms1, Ti:sapph OFF
    gate 3 = ms1, Ti:sapph ON

Args:
    [pol_ns, probe_ns, readout_ns, uwave_ind, spin_pol_vkey, readout_vkey]
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
        config["Optics"]["PhysicalLasers"]["laser_TISAPPH"]["delay"],
    )

    meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
    transient = np.int64(1000)

    front_buffer = np.int64(
        max(uwave_delay, spin_pol_delay, readout_delay, tisapph_aom_delay)
    )

    # one block = pol -> optional MW -> optional Ti:sapph probe -> readout
    block_ns = np.int64(
        pol_ns + transient +
        uwave_ns + transient +
        probe_ns + readout_ns + meas_buffer
    )

    period = np.int64(front_buffer + 4 * block_ns)

    seq = Sequence()

    # sample clock
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # block logic
    block_use_mw = [False, False, True, True]
    block_use_tisapph = [False, True, False, True]

    # ---------------- APD gate ----------------
    apd_train = [(int(front_buffer), LOW)]
    for _ in range(4):
        apd_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(uwave_ns), LOW),
            (int(transient), LOW),
            (int(probe_ns), LOW),
            (int(readout_ns), HIGH),
            (int(meas_buffer), LOW),
        ])
    seq.setDigital(do_apd_gate, apd_train)

    # ---------------- MW gate ----------------
    mw_train = [(int(front_buffer - uwave_delay), LOW)]
    for use_mw in block_use_mw:
        mw_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(uwave_ns), HIGH if use_mw else LOW),
            (int(transient), LOW),
            (int(probe_ns), LOW),
            (int(readout_ns), LOW),
            (int(meas_buffer), LOW),
        ])
    mw_train.append((int(uwave_delay), LOW))
    seq.setDigital(do_sig_gen_gate, mw_train)

    # ---------------- Ti:sapph AOM gate ----------------
    tisapph_train = [(int(front_buffer - tisapph_aom_delay), LOW)]
    for use_tisapph in block_use_tisapph:
        tisapph_train.extend([
            (int(pol_ns), LOW),
            (int(transient), LOW),
            (int(uwave_ns), LOW),
            (int(transient), LOW),
            (int(probe_ns), HIGH if use_tisapph else LOW),
            (int(readout_ns), LOW),
            (int(meas_buffer), LOW),
        ])
    tisapph_train.append((int(tisapph_aom_delay), LOW))
    seq.setDigital(do_tisapph_aom, tisapph_train)

    # ---------------- Spin/readout laser(s) ----------------
    # If spin polarization and readout use the same physical laser channel,
    # combine them into one train so the second call does not overwrite the first.
    if spin_pol_laser_name == readout_laser_name:
        shared_delay = max(spin_pol_delay, readout_delay)

        combined_laser_train = [(int(front_buffer - shared_delay), LOW)]
        for _ in range(4):
            combined_laser_train.extend([
                (int(pol_ns), HIGH),   # polarization pulse
                (int(transient), LOW),
                (int(uwave_ns), LOW),
                (int(transient), LOW),
                (int(probe_ns), LOW),
                (int(readout_ns), HIGH),  # readout pulse
                (int(meas_buffer), LOW),
            ])
        combined_laser_train.append((int(shared_delay), LOW))

        tb.process_laser_seq(seq, readout_vkey, combined_laser_train)

    else:
        spin_pol_train = [(int(front_buffer - spin_pol_delay), LOW)]
        for _ in range(4):
            spin_pol_train.extend([
                (int(pol_ns), HIGH),
                (int(transient), LOW),
                (int(uwave_ns), LOW),
                (int(transient), LOW),
                (int(probe_ns), LOW),
                (int(readout_ns), LOW),
                (int(meas_buffer), LOW),
            ])
        spin_pol_train.append((int(spin_pol_delay), LOW))
        tb.process_laser_seq(seq, spin_pol_vkey, spin_pol_train)

        readout_train = [(int(front_buffer - readout_delay), LOW)]
        for _ in range(4):
            readout_train.extend([
                (int(pol_ns), LOW),
                (int(transient), LOW),
                (int(uwave_ns), LOW),
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
    args = [2000, 500, 440, 0, "SPIN_POL", "SPIN_READOUT"]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()