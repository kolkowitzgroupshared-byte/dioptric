# -*- coding: utf-8 -*-
"""
Single-point Rabi sequence.

Two gated measurements per repetition:
    gate 0 = reference  (no MW pulse)
    gate 1 = signal     (MW pulse of duration tau_ns)

Args:
    [tau_ns, polarization_ns, readout_ns, uwave_ind, pol_vkey, readout_vkey, laser_power]
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
        return VirtualLaserKey[name]
    raise TypeError(f"Bad virtual laser key: {x!r}")


def get_seq(pulse_streamer, config, args):
    # args = [tau_ns, polarization_ns, readout_ns, uwave_ind, pol_vkey, readout_vkey, laser_power]
    tau_ns, polarization_ns, readout_ns, uwave_ind, pol_vkey_arg, readout_vkey_arg, laser_power = args

    tau_ns = _as_int64("tau_ns", tau_ns)
    polarization_ns = _as_int64("polarization_ns", polarization_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)
    uwave_ind = int(uwave_ind)

    readout_vkey = _vkey_from_arg(readout_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]

    # Microwave channel
    vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = vsg["physical_name"]
    do_sig_gen_gate = pulser_wiring[f"do_{sig_gen_name}_dm"]
    uwave_delay = _as_int64(
        "uwave_delay",
        config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"],
    )

    # Laser timing — use readout_vkey (same approach as working resonance.py)
    laser_name = tb.get_physical_laser_name(readout_vkey)
    laser_delay = _as_int64(
        "laser_delay",
        config["Optics"]["PhysicalLasers"][laser_name]["delay"],
    )

    # Buffers — match resonance.py exactly
    common_durations = config["CommonDurations"]
    meas_buffer = _as_int64("meas_buffer", common_durations["cw_meas_buffer"])
    transient = np.int64(200)  # Hardcoded 200 ns, same as resonance.py

    # Same front_buffer and alignment as resonance.py
    front_buffer = np.int64(max(uwave_delay, laser_delay))
    period = np.int64(
        front_buffer + 2 * (polarization_ns + tau_ns + transient + transient + readout_ns + meas_buffer)
    )

    print(f"[rabi.py] tau={tau_ns}, pol={polarization_ns}, read={readout_ns}, period={period}")

    seq = Sequence()

    # ------------------------------------------------------------------
    # Sample clock
    # ------------------------------------------------------------------
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # ------------------------------------------------------------------
    # APD gate — ALL channels use same starting delay (resonance.py style)
    # ------------------------------------------------------------------
    apd_train = [
        (int(front_buffer - uwave_delay), LOW),

        # Reference experiment
        (int(polarization_ns), LOW),
        (int(transient), LOW),
        (int(tau_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), HIGH),
        (int(meas_buffer), LOW),

        # Signal experiment
        (int(polarization_ns), LOW),
        (int(transient), LOW),
        (int(tau_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), HIGH),
        (int(meas_buffer + uwave_delay), LOW),
    ]
    seq.setDigital(do_apd_gate, apd_train)

    # ------------------------------------------------------------------
    # Microwave gate — OFF in reference, ON during signal tau
    # ------------------------------------------------------------------
    mw_train = [
        (int(front_buffer - uwave_delay), LOW),

        # Reference experiment — MW OFF
        (int(polarization_ns), LOW),
        (int(transient), LOW),
        (int(tau_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), LOW),
        (int(meas_buffer), LOW),

        # Signal experiment — MW ON during tau only
        (int(polarization_ns), LOW),
        (int(transient), LOW),
        (int(tau_ns), HIGH),
        (int(transient), LOW),
        (int(readout_ns), LOW),
        (int(meas_buffer + uwave_delay), LOW),
    ]
    seq.setDigital(do_sig_gen_gate, mw_train)

    # ------------------------------------------------------------------
    # Laser train — same alignment as APD and MW
    # ------------------------------------------------------------------
    laser_train = [
        (int(front_buffer - uwave_delay), LOW),

        # Reference experiment
        (int(polarization_ns), HIGH),
        (int(transient), LOW),
        (int(tau_ns), LOW),
        (int(transient), LOW),
        (int(readout_ns), HIGH),
        (int(meas_buffer), LOW),

        # Signal experiment
        (int(polarization_ns), HIGH),
        (int(transient), LOW),
        (int(tau_ns), LOW),
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
    args = [200, 2000, 440, 0, "SPIN_POL", "SPIN_READOUT", None]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()


# # -*- coding: utf-8 -*-
# """
# Single-point Rabi sequence.

# Two APD-gated experiments per repetition:
#     gate 0 = reference  (no MW pulse)
#     gate 1 = signal     (MW pulse of duration tau_ns)

# Args:
#     [tau_ns, polarization_ns, readout_ns, uwave_ind, readout_vkey, laser_power]
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


# def get_seq(pulse_streamer, config, args):
#     # args = [tau_ns, polarization_ns, readout_ns, uwave_ind, readout_vkey, laser_power]
#     tau_ns, polarization_ns, readout_ns, uwave_ind, readout_vkey_arg, laser_power = args

#     tau_ns = _as_int64("tau_ns", tau_ns)
#     polarization_ns = _as_int64("polarization_ns", polarization_ns)
#     readout_ns = _as_int64("readout_ns", readout_ns)
#     uwave_ind = int(uwave_ind)
#     readout_vkey = _vkey_from_arg(readout_vkey_arg)

#     pulser_wiring = config["Wiring"]["PulseGen"]

#     do_sample_clock = pulser_wiring["do_sample_clock"]
#     do_apd_gate = pulser_wiring["do_apd_gate"]

#     # MW source from virtual sig gen
#     vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
#     sig_gen_name = vsg["physical_name"]
#     uwave_delay = _as_int64(
#         "uwave_delay",
#         config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"],
#     )
#     do_sig_gen_gate = pulser_wiring[f"do_{sig_gen_name}_dm"]

#     # Laser timing
#     laser_name = tb.get_physical_laser_name(readout_vkey)
#     laser_delay = _as_int64(
#         "laser_delay",
#         config["Optics"]["PhysicalLasers"][laser_name]["delay"],
#     )

#     # Buffers
#     common_durations = config["CommonDurations"]
#     meas_buffer = _as_int64("meas_buffer", common_durations["cw_meas_buffer"])
#     init_to_mw_buffer = _as_int64(
#         "init_to_mw_buffer",
#         common_durations.get("uwave_buffer", common_durations["cw_meas_buffer"]),
#     )

#     front_buffer = np.int64(max(uwave_delay, laser_delay))

#     # Each experiment:
#     #   laser init/polarize
#     #   dark buffer
#     #   dummy tau (ref) OR MW tau (sig)
#     #   laser readout + APD gate
#     #   meas buffer
#     exp_period = np.int64(
#         polarization_ns + init_to_mw_buffer + tau_ns + readout_ns + meas_buffer
#     )
#     period = np.int64(front_buffer + 2 * exp_period)

#     seq = Sequence()

#     # Sample clock
#     clk_train = (
#         [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
#         if period >= 300
#         else [(int(period), LOW)]
#     )
#     seq.setDigital(do_sample_clock, clk_train)

#     # APD gate
#     # gate 0 = reference readout
#     # gate 1 = signal readout
#     apd_train = [
#         (int(front_buffer), LOW),
#         # reference experiment
#         (int(polarization_ns), LOW),
#         (int(init_to_mw_buffer), LOW),
#         (int(tau_ns), LOW),
#         (int(readout_ns), HIGH),
#         (int(meas_buffer), LOW),
#         # signal experiment
#         (int(polarization_ns), LOW),
#         (int(init_to_mw_buffer), LOW),
#         (int(tau_ns), LOW),
#         (int(readout_ns), HIGH),
#         (int(meas_buffer), LOW),
#     ]
#     seq.setDigital(do_apd_gate, apd_train)

#     # MW gate
#     # OFF during ref experiment, ON only for tau_ns in signal experiment
#     mw_train = [
#         (int(front_buffer - uwave_delay), LOW),
#         # reference experiment: all LOW
#         (int(exp_period), LOW),
#         # signal experiment
#         (int(polarization_ns), LOW),
#         (int(init_to_mw_buffer), LOW),
#         (int(tau_ns), HIGH),
#         (int(readout_ns), LOW),
#         (int(meas_buffer + uwave_delay), LOW),
#     ]
#     seq.setDigital(do_sig_gen_gate, mw_train)

#     # Laser train:
#     # init pulse + readout pulse for both ref and sig experiments
#     laser_train = [
#         (int(front_buffer), LOW),
#         # reference experiment
#         (int(polarization_ns), HIGH),
#         (int(init_to_mw_buffer), LOW),
#         (int(tau_ns), LOW),
#         (int(readout_ns), HIGH),
#         (int(meas_buffer), LOW),
#         # signal experiment
#         (int(polarization_ns), HIGH),
#         (int(init_to_mw_buffer), LOW),
#         (int(tau_ns), LOW),
#         (int(readout_ns), HIGH),
#         (int(meas_buffer), LOW),
#     ]
#     tb.process_laser_seq(seq, readout_vkey, laser_train)

#     final = OutputState([], 0.0, 0.0)
#     return seq, final, [int(period)]


# if __name__ == "__main__":
#     from utils import common

#     cfg = common.get_config_dict()
#     args = [200, 3e3, 300, 0, "SPIN_READOUT", None]
#     seq, final, ret = get_seq(None, cfg, args)
#     print("Period (ns):", ret[0])
#     seq.plot()


# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr 23 17:39:27 2019

# @author: mccambria
# modified by Saroj Chand on August 2, 2025
# """

# import numpy as np
# from pulsestreamer import OutputState, Sequence

# import utils.tool_belt as tb
# from utils import common
# from utils.constants import VirtualLaserKey
# from utils.tool_belt import Digital


# def get_seq(pulse_streamer, config, args):
#     ### Unpack and get what we need from config

#     # Unpack the durations
#     tau, max_tau, uwave_ind = args
#     # The pulse streamer expects 64 bit ints
#     tau = np.int64(tau)
#     max_tau = np.int64(max_tau)

#     # Signify which signal generator to use
#     virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
#     sig_gen_name = virtual_sig_gen_dict["physical_name"]

#     # Get which laser to use. Same laser will also be used for readout and polarization
#     laser_name = tb.get_physical_laser_name(VirtualLaserKey.SPIN_READOUT)
#     readout_dur = tb.get_virtual_laser_dict(VirtualLaserKey.SPIN_READOUT)["duration"]
#     polarization_dur = tb.get_virtual_laser_dict(VirtualLaserKey.SPIN_POL)["duration"]
#     if readout_dur > polarization_dur:
#         raise ValueError("Readout duration must be shorter than polarization duration")

#     # Get what we need out of the wiring dictionary
#     pulser_wiring = config["Wiring"]["PulseGen"]

#     pulser_do_apd_gate = pulser_wiring["do_apd_gate"]
#     pulser_do_sig_gen_dm = pulser_wiring[f"do_{sig_gen_name}_dm"]

#     # Get the other durations we need
#     # print(laser_name)
#     laser_delay = config["Optics"]["PhysicalLasers"][laser_name]["delay"]
#     uwave_delay = config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"]
#     common_delay = max(laser_delay, uwave_delay)
#     uwave_buffer = config["CommonDurations"]["uwave_buffer"]

#     ### Define the sequence

#     seq = Sequence()

#     # APD gating - first high is for signal, second high is for reference
#     train = [
#         (common_delay, Digital.LOW),
#         (polarization_dur - readout_dur, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (readout_dur, Digital.HIGH),
#         (polarization_dur - readout_dur, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (readout_dur, Digital.HIGH),
#     ]
#     seq.setDigital(pulser_do_apd_gate, train)
#     # Track the total duration for one rep
#     total_dur = 0
#     for el in train:
#         total_dur += el[0]
#     print(total_dur)

#     # Laser for polarization and readout
#     train = [
#         (common_delay - laser_delay, Digital.HIGH),
#         (polarization_dur - readout_dur, Digital.HIGH),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (polarization_dur, Digital.HIGH),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (readout_dur, Digital.HIGH),
#         (laser_delay, Digital.HIGH),
#     ]
#     tb.process_laser_seq(seq, VirtualLaserKey.SPIN_READOUT, train)
#     total_dur = 0
#     for el in train:
#         total_dur += el[0]
#     print(total_dur)

#     # Pulse the microwave for tau
#     train = [
#         (common_delay - uwave_delay, Digital.LOW),
#         (polarization_dur - readout_dur, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (tau, Digital.HIGH),
#         (max_tau - tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (polarization_dur, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (max_tau, Digital.LOW),
#         (uwave_buffer, Digital.LOW),
#         (readout_dur, Digital.LOW),
#         (uwave_delay, Digital.LOW),
#     ]
#     seq.setDigital(pulser_do_sig_gen_dm, train)
#     total_dur = 0
#     for el in train:
#         total_dur += el[0]
#     print(total_dur)

#     final_digital = [pulser_wiring["do_sample_clock"]]
#     final = OutputState(final_digital, 0.0, 0.0)
#     return seq, final, [total_dur]


# if __name__ == "__main__":
#     config = common.get_config_dict()
#     # tb.set_delays_to_zero(config)
#     args = [100, 1000.0, 0]
#     seq = get_seq(None, config, args)[0]
#     seq.plot()
