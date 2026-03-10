# -*- coding: utf-8 -*-
"""
Single resonance sequence.

Two APD-gated experiments per repetition:
    gate 0 = reference  (MW OFF)
    gate 1 = signal     (MW ON)

Args:
    [readout_ns, uwave_ind, readout_vkey, laser_power]
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


def get_seq(pulse_streamer, config, args):
    # args = [pol_ns, readout_ns, uwave_ind, readout_vkey, laser_power]
    readout_ns, uwave_ind, readout_vkey_arg, laser_power = args
    print(f"Got args: {args}")
    readout_ns = _as_int64("readout_ns", readout_ns)
    uwave_ind = int(uwave_ind)
    readout_vkey = _vkey_from_arg(readout_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]

    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]

    # MW source from virtual sig gen
    vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
    uwave_ns = _as_int64("uwave_ns", vsg["pi_pulse"])
    print(f"Microwave duration (ns): {uwave_ns}")
    sig_gen_name = vsg["physical_name"]
    uwave_delay = _as_int64(
        "uwave_delay",
        config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"],
    )
    do_sig_gen_gate = pulser_wiring[f"do_{sig_gen_name}_dm"]

    # Laser timing
    laser_name = tb.get_physical_laser_name(readout_vkey)
    laser_delay = _as_int64(
        "laser_delay",
        config["Optics"]["PhysicalLasers"][laser_name]["delay"],
    )

    meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
    transient = np.int64(0)
    # pol_ns = _as_int64("pol_ns", pol_ns)
    front_buffer = np.int64(max(uwave_delay, laser_delay))
    # period = np.int64(front_buffer + 2 * (pol_ns +  transient + readout_ns + meas_buffer))
    period = np.int64(front_buffer + 2 * (  transient + readout_ns + meas_buffer))

    print(f"Total period (ns): {period}")
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
        (front_buffer, LOW),
        # (pol_ns, LOW),
        (transient, LOW),
        (readout_ns, HIGH),
        (meas_buffer, LOW),  
        # ref
        # (pol_ns, LOW),
        (transient, LOW),
        (uwave_ns, LOW),
        (readout_ns, HIGH),
        (meas_buffer, LOW),  # sig
    ]
    seq.setDigital(do_apd_gate, apd_train)

    # MW gate: OFF in first readout window, ON in second
    # mw_train = [
    #     (front_buffer - uwave_delay, LOW),
    #     (pol_ns, LOW),
    #     (transient, LOW), 
    #     (readout_ns, LOW),
    #     (meas_buffer, LOW),  
    #     # ref
    #     (pol_ns, LOW),
    #     (transient, HIGH),
    #     (uwave_ns, HIGH),
    #     (readout_ns, LOW),
    #     (meas_buffer + uwave_delay, LOW),  # sig
    # ]
    mw_train  = [(int(period/2), LOW), (int(period/2), HIGH)]
    seq.setDigital(do_sig_gen_gate, mw_train)

    # Laser train: on continuously during both measurements
    laser_train = [(int(period), HIGH)]
   
    # laser_train = [(front_buffer - uwave_delay, LOW), 
    #                (pol_ns, LOW)]
    # laser_train = [
    # (front_buffer - uwave_delay, LOW),
    # (pol_ns, LOW),
    # (transient, LOW), 
    # (readout_ns, HIGH),
    # (meas_buffer, LOW),  
    # # ref
    # (pol_ns, LOW),
    # (transient, HIGH),
    # (uwave_ns, HIGH),
    # (readout_ns, HIGH),
    # (meas_buffer + uwave_delay, LOW),  # sig
    # ]

    tb.process_laser_seq(seq, readout_vkey, laser_train)

    # tb.process_laser_seq(seq, VirtualLaserKey.SPIN_READOUT, laser_train)
    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common

    cfg = common.get_config_dict()
    # args = [5000,300, 0, "IMAGING", None]
    args = [300, 0, "IMAGING", None]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()


# # -*- coding: utf-8 -*-
# """
# Created on Thu Apr 11 16:19:44 2019

# @author: mccambria
# """

# from pulsestreamer import Sequence
# from pulsestreamer import OutputState
# import numpy
# import utils.tool_belt as tool_belt

# LOW = 0
# HIGH = 1


# def get_seq(pulse_streamer, config, args):

#     # Unpack the args
#     readout, state, laser_name, laser_power = args

#     # state = States(state)
#     pulser_wiring = config['Wiring']['PulseGen']
#     sig_gen_name = config['Servers'][f'sig_gen_{state.name}']
#     uwave_delay = config['Microwaves'][sig_gen_name]['delay']
#     laser_delay = config['Optics'][laser_name]['delay']
#     meas_buffer = config['CommonDurations']['cw_meas_buffer']
#     transient = 0

#     readout = numpy.int64(readout)
#     front_buffer = max(uwave_delay, laser_delay)
#     period = front_buffer + 2 * (transient + readout + meas_buffer)

#     # Get what we need out of the wiring dictionary
#     pulser_do_daq_clock = pulser_wiring['do_sample_clock']
#     pulser_do_apd_gate = pulser_wiring['do_apd_gate']
#     sig_gen_gate_chan_name = 'do_{}_gate'.format(sig_gen_name)
#     pulser_do_sig_gen_gate = pulser_wiring[sig_gen_gate_chan_name]
#     laser_chan = pulser_wiring['do_{}_dm'.format(laser_name)]

#     seq = Sequence()

#     train = [(period-200, LOW), (100, HIGH), (100, LOW)]
#     seq.setDigital(pulser_do_daq_clock, train)

#     # Ungate the APD channel for the readouts
#     train = [(front_buffer, LOW),
#              (transient, LOW), (readout, HIGH), (meas_buffer, LOW),
#              (transient, LOW), (readout, HIGH), (meas_buffer, LOW)]
#     seq.setDigital(pulser_do_apd_gate, train)

#     # Uwave should be on for the first measurement and off for the second
#     train = [(front_buffer-uwave_delay, LOW),
#              (transient, LOW), (readout, LOW), (meas_buffer, LOW),
#              (transient, LOW), (readout, HIGH), (meas_buffer+uwave_delay, LOW)]
#     seq.setDigital(pulser_do_sig_gen_gate, train)

#     train = [(period, HIGH)]
#     tool_belt.process_laser_seq(pulse_streamer, seq, config,
#                                 laser_name, laser_power, train)

#     final = OutputState([laser_chan], 0.0, 0.0)
#     return seq, final, [period]


# if __name__ == '__main__':
#     config = tool_belt.get_config_dict()
#     args = [10000000.0, 1, 'integrated_520', None, 1]
#     seq, final, ret_vals = get_seq(None, config, args)
#     seq.plot()
