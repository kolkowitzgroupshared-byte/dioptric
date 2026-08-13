# -*- coding: utf-8 -*-
"""
Single-point T1 (longitudinal relaxation) sequence.

Two APD-gated experiments per repetition:
    gate 0 = reference  (laser polarize → immediate readout, no wait)
    gate 1 = signal     (laser polarize → dark wait τ → readout)

The reference measures the fully-polarized (ms=0) fluorescence level.
The signal decays toward the thermal equilibrium value as τ increases.
Fitting norm = sig/ref gives T1 from:
    norm(τ) = A * exp(-τ / T1) + C

Args (passed via confocal_t1.py → pulse_streamer.stream_load):
    [tau_ns, polarization_ns, readout_ns, pol_vkey, readout_vkey, laser_power]

Timing diagram (one period):
  |<---front_buffer--->|<--------- reference ---------->|<--------- signal ----------->|
                        pol_pulse | transient | readout | pol_pulse | transient | tau | transient | readout | meas_buffer+laser_delay

Created: 2026
@author: Yael
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
    """
    Build the T1 PulseStreamer sequence.

    Parameters
    ----------
    pulse_streamer : PulseStreamer server proxy (unused here, kept for API compatibility)
    config : dict — full lab config dict (from cryo.py / common.get_config_dict())
    args : list — [tau_ns, polarization_ns, readout_ns, pol_vkey, readout_vkey, laser_power]

    Returns
    -------
    seq    : Sequence
    final  : OutputState  (all outputs LOW at end)
    [period] : list of one int — total period in ns (required by LabRAD stream_load protocol)
    """
    tau_ns, polarization_ns, readout_ns, pol_vkey_arg, readout_vkey_arg, laser_power = args

    tau_ns         = _as_int64("tau_ns",         tau_ns)
    polarization_ns = _as_int64("polarization_ns", polarization_ns)
    readout_ns     = _as_int64("readout_ns",     readout_ns)

    readout_vkey = _vkey_from_arg(readout_vkey_arg)

    pulser_wiring  = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate    = pulser_wiring["do_apd_gate"]

    # Laser timing
    laser_name  = tb.get_physical_laser_name(readout_vkey)
    laser_delay = _as_int64(
        "laser_delay",
        config["Optics"]["PhysicalLasers"][laser_name]["delay"],
    )

    # Buffers — mirrors rabi.py exactly
    common_durations = config["CommonDurations"]
    meas_buffer = _as_int64("meas_buffer", common_durations["cw_meas_buffer"])
    transient   = np.int64(200)  # 200 ns dead-time between laser and MW / dark period

    # front_buffer: ensures all channels start at the same logical zero.
    # T1 has no MW, so only laser_delay matters here.
    front_buffer = np.int64(laser_delay)

    # Reference experiment duration = pol + transient + readout + meas_buffer
    # Signal    experiment duration = pol + transient + tau + transient + readout + meas_buffer
    # The two experiments must have the SAME duration so the sample clock fires once per period.
    # We pad the reference's dark section with tau + transient to match the signal.
    ref_dark   = transient                            # reference: no wait (just the transient)
    sig_dark   = transient + tau_ns + transient       # signal: transient + tau + transient

    exp_duration = np.int64(
        polarization_ns + sig_dark + readout_ns + meas_buffer
    )  # longest experiment — signal side

    period = np.int64(front_buffer + 2 * exp_duration)

    print(
        f"[t1.py] tau={tau_ns} ns, pol={polarization_ns} ns, "
        f"read={readout_ns} ns, period={period} ns"
    )

    seq = Sequence()

    # ------------------------------------------------------------------
    # Sample clock  — one HIGH pulse per period (100 ns wide)
    # ------------------------------------------------------------------
    clk_train = (
        [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
        if period >= 300
        else [(int(period), LOW)]
    )
    seq.setDigital(do_sample_clock, clk_train)

    # ------------------------------------------------------------------
    # APD gate
    #   Reference: gate HIGH during readout, AFTER pol + (transient + tau + transient) padding
    #   Signal:    gate HIGH during readout, AFTER pol + transient + tau + transient
    # Both experiments are time-aligned so the readout windows are at the same
    # relative position within each experiment slot.
    # ------------------------------------------------------------------
    apd_train = [
        (int(front_buffer), LOW),

        # Reference experiment
        (int(polarization_ns), LOW),          # polarization — APD OFF
        (int(sig_dark),        LOW),          # matched dark period — APD OFF
        (int(readout_ns),      HIGH),         # readout gate — APD ON
        (int(meas_buffer),     LOW),          # measurement buffer

        # Signal experiment
        (int(polarization_ns), LOW),          # polarization — APD OFF
        (int(sig_dark),        LOW),          # dark wait τ — APD OFF
        (int(readout_ns),      HIGH),         # readout gate — APD ON
        (int(meas_buffer),     LOW),          # measurement buffer
    ]
    seq.setDigital(do_apd_gate, apd_train)

    # ------------------------------------------------------------------
    # Laser train
    #   Reference: laser ON for polarization, OFF during dark, ON for readout
    #   Signal:    laser ON for polarization, OFF during dark+tau, ON for readout
    # No MW in T1 — sig_gen_gate is not set (stays LOW by default).
    # ------------------------------------------------------------------
    laser_train = [
        (int(front_buffer - laser_delay), LOW),  # pre-buffer minus delay (often 0 for green)

        # Reference experiment
        (int(polarization_ns), HIGH),         # polarization pulse — laser ON
        (int(sig_dark),        LOW),          # matched dark period — laser OFF
        (int(readout_ns),      HIGH),         # readout pulse — laser ON
        (int(meas_buffer),     LOW),

        # Signal experiment
        (int(polarization_ns), HIGH),         # polarization pulse — laser ON
        (int(sig_dark),        LOW),          # dark wait τ — laser OFF
        (int(readout_ns),      HIGH),         # readout pulse — laser ON
        (int(meas_buffer + laser_delay), LOW),
    ]
    tb.process_laser_seq(seq, readout_vkey, laser_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common

    cfg = common.get_config_dict()
    # Test with tau = 1 µs, typical NV T1 is ~ms range
    args = [1000, 2000, 440, "SPIN_POL", "SPIN_READOUT", None]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()
