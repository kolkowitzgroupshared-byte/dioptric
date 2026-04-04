# -*- coding: utf-8 -*-
"""
Generic square-wave sequence for the Swabian Pulse Streamer.

Matches the QM_opx-style interface:
    [digital_channels, analog_channels, analog_voltages, period]

digital channel behavior:
    first half-period  -> HIGH
    second half-period -> LOW

analog channel behavior:
    first half-period  -> requested voltage
    second half-period -> 0.0 V
"""

from pulsestreamer import Sequence, OutputState
import numpy as np

LOW = 0
HIGH = 1


def get_seq(pulse_streamer, config, args):
    digital_channels, analog_channels, analog_voltages, period = args

    digital_channels = [int(ch) for ch in digital_channels]
    analog_channels = [int(ch) for ch in analog_channels]
    analog_voltages = [float(v) for v in analog_voltages]
    period = int(round(float(period)))

    if len(analog_channels) != len(analog_voltages):
        raise ValueError("analog_channels and analog_voltages must have the same length")

    if period < 2:
        raise ValueError("period must be >= 2 ns")

    half_period = period // 2
    second_half = period - half_period

    seq = Sequence()

    # Digital outputs: HIGH then LOW
    digital_train = [(half_period, HIGH), (second_half, LOW)]
    for ch in digital_channels:
        seq.setDigital(ch, digital_train)

    # Analog outputs: requested voltage then 0
    for ch, voltage in zip(analog_channels, analog_voltages):
        if ch not in (0, 1):
            raise ValueError(f"Pulse Streamer analog channel must be 0 or 1, got {ch}")
        analog_train = [(half_period, voltage), (second_half, 0.0)]
        seq.setAnalog(ch, analog_train)

    # Final state: everything off
    final = OutputState([], 0.0, 0.0)

    return seq, final, [period]


if __name__ == "__main__":
    config = None
    args = [[0, 3], [0], [0.5], 200]
    seq, final, ret_vals = get_seq(None, config, args)
    seq.plot()