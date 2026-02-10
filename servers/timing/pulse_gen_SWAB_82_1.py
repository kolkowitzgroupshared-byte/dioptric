# -*- coding: utf-8 -*-
"""
Timing server for the Swabian Pulse Stream 8/2.

Created on Tue Apr  9 17:12:27 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = pulse_gen_SWAB_82_1
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import importlib
import logging
import os
import socket
import sys
from pathlib import Path

from labrad.server import LabradServer, setting
from pulsestreamer import OutputState, PulseStreamer, TriggerStart
from twisted.internet.defer import ensureDeferred

from servers.timing.interfaces.pulse_gen import PulseGen
from utils import common
from utils import tool_belt as tb


class PulseGenSwab82(PulseGen, LabradServer):
    name = "pulse_gen_SWAB_82_1"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)

        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_ip_1"]
        # logging.info(f"Connecting to PulseStreamer at {device_id}")
        self.pulse_streamer = PulseStreamer(device_id)
        calibration = self.pulse_streamer.getAnalogCalibration()
        logging.info(calibration)
        collection_mode = config["collection_mode"]
        collection_mode_str = collection_mode.name.lower()

        repo_path = common.get_repo_path()
        sequence_library_path = (
            repo_path
            / f"servers/timing/sequencelibrary/{self.name}/{collection_mode_str}"
        )

        sys.path.append(str(sequence_library_path))

        self.config = config
        self.pulse_streamer_wiring = self.config["Wiring"]["PulseGen"]

        # Initialize state variables and reset
        self.seq = None
        self.loaded_seq_streamed = False
        self.reset(None)

        logging.info("Init complete")

    def get_seq(self, seq_file, seq_args_string):
        seq = None
        file_name, file_ext = os.path.splitext(seq_file)
        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tb.decode_seq_args(seq_args_string)
            seq, final, ret_vals = seq_module.get_seq(self, self.config, args)
        return seq, final, ret_vals

    @setting(2, seq_file="s", seq_args_string="s", returns="*?")
    def stream_load(self, c, seq_file, seq_args_string="", num_reps=1):
        """See pulse_gen interface"""

        self.pulse_streamer.setTrigger(start=TriggerStart.SOFTWARE)
        seq, final, ret_vals = self.get_seq(seq_file, seq_args_string)
        if seq is not None:
            self.seq = seq
            self.loaded_seq_streamed = False
            self.final = final
        return ret_vals

    @setting(3, num_reps="i")
    def stream_start(self, c, num_reps=1):
        """See pulse_gen interface"""

        if self.seq == None:
            raise RuntimeError("Stream started with no sequence.")
        if not self.loaded_seq_streamed:
            self.pulse_streamer.stream(self.seq, num_reps, self.final)
            self.loaded_seq_streamed = True
        self.pulse_streamer.startNow()

    @setting(4, digital_channels="*i", analog_channels="*i", analog_voltages="*v[]")
    def constant(self, c, digital_channels=[], analog_channels=[], analog_voltages=[]):
        """See pulse_gen interface. Default is everything off"""

        # Digital
        digital_channels = [int(el) for el in digital_channels]

        # Analog
        analog_0_voltage = 0.0
        analog_1_voltage = 0.0
        for chan in [0, 1]:
            if chan in analog_channels:
                ind = analog_channels.index(chan)
                voltage = analog_voltages[ind]
                if chan == 0:
                    analog_0_voltage = voltage
                elif chan == 1:
                    analog_1_voltage = voltage

        # Run the operation
        state = OutputState(digital_channels, analog_0_voltage, analog_1_voltage)
        self.pulse_streamer.constant(state)

    @setting(5)
    def force_final(self, c):
        """
        Force the PulseStreamer its current final output state.
        Essentially a stop command.
        """

        self.pulse_streamer.forceFinal()

    @setting(6)
    def reset(self, c):
        # Probably don't need to force_final right before constant but...
        self.force_final(c)
        self.constant(c)
        self.seq = None
        self.loaded_seq_streamed = False


__server__ = PulseGenSwab82()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
