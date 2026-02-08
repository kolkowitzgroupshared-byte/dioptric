# -*- coding: utf-8 -*-
"""
Input server for APDs running into the Time Tagger.
Created on Wed Apr 24 22:07:25 2019
@author: mccambria
### BEGIN NODE INFO
[info]
name = tagger_SWAB_20_1
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

import logging
import re
import socket

import numpy as np
import TimeTagger
from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from servers.inputs.interfaces.tagger import Tagger, tags_to_counts
from utils import common


class TaggerSwab20(Tagger, LabradServer):
    name = "tagger_SWAB_20_1"
    pc_name = socket.gethostname()

    def initServer(self):
        ### Configure
        config = common.get_config_dict()
        self.config_apd_indices = config["apd_indices"]
        tagger_serial = config["DeviceIDs"][f"{self.name}_serial"]
        try:
            self.tagger = TimeTagger.createTimeTagger(tagger_serial)
        except Exception as e:
            logging.info(e)
        self.tagger.reset()

        # Wiring
        wiring = config["Wiring"]["Tagger"]
        self.tagger_di_clock = wiring["di_clock"]
        self.tagger_di_apd_gate = wiring["di_apd_gate"]

        # Get the APD channels
        apd_indices = []
        self.tagger_di_apd = {}
        keys = wiring.keys()
        for key in keys:
            if re.fullmatch(r"di_apd_[0-9]+", key):
                apd_index = int(key.split("_")[2])
                di_apd = wiring[key]
                self.tagger_di_apd[apd_index] = di_apd

        ### Wrap up
        self.reset_tag_stream_state()
        self.reset(None)
        logging.info("init complete")

    def read_raw_stream(self):
        if self.stream is None:
            logging.error("read_raw_stream attempted while stream is None.")
            return
        buffer = self.stream.getData()
        # Monitor overflows for both the Time Tagger's onboard buffer
        # and the software buffer that the stream feeds into on our PC
        num_hardware_overflows = self.tagger.getOverflowsAndClear()
        has_software_overflows = buffer.hasOverflows
        if (num_hardware_overflows > 0) or has_software_overflows:
            logging.info(f"Num hardware overflows: {num_hardware_overflows}")
            logging.info(f"Has software overflows: {has_software_overflows}")
        timestamps = buffer.getTimestamps()
        channels = buffer.getChannels()
        return timestamps, channels

    def read_counter_internal(self):
        if self.stream is None:
            logging.error("read_counter_internal attempted while stream is None.")
            return
        _, buffer_channels = self.read_raw_stream()

        # Do the hard work in the fast sub function
        apd_channels = [self.tagger_di_apd[val] for val in self.stream_apd_indices]
        return_counts, leftover_channels = tags_to_counts(
            buffer_channels,
            self.tagger_di_clock,
            self.tagger_di_apd_gate,
            apd_channels,
            self.leftover_channels,
        )
        # MCC: Bounce check
        # if 0 in np.array(return_counts):
        #     double_samples = [self.tagger_di_clock, self.tagger_di_clock]
        #     errors = [(buffer_times[i], buffer_times[i+1]) for i in range(len(buffer_channels)) if buffer_channels[i:i+2].tolist() == double_samples]
        #     logging.info(errors)
        self.leftover_channels = leftover_channels
        return return_counts

    def stop_tag_stream_internal(self):
        if self.stream is not None:
            self.stream.stop()
        self.reset_tag_stream_state()

    def reset_tag_stream_state(self):
        self.stream = None
        self.stream_apd_indices = []
        self.stream_channels = []
        self.leftover_channels = np.empty((0), dtype=np.int32)

    @setting(0, returns="*i")
    def get_channel_mapping(self, c):
        """As a regexp, the order is:
        [+APD, ?gate open, ?gate close, ?clock]
        Whether certain channels will be present/how many channels of a given
        type will be present is based on the channels passed to
        start_tag_stream.
        """
        return self.stream_channels

    @setting(1, apd_indices="*i", apd_gate="b", clock="b")
    def start_tag_stream(self, c, apd_indices=None, apd_gate=True, clock=True):
        """Expose a raw tag stream which can be read with read_tag_stream and
        closed with stop_tag_stream.
        """

        # Make sure the existing stream is stopped and we have fresh state
        if self.stream is not None:
            logging.warning(
                "New stream started before existing stream was "
                "stopped. Stopping existing stream."
            )
            self.stop_tag_stream_internal()
        else:
            self.reset_tag_stream_state()

        if apd_indices is None:
            apd_indices = self.config_apd_indices

        channels = []
        for ind in apd_indices:
            channels.append(self.tagger_di_apd[ind])
        if apd_gate:
            channels.append(self.tagger_di_apd_gate)
            channels.append(-self.tagger_di_apd_gate)
        if clock:
            channels.append(self.tagger_di_clock)
        # Store in state before de-duplication to preserve order
        self.stream_channels = channels
        # De-duplicate the channels list
        channels = list(set(channels))
        self.stream = TimeTagger.TimeTagStream(self.tagger, 10**8, channels)
        # When you set up a measurement, it will not start recording data
        # immediately. It takes some time for the tagger to configure the fpga,
        # etc. The sync call waits until this process is complete.
        self.tagger.sync()
        self.stream_apd_indices = apd_indices

    @setting(2)
    def stop_tag_stream(self, c):
        """Closes the stream started with start_tag_stream. Resets
        leftovers.
        """
        self.stop_tag_stream_internal()

    @setting(3)
    def clear_buffer(self, c):
        """Clear the hardware's internal buffer. Should be called before
        starting a pulse sequence."""
        buffer = self.stream.getData()
        # We also don't care about overflows here, so toss (but log) those
        num_hardware_overflows = self.tagger.getOverflowsAndClear()
        has_software_overflows = buffer.hasOverflows
        if (num_hardware_overflows > 0) or has_software_overflows:
            logging.info(f"Num hardware overflows: {num_hardware_overflows}")
            logging.info(f"Has software overflows: {has_software_overflows}")

    @setting(5)
    def reset(self, c):
        self.stop_tag_stream_internal()


__server__ = TaggerSwab20()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
