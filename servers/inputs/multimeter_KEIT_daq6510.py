# -*- coding: utf-8 -*-
"""
Input server for a Keithley DAQ6510 benchtop multimeter

Created on August 7th, 2025

@author: mccambria

### BEGIN NODE INFO
[info]
name = multimeter_KEIT_daq6510
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout =
### END NODE INFO
"""

import logging
import socket
import time
from typing import Literal

import numpy
import pyvisa as visa
from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb


class MultimeterKeitDaq6510(LabradServer):
    name = "multimeter_KEIT_daq6510"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        config = common.get_config_dict()

        resource_manager = visa.ResourceManager()
        visa_address = config["DeviceIDs"][f"{self.name}_visa"]
        # logging.info(visa_address)
        self.multimeter = resource_manager.open_resource(visa_address)
        self.multimeter.read_termination = "\n"
        self.multimeter.write_termination = "\n"
        self.multimeter.write("*RST")
        logging.info("Init complete")

    # def measure_internal(self):
    #     value = self.multimeter.query("MEAS1?")
    #     if value == "":
    #         logging.info("Read blank string in measure_internal!")
    #         while value == "":
    #             time.sleep(0.1)
    #             value = self.multimeter.query("MEAS1?")
    #         logging.info("Recovered!")
    #     return float(value)

    @setting(5, returns="v[]", channel="i")
    def read_voltage(self, c, channel):
        """Return the value from the main display."""
        #value = self.multimeter.query(":MEASure:VOLTage:DC?")

        multimeter = self.multimeter
        multimeter.write("*RST")
        multimeter.write("SENS:FUNC 'VOLT:DC', (@{})".format(channel))
        multimeter.write(":ROUT:CLOS (@{})".format(channel))
        output = float(multimeter.query(":READ?".format(channel)))
        return output

    @setting(6)
    def reset_cfm_opt_out(self, c):
        """This setting is just a flag for the client. If you include this
        setting on a server, then the server won't be reset along with the
        rest of the instruments when we call tool_belt.reset_cfm.
        """
        pass

    @setting(7)
    def reset(self, c):
        """Fully reset to factory defaults"""
        self.multimeter.write("*RST")

    @setting(8, nplc="v[]")
    def set_nplc(self, c, nplc):
        self.multimeter.write(f"VOLT:NPLC {nplc}")

    @setting(9, num_meas_to_avg="i")
    def set_avg_window_size(self, c, num_meas_to_avg):
        self.multimeter.write(f"VOLT:AVER:COUNT {num_meas_to_avg}")

    @setting(10, filter_type="s")
    def set_filter_type(self, c, filter_type: Literal["repeating", "moving", "hybrid"]):
        if filter_type == "repeating":
            self.multimeter.write("VOLT:AVER:TCON REP")
        elif filter_type == "moving":
            self.multimeter.write("VOLT:AVER:TCON MOV")
        elif filter_type == "hybrid":
            self.multimeter.write("VOLT:AVER:TCON HYBR")

    @setting(11, nplc="v[]", num_meas_to_avg="i", filter_type="s")
    def set_averaging_params(
        self,
        c,
        nplc,
        num_meas_to_avg,
        filter_type: Literal["repeating", "moving", "hybrid"],
    ):
        self.set_nplc(c, nplc)
        self.set_avg_window_size(c, num_meas_to_avg)
        self.set_filter_type(c, filter_type)
        self.turn_on_averaging(c)

    @setting(12)
    def turn_on_averaging(self, c):
        self.multimeter.write("VOLT:AVER ON")

    @setting(13)
    def turn_off_averaging(self, c):
        self.multimeter.write("VOLT:AVER OFF")

    @setting(14, returns="v[]", channel="i")
    def read_temperature(self, c, channel):
        multimeter = self.multimeter
        multimeter.write("*RST")
        multimeter.write(":FUNCtion 'TEMPerature', (@{})".format(channel))
        multimeter.write(":SENSe:TEMPerature:TRANsducer TCouple, (@{})".format(channel))
        multimeter.write(":SENSe:TEMPerature:TCouple:TYPE K,(@{})".format(channel))
        multimeter.write(":SENSe:TEMPerature:TCouple:RJUNction:RSELect INTernal,(@{})".format(channel))
        multimeter.write(":SENSe:TEMPerature:ODETector ON,(@{})".format(channel))
        multimeter.write(":ROUT:CLOS (@{})".format(channel))
        output = float(multimeter.query(":READ?".format(channel)))
        return output

__server__ = MultimeterKeitDaq6510()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
