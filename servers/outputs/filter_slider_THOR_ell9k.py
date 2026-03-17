# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs ELL9K filter slider.

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = filter_slider_THOR_ell9k
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
import socket
import time

import serial
from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb


class FilterSliderThorEll9k(LabradServer):
    name = "filter_slider_THOR_ell9k"
    pc_name = socket.gethostname()

    def initServer(self):
        # filename = (
        #     "E:/Shared drives/Kolkowitz Lab Group/nvdata/pc_{}/labrad_logging/{}.log"
        # )
        # filename = filename.format(self.pc_name, self.name)
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s %(levelname)-8s %(message)s",
        #     datefmt="%y-%m-%d_%H-%M-%S",
        #     filename=filename,
        # )
        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_com"]
        try:
            logging.info("here")
            self.slider = serial.Serial(
                device_id,
                9600,
                serial.EIGHTBITS,
                serial.PARITY_NONE,
                serial.STOPBITS_ONE,
            )
        except Exception as e:
            logging.debug(e)
            del self.slider

        time.sleep(0.1)
        self.slider.flush()
        time.sleep(0.1)
        # Find the resonant frequencies of the motor
        cmd = "0s1".encode()
        self.slider.write(cmd)
        time.sleep(0.1)
        # Set up the mapping from filter position to move command
        self.move_commands = {
            0: "0ma00000000".encode(),
            1: "0ma00000020".encode(),
            2: "0ma00000040".encode(),
            3: "0ma00000060".encode(),
        }
        logging.info("Init complete")

    @setting(0, pos="i")
    def set_filter(self, c, pos):
        cmd = self.move_commands[pos]
        # self.slider.write(cmd)
        incomplete = True
        while incomplete:
            self.slider.write(cmd)
            time.sleep(0.1)
            res = self.slider.readline()
            # The device returns a status message if it's not done moving. It
            # returns the current position if it is done moving.
            incomplete = "0GS" in res.decode()
            # if incomplete:
            #     logging.info("huh")


__server__ = FilterSliderThorEll9k()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)

    # with serial.Serial("COM5", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE) as slider:
    #     cmd = "0ma00000060".encode()
    #     slider.write(cmd)
    #     res = slider.readline()
    #     print(res)
