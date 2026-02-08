# -*- coding: utf-8 -*-
"""
Output server for the Coherent Verdi, used to pump the M^2 Solstis TiSapph

Created on August 5th, 2025

@author: mccambria

### BEGIN NODE INFO
[info]
name = tisapph_pump_COHE_verdi
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

import numpy as np
import serial
from labrad.server import LabradServer, setting

from utils import common
from utils import tool_belt as tb


class TisapphPumpCoheVerdi(LabradServer):
    name = "tisapph_pump_COHE_verdi"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_com"]
        try:
            self.laser = serial.Serial(
                device_id,
                19200,
                serial.EIGHTBITS,
                serial.PARITY_NONE,
                serial.STOPBITS_ONE,
                timeout=1,
            )
        except Exception as e:
            logging.debug(e)
            del self.laser
        logging.debug("Init complete")

    @setting(0)
    def send(self, c, cmd):
        self.laser.write(f"{cmd}\r\n".encode("ascii"))

    # @setting(0, wavelength_nm="v[]")
    # def set_wavelength_nm(self, c, wavelength_nm):
    #     wavelength = wavelength_nm * 1e-9
    #     self.tisapph.coarse_tune_wavelength(wavelength=wavelength)

    @setting(6)
    def reset(self, c):
        pass


__server__ = TisapphPumpCoheVerdi()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
