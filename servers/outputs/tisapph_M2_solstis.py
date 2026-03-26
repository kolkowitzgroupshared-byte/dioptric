# -*- coding: utf-8 -*-
"""
Output server for the M^2 Solstis TiSapph laser

Created on August 5th, 2025

@author: jchen

### BEGIN NODE INFO
[info]
name = tisapph_M2_solstis
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
import pyvisa as visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
import websocket
from labrad.server import LabradServer, setting
from pylablib.devices import M2
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb


class TisapphM2Solstis(LabradServer):
    name = "tisapph_M2_solstis"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_ip"]
        self.tisapph = M2.Solstis(device_id, 62566, use_websocket=True)
        logging.info("Init complete")

    @setting(0, wavelength_nm="v[]")
    def set_wavelength_nm(self, c, wavelength_nm):
        wavelength = wavelength_nm * 1e-9
        self.tisapph.coarse_tune_wavelength(wavelength=wavelength)

    @setting(1, returns="v[]")
    def get_wavelength_nm(self, c):
        wavelength = self.tisapph.get_coarse_wavelength()
        wavelength_nm = wavelength * 1e9
        return wavelength_nm

    @setting(2, value="v[]")
    def tune_etalon_relative(self, c, value):
        if not (value > 0.0 and value < 100.0):
            raise ValueError("Value must be between 0 and 100")
        else:
            # self.etalonTune = self.tisapph.get_full_web_status()["etalon_tune"]
            # print("Etalon tune status = ", self.etalonTune)
            self.tisapph.tune_etalon(value)
            # self.etalonTune = self.tisapph.get_full_web_status()["etalon_tune"]
            # print("Etalon tune status = ", self.etalonTune)

    @setting(3)
    def reset_etalon(self, c):
        self.tisapph.tune_etalon(50)

    @setting(4, returns="v[]")
    def get_etalon_tune_status(self, c):
        return self.tisapph.get_full_web_status()["etalon_tune"]

    @setting(6)
    def reset(self, c):
        pass


__server__ = TisapphM2Solstis()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
