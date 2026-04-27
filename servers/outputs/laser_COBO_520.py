# -*- coding: utf-8 -*-
"""
Output server for the Cobolt 520 nm laser.

Created on Mon Apr  8 19:50:12 2019

@author: mccambria

Edited 4/20/2026
@author: chemistatcode

### BEGIN NODE INFO
[info]
name = laser_COBO_520
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

import serial
from labrad.server import setting

from laser_COBO_base import LaserCoboBase
from utils import common


class LaserCobo520(LaserCoboBase):
    wavelength = 520
    name = f"laser_COBO_{wavelength}"

    def initServer(self):
        super().initServer()
        config = common.get_config_dict()
        self._com_port = config["DeviceIDs"][f"{self.name}_com"]

    def _open_serial(self):
        s = serial.Serial(
            self._com_port,
            115200,
            serial.EIGHTBITS,
            serial.PARITY_NONE,
            serial.STOPBITS_ONE,
            timeout=1,
        )
        s.reset_input_buffer()
        s.reset_output_buffer()
        return s

    def _query(self, cmd):
        s = self._open_serial()
        try:
            s.reset_input_buffer()
            s.write(f"{cmd}\r".encode("ascii"))
            return s.readline().decode("ascii").strip()
        finally:
            s.close()

    def _write(self, cmd):
        s = self._open_serial()
        try:
            s.write(f"{cmd}\r".encode("ascii"))
            s.readline()
        finally:
            s.close()

    @setting(10, power="v[]")
    def set_power(self, c, power):
        """Set the laser output power setpoint in Watts."""
        self._write(f"p {power}")

    @setting(11, returns="v[]")
    def get_power(self, c):
        """Read the laser output power setpoint in Watts."""
        return float(self._query("p?"))

    @setting(12, returns="v[]")
    def get_actual_power(self, c):
        """Read the actual laser output power in Watts."""
        return float(self._query("pa?"))


__server__ = LaserCobo520()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
