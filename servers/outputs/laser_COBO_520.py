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
        self.laser_serial = None
        try:
            config = common.get_config_dict()
            device_id = config["DeviceIDs"][f"{self.name}_com"]
            self.laser_serial = serial.Serial(
                device_id,
                115200,
                serial.EIGHTBITS,
                serial.PARITY_NONE,
                serial.STOPBITS_ONE,
                timeout=1,
            )
            self.laser_serial.reset_input_buffer()
            self.laser_serial.reset_output_buffer()
        except Exception as e:
            logging.error(f"Failed to open serial connection: {e}")
            print(f"laser_COBO_520: Failed to open serial connection: {e}")

    def stopServer(self):
        super().stopServer()
        if self.laser_serial is not None:
            self.laser_serial.close()

    def _query(self, cmd):
        self.laser_serial.reset_input_buffer()
        self.laser_serial.write(f"{cmd}\r".encode("ascii"))
        return self.laser_serial.readline().decode("ascii").strip()

    def _write(self, cmd):
        self.laser_serial.write(f"{cmd}\r".encode("ascii"))
        self.laser_serial.readline()

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
