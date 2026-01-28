"""
Virtual LabRAD server to measure power.

Created on June 25, 2025

@author: Eric Gediman, Alyssa Matthews

### BEGIN NODE INFO
[info]
name = power_meter_THOR_pm100d
version = 1.0
description = Virtual LabRAD server to monitor laser power.

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import datetime
import os
import socket
import time

import pyvisa
from labrad.server import LabradServer, setting


class PowerMonitorThorPm100D(LabradServer):
    # writes command to get output of 4A -- where the temp controller is connected to
    # See the ptc10 manual for more info

    name = "power_meter_THOR_pm100d"
    rm = pyvisa.ResourceManager()
    pc_name = socket.gethostname()

    # might be subject to change
    instrument_name = "USB0::0x1313::0x8078::P0051482::INSTR"

    # We will be running this constantly
    def initServer(self):
        """
        Initialize the server and Variables.

        """
        self.inst = self.rm.open_resource(self.instrument_name)
        # self.inst.write("INITiate[:IMMediate]")
        # self.inst.write("MEASure[:SCALar][:POWer]?")

    # @setting(0, cmd="y", val="v")
    # def set_param(self, c, cmd, val):
    #   self.ser.write(cmd + b"=" + bytes(str(val), "ascii") + b"\n")
    #  time.sleep(1)

    @setting(1)
    def get_power(self, c):
        self.inst.write("MEAS:POW?")
        power = self.inst.read()
        return power

    @setting(2)
    def set_wavelength(self, c, wavelength):
        return self.inst.write(f"CORR:WAV {wavelength:.0f}")

    @setting(3)
    def get_wavelength(self, c):
        self.inst.write("CORR:WAV?")
        curr_wavelength = self.inst.read()
        return curr_wavelength

    def stopServer(self):
        """Ensure all everything is closed."""
        self.inst.write("ABORt")
        self.inst.close()


__server__ = PowerMonitorThorPm100D()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
