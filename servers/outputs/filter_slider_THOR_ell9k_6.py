# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs ELL9K filter slider.

Created on Wed Oct 29 2025

@author: Alyssa Matthews

### BEGIN NODE INFO
[info]
name = filter_slider_THOR_ell9k_6
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


class FilterSliderThorEll9k(LabradServer):
    name = "filter_slider_THOR_ell9k_6"
    pc_name = socket.gethostname()
    port = "COM10"
    baudrate = 9600
    reset_cfm_opt_out = True

    def initServer(self):
        self._open_serial()
        self.move_commands = {
            0: "0ma00000000".encode(),
            1: "0ma00000020".encode(),
            2: "0ma00000040".encode(),
            3: "0ma00000060".encode(),
        }
        logging.info("Init complete")

    def _open_serial(self):
        try:
            if hasattr(self, "slider") and self.slider.is_open:
                self.slider.close()
        except Exception:
            pass
        self.slider = serial.Serial(self.port, baudrate=self.baudrate, timeout=2.0)
        time.sleep(0.1)
        self.slider.flush()
        self.slider.reset_input_buffer()
        time.sleep(0.1)
        self.slider.write("0s1".encode())
        time.sleep(0.1)

    @setting(0, pos="i")
    def set_filter(self, c, pos):
        cmd = self.move_commands[pos]
        for attempt in range(3):
            try:
                self.slider.reset_input_buffer()
                self.slider.write(cmd)
                deadline = time.monotonic() + 15.0
                while True:
                    if time.monotonic() > deadline:
                        raise RuntimeError(
                            f"{self.name} timed out: slider did not reach position {pos} within 15s on {self.port}. "
                            "Check that the slider is powered and not obstructed."
                        )
                    res = self.slider.readline()
                    if not res:
                        continue
                    if "0GS" not in res.decode(errors="replace"):
                        return
            except serial.SerialException as e:
                if attempt == 2:
                    raise RuntimeError(
                        f"{self.name} serial error on {self.port} after 3 attempts: {e}"
                    ) from e
                logging.warning(f"{self.name}: serial error, reconnecting (attempt {attempt + 1})...")
                self._open_serial()


# make a way to shut off serial connection when we choose to
# restarting labrat connection without closing serial is bad
__server__ = FilterSliderThorEll9k()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)

    # with serial.Serial("COM5", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE) as slider:
    #     cmd = "0ma00000060".encode()
    #     slider.write(cmd)
    #     res = slider.readline()
    #     print(res)
