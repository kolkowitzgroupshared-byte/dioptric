# -*- coding: utf-8 -*-
"""
Output server for the Keysight 33600A series waveform generator.

Created on Mon Mar 2 19:48:00 2026

@author: j-chen1

### BEGIN NODE INFO
[info]
name = awg_KEYS_33600A
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
from typing import Literal

import numpy
import pyvisa as visa  # Docs here: https://pyvisa.readthedocs.io/en/master/
import serial
from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from servers.outputs.interfaces.awg import AWG
from servers.outputs.interfaces.sig_gen_vector import SigGenVector
from utils import common, tool_belt


# class AwgKeys33600A(LabradServer, AWG):
class AwgKeys33600A(LabradServer):
    name = "awg_KEYS_33600A"
    pc_name = socket.gethostname()

    def initServer(self):
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s %(levelname)-8s %(message)s",
        #     datefmt="%y-%m-%d_%H-%M-%S",
        #     filename=filename,
        # )
        # config = common.get_config_dict()
        # self.do_arb_wave_trigger = int(
        #     config["Wiring"]["PulseGen"]["do_arb_wave_trigger"]
        # )
        # resource_manager = visa.ResourceManager()
        # device_id = config["DeviceIDs"]["arb_wave_gen_visa_address"]
        # self.wave_gen = resource_manager.open_resource(device_id)
        # self.iq_comp_amp = config["Microwaves"]["iq_comp_amp"]
        # self.reset(None)
        # logging.info("Init complete")

        logging.basicConfig(filename="awg.log", level=logging.DEBUG)

        tool_belt.configure_logging(self)
        config = common.get_config_dict()
        device_id = config["DeviceIDs"][f"{self.name}_com"]
        # try:
        #     resource_manager = visa.ResourceManager()
        #     self.wave_gen = resource_manager.open_resource(device_id)
        #     self.wave_gen.timeout = 5000  # 5 seconds
        #     self.wave_gen.write_termination = "\n"
        #     self.wave_gen.read_termination = "\n"

        #     # Query identification
        #     idn = self.wave_gen.query("*IDN?")
        #     print("Instrument ID:", idn)
        #     logging.debug("Instrument ID: {}".format(idn))
        #     self.wave_gen.write("SOUR1:FUNC TRI")
        #     self.reset(None)

        # except Exception as e:
        #     self.wave_gen.close()
        #     logging.debug("Ran into Exception! -- {}".format(e))
        #     del self.wave_gen

        # logging.debug("Init complete")

        resource_manager = visa.ResourceManager()
        self.wave_gen = resource_manager.open_resource(device_id)
        self.wave_gen.timeout = 5000  # 5 seconds
        self.wave_gen.write_termination = "\n"
        self.wave_gen.read_termination = "\n"

        # Query identification
        idn = self.wave_gen.query("*IDN?")
        print("Instrument ID:", idn)
        logging.debug("Instrument ID: {}".format(idn))
        self.reset(None)
        logging.info("Init complete")

    @setting(3)
    def set_waveform(
        self, c, ch: int, waveform_type: Literal["SIN", "SQU", "RAMP", "NRAM", "TRI"]
    ):
        source_name = "SOUR{}:".format(ch)
        self.wave_gen.write("{}FUNC {}".format(source_name, waveform_type))

    @setting(4, volt="v[]")
    def test_sin(self, c, volt):
        # for chan in [1, 2]:
        for chan in [1]:
            source_name = "SOUR{}:".format(chan)
            self.wave_gen.write("{}FUNC SIN".format(source_name))
            self.wave_gen.write("{}FREQ 1000".format(source_name))
            self.wave_gen.write("{}VOLT {}".format(source_name, volt))
            # self.wave_gen.write("{}VOLT:LOW -0.5".format(source_name))
        self.wave_gen.write("OUTP1 ON")
        # self.wave_gen.write("SOUR2:PHAS 0")
        # self.wave_gen.write("OUTP2 ON")

    @setting(5)
    def wave_off(self, c):
        self.wave_gen.write("OUTP1 OFF")
        self.wave_gen.write("OUTP2 OFF")

    @setting(6)
    def reset(self, c):
        self.wave_off(c)
        self.wave_gen.write("SOUR1:DATA:VOL:CLE")
        self.wave_gen.write("SOUR2:DATA:VOL:CLE")
        self.wave_gen.write("OUTP1:LOAD 50")
        self.wave_gen.write("OUTP2:LOAD 50")

    @setting(7)
    def force_trigger(self, c):
        # self.wave_gen.write("TRIG")
        self.wave_gen.write("TRIG1")
        self.wave_gen.write("TRIG2")

    @setting(8)
    def load_iq(self, c):
        pass

    @setting(9, ch="v[]", pulse_len="v[]", output_volt="v[]", threshold_volt="v[]")
    def set_TTL(
        self,
        c,
        ch: Literal[1, 2],
        pulse_len: float,
        output_volt: float,
        threshold_volt: float,
    ):
        # Minimum Burst Period of 1 μs
        # Define burst period:
        period = 1 * pulse_len
        if period < 1e-6:
            period = 1e-6
        else:
            period += 500e-9

        ch = int(ch)
        source_name = "SOUR{}:".format(ch)

        self.wave_gen.write("{}FUNC SQU".format(source_name))

        # Set frequency
        freq = f"{1 / pulse_len: .9e}"
        self.wave_gen.write("{}FREQ {}".format(source_name, freq))

        # Set voltage amplitude
        self.wave_gen.write("{}VOLT {}".format(source_name, output_volt))

        # Set offset
        self.wave_gen.write("{}VOLT:OFFS {}".format(source_name, output_volt / 2))

        # Set trigger type
        self.wave_gen.write("TRIG{}:SOUR EXT".format(ch))

        # Set threshold voltage for trigger
        self.wave_gen.write("TRIG{}:LEV {}".format(ch, threshold_volt))

        # Set burst mode
        self.wave_gen.write("{}BURS:MODE TRIG".format(source_name))

        # Set burst period
        self.wave_gen.write("{}BURS:INT:PER {}".format(source_name, period))

        # Set number of cycles
        self.wave_gen.write("{}BURS:NCYC {}".format(source_name, 1))

        # Set number of cycles
        self.wave_gen.write("{}BURS:PHAS 0".format(source_name))

        # Enable burst mode
        self.wave_gen.write("{}BURS:STAT ON".format(source_name))

        self.wave_gen.write("OUTP{} ON".format(ch))

        return


__server__ = AwgKeys33600A()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
