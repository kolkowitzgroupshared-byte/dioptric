# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs GVS212 galvanometer. Controlled by an NI DAQ.

Created on April 8th, 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = pos_xy_THOR_gvs212
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

import nidaqmx
import nidaqmx.stream_writers as stream_writers
import numpy as np
from labrad.server import LabradServer, setting
from nidaqmx.constants import AcquisitionType
from twisted.internet.defer import ensureDeferred

from servers.outputs.interfaces.pos_xy_stream import PosXyStream
from utils import common
from utils import tool_belt as tb


class PosXyThorGvs212(LabradServer, PosXyStream):
    name = "pos_xy_THOR_gvs212"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        self.task = None
        self._last_x = 0.0
        self._last_y = 0.0
        self.sub_init_server_xy()
        # --- add file logging just for this server ---
        from pathlib import Path
        try:
            log_dir = Path(common.get_repo_path()) / "labrad_logging"
        except Exception:
            # fallback if common.get_repo_path() not available
            log_dir = Path.home() / "GitHub" / "dioptric" / "labrad_logging"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / f"{self.name}.log"

        logger = logging.getLogger(self.name)         # server-named logger
        logger.setLevel(logging.DEBUG)                # capture everything
        fh = logging.FileHandler(str(logfile), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        fh.setFormatter(fmt)
        # Avoid duplicate handlers if server restarts in same process
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(logfile) 
                for h in logger.handlers):
            logger.addHandler(fh)
        # ------------------------------------------------
        self.sub_init_server_xy()
        logging.getLogger(self.name).info("Init complete; logging to %s", logfile)

    def sub_init_server_xy(self):
        """Sub-routine to be called by xyz server"""
        config = common.get_config_dict()
        wiring = config["Wiring"]["Daq"]
        self.daq_ao_galvo_x = wiring["ao_galvo_x"]
        self.daq_ao_galvo_y = wiring["ao_galvo_y"]
        self.daq_di_clock = wiring["di_clock"]
        logging.debug("Init complete")

    def stopServer(self):
        self.close_task_internal()

    def close_task_internal(self, task_handle=None, status=None, callback_data=None):
        task = self.task
        if task is not None:
            task.close()
            self.task = None
        return 0

    @setting(0, xVoltage="v[]", yVoltage="v[]")
    def write_xy(self, c, xVoltage, yVoltage):
        """Write the specified voltages to the galvo.

        Params
            xVoltage: float
                Voltage to write to the x channel
            yVoltage: float
                Voltage to write to the y channel
        """

        # Close the stream task if it exists
        # This can happen if we quit out early
        if self.task is not None:
            self.close_task_internal()

        with nidaqmx.Task() as task:
            # Set up the output channels
            task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_galvo_x, min_val=-10.0, max_val=10.0
            )
            task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_galvo_y, min_val=-10.0, max_val=10.0
            )
            task.write([xVoltage, yVoltage])
            self._last_x = float(xVoltage)
            self._last_y = float(yVoltage)
            logging.info(f"write_xy: x={float(xVoltage):.4f}, y={float(yVoltage):.4f}")


    def _write_single_ax(self, voltage, chan):
        if self.task is not None:
            self.close_task_internal()

        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(chan, min_val=-10.0, max_val=10.0)
            task.write(voltage)

    @setting(30, xVoltage="v[]")
    def write_x(self, c, xVoltage):
        return self._write_single_ax(xVoltage, self.daq_ao_galvo_x)

    @setting(31, yVoltage="v[]")
    def write_y(self, c, yVoltage):
        return self._write_single_ax(yVoltage, self.daq_ao_galvo_y)

    # @setting(1, returns="*v[]")
    # def read_xy(self, c):
    #     """Return the current voltages on the x and y channels.

    #     Returns
    #         list(float)
    #             Current voltages on the x and y channels

    #     """
    #     with nidaqmx.Task() as task:
    #         # Set up the internal channels - to do: actual parsing...
    #         if self.daq_ao_galvo_x == "dev1/AO0":
    #             chan_name = "dev1/_ao0_vs_aognd"
    #         task.ai_channels.add_ai_voltage_chan(chan_name, min_val=-10.0, max_val=10.0)
    #         if self.daq_ao_galvo_y == "dev1/AO1":
    #             chan_name = "dev1/_ao1_vs_aognd"
    #         task.ai_channels.add_ai_voltage_chan(chan_name, min_val=-10.0, max_val=10.0)
    #         voltages = task.read()

    #     return voltages[0], voltages[1]
    @setting(1, returns="*v[]")
    def read_xy(self, c):
        """Return last commanded XY (no AI readback)."""
        return [self._last_x, self._last_y]

    @setting(2, coords_x="*v[]", coords_y="*v[]", continuous="b")
    def load_stream_xy(self, c, coords_x, coords_y, continuous=False):
        voltages = np.vstack((coords_x, coords_y))

        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Write the initial voltages and stream the rest
        num_voltages = voltages.shape[1]
        self.write_xy(c, voltages[0, 0], voltages[1, 0])
        if continuous:
            # Perfect loop for continuous
            stream_voltages = np.roll(voltages, -1, axis=1)
            num_stream_voltages = num_voltages
        else:
            stream_voltages = voltages[:, 1:num_voltages]
            num_stream_voltages = num_voltages - 1
        stream_voltages = np.ascontiguousarray(stream_voltages)
        # Create a new task
        task = nidaqmx.Task(f"{self.name}-load_stream_xy")
        self.task = task

        # Set up the output channels
        task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_galvo_x, min_val=-10.0, max_val=10.0
        )
        task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_galvo_y, min_val=-10.0, max_val=10.0
        )

        # Set up the output stream
        writer = stream_writers.AnalogMultiChannelWriter(task.out_stream)

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = 100  # Just guess a data rate of 100 Hz
        if continuous:
            task.timing.cfg_samp_clk_timing(
                freq,
                source=self.daq_di_clock,
                samps_per_chan=num_stream_voltages,
                sample_mode=AcquisitionType.CONTINUOUS,
            )
        else:
            task.timing.cfg_samp_clk_timing(
                freq,
                source=self.daq_di_clock,
                samps_per_chan=num_stream_voltages,
            )

        writer.write_many_sample(stream_voltages)

        # Close the task once we've written all the samples
        task.register_done_event(self.close_task_internal)
        task.start()

    def _load_stream_single_ax(self, coords, chan, continuous=False):
        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Write the initial voltage and stream the rest
        num_voltages = len(coords)
        self._write_single_ax(coords[0], chan)
        if continuous:
            # Perfect loop for continuous
            stream_voltages = np.roll(coords, -1)
        else:
            stream_voltages = coords[1:num_voltages]
        stream_voltages = np.ascontiguousarray(stream_voltages)
        num_stream_voltages = num_voltages - 1
        # Create a new task
        task = nidaqmx.Task(f"{self.name}-load_stream_single_ax")
        self.task = task

        # Set up the output channels
        task.ao_channels.add_ao_voltage_chan(chan, min_val=-10.0, max_val=10.0)

        # Set up the output stream
        writer = stream_writers.AnalogSingleChannelWriter(task.out_stream)

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = 100  # Just guess a data rate of 100 Hz
        if continuous:
            task.timing.cfg_samp_clk_timing(
                freq,
                source=self.daq_di_clock,
                samps_per_chan=num_stream_voltages,
                sample_mode=AcquisitionType.CONTINUOUS,
            )
        else:
            task.timing.cfg_samp_clk_timing(
                freq,
                source=self.daq_di_clock,
                samps_per_chan=num_stream_voltages,
            )

        writer.write_many_sample(stream_voltages)

        # Close the task once we've written all the samples
        task.register_done_event(self.close_task_internal)
        task.start()

    @setting(33, coords_x="*v[]", continuous="b")
    def load_stream_x(self, c, coords_x, continuous=False):
        return self._load_stream_single_ax(coords_x, self.daq_ao_galvo_x, continuous)

    @setting(34, coords_y="*v[]", continuous="b")
    def load_stream_y(self, c, coords_y, continuous=False):
        return self._load_stream_single_ax(coords_y, self.daq_ao_galvo_y, continuous)
    
    @setting(35, returns="*v[]")
    def read_x(self, c):
        """Return last commanded X"""
        return [self._last_x]

    @setting(36, returns="*v[]")
    def read_y(self, c):
        """Return last commanded Y"""
        return [self._last_y]

    @setting(3)
    def reset(self, c):
        self.close_task_internal()


__server__ = PosXyThorGvs212()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
