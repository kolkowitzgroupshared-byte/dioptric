# -*- coding: utf-8 -*-
"""
Output server for the PI E709 objective piezo.

Created on Thu Apr  4 15:58:30 2019

@author: mccambria

### BEGIN NODE INFO
[info]
name = pos_z_PI_pifoc
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
from pathlib import Path

import nidaqmx
import nidaqmx.stream_writers as stream_writers
import numpy
from labrad.server import LabradServer, setting
from pipython import GCSDevice
from twisted.internet.defer import ensureDeferred

from utils import common
from utils import tool_belt as tb


class PosZPiPifoc(LabradServer):
    name = "pos_z_PI_pifoc"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        self.task = None
        self.sub_init_server_z()

    def sub_init_server_z(self):
        """Sub-routine to be called by xyz server"""

        self.z_last_position = None
        self.z_current_direction = None
        self.z_last_turning_position = None

        config = common.get_config_dict()

        # try:
        gcs_dll_path = config["DeviceIDs"]["gcs_dll_path"]
        model = config["DeviceIDs"]["objective_piezo_model"]
        self.piezo = GCSDevice(devname=model, gcsdll=gcs_dll_path)
        # Connect the specific device with the serial number
        serial = config["DeviceIDs"]["objective_piezo_serial"]
        self.piezo.ConnectUSB(serial)
        # Just one axis for this device
        self.axis = self.piezo.axes[0]
        self.piezo.SPA(self.axis, 0x06000500, 2)  # External control mode
        # except Exception as exc:
        #     self.piezo = None
        #     self.axis = None
        #     logging.info("Failed to connect to piezo")
        #     logging.info(exc)

        daq_ao = config["Wiring"]["Daq"]["ao_objective_piezo"]
        self.daq_ao_objective_piezo = daq_ao
        daq_clock = config["Wiring"]["Daq"]["di_clock"]
        self.daq_di_clock = daq_clock

        if "z_hysteresis_linearity" in config["Positioning"]:
            linearity = config["Positioning"]["xy_hysteresis_linearity"]
        else:
            linearity = 1
        self.z_hysteresis_b = linearity
        # Define "a" such that 1 nominal volt corresponds to
        # 1 post-compensation volt
        # p(v) = a * v**2 + b * v ==> 1 = a + b ==> a = 1 - b
        self.z_hysteresis_a = 1 - self.z_hysteresis_b

        logging.info("Init complete")

    def stopServer(self):
        if self.piezo is not None:
            self.piezo.CloseConnection()

    def compensate_hysteresis_z(self, position):
        """
        The hysteresis curve is p(v) = a * v**2 + b * v.
        We want to feedforward using this curve to set the piezo voltage
        such that the nominal voltage passed by the user functions
        linearly and without hysteresis. The goal is to prevent the
        accumulation of small errors until active feedback (eg
        optimizing on an NV) can be performed

        Parameters
        ----------
        position : float or ndarray(float)
            Position (in this case the nominal voltage) the user intends
            to move to for a linear response without hysteresis

        Returns
        -------
        float or ndarray(float)
            Compensated voltage to set
        """

        if self.z_hysteresis_b == 1:
            return position

        single_value = False
        if type(position) not in [numpy.ndarray, list]:
            single_value = True
            position = [position]

        # Pull everything out of self to save some lookup time
        last_position = self.z_last_position
        current_direction = self.z_current_direction
        last_turning_position = self.z_last_turning_position
        # If values are uninitialized, assume we just started heading up
        # (ie don't adjust the first position at all)
        if None in [last_position, current_direction, last_turning_position]:
            last_position = position[0]
            current_direction = +1
            last_turning_position = position[0]
        a = self.z_hysteresis_a
        b = self.z_hysteresis_b

        # We'll have to for loop because the (n+1)th value depends
        # on the nth value
        compensated_voltage = []
        for val in position:
            # First determine if we're turning around - if we're not moving,
            # don't consider us to be turning around
            movement_direction = numpy.sign(val - last_position)
            if movement_direction == 0:
                movement_direction = current_direction
            elif movement_direction == -current_direction:
                last_turning_position = last_position
                current_direction = movement_direction

            # The adjustment voltage we need is obtained by inverting p(v)
            abs_p = abs(val - last_turning_position)
            v = (-b + numpy.sqrt(b**2 + 4 * a * abs_p)) / (2 * a)
            result = last_turning_position + (movement_direction * v)
            # result = val
            compensated_voltage.append(result)

            # Cache the last position
            last_position = val

        self.z_last_position = last_position
        self.z_current_direction = movement_direction
        self.z_last_turning_position = last_turning_position

        # return position

        if single_value:
            return compensated_voltage[0]
        else:
            return numpy.array(compensated_voltage)

    def load_stream_writer_z(self, c, task_name, voltages, period):
        # Close the existing task if there is one
        if self.task is not None:
            self.close_task_internal()

        # Make sure the voltages are an array
        voltages = numpy.array(voltages)

        # Write the initial voltages and stream the rest
        num_voltages = len(voltages)
        self.write_z(c, voltages[0])
        stream_voltages = voltages[1:num_voltages]
        # Compensate the remaining voltages
        stream_voltages = self.compensate_hysteresis_z(stream_voltages)
        stream_voltages = numpy.ascontiguousarray(stream_voltages)
        num_stream_voltages = num_voltages - 1

        # Create a new task
        task = nidaqmx.Task(task_name)
        self.task = task

        # Set up the output channels
        task.ao_channels.add_ao_voltage_chan(
            self.daq_ao_objective_piezo, min_val=1.0, max_val=9.0
        )

        # Set up the output stream
        writer = stream_writers.AnalogSingleChannelWriter(task.out_stream)

        # Configure the sample to advance on the rising edge of the PFI input.
        # The frequency specified is just the max expected rate in this case.
        # We'll stop once we've run all the samples.
        freq = float(1 / (period * (10**-9)))  # freq in seconds as a float
        task.timing.cfg_samp_clk_timing(
            freq, source=self.daq_di_clock, samps_per_chan=num_stream_voltages
        )

        writer.write_many_sample(stream_voltages)

        # Close the task once we've written all the samples
        task.register_done_event(self.close_task_internal)

        task.start()

    def close_task_internal(self, task_handle=None, status=None, callback_data=None):
        task = self.task
        if task is not None:
            task.close()
            self.task = None
        return 0

    @setting(22, voltage="v[]")
    def write_z(self, c, voltage):
        """Write the specified voltage to the piezo"""

        # Close the stream task if it exists
        # This can happen if we quit out early
        if self.task is not None:
            self.close_task_internal()

        # Adjust voltage turn for hysteresis
        compensated_voltage = self.compensate_hysteresis_z(voltage)

        with nidaqmx.Task() as task:
            # Set up the output channels
            task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_objective_piezo, min_val=0.0, max_val=9.0
            )
            task.write(compensated_voltage)

    @setting(21, returns="v[]")
    def read_z(self, c):
        """Return the current voltages on the piezo's DAQ channel"""
        with nidaqmx.Task() as task:
            # Parse AO channel name to get internal readback channel
            # e.g., "dev1/AO21" -> "dev1/_ao21_vs_aognd"
            ao_chan = self.daq_ao_objective_piezo  # e.g., "dev1/AO21"
            parts = ao_chan.split("/")
            device = parts[0]  # e.g., "dev1"
            ao_num = parts[1].lower()  # e.g., "ao21"
            chan_name = f"{device}/_{ao_num}_vs_aognd"  # e.g., "dev1/_ao21_vs_aognd"
            task.ai_channels.add_ai_voltage_chan(chan_name, min_val=0.0, max_val=10.0)
            voltage = task.read()
        return voltage

    @setting(23, coords_z="*v[]")
    def load_stream_z(self, c, coords_z):
        """Load a linear sweep with the DAQ"""

        period = 1e6
        self.load_stream_writer_z(c, "ObjectivePiezo-load_scan_z", coords_z, period)

    @setting(24, returns="v[]")
    def get_z_position(self, c):
        """
        Read actual position from PI E-709 via GCS qPOS().

        Returns the current piezo position in micrometers (µm).
        The E-709 PIFOC has approximately 100µm travel range.

        Returns
        -------
        float
            Current Z position in micrometers
        """
        ordered_dict = self.piezo.qPOS(self.axis)
        position_um = ordered_dict[str(self.axis)]
        return position_um


__server__ = PosZPiPifoc()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)

    # config = common.get_config_dict()
    # gcs_dll_path = config["DeviceIDs"]["gcs_dll_path"]
    # model = config["DeviceIDs"]["objective_piezo_model"]
    # with GCSDevice(devname=model, gcsdll=gcs_dll_path) as piezo:
    #     serial = config["DeviceIDs"]["objective_piezo_serial"]
    #     print(piezo.EnumerateUSB())
    #     piezo.ConnectUSB(serial)
    #     # Just one axis for this device
    #     axis = piezo.axes[0]
    #     piezo.SPA(axis, 0x06000500, 2)  # External control mode
    #     daq_ao = config["Wiring"]["Daq"]["ao_objective_piezo"]
    #     daq_ao_objective_piezo = daq_ao
    #     daq_clock = config["Wiring"]["Daq"]["di_clock"]
    #     daq_di_clock = daq_clock
