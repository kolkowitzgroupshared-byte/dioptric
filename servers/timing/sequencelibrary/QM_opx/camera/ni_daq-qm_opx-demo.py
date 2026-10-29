# -*- coding: utf-8 -*-
"""
This file contains a few functions used in the Kolkowitz group NV experiment's codebase that
showcase how we use our NI DAQ and our QM OPX. The functions have been modified somewhat to
make sense in this simplified context.

"""

import nidaqmx
import nidaqmx.stream_writers as stream_writers
import numpy as np
from nidaqmx.constants import AcquisitionType
from qm import QuantumMachinesManager, qua

qm_opx_args = {"host": "192.168.0.xxx", "port": 9510, "cluster_name": "demo"}
default_int_freq = int(75e6)
default_duration = 100
opx_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": 0},
                2: {"offset": 0.0, "delay": 0},
            },
            "digital_outputs": {1: {}, 2: {}},
            "analog_inputs": {1: {"offset": 0}},
        },
    },
    "elements": {
        "do_camera_trigger": {
            "digitalInputs": {"chan": {"port": ("con1", 5), "delay": 0, "buffer": 0}},
            "sticky": {"analog": True, "digital": True, "duration": default_duration},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do_laser_OPTO_589_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 6), "delay": 0, "buffer": 0}},
            "sticky": {"analog": True, "digital": True, "duration": default_duration},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do_laser_INTE_520_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 4), "delay": 0, "buffer": 0}},
            "operations": {"charge_pol": "do_charge_pol"},
        },
        "ao_laser_INTE_520_x": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": int(110e6),
            "sticky": {"analog": True, "duration": default_duration},
            "operations": {
                "aod_cw": "green_aod_cw",
                "continue": "ao_off",
            },
        },
        "ao_laser_INTE_520_y": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": int(110e6),
            "sticky": {"analog": True, "duration": default_duration},
            "operations": {
                "aod_cw": "green_aod_cw",
                "continue": "ao_off",
            },
        },
    },
    "pulses": {
        "green_aod_cw-opti": {
            "operation": "control",
            "length": default_duration,
            "waveforms": {"single": "green_aod_cw"},
        },
        "do_charge_pol": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "on",
        },
        "do_on": {
            "operation": "control",
            "length": default_duration,
            "digital_marker": "on",
        },
        "do_off": {
            "operation": "control",
            "length": default_duration,
            "digital_marker": "off",
        },
    },
    "waveforms": {
        "green_aod_cw": {"type": "constant", "sample": 0.11},
    },
    "digital_waveforms": {  # Digital, format is list of tuples: (on/off, ns)
        "on": {"samples": [(1, 0)]},
        "off": {"samples": [(0, 0)]},
    },
}


class opx:
    def init_server(self):
        """Get connection to hardware"""
        qmm = QuantumMachinesManager(**qm_opx_args)
        self.opx = qmm.open_qm(opx_config)

    def init_seq(self, num_nvs):
        """Initialize a sequence. Create necessary QUA variables and start the AODs running.
        Must call within 'with qua.program() as seq:' block


        Parameters
        ----------
        num_nvs : int
            Number of NV centers that will be targeted
        """
        self._charge_pol_incomplete = qua.declare_input_stream(
            bool, name="charge_pol_incomplete"
        )
        self._target_list = qua.declare_input_stream(
            bool, name="target_list", size=num_nvs
        )
        self._target = qua.declare(bool)
        self._x_freq = qua.declare(int)
        self._y_freq = qua.declare(int)

        # Turn on the AOD outputs
        qua.play("aod_cw", "ao_laser_INTE_520_x")
        qua.play("aod_cw", "ao_laser_INTE_520_y")

    def charge_state_readout(self):
        """Trigger a laser to read out the charge states of the NV centers. Simultaneously
        trigger the camera to start an exposure. Pause execution until the camera signals
        that it's ready for a new exposure
        """

        readout_laser_el = "do_laser_OPTO_589_dm"
        camera_el = "do_camera_trigger"

        ###
        # For very long pulses it is easier to compile a sequence that uses "sticky" channels
        # and wait statements rather than very long pulses directly. Sticky channels stay at
        # their last sample until some other pulse is played or ramp_to_zero is called.

        qua.align()
        qua.play("charge_readout", readout_laser_el)
        # Start an exposure on the camera. The camera set up to expose for as long as the TTL is high.
        qua.play("on", camera_el)

        duration = round(50e6 / 4)  # 50 ms exposure in units of clock cycles (4 ns)
        qua.wait(duration, readout_laser_el)
        qua.wait(duration, camera_el)

        qua.ramp_to_zero(readout_laser_el)
        qua.ramp_to_zero(camera_el)

        ###

        # Pause execution until the camera is ready to expose again. The camera has a TTL output that
        # signals this.
        qua.align()
        # wait_for_trigger requires us to pass some element so use an arbitrary dummy
        dummy_element = "do_camera_trigger"
        qua.wait_for_trigger(dummy_element)

    def charge_polarize_one(self, x_coord, y_coord):
        """Charge polarize a single NV center at (x_coord, y_coord), where the coords
        correspond to AOD frequencies.

        Parameters
        ----------
        x_coord : int
            x coordinate of target NV in Hz
        y_coord : int
            y coordinate of target NV in Hz
        """
        qua.align()
        qua.update_frequency("ao_laser_INTE_520_x", x_coord)
        qua.update_frequency("ao_laser_INTE_520_y", y_coord)
        qua.play("charge_pol", "do_laser_INTE_520_dm")

    def conditional_charge_polarize_all(self, num_nvs, nv_x_coords, nv_y_coords):
        """Charge polarize all num_nvs NV centers using conditional logic

        Parameters
        ----------
        num_nvs : int
            Number of NV centers that will be targeted
        nv_x_coords : list(int)
            List of x coordinates for all NVs
        nv_y_coords : list(int)
            List of y coordinates for all NVs
        """

        # All the sequence code goes here. This is just script that gets passed to the OPX.
        # Nothing gets executed until the compile and add_compiled commands at the end of
        # this function.
        with qua.program() as seq:
            self.init_seq(num_nvs)

            # Check the NV charge states. Flow will stop on advance_input_stream until
            # the client populates the "charge_pol_incomplete" input stream variable
            self.charge_state_readout()
            qua.advance_input_stream(self._charge_pol_incomplete)

            # If not all the NVs are in the right charge state, self._charge_pol_incomplete should be
            # True, and we will attempt to polarize the NVs in the wrong charge state.
            with qua.while_(self._charge_pol_incomplete):
                # Determine which NVs to target for charge polarization based on the "target_list"
                # input stream.
                qua.advance_input_stream(self._target_list)

                # Loop through the NVs, skipping those that are already in the correct charge state
                qua_vars = (self._target, self._x_freq, self._y_freq)
                qua_vals = (self._target_list, nv_x_coords, nv_y_coords)
                with qua._for_each(qua_vars, qua_vals):
                    with qua.if_(self._target):
                        self.charge_polarize_one(self._x_freq, self._y_freq)

                # Check again and loop back
                self.charge_state_readout()
                qua.advance_input_stream(self._charge_pol_incomplete)

        program_id = self.opx.compile(seq)
        pending_job = self.opx.queue.add_compiled(program_id)
        # Only return once the job has actually started
        self.running_job = pending_job.wait_for_execution()


class daq:
    def init_server(self):
        """Initialize the server."""

        # Necessary channels
        self.daq_di_clock = "PFI1"
        self.daq_ao_galvo_x = "dev1/AO3"
        self.daq_ao_galvo_y = "dev1/AO4"

    def _close_task(self):
        """Close an open task. Used as a callback as well."""
        if self.task is not None:
            self.task.close()
            self.task = None
        return 0

    def write_xy(self, x_voltage, y_voltage):
        """Write the specified voltages to the galvo.

        Parameters
        ----------
            x_voltage : float
                Voltage to write to the x channel
            y_voltage : float
                Voltage to write to the y channel
        """

        # Close the stream task if it exists
        # This can happen if we quit out early
        if self.task is not None:
            self._close_task()

        with nidaqmx.Task() as task:
            # Set up the output channels
            task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_galvo_x, min_val=-10.0, max_val=10.0
            )
            task.ao_channels.add_ao_voltage_chan(
                self.daq_ao_galvo_y, min_val=-10.0, max_val=10.0
            )
            task.write([x_voltage, y_voltage])

    def galvo_advance_on_trigger(self, coords_x, coords_y, continuous=False):
        """Advance the position of a galvo based on a TTL trigger. The galvo just steps between
        DC voltages - very basic, no DAQ-side clock is required

        Parameters
        ----------
        coords_x : list(float)
            List of x coords to scan through
        coords_y : list(float)
            List of y coords to scan through
        continuous : bool, optional
            Whether to loop continuously on the passed coordinates, by default False
        """

        # Write function expects this arrangement of voltages. First index is x/y,
        # second index is which sample
        voltages = np.vstack((coords_x, coords_y))

        # Close the existing task
        self._close_task()

        # Write the initial voltages now and stream the rest
        num_voltages = voltages.shape[1]
        self.write_xy(voltages[0, 0], voltages[1, 0])
        if continuous:
            # Perfect loop for continuous
            stream_voltages = np.roll(voltages, -1, axis=1)
            num_stream_voltages = num_voltages
        else:
            stream_voltages = voltages[:, 1:num_voltages]
            num_stream_voltages = num_voltages - 1
        stream_voltages = np.ascontiguousarray(stream_voltages)
        # Create a new task
        task = nidaqmx.Task("galvo_advance_on_trigger")
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

        # Configure the sample to advance on the rising edge of the PFI input, which may come
        # from the OPX or some other system. As far as I could ever tell the frequency specified
        # in this case is just the max expected rate
        freq = 100  # Just guess a data rate of 100 Hz
        sample_mode = (
            AcquisitionType.CONTINUOUS if continuous else AcquisitionType.FINITE
        )
        task.timing.cfg_samp_clk_timing(
            freq,
            source=self.daq_di_clock,
            samps_per_chan=num_stream_voltages,
            sample_mode=sample_mode,
        )

        writer.write_many_sample(stream_voltages)

        # Close the task once we've written all the samples
        task.register_done_event(self._close_task)
        task.start()
