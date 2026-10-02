# -*- coding: utf-8 -*-
"""
Server for the Quantum Machines OPX

Created on August 29th, 2022

@author: carter fox

### BEGIN NODE INFO
[info]
name = QM_opx
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 120

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import importlib
import logging
import os
import socket
import sys
import time

import numpy
import numpy as np
from labrad.server import LabradServer, setting
from qm import CompilerOptionArguments, QuantumMachinesManager
from qualang_tools.results import fetching_tool, progress_counter

from servers.inputs.interfaces.tagger import Tagger
from servers.timing.interfaces.pulse_gen import PulseGen
from utils import common
from utils import tool_belt as tb
from utils.constants import CollectionMode


def get_compiled_program_key(seq_file, seq_args_string, num_reps):
    """
    Take the arguments required to generate a compiled QUA program and turn
    them into a key that we can use to lookup a pre-compiled version of the
    program if it exists
    """
    if num_reps is not None:
        num_reps = int(num_reps)
    return f"{seq_file}-{seq_args_string}-{num_reps}"


class QmOpx(Tagger, PulseGen, LabradServer):
    # region Setup and utils

    name = "QM_opx"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self, level=logging.ERROR)

        config = common.get_config_dict()

        # Get manager and OPX
        qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
        self.qmm = QuantumMachinesManager(**qm_opx_args)
        self.qmm.clear_all_job_results()

        self.running_job = None

        self.opx_config = common.get_opx_config_dict()
        self.opx = self.qmm.open_qm(self.opx_config)

        # Sequence tracking variables to prevent redundant compiles of sequences
        self.program_id = None
        self.compiled_programs = {}

        # Add sequence directory to path
        collection_mode = config["collection_mode"]
        collection_mode_str = collection_mode.name.lower()
        path_from_repo = (
            f"servers/timing/sequencelibrary/{self.name}/{collection_mode_str}"
        )
        repo_path = common.get_repo_path()
        opx_sequence_library_path = repo_path / path_from_repo
        sys.path.append(str(opx_sequence_library_path))

        # Tagger setup
        if collection_mode == CollectionMode.COUNTER:
            self.apd_indices = config["apd_indices"]
            self.tagger_di_clock = int(config["Wiring"]["Tagger"]["di_apd_gate"])

        logging.info("Init complete")

    def stopServer(self):
        self.qmm.clear_all_job_results()
        self.qmm.close_all_quantum_machines()
        self.qmm.close()

    # @setting(41)
    # def update_config(self, c):
    #     self.reset(None)

    #     # Get the latest config
    #     new_config = common.get_opx_config_dict(reload=True)

    #     # Only go through with the update if it's necessary
    #     if new_config != self.opx_config:
    #         self.opx_config = new_config
    #         self.opx = self.qmm.open_qm(self.opx_config)

    #         # Sequence tracking variables to prevent redundant compiles of sequences
    #         self.program_id = None
    #         self.compiled_programs = {}

    # endregion
    # region Sequencing

    def get_seq(self, seq_file, seq_args_string, num_reps):
        """Construct a sequence in the passed seq file
            seq_file: str
                A qua sequence file from the sequence library
            seq_args_string: list(any)
                Arbitrary list used to modulate a sequence from the sequence
                library - see simple_readout.py for an example. Default is
                None

        Returns
            seq: Qua program
                A sequence written in Qua for the OPX
            final: unsure.
            list(any)
                Arbitrary list returned by the sequence file
        """

        seq = None
        file_name, file_ext = os.path.splitext(seq_file)

        if file_ext == ".py":  # py: import as a module
            seq_module = importlib.import_module(file_name)
            args = tb.decode_seq_args(seq_args_string)
            seq, seq_ret_vals = seq_module.get_seq(*args, num_reps)

        return seq, seq_ret_vals

    @setting(13, seq_file="s", seq_args_string="s", returns="*?")
    def stream_load(self, c, seq_file, seq_args_string="", num_reps=None):
        """See pulse_gen interface"""

        seq_ret_vals = self._stream_load(seq_file, seq_args_string, num_reps)
        return seq_ret_vals

    def _stream_load(self, seq_file=None, seq_args_string=None, num_reps=None):
        """
        Internal version of stream_load with support for num_reps since that's
        handled in the sequence build for the OPX
        """

        self._halt()

        # Just do nothing if the sequence has already been compiled previously
        key = get_compiled_program_key(seq_file, seq_args_string, num_reps)
        if key in self.compiled_programs:
            program_id, seq_ret_vals = self.compiled_programs[key]
        else:  # Compile and store for next time
            seq, seq_ret_vals = self.get_seq(seq_file, seq_args_string, num_reps)
            # These options allow for faster compiles at the expense of some extra memory usage
            compiler_options = CompilerOptionArguments(
                flags=["skip-loop-unrolling", "skip-loop-rolling"]
            )
            program_id = self.opx.compile(seq, compiler_options=compiler_options)
            self.compiled_programs.clear()  # MCC just store one program for now, the most recent
            self.compiled_programs[key] = [program_id, seq_ret_vals]

        self.program_id = program_id
        return seq_ret_vals

        # Serialize to file for debugging
        # sourceFile = open('debug3.py', 'w')
        # print(generate_qua_script(seq, self.opx_config), file=sourceFile)
        # sourceFile.close()

    @setting(14)
    def stream_start(self, c, num_reps=None):
        """See pulse_gen interface"""

        # Stop the currently running job if there is one
        self._halt()

        pending_job = self.opx.queue.add_compiled(self.program_id)
        # Only return once the job has started
        self.running_job = pending_job.wait_for_execution()
        self.counter_index = 0

    @setting(15, digital_channels="*i", analog_channels="*i", analog_voltages="*v[]")
    def constant(self, c, digital_channels=[], analog_channels=[], analog_voltages=[]):
        """See pulse_gen interface"""

        analog_freqs = [0.0 for el in analog_channels]
        self.constant_ac(
            c, digital_channels, analog_channels, analog_voltages, analog_freqs
        )

    # fmt: off
    @setting(16, digital_channels="*i", analog_channels="*i", analog_voltages="*v[]", analog_freqs="*v[]")
    def constant_ac(self, c, digital_channels=[], analog_channels=[], analog_voltages=[], analog_freqs=[]):
    # fmt: on
        """
        Version of constant() with support for AC signals on the analog outputs.
        Freqs in Hz
        """

        digital_channels = [int(el) for el in digital_channels]
        analog_channels = [int(el) for el in analog_channels]
        analog_voltages = [float(el) for el in analog_voltages]
        analog_freqs = [float(el) for el in analog_freqs]

        args = [digital_channels, analog_channels, analog_voltages, analog_freqs]
        seq_args_string = tb.encode_seq_args(args)

        self.stream_immediate(
            c, seq_file="constant_ac.py", seq_args_string=seq_args_string, num_reps=-1
        )

    # fmt: off
    @setting(17, digital_channels="*i", analog_channels="*i", analog_voltages="*v[]", period="v[]")
    def square_wave(self, c, digital_channels=[], analog_channels=[], analog_voltages=[], period=1000):
    # fmt: on

        digital_channels = [int(el) for el in digital_channels]
        analog_channels = [int(el) for el in analog_channels]
        analog_voltages = [float(el) for el in analog_voltages]
        period = float(period)

        args = [digital_channels, analog_channels, analog_voltages, period]
        seq_args_string = tb.encode_seq_args(args)

        self.stream_immediate(
            c, seq_file="square_wave.py", seq_args_string=seq_args_string, num_reps=-1
        )
        
    @setting(18)
    def resume(self, c):
        
        # Make sure we've gotten to the pause
        timeout = 1
        time_step = 0.001
        start = time.time()
        while time.time() - start < timeout:
            if self.running_job.is_paused():
                break
            time.sleep(time_step)
            
        self.running_job.resume()
        
        
    @setting(31, input_stream_name="s", val="?")
    def insert_input_stream(self, c, input_stream_name, val):
        if isinstance(val, np.ndarray):
            val = val.tolist()
        self.running_job.insert_input_stream(input_stream_name, val)
        

    # endregion
    # region Time tagging

    def read_counter_internal(self):
        """
        This is the core function that any tagger needs in order to function
        as a counter.
        For the OPX this fetches the data from the job that was created when
        the program was executed. Assumes "counts" is one of the data streams
        The count stream should be a three level list. First level is the
        sample, second is the apds, third is the different gates. First index
        gives the sample. next level gives the gate. next level gives which apd
        [  [ [],[] ] , [ [],[] ], [ [],[] ]  ]

        Params
            num_to_read: int
                This is not needed for the OPX
        Returns
            return_counts: array
                This is an array of the counts
        """
        # st = time.time()
        if self.sample_size == "one_rep":
            num_gates_per_sample = self.num_gates_per_rep
            results = fetching_tool(
                self.experiment_job,
                data_list=["counts_apd0", "counts_apd1"],
                mode="live",
            )

        elif self.sample_size == "all_reps":
            num_gates_per_sample = self.num_reps * self.num_gates_per_rep
            results = fetching_tool(
                self.experiment_job,
                data_list=["counts_apd0", "counts_apd1"],
                mode="wait_for_all",
            )

        (
            counts_apd0,
            counts_apd1,
        ) = (
            results.fetch_all()
        )  # just not sure if its gonna put it into the list structure we want
        # now we need to combine into our data structure. they have different lengths because the fpga may
        # save one faster than the other. So just go as far as we have samples on both
        num_new_samples_both = min(
            int(len(counts_apd0) / num_gates_per_sample),
            int(len(counts_apd1) / num_gates_per_sample),
        )
        max_length = num_new_samples_both * num_gates_per_sample

        # get only the number of samples that both have
        counts_apd0 = counts_apd0[self.counter_index : max_length]
        counts_apd1 = counts_apd1[self.counter_index : max_length]
        # now we need to sum over all the iterative readouts that occur if the readout time is longer than 1ms
        counts_apd0 = np.sum(counts_apd0, 1).tolist()
        counts_apd1 = np.sum(counts_apd1, 1).tolist()

        ### now I buffer the list
        n = num_gates_per_sample

        counts_apd0 = [
            counts_apd0[i * n : (i + 1) * n]
            for i in range((len(counts_apd0) + n - 1) // n)
        ]
        counts_apd1 = [
            counts_apd1[i * n : (i + 1) * n]
            for i in range((len(counts_apd1) + n - 1) // n)
        ]

        return_counts = []

        if len(self.apd_indices) == 2:
            for i in range(len(counts_apd0)):
                return_counts.append([counts_apd0[i], counts_apd1[i]])

        elif len(self.apd_indices) == 1:
            for i in range(len(counts_apd0)):
                return_counts.append([counts_apd0[i]])

        self.counter_index = max_length  # make the counter indix the new max length (-1) so the samples start there
        return return_counts

    def read_raw_stream(self):
        """
        Read the raw stream. currently it waits for all data in the job to
        come in and reports it all. Ideally it would do it live
        """
        results = fetching_tool(
            self.experiment_job,
            data_list=["counts_apd0", "counts_apd1", "times_apd0", "times_apd1"],
            mode="wait_for_all",
        )

        counts_apd0, counts_apd1, times_apd0, times_apd1 = results.fetch_all()
        c1 = counts_apd0.tolist()
        c2 = counts_apd1.tolist()

        # *1000 to convert ps to ns
        t1 = (times_apd0[1::] * 1000).tolist()
        t2 = (times_apd1[1::] * 1000).tolist()

        config = self.config
        max_readout_time = config["PhotonCollection"]["qm_opx_max_readout_time"]
        max_readout_time *= 1000  # To ns
        gate_open_channel = config["Wiring"]["Tagger"]["di_apd_gate"]
        gate_close_channel = int(-1 * gate_open_channel)

        all_time_tags = []
        all_channels = []
        running_sum_c1 = 0
        running_sum_c2 = 0

        for gate_ind in range(len(c1)):
            cur_gate_counts_list = c1[gate_ind]
            all_channels.append(gate_open_channel)
            all_time_tags.append(0)

            for sub_gate_ind in range(len(cur_gate_counts_list)):
                sub_gate_counts1 = c1[gate_ind][sub_gate_ind]
                sub_gate_counts2 = c2[gate_ind][sub_gate_ind]

                all_past_counts1 = int(
                    running_sum_c1 + np.sum(c1[gate_ind][0:sub_gate_ind])
                )
                all_past_counts2 = int(
                    running_sum_c2 + np.sum(c2[gate_ind][0:sub_gate_ind])
                )

                sub_gate_time_tags1 = t1[
                    all_past_counts1 : all_past_counts1 + sub_gate_counts1
                ]
                sub_gate_time_tags2 = t1[
                    all_past_counts2 : all_past_counts2 + sub_gate_counts2
                ]

                sub_gate_time_tags1 = np.array(sub_gate_time_tags1) + (
                    sub_gate_ind * max_readout_time
                )
                sub_gate_time_tags2 = np.array(sub_gate_time_tags2) + (
                    sub_gate_ind * max_readout_time
                )

                total_sub_gate_tags = np.append(
                    sub_gate_time_tags1, sub_gate_time_tags2
                )
                total_sub_gate_channels = np.append(
                    np.full(len(sub_gate_time_tags1), 0),
                    np.full(len(sub_gate_time_tags2), 1),
                )

                sort_inds = np.argsort(total_sub_gate_tags)

                all_time_tags.extend((total_sub_gate_tags[sort_inds]).tolist())
                all_channels.extend((total_sub_gate_channels[sort_inds]).tolist())
                # all_time_tags = all_time_tags + (total_sub_gate_tags[sort_inds]).tolist()
                # all_channels = all_channels + (total_sub_gate_channels[sort_inds]).tolist()

            running_sum_c1 = running_sum_c1 + np.sum(c1[gate_ind])
            running_sum_c2 = running_sum_c2 + np.sum(c2[gate_ind])

            all_channels.append(gate_close_channel)
            all_time_tags.append(0)

        t_return = np.array(all_time_tags).astype(np.int64).astype(str)

        return t_return, np.array(all_channels)

    @setting(20, gate_indices="*i", clock="b")  # from apd tagger.
    def start_tag_stream(self, c, gate_indices=None, clock=True):
        self.stream = True
        pass

    @setting(21)  # from apd tagger.
    def stop_tag_stream(self, c):
        self.stream = None
        pass

    @setting(22)
    def clear_buffer(self, c):
        """OPX does not need this - used by Swabian Time Tagger"""
        pass

    # endregion
    # region Conditional logic

    @setting(30, num_streams="i")
    def get_cond_logic_num_ops(self, c, num_streams):
        """
        This function assumes you are trying to save a num_ops stream in the sequence to keep track of how many conditional logic
        operations you do in each run. For instance, how many readouts did you do in each iteration of the sequence, where you stop reading
        out if you met some condition.
        """

        data_list = ["num_ops_{}".format(1 + i) for i in range(num_streams)]
        results = fetching_tool(self.experiment_job, data_list, mode="wait_for_all")
        return_streams = results.fetch_all()
        return return_streams

    # endregion
    # region AWG and RF signal generator

    # all the 'load' functions are not necessary on the OPX
    # the pulses need to exist in the configuration file and they are used in the qua sequence

    # def iq_comps(phase, amp):
    #     if type(phase) is list:
    #         ret_vals = []
    #         for val in phase:
    #             ret_vals.append(numpy.round(amp * numpy.exp((0+1j) * val), 5))
    #         return (numpy.real(ret_vals).tolist(), numpy.imag(ret_vals).tolist())
    #     else:
    #         ret_val = numpy.round(amp * numpy.exp((0+1j) * phase), 5)
    #         return (numpy.real(ret_val), numpy.imag(ret_val))

    # @setting(20, amp="v[]")
    # def set_i_full(self, c, amp):
    #     pass

    # @setting(21)
    # def load_knill(self, c):
    #     pass

    # @setting(22, phases="*v[]")
    # def load_arb_phases(self, c, phases):
    #     pass

    # @setting(23, num_dd_reps="i")
    # def load_xy4n(self, c, num_dd_reps):
    #     pass

    # @setting(24, num_dd_reps="i")
    # def load_xy8n(self, c, num_dd_reps):
    #     pass

    # @setting(25, num_dd_reps="i")
    # def load_cpmg(self, c, num_dd_reps):
    #     pass
    # endregion

    @setting(40)
    def reset(self, c):
        """Stop whatever job is currently running"""
        self._halt()
        # self.qmm.clear_all_job_results()
        # self.qmm.reset_data_processing()
        # self.qmm.close_all_quantum_machines()
        
        
    @setting(42)
    def halt(self, c):
        self._halt()
        
        
    def _halt(self):
        if self.running_job is not None:
            self.running_job.halt()
        self.running_job = None


__server__ = QmOpx()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
