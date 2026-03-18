# -*- coding: utf-8 -*-
"""
Interface for TTL pulse counters

Created on August 29th, 2022

@author: mccambria
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from labrad.server import LabradServer, setting


class Counter(LabradServer, ABC):
    def read_counter_setting_internal(self, num_to_read):
        if self.stream is None:
            logging.error("read_counter attempted while stream is None.")
            return
        if num_to_read is None:
            # Poll once and return the result
            counts = self.read_counter_internal()
        else:
            # Poll until we've read the requested number of samples
            counts = []
            while len(counts) < num_to_read:
                counts.extend(self.read_counter_internal())

            if len(counts) > num_to_read:
                msg = "Read {} samples, only requested {}".format(
                    len(counts), num_to_read
                )
                logging.error(msg)

        return counts

    @setting(207, num_to_read="i", returns="*3w")
    def read_counter_complete(self, c, num_to_read=None):
        return self.read_counter_setting_internal(num_to_read)

    @setting(208, num_to_read="i", returns="*w")
    def read_counter_simple(self, c, num_to_read=None):
        complete_counts = self.read_counter_setting_internal(num_to_read)

        # To combine APDs we assume all the APDs have the same gate
        # gate_channels = list(self.tagger_di_gate.values())
        # first_gate_channel = gate_channels[0]
        # if not all(val == first_gate_channel for val in gate_channels):
        #     logging.critical("Combined counts from APDs with different gates.")

        # Just find the sum of each sample in complete_counts
        return_counts = [np.sum(sample, dtype=int) for sample in complete_counts]

        return return_counts

    @setting(209, num_to_read="i", returns="*2w")
    def read_counter_separate_gates(self, c, num_to_read=None):
        complete_counts = self.read_counter_setting_internal(num_to_read)
        # logging.info(complete_counts)

        # To combine APDs we assume all the APDs have the same gate
        # gate_channels = list(self.tagger_di_gate.values())
        # first_gate_channel = gate_channels[0]
        # if not all(val == first_gate_channel for val in gate_channels):
        #     logging.critical("Combined counts from APDs with different gates.")

        # Add the APD counts as vectors for each sample in complete_counts
        return_counts = [
            np.sum(sample, 0, dtype=int).tolist() for sample in complete_counts
        ]

        return return_counts

    @setting(210, modulus="i", num_to_read="i", returns="*2w")
    def read_counter_modulo_gates(self, c, modulus, num_to_read=None):
        complete_counts = self.read_counter_setting_internal(num_to_read)

        # To combine APDs we assume all the APDs have the same gate
        try:
            gate_channels = list(self.tagger_di_gate.values())
            first_gate_channel = gate_channels[0]
            if not all(val == first_gate_channel for val in gate_channels):
                logging.critical("Combined counts from APDs with different gates.")
        except:
            pass
        # Add the APD counts as vectors for each sample in complete_counts
        # sum_lambda = lambda arg: np.sum(arg, 0, dtype=int).tolist()
        # with Pool() as p:
        #     separate_gate_counts = p.map(sum_lambda, complete_counts)
        separate_gate_counts = []
        for el in complete_counts:
            summed = np.sum(el, axis=0, dtype=np.int64)
            separate_gate_counts.append(np.asarray(summed, dtype=np.int64).reshape(-1))

        # Run the modulus
        return_counts = []
        for sample in separate_gate_counts:
            sample_list = []
            for ind in range(modulus):
                vals = sample[ind::modulus]
                if vals.size == 0:
                    total = 0
                else:
                    total = int(np.sum(vals, dtype=np.int64))
                sample_list.append(total)
            return_counts.append(sample_list)

        # Force plain Python ints for LabRAD
        return_counts = [[int(a), int(b)] for a, b in return_counts]

        return return_counts

    @setting(211, num_to_read="i", returns="*2w")
    def read_counter_separate_apds(self, c, num_to_read=None):
        complete_counts = self.read_counter_setting_internal(num_to_read)

        # Just find the sum of the counts for each APD for each
        # sample in complete_counts
        return_counts = [
            [np.sum(apd_counts, dtype=int) for apd_counts in sample]
            for sample in complete_counts
        ]

        return return_counts

    @abstractmethod
    def reset(self, c):
        """
        Reset the tagger
        """
        pass

    # @abstractmethod
    # def get_channel_mapping(self, c):
    #     """
    #     do we need this???
    #     """
    #     pass

    @abstractmethod
    def clear_buffer(self, c):
        """
        Clear the buffer of the time tagger if necessary
        """
        pass
