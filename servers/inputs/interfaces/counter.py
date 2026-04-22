# -*- coding: utf-8 -*-
"""
Interface for TTL pulse counters

Created on August 29th, 2022

@author: mccambria

Updated on March 18th, 2026

@author: sbchand
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

    @setting(210, modulus="i", num_to_read="i", returns="**w")
    def read_counter_modulo_gates(self, c, modulus, num_to_read=None):
        complete_counts = self.read_counter_setting_internal(num_to_read)

        separate_gate_counts = []
        for el in complete_counts:
            summed = np.sum(el, axis=0, dtype=np.int64)
            separate_gate_counts.append(np.asarray(summed, dtype=np.int64).reshape(-1))

        return_counts = []
        for sample in separate_gate_counts:
            sample_list = []
            for ind in range(modulus):
                vals = sample[ind::modulus]
                total = int(np.sum(vals, dtype=np.int64)) if vals.size > 0 else 0
                if total < 0:
                    total = 0
                sample_list.append(total)
            return_counts.append(sample_list)

        return_counts = [[int(v) for v in sample] for sample in return_counts]
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

    @setting(212, num_to_read="i", returns="*i")
    def read_counter_summed(self, c, num_to_read=None):
        """Sum all samples server-side, returning one total per gate.

        Transfers only num_gates integers instead of a (num_to_read, num_gates)
        array, making transfer cost constant and negligible regardless of
        num_reps. Works for any number of gates (2 for Rabi/resonance,
        4 for singlet scan, etc.).

        Returns
        -------
        list of ints, length = num_gates
            [gate0_total, gate1_total, ...] summed across all reps and all APDs.
        """
        if self.stream is None:
            logging.error("read_counter attempted while stream is None.")
            return [0, 0]

        totals = None
        num_read = 0
        while num_read < num_to_read:
            chunk = self.read_counter_internal()
            for sample in chunk:
                # sample shape: (num_apds, num_gates)
                # sum APDs axis first → (num_gates,)
                gate_sums = np.sum(sample, axis=0, dtype=np.int64)
                if totals is None:
                    totals = gate_sums.copy()
                else:
                    totals += gate_sums
            num_read += len(chunk)

        if totals is None:
            return [0, 0]
        return totals.tolist()

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
