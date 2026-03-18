# -*- coding: utf-8 -*-
"""
Interface for TTL pulse time taggers

Created on August 29th, 2022

@author: mccambria
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from labrad.server import setting
from numba import jit, njit

from servers.inputs.interfaces.counter import Counter


class Tagger(Counter, ABC):
    @abstractmethod
    def start_tag_stream(self, c, apd_indices=None, gate_indices=None, clock=True):
        """
        Start a tag stream
        Note: These inputs are necessary for the swabian time taggers. The OPX just needs
        the apd_indices to know which apds to play measure() statements on, but that can live in the config and be pulled from there in the sequence.

        Parameters
        ----------
        apd_indices : list
            Indicates the channels for which apds we are using
        gate_indices : list, optional
            Indicates the channels for the gates corresponding to the apds
        clock : boolean, optional
            Indicates if using a clock with the tagger
        """

        pass

    @abstractmethod
    def stop_tag_stream(self, c):
        """
        Stop a tag stream
        """
        pass

    @setting(301, num_to_read="i", returns="*s*i")
    def read_tag_stream(self, c, num_to_read=None):
        """Read the stream started with start_tag_stream. Returns two lists,
        each as long as the number of counts that have occurred since the
        buffer was refreshed. First list is timestamps in ps, second is
        channel names
        """
        if self.stream is None:
            logging.error("read_tag_stream attempted while stream is None.")
            return
        if num_to_read is None:
            timestamps, channels = self.read_raw_stream()
        else:
            timestamps = np.array([], dtype=np.int64)
            channels = np.array([], dtype=int)
            num_read = 0
            while True:
                # logging.info('in the while loop')
                # logging.info(num_read)
                timestamps_chunk, channels_chunk = self.read_raw_stream()
                timestamps = np.append(timestamps, timestamps_chunk)
                channels = np.append(channels, channels_chunk)
                # Check if we've read enough samples
                new_num_read = np.count_nonzero(channels_chunk == self.tagger_di_clock)
                num_read += new_num_read
                if num_read >= num_to_read:
                    break
        # Convert timestamps to strings since labrad does not support int64s
        # It must be converted to int64s back on the client
        timestamps = timestamps.astype(str).tolist()
        return timestamps, channels


@njit
def tags_to_counts(
    buffer_channels,
    clock_channel,
    apd_gate_channel,
    apd_channels,
    leftover_channels,
):
    open_channel = apd_gate_channel
    close_channel = -open_channel

    clock_click_inds = np.flatnonzero(buffer_channels == clock_channel)

    previous_sample_end_ind = None
    sample_end_ind = None

    num_samples_max = len(clock_click_inds)
    num_apds = len(apd_channels)

    data_structure_allocated = False
    valid_sample_count = 0
    num_reps = 0

    for dim1 in range(num_samples_max):
        clock_click_ind = clock_click_inds[dim1]
        sample_end_ind = clock_click_ind + 1

        if previous_sample_end_ind is None:
            n_left = len(leftover_channels)
            n_new = sample_end_ind
            sample_channels = np.empty(n_left + n_new, dtype=np.int32)
            if n_left > 0:
                sample_channels[:n_left] = leftover_channels
            if n_new > 0:
                sample_channels[n_left:] = buffer_channels[0:sample_end_ind]
        else:
            sample_channels = buffer_channels[previous_sample_end_ind:sample_end_ind]

        open_inds = np.flatnonzero(sample_channels == open_channel)
        close_inds = np.flatnonzero(sample_channels == close_channel)

        num_reps_this_sample = min(len(open_inds), len(close_inds))
        if num_reps_this_sample == 0:
            previous_sample_end_ind = sample_end_ind
            continue

        if not data_structure_allocated:
            num_reps = num_reps_this_sample
            return_counts = np.zeros((num_samples_max, num_apds, num_reps), dtype=np.int32)
            data_structure_allocated = True

        reps_to_use = min(num_reps, num_reps_this_sample)

        for dim2 in range(num_apds):
            apd_channel = apd_channels[dim2]

            for dim3 in range(num_reps):
                if dim3 < reps_to_use:
                    start_ind = open_inds[dim3]
                    stop_ind = close_inds[dim3]

                    if stop_ind > start_ind:
                        num_counts = np.count_nonzero(
                            sample_channels[start_ind:stop_ind] == apd_channel
                        )
                    else:
                        num_counts = 0
                else:
                    num_counts = 0

                return_counts[valid_sample_count, dim2, dim3] = num_counts

        valid_sample_count += 1
        previous_sample_end_ind = sample_end_ind

    if not data_structure_allocated:
        return_counts = np.empty((0, 0, 0), dtype=np.int32)
        leftover_channels = np.append(leftover_channels, buffer_channels)
    else:
        return_counts = return_counts[:valid_sample_count]
        leftover_channels = buffer_channels[sample_end_ind:]

    return return_counts, leftover_channels
