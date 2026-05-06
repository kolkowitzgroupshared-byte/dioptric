# -*- coding: utf-8 -*-
"""
Interface for TTL pulse time taggers

Created on August 29th, 2022

@author: mccambria

Updated on March 18th, 2026

@author: sbchand

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
    """Convert raw time-tagger channel events into per-sample APD counts.

    This is the core counter-parsing function used by the tagger server.
    It is kept in a standalone numba-jitted function so it runs fast enough
    for streaming measurements.

    Parameters
    ----------
    buffer_channels : array(int)
        Channel IDs returned by the most recent tagger read.
    clock_channel : int
        Channel used as the sample delimiter. Each clock click ends one sample.
    apd_gate_channel : int
        Virtual gate channel for the APD gate. A positive edge opens the gate,
        and the corresponding negative channel closes it.
    apd_channels : array(int)
        Physical APD input channels to count within each gate window.
    leftover_channels : array(int)
        Channel events left over from the previous read. These are events that
        arrived after the last fully clocked sample and therefore must be
        prepended to the first sample in the current buffer.

    Returns
    -------
    3D array(int)
        return_counts with shape:
            (num_valid_samples, num_apds, num_reps)
        where:
            - first dimension indexes clocked samples
            - second dimension indexes APDs
            - third dimension indexes gate/repetition number within a sample
    array(int)
        Updated leftover_channels for the next read.
    """

    # The APD gate is represented by a rising edge on apd_gate_channel and
    # a falling edge on the corresponding negative channel.
    open_channel = apd_gate_channel
    close_channel = -apd_gate_channel

    # Clock clicks mark the end of each sample.
    clock_click_inds = np.flatnonzero(buffer_channels == clock_channel)

    previous_sample_end_ind = None
    sample_end_ind = None

    # Maximum possible number of samples is the number of clock clicks.
    num_samples_max = len(clock_click_inds)
    num_apds = len(apd_channels)

    # We do not know num_reps until we see the first valid sample that contains
    # at least one matched open/close gate pair.
    data_structure_allocated = False
    valid_sample_count = 0
    num_reps = 0

    for dim1 in range(num_samples_max):
        clock_click_ind = clock_click_inds[dim1]

        # A clock click terminates the sample, so include it in the slice end.
        sample_end_ind = clock_click_ind + 1

        # For the first sample in this buffer, prepend leftovers from the
        # previous read. For later samples, just slice between clock boundaries.
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

        # Find all APD gate open and close edges inside this sample.
        open_inds = np.flatnonzero(sample_channels == open_channel)
        close_inds = np.flatnonzero(sample_channels == close_channel)

        # A valid repetition requires both an open and a close edge.
        # If the sample has incomplete gate information, skip it.
        num_reps_this_sample = min(len(open_inds), len(close_inds))
        if num_reps_this_sample == 0:
            previous_sample_end_ind = sample_end_ind
            continue

        # Allocate the output array once we know how many repetitions/gates
        # are present in a valid sample.
        if not data_structure_allocated:
            num_reps = num_reps_this_sample
            return_counts = np.zeros(
                (num_samples_max, num_apds, num_reps), dtype=np.int32
            )
            data_structure_allocated = True

        # If later samples have fewer complete gates than the first valid sample,
        # only use the gates that exist and leave the rest as zero.
        reps_to_use = min(num_reps, num_reps_this_sample)

        for dim2 in range(num_apds):
            apd_channel = apd_channels[dim2]

            for dim3 in range(num_reps):
                if dim3 < reps_to_use:
                    start_ind = open_inds[dim3]
                    stop_ind = close_inds[dim3]

                    # Only count photons if the close edge comes after the open edge.
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

    # If no clocked sample contained a complete gate pair, return an empty count
    # array and keep everything as leftover for the next read.
    if not data_structure_allocated:
        return_counts = np.empty((0, 0, 0), dtype=np.int32)
        leftover_channels = np.append(leftover_channels, buffer_channels)
    else:
        # Trim off unused rows from preallocation.
        return_counts = return_counts[:valid_sample_count]

        # Anything after the last fully processed sample becomes leftover for
        # the next buffer read.
        leftover_channels = buffer_channels[sample_end_ind:]

    return return_counts, leftover_channels