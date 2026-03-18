# -*- coding: utf-8 -*-
"""
Electron spin resonance routine. Scans the microwave frequency, taking counts
at each point.

Adapted from confocal/resonance.py for CaF2.

Created on Mon Mar 16 2026

@author: j-chen1
"""

from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
from majorroutines import pulsed_resonance
from utils import common
from utils import kplotlib as kpl
from utils import positioning as positioning
from utils import tool_belt as tb
from utils.constants import NormStyle, States
from utils.kplotlib import KplColors


def main(
    nv_sig,
    freq_center,
    freq_range,
    num_steps,
    num_runs,
    uwave_power,
    state=States.LOW,
):
    with common.labrad_connect() as cxn:
        main_with_cxn(
            cxn,
            nv_sig,
            freq_center,
            freq_range,
            num_steps,
            num_runs,
            uwave_power,
            state,
        )


def main_with_cxn(
    cxn,
    nv_sig,
    freq_center,
    freq_range,
    num_steps,
    num_runs,
    uwave_power,
    state=States.LOW,
):
    ### Initial calculations and setup

    tb.reset_cfm(cxn)
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter(cxn)
    pulsegen_server = tb.get_server_pulse_gen(cxn)

    # TODO: Figure out how to add all the following information to nv_sig

    # Set up the laser
    laser_key = "spin_laser"
    laser_name = nv_sig[laser_key]
    laser_power = tb.set_laser_power(cxn, nv_sig, laser_key)

    # Since this is CW we need the imaging readout rather than the spin
    # readout typically used for state detection
    readout = nv_sig["imaging_readout_dur"]
    readout_sec = readout / (10**9)
    norm_style = nv_sig["norm_style"]

    file_name = "resonance_caf.py"  # Points to a copy of the resonance.py sequence file
    seq_args = [readout, state.value, laser_name, laser_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    # print(seq_args)
    # return

    # Calculate the frequencies we need to set
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    freqs = np.linspace(freq_low, freq_high, num_steps)
    freq_ind_list = list(range(num_steps))
    freq_ind_master_list = []

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    ref_counts = np.empty([num_runs, num_steps])
    ref_counts[:] = np.nan
    sig_counts = np.copy(ref_counts)

    opti_coords_list = []

    ### Get the starting time of the function
    start_timestamp = tb.get_time_stamp()

    # Create raw data figure for incremental plotting
    raw_fig, ax_sig_ref, ax_norm = pulsed_resonance.create_raw_data_figure(
        freq_center, freq_range, num_steps
    )
    # Set up a run indicator for incremental plotting
    run_indicator_text = "Run #{}/{}"
    text = run_indicator_text.format(0, num_runs)
    run_indicator_obj = kpl.anchored_text(ax_norm, text, loc=kpl.Loc.UPPER_RIGHT)

    ### Collect the data

    # Start 'Press enter to stop...'
    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print("Run index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tb.safe_stop():
            break

        # Optimize and save the coords we found
        opti_coords = targeting.main_with_cxn(cxn, nv_sig)
        opti_coords_list.append(opti_coords)

        # Laser setup
        tb.set_filter(cxn, nv_sig, laser_key)
        laser_power = tb.set_laser_power(cxn, nv_sig, laser_key)
        # Start the laser now to get rid of transient effects
        # tb.laser_on(cxn, laser_name, laser_power)

        sig_gen_cxn = tb.get_server_sig_gen(cxn, state)
        sig_gen_cxn.set_amp(uwave_power)
        sig_gen_cxn.uwave_on()

        # Load the APD task with two samples for each frequency step
        pulsegen_server.stream_load(file_name, seq_args_string)
        counter_server.start_tag_stream()

        # Shuffle the list of frequency indices so that we step through
        # them randomly
        shuffle(freq_ind_list)
        freq_ind_master_list.append(freq_ind_list)

        # Take a sample and increment the frequency
        for step_ind in range(num_steps):
            # Break out of the while if the user says stop
            if tb.safe_stop():
                break

            freq_ind = freq_ind_list[step_ind]
            # print(freqs[freq_ind])
            sig_gen_cxn.set_freq(freqs[freq_ind])

            # Start the timing stream
            counter_server.clear_buffer()
            pulsegen_server.stream_start()

            # Read the counts using parity to distinguish signal vs ref
            new_counts = counter_server.read_counter_modulo_gates(2, 1)
            sample_counts = new_counts[0]

            cur_run_sig_counts_summed = sample_counts[1]
            cur_run_ref_counts_summed = sample_counts[0]

            sig_counts[run_ind, freq_ind] = cur_run_sig_counts_summed
            ref_counts[run_ind, freq_ind] = cur_run_ref_counts_summed
            # break
            # norm= sum(sig_gate_counts) / sum(ref_gate_counts)
            # print(norm)

        counter_server.stop_tag_stream()

        ### Incremental plotting

        # Update the run indicator
        text = run_indicator_text.format(run_ind + 1, num_runs)
        run_indicator_obj.txt.set_text(text)

        # Average the counts over the iterations
        inc_sig_counts = sig_counts[: run_ind + 1]
        inc_ref_counts = ref_counts[: run_ind + 1]
        ret_vals = tb.process_counts(
            inc_sig_counts, inc_ref_counts, 1, readout, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals

        kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
        kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
        kpl.plot_line_update(ax_norm, y=norm_avg_sig)
        # Save the data we have incrementally for long measurements

        rawData = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            # 'nv_sig-units': tb.get_nv_sig_units(),
            "opti_coords_list": opti_coords_list,
            "opti_coords_list-units": "V",
            "freq_center": freq_center,
            "freq_center-units": "GHz",
            "freq_range": freq_range,
            "freq_range-units": "GHz",
            "num_steps": num_steps,
            "num_runs": num_runs,
            "freq_ind_master_list": freq_ind_master_list,
            "uwave_power": uwave_power,
            "uwave_power-units": "dBm",
            "readout": readout,
            "readout-units": "ns",
            "sig_counts": sig_counts.astype(int).tolist(),
            "sig_counts-units": "counts",
            "ref_counts": ref_counts.astype(int).tolist(),
            "ref_counts-units": "counts",
        }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tb.get_file_path(
            __file__, start_timestamp, nv_sig["name"], "incremental"
        )
        tb.save_raw_data(rawData, file_path)

    ### Process and plot the data

    ret_vals = tb.process_counts(sig_counts, ref_counts, 1, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

    # Raw data
    kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    kpl.plot_line_update(ax_norm, y=norm_avg_sig)
    run_indicator_obj.remove()

    ### Clean up and save the data

    tb.reset_cfm(cxn)

    timestamp = tb.get_time_stamp()

    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        # 'nv_sig-units': tb.get_nv_sig_units(),
        "opti_coords_list": opti_coords_list,
        "opti_coords_list-units": "V",
        "freq_center": freq_center,
        "freq_center-units": "GHz",
        "freq_range": freq_range,
        "freq_range-units": "GHz",
        "num_steps": num_steps,
        "num_runs": num_runs,
        "freq_ind_master_list": freq_ind_master_list,
        "uwave_power": uwave_power,
        "uwave_power-units": "dBm",
        "readout": readout,
        "readout-units": "ns",
        "sig_counts": sig_counts.astype(int).tolist(),
        "sig_counts-units": "counts",
        "ref_counts": ref_counts.astype(int).tolist(),
        "ref_counts-units": "counts",
        "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
        "norm_avg_sig-units": "arb",
        #               'norm_avg_sig_ste': norm_avg_sig_ste.astype(float).tolist(),
        #               'norm_avg_sig_ste-units': 'arb',
    }

    name = nv_sig["name"]
    filePath = tb.get_file_path(__file__, timestamp, name)
    tb.save_figure(raw_fig, filePath)
    tb.save_raw_data(rawData, filePath)

    # Use the pulsed_resonance fitting functions
    fit_func = None
    if False:
        fit_func, popt, pcov = pulsed_resonance.fit_resonance(
            freq_range,
            freq_center,
            num_steps,
            norm_avg_sig,
            norm_avg_sig_ste,
            ref_counts,
        )

    fit_fig = None
    if (fit_func is not None) and (popt is not None):
        fit_fig = pulsed_resonance.create_fit_figure(
            freq_range, freq_center, num_steps, norm_avg_sig, fit_func, popt
        )
    filePath = tb.get_file_path(__file__, timestamp, name + "-fit")
    if fit_fig is not None:
        tb.save_figure(fit_fig, filePath)

    # if fit_func == pulsed_resonance.single_gaussian_dip:
    #     print('Single resonance at {:.4f} GHz'.format(popt[2]))
    #     print('\n')
    #     return popt[2], None
    # elif fit_func == pulsed_resonance.double_gaussian_dip:
    #     print('Resonances at {:.4f} GHz and {:.4f} GHz'.format(popt[2], popt[5]))
    #     print('Splitting of {:d} MHz'.format(int((popt[5] - popt[2]) * 1000)))
    #     print('\n')
    #     return popt[2], popt[5]
    # else:
    #     print('No resonances found')
    #     print('\n')
    return None, None


###

if __name__ == "__main__":
    file = "2022_12_06-15_24_46-johnson-search"
    file_path = "pc_carr/branch_master/resonance/2022_12/incremental"
    data = tb.get_raw_data(file, file_path)

    freq_center = data["freq_center"]
    freq_range = data["freq_range"]
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    ref_counts = data["ref_counts"][0:1]
    sig_counts = data["sig_counts"][0:1]
    print(len(ref_counts))
    ret_vals = tb.process_counts(ref_counts, sig_counts)
    (
        avg_ref_counts,
        avg_sig_counts,
        norm_avg_sig,
        ste_ref_counts,
        ste_sig_counts,
        norm_avg_sig_ste,
    ) = ret_vals
    # norm_avg_sig_ste = None

    fit_func, popt, pcov = pulsed_resonance.fit_resonance(
        freq_center, freq_range, num_steps, norm_avg_sig, norm_avg_sig_ste
    )

    # fit_func, popt, pcov = fit_resonance(freq_range, freq_center, num_steps,
    #                                norm_avg_sig, ref_counts)

    pulsed_resonance.create_fit_figure(
        freq_center, freq_range, num_steps, norm_avg_sig, fit_func, popt
    )
