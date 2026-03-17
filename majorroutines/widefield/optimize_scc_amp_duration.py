# -*- coding: utf-8 -*-
"""
Optimize SCC parameters
Created on December 6th, 2023
@author: mccambria
Updated on Fall 2024
@author: schand
"""

import itertools
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig, VirtualLaserKey
from utils.tool_belt import curve_fit


def process_and_plot(data):
    """
    Process and plot the results of the SCC optimization experiment.
    Parameters
    ----------
    data : dict
        Raw data object from the experiment
    Returns
    -------
    figs : list
        List of matplotlib figures generated during plotting.
    """
    # Unpack
    nv_list = data["nv_list"]
    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]
    step_vals = np.array(data["step_vals"])
    num_amp_steps = data["num_amp_steps"]
    num_dur_steps = data["num_dur_steps"]
    # num_amp_steps = 15
    # num_dur_steps = 17
    min_amp = data["min_amp"]
    max_amp = data["max_amp"]
    min_dur = data["min_duration"]
    max_dur = data["max_duration"]
    amp_vals = np.linspace(min_amp, max_amp, num_amp_steps)
    duration_vals = np.linspace(min_dur, max_dur, num_dur_steps).astype(int)
    duration_linspace = np.linspace(min_dur, max_dur, 100)
    num_nvs = len(nv_list)
    # Process the counts
    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=True
    )
    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    opti_inds = [np.argmax(avg_snr[nv_ind]) for nv_ind in range(num_nvs)]
    opti_dur_guesses = [step_vals[opti_inds[nv_ind], 0] for nv_ind in range(num_nvs)]
    opti_amp_guesses = [step_vals[opti_inds[nv_ind], 1] for nv_ind in range(num_nvs)]
    opti_snr_guesses = [avg_snr[nv_ind, opti_inds[nv_ind]] for nv_ind in range(num_nvs)]

    # print(f"SNR: {opti_snrs}")
    # print(f"Median SNR: {np.median(opti_snrs)}")
    # return
    # Fit to the linecut at each amplitude value to determine optimal parameters
    def fit_fn(tau, delay, slope, dec):
        tau = np.array(tau) - delay
        return slope * tau * np.exp(-tau / dec)

    opti_snrs = []
    opti_amps = []
    opti_durs = []
    for nv_ind in range(num_nvs):
        lines = []
        opti_snr = 0
        snrs_2d = np.reshape(avg_snr[nv_ind], (num_dur_steps, num_amp_steps)).T
        snr_errs_2d = np.reshape(avg_snr_ste[nv_ind], (num_dur_steps, num_amp_steps)).T
        for amp_ind in range(num_amp_steps):
            slope_guess = opti_snr_guesses[nv_ind] / opti_dur_guesses[nv_ind]
            # slope_guess = opti_snr_guesses[nv_ind] / opti_amp_guesses[nv_ind]
            guess_params = [20, slope_guess, 300]
            try:
                popt, pcov, red_chi_sq = curve_fit(
                    fit_fn,
                    duration_vals,
                    # amp_vals,
                    snrs_2d[amp_ind],
                    guess_params,
                    snr_errs_2d[amp_ind],
                )
            except Exception:
                lines.append(None)
                continue
            line = fit_fn(duration_linspace, *popt)
            lines.append(line)
            opti_snr_at_amp = np.max(line)
            if opti_snr_at_amp > opti_snr:
                opti_snr = opti_snr_at_amp
                opti_dur = duration_linspace[np.argmax(line)]
                opti_amp = amp_vals[amp_ind]
        # Plot
        # if opti_dur == 32:
        fig, ax = plt.subplots()
        for amp_ind in range(num_amp_steps):
            kpl.plot_points(ax, duration_vals, snrs_2d[amp_ind])
            # kpl.plot_points(ax, amp_vals, snrs_2d[amp_ind])
            line = lines[amp_ind]
            if line is not None:
                kpl.plot_line(ax, duration_linspace, line, label=amp_vals[amp_ind])
        ax.set_xlabel("Duration (ns)")
        ax.set_ylabel("SNR")
        ax.set_title(f"NV num: {widefield.get_nv_num(nv_list[nv_ind])}")
        ax.legend()
        kpl.show(block=True)
        print(opti_snr)
        print(opti_amp)
        print(opti_dur)
        opti_snrs.append(round(opti_snr, 3))
        opti_amps.append(round(opti_amp, 3))
        opti_durs.append(round(opti_dur / 4) * 4)
        # Plot
        # fig, ax = plt.subplots()
        # kpl.imshow(
        #     ax,
        #     snrs_2d,
        #     extent=(min_dur, max_dur, min_amp, max_amp),
        #     aspect="auto",
        #     origin="lower",
        #     cbar_label="SNR",
        #     x_label="Duration (ns)",
        #     y_label="Amplitude (arb.)",
        # )
        # ax.set_title(f"NV index: {nv_ind}")
        # kpl.show(block=True)
    # Report results
    opt_dur_med = round(np.median(opti_durs))
    opti_durs = [opt_dur_med if not (32 < val < 200) else val for val in opti_durs]
    print(opti_snrs)
    print(opti_amps)
    print(opti_durs)
    print(f"Median SNR: {np.median(opti_snrs)}")
    print(f"Median Durs: {np.median(opti_durs)}")


def optimize_scc_amp_and_duration(
    nv_list,
    num_amp_steps,
    num_dur_steps,
    num_reps,
    num_runs,
    min_amp,
    max_amp,
    min_duration,
    max_duration,
):
    """
    Main function to optimize SCC parameters over amplitude and duration.

    Parameters
    ----------
    nv_list : list
        List of NVs to optimize.
    num_amp_steps : int
        Number of steps for amplitude.
    num_dur_steps : int
        Number of steps for duration.
    num_reps : int
        Number of repetitions for each step.
    num_runs : int
        Number of experimental runs.
    min_step_val : tuple
        Minimum values for duration and amplitude.
    max_step_val : tuple
        Maximum values for duration and amplitude.

    Returns
    -------

    None
    """
    ### Initial setup
    seq_file = "optimize_scc_amp_duration.py"
    # Generate grid of parameter values
    duration_vals = np.linspace(min_duration, max_duration, num_dur_steps).astype(int)
    amp_vals = np.linspace(min_amp, max_amp, num_amp_steps)
    step_vals = np.array(list(itertools.product(duration_vals, amp_vals)))
    # step_vals = np.array(np.meshgrid(duration_vals, amp_vals)).T.reshape(-1, 2)
    num_steps = num_amp_steps * num_dur_steps
    uwave_ind_list = [0, 1]
    # uwave_ind_list = [1]  # iq modulated
    pulse_gen = tb.get_server_pulse_gen()

    ### Define run function
    def run_fn(shuffled_step_inds):
        shuffled_step_vals = step_vals[shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_step_vals,
        ]

        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    ### Run the experiment
    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,
    )

    ### Process and plot results

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "step_vals": step_vals,
        "step-units": ["ns", "relative"],
        "num_amp_steps": num_amp_steps,
        "num_dur_steps": num_dur_steps,
        "min_amp": min_amp,
        "max_amp": max_amp,
        "min_duration": min_duration,
        "max_duration": max_duration,
    }

    try:
        figs = process_and_plot(raw_data)
    except Exception as e:
        print("Error in process_and_plot:", e)
        print(traceback.format_exc())
        figs = None

    ### Save results

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    if "img_arrays" in raw_data:
        keys_to_compress = ["img_arrays"]
    else:
        keys_to_compress = None
    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    if figs is not None:
        for ind, fig in enumerate(figs):
            file_path = dm.get_file_path(__file__, timestamp, f"{repr_nv_name}-{ind}")
            dm.save_figure(fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()
    # data = dm.get_raw_data(file_id=1737831146138)
    # data = dm.get_raw_data(file_id=1781811582216)
    # data = dm.get_raw_data(file_id=1797149751810)  # amps variation
    # data = dm.get_raw_data(file_id=1800806173978)  # durs variation
    file_stem = "2025_09_24-11_56_41-rubin-nv0_2025_09_08"
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True)  # durs variation
    process_and_plot(data)
    plt.show(block=True)
