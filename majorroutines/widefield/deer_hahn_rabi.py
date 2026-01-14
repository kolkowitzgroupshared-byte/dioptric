# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment.

Runs a widefield Rabi scan over pulse duration (taus) for many NVs using IQ-modulated
microwaves, saves raw data, and provides optional analysis/plots:
  - mean Rabi across all NVs (and by orientation groups)
  - per-NV Rabi fits and correlation plots (optional)

Updated: Jan 6, 2025 (Saroj Chand)
"""


import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.constants import NVSig


def create_mean_figure(data):
    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    taus = data["taus"]

    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    nv_nums = [widefield.get_nv_num(nv) for nv in nv_list]
    num_nvs = len(nv_list)
    orientation_data = dm.get_raw_data(file_id=1723161184641)
    orientation_a_nums = orientation_data["orientation_indices"]["0.041"]["nv_indices"]
    orientation_b_nums = orientation_data["orientation_indices"]["0.147"]["nv_indices"]
    orientation_a_inds = [nv_nums[ind] in orientation_a_nums for ind in range(num_nvs)]
    orientation_b_inds = np.logical_not(orientation_a_inds)

    avg_counts = np.mean(norm_counts, axis=0)
    avg_counts_ste = np.mean(norm_counts_ste, axis=0) / np.sqrt(num_nvs)
    avg_counts_a = np.mean(norm_counts[orientation_a_inds], axis=0)
    avg_counts_ste_a = np.mean(norm_counts_ste[orientation_a_inds], axis=0) / np.sqrt(
        num_nvs
    )
    avg_counts_b = np.mean(norm_counts[orientation_b_inds], axis=0)
    avg_counts_ste_b = np.mean(norm_counts_ste[orientation_b_inds], axis=0) / np.sqrt(
        num_nvs
    )

    fig, ax = plt.subplots()
    kpl.plot_points(ax, taus, avg_counts, avg_counts_ste, label="Mean All NVs")
    kpl.plot_points(
        ax, taus, avg_counts_a, avg_counts_ste_a, label="Mean Orientation A"
    )
    kpl.plot_points(
        ax, taus, avg_counts_b, avg_counts_ste_b, label="Mean Orientation B"
    )

    def cos_decay(tau, amp, freq, decay, delay):
        envelope = np.exp(-(tau - delay) / abs(decay)) * amp
        cos_part = np.cos(2 * np.pi * freq * (tau - delay))
        return amp - (envelope * cos_part)

    p0 = [0.5, 1 / 100, 1000, 5]
    popt, _, _ = tb.curve_fit(cos_decay, taus, avg_counts, p0, avg_counts_ste)
    popt_a, _, _ = tb.curve_fit(cos_decay, taus, avg_counts_a, p0, avg_counts_ste_a)
    popt_b, _, _ = tb.curve_fit(cos_decay, taus, avg_counts_b, p0, avg_counts_ste_b)
    tau_linspace = np.linspace(0, max(taus), 1000)
    kpl.plot_line(ax, tau_linspace, cos_decay(tau_linspace, *popt))
    kpl.plot_line(ax, tau_linspace, cos_decay(tau_linspace, *popt_a))
    kpl.plot_line(ax, tau_linspace, cos_decay(tau_linspace, *popt_b))

    ax.legend()


def create_raw_data_figure(nv_list, taus, counts, counts_ste):
    fig, ax = plt.subplots()
    widefield.plot_raw_data(ax, nv_list, taus, counts, counts_ste)
    ax.set_xlabel("Pulse duration (ns)")
    ax.set_ylabel("Fraction in NV$^{-}$")
    return fig


def create_fit_figure(nv_list, taus, counts, counts_ste, norms):
    ### Do the fitting

    taus = np.array(taus)

    def cos_decay(tau, freq, decay, tau_phase):
        # def cos_decay(tau, ptp_amp, freq, decay, tau_offset):
        amp = 0.5
        # amp = abs(ptp_amp) / 2
        envelope = np.exp(-tau / abs(decay)) * amp
        cos_part = np.cos((2 * np.pi * freq * (tau - tau_phase)))
        sign = 1
        # sign = np.sign(ptp_amp)
        return amp - sign * (envelope * cos_part)

    def constant(tau):
        norm = 0
        if isinstance(tau, list):
            return [norm] * len(tau)
        elif type(tau) == np.ndarray:
            return np.array([norm] * len(tau))
        else:
            return norm

    num_nvs = len(nv_list)
    tau_step = taus[1] - taus[0]
    num_steps = len(taus)

    # Single-valued norm
    # # norms_newaxis = norms[:, np.newaxis]
    # norms_newaxis = norms[0][:, np.newaxis]
    # norm_counts = counts - norms_newaxis
    # norm_counts_ste = counts_ste

    # Dual-valued norm
    norms_ms0_newaxis = norms[0][:, np.newaxis]
    norms_ms1_newaxis = norms[1][:, np.newaxis]
    contrast = norms_ms1_newaxis - norms_ms0_newaxis
    norm_counts = (counts - norms_ms0_newaxis) / contrast
    norm_counts_ste = counts_ste / contrast

    fit_fns = []
    popts = []
    for nv_ind in range(num_nvs):
        nv_sig = nv_list[nv_ind]
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        transform = np.fft.rfft(nv_counts)
        freqs = np.fft.rfftfreq(num_steps, d=tau_step)
        transform_mag = np.absolute(transform)
        max_ind = np.argmax(transform_mag[1:])  # Exclude DC component
        freq_guess = freqs[max_ind + 1]
        angular_freq_guess = 2 * np.pi * freq_guess
        tau_phase_guess = -np.angle(transform[max_ind + 1]) / angular_freq_guess
        guess_params = [freq_guess, 1000, tau_phase_guess]
        # guess_params = [ptp_amp_guess, freq_guess, 1000, 0]
        fit_fn = cos_decay

        try:
            if nv_ind in [0, 1, 4, 6]:
                fit_fn = constant
                popt = None
            else:
                popt, pcov = curve_fit(
                    fit_fn,
                    taus,
                    nv_counts,
                    p0=guess_params,
                    sigma=nv_counts_ste,
                    absolute_sigma=True,
                )
        except Exception:
            fit_fn = None
            popt = None

        fit_fns.append(fit_fn)
        popts.append(popt)

        if popt is not None:
            residuals = fit_fn(taus, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
            print(f"Red chi sq: {round(red_chi_sq, 3)}")

    rabi_periods = [None if el is None else round(1 / el[0], 2) for el in popts]
    tau_offsets = [None if el is None else round(el[-1], 2) for el in popts]
    print(f"rabi_periods: {rabi_periods}")
    print(f"tau_offsets: {tau_offsets}")

    ### Make the figure

    layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)
    # layout = kpl.calc_mosaic_layout(num_nvs)
    figsize = [6.5, 6.0]
    figsize = [6.5, 5.0]
    fig, axes_pack = plt.subplot_mosaic(
        layout, figsize=figsize, sharex=True, sharey=True
    )
    axes_pack_flat = list(axes_pack.values())

    widefield.plot_fit(
        axes_pack_flat,
        nv_list,
        taus,
        norm_counts,
        norm_counts_ste,
        fit_fns,
        popts,
        xlim=[0, None],
        no_legend=True,
    )

    ax = axes_pack[layout[-1][0]]
    kpl.set_shared_ax_xlabel(ax, "Pulse duration (ns)")
    # kpl.set_shared_ax_ylabel(ax, "Change in $P($NV$^{-})$")
    # kpl.set_shared_ax_ylabel(ax, "$P($NV$^{-})$")
    kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ population")
    # kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ pop.")

    # ax.set_ylim([0.966, 1.24])
    # ax.set_yticks([1.0, 1.2])
    ax.set_xticks([0, 200])
    ax.set_yticks([0, 1])

    return fig


def create_correlation_figure(nv_list, taus, counts):
    ### Make the figure

    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(
        nrows=5, ncols=5, sharex=True, sharey=True, figsize=[10, 10]
    )

    widefield.plot_correlations(axes_pack, nv_list, taus, counts)

    ax = axes_pack[-1, 0]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Pulse duration (ns)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Correlation coefficient", va="center", rotation="vertical")
    return fig


def main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "deer_hahn_rabi.py"
    taus = np.linspace(min_tau, max_tau, num_steps)
    taus = [int(4 * round(tau / 4)) for tau in taus]

    ### Collect the data
    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
        ]

        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        # load_iq=True,
    )

    ### save the raw data
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": max_tau,
        "max_tau": max_tau,
    }
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    ### Process and plot
    try:
        raw_fig = None
        fit_fig = None
        # counts = raw_data["counts"]
        # sig_counts = counts[0]
        # ref_counts = counts[1]
        # avg_counts, avg_counts_ste, norms = widefield.process_counts(
        #     nv_list, sig_counts, ref_counts, threshold=True
        # )

        # raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
        # fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste, norms)
    except Exception:
        print(traceback.format_exc())
        raw_fig = None
        fit_fig = None

    ### Clean up and return

    tb.reset_cfm()
    kpl.show()

    if raw_fig is not None:
        dm.save_figure(raw_fig, file_path)
    if fit_fig is not None:
        file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
        dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1772297872545, load_npz=False, use_cache=True)
    create_mean_figure(data)
    kpl.show(block=True)
    sys.exit()

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    taus = data["taus"]

    counts = np.array(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    raw_fig = create_raw_data_figure(nv_list, taus, avg_counts, avg_counts_ste)
    fit_fig = create_fit_figure(nv_list, taus, avg_counts, avg_counts_ste, norms)

    kpl.show(block=True)

    ###

    pixel_drifts = data["pixel_drifts"]
    img_arrays = np.array(data["img_arrays"])
    base_pixel_drift = [15, 45]
    # base_pixel_drift = [24, 74]
    num_reps = 1

    buffer = 30
    img_array_size = 250
    cropped_size = img_array_size - 2 * buffer
    proc_img_arrays = np.empty(
        (2, num_runs, num_steps, num_reps, cropped_size, cropped_size)
    )
    for run_ind in range(num_runs):
        pixel_drift = pixel_drifts[run_ind]
        offset = [
            pixel_drift[1] - base_pixel_drift[1],
            pixel_drift[0] - base_pixel_drift[0],
        ]
        for step_ind in range(num_steps):
            for exp_ind in range(2):
                img_array = img_arrays[exp_ind, run_ind, step_ind, 0]
                cropped_img_array = widefield.crop_img_array(img_array, offset, buffer)
                proc_img_arrays[exp_ind, run_ind, step_ind, 0, :, :] = cropped_img_array

    # Average over cosmic ray
    proc_img_arrays[0, 117, -1, 0, :, :] = np.nan
    sig_img_arrays = np.nanmean(proc_img_arrays, axis=(1, 3))[0]
    ref_img_array = np.nanmean(proc_img_arrays, axis=(1, 2, 3))[1]
    proc_img_arrays = sig_img_arrays - ref_img_array

    downsample_factor = 2
    proc_img_arrays = [
        widefield.downsample_img_array(el, downsample_factor) for el in proc_img_arrays
    ]
    proc_img_arrays = np.array(proc_img_arrays)
    proc_img_arrays = np.array(proc_img_arrays)

    widefield.animate(
        taus,
        nv_list,
        avg_counts,
        avg_counts_ste,
        norms,
        proc_img_arrays,
        cmin=np.percentile(proc_img_arrays, 60),
        cmax=np.percentile(proc_img_arrays, 99.9),
        scale_bar_length_factor=downsample_factor,
    )

    ###

    plt.show(block=True)
