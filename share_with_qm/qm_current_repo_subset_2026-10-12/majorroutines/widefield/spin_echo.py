# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
@author: schand
"""

import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping, brute
from scipy.signal import lombscargle

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield
from utils.tool_belt import curve_fit


def quartic_decay(
    tau,
    baseline,
    quartic_contrast,
    revival_time,
    quartic_decay_time,
    T2_ms,
    T2_exp,
    osc_contrast=None,
    osc_freq1=None,
    osc_freq2=None,
):
    # baseline = 0.5
    # print(len(amplitudes))
    T2_us = 1000 * T2_ms
    envelope = np.exp(-((tau / T2_us) ** T2_exp))
    # envelope = 1
    num_revivals = 3
    comb = 0
    for ind in range(num_revivals):
        exp_part = np.exp(-(((tau - ind * revival_time) / quartic_decay_time) ** 4))
        comb += exp_part
    if osc_contrast is None:
        mod = quartic_contrast
    else:
        mod = (
            quartic_contrast
            - osc_contrast
            * np.sin(2 * np.pi * osc_freq1 * tau / 2) ** 2
            * np.sin(2 * np.pi * osc_freq2 * tau / 2) ** 2
        )
    val = baseline - envelope * mod * comb
    return val


def quartic_decay_fixed_revival(
    tau,
    baseline,
    quartic_contrast,
    quartic_decay_time,
    T2_ms,
    T2_exp,
    osc_contrast=None,
    osc_freq1=None,
    osc_freq2=None,
):
    return quartic_decay(
        tau,
        baseline,
        quartic_contrast,
        50,
        quartic_decay_time,
        T2_ms,
        T2_exp,
        osc_contrast,
        osc_freq1,
        osc_freq2,
    )


def constant(tau):
    norm = 1
    if isinstance(tau, list):
        return [norm] * len(tau)
    elif type(tau) == np.ndarray:
        return np.array([norm] * len(tau))
    else:
        return norm


def create_raw_data_figure(data):
    nv_list = data["nv_list"]
    taus = data["taus"]
    counts = np.array(data["states"])
    sig_counts, ref_counts = counts[0], counts[1]

    avg_counts, avg_counts_ste, norms = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=False
    )
    # avg_counts -= norms[:, np.newaxis]

    fig, ax = plt.subplots()
    total_evolution_times = 2 * np.array(taus) / 1e3
    widefield.plot_raw_data(
        ax, nv_list, total_evolution_times, avg_counts, avg_counts_ste
    )
    ax.set_xlabel("Total evolution time (µs)")
    ax.set_ylabel("Counts")
    return fig


def brute_fit_fn_cost(
    x,
    total_evolution_times,
    nv_counts,
    nv_counts_ste,
    fit_fn,
    no_c13_popt,
    osc_contrast_guess,
):
    # line = fit_fn(total_evolution_times, *no_c13_popt, *x)
    line = fit_fn(total_evolution_times, *no_c13_popt, osc_contrast_guess, *x)
    return np.sum(((nv_counts - line) ** 2) / (nv_counts_ste**2))


def fit(total_evolution_times, nv_counts, nv_counts_ste):
    # fit_fn = quartic_decay
    fit_fn = quartic_decay_fixed_revival

    ### Get good guesses

    # baseline, quartic_contrast, revival_time, quartic_decay_time, T2_ms, T2_exp, osc_contrast, osc_freq1, osc_freq2,
    baseline_guess = nv_counts[9]
    quartic_contrast_guess = baseline_guess - nv_counts[0]
    # exp(-(0.1/t)**3) == (norm_counts[-6] - baseline_guess) / quartic_contrast_guess
    log_decay = -np.log((baseline_guess - nv_counts[-6]) / quartic_contrast_guess)
    T2_guess = 0.1 * (log_decay ** (-1 / 3))
    # guess_params = [baseline_guess, quartic_contrast_guess, 50, 7, T2_guess, 3]
    guess_params = [baseline_guess, quartic_contrast_guess, 7, T2_guess, 3]
    bounds = [
        # [0, 0, 40, 0, 0, 0],
        # [1, 1, 60, 20, 1000, 10],
        [0, 0, 0, 0, 0],
        [1, 1, 20, 1000, 10],
    ]

    # FFT to determine dominant frequency
    # start = 11
    # stop = 55
    # osc_counts = nv_counts[start:stop] - np.mean(nv_counts[start:stop])
    # transform = np.fft.rfft(osc_counts)
    # time_step = total_evolution_times[start + 1] - total_evolution_times[start]
    # freqs = np.fft.rfftfreq(stop - start, d=time_step)
    # transform_mag = np.absolute(transform)
    # freq_guess = freqs[np.argmax(transform_mag[4:]) + 4]
    # guess_params[-2] = freq_guess

    ### Fit assuming no strongly coupled C13

    no_c13_popt, no_c13_pcov, no_c13_red_chi_sq = curve_fit(
        fit_fn,
        total_evolution_times,
        nv_counts,
        guess_params,
        nv_counts_ste,
        bounds=bounds,
    )

    # return popt, pcov, red_chi_sq

    ### Brute to find correct frequencies

    osc_contrast_guess = no_c13_popt[1]
    args = (
        total_evolution_times,
        nv_counts,
        nv_counts_ste,
        fit_fn,
        no_c13_popt,
        osc_contrast_guess,
    )

    # ranges = [(bounds[0][ind], bounds[1][ind]) for ind in range(len(bounds[0]))]
    # ranges.extend([(-0.5, 0.5), (0, 1.5), (0, 1.5)])
    # ranges = [(-0.5, 0.5), (0, 1.5), (0, 1.5)]
    ranges = [(0, 1.5), (0, 0.5)]
    workers = 6
    popt = brute(
        brute_fit_fn_cost, ranges, Ns=1000, finish=None, workers=workers, args=args
    )

    ### Fine tune with a final fit

    guess_params.append(osc_contrast_guess)
    guess_params.extend(popt)
    bounds[0].extend([-0.5, 0, 0])
    bounds[1].extend([0.5, 1.5, 0.5])

    popt, pcov, red_chi_sq = curve_fit(
        fit_fn,
        total_evolution_times,
        nv_counts,
        guess_params,
        nv_counts_ste,
        bounds=bounds,
    )

    if no_c13_red_chi_sq < red_chi_sq:
        return no_c13_popt, no_c13_pcov, no_c13_red_chi_sq
    else:
        return popt, pcov, red_chi_sq


def create_fit_figure(data, axes_pack=None, layout=None, no_legend=True, nv_inds=None):
    nv_list = data["nv_list"]
    if nv_inds is None:
        num_nvs = len(nv_list)
        nv_inds = list(range(num_nvs))
    else:
        num_nvs = len(nv_inds)
    num_steps = data["num_steps"]
    taus = np.array(data["taus"])
    total_evolution_times = 2 * np.array(taus) / 1e3
    num_runs = data["num_runs"]

    if "norm_counts" in data:
        norm_counts = np.array(data["norm_counts"])
        norm_counts_ste = np.array(data["norm_counts_ste"])
    else:
        counts = np.array(data["counts"])
        sig_counts = counts[0]
        ref_counts = counts[1]
        norm_counts, norm_counts_ste = widefield.process_counts(
            nv_list, sig_counts, ref_counts, threshold=True
        )

    do_fit = False
    if do_fit:
        fit_fns = []
        popts = []

        for nv_ind in nv_inds:
            nv_counts = norm_counts[nv_ind]
            nv_counts_ste = norm_counts_ste[nv_ind]

            # fig, ax = plt.subplots()
            # kpl.plot_points(ax, freqs, transform_mag)
            # kpl.show(block=True)

            try:
                # fit_fn = quartic_decay
                fit_fn = quartic_decay_fixed_revival
                popt, pcov, red_chi_sq = fit(
                    total_evolution_times, nv_counts, nv_counts_ste
                )
                print(f"Red chi sq: {round(red_chi_sq, 3)}")

                fig, ax = plt.subplots()
                kpl.plot_points(ax, total_evolution_times, nv_counts, nv_counts_ste)
                linspace_taus = np.linspace(0, np.max(total_evolution_times), 1000)
                kpl.plot_line(
                    ax,
                    linspace_taus,
                    fit_fn(linspace_taus, *popt),
                    color=kpl.KplColors.GRAY,
                )
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                ax.set_title(nv_ind)
                ax.set_xlabel("Total evolution time (µs)")
                ax.set_ylabel("Normalized NV$^{-}$ population")
                kpl.show(block=True)
            except Exception:
                print(traceback.format_exc())
                fit_fn = None
                popt = None
            fit_fns.append(fit_fn)
            popts.append(popt)
    else:
        fit_fns = None
        popts = None

    ### Make the figure

    figsize = kpl.double_figsize
    figsize[1] = 7
    for ind in range(2):
        layout = kpl.calc_mosaic_layout(num_nvs, num_cols=12)
        fig, axes_pack = plt.subplot_mosaic(
            layout,
            figsize=figsize,
            sharex=True,
            sharey=True,
            gridspec_kw={"hspace": 0.015},
        )

        widefield.plot_fit(
            axes_pack,
            [nv_list[ind] for ind in nv_inds],
            total_evolution_times,
            norm_counts[nv_inds],
            norm_counts_ste[nv_inds],
            fit_fns,
            popts,
            no_legend=no_legend,
            # linestyle="solid",
        )
        ax = axes_pack[layout[-1, 0]]
        kpl.set_shared_ax_xlabel(ax, "Total evolution time (µs)")
        # kpl.set_shared_ax_ylabel(ax, "Change in $P($NV$^{-})$")
        # kpl.set_shared_ax_ylabel(ax, "Norm. NV$^{-}$ population")
        # kpl.set_shared_ax_ylabel(ax, "Normalized NV$^{-}$ population")
        kpl.set_shared_ax_ylabel(ax, "NV$^{-}$ population (arb. units)")
        # ax.set_title(num_runs)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1], [None, None])
        if ind == 1:
            ax.set_xlim(40, 62)
            ax.set_xticks([45, 60])
        else:
            ax.set_xticks([0, 80])

        axes_pack_flat = list(axes_pack.values())
        for ax in axes_pack_flat:
            # ax.tick_params(labelsize=kpl.FontSize.SMALL.value)
            # ax.tick_params(which="both", direction="in", labelsize=kpl.FontSize.SMALL.value)
            ax.tick_params(which="both", direction="in")

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

    return fig


def create_correlation_figure(nv_list, taus, counts):
    total_evolution_times = 2 * np.array(taus) / 1e3

    # fig, ax = plt.subplots()
    fig, axes_pack = plt.subplots(
        nrows=5, ncols=5, sharex=True, sharey=True, figsize=[10, 10]
    )

    widefield.plot_correlations(axes_pack, nv_list, total_evolution_times, counts)

    ax = axes_pack[-1, 0]
    ax.set_xlabel(" ")
    fig.text(0.55, 0.01, "Total evolution time (µs)", ha="center")
    ax.set_ylabel(" ")
    fig.text(0.01, 0.55, "Correlation coefficient", va="center", rotation="vertical")
    return fig


def calc_T2_times(
    peak_total_evolution_times, peak_contrasts, peak_contrast_errs, baselines
):
    for nv_ind in range(len(peak_contrasts)):
        baseline = baselines[nv_ind]

        def envelope(total_evolution_time, T2):
            return (
                -(baseline - 1) * np.exp(-((total_evolution_time / T2) ** 3)) + baseline
            )

        guess_params = (400,)
        popt, pcov = curve_fit(
            envelope,
            peak_total_evolution_times[nv_ind],
            peak_contrasts[nv_ind],
            p0=guess_params,
            sigma=peak_contrast_errs[nv_ind],
            absolute_sigma=True,
        )
        pste = np.sqrt(np.diag(pcov))
        print(f"{round(popt[0])} +/- {round(pste[0])}")


def main(nv_list, num_steps, num_reps, num_runs, min_tau=None, max_tau=None, taus=None):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "spin_echo.py"

    uwave_ind_list = [0, 1]
    # uwave_ind_list = [1]  # iq modulated

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
        run_fn,
        uwave_ind_list=uwave_ind_list,
        # load_iq=True,
    )

    ### Process and plot

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "tau-units": "ns",
        "taus": taus,
        "min_tau": min_tau,
        "max_tau": max_tau,
    }

    # save data
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    # creat fugure and save
    # raw_fig = None
    # try:
    #     # raw_fig = create_raw_data_figure(raw_data)
    #     fit_fig = create_fit_figure(raw_data)
    # except Exception:
    #     print(traceback.format_exc())
    #     # raw_fig = None
    #     fit_fig = None

    ### Clean up and return
    tb.reset_cfm()
    kpl.show()

    # if raw_fig is not None:
    #     dm.save_figure(raw_fig, file_path)
    # if fit_fig is not None:
    #     file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
    #     dm.save_figure(fit_fig, file_path)


if __name__ == "__main__":
    kpl.init_kplotlib()

    ### Fit fn testing

    # fig, ax = plt.subplots()
    # tau_linspace = np.linspace(0, 100, 1000)
    # line = quartic_decay(tau_linspace, 0.5, 0.5, 45, 5, 1000, 1, -0.5, 100, 0.5)
    # kpl.plot_line(ax, tau_linspace, line)
    # kpl.show(block=True)
    # sys.exit()

    ###

    # data = dm.get_raw_data(file_id=1548381879624)

    # Separate files
    # # fmt: off
    # file_ids = [1734158411844, 1734273666255, 1734371251079, 1734461462293, 1734569197701, 1736117258235, 1736254107747, 1736354618206, 1736439112682]
    # file_ids2 = [1736589839249, 1736738087977, 1736932211269, 1737087466998, 1737219491182]
    # # fmt: on
    # file_ids = file_ids[:4]
    # file_ids.extend(file_ids2)
    # data = dm.get_raw_data(file_id=file_ids)

    # Combined file
    data = dm.get_raw_data(file_id=1755199883770)

    split_esr = [12, 13, 14, 61, 116]
    broad_esr = [52, 11]
    weak_esr = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    skip_inds = list(set(split_esr + broad_esr + weak_esr))
    nv_inds = [ind for ind in range(117) if ind not in skip_inds]

    # create_raw_data_figure(data)
    create_fit_figure(data, nv_inds=nv_inds)

    plt.show(block=True)
