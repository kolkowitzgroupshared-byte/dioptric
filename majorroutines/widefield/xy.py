# -*- coding: utf-8 -*-
"""
Widefield Rabi experiment

Created on November 29th, 2023

@author: mccambria
@author: sbchand
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import utils.tool_belt as tb
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield
from utils import widefield as widefield


def process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste):
    num_nvs = len(nv_list)

    # Fit function: stretched exponential decay
    def stretched_exp(tau, a, t2, n, b):
        return a * (1 - np.exp(-((tau / t2) ** n))) + b

    for nv_ind in range(num_nvs):
        nv_counts = norm_counts[nv_ind]
        nv_counts_ste = norm_counts_ste[nv_ind]

        # Initial guesses: full contrast, T2 ~ max tau, stretch ~1, baseline ~0.5
        a0 = np.ptp(nv_counts)
        t2_0 = taus[-1] / 2
        n0 = 1.0
        b0 = np.min(nv_counts)
        p0 = [a0, t2_0, n0, b0]

        try:
            popt, _ = curve_fit(
                stretched_exp,
                taus,
                nv_counts,
                p0=p0,
                sigma=nv_counts_ste,
                absolute_sigma=True,
                maxfev=10000,
            )
        except Exception as e:
            print(f"NV {nv_ind} fit failed: {e}")
            popt = None

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(
            taus,
            nv_counts,
            yerr=np.abs(nv_counts_ste),
            fmt="o",
            capsize=3,
            label=f"NV {nv_ind}",
        )

        if popt is not None:
            tau_fit = np.linspace(min(taus), max(taus), 300)
            fit_vals = stretched_exp(tau_fit, *popt)
            ax.plot(tau_fit, fit_vals, "-", label="Fit")

            # χ²
            residuals = stretched_exp(taus, *popt) - nv_counts
            chi_sq = np.sum((residuals / nv_counts_ste) ** 2)
            red_chi_sq = chi_sq / (len(nv_counts) - len(popt))
            print(
                f"NV {nv_ind} - T2 = {popt[1]:.1f} ns, n = {popt[2]:.2f}, χ² = {red_chi_sq:.2f}"
            )

        ax.set_title(f"XY8 Decay: NV {nv_ind}")
        ax.set_xlabel("τ (ns)")
        ax.set_ylabel("Normalized NV⁻ Population")
        ax.set_xscale("symlog", linthresh=2e5)
        ax.legend()
        ax.grid(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.show(block=True)


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    taus,
    uwave_ind_list,
    xy_seq,
):
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    # seq_file = "xy8.py"
    seq_file = "xy.py"
    # seq_file = "xycp.py"

    # old version of exp run using step function
    # def step_fn(tau_ind):
    #     tau = taus[tau_ind]
    #     seq_args = widefield.get_base_scc_seq_args(nv_list)
    #     seq_args.append(tau)
    #     seq_args_string = tb.encode_seq_args(seq_args)
    #     pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    def run_fn(shuffled_step_inds):
        shuffled_taus = [taus[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_taus,
            xy_seq,
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
        load_iq=True,
    )
    
    ### save data
    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "xy_seq": xy_seq,
    }

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    ### Clean up and return
    tb.reset_cfm()

    ### Process and plot
    # try:
    #     counts = np.array(raw_data["counts"])  # shape: (2, num_nvs, num_steps)
    #     sig_counts = counts[0]
    #     ref_counts = counts[1]
    #     norm_counts, norm_counts_ste = widefield.process_counts(
    #         nv_list, sig_counts, ref_counts, threshold=True
    #     )
    #     # process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste)
    # except Exception as exc:
    #     print(exc)
    #     fig = None

    # if fig is not None:
    #     dm.save_figure(fig, file_path)
    #     file_path = dm.get_file_path(__file__, timestamp, repr_nv_name + "-fit")
    #     dm.save_figure(fit_fig, file_path)
    # kpl.show()


if __name__ == "__main__":
    kpl.init_kplotlib()

    # file_name = ""
    raw_data = dm.get_raw_data(file_id=1818240906171)

    nv_list = raw_data["nv_list"]
    taus = np.array(raw_data["taus"])  # τ values (in ns)
    counts = np.array(raw_data["counts"])  # shape: (2, num_nvs, num_steps)
    sig_counts = counts[0]
    ref_counts = counts[1]

    # Normalize counts
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig_counts, ref_counts, threshold=True
    )

    num_nvs = len(nv_list)

    process_and_plot_xy8(nv_list, taus, norm_counts, norm_counts_ste)

    plt.show(block=True)
