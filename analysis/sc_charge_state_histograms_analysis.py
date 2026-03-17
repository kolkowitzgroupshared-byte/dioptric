# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera. Interleave a signal and control sequence
and plot the difference, while fitting a bimodal distribution to NV charge states.

Created on Fall 2024

@author: saroj chand
"""

import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import factorial

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
    analyze_charge_histogram_multinv_binomial,
)
from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey


# region Process and plotting functions
def plot_histograms(
    sig_counts_list, ref_counts_list, no_title=True, ax=None, density=False
):
    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    # laser_dict = tb.get_virtual_laser_dict(laser_key)
    # readout = laser_dict["duration"]
    # readout_ms = int(readout / 1e6)
    # readout_s = readout / 1e9

    ### Histograms
    num_reps = len(ref_counts_list)
    labels = ["With ionization pulse", "Without ionization pulse"]
    # colors = [kpl.KplColors.RED, kpl.KplColors.GREEN]
    colors = [kpl.KplColors.RED, kpl.KplColors.BLUE]  # MCC
    counts_lists = [sig_counts_list, ref_counts_list]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    if not no_title:
        ax.set_title(f"Charge prep hist, {num_reps} reps")
    ax.set_xlabel("Integrated counts")
    if density:
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("Number of occurrences")

    for ind in range(2):
        # if ind == 0:
        #     continue
        if counts_lists is None or len(counts_lists) == 0:
            continue
        counts_list = counts_lists[ind]
        label = labels[ind]
        color = colors[ind]
        # kpl.histogram(ax, counts_list, label=label, color=color, density=density)  # MCC
        kpl.histogram(ax, counts_list, color=color, density=density)

    # ax.legend() # MCC
    # ax.tick_params(axis="y", rotation=90)
    ax.set_xlim(-0.5, None)
    # ax.set_yticks([0, 0.04, 0.08])

    if fig is not None:
        return fig

    
def process_and_plot(
    raw_data,
    do_plot_histograms=False,
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
    max_nvs_per_position: int = 3,
    force_nvs: int | None = None,   # set 1/2/3 if you know from imaging
):
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)

    weak_esr = [72, 64, 55, 96, 112, 87, 12, 58, 36]

    counts = np.array(raw_data["counts"])
    sig_counts_lists = [counts[0, nv_ind].flatten() for nv_ind in range(num_nvs)]
    ref_counts_lists = [counts[1, nv_ind].flatten() for nv_ind in range(num_nvs)]

    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]
    num_shots = num_reps * num_runs

    threshold_list = []
    readout_fidelity_list = []
    prep_fidelity_list = []
    ion_prob_list = []
    red_chi_sq_list = []
    hist_figs = []

    # new: store extra multi-NV info
    n_nvs_est_list = []
    thresholds_list = []
    p_minus_list = []
    rate0_list = []
    delta_list = []
    fidelity_multiclass_list = []

    for ind in range(num_nvs):
        if ind in weak_esr:
            continue

        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        fit = analyze_charge_histogram_multinv_binomial(
            ref_counts_list,
            prob_dist=prob_dist,
            max_nvs=max_nvs_per_position,
            force_nvs=force_nvs,
            bic_extra_nv_penalty=2.0,  # make N-selection conservative
            seed=ind,
        )

        if not fit.get("ok", False):
            threshold = np.nan
            readout_fidelity = np.nan
            prep_fidelity = np.nan
            ion_prob = np.nan
            red_chi_sq = np.nan

            threshold_list.append(threshold)
            readout_fidelity_list.append(readout_fidelity)
            prep_fidelity_list.append(prep_fidelity)
            ion_prob_list.append(ion_prob)
            red_chi_sq_list.append(red_chi_sq)

            n_nvs_est_list.append(np.nan)
            thresholds_list.append(None)
            p_minus_list.append(np.nan)
            rate0_list.append(np.nan)
            delta_list.append(np.nan)
            fidelity_multiclass_list.append(np.nan)
            hist_figs.append(None)
            continue

        # ---- legacy outputs ----
        threshold = fit["threshold_any"]            # binary: k=0 vs >=1
        readout_fidelity = fit["fidelity_any"]      # binary fidelity
        red_chi_sq = fit.get("red_chi_sq", np.nan)

        weights = fit["weights"]                   # binomial weights over k=0..N
        prep_fidelity = 1.0 - float(weights[0])    # P(any NV-) generalizes your old (1 - popt[0]) :contentReference[oaicite:5]{index=5}

        threshold_list.append(threshold)
        readout_fidelity_list.append(readout_fidelity)
        prep_fidelity_list.append(prep_fidelity)
        red_chi_sq_list.append(red_chi_sq)

        # ion_prob is not defined in this ref-only model (set nan)
        ion_prob_list.append(np.nan)

        # ---- extra multi-NV outputs ----
        n_est = fit["n_nvs"]
        thresholds = fit["thresholds"]             # multi-class thresholds (len=N)
        p_minus, rate0, delta = [float(v) for v in fit["popt"]]
        fidelity_multiclass = fit["fidelity_multiclass"]

        n_nvs_est_list.append(n_est)
        thresholds_list.append(thresholds)
        p_minus_list.append(p_minus)
        rate0_list.append(rate0)
        delta_list.append(delta)
        fidelity_multiclass_list.append(fidelity_multiclass)

        # ---- plotting ----
        nv_num = widefield.get_nv_num(nv_list[ind])
        if do_plot_histograms:
            fig = plot_histograms(sig_counts_list, ref_counts_list, density=True)
            ax = fig.gca()

            # overlay components + combined (N+1 modes)
            x_vals = np.linspace(0, np.max(ref_counts_list), 800)
            single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)
            N = n_est
            w = fit["weights"]
            base = N * rate0
            combined = np.zeros_like(x_vals, float)

            for k in range(N + 1):
                lam_k = base + k * delta
                comp = float(w[k]) * single_pdf(x_vals, lam_k)
                combined += comp
                kpl.plot_line(ax, x_vals, comp, label=f"k={k}")

            kpl.plot_line(ax, x_vals, combined, color=kpl.KplColors.BLUE, label="Combined")
            ax.legend(loc=kpl.Loc.UPPER_RIGHT)

            # show thresholds (multi-class)
            for t in thresholds:
                ax.axvline(t, color=kpl.KplColors.GRAY, ls="dashed")
            # show legacy binary threshold (thicker)
            ax.axvline(threshold, color=kpl.KplColors.BLACK, ls="dashed", lw=2)

            txt = (
                f"NV{nv_num}\n"
                f"N_est={N}\n"
                f"fid_any={readout_fidelity:.3f}\n"
                f"fid_multi={fidelity_multiclass:.3f}\n"
                f"P(any NV-)={prep_fidelity:.3f}\n"
                f"p_minus={p_minus:.2f}"
            )
            kpl.anchored_text(ax, txt, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)

            hist_figs.append(fig)
        else:
            hist_figs.append(None)
        kpl.show(block=True)

    # Persist extra info if you want
    raw_data["charge_hist_multinv_binomial"] = {
        "prob_dist": prob_dist.name,
        "threshold_any": threshold_list,
        "readout_fidelity_any": readout_fidelity_list,
        "prep_fidelity_any": prep_fidelity_list,
        "red_chi_sq": red_chi_sq_list,
        "n_nvs_est": n_nvs_est_list,
        "thresholds_multiclass": thresholds_list,
        "p_minus": p_minus_list,
        "rate0": rate0_list,
        "delta": delta_list,
        "fidelity_multiclass": fidelity_multiclass_list,
    }

    return hist_figs
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Process and plot function and Set Seaborn theme globally for consistent styling
    data = dm.get_raw_data(
        file_stem="2026_03_02-17_30_11-qnami-nv0_2026_02_20", load_npz=True
    )
    process_and_plot(data, do_plot_histograms=True)
    kpl.show(block=True)
