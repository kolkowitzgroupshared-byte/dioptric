# -*- coding: utf-8 -*-
"""
Widefield xy experiment
Created on November 29th, 2023
@author: Saroj Chand
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from utils import widefield

import time
import sys
import numpy as np
import math
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from scipy.optimize import curve_fit, least_squares
import utils.tool_belt as tb
from utils import data_manager as dm
from utils import kplotlib as kpl
import matplotlib.pyplot as plt

kpl.init_kplotlib()


SEQ_FILES = {
    "hahn": [
        "2026_01_24-02_18_52-johnson-nv0_2025_10_21",
        "2026_01_25-20_40_44-johnson-nv0_2025_10_21",
        "2026_01_27-03_14_34-johnson-nv0_2025_10_21",
    ],
    "xy2": [
        "2026_01_24-08_36_23-johnson-nv0_2025_10_21",
        "2026_01_26-02_59_06-johnson-nv0_2025_10_21",
        "2026_01_27-09_33_49-johnson-nv0_2025_10_21",
    ],
    "xy4": [
        "2026_01_24-15_00_32-johnson-nv0_2025_10_21",
        "2026_01_26-02_59_06-johnson-nv0_2025_10_21",
        "2026_01_27-15_51_39-johnson-nv0_2025_10_21",
    ],
    "xy8": [
        "2026_01_24-20_59_55-johnson-nv0_2025_10_21",
        "2026_01_26-15_11_04-johnson-nv0_2025_10_21",
        "2026_01_28-00_19_39-johnson-nv0_2025_10_21"
    ],
    "xy16": [
        "2026_01_25-02_20_55-johnson-nv0_2025_10_21",
        "2026_01_26-20_31_48-johnson-nv0_2025_10_21",
        "2026_01_28-05_34_06-johnson-nv0_2025_10_21",
    ],
}


def load_seq(file_stems):
    raw = widefield.process_multiple_files(file_stems, load_npz=True)
    nv_list = raw["nv_list"]

    tau_us = np.asarray(raw["taus"], float) / 1e3  # your code assumes taus in ns
    seq_xy = raw.get("xy_seq", "xy8").lower()
    _, N = widefield.parse_xy_sequence(seq_xy)

    # total evolution time (standard for CPMG/XY family): t = 2N * tau
    t_us = 2.0 * N * tau_us

    counts = np.asarray(raw["counts"], float)  # (2, num_nvs, num_steps)
    sig, ref = counts[0], counts[1]
    y, yerr = widefield.process_counts(nv_list, sig, ref, threshold=True)
    yerr = np.abs(np.asarray(yerr, float))
    yerr[yerr == 0] = np.nanmedian(yerr[yerr > 0])  # guard against zero σ

    # build an index map in case nv_list objects are hashable/unique
    # nv_to_i = {nv: i for i, nv in enumerate(nv_list)}
    nv_to_i = {i: i for i in range(len(nv_list))}

    return dict(
        seq=seq_xy,
        N=int(N),
        t_us=t_us,
        y=y,
        yerr=yerr,
        nv_list=nv_list,
        nv_to_i=nv_to_i,
    )


seq_data = {name: load_seq(stems) for name, stems in SEQ_FILES.items()}

# Choose a reference nv_list (assumes all runs used same NV ordering/objects)
ref_nv_list = next(iter(seq_data.values()))["nv_list"]


# def plot_all_sequences_for_nv(nv):
#     fig, ax = plt.subplots(figsize=(7, 5))
#     for name, d in seq_data.items():
#         i = nv
#         ax.errorbar(
#             d["t_us"],
#             d["y"][i],
#             yerr=d["yerr"][i],
#             fmt="o",
#             capsize=2,
#             label=f"{name} (N={d['N']})",
#         )

#     ax.set_xscale("log")
#     ax.set_xlabel(r"Total evolution time $t=2N\tau$ (µs)")
#     ax.set_ylabel("Norm. NV⁻ population")
#     ax.set_title(f"All sequences overlay — {nv}")
#     ax.grid(True, which="both", ls="--", alpha=0.5)
#     ax.legend(fontsize=9)
#     return fig, ax


# def plot_all_sequences_for_nv(nv, ncols=2):
#     names = list(seq_data.keys())
#     n = len(names)
#     nrows = math.ceil(n / ncols)

#     fig, axs = plt.subplots(
#         nrows, ncols,
#         figsize=(4.8 * ncols, 3.6 * nrows),
#         sharex=True, sharey=True
#     )
#     axs = np.atleast_1d(axs).ravel()

#     for k, name in enumerate(names):
#         ax = axs[k]
#         d = seq_data[name]
#         i = nv

#         ax.errorbar(
#             d["t_us"],
#             d["y"][i],
#             yerr=d["yerr"][i],
#             fmt="o",
#             capsize=2,
#         )

#         ax.set_xscale("log")
#         ax.grid(True, which="both", ls="--", alpha=0.5)
#         ax.set_title(f"{name} (N={d['N']})", fontsize=10)

#     # hide any unused subplot slots
#     for k in range(n, len(axs)):
#         axs[k].axis("off")

#     fig.supxlabel(r"Total evolution time $t=2N\tau$ (µs)")
#     fig.supylabel("Norm. NV⁻ population")
#     fig.suptitle(f"Sequences (separate subplots) — NV {nv}", y=1.02)

#     fig.tight_layout()
#     return fig, axs

def plot_all_sequences_for_nv(nv):
    names = list(seq_data.keys())
    n = len(names)

    fig, axs = plt.subplots(
        nrows=n, ncols=1,
        figsize=(7.5, 2.6 * n),
        sharex=True, sharey=True
    )
    axs = np.atleast_1d(axs).ravel()

    for ax, name in zip(axs, names):
        d = seq_data[name]
        i = nv

        ax.errorbar(
            d["t_us"],
            d["y"][i],
            yerr=d["yerr"][i],
            fmt="o",
            capsize=2,
        )
        ax.set_xscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.set_title(f"{name} (N={d['N']})", fontsize=10)

    # only label bottom x-axis (since sharex=True)
    axs[-1].set_xlabel(r"Total evolution time $t=2N\tau$ (µs)")
    fig.supylabel("Norm. NV⁻ population")
    fig.suptitle(f"Sequences (stacked, shared x) — NV {nv}", y=1.01)

    fig.tight_layout()
    return fig, axs

# Example: loop all NVs
# for nv in ref_nv_list:
# for nv in range(len(ref_nv_list)):
#     plot_all_sequences_for_nv(nv)
#     plt.show(block=True)
    
for nv in range(len(ref_nv_list)):
    plot_all_sequences_for_nv(nv)
    plt.show(block=True)
