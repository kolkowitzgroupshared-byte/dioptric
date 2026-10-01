# -*- coding: utf-8 -*-
"""
Widefield conditional charge initialization / ionization study.

Runs "charge_state_conditional_init.py" over many NVs with optional charge prep, thresholds
camera counts into NV⁻/NV⁰ states, and plots the fraction of NVs in NV⁻ vs number of attempts
(with an optional simple fit). Saves raw_data + figure; includes optional image-inset analysis.

Created: Fall 2023 (mccambria)
Updated: 2024-12-05 (Saroj Chand)
"""


import os
import sys
import time
import traceback

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.optimize import curve_fit
from scipy.special import factorial

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey

# from utils.tool_belt import determine_charge_state_threshold

# region Process and plotting functions


def process_and_plot(raw_data, mean_val=None):
    ### Setup

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    counts = np.array(raw_data["counts"])[0]
    states = widefield.threshold_counts(nv_list, counts, dynamic_thresh=False)
    # states = np.array(raw_data["states"])[0]
    num_reps = raw_data["num_reps"]
    num_runs = raw_data["num_runs"]

    # for nv_ind in range(num_nvs):
    #     fig, ax = plt.subplots()
    #     kpl.histogram(ax, counts[nv_ind].flatten(), density=True)
    #     ax.axvline(nv_list[nv_ind].threshold)
    #     kpl.show(block=True)

    num_nvn = np.sum(states, axis=0)
    # num_nvn = num_nvn[3:4, 0, :]  # Just one step
    num_nvn = num_nvn[:, 0, :]  # Just one step
    avg_num_nvn = np.mean(num_nvn, axis=0)  # Average over runs
    avg_num_nvn_ste = np.std(num_nvn, axis=0, ddof=1) / np.sqrt(num_runs)

    # print("Perfect 10s")
    # print(np.count_nonzero(num_nvn[:, 1] == 10) / num_runs)
    # print(np.count_nonzero(num_nvn[:, 5] == 10) / num_runs)
    # print(np.count_nonzero(num_nvn[:, 3:10] == 10) / (7 * num_runs))

    reps_vals = np.array(range(num_reps))

    # Only use the data we're going to show
    xlim = 11
    avg_num_nvn = avg_num_nvn[:xlim]
    avg_num_nvn_ste = avg_num_nvn_ste[:xlim]
    reps_vals = reps_vals[:xlim]

    figsize = kpl.figsize
    figsize[1] *= 1.0
    fig, ax = plt.subplots(figsize=figsize)
    kpl.plot_points(
        ax, reps_vals, avg_num_nvn / num_nvs, yerr=avg_num_nvn_ste / num_nvs
    )
    ax.set_xlabel("Number of attempts")
    ax.set_ylabel("Fraction in NV$^{-}$")
    ax.set_ylim(0, 1)
    # kpl.show(block=True)
    # sys.exit()

    def fit_fn(x, y0, c1, c2):
        term1 = (c1**x) * y0
        term2 = c2 * num_nvs * (1 - (c1**x)) / (1 - c1)
        return term1 + term2

    popt, pcov = curve_fit(
        fit_fn,
        reps_vals,
        avg_num_nvn,
        p0=(0.05, 0.7, 0.9),
        sigma=avg_num_nvn_ste,
        absolute_sigma=True,
    )
    print(popt)
    print(np.diag(np.sqrt(pcov)))
    print(fit_fn(1000, *popt))
    residuals = fit_fn(reps_vals, *popt) - avg_num_nvn
    chi_sq = np.sum((residuals / avg_num_nvn_ste) ** 2)
    red_chi_sq = chi_sq / (len(avg_num_nvn) - len(popt))
    print(f"Red chi sq: {round(red_chi_sq, 3)}")

    reps_vals_linspace = np.linspace(0, xlim, 1000)
    kpl.plot_line(ax, reps_vals_linspace, fit_fn(reps_vals_linspace, *popt) / num_nvs)

    if mean_val is not None:
        ax.axhline(mean_val, color=kpl.KplColors.GREEN, linestyle="dashed", linewidth=2)

    # ax.set_xlim((-0.5, 10.5))
    # ax.set_xticks(np.array(range(11)))
    # ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.yaxis.set_minor_locator(MaxNLocator(integer=True))

    ### Inset histograms

    if False:
        # fig, ax = plt.subplots()
        ax = fig.add_axes([0.38, 0.265, 0.6, 0.48])
        nv_ind = 3
        # nv_ind = 9
        # for rep_ind in reps_vals:
        hist_reps_vals = [0, 1, 3]
        # hist_reps_vals = [0, 1, 2, 3]
        num_hist_reps_vals = len(hist_reps_vals)
        colors = [
            kpl.KplColors.GRAY,
            mpl.colors.cnames["goldenrod"],
            kpl.KplColors.BROWN,
        ]
        # colors = [
        #     kpl.KplColors.GRAY,
        #     mpl.colors.cnames["steelblue"],
        #     mpl.colors.cnames["darkslateblue"],
        # ]
        colors = [mpl.colors.rgb2hex(color) for color in colors]
        for ind in range(num_hist_reps_vals):
            rep_ind = hist_reps_vals[ind]
            # color_rgb = mpl.colormaps["inferno"](ind / (num_hist_reps_vals - 1))
            color = colors[ind]
            # color = mpl.colors.rgb2hex(color_rgb)
            label = "1 attempt" if rep_ind == 1 else f"{rep_ind} attempts"
            kpl.histogram(
                ax,
                counts[nv_ind, :, 0, rep_ind],
                color=color,
                density=True,
                # facecolor="none",
                bin_size=4,
                lw=1.5,
                label=label,
            )
        ax.legend()
        ax.set_xlabel("Integrated counts")
        ax.set_ylabel("Probability")
        ax.set_yticks([0.0, 0.02, 0.04, 0.06])

    return fig


# endregion


def main(
    nv_list,
    num_reps,
    num_runs,
):
    ### Some initial setup
    seq_file = "charge_state_conditional_init.py"
    num_steps = 1

    charge_prep_fn = base_routine.charge_prep_no_verification_skip_first_rep
    # charge_prep_fn = None

    pulse_gen = tb.get_server_pulse_gen()

    ### Collect the data

    def run_fn(shuffled_step_inds):
        ion_coords_list = widefield.get_coords_list(nv_list, VirtualLaserKey.ION)
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(nv_list, VirtualLaserKey.CHARGE_POL)
        )
        seq_args = [ion_coords_list, pol_coords_list, pol_duration_list, pol_amp_list]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=True,
        save_images_avg_reps=False,
        charge_prep_fn=charge_prep_fn,
        num_exps=1,
        # uwave_ind_list=[0, 1],  # MCC
    )

    ### Processing

    try:
        fig = process_and_plot(raw_data)
    except Exception:
        print(traceback.format_exc())
        fig = None

    ### Save and clean up

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    raw_data |= {
        "timestamp": timestamp,
        "img_array-units": "photons",
    }
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()

    data = dm.get_raw_data(file_id=1740679854243)
    process_and_plot(data)
    kpl.show(block=True)
    sys.exit()

    ### Just get a mean val
    data = dm.get_raw_data(file_id=1573560918521)
    nv_list = data["nv_list"]
    states = np.array(data["states"])[0]
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]
    mean_val = np.sum(states) / (num_runs * num_reps)
    print(mean_val)

    ## Main plot
    # data = dm.get_raw_data(file_id=1567957794147)
    # data = dm.get_raw_data(file_id=1570547331729)
    data = dm.get_raw_data(file_id=1573541903486)  # initially ionized data set
    process_and_plot(data, mean_val=mean_val)
    kpl.show(block=True)

    ### Inset images

    base_pixel_drift = [4, 35]
    buffer = 42

    # Background image
    data = dm.get_raw_data(file_id=1567907418932, load_npz=True)
    pixel_drifts = np.array(data["pixel_drifts"])
    pixel_drift = np.mean(pixel_drifts, axis=0)
    offset = [
        pixel_drift[1] - base_pixel_drift[1] - 1,
        pixel_drift[0] - base_pixel_drift[0] + 1,
    ]
    ref_img_arrays = []
    for key in ["sig_img_array", "ref_img_array"]:
        img_array = np.array(data[key])
        widefield.replace_dead_pixel(img_array)
        img_array = widefield.crop_img_array(img_array, offset, buffer)
        ref_img_arrays.append(img_array)
    bg_img_array = ref_img_arrays[0]
    mask_img_array = ref_img_arrays[1] - ref_img_arrays[0]
    del data

    # Single shot image from experiment
    data = dm.get_raw_data(file_id=1573541903486, use_cache=False, load_npz=True)
    num_runs = data["num_runs"]
    num_reps = data["num_reps"]

    # Identify good candidate images
    # states = np.array(data["states"])[0]
    # num_in_nvm = np.sum(states, axis=0)
    # for run_ind in range(num_runs):
    #     for rep_ind in range(9):
    #         # if (
    #         #     num_in_nvm[run_ind, 0, 0] == 0
    #         #     and num_in_nvm[run_ind, 0, 1] == 7
    #         #     and num_in_nvm[run_ind, 0, rep_ind] == 9
    #         #     and num_in_nvm[run_ind, 0, rep_ind + 1] == 10
    #         #     and not (
    #         #         states[:, run_ind, 0, rep_ind] == states[:, run_ind, 0, rep_ind + 1]
    #         #     ).all()
    #         # ):
    #         for rep_ind2 in range(9):
    #             if (
    #                 num_in_nvm[run_ind, 0, 0] == 0
    #                 and num_in_nvm[run_ind, 0, 1] == 7
    #                 and num_in_nvm[run_ind, 0, rep_ind] == 9
    #                 and num_in_nvm[run_ind, 0, rep_ind2] == 10
    #             ):
    #                 # if num_in_nvm[run_ind, 0, rep_ind] == 10:
    #                 print(run_ind, rep_ind, rep_ind2)
    # sys.exit()

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)
    img_arrays = np.array(data["img_arrays"])
    # bg_img_array = np.sum(img_arrays[0, :, 0, 0], axis=0) / num_runs
    # run_ind = 24
    # rep_indd = 5
    # rep_indd = 7
    # run_ind = 42
    # rep_indd = 3
    # run_ind = 74
    # rep_indd = 8
    # for rep_ind in [0, 1, rep_indd, rep_indd + 1]:
    # run_ind = 73
    # for rep_ind in [3, 6, 8]:
    run_ind = 74
    for rep_ind in [6]:
        img_array = img_arrays[0, run_ind, 0, rep_ind]
        # img_array = np.mean(img_arrays, axis=(0, 1, 2, 3))
        widefield.replace_dead_pixel(img_array)
        states = np.array(data["states"])
        states = states[0, :, run_ind, 0, rep_ind]
        print(states)
        pixel_drift = np.array(data["pixel_drifts"])[run_ind]
        offset = [
            pixel_drift[1] - base_pixel_drift[1],
            pixel_drift[0] - base_pixel_drift[0],
        ]
        proc_img_array = widefield.crop_img_array(img_array, offset, buffer)
        proc_img_array = proc_img_array - bg_img_array
        drift = [
            pixel_drift[0] - buffer - offset[1],
            pixel_drift[1] - buffer - offset[0],
        ]

        # Downsampling / smoothing
        downsample_factor = 6
        proc_img_array = widefield.downsample_img_array(
            proc_img_array, downsample_factor
        )
        proc_img_array = np.repeat(proc_img_array, downsample_factor, axis=0)
        proc_img_array = np.repeat(proc_img_array, downsample_factor, axis=1)
        # proc_img_array = gaussian_filter(proc_img_array, 4)
        # proc_img_array = uniform_filter(proc_img_array, 5)

        fig, ax = plt.subplots()
        kpl.imshow(
            ax,
            proc_img_array,
            # clim=[0, 5],
            clim=[0, 15],
            # clim=[0.0, 0.2],
            no_cbar=True,
            # cmap=mpl.colormaps["gist_gray"],
        )
        ax.axis("off")
        nvm_inds = [ind for ind in range(num_nvs) if states[ind]]
        nv0_inds = [ind for ind in range(num_nvs) if not states[ind]]
        widefield.draw_circles_on_nvs(
            ax, nv_list, drift, linestyle="solid", no_legend=True, include_inds=nvm_inds
        )
        widefield.draw_circles_on_nvs(
            ax,
            nv_list,
            drift,
            linestyle="dashed",
            no_legend=True,
            include_inds=nv0_inds,
        )
    del img_arrays
    del data

    kpl.show(block=True)
