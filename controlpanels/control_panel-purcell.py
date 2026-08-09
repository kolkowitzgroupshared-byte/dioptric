# -*- coding: utf-8 -*-
"""
Control panel for the PC Rabi

Created on June 16th, 2023

@author: mccambria
@author: Saroj B Chand
"""

### Imports
import os
import random
import sys
import time
from random import shuffle
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np

from majorroutines import targeting
from majorroutines.widefield import (
    ac_stark,
    bootstrapped_pulse_error_tomography,
    calibrate_iq_delay,
    charge_monitor,
    charge_correlation,
    charge_state_conditional_init,
    charge_state_histograms,
    charge_state_measurement_backaction,
    charge_state_particle_memory,
    charge_state_histograms_images,
    correlation_test,
    crosstalk_check,
    dmd_crosstalk_matrix,
    image_sample,
    optimize_amp_duration_charge_state_histograms,
    optimize_charge_state_histograms,
    optimize_scc,
    optimize_scc_readout,
    optimize_scc_amp_duration,
    optimize_spin_pol,
    optimize_aod_access_time,
    power_rabi,
    rabi,
    ramsey,
    relaxation_interleave,
    resonance,
    resonance_dualgen,
    deer_hahn,
    deer_hahn_rabi,
    scc_snr_check,
    simple_correlation_test,
    T2_correlation,
    two_block_hahn_spatial_correlation,
    spin_echo,
    two_block_hahn_correlation,
    dm_xy_iq_lockin_correlation,
    spin_pol_check,
    widefield_coherence,
    xy,
    dmd_turnoff_crosstalk_extinction_matrix,
    adaptive_charge_initialization,
)

# from slmsuite import optimize_slm_calibration
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey
from utils.positioning import get_scan_1d as calculate_freqs

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_aod = f"{green_laser}_aod"
red_laser_aod = f"{red_laser}_aod"

### Major Routines


def do_widefield_image_sample(nv_sig, num_reps=1):
    return image_sample.widefield_image(nv_sig, num_reps)


def do_scanning_image_sample(nv_sig):
    scan_range = 15
    num_steps =15
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_red_calibration_image(nv_sig, coords_list, force_laser_key=None, num_reps=1):
    arr = np.asarray(coords_list, dtype=float)
    x_freqs_MHz = arr[:, 0].tolist()
    y_freqs_MHz = arr[:, 1].tolist()
    # force_laser_key = VirtualLaserKey.IMAGING
    image_sample.red_widefield_calibration(
        nv_sig, x_freqs_MHz, y_freqs_MHz, force_laser_key, num_reps=1
    )


def do_scanning_image_full_roi(nv_sig):
    total_range = 60
    scan_range = 12
    num_steps = 15
    image_sample.scanning_full_roi(nv_sig, total_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.001
    num_steps = 4
    image_sample.scanning(nv_sig, scan_range, scan_range, num_steps)


def do_image_nv_list(nv_list):
    num_reps = 200
    # num_reps = 2
    return image_sample.nv_list(nv_list, num_reps)


def do_image_single_nv(nv_sig):
    num_reps = 100
    return image_sample.single_nv(nv_sig, num_reps)


def do_charge_state_histograms(nv_list):
    # 50 ms
    num_reps = 200
    num_runs = 10

    # Test
    # num_runs = 2

    return charge_state_histograms.main(
        nv_list, num_reps, num_runs, do_plot_histograms=False
    )

def do_optimize_pol_duration(nv_list):
    num_steps = 24
    min_duration = 100
    max_duration = 1940
    # num_steps = 25
    # min_duration = 200
    # max_duration = 9992
    num_reps = 10
    num_runs = 220
    # num_runs = 2
    return optimize_charge_state_histograms.optimize_pol_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_pol_amp(nv_list):
    num_steps = 24
    # num_reps = 150
    # num_runs = 5
    num_reps = 10
    num_runs = 200
    min_amp = 0.5
    max_amp = 1.5
    return optimize_charge_state_histograms.optimize_pol_amp(
        nv_list, num_steps, num_reps, num_runs, min_amp, max_amp
    )


def do_optimize_readout_duration(nv_list):
    num_steps = 16
    # num_reps = 150
    # num_runs = 5
    num_reps = 10
    num_runs = 225
    min_duration = 12e6
    max_duration = 108e6
    return optimize_charge_state_histograms.optimize_readout_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_readout_amp(nv_list):
    # num_steps = 24
    num_steps = 18
    # num_reps = 150
    # num_runs = 5
    num_reps = 10
    num_runs = 200
    min_amp = 0.7
    max_amp = 1.3
    return optimize_charge_state_histograms.optimize_readout_amp(
        nv_list, num_steps, num_reps, num_runs, min_amp, max_amp
    )
    
    
    
def do_optimize_readout_amp_repeated_readout(nv_list):
    # num_steps = 24
    num_steps = 18
    num_reps =5
    # num_runs = 5
    # num_reps = 10
    num_runs = 400
    min_amp = 0.7
    max_amp = 1.3
    return optimize_charge_state_histograms.optimize_readout_amp_repeated_readout(
        nv_list, num_steps, num_reps, num_runs, min_amp, max_amp
    )

def do_optimize_scc_readout_amp(nv_list):
    num_steps = 18
    num_reps = 16
    num_runs = 2
    min_amp = 0.8
    max_amp = 1.2
    return optimize_scc_readout.optimize_readout_amp(
        nv_list, num_steps, num_reps, num_runs, min_amp, max_amp
    )


def optimize_readout_amp_and_duration(nv_list):
    num_amp_steps = 16
    num_dur_steps = 5
    num_reps = 3
    num_runs = 1000
    min_amp = 0.9
    max_amp = 1.2
    min_duration = 12e6
    max_duration = 60e6
    return (
        optimize_amp_duration_charge_state_histograms.optimize_readout_amp_and_duration(
            nv_list,
            num_amp_steps,
            num_dur_steps,
            num_reps,
            num_runs,
            min_amp,
            max_amp,
            min_duration,
            max_duration,
        )
    )


def do_charge_state_histograms_images(nv_list, vary_pol_laser=False):
    aom_voltage_center = 1.0
    aom_voltage_range = 0.1
    num_steps = 1
    # num_reps = 15
    # num_reps = 100
    # num_runs = 50
    # num_runs = 100
    num_reps = 20
    num_runs = 60
    return charge_state_histograms_images.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        aom_voltage_center,
        aom_voltage_range,
        vary_pol_laser,
        aom_voltage_center,
        aom_voltage_range,
    )


def do_charge_state_conditional_init(nv_list):
    num_reps = 10
    num_runs = 4
    # num_runs = 400
    return charge_state_conditional_init.main(nv_list, num_reps, num_runs)



def do_adaptive_charge_initialization(nv_list):
    adaptive_charge_initialization.main(
        nv_list,
        num_reps=10,
        num_runs=4,
        mode="dmd_block_confirmed",
        # mode="dmd_all_on",
        dmd_indices=None,      # full nv_list, DMD index = nv_list index
        dmd_radius_px=11,
        dmd_plane=230,
        confirm_margin_counts=1.0,
        save_images=True,
        save_images_avg_reps=False,
        save_data=True,
        save_fig=True,
        save_image_frames=True,
        save_movie=True,
    )



def do_charge_state_particle_memory(nv_list):
    charge_state_particle_memory.main(
    nv_list,
    # Includes rep 0 ionization and subsequent adaptive attempts.
    num_init_reps=11,
    # Independent prepare-wait-read experiments.
    num_runs=100,

    # Begin with five minutes.
    dark_wait_s=300,

    mode="dmd_block_confirmed",

    dmd_indices=None,
    dmd_radius_px=8,
    dmd_plane=230,

    # Used only to confidently stop further initialization attempts.
    confirm_margin_counts=1.0,

    # Strongly recommended for particle detection.
    take_initial_check=True,

    # Black DMD during the exposure interval.
    block_all_during_wait=True,

    # Metadata describing this condition.
    exposure_label="source_off",

    # Require counts to be clearly away from the threshold.
    initial_event_margin_counts=1.0,
    final_event_margin_counts=1.0,

    # Set after checking camera-space NV spacing.
    cluster_radius_px=None,
    min_cluster_size=2,

    save_images=True,
    save_data=True,
    save_fig=True,
    verbose=True,
    )
    
def do_charge_state_particle_memory_wait_sweep(nv_list):
    wait_plan = [
        (0, 100),
        (10, 100),
        (30, 100),
        (60, 100),
        (180, 75),
        (300, 75),
        (600, 50),
        (1200, 25),
        (1800, 25),
        (3600, 25),
    ]

    results = {}

    for dark_wait_s, num_runs in wait_plan:
        label = f"source_off_wait_{dark_wait_s:g}s"

        print(
            f"\nRunning dark wait = {dark_wait_s:g} s, "
            f"num_runs = {num_runs}"
        )

        results[dark_wait_s] = charge_state_particle_memory.main(
            nv_list,
            num_init_reps=11,
            num_runs=num_runs,
            dark_wait_s=dark_wait_s,
            mode="dmd_block_confirmed",
            dmd_indices=None,
            dmd_radius_px=8,
            dmd_plane=230,
            confirm_margin_counts=1.0,
            take_initial_check=True,
            block_all_during_wait=True,
            exposure_label=label,
            initial_event_margin_counts=1.0,
            final_event_margin_counts=1.0,
            cluster_radius_px=None,
            min_cluster_size=2,
            save_images=True,
            save_data=True,
            save_fig=True,
            verbose=True,
        )

    return results
    
    
def do_charge_state_measurement_backaction(nv_list):
    charge_state_measurement_backaction.main(
    nv_list,
    num_init_reps=10,
    num_delayed_readouts=100,
    readout_interval_s=1,
    num_runs=1,
    confirm_margin_counts=1.0,
    initial_analysis_margin_counts=1.0,
    final_analysis_margin_counts=1.0,
    block_all_during_wait=True,
    save_images=True,
    save_data=True,
    save_fig=True,
    )

def unique_keep_order(vals):
    out = []
    for val in vals:
        val = int(val)
        if val not in out:
            out.append(val)
    return out


def do_dmd_crosstalk_matrix(
    nv_list_all,
    num_sources=200,
    dmd_radius_px=8,
    num_reps=200,
    num_runs=1,
    chain_path="dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd.npz",
):
    print("len(nv_list_all):", len(nv_list_all))

    source_inds = [
        int(i)
        for i in dmd_crosstalk_matrix.get_center_dmd_indices(
            n=1176,
            chain_path=chain_path,
        )
        if int(i) < len(nv_list_all)
    ][:num_sources]

    if len(source_inds) == 0:
        raise RuntimeError("No valid DMD source indices found.")

    measured_inds = dmd_crosstalk_matrix.unique_keep_order([0] + source_inds)
    measured_inds = [i for i in measured_inds if i < len(nv_list_all)]

    nv_sub = dmd_crosstalk_matrix.subset_nv_list(nv_list_all, measured_inds)

    for nv in nv_sub:
        nv.representative = False

    nv_sub[0].representative = True
    nv_sub[0].expected_counts = 1800.0

    print(f"num sources: {len(source_inds)}")
    print(f"num measured: {len(measured_inds)}")
    print("radius:", dmd_radius_px)
    print("representative:", nv_sub[0].name)
    print("first source inds:", source_inds[:20])

    return dmd_crosstalk_matrix.main(
        nv_sub,
        num_reps=num_reps,
        num_runs=num_runs,
        source_global_inds=source_inds,
        measured_global_inds=measured_inds,
        dmd_radius_px=dmd_radius_px,
        dmd_mode="pass_single",
        do_polarize=True,
        targeted_polarization=False,
        take_background=False,
        save_images=True,
        dmd_settle_s=0.10,
        dmd_plane=230,
    )
    

def do_dmd_turnoff_crosstalk_extinction_matrix(
nv_list_all
):
    inds, dmd_points = dmd_turnoff_crosstalk_extinction_matrix.choose_center_dmd_indices_from_chain(
        nv_list_all,
        num_sources=200,
        min_source_pitch_px=None,
    )

    dmd_turnoff_crosstalk_extinction_matrix.print_dmd_pitch_stats(inds, dmd_points)

    measured_inds = dmd_turnoff_crosstalk_extinction_matrix.unique_keep_order([0] + inds)
    nv_sub = dmd_turnoff_crosstalk_extinction_matrix.subset_nv_list(nv_list_all, measured_inds)

    dmd_turnoff_crosstalk_extinction_matrix.main(
        nv_sub,
        num_reps=200,
        num_runs=1,
        source_global_inds=inds,
        measured_global_inds=measured_inds,
        dmd_radius_px=6,
        do_polarize=True,
        targeted_polarization=False,
        take_background=True,
        save_images=True,
        save_raw_counts_by_source=True,
        dmd_settle_s=0.10,
        dmd_plane=230,
    )

def do_charge_correlation(nv_list):
    """
    Run charge-correlation measurement only on selected good NVs.
    """
    good_inds = [0, 190, 154, 173, 150, 102, 209, 175, 222, 254, 265, 178, 76, 288, 71, 62, 86, 226, 309, 46, 359, 345, 241, 81, 214, 292, 189, 402, 53, 350, 299, 387, 448, 191, 357, 436, 136, 174, 106, 49, 26, 370, 90, 47, 164, 403, 3, 276, 183, 568, 569, 543, 476, 107, 373, 409, 271, 554, 2, 505, 587, 5, 612, 48, 261, 9, 638, 654, 472, 176, 333, 65, 303, 19, 253, 573, 596, 124, 668]
    nv_sub = [nv_list[int(ind)] for ind in good_inds]

    # Make sure the first NV is representative for positioning.
    for nv in nv_sub:
        nv.representative = False

    nv_sub[0].representative = True
    nv_sub[0].expected_counts = 1500.0

    raw_data = charge_correlation.main(
        nv_list=nv_sub,
        num_reps=400,
        num_runs=200,
        do_drive=True,
        targeted_drive=False,
        dynamic_thresh=True,
        pixel_size_um=0.27,
        save_images=False,
    )

    return raw_data

def do_optimize_green(nv_sig):
    ret_vals = targeting.optimize(nv_sig, coords_key=green_laser_aod)
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_red(nv_sig, ref_nv_sig):
    opti_coords = []
    # axes_list = [Axes.X, Axes.Y]
    axes_list = [Axes.Y, Axes.X]
    # shuffle(axes_list)
    for ind in range(1):
        axes = axes_list[ind]
        ret_vals = targeting.optimize(nv_sig, coords_key=red_laser_aod, axes=axes)
        opti_coords.append(ret_vals[0])
        # Compensate for drift after first optimization along X axis
        if ind == 0:
            do_compensate_for_drift(ref_nv_sig)
    return opti_coords


def do_optimize_z(nv_sig):
    ret_vals = targeting.optimize(nv_sig, coords_key=CoordsKey.Z)
    opti_coords = ret_vals[0]
    return opti_coords


def do_compensate_for_drift(nv_sig):
    return targeting.compensate_for_drift(nv_sig)


def do_optimize_xyz(nv_sig, do_plot=True):
    targeting.optimize_xyz_using_piezo(
        nv_sig, do_plot=do_plot, axes_to_optimize=[0, 1, 2]
    )


def do_optimize_sample(nv_sig):
    opti_coords = targeting.optimize_sample(nv_sig)
    if not opti_coords:
        print("Optimization failed: No coordinates found.")
    return opti_coords


# def do_optimize_sample(nv_sig):
#     opti_coords = targeting.optimize_sample(nv_sig)
# return opti_coords


def do_optimize_pixel(nv_sig):
    ret_vals = targeting.optimize(nv_sig, coords_key=CoordsKey.PIXEL)
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_loop(nv_list, coords_key):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    opti_coords_list = []
    for nv in nv_list:
        if coords_key == green_laser:
            opti_coords = do_optimize_green(nv)
        elif coords_key == red_laser:
            opti_coords = do_optimize_red(nv, repr_nv_sig)
        opti_coords_list.append(opti_coords)
        
        # do_compensate_for_drift(repr_nv_sig)


    # Report back
    for opti_coords in opti_coords_list:
        r_opti_coords = [round(el, 3) for el in opti_coords[:2]]
        print(f"{r_opti_coords},")


def optimize_slm_Phase_calibration(repr_nv_sig, target_coords):
    widefield.get_repr_nv_sig(nv_list)
    np.array([[110.186, 129.281], [128.233, 88.007], [86.294, 103.0]])
    # optimize_slm_calibration.main(repr_nv_sig, target_coords)


def do_calibrate_green_red_delay():
    cxn = common.labrad_connect()
    pulse_gen = cxn.QM_opx

    seq_file = "calibrate_green_red_delay.py"

    seq_args = [2000]
    seq_args_string = tb.encode_seq_args(seq_args)
    num_reps = -1

    pulse_gen.stream_immediate(seq_file, seq_args_string, num_reps)

    input("Press enter to stop...")
    pulse_gen.halt()


def optimize_scc_amp_and_duration(nv_list):
    # # Single amp
    min_duration = 16
    max_duration = 288
    num_dur_steps = 18
    min_amp = 1.0
    max_amp = 1.0
    num_amp_steps = 1

    # # Single dur
    # min_amp = 0.6
    # max_amp = 1.4
    # num_amp_steps = 15
    # min_duration = 84
    # max_duration = 84
    # num_dur_steps = 1
    # reps and runs
    num_reps = 11
    num_runs = 200
    return optimize_scc_amp_duration.optimize_scc_amp_and_duration(
        nv_list,
        num_amp_steps,
        num_dur_steps,
        num_reps,
        num_runs,
        min_amp,
        max_amp,
        min_duration,
        max_duration,
    )


def do_optimize_scc_duration(nv_list):
    min_tau = 16
    max_tau = 220
    num_steps = 18
    num_reps = 15
    num_runs = 200
    # num_runs = 2
    optimize_scc.optimize_scc_duration(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_optimize_scc_amp(nv_list):
    min_tau = 0.8
    max_tau = 1.2
    num_steps = 16
    num_reps = 15
    num_runs = 200
    # num_runs = 2
    optimize_scc.optimize_scc_amp(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )


def do_optimize_spin_pol_amp(nv_list):
    min_tau = 0.8
    max_tau = 1.2
    num_steps = 16
    num_reps = 15
    num_runs = 200
    # num_runs = 2
    uwave_ind_list = [0, 1]
    optimize_spin_pol.optimize_spin_pol_amp(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        min_tau,
        max_tau,
        uwave_ind_list,
    )


def do_optimize_aod_access_time(nv_list):
    min_tau = 1e3
    max_tau = 5e3
    num_steps = 11
    num_reps = 15
    num_runs = 200
    # num_runs = 2
    optimize_aod_access_time.main(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    )

def do_scc_snr_check(nv_list):
    num_reps = 200
    num_runs = 40
    # num_runs = 20
    # num_runs = 160 * 4
    # num_runs = 3
    scc_snr_check.main(nv_list, num_reps, num_runs, uwave_ind_list=[0, 1])


def do_bootstrapped_pulse_error_tomography(nv_list):
    num_reps = 11
    num_runs = 200
    # num_runs = 10
    # num_runs = 1100
    # bootstrapped_pulse_error_tomography.main(
    #     nv_list, num_reps, num_runs, uwave_ind_list=[1]
    # )
    for _ in range(2):
        bootstrapped_pulse_error_tomography.main(
            nv_list, num_reps, num_runs, uwave_ind_list=[1]
        )


def do_power_rabi(nv_list):
    num_reps = 10
    num_runs = 200
    power_range = 1.5
    num_steps = 10
    uwave_ind_list = [1]
    powers = np.linspace(0, power_range, num_steps)
    # num_runs = 200
    # num_runs = 3
    power_rabi.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        powers,
        uwave_ind_list,
    )


def do_simple_correlation_test(nv_list):
    # Run this for a quick test experiment to debug.
    # num_reps = 200
    # num_runs = 5
    # simple_correlation_test.main(nv_list, num_reps, num_runs)

    # # Uncomment this to set up spin flips
    # # fmt: off    # snr_list = [0.208, 0.202, 0.186, 0.198, 0.246, 0.211, 0.062, 0.178, 0.161, 0.192, 0.246, 0.139, 0.084, 0.105, 0.089, 0.198, 0.242, 0.068, 0.134, 0.214, 0.185, 0.149, 0.172, 0.122, 0.128, 0.205, 0.202, 0.174, 0.192, 0.172, 0.145, 0.169, 0.135, 0.184, 0.204, 0.174, 0.13, 0.174, 0.06, 0.178, 0.237, 0.167, 0.198, 0.147, 0.176, 0.154, 0.118, 0.157, 0.113, 0.202, 0.084, 0.117, 0.117, 0.182, 0.157, 0.121, 0.181, 0.124, 0.135, 0.121, 0.15, 0.099, 0.107, 0.198, 0.09, 0.153, 0.159, 0.153, 0.177, 0.182, 0.139, 0.202, 0.141, 0.173, 0.114, 0.057, 0.193, 0.172, 0.191, 0.165, 0.076, 0.116, 0.072, 0.105, 0.152, 0.139, 0.186, 0.049, 0.197, 0.072, 0.072, 0.158, 0.175, 0.142, 0.132, 0.173, 0.063, 0.172, 0.141, 0.147, 0.138, 0.151, 0.169, 0.147, 0.148, 0.117, 0.149, 0.07, 0.135, 0.152, 0.163, 0.189, 0.116, 0.124, 0.129, 0.158, 0.079]
    # # fmt: on
    # snr_sorted_nv_inds = np.argsort(snr_list)[::-1]
    # parity = 1
    # for ind in snr_sorted_nv_inds:
    #     nv_list[ind].spin_flip = parity == -1
    #     parity *= -1

    selected_indices = widefield.select_half_left_side_nvs(nv_list)
    for index in selected_indices:
        nv = nv_list[index]
        nv.spin_flip = True
    print(f"Assigned spin_flip to {len(selected_indices)}")
    # print(f"Assigned spin_flip to {selected_indices}")

    # Run this for the main experiment, ~15 hours
    # num_steps = 200
    num_reps = 200
    num_runs = 400
    # num_runs = 2
    for _ in range(5):
        simple_correlation_test.main(nv_list, num_reps, num_runs)

def do_T2_correlation_test(nv_list):
    num_reps = 200
    num_runs = 1000
    # num_runs = 2
    # tau = 19.6e3 # gap
    tau = 228 # gap between pulses
    T2_correlation.main(nv_list, num_reps, num_runs, tau)
    # for _ in range(1):
    #     T2_correlation.main(nv_list, num_reps, num_runs, tau)

def do_two_block_hahn_spatial_correlation(nv_list):
    num_reps = 200
    num_runs = 1000
    # num_runs = 2
    tau = 228 # gap between pulses
    # T_lag = 364 # gap between two blocks for trough
    T_lag = 264 # gap between two blocks for zero crodding
    two_block_hahn_spatial_correlation.main(nv_list, num_reps, num_runs, tau, T_lag)
    # for _ in range(1):
    #     T2_correlation.main(nv_list, num_reps, num_runs, tau)

def do_dm_xy_iq_lockin(nv_list):
    # tau_ns = int(3.75e3 / 4) * 4
    tau_ns = int(15e3 / 4) * 4 # for single pi pulse/echo
    n_pi = 1
    num_reps = 75
    num_runs = 2000   # 200*90 = 18000 reps -> ~1 hour
    for _ in range(2):
        dm_xy_iq_lockin_correlation.main(
            nv_list=nv_list,
            num_reps=num_reps,
            num_runs=num_runs,
            tau_ns=tau_ns,
            n_pi=n_pi,
            uwave_ind_list=(0, 1),
        )

def do_calibrate_iq_delay(nv_list):
    min_tau = 20
    max_tau = 292
    num_steps = 18
    num_reps = 10
    num_runs = 100
    uwave_ind_list = [1]
    taus = np.linspace(min_tau, max_tau, num_steps)
    calibrate_iq_delay.main(
        nv_list, num_steps, num_reps, num_runs, taus, uwave_ind_list
    )


def do_resonance(nv_list):
    freq_center = 2.8785
    # freq_range = 0.36
    # num_steps = 65
    freq_range = 0.260
    num_steps = 45
    num_reps = 4
    num_runs = 300
    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    ##
    # Remove duplicates and sort
    freqs = sorted(set(freqs))
    num_steps = len(freqs)
    for _ in range(1):
        resonance.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            freqs=freqs,
            uwave_ind_list=[1],
        )
    # for _ in range(2):
    #     resonance.main(nv_list, num_steps, num_reps, num_runs, freqs=freqs)

# def do_deer_hahn(nv_list):
#     # freq_center = 0.174
#     # freq_range = 0.024
#     # num_steps =  48
#     # num_reps = 6
#     num_reps =2
#     num_runs =300
#     # num_runs = 2
#     # freqs = calculate_freqs(freq_center, freq_range, num_steps)
#     # freqs = np.arange(20, 330 + 2, 2)
#     freqs = np.arange(40, 300 + 1, 1)
#     freqs = freqs / 1000
#     # Remove duplicates and sort
#     freqs = sorted(set(freqs))
#     num_steps = len(freqs)
#     for _ in range(2):
#         do_widefield_image_sample(nv_sig, 50)
#         deer_hahn.main(
#             nv_list,
#             num_steps,
#             num_reps,
#             num_runs,
#             freqs=freqs,
#             uwave_ind_list=[0,1,2],
#         )

def do_deer_hahn(nv_list):
    num_reps = 1
    num_runs = 600
    bands_mhz = [
        (76, 94, 0.2),  
        (188, 210, 0.2),
        (248, 276, 0.2),
    ]
    for _ in range(2):
        do_widefield_image_sample(nv_sig, 50)

        for f0, f1, df in bands_mhz:
            freqs = np.arange(f0, f1 + df, df) / 1000  # MHz -> GHz
            freqs = sorted(set(freqs))
            num_steps = len(freqs)

            deer_hahn.main(
                nv_list,
                num_steps,
                num_reps,
                num_runs,
                freqs=freqs,
                uwave_ind_list=[0, 1, 2],
            )

def do_deer_hahn_rabi(nv_list):
    min_tau = 16
    max_tau = 996
    num_steps = 50
    num_reps = 5
    num_runs = 200
    uwave_ind_list = [0, 1, 2]
    deer_hahn_rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    for _ in range(3):
        rabi.main(
            nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list
        )
    # uwave_ind_list = [0]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # uwave_ind_list = [1]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)


def do_resonance_zoom(nv_list):
    # for freq_center in (2.85761751, 2.812251747511455):
    for freq_center in (2.87 + (2.87 - 2.85856), 2.87 + (2.87 - 2.81245)):
        freq_range = 0.030
        num_steps = 20
        num_reps = 15
        num_runs = 60
        resonance.main(nv_list, num_steps, num_reps, num_runs, freq_center, freq_range)

def do_resonance_dualgen(nv_list, uwave_ind_list=[0, 1]):
    freq_center = 2.87
    freq_range  = 0.36
    num_steps   = 60

    # outer reps = drift tracking cadence
    num_reps = 2
    num_runs = 400

    # inner reps for averaging
    avg_reps_sig = 8   # signal quarters
    avg_reps_ref = 2   # reference quarters

    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    freqs = sorted(set(freqs))
    num_steps = len(freqs)

    resonance_dualgen.main(
        nv_list,
        num_steps=num_steps,
        num_reps=num_reps,   # keep this for drift tracking
        num_runs=num_runs,
        freqs=freqs,
        uwave_ind_list=uwave_ind_list,
        num_reps_sig=avg_reps_sig,   # optional if you expose it in main()
        num_reps_ref=avg_reps_ref,   # optional if you expose it in main()
    )


def do_rabi(nv_list):
    min_tau = 16
    # max_tau = 240 + min_tau
    # max_tau = 360 + min_tau
    max_tau = 480 + min_tau
    num_steps = 31
    num_reps = 10
    num_runs = 400
    # num_runs = 5
    uwave_ind_list = [0, 1]
    # uwave_ind_list = [2]
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)


def do_widefield_coherence_test(nv_list, evol_time, seq_type):
    num_reps = 15
    num_runs = 150
    # num_runs = 2
    # fmt: off
    # phi_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    phi_list = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342, 360]
    # fmt: on
    num_steps = len(phi_list)
    uwave_ind_list = [0, 1]  # both are has iq modulation
    widefield_coherence.main(
        nv_list, num_steps, num_reps, num_runs, phi_list, evol_time, seq_type, uwave_ind_list
    )

def do_ac_stark(nv_list):
    min_tau = 0
    # max_tau = 240 + min_tau
    # max_tau = 360 + min_tau
    max_tau = 480 + min_tau
    num_steps = 31
    num_reps = 10
    # num_runs = 100
    num_runs = 50
    # num_runs = 2

    # uwave_ind_list = [1]
    uwave_ind_list = [0, 1]

    ac_stark.main(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list
    )


def do_spin_echo(nv_list):
    # revival_period = int(51.5e3 / 2) ### ~37.0 G
    # revival_period = int(38.5e3 / 2)### 49.68 G
    # revival_period = int(28.6e3 / 2) ### 65.14G
    revival_period = int(29.90e3 / 2) ### 62.14G
    # revival_period = int(31.2e3 / 2) ### 59.69G
    min_tau = 200
    taus = []
    revival_width = 6e3
    # revival_width = 4e3
    decay = np.linspace(min_tau, min_tau + revival_width, 6)
    taus.extend(decay.tolist())
    gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 8)
    taus.extend(gap[1:-1].tolist())
    first_revival = np.linspace(
        revival_period - revival_width, revival_period + revival_width, 65
    )
    taus.extend(first_revival.tolist())
    gap = np.linspace(
        revival_period + revival_width, 2 * revival_period - revival_width, 8
    )
    taus.extend(gap[1:-1].tolist())
    second_revival = np.linspace(
        2 * revival_period - revival_width, 2 * revival_period + revival_width, 11
    )
    taus.extend(second_revival.tolist())
    taus = [round(el / 4) * 4 for el in taus]

    # Remove duplicates and sort
    taus = sorted(set(taus))

    # Experiment settings
    num_steps = len(taus)

    # Automatic taus setup, linear spacing
    # min_tau = 200
    # max_tau = 84e3 + min_tau
    # num_steps = 29

    num_reps = 3
    num_runs = 600
    # num_runs = 2
    # spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)
    # spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)
    for ind in range(6):
        do_widefield_image_sample(nv_sig, 50)
        spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)


# def do_spin_echo(nv_list):
#     min_tau = 200  # ns
#     revival_period = int(20e3) ##20 gauss
#     taus = []
#     revival_width = 6e3
#     decay = np.linspace(min_tau, min_tau + revival_width, 6)
#     taus.extend(decay.tolist())
#     gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 6)
#     taus.extend(gap[1:-1].tolist())
#     first_revival = np.linspace(
#         revival_period - revival_width, revival_period + revival_width, 61
#     )
#     taus.extend(first_revival.tolist())
#     # Round to clock-cycle-compatible units
#     taus = [round(el / 4) * 4 for el in taus]
#     # Remove duplicates and sort
#     taus = sorted(set(taus))
#     num_steps = len(taus)
#     num_reps = 3
#     num_runs = 600

#     print(
#         f"[Spin Echo] Running with {num_steps} τ values, revival_period={revival_period}"
#     )

#     for _ in range(1):
#         spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)

def do_two_block_hahn_correlation(nv_list):
    tau = 44
    # lag_taus = [16, 24, 40, 64, 100, 160, 250, 400, 640, 1000, 1500, 2000]
    # lag_taus = [16, 40, 64, 88, 108, 132, 156, 180, 208, 236, 272, 316, 364, 424, 488, 568, 640, 740, 856, 988, 1144, 1292, 1496, 1728, 2000]
    lag_taus = widefield.generate_divisible_by_4(16, 2000, 45)
    # print(lag_taus)
    # sys.exit()
    num_steps = len(lag_taus)
    num_reps = 4
    num_runs = 600
    for _ in range(2):
        two_block_hahn_correlation.main(nv_list, num_steps, num_reps, num_runs, tau, lag_taus)

def do_two_block_hahn_correlation_dm(nv_list):
    tau = 15e3  # your revival tau (ns)
    # tau = 44  # your revival tau (ns)

    # def lags_log_div4_ns(tmin_ns, tmax_ns, n):
    #     # logspace, then round to nearest multiple of 4 ns
    #     l = np.logspace(np.log10(tmin_ns), np.log10(tmax_ns), n)
    #     l = np.unique((np.round(l / 4) * 4).astype(int))
    #     l = l[(l >= tmin_ns) & (l <= tmax_ns)]
    #     return l.tolist()

    lags_A = widefield.generate_divisible_by_4(int(0.2e3), int(20e3), 66)

    # Bands
    # lags_A = lags_log_div4_ns(16, int(50e3),  45)
    # lags_B = lags_log_div4_ns(int(50e3), int(50e6), 35)
    # lags_C = lags_log_div4_ns(int(50e6), int(2e9), 25)
    # lags_A = lags_log_div4_ns(int(0.2e3), int(200e3), 45)  # 0.25–200 us
    # lags_B = lags_log_div4_ns(int(200e3), int(20e6), 35)    # 0.2 ms–20 ms

    num_reps = 4

    # Fast band: cheap waits
    two_block_hahn_correlation.main(nv_list, len(lags_A), num_reps, num_runs=2000, tau=tau, lag_taus=lags_A)

    # Mid band
    # two_block_hahn_correlation.main(nv_list, len(lags_B), num_reps, num_runs=200, tau=tau, lag_taus=lags_B)

    # Slow band: waits dominate
    # two_block_hahn_correlation.main(nv_list, len(lags_C), num_reps, num_runs=30,  tau=tau, lag_taus=lags_C)

def do_spin_echo_1(nv_lis):
    min_tau = 200  # ns
    # max_tau = 20e3  # fallback
    revival_period = int(15e3)
    # revival_period = int(13e3)
    taus = []
    revival_width = 5e3
    decay = np.linspace(min_tau, min_tau + revival_width, 6)
    taus.extend(decay.tolist())
    gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 6)
    taus.extend(gap[1:-1].tolist())
    first_revival = np.linspace(
        revival_period - revival_width, revival_period + revival_width, 61
    )
    taus.extend(first_revival.tolist())
    # Round to clock-cycle-compatible units
    taus = [round(el / 4) * 4 for el in taus]
    # Remove duplicates and sort
    taus = sorted(set(taus))
    num_steps = len(taus)
    num_reps = 3
    num_runs = 600

    print(
        f"[Spin Echo] Running with {num_steps} τ values, revival_period={revival_period}"
    )

    for _ in range(1):
        spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)


def do_ramsey(nv_list):
    min_tau = 100
    max_tau = 3200 + min_tau
    detuning = 3
    num_steps = 101
    num_reps = 3
    num_runs = 1600
    # num_runs = 2
    ramsey.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, detuning)


def _quant4_ns(x):
    x = np.asarray(x, dtype=float)
    return (np.round(x / 4.0) * 4.0).astype(int)

def _arange_int(start_ns, stop_ns, step_ns):
    """Inclusive integer arange (like np.arange) but robust."""
    start_ns = int(start_ns)
    stop_ns  = int(stop_ns)
    step_ns  = int(step_ns)
    if step_ns <= 0:
        raise ValueError("step_ns must be > 0")
    if stop_ns < start_ns:
        return np.array([], dtype=int)
    # include endpoint
    n = (stop_ns - start_ns) // step_ns
    return start_ns + step_ns * np.arange(n + 1, dtype=int)

def build_xy8_dip_taus(
    revival_2tau_us=36.0,
    centers=(1, 3),
    include_global=True,
    global_step_ns=400,       # coarser than 200 to keep point count down
    coarse_step_ns=200,
    coarse_margin_us=3.0,     # 6.0 is usually overkill if you already know centers
    fine_window_us=1.0,
    fine_step_ns=20,          # denser -> reduce to 12/8/4 if you want
    ultra_window_us=0.25,     # optional extra density near the dip
    ultra_step_ns=8,          # must be multiple of 4
    use_ultra=True,
    min_tau_ns=200,
    max_tau_ns=40000
):
    # sanity: enforce multiples of 4 ns for steps
    for s in (global_step_ns, coarse_step_ns, fine_step_ns, ultra_step_ns):
        if s % 4 != 0:
            raise ValueError(f"step {s} ns must be multiple of 4 ns")

    TL_us = float(revival_2tau_us)      # echo revival in 2τ
    base_center_us = TL_us / 4.0        # XY8 dip fundamental near τ ~ TL/4

    taus_all = []

    # (A) optional global sweep
    if include_global:
        taus_all.append(_arange_int(min_tau_ns, max_tau_ns, global_step_ns))

    # (B) windows around each dip
    for k in centers:
        tau_c_us = k * base_center_us
        tau_c_ns = int(round(tau_c_us * 1e3))

        # coarse around center
        t0c = max(min_tau_ns, int(round(tau_c_ns - coarse_margin_us * 1e3)))
        t1c = min(max_tau_ns, int(round(tau_c_ns + coarse_margin_us * 1e3)))
        taus_all.append(_arange_int(t0c, t1c, coarse_step_ns))

        # fine around center
        t0f = max(min_tau_ns, int(round(tau_c_ns - fine_window_us * 1e3)))
        t1f = min(max_tau_ns, int(round(tau_c_ns + fine_window_us * 1e3)))
        taus_all.append(_arange_int(t0f, t1f, fine_step_ns))

        # ultra-fine micro-window (optional)
        if use_ultra and ultra_window_us is not None and ultra_window_us > 0:
            t0u = max(min_tau_ns, int(round(tau_c_ns - ultra_window_us * 1e3)))
            t1u = min(max_tau_ns, int(round(tau_c_ns + ultra_window_us * 1e3)))
            taus_all.append(_arange_int(t0u, t1u, ultra_step_ns))

    taus = np.unique(np.concatenate(taus_all)) if taus_all else np.array([], dtype=int)
    taus = taus[(taus >= min_tau_ns) & (taus <= max_tau_ns)]
    taus = _quant4_ns(taus)
    taus = sorted(set(int(t) for t in taus))
    return taus


def _quantize_ns(x_ns, q_ns=4):
    x = np.asarray(x_ns, dtype=float)
    return (np.round(x / q_ns) * q_ns).astype(int)

def build_log_taus(min_tau_ns=200, max_tau_ns=33000, n_points=60, q_ns=4):
    taus = np.logspace(np.log10(min_tau_ns), np.log10(max_tau_ns), n_points)
    taus = _quantize_ns(taus, q_ns=q_ns)
    taus = np.unique(taus)
    taus = taus[(taus >= min_tau_ns) & (taus <= max_tau_ns)]
    taus.sort()
    return taus.astype(int)

def _parse_xy_seq(xy_seq: str):
    m = re.match(r"([a-zA-Z]+\d*)(?:-(\d+))?$", xy_seq.strip().lower())
    if not m:
        raise ValueError(f"Bad xy_seq: {xy_seq}")
    base = m.group(1)
    blocks = int(m.group(2)) if m.group(2) else 1
    return base, blocks

def _tau_max_ns_for_seq(xy_seq, hahn_max_tau_ns=1_000_000):
    # keep same max TOTAL evolution time across sequences
    coeff = {"hahn": 2, "xy2": 4, "xy4": 8, "xy8": 16, "xy16": 32}
    base, blocks = _parse_xy_seq(xy_seq)
    if base not in coeff:
        raise ValueError(f"Unknown base seq: {base}")

    max_total_evol_ns = 2 * int(hahn_max_tau_ns)   # Hahn total evol ~ 2*tau
    tau_max_ns = max_total_evol_ns / (coeff[base] * blocks)
    return int(tau_max_ns)

def do_xy(nv_list, xy_seq="xy8-1", min_tau_ns=200, hahn_max_tau_ns=1_000_000, n_points=70, q_ns=4):
    num_reps = 4
    uwave_ind_list = [0, 1]
    num_runs = 600

    max_tau_ns = _tau_max_ns_for_seq(xy_seq, hahn_max_tau_ns=hahn_max_tau_ns)
    max_tau_ns = max(max_tau_ns, min_tau_ns)

    taus = build_log_taus(min_tau_ns=min_tau_ns, max_tau_ns=max_tau_ns, n_points=n_points, q_ns=q_ns)
    taus = [int(t) for t in taus]

    print("xy_seq:", xy_seq, "num_steps:", len(taus), "ns range:", taus[0], "to", taus[-1])
    # for _ in range(2):
    do_widefield_image_sample(nv_sig, 50)
    xy.main(nv_list, len(taus), num_reps, num_runs, taus, uwave_ind_list, xy_seq)


def do_xy_uniform_revival_scan(nv_list, xy_seq="xy8-1"):
    min_tau = 1e3
    dip = 19.6/2 # us
    dip_width = 2e3
    taus = []
    gap = np.linspace(min_tau, dip - dip_width, 11)
    taus.extend(gap.tolist())
    first_dip = np.linspace(dip - dip_width, dip + dip_width, 31)
    taus.extend(first_dip[1:-1].tolist())
    gap = np.linspace(
        dip + dip_width, 3*dip - dip_width, 11
    )
    taus.extend(gap[1:-1].tolist())
    second_dip = np.linspace(3*dip - dip_width, 3*dip + dip_width, 21)
    taus.extend(second_dip[1:-1].tolist())
    second_dip = np.linspace(3*dip + dip_width, 5*dip + dip_width, 21)
    # Round τ to 4 ns resolution
    # taus = [round(tau / 4) * 4 for tau in taus]
    # taus = sorted(set(taus))  # remove duplicates
    num_reps = 2
    num_runs = 600
    num_steps = len(taus)
    uwave_ind_list = [0, 1]

    print(
        f"[XY8 Uniform] Scanning {num_steps} τ values from {taus[0]} to {taus[-1]} ns"
    )
    for _ in range(4):
        xy.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            uwave_ind_list=uwave_ind_list,
            taus=taus,
            xy_seq=xy_seq,
        )


def do_xy_revival_scan(nv_list, xy_seq="xy8-1"):
    min_total_time = 100  # ns
    revival_time = 14.1e3  # ns
    revival_width = 4e3  # ns
    high_res_points = 24
    gap_points = 6
    decay_points = 6
    num_revivals = 4
    taus = []
    # Initial coherence decay region
    decay = np.linspace(min_total_time, min_total_time + revival_width, decay_points)
    taus.extend(decay.tolist())

    for i in range(1, num_revivals + 1):
        center = i * revival_time

        # Gap before revival
        if i == 1:
            gap_start = min_total_time + revival_width
        else:
            gap_start = (i - 1) * revival_time + revival_width
        gap_end = center - revival_width

        if gap_end > gap_start:
            gap = np.linspace(gap_start, gap_end, gap_points)
            taus.extend(gap[1:-1].tolist())  # exclude endpoints to avoid duplication

        # High-resolution scan across revival
        revival_scan = np.linspace(
            center - revival_width, center + revival_width, high_res_points
        )
        taus.extend(revival_scan.tolist())

    # Round to 4 ns granularity
    taus = sorted(set(round(tau / 4) * 4 for tau in taus))

    num_steps = len(taus)
    num_reps = 1
    num_runs = 2000
    uwave_ind_list = [1]

    print(
        f"[{xy_seq}] Running with {num_steps} τ values, targeting {num_revivals} revivals starting at {revival_time} ns"
    )

    for _ in range(4):
        xy.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            uwave_ind_list=uwave_ind_list,
            taus=taus,
            xy_seq=xy_seq,
        )


def do_correlation_test(nv_list):
    min_tau = 16
    max_tau = 72
    num_steps = 15

    num_reps = 10
    num_runs = 400

    # MCC
    # min_tau = 16
    # max_tau = 240 + min_tau
    # num_steps = 31
    # num_reps = 20
    # num_runs = 30

    # anticorrelation_inds = None
    anticorrelation_inds = [2, 3]

    correlation_test.main(
        nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, anticorrelation_inds
    )


def do_sq_relaxation(nv_list):
    # min_tau = 1e3
    min_tau = 5e2
    max_tau = 10e6 + min_tau
    num_steps = 21
    num_reps = 10
    num_runs = 800
    # num_runs = 2
    # relaxation_interleave.sq_relaxation(
    #     nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    # )
    for _ in range(1):
        relaxation_interleave.sq_relaxation(
            nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
        )


def do_dq_relaxation(nv_list):
    min_tau = 5e2
    max_tau = 10e6 + min_tau
    num_steps = 21
    num_reps = 10
    num_runs = 800

    # relaxation_interleave.dq_relaxation(
    #     nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
    # )
    for _ in range(1):
        relaxation_interleave.dq_relaxation(
            nv_list, num_steps, num_reps, num_runs, min_tau, max_tau
        )


def do_opx_square_wave():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    # Yellow
    opx.square_wave(
        [],  # Digital channels
        [7],  # Analog channels
        [0.5],  # Analog voltages
        10000,  # Period (ns)
        # 1e9,  # Period (ns)
    )
    # Camera trigger
    # opx.square_wave(
    #     [4],  # Digital channels
    #     [],  # Analog channels
    #     [],  # Analog voltages
    #     100000,  # Period (ns)
    # )
    input("Press enter to stop...")
    # sig_gen.uwave_off()


def do_crosstalk_check(nv_sig):
    num_steps = 21
    num_reps = 10
    num_runs = 150
    # aod_freq_range = 3.0
    laser_name = red_laser
    # laser_name = green_laser
    # axis_ind = 0  # 0: x, 1: y, 2: z
    uwave_ind = [0, 1]

    if laser_name is red_laser:
        aod_freq_range = 2.0
    elif laser_name is green_laser:
        aod_freq_range = 3.0
    for axis_ind in [0, 1]:
        crosstalk_check.main(
            nv_sig,
            num_steps,
            num_reps,
            num_runs,
            aod_freq_range,
            laser_name,
            axis_ind,  # 0: x, 1: y, 2: z
            uwave_ind,
        )


def do_spin_pol_check(nv_sig):
    num_steps = 16
    num_reps = 10
    num_runs = 40
    aod_min_voltage = 0.01
    aod_max_voltage = 0.05
    uwave_ind = 0

    spin_pol_check.main(
        nv_sig,
        num_steps,
        num_reps,
        num_runs,
        aod_min_voltage,
        aod_max_voltage,
        uwave_ind,
    )


def do_detect_cosmic_rays(nv_list):
    num_reps = 4
    num_runs = 600
    num_runs = 2
    # dark_time = 1e9 # 1s
    # dark_time = 10e6  # 10ms
    dark_time_1 = 8e6  # 1 ms in nanoseconds
    dark_time_2 = 8e9  # 8 s in nanoseconds
    # charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)
    for _ in range(6):
        charge_monitor.detect_cosmic_rays(
            nv_list, num_reps, num_runs, dark_time_1, dark_time_2
        )
    # dark_times = [100e6, 500e6, 5e6, 506, 250e6]
    # for dark_time in dark_times:
    #     charge_monitor.detect_cosmic_rays(nv_list, num_reps, num_runs, dark_time)


def do_check_readout_fidelity(nv_list):
    num_reps = 200
    num_runs = 20

    charge_monitor.check_readout_fidelity(nv_list, num_reps, num_runs)


def do_charge_quantum_jump(nv_list):
    num_reps = 3000
    charge_monitor.charge_quantum_jump(nv_list, num_reps)

def do_opx_constant_ac():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    # num_reps = 1000
    # start = time.time()
    # for ind in range(num_reps):
    #     opx.test("_cache_charge_pol_incomplete", False)
    # stop = time.time()
    # print((stop - start) / num_reps)

    # Microwave test
    # if True:
    #     sig_gen = cxn.sig_gen_STAN_sg394_3
    #     amp = 2
    #     chan = 3
    # else:
    #     sig_gen = cxn.sig_gen_STAN_sg394_2
    #     amp = 10
    #     chan = 10
    # sig_gen.set_amp(amp)  # 12
    # sig_gen.set_freq(1.0)
    # sig_gen.uwave_on()
    # opx.constant_ac([chan])

    # Camera frame rate test
    # seq_args = [500]
    # seq_args_string = tb.encode_seq_args(seq_args)
    # opx.stream_load("camera_test.py", seq_args_string)
    # opx.stream_start()

    # Yellow
    opx.constant_ac(
        [],  # Digital channels
        [7],  # Analog channels
        [0.15],  # Analog voltages
        [0],  # Analog frequencies
    )
    # opx.constant_ac([4])  # Just laser
    # Red
    # freqs = [65, 75, 85]
    # # freqs = [73, 75, 77]
    # while not keyboard.is_pressed("q"):
    #     for freq in freqs:
    #         opx.constant_ac(
    #             [1],  # Digital channels
    #             [2, 6],  # Analog channels
    #             [0.17, 0.17],  # Analog voltages
    #             [
    #                 75,
    #                 freq,
    #             ],  # Analog frequencies                                                                                                                                                                              uencies
    #         )
    #         time.sleep(0.5)
    #     opx.halt()
    # opx.constant_ac(
    #     [1],  # Digital channels
    #     0# [2, 6],  # Analog channels
    #     # [0.19, 0.19],  # Analog voltages
    #     # [
    #     #     75,
    #     #     75,
    #     # ],  # Analog frequencies                                                                                                                                                                       uencies
    # )
    # opx.constant_ac([1])  # Just laser
    # Green
    opx.constant_ac(
        [4],  # Digital channels
        [3, 4],  # Analog channels
        [0.02, 0.02],  # Analog voltages
        [101.0, 101.0],  # Analog frequencies
    )
    # Green + red
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6],  # Analog channels
    #     [0.08, 0.08, 0.08, 0.08],  # Analog voltages;
    #     [126.958, 127.769, 90.817, 92.654],
    # )
    # green_coords_list = [
    #     [97.763, 98.184],
    #     [73.727, 115.462],
    #     [100.738, 68.901],
    #     [126.958, 127.769],
    # ]
    # red_coords_list = [
    #     [66.845,  67.629],
    #     [47.253, 81.361], 
    #     [69.57, 44.316], 
    #     [90.817, 92.654]
    # ]
    # red
    # opx.constant_ac(
    #     [1],  # Digital channels
    #     [2, 6],  # Analog channels
    #     [0.16, 0.16],  # Analog voltages
    #     [72.0, 72.0],  # Analog frequencies
    # )

    # # Green + yellow
    # opx.constant_ac(
    #     [4],  # Digital channels11
    #     [3, 4, 7],  # Analog channels
    #     [0.08, 0.08, 0.35],  # Analog voltages
    #     [99.0, 99.0, 0],  # Analog frequencies
    # )
    # # Red + green + Yellow
    # opx.constant_ac(
    #     [4, 1],  # Digital channels1
    #     [3, 4, 2, 6, 7],  # Analog channels
    #     [0.10, 0.10, 0.10, 0.10, 0.35],  # Analog voltages
    #     [99, 99, 67, 67, 0],  # Analog frequencies
    # )
    input("Press enter to stop...")
    # sig_gen.uwave_off()

def do_green_red_triplet_time_mux():
    cxn = common.labrad_connect()
    opx = cxn.QM_opx

    green_center = np.array([102.0, 102.0], dtype=float)
    red_center   = np.array([67.0,  67.0], dtype=float)

    green_d = 25.0   # MHz
    red_d   = 20.0   # MHz

    green_square = [
        [green_center[0] - green_d, green_center[1] - green_d],  # bottom-left
        [green_center[0] - green_d, green_center[1] + green_d],  # top-left
        [green_center[0] + green_d, green_center[1] - green_d],  # bottom-right
        [green_center[0] + green_d, green_center[1] + green_d],  # top-right
    ]

    red_square = [
        [red_center[0] - red_d, red_center[1] - red_d],  # bottom-left
        [red_center[0] - red_d, red_center[1] + red_d],  # top-left
        [red_center[0] + red_d, red_center[1] - red_d],  # bottom-right
        [red_center[0] + red_d, red_center[1] + red_d],  # top-right
    ]
    # Start low; raise carefully if needed
    green_amp = 0.06
    red_amp = 0.06
    yellow_amp = None
    dwell_us = 200

    seq_args = [
        green_coords_list,
        red_coords_list,
        green_amp,
        red_amp,
        yellow_amp,
        dwell_us,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)

    opx.stream_load("constant_aod_time_mux.py", seq_args_string)
    opx.stream_start()

    input("Press enter to stop...")
#     opx.halt()

def compile_speed_test(nv_list):
    cxn = common.labrad_connect()
    pulse_gen = cxn.QM_opx

    seq_file = "resonance_ref.py"
    num_reps = 20
    uwave_index = 0

    seq_args = widefield.get_base_scc_seq_args(nv_list)
    seq_args.append(uwave_index)
    seq_args.append([2.1, 2.3, 2.5, 2.7, 2.9])
    seq_args_string = tb.encode_seq_args(seq_args)

    start = time.time()
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    stop = time.time()
    print(stop - start)

    seq_args[-2] = 1
    seq_args_string = tb.encode_seq_args(seq_args)

    start = time.time()
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    stop = time.time()
    print(stop - start)


def piezo_voltage_to_pixel_calibration():
    cal_voltage_coords = np.array(
        [(1.1, 0.2), (0.20000000000000012, 0.7196152422706632), (0.19999999999999973, -0.319615242270663)], dtype="float32"
    )
    cal_pixel_coords = np.array(
        [[247.886, 242.951], [234.037, 253.065], [232.579, 236.378]], dtype="float32"
    )
    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(cal_voltage_coords, cal_pixel_coords)
    # Convert the 2x3 matrix to a 3x3 matrix
    M = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M)

    # Format and print the affine matrix as a list of lists
    affine_voltage2pixel = M.tolist()
    inverse_affine_voltage2pixel = M_inv.tolist()
    print("affine_voltage2pixel = [")
    for row in affine_voltage2pixel:
        print("    [{:.8f}, {:.8f}, {:.8f}],".format(row[0], row[1], row[2]))
    print("]")

    print("\nInverse affine matrix (M_inv) as a list of lists:")
    print("[")
    for row in inverse_affine_voltage2pixel:
        print(f"    [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}],")
    print("]")
    return M_inv


# Load the saved NV coordinates and radii from the .npz file
def load_nv_coords(
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_multiple_nv302.npz",
    x_min=0,
    x_max=450,
    y_min=0,
    y_max=450,
    edge_tol=1.0,
):
    with np.load(file_path, allow_pickle=True) as data:
        nv_coordinates = np.asarray(data["nv_coordinates"], dtype=np.float32)

    mask = (
        (nv_coordinates[:, 0] >= x_min - edge_tol)
        & (nv_coordinates[:, 0] <= x_max + edge_tol)
        & (nv_coordinates[:, 1] >= y_min - edge_tol)
        & (nv_coordinates[:, 1] <= y_max + edge_tol)
    )

    nv_coordinates_clean = nv_coordinates[mask]

    print("Original NVs:", len(nv_coordinates))
    print("After x/y filter:", len(nv_coordinates_clean))

    dropped_inds = np.where(~mask)[0]
    print("Dropped NVs:", len(dropped_inds))

    if len(dropped_inds) > 0:
        print("Dropped indices:", dropped_inds.tolist())
        print("Dropped coordinates:")
        print(nv_coordinates[dropped_inds])

    return nv_coordinates_clean


def load_thresholds(file_path="slmsuite/nv_blob_detection/threshold_list_nvs_162.npz"):
    with np.load(file_path) as data_file:
        thresholds = data_file["arr_0"]
    return thresholds


def estimate_z(x, y, z0=0.15, slope=-0.0265):
    """Estimate Z from (x, y) using diagonal slope."""
    return z0 + slope * (x + y) / np.sqrt(2)


def generate_equilateral_triangle_around_center(center=(0, 0), r=2.0):
    angles = [0, 120, 240]  # degrees
    points = []
    for angle_deg in angles:
        theta = np.radians(angle_deg)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points


def scan_equilateral_triangle(nv_sig, center_coord=(0, 0), radius=0.2):
    triangle_coords = generate_equilateral_triangle_around_center(
        center_coord, r=radius
    )
    triangle_coords.append(center_coord)  # Return to center
    print(triangle_coords)
    for sample_coord in triangle_coords:
        # z = estimate_z(*sample_coord)
        nv_sig.coords[CoordsKey.SAMPLE] = sample_coord
        # nv_sig.coords[CoordsKey.Z] = z
        # print(f"Scanning SAMPLE: {sample_coord}, estimated Z: {z:.3f}")
        do_scanning_image_sample(nv_sig)


### Run the file
if __name__ == "__main__":
    # region Shared parameters
    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"
    sample_name = "qnami"
    magnet_angle = 90
    date_str = "2026_02_20"
    sample_coords = [-1.20, -0.75]
    z_coord = -1.7
    # z_coord = -3.9
    
    config = common.get_config_dict()
    file_path = config["SpatialCalibrations"]["active_nv_coords_path"]
    pixel_coords_list = load_nv_coords(file_path=file_path).tolist()

    green_coords_list = [
        [
            round(coord, 3)
            for coord in pos.transform_coords(
                nv_pixel_coords, CoordsKey.PIXEL, green_laser_aod
            )
        ]
        for nv_pixel_coords in pixel_coords_list
    ]

    red_coords_list = [
        [
            round(coord, 3)
            for coord in pos.transform_coords(
                nv_pixel_coords, CoordsKey.PIXEL, red_laser_aod
            )
        ]
        for nv_pixel_coords in pixel_coords_list
    ]

    # print(red_coords_list)
    # print(green_coords_list)
    # sys.exit()
    print(f"Number of NVs: {len(pixel_coords_list)}")
    print(f"Reference NV:{pixel_coords_list[0]}")
    print(f"Green Laser Coordinates: {green_coords_list[0]}")
    print(f"Red Laser Coordinates: {red_coords_list[0]}")

    # --------------------------------------------------
    # Print min/max of pixel, green AOD, and red AOD coords
    # --------------------------------------------------
    # green_arr = np.asarray(green_coords_list, dtype=float)
    # red_arr = np.asarray(red_coords_list, dtype=float)

    # print("green fx range:", green_arr[:, 0].min(), green_arr[:, 0].max())
    # print("green fy range:", green_arr[:, 1].min(), green_arr[:, 1].max())
    # print("red fx range:", red_arr[:, 0].min(), red_arr[:, 0].max())
    # print("red fy range:", red_arr[:, 1].min(), red_arr[:, 1].max())

    # sys.exit()

    # pixel_coords_list =[
    #     [194.039, 189.963], 
    #     [319.015, 83.106], 
    #     [192.998, 353.981], 
    #     [17.982, 41.943],
    # ]
    # green_coords_list = [
    #     [97.74, 98.166],
    #     [73.715, 115.446],
    #     [100.72, 68.894],
    #     [126.953, 127.76],
    # ]
    # red_coords_list = [
    #     [66.826, 67.772],
    #     [47.273, 81.38],
    #     [69.645, 44.033],
    #     [90.325, 92.229],
    # ]
    
    analysis_data = dm.get_raw_data(
        # file_stem="2026_06_23-16_12_54-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", ##1176NVs
        # file_stem="2026_07_09-21_39_43-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", ##814NVs
        # file_stem="2026_07_10-12_52_52-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", ##814NVs
        # file_stem="2026_07_10-17_01_26-single_step_charge_hist_single_cpu_2026_07_10-16_57_47-qnami-nv0_2026_02_20", ##814NVs
        # file_stem="2026_07_10-17_25_18-single_step_charge_hist_single_cpu_2026_07_10-16_57_47-qnami-nv0_2026_02_20", ##814NVs
        # file_stem="2026_07_13-15_34_49-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", ##814NVs
        # file_stem="2026_07_13-16_02_12-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", ##814NVs
        # file_stem="2026_07_13-17_11_02-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", ##814NVs
        # file_stem= "2026_07_13-22_24_18-single_step_charge_hist_single_cpu_2026_07_13-17_00_15-qnami-nv0_2026_02_20",
        # file_stem = "2026_07_14-10_39_06-single_step_charge_hist_single_cpu_2026_07_13-17_00_15-qnami-nv0_2026_02_20",
        # file_stem = "2026_07_15-19_48_48-single_step_charge_hist_single_cpu_2026_07_15-19_42_19-qnami-nv0_2026_02_20", 
        # file_stem = "2026_07_16-22_53_11-single_step_charge_hist_single_cpu_2026_07_16-22_48_08-qnami-nv0_2026_02_20", 
        # file_stem = "2026_07_19-00_47_30-single_step_charge_hist_single_cpu_2026_07_19-00_17_00-qnami-nv0_2026_02_20",
        # file_stem = "2026_07_20-17_07_53-single_step_charge_hist_single_cpu_2026_07_20-17_04_32-qnami-nv0_2026_02_20",
        # file_stem = "2026_07_21-16_11_27-single_step_charge_hist_single_cpu_2026_07_21-16_08_28-qnami-nv0_2026_02_20",
        # file_stem = "2026_07_23-00_37_58-single_step_charge_hist_single_cpu_2026_07_22-22_20_35-qnami-nv0_2026_02_20",
        # file_stem = "2026_08_04-13_25_01-single_step_charge_hist_single_cpu_2026_08_04-13_21_07-qnami-nv0_2026_02_20",
        # file_stem = "2026_08_04-16_28_51-single_step_charge_hist_single_cpu_2026_08_04-16_25_34-qnami-nv0_2026_02_20",
        # file_stem = "2026_08_08-16_47_43-single_step_charge_hist_single_cpu_2026_08_08-16_44_44-qnami-nv0_2026_02_20",
        file_stem = "2026_08_08-17_24_56-single_step_charge_hist_single_cpu_2026_08_08-17_21_48-qnami-nv0_2026_02_20",
        load_npz=True,
    )
    # print (analysis_data.keys())
    # sys.exit()
    # analysis = analysis_data["charge_hist_multinv_binomial"]
    # threshold_list = analysis["threshold_any"]
    
    analysis = analysis_data["single_step_charge_histogram"]
    threshold_list = analysis["threshold"]
    # Convert None/bad values to np.nan
    threshold_arr = np.asarray(
        [
            np.nan if val is None else float(val)
            for val in threshold_list
        ],
        dtype=float,
    )

    # Compute median from valid values only
    median_threshold = float(np.nanmedian(threshold_arr))

    # Replace None/nan values with median
    threshold_arr_filled = threshold_arr.copy()
    bad = ~np.isfinite(threshold_arr_filled)
    threshold_arr_filled[bad] = median_threshold
    threshold_list = threshold_arr_filled
    # Then apply your MAD condition
    # selected_inds = inds_good[mad_thr]

    # threshold = np.asarray(analysis["threshold_any"], dtype=float)
    # ok = np.asarray(analysis["ok"], dtype=bool)
    # n_est = np.rint(np.asarray(analysis["n_nvs_est"], dtype=float)).astype(int)
    # fidelity = np.asarray(analysis["readout_fidelity_any"], dtype=float)


    # selected_inds = inds_good[mad_thr]

    # Sort by fidelity, best first
    # selected_inds = selected_inds[np.argsort(fidelity[selected_inds])[::-1]]

    # Start small
    # selected_inds = selected_inds[:50]

    # print("median threshold:", median_thr)
    # print("MAD threshold:", mad_thr)
    # print("num good 1-NV:", len(inds_good))
    # print("num near median:", np.sum(near_median_mask_local))
    # print("num selected:", len(selected_inds))
    # print("selected inds:", selected_inds)
    # print("selected threshold range:", np.nanmin(threshold[selected_inds]), np.nanmax(threshold[selected_inds]))
    # print("selected fidelity range:", np.nanmin(fidelity[selected_inds]), np.nanmax(fidelity[selected_inds]))

    # plt.figure()
    # plt.hist(thr_good, bins=60, alpha=0.5, label="good 1-NV")
    # plt.axvline(median_thr, color="k", linestyle="--", label="median")
    # plt.hist(threshold[selected_inds], bins=20, alpha=0.8, label="selected")
    # plt.xlabel("threshold_any")
    # plt.ylabel("count")
    # plt.legend()
    # plt.show(block=True)
    # sys.exit()
    num_nvs = len(pixel_coords_list)
    # threshold_list = [None] * num_nvs
    scc_duration_list = [88] * num_nvs
    pol_duration_list = [10000] * num_nvs

    # -------------------------------------------
    # amplitudes
    # -------------------------------------------
    charge_pol_amps = [
        round(
            widefield.green_qua_amp_fn_2d(green_coords_list[i]),
            4,
        )
        for i in range(num_nvs)
    ]

    scc_amp_list = [
        round(
            widefield.red_qua_amp_fn_2d(red_coords_list[i]),
            4,
        )
        for i in range(num_nvs)
    ]

    print("charge_pol QUA multipliers range:", min(charge_pol_amps), max(charge_pol_amps))
    print("scc QUA multipliers range:", min(scc_amp_list), max(scc_amp_list))
    # sys.exit()
    # nv_list[i] will have the ith coordinates from the above lists
    nv_list: list[NVSig] = []
    for ind in range(num_nvs):
        # if ind not in selected_inds:
        #     continue
        coords = {
            CoordsKey.SAMPLE: sample_coords,
            CoordsKey.Z: z_coord,
            CoordsKey.PIXEL: pixel_coords_list[ind],
            green_laser_aod: green_coords_list[ind],    
            red_laser_aod: red_coords_list[ind],
        }
        nv_sig = NVSig(
            name=f"{sample_name}-nv{ind}_{date_str}",
            coords=coords,
            threshold=threshold_list[ind],
            pulse_durations={
                VirtualLaserKey.SCC: scc_duration_list[ind],
                VirtualLaserKey.CHARGE_POL: pol_duration_list[ind],
            },
            pulse_amps={
                VirtualLaserKey.SCC: scc_amp_list[ind],
                VirtualLaserKey.ION: scc_amp_list[ind],
                VirtualLaserKey.CHARGE_POL: charge_pol_amps[ind],
            },
        )
        nv_list.append(nv_sig)
    # print(nv_sig)
    # Additional properties for the representative NV
    nv_list[0].representative = True
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    # print(f"Created NV: {nv_sig.name}, Coords: {nv_sig.coords}")
    # nv_sig.expected_counts =  3093.0
    # nv_sig.expected_counts = 1850
    # nv_list = nv_list[::-1]  # flipping the order of NVs
    # nv_list = nv_list[:200]
    print(f"length of NVs list:{len(nv_list)}")
    # sys.exit()
    # endregion

    # region Functions to run
    email_recipient = "schand@berkeley.edu"
    do_email = False
    try:
        # this is to create a flag that tell expt is runnig
        with open("experiment_running.flag", "w") as f:
            f.write("running")
        # pass
        kpl.init_kplotlib()
        # tb.init_safe_stop()
        # widefield.reset_all_drift()1
        # do_optimize_z(nv_sig)
        # do_optimize_xyz(nv_sig) 
        # pos.set_xyz_on_nv(nv_sig)
        # piezo_voltage_to_pixel_calibration()
        
        # widefield.plot_aod_freqs_over_space(
        #     pixel_coords_list,
        #     green_coords_list,
        #     red_coords_list,
        #     marker_size=10,
        # )
        
        # widefield.plot_amp_distribution_over_space(
        #     pixel_coords_list,
        #     charge_pol_amps,
        #     scc_amp_list,
        # )

        do_compensate_for_drift(nv_sig)
        
        # do_red_calibration_image(
        #     nv_sig,
        #     green_coords_list,
        #     force_laser_key=VirtualLaserKey.IMAGING,
        # )

        # do_widefield_image_sample(nv_sig, 50)     
        # do_widefield_image_sample(nv_sig, 200)

        # for nv in nv_list: 
        #     do_scanning_image_sample_zoom(nv)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        
        # do_scanning_image_full_roi(nv_sig)

        # scan_equilateral_triangle(nv_sig, center_coord=sample_coords, radius=0.6)
        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)
        # z_range = np.linspace(-4.2, -3.2, 11)
        # for z in z_range:
        #     nv_sig.coords[CoordsKey.Z] = z
        #     # do_scanning_image_sample(nv_sig)
        #     do_widefield_image_sample(nv_sig, 50)
        
        # x_range = np.linspace(-2.0, 6.0, 6)
        # y_range = np.linspace(-2.0, 6.0, 6)
        # # --- Step 1: Start at (0, 0) ---
        # sample_coord = [0.0, 0.0]
        # z = estimate_z(*sample_coord)
        # nv_sig.coords[Coo/*rdsKey.SAMPLE] = sample_coord
        # nv_sig.coords[CoordsKey.Z] = z
        
        # # --- Step 2: Loop over all other (x, y) positions ---
        # for x in x_range:
        #     for y in y_range:
        #         if np.isclose(x, 0.0) and np.isclose(y, 0.0):
        #             continue  # already scanned at (0, 0)
        #         sample_coord = [x, y]
        #         z = estimate_z(x, y)
        #         nv_sig.coords[CoordsKey.SAMPLE] = sample_coord
        #         nv_sig.coords[CoordsKey.Z] = z
        #         print(f"Scanning SAMPLE: {sample_coord}, estimated Z: {z:.3f}")
        #         do_scanning_image_sample(nv_sig)

        # do_opx_constant_ac()
        # do_opx_square_wave()
        
        # do_green_red_triplet_time_mux()
        
        # do_optimize_pixel(nv_sig)
        # do_optimize_green(nv_sig)
        # repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
        # do_optimize_red(nv_sig, repr_nv_sig)
        # do_optimize_z(nv_sig)

        # do_optimize_sample(nv_sig)
        # optimize.optimize_pixel_and_z(nv_sig, do_plot=True)
        # coords_key = None
        # coords_key = green_laser
        # coords_key = red_laser
        # do_optimize_loop(np.array(nv_list), np.array(coords_key))
 
        do_charge_state_histograms(nv_list)
        # do_charge_state_conditional_init(nv_list)
        # do_adaptive_charge_initialization(nv_list)
        # do_charge_state_particle_memory(nv_list)
        # do_charge_state_particle_memory_wait_sweep(nv_list)
        # do_charge_state_measurement_backaction(nv_list)
        
        # do_dmd_crosstalk_matrix(
        #     nv_list_all=nv_list,
        #     num_sources=200,
        #     dmd_radius_px=6,
        # )
        
        # do_dmd_turnoff_crosstalk_extinction_matrix(nv_list)
        
        # do_charge_correlation(nv_list)
        # do_charge_state_histograms_images(nv_list, vary_pol_laser=True)

        # do_optimize_pol_amp(nv_list)
        # do_optimize_pol_duration(nv_list)
        # do_optimize_readout_amp(nv_list)
        # do_optimize_readout_amp_repeated_readout(nv_list)
        # do_optimize_pol_duration(nv_list)
    
        # do_optimize_readout_duration(nv_list)
        # optimize_readout_amp_and_duration(nv_list)
        # do_optimize_spin_pol_amp(nv_list)
        # do_check_readout_fidelity(nv_list)
        # do_optimize_aod_access_time(nv_list)

        # do_scc_snr_check(nv_list)
        # do_optimize_scc_duration(nv_list)
        # do_optimize_scc_amp(nv_list)
        # optimize_scc_amp_and_duration(nv_list)
        # do_optimize_scc_readout_amp(nv_list)
        # do_crosstalk_check(nv_sig)
        # do_spin_pol_check(nv_sig)

        # do_calibrate_green_red_delay()
        # do_spin_echo_phase_scan_test(nv_list)  # for iq mod test
        # evol_time_list = [18000, 19600, 21000]

        # evol_time_list = [16]
        # seq_types = ["hahn", "xy4", "xy8"]
        # for seq_type in seq_types:
        #     for evol_time in evol_time_list:
        #         print(f"Running {seq_type} at evol_time={evol_time} ns")
        #         do_widefield_coherence_test(nv_list, evol_time, seq_type)

        # do_widefield_coherence_test(nv_list, 800, "xy8")

        # do_bootstrapped_pulse_error_tomography(nv_list)
        # do_calibrate_iq_delay(nv_list)
        # do_rabi(nv_list)
        # do_power_rabi(nv_list)
        # do_resonance(nv_list)
        # do_optimize_pol_duration(nv_list)
        # do_rabi(nv_list)
        # do_deer_hahn(nv_list)
        # do_deer_hahn_rabi(nv_list)
        # do_resonance_zoom(nv_list)
        # do_spin_echo(nv_list)
        # do_spin_echo_1(nv_list)
        # do_ramsey(nv_list)

        # do_simple_correlation_test(nv_list)
        # do_two_block_hahn_spatial_correlation(nv_list)
        # do_T2_correlation_test(nv_list)
        # do_two_block_hahn_correlation(nv_list)
        # do_reson1ance(nv_list)
        # do_sq_relaxation(nv_list)
        # do_dq_relaxation(nv_list)
        # do_detect_cosmic_rays(nv_list)
        # do_check_readout_fidelity(nv_list)
        # do_charge_quantum_jump(nv_list)
        # do_ac_stark(nv_list)
        # do_dm_xy_iq_lockin(nv_list)
        # do_two_block_hahn_correlation_dm(nv_list)

        # do_two_block_hahn_spatial_correlation(nv_list)

        # AVAILABLE_XY = ["hahn-n", "xy2-n", "xy4-n", "xy8-n", "xy16-n"]
        # run all (same style as before)
        # do_xy(nv_list, xy_seq="xy8-1")
        # do_xy_uniform_revival_scan(nv_list, xy_seq="xy4-1")
        # do_xy_revival_scan(nv_list, xy_seq="xy4-1")
        # do_all_xy_log(nv_list, T2_us=600, blocks=1)
        # same calling style as before
        # AVAILABLE_XY = ["xy8-1", "hahn-1", "xy2-1", "xy4-1", "xy16-1"]
        # for seq in AVAILABLE_XY:
        #     do_xy(nv_list, xy_seq=seq, min_tau_ns=200, hahn_max_tau_ns=600_000)  # 1 ms max tau for Hahn
        # for nv in nv_list:
        #     nv.spin_flip = False
        # for nv in nv_list[: num_nvs // 2]:
        #     nv.spin_flip = True
        # do_simple_correlation_test(nv_list)
        # do_correlation_test(nv_list)

        # region Cleanup
    except Exception as exc:
        if do_email:
            recipient = email_recipient
            tb.send_exception_email(email_to=recipient)
        raise exc

    finally:
        if os.path.exists("experiment_running.flag"):
            os.remove("experiment_running.flag")  # Clear flag

        if do_email:
            msg = "Experiment complete!"
            recipient = email_recipient
            tb.send_email(msg, email_to=recipient)

        print()
        print("Routine complete")

        # Maybe necessary to make sure we don't interrupt a sequence prematurely
        # tb.poll_safe_stop()

        # Make sure everything is reset
        tb.reset_cfm()
        cxn = common.labrad_connect()
        cxn.disconnect()
        tb.reset_safe_stop()
        plt.show(block=True)

    # endregion