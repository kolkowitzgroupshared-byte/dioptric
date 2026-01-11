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

import cv2
import matplotlib.pyplot as plt
import numpy as np

from majorroutines import targeting
from majorroutines.widefield import (
    ac_stark,
    bootstrapped_pulse_error_tomography,
    calibrate_iq_delay,
    charge_monitor,
    charge_state_conditional_init,
    charge_state_histograms,
    charge_state_histograms_images,
    correlation_test,
    crosstalk_check,
    image_sample,
    optimize_amp_duration_charge_state_histograms,
    optimize_charge_state_histograms_mcc,
    optimize_scc,
    optimize_scc_readout,
    optimize_scc_amp_duration,
    optimize_spin_pol,
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
    num_steps = 15
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
    total_range = 30
    scan_range = 10
    num_steps = 10
    image_sample.scanning_full_roi(nv_sig, total_range, scan_range, num_steps)


def do_scanning_image_sample_zoom(nv_sig):
    scan_range = 0.001
    num_steps = 5
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

    # 100 ms
    # num_reps = 100
    # num_runs = 20

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
    return optimize_charge_state_histograms_mcc.optimize_pol_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_pol_amp(nv_list):
    num_steps = 24
    # num_reps = 150
    # num_runs = 5
    num_reps = 10
    num_runs = 220
    min_amp = 0.7
    max_amp = 1.2
    return optimize_charge_state_histograms_mcc.optimize_pol_amp(
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
    return optimize_charge_state_histograms_mcc.optimize_readout_duration(
        nv_list, num_steps, num_reps, num_runs, min_duration, max_duration
    )


def do_optimize_readout_amp(nv_list):
    # num_steps = 21
    num_steps = 18
    # num_reps = 150
    # num_runs = 5
    num_reps = 12
    # num_runs = 200
    num_runs = 400
    min_amp = 0.8 
    max_amp = 1.2
    return optimize_charge_state_histograms_mcc.optimize_readout_amp(
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
    num_reps = 20
    num_runs = 10
    # num_runs = 400
    return charge_state_conditional_init.main(nv_list, num_reps, num_runs)


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
        # Adjust for the drift that may have occurred since beginning the loop
        do_compensate_for_drift(repr_nv_sig)
        opti_coords_list.append(opti_coords)

    # Report back
    for opti_coords in opti_coords_list:
        r_opti_coords = [round(el, 3) for el in opti_coords[:2]]
        print(f"{r_opti_coords},")


def optimize_slm_Phase_calibration(repr_nv_sig, target_coords):
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    target_coords = np.array([[110.186, 129.281], [128.233, 88.007], [86.294, 103.0]])
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
    freq_center = 2.87
    
    freq_range = 0.36
    num_steps = 65
    
    # freq_range = 0.260
    # num_steps = 40
    num_reps = 2
    num_runs = 800
    # num_runs = 1
    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    ##
    # Remove duplicates and sort
    freqs = sorted(set(freqs))
    num_steps = len(freqs)
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


def do_deer_hahn(nv_list):
    freq_center = 0.174
    freq_range = 0.024
    # num_steps =  48
    # num_reps = 6
    num_reps = 4
    num_runs = 400
    # num_runs = 2
    # freqs = calculate_freqs(freq_center, freq_range, num_steps)
    # freqs = np.arange(20, 320 + 2, 2)
    freqs = np.arange(130, 190 + 1, 1)
    freqs = freqs / 1000
    ##
    # Remove duplicates and sort
    freqs = sorted(set(freqs))
    num_steps = len(freqs)
    for _ in range(1):
        do_widefield_image_sample(nv_sig, 50)
        deer_hahn.main(
            nv_list, 
            num_steps,
            num_reps,
            num_runs,
            freqs=freqs,
            uwave_ind_list=[0,1,2],
        )

def do_deer_hahn_rabi(nv_list):
    min_tau = 16
    max_tau = 240 + min_tau
    # max_tau = 360 + min_tau
    # max_tau = 480 + min_tau
    num_steps = 31
    num_reps = 10
    num_runs = 400
    # num_runs = 5
    uwave_ind_list = [0, 1]
    deer_hahn_rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # for _ in range(2):
    #     rabi.main(
    #         nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list
    #     )
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
    max_tau = 240 + min_tau
    # max_tau = 360 + min_tau
    # max_tau = 480 + min_tau
    num_steps = 31
    num_reps = 10
    num_runs = 400
    # num_runs = 5
    uwave_ind_list = [0, 1]
    rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # for _ in range(2):
    #     rabi.main(
    #         nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list
    #     )
    # uwave_ind_list = [0]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)
    # uwave_ind_list = [1]
    # rabi.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau, uwave_ind_list)


def do_widefield_coherence_test(nv_list, evol_time, seq_type):
    # num_reps = 11
    num_reps = 15
    num_runs = 150
    # num_runs = 2
    # phi_list = np.linspace(0, 360, num_steps)
    # fmt: off
    # phi_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    phi_list = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342, 360]
    # phi_list = [-351, -333, -315, -297, -279, -261, -243, -225, -207, -189, -171, -153, -135, -117, -99, -81, -63, -45, -27, -9, 9, 27, 45, 63, 81, 99, 117, 135, 153, 171, 189, 207, 225, 243, 261, 279, 297, 315, 333, 351]
    # fmt: on
    num_steps = len(phi_list)
    uwave_ind_list = [0, 1]  # both are has iq modulation
    widefield_coherence.main(
        nv_list, num_steps, num_reps, num_runs, phi_list, evol_time, seq_type, uwave_ind_list
    )
    # for _ in range(2):
    #     spin_echo_phase_scan_test.main(
    #         nv_list, num_steps, num_reps, num_runs, min_phi, max_phi, uwave_ind_list
    #     )


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


def do_xy(nv_list, xy_seq="xy8"):
    min_tau = 200
    max_tau = 1e6 + min_tau
    # num_steps = 24
    num_reps = 2
    uwave_ind_list = [0, 1]  # iq modulated
    num_runs = 400
    # taus calculation
    # taus = widefield.generate_log_spaced_taus(min_tau, max_tau, num_steps, base=4)
    taus = np.arange(200, 20000 + 1, 200)   # all divisible by 4
    taus = [int(t) for t in taus]
    num_steps = len(taus)
    # print(taus)
    # sys.exit()
    # num_runs = 2
    # xy8.main(nv_list, num_steps, num_reps, num_runs, taus , uwave_ind_list)
    for _ in range(3):
        xy.main(
            nv_list,
            num_steps,
            num_reps,
            num_runs,
            taus,
            uwave_ind_list,
            xy_seq,
        )


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
    taus = [round(tau / 4) * 4 for tau in taus]
    taus = sorted(set(taus))  # remove duplicates
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
    if True:
        sig_gen = cxn.sig_gen_STAN_sg394_3
        amp = 0
        chan = 3
    else:
        sig_gen = cxn.sig_gen_STAN_sg394_2
        amp = 10
        chan = 10
    sig_gen.set_amp(amp)  # 12
    sig_gen.set_freq(0.175)
    sig_gen.uwave_on()
    opx.constant_ac([chan])

    # Camera frame rate test
    # seq_args = [500]
    # seq_args_string = tb.encode_seq_args(seq_args)
    # opx.stream_load("camera_test.py", seq_args_string)
    # opx.stream_start()

    # Yellow
    # opx.constant_ac(
    #     [],  # Digital channels
    #     [7],  # Analog channels
    #     [0.35],  # Analog voltages
    #     [0],  # Analog frequencies
    # )
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
    # opx.constant_ac(
    #     [4],  # Digital channels
    #     [3, 4],  # Analog channels
    #     [0.02, 0.02],  # Analog voltages
    #     [107.0, 107.0],  # Analog frequencies
    # )
    # Green + red
    # opx.constant_ac(
    #     [4, 1],  # Digital channels
    #     [3, 4, 2, 6],  # Analog channels
    #     [0.15, 0.15, 0.15, 0.15],  # Analog voltages;
    #     [107, 107, 72, 72],
    # )
    # green_coords_list = [
    #     [108.302, 107.046],
    #     [122.658, 98.967],
    #     [96.376, 95.86],
    #     [106.999, 119.23],
    # ]
    # red_coords_list = [
    #     [72.722, 71.926],
    #     [84.601, 66.136],
    #     [63.555, 62.304],
    #     [71.244, 81.727],
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
    #     [4],  # Digital channels
    #     [3, 4, 7],  # Analog channels
    #     [0.11, 0.11, 0.30],  # Analog voltages
    #     [107, 107, 0],  # Analog frequencies
    # )
    # Red + green + Yellow
    # opx.constant_ac(
    #     [4, 1],  # Digital channels1
    #     [3, 4, 2, 6, 7],  # Analog channels
    #     [0.19, 0.19, 0.17, 0.17, 0.25],  # Analog voltages
    #     [107, 107, 72, 72, 0],  # Analog frequencies
    # )
    input("Press enter to stop...")
    # sig_gen.uwave_off()


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
        [[0.4, 0.2], [-0.1999, 0.5464], [-0.2, -0.1464]], dtype="float32"
    )
    cal_pixel_coords = np.array(
        [[135.141, 117.788], [97.234, 144.799], [92.568, 98.422]], dtype="float32"
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
    x_max=250,
    y_min=0,
    y_max=250,
):
    data = np.load(file_path, allow_pickle=True)
    nv_coordinates = data["nv_coordinates"]

    # Create a mask based on the min/max thresholds for x and y
    mask = (
        (nv_coordinates[:, 0] >= x_min)
        & (nv_coordinates[:, 0] <= x_max)
        & (nv_coordinates[:, 1] >= y_min)
        & (nv_coordinates[:, 1] <= y_max)
    )
    nv_coordinates_clean = nv_coordinates[mask]
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
    sample_name = "johnson"
    # magnet_angle = 90
    date_str = "2025_10_21"
    sample_coords = [0.4, 0.8]
    z_coord = 0.0
    # Load NV pixel coordinates1
    pixel_coords_list = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_308nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_254nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_151nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_136nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_118nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_312nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_230nvs_reordered.npz",
        # file_path="slmsuite/nv_blob_detection/nv_blob_223nvs_reordered.npz",
        file_path="slmsuite/nv_blob_detection/nv_blob_204nvs_reordered.npz",
    ).tolist()
    # pixel_coords_list = [[124.195, 127.341],[14.043, 37.334],[106.538, 237.374],[218.314, 23.302]]
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

    # Print first coordinate set for verification
    # print(f"Number of NVs: {green_coords_list}")
    # print(f"Number of NVs: {red_coords_list}")
    # sys.exit()
    print(f"Number of NVs: {len(pixel_coords_list)}")
    print(f"Reference NV:{pixel_coords_list[0]}")
    print(f"Green Laser Coordinates: {green_coords_list[0]}")
    print(f"Red Laser Coordinates: {red_coords_list[0]}")

    # pixel_coords_list = [[124.195, 127.341],[14.043, 37.334],[106.538, 237.374],[218.314, 23.302]]
    # green_coords_list = [[107.884, 107.983],[119.262, 119.511],[111.272, 95.718],[95.966, 118.875]]
    # red_coords_list = [[73.27, 72.27],[82.164, 82.223],[76.471, 62.475],[63.144, 80.513]]

    num_nvs = len(pixel_coords_list)
    threshold_list = [None] * num_nvs
    # fmt: off
    ## 308nvs
    # pol_duration_list = [336, 336, 308, 308, 428, 428, 504, 504, 816, 816, 528, 528, 372, 372, 1060, 1060, 852, 852, 852, 852, 612, 612, 484, 484, 1120, 1120, 852, 852, 404, 404, 812, 812, 672, 672, 560, 560, 644, 644, 352, 352, 380, 380, 852, 852, 400, 400, 620, 620, 628, 628, 292, 292, 528, 528, 392, 392, 524, 524, 680, 680, 504, 504, 396, 396, 324, 324, 428, 428, 240, 240, 504, 504, 540, 540, 852, 852, 1188, 1188, 764, 764, 976, 976, 820, 820, 444, 444, 1100, 1100, 488, 488, 604, 604, 972, 972, 380, 380, 352, 352, 660, 660, 592, 592, 416, 416, 452, 452, 620, 620, 576, 576, 316, 316, 660, 660, 660, 660, 720, 720, 620, 620, 1024, 1024, 320, 320, 852, 852, 1396, 1396, 464, 464, 416, 416, 624, 624, 1008, 1008, 460, 460, 508, 508, 668, 668, 448, 448, 440, 440, 668, 668, 852, 852, 852, 852, 844, 844, 1048, 1048, 320, 320, 780, 780, 492, 492, 1476, 1476, 656, 656, 1064, 1064, 456, 456, 344, 344, 852, 852, 540, 540, 352, 352, 524, 524, 852, 852, 1156, 1156, 1388, 1388, 308, 308, 852, 852, 1360, 1360, 572, 572, 204, 204, 316, 316, 696, 696, 504, 504, 1332, 1332, 1012, 1012, 708, 708, 852, 852, 912, 912, 804, 804, 608, 608, 948, 948, 596, 596, 1256, 1256, 808, 808, 852, 852, 392, 392, 568, 568, 872, 872, 1268, 1268, 780, 780, 852, 852, 476, 476, 508, 508, 640, 640, 392, 392, 512, 512, 700, 700, 700, 700, 932, 932, 840, 840, 852, 852, 1248, 1248, 852, 852, 852, 852, 1444, 1444, 620, 620, 852, 852, 852, 852, 660, 660, 752, 752, 1052, 1052, 592, 592, 852, 852, 852, 852, 1248, 1248, 860, 860, 520, 520, 1320, 1320, 1096, 1096, 568, 568, 488, 488, 852, 852, 556, 556, 420, 420, 1192, 1192, 552, 552, 1032, 1032, 508, 508, 1268, 1268, 872, 872, 852, 852, 852, 852, 560, 560, 328, 328, 1232, 1232, 1288, 1288, 500, 500, 356, 356, 836, 836, 852, 852, 392, 392, 940, 940, 1252, 1252, 1428, 1428, 896, 896, 1260, 1260, 1260, 1260, 852, 852, 776, 776, 796, 796, 368, 368, 1164, 1164, 1276, 1276, 1472, 1472, 448, 448, 1000, 1000, 504, 504, 1096, 1096, 612, 612, 584, 584, 660, 660, 776, 776, 684, 684, 1424, 1424, 852, 852, 416, 416, 1452, 1452, 996, 996, 668, 668, 484, 484, 364, 364, 548, 548, 472, 472, 852, 852, 1080, 1080, 852, 852, 1276, 1276, 1188, 1188, 852, 852, 852, 852, 324, 324, 1124, 1124, 300, 300, 512, 512, 884, 884, 852, 852, 1140, 1140, 852, 852, 1124, 1124, 852, 852, 1144, 1144, 852, 852, 824, 824, 852, 852, 1080, 1080, 1000, 1000, 1296, 1296, 852, 852, 1284, 1284, 852, 852, 852, 852, 1196, 1196, 432, 432, 1112, 1112, 696, 696, 400, 400, 852, 852, 852, 852, 440, 440, 852, 852, 1260, 1260, 808, 808, 572, 572, 852, 852, 772, 772, 428, 428, 940, 940, 852, 852, 480, 480, 1196, 1196, 1020, 1020, 492, 492, 1012, 1012, 852, 852, 964, 964, 1284, 1284, 852, 852, 852, 852, 852, 852, 852, 852, 820, 820, 852, 852, 944, 944, 1180, 1180, 852, 852, 528, 528, 1432, 1432, 852, 852, 976, 976, 764, 764, 1048, 1048, 852, 852, 852, 852, 852, 852, 352, 352, 852, 852, 1408, 1408, 564, 564, 852, 852, 852, 852, 1460, 1460, 1072, 1072, 548, 548, 852, 852, 688, 688, 852, 852, 852, 852, 488, 488, 1028, 1028, 540, 540, 1400, 1400, 852, 852, 852, 852, 1000, 1000, 852, 852, 892, 892, 852, 852, 852, 852, 1056, 1056, 852, 852, 1496, 1496, 852, 852, 852, 852, 1316, 1316, 1396, 1396, 1172, 1172, 852, 852, 852, 852, 852, 852, 708, 708]
    # scc_duration_list = [142, 142, 142, 142, 64, 142, 142, 142, 136, 142, 142, 142, 142, 142, 142, 142, 80, 142, 142, 142, 142, 196, 88, 108, 108, 142, 142, 142, 72, 142, 142, 142, 142, 142, 64, 142, 142, 142, 142, 142, 104, 160, 84, 142, 36, 36, 92, 142, 142, 142, 48, 56, 48, 172, 142, 142, 80, 142, 142, 48, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 72, 112, 142, 142, 142, 140, 142, 142, 52, 72, 72, 142, 142, 36, 36, 68, 142, 142, 142, 142, 36, 48, 142, 142, 142, 152, 142, 142, 104, 72, 68, 124, 112, 108, 164, 168, 142, 142, 142, 142, 142, 142, 64, 64, 142, 132, 142, 142, 72, 142, 152, 142, 164, 164, 164, 142, 142, 142, 156, 142, 142, 142, 142, 142, 142, 142, 142, 142, 124, 142, 142, 142, 64, 142, 108, 108, 142, 142, 142, 142, 142, 142, 140, 142, 142, 142, 100, 142, 142, 142, 188, 188, 76, 142, 142, 100, 142, 160, 160, 124, 142, 142, 136, 142, 142, 142, 142, 142, 142, 64, 142, 142, 142, 142, 132, 172, 56, 142, 64, 64, 196, 68, 142, 92, 92, 142, 142, 142, 142, 142, 142, 142, 48, 142, 142, 144, 142, 142, 142, 142, 142, 142, 142, 142, 196, 142, 142, 142, 76, 142, 142, 142, 64, 142, 142, 136, 136, 142, 142, 142, 142, 100, 142, 142, 142, 142, 142, 96, 142, 142, 124, 124, 124, 142, 142, 142, 142, 142, 56, 142, 142, 142, 142, 142, 88, 142, 142, 196, 142, 120, 120, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 72, 142, 142, 142, 116, 116, 48, 36, 36, 142, 142, 142, 36, 104, 56, 40, 142, 142, 142, 142, 40, 142, 142, 142, 142, 92, 142, 192, 142, 68, 142, 142, 142, 142, 168, 142, 142]
    #254NVs
    # pol_duration_list = [336, 336, 308, 308, 428, 428, 504, 528, 372, 1060, 1060, 852, 852, 852, 612, 484, 1120, 852, 404, 404, 812, 672, 672, 560, 644, 644, 352, 352, 380, 380, 852, 852, 620, 620, 628, 628, 292, 292, 528, 528, 392, 392, 524, 524, 680, 504, 396, 396, 324, 324, 428, 240, 240, 504, 540, 540, 852, 1188, 1188, 764, 764, 976, 976, 820, 820, 444, 444, 1100, 488, 488, 604, 972, 380, 380, 352, 352, 660, 592, 592, 416, 416, 452, 452, 620, 620, 576, 576, 316, 316, 660, 660, 660, 720, 720, 620, 1024, 852, 852, 1396, 1396, 464, 464, 416, 416, 624, 624, 1008, 1008, 460, 460, 508, 508, 668, 668, 448, 448, 440, 668, 668, 852, 852, 852, 852, 844, 1048, 1048, 320, 320, 780, 780, 492, 1476, 1476, 656, 656, 1064, 1064, 456, 344, 344, 852, 540, 352, 524, 852, 852, 1156, 1388, 308, 852, 852, 1360, 1360, 572, 572, 204, 696, 504, 504, 1332, 1332, 1012, 1012, 708, 708, 852, 912, 912, 804, 804, 608, 608, 948, 948, 596, 596, 1256, 1256, 808, 808, 852, 392, 392, 568, 568, 872, 872, 1268, 1268, 780, 780, 852, 852, 476, 476, 508, 508, 640, 640, 392, 392, 512, 700, 700, 700, 700, 932, 932, 840, 840, 852, 852, 1248, 1248, 852, 852, 852, 1444, 1444, 620, 620, 852, 852, 752, 752, 1052, 1052, 852, 852, 852, 852, 1248, 860, 860, 520, 520, 1320, 1320, 1096, 1096, 568, 568, 488, 488, 852, 852, 556, 556, 420, 420, 1192, 1192, 552, 552]
    # scc_duration_list = [142, 142, 142, 142, 64, 142, 142, 142, 142, 142, 142, 142, 142, 142, 196, 108, 108, 142, 72, 142, 142, 142, 142, 64, 142, 142, 142, 142, 104, 160, 84, 142, 92, 142, 142, 142, 48, 56, 48, 172, 142, 142, 80, 142, 48, 142, 142, 142, 142, 142, 142, 142, 142, 72, 142, 142, 142, 142, 142, 52, 72, 72, 142, 142, 36, 36, 68, 142, 142, 142, 36, 142, 142, 152, 142, 142, 104, 68, 124, 112, 108, 164, 168, 142, 142, 142, 142, 142, 142, 64, 64, 132, 142, 142, 72, 142, 164, 142, 142, 142, 156, 142, 142, 142, 142, 142, 142, 142, 142, 142, 124, 142, 142, 142, 64, 142, 108, 142, 142, 142, 142, 142, 142, 140, 142, 142, 100, 142, 142, 142, 188, 76, 142, 142, 100, 142, 160, 124, 142, 142, 142, 142, 142, 64, 142, 142, 142, 132, 56, 64, 64, 196, 68, 142, 92, 92, 142, 142, 142, 48, 142, 142, 144, 142, 142, 142, 142, 142, 142, 142, 196, 142, 142, 142, 76, 142, 142, 142, 64, 142, 136, 136, 142, 142, 142, 142, 100, 142, 142, 142, 142, 142, 96, 142, 142, 124, 124, 124, 142, 142, 142, 142, 56, 142, 142, 142, 142, 142, 88, 142, 142, 196, 142, 120, 120, 142, 142, 142, 142, 142, 142, 142, 72, 142, 116, 116, 48, 142, 142, 142, 36, 56, 40, 142, 142, 142, 142, 40, 142, 142, 142, 142, 92, 142, 192, 142, 68, 142, 142, 142, 142, 168, 142, 142]
    # 136NVs
    # pol_duration_list = [504, 504, 648, 648, 592, 592, 608, 608, 680, 680, 884, 884, 652, 652, 556, 556, 408, 408, 680, 680, 304, 304, 396, 396, 368, 368, 708, 708, 592, 592, 724, 724, 412, 412, 324, 324, 352, 352, 360, 360, 428, 428, 316, 316, 420, 420, 728, 728, 680, 680, 360, 360, 504, 504, 300, 300, 420, 420, 400, 400, 552, 552, 272, 272, 568, 568, 516, 516, 512, 512, 300, 300, 680, 680, 380, 380, 304, 304, 580, 580, 648, 648, 764, 764, 596, 596, 852, 852, 928, 928, 496, 496, 444, 444, 620, 620, 640, 640, 588, 588, 572, 572, 768, 768, 996, 996, 616, 616, 908, 908, 752, 752, 644, 644, 1508, 1508, 664, 664, 928, 928, 1092, 1092, 468, 468, 416, 416, 444, 444, 760, 760, 760, 760, 1052, 1052, 844, 844, 492, 492, 324, 324, 516, 516, 676, 676, 964, 964, 528, 528, 684, 684, 820, 820, 1084, 1084, 552, 552, 752, 752, 952, 952, 956, 956, 968, 968, 1428, 1428, 892, 892, 788, 788, 500, 500, 416, 416, 808, 808, 656, 656, 240, 240, 1352, 1352, 1084, 1084, 964, 964, 680, 680, 592, 592, 680, 680, 1204, 1204, 656, 656, 656, 656, 972, 972, 660, 660, 1476, 1476, 1500, 1500, 808, 808, 568, 568, 832, 832, 520, 520, 1272, 1272, 1152, 1152, 572, 572, 1020, 1020, 680, 680, 1292, 1292, 740, 740, 1264, 1264, 864, 864, 1060, 1060, 1188, 1188, 656, 656, 1392, 1392, 980, 980, 1308, 1308, 868, 868, 1092, 1092, 1784, 1784, 956, 956, 1076, 1076, 680, 680, 1372, 1372, 680, 680, 1924, 1924, 1640, 1640, 1176, 1176, 1676, 1676, 1476, 1476, 972, 972]
    # scc_duration_list = [76, 88, 116, 92, 88, 104, 108, 80, 80, 88, 100, 96, 96, 92, 72, 112, 96, 92, 92, 140, 96, 76, 100, 112, 72, 96, 92, 72, 92, 80, 100, 76, 80, 104, 80, 76, 96, 60, 68, 80, 80, 128, 92, 112, 80, 96, 80, 112, 72, 76, 76, 124, 124, 104, 108, 72, 84, 100, 112, 92, 180, 116, 76, 108, 112, 140, 120, 100, 72, 84, 128, 96, 100, 140, 96, 120, 136, 100, 128, 108, 92, 96, 96, 96, 96, 84, 92, 164, 88, 100, 132, 124, 100, 88, 84, 96, 124, 80, 88, 176, 128, 112, 172, 88, 140, 112, 108, 144, 104, 104, 112, 108, 244, 140, 108, 120, 100, 96, 164, 100, 140, 180, 108, 180, 92, 112, 124, 108, 176, 132, 120, 192, 232, 128, 104, 144]
    #118NVs
    # pol_duration_list =[504, 504, 648, 592, 608, 608, 680, 680, 884, 884, 652, 652, 556, 556, 408, 680, 304, 396, 396, 368, 708, 708, 592, 592, 724, 724, 412, 324, 324, 352, 360, 360, 428, 428, 316, 316, 420, 420, 728, 728, 680, 680, 360, 360, 504, 300, 300, 420, 420, 400, 400, 552, 272, 272, 568, 568, 516, 516, 512, 512, 300, 300, 680, 680, 380, 380, 304, 304, 580, 648, 648, 764, 764, 596, 596, 852, 852, 928, 928, 496, 496, 444, 620, 640, 640, 588, 588, 572, 572, 768, 768, 996, 996, 616, 908, 752, 752, 644, 1508, 1508, 664, 664, 928, 928, 1092, 1092, 468, 416, 444, 444, 760, 760, 760, 760, 1052, 1052, 844, 844]
    # scc_duration_list =[76, 88, 92, 88, 108, 80, 80, 88, 100, 96, 96, 92, 72, 112, 92, 92, 76, 100, 112, 72, 92, 72, 92, 80, 100, 76, 104, 80, 76, 96, 68, 80, 80, 128, 92, 112, 80, 96, 80, 112, 72, 76, 76, 124, 104, 108, 72, 84, 100, 112, 92, 116, 76, 108, 112, 140, 120, 100, 72, 84, 128, 96, 100, 140, 96, 120, 136, 100, 108, 92, 96, 96, 96, 96, 84, 92, 164, 88, 100, 132, 124, 88, 96, 124, 80, 88, 176, 128, 112, 172, 88, 140, 112, 108, 104, 112, 108, 140, 108, 120, 100, 96, 164, 100, 140, 180, 108, 112, 124, 108, 176, 132, 120, 192, 232, 128, 104, 144]
    ##johnson 312 NVs
    # pol_duration_list = [392, 392, 736, 736, 720, 720, 708, 708, 1092, 1092, 772, 772, 1108, 1108, 868, 868, 1204, 1204, 936, 936, 704, 704, 288, 288, 800, 800, 732, 732, 352, 352, 864, 864, 804, 804, 580, 580, 708, 708, 772, 772, 920, 920, 1020, 1020, 1076, 1076, 676, 676, 480, 480, 684, 684, 748, 748, 300, 300, 764, 764, 572, 572, 704, 704, 806, 806, 856, 856, 380, 380, 736, 736, 820, 820, 660, 660, 1888, 1888, 392, 392, 704, 704, 420, 420, 464, 464, 700, 700, 806, 806, 304, 304, 806, 806, 724, 724, 708, 708, 520, 520, 776, 776, 384, 384, 680, 680, 724, 724, 1740, 1740, 716, 716, 952, 952, 876, 876, 660, 660, 1212, 1212, 936, 936, 806, 806, 392, 392, 828, 828, 796, 796, 308, 308, 692, 692, 492, 492, 806, 806, 288, 288, 806, 806, 944, 944, 660, 660, 728, 728, 806, 806, 806, 806, 806, 806, 1360, 1360, 596, 596, 724, 724, 412, 412, 806, 806, 806, 806, 806, 806, 564, 564, 652, 652, 696, 696, 1540, 1540, 728, 728, 652, 652, 392, 392, 448, 448, 304, 304, 392, 392, 968, 968, 772, 772, 640, 640, 548, 548, 664, 664, 468, 468, 806, 806, 504, 504, 806, 806, 700, 700, 1128, 1128, 672, 672, 300, 300, 1536, 1536, 836, 836, 304, 304, 1036, 1036, 1576, 1576, 804, 804, 806, 806, 640, 640, 806, 806, 1776, 1776, 612, 612, 844, 844, 732, 732, 684, 684, 928, 928, 628, 628, 972, 972, 988, 988, 680, 680, 1064, 1064, 708, 708, 780, 780, 1004, 1004, 1484, 1484, 912, 912, 600, 600, 484, 484, 636, 636, 720, 720, 1088, 1088, 806, 806, 824, 824, 608, 608, 956, 956, 708, 708, 692, 692, 428, 428, 836, 836, 806, 806, 396, 396, 760, 760, 384, 384, 524, 524, 852, 852, 1600, 1600, 806, 806, 1268, 1268, 788, 788, 308, 308, 728, 728, 304, 304, 832, 832, 288, 288, 728, 728, 728, 728, 912, 912, 452, 452, 1452, 1452, 588, 588, 1892, 1892, 1896, 1896, 476, 476, 1072, 1072, 580, 580, 1244, 1244, 1656, 1656, 1792, 1792, 1600, 1600, 1172, 1172, 540, 540, 1556, 1556, 668, 668, 704, 704, 504, 504, 792, 792, 376, 376, 716, 716, 684, 684, 692, 692, 1100, 1100, 806, 806, 444, 444, 520, 520, 1404, 1404, 736, 736, 1696, 1696, 428, 428, 600, 600, 1608, 1608, 806, 806, 1220, 1220, 1644, 1644, 1040, 1040, 1376, 1376, 752, 752, 468, 468, 652, 652, 908, 908, 806, 806, 748, 748, 528, 528, 440, 440, 848, 848, 656, 656, 696, 696, 792, 792, 888, 888, 808, 808, 584, 584, 832, 832, 680, 680, 806, 806, 1060, 1060, 806, 806, 1000, 1000, 940, 940, 712, 712, 796, 796, 996, 996, 806, 806, 812, 812, 748, 748, 692, 692, 806, 806, 556, 556, 1032, 1032, 692, 692, 806, 806, 872, 872, 980, 980, 806, 806, 1540, 1540, 768, 768, 308, 308, 806, 806, 872, 872, 1020, 1020, 1512, 1512, 1080, 1080, 960, 960, 1920, 1920, 728, 728, 804, 804, 648, 648, 806, 806, 1180, 1180, 640, 640, 1188, 1188, 1052, 1052, 928, 928, 1816, 1816, 960, 960, 532, 532, 1396, 1396, 1032, 1032, 1336, 1336, 444, 444, 436, 436, 532, 532, 424, 424, 884, 884, 904, 904, 812, 812, 884, 884, 888, 888, 684, 684, 1176, 1176, 636, 636, 460, 460, 924, 924, 984, 984, 1244, 1244, 806, 806, 1552, 1552, 1040, 1040, 896, 896, 980, 980, 940, 940, 1144, 1144, 784, 784, 806, 806, 912, 912, 764, 764, 1120, 1120, 692, 692, 1840, 1840, 996, 996, 1644, 1644, 1796, 1796, 1064, 1064, 816, 816, 732, 732, 848, 848, 952, 952, 806, 806, 1028, 1028, 720, 720, 796, 796, 1268, 1268, 1584, 1584, 1720, 1720, 1768, 1768, 806, 806, 1388, 1388, 924, 924, 1496, 1496]
    # pol_duration_list = [ round(val/4) * 4 for val in pol_duration_list]
    # scc_duration_list = [180, 176, 178, 168, 104, 108, 156, 136, 180, 188, 304, 144, 252, 164, 172, 148, 244, 152, 172, 116, 180, 172, 128, 124, 108, 304, 64, 144, 84, 176, 148, 192, 144, 178, 160, 208, 248, 284, 120, 152, 268, 176, 124, 252, 204, 304, 178, 132, 156, 160, 124, 180, 124, 200, 176, 160, 144, 204, 204, 188, 88, 196, 148, 168, 136, 148, 280, 120, 92, 120, 216, 232, 156, 96, 180, 164, 152, 192, 172, 108, 178, 192, 144, 168, 264, 108, 196, 176, 112, 304, 124, 136, 120, 136, 144, 128, 124, 68, 192, 112, 268, 160, 144, 160, 212, 204, 304, 196, 136, 156, 104, 140, 88, 128, 140, 304, 128, 304, 116, 148, 304, 116, 120, 304, 148, 92, 204, 200, 304, 252, 268, 178, 212, 100, 148, 164, 216, 304, 216, 148, 152, 180, 164, 148, 178, 120, 156, 204, 96, 152, 304, 304, 304, 144, 304, 208, 220, 168, 164, 212, 252, 100, 112, 160, 144, 304, 244, 178, 148, 304, 304, 212, 216, 304, 120, 220, 232, 124, 192, 76, 152, 172, 124, 172, 212, 236, 200, 208, 164, 304, 240, 304, 176, 178, 292, 304, 148, 160, 168, 304, 172, 204, 232, 178, 304, 232, 304, 116, 178, 192, 216, 88, 160, 128, 120, 148, 144, 236, 276, 156, 132, 212, 172, 148, 156, 192, 104, 232, 288, 200, 300, 180, 232, 216, 304, 208, 84, 196, 292, 304, 304, 148, 304, 304, 160, 208, 304, 304, 304, 180, 144, 304, 168, 304, 172, 276, 304, 236, 304, 178, 176, 288, 178, 212, 184, 208, 180, 216, 252, 212, 188, 176, 304, 304, 304, 156, 200, 304, 136, 100, 136, 80, 276, 304, 240, 178, 180, 304, 178, 220, 304, 176, 196, 280, 304, 304, 200, 304, 304, 188, 304, 204, 178, 208, 236, 304, 192, 232, 304, 304, 168, 304]
    # scc_duration_list = [ round(val/4) * 4 for val in scc_duration_list]
    ##johnson 230 NVs
    # pol_duration_list = [760, 760, 668, 668, 608, 608, 700, 700, 1008, 1008, 616, 616, 492, 492, 836, 836, 392, 392, 1028, 1028, 312, 312, 772, 772, 600, 600, 1036, 1036, 840, 840, 728, 728, 728, 728, 1076, 1076, 412, 412, 440, 440, 860, 860, 848, 848, 704, 704, 508, 508, 652, 652, 836, 836, 796, 796, 728, 728, 712, 712, 696, 696, 436, 436, 612, 612, 612, 612, 748, 748, 956, 956, 676, 676, 668, 668, 404, 404, 776, 776, 468, 468, 688, 688, 548, 548, 1652, 1652, 652, 652, 1064, 1064, 488, 488, 616, 616, 744, 744, 368, 368, 468, 468, 744, 744, 740, 740, 1252, 1252, 668, 668, 536, 536, 820, 820, 400, 400, 812, 812, 1616, 1616, 984, 984, 576, 576, 920, 920, 624, 624, 548, 548, 692, 692, 692, 692, 536, 536, 552, 552, 508, 508, 684, 684, 672, 672, 492, 492, 388, 388, 496, 496, 1688, 1688, 652, 652, 1112, 1112, 756, 756, 480, 480, 556, 556, 1628, 1628, 1016, 1016, 664, 664, 716, 716, 780, 780, 624, 624, 1320, 1320, 644, 644, 620, 620, 688, 688, 880, 880, 576, 576, 1788, 1788, 744, 744, 1940, 1940, 676, 676, 696, 696, 1940, 1940, 716, 716, 668, 668, 680, 680, 1940, 1940, 692, 692, 712, 712, 944, 944, 776, 776, 796, 796, 732, 732, 684, 684, 668, 668, 752, 752, 856, 856, 596, 596, 776, 776, 1220, 1220, 616, 616, 308, 308, 520, 520, 808, 808, 740, 740, 952, 952, 1112, 1112, 1940, 1940, 1236, 1236, 1140, 1140, 656, 656, 1276, 1276, 1124, 1124, 100, 100, 1940, 1940, 1352, 1352, 640, 640, 1940, 1940, 1044, 1044, 468, 468, 780, 780, 456, 456, 536, 536, 584, 584, 1276, 1276, 760, 760, 396, 396, 600, 600, 1568, 1568, 764, 764, 796, 796, 780, 780, 1828, 1828, 1292, 1292, 1256, 1256, 840, 840, 516, 516, 760, 760, 1940, 1940, 876, 876, 684, 684, 756, 756, 808, 808, 616, 616, 604, 604, 844, 844, 588, 588, 800, 800, 1132, 1132, 1372, 1372, 976, 976, 836, 836, 1556, 1556, 628, 628, 944, 944, 1940, 1940, 460, 460, 892, 892, 444, 444, 960, 960, 700, 700, 1940, 1940, 528, 528, 1364, 1364, 448, 448, 468, 468, 1608, 1608, 1136, 1136, 1140, 1140, 648, 648, 984, 984, 936, 936, 808, 808, 1412, 1412, 600, 600, 748, 748, 912, 912, 1272, 1272, 1024, 1024, 596, 596, 980, 980, 1848, 1848, 640, 640, 912, 912, 1052, 1052, 868, 868, 580, 580, 836, 836, 792, 792, 600, 600, 972, 972, 1076, 1076, 1204, 1204, 1168, 1168, 1116, 1116, 988, 988, 800, 800, 988, 988, 1112, 1112, 1832, 1832, 1656, 1656, 1940, 1940, 964, 964, 1304, 1304, 728, 728, 892, 892, 1940, 1940, 940, 940, 1340, 1340, 1368, 1368, 1416, 1416, 1736, 1736, 1940, 1940, 1636, 1636, 1484, 1484]
    # scc_duration_list = [88, 92, 84, 112, 88, 92, 156, 88, 80, 100, 84, 84, 92, 72, 104, 92, 92, 68, 100, 72, 100, 96, 92, 76, 88, 88, 88, 92, 84, 108, 116, 92, 72, 96, 116, 112, 76, 92, 100, 88, 84, 72, 64, 88, 76, 68, 72, 88, 120, 80, 96, 88, 92, 116, 80, 92, 112, 104, 156, 116, 80, 80, 92, 92, 92, 84, 100, 96, 116, 80, 76, 72, 76, 84, 84, 88, 176, 88, 96, 92, 92, 76, 68, 84, 128, 80, 84, 116, 88, 36, 84, 88, 96, 76, 112, 96, 80, 140, 92, 96, 100, 76, 84, 72, 80, 120, 80, 88, 124, 100, 76, 116, 68, 80, 92, 84, 96, 92, 104, 92, 136, 116, 136, 112, 76, 84, 92, 176, 108, 104, 120, 96, 92, 92, 88, 88, 84, 92, 124, 84, 112, 92, 68, 88, 88, 92, 80, 136, 92, 92, 124, 88, 72, 104, 100, 120, 108, 108, 84, 88, 92, 112, 112, 88, 112, 96, 132, 96, 88, 112, 116, 108, 100, 84, 96, 116, 100, 88, 132, 88, 92, 148, 96, 100, 92, 140, 88, 84, 84, 92, 96, 144, 112, 100, 100, 112, 104, 96, 84, 104, 104, 116, 76, 120, 148, 128, 92, 92, 100, 92, 108, 108, 92, 108, 112, 104, 112, 120, 144, 88, 100, 120, 100, 116, 144, 112, 104, 116, 132, 108]
    ##johnson 223 NVs
    # pol_duration_list = [760, 760, 668, 668, 608, 608, 700, 700, 1008, 1008, 616, 616, 492, 492, 836, 836, 392, 392, 1028, 1028, 312, 312, 772, 772, 600, 600, 1036, 1036, 840, 840, 728, 728, 728, 1076, 1076, 412, 440, 440, 860, 860, 848, 704, 704, 508, 508, 652, 652, 836, 836, 796, 796, 728, 728, 712, 712, 696, 696, 436, 436, 612, 612, 612, 612, 748, 748, 956, 956, 676, 676, 668, 668, 404, 404, 776, 776, 468, 468, 688, 688, 548, 548, 1652, 1652, 652, 652, 1064, 488, 488, 616, 616, 744, 368, 368, 468, 468, 744, 744, 740, 740, 1252, 1252, 668, 668, 536, 536, 820, 400, 400, 812, 812, 1616, 1616, 984, 984, 576, 576, 920, 920, 624, 624, 548, 548, 692, 692, 692, 692, 536, 536, 552, 552, 508, 508, 684, 684, 672, 672, 492, 492, 388, 496, 496, 1688, 1688, 652, 652, 1112, 1112, 756, 756, 480, 480, 556, 556, 1628, 1628, 1016, 1016, 664, 664, 716, 716, 780, 780, 624, 624, 1320, 1320, 644, 644, 620, 620, 688, 688, 880, 880, 576, 576, 1788, 1788, 744, 744, 1940, 1940, 676, 676, 696, 696, 1940, 1940, 716, 716, 668, 668, 680, 680, 1940, 1940, 692, 692, 712, 712, 944, 944, 776, 776, 796, 796, 732, 732, 684, 684, 668, 668, 752, 752, 856, 856, 596, 596, 776, 776, 1220, 1220]
    # scc_duration_list = [88, 92, 84, 112, 88, 92, 156, 88, 80, 100, 84, 84, 92, 72, 104, 92, 92, 68, 100, 72, 100, 96, 92, 76, 88, 88, 88, 92, 84, 108, 116, 72, 96, 116, 112, 76, 100, 88, 84, 72, 88, 76, 68, 72, 88, 120, 80, 96, 88, 92, 116, 80, 92, 112, 104, 156, 116, 80, 80, 92, 92, 92, 84, 100, 96, 116, 80, 76, 72, 76, 84, 84, 88, 176, 88, 96, 92, 92, 76, 68, 84, 128, 80, 84, 116, 88, 84, 88, 96, 76, 96, 80, 140, 92, 96, 100, 76, 84, 72, 80, 120, 80, 88, 124, 100, 76, 68, 80, 92, 84, 96, 92, 104, 92, 136, 116, 136, 112, 76, 84, 92, 176, 108, 104, 120, 96, 92, 92, 88, 88, 84, 92, 124, 84, 112, 92, 68, 88, 88, 80, 136, 92, 92, 124, 88, 72, 104, 100, 120, 108, 108, 84, 88, 92, 112, 112, 88, 112, 96, 132, 96, 88, 112, 116, 108, 100, 84, 96, 116, 100, 88, 132, 88, 92, 148, 96, 100, 92, 140, 88, 84, 84, 92, 96, 144, 112, 100, 100, 112, 104, 96, 84, 104, 104, 116, 76, 120, 148, 128, 92, 92, 100, 92, 108, 108, 92, 108, 112, 104, 112, 120, 144, 88, 100, 120, 100, 116, 144, 112, 104, 116, 132, 108]
    ### Johnso 204NVs
    # pol_duration_list = [760, 760, 668, 668, 608, 608, 700, 1008, 1008, 616, 616, 492, 492, 836, 836, 392, 392, 1028, 1028, 312, 312, 772, 600, 600, 1036, 1036, 840, 840, 728, 728, 728, 412, 440, 440, 860, 860, 848, 704, 704, 508, 508, 652, 652, 836, 836, 796, 728, 728, 712, 696, 436, 436, 612, 612, 612, 612, 748, 956, 956, 676, 676, 668, 668, 404, 404, 776, 776, 468, 468, 688, 688, 548, 548, 1652, 1652, 652, 652, 1064, 488, 488, 616, 616, 744, 368, 368, 468, 468, 744, 740, 740, 1252, 1252, 668, 668, 536, 820, 400, 400, 812, 812, 1616, 1616, 984, 984, 576, 576, 920, 920, 624, 624, 548, 692, 692, 692, 536, 552, 552, 508, 508, 684, 684, 672, 492, 492, 388, 496, 496, 1688, 1688, 652, 652, 1112, 1112, 756, 756, 480, 556, 556, 1628, 1628, 1016, 1016, 664, 664, 716, 780, 780, 624, 624, 1320, 1320, 644, 644, 620, 620, 688, 688, 880, 880, 576, 576, 1788, 1788, 744, 744, 1940, 1940, 676, 676, 696, 696, 1940, 1940, 716, 716, 668, 668, 680, 1940, 1940, 692, 712, 712, 944, 944, 776, 776, 796, 796, 732, 684, 684, 668, 668, 752, 752, 856, 856, 596, 596, 776, 776, 1220, 1220]
    pol_duration_list = [740, 740, 948, 948, 1112, 1112, 556, 556, 948, 948, 756, 756, 756, 756, 824, 824, 1184, 1184, 804, 804, 744, 744, 828, 828, 1644, 1644, 948, 948, 560, 560, 876, 876, 1320, 1320, 948, 948, 972, 972, 748, 748, 1084, 1084, 948, 948, 1076, 1076, 1196, 1196, 840, 840, 1264, 1264, 760, 760, 936, 936, 864, 864, 856, 856, 812, 812, 852, 852, 872, 872, 760, 760, 732, 732, 800, 800, 952, 952, 1556, 1556, 892, 892, 936, 936, 1472, 1472, 720, 720, 700, 700, 944, 944, 788, 788, 808, 808, 768, 768, 1072, 1072, 784, 784, 832, 832, 776, 776, 1516, 1516, 996, 996, 972, 972, 864, 864, 940, 940, 800, 800, 980, 980, 916, 916, 836, 836, 936, 936, 764, 764, 788, 788, 760, 760, 800, 800, 1888, 1888, 692, 692, 876, 876, 932, 932, 948, 948, 1420, 1420, 728, 728, 928, 928, 848, 848, 912, 912, 876, 876, 884, 884, 1224, 1224, 1308, 1308, 856, 856, 1172, 1172, 960, 960, 1048, 1048, 1060, 1060, 824, 824, 844, 844, 800, 800, 1560, 1560, 756, 756, 1652, 1652, 1080, 1080, 816, 816, 864, 864, 876, 876, 900, 900, 812, 812, 892, 892, 1224, 1224, 1608, 1608, 960, 960, 780, 780, 504, 504, 1188, 1188, 972, 972, 968, 968, 1036, 1036, 924, 924, 852, 852, 948, 948, 1016, 1016, 1744, 1744, 924, 924, 884, 884, 816, 816, 796, 796, 816, 816, 1008, 1008, 952, 952, 796, 796, 936, 936, 1220, 1220, 948, 948, 888, 888, 1672, 1672, 848, 848, 880, 880, 1748, 1748, 1752, 1752, 1008, 1008, 824, 824, 844, 844, 980, 980, 948, 948, 1116, 1116, 784, 784, 988, 988, 904, 904, 804, 804, 832, 832, 864, 864, 1196, 1196, 1844, 1844, 948, 948, 1000, 1000, 896, 896, 936, 936, 1100, 1100, 948, 948, 824, 824, 1064, 1064, 1000, 1000, 1128, 1128, 912, 912, 948, 948, 1424, 1424, 948, 948, 832, 832, 928, 928, 948, 948, 908, 908, 940, 940, 1076, 1076, 976, 976, 1164, 1164, 1112, 1112, 1488, 1488, 920, 920, 944, 944, 960, 960, 948, 948, 1280, 1280, 880, 880, 1064, 1064, 1720, 1720, 848, 848, 1064, 1064, 1064, 1064, 1116, 1116, 1216, 1216, 1028, 1028, 880, 880, 1084, 1084, 888, 888, 1416, 1416, 1272, 1272, 1144, 1144, 948, 948, 1492, 1492, 1440, 1440, 948, 948, 948, 948, 1224, 1224, 976, 976, 1000, 1000, 940, 940, 948, 948, 1076, 1076, 1388, 1388, 1656, 1656, 1228, 1228, 1496, 1496, 948, 948, 1224, 1224, 1532, 1532]
    # scc_duration_list = [88, 92, 84, 112, 88, 92, 88, 80, 100, 84, 84, 92, 72, 104, 92, 92, 68, 100, 72, 100, 96, 92, 88, 88, 88, 92, 84, 108, 116, 72, 96, 76, 100, 88, 84, 72, 88, 76, 68, 72, 88, 120, 80, 96, 88, 92, 80, 92, 112, 156, 80, 80, 92, 92, 92, 84, 96, 116, 80, 76, 72, 76, 84, 84, 88, 176, 88, 96, 92, 92, 76, 68, 84, 128, 80, 84, 116, 88, 84, 88, 96, 76, 96, 80, 140, 92, 96, 76, 84, 72, 80, 120, 80, 88, 100, 76, 68, 80, 92, 84, 96, 92, 104, 92, 136, 116, 136, 112, 76, 84, 92, 108, 120, 96, 92, 88, 88, 84, 92, 124, 84, 92, 68, 88, 88, 80, 136, 92, 92, 124, 88, 72, 104, 100, 120, 108, 84, 88, 92, 112, 112, 88, 112, 96, 96, 88, 112, 116, 108, 100, 84, 96, 116, 100, 88, 132, 88, 92, 148, 96, 100, 92, 140, 88, 84, 84, 92, 96, 144, 112, 100, 100, 112, 104, 96, 84, 104, 104, 76, 120, 128, 92, 92, 100, 92, 108, 108, 92, 108, 104, 112, 120, 144, 88, 100, 120, 100, 116, 144, 112, 104, 116, 132, 108]
    # scc_duration_list = [88, 92, 80, 88, 96, 96, 92, 108, 96, 100, 88, 88, 88, 92, 88, 108, 64, 96, 72, 116, 84, 88, 80, 76, 92, 88, 84, 104, 120, 84, 100, 112, 140, 84, 100, 76, 88, 76, 88, 76, 80, 156, 84, 96, 88, 100, 72, 84, 84, 188, 88, 96, 84, 108, 104, 84, 84, 96, 84, 88, 76, 80, 96, 84, 100, 160, 104, 80, 96, 152, 84, 80, 80, 120, 80, 96, 108, 80, 84, 100, 100, 80, 96, 72, 128, 72, 76, 100, 92, 76, 76, 124, 88, 96, 84, 100, 88, 76, 100, 96, 80, 80, 120, 96, 120, 160, 148, 100, 80, 96, 92, 96, 112, 116, 92, 88, 92, 76, 108, 124, 84, 92, 88, 120, 100, 84, 124, 96, 112, 116, 88, 92, 96, 76, 148, 96, 92, 88, 92, 104, 116, 96, 116, 96, 92, 88, 108, 100, 104, 88, 80, 96, 112, 108, 108, 120, 96, 88, 112, 116, 116, 108, 140, 96, 92, 92, 96, 164, 160, 92, 108, 116, 96, 92, 100, 124, 100, 108, 96, 112, 148, 92, 84, 96, 100, 124, 96, 92, 84, 104, 108, 108, 124, 76, 92, 132, 104, 104, 160, 112, 160, 148, 220, 140]
    scc_duration_list = [60, 72, 64, 72, 68, 76, 68, 72, 88, 76, 64, 76, 72, 72, 64, 68, 68, 76, 68, 80, 92, 64, 72, 68, 76, 76, 64, 76, 88, 68, 60, 64, 100, 76, 68, 68, 84, 64, 64, 60, 80, 84, 72, 68, 84, 72, 72, 84, 80, 112, 96, 88, 76, 68, 72, 72, 72, 84, 68, 64, 60, 52, 68, 68, 76, 76, 64, 60, 76, 92, 72, 56, 64, 80, 60, 88, 76, 68, 92, 80, 68, 64, 88, 64, 92, 68, 72, 68, 72, 68, 60, 92, 60, 88, 76, 60, 60, 64, 92, 72, 68, 72, 84, 72, 76, 112, 108, 92, 76, 72, 100, 88, 76, 84, 88, 64, 80, 68, 80, 104, 76, 80, 68, 100, 76, 68, 96, 76, 120, 96, 72, 56, 80, 68, 104, 88, 76, 60, 68, 76, 100, 76, 100, 88, 68, 76, 88, 80, 80, 72, 88, 84, 92, 96, 84, 84, 80, 60, 120, 80, 80, 76, 124, 88, 72, 68, 76, 76, 104, 80, 88, 92, 100, 76, 80, 96, 72, 88, 76, 96, 112, 84, 76, 104, 96, 96, 100, 84, 84, 92, 96, 100, 124, 76, 96, 120, 76, 88, 112, 100, 108, 128, 120, 112] 
    # median = np.median(scc_duration_list)
    # scc_duration_list = [int(median) if (val < 24 or val > 200) else val for val in scc_duration_list]
    # print(scc_duration_list)
    # sys.exit()
    # arranged_scc_amp_list = [None] * num_nvs
    # arranged_scc_duration_list = [None] * num_nvs
    # arranged_pol_duration_list = [None] * len(pol_duration_list)
    # for i, idx in enumerate(include_indices):
    # arranged_scc_duration_list[idx] = scc_duration_list[i]
    # arranged_pol_duration_list[idx] = pol_duration_list[i]
    # arranged_scc_amp_list[idx] = scc_amp_list[i]
    # # # Assign back to original lists
    # scc_duration_list = arranged_scc_duration_list
    # pol_duration_list = arranged_pol_duration_list
    # scc_amp_list = arranged_scc_amp_list 
    indices_113_MHz = [0, 1, 3, 6, 10, 14, 16, 17, 19, 23, 24, 25, 26, 27, 32, 33, 34, 35, 37, 38, 41, 49, 50, 51, 53, 54, 55, 60, 62, 63, 64, 66, 67, 68, 70, 72, 73, 74, 75, 76, 78, 80, 81, 82, 83, 84, 86, 88, 90, 92, 93, 95, 96, 99, 100, 101, 102, 103, 105, 108, 109, 111, 113, 114]
    indices_217_MHz = [0, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 18, 20, 21, 22, 28, 29, 30, 31, 36, 39, 40, 42, 43, 44, 45, 46, 47, 48, 52, 56, 57, 58, 59, 61, 65, 69, 71, 77, 79, 85, 87, 89, 91, 94, 97, 98, 104, 106, 107, 110, 112, 115, 116, 117]
    # scc_amp_list = [1.0] * num_nv
    # scc_duration_list = [100] * num_nvs
    # pol_duration_list = [600] * num_nvs
    # pol_duration_list = [1000] * num_nvs
    # nv_list[i] will have the ith coordinates from the above lists
    nv_list: list[NVSig] = []
    for ind in range(num_nvs):
        # if ind not in indices_113_MHz:
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
                # VirtualLaserKey.SCC: scc_amp_list[ind],
                # VirtualLaserKey.CHARGE_POL: charge_pol_amps[ind],
            },
        )
        nv_list.append(nv_sig)
    # print(nv_sig)
    # Additional properties for the representative NV
    nv_list[0].representative = True
    # nv_list[1].representative = True
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    # print(f"Created NV: {nv_sig.name}, Coords: {nv_sig.coords}")
    # nv_sig.expected_counts = 900
    # nv_sig.expected_counts = 1400
    nv_sig.expected_counts = 1300
    # nv_sig.expected_counts = 1200
    # nv_sig.expected_counts = 1800

    # nv_list = nv_list[::-1]  # flipping the order of NVs
    # nv_list = nv_list[:1]
    print(f"length of NVs list:{len(nv_list)}")
    # sys.exit()
    # endregion

    # region Functions to run
    email_recipient = "mccambria@berkeley.edu"
    do_email = False
    try:
        # this is to create a flag that tell expt is runnig
        with open("experiment_running.flag", "w") as f:
            f.write("running")
        # pass
        kpl.init_kplotlib()
        # tb.init_safe_stop()
        # widefield.reset_all_drift()
        # do_optimize_z(nv_sig)
        # do_optimize_xyz(nv_sig)
        # pos.set_xyz_on_nv(nv_sig)
        # piezo_voltage_to_pixel_calibration()

        ### warning: this direclty iamge the laser spo, boftfor starign this makesure the red laser so set to 1mw on GUI
        ### ⚠️ CAUTION: direct laser imaging, check power
        ### ⚠️ CAUTION Set RED ≈ 0.1 mW • Exposure ≤ 0.1ms • Low em gain ≤ 10 / ND filter if needed
        # do_red_calibration_image(
        #     nv_sig,
        #     red_coords_list,
        #     force_laser_key=VirtualLaserKey.RED_IMAGING,
        # )

        do_compensate_for_drift(nv_sig)
        # do_widefield_image_sample(nv_sig, 50)
        # do_widefield_image_sample(nv_sig, 400)

        # for nv in nv_list:
            # do_scanning_image_sample_zoom(nv)

        # do_scanning_image_sample(nv_sig)
        # do_scanning_image_sample_zoom(nv_sig)
        # do_scanning_image_full_roi(nv_sig)

        # scan_equilateral_triangle(nv_sig, center_coord=sample_coords, radius=0.4)
        # do_image_nv_list(nv_list)
        # do_image_single_nv(nv_sig)
        # z_range = np.linspace(1.5, 1.9, 11)
        # for z in z_range:
        #     nv_sig.coords[CoordsKey.Z] = z
        #     do_scanning_image_sample(nv_sig)
        # x_range = np.linspace(-2.0, 6.0, 6)
        # y_range = np.linspace(-2.0, 6.0, 6)
        # # --- Step 1: Start at (0, 0) ---
        # sample_coord = [0.0, 0.0]
        # z = estimate_z(*sample_coord)
        # nv_sig.coords[CoordsKey.SAMPLE] = sample_coord
        # nv_sig.coords[CoordsKey.Z] = z
        # print(f"[START] Scanning SAMPLE: {sample_coord}, estimated Z: {z:.3f}")
        # do_scanning_image_sample(nv_sig)

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

        # do_optimize_pixel(nv_sig)
        # do_optimize_green(nv_sig)
        # repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
        # do_optimize_red(nv_sig, repr_nv_sig)
        # do_optimize_z(nv_sig).

        # do_optimize_sample(nv_sig)
        # optimize.optimize_pixel_and_z(nv_sig, do_plot=True)
        # coords_key = None
        # coords_key = green_laser
        # coords_key = red_laser
        # do_optimize_loop(np.array(nv_list), np.array(coords_key))

        # do_charge_state_histograms(nv_list)
        # do_charge_state_conditional_init(nv_list)
        # do_charge_state_histograms_images(nv_list, vary_pol_laser=True)

        # do_optimize_pol_amp(nv_list)
        # do_optimize_pol_duration(nv_list)
        # do_optimize_readout_amp(nv_list)
        # do_optimize_readout_duration(nv_list)
        # optimize_readout_amp_and_duration(nv_list) 
        # do_optimize_spin_pol_amp(nv_list)
        # do_check_readout_fidelity(nv_list)

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

        # evol_time_list = [15000]  # ns
        # seq_types = ["hahn", "xy4", "xy8"]  # or add "ramsey", "xy16"
        # for seq_type in seq_types:
        #     for evol_time in evol_time_list:
        #         print(f"Running {seq_type} at evol_time={evol_time} ns")
        #         do_widefield_coherence_test(nv_list, evol_time, seq_type)

        # do_bootstrapped_pulse_error_tomography(nv_list)
        # do_calibrate_iq_delay(nv_list)
        # do_rabi(nv_list)
        # do_power_rabi(nv_list)
        # do_resonance(nv_list)
        # do_rabi(nv_list)
        do_deer_hahn(nv_list)
        # do_deer_hahn_rabi(nv_list)
        # do_resonance_zoom(nv_list)
        # do_spin_echo(nv_list)
        # do_spin_echo_1(nv_list)
        # do_ramsey(nv_list)

        # do_simple_correlation_test(nv_list)
        # do_two_block_hahn_spatial_correlation(nv_list)
        # do_T2_correlation_test(nv_list)
        # do_two_block_hahn_correlation(nv_list)
        # do_resonance(nv_list) 
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
        # do_xy(nv_list, xy_seq="xy4-1")
        # do_xy_uniform_revival_scan(nv_list, xy_seq="xy4-1")
        # do_xy_revival_scan(nv_list, xy_seq="xy4-1")

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
