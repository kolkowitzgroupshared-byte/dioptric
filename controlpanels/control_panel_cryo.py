# -*- coding: utf-8 -*-
"""This file contains functions to control the CFM. Just change the function call
in the main section at the bottom of this file and run the file. Shared or
frequently changed parameters are in the __main__ body and relatively static
parameters are in the function definitions.

Created on Oct 7th, 2025

@author: chemistatcode
@author: Saroj B Chand
@author: ericvin
@author: mccambrias
"""


# region Imports and constants

import copy
import sys
import time

import labrad
import numpy as np

import majorroutines.calibration.calibrate_z_axis as calibrate_z_axis
import majorroutines.calibration.optimize_xy as optimize_xy
import majorroutines.calibration.optimize_z_PI as optimize_z_PI

# import majorroutines.confocal.determine_standard_readout_params as determine_standard_readout_params
# import majorroutines.confocal.g2_measurement as g2_measurement
import majorroutines.confocal.confocal_image_sample as image_sample

# import majorroutines.confocal.image_sample as image_sample
# import majorroutines.confocal.optimize_magnet_angle as optimize_magnet_angle
# import majorroutines.confocal.pulsed_resonance as pulsed_resonance
import majorroutines.confocal.confocal_rabi as rabi
import majorroutines.confocal.confocal_test_simple_spin_contrast as test_simple_spin_contrast

# import majorroutines.confocal.confocal_resonance as resonance

# import majorroutines.confocal.ramsey as ramsey
import majorroutines.confocal.confocal_resonance as resonance
import majorroutines.confocal.confocal_optimize_green_readout as optimize_green_readout_time
import majorroutines.confocal.confocal_optimize_apd_gate_width as optimize_apd_gate_width
import majorroutines.confocal.confocal_optimize_transient as optimize_transient
import majorroutines.confocal.optimize_green_power as optimize_green_power
import majorroutines.confocal.confocal_resonance_singlet_scan as resonance_tisapph_singlet_scan
import majorroutines.confocal.confocal_odmr_tisapph_short as odmr_tisapph_short
import majorroutines.confocal.confocal_apd_gate_overlap_scan as find_apd_gate_overlap

# import majorroutines.confocal.spin_echo as spin_echo
import majorroutines.confocal.confocal_stationary_count as stationary_count
import majorroutines.confocal.confocal_stationary_count_Tisapph as stationary_count_Tisapph
import majorroutines.confocal.z_scan_1d as z_scan_1d
import majorroutines.confocal.z_scan_2d as z_scan_2d

# import majorroutines.confocal.t1_dq_main as t1_dq_main
import majorroutines.targeting as targeting
import utils.tool_belt as tool_belt
from majorroutines.calibration import approach_surface, diagnose_z_direction
from majorroutines.confocal.confocal_2D_scan import confocal_scan_2D_xz
from majorroutines.confocal.z_scan_1d import main as scan_1D
from utils import common, kplotlib as kpl
from utils import positioning as pos
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey

# from utils.tool_belt import States

# endregion
# region Routines


def do_image_sample(nv_sig):
    """
    A 2D galvo scan while the piezo holds a fixed z position. The output figure shows
    photon counts at defined x,y galvo positions. Photon count is displayed as a color map.
    """

    scan_range = 0.2  # voltage
    num_steps = 90

    # For now we only support square scans so pass scan_range twice
    image_sample.confocal_scan(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
    )


def do_image_sample_zoom(nv_sig):
    scan_range = 0.08  #0.05 cryo iimage conversion: 37um/V; step size: x,y,z=30,30,40V
    num_steps = 35

    image_sample.confocal_scan(
        nv_sig,
        scan_range,
        scan_range,
        num_steps,
    )


# def do_image_sample_Hahn( # From Hahn control panel, should not work with current version of image_sample
#     nv_sig,
#     nv_minus_initialization=False,
#     cbarmin=None,
#     cbarmax=None,
# ):
#     # scan_range = 0.2
#     # num_steps = 60

#     scan_range = 0.5
#     num_steps = 90

#     # For now we only support square scans so pass scan_range twice
#     image_sample.main(
#         nv_sig,
#         scan_range,
#         scan_range,
#         num_steps,
#         nv_minus_initialization=nv_minus_initialization,
#         cmin=cbarmin,
#         cmax=cbarmax,
#     )


def do_2D_xz_scan(nv_sig):
    """
    A 2D z-scan of the piezo that sweeps the x-axis of the galvo.
    This is a modified version of the 1D scan designed for when NVs location
    are not known. Plots a line plot.

    This routine:
    1. Starts at the defined Galvo position
    2. Sweeps the galvo in X over the defined range (scan_range)
    3. Reads out photon counts at that position
    4. Plots the data in real-time for a position z set by the piezo
    5. When the plot is complete, the piezo will move down a defined step z and
       repeat the scan until the final defined z step is reached.

    """
    scan_range = 0.4  # voltage range for X axis
    num_steps = 60  # number of points along X

    # 1D scan function
    counts, x_positions = confocal_scan_2D_xz(
        nv_sig,
        scan_range,
        num_steps,
    )

    return counts, x_positions


def do_optimize_z_atto(nv_sig, num_steps=20, step_size=1, scan_direction="down"):
    """
    Optimize Z position by scanning and fitting a Gaussian to find the focus peak.

    # Uses the step-based scanning pattern from calibrate_z_axis.optimize_z which
    is compatible with the Attocube piezo (unlike targeting.optimize which requires
    streaming support).

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (pulse durations, laser settings)
    num_steps : int, optional
        Total number of Z positions to scan. Default: 40
    step_size : int, optional
        Step size in piezo units between positions. Default: 1
    scan_direction : str, optional
        Direction to scan: "up" starts low and scans upward (away from sample),
        "down" starts high and scans downward (toward sample). Default: "down"

    Returns
    -------
    float or None
        Optimal Z position (piezo steps), or None if optimization failed
    """
    results = calibrate_z_axis.optimize_z(
        nv_sig,
        num_steps=num_steps,
        step_size=step_size,
        num_averages=5,
        move_to_optimal=True,
        save_data=True,
        scan_direction=scan_direction,
    )

    opti_z = results.get("opti_z")  # Actual final position
    opti_z_fit = results.get("opti_z_fit")  # Gaussian fit estimate
    opti_counts = results.get("opti_counts")

    print(f"Z optimization complete: Final Z={opti_z}, Counts={opti_counts}")
    if opti_z_fit is not None:
        print(f"  (Gaussian fit estimated Z={opti_z_fit:.1f})")

    return opti_z


def do_optimize_z_PI(
    nv_sig, voltage_start, voltage_end, step_size=0.01, num_averages=3
):
    """
    Optimize Z position for PI E-709 piezo using voltage scan + Gaussian fit.

    Scans through the specified voltage range, collects photon counts,
    fits a Gaussian to find the optimal Z voltage, and moves to the peak.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (pulse durations, laser settings)
    voltage_start : float
        Starting voltage (V). Required. Must be in range [1.0, 9.0].
    voltage_end : float
        Ending voltage (V). Required. Must be in range [1.0, 9.0].
    step_size : float, optional
        Voltage step size (V). Default: 0.01V (10mV)
    num_averages : int, optional
        Photon count samples per position. Default: 3

    Returns
    -------
    float or None
        Optimal voltage (V), or None if optimization failed

    Examples
    --------
    do_optimize_z_PI(nv_sig, 3.90, 4.10)  # 20 steps at 10mV each
    do_optimize_z_PI(nv_sig, 3.90, 4.02, step_size=0.005)  # Fine: 24 steps at 5mV
    do_optimize_z_PI(nv_sig, 1.0, 9.0, step_size=0.1)  # Full range: 80 steps
    """
    results = optimize_z_PI.optimize_z_PI(
        nv_sig,
        voltage_start=voltage_start,
        voltage_end=voltage_end,
        step_size=step_size,
        num_averages=num_averages,
        move_to_optimal=True,
        save_data=True,
        use_position_feedback=False,  # qPOS() times out in external control mode
    )

    opti_voltage = results.get("opti_voltage")
    opti_counts = results.get("opti_counts")

    print(f"Z optimization complete: V={opti_voltage:.4f}")
    if opti_counts is not None:
        print(f"  Counts at optimal: {opti_counts}")

    return opti_voltage


def do_optimize_xy_loop(
    nv_sig, num_iterations=3, num_steps=16, scan_range=0.008, fit_method="gaussian"
):
    for i in range(num_iterations):
        if tool_belt.safe_stop():
            break

        results = optimize_xy.main(
            nv_sig,
            num_steps=num_steps,
            scan_range=scan_range,
            fit_method=fit_method,
            move_to_optimal=True,
            save_data=True,
        )

        opti_x = results.get("opti_x")
        opti_y = results.get("opti_y")
        opti_counts = results.get("opti_counts")

        if opti_x is not None and opti_y is not None:
            # Update nv_sig so next iteration re-centers on optimal position
            nv_sig.coords[CoordsKey.PIXEL] = [opti_x, opti_y]

        optimize_xy.plt.close("all")  # Close figure to prevent accumulation

        print(
            f"Iteration {i+1}/{num_iterations}: X={opti_x:.4f}, Y={opti_y:.4f}, Counts={opti_counts}"
        )


def do_green_optimize_loop(nv_sig, num_iterations=3): # Not actually moving to new position
    for i in range(num_iterations):
        if tool_belt.safe_stop():
            break

        piezo = pos.get_positioner_server(CoordsKey.Z)
        galvo = pos.get_positioner_server(CoordsKey.PIXEL)

        print(f"Starting position: Z={piezo.read_z()}, XY={galvo.read_xy()}")
        do_optimize_z(nv_sig)  # Optimize Z using piezo
        piezo.read_z()
        piezo.write_z(piezo.read_z())
        # pos.get_positioner_server(CoordsKey.Z).read_z()
        # positioner = CoordsKey.Z
        # pos._set_xyz(nv_sig)  # Update nv_sig coords with current position

        print(
            f"Z position: {piezo.read_z(), nv_sig.coords[CoordsKey.Z]}"
        )  # Write current Z back to trigger any necessary updates
        
        do_optimize_galvo(nv_sig)
        galvo.read_xy()
        galvo.write_xy(galvo.read_xy())
        print(
            f"Galvo position: {galvo.read_xy()}"
        )  # Write current XY back to trigger any necessary updates
        

        print(f"Optimized position: Z={piezo.read_z()}, XY={galvo.read_xy()}")


# def do_optimize_z_PI(nv_sig, num_steps=20, step_size=1, scan_direction="down"):  # Old placeholder


def do_optimize_galvo(nv_sig):
    # Use whatever coords key the imaging laser uses (PIXEL in cryo, AOD in widefield)
    coords_key = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
    opti_coords, final_counts = targeting.optimize(nv_sig, coords_key=coords_key)

    if getattr(nv_sig, "expected_counts", None) is None:
        nv_sig.expected_counts = final_counts

    return opti_coords


def do_optimize_z(nv_sig):
    ret_vals = targeting.optimize(nv_sig, coords_key=CoordsKey.Z)
    opti_coords = ret_vals[0]
    return opti_coords


def do_optimize_xy(nv_sig, num_steps=15, scan_range=None, fit_method="gaussian"):
    """
    Optimize XY position using a grid/raster scan pattern.

    Uses the galvo to scan a small grid around the current position,
    collects photon counts, and finds the optimal XY position using either
    2D Gaussian fitting or maximum counts.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (pulse durations, laser settings)
    num_steps : int, optional
        Number of steps per axis (creates num_steps x num_steps grid). Default: 15
    scan_range : float, optional
        Total scan range in volts. If None, uses config's optimize_range.
        Step size = scan_range / (num_steps - 1). Default: None
    fit_method : str, optional
        Method to find optimal position: "gaussian" or "max_counts". Default: "gaussian"

    Returns
    -------
    tuple
        (opti_x, opti_y) - Optimal XY coordinates in volts
    """
    results = optimize_xy.main(
        nv_sig,
        num_steps=num_steps,
        scan_range=scan_range,
        fit_method=fit_method,
        move_to_optimal=True,
        save_data=True,
    )

    opti_x = results.get("opti_x")
    opti_y = results.get("opti_y")
    opti_counts = results.get("opti_counts")

    print(f"XY optimization complete: X={opti_x:.4f}, Y={opti_y:.4f}")
    if opti_counts is not None:
        print(f"  Counts at optimal position: {opti_counts}")

    return opti_x, opti_y


def do_optimize_xy_loop(
    nv_sig, num_iterations=3, num_steps=16, scan_range=0.008, fit_method="gaussian"
):
    for i in range(num_iterations):
        if tool_belt.safe_stop():
            break

        results = optimize_xy.main(
            nv_sig,
            num_steps=num_steps,
            scan_range=scan_range,
            fit_method=fit_method,
            move_to_optimal=True,
            save_data=True,
        )

        opti_x = results.get("opti_x")
        opti_y = results.get("opti_y")
        opti_counts = results.get("opti_counts")

        if opti_x is not None and opti_y is not None:
            # Update nv_sig so next iteration re-centers on optimal position
            nv_sig.coords[CoordsKey.PIXEL] = [opti_x, opti_y]

        optimize_xy.plt.close("all")  # Close figure to prevent accumulation

        print(
            f"Iteration {i+1}/{num_iterations}: X={opti_x:.4f}, Y={opti_y:.4f}, Counts={opti_counts}"
        )


# def do_optimize_pixel(nv_sig):
#     ret_vals = targeting.optimize(nv_sig, coords_key=CoordsKey.PIXEL)
#     opti_coords = ret_vals[0]
#     return opti_coords


def do_compensate_for_drift(nv_sig):
    targeting.compensate_for_drift(nv_sig, no_crash=True)


# def do_optimize(nv_sig):
#     targeting.main(
#         nv_sig,
#         set_to_opti_coords=False,
#         save_data=True,
#         plot_data=True,
#     )


# def do_stationary_count(
#     nv_sig,
#     disable_opt=None,
# ):
#     """
#     A 1D scan which holds the galvo and piezo at a fixed position while collecting photon counts.

#     Movement can be done during this scan using cryo_position_control.py file and running in
#     a dedicated terminal.

#     """
#     run_time = 3 * 60 * 10**9  # ns

#     stationary_count.main(
#         nv_sig,
#         run_time,
#         disable_opt=disable_opt,
#         # nv_minus_initialization=nv_minus_initialization,
#         # nv_zero_initialization=nv_zero_initialization,
#     )


def do_stationary_count(nv_sig, disable_opt=None):
    """
    A 1D scan which holds the galvo and piezo at a fixed position while collecting photon counts.

    Movement can be done during this scan using cryo_position_control.py file and running in
    a dedicated terminal.

    """
    run_time = 3 * 60 * 10**9  # ns

    stationary_count.main(
        nv_sig,
        run_time,
        disable_opt=disable_opt,
        # nv_minus_initialization=nv_minus_initialization,
        # nv_zero_initialization=nv_zero_initialization,
    )

def do_stationary_count_Tisapph(
    nv_sig,
    disable_opt=None,
):
    run_time = 3 * 60 * 10**9  # ns
    stationary_count_Tisapph.main(nv_sig, run_time, disable_opt=disable_opt)

def do_calibrate_z_axis(nv_sig):
    """
    Calibrate the Z-axis to find the sample surface.

    This routine:
    1. Moves the piezo to the top of the Z range
    2. Scans downward while monitoring photon counts
    3. Finds the peak photon count position (surface)
    4. Sets that position as Z=0 reference

    Returns the calibration results dictionary.
    """
    # Go down to find approx. surface (target count=stopping point)
    # results = approach_surface.main(
    #     nv_sig,
    #     target_counts=500,  # Stop at surface counts (needs to be updated)
    #     direction="down"      # Move down toward surface
    # )
    # Continously go up for x amount of steps, stops after max steps (limited to 0.5mm, sample size)
    results = diagnose_z_direction.main(
        nv_sig,
        step_size=10,  # 10 steps at a time
        max_steps=500,  # Stop after 100k steps max
    )
    # Under construction, will combine above (and account for hysteresis)
    # results = calibrate_z_axis.main(
    #     nv_sig,
    #     scan_range=600,  # Can be overridden by config
    #     step_size=5,
    #     num_averages=100,
    #     safety_threshold=150,
    # )
    return results


# region 1D Scan
def do_z_scan_1d(nv_sig, num_steps=60, step_size=1, num_averages=1, min_threshold=1):
    """
    Perform a 1D Z-axis scan without calibration.

    Scans along Z-axis, collecting photon counts at each position.
    Does NOT move X or Y coordinates.
    Displays real-time line plot of counts vs Z position.

    Parameters
    ----------
    nv_sig : dict
        NV center parameters
    z_start : int
        Starting Z position in steps
    z_end : int
        Ending Z position in steps
    num_steps : int
        Number of Z positions to scan
    num_averages : int
        Number of photon count samples to average at each Z position

    Returns
    -------
    tuple
        (counts, z_positions) - counts in kcps or raw depending on config
    """

    results = z_scan_1d.main(
        nv_sig,
        num_steps=num_steps,
        step_size=step_size,
        num_averages=num_averages,
        min_threshold=min_threshold,
    )
    return results


def do_z_scan_3d(nv_sig):
    """
    Perform a 3D scan: 2D XY confocal images at multiple Z depths.

    At each Z position, performs a complete 2D XY confocal scan using galvo mirrors.
    Generates one image per Z slice, displayed as subplots in a single figure.

    This routine:
    1. Starts at the current Z position
    2. Moves Z relatively by z_step_size using piezo controls
    3. Performs complete 2D XY galvo scan (like do_image_sample)
    4. Generates and displays 2D image for this Z position
    5. Checks safety threshold (pauses if mean counts drop too low)
    6. Repeats for all Z steps

    Z Direction Convention (absolute positioning):
    - Negative z_step_size: moves TOWARD sample (closer)
    - Positive z_step_size: moves AWAY FROM sample (farther)

    Returns the 3D image array and Z positions.
    """
    # XY scan parameters (matching do_image_sample defaults)
    scan_range = 0.2  # XY range in volts
    num_steps = 90  # XY resolution

    # Z scan parameters
    num_z_steps = 42  # Number of Z slices
    z_step_size = 3  # Each step ~100nm RT (+/- or up/down for direction)

    # Safety and acquisition
    num_averages = 1  # Samples per pixel
    min_threshold = 0  # Pause if counts per image drops below this

    return z_scan_2d.main(
        nv_sig,
        x_range=scan_range,
        y_range=scan_range,
        num_steps=num_steps,
        num_z_steps=num_z_steps,
        z_step_size=z_step_size,
        num_averages=num_averages,
        min_threshold=min_threshold,
    )


# end region

# def do_z_scan_calibrated(nv_sig, z_start=50, z_end=-350, num_steps=61, num_averages=1):
#     """
#     Perform a 1D Z-axis scan with automatic calibration.

#     This function:
#     1. Calibrates the Z-axis to find surface (Z=0)
#     2. Performs a 1D scan along Z-axis collecting photon counts
#     3. Displays real-time plot of counts vs Z position
#     4. Saves data and plot

#     Parameters
#     ----------
#     nv_sig : dict
#         NV center parameters
#     z_start : int
#         Starting Z position in steps (positive = above surface)
#     z_end : int
#         Ending Z position in steps (negative = below surface)
#     num_steps : int
#         Number of Z positions to scan
#     num_averages : int
#         Number of photon count samples to average at each Z position

#     Returns
#     -------
#     tuple
#         (counts, z_positions) - counts in kcps or raw depending on config
#     """
#     # First calibrate to find surface
#     print("=== Starting Z-axis calibration ===")
#     cal_results = do_calibrate_z_axis(nv_sig)

#     if cal_results is None:
#         print("ERROR: Calibration failed, aborting Z scan")
#         return None, None

#     print(f"Calibration complete. Surface set at Z=0")
#     print()

#     # Now perform 1D Z scan using the dedicated routine
#     counts, z_positions = z_scan_1d.main(
#         nv_sig,
#         z_start=z_start,
#         z_end=z_end,
#         num_steps=num_steps,
#         num_averages=num_averages,
#         save_data=True,
#     )

#     return counts, z_positions

# end of construction

# def do_g2_measurement(nv_sig, apd_a_index, apd_b_index):
#     run_time = 60 * 10  # s
#     diff_window = 200  # ns

#     g2_measurement.main(nv_sig, run_time, diff_window, apd_a_index, apd_b_index)


# def do_determine_standard_readout_params(nv_sig):
#     num_reps = 1e5
#     max_readouts = [1e6]
#     filters = ["nd_0"]
#     state = States.LOW

#     determine_standard_readout_params.main(
#         nv_sig,
#         num_reps,
#         max_readouts,
#         filters=filters,
#         state=state,
#     )


# def do_pulsed_resonance(nv_sig, freq_center=2.87, freq_range=0.2):
#     num_steps = 51

#     num_reps = 2e4
#     num_runs = 16

#     # num_reps = 1e3
#     # num_runs = 8

#     uwave_power = 16.5
#     uwave_pulse_dur = 400

#     pulsed_resonance.main(
#         nv_sig,
#         freq_center,
#         freq_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         uwave_power,
#         uwave_pulse_dur,
#     )


# def do_pulsed_resonance_state(nv_sig, state):
#     freq_range = 0.020
#     num_steps = 51
#     num_reps = 2e4
#     num_runs = 16

#     # Zoom
#     # freq_range = 0.035
#     # # freq_range = 0.120
#     # num_steps = 51
#     # num_reps = 8000
#     # num_runs = 3

#     composite = False

#     res, _ = pulsed_resonance.state(
#         nv_sig,
#         state,
#         freq_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         composite,
#     )
#     nv_sig["resonance_{}".format(state.name)] = res
#     return res


# def do_scc_pulsed_resonance(nv_sig, state):
#     opti_nv_sig = nv_sig
#     freq_center = nv_sig["resonance_{}".format(state)]
#     uwave_power = nv_sig["uwave_power_{}".format(state)]
#     uwave_pulse_dur = tool_belt.get_pi_pulse_dur(nv_sig["rabi_{}".format(state)])
#     freq_range = 0.020
#     num_steps = 25
#     num_reps = int(1e3)
#     num_runs = 5

#     scc_pulsed_resonance.main(
#         nv_sig,
#         opti_nv_sig,
#         freq_center,
#         freq_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         uwave_power,
#         uwave_pulse_dur,
#     )


# def do_determine_charge_readout_params(nv_sig):
#     readout_durs = [10e6]
#     readout_durs = [int(el) for el in readout_durs]
#     max_readout_dur = max(readout_durs)

#     readout_powers = [1.0]
#     readout_powers = [round(val, 3) for val in readout_powers]

#     num_reps = 1000

#     determine_charge_readout_params.main(
#         nv_sig,
#         num_reps,
#         readout_powers,
#         max_readout_dur,
#         plot_readout_durs=readout_durs,
#     )


# def do_optimize_magnet_angle(nv_sig):
#     angle_range = [0, 150]
#     num_angle_steps = 6
#     freq_center = 2.87
#     freq_range = 0.200
#     num_freq_steps = 51
#     num_freq_runs = 15

#     # Pulsed
#     uwave_power = 16.5
#     uwave_pulse_dur = 85
#     num_freq_reps = 5000

#     # CW
#     # uwave_power = -5.0
#     # uwave_pulse_dur = None
#     # num_freq_reps = None

#     optimize_magnet_angle.main(
#         nv_sig,
#         angle_range,
#         num_angle_steps,
#         freq_center,
#         freq_range,
#         num_freq_steps,
#         num_freq_reps,
#         num_freq_runs,
#         uwave_power,
#         uwave_pulse_dur,
#     )


def do_rabi(nv_sig):
    rabi.main(
        nv_sig=nv_sig,
        num_reps=int(20e4),
        num_runs=10, #testing
        min_tau=20,  # ns
        max_tau=500,  # ns (480+min_tau)
        num_steps=40,  # 1 step every ~5-10ns
        uwave_ind=0,
        uwave_freq_ghz= 2.8320, #2.8513,  # Change to target ms=+1 or ms=-1 transition
        optimize_between_runs=True,  # Set to false to turn off optimize between runs
    )


def do_resonance(nv_sig):
    resonance.main(
        nv_sig,
        freq_center_ghz= 2.8474,#2.8333,#2.869332,
        freq_span_mhz=50.0,
        num_steps=51,
        num_reps=1,#20e4,
        num_runs=5,
        uwave_ind=0,
        optimize_between_runs=True,
    )


def do_optimize_green_readout_time(nv_sig):
    """Sweep green readout duration and pick the one that gives the best
    ODMR contrast. Set `freq_center_ghz` / `freq_span_mhz` below so the
    scan window contains exactly one peak.
    """
    optimize_green_readout_time.main(
        nv_sig,
        readout_times_ns=[550,570,590,610,630,650],
        # readout_times_ns=[400],
        freq_center_ghz= 2.8320,#2.8316,#2.8513,   # park on peak
        num_reps=int(1e6),
        num_runs=15, #per readout time
        uwave_ind=0,
        optimize_between_runs=True,
    )


def do_optimize_apd_gate_width(nv_sig):
    """Sweep APD gate width (holding green readout pulse duration and gate
    delay fixed) and pick the width that maximizes SNR per rep. Set
    laser_on_ns >= max(gate_widths_ns) + gate_delay_ns.
    """
    optimize_apd_gate_width.main(
        nv_sig,
        gate_widths_ns=[550,570,590,610,630,650],
        freq_center_ghz=2.869332,
        laser_on_ns=1000,
        gate_delay_ns=0,
        num_reps=int(20e4),
        num_runs=3,
        uwave_ind=0,
        uwave_power_dbm=10.0,
        pi_pulse_ns=146,
        optimize_between_runs=False,
    )


def do_optimize_transient(nv_sig):
    """Sweep the dark transient gap between the green polarization pulse
    and green readout pulse, and pick the value that maximizes SNR per rep.
    Too short → laser/MW leakage contaminates the measurement.
    Too long  → T1 relaxation decays the spin state before readout.
    """
    optimize_transient.main(
        nv_sig,
        transient_times_ns=[100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000],
        freq_center_ghz=2.869332,
        num_reps=int(20e4),
        num_runs=3,
        uwave_ind=0,
        uwave_power_dbm=10.0,
        pi_pulse_ns=146,
        optimize_between_runs=False,
    )


def do_tisapph_singlet_scan(nv_sig):
    resonance_tisapph_singlet_scan.main(
        nv_sig,
        wavelength_start_nm=805,
        wavelength_stop_nm=815,
        num_steps=51, #51=0.2nm steps, 105=0.1nm steps 
        num_reps=1e4,
        num_runs=3, 
        uwave_ind=0,
        uwave_power_dbm=10.0,
        probe_ns=100e3,
        do_plot=True,
        shuffle=False,
        settle_s=0.3,
        optimize_between_runs=True,
    )


def do_odmr_tisapph_short(nv_sig):
    odmr_tisapph_short.main(
        nv_sig,
        freq_center_ghz=2.87,
        freq_span_mhz=200.0,
        num_steps=30,
        num_reps=int(20e4),
        num_runs=10,
        uwave_ind=0,
        probe_ns=100e3,
        optimize_between_runs=True,
    )

def do_test_simple_spin_contrast(nv_sig):
    test_simple_spin_contrast.main(
    nv_sig,
    uwave_freq_ghz=2.8214,
    num_reps=200000,
    num_runs=10,
    uwave_ind=0,
    optimize_between_runs=True,
    do_plot=True,
)


def do_optimize_green_power(nv_sig):
    """Sweep green-laser power (and optionally readout duration) and measure
    spin-readout SNR at each point. The optimal power is where SNR per rep
    peaks; change the sweep range to find it, then fine-tune.

    Microwave settings (freq, power, pi_pulse) override the VirtualSigGens
    config for this run only; laser power is restored on exit.
    """
    optimize_green_power.main(
        nv_sig,
        # powers_mW=np.linspace(0.05, 5.0, 10),
        powers_mW=[1, 3, 5, 7, 9, 10, 15, 20, 25],
        readout_times_ns=[610],
        num_reps=int(1e6),
        num_runs=10,
        uwave_ind=0,
        uwave_freq_ghz=2.8508,
        uwave_power_dbm=10.0,
        pi_pulse_ns=107.3,
        laser_name="laser_COBO_520",
        settle_time=0.2,
        optimize_between_runs=True,
        do_plot=True,
    )


def do_optimize_spin_readout(
    nv_sig,
    # --- Which routines to run ---
    do_green_power=False,
    do_readout_time=True,
    do_apd_gate_width=False,
    do_transient=False,
    # --- Shared microwave params ---
    freq_center_ghz=2.869332,
    uwave_power_dbm=10.0,
    pi_pulse_ns=146,
    uwave_ind=0,
    # --- Shared acquisition ---
    num_reps=int(20e4),
    num_runs=3,
    optimize_between_runs=False,
    # --- Green power routine ---
    powers_mW=None,
    power_readout_times_ns=None,
    laser_name="laser_COBO_520",
    settle_time=0.2,
    # --- Readout time routine ---
    readout_times_ns=None,
    # --- APD gate width routine ---
    gate_widths_ns=None,
    laser_on_ns=1000,
    gate_delay_ns=0,
    # --- Transient routine ---
    transient_times_ns=None,
):
    """Unified entry point for the four SNR-based spin-readout optimization
    routines. Toggle each on/off with the do_* flags; shared microwave and
    acquisition parameters are set once. Routine-specific sweep ranges use
    sensible defaults when left as None.

    Routines run in order: green power -> readout time -> APD gate -> transient.
    """
    if powers_mW is None:
        powers_mW = np.linspace(0.1, 5.0, 10)
    if power_readout_times_ns is None:
        power_readout_times_ns = [300, 500, 1000]
    if readout_times_ns is None:
        readout_times_ns = [200, 250, 300, 350, 400, 450, 500, 550]
    if gate_widths_ns is None:
        gate_widths_ns = [50, 100, 150, 200, 300, 400, 500, 700, 1000]
    if transient_times_ns is None:
        transient_times_ns = [100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000]

    selected = any([do_green_power, do_readout_time, do_apd_gate_width, do_transient])
    if not selected:
        print("No optimization routines selected. Set at least one do_* flag to True.")
        return

    # 1. Green power
    if do_green_power:
        print("\n" + "#" * 72)
        print("# OPTIMIZE GREEN POWER")
        print("#" * 72)
        optimize_green_power.main(
            nv_sig,
            powers_mW=powers_mW,
            readout_times_ns=power_readout_times_ns,
            num_reps=num_reps,
            uwave_ind=uwave_ind,
            uwave_freq_ghz=freq_center_ghz,
            uwave_power_dbm=uwave_power_dbm,
            pi_pulse_ns=pi_pulse_ns,
            laser_name=laser_name,
            settle_time=settle_time,
            do_plot=True,
        )

    # 2. Readout time
    if do_readout_time:
        print("\n" + "#" * 72)
        print("# OPTIMIZE GREEN READOUT TIME")
        print("#" * 72)
        optimize_green_readout_time.main(
            nv_sig,
            readout_times_ns=readout_times_ns,
            freq_center_ghz=freq_center_ghz,
            num_reps=num_reps,
            num_runs=num_runs,
            uwave_ind=uwave_ind,
            uwave_power_dbm=uwave_power_dbm,
            pi_pulse_ns=pi_pulse_ns,
            optimize_between_runs=optimize_between_runs,
        )

    # 3. APD gate width
    if do_apd_gate_width:
        print("\n" + "#" * 72)
        print("# OPTIMIZE APD GATE WIDTH")
        print("#" * 72)
        optimize_apd_gate_width.main(
            nv_sig,
            gate_widths_ns=gate_widths_ns,
            freq_center_ghz=freq_center_ghz,
            laser_on_ns=laser_on_ns,
            gate_delay_ns=gate_delay_ns,
            num_reps=num_reps,
            num_runs=num_runs,
            uwave_ind=uwave_ind,
            uwave_power_dbm=uwave_power_dbm,
            pi_pulse_ns=pi_pulse_ns,
            optimize_between_runs=optimize_between_runs,
        )

    # 4. Transient dark gap
    if do_transient:
        print("\n" + "#" * 72)
        print("# OPTIMIZE TRANSIENT DARK GAP")
        print("#" * 72)
        optimize_transient.main(
            nv_sig,
            transient_times_ns=transient_times_ns,
            freq_center_ghz=freq_center_ghz,
            num_reps=num_reps,
            num_runs=num_runs,
            uwave_ind=uwave_ind,
            uwave_power_dbm=uwave_power_dbm,
            pi_pulse_ns=pi_pulse_ns,
            optimize_between_runs=optimize_between_runs,
        )

    print("\n" + "#" * 72)
    print("# SPIN READOUT OPTIMIZATION COMPLETE")
    print("#" * 72)


#  def do_determine_standard_readout_params(nv_sig)
    


# def do_t1_dq(nv_sig):
#     # T1 experiment parameters, formatted:
#     # [[init state, read state], relaxation_time_range, num_steps, num_reps]
#     num_runs = 500
#     num_reps = 1000
#     num_steps = 12
#     min_tau = 10e3
#     max_tau_omega = int(18e6)
#     max_tau_gamma = int(8.5e6)
#     # fmt: off
#     t1_exp_array = np.array(
#         [[[States.ZERO, States.HIGH], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
#         [[States.ZERO, States.ZERO], [min_tau, max_tau_omega], num_steps, num_reps, num_runs],
#         [[States.ZERO, States.HIGH], [min_tau, max_tau_omega // 3], num_steps, num_reps, num_runs],
#         [[States.ZERO, States.ZERO], [min_tau, max_tau_omega // 3], num_steps, num_reps, num_runs],
#         [[States.LOW, States.HIGH], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
#         [[States.LOW, States.LOW], [min_tau, max_tau_gamma], num_steps, num_reps, num_runs],
#         [[States.LOW, States.HIGH], [min_tau, max_tau_gamma // 3], num_steps, num_reps, num_runs],
#         [[States.LOW, States.LOW], [min_tau, max_tau_gamma // 3], num_steps, num_reps, num_runs]],
#         dtype=object,
#     )
#     # fmt: on

#     t1_dq_main.main(nv_sig, t1_exp_array, num_runs)


# def do_ramsey(nv_sig):
#     detuning = 2.5  # MHz
#     precession_time_range = [0, 4 * 10**3]
#     num_steps = 151
#     num_reps = 3 * 10**5
#     num_runs = 1

#     ramsey.main(
#         nv_sig,
#         detuning,
#         precession_time_range,
#         num_steps,
#         num_reps,
#         num_runs,
#     )


# def do_spin_echo(nv_sig):
#     # T2* in nanodiamond NVs is just a couple us at 300 K
#     # In bulk it"s more like 100 us at 300 K
#     max_time = 120  # us
#     num_steps = max_time  # 1 point per us
#     precession_time_range = [1e3, max_time * 10**3]
#     num_reps = 4e3
#     num_runs = 20

#     state = States.LOW

#     angle = spin_echo.main(
#         nv_sig,
#         precession_time_range,
#         num_steps,
#         num_reps,
#         num_runs,
#         state,
#     )
#     return angle

def do_pulse_gen_constant(digital_channels=(3,), analog0=None, analog1=None):
    pulse_gen = tool_belt.get_server_pulse_streamer()
    # Build args for the LabRAD setting
    digital_channels = [int(ch) for ch in digital_channels]

    analog_channels = []
    analog_voltages = []
    if analog0 is not None:
        analog_channels.append(0)
        analog_voltages.append(float(analog0))
    if analog1 is not None:
        analog_channels.append(1)
        analog_voltages.append(float(analog1))

    # Turn on constant outputs
    pulse_gen.constant(digital_channels, analog_channels, analog_voltages)

    try:
        input("Constant state applied. Press Enter to stop...")
    finally:
        # Safest cleanup: forces final + sets everything off
        pulse_gen.reset()

def do_pulse_gen_square_wave(period, digital_channels=(3,), analog0=None, analog1=None):
    pulse_gen = tool_belt.get_server_pulse_streamer()

    # Digital channels
    digital_channels = [int(ch) for ch in digital_channels]

    # Analog channels
    analog_channels = []
    analog_voltages = []
    if analog0 is not None:
        analog_channels.append(0)
        analog_voltages.append(float(analog0))
    if analog1 is not None:
        analog_channels.append(1)
        analog_voltages.append(float(analog1))

    # Start square wave
    pulse_gen.square_wave(digital_channels, analog_channels, analog_voltages, period)

    try:
        input("Square wave running. Press Enter to stop...")
    finally:
        pulse_gen.reset()
        
def piezo_pest():
    cxn = labrad.connect()
    s = cxn.pos_z_PI_pifoc
    voltages_to_write = np.linspace(1, 4, 5)
    for v in voltages_to_write:
        s.write_z(v)
        time.sleep(2)


def get_sample_name() -> str:
    sample = "Wu"  # lovelace
    return sample


def do_constant_ac(digital_channels=(4,), analog0=None, analog1=None):
    cxn = common.labrad_connect()
    sig_gen = cxn.sig_gen_STAN_sg394_3
    pulse_gen = tool_belt.get_server_pulse_streamer()
    # Build args for the LabRAD setting
    digital_channels = [int(ch) for ch in digital_channels]

    analog_channels = []
    analog_voltages = []
    if analog0 is not None:
        analog_channels.append(0)
        analog_voltages.append(float(analog0))
    if analog1 is not None:
        analog_channels.append(1)
        analog_voltages.append(float(analog1))

    # Microwave test
    amp = 5
    sig_gen.set_amp(amp)  # 12
    sig_gen.set_freq(2.87)  # Ghz
    sig_gen.uwave_on()
    # Turn on constant outputs
    pulse_gen.constant(digital_channels, analog_channels, analog_voltages)
    try:
        input("Constant state applied. Press Enter to stop...")
    finally:
        # Safest cleanup: forces final + sets everything off
        pulse_gen.reset()
    # input("Press enter to stop...")
    # pulse_gen.reset()


def do_tisapph_constant_wavelength(wavelength_nm=780.0):
    cxn = common.labrad_connect()
    tisapph = cxn.tisapph_m2_solstis

    try:
        current_wavelength = tisapph.get_wavelength_nm()
        print(f"Current wavelength: {current_wavelength:.6f} nm")

        print(f"Setting wavelength to {wavelength_nm:.6f} nm...")
        tisapph.set_wavelength_nm(wavelength_nm)

        time.sleep(1.0)

        new_wavelength = tisapph.get_wavelength_nm()
        print(f"Updated wavelength: {new_wavelength:.6f} nm")

        input("Ti:Sapph wavelength set. Press Enter to finish...")

    finally:
        # No hard reset here unless you really want it
        pass

def do_find_apd_gate_overlap(nv_sig):
    """Sweep APD gate delay; find where counts match nv_sig.expected_counts."""
    TOLERANCE = 0.10       # +/- band around expected_counts
    ALLOW_OVERLAP = True  # True to probe negative delays (APD inside laser pulse)

    # Stay just inside the pre-laser dark region: gate_delay < -(laser_on + gate_width)
    # is entirely before the laser turns on and contributes no signal.
    delay_min_ns = -(LASER_ON_NS + 100) if ALLOW_OVERLAP else 0
    num_steps = 45 if ALLOW_OVERLAP else 21

    find_apd_gate_overlap.main(
        nv_sig,
        num_reps=int(2e5),
        num_runs=3,
        delay_min_ns=delay_min_ns,
        delay_max_ns=500,
        num_steps=num_steps,
        laser_on_ns=LASER_ON_NS,
        gate_width_ns=300,
        laser_vkey=VirtualLaserKey.SPIN_READOUT,
        tolerance=TOLERANCE,
        allow_overlap=ALLOW_OVERLAP,
    )


if __name__ == "__main__":
    ### Shared parameters

    green_laser = "laser_COBO_520"
    # yellow_laser = "laserglow_589"
    # red_laser = "cobolt_638"

    # fmt: off
    # lovelace"
    # nv_sig = {
    #     "coords": [0.240, -0.426, 1], "name": "{}-nv8_2022_11_14".format(sample_name),
    #     "disable_opt": False, "disable_z_opt": True, "expected_count_rate": 13,

    #     "imaging_laser": green_laser, "imaging_laser_filter": "nd_0", "imaging_readout_dur": 1e7,
    #     "spin_laser": green_laser, "spin_laser_filter": "nd_0", "spin_pol_dur": 2e3, "spin_readout_dur": 440,

    #     "nv-_reionization_laser": green_laser, "nv-_reionization_dur": 1e6, "nv-_reionization_laser_filter": "nd_1.0",
    #     "nv-_prep_laser": green_laser, "nv-_prep_laser_dur": 1e6, "nv-_prep_laser_filter": "nd_0",
    #     "nv0_ionization_laser": red_laser, "nv0_ionization_dur": 75, "nv0_prep_laser": red_laser, "nv0_prep_laser_dur": 75,
    #     "spin_shelf_laser": yellow_laser, "spin_shelf_dur": 0, "spin_shelf_laser_power": 1.0,
    #     "initialize_laser": green_laser, "initialize_dur": 1e4,
    #     "charge_readout_laser": yellow_laser, "charge_readout_dur": 100e6, "charge_readout_laser_power": 1.0,

    #     "collection_filter": None, "magnet_angle": None,
    #     "resonance_LOW": 2.878, "rabi_LOW": 400, "uwave_power_LOW": 16.5,
    #     "resonance_HIGH": 2.882, "rabi_HIGH": 400, "uwave_power_HIGH": 16.5,
    #     }
    # fmt: on
    #region Position
    # coords: SAMPLE (piezo) xyz
    # current step rate: 30.0V XY
    # current step rate: 40.0V Z (atto)
    sample_xy = [0, 0]  # piezo XY voltage input (1.0=1V) (coordinates)
    coord_z = 5.673 #4.828+1.25 #6.4988 #5.5471  # atto=rel (set to 0 between measurements) PI=absolute, start at 4.00V for lovelace, minimum step size = 0.005
    # coord_z = 3.4318
    # pixel_xy = [-0.026, 0.036]  # Old Wu NV 4/14
    pixel_xy = [-0.324,0.28]  # candidate 1 z=5.673,ms=2.513,
    # pixel_xy = [-0.14, 0.164]  # candidate 2 z=6

    #region Params
    # return
    nv_sig = NVSig(
        name=f"({get_sample_name()})",
        coords={
            CoordsKey.SAMPLE: sample_xy,
            CoordsKey.Z: coord_z,
            CoordsKey.PIXEL: pixel_xy,  # galvo
        },
        disable_opt=False,
        disable_z_opt=True,
        # expected_counts=13,
        pulse_durations={
            VirtualLaserKey.IMAGING: int(10e6),  # readout is in ns (5e6 = 5ms)
            VirtualLaserKey.SPIN_READOUT: int(610), #Pulsed: int(610) #CW=int(10e6),10ms # readout is in ns (5e6 = 5ms)
            VirtualLaserKey.SPIN_POL: 2000,
            VirtualLaserKey.SINGLET_DRIVE: 100e3,  # placeholder
        },
    )
    # nv_sig.expected_counts = None
    nv_sig.expected_counts = 86.2

    # cxn = labrad.connect()
    # s = cxn.pos_z_PI_pifocss
    # print(sorted(s.settings.keys()))
    # sys.exit()
    # endregion
    ### Routines to execute

    try:
        tool_belt.init_safe_stop()
        # pos.set_drift([0.0, 0.0, 0.0])  # Reset drift to clean state
        # drift = tool_belt.get_drift()
        # tool_belt.set_drift([0.0, 0.0, drift[2]])  # Keep z
        # tool_belt.set_drifts([drift[0], drift[1], 0.0])  # Keep xy

        # print("PIXEL coords going to galvo:", nv_sig.coords[CoordsKey.PIXEL])
        # print("SAMPLE coords going to piezo:", nv_sig.coords[CoordsKey.SAMPLE])
        # pos.set_xyz_on_nv(nv_sig)  # Leave this line out when calibrating z
        pos.set_xyz_on_nv(nv_sig)  # Leave this line out when calibrating z

        # region Pulse Gen
        # do_pulse_gen_constant()
        # do_pulse_gen_constant(digital_channels=(2,))
        # do_pulse_gen_constant(digital_channels=(3,))
        # do_tisapph_constant_wavelength(wavelength_nm=780.0)
        # do_pulse_gen_square_wave(10000, digital_channels=(3,))
        # do_constant_ac()
        # do_pulse_gen_constant(digital_channels=(4,), analog0=None, analog1=None)

        # # # Manually set Z reference to current position
        # piezo = pos.get_positioner_server(CoordsKey.Z)
        # # print(piezo.get_z_position())
        # piezo.set_z_reference()

        # region 1D scan + Calibrate
        # do_calibrate_z_axis(nv_sig)
        # do_z_scan_1d(nv_sig)
        # endregion 1D scan + Calibrate
        # do_image_sample_zoom(nv_sig)
        # region 2D scan (x galvo, z piezo)
        # # do_2D_xz_scan(nv_sig)
        # z_range = np.linspace(0, 3, 31)
        # for z in z_range:
        #     nv_sig.coords[CoordsKey.Z] = z
        #     pos.set_xyz_on_nv(nv_sig)
        #     # do_image_sample_zoom(nv_sig)
        # do_image_sample(nv_sig)
        # do_2D_xz_scan(nv_sig)

        # endregion 2D scan

        # region Image / 3D scan
    
        # do_z_scan_3d(nv_sig) # (xy gavo, z piezo)
        # do_image_sample_zoom(nv_sig)
        # do_image_sample(nv_sig)
        # do_image_sample(nv_sig, nv_minus_initialization=True)
        # do_image_sample_zoom(nv_sig, nv_minus_initialization=True)
        # end region Image sample
        #
        # region Optimize
        # do_optimize_z_PI(nv_sig, voltage_start=5.6, voltage_end=6.3, step_size=0.02) #must be between 1-9V
        # do_optimize_z_atto(nv_sig) # z position optimize atto
        # do_optimize_xy(nv_sig, num_steps=8, scan_range=0.008) #xy galvo optimize but it works :)
        # do_optimize_xy_loop(nv_sig, num_iterations=3, num_steps=16, scan_range=0.008)

        # do_compensate_for_drift(nv_sig)
        # do_optimize_galvo(nv_sig) # optimize xy for drift
        # do_optimize_z(nv_sig) # optimize z for drift
        # do_green_optimize_loop(nv_sig, num_iterations=3)  # Optimize before resonance scans to ensure we're on target

        #Optimize seq. parameters 
        # do_optimize_green_power(nv_sig)
        do_optimize_green_readout_time(nv_sig)
        # do_find_apd_gate_overlap(nv_sig)
        # endregion Optimize

        # region Stationary count
        # do_stationary_count(nv_sig, disable_opt=True) #Note there is a slow response time w/ the APD
        # do_stationary_count_Tisapph(nv_sig, disable_opt=True)
        # do_stationary_count(nv_sig, disable_opt=True, nv_minus_initialization=True)
        # do_stationary_count(nv_sig, disable_opt=True, nv_zero_initialization=True)
        # endregion Stationary count

        # region Resonance, Pulse Seq., Singlet
        # do_tisapph_singlet_scan(nv_sig)

        # probe_ns = [2e3, 5e3, 10e3, 20e3, 50e3, 100e3]
        # for probe in probe_ns:
            # do_tisapph_singlet_scan(nv_sig, probe_ns=probe)

        # do_resonance(nv_sig)
        # do_rabi(nv_sig)

  # do_rabi(nv_sig)
        # try:
        #     tool_belt.init_safe_stop()
        #     pos.set_xyz_on_nv(nv_sig)

            # do_rabi(nv_sig)

        # except Exception as exc:
        #     tool_belt.traceback.print_exc()
        #     raise exc
        # finally:
        #     tool_belt.reset_cfm()
        #     tool_belt.reset_safe_stop()
        #     kpl.show(block=True)
        
        
        # for i in range(3):
        # do_resonance(nv_sig)
        #     do_green_optimize_loop(nv_sig, num_iterations=1)
        #     print(f"Completed resonance scan {i+1}/3, optimizing Z and galvo before next scan")
        #     do_green_optimize_loop(nv_sig, num_iterations=2)  # Optimsize after each resonance scan to keep on target

        # do_resonance_state(nv_sig , States.LOW)
        # do_resonance_state(nv_sig, States.HIGH)
        # do_pulsed_resonance(nv_sig, 2.87, 0.200)
        # do_pulsed_re2.sonance_state(nv_sig, States.LOW)
        # do_pulsed_resonance_state(nv_sig, States.HIGH)
        
        # do_tisapph_singlet_scan(nv_sig)
        # do_test_simple_spin_contrast(nv_sig)

        # probe_ns = [2e3, 5e3, 10e3, 20e3, 50e3, 100e3]
        # for probe in probe_ns:
            # do_tisapph_singlet_scan(nv_sig, probe_ns=probe)
          
        # do_rabi(nv_sig, uwave_time_range=[0, 400])
        # do_spin_echo(nv_sig)
        # do_g2_measurement(nv_sig, 0, 1)
        # do_determine_standard_readout_params(nv_sig)
        
        # region SCC
        # do_determine_charge_readout_params(nv_sig,nbins=200,nreps=100)
        # do_scc_pulsed_resonance(nv_sig)


    ### Error handling and wrap-up
    except Exception as exc:
        recipient = "cmreiter@berkeley.edu"
        # tool_belt.send_exception_email(email_to=recipient)
        tool_belt.traceback.print_exc()
        raise exc
    finally:
        tool_belt.reset_cfm()
        tool_belt.reset_safe_stop()
        kpl.show(block=True)

# endregion
