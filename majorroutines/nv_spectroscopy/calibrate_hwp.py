# -*- coding: utf-8 -*-
"""
Search for NV triplet-to-singlet wavelength

Created on August 9th, 2025

@author: jchen-1
"""

import os
import sys
import time
from pathlib import Path
from random import shuffle

import clr
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from System import *
from System.Collections.Generic import List
from System.Runtime.InteropServices import GCHandle, GCHandleType, Marshal

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb


def sine_func(theta_deg, phase_shift_deg, amplitude, vertical_displacement):
    return (
        amplitude * (np.cos(2 * (theta_deg - phase_shift_deg) * np.pi / 180)) ** 2
        + vertical_displacement
    )


def fit_calibration_curve(angles_deg, powers):
    lower_bounds = [-180, 0, -np.inf]
    upper_bounds = [180, np.inf, np.inf]
    bounds = (lower_bounds, upper_bounds)

    popt, pcov = curve_fit(sine_func, xdata=angles_deg, ydata=powers, bounds=bounds)
    # perr = np.sqrt(np.diag(pcov))
    # phase_shift_fit, amp_fit, vertical_displace_fit = popt
    return popt


def calibrate_hwp(
    wavelength: int = None,
):
    print("hi")
    # Initialize the devices
    slider = tb.get_server_slider_2()
    power_meter = tb.get_server_power_meter()
    rotator = tb.get_server_rotation_mount()
    tisapph = tb.get_tisapph()
    print("Initialized devices")

    if wavelength is not None:
        tisapph.set_wavelength_nm(wavelength)

    actual_wavelength = tisapph.get_wavelength_nm()

    # Set power meter to correct wavelength setting
    current_wavelength = tisapph.get_wavelength_nm()
    power_meter.set_wavelength(int(current_wavelength))
    print(f"Set power meter to {current_wavelength:.0f}")

    # Set up data taking folder
    timestamp = dm.get_time_stamp()

    # Sweep HWP angles
    angle_start, angle_end, da = 0, 120, 5
    angles_deg = np.arange(angle_start, angle_end, da)

    # Data Folder
    measured_powers = np.zeros_like(angles_deg, dtype=np.float64)

    # Move slider to deflect light to power meter
    print("Have not set slider yet")
    slider.set_filter(3)
    print("Set slider to mirror position")

    for i in range(len(angles_deg)):
        # Set HWP to angle
        angle = angles_deg[i]
        rotator.set_angle(angle)
        time.sleep(0.4)

        # Take power measurement
        measured_power = power_meter.get_power()
        # measured_power = (
        #     50 * (np.cos(2 * (angle - 10) * np.pi / 180)) ** 2
        # ) + np.random.normal(scale=3)  # Mimic what's happening
        measured_powers[i] = measured_power

    # Move slider back to throughput
    slider.set_filter(2)

    # Save data
    raw_data = {
        "timestamp": timestamp,
        "angle_start": angle_start,
        "angle_end": angle_end,
        "angle_div": da,
        "tisapph_wavelength": float(current_wavelength),
        "powers": measured_powers,
    }
    repr_nv_name = f"Calibration-{actual_wavelength:.1f}nm"
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    # Fit data
    fit_param = fit_calibration_curve(angles_deg, measured_powers)
    phase, amp, yint = fit_param

    # Plot data
    fig, ax = plt.subplots()
    ax.plot(
        angles_deg,
        sine_func(angles_deg, *fit_param),
        label=f"Fit\n{amp:.3f}(cos^2(2(θ - {phase:.2f}))) + {yint:.3e}",
    )
    ax.scatter(angles_deg, measured_powers, label="Data")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"HWP Calibration, {int(current_wavelength)}nm")
    ax.legend()
    plt.show()

    raw_data |= {
        "phase_shift_fit": fit_param[0],
        "amplitude_fit": fit_param[1],
        "vertical_displace_fit": fit_param[2],
    }

    # Save data with fit again
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    return file_path


wavelengths = [696]
# wavelengths = [696, 698, 700, 702.5, 705, 707.5]
# wavelengths = [710, 712.5, 715, 720, 725]
# wavelengths = [730, 735, 740, 745, 750, 760, 770, 780, 790, 800]

file_paths = []

for i in range(len(wavelengths)):
    file_paths.append(calibrate_hwp(wavelengths[i]))

for f in file_paths:
    print(f)


# calibrate_hwp(698)
# calibrate_hwp(700)
# calibrate_hwp(702.5)
# calibrate_hwp(705)
# calibrate_hwp(707.5)
# calibrate_hwp(710)
# calibrate_hwp(712.5)
# calibrate_hwp(715)


# calibrate_hwp(710)
# calibrate_hwp(707.5)
# calibrate_hwp(712.5)
# calibrate_hwp(705)
# calibrate_hwp(715)
# calibrate_hwp(720)
# calibrate_hwp(725)
# calibrate_hwp(730)
# calibrate_hwp(735)
# calibrate_hwp(745)
# calibrate_hwp(755)
# calibrate_hwp(765)
# calibrate_hwp(775)
# calibrate_hwp(795)
# calibrate_hwp(805)
# calibrate_hwp(790)
# calibrate_hwp(820)
# calibrate_hwp(850)
