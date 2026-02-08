# -*- coding: utf-8 -*-
"""
Search for NV triplet-to-singlet wavelength

Created on November 23rd, 2025

@author: Alyssa Matthews

Map out power range for TiSapph wavelength

take two msmts at two angles, then fit to the sine fxn and extract

"""

import os
import sys
import time
from pathlib import Path
from random import shuffle

import clr
import numpy as np
from matplotlib import pyplot as plt
from System import *
from System.Collections.Generic import List
from System.Runtime.InteropServices import GCHandle, GCHandleType, Marshal

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb

# --- EXACT HELPER FUNCTIONS AS REQUESTED ---


def sine_func(theta_deg, phase_shift_deg, amplitude, vertical_displacement):
    return (
        amplitude * (np.cos(2 * (theta_deg - phase_shift_deg) * np.pi / 180)) ** 2
        + vertical_displacement
    )


def find_hwp_fit_params(
    slider_2,
    power_meter,
    rotator,
    calibration_filename: str,
):
    calibration_dict = dm.get_raw_data(calibration_filename)
    phase_shift_deg = calibration_dict["phase_shift_fit"]

    # Function: Acos^2(2(theta - shift)) + B

    # Move slider to correct location
    slider_2.set_filter(3)  # Mirror
    time.sleep(0.2)

    # Measurement 1
    angle_1 = 25
    rotator.set_angle(angle_1)
    time.sleep(0.3)
    power_1 = float(power_meter.get_power())

    # Measurement 2
    angle_2 = 40
    rotator.set_angle(angle_2)
    time.sleep(1)
    power_2 = float(power_meter.get_power())

    # Solve for A and B
    A = (power_1 - power_2) / (
        (np.cos(2 * (angle_1 - phase_shift_deg) * np.pi / 180)) ** 2
        - (np.cos(2 * (angle_2 - phase_shift_deg) * np.pi / 180)) ** 2
    )
    B = power_1 - A * (np.cos(2 * (angle_1 - phase_shift_deg) * np.pi / 180)) ** 2

    print(f"Power 1 = {power_1}, Power 2 = {power_2}")
    print(f"A = {A}, B = {B}")
    fit_param = [phase_shift_deg, A, B]

    slider_2.set_filter(2)
    return fit_param


def get_angle_from_params(target_power, params):
    phase, amp, vert = params
    print(f"Phase = {phase}, Amp = {amp}, Vert = {vert}")
    hwp_angle = (
        0.5 * np.arccos(np.sqrt((target_power - vert) / amp)) * 180 / np.pi + phase
    )
    print(f"HWP angle = {hwp_angle: .3f}")
    return hwp_angle


def get_max_power_prediction(params):
    phase, amp, vert = params
    # Max power occurs when cos^2 term is 1
    max_power = amp + vert
    print(f"Max Theoretical Power: {max_power:.4f}")
    return max_power


# --- MAIN EXECUTION SCRIPT ---


def main(
    wavelength_list: list,
    filename_list: list,
    pump_power_note,
):
    # 0. INPUT VALIDATION
    if len(wavelength_list) != len(filename_list):
        print("ERROR: The number of wavelengths and filenames must match!")
        return

    # Pair them up and sort by wavelength (safest for laser tuning)
    scan_pairs = sorted(zip(wavelength_list, filename_list))

    # 1. INITIALIZATION
    print("Initializing devices...")
    slider_2 = tb.get_server_slider_2()
    power_meter = tb.get_server_power_meter()
    rotator = tb.get_server_rotation_mount()
    tisapph = tb.get_tisapph()

    # Lists to store our results
    max_powers_list = []
    fit_params_library = {}  # {wavelength: [phase, A, B]}
    wavelengths = []

    print(f"\n--- TiSapph Bottleneck Scan ---")
    print(f"Scanning {len(scan_pairs)} specific wavelengths.")
    print("This script uses pre-existing HWP calibration files to calculate max power.")

    # Safety Check
    print("\n" + "!" * 40)
    print("SAFETY CHECK:")
    print("1. Ensure Pump Laser is ON (Nominal Power).")
    print("2. Ensure High Power Meter Head is installed.")
    print("!" * 40)
    input("Press Enter to begin sweep >> ")

    timestamp = dm.get_time_stamp()

    try:
        # 2. WAVELENGTH LOOP
        for i, (wl, calib_filename) in enumerate(scan_pairs):
            print(
                f"\n[{i + 1}/{len(scan_pairs)}] Testing {wl} nm (File: {calib_filename})"
            )

            # B. Move Laser
            print(f"  -> Tuning TiSapph to {wl} nm...")
            tisapph.set_wavelength_nm(wl)
            power_meter.set_wavelength(wl)

            # Wait for lock
            time.sleep(3.0)

            # C. Run the 2-Point Fit Routine
            try:
                params = find_hwp_fit_params(
                    slider_2, power_meter, rotator, calib_filename
                )

                # --- PREDICT MAX POWER ---
                max_p = get_max_power_prediction(params)

                # Store data
                max_powers_list.append(max_p)
                fit_params_library[wl] = params
                wavelengths.append(wl)

                print(f"  -> Result: Max Power = {max_p:.3f} W")

            except Exception as e:
                print(f"  ! Error processing {wl}nm: {e}")
                # We continue to the next wavelength even if one fails
                continue

    except KeyboardInterrupt:
        print("\nSweep interrupted by user.")

    finally:
        print("\nMoving slider to throughput...")
        slider_2.set_filter(2)

    # 3. ANALYZE BOTTLENECK
    if not max_powers_list:
        print("No valid data collected.")
        return

    max_powers_arr = np.array(max_powers_list)

    # The Bottleneck is the lowest maximum found across the spectrum
    bottleneck_power = np.min(max_powers_arr)
    bottleneck_idx = np.argmin(max_powers_arr)
    bottleneck_wl = wavelengths[bottleneck_idx]

    highest_peak = np.max(max_powers_arr)

    print("\n" + "=" * 40)
    print(f"SCAN RESULTS")
    print(f"Peak Power in Range:    {highest_peak:.4f} W")
    print(f"BOTTLENECK (Low Limit): {bottleneck_power:.4f} W")
    print(f"Bottleneck occurs at:   {bottleneck_wl} nm")
    print(f"--> To run a flat spectrum, set power <= {bottleneck_power:.4f} W")
    print("=" * 40)

    # 4. SAVE DATA
    raw_data = {
        "timestamp": timestamp,
        "pump_power_note": pump_power_note,
        "wavelengths": wavelengths,
        "max_powers": max_powers_list,
        "fit_params_library": fit_params_library,
        "bottleneck_power": bottleneck_power,
        "bottleneck_wavelength": bottleneck_wl,
    }

    # Generate filename based on range found in map
    start_wl = min(wavelengths)
    end_wl = max(wavelengths)
    file_name = f"TiSapph_BottleneckScan_{int(start_wl)}-{int(end_wl)}nm"

    file_path = dm.get_file_path(__file__, timestamp, file_name)
    dm.save_raw_data(raw_data, file_path)
    print(f"Data saved to: {file_path}")

    # 5. PLOT
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        wavelengths,
        max_powers_list,
        "o",
        color="blue",
        markersize=8,
        label="Max Possible Power",
    )

    # The bottleneck line
    ax.axhline(
        y=bottleneck_power,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Bottleneck ({bottleneck_power:.3f} W)",
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"Ti:Sapph Max Power Envelope (Pump: {pump_power_note})")

    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # List 1: Wavelengths
    my_wavelengths = [730, 760, 790, 819, 775, 805]

    # List 2: The exact filename corresponding to each wavelength above
    # (Order must match List 1 exactly)
    my_files = [
        "2025_11_21-15_58_31-Calibration-730nm"
        "2025_11_21-15_58_52-Calibration-760nm"
        "2025_11_21-15_59_13-Calibration-790nm"
        "2025_11_21-15_59_34-Calibration-819nm"
        "2025_11_21-16_34_44-Calibration-775nm"
        "2025_11_21-16_35_10-Calibration-805nm"
    ]

    main(
        wavelength_list=my_wavelengths, filename_list=my_files, pump_power_note="18.0W"
    )
