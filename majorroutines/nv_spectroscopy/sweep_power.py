# -*- coding: utf-8 -*-
"""
Search for NV triplet-to-singlet wavelength

Created on August 9th, 2025

@author: jchen-1, alyssa matthews
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

# Add needed dll references
sys.path.append(os.environ["LIGHTFIELD_ROOT"])
sys.path.append(os.environ["LIGHTFIELD_ROOT"] + "\\AddInViews")
clr.AddReference("PrincetonInstruments.LightFieldViewV5")
clr.AddReference("PrincetonInstruments.LightField.AutomationV5")
clr.AddReference("PrincetonInstruments.LightFieldAddInSupportServices")

# PI imports
import traceback

from PrincetonInstruments.LightField.AddIns import *
from PrincetonInstruments.LightField.Automation import *
from tqdm import tqdm

from utils import positioning as pos

"""
HWP Calibration: Power versus angle of Half wave plate
    a) Deflect light towards the power meter
    b) Set the power meter to the correct wavelength setting
    c) Set the half wave plate to a specific angle
    d) Take reading from power meter
    e) Repeat c) and d) until sufficient points are taken

Power sweep
1. Place filter translators in correct configuration for wavelength
2. Set the Tisapph at the wavelength
3. Set Half wave plate to angles which minimize and maximize the power, and measure
4. Define a function which outputs HWP angle versus desired power using HWP calibration
5. Move mirror back and let light through to NV sample
6. Take spectra
    a) Set half wave plate at desired angle based on fit
    b) Move mirror into light path; Deflect light to power meter and measure
    c) Move mirror back
    d) Take spectra on the spectrometer and save
    e) Repeat steps a)-d)


"""


# Helper functions for spectrometer
def validate_camera(_experiment):
    camera = None

    # Find connected device
    for device in _experiment.ExperimentDevices:
        if device.Type == DeviceType.Camera and _experiment.IsReadyToRun:
            camera = device

    if camera == None:
        print("This sample requires a camera.")
        return False

    if not _experiment.IsReadyToRun:
        print("The system is not ready for acquisition, is there an error?")

    return True


# Creates a numpy array from our acquired buffer
def convert_buffer(net_array, image_format):
    src_hndl = GCHandle.Alloc(net_array, GCHandleType.Pinned)
    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()

        # Possible data types returned from acquisition
        if image_format == ImageDataFormat.MonochromeUnsigned16:
            buf_type = ctypes.c_ushort * len(net_array)
        elif image_format == ImageDataFormat.MonochromeUnsigned32:
            buf_type = ctypes.c_uint * len(net_array)
        elif image_format == ImageDataFormat.MonochromeFloating32:
            buf_type = ctypes.c_float * len(net_array)

        cbuf = buf_type.from_address(src_ptr)
        resultArray = np.frombuffer(cbuf, dtype=cbuf._type_)

    # Free the handle
    finally:
        if src_hndl.IsAllocated:
            src_hndl.Free()

    # Make a copy of the buffer
    return np.copy(resultArray)


# Helper functions to calibrate the half wave plate
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


def main(
    calibration_files,
    wavelengths,
    slider_1_position,
    slider_3_position,
    min_power,
    max_power,
    num_points,
    start_wavelength,
    end_wavelength,
    RF_on: bool = False,
    spect_wavelengths=r"G:\nvdata\pc_Nuclear\branch_master\sweep_power\2025_12\800nm_center_wavelengths.npy",
):
    wavelengths_data = np.load(spect_wavelengths)
    start_wavelength_i = np.argmin(np.abs(wavelengths_data - start_wavelength))
    end_wavelength_i = np.argmin(np.abs(wavelengths_data - end_wavelength))

    # Initialize all the devices
    slider_1 = tb.get_server_slider_1()
    slider_2 = tb.get_server_slider_2()
    slider_3 = tb.get_server_slider_3()

    # Set filter translators (protect the spectrometr)
    slider_1.set_filter(slider_1_position)
    slider_3.set_filter(slider_3_position)

    # return

    tisapph = tb.get_tisapph()
    hwp_rotator = tb.get_server_rotation_mount()
    power_meter = tb.get_server_power_meter()

    # Set up the spectrometer
    ## Create the LightField Application (true for visible)
    ## The 2nd parameter forces LF to load with no experiment
    _auto = Automation(True, List[String]())

    ## Get LightField Application object
    _application = _auto.LightFieldApplication

    ## Get experiment object
    _experiment = _application.Experiment
    _experiment.Load("11212025_NV")

    sensor_height = _experiment.GetValue(
        CameraSettings.SensorInformationActiveAreaHeight
    )
    sensor_width = _experiment.GetValue(CameraSettings.SensorInformationActiveAreaWidth)
    grating_center_wavelength = _experiment.GetValue(
        SpectrometerSettings.GratingCenterWavelength
    )

    print(f"Sensor = ({sensor_height}, {sensor_width})")
    print(f"Center wavelength = {grating_center_wavelength} nm")

    ## Start data collection

    for i in range(len(wavelengths)):
        wavelength = wavelengths[i]
        calibration_file = calibration_files[i]

        # Set wavelength
        tisapph.set_wavelength_nm(wavelength)
        power_meter.set_wavelength(wavelength)
        time.sleep(1)

        # Make data arrays
        actual_powers_swept = np.zeros(num_points)
        total_counts = np.zeros(num_points)

        # Determine half wave plate sweep parameters
        hwp_params = find_hwp_fit_params(
            slider_2, power_meter, hwp_rotator, calibration_file
        )
        _, amp, vert = hwp_params
        if max_power > vert + amp:  # TODO: Fix later...
            raise ValueError(
                f"Provided power sweep range [{min_power:.3e}, {max_power:.3e}] outside of possible powers [{vert:.3e}, {amp + vert:.3e}]"
            )
        # min_power_hwp_angle = get_angle_from_params(min_power, hwp_params)
        min_power_hwp_angle = hwp_params[0] + 45
        max_power_hwp_angle = get_angle_from_params(max_power, hwp_params)

        hwp_angles = np.round(
            np.linspace(
                min(min_power_hwp_angle, max_power_hwp_angle),
                max(min_power_hwp_angle, max_power_hwp_angle),
                num_points,
            ),
            decimals=1,
        )

        if max_power_hwp_angle < min_power_hwp_angle:
            hwp_angles = hwp_angles[::-1]

        print(
            f"Min HWP angle = {min_power_hwp_angle: .1f}deg, Max HWP angle = {max_power_hwp_angle: .1f}deg"
        )
        print(hwp_angles)

        # Set up data taking folder
        timestamp = dm.get_time_stamp()
        date, _ = timestamp.split("-")
        year, month, _ = date.split("_")

        # Make unique folder for each wavelength and timestamp
        data_path = r"G:\nvdata\pc_Nuclear\branch_master\sweep_power"
        data_path = f"{data_path}/{year}_{month}"

        # Create numpy array to store all the spectra
        spectra_data = np.zeros((num_points, sensor_width))

        try:
            # Validate camera state
            if validate_camera(_experiment):
                frames = 1

                for i in range(num_points):
                    ## Set half wave plate
                    hwp_angle = hwp_angles[i]
                    print(i, hwp_angle)
                    hwp_rotator.set_angle(hwp_angle)

                    ## Measure power with power meter
                    slider_2.set_filter(3)  # Move to mirror setting
                    time.sleep(0.3)
                    meas_power = power_meter.get_power()
                    actual_powers_swept[i] = meas_power
                    slider_2.set_filter(2)  # Move back to throughput setting
                    time.sleep(0.3)

                    ## Take data with spectrometer
                    dataset = _experiment.Capture(frames)
                    # Stop processing if we do not have all frames
                    if dataset.Frames != frames:
                        # Clean up the image data set
                        dataset.Dispose()
                        raise Exception("Frames are not equal.")

                    # Get the data from the current frame
                    image_data = list(dataset.GetFrame(0, frames - 1).GetData())

                    # Do some data analysis, calculate total counts
                    total_counts[i] = np.sum(
                        image_data[start_wavelength_i:end_wavelength_i]
                    )

                    ## Save data as numpy array
                    spectra_data[i] = image_data
                    if i % 10 == 0 or i == num_points - 1:
                        # Write the spectra
                        np.save(
                            f"{data_path}/{timestamp}_{wavelength:.1f}nm_spectra.npy",
                            spectra_data,
                        )

                        # Write the measured powers
                        np.save(
                            f"{data_path}/{timestamp}_{wavelength:.1f}nm_powers.npy",
                            actual_powers_swept,
                        )

        except Exception:
            print(traceback.format_exc())

        fig, ax = plt.subplots()
        ax.scatter(actual_powers_swept, total_counts)
        ax.set_xlabel("Power (W)")
        ax.set_ylabel(f"Total counts ({start_wavelength:.1f}-{end_wavelength:.1f}nm)")
        ax.set_title(f"Wavelength: {int(wavelength)}nm")
        # plt.show()
        plt.savefig(f"{data_path}/{timestamp}_{wavelength:.1f}nm_plot.png")

        # Save the parameters
        raw_data = {
            "timestamp": timestamp,
            "calibration_file": calibration_file,
            "num_points": num_points,
            "num_runs": 1,
            "wavelength": wavelength,
            "min_power": min_power,
            "max_power": max_power,
            "slider_1_pos": slider_1_position,
            "slider_3_pos": slider_3_position,
            "voltages-units": "photons",
            "grating_center_wavelength": grating_center_wavelength,
            "RF_on": RF_on,
        }
        file_path = dm.get_file_path(__file__, timestamp, f"{wavelength:.1f}nm")
        dm.save_raw_data(raw_data, file_path)

    return


# 12/05/2025
# file_name = "2025_12_05-08_05_18-Calibration-695nm"
# file_name = "2025_12_05-08_05_46-Calibration-697nm"
# file_name = "2025_12_05-08_06_07-Calibration-700nm"
# file_name = "2025_12_05-08_06_29-Calibration-702nm"
# file_name = "2025_12_05-08_06_49-Calibration-705nm"
# file_name = "2025_12_05-08_07_08-Calibration-707nm"
# file_name = "2025_12_05-08_07_28-Calibration-710nm"
# file_name = "2025_12_05-08_07_48-Calibration-712nm"
# file_name = "2025_12_05-08_09_16-Calibration-714nm"

# calib_files = [
# "2025_12_08-13_14_57-Calibration-696.0nm",
# "2025_12_08-13_15_10-Calibration-698.0nm",
# "2025_12_08-13_15_23-Calibration-700.1nm",
# "2025_12_08-13_15_36-Calibration-702.6nm",
# "2025_12_08-13_15_49-Calibration-705.0nm",
# "2025_12_08-13_16_02-Calibration-707.6nm",
# ]
# wavelengths = [696, 698, 700, 702.5, 705, 707.5]
# wavelengths = [705]

# calib_files = [
#     "2025_12_08-14_14_21-Calibration-710.1nm",
# "2025_12_08-14_14_35-Calibration-712.6nm",
# "2025_12_08-14_14_48-Calibration-715.0nm",
# ]
# wavelengths = [710, 712.5, 715]
# wavelengths = [710, 715]

# calib_files = [
#     "2025_12_08-14_14_48-Calibration-715.0nm",
#     "2025_12_08-14_15_02-Calibration-720.1nm",
#     "2025_12_08-14_15_15-Calibration-725.0nm",
# ]
# wavelengths = [715, 720, 725]

calib_files = [
    "2025_12_08-14_35_56-Calibration-730.1nm",
    # "2025_12_08-14_36_09-Calibration-735.0nm",
    "2025_12_08-14_36_23-Calibration-740.1nm",
    # "2025_12_08-14_36_36-Calibration-745.1nm",
    "2025_12_08-14_36_49-Calibration-750.1nm",
    "2025_12_08-14_37_03-Calibration-760.0nm",
    "2025_12_08-14_37_16-Calibration-770.0nm",
    "2025_12_08-14_37_29-Calibration-780.1nm",
    "2025_12_08-14_37_43-Calibration-790.0nm",
    "2025_12_08-14_37_56-Calibration-800.0nm",
]
# wavelengths = [730, 735, 740, 745, 750, 760, 770, 780, 790, 800]
wavelengths = [730, 740, 750, 760, 770, 780, 790, 800]

# calib_files = []
# wavelengths = []


if __name__ == "__main__":
    main(
        calibration_files=calib_files,
        wavelengths=wavelengths,
        slider_1_position=2,
        slider_3_position=1,
        min_power=0.000,
        max_power=0.090,
        num_points=50,
        start_wavelength=675,
        end_wavelength=705,
    )
