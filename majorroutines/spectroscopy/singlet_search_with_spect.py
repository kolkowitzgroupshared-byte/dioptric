# -*- coding: utf-8 -*-
"""
Search for NV triplet-to-singlet wavelength

Created on August 9th, 2025

@author: jchen-1
"""

import ctypes
import os

# Import python sys module
import sys
from pathlib import Path

import clr
import matplotlib.pyplot as plt

# numpy import
import numpy as np

# Import c compatible List and String
from System import *
from System.Collections.Generic import List
from System.Runtime.InteropServices import GCHandle, GCHandleType, Marshal

# Add needed dll references
sys.path.append(os.environ["LIGHTFIELD_ROOT"])
sys.path.append(os.environ["LIGHTFIELD_ROOT"] + "\\AddInViews")
clr.AddReference("PrincetonInstruments.LightFieldViewV5")
clr.AddReference("PrincetonInstruments.LightField.AutomationV5")
clr.AddReference("PrincetonInstruments.LightFieldAddInSupportServices")

# PI imports
import time
import traceback
from datetime import datetime
from random import shuffle

import numpy as np
from matplotlib import pyplot as plt
from PrincetonInstruments.LightField.AddIns import *
from PrincetonInstruments.LightField.Automation import *
from tqdm import tqdm

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb


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


def manipulate_image_data(dat, buff_format, sensor_height, sensor_width):
    im = convert_buffer(dat, buff_format)
    im = np.reshape(im, (sensor_height, sensor_width))
    return im


def get_ROI_from_image(arr, start_col_index, end_col_index):
    if end_col_index is None and start_col_index is not None:
        return arr[:, start_col_index:]

    elif start_col_index is None and end_col_index is not None:
        return arr[:, :end_col_index]

    elif start_col_index is not None and end_col_index is not None:
        return arr[:, start_col_index:end_col_index]

    elif end_col_index is None and start_col_index is None:
        return arr


def create_raw_figure(wavelengths, relative_diff, relative_diff_err):
    fig, ax = plt.subplots()
    kpl.plot_points(ax, wavelengths, relative_diff, relative_diff_err)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Contrast (arb. units)")


def create_fit_figure():
    pass


def main(
    min_wavelength,
    max_wavelength,
    num_steps,
    num_runs,
    RF_on,
    etalon_range,
    start_col_index: int = None,
    end_col_index: int = None,
    exposure_time: float = 1000.0,  # milliseconds
    etalon_spacing: float = 1,
    tisaph_power: float = None,
):
    # Create the LightField Application (true for visible)
    # The 2nd parameter forces LF to load with no experiment
    _auto = Automation(True, List[String]())

    # Get LightField Application object
    _application = _auto.LightFieldApplication

    # Get experiment object
    _experiment = _application.Experiment
    _experiment.Load("test_130825")

    sensor_height = _experiment.GetValue(
        CameraSettings.SensorInformationActiveAreaHeight
    )
    sensor_width = _experiment.GetValue(CameraSettings.SensorInformationActiveAreaWidth)
    grating_center_wavelength = _experiment.GetValue(
        SpectrometerSettings.GratingCenterWavelength
    )

    dummy_ROI = get_ROI_from_image(
        np.zeros((sensor_height, sensor_width)), start_col_index, end_col_index
    )
    ROI_height, ROI_width = dummy_ROI.shape

    print(f"Sensor = ({sensor_height}, {sensor_width})")
    print(f"Center wavelength = {grating_center_wavelength} nm")

    # Create sweep parameters
    wavelengths = np.linspace(min_wavelength, max_wavelength, num_steps)
    etalon_settings = np.arange(
        50 - int(etalon_range / 2),
        50 + (etalon_range - int(etalon_range / 2)) + 1,
        etalon_spacing,
    )
    num_etalon = len(etalon_settings)
    etalon_settings_rearr = np.copy(etalon_settings)
    etalon_settings_rearr[: int(num_etalon / 2)] = np.flip(
        etalon_settings_rearr[: int(num_etalon / 2)]
    )

    step_ind_list = list(range(0, num_steps))
    step_ind_master_list = [None for ind in range(num_runs)]
    tisapph = common.get_server_by_name("tisapph_M2_solstis")

    ### Make folders to store data
    dm_folder = common.get_data_manager_folder()
    timestamp = dm.get_time_stamp()
    date, _ = timestamp.split("-")
    year, month, _ = date.split("_")

    # Row 0 - Wavelength, Row 1 - Total counts in ROI
    data_to_save = np.empty((2, num_runs, num_steps, num_etalon))
    start_time = time.time()

    # Make new folder for each datarun in its designated month folder
    month_path = f"G:/nvdata/pc_Nuclear/branch_master/singlet_search_with_spect/{year}_{month}/{timestamp}"
    data_path = Path(month_path)
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)

    # Start data collection

    try:
        # Validate camera state
        if validate_camera(_experiment):
            # Full Frame
            # _experiment.SetFullSensorRegion()
            frames = 1

            # Set exposure
            _experiment.SetValue(
                CameraSettings.ShutterTimingExposureTime, exposure_time
            )
            # Set number of frames
            _experiment.SetValue(
                ExperimentSettings.AcquisitionFramesToStore, Int32(frames)
            )

            for run_ind in range(num_runs):
                print(f"Run {run_ind}")
                shuffle(step_ind_list)

                # Steps loop
                num_steps_completed = 0

                for ind in tqdm(range(len(step_ind_list)), desc="Steps", position=0):
                    step_ind = step_ind_list[ind]
                    wavelength_data = np.zeros((num_etalon, ROI_height, ROI_width))

                    # Set coarse wavelength
                    wavelength = wavelengths[step_ind]
                    tisapph.set_wavelength_nm(wavelength)
                    tisapph.tune_etalon_relative(50)
                    time.sleep(1.5)

                    # Save wavelength data
                    data_to_save[0, run_ind, num_steps_completed, :] = wavelength

                    # Decrement and then increment etalon voltage from 50%
                    for eind in tqdm(
                        range(num_etalon),
                        desc="Etalon",
                        position=1,
                        leave=False,
                    ):
                        etalon_sett = etalon_settings_rearr[eind]
                        # print(f"Etalon setting = {etalon_sett}")
                        tisapph.tune_etalon_relative(etalon_settings_rearr[eind])
                        time.sleep(0.3)

                        dataset = _experiment.Capture(frames)
                        # Stop processing if we do not have all frames
                        if dataset.Frames != frames:
                            # Clean up the image data set
                            dataset.Dispose()
                            raise Exception("Frames are not equal.")

                        # Get the data from the current frame
                        image_data = dataset.GetFrame(0, frames - 1).GetData()
                        image_frame = dataset.GetFrame(0, frames - 1)

                        # fig, ax = plt.subplots()
                        # ax.plot(image_data)
                        # plt.show()
                        # return
                        image_arr = manipulate_image_data(
                            image_data, image_frame.Format, sensor_height, sensor_width
                        )

                        # Get total counts from ROI
                        total_counts_from_ROI = np.sum(image_arr[:, 1110:1160])
                        data_to_save[1, run_ind, num_steps_completed, eind] = (
                            total_counts_from_ROI
                        )

                        # Save to the NAS
                        ROI_image_arr = get_ROI_from_image(
                            image_arr, start_col_index, end_col_index
                        )

                        wavelength_data[etalon_sett - min(etalon_settings), :, :] = (
                            ROI_image_arr
                        )
                        if eind % 10 == 0 or eind == num_etalon - 1:
                            image_arr_loc = (
                                f"{data_path}/{timestamp}_{wavelength: .2f}.npy"
                            )
                            np.save(image_arr_loc, wavelength_data)
                            np.save(
                                f"{data_path}/{timestamp}_wavelength-ROIcounts.npy",
                                data_to_save,
                            )

                    # Set back to 50%
                    tisapph.tune_etalon_relative(50)
                    num_steps_completed += 1

                ### Move on to the next run
                # Record step order
                step_ind_master_list[run_ind] = step_ind_list.copy()

            print(f"Total Time Elapsed = {time.time() - start_time} s")

    except Exception:
        print(traceback.format_exc())

    ### Return

    raw_data = {
        "num_steps": num_steps,
        "num_runs": num_runs,
        "min_wavelength": min_wavelength,
        "max_wavelength": max_wavelength,
        "wavelengths": wavelengths,
        "step_ind_master_list": step_ind_master_list,
        "voltages-units": "photons",
        "etalon_setts": etalon_settings,
        "camera_start_col_index": start_col_index,
        "camera_end_col_index": end_col_index,
        "grating_center_wavelength": grating_center_wavelength,
        "tisaph_power": tisaph_power,
        "RF_on": RF_on,
    }

    ### Clean up and return

    # tb.reset_cfm()
    raw_data |= {
        "timestamp": timestamp,
    }

    repr_nv_name = "implanted_chinese"
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)


# if __name__ == "__main__":
#     main(
#         min_wavelength=800,
#         max_wavelength=850,
#         num_steps=3,
#         num_runs=2,
#         RF_on=True,
#         etalon_range=10,
#         exposure_time=1000.0,  # milliseconds
#         etalon_spacing=1,
#         tisaph_power=None,
#     )
# kpl.init_kplotlib()

# kpl.show(block=True)
