# -*- coding: utf-8 -*-
"""
Control panel for the slm

Created on Spring, 2024

@author: saroj chand
"""

import os
import sys
import ctypes
import time 
import warnings
from datetime import datetime
from pathlib import Path

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils import common
from utils import data_manager as dm
from IPython.display import Image

from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.holography import analysis, toolbox
from slmsuite.holography.algorithms import SpotHologram

# from slmsuite.nv_coords_weights_reorder_reassign import curve_extreme_weights_simple

warnings.filterwarnings("ignore")
mpl.rc("image", cmap="Blues")

def plot_phase(phase, angle):
    # Initialize the figure and axes outside the loop
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    blaze_vector = (np.cos(np.radians(angle)), np.sin(np.radians(angle)))

    # Update phase with live rotation
    delta_phase = toolbox.phase.blaze(grid=slm, vector=blaze_vector, offset=0)
    phase = None

    # Display the phase pattern on the SLM
    slm.write(phase, settle=True)

    # Capture image from the camera
    cam.set_exposure(0.0001)
    im = cam.get_image()

    # Clear the axes and plot the phase, delta phase, and camera image
    ax[0].clear()
    ax[0].imshow(phase, cmap="gray")
    ax[0].set_title("Total Phase")

    ax[1].clear()
    ax[1].imshow(delta_phase, cmap="gray")
    ax[1].set_title("Delta Phase")

    ax[2].clear()
    ax[2].imshow(im, cmap="gray")
    ax[2].set_title("Camera Image")

    plt.pause(0.01)


def cam_plot():
    cam.set_exposure(0.0001)
    img = cam.get_image()
    # Plot the result
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap="gray")  # Adjust 'cmap' as needed for color maps
    plt.show()

    # # Save the image

    file_path = r"slmsuite\cam_image"
    num_nvs = len(nuvu_pixel_coords)
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"slm_generated_spots_{num_nvs}nvs_{date_time_str}.npy"
    # Save the phase data
    save(img, file_path, filename)
    print(f"Image saved at {file_path}")


def blaze(vector_deg=(0.2, 0.2)):
    # Get .2 degrees in normalized units.
    vector = toolbox.convert_blaze_vector(vector_deg, from_units="deg", to_units="norm")
    blaze_phase = toolbox.phase.blaze(grid=slm, vector=vector)
    plot_phase(blaze_phase, title="Blaze at {} deg".format(vector_deg))


# region "calibration functions"
def fourier_calibration():
    cam.set_exposure(0.0001)  # Increase exposure because power will be split many ways
    fs.fourier_calibrate(
        array_shape=[11, 11],  # Size of the calibration grid (Nx, Ny) [knm]
        array_pitch=[100, 100],  # Pitch of the calibration grid (x, y) [knm]
        plot=True,
    )
    cam.set_exposure(0.0002)
    # Save new calibration history through the standard nvdata hierarchy.
    output_dir = dm.get_output_folder(
        __file__, subfolder="fourier_calibration", create=True
    )
    calibration_file = fs.save_fourier_calibration(path=str(output_dir))
    print("Fourier calibration saved to:", calibration_file)


def test_wavefront_calibration():
    cam.set_exposure(0.0001)
    movie = fs.wavefront_calibrate(
        interference_point=(600, 400),
        field_point=(0.25, 0),
        field_point_units="freq",
        superpixel_size=60,
        test_superpixel=(16, 16),  # Testing mode
        autoexposure=False,
        plot=3,  # Special mode to generate a phase .gif
    )
    timestamp = dm.get_time_stamp()
    gif_path = dm.get_file_path(
        __file__, timestamp, "wavefront", subfolder="wavefront_calibration"
    ).with_suffix(".gif")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, movie)
    Image(filename=str(gif_path))


def wavefront_calibration():
    cam.set_exposure(0.001)
    fs.wavefront_calibrate(
        interference_point=(600, 400),
        field_point=(0.25, 0),
        field_point_units="freq",
        superpixel_size=40,
        autoexposure=False,
    )
    # Save new calibration history through the standard nvdata hierarchy.
    output_dir = dm.get_output_folder(
        __file__, subfolder="wavefront_calibration", create=True
    )
    calibration_file = fs.save_wavefront_calibration(path=str(output_dir))
    print("Wavefront calibration saved to:", calibration_file)
    
    
# region "phase patterns"
def circles():
    # cam.set_exposure(0.1)

    center = (705, 520)

    # Use larger radii and more spacing
    radii = np.linspace(20, 120, num=6)

    circle_points = []
    for radius in radii:
        # more points per ring
        num_points = max(12, int(2 * np.pi * radius / 30))

        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)

        circle = np.vstack((x_circle, y_circle))
        circle_points.append(circle)

    circles = np.concatenate(circle_points, axis=1)

    hologram = SpotHologram(
        shape=(2048, 2048),
        spot_vectors=circles,
        basis="ij",
        cameraslm=fs,
    )

    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    
    ### save the phase data
    # file_path = r"slmsuite\phase"
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    # filename = f"slm_calibration_circle_{date_time_str}.npy"
    # # Save the phase data
    # save(phase, file_path, filename)
    # cam_plot()

def calibration_triangle():
    # Define parameters for the equilateral triangle
    center = (720, 530)  # Center of the triangle
    side_length = 140  # Length of each side of the triangle

    # Calculate the coordinates of the three vertices of the equilateral triangle
    theta = np.linspace(0, 2 * np.pi, 4)[:-1]  # Exclude the last point to avoid overlap
    x_triangle = center[0] + side_length * np.cos(theta + np.pi / 6)  # X coordinates
    y_triangle = center[1] + side_length * np.sin(theta + np.pi / 6)  # Y coordinates
    
    # x_triangle = [821.24355653, 598.75644347, 700.0]
    # sys.exit()
    # Combine the coordinates into a grid format
    triangle_points = np.vstack((x_triangle, y_triangle))

    print("thorcam coords:", triangle_points)
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=triangle_points, basis="ij", cameraslm=fs
    )

    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    
    ### save the phase data
    # file_path = r"slmsuite\phase"
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    # filename = f"slm_calibration_triangle_{date_time_str}.npy"
    # Save the phase data
    # save(phase, file_path, filename)
    # cam_plot()
    
# endregion

# region Load calibration
def load_fourier_calibration():
    calibration_file_path = (
        # "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00003.h5"
        # "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00006.h5"
        # "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00008.h5"
        "calibrations/purcell/current/slm_fourier.h5"
    )
    
    fs.load_fourier_calibration(calibration_file_path)
    print(fs.fourier_calibration)
    print("Fourier calibration loaded from:", calibration_file_path)


def load_wavefront_calibration():
    calibration_file_path = (
        "slmsuite/wavefront_calibration/26438-SLM-wavefront-calibration_00004.h5"
    )
    fs.load_wavefront_calibration(calibration_file_path)
    print("Wavefront calibration loaded from:", calibration_file_path)
  
# def nuvu2thorcam_slm(
#     coords,
#     calib_path="slmsuite/calibration/nuvu_to_thorcam_slm.npz",
# ):
#     data = np.load(calib_path, allow_pickle=True)
#     M = np.asarray(data["M_nuvu_to_thorcam_slm"], dtype=np.float32)
#     return apply_affine(M, coords)

def nuvu2thorcam_slm(coords):
    """Transform Nuvu pixels to SLM-calibrated ThorCam pixels."""
    config = common.get_config_dict()
    cal = config["SpatialCalibrations"]
    
    src = np.asarray(cal["calibration_coords_nuvu"], dtype=np.float64)
    dst = np.asarray(cal["calibration_coords_thorcam"], dtype=np.float64)

    # Fit affine transformation using all calibration points.
    X = np.column_stack((src, np.ones(len(src))))
    M = np.linalg.lstsq(X, dst, rcond=None)[0].T

    # Transform NV coordinates.
    coords = np.asarray(coords, dtype=np.float64)
    return coords @ M[:, :2].T + M[:, 2]

# endregion


# nv phase computation
def load_nv_coords():
    config = common.get_config_dict()
    file_path = config["SpatialCalibrations"]["active_nv_coords_path"]
    data = np.load(file_path, allow_pickle=True)
    nv_coordinates = data["nv_coordinates"]
    spot_weights = data["updated_spot_weights"]
    print(f"len of nv coords: {len(nv_coordinates)}")
    return nv_coordinates, spot_weights

# ----------------------------
# Load coordinates and weights
# ----------------------------
nuvu_pixel_coords, spot_weights = load_nv_coords()
# Transform Nuvu coordinates to ThorCam coordinates
thorcam_coords = nuvu2thorcam_slm(nuvu_pixel_coords)  # shape: (N, 2)
thorcam_coords_xy = thorcam_coords.T                  # shape: (2, N)

print("nuvu_pixel_coords shape:", nuvu_pixel_coords.shape)
print("thorcam_coords shape:", thorcam_coords_xy.shape)
print("spot_weights shape:", spot_weights.shape)
print("spot weight min/max:", np.nanmin(spot_weights), np.nanmax(spot_weights))

# sys.exit()

def compute_and_write_nvs_phase():
    hologram = SpotHologram(
        shape=(4096, 2048),
        spot_vectors=thorcam_coords_xy,
        basis="ij",
        spot_amp=spot_weights,
        cameraslm=fs,
    )
    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=50,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    # Precondition computationally
    initial_phase = hologram.extract_phase()
    # Define the path to save the phase data1
    file_path = r"slmsuite\computed_phase"
    num_nvs = len(nuvu_pixel_coords)
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"slm_phase_{num_nvs}nvs_{date_time_str}.npy"
    # Save the phase data
    save(initial_phase, file_path, filename)
    # write
    slm.write(initial_phase, settle=True)
    # cam_plot()
    
def write_pre_computed_nvs_phase():
    phase = np.load("slmsuite\computed_phase\slm_phase_631nvs_20260916_122929.npy")
    slm.write(phase, settle=True)
    # cam_plot()


def write_pre_computed_circles():
    phase = np.load("slmsuite\phase\slm_calibration_circles_20260607_121625.npy")
    slm.write(phase, settle=True)
    # cam_plot()
    
def write_pre_computed_triangle():
    phase = np.load("slmsuite\phase\slm_calibration_triangle_20260721_161807.npy")
    slm.write(phase, settle=True)
    # cam_plot()

# Define the save function
def save(data, path, filename):
    """Save generated SLM data through the standard nvdata hierarchy."""
    timestamp = dm.get_time_stamp()
    name = Path(filename).stem
    subfolder = Path(path).name if path else None
    file_path = dm.get_file_path(
        __file__, timestamp, name, subfolder=subfolder
    )
    saved_path = dm.save_npy(data, file_path)
    print(f"Saved to {saved_path}")
    return saved_path

#endregion
class DummyCamera:
    """
    Minimal Camera-like object for slmsuite CameraSLM/FourierSLM.

    Must have:
        name
        shape
        get_image()
        close()

    shape is in numpy/image convention: (height, width).
    For ThorCam 26438, use the real camera frame size:
        (2160, 2880)
    """

    def __init__(self, shape=(2160, 2880), name="26438"):
        self.shape = tuple(shape)
        self.name = str(name)
        self.closed = False

    def get_image(self, *args, **kwargs):
        raise RuntimeError(
            "DummyCamera has no hardware attached. "
            "You called get_image(), which is not allowed in camera-free runtime."
        )

    def close(self):
        """
        Dummy close function so cleanup code can safely call cam.close().
        """
        self.closed = True
        print(f"DummyCamera {self.name} closed.")
        
def hold_pattern():
#"""Keep the pattern on the SLM until "close" is typed.

#The Thorlabs display window only repaints while this thread processes its Windows
#messages. A plain input() blocks them, so the SLM can keep showing the window's black
#background instead of the pattern. This loop processes the messages while it waits.
#"""
    import msvcrt
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    msg = wintypes.MSG()
    typed = ""
    print("Pattern displayed and held. Type close and press Enter to close.", flush=True)
    while True:
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # 1 = PM_REMOVE
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        while msvcrt.kbhit():
            char = msvcrt.getwche()
            if char in ("\r", "\n"):
                print(flush=True)
                if typed.strip().lower() == "close":
                    return
                typed = ""
                print("Still holding. Type close and press Enter to close.", flush=True)
            elif char == "\b":
                typed = typed[:-1]
            else:
                typed += char
        time.sleep(0.02)

try:
    slm = ThorSLM()
    # slm = Meadowlark()
    thorcam_shape = (2160, 2880) 
    cam = DummyCamera(shape=thorcam_shape, name="26438")
    # cam = ThorCam(serial="26438", verbose=True)
    fs = FourierSLM(cam, slm)
    
    # cam = tb.get_server_thorcam()
    # slm = tb.get_server_thorslm()
    # fourier_calibration()
    load_fourier_calibration()
    
    # test_wavefront_calibration()
    # wavefront_calibration()
    # load_wavefront_calibration()
    
    compute_and_write_nvs_phase()  # write all the NV coordinates 
    # write_pre_computed_nvs_phase()
    
    # calibration_triangle()   # just for calibration we target only 3
    # write_pre_computed_triangle()
    
    # circles()
    # write_pre_computed_circles()
    # smiley()
    # cam_plot()
    
    hold_pattern()  # keeps the window repainting; type close + Enter to release the SLM
    # input("Pattern displayed and held. Press Enter to close...")
finally:
    print("Closing")

    if slm is not None:
        slm.close()

    if cam is not None:
        cam.close()
# endregions
