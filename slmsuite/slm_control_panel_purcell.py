# -*- coding: utf-8 -*-
"""
Control panel for the slm

Created on Spring, 2024

@author: saroj chand
"""

import os
import sys
import warnings
from datetime import datetime

# os.environ["QT_QPA_PLATFORM"] = "offscreen"
import cv2
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Generate a phase .gif
from IPython.display import Image

from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.holography import analysis, toolbox
from slmsuite.holography.algorithms import SpotHologram

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


# region "calibration"
def fourier_calibration():
    cam.set_exposure(0.0001)  # Increase exposure because power will be split many ways
    fs.fourier_calibrate(
        array_shape=[11, 11],  # Size of the calibration grid (Nx, Ny) [knm]
        array_pitch=[100, 100],  # Pitch of the calibration grid (x, y) [knm]
        plot=True,
    )
    cam.set_exposure(0.0002)
    # save calibation
    calibration_file = fs.save_fourier_calibration(path="slmsuite/fourier_calibration")
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
    imageio.mimsave("wavefront.gif", movie)
    Image(filename="wavefront.gif")


def wavefront_calibration():
    cam.set_exposure(0.001)
    fs.wavefront_calibrate(
        interference_point=(600, 400),
        field_point=(0.25, 0),
        field_point_units="freq",
        superpixel_size=40,
        autoexposure=False,
    )
    # save calibation
    calibration_file = fs.save_wavefront_calibration(
        path="slmsuite/wavefront_calibration"
    )
    print("Fourier calibration saved to:", calibration_file)


def load_fourier_calibration():
    calibration_file_path = (
        # "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00003.h5"
        # "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00006.h5"
        # "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00008.h5"
        "slmsuite/fourier_calibration/26438-SLM-fourier-calibration_00015.h5"
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


def evaluate_uniformity(vectors=None, size=25):
    # Set exposure and capture image
    cam.set_exposure(0.001)
    img = cam.get_image()
    # Extract subimages
    if vectors is None:
        subimages = analysis.take(img, vectors=None, size=size)
    else:
        subimages = analysis.take(img, vectors=vectors, size=size)

    # Plot subimages
    analysis.take_plot(subimages)
    # Normalize subimages and compute powers
    powers = analysis.image_normalization(subimages)
    # Plot histogram of powers
    plt.hist(powers / np.mean(powers))
    plt.show()


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

# region "nv phase calulation"
def calibration_triangle():
    # Define parameters for the equilateral triangle
    center = (705, 520)  # Center of the triangle
    # side_length = 80  # Length of each side of the triangle
    side_length = 150  # Length of each side of the triangle

    # Calculate the coordinates of the three vertices of the equilateral triangle
    theta = np.linspace(0, 2 * np.pi, 4)[:-1]  # Exclude the last point to avoid overlap
    x_triangle = center[0] + side_length * np.cos(theta + np.pi / 6)  # X coordinates
    y_triangle = center[1] + side_length * np.sin(theta + np.pi / 6)  # Y coordinates
    
    # x_triangle = [834.90381057, 475.09618943, 705.0]
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
    # file_path = r"slmsuite\calibration"
    # num_nvs = len(nuvu_pixel_coords)
    # now = datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    # filename = f"slm_calibration_{num_nvs}nvs_{date_time_str}.npy"
    # Save the phase data
    # save(phase, file_path, filename)
    # cam_plot()
    
def nuvu2thorcam_calibration(coords):
    """
    Calibrates and transforms coordinates from the Nuvu camera's coordinate system
    to the Thorlabs camera's coordinate system using an affine transformation.
    """
    # cal_coords_thorcam = np.array(
    #     [[883.205,   640. ], [536.795,  640. ], [710., 340.]], dtype="float32"
    # )

    # cal_coords_nuvu = np.array(
    #     [[338.04, 361.354], [311.711, 11.032], [18.043, 209.58]], dtype="float32"
    # )
    
    cal_coords_thorcam = np.array(
        [                
            [848.56406461, 620.0],
            [571.43593539, 620.0],
            [710.0,        380.0],
        ], dtype="float32"
    )

    cal_coords_nuvu = np.array(
        [
            [84.57, 7.014],
            [92.258, 353.467],
            [337.832, 184.63],
        ], dtype="float32"
    )
    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(cal_coords_nuvu, cal_coords_thorcam)
    # Append a column of ones to the input coordinates to facilitate affine transformation
    ones_column = np.ones((coords.shape[0], 1))
    coords_homogeneous = np.hstack((coords, ones_column))
    thorcam_coords = np.dot(coords_homogeneous, M.T)

    return thorcam_coords

def apply_affine(M, coords):
    coords = np.asarray(coords, dtype=np.float32)

    single_point = False
    if coords.ndim == 1:
        coords = coords[None, :]
        single_point = True

    ones = np.ones((coords.shape[0], 1), dtype=np.float32)
    coords_h = np.hstack([coords, ones])
    out = coords_h @ M.T

    if single_point:
        return out[0]

    return out


def nuvu2thorcam_slm(
    coords,
    calib_path="slmsuite/calibration/nuvu_to_thorcam_slm.npz",
):
    data = np.load(calib_path, allow_pickle=True)
    M = np.asarray(data["M_nuvu_to_thorcam_slm"], dtype=np.float32)
    return apply_affine(M, coords)

def load_nv_coords( 
    # file_path="slmsuite/nv_blob_detection/nv_blob_1460nvs_reordered.npz",   
    # file_path="slmsuite/nv_blob_detection/nv_blob_1348nvs_reordered.npz",   
    # file_path="slmsuite/nv_blob_detection/nv_blob_1306nvs_reordered.npz",   
    # file_path="slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz",   
    # file_path="slmsuite/nv_blob_detection/nv_blob_1274nvs_reordered.npz",   
    file_path="slmsuite/nv_blob_detection/nv_blob_1267nvs_reordered.npz",   
):
    data = np.load(file_path, allow_pickle=True)
    nv_coordinates = data["nv_coordinates"]
    spot_weights = data["updated_spot_weights"]
    print(f"len of nv coords: {len(nv_coordinates)}")
    return nv_coordinates, spot_weights

# nuvu_pixel_coords, spot_weights = load_nv_coords()
# # nuvu_pixel_coords = np.array([[215.025, 203.863], [308.628, 103.893], [238.142, 328.739], [63.706, 100.683]])
# thorcam_coords_xy = nuvu2thorcam_calibration(nuvu_pixel_coords).T
# thorcam_coords_xy = nuvu2thorcam_calibration(nuvu_pixel_coords).T

nuvu_pixel_coords, spot_weights = load_nv_coords()

thorcam_coords = nuvu2thorcam_slm(nuvu_pixel_coords)   # shape: (N, 2)
thorcam_coords_xy = thorcam_coords.T                   # shape: (2, N)
# test_inds = [0, 10, 100, 500, 1000]
# thorcam_coords_xy = thorcam_coords[test_inds].T
print("thorcam_coords shape:", thorcam_coords.shape)
print("thorcam_coords_xy shape:", thorcam_coords_xy.shape)
print("first ThorCam coord:", thorcam_coords[0])

zero_nuvu = np.array([186.796, 186.907], dtype=np.float32)
zero_thor = nuvu2thorcam_slm(zero_nuvu)

print("zero_nuvu:", zero_nuvu)
print("mapped zero_thorcam_slm:", zero_thor)
# sys.exit()

def compute_and_write_nvs_phase():
    hologram = SpotHologram(
        shape=(4096, 2048),
        spot_vectors=thorcam_coords_xy,
        basis="ij",
        # spot_amp=spot_weights,
        cameraslm=fs,
    )
    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
# 
    initial_phase = hologram.extract_phase()
    # Define the path to save the phase data1
    file_path = r"slmsuite\computed_phase"
    num_nvs = len(nuvu_pixel_coords)
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"slm_phase_{num_nvs}nvs_{date_time_str}.npy"
    # file_path = dm.get_file_path(__file__, filename)
    # Save the phase data
    save(initial_phase, file_path, filename)
    slm.write(initial_phase, settle=True)
    # cam_plot()
    
def write_pre_computed_nvs_phase():
    phase = np.load("slmsuite\computed_phase\slm_phase_75nvs_20250605_181402.npy")
    slm.write(phase, settle=True)
    # cam_plot()


def write_pre_computed_circles():
    phase = np.load("slmsuite\circles\slm_phase_circles_20250606_163346.npy")
    slm.write(phase, settle=True)
    # cam_plot()

# Define the save function
def save(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, filename), data)
    
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
    
    # compute_and_write_nvs_phase()
    calibration_triangle()
    
    # circles()
    # smiley()
    # cam_plot()
    
    input("Pattern displayed and held. Press Enter to close...")
finally:
    print("Closing")

    if slm is not None:
        slm.close()

    if cam is not None:
        cam.close()
# endregions
