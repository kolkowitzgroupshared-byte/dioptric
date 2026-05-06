import os
import random
import sys
import time

# Add slmsuite to the python path (for the case where it isn't installed via pip).
sys.path.append(os.path.join(os.getcwd(), "c:/Users/Saroj Chand/Documents/dioptric"))

import warnings

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

warnings.filterwarnings("ignore")
import io

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt

# Generate a phase .gif
from IPython.display import Image
from scipy.optimize import curve_fit

from utils import tool_belt as tb

mpl.rc("image", cmap="Blues")

from slmsuite import example_library
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.thorlabs import ThorSLM
from slmsuite.holography import analysis, toolbox
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.misc import fitfunctions


# funtions
# region "plot_phase" function
def plot_phase(phase, title="", zoom=True):
    # One plot if no camera; two otherwise.
    _, axs = plt.subplots(1, 2 - (cam is None), figsize=(12, 6))

    if cam is None:
        axs = [axs]

    # Plot the phase.
    axs[0].set_title("SLM Phase")
    im = axs[0].imshow(
        np.mod(phase, 2 * np.pi),
        vmin=0,
        vmax=2 * np.pi,
        interpolation="none",
        cmap="gray",
    )
    plt.colorbar(im, ax=axs[0])

    # Grab an image of the resulting pattern and plot.
    slm.write(phase, settle=True)
    img = cam.get_image()

    axs[1].set_title("Camera Result")
    axs[1].imshow(img)
    if zoom:
        xlim = axs[1].get_xlim()
        ylim = axs[1].get_ylim()
        axs[1].set_xlim([xlim[0] * 0.7 + xlim[1] * 0.3, xlim[0] * 0.3 + xlim[1] * 0.7])
        axs[1].set_ylim([ylim[0] * 0.7 + ylim[1] * 0.3, ylim[0] * 0.3 + ylim[1] * 0.7])

    # Make a title, if given.
    plt.suptitle(title)
    plt.show()


def update_plot(phase, angle):
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


# region "blaze" function
def blaze():
    # Get .2 degrees in normalized units.
    vector_deg = (0.2, 0.2)
    vector = toolbox.convert_blaze_vector(vector_deg, from_units="deg", to_units="norm")

    # Get the phase for the new vector
    blaze_phase = toolbox.phase.blaze(grid=slm, vector=vector)

    plot_phase(blaze_phase, title="Blaze at {} deg".format(vector_deg))


# region "calibration"
def fourier_calibration():
    cam.set_exposure(0.03)  # Increase exposure because power will be split many ways
    fs.fourier_calibrate(
        array_shape=[20, 12],  # Size of the calibration grid (Nx, Ny) [knm]
        array_pitch=[20, 30],  # Pitch of the calibration grid (x, y) [knm]
        plot=True,
    )
    cam.set_exposure(0.01)
    # save calibation
    calibration_file = fs.save_fourier_calibration(
        path=r"C:\Users\matth\GitHub\dioptric\slmsuite\fourier_calibration"
    )
    print("Fourier calibration saved to:", calibration_file)


def test_wavefront_calibration():
    cam.set_exposure(0.001)
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
        path=r"C:\Users\matth\GitHub\dioptric\slmsuite\wavefront_calibration"
    )
    print("Fourier calibration saved to:", calibration_file)


# region "load calibration"
def load_fourier_calibration():
    calibration_file_path = r"C:\Users\matth\GitHub\dioptric\slmsuite\fourier_calibration\26438-SLM-fourier-calibration_00003.h5"
    fs.load_fourier_calibration(calibration_file_path)
    print("Fourier calibration loaded from:", calibration_file_path)


def load_wavefront_calibration():
    calibration_file_path = r"C:\Users\matth\GitHub\dioptric\slmsuite\wavefront_calibration\26438-SLM-wavefront-calibration_00004.h5"
    fs.load_wavefront_calibration(calibration_file_path)
    print("Wavefront calibration loaded from:", calibration_file_path)


# region "cam_plot" function
def cam_plot():
    cam.set_exposure(0.0001)
    img = cam.get_image()
    # Plot the result
    plt.figure(figsize=(12, 9))
    plt.imshow(img)
    plt.show()


# def cam_plot():
#     cam.set_exposure(0.0001)
#     img = cam.get_image()

#     # Check if img is a numpy array and has a valid dtype
#     if isinstance(img, np.ndarray):
#         if img.dtype == object:
#             print("Converting image data from dtype object to float32.")
#             img = np.array(img, dtype=np.float32)
#         elif img.dtype != np.uint8 and img.dtype != np.float32:
#             print(f"Converting image data from dtype {img.dtype} to float32.")
#             img = img.astype(np.float32)
#     else:
#         raise TypeError("Captured image is not a numpy array.")

#     # Plot the result
#     plt.figure(figsize=(12, 9))
#     plt.imshow(img, cmap="gray")  # Adding cmap='gray' in case the image is grayscale
#     plt.show()


# region "evaluate" function
def evaluate_uniformity(vectors=None, size=25):
    # Set exposure and capture image
    cam.set_exposure(0.0001)
    img = cam.get_image()
    # cam.set_exposure(0.001)
    # img = cam.get_image()
    print("Images type:", type(img))
    print("Images shape:", np.shape(img))
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

    # return subimages, powers


# region "square" function
def square_array():
    xlist = np.arange(750, 950, 25)  # Get the coordinates for one edge
    ylist = np.arange(400, 750, 25)
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack(
        (xgrid.ravel(), ygrid.ravel())
    )  # Make an array of points in a grid
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=square, basis="ij", cameraslm=fs
    )

    # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    # hologram.plot_nearfield(title="Padded", padded=True)
    # hologram.plot_nearfield(title="Unpadded")
    phase = hologram.extract_phase()
    # plot_phase(phase)
    slm.write(phase, settle=True)
    cam_plot()
    # for ind in range(phase.shape[0]):
    #     for jnd in range(phase.shape[1]):
    #         phase[ind, jnd] += np.dot((ind, jnd), (0.03, 0.03))
    quadratic_phase = toolbox.phase.quadratic_phase(grid=slm, focal_length=25)
    gaussian_phase = toolbox.phase.gaussian(grid=slm, wx=60000, wy=60000)
    # Generate the initial phase with the gradient
    # phase += gaussian_phase
    # plot_phase(quadratic_phase)
    # slm.write(phase, settle=True)
    # cam_plot()
    # evaluate_uniformity(vectors=square)
    # Hone the result with experimental feedback.
    # hologram.optimize(
    #     'WGS-Kim',
    #     maxiter=20,
    #     feedback='experimental_spot',
    #     stat_groups=['computational_spot', 'experimental_spot'],
    #     fixed_phase=False
    # )
    # phase = hologram.extract_phase()
    # slm.write(phase, settle=True)
    # cam_plot()
    # evaluate_uniformity(vectors=square)


def square_array_cirle():
    xlist = np.arange(450, 1050, 25)  # X coordinates for the grid
    ylist = np.arange(350, 750, 25)  # Y coordinates for the grid
    xgrid, ygrid = np.meshgrid(xlist, ylist)

    # Flatten the grids to 1D arrays
    xpoints = xgrid.ravel()
    ypoints = ygrid.ravel()

    # Calculate the center and radius for the circle
    x_center = np.mean(xlist)
    y_center = np.mean(ylist)
    radius = min((xlist[-1] - xlist[0]), (ylist[-1] - ylist[0])) / 2

    # Create a boolean mask for points within the circle
    mask = (xpoints - x_center) ** 2 + (ypoints - y_center) ** 2 <= radius**2

    # Apply the mask to filter the points
    circular_points = np.vstack((xpoints[mask], ypoints[mask]))

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=circular_points, basis="ij", cameraslm=fs
    )

    # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()


# region "circle" function
def circle_pattern():
    cam.set_exposure(0.001)
    # Define parameters for the circle
    center = (850, 540)  # Center of the circle
    radius = 200  # Radius of the circle

    # Generate points within the circle using polar coordinates
    num_points = 30  # Number of points to generate
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angle values
    x_circle = center[0] + radius * np.cos(theta)  # X coordinates
    y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

    # Convert to grid format if needed
    circle = np.vstack((x_circle, y_circle))

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=circle, basis="ij", cameraslm=fs
    )

    # # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)

    # Hone the result with experimental feedback.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="experimental_spot",
        stat_groups=["computational_spot", "experimental_spot"],
        fixed_phase=False,
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    evaluate_uniformity(vectors=circle)


def circles():
    cam.set_exposure(0.1)
    # Define parameters for the circle
    center = (750, 530)  # Center of the circle
    # Generate circles with radii ranging from 10 to 200
    radii = np.linspace(50, 200, num=4)  # Adjust the number of circles as needed
    # Generate points for each circle and create the hologram
    circle_points = []
    for radius in radii:
        num_points = int(
            2 * np.pi * radius / 60
        )  # Adjust the number of points based on the radius

        # Generate points within the circle using polar coordinates
        theta = np.linspace(0, 2 * np.pi, num_points)  # Angle values
        x_circle = center[0] + radius * np.cos(theta)  # X coordinates
        y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

        # Convert to grid format for the current circle
        circle = np.vstack((x_circle, y_circle))

        circle_points.append(circle)

    # Combine the points of all circles
    circles = np.concatenate(circle_points, axis=1)
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=circles, basis="ij", cameraslm=fs
    )

    # # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    # evaluate_uniformity(vectors=circle)

    # Hone the result with experimental feedback.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="experimental_spot",
        stat_groups=["computational_spot", "experimental_spot"],
        fixed_phase=False,
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    # evaluate_uniformity(vectors=circle)


def calibration_triangle():
    cam.set_exposure(0.1)

    # Define parameters for the equilateral triangle
    center = (750, 630)  # Center of the triangle
    side_length = 240  # Length of each side of the triangle

    # Calculate the coordinates of the three vertices of the equilateral triangle
    theta = np.linspace(0, 2 * np.pi, 4)[:-1]  # Exclude the last point to avoid overlap
    x_triangle = center[0] + side_length * np.cos(theta + np.pi / 6)  # X coordinates
    y_triangle = center[1] + side_length * np.sin(theta + np.pi / 6)  # Y coordinates

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
    cam_plot()


# region "scatter" function
def scatter_pattern():
    lloyds_points = (
        toolbox.lloyds_points(
            grid=tuple(int(s / 5) for s in fs.cam.shape), n_points=100, iterations=40
        )
        * 5
    )

    hologram = SpotHologram((2048, 2048), lloyds_points, basis="ij", cameraslm=fs)

    # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot", "experimental_spot"],
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()
    # # Hone the result with experimental feedback.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="experimental_spot",
        stat_groups=["computational_spot", "experimental_spot"],
        fixed_phase=False,
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()


# region "smiley" function


def smiley_pattern():
    # Define points for the smiley face
    x_eyes = [1100, 1200]  # X coordinates for the eyes
    y_eyes = [400, 400]  # Y coordinates for the eyes

    # Define points for the mouth (a semi-circle)
    theta = np.linspace(0, np.pi, 15)  # Angle values for the semi-circle
    mouth_radius = 150  # Radius of the mouth
    x_mouth = 1150 + mouth_radius * np.cos(theta)  # X coordinates for the mouth
    y_mouth = 500 + mouth_radius * np.sin(theta)  # Y coordinates for the mouth

    # Combine all points into a single array
    smiley = np.vstack(
        (np.concatenate((x_eyes, x_mouth)), np.concatenate((y_eyes, y_mouth)))
    )
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=smiley, basis="ij", cameraslm=fs
    )

    # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()


# region "UCB"  function
def UCB_pattern():
    # Define coordinates for each letter in "UCB"
    letters = {
        "U": [(700, 400), (650, 500), (650, 600), (700, 700), (750, 600), (750, 500)],
        "C": [(800, 500), (800, 600), (850, 600), (900, 550), (850, 500), (900, 500)],
        "B": [
            (950, 500),
            (950, 600),
            (1000, 600),
            (1050, 550),
            (1000, 500),
            (1050, 500),
        ],
    }

    # Combine coordinates for "UCB"
    ucb = np.vstack([letters[letter] for letter in "UCB"]).T

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=ucb, basis="ij", cameraslm=fs
    )

    # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()


def pattern_from_image():
    # Load the image of the letters "UCB"
    image = cv2.imread("cgn30eDI_400x400.png", cv2.IMREAD_GRAYSCALE)

    # Threshold the image to obtain binary representation
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Display the thresholded image for debugging
    cv2.imshow("Thresholded Image", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours of the letters in the image
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Display the contours for debugging
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255), thickness=cv2.FILLED)
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Extract the coordinates of the contours
    ucb_coords = []
    for contour in contours:
        for point in contour:
            ucb_coords.append([point[0][0], point[0][1]])

    # Print out ucb_coords for debugging
    print("ucb_coords:", ucb_coords)

    # Convert ucb_coords to numpy array and transpose it
    ucb_coords = np.vstack(ucb_coords).T

    # Print out ucb_coords after conversion for debugging
    print("ucb_coords after conversion:", ucb_coords)

    # Transpose the coordinates array to match the expected shape (2, N)
    ucb = ucb_coords

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=ucb, basis="ij", cameraslm=fs
    )
    # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    phase = hologram.extract_phase()
    slm.write(phase, settle=True)
    cam_plot()


# integrate Intensity
def integrate_intensity():
    xlist = np.arange(650, 750, 25)  # Get the coordinates for one edge
    ylist = np.arange(340, 440, 25)
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack(
        (xgrid.ravel(), ygrid.ravel())
    )  # Make an array of points in a grid
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=square, basis="ij", cameraslm=fs
    )

    # Precondition computationally.
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    phase = hologram.extract_phase()
    # for ind in range(phase.shape[0]):
    #     for jnd in range(phase.shape[1]):
    #         phase[ind, jnd] *= np.dot((ind, jnd), (0.5, 0))
    # print(phase)
    # phase *= np.exp(1j * np.dot(np.meshgrid(phase.shape), (10,15))), 855,502.5
    slm.write(phase, settle=True)
    cam_plot()
    cam.set_exposure(0.0001)
    img = cam.get_image()
    # Define the region of interest (ROI) around the center spot and compute intensity
    x, y = 700, 390  # Center spot coordinates, adjust as necessary
    roi_size = 110  # Size of the region around the center spot to analyze
    center_intensity = img[
        y - roi_size : y + roi_size, x - roi_size : x + roi_size
    ].sum()
    roi_size = 50
    x, y = 830, 540
    # Compute intensities for each spot in the grid
    total_intensity = img[
        y - roi_size : y + roi_size, x - roi_size : x + roi_size
    ].sum()
    # Print or plot the intensities for analysis
    print("Center spot intensity:", center_intensity)
    print(f"all intensity: {total_intensity}")


# region dynamical optical tweezers
def animate_wavefront_shifts():
    initial_phase = np.load("initial_phase.npy")  # Load the saved phase

    frames = []
    shifts = np.linspace(
        0, 0.3, num=10
    )  # Define the range and number of steps for the shifts

    # Define the square path
    path = example_library.tweezers_square_path(shifts)

    for shift_x, shift_y in path:
        shifted_phase = example_library.shift_phase(
            np.copy(initial_phase), shift_x=shift_x, shift_y=shift_y
        )
        slm.write(shifted_phase, settle=True)
        cam.set_exposure(0.0001)
        im = cam.get_image()
        if im is not None:
            plt.imshow(im)  # Use custom colormap
            plt.draw()
            # Save the current figure to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image = imageio.imread(buffer)
            frames.append(image)
            buffer.close()
            plt.pause(0.01)  # Small pause to update the plot

    # Save frames as a GIF
    imageio.mimsave("wavefront_shift_animation.gif", frames, fps=5)
    print("Animation saved as wavefront_shift_animation.gif")

    # # Ensure frames are in the correct format for imageio
    # frames = [np.array(frame, dtype=np.uint8) for frame in frames]
    # imageio.mimsave('wavefront_shift_animation_1.gif', frames, fps=30)


def real_time_dynamical_tweezers():
    # Define parameters for the sqaure array
    # xlist = np.arange(750, 850, 25)  # Get the coordinates for one edge
    # ylist = np.arange(440, 540, 25)
    # xgrid, ygrid = np.meshgrid(xlist, ylist)
    # square = np.vstack((xgrid.ravel(), ygrid.ravel()))  # Make an array of points in a grid
    # hologram = SpotHologram(shape=(2048, 2048), spot_vectors=square, basis='ij', cameraslm=fs)

    #    # Define parameters for the circle
    #     center = (850, 540)  # Center of the circle
    #     radius = 100  # Radius of the circle

    #     # Generate points within the circle using polar coordinates
    #     num_points = 30  # Number of points to generate
    #     theta = np.linspace(0, 2*np.pi, num_points)  # Angle values
    #     x_circle = center[0] + radius * np.cos(theta)  # X coordinates
    #     y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

    #     # Convert to grid format if needed
    #     circle = np.vstack((x_circle, y_circle))

    #     hologram = SpotHologram(shape=(2048, 2048), spot_vectors=circle, basis='ij', cameraslm=fs)

    # Define parameters for the circles
    # center = (800, 550)  # Center of the circle
    # radii = np.linspace(10, 100, num=4)  # Adjust the number of circles as needed
    # circle_points = []
    # for radius in radii:
    #     num_points = int(2 * np.pi * radius / 50)  # Adjust the number of points based on the radius

    #     # Generate points within the circle using polar coordinates
    #     theta = np.linspace(0, 2*np.pi, num_points)  # Angle values
    #     x_circle = center[0] + radius * np.cos(theta)  # X coordinates
    #     y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

    #     # Convert to grid format for the current circle
    #     circle = np.vstack((x_circle, y_circle))
    #     circle_points.append(circle)
    # # Combine the points of all circles
    # circles = np.concatenate(circle_points, axis=1)
    # hologram = SpotHologram(shape=(2048, 2048), spot_vectors=circles, basis='ij', cameraslm=fs)
    # Define parameters for the spiral pattern
    center = (750, 540)
    num_turns = 6
    num_points = 60
    theta = np.linspace(0, 2 * np.pi * num_turns, num_points)
    radius = np.linspace(0, 100, num_points)  # Adjust radius array to match num_points

    # Generate spiral coordinates
    x_spiral = center[0] + radius * np.cos(theta)
    y_spiral = center[1] + radius * np.sin(theta)

    # Convert to grid format
    spiral = np.vstack((x_spiral, y_spiral))

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=spiral, basis="ij", cameraslm=fs
    )

    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=30,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    initial_phase = hologram.extract_phase()
    slm.write(initial_phase, settle=True)
    # time.sleep(10)
    # # Generate the square path for the tweezers
    # x1 = -0.5  # Lower bound of shift range
    # x2 = 0.7  # Upper bound of shift range
    # num_points = 31  # Number of points
    # path = example_library.square_tweezer_path(x1, x2, num_points)

    # # Generate the circular path for the tweezers
    # num_points = 100  # Number of points in the path
    # radius = 0.5  # Maximum distance from the center
    # path = example_library.circular_tweezer_path(num_points, radius)

    # Generate the spiral path for the tweezers
    num_points = 1000  # Number of points in the path
    max_radius = 1  # Maximum distance from the center
    num_turns = 6  # Number of turns in the spiral
    path = example_library.spiral_tweezer_path(num_points, max_radius, num_turns)

    # Precompute shifted phases with verbose output
    # shifted_phases = []
    # print("Precomputing phase shifts...")
    # for idx, (shift_x, shift_y) in enumerate(path):
    #     shifted_phase = shift_phase(np.copy(initial_phase), shift_x=shift_x, shift_y=shift_y)
    #     shifted_phases.append(shifted_phase)
    #     print(f"Precomputed phase shift {idx+1}/{len(path)}: shift_x={shift_x}, shift_y={shift_y}")

    # try:
    #     while True:  # Loop indefinitely until interrupted
    #         for idx, shifted_phase in enumerate(shifted_phases):
    #             slm.write(shifted_phase, settle=True)  # Adjust 'settle=True' to 'settle=False' if applicable
    #             print(f"Written phase shift {idx+1}/{len(shifted_phases)} to SLM")
    #             # time.sleep(0.01)  # Small delay to allow for KeyboardInterrupt
    #         print("Completed one loop of the square path, starting again...")
    frames = []
    for shift_x, shift_y in path:
        shifted_phase = example_library.shift_phase(
            np.copy(initial_phase), shift_x=shift_x, shift_y=shift_y
        )
        slm.write(shifted_phase, settle=True)
        cam.set_exposure(0.0001)
        im = cam.get_image()
        if im is not None:
            plt.imshow(im)  # Use custom colormap
            plt.draw()
            # Save the current figure to a buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image = imageio.imread(buffer)
            frames.append(image)
            buffer.close()
            plt.pause(0.01)  # Small pause to update the plot

    # Save frames as a GIF
    imageio.mimsave("spiral_animation_1.gif", frames, fps=30)
    print("Animation saved as spira_animation.gif")

    # try:
    #     while True:  # Loop indefinitely until interrupted
    #         for shift_x, shift_y in path:
    #             shifted_phase = example_library.shift_phase(np.copy(initial_phase), shift_x, shift_y)
    #             # plot_phase(shifted_phase)
    #             slm.write(shifted_phase, settle=False)
    #             # time.sleep(0.03)
    #         print("Completed the loop, starting again...")

    # except KeyboardInterrupt:
    #     print("Real-time dynamical tweezers operation interrupted and stopped.")


def selected_dynamical_tweezers():
    # Define parameters for the circle
    center = (550, 740)  # Center of the circle
    radius = 100  # Radius of the bigger circle

    # Generate points within the circle using polar coordinates
    num_points = 12  # Number of points to generate
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angle values
    x_circle = center[0] + radius * np.cos(theta)  # X coordinates
    y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

    # Combine x and y coordinates into an array of spot indices
    spot_indices = np.vstack((y_circle, x_circle)).astype(int)

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=spot_indices, basis="ij", cameraslm=fs
    )
    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    initial_phase = hologram.extract_phase()

    # Initialize the figure and axes outside the loop
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    frames = []

    try:
        while True:
            for angle in range(0, 360, 30):  # Rotate in steps of 30 degrees
                blaze_vector = (np.cos(np.radians(angle)), np.sin(np.radians(angle)))

                # Update phase with live rotation
                delta_phase = toolbox.phase.blaze(
                    grid=slm, vector=blaze_vector, offset=0
                )
                phase = initial_phase + delta_phase

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

                plt.show()
    finally:
        print("Real-time dynamical tweezers operation stopped.")


def camp2phase_calibration():
    # Define parameters for the circle
    center = (550, 740)  # Center of the circle
    radius = 30  # Radius of the bigger circle

    # Generate points within the circle using polar coordinates
    num_points = 3  # Number of points to generate
    theta = np.linspace(0, 2 * np.pi, num_points)  # Angle values
    x_circle = center[0] + radius * np.cos(theta)  # X coordinates
    y_circle = center[1] + radius * np.sin(theta)  # Y coordinates

    # Combine x and y coordinates into an array of spot indices
    spot_indices = np.vstack((y_circle, x_circle)).astype(int)

    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=spot_indices, basis="ij", cameraslm=fs
    )
    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    initial_phase = hologram.extract_phase()

    nuvu_pixel_coords = [
        [131.144, 129.272],
        [261.477, 205.335],
        [435.139, 304.013],
        [310.023, 187.942],
        [444.169, 463.787],
    ]
    # Convert the list to a numpy array
    nuvu_pixel_coords_array = np.array(nuvu_pixel_coords)
    # calibration Path
    # path = example_library.triangular(0.6,0.6)
    # Calibrate the coordinates from Nuvu to Thorlabs system
    thorcam_coords = example_library.nuvu2thorcam_calibration(nuvu_pixel_coords_array)
    print(thorcam_coords)
    # calculate the phase shifts based on camera pixels cange
    phase_shifts = example_library.calculate_phaseshifts(thorcam_coords)
    for shift_x, shift_y in phase_shifts:
        shifted_phase = example_library.shift_phase(
            np.copy(initial_phase), shift_x=shift_x, shift_y=shift_y
        )
        print(shift_x)
        print(shift_y)
        slm.write(shifted_phase, settle=True)
        cam_plot()


# Load the saved NV coordinates and radii from the .npz file

# def load_nv_coords(
#     file_path="slmsuite/nv_blob_detection/nv_blob_filtered_162nvs_ref.npz",
#     x_min=0,
#     x_max=250,
#     y_min=0,
#     y_max=250,
# ):
#     """
#     Load the NV coordinates and radii from a .npz file and remove outliers based on min/max thresholds.

#     Args:
#     file_path: The file path to the .npz file.
#     x_min: Minimum allowed value for x-coordinate.
#     x_max: Maximum allowed value for x-coordinate.
#     y_min: Minimum allowed value for y-coordinate.
#     y_max: Maximum allowed value for y-coordinate.

#     Returns:
#     nv_coordinates_clean: The NV coordinates with outliers removed based on the given thresholds.
#     """
#     data = np.load(file_path)
#     nv_coordinates = data["nv_coordinates"]

#     # Create a mask based on the min/max thresholds for x and y
#     mask = (
#         (nv_coordinates[:, 0] >= x_min)
#         & (nv_coordinates[:, 0] <= x_max)
#         & (nv_coordinates[:, 1] >= y_min)
#         & (nv_coordinates[:, 1] <= y_max)
#     )

#     # Filter out coordinates that fall outside the thresholds
#     nv_coordinates_clean = nv_coordinates[mask]

#     print(f"Filtered {np.sum(~mask)} outliers.")

#     return nv_coordinates_clean


def load_nv_coords(
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_162nvs_ref.npz",
    # file_path="slmsuite/nv_blob_detection/nv_coords_integras_counts_162nvs.npz",
    file_path="slmsuite/nv_blob_detection/nv_coords_integras_counts_filtered.npz",
):
    data = np.load(file_path)
    nv_coordinates = data["nv_coordinates"]
    spot_weights = data["spot_weights"]
    return nv_coordinates, spot_weights


# Set the threshold for x and y coordinates, assuming the SLM has a 2048x2048 pixel grid
nuvu_pixel_coords, spot_weights = load_nv_coords()
print(f"Total NV coordinates: {len(nuvu_pixel_coords)}")
print(f"Total spot weigths: {len(spot_weights)}")
thorcam_coords = example_library.nuvu2thorcam_calibration(nuvu_pixel_coords).T

def nvs_demo():
    hologram = SpotHologram(
        shape=(4096, 2048),
        spot_vectors=thorcam_coords,
        basis="ij",
        spot_amp=spot_weights,
        cameraslm=fs,
    )
    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=30,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )

    initial_phase = hologram.extract_phase()
    # Define the path to save the phase data
    path = r"C:\Users\matth\GitHub\dioptric\slmsuite\Initial_phase"
    filename = "slm_phase_162nvs.npy"
    # Save the phase data
    save(initial_phase, path, filename)
    slm.write(initial_phase, settle=True)
    cam_plot()


def nvs_phase():
    # phase = np.load(
    #     r"C:\Users\matth\GitHub\dioptric\slmsuite\Initial_phase\initial_phase.npy"
    # )
    phase = np.load("slmsuite\optimized_phases\optimized_phase_nv_0.npy")
    slm.write(phase, settle=True)
    cam_plot()


def initial_phase():
    xlist = np.arange(650, 750, 10)  # X coordinates for the grid
    ylist = np.arange(340, 440, 10)  # Y coordinates for the grid
    xgrid, ygrid = np.meshgrid(xlist, ylist)
    square = np.vstack((xgrid.ravel(), ygrid.ravel()))
    hologram = SpotHologram(
        shape=(2048, 2048), spot_vectors=square, basis="ij", cameraslm=fs
    )

    # Precondition computationally
    hologram.optimize(
        "WGS-Kim",
        maxiter=20,
        feedback="computational_spot",
        stat_groups=["computational_spot"],
    )
    initial_phase = hologram.extract_phase()
    # Define the path to save the phase data
    path = r"C:\Users\matth\GitHub\dioptric\slmsuite\Initial_phase"
    filename = "initial_phase.npy"
    # Save the phase data
    save(initial_phase, path, filename)


# Define the save function
def save(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, filename), data)


def optimize_array():
    # Import saved the phase data
    path = r"C:\Users\Saroj Chand\Documents\slm_phase"
    filename = "initial_phase.npy"
    initial_phase = np.load(os.path.join(path, filename))
    optimized_coords = []
    phase_shifts = example_library.calculate_phaseshifts(thorcam_coords)

    print("Shape of phase_shifts:", phase_shifts.shape)
    print("Shape of thorcam_coords:", thorcam_coords.shape)
    print("phase shifts:", phase_shifts)
    print("thorcam coords:", thorcam_coords)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.ion()

    for (shift_x, shift_y), (x0, y0) in zip(phase_shifts, thorcam_coords):
        shifted_phase = example_library.shift_phase(
            np.copy(initial_phase), shift_x=shift_x, shift_y=shift_y
        )
        slm.write(shifted_phase, settle=True)
        im = cam.get_image()

        fit_params = example_library.fit_gaussian2d(im, x0, y0)
        if fit_params is not None:
            x0, y0, a, c, wx, wy, wxy = fit_params
            optimized_coords.append((x0, y0))
            print(f"Fitted coordinates: x={x0}, y={y0}")

            ax[0].clear()
            ax[0].imshow(im)
            ax[0].set_title("Captured Image")

            x = np.linspace(0, im.shape[1] - 1, im.shape[1])
            y = np.linspace(0, im.shape[0] - 1, im.shape[0])
            x, y = np.meshgrid(x, y)
            fitted_data = example_library.gaussian2d((x, y), *fit_params).reshape(
                im.shape
            )

            ax[1].clear()
            ax[1].imshow(fitted_data)
            ax[1].set_title("Fitted Gaussian")

            plt.pause(0.01)

    plt.ioff()
    print("Optimized coordinates list:")
    print(optimized_coords)


def plot_laguerre_gaussian_phase():
    # Assuming `slm` is a valid instance and `toolbox` is properly defined with the phase method
    laguerre_gaussian_phase = toolbox.phase.laguerre_gaussian(
        grid=slm, l=12, p=0
    )  # Example values for l and p
    plot_phase(laguerre_gaussian_phase)


# region run funtions
try:
    slm = ThorSLM(serialNumber="00429430")
    cam = ThorCam(serial="26438", verbose=True)
    # cam = tb.get_server_thorcam()
    # slm = tb.get_server_thorslm()

    fs = FourierSLM(cam, slm)

    # blaze()
    # fourier_calibration()
    load_fourier_calibration()
    # test_wavefront_calibration()
    # wavefront_calibration()
    # load_wavefront_calibration()
    # fs.process_wavefront_calibration(r2_threshold=.9, smooth=True, plot=True)
    # square_array()
    # square_array_cirle()
    # save_initial_phase()
    # animate_wavefront_shifts()
    real_time_dynamical_tweezers()
    # selected_dynamical_tweezers()
    # camp2phase_calibration()
    # initial_phase()
    # optimize_array()
    # plot_laguerre_gaussian_phase()
    nvs_demo()
    # nvs_phase()
    # circles()
    calibration_triangle()
    # circle_pattern()
    # smiley()
    # scatter_pattern()
    # UCB_pattern()
    # pattern_from_image()
    # cam_plot()
    # integrate_intensity()

finally:
    print("Closing")
    # After you're done using the slm and camera
    slm.close_window()
    slm.close_device()
    cam.close()
    # ThorCam.close_sdk()
# endregions
