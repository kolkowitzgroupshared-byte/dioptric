import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl


# Define the 2D Gaussian function
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


# Fit a 2D Gaussian to a local region of the image data
def fit_gaussian_2d_local(image, center, size=20):
    """Fit a 2D Gaussian to a local region around the initial peak coordinates.

    Args:
        image (ndarray): 2D image array.
        center (tuple): Initial center (x, y) of the peak.
        size (int): Size of the local region to consider for the fit.

    Returns:
        tuple: Fitted coordinates (x, y) and Gaussian parameters.
    """
    x0, y0 = center
    x_min, x_max = int(x0 - size), int(x0 + size)
    y_min, y_max = int(y0 - size), int(y0 + size)

    # Extract the local region of the image
    local_image = image[y_min:y_max, x_min:x_max]

    # Meshgrid for the local region
    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    x, y = np.meshgrid(x, y)

    # Initial guess for the Gaussian fit
    initial_guess = (
        local_image.max(),
        x0,
        y0,
        1,
        1,
        0,
        np.min(local_image),
    )

    # Perform the Gaussian fit
    popt, _ = curve_fit(gaussian_2d, (x, y), local_image.ravel(), p0=initial_guess)

    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt
    return (round(xo, 3), round(yo, 3)), popt


def plot_fitting(
    image, centers, optimized_coords, size=20, colormap="hot", vmin=None, vmax=None
):
    """Plot the original image with the initial and optimized peak coordinates."""
    fig, ax = plt.subplots()
    img_plot = kpl.imshow(ax, image, cbar_label="Photons")
    # Plot initial peaks
    ax.scatter(
        centers[:, 0], centers[:, 1], c="black", marker="x", label="Initial Peaks"
    )
    # Plot optimized peaks
    opt_x = [coord[0] for coord in optimized_coords]
    opt_y = [coord[1] for coord in optimized_coords]
    ax.scatter(opt_x, opt_y, c="blue", marker="o", label="Optimized Peaks")

    ax.legend()

    plt.title("2D Gaussian Fit: Initial vs Optimized Peaks")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    # Automatically save the plot using the same approach for file paths
    plt.show(block=True)


# Example usage
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Load the image array (replace this with your own method for loading image data)
    data = dm.get_raw_data(
        file_stem="2026_01_09-18_19_55-johnson-nv0_2025_10_21", load_npz=True
    )

    img_array = np.array(data["img_array"])
    fig, ax = plt.subplots()
    img_plot = kpl.imshow(ax, img_array, cbar_label="Photons")
    plt.show(block=True)
    # List of initial peak yellow coordinates
    initial_peaks = np.array([[223.098, 240.604], [190.265, 20.887], [18.737, 157.715]])
    # red calibration
    # initial_peaks = np.array([[123.486, 9.115], [16.951, 242.751], [237.556, 240.186]])
    # gree calibration
    # initial_peaks = np.array([[27.805, 30.303], [138.235, 235.17], [228.655, 15.301]])
    # Fit Gaussian to each peak
    optimized_coords = []
    for peak in initial_peaks:
        coords, _ = fit_gaussian_2d_local(img_array, peak, size=7)
        optimized_coords.append(coords)

    optimized_coords = np.array(optimized_coords)
    # Print optimized peak coordinates rounded to three digits
    print("Optimized peak coordinates (3 digits):", optimized_coords.tolist())
    # Plot the fitting results and save the figure
    plot_fitting(img_array, initial_peaks, optimized_coords)
