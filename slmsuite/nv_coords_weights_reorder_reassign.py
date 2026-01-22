import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from skimage.draw import disk

# from tabulate import tabulate
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


# Function to fit a 2D Gaussian around NV coordinates
def fit_gaussian(image, coord, window_size=2):
    x0, y0 = coord
    img_shape_y, img_shape_x = image.shape

    # Ensure the window is within image bounds
    x_min = max(int(x0 - window_size), 0)
    x_max = min(int(x0 + window_size + 1), img_shape_x)
    y_min = max(int(y0 - window_size), 0)
    y_max = min(int(y0 + window_size + 1), img_shape_y)

    if (x_max - x_min) <= 1 or (y_max - y_min) <= 1:
        print(
            f"Invalid cutout for NV at ({x0}, {y0}): Region too small or out of bounds"
        )
        return x0, y0

    # Extract cutout and mesh grid
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    x, y = np.meshgrid(x_range, y_range)
    image_cutout = image[y_min:y_max, x_min:x_max]

    # Check for valid cutout size
    if image_cutout.size == 0:
        print(f"Zero-size cutout for NV at ({x0}, {y0})")
        return x0, y0

    # Normalize the image cutout
    image_cutout = (image_cutout - np.min(image_cutout)) / (
        np.max(image_cutout) - np.min(image_cutout)
    )

    # Initial guess parameters
    initial_guess = (1, x0, y0, 3, 3, 0, 0)  # Amplitude normalized to 1

    try:
        # Apply bounds to avoid unreasonable parameter values
        bounds = (
            (0, x_min, y_min, 0, 0, -np.pi, 0),  # Lower bounds
            (np.inf, x_max, y_max, np.inf, np.inf, np.pi, np.inf),  # Upper bounds
        )

        # Perform the Gaussian fit
        popt, _ = curve_fit(
            gaussian_2d, (x, y), image_cutout.ravel(), p0=initial_guess, bounds=bounds
        )
        amplitude, fitted_x, fitted_y, _, _, _, _ = popt

        return fitted_x, fitted_y, amplitude

    except Exception as e:
        print(f"Fit failed for NV at ({x0}, {y0}): {e}")
        return x0, y0, 0


def integrate_intensity(image_array, nv_coords, sigma):
    """
    Integrate the intensity around each NV coordinate within a circular region
    defined by sigma, with a Gaussian weighting if needed.
    """
    intensities = []
    for coord in nv_coords:
        # Define a larger radius to ensure full capture of intensity around bright spots
        rr, cc = disk((coord[0], coord[1]), radius=sigma, shape=image_array.shape)

        # Integrate (sum) the intensity values within the disk
        intensity = np.sum(image_array[rr, cc])

        # Append integrated intensity to the list
        intensities.append(intensity)
    return intensities


def remove_outliers(intensities, nv_coords):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(intensities, 25)
    Q3 = np.percentile(intensities, 75)
    IQR = Q3 - Q1

    # Define bounds for identifying outliers
    lower_bound = Q1 - 1.0 * IQR
    upper_bound = Q3 + 6.5 * IQR
    # lower_bound = 10
    # upper_bound = 100

    # Filter out the outliers and corresponding NV coordinates
    filtered_intensities = []
    filtered_nv_coords = []

    for intensity, coord in zip(intensities, nv_coords):
        if lower_bound <= intensity <= upper_bound:
            filtered_intensities.append(intensity)
            filtered_nv_coords.append(coord)

    return filtered_intensities, filtered_nv_coords


def remove_manual_indices(nv_coords, indices_to_remove):
    """Remove NVs based on manually specified indices"""
    return [
        coord for idx, coord in enumerate(nv_coords) if idx not in indices_to_remove
    ]


def filter_and_reorder_nv_coords(
    nv_coordinates, integrated_intensities, reference_nv, min_distance=3
):
    """
    Filters NV coordinates based on distance from each other and reorders based on distance from a reference NV.

    """
    nv_coords = [reference_nv]  # Store as list for later operations
    # Find the closest NV to the reference_nv in case it's not an exact match
    distances_to_ref = np.linalg.norm(
        np.array(nv_coordinates) - np.array(reference_nv), axis=1
    )
    closest_index = np.argmin(distances_to_ref)  # Get the index of the closest match
    reference_nv = nv_coordinates[closest_index]  # Use this as the reference
    included_indices = [closest_index]  # Track included indices

    # Filter NV coordinates based on minimum distance
    for idx, coord in enumerate(nv_coordinates):
        keep_coord = True
        for existing_coord in nv_coords:
            distance = np.linalg.norm(np.array(existing_coord) - np.array(coord))
            if distance < min_distance:
                keep_coord = False
                break
        if keep_coord:
            nv_coords.append(coord)
            included_indices.append(idx)
            # intensities.append(integrated_intensities[idx])  # Store matching intensity
    print(included_indices)
    # Reorder based on distance to the reference NV
    distances = [
        np.linalg.norm(np.array(coord) - np.array(reference_nv)) for coord in nv_coords
    ]
    sorted_indices = np.argsort(distances)
    reordered_coords = [nv_coords[idx] for idx in sorted_indices]
    reordered_intensities = [integrated_intensities[idx] for idx in sorted_indices]

    return reordered_coords, reordered_intensities, included_indices


def sigmoid_weights(intensities, threshold, beta=1):
    weights = np.exp(beta * (intensities - threshold))
    return weights / np.max(weights)  # Normalize the weights


def linear_weights(intensities, alpha=1):
    weights = 1 / np.power(intensities, alpha)
    weights = weights / np.max(weights)  # Normalize to avoid extreme values
    return weights


def non_linear_weights_adjusted(intensities, alpha=1, beta=0.5, threshold=0.5):
    # Normalize the intensities between 0 and 1
    norm_intensities = intensities / np.max(intensities)

    # Apply a non-linear transformation to only the lower intensities
    weights = np.where(
        norm_intensities > threshold,
        1,  # Keep bright NVs the same
        1
        / (1 + np.exp(-beta * (norm_intensities - threshold)))
        ** alpha,  # Non-linear scaling for low intensities
    )

    # Ensure that the weights are normalized
    weights = weights / np.max(weights)

    return weights


# Save the results to a file
def save_results(nv_coordinates, updated_spot_weights, filename):
    # Ensure the directory exists
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)  # Create the directory if it doesn't exist

    # Save the data to a .npz file
    np.savez(
        filename,
        nv_coordinates=nv_coordinates,
        # integrated_counts=integrated_intensities,
        # spot_weights=spot_weights,
        # nv_powers=nv_powers,
        updated_spot_weights=updated_spot_weights,
    )


def filter_by_snr(snr_list, threshold=0.5):
    """Filter out NVs with SNR below the threshold."""
    return [i for i, snr in enumerate(snr_list) if snr >= threshold]


def load_nv_coords(
    # file_path="slmsuite/nv_blob_detection/nv_blob_filtered_77nvs_new.npz",
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_240nvs.npz",
):
    data = np.load(file_path)
    print(data.keys())
    nv_coordinates = data["nv_coordinates"]
    # spot_weights = data["spot_weights"]
    spot_weights = data["updated_spot_weights"]
    # spot_weights = data["integrated_counts"]
    # spot_weights = data["integrated_counts"]
    return nv_coordinates, spot_weights


def load_nv_weights(file_path="optimal_separation_and_goodness.txt"):
    # Load data, skipping the header row
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    # Extract the step values for separation
    nv_weights = data[:, 2]  # Step Val (Separation) is the 3rd column (index 2)
    return nv_weights


def sigmoid_weight_update(
    fidelities, spot_weights, intensities, alpha=1, beta=10, fidelity_threshold=0.90
):
    # Normalize intensities between 0 and 1
    norm_intensities = intensities / np.max(intensities)

    # Initialize updated weights as 1 (i.e., no change for high-fidelity NVs)
    updated_weights = np.copy(spot_weights)

    # Loop over each NV and update weights for those with fidelity < fidelity_threshold
    for i, fidelity in enumerate(fidelities):
        if fidelity < fidelity_threshold:
            # Use a sigmoid to adjust the weight based on intensity
            updated_weights[i] = (
                1 / (1 + np.exp(-beta * (norm_intensities[i]))) ** alpha
            )

    # Normalize the updated weights to avoid extreme values
    updated_weights = updated_weights / np.max(updated_weights)

    return updated_weights


def manual_sigmoid_weight_update(
    spot_weights, intensities, alpha, beta, update_indices
):
    updated_spot_weights = (
        spot_weights.copy()
    )  # Make a copy to avoid mutating the original list
    norm_intensities = intensities / np.max(intensities)
    for idx in update_indices:
        print(f"NV Index {idx}: Weight before update: {updated_spot_weights[idx]}")

        # Apply the sigmoid weight update for the specific NV
        weight_update = 1 / (1 + np.exp(-beta * (norm_intensities[idx]))) ** alpha
        updated_spot_weights[idx] = weight_update  # Update weight for this NV

        print(f"NV Index {idx}: Weight after update: {updated_spot_weights[idx]}")

    return updated_spot_weights


# Adjust weights based on SNR values
def adjust_weights_sigmoid(spot_weights, snr_values, alpha=1.0, beta=0.001):
    """Apply sigmoid adjustment to spot weights based on SNR values."""
    updated_weights = np.copy(spot_weights)
    for i, value in enumerate(snr_values):
        if value < 0.9:
            # Sigmoid-based weight adjustment
            updated_weights[i] = 1 / (1 + np.exp(-beta * (value - alpha)))
    return updated_weights


def filter_by_peak_intensity(fitted_data, threshold=0.5):
    filtered_coords = []
    filtered_intensities = []

    for x, y, intensity in fitted_data:
        if intensity >= threshold:
            filtered_coords.append((x, y))
            filtered_intensities.append(intensity)

    return filtered_coords, filtered_intensities


def adjust_aom_voltage_for_slm(nv_amp, aom_voltage, power_law_params):
    nv_amp = np.array(nv_amp)
    a, b, c = power_law_params

    aom_voltages = nv_amp * aom_voltage

    nv_powers = a * (aom_voltages**b) + c
    scaled_nv_powers = nv_powers / (len(nv_powers))
    # Normalize powers across all spots
    total_power = np.sum(scaled_nv_powers)
    nv_weights = nv_powers / total_power
    # Compute adjusted AOM voltage for the total power
    adjusted_aom_voltage = ((total_power - c) / a) ** (1 / b)
    return nv_weights, adjusted_aom_voltage


def curve_extreme_weights_simple(weights, scaling_factor=1.0):
    median = np.median(weights)

    curved_weights = [1 / (1 + np.exp(-scaling_factor * (w - median))) for w in weights]

    return curved_weights


def curve_inverse_counts(counts, scaling_factor=0.5):
    median_count = np.median(counts)
    adjusted_weights = np.exp(-scaling_factor * (counts / median_count))
    adjusted_weights /= np.max(adjusted_weights)
    return adjusted_weights


def select_half_left_side_nvs_and_plot(nv_coordinates):
    # Filter NVs on the left side (x < median x)
    median_x = np.median(nv_coordinates[:, 0])
    left_side_indices = [
        i for i, coord in enumerate(nv_coordinates) if coord[0] < median_x
    ]

    # Randomly select half of the NVs from the left side
    print(f"Selected {len(left_side_indices)} NVs from the left side.")

    # Plot distribution
    plt.figure(figsize=(10, 7))

    # Plot all NVs
    plt.scatter(
        nv_coordinates[:, 0], nv_coordinates[:, 1], color="gray", label="All NVs"
    )

    # Highlight left-side NVs
    left_coords = nv_coordinates[left_side_indices]
    plt.scatter(
        left_coords[:, 0], left_coords[:, 1], color="blue", label="Left Side NVs"
    )

    # Add median line
    plt.axvline(
        median_x, color="green", linestyle="--", label=f"Median X = {median_x:.2f}"
    )

    # Labels and legend
    plt.title("NV Distribution with Left Side Selection", fontsize=16)
    plt.xlabel("X Coordinate", fontsize=14)
    plt.ylabel("Y Coordinate", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    return


# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Parameters
    remove_outliers_flag = False  # Set this flag to enable/disable outlier removal
    reorder_coords_flag = True  # Set this flag to enable/disable reordering of NVs
    data = dm.get_raw_data(
        file_stem="2025_10_30-16_04_15-johnson-nv0_2025_10_21", load_npz=True
    )
    img_array = np.array(data["ref_img_array"])
    # img_array = data["img_array"]
    nv_coordinates, spot_weights = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_327nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_308nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_254nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_151nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_136nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_312nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_230nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_223nvs_reordered.npz"
        file_path="slmsuite/nv_blob_detection/nv_blob_204nvs_reordered.npz"
    )
    # Convert coordinates to a standard format (lists of lists)
    # nv_coordinates = [[coord[0] - 3, coord[1] + 3] for coord in nv_coordinates]
    nv_coordinates = [list(coord) for coord in nv_coordinates]
    # Filter NV coordinates: Keep only those where both x and y are in [0, 250]
    nv_coordinates_filtered = [
        coord
        for coord in nv_coordinates
        if isinstance(coord, (list, tuple))
        and len(coord) == 2
        and all(2 <= x <= 248 for x in coord)
    ]

    # Ensure spot weights are filtered accordingly
    spot_weights_filtered = [
        weight
        for coord, weight in zip(nv_coordinates, spot_weights)
        if isinstance(coord, (list, tuple))
        and len(coord) == 2
        and all(2 <= x <= 248 for x in coord)
    ]

    # Replace original lists with filtered versions
    nv_coordinates = nv_coordinates_filtered
    spot_weights = spot_weights_filtered

    print(f"After filtering: {len(spot_weights)} NVs")

    # Filter and reorder NV coordinates based on reference NV
    # integrated_intensities = []
    sigma = 3.0
    reference_nv = [124.195, 127.341]
    filtered_reordered_coords, filtered_reordered_spot_weights, include_indices = (
        filter_and_reorder_nv_coords(
            nv_coordinates, spot_weights, reference_nv, min_distance=3
        )
    )
    print(len(filtered_reordered_coords))
    # filtered_reordered_coords = [
    #     [coord[0] - 5, coord[1] - 0] for coord in filter_and_reorder_nv_coords
    # ]

    # Integration over disk region around each NV coordinate
    # filtered_reordered_counts = []
    # integration_radius = 3.0
    # for coord in filtered_reordered_coords:
    #     x, y = coord[:2]  # Assuming `coord` contains at least two elements (y, x)
    #     rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
    #     sum_value = np.sum(img_array[rr, cc])
    #     filtered_reordered_counts.append(sum_value)

    # # calcualte spot weight  based on
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # filtered_reordered_spot_weights = calcualted_spot_weights
    # Manually remove NVs with specified indices
    indices_to_remove = []
    filtered_reordered_coords_0 = [
        coord
        for i, coord in enumerate(filtered_reordered_coords)
        if i not in indices_to_remove
    ]
    filtered_reordered_spot_weights_0 = [
        count
        for i, count in enumerate(filtered_reordered_spot_weights)
        if i not in indices_to_remove
    ]
    filtered_reordered_coords = filtered_reordered_coords_0
    filtered_reordered_spot_weights = filtered_reordered_spot_weights_0

    # print(filtered_reordered_coords)
    # print("Filter:", filtered_reordered_counts)
    # print("Filtered and Reordered NV Coordinates:", filtered_reordered_coords)
    # print("Filtered and Reordered NV Coordinates:", integrated_intensities)

    # Initialize lists to store the results
    # fitted_amplitudes = []
    # for coord in filtered_reordered_coords:
    #     fitted_x, fitted_y, amplitude = fit_gaussian(img_array, coord, window_size=2)
    #     fitted_amplitudes.append(amplitude)

    # fmt: off
    snr = ['0.078', '0.090', '0.019', '0.067', '0.015', '0.014', '0.013', '0.009', '0.046', '0.047', '0.050', '0.042', '-0.005', '0.065', '0.064', '0.052', '0.051', '0.002', '0.059', '0.050', '0.048', '0.072', '0.063', '0.017', '0.028', '0.006', '0.029', '0.088', '0.013', '0.046', '0.055', '0.078', '0.051', '0.029', '0.064', '0.049', '0.053', '0.055', '0.038', '0.030', '0.039', '0.035', '0.037', '0.019', '0.052', '0.036', '0.036', '0.085', '0.029', '0.045', '0.071', '0.041', '0.067', '0.025', '0.058', '0.030', '0.044', '0.094', '0.063', '0.010', '0.015', '0.064', '0.004', '0.013', '0.051', '0.067', '0.053', '-0.003', '0.037', '0.077', '0.052', '0.028', '0.054', '0.025', '-0.004', '0.008', '0.027', '0.003', '0.020', '0.055', '-0.009', '0.008', '0.051', '0.037', '0.045', '0.054', '0.011', '0.064', '0.025', '0.031', '0.043', '0.022', '0.057', '0.051', '0.093', '0.047', '0.044', '0.003', '0.069', '0.018', '0.006', '0.028', '0.018', '0.057', '0.067', '0.035', '0.024', '0.042', '0.065', '0.031', '0.036', '0.064', '0.004', '-0.001', '0.054', '0.009', '0.069', '0.038', '0.023', '0.015', '0.039', '-0.019', '0.042', '0.053', '0.102', '0.020', '0.069', '0.061', '0.002', '0.049', '0.002', '0.067', '0.082', '0.011', '0.037', '0.044', '0.076', '0.028', '0.063', '0.048', '0.082', '0.020', '0.029', '0.029', '-0.002', '0.047', '0.030', '0.044', '0.005', '0.056', '0.009', '0.019', '0.030', '0.071', '0.011', '0.096', '0.008', '0.088', '0.016', '0.051', '0.070', '0.033', '-0.021', '0.029', '0.062', '0.037', '0.014', '0.018', '0.052', '0.062', '0.029', '0.039', '0.035', '0.023', '0.037', '0.069', '0.029', '0.053', '0.078', '0.005', '0.037', '0.067', '0.011', '0.055', '0.067', '0.039', '0.028', '0.056', '0.077', '0.054', '0.053', '0.028', '0.088', '-0.014', '0.026', '0.032', '-0.014', '-0.010', '-0.002', '0.043', '0.054', '0.059', '0.069', '0.008', '0.064', '-0.003', '0.043', '0.016', '0.004', '0.072', '0.072', '0.005', '0.023', '0.037', '0.063', '0.064', '0.063', '0.035', '0.042', '0.080', '0.043', '0.052', '0.032', '0.062', '0.054', '0.034', '0.025', '0.085', '0.018', '0.015', '0.075', '0.061', '0.067', '0.036', '0.018', '0.025', '0.014', '0.053', '0.066', '0.057', '0.034', '0.049', '0.019', '0.000', '0.042', '0.059', '0.027', '-0.000', '0.062', '0.052', '0.019', '0.039', '0.046', '0.058', '-0.012', '0.080', '0.054', '0.082', '0.036', '-0.012', '0.083', '0.042', '0.004', '0.103', '0.064', '0.009', '0.083', '0.100', '0.029', '0.032', '-0.001', '-0.022', '0.034', '0.001', '0.037', '0.032', '0.059', '0.045', '0.014', '0.017', '-0.009', '0.005', '0.043', '0.089', '0.109', '-0.004', '0.069', '0.052', '0.010', '0.081', '0.004', '0.023', '0.006', '0.031', '0.054', '0.058', '0.055', '0.003', '0.071', '0.068', '0.037', '0.095', '-0.010', '0.009', '0.123', '0.077', '0.049', '0.061', '0.060', '0.006', '0.056', '0.076']
    ## 230 NVs
    snr = ['0.108', '0.121', '0.102', '0.092', '0.100', '0.096', '0.071', '0.130', '0.089', '0.138', '0.124', '0.116', '0.079', '0.058', '0.139', '0.102', '0.090', '0.062', '0.130', '0.104', '0.083', '0.130', '0.072', '0.033', '0.094', '0.071', '0.122', '0.121', '0.059', '0.135', '0.070', '0.009', '0.093', '0.126', '0.055', '0.038', '0.116', '0.011', '0.049', '0.083', '0.075', '0.107', '-0.005', '0.116', '0.102', '0.105', '0.104', '0.072', '0.117', '0.115', '0.116', '0.090', '0.084', '0.103', '0.088', '0.094', '0.101', '0.090', '0.044', '0.102', '0.103', '0.073', '0.074', '0.088', '0.106', '0.097', '0.082', '0.086', '0.086', '0.094', '0.039', '0.100', '0.095', '0.072', '0.075', '0.120', '0.045', '0.094', '0.125', '0.109', '0.087', '0.089', '0.108', '0.090', '0.035', '0.117', '0.123', '0.120', '0.081', '0.011', '0.070', '0.112', '0.074', '0.092', '0.004', '0.103', '0.089', '0.078', '0.111', '0.105', '0.033', '0.096', '0.123', '0.103', '0.101', '0.082', '0.110', '0.106', '0.031', '0.082', '0.097', '0.012', '0.123', '0.110', '0.098', '0.098', '0.107', '0.118', '0.071', '0.130', '0.091', '0.067', '0.061', '0.060', '0.107', '0.083', '0.050', '0.033', '0.121', '0.046', '0.046', '0.108', '0.021', '0.121', '0.126', '0.108', '0.114', '0.123', '0.163', '0.047', '0.041', '0.095', '0.108', '0.083', '0.099', '0.000', '0.110', '0.061', '0.104', '0.119', '0.034', '0.102', '0.110', '0.102', '0.057', '0.105', '0.109', '0.067', '0.108', '0.126', '0.143', '0.101', '0.114', '0.085', '0.145', '0.106', '0.062', '0.075', '0.130', '0.132', '0.048', '0.086', '0.131', '0.107', '0.119', '0.104', '0.070', '0.117', '0.054', '0.135', '0.111', '0.061', '0.102', '0.070', '0.058', '0.137', '0.120', '0.122', '0.090', '0.121', '0.100', '0.126', '0.049', '0.098', '0.154', '0.100', '0.093', '0.133', '0.090', '0.119', '0.137', '0.049', '0.048', '0.040', '0.030', '0.099', '0.105', '0.122', '0.054', '0.123', '0.112', '0.139', '0.113', '0.113', '0.035', '0.071', '0.082', '0.141', '0.093', '0.141', '0.142', '0.055', '0.128', '0.116', '0.105', '0.067', '0.092', '0.110', '0.049', '0.122']
    ## 223 NVs
    # snr = ['0.105', '0.125', '0.075', '0.106', '0.100', '0.120', '0.043', '0.084', '0.079', '0.123', '0.121', '0.121', '0.053', '0.037', '0.099', '0.095', '0.090', '0.047', '0.128', '0.105', '0.095', '0.093', '0.098', '0.014', '0.082', '0.088', '0.097', '0.133', '0.075', '0.135', '0.100', '0.090', '0.052', '0.038', '0.021', '0.120', '0.071', '0.068', '0.089', '0.111', '0.124', '0.104', '0.119', '0.065', '0.100', '0.087', '0.131', '0.122', '0.123', '0.089', '0.070', '0.124', '0.086', '0.106', '0.008', '0.080', '0.088', '0.146', '0.076', '0.095', '0.105', '0.094', '0.103', '0.071', '0.088', '0.083', '0.090', '0.117', '0.098', '0.105', '0.084', '0.127', '0.072', '0.020', '0.112', '0.076', '0.118', '0.105', '0.104', '0.118', '0.095', '0.055', '0.126', '0.152', '0.133', '0.097', '0.091', '0.112', '0.088', '0.101', '0.094', '0.095', '0.124', '0.112', '0.123', '0.058', '0.086', '0.135', '0.083', '0.084', '0.066', '0.134', '0.117', '0.017', '0.081', '0.133', '0.123', '0.124', '0.110', '0.101', '0.127', '0.107', '0.093', '0.096', '0.065', '0.068', '0.032', '0.057', '0.103', '0.102', '0.060', '0.010', '0.152', '0.053', '0.022', '0.121', '0.014', '0.133', '0.066', '0.139', '0.102', '0.091', '0.119', '0.067', '0.049', '0.115', '0.130', '0.113', '0.107', '0.113', '0.065', '0.089', '0.107', '0.059', '0.101', '0.109', '0.117', '0.088', '0.088', '0.104', '0.052', '0.112', '0.121', '0.133', '0.127', '0.122', '0.110', '0.175', '0.096', '0.063', '0.098', '0.101', '0.140', '0.061', '0.084', '0.133', '0.143', '0.107', '0.077', '0.083', '0.138', '0.055', '0.115', '0.096', '0.074', '0.109', '0.089', '0.106', '0.094', '0.126', '0.130', '0.085', '0.110', '0.135', '0.115', '0.075', '0.140', '0.106', '0.091', '0.095', '0.132', '0.109', '0.127', '0.123', '0.055', '0.039', '0.059', '0.011', '0.101', '0.124', '0.120', '0.055', '0.129', '0.150', '0.132', '0.104', '0.148', '0.042', '0.088', '0.078', '0.129', '0.110', '0.112', '0.137', '0.071', '0.132', '0.136', '0.112', '0.095', '0.110', '0.102', '0.084', '0.136']
    snr = ['0.126', '0.127', '0.090', '0.095', '0.090', '0.114', '0.008', '0.130', '0.092', '0.124', '0.119', '0.138', '0.128', '0.080', '0.136', '0.099', '0.087', '0.094', '0.109', '0.090', '0.082', '0.093', '0.099', '0.000', '0.102', '0.078', '0.130', '0.109', '0.059', '0.099', '0.066', '0.106', '0.108', '0.024', '0.025', '0.114', '0.116', '0.134', '0.087', '0.086', '0.153', '0.070', '0.079', '0.091', '0.056', '0.054', '0.118', '0.117', '0.084', '0.085', '0.042', '0.098', '0.103', '0.110', '0.007', '0.062', '0.038', '0.085', '0.080', '0.091', '0.073', '0.112', '0.088', '0.037', '0.099', '0.083', '0.097', '0.086', '0.084', '0.087', '0.071', '0.083', '0.145', '0.054', '0.097', '0.086', '0.115', '0.105', '0.114', '0.118', '0.067', '0.061', '0.097', '0.122', '0.062', '0.079', '0.073', '0.114', '0.080', '0.113', '0.097', '0.084', '0.073', '0.115', '0.071', '0.045', '0.069', '0.112', '0.092', '0.103', '0.113', '0.103', '0.079', '0.038', '0.100', '0.098', '0.109', '0.121', '0.086', '0.102', '0.092', '0.111', '0.105', '0.121', '0.082', '0.051', '0.066', '0.066', '0.107', '0.094', '0.067', '0.027', '0.114', '0.046', '0.064', '0.129', '0.011', '0.112', '0.149', '0.100', '0.118', '0.147', '0.136', '0.057', '0.029', '0.112', '0.098', '0.106', '0.123', '0.123', '0.071', '0.079', '0.126', '0.096', '0.128', '0.086', '0.112', '0.103', '0.093', '0.132', '0.034', '0.123', '0.142', '0.131', '0.105', '0.141', '0.104', '0.196', '0.085', '0.025', '0.124', '0.135', '0.140', '0.061', '0.091', '0.128', '0.092', '0.137', '0.141', '0.078', '0.133', '0.074', '0.139', '0.088', '0.075', '0.111', '0.084', '0.100', '0.082', '0.111', '0.111', '0.102', '0.108', '0.122', '0.100', '0.062', '0.147', '0.145', '0.116', '0.095', '0.153', '0.117', '0.114', '0.144', '0.037', '0.048', '0.062', '0.015', '0.088', '0.139', '0.155', '0.067', '0.108', '0.081', '0.125', '0.120', '0.139', '0.029', '0.060', '0.118', '0.133', '0.093', '0.129', '0.135', '0.062', '0.138', '0.126', '0.113', '0.084', '0.102', '0.117', '0.068', '0.137']
    snr_float = [float(el) for el in snr]

    ### 204NVs
    snr_float = [0.132, 0.115, 0.095, 0.104, 0.124, 0.145, 0.107, 0.091, 0.177, 0.119, 0.136, 0.128, 0.077, 0.136, 0.103, 0.118, 0.105, 0.179, 0.099, 0.109, 0.163, 0.107, 0.112, 0.119, 0.125, 0.169, 0.064, 0.106, 0.104, 0.133, 0.108, 0.111, 0.132, 0.008, 0.099, 0.121, 0.143, 0.101, 0.111, 0.101, 0.156, 0.127, 0.132, 0.117, 0.139, 0.11, 0.135, 0.114, 0.09, 0.088, 0.136, 0.157, 0.075, 0.096, 0.143, 0.132, 0.094, 0.092, 0.123, 0.075, 0.111, 0.135, 0.117, 0.057, 0.14, 0.025, 0.104, 0.119, 0.101, 0.142, 0.114, 0.146, 0.095, 0.034, 0.148, 0.201, 0.094, 0.109, 0.087, 0.11, 0.121, 0.115, 0.138, 0.099, 0.139, 0.141, 0.114, 0.083, 0.109, 0.119, 0.121, 0.171, 0.121, 0.113, 0.088, 0.144, 0.132, 0.101, 0.135, 0.124, 0.122, 0.137, 0.116, 0.127, 0.067, 0.118, 0.134, 0.077, 0.133, 0.11, 0.111, 0.138, 0.069, 0.146, 0.125, 0.125, 0.136, 0.134, 0.14, 0.151, 0.063, 0.125, 0.115, 0.124, 0.145, 0.125, 0.127, 0.139, 0.163, 0.118, 0.122, 0.111, 0.156, 0.099, 0.09, 0.14, 0.138, 0.107, 0.142, 0.135, 0.155, 0.12, 0.221, 0.139, 0.149, 0.142, 0.12, 0.128, 0.106, 0.127, 0.112, 0.143, 0.162, 0.111, 0.18, 0.124, 0.118, 0.111, 0.138, 0.074, 0.067, 0.12, 0.088, 0.145, 0.154, 0.094, 0.122, 0.113, 0.099, 0.128, 0.129, 0.133, 0.126, 0.127, 0.11, 0.088, 0.12, 0.143, 0.08, 0.046, 0.16, 0.14, 0.149, 0.067, 0.155, 0.1, 0.155, 0.142, 0.151, 0.083, 0.146, 0.15, 0.091, 0.128, 0.167, 0.111, 0.135, 0.15, 0.147, 0.091, 0.105, 0.127, 0.019, 0.136]
    #308NVs
    pol_duration_list = [336, 336, 308, 308, 428, 428, 504, 504, 816, 816, 528, 528, 372, 372, 1060, 1060, 852, 852, 852, 852, 612, 612, 484, 484, 1120, 1120, 852, 852, 404, 404, 812, 812, 672, 672, 560, 560, 644, 644, 352, 352, 380, 380, 852, 852, 400, 400, 620, 620, 628, 628, 292, 292, 528, 528, 392, 392, 524, 524, 680, 680, 504, 504, 396, 396, 324, 324, 428, 428, 240, 240, 504, 504, 540, 540, 852, 852, 1188, 1188, 764, 764, 976, 976, 820, 820, 444, 444, 1100, 1100, 488, 488, 604, 604, 972, 972, 380, 380, 352, 352, 660, 660, 592, 592, 416, 416, 452, 452, 620, 620, 576, 576, 316, 316, 660, 660, 660, 660, 720, 720, 620, 620, 1024, 1024, 320, 320, 852, 852, 1396, 1396, 464, 464, 416, 416, 624, 624, 1008, 1008, 460, 460, 508, 508, 668, 668, 448, 448, 440, 440, 668, 668, 852, 852, 852, 852, 844, 844, 1048, 1048, 320, 320, 780, 780, 492, 492, 1476, 1476, 656, 656, 1064, 1064, 456, 456, 344, 344, 852, 852, 540, 540, 352, 352, 524, 524, 852, 852, 1156, 1156, 1388, 1388, 308, 308, 852, 852, 1360, 1360, 572, 572, 204, 204, 316, 316, 696, 696, 504, 504, 1332, 1332, 1012, 1012, 708, 708, 852, 852, 912, 912, 804, 804, 608, 608, 948, 948, 596, 596, 1256, 1256, 808, 808, 852, 852, 392, 392, 568, 568, 872, 872, 1268, 1268, 780, 780, 852, 852, 476, 476, 508, 508, 640, 640, 392, 392, 512, 512, 700, 700, 700, 700, 932, 932, 840, 840, 852, 852, 1248, 1248, 852, 852, 852, 852, 1444, 1444, 620, 620, 852, 852, 852, 852, 660, 660, 752, 752, 1052, 1052, 592, 592, 852, 852, 852, 852, 1248, 1248, 860, 860, 520, 520, 1320, 1320, 1096, 1096, 568, 568, 488, 488, 852, 852, 556, 556, 420, 420, 1192, 1192, 552, 552, 1032, 1032, 508, 508, 1268, 1268, 872, 872, 852, 852, 852, 852, 560, 560, 328, 328, 1232, 1232, 1288, 1288, 500, 500, 356, 356, 836, 836, 852, 852, 392, 392, 940, 940, 1252, 1252, 1428, 1428, 896, 896, 1260, 1260, 1260, 1260, 852, 852, 776, 776, 796, 796, 368, 368, 1164, 1164, 1276, 1276, 1472, 1472, 448, 448, 1000, 1000, 504, 504, 1096, 1096, 612, 612, 584, 584, 660, 660, 776, 776, 684, 684, 1424, 1424, 852, 852, 416, 416, 1452, 1452, 996, 996, 668, 668, 484, 484, 364, 364, 548, 548, 472, 472, 852, 852, 1080, 1080, 852, 852, 1276, 1276, 1188, 1188, 852, 852, 852, 852, 324, 324, 1124, 1124, 300, 300, 512, 512, 884, 884, 852, 852, 1140, 1140, 852, 852, 1124, 1124, 852, 852, 1144, 1144, 852, 852, 824, 824, 852, 852, 1080, 1080, 1000, 1000, 1296, 1296, 852, 852, 1284, 1284, 852, 852, 852, 852, 1196, 1196, 432, 432, 1112, 1112, 696, 696, 400, 400, 852, 852, 852, 852, 440, 440, 852, 852, 1260, 1260, 808, 808, 572, 572, 852, 852, 772, 772, 428, 428, 940, 940, 852, 852, 480, 480, 1196, 1196, 1020, 1020, 492, 492, 1012, 1012, 852, 852, 964, 964, 1284, 1284, 852, 852, 852, 852, 852, 852, 852, 852, 820, 820, 852, 852, 944, 944, 1180, 1180, 852, 852, 528, 528, 1432, 1432, 852, 852, 976, 976, 764, 764, 1048, 1048, 852, 852, 852, 852, 852, 852, 352, 352, 852, 852, 1408, 1408, 564, 564, 852, 852, 852, 852, 1460, 1460, 1072, 1072, 548, 548, 852, 852, 688, 688, 852, 852, 852, 852, 488, 488, 1028, 1028, 540, 540, 1400, 1400, 852, 852, 852, 852, 1000, 1000, 852, 852, 892, 892, 852, 852, 852, 852, 1056, 1056, 852, 852, 1496, 1496, 852, 852, 852, 852, 1316, 1316, 1396, 1396, 1172, 1172, 852, 852, 852, 852, 852, 852, 708, 708]
    scc_duration_list = [142, 142, 142, 142, 64, 142, 142, 142, 136, 142, 142, 142, 142, 142, 142, 142, 80, 142, 142, 142, 142, 196, 88, 108, 108, 142, 142, 142, 72, 142, 142, 142, 142, 142, 64, 142, 142, 142, 142, 142, 104, 160, 84, 142, 36, 36, 92, 142, 142, 142, 48, 56, 48, 172, 142, 142, 80, 142, 142, 48, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 72, 112, 142, 142, 142, 140, 142, 142, 52, 72, 72, 142, 142, 36, 36, 68, 142, 142, 142, 142, 36, 48, 142, 142, 142, 152, 142, 142, 104, 72, 68, 124, 112, 108, 164, 168, 142, 142, 142, 142, 142, 142, 64, 64, 142, 132, 142, 142, 72, 142, 152, 142, 164, 164, 164, 142, 142, 142, 156, 142, 142, 142, 142, 142, 142, 142, 142, 142, 124, 142, 142, 142, 64, 142, 108, 108, 142, 142, 142, 142, 142, 142, 140, 142, 142, 142, 100, 142, 142, 142, 188, 188, 76, 142, 142, 100, 142, 160, 160, 124, 142, 142, 136, 142, 142, 142, 142, 142, 142, 64, 142, 142, 142, 142, 132, 172, 56, 142, 64, 64, 196, 68, 142, 92, 92, 142, 142, 142, 142, 142, 142, 142, 48, 142, 142, 144, 142, 142, 142, 142, 142, 142, 142, 142, 196, 142, 142, 142, 76, 142, 142, 142, 64, 142, 142, 136, 136, 142, 142, 142, 142, 100, 142, 142, 142, 142, 142, 96, 142, 142, 124, 124, 124, 142, 142, 142, 142, 142, 56, 142, 142, 142, 142, 142, 88, 142, 142, 196, 142, 120, 120, 142, 142, 142, 142, 142, 142, 142, 142, 142, 142, 72, 142, 142, 142, 116, 116, 48, 36, 36, 142, 142, 142, 36, 104, 56, 40, 142, 142, 142, 142, 40, 142, 142, 142, 142, 92, 142, 192, 142, 68, 142, 142, 142, 142, 168, 142, 142]
    #254NVs
    pol_duration_list = [336, 336, 308, 308, 428, 428, 504, 528, 372, 1060, 1060, 852, 852, 852, 612, 484, 1120, 852, 404, 404, 812, 672, 672, 560, 644, 644, 352, 352, 380, 380, 852, 852, 620, 620, 628, 628, 292, 292, 528, 528, 392, 392, 524, 524, 680, 504, 396, 396, 324, 324, 428, 240, 240, 504, 540, 540, 852, 1188, 1188, 764, 764, 976, 976, 820, 820, 444, 444, 1100, 488, 488, 604, 972, 380, 380, 352, 352, 660, 592, 592, 416, 416, 452, 452, 620, 620, 576, 576, 316, 316, 660, 660, 660, 720, 720, 620, 1024, 852, 852, 1396, 1396, 464, 464, 416, 416, 624, 624, 1008, 1008, 460, 460, 508, 508, 668, 668, 448, 448, 440, 668, 668, 852, 852, 852, 852, 844, 1048, 1048, 320, 320, 780, 780, 492, 1476, 1476, 656, 656, 1064, 1064, 456, 344, 344, 852, 540, 352, 524, 852, 852, 1156, 1388, 308, 852, 852, 1360, 1360, 572, 572, 204, 696, 504, 504, 1332, 1332, 1012, 1012, 708, 708, 852, 912, 912, 804, 804, 608, 608, 948, 948, 596, 596, 1256, 1256, 808, 808, 852, 392, 392, 568, 568, 872, 872, 1268, 1268, 780, 780, 852, 852, 476, 476, 508, 508, 640, 640, 392, 392, 512, 700, 700, 700, 700, 932, 932, 840, 840, 852, 852, 1248, 1248, 852, 852, 852, 1444, 1444, 620, 620, 852, 852, 752, 752, 1052, 1052, 852, 852, 852, 852, 1248, 860, 860, 520, 520, 1320, 1320, 1096, 1096, 568, 568, 488, 488, 852, 852, 556, 556, 420, 420, 1192, 1192, 552, 552]
    scc_duration_list = [142, 142, 142, 142, 64, 142, 142, 142, 142, 142, 142, 142, 142, 142, 196, 108, 108, 142, 72, 142, 142, 142, 142, 64, 142, 142, 142, 142, 104, 160, 84, 142, 92, 142, 142, 142, 48, 56, 48, 172, 142, 142, 80, 142, 48, 142, 142, 142, 142, 142, 142, 142, 142, 72, 142, 142, 142, 142, 142, 52, 72, 72, 142, 142, 36, 36, 68, 142, 142, 142, 36, 142, 142, 152, 142, 142, 104, 68, 124, 112, 108, 164, 168, 142, 142, 142, 142, 142, 142, 64, 64, 132, 142, 142, 72, 142, 164, 142, 142, 142, 156, 142, 142, 142, 142, 142, 142, 142, 142, 142, 124, 142, 142, 142, 64, 142, 108, 142, 142, 142, 142, 142, 142, 140, 142, 142, 100, 142, 142, 142, 188, 76, 142, 142, 100, 142, 160, 124, 142, 142, 142, 142, 142, 64, 142, 142, 142, 132, 56, 64, 64, 196, 68, 142, 92, 92, 142, 142, 142, 48, 142, 142, 144, 142, 142, 142, 142, 142, 142, 142, 196, 142, 142, 142, 76, 142, 142, 142, 64, 142, 136, 136, 142, 142, 142, 142, 100, 142, 142, 142, 142, 142, 96, 142, 142, 124, 124, 124, 142, 142, 142, 142, 56, 142, 142, 142, 142, 142, 88, 142, 142, 196, 142, 120, 120, 142, 142, 142, 142, 142, 142, 142, 72, 142, 116, 116, 48, 142, 142, 142, 36, 56, 40, 142, 142, 142, 142, 40, 142, 142, 142, 142, 92, 142, 192, 142, 68, 142, 142, 142, 142, 168, 142, 142]
    #151NVs
    pol_duration_list = [336, 336, 308, 428, 1060, 1060, 852, 612, 484, 852, 404, 812, 672, 560, 644, 352, 380, 380, 852, 620, 628, 292, 392, 392, 524, 504, 396, 324, 428, 240, 540, 540, 1188, 1188, 764, 764, 976, 444, 1100, 488, 488, 380, 380, 352, 592, 416, 416, 452, 452, 576, 576, 316, 316, 660, 620, 1024, 1396, 1396, 464, 416, 624, 1008, 1008, 508, 668, 448, 440, 668, 852, 852, 844, 1048, 320, 320, 780, 492, 1476, 656, 656, 1064, 1064, 456, 352, 524, 852, 852, 1156, 308, 852, 1360, 572, 204, 504, 1012, 708, 708, 852, 912, 912, 608, 948, 948, 596, 596, 808, 392, 568, 872, 872, 1268, 852, 852, 476, 508, 508, 640, 392, 392, 512, 700, 700, 932, 852, 852, 1248, 852, 852, 1444, 620, 852, 752, 1052, 1052, 852, 852, 852, 852, 860, 520, 1320, 1096, 1096, 488, 852, 852, 556, 556, 420, 1192, 552, 552]
    scc_duration_list = [76, 88, 116, 92, 88, 72, 104, 108, 80, 80, 88, 100, 96, 96, 96, 92, 72, 112, 96, 92, 92, 140, 96, 76, 100, 112, 72, 96, 92, 72, 92, 80, 100, 76, 80, 64, 104, 80, 304, 76, 96, 60, 68, 56, 60, 80, 80, 128, 92, 112, 80, 96, 72, 80, 112, 72, 76, 76, 124, 124, 104, 108, 72, 84, 100, 112, 92, 180, 116, 76, 92, 108, 112, 140, 120, 100, 72, 124, 84, 128, 96, 100, 140, 96, 120, 136, 100, 128, 108, 92, 96, 96, 96, 96, 96, 84, 92, 164, 88, 100, 132, 124, 100, 88, 84, 96, 124, 80, 88, 176, 128, 112, 172, 116, 304, 88, 96, 140, 112, 108, 144, 104, 104, 92, 112, 128, 108, 244, 140, 108, 120, 100, 96, 164, 100, 140, 180, 108, 180, 92, 112, 124, 108, 176, 132, 120, 192, 232, 128, 104, 144]
    ## 136NV
    pol_duration_list = [504, 504, 648, 648, 592, 592, 608, 608, 680, 680, 884, 884, 652, 652, 556, 556, 408, 408, 680, 680, 304, 304, 396, 396, 368, 368, 708, 708, 592, 592, 724, 724, 412, 412, 324, 324, 352, 352, 360, 360, 428, 428, 316, 316, 420, 420, 728, 728, 680, 680, 360, 360, 504, 504, 300, 300, 420, 420, 400, 400, 552, 552, 272, 272, 568, 568, 516, 516, 512, 512, 300, 300, 680, 680, 380, 380, 304, 304, 580, 580, 648, 648, 764, 764, 596, 596, 852, 852, 928, 928, 496, 496, 444, 444, 620, 620, 640, 640, 588, 588, 572, 572, 768, 768, 996, 996, 616, 616, 908, 908, 752, 752, 644, 644, 1508, 1508, 664, 664, 928, 928, 1092, 1092, 468, 468, 416, 416, 444, 444, 760, 760, 760, 760, 1052, 1052, 844, 844, 492, 492, 324, 324, 516, 516, 676, 676, 964, 964, 528, 528, 684, 684, 820, 820, 1084, 1084, 552, 552, 752, 752, 952, 952, 956, 956, 968, 968, 1428, 1428, 892, 892, 788, 788, 500, 500, 416, 416, 808, 808, 656, 656, 240, 240, 1352, 1352, 1084, 1084, 964, 964, 680, 680, 592, 592, 680, 680, 1204, 1204, 656, 656, 656, 656, 972, 972, 660, 660, 1476, 1476, 1500, 1500, 808, 808, 568, 568, 832, 832, 520, 520, 1272, 1272, 1152, 1152, 572, 572, 1020, 1020, 680, 680, 1292, 1292, 740, 740, 1264, 1264, 864, 864, 1060, 1060, 1188, 1188, 656, 656, 1392, 1392, 980, 980, 1308, 1308, 868, 868, 1092, 1092, 1784, 1784, 956, 956, 1076, 1076, 680, 680, 1372, 1372, 680, 680, 1924, 1924, 1640, 1640, 1176, 1176, 1676, 1676, 1476, 1476, 972, 972]
    scc_duration_list = [76, 88, 116, 92, 88, 104, 108, 80, 80, 88, 100, 96, 96, 92, 72, 112, 96, 92, 92, 140, 96, 76, 100, 112, 72, 96, 92, 72, 92, 80, 100, 76, 80, 104, 80, 76, 96, 60, 68, 80, 80, 128, 92, 112, 80, 96, 80, 112, 72, 76, 76, 124, 124, 104, 108, 72, 84, 100, 112, 92, 180, 116, 76, 108, 112, 140, 120, 100, 72, 84, 128, 96, 100, 140, 96, 120, 136, 100, 128, 108, 92, 96, 96, 96, 96, 84, 92, 164, 88, 100, 132, 124, 100, 88, 84, 96, 124, 80, 88, 176, 128, 112, 172, 88, 140, 112, 108, 144, 104, 104, 112, 108, 244, 140, 108, 120, 100, 96, 164, 100, 140, 180, 108, 180, 92, 112, 124, 108, 176, 132, 120, 192, 232, 128, 104, 144]
    # spot_weights = [1.159808521156943, 0.9339705451509978, 1.0815148619637092, 0.9339705451509978, 0.7345159774420897, 0.7981528012511167, 0.7981528012511167, 0.7345159774420897, 1.32568384783322, 1.159808521156943, 1.2411819619044773, 0.7345159774420897, 0.6736595263587027, 1.0062519850229286, 0.8646208405138835, 0.7345159774420897, 0.6736595263587027, 0.7981528012511167, 0.9339705451509978, 0.9339705451509978, 1.159808521156943, 0.8646208405138835, 1.32568384783322, 0.9339705451509978, 0.7345159774420897, 0.7981528012511167, 0.8646208405138835, 1.5042659890001955, 1.0062519850229286, 1.0062519850229286, 1.0062519850229286, 0.9339705451509978, 0.7981528012511167, 0.6736595263587027, 0.7981528012511167, 1.0815148619637092, 0.7981528012511167, 1.0062519850229286, 1.4133625165812045, 1.0815148619637092, 1.2411819619044773, 1.2411819619044773, 0.8646208405138835, 1.0062519850229286, 1.5042659890001955, 1.0815148619637092, 0.6736595263587027, 1.0815148619637092, 0.9339705451509978, 0.9339705451509978, 1.2411819619044773, 1.4133625165812045, 0.6155321989420328, 0.6736595263587027, 1.0815148619637092, 0.6155321989420328, 1.0815148619637092, 1.159808521156943, 0.9339705451509978, 0.6736595263587027, 1.2411819619044773, 0.9339705451509978, 1.0815148619637092, 1.5984419779087295, 0.9339705451509978, 1.0815148619637092, 0.7345159774420897, 0.7981528012511167, 1.32568384783322, 0.8646208405138835, 0.6155321989420328, 0.7981528012511167, 1.159808521156943, 1.0062519850229286, 0.7981528012511167, 0.8646208405138835, 1.0815148619637092, 0.7981528012511167, 0.8646208405138835, 0.7981528012511167, 0.7981528012511167, 0.9339705451509978, 1.0062519850229286, 1.159808521156943, 0.8646208405138835, 0.7981528012511167, 0.9339705451509978, 1.5984419779087295, 1.0815148619637092, 0.6155321989420328, 0.9339705451509978, 1.0815148619637092, 1.0062519850229286, 1.6959378964264216, 0.9339705451509978, 0.8646208405138835, 1.159808521156943, 0.9339705451509978, 0.7981528012511167, 0.7345159774420897, 0.7345159774420897, 1.5042659890001955, 1.4133625165812045, 1.0815148619637092, 0.8646208405138835, 0.9339705451509978, 0.7981528012511167, 0.6736595263587027, 0.6155321989420328, 0.7345159774420897, 1.0815148619637092, 1.5984419779087295, 1.5984419779087295, 1.159808521156943, 0.7981528012511167, 0.6155321989420328, 1.0815148619637092, 1.2411819619044773, 1.0062519850229286, 0.8646208405138835, 1.5984419779087295, 1.2411819619044773, 1.159808521156943, 0.7981528012511167, 1.32568384783322, 1.0062519850229286, 0.8646208405138835, 0.9339705451509978, 1.159808521156943, 0.8646208405138835, 1.0062519850229286, 1.7968008659176977, 1.5042659890001955, 0.9339705451509978, 0.6736595263587027, 1.32568384783322]
    # spot_weights = [1.240943198967792, 1.1740251843072143, 1.240943198967792, 1.1740251843072143, 1.1093410555568934, 0.9865492106513315, 1.1093410555568934, 1.240943198967792, 0.7164498009601598, 0.9865492106513315, 0.7663833130104475, 0.9283783827420362, 1.240943198967792, 0.9865492106513315, 0.9865492106513315, 0.6224839475607075, 0.8183276118958405, 0.7663833130104475, 1.240943198967792, 1.240943198967792, 1.1740251843072143, 0.9865492106513315, 1.240943198967792, 0.8723152064351439, 1.1740251843072143, 0.7164498009601598, 0.9865492106513315, 0.8723152064351439, 0.8183276118958405, 0.7663833130104475, 1.240943198967792, 0.8183276118958405, 1.1740251843072143, 0.6684943374492833, 1.0468595498244968, 1.1740251843072143, 0.9865492106513315, 1.0468595498244968, 1.240943198967792, 0.6684943374492833, 1.1093410555568934, 0.7164498009601598, 1.240943198967792, 0.7663833130104475, 0.7164498009601598, 0.5783854122084401, 1.1093410555568934, 1.0468595498244968, 0.7663833130104475, 1.0468595498244968, 1.0468595498244968, 0.8183276118958405, 1.1740251843072143, 1.1740251843072143, 1.240943198967792, 1.1093410555568934, 0.9865492106513315, 1.1093410555568934, 1.1093410555568934, 1.0468595498244968, 1.0468595498244968, 1.240943198967792, 1.1093410555568934, 0.8723152064351439, 1.240943198967792, 1.1093410555568934, 0.9865492106513315, 1.0468595498244968, 1.1093410555568934, 1.1740251843072143, 1.1740251843072143, 1.1093410555568934, 1.0468595498244968, 1.0468595498244968, 0.9283783827420362, 0.5783854122084401, 1.0468595498244968, 0.9865492106513315, 0.6684943374492833, 1.0468595498244968, 0.5361652601537913, 0.49578975954808036, 1.240943198967792, 0.9283783827420362, 1.1740251843072143, 1.1093410555568934, 1.0468595498244968, 0.7663833130104475, 1.240943198967792, 1.1740251843072143, 1.1093410555568934, 1.1093410555568934, 0.7663833130104475, 0.6224839475607075, 1.0468595498244968, 1.1740251843072143, 0.9865492106513315, 0.7164498009601598, 0.9865492106513315, 0.5361652601537913, 1.1740251843072143, 0.49578975954808036, 0.8723152064351439, 0.8183276118958405, 0.9283783827420362, 1.240943198967792, 0.8723152064351439, 1.1093410555568934, 0.7663833130104475, 1.240943198967792, 1.240943198967792, 0.9283783827420362, 1.1740251843072143, 0.9283783827420362, 1.240943198967792, 0.7663833130104475, 1.1740251843072143, 0.8723152064351439, 0.8723152064351439, 1.240943198967792, 1.0468595498244968, 0.8183276118958405, 1.1740251843072143, 1.1093410555568934, 1.0468595498244968, 1.0468595498244968, 0.8723152064351439, 0.7164498009601598, 1.240943198967792, 0.9865492106513315, 0.9283783827420362, 0.9865492106513315, 1.1740251843072143, 0.6684943374492833, 1.1740251843072143, 1.1740251843072143, 0.9865492106513315, 1.240943198967792, 1.0468595498244968, 0.8183276118958405, 0.8723152064351439, 0.9865492106513315, 1.0468595498244968, 0.6224839475607075, 0.7663833130104475, 0.8183276118958405, 0.9283783827420362, 1.240943198967792, 1.1093410555568934, 1.1740251843072143, 1.1093410555568934, 0.5361652601537913, 1.1740251843072143, 1.1093410555568934, 0.7663833130104475, 1.1740251843072143, 0.9865492106513315, 0.7164498009601598, 0.8723152064351439, 0.8183276118958405, 0.8183276118958405, 1.0468595498244968, 1.1740251843072143, 0.9283783827420362, 0.9283783827420362, 0.9865492106513315, 0.8723152064351439, 0.9283783827420362, 0.8183276118958405, 0.8183276118958405, 0.8183276118958405, 0.7663833130104475, 0.7663833130104475, 0.9865492106513315, 1.1740251843072143, 1.240943198967792, 1.240943198967792, 1.1093410555568934, 0.8183276118958405, 0.9283783827420362, 0.7164498009601598, 1.1740251843072143, 1.0468595498244968, 0.8183276118958405, 0.8183276118958405, 0.7164498009601598, 0.5783854122084401, 1.1740251843072143, 1.1740251843072143, 0.8723152064351439, 0.8723152064351439, 0.9283783827420362, 0.7164498009601598, 1.1740251843072143, 1.0468595498244968, 1.240943198967792, 1.1093410555568934, 0.9865492106513315, 1.0468595498244968, 1.1740251843072143, 1.1740251843072143, 1.240943198967792, 0.8183276118958405, 0.9865492106513315, 1.1740251843072143, 0.9283783827420362, 1.1093410555568934, 1.1740251843072143, 1.240943198967792, 1.1740251843072143, 1.1093410555568934, 1.1093410555568934, 1.240943198967792, 0.7164498009601598, 1.240943198967792, 1.240943198967792, 1.240943198967792, 0.7663833130104475, 1.0468595498244968, 1.1093410555568934, 0.8183276118958405, 0.8723152064351439, 0.7663833130104475, 0.9865492106513315, 1.0468595498244968, 0.9865492106513315, 1.1740251843072143, 1.0468595498244968, 1.1740251843072143, 1.1740251843072143, 1.1740251843072143, 0.9283783827420362, 1.1093410555568934, 1.1093410555568934, 1.1740251843072143, 0.457224908960518, 0.7164498009601598, 0.9283783827420362, 1.1093410555568934, 0.8183276118958405, 1.240943198967792, 1.1740251843072143, 1.240943198967792, 0.9283783827420362, 1.1093410555568934, 1.240943198967792, 0.5361652601537913, 1.1740251843072143, 0.9865492106513315, 1.240943198967792, 1.0468595498244968, 0.9865492106513315, 0.8183276118958405, 1.1093410555568934, 1.1093410555568934, 1.1740251843072143, 1.1093410555568934, 0.9283783827420362, 0.9865492106513315, 1.240943198967792, 1.1740251843072143, 0.6684943374492833, 1.0468595498244968, 1.240943198967792, 1.1740251843072143, 0.5361652601537913, 1.240943198967792, 0.8183276118958405, 1.1740251843072143, 0.9283783827420362, 1.1093410555568934, 1.0468595498244968, 0.8723152064351439, 0.9283783827420362, 0.7164498009601598, 0.7663833130104475, 1.0468595498244968, 1.0468595498244968, 0.5361652601537913, 0.8723152064351439, 0.457224908960518, 1.240943198967792, 1.240943198967792, 0.8723152064351439, 1.240943198967792, 1.1093410555568934, 0.6224839475607075, 1.1093410555568934, 1.1740251843072143, 0.9865492106513315, 0.9283783827420362, 1.0468595498244968, 0.9865492106513315, 1.1740251843072143, 1.0468595498244968, 0.9283783827420362, 0.9283783827420362, 0.7663833130104475, 0.7164498009601598, 1.1093410555568934, 1.240943198967792, 0.9283783827420362, 1.1740251843072143, 1.1093410555568934, 1.240943198967792, 1.0468595498244968, 1.1740251843072143, 1.240943198967792, 1.240943198967792, 0.9865492106513315, 0.9865492106513315, 1.1093410555568934]
    pol_duration_list = [392, 392, 736, 736, 720, 720, 708, 708, 1092, 1092, 772, 772, 1108, 1108, 868, 868, 1204, 1204, 936, 936, 704, 704, 288, 288, 800, 800, 732, 732, 352, 352, 864, 864, 804, 804, 580, 580, 708, 708, 772, 772, 920, 920, 1020, 1020, 1076, 1076, 676, 676, 480, 480, 684, 684, 748, 748, 300, 300, 764, 764, 572, 572, 704, 704, 806, 806, 856, 856, 380, 380, 736, 736, 820, 820, 660, 660, 1888, 1888, 392, 392, 704, 704, 420, 420, 464, 464, 700, 700, 806, 806, 304, 304, 806, 806, 724, 724, 708, 708, 520, 520, 776, 776, 384, 384, 680, 680, 724, 724, 1740, 1740, 716, 716, 952, 952, 876, 876, 660, 660, 1212, 1212, 936, 936, 806, 806, 392, 392, 828, 828, 796, 796, 308, 308, 692, 692, 492, 492, 806, 806, 288, 288, 806, 806, 944, 944, 660, 660, 728, 728, 806, 806, 806, 806, 806, 806, 1360, 1360, 596, 596, 724, 724, 412, 412, 806, 806, 806, 806, 806, 806, 564, 564, 652, 652, 696, 696, 1540, 1540, 728, 728, 652, 652, 392, 392, 448, 448, 304, 304, 392, 392, 968, 968, 772, 772, 640, 640, 548, 548, 664, 664, 468, 468, 806, 806, 504, 504, 806, 806, 700, 700, 1128, 1128, 672, 672, 300, 300, 1536, 1536, 836, 836, 304, 304, 1036, 1036, 1576, 1576, 804, 804, 806, 806, 640, 640, 806, 806, 1776, 1776, 612, 612, 844, 844, 732, 732, 684, 684, 928, 928, 628, 628, 972, 972, 988, 988, 680, 680, 1064, 1064, 708, 708, 780, 780, 1004, 1004, 1484, 1484, 912, 912, 600, 600, 484, 484, 636, 636, 720, 720, 1088, 1088, 806, 806, 824, 824, 608, 608, 956, 956, 708, 708, 692, 692, 428, 428, 836, 836, 806, 806, 396, 396, 760, 760, 384, 384, 524, 524, 852, 852, 1600, 1600, 806, 806, 1268, 1268, 788, 788, 308, 308, 728, 728, 304, 304, 832, 832, 288, 288, 728, 728, 728, 728, 912, 912, 452, 452, 1452, 1452, 588, 588, 1892, 1892, 1896, 1896, 476, 476, 1072, 1072, 580, 580, 1244, 1244, 1656, 1656, 1792, 1792, 1600, 1600, 1172, 1172, 540, 540, 1556, 1556, 668, 668, 704, 704, 504, 504, 792, 792, 376, 376, 716, 716, 684, 684, 692, 692, 1100, 1100, 806, 806, 444, 444, 520, 520, 1404, 1404, 736, 736, 1696, 1696, 428, 428, 600, 600, 1608, 1608, 806, 806, 1220, 1220, 1644, 1644, 1040, 1040, 1376, 1376, 752, 752, 468, 468, 652, 652, 908, 908, 806, 806, 748, 748, 528, 528, 440, 440, 848, 848, 656, 656, 696, 696, 792, 792, 888, 888, 808, 808, 584, 584, 832, 832, 680, 680, 806, 806, 1060, 1060, 806, 806, 1000, 1000, 940, 940, 712, 712, 796, 796, 996, 996, 806, 806, 812, 812, 748, 748, 692, 692, 806, 806, 556, 556, 1032, 1032, 692, 692, 806, 806, 872, 872, 980, 980, 806, 806, 1540, 1540, 768, 768, 308, 308, 806, 806, 872, 872, 1020, 1020, 1512, 1512, 1080, 1080, 960, 960, 1920, 1920, 728, 728, 804, 804, 648, 648, 806, 806, 1180, 1180, 640, 640, 1188, 1188, 1052, 1052, 928, 928, 1816, 1816, 960, 960, 532, 532, 1396, 1396, 1032, 1032, 1336, 1336, 444, 444, 436, 436, 532, 532, 424, 424, 884, 884, 904, 904, 812, 812, 884, 884, 888, 888, 684, 684, 1176, 1176, 636, 636, 460, 460, 924, 924, 984, 984, 1244, 1244, 806, 806, 1552, 1552, 1040, 1040, 896, 896, 980, 980, 940, 940, 1144, 1144, 784, 784, 806, 806, 912, 912, 764, 764, 1120, 1120, 692, 692, 1840, 1840, 996, 996, 1644, 1644, 1796, 1796, 1064, 1064, 816, 816, 732, 732, 848, 848, 952, 952, 806, 806, 1028, 1028, 720, 720, 796, 796, 1268, 1268, 1584, 1584, 1720, 1720, 1768, 1768, 806, 806, 1388, 1388, 924, 924, 1496, 1496]
    pol_duration_list = [ round(val/4) * 4 for val in pol_duration_list]
    scc_duration_list = [180, 176, 178, 168, 104, 108, 156, 136, 180, 188, 304, 144, 252, 164, 172, 148, 244, 152, 172, 116, 180, 172, 128, 124, 108, 304, 64, 144, 84, 176, 148, 192, 144, 178, 160, 208, 248, 284, 120, 152, 268, 176, 124, 252, 204, 304, 178, 132, 156, 160, 124, 180, 124, 200, 176, 160, 144, 204, 204, 188, 88, 196, 148, 168, 136, 148, 280, 120, 92, 120, 216, 232, 156, 96, 180, 164, 152, 192, 172, 108, 178, 192, 144, 168, 264, 108, 196, 176, 112, 304, 124, 136, 120, 136, 144, 128, 124, 68, 192, 112, 268, 160, 144, 160, 212, 204, 304, 196, 136, 156, 104, 140, 88, 128, 140, 304, 128, 304, 116, 148, 304, 116, 120, 304, 148, 92, 204, 200, 304, 252, 268, 178, 212, 100, 148, 164, 216, 304, 216, 148, 152, 180, 164, 148, 178, 120, 156, 204, 96, 152, 304, 304, 304, 144, 304, 208, 220, 168, 164, 212, 252, 100, 112, 160, 144, 304, 244, 178, 148, 304, 304, 212, 216, 304, 120, 220, 232, 124, 192, 76, 152, 172, 124, 172, 212, 236, 200, 208, 164, 304, 240, 304, 176, 178, 292, 304, 148, 160, 168, 304, 172, 204, 232, 178, 304, 232, 304, 116, 178, 192, 216, 88, 160, 128, 120, 148, 144, 236, 276, 156, 132, 212, 172, 148, 156, 192, 104, 232, 288, 200, 300, 180, 232, 216, 304, 208, 84, 196, 292, 304, 304, 148, 304, 304, 160, 208, 304, 304, 304, 180, 144, 304, 168, 304, 172, 276, 304, 236, 304, 178, 176, 288, 178, 212, 184, 208, 180, 216, 252, 212, 188, 176, 304, 304, 304, 156, 200, 304, 136, 100, 136, 80, 276, 304, 240, 178, 180, 304, 178, 220, 304, 176, 196, 280, 304, 304, 200, 304, 304, 188, 304, 204, 178, 208, 236, 304, 192, 232, 304, 304, 168, 304]
    scc_duration_list = [ round(val/4) * 4 for val in scc_duration_list]
    ### 230NVs
    pol_duration_list = [760, 760, 668, 668, 608, 608, 700, 700, 1008, 1008, 616, 616, 492, 492, 836, 836, 392, 392, 1028, 1028, 312, 312, 772, 772, 600, 600, 1036, 1036, 840, 840, 728, 728, 728, 728, 1076, 1076, 412, 412, 440, 440, 860, 860, 848, 848, 704, 704, 508, 508, 652, 652, 836, 836, 796, 796, 728, 728, 712, 712, 696, 696, 436, 436, 612, 612, 612, 612, 748, 748, 956, 956, 676, 676, 668, 668, 404, 404, 776, 776, 468, 468, 688, 688, 548, 548, 1652, 1652, 652, 652, 1064, 1064, 488, 488, 616, 616, 744, 744, 368, 368, 468, 468, 744, 744, 740, 740, 1252, 1252, 668, 668, 536, 536, 820, 820, 400, 400, 812, 812, 1616, 1616, 984, 984, 576, 576, 920, 920, 624, 624, 548, 548, 692, 692, 692, 692, 536, 536, 552, 552, 508, 508, 684, 684, 672, 672, 492, 492, 388, 388, 496, 496, 1688, 1688, 652, 652, 1112, 1112, 756, 756, 480, 480, 556, 556, 1628, 1628, 1016, 1016, 664, 664, 716, 716, 780, 780, 624, 624, 1320, 1320, 644, 644, 620, 620, 688, 688, 880, 880, 576, 576, 1788, 1788, 744, 744, 1940, 1940, 676, 676, 696, 696, 1940, 1940, 716, 716, 668, 668, 680, 680, 1940, 1940, 692, 692, 712, 712, 944, 944, 776, 776, 796, 796, 732, 732, 684, 684, 668, 668, 752, 752, 856, 856, 596, 596, 776, 776, 1220, 1220, 616, 616, 308, 308, 520, 520, 808, 808, 740, 740, 952, 952, 1112, 1112, 1940, 1940, 1236, 1236, 1140, 1140, 656, 656, 1276, 1276, 1124, 1124, 100, 100, 1940, 1940, 1352, 1352, 640, 640, 1940, 1940, 1044, 1044, 468, 468, 780, 780, 456, 456, 536, 536, 584, 584, 1276, 1276, 760, 760, 396, 396, 600, 600, 1568, 1568, 764, 764, 796, 796, 780, 780, 1828, 1828, 1292, 1292, 1256, 1256, 840, 840, 516, 516, 760, 760, 1940, 1940, 876, 876, 684, 684, 756, 756, 808, 808, 616, 616, 604, 604, 844, 844, 588, 588, 800, 800, 1132, 1132, 1372, 1372, 976, 976, 836, 836, 1556, 1556, 628, 628, 944, 944, 1940, 1940, 460, 460, 892, 892, 444, 444, 960, 960, 700, 700, 1940, 1940, 528, 528, 1364, 1364, 448, 448, 468, 468, 1608, 1608, 1136, 1136, 1140, 1140, 648, 648, 984, 984, 936, 936, 808, 808, 1412, 1412, 600, 600, 748, 748, 912, 912, 1272, 1272, 1024, 1024, 596, 596, 980, 980, 1848, 1848, 640, 640, 912, 912, 1052, 1052, 868, 868, 580, 580, 836, 836, 792, 792, 600, 600, 972, 972, 1076, 1076, 1204, 1204, 1168, 1168, 1116, 1116, 988, 988, 800, 800, 988, 988, 1112, 1112, 1832, 1832, 1656, 1656, 1940, 1940, 964, 964, 1304, 1304, 728, 728, 892, 892, 1940, 1940, 940, 940, 1340, 1340, 1368, 1368, 1416, 1416, 1736, 1736, 1940, 1940, 1636, 1636, 1484, 1484]
    scc_duration_list = [88, 92, 84, 112, 88, 92, 156, 88, 80, 100, 84, 84, 92, 72, 104, 92, 92, 68, 100, 72, 100, 96, 92, 76, 88, 88, 88, 92, 84, 108, 116, 92, 72, 96, 116, 112, 76, 92, 100, 88, 84, 72, 64, 88, 76, 68, 72, 88, 120, 80, 96, 88, 92, 116, 80, 92, 112, 104, 156, 116, 80, 80, 92, 92, 92, 84, 100, 96, 116, 80, 76, 72, 76, 84, 84, 88, 176, 88, 96, 92, 92, 76, 68, 84, 128, 80, 84, 116, 88, 36, 84, 88, 96, 76, 112, 96, 80, 140, 92, 96, 100, 76, 84, 72, 80, 120, 80, 88, 124, 100, 76, 116, 68, 80, 92, 84, 96, 92, 104, 92, 136, 116, 136, 112, 76, 84, 92, 176, 108, 104, 120, 96, 92, 92, 88, 88, 84, 92, 124, 84, 112, 92, 68, 88, 88, 92, 80, 136, 92, 92, 124, 88, 72, 104, 100, 120, 108, 108, 84, 88, 92, 112, 112, 88, 112, 96, 132, 96, 88, 112, 116, 108, 100, 84, 96, 116, 100, 88, 132, 88, 92, 148, 96, 100, 92, 140, 88, 84, 84, 92, 96, 144, 112, 100, 100, 112, 104, 96, 84, 104, 104, 116, 76, 120, 148, 128, 92, 92, 100, 92, 108, 108, 92, 108, 112, 104, 112, 120, 144, 88, 100, 120, 100, 116, 144, 112, 104, 116, 132, 108]
    # spot_weights = [1.1238878502462768, 0.85499400707105, 0.6810425686196978, 0.85499400707105, 1.3548565279822078, 0.9181663879858525, 0.6280929318715672, 0.9840093598709314, 0.85499400707105, 0.85499400707105, 0.85499400707105, 0.9840093598709314, 1.0525681973703258, 1.6122615108316045, 1.0525681973703258, 1.3548565279822078, 0.7364782323918739, 1.6122615108316045, 0.7364782323918739, 0.7944466072379957, 1.1980129531886803, 0.6810425686196978, 1.1238878502462768, 0.6810425686196978, 1.0525681973703258, 1.1980129531886803, 1.1238878502462768, 0.9181663879858525, 1.1980129531886803, 1.3548565279822078, 1.0525681973703258, 0.6280929318715672, 0.9840093598709314, 0.9840093598709314, 1.2749878351127324, 0.6280929318715672, 0.85499400707105, 1.2749878351127324, 1.1238878502462768, 1.5234500395432031, 0.9840093598709314, 0.9181663879858525, 1.704140113087266, 0.9840093598709314, 1.2749878351127324, 0.6810425686196978, 0.9181663879858525, 1.4376627751923434, 1.0525681973703258, 1.1980129531886803, 0.85499400707105, 0.6810425686196978, 1.0525681973703258, 1.4376627751923434, 0.85499400707105, 1.0525681973703258, 1.0525681973703258, 1.1980129531886803, 1.2749878351127324, 0.9181663879858525, 1.2749878351127324, 1.5234500395432031, 0.85499400707105, 1.1980129531886803, 0.7944466072379957, 0.9181663879858525, 1.5234500395432031, 0.9181663879858525, 1.1238878502462768, 1.0525681973703258, 0.9181663879858525, 0.7364782323918739, 0.7944466072379957, 0.7364782323918739, 0.6810425686196978, 0.85499400707105, 1.704140113087266, 1.1980129531886803, 0.9840093598709314, 0.85499400707105, 1.0525681973703258, 0.9181663879858525, 0.9840093598709314, 1.1238878502462768, 0.85499400707105, 1.0525681973703258, 0.7364782323918739, 1.3548565279822078, 0.7364782323918739, 0.9840093598709314, 0.6810425686196978, 1.1980129531886803, 1.1980129531886803, 0.7364782323918739, 0.7944466072379957, 1.1980129531886803, 1.0525681973703258, 1.2749878351127324, 1.1238878502462768, 0.9181663879858525, 0.9840093598709314, 0.9181663879858525, 0.7944466072379957, 0.9840093598709314, 0.85499400707105, 1.3548565279822078, 0.85499400707105, 0.9181663879858525, 0.9840093598709314, 1.0525681973703258, 0.7944466072379957, 0.85499400707105, 1.1980129531886803, 0.6810425686196978, 0.6810425686196978, 1.0525681973703258, 0.9181663879858525, 0.7364782323918739, 1.1238878502462768, 0.9181663879858525, 1.4376627751923434, 1.6122615108316045, 0.7944466072379957, 0.9181663879858525, 0.6810425686196978, 0.9181663879858525, 1.704140113087266, 0.6280929318715672, 0.9181663879858525, 0.85499400707105, 0.7364782323918739, 0.9181663879858525, 0.6810425686196978, 1.4376627751923434, 1.1980129531886803, 0.6810425686196978, 0.85499400707105, 1.1980129531886803, 1.1980129531886803, 0.7944466072379957, 0.7944466072379957, 1.0525681973703258, 0.7364782323918739, 1.0525681973703258, 0.6810425686196978, 1.5234500395432031, 0.9840093598709314, 1.2749878351127324, 0.6810425686196978, 1.1238878502462768, 0.6280929318715672, 1.0525681973703258, 0.7944466072379957, 1.0525681973703258, 0.9181663879858525, 0.6810425686196978, 0.9840093598709314, 1.1238878502462768, 0.9840093598709314, 1.0525681973703258, 1.1980129531886803, 0.9181663879858525, 1.5234500395432031, 0.9840093598709314, 0.85499400707105, 0.6280929318715672, 0.85499400707105, 1.5234500395432031, 0.6810425686196978, 1.0525681973703258, 0.6810425686196978, 0.85499400707105, 1.0525681973703258, 0.9181663879858525, 0.6810425686196978, 0.7364782323918739, 1.1980129531886803, 0.6810425686196978, 0.9181663879858525, 0.6280929318715672, 0.9840093598709314, 0.7944466072379957, 1.1980129531886803, 0.85499400707105, 0.6280929318715672, 1.704140113087266, 0.85499400707105, 0.7944466072379957, 0.9181663879858525, 0.7364782323918739, 1.1980129531886803, 1.1238878502462768, 1.2749878351127324, 0.9181663879858525, 1.1980129531886803, 0.9181663879858525, 1.0525681973703258, 0.7944466072379957, 0.85499400707105, 1.1238878502462768, 0.7364782323918739, 1.1980129531886803, 0.7364782323918739, 1.6122615108316045, 0.85499400707105, 0.7944466072379957, 0.9840093598709314, 0.9840093598709314, 0.9840093598709314, 0.6810425686196978, 0.9840093598709314, 0.7364782323918739, 0.6810425686196978, 0.7944466072379957, 0.85499400707105, 1.0525681973703258, 1.0525681973703258, 0.85499400707105, 1.6122615108316045, 0.6810425686196978, 1.0525681973703258, 1.4376627751923434, 0.7364782323918739, 0.9181663879858525, 1.4376627751923434, 1.1980129531886803, 0.7944466072379957, 1.5234500395432031, 1.3548565279822078, 1.2749878351127324]
    # pol_duration_list = [392, 392, 736, 1092, 1092, 772, 772, 1108, 868, 868, 1204, 936, 936, 704, 704, 288, 800, 732, 732, 352, 864, 864, 804, 804, 580, 580, 708, 708, 772, 772, 920, 920, 1020, 1076, 1076, 676, 676, 480, 480, 684, 684, 748, 748, 300, 300, 764, 764, 572, 704, 856, 856, 380, 736, 736, 820, 820, 660, 660, 392, 704, 704, 464, 464, 700, 700, 808, 304, 304, 808, 808, 724, 724, 708, 708, 520, 776, 384, 680, 724, 724, 1740, 1740, 716, 716, 952, 952, 660, 1212, 1212, 936, 808, 392, 392, 828, 828, 796, 796, 308, 692, 492, 808, 808, 288, 288, 808, 808, 944, 944, 660, 660, 728, 808, 808, 808, 1360, 1360, 596, 724, 412, 808, 808, 808, 808, 808, 652, 652, 696, 696, 1540, 1540, 728, 728, 652, 652, 392, 448, 448, 304, 392, 392, 968, 968, 772, 772, 640, 640, 548, 664, 664, 808, 504, 504, 808, 700, 1128, 672, 300, 1536, 1536, 836, 836, 304, 304, 1036, 1036, 1576, 1576, 804, 804, 808, 808, 640, 640, 1776, 1776, 612, 612, 844, 732, 684, 684, 928, 928, 972, 972, 988, 680, 680, 1064, 708, 708, 780, 1004, 1004, 1484, 912, 912, 600, 484, 636, 636, 720, 720, 808, 824, 824, 608, 608, 692, 692, 428, 836, 836, 808, 396, 760, 384, 384, 524, 852, 852, 1600, 1600, 1268, 1268, 788, 788, 308, 728, 728]
    # scc_duration_list = [180, 176, 168, 180, 188, 304, 144, 164, 172, 148, 244, 172, 116, 180, 172, 128, 108, 64, 144, 176, 148, 192, 144, 176, 160, 208, 248, 284, 120, 152, 268, 176, 124, 204, 304, 176, 132, 156, 160, 124, 180, 124, 200, 176, 160, 144, 204, 204, 196, 136, 148, 280, 92, 120, 216, 232, 156, 96, 152, 172, 108, 144, 168, 264, 108, 176, 112, 304, 124, 136, 120, 136, 144, 128, 124, 192, 160, 160, 212, 204, 304, 196, 136, 156, 104, 140, 140, 128, 304, 116, 304, 120, 304, 148, 92, 204, 200, 252, 176, 212, 148, 164, 216, 304, 216, 148, 152, 180, 164, 148, 120, 156, 204, 152, 304, 144, 208, 168, 212, 252, 100, 160, 144, 304, 148, 304, 304, 212, 216, 304, 120, 220, 232, 124, 192, 152, 172, 172, 212, 236, 200, 208, 164, 304, 240, 304, 176, 292, 304, 304, 172, 204, 232, 304, 304, 192, 216, 160, 128, 120, 148, 144, 236, 276, 156, 132, 212, 172, 148, 156, 192, 104, 232, 300, 180, 232, 216, 208, 196, 292, 304, 304, 148, 160, 208, 304, 304, 180, 304, 168, 304, 276, 304, 236, 304, 176, 288, 212, 184, 180, 216, 252, 212, 304, 304, 156, 200, 304, 276, 304, 240, 180, 304, 220, 176, 280, 304, 304, 200, 304, 188, 304, 204, 236, 304, 192, 232, 304, 168, 304]
    ### 223NV s
    pol_duration_list = [760, 760, 668, 668, 608, 608, 700, 700, 1008, 1008, 616, 616, 492, 492, 836, 836, 392, 392, 1028, 1028, 312, 312, 772, 772, 600, 600, 1036, 1036, 840, 840, 728, 728, 728, 1076, 1076, 412, 440, 440, 860, 860, 848, 704, 704, 508, 508, 652, 652, 836, 836, 796, 796, 728, 728, 712, 712, 696, 696, 436, 436, 612, 612, 612, 612, 748, 748, 956, 956, 676, 676, 668, 668, 404, 404, 776, 776, 468, 468, 688, 688, 548, 548, 1652, 1652, 652, 652, 1064, 488, 488, 616, 616, 744, 368, 368, 468, 468, 744, 744, 740, 740, 1252, 1252, 668, 668, 536, 536, 820, 400, 400, 812, 812, 1616, 1616, 984, 984, 576, 576, 920, 920, 624, 624, 548, 548, 692, 692, 692, 692, 536, 536, 552, 552, 508, 508, 684, 684, 672, 672, 492, 492, 388, 496, 496, 1688, 1688, 652, 652, 1112, 1112, 756, 756, 480, 480, 556, 556, 1628, 1628, 1016, 1016, 664, 664, 716, 716, 780, 780, 624, 624, 1320, 1320, 644, 644, 620, 620, 688, 688, 880, 880, 576, 576, 1788, 1788, 744, 744, 1940, 1940, 676, 676, 696, 696, 1940, 1940, 716, 716, 668, 668, 680, 680, 1940, 1940, 692, 692, 712, 712, 944, 944, 776, 776, 796, 796, 732, 732, 684, 684, 668, 668, 752, 752, 856, 856, 596, 596, 776, 776, 1220, 1220]
    scc_duration_list = [88, 92, 84, 112, 88, 92, 156, 88, 80, 100, 84, 84, 92, 72, 104, 92, 92, 68, 100, 72, 100, 96, 92, 76, 88, 88, 88, 92, 84, 108, 116, 72, 96, 116, 112, 76, 100, 88, 84, 72, 88, 76, 68, 72, 88, 120, 80, 96, 88, 92, 116, 80, 92, 112, 104, 156, 116, 80, 80, 92, 92, 92, 84, 100, 96, 116, 80, 76, 72, 76, 84, 84, 88, 176, 88, 96, 92, 92, 76, 68, 84, 128, 80, 84, 116, 88, 84, 88, 96, 76, 96, 80, 140, 92, 96, 100, 76, 84, 72, 80, 120, 80, 88, 124, 100, 76, 68, 80, 92, 84, 96, 92, 104, 92, 136, 116, 136, 112, 76, 84, 92, 176, 108, 104, 120, 96, 92, 92, 88, 88, 84, 92, 124, 84, 112, 92, 68, 88, 88, 80, 136, 92, 92, 124, 88, 72, 104, 100, 120, 108, 108, 84, 88, 92, 112, 112, 88, 112, 96, 132, 96, 88, 112, 116, 108, 100, 84, 96, 116, 100, 88, 132, 88, 92, 148, 96, 100, 92, 140, 88, 84, 84, 92, 96, 144, 112, 100, 100, 112, 104, 96, 84, 104, 104, 116, 76, 120, 148, 128, 92, 92, 100, 92, 108, 108, 92, 108, 112, 104, 112, 120, 144, 88, 100, 120, 100, 116, 144, 112, 104, 116, 132, 108]
    # spot_weights = [0.8500450744981615, 0.9130662981054044, 1.1182955151489753, 0.8500450744981615, 0.7896425509829849, 1.1182955151489753, 0.8500450744981615, 0.7896425509829849, 0.8500450744981615, 0.7318128814276639, 1.1182955151489753, 0.8500450744981615, 1.5169016419410182, 0.9787517225590299, 1.5169016419410182, 1.2690339517160951, 1.2690339517160951, 0.8500450744981615, 0.8500450744981615, 0.8500450744981615, 0.9130662981054044, 0.9130662981054044, 0.9787517225590299, 1.0471465141710325, 0.9130662981054044, 0.7896425509829849, 0.7318128814276639, 0.7896425509829849, 0.9130662981054044, 0.9130662981054044, 1.1182955151489753, 0.9130662981054044, 0.9130662981054044, 0.6236869225685145, 1.2690339517160951, 0.9130662981054044, 0.9130662981054044, 1.0471465141710325, 0.8500450744981615, 0.8500450744981615, 0.9130662981054044, 1.1182955151489753, 0.9787517225590299, 0.9787517225590299, 0.7896425509829849, 0.9787517225590299, 1.3487115367565212, 0.9787517225590299, 0.6236869225685145, 1.1182955151489753, 1.1922432533819738, 0.8500450744981615, 0.9787517225590299, 0.6765098627379946, 0.6236869225685145, 1.3487115367565212, 0.9787517225590299, 0.9130662981054044, 1.1922432533819738, 0.9130662981054044, 0.8500450744981615, 0.9130662981054044, 0.7318128814276639, 1.1182955151489753, 0.7896425509829849, 1.3487115367565212, 0.9130662981054044, 0.8500450744981615, 1.0471465141710325, 0.7318128814276639, 1.2690339517160951, 0.6765098627379946, 0.7896425509829849, 0.8500450744981615, 0.6765098627379946, 0.9787517225590299, 0.7896425509829849, 1.4313196472303777, 0.9787517225590299, 1.0471465141710325, 0.9130662981054044, 0.9787517225590299, 0.9130662981054044, 0.7318128814276639, 1.1182955151489753, 0.7318128814276639, 0.7318128814276639, 1.5169016419410182, 1.2690339517160951, 0.9130662981054044, 0.8500450744981615, 1.3487115367565212, 0.8500450744981615, 1.0471465141710325, 1.2690339517160951, 0.7896425509829849, 1.2690339517160951, 1.0471465141710325, 0.7896425509829849, 1.0471465141710325, 1.0471465141710325, 0.8500450744981615, 0.6765098627379946, 1.0471465141710325, 1.6055006073417841, 0.8500450744981615, 0.8500450744981615, 1.1182955151489753, 0.8500450744981615, 1.3487115367565212, 0.9787517225590299, 0.8500450744981615, 0.8500450744981615, 0.9787517225590299, 1.4313196472303777, 1.6055006073417841, 1.5169016419410182, 1.3487115367565212, 0.6765098627379946, 0.8500450744981615, 0.6236869225685145, 0.8500450744981615, 1.697159364754839, 0.6765098627379946, 1.4313196472303777, 0.9130662981054044, 1.5169016419410182, 1.6055006073417841, 0.7318128814276639, 0.9130662981054044, 0.7896425509829849, 0.8500450744981615, 1.4313196472303777, 1.1922432533819738, 1.0471465141710325, 0.7896425509829849, 0.9130662981054044, 0.7318128814276639, 0.8500450744981615, 0.9787517225590299, 1.4313196472303777, 0.9130662981054044, 1.1922432533819738, 1.1922432533819738, 0.7896425509829849, 1.0471465141710325, 0.7896425509829849, 0.8500450744981615, 1.6055006073417841, 0.8500450744981615, 0.8500450744981615, 1.2690339517160951, 0.8500450744981615, 0.9130662981054044, 0.9787517225590299, 0.7896425509829849, 0.8500450744981615, 1.1922432533819738, 0.7896425509829849, 0.7896425509829849, 0.6236869225685145, 0.9787517225590299, 0.6765098627379946, 1.3487115367565212, 1.1182955151489753, 0.7896425509829849, 0.9787517225590299, 0.9130662981054044, 1.1182955151489753, 1.6055006073417841, 0.7318128814276639, 1.1922432533819738, 0.9130662981054044, 0.7896425509829849, 0.7318128814276639, 0.9787517225590299, 1.1182955151489753, 0.9787517225590299, 0.9130662981054044, 0.8500450744981615, 0.7318128814276639, 1.3487115367565212, 0.6765098627379946, 1.3487115367565212, 1.1182955151489753, 0.7896425509829849, 0.9130662981054044, 0.9130662981054044, 1.4313196472303777, 0.7318128814276639, 0.8500450744981615, 0.9130662981054044, 1.4313196472303777, 0.9787517225590299, 0.8500450744981615, 1.1922432533819738, 1.0471465141710325, 0.8500450744981615, 1.697159364754839, 1.0471465141710325, 0.7896425509829849, 1.2690339517160951, 0.7896425509829849, 0.9130662981054044, 1.697159364754839, 0.8500450744981615, 0.9130662981054044, 1.2690339517160951, 0.9787517225590299, 0.9787517225590299, 1.0471465141710325, 1.697159364754839, 0.9130662981054044, 1.1922432533819738, 1.5169016419410182, 0.9787517225590299, 1.0471465141710325, 1.0471465141710325, 0.9130662981054044, 0.6765098627379946, 0.8500450744981615, 1.3487115367565212, 0.6765098627379946]
    # include_indices = [
    #     i
    #     for i, (v1, v2) in enumerate(zip(readout_fidelity_list, prep_fidelity_list))
    #     if (v1 is not None and v2 is not None)
    #     and all(isinstance(x, (int, float)) for x in (v1, v2))
    #     and not (math.isnan(v1) or math.isnan(v2))
    #     and v1 >= 0.7 and v2 >= 0.2
    # ]
    # include_indices = [0, 1, 2, 3, 4, 5, 6, 10, 12, 14, 15, 17, 18, 19, 21, 23, 24, 26, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 121, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 169, 170, 171, 173, 174, 177, 179, 180, 181, 182, 184, 186, 188, 189, 190, 191, 192, 193, 194, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 270, 271, 274, 275, 276, 277, 280, 281, 282, 283, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307]
    ## 2 oruen
    # include_indices = [0, 1, 2, 4, 9, 10, 12, 14, 15, 17, 18, 20, 22, 23, 24, 27, 28, 29, 30, 33, 34, 37, 40, 41, 42, 45, 46, 49, 50, 51, 54, 55, 57, 58, 59, 60, 62, 65, 67, 68, 69, 72, 73, 75, 78, 79, 80, 81, 82, 85, 86, 87, 88, 90, 94, 95, 98, 99, 101, 102, 104, 106, 107, 111, 113, 114, 116, 117, 119, 122, 123, 125, 126, 127, 128, 130, 131, 133, 134, 135, 136, 137, 142, 143, 144, 145, 146, 148, 149, 151, 153, 155, 158, 161, 163, 164, 165, 166, 167, 170, 172, 173, 174, 175, 178, 181, 183, 185, 186, 187, 191, 192, 193, 195, 196, 197, 199, 200, 201, 203, 205, 207, 210, 211, 212, 214, 216, 218, 220, 221, 223, 225, 226, 227, 228, 229, 230, 233, 235, 237, 238, 239, 242, 244, 245, 246, 247, 249, 250, 252, 253]
    # include_indices = [i for i, val in enumerate(prep_fidelity_list) if val >= 0.4 or val is None]
    include_indices =  [i for i, val in enumerate(snr_float) if val >= 0.00]
    # import math
    # indices_113_MHz = [0, 1, 4, 8, 12, 17, 21, 22, 24, 29, 30, 31, 33, 34, 40, 41, 42, 43, 45, 46, 49, 58, 59, 61, 63, 64, 65, 70, 72, 73, 74, 76, 77, 79, 81, 83, 84, 85, 86, 87, 89, 91, 93, 95, 96, 97, 99, 101, 103, 105, 106, 110, 111, 115, 116, 117, 118, 119, 121, 126, 127, 129, 131, 132]
    # indices_217_MHz = [3, 6, 7, 9, 10, 11, 13, 14, 15, 18, 23, 26, 27, 28, 35, 36, 38, 39, 44, 47, 48, 50, 51, 53, 54, 55, 56, 57, 62, 66, 67, 68, 69, 71, 75, 80, 82, 88, 90, 98, 100, 102, 104, 109, 113, 114, 120, 122, 125, 128, 130, 133, 134, 135]
    # include_indices = np.sort(indices_113_MHz + indices_217_MHz)
    # include_indices = [
    #     i for i, val in enumerate(prep_fidelity_list)
    #     if (val is None) or (isinstance(val, (int, float)) and not math.isnan(val) and val >= 0.5)
    # ]

    print(np.sort(list(include_indices)))

    indices = np.sort(list(include_indices))
    print(", ".join(str(i) for i in indices))
    # print("[" + ", ".join(map(str, np.sort(list(include_indices)))) + "]")

    # include_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 40, 41, 43, 46, 48, 49, 50, 51, 53, 55, 58, 60, 61, 62, 63, 64, 65, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 82, 83, 88, 90, 91, 93, 94, 96, 98, 99, 102, 103, 104, 105, 107, 108, 109, 110, 113, 114, 115, 116, 117, 118, 121, 123, 125, 126, 127, 128]
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind not in drop_indices]
    # print(len(include_indices))
    # fmt: on
    filtered_reordered_coords = [filtered_reordered_coords[i] for i in include_indices]
    print(f"len filtered_reordered_coords: {len(filtered_reordered_coords)}")
    # # # select_half_left_side_nvs_and_plot(nv_coordinates_filtered)
    spot_weights = np.array(
        [weight for i, weight in enumerate(spot_weights) if i in include_indices]
    )
    print(f"len spot_weights: {len(spot_weights)}")
    # filtered_pol_durs = [pol_duration_list[i] for i in include_indices]
    # filtered_scc_durs = [scc_duration_list[i] for i in include_indices]
    # print(filtered_pol_durs)
    # print(f"len spot_weights: {len(spot_weights)}")
    # print(filtered_scc_durs)


    # sys.exit()
    # print(len(spot_weights_filtered))
    # # sys.exit()
    # aom_voltage = 0.2861
    aom_voltage = 0.3084 ### 223NVs
    aom_voltage = 0.299064 ###204NVs
    # a, b, c = [3.7e5, 6.97, 8e-14]
    # a, b, c = 161266.751, 6.617, -19.492
    a, b, c = 1.5133e04, 2.6976, -38.63  # UPDATED 2025-09-17

    total_power = a * (aom_voltage) ** b + c
    # print(total_power)
    normalized_spot_weigths = spot_weights / np.sum(spot_weights)
    nv_powers = total_power * normalized_spot_weigths
    # print(nv_powers)
    # print(nv_powers)
    # calcualted_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.3)
    # updated_spot_weights = linear_weights(filtered_reordered_counts, alpha=0.6)
    nv_powers_filtered = np.array(
        [power for i, power in enumerate(nv_powers) if i in include_indices]
    )
    print(nv_powers_filtered)
    # Create a copy or initialize spot weights for modification
    # updated_spot_weights = curve_extreme_weights_simple(
    #     spot_weights, scaling_factor=1.0
    # )
    # filtered_reordered_spot_weights = np.array(
    #     [
    #         weight
    #         for i, weight in enumerate(updated_spot_weights)
    #         if i in include_indices
    #     ]
    # )
    # updated_spot_weights = spot_weights
    # print(filter_and_reorder_nv_coords)
    # updated_spot_weights = np.array(
    #     [w for i, w in enumerate(updated_spot_weights_0) if i in include_indices]
    # )
    # updated_spot_weights = spot_weights
    # updated_spot_weights = curve_extreme_weights_simple(nv_powers)
    # updated_spot_weights = curve_inverse_counts(filtered_reordered_spot_weights)
    # drop_indices = [150, 161, 392, 403]
    # updated_spot_weights = [
    #     val for ind, val in enumerate(updated_spot_weights) if ind not in drop_indices
    # ]
    # filtered_reordered_coords = [
    #     filtered_reordered_coords[ind]
    #     for ind in range(len(filtered_reordered_coords))
    #     if ind not in drop_indices
    # ]
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind not in drop_indices]

    ### new weithgs for 75 NVs
    # fmt: off
    # spot_weights = [1.5095766644650288, 0.9618852387815993, 1.0430775922714643, 0.4436039796130322, 1.0430775922714643, 1.5095766644650288, 0.30376599435099677, 1.310540492774533, 1.4079312952605532, 0.6747045661156417, 0.4964211566919598, 0.7410295032188732, 1.0430775922714643, 1.0430775922714643, 0.6118852419676627, 0.552486087697023, 0.552486087697023, 0.3939475438077535, 0.7410295032188732, 0.9618852387815993, 0.8845355774031071, 0.3939475438077535, 0.8109450107566616, 1.1281958109476427, 1.5095766644650288, 1.4079312952605532, 1.4079312952605532, 0.552486087697023, 0.4436039796130322, 0.3939475438077535, 1.2173226567254087, 0.7410295032188732, 0.8845355774031071, 0.552486087697023, 1.0430775922714643, 0.4964211566919598, 1.5095766644650288, 1.4079312952605532, 0.6747045661156417, 0.6118852419676627, 1.5095766644650288, 0.6118852419676627, 0.3939475438077535, 1.1281958109476427, 0.8845355774031071, 1.2173226567254087, 1.4079312952605532, 0.6118852419676627, 1.310540492774533, 1.5095766644650288, 1.1281958109476427, 1.310540492774533, 0.7410295032188732, 1.310540492774533, 1.5095766644650288, 0.8845355774031071, 1.5095766644650288, 0.7410295032188732, 0.9618852387815993, 0.3473642711816343, 0.6747045661156417, 0.6747045661156417, 1.310540492774533, 0.6118852419676627, 0.6118852419676627, 1.4079312952605532, 0.3939475438077535, 0.6118852419676627, 1.2173226567254087, 1.4079312952605532, 1.4079312952605532, 0.6747045661156417, 1.310540492774533, 1.0430775922714643, 1.0430775922714643, 1.0430775922714643, 1.4079312952605532, 1.2173226567254087, 1.4079312952605532, 0.3473642711816343, 1.2173226567254087, 0.8109450107566616, 0.9618852387815993, 1.5095766644650288, 1.4079312952605532, 1.2173226567254087, 0.8109450107566616, 1.310540492774533, 0.552486087697023, 1.4079312952605532, 1.0430775922714643, 0.7410295032188732, 0.4964211566919598, 0.7410295032188732, 0.6747045661156417, 1.310540492774533, 0.9618852387815993, 1.5095766644650288, 0.8845355774031071, 0.8109450107566616, 1.310540492774533, 1.4079312952605532, 0.8109450107566616, 1.310540492774533, 1.1281958109476427, 1.4079312952605532, 1.2173226567254087, 0.8845355774031071, 0.7410295032188732, 0.6118852419676627, 1.1281958109476427, 0.8109450107566616, 1.1281958109476427, 0.7410295032188732, 1.1281958109476427, 1.5095766644650288, 0.6118852419676627, 1.5095766644650288, 1.2173226567254087, 0.7410295032188732, 0.6118852419676627, 0.7410295032188732, 1.310540492774533, 1.0430775922714643, 1.4079312952605532, 0.7410295032188732, 1.2173226567254087, 1.5095766644650288, 1.310540492774533, 0.9618852387815993, 1.310540492774533, 0.6747045661156417, 0.7410295032188732, 1.0430775922714643, 1.4079312952605532, 0.6747045661156417, 0.3939475438077535, 0.6118852419676627, 0.7410295032188732, 1.310540492774533, 1.5095766644650288, 1.2173226567254087, 1.2173226567254087, 0.9618852387815993, 0.6747045661156417, 1.2173226567254087, 1.2173226567254087, 0.4964211566919598, 1.2173226567254087, 0.6118852419676627, 0.8109450107566616, 1.0430775922714643, 1.4079312952605532, 0.3939475438077535, 0.7410295032188732, 1.4079312952605532, 0.9618852387815993, 1.310540492774533, 1.1281958109476427, 1.4079312952605532, 1.5095766644650288, 1.0430775922714643, 0.7410295032188732, 1.4079312952605532, 1.5095766644650288, 1.4079312952605532, 1.4079312952605532, 0.552486087697023, 0.3939475438077535, 0.8845355774031071, 0.552486087697023, 0.6118852419676627, 0.6747045661156417, 0.6747045661156417, 0.552486087697023, 1.4079312952605532, 1.5095766644650288, 1.4079312952605532, 1.1281958109476427, 0.8109450107566616, 0.8845355774031071, 1.1281958109476427, 0.9618852387815993, 1.1281958109476427, 1.0430775922714643, 1.310540492774533, 0.6747045661156417, 1.0430775922714643, 1.5095766644650288, 1.2173226567254087, 1.4079312952605532, 1.1281958109476427, 1.0430775922714643, 0.9618852387815993, 0.8109450107566616, 0.552486087697023, 0.30376599435099677, 1.310540492774533, 0.4964211566919598, 0.6118852419676627, 0.7410295032188732, 0.7410295032188732, 1.2173226567254087, 0.8845355774031071, 0.8109450107566616, 1.5095766644650288, 0.8845355774031071, 0.9618852387815993, 1.2173226567254087, 1.2173226567254087, 1.1281958109476427, 0.8109450107566616, 0.552486087697023, 1.2173226567254087, 1.2173226567254087, 0.9618852387815993, 0.6118852419676627, 1.310540492774533, 0.9618852387815993, 1.4079312952605532, 0.8109450107566616, 0.6118852419676627, 0.8109450107566616, 1.4079312952605532, 0.7410295032188732, 1.2173226567254087, 1.310540492774533, 0.7410295032188732, 0.4436039796130322, 1.0430775922714643, 1.5095766644650288, 1.1281958109476427, 0.8109450107566616, 0.8109450107566616, 0.7410295032188732, 1.0430775922714643, 1.310540492774533, 1.2173226567254087, 0.8109450107566616, 0.7410295032188732, 1.4079312952605532, 1.4079312952605532, 1.2173226567254087, 0.4964211566919598, 1.4079312952605532, 0.9618852387815993, 1.5095766644650288, 1.2173226567254087, 0.6118852419676627, 1.0430775922714643, 0.7410295032188732, 0.9618852387815993, 0.6118852419676627, 0.4964211566919598, 1.5095766644650288, 0.8845355774031071, 1.1281958109476427, 1.4079312952605532, 0.9618852387815993, 0.6747045661156417, 1.310540492774533, 0.8845355774031071, 0.4436039796130322, 1.2173226567254087, 0.6747045661156417, 0.3939475438077535, 0.6118852419676627, 0.9618852387815993, 0.6747045661156417, 0.7410295032188732, 1.4079312952605532, 0.8845355774031071, 0.7410295032188732, 0.9618852387815993, 1.1281958109476427, 1.2173226567254087, 0.8845355774031071, 0.7410295032188732, 0.552486087697023, 1.1281958109476427, 1.1281958109476427, 1.4079312952605532, 1.0430775922714643, 0.8845355774031071, 0.552486087697023, 0.8109450107566616, 0.6747045661156417, 1.5095766644650288, 1.310540492774533, 1.1281958109476427, 0.8109450107566616, 1.2173226567254087, 1.0430775922714643, 1.2173226567254087, 0.7410295032188732, 1.4079312952605532, 1.5095766644650288, 1.0430775922714643, 1.1281958109476427, 1.4079312952605532, 1.310540492774533, 0.9618852387815993, 1.4079312952605532, 1.5095766644650288, 0.6118852419676627, 0.8109450107566616, 1.5095766644650288, 1.310540492774533, 0.6747045661156417, 1.310540492774533, 0.552486087697023, 1.1281958109476427, 1.5095766644650288, 0.6747045661156417, 0.9618852387815993, 0.6747045661156417, 1.310540492774533, 0.8845355774031071, 1.4079312952605532, 0.6118852419676627, 1.2173226567254087, 0.8109450107566616, 0.552486087697023, 1.5095766644650288, 0.6747045661156417, 0.7410295032188732, 1.310540492774533, 0.6118852419676627, 0.4436039796130322, 1.310540492774533, 0.9618852387815993, 1.2173226567254087, 1.310540492774533, 1.310540492774533, 1.310540492774533, 1.310540492774533, 1.310540492774533, 0.7410295032188732, 0.9618852387815993, 1.4079312952605532, 1.310540492774533, 0.8845355774031071, 0.9618852387815993, 0.8845355774031071, 1.4079312952605532, 0.4964211566919598, 1.0430775922714643, 1.4079312952605532, 0.6747045661156417, 0.7410295032188732, 0.7410295032188732, 1.2173226567254087, 1.2173226567254087, 0.552486087697023, 1.310540492774533, 0.7410295032188732, 1.4079312952605532, 1.0430775922714643, 0.6118852419676627, 1.310540492774533, 0.8109450107566616, 1.5095766644650288, 0.3939475438077535, 0.8845355774031071, 0.9618852387815993, 0.8109450107566616, 0.8845355774031071, 1.1281958109476427, 1.4079312952605532, 0.8109450107566616]
    #fmt: off
    # spot_weights_med = np.median(spot_weights)
    # spot_weights = [spot_weights_med if spot_weights[ind] > 3 else spot_weights[ind] for ind in range(len(spot_weights))]
    updated_spot_weights = curve_extreme_weights_simple(
    spot_weights, scaling_factor=1.0
    )
    # updated_spot_weights = spot_weights
    # Update weights for the specified indices using the calculated weights
    # fmt: off
    # selected_indices_68MHz = [0, 7, 8, 9, 11, 14, 18, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 35, 38, 44, 45, 46, 47, 48, 49, 53, 55, 57, 58, 60, 62, 64, 66, 67, 68, 69, 70, 71, 72, 73]
    # selected_indices_185MHz  =[0, 1, 2, 3, 4, 5, 6, 10, 12, 13, 15, 16, 17, 19, 20, 21, 23, 29, 34, 36, 39, 40, 41, 42, 43, 50, 51, 52, 54, 56, 59, 61, 63, 65, 74]
    # selected_indices = selected_indices_185MHz
    # fmt: on
    # updated_spot_weights = [
    #     val for ind, val in enumerate(updated_spot_weights) if ind in selected_indices
    # ]
    # filtered_reordered_coords = [
    #     filtered_reordered_coords[ind]
    #     for ind in range(len(filtered_reordered_coords))
    #     if ind in selected_indices
    # ]
    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind in selected_indices]

    ####
    filtered_total_power = np.sum(nv_powers_filtered)
    print(total_power)
    adjusted_aom_voltage = ((filtered_total_power - c) / a) ** (1 / b)
    print("Adjusted Voltages (V):", adjusted_aom_voltage)
    # sys.exit()
    filtered_reordered_spot_weights = updated_spot_weights
    print("filtered_reordered_spot_weights_len:", len(filtered_reordered_spot_weights))
    print("filtered_reordered_coords_len:", len(filtered_reordered_coords))
    print("filtered_nv_power_len:", len(nv_powers_filtered))
    print("NV Index | Coords    |   previous weights")
    print("-" * 60)
    for idx, (coords, weight) in enumerate(
        zip(filtered_reordered_coords, filtered_reordered_spot_weights)
    ):
        print(f"{idx + 1:<8} | {coords} | {weight:.3f}")

    print(adjusted_aom_voltage)

    # print(np.max(filtered_reordered_spot_weights))
    # print(np.median(filter_and_reorder_nv_coords))
    # sys.exit()
    # print(len(spot_weights))
    # updated_spot_weights = filtered_reordered_counts
    # spot_weights = updated_spot_weights
    # spot_weights = linear_weights(filtered_reordered_counts, alpha=0.9)
    # spot_weights = non_linear_weights_adjusted(
    #     filtered_intensities, alpha=0.9, beta=0.3, threshold=0.9
    # )
    # spot_weights = sigmoid_weights(filtered_intensities, threshold=0, beta=0.005)
    # Print some diagnostics
    # Update spot weights for NVs with low fidelity

    # Calculate the spot weights based on the integrated intensities
    # spot_weights = non_linear_weights(filtered_intensities, alpha=0.9)

    # # Save the filtered results
    # save_results(
    #     filtered_reordered_coords,
    #     filtered_reordered_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_204nvs_reordered.npz",
    # )

    # # Plot the original image with circles around each NV
    fig, ax = plt.subplots()
    title = "LASER_589_Ref, 50ms"
    kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # Draw circles and index numbers
    for idx, coord in enumerate(filtered_reordered_coords):
        circ = plt.Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
        ax.add_patch(circ)
        # Place text just above the circle
        ax.text(
            coord[0],
            coord[1] - sigma - 1,
            str(idx),
            color="white",
            fontsize=8,
            ha="center",
        )

    plt.show(block=True)
