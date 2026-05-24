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

from matplotlib.path import Path

# -----------------------------
# 2D rotated Gaussian model
# -----------------------------
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    2D rotated Gaussian:
      g(x,y) = offset + amplitude * exp(-Q)
    where Q is the rotated quadratic form.

    NOTE: amplitude is peak height above offset.
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)

    # Prevent zero / negative sigmas from blowing up numerically
    sigma_x = max(float(sigma_x), 1e-6)
    sigma_y = max(float(sigma_y), 1e-6)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    a = (cos_t**2) / (2 * sigma_x**2) + (sin_t**2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (sin_t**2) / (2 * sigma_x**2) + (cos_t**2) / (2 * sigma_y**2)

    g = offset + amplitude * np.exp(
        -(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
    )
    return g.ravel()


# -----------------------------
# Fit Gaussian around a coordinate
# -----------------------------
def fit_gaussian(image, coord, window_size=12, normalize=False, maxfev=5000, return_fit=False):
    """
    Fit a 2D Gaussian in a window around coord=(x0,y0).

    Coordinate convention:
      - coord is (x, y) in pixel units
      - image is indexed as image[y, x]

    Returns by default:
      (fitted_x, fitted_y, amplitude)

    If return_fit=True:
      (fitted_x, fitted_y, amplitude, gaussian_weight, popt)

    gaussian_weight = 2*pi*amplitude*sigma_x*sigma_y  (integral above offset)
    """
    x0, y0 = coord
    img_h, img_w = image.shape  # (y, x)

    # window bounds (ensure within image)
    x_min = max(int(np.floor(x0 - window_size)), 0)
    x_max = min(int(np.ceil(x0 + window_size + 1)), img_w)
    y_min = max(int(np.floor(y0 - window_size)), 0)
    y_max = min(int(np.ceil(y0 + window_size + 1)), img_h)

    # region must be at least 2x2
    if (x_max - x_min) <= 1 or (y_max - y_min) <= 1:
        if return_fit:
            return float(x0), float(y0), 0.0, 0.0, None
        return float(x0), float(y0), 0.0

    # cutout
    image_cutout = np.asarray(image[y_min:y_max, x_min:x_max], dtype=float)
    if image_cutout.size == 0:
        if return_fit:
            return float(x0), float(y0), 0.0, 0.0, None
        return float(x0), float(y0), 0.0

    # mesh grid in ABSOLUTE pixel coords
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    x, y = np.meshgrid(x_range, y_range)

    # optional normalization (NOTE: if normalize=True, amplitude becomes unitless)
    if normalize:
        den = float(np.max(image_cutout) - np.min(image_cutout))
        if den <= 0:
            if return_fit:
                return float(x0), float(y0), 0.0, 0.0, None
            return float(x0), float(y0), 0.0
        image_fit = (image_cutout - np.min(image_cutout)) / den
    else:
        image_fit = image_cutout

    # initial guesses from data
    offset0 = float(np.percentile(image_fit, 10))
    peak0 = float(np.max(image_fit))
    amp0 = max(peak0 - offset0, 1e-6)

    # initial center from local maximum in cutout
    iy, ix = np.unravel_index(np.argmax(image_fit), image_fit.shape)
    xo0 = float(x_min + ix)
    yo0 = float(y_min + iy)

    # initial sigma guess
    sig0 = max(1.0, window_size / 4.0)
    initial_guess = (amp0, xo0, yo0, sig0, sig0, 0.0, offset0)

    # bounds (avoid sigma=0; restrict theta to remove degeneracy)
    eps = 0.3
    bounds = (
        (0.0, x_min, y_min, eps, eps, -np.pi / 2, -np.inf),  # lower
        (np.inf, x_max, y_max, window_size * 2, window_size * 2, np.pi / 2, np.inf),  # upper
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (x, y),
            image_fit.ravel(),
            p0=initial_guess,
            bounds=bounds,
            maxfev=maxfev,
        )
        amplitude, fitted_x, fitted_y, sigma_x, sigma_y, theta, offset = popt
        gaussian_weight = float(2 * np.pi * amplitude * sigma_x * sigma_y)

        if return_fit:
            return float(fitted_x), float(fitted_y), float(amplitude), gaussian_weight, popt
        return float(fitted_x), float(fitted_y), float(amplitude)

    except Exception as e:
        print(f"Fit failed for NV at ({x0}, {y0}): {e}")
        if return_fit:
            return float(x0), float(y0), 0.0, 0.0, None
        return float(x0), float(y0), 0.0


# -----------------------------
# Disk integration weight (sum in radius)
# -----------------------------
def integrate_intensity(image_array, nv_coords, sigma):
    """
    Sum intensity around each (x,y) coord within a disk of radius=sigma.
    Returns a list of sums in original image units.

    NOTE: skimage.draw.disk expects center=(row, col) = (y, x),
          so we convert (x,y) -> (y,x).
    """
    intensities = []
    for (x, y) in nv_coords:
        rr, cc = disk((y, x), radius=sigma, shape=image_array.shape)
        intensities.append(float(np.sum(image_array[rr, cc])))
    return intensities


# -----------------------------
# Example: filter + refit coords + keep weights aligned
# -----------------------------
def refine_coords_after_fitting(
    img_array,
    filtered_reordered_coords,
    filtered_reordered_spot_weights=None,
    window_size=12,
    min_amplitude=0.0,
    replace_weights_with="none",  # "none" | "amplitude" | "gaussian_weight"
    normalize=False,
):
    """
    Returns:
      new_coords, new_weights, fitted_amplitudes, fitted_gaussian_weights
    """
    new_coords = []
    new_weights = [] if filtered_reordered_spot_weights is not None else None
    fitted_amplitudes = []
    fitted_gaussian_weights = []

    if filtered_reordered_spot_weights is None:
        iterable = [(c, None) for c in filtered_reordered_coords]
    else:
        iterable = list(zip(filtered_reordered_coords, filtered_reordered_spot_weights))

    for coord, w in iterable:
        fx, fy, amp, gwt, popt = fit_gaussian(
            img_array,
            coord,
            window_size=window_size,
            normalize=normalize,
            return_fit=True,
        )

        # reject bad fits
        if amp <= min_amplitude or popt is None:
            continue

        new_coords.append([fx, fy])
        fitted_amplitudes.append(amp)
        fitted_gaussian_weights.append(gwt)

        if new_weights is not None:
            if replace_weights_with == "amplitude":
                new_weights.append(float(amp))
            elif replace_weights_with == "gaussian_weight":
                new_weights.append(float(gwt))
            else:
                new_weights.append(w)

    return new_coords, new_weights, fitted_amplitudes, fitted_gaussian_weights


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
    # print(included_indices)
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


def curve_inverse_counts(counts, scaling_factor=0.2):
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


def points_in_region(points, region):
    """
    points: (N, 2) array of [x, y]
    region: dict describing one exclusion region
    """
    pts = np.asarray(points, dtype=float)

    if region["type"] == "polygon":
        path = Path(np.asarray(region["vertices"], dtype=float))
        return path.contains_points(pts)

    elif region["type"] == "rect":
        xmin, xmax = region["xmin"], region["xmax"]
        ymin, ymax = region["ymin"], region["ymax"]
        x, y = pts[:, 0], pts[:, 1]
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    elif region["type"] == "circle":
        cx, cy = region["center"]
        r = region["radius"]
        x, y = pts[:, 0], pts[:, 1]
        return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    else:
        raise ValueError(f"Unknown region type: {region['type']}")


def remove_coords_in_regions(coords, regions):
    """
    Remove coordinates that fall inside ANY exclusion region.

    coords: list of [x, y] or (N, 2) numpy array
    regions: list of region dicts
    """
    coords = np.asarray(coords, dtype=float)
    remove_mask = np.zeros(len(coords), dtype=bool)

    for region in regions:
        remove_mask |= points_in_region(coords, region)

    kept = coords[~remove_mask]
    removed = coords[remove_mask]
    return kept, removed, remove_mask

class ManualPolygonSelector:
    def __init__(self, ax):
        self.ax = ax
        self.verts = []
        self.polygons = []
        self.line, = ax.plot([], [], "r-", lw=2)
        self.points, = ax.plot([], [], "ro", ms=4)

        self.cid_click = ax.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = ax.figure.canvas.mpl_connect("key_press_event", self.on_key)

        ax.set_title(
            "Left click = add point, Backspace = undo last point\n"
            "Enter = save polygon, q = finish"
        )

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.verts.append([event.xdata, event.ydata])
            self.redraw()

    def on_key(self, event):
        if event.key == "backspace":
            if len(self.verts) > 0:
                self.verts.pop()
                self.redraw()

        elif event.key == "enter":
            if len(self.verts) >= 3:
                poly = np.array(self.verts.copy())
                self.polygons.append(poly)
                print(f"\nSaved region #{len(self.polygons)}")
                for x, y in poly:
                    print(f"[{x:.3f}, {y:.3f}],")
                self.verts = []
                self.redraw()

        elif event.key == "q":
            plt.close(event.canvas.figure)

    def redraw(self):
        if len(self.verts) > 0:
            arr = np.array(self.verts)
            self.line.set_data(arr[:, 0], arr[:, 1])
            self.points.set_data(arr[:, 0], arr[:, 1])
        else:
            self.line.set_data([], [])
            self.points.set_data([], [])
        self.ax.figure.canvas.draw_idle()


import numpy as np


def filter_and_rescale_coords_for_new_roi(
    nv_coordinates,
    spot_weights=None,
    old_roi=(40, 45, 450, 450),
    new_roi=(40, 90, 400, 400),
    rescale=False,
    coords_are_roi_local=True,
):
    """
    Convert coordinates from old ROI to new ROI, and remove coordinates outside new ROI.

    Parameters
    ----------
    nv_coordinates : array-like, shape (N, 2)
        NV coordinates.
    spot_weights : array-like or None
        Per-NV weights. If None, all ones are used.
    old_roi : tuple
        (x0, y0, width, height) of old ROI.
    new_roi : tuple
        (x0, y0, width, height) of new ROI.
    rescale : bool
        If False: only shift coordinates into new ROI frame, then crop.
        If True: also rescale from old ROI size to new ROI size.
    coords_are_roi_local : bool
        If True, input coords are local to old ROI.
        If False, input coords are already in full-image/global coordinates.

    Returns
    -------
    new_coords : np.ndarray, shape (M, 2)
        Coordinates in the new ROI frame.
    new_weights : np.ndarray, shape (M,)
        Weights for kept coordinates.
    keep_indices : np.ndarray
        Indices of kept coordinates in the original array.
    keep_mask : np.ndarray
        Boolean mask of kept coordinates.
    global_coords_kept : np.ndarray, shape (M, 2)
        Kept coordinates in full-image/global frame.
    """
    nv_coordinates = np.asarray(nv_coordinates, dtype=float)

    if spot_weights is None:
        spot_weights = np.ones(len(nv_coordinates), dtype=float)
    else:
        spot_weights = np.asarray(spot_weights, dtype=float)

    old_x0, old_y0, old_w, old_h = old_roi
    new_x0, new_y0, new_w, new_h = new_roi

    # Step 1: move to global/full-image coordinates
    if coords_are_roi_local:
        global_coords = nv_coordinates.copy()
        global_coords[:, 0] += old_x0
        global_coords[:, 1] += old_y0
    else:
        global_coords = nv_coordinates.copy()

    # Step 2: map into new ROI frame
    if rescale:
        # Interpret coordinates relative to old ROI and scale into new ROI size
        if coords_are_roi_local:
            old_local = nv_coordinates.copy()
        else:
            old_local = global_coords.copy()
            old_local[:, 0] -= old_x0
            old_local[:, 1] -= old_y0

        sx = new_w / old_w
        sy = new_h / old_h

        new_local = np.empty_like(old_local)
        new_local[:, 0] = old_local[:, 0] * sx
        new_local[:, 1] = old_local[:, 1] * sy

        # recompute corresponding global coords after rescaling
        global_coords_rescaled = np.empty_like(new_local)
        global_coords_rescaled[:, 0] = new_local[:, 0] + new_x0
        global_coords_rescaled[:, 1] = new_local[:, 1] + new_y0

        coords_in_new_roi = new_local
        global_coords_used = global_coords_rescaled
    else:
        # No rescaling: just shift into new ROI frame
        coords_in_new_roi = np.empty_like(global_coords)
        coords_in_new_roi[:, 0] = global_coords[:, 0] - new_x0
        coords_in_new_roi[:, 1] = global_coords[:, 1] - new_y0
        global_coords_used = global_coords

    # Step 3: keep only points inside new ROI
    keep_mask = (
        (coords_in_new_roi[:, 0] >= 0)
        & (coords_in_new_roi[:, 0] < new_w)
        & (coords_in_new_roi[:, 1] >= 0)
        & (coords_in_new_roi[:, 1] < new_h)
    )

    keep_indices = np.where(keep_mask)[0]

    new_coords = coords_in_new_roi[keep_mask]
    new_weights = spot_weights[keep_mask]
    global_coords_kept = global_coords_used[keep_mask]

    return new_coords, new_weights, keep_indices, keep_mask, global_coords_kept

def remap_single_coord(coord, old_roi, new_roi, rescale=False):
    coord = np.asarray(coord, dtype=float).reshape(1, 2)
    new_coord, _, _, _, _ = filter_and_rescale_coords_for_new_roi(
        coord,
        spot_weights=np.array([1.0]),
        old_roi=old_roi,
        new_roi=new_roi,
        rescale=rescale,
        coords_are_roi_local=True,
    )
    if len(new_coord) == 0:
        raise ValueError("Reference NV is outside the new ROI.")
    return new_coord[0].tolist()




def update_calibration_pixel_coords(
    calibration_coords_pixel,
    old_roi=(40, 45, 450, 450),
    new_roi=(40, 90, 400, 400),
    rescale=False,
):
    coords = np.asarray(calibration_coords_pixel, dtype=float)

    old_x0, old_y0, old_w, old_h = old_roi
    new_x0, new_y0, new_w, new_h = new_roi

    if rescale:
        sx = new_w / old_w
        sy = new_h / old_h
        new_coords = coords.copy()
        new_coords[:, 0] *= sx
        new_coords[:, 1] *= sy
    else:
        # old ROI-local -> global -> new ROI-local
        new_coords = coords.copy()
        new_coords[:, 0] = coords[:, 0] + old_x0 - new_x0
        new_coords[:, 1] = coords[:, 1] + old_y0 - new_y0

    keep_mask = (
        (new_coords[:, 0] >= 0) & (new_coords[:, 0] < new_w) &
        (new_coords[:, 1] >= 0) & (new_coords[:, 1] < new_h)
    )

    return new_coords[keep_mask].tolist(), keep_mask




# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Parameters
    remove_outliers_flag = False  # Set this flag to enable/disable outlier removal
    reorder_coords_flag = True  # Set this flag to enable/disable reordering of NVs
    data = dm.get_raw_data(
        # file_stem="2026_03_10-16_56_54-combined_image_array", load_npz=True
        file_stem="2026_05_23-22_55_40-combined_image_array", load_npz=True
    )
    # img_array = np.array(data["ref_img_array"])
    img_array = data["img_array"]
    nv_coordinates, spot_weights = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_6837nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_6904nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_3986nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_3554nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_3366nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1460nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1487nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1348nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1306nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1275nvs_reordered.npz",
        file_path="slmsuite/nv_blob_detection/nv_blob_1274nvs_reordered_after_sample_rotation.npz"
    )
    
    # fitted_data = dm.get_raw_data(file_stem="2026_03_23-16_39_03-optimal_values_2026_03_22-21_49_52-qnami-nv0_2026_02_20")
    # saved_summary = dm.get_raw_data(file_stem="2026_03_26-18_13_04-recomputed_summary_w_1_2_1_2026_03_26-15_04_19-optimization_processed_full_raw_data")
    # optimal_weigths = saved_summary["optimal_weights"]
    # aom_voltage = saved_summary["aom_voltage"]
    # print (aom_voltage)
    # print(aom_voltage)
    
    # new_coords, keep_mask = update_calibration_pixel_coords(
    #     calibration_coords_pixel,
    #     old_roi=(40, 45, 450, 450),
    #     new_roi=(55, 85, 400, 400),
    #     rescale=False,
    # )

    # print(new_coords)
    # print(keep_mask)
    
    
    # kept_weights = spot_weights[~remove_mask]
    # removed_weights = spot_weights[remove_mask]

    # print("Original:", len(nv_coordinates))
    # print("Kept:", len(kept_coords))
    # print("Removed:", len(removed_coords))

    # Convert coordinates to a standard format (lists of lists)
    # nv_coordinates = [[coord[0] - 3, coord[1] + 3] for coord in nv_coordinates]
    # nv_coordinates = [list(coord) for coord in nv_coordinates]
    # Filter NV coordinates: Keep only those where both x and y are in [0, 250]
    
    # mask = (
    #     (nv_coordinates[:, 0] >= 5) & (nv_coordinates[:, 0] < 380) &
    #     (nv_coordinates[:, 1] >= 0) & (nv_coordinates[:, 1] < 375)
    # )

    # nv_coordinates = nv_coordinates[mask]
    # spot_weights = spot_weights[mask]
    # nv_coordinates = np.asarray([list(coord) for coord in nv_coordinates], dtype=float)
    # spot_weights = np.ones(nv_coordinates.shape[0], dtype=float)
    # nv_coordinates = np.array([
    #     [214.573, 203.991], 
    #     [360.977, 57.633], 
    #     [225.414, 361.866], 
    #     [30.961, 57.878],
    #     ])
    # spot_weights = np.ones(nv_coordinates.shape[0], dtype=float)
    # old_roi = (40, 45, 450, 450)
    # old_roi = (55, 85, 400, 400)
    # new_roi = (60, 87, 375, 375)
    # nv_coordinates, spot_weights, keep_indices, keep_mask, global_coords_kept = (
    #     filter_and_rescale_coords_for_new_roi(
    #         nv_coordinates,
    #         spot_weights=spot_weights,
    #         old_roi=old_roi,
    #         new_roi=new_roi,
    #         rescale=False,
    #         coords_are_roi_local=True,
    #     )
    # )
    
    
    filtered_reordered_coords = np.round(nv_coordinates, 3)
    filtered_reordered_spot_weights = np.round(spot_weights,3)
    # print(filtered_reordered_coords)
    # sys.exit()
    # dx = 214.573 - 216.4199
    # dy = 203.991 - 195.96799

    # nv_coordinates = np.asarray(nv_coordinates, dtype=float)
    # nv_coordinates[:, 0] += dx
    # nv_coordinates[:, 1] += dy

    
    # reference_nv = [214.573, 203.991]
    # filtered_reordered_coords, filtered_reordered_spot_weights, include_indices = (
    #     filter_and_reorder_nv_coords(
    #         nv_coordinates, spot_weights, reference_nv, min_distance=4.0
    #     )
    # )

    # print(f"After filtering: {len(nv_coordinates)} NVs")
    # cx, cy = 215, 230
    # r = 220

    # mask = (nv_coordinates[:, 0] - cx)**2 + (nv_coordinates[:, 1] - cy)**2 <= r**2

    # nv_coordinates_filtered = nv_coordinates[mask]
    # spot_weights_filtered = spot_weights[mask]

    # # Replace original lists with filtered versions
    # nv_coordinates = nv_coordinates_filtered
    # spot_weights = spot_weights_filtered

    # print(f"After filtering: {len(spot_weights)} NVs")
    # filtered_reordered_coords, filtered_reordered_spot_weights = nv_coordinates, spot_weights 
    # # Filter and reorder NV coordinates based on reference NV
    sigma = 2.0
    # reference_nv = [231.42, 235.968]
    # reference_nv_old = [231.42, 235.968]
    # reference_nv = remap_single_coord(reference_nv_old, old_roi, new_roi, rescale=False)

    # print("New reference_nv:", reference_nv)
    # reference_nv = [214.573, 203.991]
    # filtered_reordered_coords, filtered_reordered_spot_weights, include_indices = (
    #     filter_and_reorder_nv_coords(
    #         nv_coordinates, spot_weights, reference_nv, min_distance=4.0
    #     )
    # )
    
    # # Initialize lists to store the results
    # fitted_amplitudes = []
    # fitted_coords = []
    # for coord in filtered_reordered_coords:
    #     fitted_x, fitted_y, amplitude = fit_gaussian(img_array, coord, window_size=1)
    #     fitted_coords.append([fitted_x, fitted_y])
    #     fitted_amplitudes.append(amplitude)
        
    # filtered_reordered_coords = fitted_coords

    # -----------------------------
    # Example: your bar as polygon
    # -----------------------------
    bar_region = {
        "type": "polygon",
        "vertices": [
            [123.087, 302.236],
            [214.41, 224.865],
            [221.386, 232.158],
            [130.063, 311.115],
        ],
    }
    # Add more bars / regions here
    regions = [
        bar_region,
        # # Example rectangle
        # {
        #     "type": "rect",
        #     "xmin": 10,
        #     "xmax": 30,
        #     "ymin": 100,
        #     "ymax": 130,
        # },

        # # Example circle
        # {
        #     "type": "circle",
        #     "center": [200, 200],
        #     "radius": 12,
        # },
    ]

    # kept_coords, removed_coords, mask = remove_coords_in_regions(filtered_reordered_coords, regions)

    # print("Kept coords:")
    # print(kept_coords)

    # print("\nRemoved coords:")
    # print(removed_coords)
    # filtered_reordered_coords = kept_coords
    # print(f"After filtering:{len(filtered_reordered_coords)}")

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
    print(len(filtered_reordered_coords))
    # print(filtered_reordered_coords)
    # print("Filter:", filtered_reordered_counts)
    # print("Filtered and Reordered NV Coordinates:", filtered_reordered_coords)
    # print("Filtered and Reordered NV Coordinates:", integrated_intensities)

    # -----------------------------
    # Your usage pattern
    # -----------------------------
    filtered_reordered_coords_0 = [
        coord for i, coord in enumerate(filtered_reordered_coords) if i not in indices_to_remove
    ]
    filtered_reordered_spot_weights_0 = [
        w for i, w in enumerate(filtered_reordered_spot_weights) if i not in indices_to_remove
    ]
    filtered_reordered_coords = filtered_reordered_coords_0
    filtered_reordered_spot_weights = filtered_reordered_spot_weights_0
    
    # refine coords after fitting; keep original weights
    filtered_reordered_coords, filtered_reordered_spot_weights, fitted_amplitudes, fitted_gauss_w = (
        refine_coords_after_fitting(
            img_array,
            filtered_reordered_coords,
            filtered_reordered_spot_weights,
            window_size=1,
            min_amplitude=0.0,
            replace_weights_with="none",   # or "amplitude" or "gaussian_weight"
            normalize=False,              # keep amplitude in image units
        )
    )
    
    # print("Kept after fitting:", len(filtered_reordered_coords))
    
    # # If you want disk-sum weights (another notion of weight)
    # disk_weights = integrate_intensity(img_array, filtered_reordered_coords, sigma=5)

    ### 205NVs
    # spot_weights = [0.6967448366212741, 0.6431475435032771, 0.9367690489172257, 0.7528585647384544, 0.6431475435032771, 0.8115359829232205, 1.2200385726457377, 0.8728239715797121, 0.9367690489172257, 0.8115359829232205, 0.6967448366212741, 0.8115359829232205, 0.6967448366212741, 1.549456260472012, 0.8728239715797121, 0.6431475435032771, 0.6431475435032771, 1.0728148013637353, 0.7528585647384544, 1.6393540341154373, 1.2979549770364116, 0.8728239715797121, 1.2979549770364116, 1.378800588178922, 1.1450068044637591, 1.6393540341154373, 0.6431475435032771, 0.8115359829232205, 1.0728148013637353, 0.6967448366212741, 0.9367690489172257, 0.6431475435032771, 0.9367690489172257, 1.732356454505837, 0.6431475435032771, 0.7528585647384544, 0.7528585647384544, 1.549456260472012, 0.8115359829232205, 0.8728239715797121, 1.0034173827041273, 0.8115359829232205, 0.8728239715797121, 1.6393540341154373, 0.9367690489172257, 0.6967448366212741, 1.4626196845182309, 0.7528585647384544, 0.8115359829232205, 0.9367690489172257, 1.549456260472012, 1.1450068044637591, 0.8115359829232205, 1.0034173827041273, 0.8115359829232205, 1.0728148013637353, 1.2979549770364116, 1.0728148013637353, 0.9367690489172257, 0.7528585647384544, 1.1450068044637591, 0.9367690489172257, 1.732356454505837, 0.9367690489172257, 1.0034173827041273, 0.7528585647384544, 0.6431475435032771, 1.378800588178922, 0.6431475435032771, 0.7528585647384544, 1.4626196845182309, 1.6393540341154373, 1.0034173827041273, 0.6431475435032771, 0.7528585647384544, 1.1450068044637591, 0.8115359829232205, 1.0728148013637353, 1.378800588178922, 0.6967448366212741, 1.2979549770364116, 1.378800588178922, 0.8728239715797121, 0.8115359829232205, 0.8115359829232205, 1.0034173827041273, 0.7528585647384544, 0.7528585647384544, 1.4626196845182309, 1.0728148013637353, 0.9367690489172257, 0.9367690489172257, 0.6967448366212741, 0.8728239715797121, 1.0034173827041273, 0.8728239715797121, 1.378800588178922, 1.0034173827041273, 0.6967448366212741, 1.1450068044637591, 1.0034173827041273, 0.8728239715797121, 1.1450068044637591, 1.4626196845182309, 1.1450068044637591, 0.7528585647384544, 0.8728239715797121, 0.8728239715797121, 1.0034173827041273, 1.0728148013637353, 0.6967448366212741, 1.549456260472012, 0.6967448366212741, 1.1450068044637591, 0.9367690489172257, 1.0034173827041273, 0.8115359829232205, 0.9367690489172257, 0.8728239715797121, 0.8728239715797121, 1.2200385726457377, 1.0034173827041273, 0.6967448366212741, 0.8728239715797121, 0.7528585647384544, 0.8115359829232205, 1.2200385726457377, 1.0034173827041273, 1.0728148013637353, 0.8728239715797121, 1.378800588178922, 1.378800588178922, 1.0728148013637353, 1.2979549770364116, 0.6431475435032771, 0.6431475435032771, 0.9367690489172257, 1.4626196845182309, 1.0034173827041273, 0.8115359829232205, 1.0034173827041273, 1.732356454505837, 0.8728239715797121, 1.378800588178922, 1.2979549770364116, 0.6967448366212741, 0.6967448366212741, 1.6393540341154373, 0.8728239715797121, 1.6393540341154373, 0.6967448366212741, 0.8728239715797121, 0.9367690489172257, 0.7528585647384544, 0.9367690489172257, 0.8115359829232205, 1.2200385726457377, 1.0034173827041273, 0.8115359829232205, 0.9367690489172257, 0.6967448366212741, 0.6967448366212741, 1.2200385726457377, 0.7528585647384544, 1.2200385726457377, 0.7528585647384544, 0.6967448366212741, 0.8115359829232205, 0.6967448366212741, 1.1450068044637591, 0.9367690489172257, 0.8115359829232205, 0.9367690489172257, 0.7528585647384544, 0.8728239715797121, 0.8728239715797121, 1.378800588178922, 1.2979549770364116, 0.7528585647384544, 1.0034173827041273, 0.8115359829232205, 0.8115359829232205, 1.6393540341154373, 0.9367690489172257, 0.6967448366212741, 1.2200385726457377, 0.8115359829232205, 1.0034173827041273, 1.732356454505837, 0.8115359829232205, 1.2200385726457377, 0.9367690489172257, 1.732356454505837, 0.6967448366212741, 1.1450068044637591, 0.8728239715797121, 1.6393540341154373, 1.549456260472012, 0.7528585647384544, 1.0728148013637353, 0.9367690489172257, 1.2979549770364116, 0.9367690489172257, 0.8728239715797121, 1.0034173827041273]
    # prep_fidelity_list = [0.7244842538372116, 0.668176307018715, 0.8136753715942407, 0.4598940360267726, 0.6278809166823878, 0.7283280165573244, 0.7049272108862757, 0.7403875062031255, 0.6338172722809289, 0.9502363921791863, 0.8168664062046057, 0.7830749116055468, 0.6774543752234123, 0.6800967434507296, 0.8067577295137124, 0.6773473003062411, 0.6265924456880423, 0.6647949512524377, 0.6642498047435033, 0.5799437040299427, 0.6049134706925767, 0.5180695714070349, 0.6525592417116729, 0.6055283409434562, 0.5602976256147588, 0.9879253109439408, 0.6341976508818022, 0.7051656775466422, 0.7761005779470389, 0.8163302651218947, 0.6771730493070144, 0.7991676556821871, 0.7228486331345386, 0.7684607067208107, 0.8383452279962957, 0.8163571580575045, 0.823859543644205, 0.7767758769863704, 0.7, 0.7389885073586306, 0.7058689166039616, 0.5686118129229055, 0.5458077009439342, 0.8230725868098714, 0.5933373815394066, 0.7531157948973536, 0.7598755101490915, 0.598285303639225, 0.4770696743475886, 0.629945379818548, 0.6044266288634046, 0.6633460892304572, 0.7372827918904223, 0.974724777647361, 0.6379419151748946, 0.6137508890696979, 0.6297321753640704, 0.6907613258124311, 0.8353142957718657, 0.8784531948511067, 0.7946035685080868, 0.7923761207306218, 0.4635385937318277, 0.6959487283109426, 0.7438724954533029, 0.7236950237028035, 0.728142756121292, 0.669780892707452, 0.7983892632746994, 0.6235294911858871, 0.6910704061948154, 0.6410142976255013, 0.5874430732447566, 0.7650970765344955, 0.7071899998815345, 0.6658736120358244, 0.765906370628554, 0.6857840035055728, 0.7213565156605926, 0.5851728269097456, 0.6625572262454951, 0.744149054731007, 0.7133379203181531, 0.7585795335152616, 0.7146540770404471, 0.7839221610123367, 0.7403061148754196, 0.6327732360584898, 0.7467307625078503, 0.7294838239883794, 0.7048602854503451, 0.6411082918525868, 0.6870904449282573, 0.662432776080613, 0.7236240682544943, 0.7659420680091513, 0.9998903580459301, 0.8770144718988805, 0.6555604368504004, 0.7169851724694452, 0.8150953856905679, 0.6718985828039988, 0.7318542073219265, 0.6173748477446523, 0.6461482294787366, 0.6252554365440328, 0.8340775444351025, 0.6690578039962596, 0.6593457009172152, 0.6411929945874324, 0.7284568997042813, 0.5691459411874935, 0.7047241144802922, 0.7651315012755694, 0.6353771868614405, 0.7491776931668837, 0.788121980324948, 0.6774070250779753, 0.7498572277094442, 0.5702740100951389, 0.6962273397257662, 0.8622069283248135, 0.6626278628761579, 0.8214128353890614, 0.6077314257602728, 0.6765331967431049, 0.7, 0.6515825101537264, 0.7388942345156293, 0.8239191471465833, 0.8450595518363193, 0.7182797909253857, 0.7879512347089562, 0.7486334895580925, 0.7, 0.5908236034851759, 0.9541426708456908, 0.570536949450317, 0.6055288704810973, 0.5704024565196837, 0.7569701927642681, 0.7357089253744847, 0.6933078232547394, 0.7206670727864188, 0.72191872802019, 0.7196732687461994, 0.7319169685703122, 0.7127548138444302, 0.6632502008695904, 0.7628056859853998, 0.6431445347152842, 0.7238549572314006, 0.6770866887795782, 0.6318504916172036, 0.72757588750728, 0.7535715762856774, 0.8512010484668235, 0.7628560219962184, 0.6441897884847472, 0.796482723456197, 0.6160898338615877, 0.7805229565064302, 0.7865165677176365, 0.7483935731642073, 0.6959079209045878, 0.7023607273964816, 0.7176882269961415, 0.700732760615871, 0.5120756357376884, 0.6577602897506898, 0.9861105645430472, 0.8392397155600052, 0.7286932810269, 0.7930021509408708, 0.7265693059696104, 0.8491598443121586, 0.45207622768221, 0.7996723599402089, 0.740735541143823, 0.7119308864459062, 0.5858535880467133, 0.867463667468022, 0.6008931960107837, 0.7114699125114212, 0.7062433363350984, 0.7328830021475375, 0.6445341830551592, 0.5414302797128977, 0.5893052651840647, 0.9890129139063075, 0.6993337842349376, 0.566431772083579, 0.6658305135626419, 0.7059676770010812, 0.5354595519286225, 0.6876690607003557]
    # readout_fidelity_list= [0.9434835396565022, 0.9372640061665731, 0.8286172409967514, 0.9649003903665963, 0.914888165626927, 0.919757288549233, 0.9381519118257791, 0.7679108843612239, 0.9406166520617758, 0.6700316088360911, 0.8716040090663616, 0.8172508710196675, 0.8890872301018096, 0.9052674285124747, 0.8789668632945321, 0.9539324508487146, 0.9551023446290093, 0.8993496868655347, 0.8966051448476755, 0.9167567024482017, 0.9486723898737144, 0.9688547857363747, 0.9410181010680867, 0.973344660814147, 0.9664327397971195, 0.8217294133349002, 0.9611015077998372, 0.837395222928186, 0.8296867289607543, 0.8340894802492043, 0.9410450153734707, 0.8499240375575701, 0.8878855574290908, 0.9194534204493765, 0.8707464916064099, 0.7222811885077294, 0.8245231079462949, 0.8121701107298482, 0.9572236992725673, 0.9151045681919635, 0.8954850428174185, 0.9554363446161365, 0.9441144190906161, 0.7651332545873465, 0.8788387772128219, 0.9151918235302934, 0.911974747149918, 0.882369123769214, 0.9750665455931781, 0.9122442217009129, 0.952787573073191, 0.9375427862710524, 0.9303795886295904, 0.785908108784751, 0.880040154136482, 0.9202598522438208, 0.8109131336975899, 0.8814073710186493, 0.8219692197749937, 0.857393071375221, 0.8868171216111047, 0.8160817798233253, 0.9673090621085769, 0.8901846885203288, 0.8429490076388286, 0.8962619734140469, 0.8940425014864638, 0.9564382879047615, 0.8531444707905631, 0.964931819275505, 0.8897594273274101, 0.7593910104638081, 0.9719813311811567, 0.8204044261989576, 0.9197979883053877, 0.9410721292253019, 0.9094732300430024, 0.901108091348898, 0.9290364859896829, 0.9641950384613192, 0.882827950964012, 0.8690446682490136, 0.9456786955193122, 0.8947780836565871, 0.9133456009058771, 0.9059043482691944, 0.9297658934216458, 0.9698533291704405, 0.9233397726267658, 0.8317339948512845, 0.9542036059989323, 0.9340912153039349, 0.9263156842137164, 0.9504398444622723, 0.9046777619537931, 0.866804315951645, 0.5005399005778868, 0.8282670182286944, 0.9542534080510823, 0.9362941623927001, 0.7837642782215319, 0.9112147721834811, 0.9325476003020569, 0.8822438619977533, 0.8957936129410193, 0.9273109841840378, 0.8070424771006035, 0.9426546412801311, 0.9288778073136732, 0.9423647905330262, 0.9298367823613384, 0.9143565783804914, 0.9427104645901587, 0.9213370489760793, 0.9723406931351746, 0.8769906349230426, 0.8519348073694535, 0.9098742484610653, 0.925148999652528, 0.9702652177754999, 0.9361840576793263, 0.50001363003431, 0.9292132729920701, 0.7699279724339522, 0.9612386967181832, 0.837651610954483, 0.9270684986328944, 0.9369851031093845, 0.8730961403880937, 0.49999620087924285, 0.4999981634420655, 0.93050117968955, 0.8730494536931042, 0.9210074016770486, 0.9355430792181355, 0.9696306622347739, 0.4999850160268842, 0.8776632206178094, 0.9680476203464741, 0.9681994774279328, 0.786627861763822, 0.8419329291383022, 0.9493400385750737, 0.9126415182170886, 0.92998488528008, 0.9273566630024407, 0.927265876966419, 0.9367351203423103, 0.9133427376865279, 0.8566217933868573, 0.9173546667071225, 0.922350477795163, 0.7544742539383164, 0.9348396108455233, 0.9101576061242532, 0.9037855573382562, 0.8263724684983165, 0.9104929869518873, 0.959317731686854, 0.780890788378332, 0.9771266942903336, 0.8222955129730233, 0.8788435454878102, 0.7609317278169488, 0.9481382235152354, 0.9151573725345858, 0.7939541001612469, 0.866245471656627, 0.9179524604882854, 0.9650381787360292, 0.7815762358125025, 0.7984481962763585, 0.9093660382752915, 0.4999931923083893, 0.9081771121185458, 0.6943313742805974, 0.9752985611060302, 0.8939368642105059, 0.9356361349579876, 0.913079830766012, 0.9220920802715542, 0.8010208227784532, 0.9730877806238505, 0.9031195385588932, 0.9432006717419518, 0.9270188444917952, 0.8729294469045447, 0.9851079647680514, 0.9873099749652372, 0.7701768338705828, 0.9453956397635191, 0.8754771043693431, 0.8647680871373026, 0.9542611904457783, 0.9563101487606434, 0.9718617372672891]
    # include_indices = [
    #     i
    #     for i, (v1, v2) in enumerate(zip(readout_fidelity_list, prep_fidelity_list))
    #     if (v1 is not None and v2 is not None)
    #     and all(isinstance(x, (int, float)) for x in (v1, v2))
    #     and not (math.isnan(v1) or math.isnan(v2))
    #     and v1 >= 0.6 and v2 >= 0.4
    # ]
        ## qnami
    # spot_weights =[1.2266119417359693, 0.7544120991715823, 0.5597670795701869, 0.6217217256579556, 1.8188148536479478, 0.8252566404860973, 0.8991725867033584, 0.6865851910294521, 0.5597670795701869, 0.7544120991715823, 1.6073952812862813, 0.8252566404860973, 0.5597670795701869, 0.6217217256579556, 0.6865851910294521, 1.6073952812862813, 1.4101295022755422, 0.7544120991715823, 1.6073952812862813, 1.0564317681700715, 1.8188148536479478, 0.7544120991715823, 0.7544120991715823, 1.0564317681700715, 1.3166777450187614, 0.8252566404860973, 0.8991725867033584, 1.0564317681700715, 0.6217217256579556, 1.7113106924227182, 1.3166777450187614, 1.6073952812862813, 0.7544120991715823, 0.5597670795701869, 0.7544120991715823, 0.8991725867033584, 0.7544120991715823, 0.6865851910294521, 0.6217217256579556, 1.6073952812862813, 1.4101295022755422, 0.7544120991715823, 1.3166777450187614, 1.7113106924227182, 0.6865851910294521, 1.4101295022755422, 0.7544120991715823, 1.6073952812862813, 1.5070183962289345, 0.8252566404860973, 0.8252566404860973, 1.7113106924227182, 0.8991725867033584, 0.8252566404860973, 0.7544120991715823, 1.1398805720598935, 0.7544120991715823, 1.3166777450187614, 0.6865851910294521, 0.8252566404860973, 0.5597670795701869, 1.6073952812862813, 0.6865851910294521, 1.1398805720598935, 0.7544120991715823, 0.7544120991715823, 0.6865851910294521, 0.8252566404860973, 1.0564317681700715, 1.1398805720598935, 0.6217217256579556, 1.6073952812862813, 0.5597670795701869, 0.6865851910294521, 0.6217217256579556, 0.7544120991715823, 1.4101295022755422, 0.9762133044690186, 1.6073952812862813, 1.3166777450187614, 1.2266119417359693, 0.7544120991715823, 1.7113106924227182, 1.4101295022755422, 1.2266119417359693, 1.6073952812862813, 0.8991725867033584, 1.0564317681700715, 0.8252566404860973, 1.6073952812862813, 0.7544120991715823, 0.6865851910294521, 0.8991725867033584, 0.6865851910294521, 0.7544120991715823, 1.6073952812862813, 0.6865851910294521, 0.6865851910294521, 0.5597670795701869, 0.6865851910294521, 0.6865851910294521, 0.5597670795701869, 1.7113106924227182, 1.8188148536479478, 0.8991725867033584, 1.1398805720598935, 0.8991725867033584, 0.8252566404860973, 0.8991725867033584, 0.6217217256579556, 1.1398805720598935, 0.9762133044690186, 1.6073952812862813, 0.8252566404860973, 0.6865851910294521, 0.8252566404860973, 0.7544120991715823, 0.7544120991715823, 0.7544120991715823, 0.8252566404860973, 1.6073952812862813, 1.7113106924227182, 0.9762133044690186, 0.6865851910294521, 0.8252566404860973, 0.8252566404860973, 0.7544120991715823, 0.5597670795701869, 0.5597670795701869, 1.0564317681700715, 1.3166777450187614, 0.7544120991715823, 1.6073952812862813, 1.5070183962289345, 1.1398805720598935, 1.3166777450187614, 0.6865851910294521, 1.5070183962289345, 1.1398805720598935, 1.8188148536479478, 0.9762133044690186, 1.3166777450187614, 0.7544120991715823, 0.5597670795701869, 1.2266119417359693, 1.6073952812862813, 0.8252566404860973, 0.6865851910294521, 0.7544120991715823, 0.5597670795701869, 0.7544120991715823, 0.9762133044690186, 1.0564317681700715, 0.5597670795701869, 1.7113106924227182, 0.6865851910294521, 0.5597670795701869, 0.7544120991715823, 0.8252566404860973, 1.1398805720598935, 0.8991725867033584, 0.7544120991715823, 0.5597670795701869, 0.8991725867033584, 0.6865851910294521, 0.7544120991715823, 0.6217217256579556, 0.8252566404860973, 0.8991725867033584, 1.0564317681700715, 1.1398805720598935, 0.6217217256579556, 0.9762133044690186, 1.0564317681700715, 0.8252566404860973, 1.8188148536479478, 1.8188148536479478, 0.7544120991715823, 0.8252566404860973, 0.6865851910294521, 1.2266119417359693, 1.3166777450187614, 1.5070183962289345, 0.6865851910294521, 0.7544120991715823, 0.8991725867033584, 0.5597670795701869, 0.8991725867033584, 0.6865851910294521, 1.4101295022755422, 1.3166777450187614, 1.3166777450187614, 1.0564317681700715, 0.8991725867033584, 1.4101295022755422, 0.8252566404860973, 1.4101295022755422, 0.6865851910294521, 1.6073952812862813, 1.2266119417359693, 0.5597670795701869, 1.7113106924227182, 0.5597670795701869, 0.7544120991715823, 0.6865851910294521, 0.6865851910294521, 0.5597670795701869, 0.5597670795701869, 0.8991725867033584, 1.5070183962289345, 0.8991725867033584, 1.2266119417359693, 0.9762133044690186, 0.6217217256579556, 1.5070183962289345]
    # include_indices = [i for i, val in enumerate(prep_fidelity_list) if val >= 0.4 or val is None]
    # include_indices =  [i for i, val in enumerate(snr_float) if val >= 0.02]
    
    data = dm.get_raw_data(
        # file_stem="2026_03_25-16_28_08-charge_state_analysis_hist_data_raw_data", load_npz=True
        # file_stem="2026_03_25-18_15_53-charge_state_analysis_hist_data_raw_data", load_npz=True
        file_stem="2026_03_26-20_09_33-charge_state_analysis_hist_data_raw_data", load_npz=True
    )
    readout_fidelity_list = data["readout_fidelity_list"]
    prep_fidelity_list = data["prep_fidelity_list"]
    # include_indices = [
    #     i for i, val in enumerate(readout_fidelity_list)
    #     if (val is None) or (isinstance(val, (int, float)) and not math.isnan(val) and val >= 0.7)
    # ]
    include_indices = list(range(len(filtered_reordered_coords)))
    # include_indices = [
    #     i
    #     for i, (v1, v2) in enumerate(zip(readout_fidelity_list, prep_fidelity_list))
    #     if (v1 is not None and v2 is not None)
    #     and all(isinstance(x, (int, float)) for x in (v1, v2))
    #     and not (math.isnan(v1) or math.isnan(v2))
    #     and v1 >= 0.7 and v2 >= 0.2
    # ]
    # print(np.sort(list(include_indices)))
    filtered_reordered_coords = [filtered_reordered_coords[i] for i in include_indices]
    updated_spot_weights = [spot_weights[i] for i in include_indices]

    # filtered_pol_durs = [pol_duration_list[i] for i in include_indices]
    # filtered_scc_durs = [scc_duration_list[i] for i in include_indices]

    aom_voltage = 0.3614
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
    # print(nv_powers_filtered)
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

    # updated_spot_weights = curve_extreme_weights_simple(
    # optimal_weigths, scaling_factor=1.0
    # )
    # updated_spot_weights = 1/fitted_amplitudes
    # updated_spot_weights = curve_inverse_counts(fitted_amplitudes, scaling_factor=1.0)
       
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
    # for idx, (coords, weight) in enumerate(
    #     zip(filtered_reordered_coords, filtered_reordered_spot_weights)
    # ):
    #     print(f"{idx + 1:<8} | {coords} | {weight:.3f}")

    # print(adjusted_aom_voltage)
    


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
    # filtered_reordered_spot_weights = filtered_reordered_spot_weights[:4094]
    # filtered_reordered_coords = filtered_reordered_coords[:4094]
    # # Save the filtered results
    # save_results(
    #     filtered_reordered_coords,
    #     filtered_reordered_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_1275nvs_reordered.npz",
    # )

    # # Plot the original image with circles around each NV
    fig, ax = plt.subplots()
    title = "LASER_589, 50ms Ref"
    kpl.imshow(ax, img_array, title=title, cbar_label="Estimated Photons")
    # Draw circles and index numbers
    for idx, coord in enumerate(filtered_reordered_coords):
        circ = plt.Circle(coord, sigma, color="lightblue", fill=False, linewidth=0.5)
        ax.add_patch(circ)
        # Place text just above the circle
        # ax.text(
        #     coord[0],
        #     coord[1] - sigma - 1,
        #     str(idx),
        #     color="white",
        #     fontsize=8,
        #     ha="center",
        # )
    # selector = ManualPolygonSelector(ax)
    plt.show(block=True)
