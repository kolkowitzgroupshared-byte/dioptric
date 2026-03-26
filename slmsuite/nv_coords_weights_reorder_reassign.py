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
        file_stem="2026_03_25-17_02_10-qnami-nv0_2026_02_20", load_npz=True
    )
    img_array = np.array(data["ref_img_array"])
    # img_array = data["img_array"]
    nv_coordinates, spot_weights = load_nv_coords(
        # file_path="slmsuite/nv_blob_detection/nv_blob_6837nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_6904nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_3986nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_3554nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_3366nvs_reordered.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1460nvs.npz"
        # file_path="slmsuite/nv_blob_detection/nv_blob_1487nvs_reordered.npz"
        file_path="slmsuite/nv_blob_detection/nv_blob_1348nvs_reordered.npz"
    )
    
    fitted_data = dm.get_raw_data(file_stem="2026_03_23-16_39_03-optimal_values_2026_03_22-21_49_52-qnami-nv0_2026_02_20")
    # optimal_weigths = fitted_data["optimal_weigths"]
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
    # filtered_reordered_coords, filtered_reordered_spot_weights, fitted_amplitudes, fitted_gauss_w = (
    #     refine_coords_after_fitting(
    #         img_array,
    #         filtered_reordered_coords,
    #         filtered_reordered_spot_weights,
    #         window_size=6,
    #         min_amplitude=0.0,
    #         replace_weights_with="none",   # or "amplitude" or "gaussian_weight"
    #         normalize=False,              # keep amplitude in image units
    #     )
    # )
    
    # print("Kept after fitting:", len(filtered_reordered_coords))
    
    # # If you want disk-sum weights (another notion of weight)
    # disk_weights = integrate_intensity(img_array, filtered_reordered_coords, sigma=5)

    # fmt: off
    ## 210 NVs
    snr =['0.037', '0.088', '0.064', '0.039', '0.081', '0.093', '0.027', '0.070', '0.082', '0.087', '0.108', '0.064', '0.108', '0.066', '0.047', '0.098', '0.105', '0.088', '0.093', '0.092', '0.083', '0.084', '0.095', '0.125', '0.081', '0.137', '-0.005', '0.097', '0.060', '0.077', '0.109', '0.082', '0.096', '0.074', '0.061', '0.108', '0.116', '0.047', '0.087', '0.034', '0.093', '0.139', '0.005', '0.059', '0.083', '0.019', '0.054', '0.089', '0.092', '0.048', '0.039', '0.012', '0.003', '0.093', '0.107', '0.063', '0.058', '0.064', '0.092', '0.121', '0.042', '0.051', '0.032', '0.112', '0.069', '0.023', '0.083', '0.093', '0.047', '0.079', '0.092', '0.091', '0.011', '0.094', '0.103', '0.013', '0.120', '0.034', '0.103', '0.088', '0.082', '0.019', '0.075', '0.082', '0.108', '0.070', '0.087', '0.125', '0.074', '0.097', '0.123', '0.091', '0.137', '0.098', '0.119', '0.084', '0.100', '0.000', '0.129', '0.011', '0.102', '0.103', '0.004', '0.076', '0.024', '0.040', '0.104', '0.129', '0.094', '0.058', '-0.003', '0.078', '0.122', '0.081', '0.060', '-0.013', '0.124', '0.135', '0.071', '0.143', '0.063', '0.083', '0.007', '0.127', '0.105', '0.011', '0.078', '0.102', '0.095', '0.090', '0.064', '0.109', '0.014', '0.095', '0.129', '0.115', '0.132', '0.005', '0.061', '0.039', '0.066', '0.088', '0.096', '0.059', '0.108', '0.043', '0.132', '0.077', '-0.006', '0.054', '0.147', '0.089', '0.065', '0.124', '0.043', '0.057', '0.095', '0.086', '0.130', '0.064', '0.130', '0.093', '0.050', '0.091', '0.095', '0.105', '0.120', '-0.003', '0.094', '0.147', '0.068', '0.103', '0.117', '0.028', '0.123', '0.110', '0.100', '0.045', '0.129', '0.031', '0.061', '0.127', '0.066', '0.123', '0.123', '0.096', '0.040', '0.103', '0.040', '0.077', '0.041', '0.125', '0.047', '0.109', '0.056', '0.102', '0.039', '0.064', '0.017', '0.067', '0.075', '0.078', '0.004', '0.040', '0.091']
    snr_float = [float(el) for el in snr]
    ### 204NVs
    snr_float = [0.132, 0.115, 0.095, 0.104, 0.124, 0.145, 0.107, 0.091, 0.177, 0.119, 0.136, 0.128, 0.077, 0.136, 0.103, 0.118, 0.105, 0.179, 0.099, 0.109, 0.163, 0.107, 0.112, 0.119, 0.125, 0.169, 0.064, 0.106, 0.104, 0.133, 0.108, 0.111, 0.132, 0.008, 0.099, 0.121, 0.143, 0.101, 0.111, 0.101, 0.156, 0.127, 0.132, 0.117, 0.139, 0.11, 0.135, 0.114, 0.09, 0.088, 0.136, 0.157, 0.075, 0.096, 0.143, 0.132, 0.094, 0.092, 0.123, 0.075, 0.111, 0.135, 0.117, 0.057, 0.14, 0.025, 0.104, 0.119, 0.101, 0.142, 0.114, 0.146, 0.095, 0.034, 0.148, 0.201, 0.094, 0.109, 0.087, 0.11, 0.121, 0.115, 0.138, 0.099, 0.139, 0.141, 0.114, 0.083, 0.109, 0.119, 0.121, 0.171, 0.121, 0.113, 0.088, 0.144, 0.132, 0.101, 0.135, 0.124, 0.122, 0.137, 0.116, 0.127, 0.067, 0.118, 0.134, 0.077, 0.133, 0.11, 0.111, 0.138, 0.069, 0.146, 0.125, 0.125, 0.136, 0.134, 0.14, 0.151, 0.063, 0.125, 0.115, 0.124, 0.145, 0.125, 0.127, 0.139, 0.163, 0.118, 0.122, 0.111, 0.156, 0.099, 0.09, 0.14, 0.138, 0.107, 0.142, 0.135, 0.155, 0.12, 0.221, 0.139, 0.149, 0.142, 0.12, 0.128, 0.106, 0.127, 0.112, 0.143, 0.162, 0.111, 0.18, 0.124, 0.118, 0.111, 0.138, 0.074, 0.067, 0.12, 0.088, 0.145, 0.154, 0.094, 0.122, 0.113, 0.099, 0.128, 0.129, 0.133, 0.126, 0.127, 0.11, 0.088, 0.12, 0.143, 0.08, 0.046, 0.16, 0.14, 0.149, 0.067, 0.155, 0.1, 0.155, 0.142, 0.151, 0.083, 0.146, 0.15, 0.091, 0.128, 0.167, 0.111, 0.135, 0.15, 0.147, 0.091, 0.105, 0.127, 0.019, 0.136]
    pol_duration_list = [760, 760, 668, 668, 608, 608, 700, 700, 1008, 1008, 616, 616, 492, 492, 836, 836, 392, 392, 1028, 1028, 312, 312, 772, 772, 600, 600, 1036, 1036, 840, 840, 728, 728, 728, 1076, 1076, 412, 440, 440, 860, 860, 848, 704, 704, 508, 508, 652, 652, 836, 836, 796, 796, 728, 728, 712, 712, 696, 696, 436, 436, 612, 612, 612, 612, 748, 748, 956, 956, 676, 676, 668, 668, 404, 404, 776, 776, 468, 468, 688, 688, 548, 548, 1652, 1652, 652, 652, 1064, 488, 488, 616, 616, 744, 368, 368, 468, 468, 744, 744, 740, 740, 1252, 1252, 668, 668, 536, 536, 820, 400, 400, 812, 812, 1616, 1616, 984, 984, 576, 576, 920, 920, 624, 624, 548, 548, 692, 692, 692, 692, 536, 536, 552, 552, 508, 508, 684, 684, 672, 672, 492, 492, 388, 496, 496, 1688, 1688, 652, 652, 1112, 1112, 756, 756, 480, 480, 556, 556, 1628, 1628, 1016, 1016, 664, 664, 716, 716, 780, 780, 624, 624, 1320, 1320, 644, 644, 620, 620, 688, 688, 880, 880, 576, 576, 1788, 1788, 744, 744, 1940, 1940, 676, 676, 696, 696, 1940, 1940, 716, 716, 668, 668, 680, 680, 1940, 1940, 692, 692, 712, 712, 944, 944, 776, 776, 796, 796, 732, 732, 684, 684, 668, 668, 752, 752, 856, 856, 596, 596, 776, 776, 1220, 1220]
    scc_duration_list = [88, 92, 84, 112, 88, 92, 156, 88, 80, 100, 84, 84, 92, 72, 104, 92, 92, 68, 100, 72, 100, 96, 92, 76, 88, 88, 88, 92, 84, 108, 116, 72, 96, 116, 112, 76, 100, 88, 84, 72, 88, 76, 68, 72, 88, 120, 80, 96, 88, 92, 116, 80, 92, 112, 104, 156, 116, 80, 80, 92, 92, 92, 84, 100, 96, 116, 80, 76, 72, 76, 84, 84, 88, 176, 88, 96, 92, 92, 76, 68, 84, 128, 80, 84, 116, 88, 84, 88, 96, 76, 96, 80, 140, 92, 96, 100, 76, 84, 72, 80, 120, 80, 88, 124, 100, 76, 68, 80, 92, 84, 96, 92, 104, 92, 136, 116, 136, 112, 76, 84, 92, 176, 108, 104, 120, 96, 92, 92, 88, 88, 84, 92, 124, 84, 112, 92, 68, 88, 88, 80, 136, 92, 92, 124, 88, 72, 104, 100, 120, 108, 108, 84, 88, 92, 112, 112, 88, 112, 96, 132, 96, 88, 112, 116, 108, 100, 84, 96, 116, 100, 88, 132, 88, 92, 148, 96, 100, 92, 140, 88, 84, 84, 92, 96, 144, 112, 100, 100, 112, 104, 96, 84, 104, 104, 116, 76, 120, 148, 128, 92, 92, 100, 92, 108, 108, 92, 108, 112, 104, 112, 120, 144, 88, 100, 120, 100, 116, 144, 112, 104, 116, 132, 108]
    # spot_weights = [0.8500450744981615, 0.9130662981054044, 1.1182955151489753, 0.8500450744981615, 0.7896425509829849, 1.1182955151489753, 0.8500450744981615, 0.7896425509829849, 0.8500450744981615, 0.7318128814276639, 1.1182955151489753, 0.8500450744981615, 1.5169016419410182, 0.9787517225590299, 1.5169016419410182, 1.2690339517160951, 1.2690339517160951, 0.8500450744981615, 0.8500450744981615, 0.8500450744981615, 0.9130662981054044, 0.9130662981054044, 0.9787517225590299, 1.0471465141710325, 0.9130662981054044, 0.7896425509829849, 0.7318128814276639, 0.7896425509829849, 0.9130662981054044, 0.9130662981054044, 1.1182955151489753, 0.9130662981054044, 0.9130662981054044, 0.6236869225685145, 1.2690339517160951, 0.9130662981054044, 0.9130662981054044, 1.0471465141710325, 0.8500450744981615, 0.8500450744981615, 0.9130662981054044, 1.1182955151489753, 0.9787517225590299, 0.9787517225590299, 0.7896425509829849, 0.9787517225590299, 1.3487115367565212, 0.9787517225590299, 0.6236869225685145, 1.1182955151489753, 1.1922432533819738, 0.8500450744981615, 0.9787517225590299, 0.6765098627379946, 0.6236869225685145, 1.3487115367565212, 0.9787517225590299, 0.9130662981054044, 1.1922432533819738, 0.9130662981054044, 0.8500450744981615, 0.9130662981054044, 0.7318128814276639, 1.1182955151489753, 0.7896425509829849, 1.3487115367565212, 0.9130662981054044, 0.8500450744981615, 1.0471465141710325, 0.7318128814276639, 1.2690339517160951, 0.6765098627379946, 0.7896425509829849, 0.8500450744981615, 0.6765098627379946, 0.9787517225590299, 0.7896425509829849, 1.4313196472303777, 0.9787517225590299, 1.0471465141710325, 0.9130662981054044, 0.9787517225590299, 0.9130662981054044, 0.7318128814276639, 1.1182955151489753, 0.7318128814276639, 0.7318128814276639, 1.5169016419410182, 1.2690339517160951, 0.9130662981054044, 0.8500450744981615, 1.3487115367565212, 0.8500450744981615, 1.0471465141710325, 1.2690339517160951, 0.7896425509829849, 1.2690339517160951, 1.0471465141710325, 0.7896425509829849, 1.0471465141710325, 1.0471465141710325, 0.8500450744981615, 0.6765098627379946, 1.0471465141710325, 1.6055006073417841, 0.8500450744981615, 0.8500450744981615, 1.1182955151489753, 0.8500450744981615, 1.3487115367565212, 0.9787517225590299, 0.8500450744981615, 0.8500450744981615, 0.9787517225590299, 1.4313196472303777, 1.6055006073417841, 1.5169016419410182, 1.3487115367565212, 0.6765098627379946, 0.8500450744981615, 0.6236869225685145, 0.8500450744981615, 1.697159364754839, 0.6765098627379946, 1.4313196472303777, 0.9130662981054044, 1.5169016419410182, 1.6055006073417841, 0.7318128814276639, 0.9130662981054044, 0.7896425509829849, 0.8500450744981615, 1.4313196472303777, 1.1922432533819738, 1.0471465141710325, 0.7896425509829849, 0.9130662981054044, 0.7318128814276639, 0.8500450744981615, 0.9787517225590299, 1.4313196472303777, 0.9130662981054044, 1.1922432533819738, 1.1922432533819738, 0.7896425509829849, 1.0471465141710325, 0.7896425509829849, 0.8500450744981615, 1.6055006073417841, 0.8500450744981615, 0.8500450744981615, 1.2690339517160951, 0.8500450744981615, 0.9130662981054044, 0.9787517225590299, 0.7896425509829849, 0.8500450744981615, 1.1922432533819738, 0.7896425509829849, 0.7896425509829849, 0.6236869225685145, 0.9787517225590299, 0.6765098627379946, 1.3487115367565212, 1.1182955151489753, 0.7896425509829849, 0.9787517225590299, 0.9130662981054044, 1.1182955151489753, 1.6055006073417841, 0.7318128814276639, 1.1922432533819738, 0.9130662981054044, 0.7896425509829849, 0.7318128814276639, 0.9787517225590299, 1.1182955151489753, 0.9787517225590299, 0.9130662981054044, 0.8500450744981615, 0.7318128814276639, 1.3487115367565212, 0.6765098627379946, 1.3487115367565212, 1.1182955151489753, 0.7896425509829849, 0.9130662981054044, 0.9130662981054044, 1.4313196472303777, 0.7318128814276639, 0.8500450744981615, 0.9130662981054044, 1.4313196472303777, 0.9787517225590299, 0.8500450744981615, 1.1922432533819738, 1.0471465141710325, 0.8500450744981615, 1.697159364754839, 1.0471465141710325, 0.7896425509829849, 1.2690339517160951, 0.7896425509829849, 0.9130662981054044, 1.697159364754839, 0.8500450744981615, 0.9130662981054044, 1.2690339517160951, 0.9787517225590299, 0.9787517225590299, 1.0471465141710325, 1.697159364754839, 0.9130662981054044, 1.1922432533819738, 1.5169016419410182, 0.9787517225590299, 1.0471465141710325, 1.0471465141710325, 0.9130662981054044, 0.6765098627379946, 0.8500450744981615, 1.3487115367565212, 0.6765098627379946]
    
    ### 279NVs
    # spot_weights = [1.0471679790266164, 1.2401017668354632, 0.67104441621813, 1.0471679790266164, 0.7187157644338229, 0.4610266673429884, 0.7683534421103578, 1.1092793089720423, 1.2401017668354632, 1.1092793089720423, 1.2401017668354632, 0.67104441621813, 1.2401017668354632, 0.8199899935832455, 0.7683534421103578, 1.0471679790266164, 1.2401017668354632, 0.7683534421103578, 0.7683534421103578, 1.1092793089720423, 0.8199899935832455, 1.1735802123505668, 1.0471679790266164, 1.1735802123505668, 1.1735802123505668, 1.2401017668354632, 0.9293887614050709, 0.8736577350706582, 0.9293887614050709, 1.2401017668354632, 1.1735802123505668, 1.2401017668354632, 1.1735802123505668, 1.1092793089720423, 0.7187157644338229, 1.0471679790266164, 1.1735802123505668, 0.6253066179082334, 1.2401017668354632, 1.2401017668354632, 0.9872149524214544, 0.6253066179082334, 1.1735802123505668, 0.9293887614050709, 1.2401017668354632, 0.7187157644338229, 1.0471679790266164, 0.7683534421103578, 0.9872149524214544, 1.2401017668354632, 0.539499331237689, 1.0471679790266164, 1.1735802123505668, 1.1092793089720423, 0.8736577350706582, 0.5814693472267987, 1.1735802123505668, 0.8199899935832455, 1.1092793089720423, 1.0471679790266164, 1.1735802123505668, 0.9872149524214544, 1.1092793089720423, 0.9293887614050709, 1.2401017668354632, 0.539499331237689, 1.1735802123505668, 1.1092793089720423, 0.4610266673429884, 0.9293887614050709, 1.1735802123505668, 1.1735802123505668, 0.8736577350706582, 1.0471679790266164, 0.9293887614050709, 1.2401017668354632, 1.1735802123505668, 1.0471679790266164, 0.7187157644338229, 1.1735802123505668, 1.0471679790266164, 1.0471679790266164, 0.8736577350706582, 0.7683534421103578, 0.7187157644338229, 1.1735802123505668, 1.1735802123505668, 0.9293887614050709, 1.1092793089720423, 1.1092793089720423, 1.0471679790266164, 0.9293887614050709, 1.2401017668354632, 1.2401017668354632, 1.0471679790266164, 0.9293887614050709, 0.8736577350706582, 1.2401017668354632, 0.7187157644338229, 1.1092793089720423, 1.1735802123505668, 0.9872149524214544, 0.8199899935832455, 0.9872149524214544, 1.2401017668354632, 0.8199899935832455, 1.2401017668354632, 0.9293887614050709, 1.0471679790266164, 0.8736577350706582, 1.2401017668354632, 0.9872149524214544, 0.9293887614050709, 0.9293887614050709, 0.7187157644338229, 0.9872149524214544, 1.2401017668354632, 0.9872149524214544, 1.2401017668354632, 1.0471679790266164, 1.1092793089720423, 0.9293887614050709, 1.2401017668354632, 1.1735802123505668, 1.2401017668354632, 1.1092793089720423, 1.0471679790266164, 0.67104441621813, 1.2401017668354632, 1.2401017668354632, 0.7187157644338229, 0.9872149524214544, 0.8736577350706582, 1.1092793089720423, 1.2401017668354632, 1.1735802123505668, 0.7187157644338229, 0.7683534421103578, 0.9293887614050709, 0.7187157644338229, 0.8736577350706582, 0.7187157644338229, 1.0471679790266164, 0.9293887614050709, 1.2401017668354632, 1.1735802123505668, 0.5814693472267987, 0.5814693472267987, 1.1092793089720423, 1.2401017668354632, 1.2401017668354632, 1.2401017668354632, 1.1735802123505668, 0.9293887614050709, 0.9872149524214544, 0.8736577350706582, 0.8199899935832455, 0.8199899935832455, 0.9872149524214544, 0.8199899935832455, 0.9872149524214544, 0.9293887614050709, 0.9293887614050709, 1.1735802123505668, 1.2401017668354632, 0.67104441621813, 0.67104441621813, 0.9293887614050709, 1.1735802123505668, 1.2401017668354632, 0.7187157644338229, 1.1735802123505668, 1.1735802123505668, 1.1735802123505668, 1.1735802123505668, 0.9872149524214544, 1.1735802123505668, 1.1735802123505668, 0.9872149524214544, 0.8736577350706582, 1.1092793089720423, 1.2401017668354632, 1.1092793089720423, 0.9872149524214544, 0.9293887614050709, 1.1092793089720423, 0.9293887614050709, 0.8736577350706582, 0.539499331237689, 0.6253066179082334, 0.67104441621813, 0.8199899935832455, 0.67104441621813, 0.8199899935832455, 1.1092793089720423, 0.7187157644338229, 1.2401017668354632, 0.8736577350706582, 1.1735802123505668, 1.1735802123505668, 1.2401017668354632, 0.7683534421103578, 1.1092793089720423, 0.5814693472267987, 1.1092793089720423, 0.9872149524214544, 1.0471679790266164, 1.1092793089720423, 1.1092793089720423, 1.1735802123505668, 0.6253066179082334, 0.9872149524214544, 0.7683534421103578, 1.2401017668354632, 1.1735802123505668, 0.8199899935832455, 0.8736577350706582, 1.0471679790266164, 1.1092793089720423, 0.7187157644338229, 1.1092793089720423, 0.8199899935832455, 1.1735802123505668, 0.9293887614050709, 0.7683534421103578, 0.8736577350706582, 0.9872149524214544, 0.7683534421103578, 1.1092793089720423, 0.9293887614050709, 1.0471679790266164, 0.8736577350706582, 1.0471679790266164, 0.67104441621813, 1.1735802123505668, 1.2401017668354632, 0.9872149524214544, 1.2401017668354632, 0.9872149524214544, 1.1735802123505668, 1.2401017668354632, 0.9872149524214544, 1.0471679790266164, 1.1092793089720423, 0.6253066179082334, 1.0471679790266164, 1.2401017668354632, 1.2401017668354632, 1.1735802123505668, 0.9872149524214544, 0.8199899935832455, 0.8736577350706582, 1.1092793089720423, 0.7683534421103578, 0.8199899935832455, 0.9293887614050709, 1.0471679790266164, 0.9872149524214544, 1.2401017668354632, 1.1735802123505668, 0.9293887614050709, 1.2401017668354632, 1.0471679790266164, 1.2401017668354632, 1.2401017668354632, 1.1735802123505668, 1.2401017668354632, 1.1735802123505668, 0.8736577350706582, 0.9293887614050709, 0.7187157644338229, 1.1735802123505668, 1.2401017668354632, 0.7187157644338229, 0.4610266673429884, 1.0471679790266164, 0.8199899935832455, 1.2401017668354632, 1.0471679790266164]
    # readout_fidelity_list = [0.9560371346590277, 0.9590343124109253, 0.9270579204206462, 0.8026391733204805, 0.8201772135714809, 0.7154552401734582, 0.9286959899257543, 0.8866581345067346, 0.9164434120954779, 0.7708891867533793, 0.9505680348080126, 0.7397109573850376, 0.8331509841592224, 0.7726294574681829, 0.49999190255455517, 0.9548212563200495, 0.7463085060833549, 0.8376188687496097, 0.8715307163188477, 0.8648382593988495, 0.9328487845487317, 0.4999985953943063, 0.8930062877187435, 0.9344790979594411, 0.8443979984225818, 0.9503266496979066, 0.9087447975186171, 0.8630363648341985, 0.8458134359330802, 0.9204897567192765, 0.9001277215362836, 0.8392978342092258, 0.9155934432445453, 0.8655383212333185, 0.9485678398365229, 0.8909076263215807, 0.9595641050336188, 0.9458928959940602, 0.8793611226305773, 0.5000167447941374, 0.7935483029537376, 0.9157158571097177, 0.9572685413583306, 0.75772694027504, 0.9200280558099123, 0.8557292225415432, 0.7977211522770571, 0.8548181130215035, 0.6524822045872942, 0.8982712009791267, 0.9321222189789062, 0.6900544272333707, 0.8398564830264212, 0.7122010107690319, 0.8444668089443157, 0.9081503939226792, 0.5000061514692997, 0.8770652307702183, 0.9713889271495451, 0.6842342388785099, 0.5000001984274018, 0.911643578665479, 0.9551792790893877, 0.5000034545772097, 0.7158158342121026, 0.9242441247330154, 0.7529197019123905, 0.8385487791484647, 0.7867892039878636, 0.9600574085412408, 0.9208517912215926, 0.9340230985700675, 0.8839723405437165, 0.9352843761529517, 0.8064810258668801, 0.7671241419408259, 0.8260041600292753, 0.7458131390132152, 0.8812308109329423, 0.9490899264918761, 0.9598828781611579, 0.918692303726717, 0.7979791752087488, 0.8114849752883926, 0.8469283251215127, 0.899095993557127, 0.8698402717507696, 0.9330603816708483, 0.8989004967849903, 0.8438804714528421, 0.9049924388189444, 0.9019451850867559, 0.8754350483168711, 0.9028948584112145, 0.9359832848872469, 0.9240213971100844, 0.8345148708089216, 0.7418932018955281, 0.941600870238003, 0.888539231992564, 0.49999306964951207, 0.8494296103414636, 0.7553197432431766, 0.7745335984485101, 0.8153527138215232, 0.8867737095925368, 0.9156515544094813, 0.8941718656499505, 0.8291489920153893, 0.9242367119787565, 0.8431894646160761, 0.9366980725266394, 0.9007943263576954, 0.9611390765769183, 0.9113029372716803, 0.9544195570160481, 0.7566456877649606, 0.4999957573057602, 0.5000041335674852, 0.9510246982804449, 0.8567841250619983, 0.7567214982631627, 0.9029404089793396, 0.7183719086079972, 0.9101221052203539, 0.770550483369927, 0.9362232574632069, 0.7056194483089306, 0.9215135420009192, 0.8616372345901382, 0.7831425833267727, 0.8689094134425752, 0.8880095924401762, 0.9475609040171631, 0.6059903435043144, 0.9461055444756079, 0.9293426686611883, 0.8441895484901558, 0.49999487116817387, 0.9517631916450504, 0.9051309590557892, 0.9610909138147408, 0.9100864631874657, 0.9435755534126232, 0.9457351229199559, 0.88378369575564, 0.9008621692102285, 0.5000087893996614, 0.8073470627737342, 0.6439842303673275, 0.4999816983724884, 0.9164412661304187, 0.9489730370388356, 0.757546774739912, 0.9556441046293176, 0.9617076574885699, 0.6470930884936008, 0.4999997621557732, 0.9287261083176808, 0.7124095242637059, 0.940462369685516, 0.8568115477316256, 0.9481536821736564, 0.7702930034887646, 0.5000062440770694, 0.7922796951435388, 0.8563913843137159, 0.7695779286083673, 0.8653393434560284, 0.9381979229207849, 0.8492683630996916, 0.8980675080711039, 0.5000004230715585, 0.950547915034173, 0.8538189961322104, 0.9249913565647974, 0.8038250355734774, 0.9270891633154064, 0.9189744082292658, 0.5000027886066607, 0.7979417503217755, 0.7300292760885438, 0.8243369749036293, 0.4999752889978536, 0.8994012903854103, 0.7850704707358508, 0.8246040012223712, 0.9371775091417498, 0.7480466777433663, 0.5000001972025054, 0.8604640256946916, 0.9764738662230232, 0.5000075766277193, 0.8710369606448023, 0.5512180203580735, 0.7659215234271448, 0.9029087546077252, 0.8819708814280954, 0.7841517147029269, 0.9509135530785169, 0.9207169997813986, 0.500011364494659, 0.934389499402249, 0.885938259571499, 0.9089897385790964, 0.9239838397000972, 0.9276636889056429, 0.9450437547033972, 0.8874294064391496, 0.8487089770506017, 0.8006419810245211, 0.9300731990619112, 0.5000013192915844, 0.9615073189458507, 0.9493999652144827, 0.5000045476812609, 0.8816394956136144, 0.7265637276621277, 0.8152218148719261, 0.9445031319171358, 0.7909419487319282, 0.9609262327386219, 0.9082992033733763, 0.8513901192058292, 0.8034115911684374, 0.8511280513160067, 0.9152774657712666, 0.8843618327462395, 0.842061993826072, 0.7190220331598789, 0.9351577959037314, 0.9326684814960106, 0.8046107158874873, 0.5000020495574815, 0.8949779254544468, 0.7852165602845793, 0.9427642747320695, 0.8918050973545661, 0.918845811468092, 0.9388764230605506, 0.9015958010557712, 0.8634828142043021, 0.7635542569102579, 0.5000094515041054, 0.8370262575173362, 0.9274406268931614, 0.8107705723019737, 0.9560848654490228, 0.813375789179432, 0.9346968007375839, 0.9102585498612317, 0.8358124676502694, 0.8728475523952111, 0.600065719600541, 0.9430297307111253, 0.9218327681977216, 0.954918403409132, 0.8957688215659657, 0.937415567477873, 0.8830480570692011, 0.8637794292864014, 0.5000014197799821, 0.7633271732467084, 0.951635287184591, 0.8774388122530286, 0.5000043396532245, 0.9262622441110798, 0.9603839376665452, 0.8845939307506454, 0.8140681066213957]
    # prep_fidelity_list = [0.5734343068827965, 0.4250840538604531, 0.47425182343436123, 0.7209854236240996, 0.7221379059384638, 0.9739368426759898, 0.6553730332903054, 0.5249981393821257, 0.6399780179915544, 0.779263635317473, 0.5414544539027344, 0.623393783032414, 0.6968403532589094, 0.9033658517526756, 0.8209274311083566, 0.5434574766149827, 0.8957155059970158, 0.757097055326903, 0.6756888629808253, 0.6541571700421085, 0.5952783043910069, 0.8408059537474954, 0.5325392546144673, 0.5345669297915001, 0.7029825359816932, 0.4803516135705389, 0.6585444916018163, 0.7514996322198395, 0.6448776819914395, 0.6165751402520079, 0.5130424101316656, 0.7129114008903761, 0.6327472069018161, 0.7067993002946567, 0.42538027770724063, 0.7266824613933308, 0.46972204054458067, 0.466314993884955, 0.7713659827399867, 0.9294787684543314, 0.7924702510254142, 0.5320355557307557, 0.43647217566381635, 0.8249445414745893, 0.6145095847704026, 0.7316875671937484, 0.7738389366111771, 0.664900644697271, 0.9493870609197386, 0.49055164854760835, 0.575875376355034, 0.8396536847906263, 0.7787986735163339, 0.9304477146688235, 0.7353373446043743, 0.6158803580940581, 0.9000295736204657, 0.702024736606599, 0.3641714917131471, 0.8048008552180903, 0.9137228953285689, 0.49734419264644836, 0.466362415248995, 0.9376526768826758, 0.9429187247970947, 0.44467365802303205, 0.7548553024794571, 0.7208530403577685, 0.8692944752200552, 0.4314555836312082, 0.5085021685597159, 0.4107661539616575, 0.6369277932107862, 0.36111157129423976, 0.7579584093662826, 0.7944662044979767, 0.7928429573702661, 0.799330158716685, 0.7605098065926953, 0.515037501390751, 0.5026832978223024, 0.5841798862124192, 0.7715086973968738, 0.7084083404291456, 0.7132620635612589, 0.7135539760032863, 0.7040381387805266, 0.6264521773676004, 0.6429873776759991, 0.6812928615401144, 0.7078470292734758, 0.6234194327153086, 0.7233340573109661, 0.6707966050083125, 0.6692632186402596, 0.6629246407705478, 0.7282284770557186, 0.7309898672389354, 0.5426004748605908, 0.71662906468579, 0.8185001942458847, 0.8125437348709011, 0.783836197422594, 0.7652020865627552, 0.8110298592273227, 0.7396822271461363, 0.645975453657448, 0.700076924643817, 0.7762704707726829, 0.522535545249361, 0.7389431561396367, 0.5921793492478115, 0.6954877434143639, 0.39488431481910047, 0.5949004630197099, 0.44044082719463473, 0.86935048241424, 0.8363379870615248, 0.856548585511623, 0.5516807471016871, 0.7026912612532323, 0.8546749167593598, 0.7056761250428527, 0.7661726839088083, 0.6165557753172672, 0.8874663251596003, 0.5749766292115681, 0.7996619282616838, 0.5846043837686652, 0.6683693256332279, 0.9263666605741323, 0.7216103638516925, 0.703346872130242, 0.42105948009204797, 0.720816785189311, 0.5311404590118008, 0.5093920619397209, 0.8247557877858898, 0.8282977438970938, 0.5143779641746161, 0.4891569181310228, 0.5038890434798369, 0.6840533358488214, 0.580681143374483, 0.6235179292885424, 0.6772264791032698, 0.6586036001072122, 0.8577441469865019, 0.701580909318716, 0.779412745885435, 0.9065287026079948, 0.6378015750194614, 0.6071699099449617, 0.9693273352287969, 0.5173520861630314, 0.49625356517406805, 0.9883815324514232, 0.8416991158069458, 0.5681230574838689, 0.8446913577155306, 0.5033803900822797, 0.6129374686091398, 0.5483465016161178, 0.7812938512484969, 0.8255610339135885, 0.7515231532839675, 0.6270855417195613, 0.847849610771271, 0.6857264114491735, 0.5864967905136034, 0.6521398547022226, 0.4885245306928633, 0.9055424095564358, 0.5051802022948116, 0.6984787051704355, 0.4560495386174731, 0.7916349347890523, 0.6442991051479006, 0.6256856448689707, 0.9020057533814635, 0.931353468826359, 0.8569609691384608, 0.7632511587951334, 0.9205762783335218, 0.6746242131204109, 0.815212987941687, 0.7344343893315675, 0.6360270820038993, 0.6842124977913278, 0.819383471976933, 0.7431395630114703, 0.4149343590877218, 0.8153250131984381, 0.6293487242249021, 0.7396285396428348, 0.8124272234181097, 0.6861126504778086, 0.6950020533411416, 0.754911067090533, 0.4541863640802044, 0.6419084018240189, 0.9319012277802033, 0.643480066916858, 0.6798621475211787, 0.4140304759805856, 0.49162516140856594, 0.6843253024450691, 0.5747540113655475, 0.6899588813849125, 0.5881302211637565, 0.7970678728472413, 0.6306489749289038, 0.9074350893546227, 0.5191634816357333, 0.6110254166040027, 0.8150314812429561, 0.7437974825321167, 0.7939768296693418, 0.9225727533407286, 0.6007060035161358, 0.6984980578694029, 0.6080373682350815, 0.659824245213575, 0.6926300429926999, 0.8362809599680426, 0.6552740007256178, 0.3873877855572342, 0.6797925058774701, 0.7119128962269825, 0.9555962701620938, 0.3921605941671612, 0.5086340929962619, 0.7555693362216512, 0.8731969636389896, 0.7331309441045395, 0.7776928893839979, 0.592593796211202, 0.6590494151340374, 0.6340885526876325, 0.5222173878558829, 0.7199886053131068, 0.6494088681759485, 0.9634195907494302, 0.93869671994082, 0.8263550788170334, 0.6649426521938886, 0.7687457446412068, 0.626102194272444, 0.8147766751789709, 0.4297544327524896, 0.6893951026998304, 0.7897213899426426, 0.7111813544818715, 0.7232244491253204, 0.6067297657799922, 0.41467033590428726, 0.5579383919252692, 0.510764247884343, 0.6492791630837844, 0.6432899229268305, 0.7413158715535818, 0.8397166241961691, 0.9267519022494944, 0.6292200835764663, 0.7589600939956671, 0.9414563283161753, 0.5417518446176215, 0.5984258130469486, 0.7462528291581527, 0.6717788319156494]
    
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
    
    # spot_weights = [0.7577067581936708, 1.4012185764428304, 0.8677475266962307, 1.049852425278518, 0.8116153421622102, 0.9868296490145028, 0.8677475266962307, 1.2532795371945253, 0.7577067581936708, 1.2532795371945253, 0.6089505445612984, 1.4012185764428304, 0.9868296490145028, 1.4012185764428304, 0.8677475266962307, 0.9261401141751321, 0.705984706415963, 0.9261401141751321, 0.6564118439747914, 0.6564118439747914, 0.705984706415963, 0.8677475266962307, 1.4012185764428304, 0.9261401141751321, 1.049852425278518, 1.049852425278518, 1.2532795371945253, 0.7577067581936708, 1.3259934582137147, 0.9261401141751321, 0.8116153421622102, 0.7577067581936708, 1.049852425278518, 1.4012185764428304, 0.563562888904897, 1.049852425278518, 0.9868296490145028, 0.705984706415963, 0.9261401141751321, 0.8116153421622102, 0.9261401141751321, 1.3259934582137147, 1.2532795371945253, 1.4012185764428304, 1.3259934582137147, 0.9261401141751321, 1.4012185764428304, 0.705984706415963, 0.6564118439747914, 0.8677475266962307, 0.8677475266962307, 1.3259934582137147, 0.9261401141751321, 0.7577067581936708, 0.5202106546873658, 0.9261401141751321, 1.2532795371945253, 1.3259934582137147, 0.8677475266962307, 0.9868296490145028, 1.3259934582137147, 1.3259934582137147, 0.8677475266962307, 0.8677475266962307, 1.2532795371945253, 1.1830416695636192, 0.8116153421622102, 1.049852425278518, 0.7577067581936708, 1.1830416695636192, 1.4012185764428304, 1.1830416695636192, 0.9261401141751321, 0.7577067581936708, 0.8677475266962307, 0.9261401141751321, 0.7577067581936708, 1.1152444939039519, 0.8116153421622102, 1.1152444939039519, 1.3259934582137147, 0.8116153421622102, 0.9261401141751321, 1.4012185764428304, 0.8116153421622102, 0.563562888904897, 0.8677475266962307, 1.2532795371945253, 0.8116153421622102, 1.1830416695636192, 0.705984706415963, 1.3259934582137147, 0.8677475266962307, 0.8677475266962307, 1.1152444939039519, 0.8116153421622102, 0.9868296490145028, 1.3259934582137147, 1.1830416695636192, 0.705984706415963, 1.2532795371945253, 1.1830416695636192, 0.8116153421622102, 1.1152444939039519, 1.049852425278518, 1.1830416695636192, 1.4012185764428304, 1.049852425278518, 0.8116153421622102, 1.3259934582137147, 0.6089505445612984, 0.9261401141751321, 1.4012185764428304, 0.6089505445612984, 1.049852425278518, 0.7577067581936708, 1.1830416695636192, 1.1830416695636192, 1.049852425278518, 0.9261401141751321, 0.8116153421622102, 1.1152444939039519, 0.8677475266962307, 0.7577067581936708, 0.8677475266962307, 1.049852425278518, 0.7577067581936708, 0.9868296490145028, 1.049852425278518, 0.7577067581936708, 1.2532795371945253, 0.9868296490145028, 1.3259934582137147, 0.563562888904897, 0.9261401141751321, 0.8677475266962307, 1.3259934582137147, 0.9261401141751321, 1.4012185764428304, 1.4012185764428304, 0.9868296490145028, 0.9261401141751321, 0.8677475266962307, 0.8116153421622102, 0.9261401141751321, 1.4012185764428304, 0.7577067581936708, 1.3259934582137147, 0.8677475266962307, 0.563562888904897, 1.1152444939039519, 1.3259934582137147, 0.6564118439747914, 0.8116153421622102, 0.8116153421622102, 1.049852425278518, 0.9261401141751321, 0.7577067581936708, 0.8116153421622102, 1.1830416695636192, 0.8116153421622102, 1.049852425278518, 0.8677475266962307, 0.8677475266962307, 1.049852425278518, 0.9868296490145028, 0.5202106546873658, 0.9261401141751321, 1.049852425278518, 0.7577067581936708, 0.9261401141751321, 1.4012185764428304, 0.9261401141751321, 1.2532795371945253, 0.9868296490145028, 0.9261401141751321, 1.4012185764428304, 0.9261401141751321, 0.9868296490145028, 1.3259934582137147, 1.1830416695636192, 0.9868296490145028, 1.4012185764428304, 1.049852425278518, 0.9868296490145028, 0.9261401141751321, 1.3259934582137147, 1.2532795371945253, 0.9261401141751321, 0.8116153421622102, 1.4012185764428304, 0.7577067581936708, 0.6564118439747914, 1.1152444939039519, 0.8116153421622102, 1.1830416695636192, 0.8677475266962307, 1.1830416695636192, 0.8677475266962307, 0.8116153421622102, 1.2532795371945253, 0.9261401141751321, 1.049852425278518, 1.3259934582137147, 0.7577067581936708, 1.1830416695636192, 1.2532795371945253, 0.8116153421622102, 0.8116153421622102, 1.049852425278518]
    # spot_weights = [0.6736744402887587, 1.1237357868234286, 1.3595622054499192, 1.3595622054499192, 1.0510444306249145, 0.8500554104459587, 0.8500554104459587, 1.1993543719723665, 0.8500554104459587, 0.6736744402887587, 0.7885928778552385, 1.1993543719723665, 0.8500554104459587, 1.5320426973194516, 0.7298157204770973, 0.8500554104459587, 0.8500554104459587, 0.8500554104459587, 0.8500554104459587, 0.8500554104459587, 0.914252423824391, 1.1237357868234286, 1.0510444306249145, 0.914252423824391, 0.7298157204770973, 0.7298157204770973, 0.914252423824391, 0.9812326441793046, 0.914252423824391, 1.1993543719723665, 1.0510444306249145, 0.8500554104459587, 1.3595622054499192, 0.6736744402887587, 0.7298157204770973, 0.914252423824391, 0.7298157204770973, 1.62300095766451, 0.8500554104459587, 0.7298157204770973, 0.8500554104459587, 1.3595622054499192, 1.0510444306249145, 1.1237357868234286, 1.7171657149650743, 1.1993543719723665, 1.1237357868234286, 0.7885928778552385, 0.8500554104459587, 1.1237357868234286, 0.7298157204770973, 1.1993543719723665, 1.1237357868234286, 0.7885928778552385, 0.6736744402887587, 1.1237357868234286, 1.5320426973194516, 0.7885928778552385, 0.8500554104459587, 0.6736744402887587, 1.0510444306249145, 0.914252423824391, 1.1237357868234286, 0.7298157204770973, 1.1237357868234286, 0.7298157204770973, 1.0510444306249145, 0.9812326441793046, 0.9812326441793046, 1.5320426973194516, 0.6736744402887587, 0.7885928778552385, 0.8500554104459587, 0.8500554104459587, 0.7885928778552385, 0.7298157204770973, 0.8500554104459587, 1.4442451407831798, 0.7885928778552385, 0.9812326441793046, 1.3595622054499192, 0.7885928778552385, 0.9812326441793046, 1.8145824805811321, 1.0510444306249145, 0.7298157204770973, 0.8500554104459587, 1.3595622054499192, 0.914252423824391, 0.914252423824391, 0.914252423824391, 1.3595622054499192, 0.7885928778552385, 0.7298157204770973, 1.0510444306249145, 1.0510444306249145, 0.914252423824391, 0.6736744402887587, 1.5320426973194516, 0.914252423824391, 1.277947511205297, 1.5320426973194516, 1.0510444306249145, 0.9812326441793046, 0.7298157204770973, 1.5320426973194516, 1.8145824805811321, 1.0510444306249145, 0.7885928778552385, 0.7298157204770973, 0.7885928778552385, 0.8500554104459587, 0.8500554104459587, 0.7885928778552385, 0.9812326441793046, 0.9812326441793046, 0.8500554104459587, 1.1237357868234286, 0.914252423824391, 1.0510444306249145, 0.914252423824391, 1.1237357868234286, 0.7298157204770973, 0.914252423824391, 0.8500554104459587, 0.914252423824391, 0.9812326441793046, 1.0510444306249145, 0.7885928778552385, 0.914252423824391, 0.8500554104459587, 0.7885928778552385, 0.6736744402887587, 0.7298157204770973, 0.8500554104459587, 0.7298157204770973, 0.6736744402887587, 0.8500554104459587, 1.8145824805811321, 0.8500554104459587, 1.0510444306249145, 1.1993543719723665, 1.1237357868234286, 1.0510444306249145, 0.8500554104459587, 1.277947511205297, 0.9812326441793046, 0.6736744402887587, 0.7298157204770973, 0.6736744402887587, 1.4442451407831798, 0.6736744402887587, 0.6736744402887587, 1.0510444306249145, 0.7885928778552385, 1.3595622054499192, 0.9812326441793046, 1.1993543719723665, 0.9812326441793046, 1.5320426973194516, 0.914252423824391, 0.6736744402887587, 1.1237357868234286, 0.8500554104459587, 0.8500554104459587, 0.914252423824391, 0.6736744402887587, 1.0510444306249145, 0.914252423824391, 0.9812326441793046, 0.9812326441793046, 1.7171657149650743, 0.9812326441793046, 0.6736744402887587, 1.277947511205297, 0.9812326441793046, 1.0510444306249145, 0.914252423824391, 1.277947511205297, 1.7171657149650743, 0.914252423824391, 1.277947511205297, 1.4442451407831798, 1.3595622054499192, 1.277947511205297, 1.1993543719723665, 1.7171657149650743, 1.62300095766451, 0.914252423824391, 0.6736744402887587, 0.9812326441793046, 0.8500554104459587, 0.8500554104459587, 0.914252423824391, 1.0510444306249145, 1.277947511205297, 0.9812326441793046, 1.1237357868234286, 0.914252423824391, 0.7885928778552385, 1.0510444306249145, 0.914252423824391, 0.9812326441793046, 1.5320426973194516, 0.9812326441793046, 1.1237357868234286, 0.7298157204770973, 0.8500554104459587, 0.7298157204770973, 0.8500554104459587]
    # include_indices = [0, 1, 2, 3, 4, 5, 6, 10, 12, 14, 15, 17, 18, 19, 21, 23, 24, 26, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 92, 94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 121, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 169, 170, 171, 173, 174, 177, 179, 180, 181, 182, 184, 186, 188, 189, 190, 191, 192, 193, 194, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 264, 265, 266, 267, 270, 271, 274, 275, 276, 277, 280, 281, 282, 283, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307]
    ## qnami
    # spot_weights =[1.2266119417359693, 0.7544120991715823, 0.5597670795701869, 0.6217217256579556, 1.8188148536479478, 0.8252566404860973, 0.8991725867033584, 0.6865851910294521, 0.5597670795701869, 0.7544120991715823, 1.6073952812862813, 0.8252566404860973, 0.5597670795701869, 0.6217217256579556, 0.6865851910294521, 1.6073952812862813, 1.4101295022755422, 0.7544120991715823, 1.6073952812862813, 1.0564317681700715, 1.8188148536479478, 0.7544120991715823, 0.7544120991715823, 1.0564317681700715, 1.3166777450187614, 0.8252566404860973, 0.8991725867033584, 1.0564317681700715, 0.6217217256579556, 1.7113106924227182, 1.3166777450187614, 1.6073952812862813, 0.7544120991715823, 0.5597670795701869, 0.7544120991715823, 0.8991725867033584, 0.7544120991715823, 0.6865851910294521, 0.6217217256579556, 1.6073952812862813, 1.4101295022755422, 0.7544120991715823, 1.3166777450187614, 1.7113106924227182, 0.6865851910294521, 1.4101295022755422, 0.7544120991715823, 1.6073952812862813, 1.5070183962289345, 0.8252566404860973, 0.8252566404860973, 1.7113106924227182, 0.8991725867033584, 0.8252566404860973, 0.7544120991715823, 1.1398805720598935, 0.7544120991715823, 1.3166777450187614, 0.6865851910294521, 0.8252566404860973, 0.5597670795701869, 1.6073952812862813, 0.6865851910294521, 1.1398805720598935, 0.7544120991715823, 0.7544120991715823, 0.6865851910294521, 0.8252566404860973, 1.0564317681700715, 1.1398805720598935, 0.6217217256579556, 1.6073952812862813, 0.5597670795701869, 0.6865851910294521, 0.6217217256579556, 0.7544120991715823, 1.4101295022755422, 0.9762133044690186, 1.6073952812862813, 1.3166777450187614, 1.2266119417359693, 0.7544120991715823, 1.7113106924227182, 1.4101295022755422, 1.2266119417359693, 1.6073952812862813, 0.8991725867033584, 1.0564317681700715, 0.8252566404860973, 1.6073952812862813, 0.7544120991715823, 0.6865851910294521, 0.8991725867033584, 0.6865851910294521, 0.7544120991715823, 1.6073952812862813, 0.6865851910294521, 0.6865851910294521, 0.5597670795701869, 0.6865851910294521, 0.6865851910294521, 0.5597670795701869, 1.7113106924227182, 1.8188148536479478, 0.8991725867033584, 1.1398805720598935, 0.8991725867033584, 0.8252566404860973, 0.8991725867033584, 0.6217217256579556, 1.1398805720598935, 0.9762133044690186, 1.6073952812862813, 0.8252566404860973, 0.6865851910294521, 0.8252566404860973, 0.7544120991715823, 0.7544120991715823, 0.7544120991715823, 0.8252566404860973, 1.6073952812862813, 1.7113106924227182, 0.9762133044690186, 0.6865851910294521, 0.8252566404860973, 0.8252566404860973, 0.7544120991715823, 0.5597670795701869, 0.5597670795701869, 1.0564317681700715, 1.3166777450187614, 0.7544120991715823, 1.6073952812862813, 1.5070183962289345, 1.1398805720598935, 1.3166777450187614, 0.6865851910294521, 1.5070183962289345, 1.1398805720598935, 1.8188148536479478, 0.9762133044690186, 1.3166777450187614, 0.7544120991715823, 0.5597670795701869, 1.2266119417359693, 1.6073952812862813, 0.8252566404860973, 0.6865851910294521, 0.7544120991715823, 0.5597670795701869, 0.7544120991715823, 0.9762133044690186, 1.0564317681700715, 0.5597670795701869, 1.7113106924227182, 0.6865851910294521, 0.5597670795701869, 0.7544120991715823, 0.8252566404860973, 1.1398805720598935, 0.8991725867033584, 0.7544120991715823, 0.5597670795701869, 0.8991725867033584, 0.6865851910294521, 0.7544120991715823, 0.6217217256579556, 0.8252566404860973, 0.8991725867033584, 1.0564317681700715, 1.1398805720598935, 0.6217217256579556, 0.9762133044690186, 1.0564317681700715, 0.8252566404860973, 1.8188148536479478, 1.8188148536479478, 0.7544120991715823, 0.8252566404860973, 0.6865851910294521, 1.2266119417359693, 1.3166777450187614, 1.5070183962289345, 0.6865851910294521, 0.7544120991715823, 0.8991725867033584, 0.5597670795701869, 0.8991725867033584, 0.6865851910294521, 1.4101295022755422, 1.3166777450187614, 1.3166777450187614, 1.0564317681700715, 0.8991725867033584, 1.4101295022755422, 0.8252566404860973, 1.4101295022755422, 0.6865851910294521, 1.6073952812862813, 1.2266119417359693, 0.5597670795701869, 1.7113106924227182, 0.5597670795701869, 0.7544120991715823, 0.6865851910294521, 0.6865851910294521, 0.5597670795701869, 0.5597670795701869, 0.8991725867033584, 1.5070183962289345, 0.8991725867033584, 1.2266119417359693, 0.9762133044690186, 0.6217217256579556, 1.5070183962289345]
    # include_indices = [0, 1, 2, 4, 9, 10, 12, 14, 15, 17, 18, 20, 22, 23, 24, 27, 28, 29, 30, 33, 34, 37, 40, 41, 42, 45, 46, 49, 50, 51, 54, 55, 57, 58, 59, 60, 62, 65, 67, 68, 69, 72, 73, 75, 78, 79, 80, 81, 82, 85, 86, 87, 88, 90, 94, 95, 98, 99, 101, 102, 104, 106, 107, 111, 113, 114, 116, 117, 119, 122, 123, 125, 126, 127, 128, 130, 131, 133, 134, 135, 136, 137, 142, 143, 144, 145, 146, 148, 149, 151, 153, 155, 158, 161, 163, 164, 165, 166, 167, 170, 172, 173, 174, 175, 178, 181, 183, 185, 186, 187, 191, 192, 193, 195, 196, 197, 199, 200, 201, 203, 205, 207, 210, 211, 212, 214, 216, 218, 220, 221, 223, 225, 226, 227, 228, 229, 230, 233, 235, 237, 238, 239, 242, 244, 245, 246, 247, 249, 250, 252, 253]
    # include_indices = [i for i, val in enumerate(prep_fidelity_list) if val >= 0.4 or val is None]
    # include_indices =  [i for i, val in enumerate(snr_float) if val >= 0.02]
    
    data = dm.get_raw_data(
        # file_stem="2026_03_25-16_28_08-charge_state_analysis_hist_data_raw_data", load_npz=True
        file_stem="2026_03_25-18_15_53-charge_state_analysis_hist_data_raw_data", load_npz=True
    )
    readout_fidelity_list = data["readout_fidelity_list"]
    include_indices = [
        i for i, val in enumerate(readout_fidelity_list)
        if (val is None) or (isinstance(val, (int, float)) and not math.isnan(val) and val >= 0.6)
    ]
    
    # print(np.sort(list(include_indices)))
    # sys.exit()
    # indices = np.sort(list(include_indices))
    # print(", ".join(str(i) for i in indices))
    # print("[" + ", ".join(map(str, np.sort(list(include_indices)))) + "]")

    # nv_powers = [val for ind, val in enumerate(nv_powers) if ind not in drop_indices]
    # print(len(include_indices))
    # fmt: on
    filtered_reordered_coords = [filtered_reordered_coords[i] for i in include_indices]
    spot_weights = [spot_weights[i] for i in include_indices]
    
    # print(f"len filtered_reordered_coords: {len(filtered_reordered_coords)}")
    # # # select_half_left_side_nvs_and_plot(nv_coordinates_filtered)
    # spot_weights = np.array(
    #     [weight for i, weight in enumerate(spot_weights) if i in include_indices]
    # )
    # print(f"len spot_weights: {len(spot_weights)}")
    # filtered_pol_durs = [pol_duration_list[i] for i in include_indices]
    # filtered_scc_durs = [scc_duration_list[i] for i in include_indices]
    # print(filtered_pol_durs)
    # print(f"len spot_weights: {len(spot_weights)}")
    # print(filtered_scc_durs)


    # sys.exit()
    # print(len(spot_weights_filtered))
    # # sys.exit()
    # aom_voltage = 0.2861
    # aom_voltage = 0.3084 ### 223NVs
    # aom_voltage = 0.299064 ###204NVs
    # aom_voltage = 0.3707 ### 279NVs
    aom_voltage = 0.3344 ### 279NVs
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
    filtered_reordered_spot_weights = spot_weights
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
    # filtered_reordered_spot_weights = filtered_reordered_spot_weights[:4094]
    # filtered_reordered_coords = filtered_reordered_coords[:4094]
    # # Save the filtered results
    # save_results(
    #     filtered_reordered_coords,
    #     filtered_reordered_spot_weights,
    #     filename="slmsuite/nv_blob_detection/nv_blob_1306nvs_reordered.npz",
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
