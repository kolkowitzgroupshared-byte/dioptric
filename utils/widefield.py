# -*- coding: utf-8 -*-
"""Various utility functions for widefield imaging and camera data processing

Created on August 15th, 2023
@author: mccambria
Updated on September 16th, 2024
@author: sbchand
"""

# region Imports and constants

import pickle
import sys
from datetime import datetime
from functools import cache
from importlib import import_module
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib import animation
from numpy import inf
from scipy.linalg import expm
from scipy.special import gamma
from scipy.stats import poisson
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle
from skimage.measure import ransac
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey

# endregion
# region Image processing


def crop_img_array(img_array, offset=[0, 0], buffer=20):
    offset = [round(el) for el in offset]
    size = img_array.shape[-1]
    if size == 250:
        replace_dead_pixel(img_array)
    # print([buffer + offset[0], buffer + offset[1]])
    img_array = img_array[
        buffer + offset[0] : size - buffer + offset[0],
        buffer + offset[1] : size - buffer + offset[1],
    ]
    return img_array


def mask_img_array(img_array, nv_list, pixel_drift):
    shape = img_array.shape
    x_mesh, y_mesh = np.meshgrid(list(range(shape[0])), list(range(shape[1])))
    mask = np.zeros(shape)
    radius = _get_camera_spot_radius()
    for nv in nv_list:
        pixel_coords = pos.get_nv_coords(
            nv, CoordsKey.PIXEL, drift_adjust=True, drift=pixel_drift
        )
        dist = np.sqrt(
            (x_mesh - pixel_coords[0]) ** 2 + (y_mesh - pixel_coords[1]) ** 2
        )
        mask += np.where(dist < radius, 1, 0)
    # fig, ax = plt.subplots()
    # kpl.imshow(ax, mask)
    return img_array * mask


def crop_img_arrays(img_arrays, offsets=[0, 0], buffer=20):
    shape = img_arrays.shape
    cropped_shape = list(shape)
    cropped_shape[-1] -= 2 * buffer
    cropped_shape[-2] -= 2 * buffer
    cropped_img_arrays = np.empty(cropped_shape)
    for exp_ind in range(shape[0]):
        for run_ind in range(shape[1]):
            offset = offsets[run_ind]
            for step_ind in range(shape[2]):
                for rep_ind in range(shape[2]):
                    img_array = img_arrays[exp_ind, run_ind, step_ind, rep_ind]
                    cropped_img_array = crop_img_array(img_array, offset, buffer)
                    cropped_img_arrays[exp_ind, run_ind, step_ind, rep_ind] = (
                        cropped_img_array
                    )
    return cropped_img_arrays


def integrate_counts_from_adus(img_array, pixel_coords, radius=None):
    img_array_photons = adus_to_photons(img_array)
    return integrate_counts(img_array_photons, pixel_coords, radius)


@cache
def _calc_dist_matrix(radius=None):
    if radius is None:
        radius = _get_camera_spot_radius()
    left = -radius
    right = +radius
    top = -radius
    bottom = +radius
    x_crop = list(range(left, right + 1))
    y_crop = list(range(top, bottom + 1))
    x_crop_mesh, y_crop_mesh = np.meshgrid(x_crop, y_crop)
    return np.sqrt((x_crop_mesh) ** 2 + (y_crop_mesh) ** 2)

# SBC: update on 9/9/2024
def integrate_counts(img_array, pixel_coords, radius=None):
    """
    Add up the counts around a target set of pixel coordinates in the passed image array.
    Use for getting the total number of photons coming from a target NV.

    Parameters
    ----------
    img_array : ndarray
        Image array in units of photons (convert from ADUs with adus_to_photons).
    pixel_coords : 2-tuple
        Pixel coordinates to integrate around.
    radius : float, optional
        Radius of disk to integrate over. Default is retrieved from config.

    Returns
    -------
    float
        Integrated counts (estimate, as adus_to_photons is also an estimate).
    """
    pixel_x = pixel_coords[0]
    pixel_y = pixel_coords[1]

    if radius is None:
        radius = _get_camera_spot_radius()

    # Crop the region of interest based on the radius around the target pixel
    left = max(0, round(pixel_x - radius))
    right = min(img_array.shape[1] - 1, round(pixel_x + radius))
    top = max(0, round(pixel_y - radius))
    bottom = min(img_array.shape[0] - 1, round(pixel_y + radius))

    img_array_crop = img_array[top : bottom + 1, left : right + 1]

    # Calculate a distance matrix for the cropped image
    y_coords, x_coords = np.ogrid[top : bottom + 1, left : right + 1]
    dist = np.sqrt((x_coords - pixel_x) ** 2 + (y_coords - pixel_y) ** 2)

    # Sum the pixel values within the specified radius
    mask = dist < radius
    counts = np.sum(img_array_crop[mask])

    return counts

def integrate_counts_bg_subtracted(
    img_array,
    pixel_coords,
    radius=None,
    bg_inner_radius=None,
    bg_outer_radius=None,
):
    """
    Integrate counts in a disk around the NV and subtract local background
    estimated from an annulus around the same NV.
    """
    pixel_x = pixel_coords[0]
    pixel_y = pixel_coords[1]

    if radius is None:
        radius = _get_camera_spot_radius()

    if bg_inner_radius is None:
        bg_inner_radius = radius + 2.0
    if bg_outer_radius is None:
        bg_outer_radius = radius + 3.0

    half_width = int(np.ceil(bg_outer_radius))
    left = max(0, round(pixel_x - half_width))
    right = min(img_array.shape[1] - 1, round(pixel_x + half_width))
    top = max(0, round(pixel_y - half_width))
    bottom = min(img_array.shape[0] - 1, round(pixel_y + half_width))

    img_crop = img_array[top : bottom + 1, left : right + 1]

    y_coords, x_coords = np.ogrid[top : bottom + 1, left : right + 1]
    dist = np.sqrt((x_coords - pixel_x) ** 2 + (y_coords - pixel_y) ** 2)

    sig_mask = dist < radius
    bg_mask = (dist >= bg_inner_radius) & (dist < bg_outer_radius)

    sig_sum = np.sum(img_crop[sig_mask])

    if np.count_nonzero(bg_mask) == 0:
        return sig_sum

    bg_mean = np.mean(img_crop[bg_mask])
    bg_subtracted_counts = sig_sum - bg_mean * np.count_nonzero(sig_mask)

    return bg_subtracted_counts

def fit_max_counts(img_array, pixel_coords, radius=None):
    """
    Fit a 2D Gaussian to the image around the NV site to determine the peak count.
    """
    pixel_x, pixel_y = pixel_coords

    if radius is None:
        radius = _get_camera_spot_radius()

    # Crop the region of interest based on the radius around the target pixel
    left = max(0, round(pixel_x - radius))
    right = min(img_array.shape[1] - 1, round(pixel_x + radius))
    top = max(0, round(pixel_y - radius))
    bottom = min(img_array.shape[0] - 1, round(pixel_y + radius))

    img_array_crop = img_array[top : bottom + 1, left : right + 1]

    # Define 2D Gaussian function
    def gaussian_2d(coords, amp, x0, y0, sigma_x, sigma_y, offset):
        x, y = coords
        return (
            amp
            * np.exp(
                -(
                    ((x - x0) ** 2) / (2 * sigma_x**2)
                    + ((y - y0) ** 2) / (2 * sigma_y**2)
                )
            )
            + offset
        )

    # Generate coordinate grids
    y_vals, x_vals = np.mgrid[top : bottom + 1, left : right + 1]
    x_flat, y_flat = x_vals.ravel(), y_vals.ravel()
    img_flat = img_array_crop.ravel()

    # Initial guess (Amplitude, x0, y0, σx, σy, offset)
    p0 = [np.max(img_flat), pixel_x, pixel_y, radius / 2, radius / 2, np.min(img_flat)]

    try:
        popt, _ = opt.curve_fit(gaussian_2d, (x_flat, y_flat), img_flat, p0=p0)
        max_counts = popt[0]  # Extract fitted peak amplitude
    except RuntimeError:
        max_counts = np.max(img_array_crop)  # Fallback to raw max if fitting fails

    return max_counts


def adus_to_photons(adus, k_gain=None, em_gain=None, baseline=None):
    """Convert camera analog-to-digital converter units (ADUs) to an
    estimated number of photons. Since the gain stages are noisy, this is
    just an estimate

    Parameters
    ----------
    adus : numeric
        Quantity to convert in ADUs
    k_gain : numeric, optional
        k gain of the camera in e- / ADU, by default retrieved from config
    em_gain : numeric, optional
        Electron-multiplying gain of the camera, by default retrieved from config
    baseline : numeric, optional
        Baseline, or bias clamp level, i.e. the ADU value for a pixel which receives no light.
        Used to ensure the camera does not return negative values. By default retrieved from config

    Returns
    -------
    numeric
        Quantity converted to photons
    """
    if k_gain is None:
        k_gain = _get_camera_k_gain()
    if em_gain is None:
        em_gain = _get_camera_em_gain()
    if baseline is None:
        baseline = _get_camera_bias_clamp()

    total_gain = k_gain / em_gain
    photons = (adus - baseline) * total_gain
    return photons


@cache
def _img_array_iris(shape):
    roi = _get_camera_roi()  # offsetX, offsetY, width, height
    offsetX, offsetY, width, height = roi
    iris_radius = np.sqrt((height / 2) ** 2 + (width / 2) ** 2) + 25
    center_x = offsetX + width // 2
    center_y = height // 2

    iris = np.empty(shape)
    for ind in range(shape[0]):
        for jnd in range(shape[1]):
            dist = np.sqrt((jnd - center_x) ** 2 + (ind - center_y) ** 2)
            iris[ind, jnd] = dist > iris_radius

    return iris


def img_str_to_array(img_str):
    """Convert an img_array from a uint16-valued byte string (returned by the camera
    labrad server for speed) into a usable int-valued 2D array. Also subtracts off bias
    not captured on Nuvu's end camera software

    Parameters
    ----------
    img_str : byte string
        Image array as a byte string - the return value of a camera.read() call

    Returns
    -------
    ndarray
        Image array contructed from the byte string
    """
    shape = _get_img_str_shape()
    img_array = np.frombuffer(img_str, dtype=np.uint16).reshape(*shape)
    img_array = img_array.astype(int)

    # Subtract off correlated readout noise (see wiki 4/19/24)
    roi = _get_camera_roi()  # offsetX, offsetY, width, height
    if roi is None:
        baseline = _get_camera_bias_clamp()
    else:
        offset_x = roi[0]
        width = roi[2]
        iris = _img_array_iris(img_array.shape)
        bg_pixels = np.where(iris, img_array, np.nan)
        baseline = np.nanmean(bg_pixels)
        img_array = img_array[0:, offset_x : offset_x + width]
    return img_array, baseline


def _validate_counts_structure(counts):
    """Validate counts with structure [nv_ind, run_ind, step_ind, rep_ind]."""
    if counts is None:
        return
    if not isinstance(counts, np.ndarray):
        raise RuntimeError("Passed counts object is not a numpy array.")
    if counts.ndim != 4:
        raise RuntimeError("Passed counts object has the wrong number of dimensions.")


# Axes constants
run_ax = 1
rep_ax = 3
run_rep_axes = (run_ax, rep_ax)


def average_counts(sig_counts, ref_counts=None):
    """
    Gets average and standard error for counts data structure.
    Counts arrays must have structure [nv_ind, run_ind, step_ind, rep_ind].
    Returns arrays shaped [nv_ind, step_ind] for avg_counts and avg_counts_ste.
    Returns norms = [ms0_mean_per_nv, ms1_mean_per_nv] if ref_counts provided.
    """
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)

    sig_counts = _safe_to_f64(sig_counts)

    # Mean across (run, rep) -> keeps [nv_ind, step_ind]
    avg_counts = _mean_nan(sig_counts, axis=run_rep_axes)

    # STD across (run, rep) with ddof guard
    avg_counts_std = _std_ddof_guard(sig_counts, run_rep_axes)

    # Number of shots across run & rep; if it's 0, fall back to 1 to avoid division by zero
    num_shots = int(sig_counts.shape[rep_ax]) * int(sig_counts.shape[run_ax])
    denom = np.sqrt(max(num_shots, 1))
    avg_counts_ste = _nan0(avg_counts_std / denom)

    norms = None
    if ref_counts is not None:
        ref_counts = _safe_to_f64(ref_counts)

        # Even indices -> ms=0, odd indices -> ms=±1 (your original convention)
        ms0_ref_counts = ref_counts[:, :, :, 0::2]
        ms1_ref_counts = ref_counts[:, :, :, 1::2]

        # Means per NV across all other axes
        ms0_mean = _mean_nan(ms0_ref_counts, axis=(1, 2, 3))
        ms1_mean = _mean_nan(ms1_ref_counts, axis=(1, 2, 3))
        norms = [_nan0(ms0_mean), _nan0(ms1_mean)]

    return _nan0(avg_counts), _nan0(avg_counts_ste), norms


_SAFE_EPS = 1e-12


def _safe_to_f64(x):
    return np.asarray(x, dtype=np.float64)


def _nan0(x):
    # Replace NaN/Inf with 0 to avoid propagation
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _std_ddof_guard(x, axes):
    """
    nanstd with ddof=1 when we have >=2 samples total on (run, rep),
    else ddof=0 to avoid NaNs from Bessel correction.
    """
    x = _safe_to_f64(x)
    # Estimate shots across the axes we reduce (assumes axes are run & rep)
    # If axes can be in any order, this still multiplies correct counts.
    n = 1
    for ax in axes:
        n *= x.shape[ax]
    ddof = 1 if n >= 2 else 0
    with np.errstate(all="ignore"):
        return np.nanstd(x, axis=axes, ddof=ddof)


def _mean_nan(x, axis=None):
    x = _safe_to_f64(x)
    with np.errstate(all="ignore"):
        return np.nanmean(x, axis=axis)


def _safe_div(num, den):
    num = _safe_to_f64(num)
    den = _safe_to_f64(den)
    # Regularize denominator and avoid divide-by-zero/NaN
    den = np.hypot(den, _SAFE_EPS)
    out = np.zeros_like(num, dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (den > 0.0)
    np.divide(num, den, out=out, where=valid)
    return _nan0(out)


def calc_snr(sig_counts, ref_counts):
    """Calculate SNR for a single shot"""
    avg_contrast, avg_contrast_ste = calc_contrast(sig_counts, ref_counts)

    # Noise is quadrature sum of sig/ref standard deviations over (run, rep)
    std_sig = _std_ddof_guard(sig_counts, run_rep_axes)
    std_ref = _std_ddof_guard(ref_counts, run_rep_axes)
    noise = np.hypot(std_sig, std_ref)  # sqrt(sig^2 + ref^2)

    avg_snr = _safe_div(avg_contrast, noise)
    avg_snr_ste = _safe_div(avg_contrast_ste, noise)
    return avg_snr, avg_snr_ste


def calc_contrast(sig_counts, ref_counts):
    """Calculate contrast for a single shot"""
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)

    avg_sig_counts, avg_sig_counts_ste, _ = average_counts(sig_counts)
    avg_ref_counts, avg_ref_counts_ste, _ = average_counts(ref_counts)

    avg_contrast = _safe_to_f64(avg_sig_counts) - _safe_to_f64(avg_ref_counts)
    # STEs add in quadrature
    avg_contrast_ste = np.hypot(
        _safe_to_f64(avg_sig_counts_ste), _safe_to_f64(avg_ref_counts_ste)
    )
    return _nan0(avg_contrast), _nan0(avg_contrast_ste)


def threshold_counts(nv_list, sig_counts, ref_counts=None, dynamic_thresh=False):
    """Only actually thresholds counts for NVs with thresholds specified in their sigs.
    If there's no threshold, then the raw counts are just averaged as normal."""
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)

    num_nvs = len(nv_list)
    if dynamic_thresh:
        thresholds = []
        for nv_ind in range(num_nvs):
            if ref_counts is not None:
                combined_counts = np.append(
                    sig_counts[nv_ind].flatten(), ref_counts[nv_ind].flatten()
                )
            else:
                combined_counts = sig_counts[nv_ind].flatten()
            prob_dist = ProbDist.COMPOUND_POISSON
            popt, _, _ = fit_bimodal_histogram(combined_counts, prob_dist, no_plot=True)
            threshold = determine_threshold(popt, prob_dist, dark_mode_weight=0.5)
            thresholds.append(threshold)
    else:
        thresholds = [nv.threshold for nv in nv_list]
        # thresholds = [2.5 for nv in nv_list]
        # thresholds = adaptive_threshold_counts()
    # print(thresholds)

    shape = sig_counts.shape
    sig_states = np.empty(shape)
    for nv_ind in range(num_nvs):
        sig_states[nv_ind] = tb.threshold(sig_counts[nv_ind], thresholds[nv_ind])

    if ref_counts is not None:
        ref_states = np.empty(shape)
        for nv_ind in range(num_nvs):
            ref_states[nv_ind] = tb.threshold(ref_counts[nv_ind], thresholds[nv_ind])
        return sig_states, ref_states
    else:
        return sig_states


def process_counts(nv_list, sig_counts, ref_counts=None, threshold=True):
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)
    if threshold:
        if ref_counts is not None:
            sig_counts, ref_counts = threshold_counts(
                nv_list, sig_counts, ref_counts, dynamic_thresh=True
            )
        else:
            sig_counts = threshold_counts(nv_list, sig_counts, dynamic_thresh=True)
    avg_counts, avg_counts_ste, norms = average_counts(sig_counts, ref_counts)

    if ref_counts is None:
        return avg_counts, avg_counts_ste

    norms_ms0_newaxis = norms[0][:, np.newaxis]
    norms_ms1_newaxis = norms[1][:, np.newaxis]
    contrast = norms_ms1_newaxis - norms_ms0_newaxis
    contrast = np.where(contrast > 0.03, contrast, 0.03)
    norm_counts = (avg_counts - norms_ms0_newaxis) / contrast
    norm_counts_ste = avg_counts_ste / contrast

    return norm_counts, norm_counts_ste


# Combine data from multiple file IDs
def process_multiple_files(file_ids, load_npz=True):
    """
    Load and combine data from multiple file IDs.

    """
    # print(file_ids[0])
    combined_data = dm.get_raw_data(
        file_stem=file_ids[0], load_npz=load_npz, use_cache=False
    )
    counts = np.array(combined_data["counts"])
    print(f"combined data shape : {counts.shape}")
    seq_xy = combined_data.get("xy_seq", "xy8").lower()
    print(seq_xy)
    for file_id in file_ids[1:]:
        new_data = dm.get_raw_data(
            file_stem=file_id, load_npz=load_npz, use_cache=False
        )
        seq_xy = combined_data.get("xy_seq", "xy8").lower()
        print(seq_xy)
        new_counts = np.array(new_data["counts"])
        print(f"new data shape : {new_counts.shape}")
        combined_data["num_runs"] += new_data["num_runs"]
        combined_data["counts"] = np.append(
            combined_data["counts"], new_data["counts"], axis=2
        )
    return combined_data


def combined_filename(file_ids):
    all_file_ids_str = "_".join(map(str, file_ids))
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    file_name = dm.get_file_name(file_id=file_ids[0])
    file_path = dm.get_file_path(
        __file__, file_name, f"{all_file_ids_str}_{date_time_str}"
    )
    return file_path, all_file_ids_str


def combined_filenames(file_stems):
    all_file_ids_str = "_".join(file_stems)
    return all_file_ids_str


def combined_filenames_numbered(file_stems):
    all_file_ids_str = "_".join(f"F{i}={stem}" for i, stem in enumerate(file_stems))
    return all_file_ids_str


def parse_xy_sequence(seq_name: str):
    """
    Parses sequence names like 'xy8' or 'xy8-2' into base sequence and number of blocks (N).
    Returns (base_seq, N, total_pulses, total_evolution_time_units).
    """
    # Comprehensive code to parse XY sequence strings (e.g., 'xy8', 'xy8-2')
    # and calculate key sequence parameters.
    available = ["hahn-n", "xy2-n", "xy4-n", "xy8-n", "xy16-n"]
    # Default values
    default_seq = "xy8"
    default_n = 1
    seq_name = seq_name.lower().strip()
    base_seq, N = default_seq, default_n

    # Try to parse the sequence string
    if "-" in seq_name:
        base_seq, n_str = seq_name.split("-", 1)
        try:
            N = int(n_str)
        except ValueError:
            print(f"Invalid N in '{seq_name}'. Defaulting to N={default_n}.")
            base_seq = default_seq
    else:
        base_seq = seq_name

    # Check if it's allowed
    if f"{base_seq}-n" not in available:
        print(
            f"Sequence '{base_seq}' not recognized. Defaulting to '{default_seq}-{default_n}'."
        )
        base_seq = default_seq
        N = default_n

    # Define pulses per block for each sequence type
    pulses_per_block = {
        "hahn": 1,
        "xy2": 2,
        "xy4": 4,
        "xy8": 8,
        "xy16": 16,
    }
    pulses_per = pulses_per_block.get(base_seq, 8)  # Default to xy8 if unknown

    N = pulses_per * N

    return base_seq, N


def threshold_counts_selected_method(
    nv_list, sig_counts, ref_counts=None, method="otsu"
):
    """Threshold counts for NVs based on the selected method."""
    _validate_counts_structure(sig_counts)
    _validate_counts_structure(ref_counts)

    num_nvs = len(nv_list)
    sig_thresholds, ref_thresholds = [], []

    # Process thresholds based on the selected method
    for nv_ind in range(num_nvs):
        combined_counts = (
            np.append(sig_counts[nv_ind].flatten(), ref_counts[nv_ind].flatten())
            if ref_counts is not None
            else sig_counts[nv_ind].flatten()
        )

        # Choose method for thresholding
        if method == "otsu":
            threshold = threshold_otsu(combined_counts)
        elif method == "triangle":
            threshold = threshold_triangle(combined_counts)
        elif method == "entropy":
            threshold = threshold_li(combined_counts)
        elif method == "mean":
            threshold = np.mean(combined_counts)
        elif method == "median":
            threshold = np.median(combined_counts)
        elif method == "kmeans":
            threshold = kmeans_threshold(combined_counts)
        elif method == "gmm":
            threshold = gmm_threshold(combined_counts, use_intersection=False)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")

        sig_thresholds.append(threshold)
        if ref_counts is not None:
            ref_thresholds.append(threshold)

    # Apply thresholds to signal counts (assuming single threshold value per NV)
    sig_states = np.array(
        [sig_counts[nv_ind] > sig_thresholds[nv_ind] for nv_ind in range(num_nvs)]
    )
    print(sig_thresholds)
    if ref_counts is not None:
        ref_states = np.array(
            [ref_counts[nv_ind] > ref_thresholds[nv_ind] for nv_ind in range(num_nvs)]
        )
        return sig_states, ref_states
    else:
        return sig_states


# Adaptive clustering-based thresholding (K-means or GMM)
def adaptive_clustering_thresholding(counts, method="kmeans", block_size=11):
    """Apply adaptive K-means or GMM thresholding to each block of data."""
    adaptive_counts = np.zeros_like(counts)

    # Divide data into blocks and apply thresholding
    for i in range(0, counts.shape[0], block_size):
        for j in range(0, counts.shape[1], block_size):
            block = counts[i : i + block_size, j : j + block_size].flatten()

            if method == "kmeans":
                threshold = kmeans_threshold(block)
            elif method == "gmm":
                threshold = gmm_threshold(block)
            else:
                raise ValueError(f"Unknown adaptive clustering method: {method}")

            # Apply threshold to the block
            adaptive_counts[i : i + block_size, j : j + block_size] = block > threshold

    return adaptive_counts


# K-means thresholding function
def kmeans_threshold(data):
    kmeans = KMeans(n_clusters=2).fit(data.reshape(-1, 1))
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    return np.mean(cluster_centers)


# GMM thresholding function
def gmm_threshold(data, use_intersection=False):
    """
    Finds a threshold for separating data using a Gaussian Mixture Model (GMM).

    Parameters:
    - data: Input data to find threshold for (1D array).
    - use_intersection: If True, computes the intersection point between the two Gaussians.

    Returns:
    - threshold: The calculated threshold (weighted mean or intersection of the two Gaussians).
    """
    data = data.reshape(-1, 1)  # Reshape data to 2D for GMM fitting
    gmm = GaussianMixture(n_components=2).fit(data)

    means = sorted(gmm.means_.flatten())  # Get the means of the two Gaussians
    weights = gmm.weights_  # Get the mixing weights of the two Gaussians
    covariances = np.sqrt(gmm.covariances_).flatten()  # Standard deviations

    if use_intersection:
        # Find the intersection of the two Gaussian distributions
        mean1, mean2 = means
        std1, std2 = covariances
        intersection = (mean1 * std2**2 - mean2 * std1**2) / (std2**2 - std1**2)
        return intersection

    # Compute a weighted average of the means (more robust if one Gaussian dominates)
    weighted_mean = np.average(means, weights=weights)

    return weighted_mean


def poisson_pmf_cont(k, mean):
    return mean**k * np.exp(-mean) / gamma(k + 1)


@cache
def _get_mle_radius():
    return round(1.5 * _get_camera_spot_radius())


@cache
def _calc_mesh_grid(radius=None):
    radius = _get_mle_radius()
    half_range = radius
    one_ax_linspace = np.linspace(-half_range, half_range, 2 * half_range + 1)
    x_crop_mesh, y_crop_mesh = np.meshgrid(one_ax_linspace, one_ax_linspace)
    return x_crop_mesh, y_crop_mesh


@cache
def _calc_nvn_count_distribution(nvn_dist_params, subpixel_offset=(0, 0)):
    x_crop_mesh, y_crop_mesh = _calc_mesh_grid()
    bg, amp, sigma = nvn_dist_params
    return bg + amp * np.exp(
        -(
            ((x_crop_mesh - subpixel_offset[0]) ** 2)
            + ((y_crop_mesh - subpixel_offset[1]) ** 2)
        )
        / (2 * sigma**2)
    )


@cache
def _calc_nv0_count_distribution(nvn_dist_params):
    bg = nvn_dist_params[0]
    return bg


def charge_state_mle_single(nv_sig, img_array):
    nvn_dist_params = nv_sig.nvn_dist_params
    if nvn_dist_params is None:
        return None

    x0, y0 = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
    radius = _get_mle_radius()
    half_range = radius
    left = round(x0 - half_range)
    right = round(x0 + half_range)
    top = round(y0 - half_range)
    bottom = round(y0 + half_range)
    img_array_crop = img_array[top : bottom + 1, left : right + 1]
    img_array_crop = np.where(img_array_crop >= 0, img_array_crop, 0)

    subpixel_offset = (x0 - round(x0), y0 - round(y0))
    nvn_count_distribution = _calc_nvn_count_distribution(
        nvn_dist_params, subpixel_offset
    )
    nv0_count_distribution = _calc_nv0_count_distribution(nvn_dist_params)

    nvn_probs = poisson_pmf_cont(img_array_crop, nvn_count_distribution)
    nv0_probs = poisson_pmf_cont(img_array_crop, nv0_count_distribution)
    nvn_prob = np.nanprod(nvn_probs)
    nv0_prob = np.nanprod(nv0_probs)
    return int(nvn_prob > nv0_prob)


def charge_state_mle(nv_list, img_array):
    """Maximum likelihood estimator of state based on image"""

    states = [charge_state_mle_single(nv, img_array) for nv in nv_list]

    return states


# def calc_snr(sig_counts, ref_counts):
#     """Calculate SNR for a single shot"""
#     avg_contrast, avg_contrast_ste = calc_contrast(sig_counts, ref_counts)
#     eps = 1e-12
#     noise = np.sqrt(
#         np.std(sig_counts, axis=run_rep_axes, ddof=1) ** 2
#         + np.std(ref_counts, axis=run_rep_axes, ddof=1) ** 2
#     )
#     avg_snr = avg_contrast / noise
#     avg_snr_ste = avg_contrast_ste / noise
#     return avg_snr, avg_snr_ste


# def calc_contrast(sig_counts, ref_counts):
#     """Calculate contrast for a single shot"""
#     _validate_counts_structure(sig_counts)
#     _validate_counts_structure(ref_counts)
#     avg_sig_counts, avg_sig_counts_ste, _ = average_counts(sig_counts)
#     avg_ref_counts, avg_ref_counts_ste, _ = average_counts(ref_counts)
#     avg_contrast = avg_sig_counts - avg_ref_counts
#     avg_contrast_ste = np.sqrt((avg_sig_counts_ste**2 + avg_ref_counts_ste**2))
#     return avg_contrast, avg_contrast_ste


# def _validate_counts_structure(counts):
#     """Make sure the structure of the counts object for a single experiment is valid
#     for further processing. The structure is [nv_ind, run_ind, step_ind, rep_ind]."""
#     if counts is None:
#         return
#     if not isinstance(counts, np.ndarray):
#         raise RuntimeError("Passed counts object is not a numpy array.")
#     if counts.ndim != 4:
#         raise RuntimeError("Passed counts object has the wrong number of dimensions.")


# endregion


# region Miscellaneous public functions


def rep_loop(num_reps, rep_fn):
    """Loop through the reps for a routine. Handles errors from the camera,
    as well as starting and stopping the pulse generator. Does not handle
    loading the sequence or arming and disarming the camera.

    Parameters
    ----------
    num_reps : int
        Number of reps
    rep_fn : function
        Function to run for each rep
    """
    pulse_gen = tb.get_server_pulse_gen()
    camera = tb.get_server_camera()

    # Try 5 times then give up
    num_attempts = 5
    attempt_ind = 0
    start_rep = 0
    while True:
        try:
            pulse_gen.stream_start()

            for rep_ind in range(start_rep, num_reps):
                rep_fn(rep_ind)

            break

        except Exception as exc:
            pulse_gen.halt()

            nuvu_237 = "NuvuException: 237"
            nuvu_214 = "NuvuException: 214"
            if "NuvuException: 237" in str(exc):
                print(nuvu_237)
            elif "NuvuException: 214" in str(exc):
                print(nuvu_214)
            else:
                raise exc

            attempt_ind += 1
            if attempt_ind == num_attempts:
                raise RuntimeError("Maxed out number of attempts")

            camera.arm()
            start_rep = rep_ind

    pulse_gen.halt()

    if attempt_ind > 0:
        print(f"{attempt_ind} crashes occurred")


def get_repr_nv_sig(nv_list: list[NVSig]) -> NVSig:
    for nv in nv_list:
        try:
            if nv.representative:
                return nv
        except Exception:
            pass


def get_nv_num(nv_sig):
    nv_name = nv_sig.name  # of the form <sample_name>-nv<num>_<date_str>
    nv_name_part = nv_name.split("-")[1]  # nv<num>_<date_str>
    nv_num = nv_name_part.split("_")[0][2:]
    nv_num = int(nv_num)
    return nv_num


def get_base_scc_seq_args(nv_list: list[NVSig], uwave_ind_list: list[int]):
    """Return base seq_args for any SCC routine. The base sequence arguments
    are the minimum info required for state preparation and SCC.

    Parameters
    ----------
    nv_list : list[NVSig]
        List of nv signatures to target
    uwave_ind_list : list[int]
        List of indices of the microwave chains to run for state prep

    Returns
    -------
    list
        Sequence arguments
    """

    pol_coords_list, pol_duration_list, pol_amp_list = get_pulse_parameter_lists(
        nv_list, VirtualLaserKey.CHARGE_POL
    )

    scc_coords_list, scc_duration_list, scc_amp_list = get_pulse_parameter_lists(
        nv_list, VirtualLaserKey.SCC
    )

    spin_flip_ind_list = get_spin_flip_do_target_list(nv_list)
    # threshold_list = get_threshold_list(nv_list, include_inds=scc_include_inds)

    # Create list of arguments
    seq_args = [
        pol_coords_list,
        pol_duration_list,
        pol_amp_list,
        scc_coords_list,
        scc_duration_list,
        scc_amp_list,
        spin_flip_ind_list,
        uwave_ind_list,
    ]
    # print(seq_args)

    return seq_args


def get_pulse_parameter_lists(nv_list: list[NVSig], virtual_laser_key: VirtualLaserKey):
    coords_list = get_coords_list(nv_list, virtual_laser_key)
    duration_list = []
    amp_list = []
    for nv in nv_list:
        # Retrieve duration and amplitude using .get to avoid KeyError
        duration = nv.pulse_durations.get(virtual_laser_key)
        amp = nv.pulse_amps.get(virtual_laser_key)
        # print(f"DEBUG: nv={nv}, duration={duration}, amp={amp}")
        duration_list.append(duration)
        amp_list.append(amp)

    # The lists will be passed to qua.for_each in the sequence, so each entry needs
    # to be a proper number, not None
    default_duration = int(tb.get_virtual_laser_dict(virtual_laser_key)["duration"])
    default_amp = 1.0

    duration_list = [
        val if val is not None else default_duration for val in duration_list
    ]
    amp_list = [float(val) if val is not None else default_amp for val in amp_list]

    # Debugging: Ensure all values are correct
    # print(f"DEBUG: Final duration list: {duration_list}")
    # print(f"DEBUG: Final amplitude list: {amp_list}")

    return coords_list, duration_list, amp_list


def get_coords_list(nv_list: list[NVSig], laser_key, drift_adjust=None):
    laser_positioner = pos.get_laser_positioner(laser_key)
    coords_list = [
        pos.get_nv_coords(nv, coords_key=laser_positioner, drift_adjust=drift_adjust)
        for nv in nv_list
    ]
    return coords_list


def get_spin_flip_do_target_list(nv_list: list[NVSig]):
    return [nv.spin_flip for nv in nv_list]


def get_threshold_list(nv_list: list[NVSig], include_inds=None):
    threshold_list = []
    for nv in nv_list:
        threshold = nv.threshold
        if threshold is None:
            config = common.get_config_dict()
            threshold = config["Default"]["threshold"]
        threshold_list.append(threshold)
    if include_inds is not None:
        threshold_list = [threshold_list[ind] for ind in include_inds]
    return threshold_list


# endregion


# region public funtion for control panel
# Assuming nv_list is a list of NV centers with their (x, y, z) coordinates
def select_well_separated_nvs(nv_list, num_nvs_to_select):
    """
    Selects a specified number of well-separated NVs from a list based on their pixel coordinates.

    Args:
        nv_list (list[NVSig]): List of NVs.
        num_nvs_to_select (int): Number of NVs to select.

    Returns:
        list: Indices of the selected NVs.
    """
    # Extract pixel coordinates from nv_list
    pixel_coords_list = [nv.coords[CoordsKey.PIXEL] for nv in nv_list]
    coords_array = np.array(pixel_coords_list)

    # Calculate pairwise Euclidean distances between NVs
    distances = np.linalg.norm(
        coords_array[:, np.newaxis] - coords_array[np.newaxis, :], axis=2
    )

    # List to hold indices of selected NVs
    selected_indices = []

    # Initialize by selecting the first NV
    selected_indices.append(0)

    # Greedily select NVs that maximize the minimum distance to the already selected NVs
    for _ in range(1, num_nvs_to_select):
        max_min_distance = -np.inf
        next_index = -1
        for i in range(len(nv_list)):
            if i in selected_indices:
                continue
            # Calculate the minimum distance of this NV to any of the already selected NVs
            min_distance = min(distances[i, idx] for idx in selected_indices)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_index = i
        selected_indices.append(next_index)

    return selected_indices


def select_half_left_side_nvs(nv_list):
    """
    Assign spin_flip states to NVs based on their x-coordinate.
    - NVs on the left side (x < median_x) are assigned spin_flip = True.

    Parameters:
    - nv_list: List of NV objects, each with a 'coords' dictionary containing 'pixel'.

    Returns:
    - left_indices: Indices of NVs on the left side.
    """
    # Extract NV pixel coordinates
    nv_coordinates = np.array([nv.coords["pixel"][:2] for nv in nv_list])

    # Find the median x-coordinate
    median_x = np.median(nv_coordinates[:, 0])

    # Separate NVs into left and right based on the median x-coordinate
    left_indices = [i for i, coord in enumerate(nv_coordinates) if coord[0] < median_x]

    return left_indices


def hybrid_tau_spacing(min_tau, max_tau, num_steps, log_frac=0.6):
    N_log = int(num_steps * log_frac)
    N_lin = num_steps - N_log

    log_max = 10 ** (
        np.log10(min_tau) + (np.log10(max_tau) - np.log10(min_tau)) * log_frac
    )
    taus_log = np.logspace(np.log10(min_tau), np.log10(log_max), N_log, endpoint=False)
    taus_lin = np.linspace(log_max, max_tau, N_lin)

    taus = np.unique(np.concatenate([taus_log, taus_lin]))
    taus = [round(tau / 4) * 4 for tau in taus]

    return taus


def generate_log_spaced_taus(min_tau, max_tau, num_steps, base=4):
    taus = np.logspace(np.log10(min_tau), np.log10(max_tau), num_steps)
    taus = np.floor(taus / base) * base
    return taus


def generate_divisible_by_4(min_val, max_val, num_steps):
    step_size = (max_val - min_val) / (num_steps - 1)
    step_size = round(step_size / 4) * 4  # Ensure step size is divisible by 4

    values = [min_val + i * step_size for i in range(num_steps)]

    # Ensure values stay within bounds
    values = [x for x in values if x <= max_val]

    return values


# Power equation constants for yellow aom
a, b, c = 161266.751, 6.617, -19.492


def power_from_voltage(v):
    return a * v**b + c


def voltage_from_power(p):
    return ((p - c) / a) ** (1 / b)


# endregion


# region Camera getters - probably only needed internally


@cache
def _get_camera_spot_radius():
    return _get_camera_config_val("spot_radius")


@cache
def _get_camera_bias_clamp():
    return _get_camera_config_val("bias_clamp")


@cache
def _get_camera_resolution():
    return _get_camera_config_val("resolution")


@cache
def _get_camera_em_gain():
    return _get_camera_config_val("em_gain")


@cache
def _get_camera_k_gain():
    readout_mode = _get_camera_readout_mode()
    camera_server_name = common.get_server_name("camera")
    camera_module = import_module(f"servers.inputs.{camera_server_name}")
    k_gain_dict = camera_module.k_gain_dict
    _camera_k_gain = k_gain_dict[readout_mode]
    return _camera_k_gain


@cache
def _get_camera_readout_mode():
    return _get_camera_config_val("readout_mode")


@cache
def _get_camera_timeout():
    return _get_camera_config_val("timeout")


@cache
def _get_camera_temp():
    return _get_camera_config_val("temp")


@cache
def _get_camera_readout_mode():
    return _get_camera_config_val("readout_mode")


@cache
def _get_camera_roi():
    try:
        return _get_camera_config_val("roi")
    except Exception:
        return None


def _get_camera_config_val(key):
    config = common.get_config_dict()
    return config["Camera"][key]


def _get_img_str_shape():
    resolution = _get_camera_resolution()
    roi = _get_camera_roi()
    if roi is None:
        shape = resolution
    else:
        shape = (roi[3], resolution[0])
    return shape


def get_img_array_shape():
    resolution = _get_camera_resolution()
    roi = _get_camera_roi()
    if roi is None:
        shape = resolution
    else:
        shape = roi[2:]
    return shape


@cache
def get_camera_scale():
    return _get_camera_config_val("scale")


# endregion
# region Plotting


def replace_dead_pixel(img_array):
    dead_pixel = [132, 100]
    dead_pixel_x = dead_pixel[1]
    dead_pixel_y = dead_pixel[0]
    img_array[dead_pixel_y, dead_pixel_x] = np.mean(
        img_array[
            dead_pixel_y - 1 : dead_pixel_y + 1 : 2,
            dead_pixel_x - 1 : dead_pixel_x + 1 : 2,
        ]
    )


def draw_circles_on_nvs(
    ax,
    nv_list=None,
    drift=None,
    pixel_coords_list=None,
    color=None,
    linestyle="solid",
    no_legend=False,
    include_inds=None,
):
    scale = get_camera_scale()
    passed_color = color
    if pixel_coords_list is None:
        pixel_coords_list = []
        for nv in nv_list:
            pixel_coords = pos.get_nv_coords(
                nv, CoordsKey.PIXEL, drift_adjust=True, drift=drift
            )
            pixel_coords_list.append(pixel_coords)
    num_nvs = len(pixel_coords_list)
    points = []
    for ind in range(num_nvs):
        pixel_coords = pixel_coords_list[ind]
        if passed_color is None:
            color = kpl.data_color_cycler[ind]
        else:
            color = passed_color
        point = kpl.draw_circle(
            ax,
            pixel_coords,
            color=color,
            radius=0.6 * scale,
            label=ind,
            linestyle=linestyle,
        )
        points.append(point)
    if not no_legend:
        # ncols = (num_nvs // 5) + (1 if num_nvs % 5 > 0 else 0)
        # ncols = 6
        nrows = 2
        ncols = np.ceil(num_nvs / nrows)
        # ax.legend(loc=kpl.Loc.LOWER_CENTER, ncols=ncols, markerscale=0.9)
        ax.legend(loc=kpl.Loc.UPPER_CENTER, ncols=ncols, markerscale=0.5)
        # ax.legend(loc=kpl.Loc.UPPER_LEFT, ncols=ncols, markerscale=0.5)

    if include_inds is not None:
        for ind in range(num_nvs):
            if ind not in include_inds:
                points[ind].remove()


def plot_raw_data(ax, nv_list, x, ys, yerrs=None, subset_inds=None):
    """Plot multiple data sets (with a common set of x vals) with an offset between
    the sets such that they are easier to interpret. Useful for plotting simultaneous
    data from multiple NVs.

    Parameters
    ----------
    ax : matplotlib axes
        axes to plot on
    nv_list : list(nv_sig)
        List of NV sigs to plot
    x : 1D array
        x values to plot
    ys : 2D array
        y values to plot - first dimension divides the data sets up
    yerrs : 2D array
        y errors to plot
    subset_inds : list
        Specify subset_inds if we want to plot just a specific subset of the NVs in nv_list
    """
    num_nvs = len(nv_list)
    if subset_inds is None:
        nv_inds = range(num_nvs)
    else:
        nv_inds = subset_inds
    for nv_ind in nv_inds:
        # if nv_ind not in [3]:
        #     continue
        # if nv_ind not in [0, 1, 2, 4, 6, 11, 14]:
        #     continue
        yerr = None if yerrs is None else yerrs[nv_ind]
        nv_sig = nv_list[nv_ind]
        # nv_num = get_nv_num(nv_sig)
        nv_num = nv_ind
        num_colors = len(kpl.data_color_cycler)
        color = kpl.data_color_cycler[nv_num % num_colors]
        kpl.plot_points(
            ax,
            x,
            ys[nv_ind],
            yerr=yerr,
            label=str(nv_num),
            size=kpl.Size.SMALL,
            color=color,
        )

        # MCC
        # ax.legend()
        # kpl.show(block=True)
        # fig, ax = plt.subplots()

    # min_x = min(x)
    # max_x = max(x)
    # excess = 0.08 * (max_x - min_x)
    # ax.set_xlim(min_x - excess, max_x + excess)
    ncols = round(num_nvs / 5)
    ax.legend(ncols=ncols)


# Separate plots, shared axes
def plot_fit(
    axes_pack,
    nv_list,
    x,
    ys,
    yerrs=None,
    fns=None,
    popts=None,
    xlim=None,
    norms=None,
    no_legend=False,
    linestyle="none",
    nv_inds=None,
):
    """Plot multiple data sets (with a common set of x vals) with an offset between
    the sets such that they are separated and easier to interpret. Useful for
    plotting simultaneous data from multiple NVs.

    Parameters
    ----------
    ax : matplotlib axes
        axes to plot on, flattened
    nv_list : list(nv_sig)
        List of NV sigs to plot
    x : 1D array
        x values to plot
    ys : 2D array
        y values to plot - first dimension divides the data sets up
    yerrs : 2D array
        y errors to plot
    fns : list(function)
        The ith fn is the fit function used to fit the data for the ith NV
    popts : list(list(numeric))
        The ith popt is the curve fit results for the ith NV
    """
    if isinstance(axes_pack, dict):
        axes_pack = list(axes_pack.values())
    if xlim is None:
        xlim = (min(x), max(x))
    if xlim[0] is None:
        xlim[0] = min(x)
    if xlim[1] is None:
        xlim[1] = max(x)
    x_linspace = np.linspace(*xlim, 1000)
    num_nvs = len(nv_list)
    for nv_ind in range(num_nvs):
        fn = None if fns is None else fns[nv_ind]
        popt = None if popts is None else popts[nv_ind]

        nv_sig = nv_list[nv_ind]
        nv_num = nv_ind
        num_colors = len(kpl.data_color_cycler)
        color = kpl.data_color_cycler[nv_num % num_colors]

        ax = axes_pack[nv_ind]

        # Include the norm if there is one
        y = np.copy(ys[nv_ind])
        yerr = np.copy(yerrs[nv_ind]) if yerrs is not None else None
        if norms is not None:
            norm = norms[nv_ind]
            y /= norm
            yerr /= norm
        # yerr = None  # MCC

        # Plot the points
        # ls = "none" if fn is not None else "solid"
        # size = kpl.Size.SMALL
        # size = kpl.Size.XSMALL
        size = kpl.Size.TINY
        label = str(nv_num)
        kpl.plot_points(
            ax,
            x,
            y,
            yerr=yerr,
            label=label,
            size=size,
            color=color,
            linestyle=linestyle,
        )
        # nv_num = get_nv_num(nv_sig)
        # kpl.anchored_text(ax, nv_inds[nv_ind], size=kpl.Size.TINY)

        # Plot the fit
        if fn is not None:
            # Include the norm if there is one
            fit_vals = fn(x_linspace) if popt is None else fn(x_linspace, *popt)
            if norms is not None:
                fit_vals /= norm
            kpl.plot_line(ax, x_linspace, fit_vals, color=color)
            # kpl.plot_line(ax, x_linspace, fit_vals, color=kpl.lighten_color_hex(color, 0.5))

        if not no_legend:
            loc = kpl.Loc.UPPER_LEFT
            # loc = kpl.Loc.UPPER_LEFT if nv_ind in [0, 1, 4, 6] else "upper center"
            ax.legend(loc=loc)

    for ax in axes_pack:
        ax.spines[["right", "top"]].set_visible(False)

    fig = axes_pack[0].get_figure()
    fig.get_layout_engine().set(h_pad=0, hspace=0, w_pad=0, wspace=0)


def downsample_img_array(img_array, downsample_factor):
    shape = img_array.shape
    downsampled_shape = (
        int(np.floor(shape[0] / downsample_factor)),
        int(np.floor(shape[1] / downsample_factor)),
    )

    # Clip the original img_array so that its dimensions are an integer
    # multiple of downsample_factor
    clip_shape = (
        downsample_factor * downsampled_shape[0],
        downsample_factor * downsampled_shape[1],
    )
    img_array = img_array[: clip_shape[0], : clip_shape[1]]

    downsampled_img_array = np.zeros(downsampled_shape)
    for ind in range(downsample_factor):
        for jnd in range(downsample_factor):
            downsampled_img_array += img_array[
                ind::downsample_factor, jnd::downsample_factor
            ]

    return downsampled_img_array


def smooth_img_array(img_array, downsample_factor):
    shape = img_array.shape
    downsampled_shape = (
        int(np.floor(shape[0] / downsample_factor)),
        int(np.floor(shape[1] / downsample_factor)),
    )

    # Clip the original img_array so that its dimensions are an integer
    # multiple of downsample_factor
    clip_shape = (
        downsample_factor * downsampled_shape[0],
        downsample_factor * downsampled_shape[1],
    )
    img_array = img_array[: clip_shape[0], : clip_shape[1]]

    downsampled_img_array = np.zeros(downsampled_shape)
    for ind in range(downsample_factor):
        for jnd in range(downsample_factor):
            downsampled_img_array += img_array[
                ind::downsample_factor, jnd::downsample_factor
            ]

    return downsampled_img_array


def animate_images(x, img_arrays, cmin=None, cmax=None):
    num_steps = img_arrays.shape[0]
    step_size = x[1] - x[0]
    left_xlim = x[0] - step_size
    right_xlim = x[-1] + step_size
    figsize = [5.0, 5.9]
    fig, axes_pack = plt.subplots(2, 1, height_ratios=(5, 0.2), figsize=figsize)
    im_ax = axes_pack[0]
    prog_bar_ax = axes_pack[1]

    # Set up the actual image
    kpl.imshow(im_ax, np.zeros(img_arrays[0].shape), no_cbar=True)
    # 5 um, downsampled by 2X
    scale = 5 * get_camera_scale() // 2
    # kpl.scale_bar(im_ax, scale, "5 µm", kpl.Loc.LOWER_RIGHT)

    def prog_bar_ax_relim():
        xlabel = "Frequency (GHz)"
        prog_bar_ax.set_xlabel(xlabel)
        prog_bar_ax.set_xlim([left_xlim, right_xlim])
        prog_bar_ax.set_ylim([0, 2])

    def animate_sub(step_ind):
        kpl.imshow_update(im_ax, img_arrays[step_ind], cmin, cmax)
        im_ax.axis("off")

        prog_bar_ax.clear()
        prog_bar_ax.spines["top"].set_visible(False)
        prog_bar_ax.spines["right"].set_visible(False)
        prog_bar_ax.spines["left"].set_visible(False)
        prog_bar_ax.axes.get_yaxis().set_visible(False)
        kpl.plot_points(
            prog_bar_ax,
            [x[step_ind]],
            [1.3],
            color=kpl.KplColors.BLUE,
            size=kpl.Size.BIG,
        )
        prog_bar_ax_relim()
        return axes_pack

    anim = animation.FuncAnimation(
        fig, animate_sub, frames=num_steps, interval=200, blit=False
    )
    timestamp = dm.get_time_stamp()
    anim.save(Path.home() / f"lab/movies/{timestamp}.gif")


def animate_images_and_data(
    x,
    nv_list,
    norm_counts,
    norm_counts_ste,
    img_arrays,
    cmin=None,
    cmax=None,
):
    num_steps = img_arrays.shape[0]

    just_plot_figsize = [6.5, 5.0]
    figsize = [just_plot_figsize[0] + just_plot_figsize[1], just_plot_figsize[1]]
    # fig, axes_pack = plt.subplots(2, 1, height_ratios=(1, 1), figsize=figsize)
    fig = plt.figure(figsize=figsize)
    im_fig, data_fig = fig.subfigures(1, 2, width_ratios=just_plot_figsize[::-1])
    im_ax = im_fig.add_subplot()
    all_axes = [im_ax]
    num_nvs = len(nv_list)
    layout = kpl.calc_mosaic_layout(num_nvs, num_rows=2)
    data_axes = data_fig.subplot_mosaic(layout, sharex=True, sharey=True)
    data_axes_flat = list(data_axes.values())
    rep_data_ax = data_axes[layout[-1, 0]]
    all_axes.extend(data_axes_flat)

    # Set up the actual image
    kpl.imshow(im_ax, np.zeros(img_arrays[0].shape), no_cbar=True)
    # 5 um, downsampled by 2X
    scale = 5 * get_camera_scale() // 2
    kpl.scale_bar(im_ax, scale, "5 µm", kpl.Loc.LOWER_RIGHT)

    def data_ax_relim():
        # Update this manually to match the final plot

        # ESR
        xlabel = "Frequency (GHz)"
        # rep_data_ax.set_xticks([0, 200])
        rep_data_ax.set_xlim([2.771, 2.969])
        rep_data_ax.set_yticks([0, 1])
        rep_data_ax.set_ylim([-0.2698272503744738, 1.2643312422797126])

        # Rabi
        # xlabel = "Pulse duration (ns)"
        # rep_data_ax.set_xticks([0, 200])
        # rep_data_ax.set_xlim([-12.8, 268.8])
        # rep_data_ax.set_yticks([0, 1])
        # rep_data_ax.set_ylim([-0.33, 1.25])

        # Leave fixed
        kpl.set_shared_ax_xlabel(rep_data_ax, xlabel)
        ylabel = "Norm. NV$^{-}$ population"
        kpl.set_shared_ax_ylabel(rep_data_ax, ylabel)

    data_ax_relim()

    def animate_sub(step_ind):
        kpl.imshow_update(im_ax, img_arrays[step_ind], cmin, cmax)
        im_ax.axis("off")

        for ax in data_axes_flat:
            ax.clear()
        plot_fit(
            data_axes_flat,
            nv_list,
            x[: step_ind + 1],
            norm_counts[:, : step_ind + 1],
            norm_counts_ste[:, : step_ind + 1],
            no_legend=True,
        )
        data_ax_relim()

        return all_axes

    anim = animation.FuncAnimation(
        fig, animate_sub, frames=num_steps, interval=200, blit=False
    )
    timestamp = dm.get_time_stamp()
    anim.save(Path.home() / f"lab/movies/{timestamp}.gif")


def plot_correlations(axes_pack, nv_list, x, counts):
    num_nvs = len(nv_list)

    background_nv_ind = 1

    for nv_ind_1 in range(num_nvs):
        if nv_ind_1 == background_nv_ind:
            continue
        for nv_ind_2 in range(num_nvs):
            if nv_ind_2 == background_nv_ind:
                continue

            ax = axes_pack[
                nv_ind_1 if nv_ind_1 < 1 else nv_ind_1 - 1,
                nv_ind_2 if nv_ind_2 < 1 else nv_ind_2 - 1,
            ]

            if nv_ind_1 == nv_ind_2:
                color = kpl.data_color_cycler[nv_ind_1]
                ax.set_facecolor(color)
                continue

            nv_counts_1 = counts[nv_ind_1]
            nv_counts_2 = counts[nv_ind_2]
            corrs = []
            for step_ind in range(len(x)):
                step_counts_1 = nv_counts_1[:, step_ind, :].flatten()
                step_counts_2 = nv_counts_2[:, step_ind, :].flatten()
                corrs.append(np.corrcoef(step_counts_1, step_counts_2)[0, 1])

            # size = kpl.Size.SMALL
            # kpl.plot_points(ax, x, corrs, size=size)


# endregion

# Add this function to your widefield module
def draw_circle_on_nv(
    ax, center, radius, color=kpl.KplColors.RED, linestyle="solid", no_legend=True
):
    """
    Draw a single circle on a detected NV center in the given axis.

    Args:
    ax: Matplotlib axis to draw on.
    center: Tuple (x, y) representing the center coordinates of the NV.
    radius: Radius of the circle to be drawn.
    color: Color of the circle (default: RED).
    linestyle: Line style of the circle (default: solid).
    no_legend: Boolean to indicate whether to add the legend or not (default: True).
    """
    # Create a circle patch
    circle = patches.Circle(
        center, radius, edgecolor=color, facecolor="none", linestyle=linestyle
    )

    # Add the circle to the axis
    ax.add_patch(circle)

    # Optionally handle the legend, here no_legend=True means no legend
    if not no_legend:
        ax.legend()

# region AOD AMP 

# ----------------------------
# Empirical calibration data
# ----------------------------
GREEN_AMP = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16], dtype=float)
GREEN_PWR = np.array([12, 162, 752, 2170, 4520, 7660, 11500, 15400], dtype=float)

RED_AMP = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16], dtype=float)
RED_PWR = np.array([24, 303, 1430, 4160, 9000, 16200, 25000, 34200], dtype=float)

GREEN_FREQ = np.array([90, 95, 100, 105, 110, 115, 120, 125], dtype=float)
GREEN_FREQ_PWR = np.array([260, 310, 330, 350, 460, 340, 240, 140], dtype=float)

RED_FREQ = np.array([55, 60, 65, 70, 75, 80, 85, 90], dtype=float)
RED_FREQ_PWR = np.array([112, 200, 255, 260, 270, 260, 205, 110], dtype=float)


def _interp_clipped(x, xp, fp):
    """
    1D interpolation with clipping at the calibration range.
    Works with scalar or array x.
    """
    x = np.asarray(x, dtype=float)
    x_clip = np.clip(x, xp[0], xp[-1])
    return np.interp(x_clip, xp, fp)


def make_aod_amp_fn_2d_same_curve(
    amp_pts,
    pwr_pts,
    freq_pts,
    freq_pwr_pts,
    ref_freq,
    min_amp=None,
    max_amp=None,
    min_scale=0.5,
    max_scale=2.0,
):
    """
    Return a function final_amp(coords, base_amp).

    coords can be:
        scalar frequency
        [fx, fy]

    Model:
        P(amp, fx, fy) ≈ g(amp) * h(fx) * h(fy)

    This assumes the two AOD axes have the same frequency-efficiency curve h(f).
    If you later measure x-axis and y-axis separately, use separate h_x and h_y.

    Returns:
        final amplitude, not scale factor.
    """
    amp_pts = np.asarray(amp_pts, dtype=float)
    pwr_pts = np.asarray(pwr_pts, dtype=float)
    freq_pts = np.asarray(freq_pts, dtype=float)
    freq_pwr_pts = np.asarray(freq_pwr_pts, dtype=float)

    if min_amp is None:
        min_amp = float(amp_pts[0])

    if max_amp is None:
        max_amp = float(amp_pts[-1])

    ref_freq_pwr = _interp_clipped(ref_freq, freq_pts, freq_pwr_pts)
    rel_eff_pts = freq_pwr_pts / ref_freq_pwr

    def final_amp(coords, base_amp):
        coords = np.asarray(coords, dtype=float).ravel()
        base_amp = float(base_amp)

        if len(coords) == 1:
            fx = float(coords[0])
            fy = float(ref_freq)
        else:
            fx = float(coords[0])
            fy = float(coords[1])

        # Target power at reference frequency for the chosen base amplitude.
        target_pwr = _interp_clipped(base_amp, amp_pts, pwr_pts)

        # Relative AOD efficiency from both axes.
        rel_eff_x = _interp_clipped(fx, freq_pts, rel_eff_pts)
        rel_eff_y = _interp_clipped(fy, freq_pts, rel_eff_pts)

        rel_eff = rel_eff_x * rel_eff_y
        rel_eff = max(float(rel_eff), 1e-9)

        # Required power from amplitude curve.
        needed_pwr = target_pwr / rel_eff
        needed_pwr = np.clip(needed_pwr, pwr_pts[0], pwr_pts[-1])

        # Invert power -> amplitude.
        needed_amp = _interp_clipped(needed_pwr, pwr_pts, amp_pts)

        # Also limit by allowed scale relative to base_amp.
        min_allowed_amp = base_amp * min_scale
        max_allowed_amp = base_amp * max_scale

        needed_amp = np.clip(
            needed_amp,
            max(min_amp, min_allowed_amp),
            min(max_amp, max_allowed_amp),
        )

        return float(needed_amp)

    return final_amp


# -------------------------------------------
# Build 2D compensators
# -------------------------------------------
green_amp_fn_2d = make_aod_amp_fn_2d_same_curve(
    GREEN_AMP,
    GREEN_PWR,
    GREEN_FREQ,
    GREEN_FREQ_PWR,
    ref_freq=110.0,
    min_scale=0.5,
    max_scale=2.0,
)

red_amp_fn_2d = make_aod_amp_fn_2d_same_curve(
    RED_AMP,
    RED_PWR,
    RED_FREQ,
    RED_FREQ_PWR,
    ref_freq=75.0,
    min_scale=0.5,
    max_scale=2.0,
)


def get_freq_from_coords(coords, freq_index=0):
    """
    Extract the frequency coordinate used for this calibration.
    If coords is scalar, returns it directly.
    If coords is [fx, fy], pick freq_index.
    """
    arr = np.asarray(coords, dtype=float).ravel()
    if len(arr) == 1:
        return float(arr[0])
    return float(arr[freq_index])


def fit_power_law_amp_curve(amp_pts, pwr_pts):
    """
    Fit P = C * amp^gamma using log-log fit.
    """
    amp_pts = np.asarray(amp_pts, dtype=float)
    pwr_pts = np.asarray(pwr_pts, dtype=float)

    good = (amp_pts > 0) & (pwr_pts > 0)

    log_amp = np.log(amp_pts[good])
    log_pwr = np.log(pwr_pts[good])

    gamma, log_C = np.polyfit(log_amp, log_pwr, 1)
    C = np.exp(log_C)

    print("Power-law fit:")
    print("C =", C)
    print("gamma =", gamma)

    return C, gamma


def power_from_amp_powerlaw(amp, C, gamma):
    amp = np.asarray(amp, dtype=float)
    return C * amp**gamma


def amp_from_power_powerlaw(power, C, gamma):
    power = np.asarray(power, dtype=float)
    power = np.maximum(power, 1e-12)
    return (power / C) ** (1.0 / gamma)


def make_aod_amp_fn_2d_powerlaw(
    amp_pts,
    pwr_pts,
    freq_pts,
    freq_pwr_pts,
    ref_freq,
    min_amp=0.02,
    max_amp=0.16,
    min_scale=0.5,
    max_scale=2.0,
):
    """
    Return final_amp(coords, base_amp).

    Model:
        P(amp, fx, fy) = C * amp^gamma * h(fx) * h(fy)

    Assumes both AOD axes have the same frequency-efficiency curve h(f).
    """
    C, gamma = fit_power_law_amp_curve(amp_pts, pwr_pts)

    ref_freq_pwr = _interp_clipped(ref_freq, freq_pts, freq_pwr_pts)
    rel_eff_pts = freq_pwr_pts / ref_freq_pwr

    def final_amp(coords, base_amp):
        coords = np.asarray(coords, dtype=float).ravel()
        base_amp = float(base_amp)

        if len(coords) == 1:
            fx = float(coords[0])
            fy = float(ref_freq)
        else:
            fx = float(coords[0])
            fy = float(coords[1])

        # Target power at reference frequency using power-law amp calibration.
        target_pwr = power_from_amp_powerlaw(base_amp, C, gamma)

        # Frequency efficiency correction.
        rel_eff_x = _interp_clipped(fx, freq_pts, rel_eff_pts)
        rel_eff_y = _interp_clipped(fy, freq_pts, rel_eff_pts)

        rel_eff = float(rel_eff_x * rel_eff_y)
        rel_eff = max(rel_eff, 1e-9)

        # Required amplitude power before frequency loss.
        needed_pwr = target_pwr / rel_eff

        # Invert power-law.
        needed_amp = amp_from_power_powerlaw(needed_pwr, C, gamma)

        # Safety limits.
        needed_amp = np.clip(
            needed_amp,
            max(min_amp, base_amp * min_scale),
            min(max_amp, base_amp * max_scale),
        )

        return float(needed_amp)

    return final_amp



def plot_amp_distribution_over_space(
    pixel_coords_list,
    charge_pol_amps,
    scc_amp_list,
    figsize=(7, 6),
    marker_size=28,
):
    pixel_coords = np.asarray(pixel_coords_list, dtype=float)
    x = pixel_coords[:, 0]
    y = pixel_coords[:, 1]

    charge_pol_amps = np.asarray(charge_pol_amps, dtype=float)
    scc_amp_list = np.asarray(scc_amp_list, dtype=float)

    # --------------------------------------------------
    # Plot 1: Green amplitude over NV space
    # --------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=figsize)
    sc1 = ax1.scatter(
        x,
        y,
        c=charge_pol_amps,
        s=marker_size,
        cmap="viridis",
    )
    ax1.set_title("Green charge-polarization amplitude over NV space")
    ax1.set_xlabel("Pixel x")
    ax1.set_ylabel("Pixel y")
    ax1.invert_yaxis()   # image-style coordinates
    cbar1 = fig1.colorbar(sc1, ax=ax1)
    cbar1.set_label("Green amplitude")

    # --------------------------------------------------
    # Plot 2: Red amplitude over NV space
    # --------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=figsize)
    sc2 = ax2.scatter(
        x,
        y,
        c=scc_amp_list,
        s=marker_size,
        cmap="plasma",
    )
    ax2.set_title("Red SCC amplitude over NV space")
    ax2.set_xlabel("Pixel x")
    ax2.set_ylabel("Pixel y")
    ax2.invert_yaxis()
    cbar2 = fig2.colorbar(sc2, ax=ax2)
    cbar2.set_label("Red amplitude")

    print("Green amplitude range:", np.min(charge_pol_amps), "to", np.max(charge_pol_amps))
    print("Red amplitude range:", np.min(scc_amp_list), "to", np.max(scc_amp_list))

    return fig1, fig2


def plot_aod_freqs_over_space(
    pixel_coords_list,
    green_coords_list,
    red_coords_list,
    marker_size=20,
):
    pixel_arr = np.asarray(pixel_coords_list, dtype=float)
    green_arr = np.asarray(green_coords_list, dtype=float)
    red_arr = np.asarray(red_coords_list, dtype=float)

    x = pixel_arr[:, 0]
    y = pixel_arr[:, 1]

    plots = [
        (green_arr[:, 0], "Green AOD fx over NV space", "Green fx [MHz]"),
        (green_arr[:, 1], "Green AOD fy over NV space", "Green fy [MHz]"),
        (red_arr[:, 0], "Red AOD fx over NV space", "Red fx [MHz]"),
        (red_arr[:, 1], "Red AOD fy over NV space", "Red fy [MHz]"),
    ]

    figs = []

    for values, title, cbar_label in plots:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        sc = ax.scatter(
            x,
            y,
            c=values,
            s=marker_size,
        )

        ax.set_title(title)
        ax.set_xlabel("Pixel x")
        ax.set_ylabel("Pixel y")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

        figs.append(fig)

    return figs

# endregion

if __name__ == "__main__":
    print(adus_to_photons(700))
    sys.exit()
    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    kpl.imshow(ax, _img_array_iris((250, 512)))
    kpl.show(block=True)
