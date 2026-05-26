import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.filters import gaussian
from utils import data_manager as dm
from utils import kplotlib as kpl
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max


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

def fit_gaussian_2d(image, center, size=12, maxfev=20000):
    """
    size = half-size of the patch in pixels (so patch is ~2*size x 2*size)
    """
    x0, y0 = center  # x, y in global coords (float ok)

    x_min, x_max = int(np.floor(x0 - size)), int(np.ceil(x0 + size))
    y_min, y_max = int(np.floor(y0 - size)), int(np.ceil(y0 + size))

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])

    patch = image[y_min:y_max, x_min:x_max].astype(float)
    if patch.size == 0:
        return (x0, y0), (None, None), None

    # Use local coordinate system for numerical stability
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]

    # Better initial guess: use brightest pixel in patch
    iy, ix = np.unravel_index(np.argmax(patch), patch.shape)

    offset0 = np.percentile(patch, 20)               # robust-ish background
    amp0 = max(patch[iy, ix] - offset0, 1.0)

    # initial sigmas: a couple pixels (tune if needed)
    sigma_x0 = 2.0
    sigma_y0 = 2.0
    theta0 = 0.0

    p0 = (amp0, ix, iy, sigma_x0, sigma_y0, theta0, offset0)

    # Bounds: keep center inside patch; keep sigmas positive and reasonable; limit theta
    eps = 1e-6
    lower = (0.0, 0.0, 0.0, 0.5, 0.5, -np.pi/4, np.min(patch) - abs(amp0))
    upper = (np.inf, patch.shape[1]-1 + eps, patch.shape[0]-1 + eps,
             float(size), float(size), np.pi/4, np.max(patch) + abs(amp0))

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (xx, yy),
            patch.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
        amp, xo_l, yo_l, sx, sy, theta, offset = popt

        # Convert local fitted center back to global coordinates
        xo_g = x_min + xo_l
        yo_g = y_min + yo_l

        fwhm_x = 2.355 * sx
        fwhm_y = 2.355 * sy

        return (round(xo_g, 3), round(yo_g, 3)), (fwhm_x, fwhm_y), popt

    except Exception:
        return (x0, y0), (None, None), None

# Apply the blob detection algorithm and estimate spot size in pixels
def detect_nv_coordinates_blob(
    img_array,
    sigma=2.0,
    lower_threshold=15.0,
    upper_threshold=None,
    smoothing_sigma=0,
    integration_radius=2,
):
    smoothed_img = gaussian(img_array, sigma=smoothing_sigma)

    blobs = blob_log(
        smoothed_img,
        min_sigma=sigma,
        max_sigma=sigma,
        num_sigma=1,
        threshold=lower_threshold,
    )

    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)

    valid_blobs = []
    optimized_coords = []
    spot_sizes = []  # List to store FWHM sizes for each spot
    integrated_counts = []
    for blob in blobs:
        y, x, r = blob
        rr, cc = disk((y, x), integration_radius, shape=img_array.shape)
        integrated_intensity = np.sum(smoothed_img[rr, cc])

        if integrated_intensity >= lower_threshold and (
            upper_threshold is None or integrated_intensity <= upper_threshold
        ):
            valid_blobs.append(blob)
            fit_size = max(2, int(integration_radius))   # good default
            # or: fit_size = max(10, int(2.5 * sigma))

            optimized_coord, fwhm, _ = fit_gaussian_2d(smoothed_img, (x, y), size=fit_size)
            optimized_coord = (x,y)
            # Perform Gaussian fitting and get the FWHM
            optimized_coords.append(optimized_coord)
            spot_sizes.append(fwhm)  # Append the FWHM for the spot
            integrated_counts.append(integrated_intensity)

    valid_blobs = np.array(valid_blobs)
    print(valid_blobs)
    optimized_coords = np.array(optimized_coords)

    fig, ax = plt.subplots()
    title = "24ms, Ref"
    cax = kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # cax = ax.imshow(img_array, cmap="hot")
    ax.set_title("NV Detection with Blob")
    ax.axis("off")

    # fig.colorbar(cax, ax=ax, orientation="vertical", label="Intensity")

    for idx, blob in enumerate(valid_blobs, start=1):
        y, x, r = blob
        circ = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
        ax.add_patch(circ)

        ax.text(
            x, y - r - 1, f"{idx}", color="black", fontsize=8, ha="center", va="center"
        )

    # kpl.show(block=True)

    return optimized_coords, integrated_counts, spot_sizes


def local_hex_metrics(points_xy, d_min=4.5, d_max=8.5):
    """
    For each point:
      - count neighbors in the first lattice shell
      - compute hexagonal bond-order parameter |psi6|

    For a good interior hex point:
      - nn_count is usually ~3-6
      - |psi6| is relatively high
    """
    if len(points_xy) == 0:
        return np.array([]), np.array([])

    tree = cKDTree(points_xy)
    nn_count = np.zeros(len(points_xy), dtype=int)
    psi6 = np.zeros(len(points_xy), dtype=float)

    for i, p in enumerate(points_xy):
        idx = tree.query_ball_point(p, r=d_max + 0.25)
        idx = [j for j in idx if j != i]
        if len(idx) == 0:
            continue

        vec = points_xy[idx] - p
        dist = np.linalg.norm(vec, axis=1)

        keep = (dist >= d_min) & (dist <= d_max)
        vec = vec[keep]

        nn_count[i] = len(vec)

        if len(vec) >= 2:
            ang = np.arctan2(vec[:, 1], vec[:, 0])
            psi6[i] = np.abs(np.mean(np.exp(1j * 6 * ang)))
        else:
            psi6[i] = 0.0

    return nn_count, psi6


def detect_nv_coordinates_hex(
    img_array,
    # detection
    dog_sigma_small=0.9,
    dog_sigma_large=3.2,
    min_distance=3,
    peak_threshold_abs=0.35,
    # Gaussian refinement
    fit_size=8,
    # local lattice filter
    d_min=4.5,
    d_max=8.5,
    min_neighbors=2,
    psi6_min=0.30,
    # broad-wall / boundary rejection
    wall_sigma=6.0,
    wall_percentile=93,
    wall_dilate_iters=3,
    # optional extra filters
    min_amp=0.0,
    max_sigma_px=4.0,
    show_debug=True,
):
    """
    Better detector for dense hex-like NV arrays with boundaries.

    Strategy:
      1) bandpass image with DoG
      2) liberal peak detection (includes weak spots)
      3) Gaussian-fit each candidate
      4) reject broad bright boundary-wall regions
      5) keep only candidates that live in a locally hex-like neighborhood
    """

    img = img_array.astype(float)

    # --------------------------------------------------------
    # 1) bandpass / DoG image
    # --------------------------------------------------------
    img_small = gaussian_filter(img, dog_sigma_small)
    img_large = gaussian_filter(img, dog_sigma_large)
    dog = img_small - img_large

    # --------------------------------------------------------
    # 2) broad-wall / boundary mask
    #    These bright thick walls are NOT the small NV spots.
    # --------------------------------------------------------
    low = gaussian_filter(img, wall_sigma)
    wall_thresh = np.percentile(low, wall_percentile)
    wall_mask = low > wall_thresh
    wall_mask = binary_dilation(wall_mask, iterations=wall_dilate_iters)

    # --------------------------------------------------------
    # 3) liberal peak detection
    #    Much better than blob_log here for your dense lattice.
    # --------------------------------------------------------
    peak_rc = peak_local_max(
        dog,
        min_distance=min_distance,
        threshold_abs=peak_threshold_abs,
        exclude_border=False,
    )

    # --------------------------------------------------------
    # 4) Gaussian refinement + basic fit quality info
    # --------------------------------------------------------
    refined_xy = []
    amp_list = []
    bg_list = []
    sigma_list = []
    raw_rc_keep = []

    for r, c in peak_rc:
        # reject immediately if candidate lies on broad wall
        if wall_mask[r, c]:
            continue

        optimized_coord, fwhm, popt = fit_gaussian_2d(img, (c, r), size=fit_size)

        if popt is None:
            continue

        amp, xo_l, yo_l, sx, sy, theta, offset = popt
        sigma_mean = 0.5 * (sx + sy)

        # simple fit sanity cuts
        if amp < min_amp:
            continue
        if sigma_mean <= 0 or sigma_mean > max_sigma_px:
            continue

        refined_xy.append(optimized_coord)
        amp_list.append(amp)
        bg_list.append(offset)
        sigma_list.append(sigma_mean)
        raw_rc_keep.append((r, c))

    if len(refined_xy) == 0:
        return np.empty((0, 2)), [], [], {}

    refined_xy = np.array(refined_xy, dtype=float)
    amp_list = np.array(amp_list, dtype=float)
    bg_list = np.array(bg_list, dtype=float)
    sigma_list = np.array(sigma_list, dtype=float)
    raw_rc_keep = np.array(raw_rc_keep, dtype=int)

    # --------------------------------------------------------
    # 5) local hex-order filter
    #    This is what lets weak in-between spots survive,
    #    while noise / wall / boundary junk gets removed.
    # --------------------------------------------------------
    nn_count, psi6 = local_hex_metrics(refined_xy, d_min=d_min, d_max=d_max)

    keep = (
        (nn_count >= min_neighbors) &
        (psi6 >= psi6_min)
    )

    final_xy = refined_xy[keep]
    final_amp = amp_list[keep]
    final_sigma = sigma_list[keep]

    # --------------------------------------------------------
    # 6) integrated counts on original image
    # --------------------------------------------------------
    integrated_counts = []
    spot_sizes = []

    integration_radius = 2
    for x, y in final_xy:
        rr, cc = disk((y, x), integration_radius, shape=img.shape)
        integrated_counts.append(np.sum(img[rr, cc]))
        # convert sigma_mean -> approximate FWHM if desired
        spot_sizes.append((2.355 * final_sigma[len(spot_sizes)],
                           2.355 * final_sigma[len(spot_sizes)]))

    integrated_counts = np.array(integrated_counts, dtype=float)
    spot_sizes = np.array(spot_sizes, dtype=float)

    debug = {
        "dog": dog,
        "low": low,
        "wall_mask": wall_mask,
        "all_peak_rc": peak_rc,
        "refined_xy_all": refined_xy,
        "nn_count_all": nn_count,
        "psi6_all": psi6,
        "keep_mask_after_hex": keep,
    }

    # --------------------------------------------------------
    # 7) debug plot
    # --------------------------------------------------------
    if show_debug:
        fig, ax = plt.subplots(figsize=(7, 7))
        kpl.imshow(ax, img, title="NV detection: hex-aware", cbar_label="Photons")
        ax.set_title("Hex-aware NV detection")
        ax.axis("off")

        # show rejected refined points faintly
        rejected_xy = refined_xy[~keep]
        if len(rejected_xy) > 0:
            ax.scatter(
                rejected_xy[:, 0], rejected_xy[:, 1],
                s=12, facecolors="none", edgecolors="cyan", linewidths=0.6, alpha=0.35
            )

        # show final kept points strongly
        if len(final_xy) > 0:
            ax.scatter(
                final_xy[:, 0], final_xy[:, 1],
                s=20, facecolors="none", edgecolors="lime", linewidths=0.9
            )

        # optional wall overlay
        yy, xx = np.where(wall_mask)
        ax.scatter(xx, yy, s=1, c="magenta", alpha=0.04)

    return final_xy, integrated_counts, spot_sizes, debug

# Save the results to a file
def save_results(
    nv_coordinates,
    integrated_counts,
    path="slmsuite/nv_blob_detection",
    filename="nv_detection_results_with_gaussian_fit.npz",
):
    if not os.path.exists(path):
        os.makedirs(path)

    full_filepath = os.path.join(path, filename)

    np.savez(
        full_filepath,
        nv_coordinates=nv_coordinates,
        updated_spot_weights=integrated_counts,
    )


# Calculate the diffraction-limited resolution
def calculate_resolution(wavelength, NA):
    resolution = (0.61 * wavelength) / NA
    return resolution  # in micrometers


# Estimate pixel-to-µm conversion factor using FWHM
def pixel_to_um_conversion_factor(avg_fwhm, resolution):
    # Use the average FWHM to estimate the conversion factor
    conversion_factor = resolution / avg_fwhm  # µm per pixel
    return conversion_factor


# Function to calculate the Euclidean distance between two coordinates
def euclidean_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


# Function to remove duplicates based on the Euclidean distance threshold
def remove_duplicates(coords, threshold=3):
    unique_coords = []
    for coord in coords:
        if not any(
            euclidean_distance(coord, unique_coord) < threshold
            for unique_coord in unique_coords
        ):
            unique_coords.append(coord)
    return unique_coords


# Process multiple images and remove duplicate NV coordinates
def process_multiple_images(
    image_ids, sigma=2.0, lower_threshold=30.0, upper_threshold=None, smoothing_sigma=1
):
    all_nv_coordinates = []
    all_spot_sizes = []

    for image_id in image_ids:
        print(f"Processing image ID: {image_id}")
        data = dm.get_raw_data(file_id=image_id, load_npz=True)
        img_array = np.array(data["img_array"])  # Load image data

        nv_coordinates, spot_sizes, *_ = detect_nv_coordinates_blob(
            img_array,
            sigma=sigma,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            smoothing_sigma=smoothing_sigma,
        )

        # Append new coordinates and spot sizes
        all_nv_coordinates.extend(nv_coordinates)
        all_spot_sizes.extend(spot_sizes)

    # Remove duplicates based on Euclidean distance
    unique_nv_coordinates = remove_duplicates(all_nv_coordinates, threshold=3)
    print(
        f"Total unique NV coordinates after removing duplicates: {len(unique_nv_coordinates)}"
    )

    return unique_nv_coordinates, all_spot_sizes


def reorder_coords(nv_coords):
    # Calculate Euclidean distances from the first NV coordinate
    distances = [
        np.linalg.norm(np.array(coord) - np.array(nv_coords[0])) for coord in nv_coords
    ]
    # Get sorted indices based on distances
    sorted_indices = np.argsort(distances)
    # Reorder NV coordinates based on sorted distances
    reordered_coords = [nv_coords[idx] for idx in sorted_indices]
    return reordered_coords


def process_scan_file(file_stem):
    """Processes a saved scan file, extracts NV coordinates from each scan entry,
    and creates a combined image using max projection.
    """
    raw_data = dm.get_raw_data(file_stem=file_stem, load_npz=True, allow_pickle=True)

    # Extract scanned data
    scanned_data = raw_data["scanned_data"]
    # Preallocate a list for image processing
    blob_coords, spot_weights, img_arrays = [], [], []

    for index, scan in enumerate(scanned_data):
        img_array = np.array(scan["scan_data"], dtype=np.float64)

        # Detect NVs
        # optimized_coords, integrated_counts, _ = detect_nv_coordinates_blob(
        #     img_array,
        #     lower_threshold=11,
        # )

        # Only store detected NVs if not empty
        # if optimized_coords.size > 0:
        #     blob_coords.extend(optimized_coords)  # Ensure list format
        #     spot_weights.extend(integrated_counts)

        # Normalize image
        img_array = (img_array) / max(1, np.median(img_array))
        # img_array = widefie
        # Store processed image
        img_arrays.append(img_array)

    # Convert to NumPy array if images exist
    if img_arrays:
        img_arrays = np.array(img_arrays)
        combined_img = np.max(img_arrays, axis=0)  # Maximum intensity projection

        # Plot final image
        fig, ax = plt.subplots()
        kpl.imshow(
            ax, combined_img, title="Max_Int_Proj_laser_INTI_520", cbar_label="Photons"
        )
        ax.axis("off")
        print(f"Final detected NV count: {len(blob_coords)}")
    else:
        print("No valid images found.")

    # **Save the results (uncomment if needed)**
    # save_results(
    #     blob_coords,
    #     spot_weights,
    #     path="slmsuite/nv_blob_detection",
    #     filename=f"nv_blob_{len(blob_coords)}nvs.npz",
    # )

    timestamp = dm.get_time_stamp()
    data = {
        "timestamp": timestamp,
        "img_array": combined_img,
    }

    file_path = dm.get_file_path(__file__, timestamp, "combined_image_array")
    dm.save_raw_data(data, file_path, keys_to_compress=["img_array"])
    kpl.show(block=True)


# Plot NV detection results
def plot_nv_detection(img_array, nv_coords):
    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array, title="NV Detection", cbar_label="Photons")

    for x, y in nv_coords:
        circ = plt.Circle((x, y), 2.4, color="red", linewidth=1, fill=False)
        ax.add_patch(circ)
        ax.text(x, y - 3, f"{x:.1f}, {y:.1f}", color="white", fontsize=8, ha="center")

    kpl.show(block=True)

# Main section of the code
if __name__ == "__main__":
    kpl.init_kplotlib()
    # Load the image data
    # data = dm.get_raw_data(
    #     file_stem="2026_03_20-18_05_10-qnami-nv0_2026_02_20", load_npz=True
    # )
    # img_array = np.array(data["ref_img_array"])
    # img_array = np.array(data["img_array"])
    
    # Apply the blob detection and Gaussian fitting
    # sigma = 1.0
    # lower_threshold = 0.1
    # upper_threshold = None
    # smoothing_sigma = 0.0
    # integration_radius= 2
    # nv_coordinates, integrated_counts, spot_sizes = detect_nv_coordinates_blob(
    #     img_array,
    #     sigma=sigma,
    #     lower_threshold=lower_threshold,
    #     upper_threshold=upper_threshold,
    #     smoothing_sigma=smoothing_sigma,
    #     integration_radius=integration_radius,
    # )
    # filtered_nv_coords = nv_coordinates
    # filtered_counts = integrated_counts
    # # Verify if reversing coordinates resolves the offset
    # default_radius = 2
    # fig, ax = plt.subplots()
    # title = "24ms, Ref"
    # cax = kpl.imshow(ax, img_array, title=title, cbar_label="Photons")
    # ax.set_title("NV Detection with Blob")
    # ax.axis("off")

    # for idx, (x, y) in enumerate(filtered_nv_coords, start=1):  # Swapped y, x to x, y
    #     circ = plt.Circle((x, y), default_radius, color="red", linewidth=1, fill=False)
    #     ax.add_patch(circ)
    #     ax.text(
    #         x,
    #         y - default_radius - 1,
    #         f"{idx}",
    #         # color="black",
    #         fontsize=8,
    #         ha="center",
    #         va="center",
    #     )

    # print(f"Detected NV coordinates (optimized): {len(filtered_nv_coords)}")

    # Save the results
    # save_results(
    #     filtered_nv_coords,
    #     filtered_counts,
    #     path="slmsuite/nv_blob_detection",
    #     filename="nv_blob_1487nvs.npz",
    # )

    # full ROI -- multiple images save in the same file
    # process_scan_file(file_stem="2026_03_13-00_01_52-qnami-nv0_2026_02_20")
    process_scan_file(file_stem="2026_05_26-07_18_16-qnami-nv0_2026_02_20")
    
    kpl.show(block=True)

