import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from skimage.draw import disk
from skimage.feature import blob_log
from skimage.filters import gaussian

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


# # Fit a 2D Gaussian to a local region of the image data and return FWHM
# def fit_gaussian_2d_local(image, center, size=10):
#     x0, y0 = center
#     x_min, x_max = int(x0 - size), int(x0 + size)
#     y_min, y_max = int(y0 - size), int(y0 + size)

#     x_min = max(x_min, 0)
#     x_max = min(x_max, image.shape[1])
#     y_min = max(y_min, 0)
#     y_max = min(y_max, image.shape[0])

#     local_image = image[y_min:y_max, x_min:x_max]

#     x = np.arange(x_min, x_max)
#     y = np.arange(y_min, y_max)
#     x, y = np.meshgrid(x, y)

#     initial_guess = (
#         local_image.max(),
#         x0,
#         y0,
#         1,
#         1,
#         0,
#         np.min(local_image),
#     )

#     try:
#         popt, _ = curve_fit(gaussian_2d, (x, y), local_image.ravel(), p0=initial_guess)
#         amplitude, xo, yo, sigma_x, sigma_y, theta, offset = popt

#         # Calculate FWHM in pixels for both x and y directions
#         fwhm_x = 2.355 * sigma_x  # FWHM = 2.355 * sigma
#         fwhm_y = 2.355 * sigma_y

#         return (round(xo, 3), round(yo, 3)), (fwhm_x, fwhm_y), popt
#     except RuntimeError:
#         return center, (None, None), None


def fit_gaussian_2d_local(image, center, size=12, maxfev=20000):
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
    smoothing_sigma=1,
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
            fit_size = max(8, int(2 * integration_radius))   # good default
            # or: fit_size = max(10, int(2.5 * sigma))

            optimized_coord, fwhm, _ = fit_gaussian_2d_local(smoothed_img, (x, y), size=fit_size)

            # Perform Gaussian fitting and get the FWHM
            # optimized_coord, fwhm, _ = fit_gaussian_2d_local(img_array, (x, y), size=3)
            optimized_coords.append(optimized_coord)
            spot_sizes.append(fwhm)  # Append the FWHM for the spot
            integrated_counts.append(integrated_intensity)

    valid_blobs = np.array(valid_blobs)
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
            x, y - r - 2, f"{idx}", color="black", fontsize=8, ha="center", va="center"
        )

    # kpl.show(block=True)

    return optimized_coords, integrated_counts, spot_sizes


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
        img_array = (img_array - 300) / max(1, np.median(img_array))
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
    data = dm.get_raw_data(
        file_stem="2026_03_05-08_55_37-combined_image_array", load_npz=True
    )
    # img_array = np.array(data["ref_img_array"])
    img_array = np.array(data["img_array"])

    # Apply the blob detection and Gaussian fitting
    sigma = 2.0
    lower_threshold = 0.01
    upper_threshold = None
    smoothing_sigma = 0.0
    integration_radius= 2
    nv_coordinates, integrated_counts, spot_sizes = detect_nv_coordinates_blob(
        img_array,
        sigma=sigma,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        smoothing_sigma=smoothing_sigma,
        integration_radius=integration_radius,
    )
    filtered_nv_coords = nv_coordinates
    filtered_counts = integrated_counts
    # # List to store valid NV coordinates after filtering
    # filtered_nv_coords = []
    # filtered_counts = []
    # # Iterate through detected NV coordinates and apply distance filtering
    # for coord, count in zip(nv_coordinates, integrated_counts):
    #     # Assume the coordinate is valid initially
    #     keep_coord = True

    #     # Check distance with all previously accepted NVs
    #     for existing_coord in filtered_nv_coords:
    #         distance = np.linalg.norm(np.array(existing_coord) - np.array(coord))

    #         if distance < 1:
    #             keep_coord = False  # Mark it for exclusion if too close
    #             break  # No need to check further distances

    #     # If the coordinate passes the distance check, add it to the list
    #     if keep_coord:
    #         filtered_nv_coords.append(coord)
    #         filtered_counts.append(count)

    # print(f"Number of NVs detected: {len(filtered_nv_coords)}")
    # for idx, (coord, count) in enumerate(
    #     zip(filtered_nv_coords, filtered_counts), start=1
    # ):
    #     print(f"NV {idx}: {coord}, {count}:.2f")
    # # Plotting the results
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

    kpl.show(block=True)

    print(f"Detected NV coordinates (optimized): {len(filtered_nv_coords)}")

    # Save the results
    # save_results(
    #     filtered_nv_coords,
    #     filtered_counts,
    #     path="slmsuite/nv_blob_detection",
    #     filename="nv_blob_237nvs.npz",
    # )

    # full ROI -- multiple images save in the same file
    process_scan_file(file_stem="2026_03_04-23_42_24-qnami-nv0_2026_02_20")
    # process_scan_file(file_stem="2026_03_05-09_46_19-qnami-nv0_2026_02_20")
