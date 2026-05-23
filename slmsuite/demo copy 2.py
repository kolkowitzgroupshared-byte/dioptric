# -*- coding: utf-8 -*-
"""
Update NV coordinates after sample/sample-mount rotation.

This script:
    1. Loads before/after widefield images.
    2. Estimates rotation angle using ECC image registration.
    3. Rotates old NV coordinates around a fixed reference NV.
    4. Generates BOTH +theta and -theta coordinate files.
    5. Saves a chosen coordinate file for later use.

Why both signs?
---------------
OpenCV/ECC image-warp convention can be confusing. If the updated coordinates
look rotated opposite to the actual image, use the opposite sign.

Reference-NV model:
-------------------
If the reference NV has the same coordinate before and after, then all other
coordinates should be updated by:

    p_new = p_ref + R(theta) @ (p_old - p_ref)

No translation is applied.
"""

from pathlib import Path
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# Inputs
# =============================================================================

before_file = "2026_05_21-20_51_21-combined_image_array"
after_file = "2026_05_23-13_25_26-combined_image_array"

old_npz_path = Path("slmsuite/nv_blob_detection/nv_blob_1275nvs_reordered.npz")

# The chosen output file used by your AOD-axis script.
chosen_npz_path = Path(
    "slmsuite/nv_blob_detection/nv_blob_1275nvs_reordered_after_sample_rotation.npz"
)

# Also save both sign candidates.
plus_npz_path = Path(
    "slmsuite/nv_blob_detection/nv_blob_1275nvs_reordered_after_sample_rotation_PLUS.npz"
)
minus_npz_path = Path(
    "slmsuite/nv_blob_detection/nv_blob_1275nvs_reordered_after_sample_rotation_MINUS.npz"
)

# Reference NV that stayed fixed before/after.
ref_nv = np.array([199.6929931640625, 201.93699645996094], dtype=np.float32)

# Because your previous update looked opposite, default to "minus".
# Options:
#     "plus"
#     "minus"
CHOSEN_SIGN = "minus"

# Optional manual override.
# If you already know the rotation angle in image coordinates, set this.
manual_theta_image_deg = None
# manual_theta_image_deg = -5.0

# Optional manual matched-point rotation.
# Use this if you can identify one non-reference point before and after.
# If both are not None, this overrides ECC.
manual_point_before = None
manual_point_after = None
# manual_point_before = np.array([342.930, 44.107], dtype=float)
# manual_point_after = np.array([YOUR_AFTER_X, YOUR_AFTER_Y], dtype=float)

SHOW_OVERLAY = True
MAX_OVERLAY_POINTS = 700


# =============================================================================
# Image loading / preprocessing
# =============================================================================

def load_combined_image(file_stem):
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True)

    if "img_array" in data:
        img = np.asarray(data["img_array"], dtype=np.float32)
    elif "ref_img_array" in data:
        img = np.asarray(data["ref_img_array"], dtype=np.float32)
    elif "sig_img_array" in data:
        img = np.asarray(data["sig_img_array"], dtype=np.float32)
    else:
        raise KeyError(f"No recognized image key in {file_stem}. Keys: {data.keys()}")

    return img


def preprocess_for_registration(img, sigma_bg=20, sigma_smooth=1.0):
    img = np.asarray(img, dtype=np.float32)

    bg = gaussian_filter(img, sigma=sigma_bg)
    proc = img - bg

    if sigma_smooth is not None and sigma_smooth > 0:
        proc = gaussian_filter(proc, sigma=sigma_smooth)

    lo, hi = np.percentile(proc, [1, 99.8])
    proc = np.clip(proc, lo, hi)

    proc -= np.min(proc)
    denom = np.max(proc)
    if denom > 0:
        proc /= denom

    return proc.astype(np.float32)


# =============================================================================
# Rotation estimation
# =============================================================================

def estimate_rotation_ecc(
    img_before,
    img_after,
    motion_model=cv2.MOTION_EUCLIDEAN,
    num_iters=5000,
    eps=1e-7,
):
    """
    Estimate image registration transform.

    theta_image_deg is in image coordinates:
        +x right
        +y down

    Positive image-coordinate rotation visually appears clockwise.
    """
    before = preprocess_for_registration(img_before)
    after = preprocess_for_registration(img_after)

    if before.shape != after.shape:
        raise ValueError(f"Image shapes differ: {before.shape} vs {after.shape}")

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        num_iters,
        eps,
    )

    cc, warp_matrix = cv2.findTransformECC(
        templateImage=after,
        inputImage=before,
        warpMatrix=warp_matrix,
        motionType=motion_model,
        criteria=criteria,
        inputMask=None,
        gaussFiltSize=5,
    )

    a = warp_matrix[0, 0]
    b = warp_matrix[1, 0]

    theta_image_deg = np.degrees(np.arctan2(b, a))
    theta_math_deg = -theta_image_deg

    return {
        "cc": float(cc),
        "warp_matrix": warp_matrix.astype(np.float32),
        "theta_image_deg": float(theta_image_deg),
        "theta_math_deg": float(theta_math_deg),
        "tx_px": float(warp_matrix[0, 2]),
        "ty_px": float(warp_matrix[1, 2]),
        "before_proc": before,
        "after_proc": after,
    }


def rotation_angle_from_matched_point(ref, p_before, p_after):
    """
    Compute image-coordinate rotation angle around ref that maps p_before -> p_after.

    This is often more intuitive than ECC if the rotation center is known.
    """
    ref = np.asarray(ref, dtype=float)
    p_before = np.asarray(p_before, dtype=float)
    p_after = np.asarray(p_after, dtype=float)

    v0 = p_before - ref
    v1 = p_after - ref

    cross = v0[0] * v1[1] - v0[1] * v1[0]
    dot = np.dot(v0, v1)

    theta_deg = np.degrees(np.arctan2(cross, dot))
    return float(theta_deg)


# =============================================================================
# Coordinate transform
# =============================================================================

def rotation_matrix_image_coords(theta_deg):
    """
    Rotation matrix in image coordinates.

    x right, y down.

    R = [[cos, -sin],
         [sin,  cos]]
    """
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s],
            [s,  c],
        ],
        dtype=np.float32,
    )


def rotate_points_about_ref(points, ref_point, theta_deg):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ref_point = np.asarray(ref_point, dtype=np.float32).reshape(1, 2)

    R = rotation_matrix_image_coords(theta_deg)

    shifted = points - ref_point
    rotated = shifted @ R.T + ref_point

    return rotated.astype(np.float32)


def get_inside_image_mask(coords, image_shape, margin_px=0):
    """
    Return mask for coords inside image boundary.

    coords are [x, y].
    image_shape is image.shape = (height, width).
    """
    coords = np.asarray(coords, dtype=np.float32).reshape(-1, 2)

    height, width = image_shape[:2]

    x = coords[:, 0]
    y = coords[:, 1]

    inside = (
        (x >= margin_px)
        & (x < width - margin_px)
        & (y >= margin_px)
        & (y < height - margin_px)
    )

    return inside


def filter_npz_per_nv_arrays(npz_data, keep_mask, old_num_nvs):
    """
    Filter all per-NV arrays whose first dimension equals old_num_nvs.

    This keeps nv_coordinates, spot_weights, updated_spot_weights, etc.
    Other metadata is unchanged.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    out = {}

    for key, val in npz_data.items():
        arr = np.asarray(val)

        if arr.shape[:1] == (old_num_nvs,):
            out[key] = arr[keep_mask]
            print(f"Filtered key '{key}': {old_num_nvs} -> {len(out[key])}")
        else:
            out[key] = val

    return out


def make_filtered_rotation_data(
    npz_data,
    rotated_coords,
    keep_mask,
    theta_used_deg,
    sign_label,
    image_shape,
):
    """
    Build filtered output dictionary for one rotation-sign candidate.
    """
    old_num_nvs = len(rotated_coords)

    out = dict(npz_data)
    out["nv_coordinates"] = rotated_coords.astype(np.float32)

    # Filter all per-NV arrays consistently.
    out = filter_npz_per_nv_arrays(out, keep_mask, old_num_nvs)

    # Save provenance.
    out["original_global_indices"] = np.where(keep_mask)[0].astype(np.int32)
    out["inside_image_mask"] = keep_mask.astype(bool)
    out["image_shape_yx"] = np.asarray(image_shape[:2], dtype=np.int32)

    out["rotation_ref_nv"] = ref_nv.astype(np.float32)
    out["rotation_theta_image_deg"] = np.array(theta_used_deg, dtype=np.float32)
    out["rotation_sign"] = np.array(sign_label)
    out["rotation_source_before_file"] = np.array(before_file)
    out["rotation_source_after_file"] = np.array(after_file)

    return out
    
def load_npz_as_dict(path):
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    return out


def save_npz(path, data_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data_dict)
    print(f"Saved: {path}")


def backup_original_once(path):
    backup_path = path.with_suffix(".backup_before_sample_rotation.npz")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        print("Backup saved:", backup_path)
    else:
        print("Backup already exists:", backup_path)


# =============================================================================
# Diagnostics
# =============================================================================

def print_coord_examples(old_coords, new_coords, ref_point, label, num=8):
    print(f"\n=== Coordinate examples: {label} ===")
    for i in range(min(num, len(old_coords))):
        print(f"{i}: old {old_coords[i]}  ->  new {new_coords[i]}")

    nearest_ref_ind = int(np.argmin(np.linalg.norm(old_coords - ref_point[None, :], axis=1)))

    print("\nReference check:")
    print("nearest ref index:", nearest_ref_ind)
    print("old nearest ref coord:", old_coords[nearest_ref_ind])
    print("new nearest ref coord:", new_coords[nearest_ref_ind])
    print(
        "ref movement [px]:",
        np.linalg.norm(new_coords[nearest_ref_ind] - old_coords[nearest_ref_ind]),
    )


def plot_coord_overlay(img_after, coords_plus, coords_minus, ref_point, theta_deg, max_points=700):
    proc = preprocess_for_registration(img_after)

    n = len(coords_plus)
    step = max(1, n // max_points)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(proc, cmap="gray")
    axes[0].scatter(
        coords_plus[::step, 0],
        coords_plus[::step, 1],
        s=10,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.8,
    )
    axes[0].scatter([ref_point[0]], [ref_point[1]], s=90, marker="x", color="red")
    axes[0].set_title(f"After image + coords rotated by +theta\n+theta = {theta_deg:+.4f} deg")
    axes[0].axis("off")

    axes[1].imshow(proc, cmap="gray")
    axes[1].scatter(
        coords_minus[::step, 0],
        coords_minus[::step, 1],
        s=10,
        facecolors="none",
        edgecolors="yellow",
        linewidths=0.8,
    )
    axes[1].scatter([ref_point[0]], [ref_point[1]], s=90, marker="x", color="red")
    axes[1].set_title(f"After image + coords rotated by -theta\n-theta = {-theta_deg:+.4f} deg")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    kpl.init_kplotlib()

    img_after = load_combined_image(after_file)

    # -------------------------------------------------------------------------
    # Determine rotation angle.
    # -------------------------------------------------------------------------
    if manual_point_before is not None and manual_point_after is not None:
        theta_image_deg = rotation_angle_from_matched_point(
            ref_nv,
            manual_point_before,
            manual_point_after,
        )

        print("\n=== Manual matched-point rotation ===")
        print("manual_point_before:", manual_point_before)
        print("manual_point_after: ", manual_point_after)
        print(f"theta_image_deg from matched point: {theta_image_deg:+.6f} deg")

    elif manual_theta_image_deg is not None:
        theta_image_deg = float(manual_theta_image_deg)

        print("\n=== Manual rotation angle ===")
        print(f"theta_image_deg: {theta_image_deg:+.6f} deg")

    else:
        img_before = load_combined_image(before_file)
        result = estimate_rotation_ecc(img_before, img_after)
        theta_image_deg = result["theta_image_deg"]

        print("\n=== ECC rotation estimate ===")
        print("Maps BEFORE image -> AFTER image")
        print("ECC correlation:", result["cc"])
        print("warp_matrix:")
        print(result["warp_matrix"])
        print(f"theta_image_deg: {theta_image_deg:+.6f} deg")
        print(f"theta_math_deg:  {result['theta_math_deg']:+.6f} deg")
        print(f"ECC translation: tx={result['tx_px']:+.3f}, ty={result['ty_px']:+.3f}")
        print("Translation is ignored for coordinate update because rotation center is fixed ref NV.")

    print("\nRotation sign convention:")
    print(f"+theta rotates points by {theta_image_deg:+.6f} deg in image coordinates.")
    print(f"-theta rotates points by {-theta_image_deg:+.6f} deg in image coordinates.")
    print("In image coordinates, positive angle appears clockwise because +y is downward.")

    # -------------------------------------------------------------------------
    # Load old coordinates.
    # -------------------------------------------------------------------------
    npz_data = load_npz_as_dict(old_npz_path)

    if "nv_coordinates" not in npz_data:
        raise KeyError(f"{old_npz_path} does not contain key 'nv_coordinates'.")

    old_coords = np.asarray(npz_data["nv_coordinates"], dtype=np.float32)

    print("\n=== Coordinate file ===")
    print("old_npz_path:", old_npz_path)
    print("number of coords:", len(old_coords))
    print("reference NV:", ref_nv)
    print("old first coord:", old_coords[0])

    # -------------------------------------------------------------------------
    # Generate both sign candidates.
    # -------------------------------------------------------------------------
    coords_plus = rotate_points_about_ref(old_coords, ref_nv, +theta_image_deg)
    coords_minus = rotate_points_about_ref(old_coords, ref_nv, -theta_image_deg)

    print_coord_examples(old_coords, coords_plus, ref_nv, "+theta")
    print_coord_examples(old_coords, coords_minus, ref_nv, "-theta")

    # -------------------------------------------------------------------------
    # Overlay diagnostic.
    # -------------------------------------------------------------------------
    if SHOW_OVERLAY:
        fig = plot_coord_overlay(
            img_after,
            coords_plus,
            coords_minus,
            ref_nv,
            theta_image_deg,
            max_points=MAX_OVERLAY_POINTS,
        )

        print("\nOverlay check:")
        print("LEFT  = +theta candidate")
        print("RIGHT = -theta candidate")
        print("Use the one whose points sit on the after-image NV/pillar spots.")

        plt.show(block=True)

    # -------------------------------------------------------------------------
    # Save both candidates, after removing out-of-bound coordinates.
    # -------------------------------------------------------------------------
    backup_original_once(old_npz_path)

    image_shape = img_after.shape
    print("\n=== Image boundary filtering ===")
    print("image shape:", image_shape)
    print("valid boundary: 0 <= x < width, 0 <= y < height")

    keep_plus = get_inside_image_mask(
        coords_plus,
        image_shape=image_shape,
        margin_px=0,
    )

    keep_minus = get_inside_image_mask(
        coords_minus,
        image_shape=image_shape,
        margin_px=0,
    )

    print("\n+theta candidate:")
    print(f"inside image: {np.sum(keep_plus)} / {len(coords_plus)}")
    print(f"removed: {len(coords_plus) - np.sum(keep_plus)}")

    print("\n-theta candidate:")
    print(f"inside image: {np.sum(keep_minus)} / {len(coords_minus)}")
    print(f"removed: {len(coords_minus) - np.sum(keep_minus)}")

    plus_data = make_filtered_rotation_data(
        npz_data=npz_data,
        rotated_coords=coords_plus,
        keep_mask=keep_plus,
        theta_used_deg=+theta_image_deg,
        sign_label="plus",
        image_shape=image_shape,
    )

    minus_data = make_filtered_rotation_data(
        npz_data=npz_data,
        rotated_coords=coords_minus,
        keep_mask=keep_minus,
        theta_used_deg=-theta_image_deg,
        sign_label="minus",
        image_shape=image_shape,
    )

    save_npz(plus_npz_path, plus_data)
    save_npz(minus_npz_path, minus_data)

    # -------------------------------------------------------------------------
    # Save chosen candidate.
    # -------------------------------------------------------------------------
    if CHOSEN_SIGN.lower() == "plus":
        chosen_data = plus_data
        chosen_coords = plus_data["nv_coordinates"]
        chosen_keep = keep_plus

    elif CHOSEN_SIGN.lower() == "minus":
        chosen_data = minus_data
        chosen_coords = minus_data["nv_coordinates"]
        chosen_keep = keep_minus

    else:
        raise ValueError("CHOSEN_SIGN must be 'plus' or 'minus'.")

    save_npz(chosen_npz_path, chosen_data)

    print("\n=== Chosen coordinate file ===")
    print("CHOSEN_SIGN:", CHOSEN_SIGN)
    print("chosen_npz_path:", chosen_npz_path)
    print("number of kept NVs:", len(chosen_coords))
    print("number removed:", len(chosen_keep) - np.sum(chosen_keep))
    print("chosen first coord:", chosen_coords[0])
    print("")
    print("Use this in later scripts:")
    print(f'npz_path = "{chosen_npz_path.as_posix()}"')


if __name__ == "__main__":
    main()