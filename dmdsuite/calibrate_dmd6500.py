# -*- coding: utf-8 -*-
"""
DMD triangle ON-pass calibration with data_manager saving.

This version separates three things:
    1. Compact zero-order calibration:
       dmdsuite/calibration/zero_order_onpass.npz
    2. Compact affine calibration for the DMD LabRAD server:
       dmdsuite/calibration/triangle_affine_onpass.npz
    3. Heavy raw acquisition data, including optional scan images:
       saved as compressed .npz using utils.data_manager path conventions.

This means zero-order calibration is saved as soon as it succeeds. On later
runs, you can reuse it and skip directly to the triangle affine calibration.

ON-pass convention:
    white = pass
    black stripe = block
    DMD coordinate = stripe position where spot intensity drops maximally.
"""

import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import labrad
import numpy as np
import matplotlib.pyplot as plt

from slmsuite.hardware.cameras.thorlabs import ThorCam
from utils import data_manager as dm
from utils import kplotlib as kpl

from matplotlib.colors import LinearSegmentedColormap

blue_cmap = LinearSegmentedColormap.from_list(
    "white_to_C0",
    ["white", "#1f77b4"],
    )
# =============================================================================
# Small helpers
# =============================================================================


def _arr(x, dtype=np.float32):
    """Robust conversion for plotting after dm.get_raw_data reload."""
    if x is None:
        return None
    return np.asarray(x, dtype=dtype)


def _maybe_stack_images(images):
    """Stack raw images if shapes match; otherwise keep as an object array."""
    if len(images) == 0:
        return np.asarray([])
    try:
        return np.stack(images, axis=0)
    except Exception:
        return np.asarray(images, dtype=object)


def _show_nonblocking(fig, show=True):
    if show:
        fig.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.001)


# =============================================================================
# Camera helpers
# =============================================================================


def safe_get_image(cam, exposure=0.0001, tries=200, delay_s=0.05):
    """Set exposure and keep trying until a real image is returned."""
    cam.set_exposure(exposure)
    time.sleep(0.15)

    for _ in range(tries):
        img = cam.get_image()
        if img is not None:
            return img
        time.sleep(delay_s)

    raise RuntimeError("Camera returned None after multiple attempts.")


def integrate_spot_intensities(img, spot_pts, roi=8):
    """Integrate local camera intensity around each spot center."""
    img = np.asarray(img).astype(np.float32)
    h, w = img.shape
    vals = []

    for x, y in spot_pts:
        x = int(round(x))
        y = int(round(y))

        x0 = max(0, x - roi)
        x1 = min(w, x + roi + 1)
        y0 = max(0, y - roi)
        y1 = min(h, y + roi + 1)

        vals.append(img[y0:y1, x0:x1].sum())

    return np.asarray(vals, dtype=np.float32)


def brightest_spot_centroid(
    img,
    threshold_percentile=99.8,
    expected_xy=None,
    half_width=120,
):
    """
    Find brightest connected component and return weighted centroid.

    If expected_xy is given, only search inside a box around that point.
    This avoids accidentally picking bright edge artifacts.
    """
    imgf = np.asarray(img).astype(np.float32)

    h, w = imgf.shape

    # ------------------------------------------------------------
    # Restrict search region if expected_xy is given.
    # ------------------------------------------------------------
    if expected_xy is not None:
        x0, y0 = expected_xy
        x0 = int(round(x0))
        y0 = int(round(y0))

        x_min = max(0, x0 - half_width)
        x_max = min(w, x0 + half_width + 1)
        y_min = max(0, y0 - half_width)
        y_max = min(h, y0 + half_width + 1)

        img_search = imgf[y_min:y_max, x_min:x_max]
    else:
        x_min = 0
        y_min = 0
        img_search = imgf

    thresh = np.percentile(img_search, threshold_percentile)
    mask = (img_search >= thresh).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    best_i = None
    best_sum = -np.inf

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if 3 <= area <= 8000:
            total = img_search[labels == i].sum()

            if total > best_sum:
                best_sum = total
                best_i = i

    if best_i is None:
        raise RuntimeError("Could not find brightest spot in selected region.")

    ys, xs = np.where(labels == best_i)
    weights = img_search[ys, xs] - thresh
    weights = np.clip(weights, 0, None)

    if weights.sum() <= 0:
        xy_local = centroids[best_i].astype(np.float32)
    else:
        xy_local = np.array(
            [
                np.sum(xs * weights) / np.sum(weights),
                np.sum(ys * weights) / np.sum(weights),
            ],
            dtype=np.float32,
        )

    # Convert crop-local coordinate back to full-image coordinate.
    xy = np.array(
        [
            xy_local[0] + x_min,
            xy_local[1] + y_min,
        ],
        dtype=np.float32,
    )

    return xy


def detect_top_n_spots(
    img,
    n=3,
    threshold_percentile=99.0,
    min_area=3,
    max_area=5000,
    zero_order_xy=None,
    zero_order_exclusion_radius=50,
    min_separation_px=25,
    refine_roi=10,
):
    """
    Connected-component detection plus local weighted centroid refinement.
    Good for the SLM triangle spots.
    """
    imgf = np.asarray(img).astype(np.float32)

    thresh = np.percentile(imgf, threshold_percentile)
    mask = (imgf >= thresh).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    candidates = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue

        x_bin, y_bin = centroids[i]

        if zero_order_xy is not None:
            zx, zy = zero_order_xy
            if np.hypot(x_bin - zx, y_bin - zy) < zero_order_exclusion_radius:
                continue

        total = imgf[labels == i].sum()
        candidates.append(
            {
                "xy": np.array([x_bin, y_bin], dtype=np.float32),
                "total": total,
                "label": i,
            }
        )

    candidates = sorted(candidates, key=lambda c: c["total"], reverse=True)

    selected = []
    for c in candidates:
        xy = c["xy"]

        if selected:
            dists = [np.linalg.norm(xy - s) for s in selected]
            if np.min(dists) < min_separation_px:
                continue

        selected.append(xy)

        if len(selected) == n:
            break

    selected = np.asarray(selected, dtype=np.float32)

    if len(selected) < n:
        print(f"WARNING: requested {n} spots, found {len(selected)}.")

    # Refine with local weighted centroid.
    refined = []
    bg = np.percentile(imgf, 50)
    h, w = imgf.shape

    for x0, y0 in selected:
        x0i = int(round(x0))
        y0i = int(round(y0))

        x1 = max(0, x0i - refine_roi)
        x2 = min(w, x0i + refine_roi + 1)
        y1 = max(0, y0i - refine_roi)
        y2 = min(h, y0i + refine_roi + 1)

        patch = imgf[y1:y2, x1:x2]
        weights = np.clip(patch - bg, 0, None)

        if weights.sum() <= 0:
            refined.append([x0, y0])
            continue

        yy, xx = np.mgrid[y1:y2, x1:x2]
        x_ref = np.sum(xx * weights) / np.sum(weights)
        y_ref = np.sum(yy * weights) / np.sum(weights)

        refined.append([x_ref, y_ref])

    return np.asarray(refined, dtype=np.float32)


# =============================================================================
# DMD scan helpers using LabRAD server
# =============================================================================


def scan_dmd_axis_for_spots_onpass(
    dmd,
    cam,
    cam_pts,
    axis="x",
    positions=None,
    stripe_width=40,
    plane=220,
    exposure=0.0001,
    roi=8,
    settle_s=0.08,
    save_scan_images=True,
):
    """
    ON-pass calibration:
        white = pass
        black stripe = block

    Coordinate is where spot intensity DROP is maximum.

    Returns
    -------
    positions : np.ndarray
    drops : np.ndarray, shape = [num_positions, num_spots]
    best_positions : np.ndarray, shape = [num_spots]
    raw : dict containing all pass image, per-step values, and optionally images
    """
    if positions is None:
        if axis == "x":
            positions = np.arange(1000, 1820, 4)
        else:
            positions = np.arange(840, 5000, 4)

    positions = np.asarray(positions, dtype=np.float32)

    # Reference pass image.
    dmd.pass_all(True)
    time.sleep(0.2)
    img_pass = safe_get_image(cam, exposure=exposure)
    pass_vals = integrate_spot_intensities(img_pass, cam_pts, roi=roi)
    pass_safe = np.maximum(pass_vals, 1.0)

    drops = []
    scan_vals = []
    scan_images = []

    for p in positions:
        dmd.show_blocking_stripe(axis, float(p), int(stripe_width), int(plane))
        time.sleep(settle_s)

        img = safe_get_image(cam, exposure=exposure)
        vals = integrate_spot_intensities(img, cam_pts, roi=roi)

        drop = np.clip(pass_vals - vals, 0, None)
        drop_frac = drop / pass_safe

        drops.append(drop_frac)
        scan_vals.append(vals)
        if save_scan_images:
            scan_images.append(np.asarray(img))

        print(
            f"{axis} scan {p}: max fractional drop {np.max(drop_frac):.3f}",
            flush=True,
        )

    drops = np.asarray(drops, dtype=np.float32)
    scan_vals = np.asarray(scan_vals, dtype=np.float32)
    best_indices = np.argmax(drops, axis=0)
    best_positions = positions[best_indices].astype(np.float32)

    raw = {
        "axis": axis,
        "positions": positions,
        "stripe_width": int(stripe_width),
        "plane": int(plane),
        "exposure": float(exposure),
        "roi": int(roi),
        "settle_s": float(settle_s),
        "pass_image": np.asarray(img_pass),
        "pass_vals": pass_vals,
        "scan_vals": scan_vals,
        "scan_images": _maybe_stack_images(scan_images) if save_scan_images else None,
        "drops": drops,
        "best_indices": best_indices.astype(np.int32),
        "best_positions": best_positions,
    }

    return positions, drops, best_positions, raw


def fit_cam_to_dmd_affine(cam_pts, dmd_pts):
    cam_pts = np.asarray(cam_pts, dtype=np.float32)
    dmd_pts = np.asarray(dmd_pts, dtype=np.float32)

    M, inliers = cv2.estimateAffine2D(cam_pts, dmd_pts)

    if M is None:
        raise RuntimeError("Affine camera->DMD fit failed.")

    return M.astype(np.float32), inliers


def apply_affine(M, pts):
    """Apply 2x3 affine matrix to Nx2 points."""
    M = np.asarray(M, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    return pts_h @ M.T


# =============================================================================
# Plotting / processing
# =============================================================================


def plot_camera_points(img, points=None, labels=None, title="Camera image", show=False):
    img = np.asarray(img)
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax.imshow(img, cmap="gray")
    ax.imshow(img, cmap=blue_cmap)

    if points is not None:
        points = np.asarray(points, dtype=np.float32)
        if points.ndim == 1:
            points = points[None, :]
        for ind, p in enumerate(points):
            ax.plot(p[0], p[1], "rx", markersize=8)
            if labels is not None:
                ax.text(p[0] + 5, p[1] + 5, labels[ind], color="red")

    ax.set_title(title)
    fig.tight_layout()
    _show_nonblocking(fig, show=show)
    return fig


def plot_response_curves(positions, responses, title="DMD scan", show=False):
    positions = np.asarray(positions, dtype=np.float32)
    responses = np.asarray(responses, dtype=np.float32)
    if responses.ndim == 1:
        responses = responses[:, None]

    fig, ax = plt.subplots(figsize=(8, 4))
    nspots = responses.shape[1]

    for i in range(nspots):
        ax.plot(positions, responses[:, i], "-o", label=f"spot {i}")

    ax.set_xlabel("DMD scan position")
    ax.set_ylabel("fractional intensity drop")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _show_nonblocking(fig, show=show)
    return fig


def plot_affine_check(data, show=False):
    tri_cam_pts = _arr(data.get("tri_cam_pts"))
    tri_dmd_pts = _arr(data.get("tri_dmd_pts"))
    M_cam_to_dmd = _arr(data.get("M_cam_to_dmd"))

    if tri_cam_pts is None or tri_dmd_pts is None or M_cam_to_dmd is None:
        return None

    predicted = apply_affine(M_cam_to_dmd, tri_cam_pts)
    residual = tri_dmd_pts - predicted
    residual_norm = np.linalg.norm(residual, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(residual_norm)), residual_norm, "o-")
    ax.set_xlabel("Triangle spot index")
    ax.set_ylabel("DMD residual magnitude [px]")
    ax.set_title("Camera→DMD affine fit residuals")
    fig.tight_layout()
    _show_nonblocking(fig, show=show)
    return fig


def process_and_plot(data, show=False):
    """
    Generate all diagnostic plots from a saved raw-data dictionary.

    This is intentionally separated from acquisition so you can later do:
        data = dm.get_raw_data(file_id=...)
        figs = process_and_plot(data, show=True)
    """
    figs = []

    if data.get("img_zero") is not None and data.get("zero_cam_xy") is not None:
        figs.append(
            plot_camera_points(
                data["img_zero"],
                points=data["zero_cam_xy"],
                labels=["0th"],
                title="0th-order / reference beam centroid",
                show=show,
            )
        )

    if data.get("zero_x") is not None:
        figs.append(
            plot_response_curves(
                data["zero_x"]["positions"],
                data["zero_x"]["drops"],
                title="0th-order DMD x scan",
                show=show,
            )
        )

    if data.get("zero_y") is not None:
        figs.append(
            plot_response_curves(
                data["zero_y"]["positions"],
                data["zero_y"]["drops"],
                title="0th-order DMD y scan",
                show=show,
            )
        )

    if data.get("img_triangle") is not None and data.get("tri_cam_pts") is not None:
        tri_pts = _arr(data["tri_cam_pts"])
        labels = [str(ind) for ind in range(len(tri_pts))]
        if data.get("zero_cam_xy") is not None:
            points = np.vstack([_arr(data["zero_cam_xy"]), tri_pts])
            labels = ["0th"] + labels
        else:
            points = tri_pts
        figs.append(
            plot_camera_points(
                data["img_triangle"],
                points=points,
                labels=labels,
                title="Triangle calibration spots",
                show=show,
            )
        )

    if data.get("triangle_x") is not None:
        figs.append(
            plot_response_curves(
                data["triangle_x"]["positions"],
                data["triangle_x"]["drops"],
                title="Triangle DMD x scan",
                show=show,
            )
        )

    if data.get("triangle_y") is not None:
        figs.append(
            plot_response_curves(
                data["triangle_y"]["positions"],
                data["triangle_y"]["drops"],
                title="Triangle DMD y scan",
                show=show,
            )
        )

    fig = plot_affine_check(data, show=show)
    if fig is not None:
        figs.append(fig)

    return figs


def _compressed_npz_path_from_dm(file_label, timestamp=None):
    """
    Build a data-manager-style path, but save as compressed .npz.

    dm.save_raw_data can be inconvenient for very large camera-image stacks.
    This keeps the usual timestamp/repr-name directory convention while using
    np.savez_compressed for the heavy DMD raw data.
    """
    if timestamp is None:
        timestamp = dm.get_time_stamp()

    base_path = Path(str(dm.get_file_path(__file__, timestamp, file_label)))
    if base_path.suffix:
        npz_path = base_path.with_suffix(".npz")
    else:
        npz_path = Path(str(base_path) + ".npz")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    return npz_path, timestamp


def _flatten_dict_for_npz(obj, prefix="", out=None):
    """
    Flatten nested dict/list/array content into np.savez-compatible keys.

    Nested dictionaries are saved with keys like:
        zero_x__positions
        triangle_y__scan_images

    This avoids saving one big pickled Python object and keeps the .npz easier
    to inspect with np.load(...).files.
    """
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if value is None:
                continue
            key = str(key).replace("/", "_")
            new_prefix = key if prefix == "" else f"{prefix}__{key}"
            _flatten_dict_for_npz(value, new_prefix, out)
        return out

    if prefix == "":
        raise ValueError("Cannot save an unnamed top-level object to npz.")

    try:
        arr = np.asarray(obj)
        # Avoid object arrays when possible. If conversion produced an object
        # array, store a string representation instead of a pickle-dependent blob.
        if arr.dtype == object:
            out[prefix] = np.asarray(str(obj))
        else:
            out[prefix] = arr
    except Exception:
        out[prefix] = np.asarray(str(obj))

    return out


def _restore_scalar_or_array(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _unflatten_npz_dict(flat):
    """Rebuild nested dictionaries from keys generated by _flatten_dict_for_npz."""
    out = {}
    for flat_key, value in flat.items():
        keys = flat_key.split("__")
        target = out
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = _restore_scalar_or_array(value)
    return out


def save_raw_data_npz_compressed(data, file_label="dmd-triangle-affine-onpass-raw"):
    """Save the full heavy DMD raw data as compressed .npz using dm path style."""
    timestamp = data.get("timestamp", dm.get_time_stamp())
    data["timestamp"] = timestamp

    npz_path, _ = _compressed_npz_path_from_dm(file_label, timestamp=timestamp)
    flat = _flatten_dict_for_npz(data)
    np.savez_compressed(npz_path, **flat)

    print(f"Saved compressed raw DMD data to: {npz_path}")
    return str(npz_path)


def load_raw_data_npz_compressed(npz_path):
    """
    Load a compressed raw DMD .npz produced by save_raw_data_npz_compressed.

    Example:
        data = load_raw_data_npz_compressed(r"path/to/file.npz")
        process_and_plot(data, show=True)
    """
    with np.load(npz_path, allow_pickle=False) as npz:
        flat = {key: npz[key] for key in npz.files}
    return _unflatten_npz_dict(flat)


def save_data_and_figures(data, file_label="dmd-triangle-affine-onpass", show=True):
    """
    Save full raw data as compressed .npz and save diagnostic figures.

    The heavy raw data is no longer saved with dm.save_raw_data; that can become
    too large when scan images are included. Instead, this uses dm.get_file_path
    for the normal timestamped location and np.savez_compressed for storage.
    """
    timestamp = data.get("timestamp", dm.get_time_stamp())
    data["timestamp"] = timestamp

    raw_npz_path = save_raw_data_npz_compressed(
        data,
        file_label=f"{file_label}-raw-compressed",
    )
    data["raw_npz_path"] = raw_npz_path

    try:
        figs = process_and_plot(data, show=show)
    except Exception:
        print(traceback.format_exc())
        figs = []

    for ind, fig in enumerate(figs):
        fig_path = dm.get_file_path(__file__, timestamp, f"{file_label}-{ind}")
        dm.save_figure(fig, fig_path)
        print(f"Saved figure {ind} to: {fig_path}")

    return raw_npz_path, figs


# =============================================================================
# Calibration routines
# =============================================================================


def calibrate_zero_order_onpass(
    dmd,
    cam,
    exposure=0.0001,
    roi=12,
    stripe_width=20,
    zero_radius_px=30,
    save_scan_images=True,
):
    """
    Find 0th-order camera position, then find corresponding DMD x/y
    by black-stripe scans in ON-pass geometry.
    """
    print("\n=== 0th-order calibration ===")

    dmd.pass_all(False)
    time.sleep(0.2)
    img_zero = safe_get_image(cam, exposure=exposure)

    # zero_cam_xy = brightest_spot_centroid(
    #     img_zero,
    #     threshold_percentile=99.8,
    # )
    zero_cam_xy = brightest_spot_centroid(
        img_zero,
        threshold_percentile=99.8,
        expected_xy=[717.849609375, 532.0283203125],
        half_width=120,
    )

    cam_pts = np.array([zero_cam_xy], dtype=np.float32)

    x_positions, x_drop, dmd_x, x_raw = scan_dmd_axis_for_spots_onpass(
        dmd=dmd,
        cam=cam,
        cam_pts=cam_pts,
        axis="x",
        positions=np.arange(850, 1020, 4),
        stripe_width=stripe_width,
        plane=220,
        exposure=exposure,
        roi=roi,
        save_scan_images=save_scan_images,
    )

    y_positions, y_drop, dmd_y, y_raw = scan_dmd_axis_for_spots_onpass(
        dmd=dmd,
        cam=cam,
        cam_pts=cam_pts,
        axis="y",
        positions=np.arange(450, 600, 4),
        stripe_width=stripe_width,
        plane=221,
        exposure=exposure,
        roi=roi,
        save_scan_images=save_scan_images,
    )

    zero_dmd_xy = np.array([dmd_x[0], dmd_y[0]], dtype=np.float32)

    print("zero camera xy:", zero_cam_xy)
    print("zero DMD xy:", zero_dmd_xy)

    update_zero_result = dmd.update_zero_block_xy(
        float(zero_dmd_xy[0]),
        float(zero_dmd_xy[1]),
        int(zero_radius_px),
    )

    zero_data = {
        "zero_cam_xy": zero_cam_xy,
        "zero_dmd_xy": zero_dmd_xy,
        "img_zero": np.asarray(img_zero),
        "zero_x": x_raw,
        "zero_y": y_raw,
        "zero_radius_px": int(zero_radius_px),
        "update_zero_block_result": str(update_zero_result),
    }

    return zero_data


def calibrate_triangle_onpass(
    dmd,
    cam,
    zero_cam_xy,
    exposure=0.0001,
    roi=8,
    stripe_width=40,
    save_scan_images=True,
):
    """
    User should already have written the SLM triangle pattern and held it.
    This function detects 3 spots and finds the camera->DMD affine map.
    """
    print("\n=== Triangle calibration ===")

    dmd.pass_all(True)
    time.sleep(0.2)
    img_triangle = safe_get_image(cam, exposure=exposure)

    tri_cam_pts = detect_top_n_spots(
        img_triangle,
        n=3,
        threshold_percentile=99.0,
        min_area=3,
        max_area=50000,
        zero_order_xy=zero_cam_xy,
        zero_order_exclusion_radius=40,
        min_separation_px=30,
        refine_roi=10,
    )

    print("Triangle camera points:")
    print(tri_cam_pts)

    x_positions, x_drop, dmd_x, x_raw = scan_dmd_axis_for_spots_onpass(
        dmd=dmd,
        cam=cam,
        cam_pts=tri_cam_pts,
        axis="x",
        positions=np.arange(300, 1500, 5),
        stripe_width=stripe_width,
        plane=222,
        exposure=exposure,
        roi=roi,
        save_scan_images=save_scan_images,
    )

    # Plot immediately so you can inspect before continuing to y-scan.
    tmp_data = {
        "img_triangle": img_triangle,
        "tri_cam_pts": tri_cam_pts,
        "zero_cam_xy": zero_cam_xy,
        "triangle_x": x_raw,
    }
    process_and_plot(tmp_data, show=True)
    input("X scan done. Press Enter for Y scan...")

    y_positions, y_drop, dmd_y, y_raw = scan_dmd_axis_for_spots_onpass(
        dmd=dmd,
        cam=cam,
        cam_pts=tri_cam_pts,
        axis="y",
        positions=np.arange(80, 1100, 5),
        stripe_width=stripe_width,
        plane=223,
        exposure=exposure,
        roi=roi,    
        save_scan_images=save_scan_images,
    )

    tri_dmd_pts = np.column_stack([dmd_x, dmd_y]).astype(np.float32)

    print("Triangle DMD points:")
    print(tri_dmd_pts)

    M_cam_to_dmd, inliers = fit_cam_to_dmd_affine(tri_cam_pts, tri_dmd_pts)

    print("M_cam_to_dmd:")
    print(M_cam_to_dmd)

    triangle_data = {
        "img_triangle": np.asarray(img_triangle),
        "tri_cam_pts": tri_cam_pts,
        "tri_dmd_pts": tri_dmd_pts,
        "triangle_x": x_raw,
        "triangle_y": y_raw,
        "M_cam_to_dmd": M_cam_to_dmd,
        "inliers": inliers,
    }

    return triangle_data


def _ensure_parent_dir(path):
    path = Path(path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_zero_order_calibration_npz(
    zero_data,
    save_path="dmdsuite/calibration/zero_order_onpass.npz",
):
    """
    Save a compact zero-order calibration immediately after it succeeds.

    This lets you skip the 0th-order scan next time and go directly to the
    triangle affine calibration.
    """
    path = _ensure_parent_dir(save_path)

    np.savez_compressed(
        path,
        zero_cam_xy=np.asarray(zero_data["zero_cam_xy"], dtype=np.float32),
        zero_dmd_xy=np.asarray(zero_data["zero_dmd_xy"], dtype=np.float32),
        zero_radius_px=np.asarray(zero_data.get("zero_radius_px", 30), dtype=np.int32),
        convention=np.asarray("ON_PASS_WHITE_PASS_BLACK_BLOCK"),
        calibration_type=np.asarray("dmd_zero_order_onpass"),
        created_timestamp=np.asarray(zero_data.get("timestamp", dm.get_time_stamp())),
    )

    print(f"Saved compact zero-order calibration to: {path}")
    return str(path)


def load_zero_order_calibration_npz(calib_path="dmdsuite/calibration/zero_order_onpass.npz"):
    """Load compact zero-order calibration without rerunning the stripe scan."""
    path = Path(calib_path)
    if not path.exists():
        raise FileNotFoundError(f"Zero-order calibration file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        zero_data = {
            "zero_cam_xy": np.asarray(npz["zero_cam_xy"], dtype=np.float32),
            "zero_dmd_xy": np.asarray(npz["zero_dmd_xy"], dtype=np.float32),
            "zero_radius_px": int(np.asarray(npz["zero_radius_px"]).item()),
            "zero_calib_path": str(path),
            "zero_calib_loaded_from_file": True,
        }

    return zero_data


def apply_zero_order_to_dmd(dmd, zero_data):
    """Update the LabRAD DMD server with a loaded zero-order coordinate."""
    zero_dmd_xy = np.asarray(zero_data["zero_dmd_xy"], dtype=np.float32).reshape(2)
    radius_px = int(zero_data.get("zero_radius_px", 30))
    result = dmd.update_zero_block_xy(
        float(zero_dmd_xy[0]),
        float(zero_dmd_xy[1]),
        radius_px,
    )
    return str(result)


def save_server_calibration_npz(
    data,
    save_path="dmdsuite/calibration/triangle_affine_onpass.npz",
):
    """
    Save the compact affine calibration expected by the DMD LabRAD server.

    This file is intentionally small and separate from the heavy raw-data .npz.
    The server currently reads M_cam_to_dmd, zero_dmd_xy, and zero_cam_xy from
    this file, with optional DMD/camera point arrays available for debugging.
    """
    path = _ensure_parent_dir(save_path)

    np.savez_compressed(
        path,
        M_cam_to_dmd=np.asarray(data["M_cam_to_dmd"], dtype=np.float32),
        zero_cam_xy=np.asarray(data["zero_cam_xy"], dtype=np.float32),
        zero_dmd_xy=np.asarray(data["zero_dmd_xy"], dtype=np.float32),
        zero_radius_px=np.asarray(data.get("zero_radius_px", 30), dtype=np.int32),
        tri_cam_pts=np.asarray(data["tri_cam_pts"], dtype=np.float32),
        tri_dmd_pts=np.asarray(data["tri_dmd_pts"], dtype=np.float32),
        inliers=np.asarray(data["inliers"]),
        convention=np.asarray("ON_PASS_WHITE_PASS_BLACK_BLOCK"),
        calibration_type=np.asarray("dmd_triangle_affine_onpass"),
        created_timestamp=np.asarray(data.get("timestamp", dm.get_time_stamp())),
    )

    print(f"Saved compact server affine calibration to: {path}")
    return str(path)


# =============================================================================
# Main acquisition / reload path
# =============================================================================


def main(
    load_file_id=None,
    load_npz_path=None,
    camera_serial="26438",
    exposure_zero=0.0001,
    exposure_triangle=0.0001,
    save_scan_images=True,
    reuse_zero_order=True,
    force_zero_order=False,
    zero_calib_path="dmdsuite/calibration/zero_order_onpass.npz",
    server_calib_path="dmdsuite/calibration/triangle_affine_onpass.npz",
):
    """
    If load_file_id is None, run a new calibration.
    If load_file_id is given, reload old dm.save_raw_data data.
    If load_npz_path is given, reload compressed raw .npz data.
    """
    if load_npz_path is not None:
        data = load_raw_data_npz_compressed(load_npz_path)
        process_and_plot(data, show=True)
        kpl.show(block=True)
        return data

    if load_file_id is not None:
        data = dm.get_raw_data(file_id=load_file_id)
        process_and_plot(data, show=True)
        kpl.show(block=True)
        return data

    timestamp = dm.get_time_stamp()

    cxn = labrad.connect(username="", password="")
    dmd = cxn.dmd_dlp6500
    cam = ThorCam(serial=camera_serial, verbose=True)

    data = {
        "timestamp": timestamp,
        "camera_serial": camera_serial,
        "calibration_type": "dmd_triangle_affine_onpass",
        "convention": "ON_PASS_WHITE_PASS_BLACK_BLOCK",
        "save_scan_images": bool(save_scan_images),
        "reuse_zero_order": bool(reuse_zero_order),
        "force_zero_order": bool(force_zero_order),
        "zero_calib_path": zero_calib_path,
        "server_calib_path": server_calib_path,
        "exposure_zero": float(exposure_zero),
        "exposure_triangle": float(exposure_triangle),
    }

    try:
        data["dmd_state_initial"] = str(dmd.get_state())
        print(data["dmd_state_initial"])

        # Make sure DMD planes are uploaded.
        # If already initialized, this is okay.
        data["initialize_pass_state_result"] = str(dmd.initialize_pass_state())
        print(data["initialize_pass_state_result"])

        zero_path_obj = Path(zero_calib_path)
        can_reuse_zero = (
            bool(reuse_zero_order)
            and not bool(force_zero_order)
            and zero_path_obj.exists()
        )

        if can_reuse_zero:
            print(f"\n=== Loading existing zero-order calibration: {zero_calib_path} ===")
            zero_data = load_zero_order_calibration_npz(zero_calib_path)
            zero_data["update_zero_block_result"] = apply_zero_order_to_dmd(dmd, zero_data)
            print("zero camera xy:", zero_data["zero_cam_xy"])
            print("zero DMD xy:", zero_data["zero_dmd_xy"])
            print(zero_data["update_zero_block_result"])
            data.update(zero_data)
        else:
            input(
                "\nStep 1: Make sure ONLY the 0th-order / reference beam is visible "
                "or no SLM pattern is written. Press Enter to calibrate 0th order..."
            )

            zero_data = calibrate_zero_order_onpass(
                dmd=dmd,
                cam=cam,
                exposure=exposure_zero,
                roi=12,
                stripe_width=50,
                zero_radius_px=30,
                save_scan_images=save_scan_images,
            )
            zero_data["timestamp"] = timestamp
            zero_saved_path = save_zero_order_calibration_npz(zero_data, zero_calib_path)
            zero_data["zero_calib_path"] = zero_saved_path
            zero_data["zero_calib_loaded_from_file"] = False
            data.update(zero_data)

            # Show zero-order diagnostics before moving to triangle.
            process_and_plot(
                {
                    "img_zero": data["img_zero"],
                    "zero_cam_xy": data["zero_cam_xy"],
                    "zero_x": data["zero_x"],
                    "zero_y": data["zero_y"],
                },
                show=True,
            )

        input(
            "\nStep 2: Now write the SLM triangle pattern and keep it held. "
            "Press Enter here when triangle spots are visible..."
        )

        triangle_data = calibrate_triangle_onpass(
            dmd=dmd,
            cam=cam,
            zero_cam_xy=data["zero_cam_xy"],
            exposure=exposure_triangle,
            roi=8,
            stripe_width=40,
            save_scan_images=save_scan_images,
        )
        data.update(triangle_data)

        server_path = save_server_calibration_npz(data, server_calib_path)
        data["server_calib_path"] = server_path

        # Load into server immediately.
        data["load_calibration_result"] = str(dmd.load_calibration(server_path))
        print(data["load_calibration_result"])

        # Save raw data and all figures with data_manager.
        save_data_and_figures(
            data,
            file_label="dmd-triangle-affine-onpass",
            show=True,
        )

        input("\nCalibration complete. Press Enter to leave DMD in bypass/pass state...")
        dmd.pass_all(False)

    except Exception:
        # Save partial data if something fails after timestamp creation.
        print(traceback.format_exc())
        data["exception"] = traceback.format_exc()
        try:
            save_data_and_figures(
                data,
                file_label="dmd-triangle-affine-onpass-FAILED",
                show=True,
            )
        except Exception:
            print("Could not save partial calibration data:")
            print(traceback.format_exc())
        raise

    finally:
        cam.close()

    kpl.show()
    return data


if __name__ == "__main__":
    kpl.init_kplotlib()

    # To reprocess old dm.save_raw_data data later, set this to a raw-data file ID.
    # Example:
    # LOAD_FILE_ID = 1732924888109
    LOAD_FILE_ID = None

    # To reprocess the new compressed DMD raw .npz, set this to the .npz path.
    LOAD_NPZ_PATH = None

    # main(
    #     load_file_id=LOAD_FILE_ID,
    #     load_npz_path=LOAD_NPZ_PATH,
    #     reuse_zero_order=True,   # use dmdsuite/calibration/zero_order_onpass.npz if present
    #     force_zero_order=False,  # set True only when you want to redo 0th-order scan
    # )
    
    ## take a qu
    cam = ThorCam(serial="26438", verbose=True)
    try:
        img = safe_get_image(cam, exposure=0.0001)
        plt.figure(figsize=(8, 5.5))
        plt.imshow(img, cmap=blue_cmap)
        plt.colorbar()
        plt.title("ThorCam Image")
        plt.show(block=True)
    finally:
        cam.close()