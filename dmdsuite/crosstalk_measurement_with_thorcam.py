# -*- coding: utf-8 -*-
"""
No-calibration SLM circle-pattern aberration analysis.

No SLM/Nuvu/DMD calibration is used.

Workflow:
    1. Turn yellow on with OPX constant_ac.
    2. Take ThorCam image of already-displayed SLM circle pattern.
    3. Detect bright/dim spots directly in ThorCam pixels.
    4. Detect zeroth-order spot.
    5. Generate the predetermined ideal circle pattern.
    6. Fit ideal circle pattern to observed spots.
    7. Quantify aberration from residuals and ring circularity.
    8. Save image, detected spots, matched pattern, and analysis.

Also includes:
    do_thorcam_sample_image_with_yellow()
for yellow-off / yellow-on / difference image testing.
"""

import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from slmsuite.hardware.cameras.thorlabs import ThorCam
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# USER SETTINGS
# =============================================================================

CAMERA_SERIAL = "26438"
EXPOSURE_S = 0.0001

YELLOW_CHANNEL = 7
YELLOW_AMP = 0.04

SAVE_LABEL = "slm-circles-predetermined-pattern-analysis"

# Predetermined circle pattern used in your SLM script.
# Only relative geometry matters here. No calibration is used.
CIRCLE_RADII_PX = np.linspace(20, 120, num=6)
CIRCLE_POINT_SPACING_PX = 30
CIRCLE_ANGLE_OFFSET_RAD = 0.0

# Zeroth-order detection.
# Options:
#   "brightest"            -> choose brightest detected component
#   "nearest_image_center" -> choose detected spot nearest image center
#   "manual"              -> use MANUAL_ZERO_XY
ZERO_MODE = "brightest"
MANUAL_ZERO_XY = np.array([705.0, 520.0], dtype=np.float32)

# Detection thresholds. For dim images, lower values help.
THRESHOLD_PERCENTILES = [99.8, 99.5, 99.2, 98.8, 98.0, 97.0, 96.0, 95.0]

MIN_AREA = 3
MAX_AREA = 50000
MIN_SEPARATION_PX = 8
REFINE_ROI = 8
MAX_REASONABLE_SPOTS = 400

# Optional crop to avoid edge artifacts.
USE_CROP = False
CROP_XYWH = [300, 100, 900, 900]  # x, y, width, height

blue_cmap = LinearSegmentedColormap.from_list(
    "white_to_C0",
    ["white", "#1f77b4"],
)


# =============================================================================
# HARDWARE HELPERS
# =============================================================================

def set_yellow(opx, yellow_channel=7, yellow_amp=0.0):
    """
    OPX constant_ac keeps this output on until you set it back to 0.
    """
    opx.constant_ac(
        [],                # digital channels
        [yellow_channel],  # analog channels
        [yellow_amp],      # analog voltages
        [0],               # analog frequencies
    )


def safe_get_image(cam, exposure=0.0001, tries=200, delay_s=0.05):
    cam.set_exposure(exposure)
    time.sleep(0.15)

    for _ in range(tries):
        img = cam.get_image()
        if img is not None:
            return np.asarray(img)
        time.sleep(delay_s)

    raise RuntimeError("Camera returned None after multiple attempts.")


def capture_thorcam_image_with_yellow(
    exposure=0.0001,
    yellow_channel=7,
    yellow_amp=0.04,
):
    """
    Turn yellow on, take one ThorCam image, then always turn yellow off.
    """
    cxn = None
    opx = None
    cam = None

    try:
        cxn = common.labrad_connect()
        opx = cxn.QM_opx
        cam = ThorCam(serial=CAMERA_SERIAL, verbose=True)

        set_yellow(opx, yellow_channel=yellow_channel, yellow_amp=yellow_amp)
        time.sleep(0.2)

        img = safe_get_image(cam, exposure=exposure)
        return img

    finally:
        try:
            if opx is not None:
                set_yellow(opx, yellow_channel=yellow_channel, yellow_amp=0.0)
                time.sleep(0.05)
        except Exception as exc:
            print("Could not turn off yellow:", exc)

        try:
            if cam is not None:
                cam.close()
        except Exception:
            pass

        try:
            if cxn is not None:
                cxn.disconnect()
        except Exception:
            pass


# =============================================================================
# SIMPLE IMAGE TEST FUNCTION
# =============================================================================

def save_thorcam_snapshot(img, label="thorcam-snapshot", exposure=0.0001):
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, label)
    npz_path = str(file_path) + ".npz"

    np.savez_compressed(
        npz_path,
        img=np.asarray(img),
        exposure=np.asarray(exposure, dtype=np.float32),
        timestamp=np.asarray(timestamp),
        label=np.asarray(label),
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    if np.nanmin(img) < 0:
        vmax = np.nanpercentile(np.abs(img), 99.5)
        ax.imshow(img, cmap="bwr", vmin=-vmax, vmax=vmax)
    else:
        vmin, vmax = np.nanpercentile(img, [1, 99.8])
        ax.imshow(img, cmap=blue_cmap, vmin=vmin, vmax=vmax)

    ax.set_title(label)
    cbar = fig.colorbar(ax.images[0], ax=ax)
    cbar.set_label("counts")
    dm.save_figure(fig, file_path)

    print("Saved image data:", npz_path)
    print("Saved figure:", file_path)

    return npz_path, fig


def do_thorcam_sample_image_with_yellow(
    label="thorcam-sample-yellow",
    exposure=0.0001,
    yellow_channel=7,
    yellow_amp=0.04,
    wait_before_cleanup=True,
):
    """
    Test function:
        yellow off image
        yellow on image
        difference image
    """
    cxn = None
    opx = None
    cam = None

    try:
        cxn = common.labrad_connect()
        opx = cxn.QM_opx
        cam = ThorCam(serial=CAMERA_SERIAL, verbose=True)

        # Yellow off.
        set_yellow(opx, yellow_channel=yellow_channel, yellow_amp=0.0)
        time.sleep(0.2)
        img_off = safe_get_image(cam, exposure=exposure)

        off_path, _ = save_thorcam_snapshot(
            img_off,
            label=f"{label}-yellow-off",
            exposure=exposure,
        )

        # Yellow on.
        set_yellow(opx, yellow_channel=yellow_channel, yellow_amp=yellow_amp)
        time.sleep(0.2)
        img_on = safe_get_image(cam, exposure=exposure)

        on_path, _ = save_thorcam_snapshot(
            img_on,
            label=f"{label}-yellow-on",
            exposure=exposure,
        )

        # Difference.
        img_diff = img_on.astype(np.float32) - img_off.astype(np.float32)

        diff_path, _ = save_thorcam_snapshot(
            img_diff,
            label=f"{label}-yellow-diff",
            exposure=exposure,
        )

        print("Saved yellow-off image:", off_path)
        print("Saved yellow-on image:", on_path)
        print("Saved difference image:", diff_path)

        plt.show(block=False)

        if wait_before_cleanup:
            input("Press Enter to turn off yellow and continue...")

        return {
            "img_off": img_off,
            "img_on": img_on,
            "img_diff": img_diff,
            "off_path": off_path,
            "on_path": on_path,
            "diff_path": diff_path,
        }

    finally:
        try:
            if opx is not None:
                set_yellow(opx, yellow_channel=yellow_channel, yellow_amp=0.0)
        except Exception as exc:
            print("Could not turn off yellow:", exc)

        try:
            if cam is not None:
                cam.close()
        except Exception:
            pass

        try:
            if cxn is not None:
                cxn.disconnect()
        except Exception:
            pass


# =============================================================================
# PREDETERMINED IDEAL PATTERN
# =============================================================================

def make_ideal_circle_pattern(
    radii=CIRCLE_RADII_PX,
    spacing_px=CIRCLE_POINT_SPACING_PX,
    angle_offset_rad=CIRCLE_ANGLE_OFFSET_RAD,
):
    """
    Generate the ideal circle pattern in arbitrary relative coordinates.

    Center/zeroth order is at [0, 0].
    Circle spots are on predetermined radii with predetermined angular spacing.
    """
    ideal_pts = []
    ring_ids = []
    ideal_angles = []

    for ring_id, radius in enumerate(radii):
        num_points = max(12, int(2 * np.pi * radius / spacing_px))

        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        theta = theta + angle_offset_rad

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        pts = np.column_stack([x, y])

        ideal_pts.append(pts)
        ring_ids.extend([ring_id] * num_points)
        ideal_angles.extend(theta)

    ideal_pts = np.vstack(ideal_pts).astype(np.float32)
    ring_ids = np.asarray(ring_ids, dtype=np.int32)
    ideal_angles = np.asarray(ideal_angles, dtype=np.float32)

    zero_ideal = np.array([[0.0, 0.0]], dtype=np.float32)

    return zero_ideal, ideal_pts, ring_ids, ideal_angles


# =============================================================================
# SPOT DETECTION
# =============================================================================

def preprocess_image(img, bg_sigma=35):
    imgf = np.asarray(img, dtype=np.float32)

    bg = cv2.GaussianBlur(imgf, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    proc = imgf - bg

    proc -= np.percentile(proc, 5)
    proc = np.clip(proc, 0, None)

    return proc


def refine_centroid(imgf, xy, roi=8):
    h, w = imgf.shape
    x0, y0 = xy

    xi = int(round(x0))
    yi = int(round(y0))

    x1 = max(0, xi - roi)
    x2 = min(w, xi + roi + 1)
    y1 = max(0, yi - roi)
    y2 = min(h, yi + roi + 1)

    patch = imgf[y1:y2, x1:x2]
    bg = np.percentile(imgf, 50)
    weights = np.clip(patch - bg, 0, None)

    if weights.sum() <= 0:
        return np.array([x0, y0], dtype=np.float32)

    yy, xx = np.mgrid[y1:y2, x1:x2]

    x_ref = np.sum(xx * weights) / np.sum(weights)
    y_ref = np.sum(yy * weights) / np.sum(weights)

    return np.array([x_ref, y_ref], dtype=np.float32)


def detect_spots_at_threshold(
    img,
    threshold_percentile,
    min_area=3,
    max_area=50000,
    min_separation_px=8,
    refine_roi=8,
):
    imgf = np.asarray(img, dtype=np.float32)
    proc = preprocess_image(imgf)

    thresh = np.percentile(proc, threshold_percentile)
    mask = (proc >= thresh).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    candidates = []

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]

        if area < min_area or area > max_area:
            continue

        total = float(proc[labels == lab].sum())
        xy = np.asarray(centroids[lab], dtype=np.float32)

        candidates.append(
            {
                "xy": xy,
                "total": total,
                "area": area,
            }
        )

    candidates = sorted(candidates, key=lambda c: c["total"], reverse=True)

    selected = []
    selected_xy = []

    for c in candidates:
        xy = c["xy"]

        if len(selected_xy) > 0:
            dists = np.linalg.norm(np.asarray(selected_xy) - xy[None, :], axis=1)
            if np.min(dists) < min_separation_px:
                continue

        selected.append(c)
        selected_xy.append(xy)

    if len(selected) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0), proc

    spots = []
    totals = []

    for c in selected:
        xy_ref = refine_centroid(imgf, c["xy"], roi=refine_roi)
        spots.append(xy_ref)
        totals.append(c["total"])

    spots = np.asarray(spots, dtype=np.float32)
    totals = np.asarray(totals, dtype=np.float32)

    return spots, totals, proc


def detect_all_spots(img):
    """
    Try several thresholds and keep a reasonable detection set.
    """
    records = []

    for pct in THRESHOLD_PERCENTILES:
        spots, totals, proc = detect_spots_at_threshold(
            img,
            threshold_percentile=pct,
            min_area=MIN_AREA,
            max_area=MAX_AREA,
            min_separation_px=MIN_SEPARATION_PX,
            refine_roi=REFINE_ROI,
        )

        print(f"threshold {pct:.1f}%: detected {len(spots)} spots")

        records.append(
            {
                "pct": pct,
                "spots": spots,
                "totals": totals,
                "proc": proc,
                "n": len(spots),
            }
        )

    valid = [r for r in records if 0 < r["n"] <= MAX_REASONABLE_SPOTS]

    if len(valid) == 0:
        best = max(records, key=lambda r: r["n"])
    else:
        best = max(valid, key=lambda r: r["n"])

    print(f"Using threshold percentile: {best['pct']}")

    return best["spots"], best["totals"], best["proc"], best["pct"]


def find_zero_order(spots, totals, img_shape, mode=ZERO_MODE):
    spots = np.asarray(spots, dtype=np.float32)
    totals = np.asarray(totals, dtype=np.float32)

    if len(spots) == 0:
        raise RuntimeError("No spots detected; cannot find zeroth order.")

    if mode == "manual":
        d = np.linalg.norm(spots - MANUAL_ZERO_XY[None, :], axis=1)
        idx = int(np.argmin(d))

    elif mode == "nearest_image_center":
        h, w = img_shape
        center = np.array([w / 2, h / 2], dtype=np.float32)
        d = np.linalg.norm(spots - center[None, :], axis=1)
        idx = int(np.argmin(d))

    elif mode == "brightest":
        idx = int(np.argmax(totals))

    else:
        raise ValueError(f"Unknown ZERO_MODE: {mode}")

    zero_xy = spots[idx]

    keep = np.ones(len(spots), dtype=bool)
    keep[idx] = False

    return zero_xy, idx, keep


# =============================================================================
# PATTERN MATCHING
# =============================================================================

def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fit_similarity(src, dst):
    """
    Fit similarity transform:
        dst = scale * R @ src + t

    Returns:
        M : 2x3 affine matrix
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    X = src - src_mean
    Y = dst - dst_mean

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = np.sum(S) / np.sum(X**2)

    t = dst_mean - scale * (R @ src_mean)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = (scale * R).astype(np.float32)
    M[:, 2] = t.astype(np.float32)

    return M


def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]

    ones = np.ones((len(pts), 1), dtype=np.float32)
    out = np.hstack([pts, ones]) @ np.asarray(M, dtype=np.float32).T

    return out


def initial_scale_from_radii(obs_pts, zero_obs, ideal_pts):
    obs_r = np.linalg.norm(obs_pts - zero_obs[None, :], axis=1)
    ideal_r = np.linalg.norm(ideal_pts, axis=1)

    obs_r = obs_r[obs_r > 2]
    ideal_r = ideal_r[ideal_r > 2]

    if len(obs_r) == 0 or len(ideal_r) == 0:
        return 1.0

    return float(np.median(obs_r) / np.median(ideal_r))


def estimate_rotation_from_angles(
    obs_pts,
    zero_obs,
    ideal_pts,
    ring_ids,
    ideal_angles,
    scale,
    num_scan=720,
):
    """
    Estimate global rotation by assigning each observed spot to nearest
    ideal angular grid on its nearest predetermined ring.
    """
    obs_vec = obs_pts - zero_obs[None, :]
    obs_r = np.linalg.norm(obs_vec, axis=1)
    obs_theta = np.arctan2(obs_vec[:, 1], obs_vec[:, 0])

    ideal_radii = np.asarray(CIRCLE_RADII_PX, dtype=np.float32) * scale

    # Assign observed points to nearest predetermined ring by radius.
    obs_ring = np.argmin(np.abs(obs_r[:, None] - ideal_radii[None, :]), axis=1)

    phis = np.linspace(-np.pi, np.pi, num_scan, endpoint=False)
    scores = []

    for phi in phis:
        err2 = []

        for k in range(len(obs_pts)):
            ring = obs_ring[k]
            inds = np.where(ring_ids == ring)[0]

            if len(inds) == 0:
                continue

            dtheta = angle_wrap(obs_theta[k] - phi - ideal_angles[inds])
            best = np.min(np.abs(dtheta))

            # Convert angular error to approximate pixel error.
            err2.append((obs_r[k] * best) ** 2)

        if len(err2) == 0:
            scores.append(np.inf)
        else:
            scores.append(np.mean(err2))

    best_phi = float(phis[int(np.argmin(scores))])

    return best_phi, obs_ring


def match_observed_to_ideal(
    obs_pts,
    zero_obs,
    ideal_pts,
    ring_ids,
    ideal_angles,
):
    """
    Match observed spots to the predetermined ideal pattern.

    Uses:
        - zeroth order as origin
        - predetermined radii
        - predetermined angular spacing
        - global scale + rotation
    """
    obs_pts = np.asarray(obs_pts, dtype=np.float32)
    zero_obs = np.asarray(zero_obs, dtype=np.float32)

    scale0 = initial_scale_from_radii(obs_pts, zero_obs, ideal_pts)
    phi0, obs_ring = estimate_rotation_from_angles(
        obs_pts,
        zero_obs,
        ideal_pts,
        ring_ids,
        ideal_angles,
        scale0,
    )

    obs_vec = obs_pts - zero_obs[None, :]
    obs_r = np.linalg.norm(obs_vec, axis=1)
    obs_theta = np.arctan2(obs_vec[:, 1], obs_vec[:, 0])

    candidate_pairs = []

    for obs_i in range(len(obs_pts)):
        ring = obs_ring[obs_i]
        inds = np.where(ring_ids == ring)[0]

        if len(inds) == 0:
            continue

        dtheta = angle_wrap(obs_theta[obs_i] - phi0 - ideal_angles[inds])
        ideal_local = int(np.argmin(np.abs(dtheta)))
        ideal_i = int(inds[ideal_local])

        ideal_r_scaled = np.linalg.norm(ideal_pts[ideal_i]) * scale0
        radial_err = abs(obs_r[obs_i] - ideal_r_scaled)
        angular_err_px = obs_r[obs_i] * abs(dtheta[ideal_local])

        score = radial_err + angular_err_px

        candidate_pairs.append((score, obs_i, ideal_i))

    # Greedy unique matching.
    candidate_pairs = sorted(candidate_pairs, key=lambda x: x[0])

    used_obs = set()
    used_ideal = set()
    matched_obs = []
    matched_ideal = []

    for score, obs_i, ideal_i in candidate_pairs:
        if obs_i in used_obs or ideal_i in used_ideal:
            continue

        used_obs.add(obs_i)
        used_ideal.add(ideal_i)

        matched_obs.append(obs_i)
        matched_ideal.append(ideal_i)

    matched_obs = np.asarray(matched_obs, dtype=np.int32)
    matched_ideal = np.asarray(matched_ideal, dtype=np.int32)

    # Add zeroth order as an anchor.
    src = np.vstack(
        [
            np.array([[0.0, 0.0]], dtype=np.float32),
            ideal_pts[matched_ideal],
        ]
    )
    dst = np.vstack(
        [
            zero_obs[None, :],
            obs_pts[matched_obs],
        ]
    )

    M_similarity = fit_similarity(src, dst)

    pred_matched = apply_affine(M_similarity, ideal_pts[matched_ideal])
    residuals = obs_pts[matched_obs] - pred_matched
    residual_norm = np.linalg.norm(residuals, axis=1)

    pred_all = apply_affine(M_similarity, ideal_pts)

    match = {
        "zero_obs": zero_obs,
        "matched_obs_indices": matched_obs,
        "matched_ideal_indices": matched_ideal,
        "M_similarity": M_similarity,
        "pred_all": pred_all,
        "pred_matched": pred_matched,
        "residuals": residuals,
        "residual_norm": residual_norm,
        "scale_initial": np.asarray(scale0, dtype=np.float32),
        "rotation_initial_rad": np.asarray(phi0, dtype=np.float32),
        "rotation_initial_deg": np.asarray(np.degrees(phi0), dtype=np.float32),
    }

    return match


# =============================================================================
# ABERRATION / CIRCULARITY METRICS
# =============================================================================

def fit_circle(points):
    pts = np.asarray(points, dtype=np.float64)

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy, d = sol
    r = np.sqrt(max(cx**2 + cy**2 + d, 0.0))

    radial = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residuals = radial - r

    return np.array([cx, cy], dtype=np.float32), float(r), residuals.astype(np.float32)


def fit_ellipse_metrics(points):
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 5:
        return np.nan, np.nan, np.nan, np.nan

    ellipse = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    _, axes, angle = ellipse

    major = float(max(axes))
    minor = float(min(axes))

    if major <= 0:
        ellipticity = np.nan
    else:
        ellipticity = float((major - minor) / major)

    return major, minor, ellipticity, float(angle)


def analyze_matched_pattern(obs_pts, ideal_pts, ring_ids, match):
    matched_obs = match["matched_obs_indices"]
    matched_ideal = match["matched_ideal_indices"]
    residual_norm = match["residual_norm"]

    num_rings = len(CIRCLE_RADII_PX)

    ring_count = []
    ring_residual_rms = []
    ring_residual_max = []
    ring_circle_radius = []
    ring_circle_rms = []
    ring_ellipticity = []

    for ring in range(num_rings):
        mask = ring_ids[matched_ideal] == ring
        obs_ring = obs_pts[matched_obs[mask]]
        res_ring = residual_norm[mask]

        ring_count.append(len(obs_ring))

        if len(obs_ring) == 0:
            ring_residual_rms.append(np.nan)
            ring_residual_max.append(np.nan)
            ring_circle_radius.append(np.nan)
            ring_circle_rms.append(np.nan)
            ring_ellipticity.append(np.nan)
            continue

        ring_residual_rms.append(float(np.sqrt(np.mean(res_ring**2))))
        ring_residual_max.append(float(np.max(res_ring)))

        if len(obs_ring) >= 3:
            _, radius, radial_res = fit_circle(obs_ring)
            ring_circle_radius.append(radius)
            ring_circle_rms.append(float(np.sqrt(np.mean(radial_res**2))))
        else:
            ring_circle_radius.append(np.nan)
            ring_circle_rms.append(np.nan)

        if len(obs_ring) >= 5:
            _, _, ell, _ = fit_ellipse_metrics(obs_ring)
            ring_ellipticity.append(ell)
        else:
            ring_ellipticity.append(np.nan)

    results = {
        "ring_count": np.asarray(ring_count, dtype=np.int32),
        "ring_residual_rms": np.asarray(ring_residual_rms, dtype=np.float32),
        "ring_residual_max": np.asarray(ring_residual_max, dtype=np.float32),
        "ring_circle_radius": np.asarray(ring_circle_radius, dtype=np.float32),
        "ring_circle_rms": np.asarray(ring_circle_rms, dtype=np.float32),
        "ring_ellipticity": np.asarray(ring_ellipticity, dtype=np.float32),
        "overall_residual_rms": np.asarray(
            np.sqrt(np.mean(residual_norm**2)), dtype=np.float32
        ),
        "overall_residual_max": np.asarray(np.max(residual_norm), dtype=np.float32),
    }

    return results


def print_summary(match, metrics):
    print("\n=== Predetermined-pattern fit summary ===")
    print("Zeroth order observed [px]:", match["zero_obs"])
    print("Initial scale:", float(match["scale_initial"]))
    print("Initial rotation [deg]:", float(match["rotation_initial_deg"]))
    print("Matched spots:", len(match["matched_obs_indices"]))
    print("Overall residual RMS [px]:", float(metrics["overall_residual_rms"]))
    print("Overall residual max [px]:", float(metrics["overall_residual_max"]))

    print("\n=== Ring summary ===")
    for ring in range(len(metrics["ring_count"])):
        print(
            f"Ring {ring}: "
            f"N={metrics['ring_count'][ring]}, "
            f"fit residual RMS={metrics['ring_residual_rms'][ring]:.2f} px, "
            f"fit residual max={metrics['ring_residual_max'][ring]:.2f} px, "
            f"circle RMS={metrics['ring_circle_rms'][ring]:.2f} px, "
            f"ellipticity={metrics['ring_ellipticity'][ring]:.4f}"
        )


# =============================================================================
# PLOTTING / SAVING
# =============================================================================

def plot_analysis(img, obs_pts, zero_obs, ideal_pts, ring_ids, match, label=SAVE_LABEL):
    fig, ax = plt.subplots(figsize=(8, 6))

    vmin, vmax = np.percentile(img, [1, 99.8])
    ax.imshow(img, cmap=blue_cmap, vmin=vmin, vmax=vmax)

    pred_all = match["pred_all"]
    matched_obs = match["matched_obs_indices"]
    matched_ideal = match["matched_ideal_indices"]

    # Plot all detected non-zero spots.
    ax.scatter(
        obs_pts[:, 0],
        obs_pts[:, 1],
        s=25,
        marker="o",
        facecolors="none",
        edgecolors="white",
        linewidths=0.8,
        label="detected spots",
    )

    # Plot zeroth order.
    ax.scatter(
        [zero_obs[0]],
        [zero_obs[1]],
        s=120,
        marker="x",
        color="red",
        linewidths=2,
        label="zeroth order",
    )

    # Plot predicted ideal pattern after fit.
    ax.scatter(
        pred_all[:, 0],
        pred_all[:, 1],
        s=20,
        marker="+",
        color="yellow",
        label="fitted ideal pattern",
    )

    # Draw residual vectors for matched points.
    for obs_i, ideal_i in zip(matched_obs, matched_ideal):
        p_obs = obs_pts[obs_i]
        p_pred = pred_all[ideal_i]

        ax.plot(
            [p_pred[0], p_obs[0]],
            [p_pred[1], p_obs[1]],
            color="orange",
            linewidth=0.7,
            alpha=0.7,
        )

    ax.set_title(label)
    ax.set_xlabel("ThorCam X [px]")
    ax.set_ylabel("ThorCam Y [px]")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()

    return fig


def save_results(
    img,
    proc,
    all_spots,
    all_totals,
    zero_obs,
    obs_pts,
    zero_ideal,
    ideal_pts,
    ring_ids,
    ideal_angles,
    match,
    metrics,
    threshold_percentile,
    label=SAVE_LABEL,
):
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, label)
    npz_path = str(file_path) + ".npz"

    # np.savez_compressed(
    #     npz_path,
    #     timestamp=np.asarray(timestamp),
    #     label=np.asarray(label),
    #     exposure_s=np.asarray(EXPOSURE_S, dtype=np.float32),
    #     yellow_channel=np.asarray(YELLOW_CHANNEL, dtype=np.int32),
    #     yellow_amp=np.asarray(YELLOW_AMP, dtype=np.float32),
    #     threshold_percentile=np.asarray(threshold_percentile, dtype=np.float32),
    #     img=np.asarray(img),
    #     proc=np.asarray(proc),
    #     all_spots=np.asarray(all_spots, dtype=np.float32),
    #     all_totals=np.asarray(all_totals, dtype=np.float32),
    #     zero_obs=np.asarray(zero_obs, dtype=np.float32),
    #     obs_pts=np.asarray(obs_pts, dtype=np.float32),
    #     zero_ideal=np.asarray(zero_ideal, dtype=np.float32),
    #     ideal_pts=np.asarray(ideal_pts, dtype=np.float32),
    #     ring_ids=np.asarray(ring_ids, dtype=np.int32),
    #     ideal_angles=np.asarray(ideal_angles, dtype=np.float32),
    #     **match,
    #     **metrics,
    # )

    fig = plot_analysis(
        img=img,
        obs_pts=obs_pts,
        zero_obs=zero_obs,
        ideal_pts=ideal_pts,
        ring_ids=ring_ids,
        match=match,
        label=label,
    )

    dm.save_figure(fig, file_path)

    print("\nSaved data:", npz_path)
    print("Saved figure:", file_path)

    return npz_path, fig


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main_circle_analysis():
    kpl.init_kplotlib()

    input(
        "\nMake sure the predetermined circle pattern is displayed. "
        "No calibration will be used. Press Enter to capture..."
    )

    img = capture_thorcam_image_with_yellow(
        exposure=EXPOSURE_S,
        yellow_channel=YELLOW_CHANNEL,
        yellow_amp=YELLOW_AMP,
    )

    print("Image shape:", img.shape)
    print("Image min/max:", np.min(img), np.max(img))

    if USE_CROP:
        x, y, w, h = CROP_XYWH
        img_work = img[y:y + h, x:x + w]
    else:
        img_work = img

    all_spots, all_totals, proc, threshold_used = detect_all_spots(img_work)

    if USE_CROP:
        x, y, _, _ = CROP_XYWH
        all_spots[:, 0] += x
        all_spots[:, 1] += y

    print("Detected total spots:", len(all_spots))

    zero_obs, zero_idx, keep_mask = find_zero_order(
        all_spots,
        all_totals,
        img.shape,
        mode=ZERO_MODE,
    )

    obs_pts = all_spots[keep_mask]

    print("Zeroth-order spot:", zero_obs)
    print("Non-zero observed spots:", len(obs_pts))

    zero_ideal, ideal_pts, ring_ids, ideal_angles = make_ideal_circle_pattern()

    print("Ideal circle spots:", len(ideal_pts))
    print("Expected rings:", len(CIRCLE_RADII_PX))

    match = match_observed_to_ideal(
        obs_pts=obs_pts,
        zero_obs=zero_obs,
        ideal_pts=ideal_pts,
        ring_ids=ring_ids,
        ideal_angles=ideal_angles,
    )

    metrics = analyze_matched_pattern(
        obs_pts=obs_pts,
        ideal_pts=ideal_pts,
        ring_ids=ring_ids,
        match=match,
    )

    print_summary(match, metrics)

    npz_path, fig = save_results(
        img=img,
        proc=proc,
        all_spots=all_spots,
        all_totals=all_totals,
        zero_obs=zero_obs,
        obs_pts=obs_pts,
        zero_ideal=zero_ideal,
        ideal_pts=ideal_pts,
        ring_ids=ring_ids,
        ideal_angles=ideal_angles,
        match=match,
        metrics=metrics,
        threshold_percentile=threshold_used,
        label=SAVE_LABEL,
    )

    plt.show(block=True)

    return {
        "img": img,
        "all_spots": all_spots,
        "zero_obs": zero_obs,
        "obs_pts": obs_pts,
        "ideal_pts": ideal_pts,
        "match": match,
        "metrics": metrics,
        "npz_path": npz_path,
    }
    


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()

    # Circle-pattern aberration analysis.
    # main_circle_analysis()

    # Simple yellow/camera test only:
    do_thorcam_sample_image_with_yellow(
        label="thorcam-sample-test",
        exposure=0.0001,
        yellow_channel=7,
        yellow_amp=0.04,
        wait_before_cleanup=True,
    )