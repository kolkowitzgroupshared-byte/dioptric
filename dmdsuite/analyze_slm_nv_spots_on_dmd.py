# -*- coding: utf-8 -*-
"""
Take or reload ThorCam image of first-200 SLM spots, detect spots directly
from the image, convert spot widths/pitch to DMD units, and save everything
using data_manager.
"""

import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from slmsuite.hardware.cameras.thorlabs import ThorCam
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# User settings
# =============================================================================

# RUN_MODE = "take_new"
RUN_MODE = "load_saved"

INPUT_FILE_STEM = "2026_06_09-19_44_52-first-200-detected-spots-dmd-radius"

DMD_AFFINE_CALIB_PATH = "dmdsuite/calibration/triangle_affine_onpass.npz"

OUTPUT_LABEL = "first-200-detected-spots-dmd-radius"

EXPECTED_N_SPOTS = 200

YELLOW_CHANNEL = 7
YELLOW_AMP = 0.08
THORCAM_SERIAL = "26438"
EXPOSURE = 0.0001

ROI_XYWH = None
WAIT_BEFORE_CLEANUP = True


blue_cmap = LinearSegmentedColormap.from_list(
    "white_to_C0",
    ["white", "#1f77b4"],
)


# =============================================================================
# Basic helpers
# =============================================================================

def clean_file_stem(file_stem_or_path):
    p = Path(str(file_stem_or_path))
    if p.suffix in [".txt", ".npz"]:
        return p.stem
    return str(file_stem_or_path)


def load_saved_data_dm(file_stem_or_path):
    file_stem = clean_file_stem(file_stem_or_path)

    print("\n=== Loading saved data with dm.get_raw_data ===")
    print("file_stem:", file_stem)

    raw_data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
        allow_pickle=True,
    )

    print("Loaded keys:")
    for key in sorted(raw_data.keys()):
        print("  ", key)

    return raw_data


def load_dmd_affine_calibration(calib_path=None):
    if calib_path is None:
        calib_path = DMD_AFFINE_CALIB_PATH

    with np.load(calib_path, allow_pickle=False) as npz:
        M_cam_to_dmd = np.asarray(npz["M_cam_to_dmd"], dtype=np.float32)
        zero_cam_xy = np.asarray(npz["zero_cam_xy"], dtype=np.float32).reshape(2)
        zero_dmd_xy = np.asarray(npz["zero_dmd_xy"], dtype=np.float32).reshape(2)

    print("\n=== Loaded DMD calibration ===")
    print("M_cam_to_dmd:")
    print(M_cam_to_dmd)
    print("zero_cam_xy:", zero_cam_xy)
    print("zero_dmd_xy:", zero_dmd_xy)

    return M_cam_to_dmd, zero_cam_xy, zero_dmd_xy


def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=np.float32)

    single = False
    if pts.ndim == 1:
        pts = pts[None, :]
        single = True

    ones = np.ones((len(pts), 1), dtype=np.float32)
    out = np.hstack([pts, ones]) @ np.asarray(M, dtype=np.float32).T

    if single:
        return out[0]

    return out


def nearest_neighbor_pitch(points):
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 2:
        return np.asarray([], dtype=np.float32)

    d = np.linalg.norm(
        pts[:, None, :] - pts[None, :, :],
        axis=2,
    )

    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1).astype(np.float32)


# =============================================================================
# Camera acquisition
# =============================================================================

def safe_get_image(cam, exposure=0.0001, tries=200, delay_s=0.005):
    cam.set_exposure(exposure)

    for _ in range(tries):
        img = cam.get_image()
        if img is not None:
            return img
        time.sleep(delay_s)

    raise RuntimeError("Camera returned None after multiple attempts.")


def do_thorcam_hardware_roi_with_yellow(
    label="thorcam-yellow-image",
    exposure=0.0001,
    yellow_channel=7,
    yellow_amp=0.08,
    roi_xywh=None,
    wait_before_cleanup=True,
):
    cxn = None
    opx = None
    cam = None

    try:
        cxn = common.labrad_connect()
        opx = cxn.QM_opx

        opx.constant_ac([], [yellow_channel], [yellow_amp], [0])
        time.sleep(0.2)

        cam = ThorCam(serial=THORCAM_SERIAL, verbose=True)

        if roi_xywh is not None:
            x, y, w, h = roi_xywh
            x, y, w, h = int(x), int(y), int(w), int(h)

            cam.set_woi((x, w, y, h))

            image_origin_xy = np.array([x, y], dtype=np.float32)
            save_roi_xywh = np.array([x, y, w, h], dtype=np.int32)
            x_label = "ROI x [px]"
            y_label = "ROI y [px]"
        else:
            image_origin_xy = np.array([0, 0], dtype=np.float32)
            save_roi_xywh = np.array([-1, -1, -1, -1], dtype=np.int32)
            x_label = "Camera x [px]"
            y_label = "Camera y [px]"

        img = safe_get_image(cam, exposure=exposure)

        img_f = img.astype(np.float32)
        bg = np.percentile(img_f, 20)
        weights = np.clip(img_f - bg, 0, None)

        if weights.sum() > 0:
            yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            cx = image_origin_xy[0] + np.sum(xx * weights) / weights.sum()
            cy = image_origin_xy[1] + np.sum(yy * weights) / weights.sum()
            centroid_xy = np.array([cx, cy], dtype=np.float32)
        else:
            centroid_xy = np.array([np.nan, np.nan], dtype=np.float32)

        print("\n=== ThorCam readout ===")
        print("roi_xywh:", None if roi_xywh is None else tuple(save_roi_xywh))
        print("image shape:", img.shape)
        print("sum:", float(np.sum(img)))
        print("mean:", float(np.mean(img)))
        print("max:", float(np.max(img)))
        print("centroid_xy full camera:", centroid_xy)

        timestamp = dm.get_time_stamp()
        file_path = dm.get_file_path(__file__, timestamp, label)

        raw_data = {
            "timestamp": timestamp,
            "experiment": "thorcam_yellow_image",
            "label": label,
            "roi_xywh": save_roi_xywh.tolist(),
            "image_shape": list(img.shape),
            "img_sum": float(np.sum(img)),
            "img_mean": float(np.mean(img)),
            "img_max": float(np.max(img)),
            "centroid_xy": centroid_xy.tolist(),
            "exposure": float(exposure),
            "yellow_channel": int(yellow_channel),
            "yellow_amp": float(yellow_amp),
            "img": np.asarray(img),
        }

        dm.save_raw_data(
            raw_data,
            file_path,
            keys_to_compress=["img"],
        )

        fig, ax = plt.subplots(figsize=(8, 7))
        vmin, vmax = np.percentile(img, [0, 100.0])

        im = ax.imshow(
            img,
            cmap=blue_cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_title(label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, label="counts")
        fig.tight_layout()

        dm.save_figure(fig, file_path)

        print("Saved ThorCam image with dm:")
        print(file_path)

        plt.show(block=False)

        if wait_before_cleanup:
            input("Press Enter to turn off yellow...")

        return img, centroid_xy, str(file_path)

    finally:
        try:
            if opx is not None:
                opx.constant_ac([], [yellow_channel], [0.0], [0])
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
# Spot detection and measurement
# =============================================================================

def preprocess_for_spots(img, bg_sigma=25):
    imgf = np.asarray(img, dtype=np.float32)

    bg = cv2.GaussianBlur(
        imgf,
        (0, 0),
        sigmaX=bg_sigma,
        sigmaY=bg_sigma,
    )

    proc = imgf - bg
    proc -= np.percentile(proc, 5)

    return np.clip(proc, 0, None).astype(np.float32)


def detect_spots_from_image(
    img,
    threshold_percentiles=(99.8, 99.5, 99.2, 98.8, 98.0, 97.0, 96.0),
    expected_n=200,
    min_area=2,
    max_area=5000,
    min_separation_px=6,
    zero_cam_xy=None,
    zero_exclusion_radius=40,
):
    imgf = np.asarray(img, dtype=np.float32)
    proc = preprocess_for_spots(imgf)

    best = None

    for pct in threshold_percentiles:
        thresh = np.percentile(proc, pct)
        mask = (proc >= thresh).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        candidates = []

        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]

            if area < min_area or area > max_area:
                continue

            xy0 = np.asarray(centroids[lab], dtype=np.float32)

            if zero_cam_xy is not None:
                if np.linalg.norm(xy0 - zero_cam_xy) < zero_exclusion_radius:
                    continue

            ys, xs = np.where(labels == lab)
            vals = proc[ys, xs]
            total = float(np.sum(vals))

            candidates.append(
                {
                    "xy": xy0,
                    "total": total,
                    "label": int(lab),
                    "area": int(area),
                }
            )

        candidates = sorted(candidates, key=lambda c: c["total"], reverse=True)

        selected = []
        selected_xy = []

        for c in candidates:
            xy = c["xy"]

            if len(selected_xy) > 0:
                d = np.linalg.norm(
                    np.asarray(selected_xy, dtype=np.float32) - xy[None, :],
                    axis=1,
                )

                if np.min(d) < min_separation_px:
                    continue

            selected.append(c)
            selected_xy.append(xy)

        n = len(selected)
        score = abs(n - expected_n) if expected_n is not None else -n

        print(f"threshold {pct:.1f}%: detected {n} spots")

        if best is None or score < best["score"]:
            best = {
                "pct": float(pct),
                "selected": selected,
                "labels": labels,
                "proc": proc,
                "score": score,
                "n_detected": n,
            }

    print("Using threshold:", best["pct"])
    print("Detected spots:", best["n_detected"])

    return best["selected"], best["labels"], best["proc"], best["pct"]


def measure_component_widths(
    img,
    proc,
    selected,
    labels,
    M_cam_to_dmd,
):
    imgf = np.asarray(img, dtype=np.float32)
    A = np.asarray(M_cam_to_dmd[:, :2], dtype=np.float32)

    measured = []

    for i, c in enumerate(selected):
        lab = int(c["label"])
        ys, xs = np.where(labels == lab)

        if len(xs) < 2:
            continue

        vals = proc[ys, xs].astype(np.float32)
        vals = np.clip(vals, 0, None)

        if vals.sum() <= 0:
            continue

        norm = vals.sum()

        cx = np.sum(xs * vals) / norm
        cy = np.sum(ys * vals) / norm

        dx = xs - cx
        dy = ys - cy

        cov_cam = np.array(
            [
                [
                    np.sum(vals * dx * dx) / norm,
                    np.sum(vals * dx * dy) / norm,
                ],
                [
                    np.sum(vals * dx * dy) / norm,
                    np.sum(vals * dy * dy) / norm,
                ],
            ],
            dtype=np.float32,
        )

        evals_cam, _ = np.linalg.eigh(cov_cam)
        evals_cam = np.clip(evals_cam, 0, None)

        sigma_minor_cam, sigma_major_cam = np.sqrt(evals_cam)

        cov_dmd = A @ cov_cam @ A.T

        evals_dmd, _ = np.linalg.eigh(cov_dmd)
        evals_dmd = np.clip(evals_dmd, 0, None)

        sigma_minor_dmd, sigma_major_dmd = np.sqrt(evals_dmd)

        cam_xy = np.array([cx, cy], dtype=np.float32)
        dmd_xy = apply_affine(M_cam_to_dmd, cam_xy)

        measured.append(
            {
                "index": int(i),
                "thorcam_centroid": cam_xy,
                "dmd_centroid": dmd_xy,
                "sum": float(norm),
                "peak": float(np.max(imgf[ys, xs])),
                "area": int(len(xs)),
                "sigma_major_cam": float(sigma_major_cam),
                "sigma_minor_cam": float(sigma_minor_cam),
                "fwhm_major_cam": float(2.355 * sigma_major_cam),
                "fwhm_minor_cam": float(2.355 * sigma_minor_cam),
                "sigma_major_dmd": float(sigma_major_dmd),
                "sigma_minor_dmd": float(sigma_minor_dmd),
                "fwhm_major_dmd": float(2.355 * sigma_major_dmd),
                "fwhm_minor_dmd": float(2.355 * sigma_minor_dmd),
            }
        )

    return measured


def measured_from_saved_arrays(raw_data):
    measured_thorcam = np.asarray(raw_data["measured_thorcam"], dtype=np.float32)
    measured_dmd = np.asarray(raw_data["measured_dmd"], dtype=np.float32)

    fwhm_major_dmd = np.asarray(raw_data["fwhm_major_dmd"], dtype=np.float32)
    fwhm_minor_dmd = np.asarray(raw_data["fwhm_minor_dmd"], dtype=np.float32)

    if "spot_sums" in raw_data:
        spot_sums = np.asarray(raw_data["spot_sums"], dtype=np.float32)
    else:
        spot_sums = np.ones(len(measured_dmd), dtype=np.float32)

    if "fwhm_major_cam" in raw_data:
        fwhm_major_cam = np.asarray(raw_data["fwhm_major_cam"], dtype=np.float32)
    else:
        fwhm_major_cam = np.full(len(measured_dmd), np.nan, dtype=np.float32)

    if "fwhm_minor_cam" in raw_data:
        fwhm_minor_cam = np.asarray(raw_data["fwhm_minor_cam"], dtype=np.float32)
    else:
        fwhm_minor_cam = np.full(len(measured_dmd), np.nan, dtype=np.float32)

    measured = []

    for i in range(len(measured_dmd)):
        measured.append(
            {
                "index": int(i),
                "thorcam_centroid": measured_thorcam[i],
                "dmd_centroid": measured_dmd[i],
                "sum": float(spot_sums[i]),
                "peak": np.nan,
                "area": -1,
                "fwhm_major_cam": float(fwhm_major_cam[i]),
                "fwhm_minor_cam": float(fwhm_minor_cam[i]),
                "sigma_major_dmd": float(fwhm_major_dmd[i] / 2.355),
                "sigma_minor_dmd": float(fwhm_minor_dmd[i] / 2.355),
                "fwhm_major_dmd": float(fwhm_major_dmd[i]),
                "fwhm_minor_dmd": float(fwhm_minor_dmd[i]),
            }
        )

    return measured


# =============================================================================
# Summary
# =============================================================================

def summarize_dmd_radius_and_pitch(measured):
    dmd_centers = np.asarray(
        [m["dmd_centroid"] for m in measured],
        dtype=np.float32,
    )

    fwhm_major = np.asarray(
        [m["fwhm_major_dmd"] for m in measured],
        dtype=np.float32,
    )

    fwhm_minor = np.asarray(
        [m["fwhm_minor_dmd"] for m in measured],
        dtype=np.float32,
    )

    sigma_major = np.asarray(
        [m["sigma_major_dmd"] for m in measured],
        dtype=np.float32,
    )

    nn_pitch = nearest_neighbor_pitch(dmd_centers)

    radius_95 = 2.45 * sigma_major
    radius_99 = 3.03 * sigma_major

    safe_radius_from_pitch = 0.4 * np.nanpercentile(nn_pitch, 5)

    recommended_radius = min(
        float(np.nanpercentile(radius_99, 90)),
        float(safe_radius_from_pitch),
    )

    print("\n=== DMD spot-size summary ===")
    print("Measured spots:", len(measured))
    print("DMD FWHM major median:", float(np.nanmedian(fwhm_major)))
    print("DMD FWHM minor median:", float(np.nanmedian(fwhm_minor)))
    print("DMD FWHM major 90%:", float(np.nanpercentile(fwhm_major, 90)))
    print("DMD nearest-neighbor pitch min:", float(np.nanmin(nn_pitch)))
    print("DMD nearest-neighbor pitch 5%:", float(np.nanpercentile(nn_pitch, 5)))
    print("DMD nearest-neighbor pitch median:", float(np.nanmedian(nn_pitch)))

    print("\nRecommended DMD radius estimates:")
    print("median radius for ~95% power:", float(np.nanmedian(radius_95)))
    print("median radius for ~99% power:", float(np.nanmedian(radius_99)))
    print("90% radius for ~99% power:", float(np.nanpercentile(radius_99, 90)))
    print("safe radius from pitch, 0.4 × pitch_5%:", float(safe_radius_from_pitch))
    print("recommended starting DMD radius:", float(recommended_radius))

    return {
        "dmd_centers": dmd_centers,
        "fwhm_major_dmd": fwhm_major,
        "fwhm_minor_dmd": fwhm_minor,
        "sigma_major_dmd": sigma_major,
        "nn_pitch_dmd": nn_pitch,
        "radius_95_dmd": radius_95,
        "radius_99_dmd": radius_99,
        "recommended_radius_dmd": np.asarray(recommended_radius, dtype=np.float32),
    }


# =============================================================================
# Plotting
# =============================================================================

def make_main_detection_plot(img, measured, zero_cam_xy):
    fig, ax = plt.subplots(figsize=(9, 7))

    vmin, vmax = np.percentile(img, [1, 99.99])

    ax.imshow(
        img,
        cmap=blue_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    cam_pts = np.asarray(
        [m["thorcam_centroid"] for m in measured],
        dtype=np.float32,
    )

    ax.scatter(
        cam_pts[:, 0],
        cam_pts[:, 1],
        s=25,
        marker="o",
        facecolors="none",
        edgecolors="red",
        label="detected spots",
    )

    ax.scatter(
        [zero_cam_xy[0]],
        [zero_cam_xy[1]],
        s=100,
        marker="x",
        color="yellow",
        label="zero order",
    )

    ax.set_title("Detected first-200 SLM spots from image")
    ax.set_xlabel("ThorCam x [px]")
    ax.set_ylabel("ThorCam y [px]")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig


def make_extra_scatter_plots(
    measured,
    summary,
    zero_dmd_xy=None,
    label_prefix="first-200-detected-spots",
):
    figs = []

    dmd_pts = np.asarray([m["dmd_centroid"] for m in measured], dtype=np.float32)
    cam_pts = np.asarray([m["thorcam_centroid"] for m in measured], dtype=np.float32)
    spot_sums = np.asarray([m["sum"] for m in measured], dtype=np.float32)

    fwhm_major = np.asarray(summary["fwhm_major_dmd"], dtype=np.float32)
    fwhm_minor = np.asarray(summary["fwhm_minor_dmd"], dtype=np.float32)
    nn_pitch = np.asarray(summary["nn_pitch_dmd"], dtype=np.float32)
    radius_99 = np.asarray(summary["radius_99_dmd"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(dmd_pts[:, 0], dmd_pts[:, 1], c=fwhm_major, s=25)
    if zero_dmd_xy is not None:
        ax.scatter([zero_dmd_xy[0]], [zero_dmd_xy[1]], s=100, marker="x", label="zero order")
        ax.legend(fontsize=8)
    ax.set_title(f"{label_prefix}: DMD position colored by FWHM major")
    ax.set_xlabel("DMD x [mirrors]")
    ax.set_ylabel("DMD y [mirrors]")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, label="FWHM major [DMD mirrors]")
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(dmd_pts[:, 0], dmd_pts[:, 1], c=nn_pitch, s=25)
    if zero_dmd_xy is not None:
        ax.scatter([zero_dmd_xy[0]], [zero_dmd_xy[1]], s=100, marker="x", label="zero order")
        ax.legend(fontsize=8)
    ax.set_title(f"{label_prefix}: DMD position colored by nearest-neighbor pitch")
    ax.set_xlabel("DMD x [mirrors]")
    ax.set_ylabel("DMD y [mirrors]")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, label="Nearest-neighbor pitch [DMD mirrors]")
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(spot_sums, fwhm_major, s=25, label="major")
    ax.scatter(spot_sums, fwhm_minor, s=25, label="minor")
    ax.set_xlabel("Spot integrated signal [arb.]")
    ax.set_ylabel("FWHM [DMD mirrors]")
    ax.set_title(f"{label_prefix}: signal versus width")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(nn_pitch, radius_99, s=25)
    ax.axline((0, 0), slope=0.5, linestyle="--", label="radius = 0.5 × pitch")
    ax.axline((0, 0), slope=0.4, linestyle="--", label="radius = 0.4 × pitch")
    ax.set_xlabel("Nearest-neighbor pitch [DMD mirrors]")
    ax.set_ylabel("99% Gaussian radius [DMD mirrors]")
    ax.set_title(f"{label_prefix}: radius versus pitch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fwhm_minor, fwhm_major, s=25)
    lim_max = float(np.nanmax([np.nanmax(fwhm_minor), np.nanmax(fwhm_major)]))
    ax.plot([0, lim_max], [0, lim_max], linestyle="--")
    ax.set_xlabel("FWHM minor [DMD mirrors]")
    ax.set_ylabel("FWHM major [DMD mirrors]")
    ax.set_title(f"{label_prefix}: ellipticity check")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(cam_pts[:, 0], cam_pts[:, 1], c=spot_sums, s=25)
    ax.set_xlabel("ThorCam x [px]")
    ax.set_ylabel("ThorCam y [px]")
    ax.set_title(f"{label_prefix}: ThorCam position colored by signal")
    fig.colorbar(sc, ax=ax, label="Spot integrated signal [arb.]")
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(nn_pitch, bins=30)
    ax.axvline(np.nanpercentile(nn_pitch, 5), linestyle="--", label="5%")
    ax.axvline(np.nanmedian(nn_pitch), linestyle="--", label="median")
    ax.set_xlabel("Nearest-neighbor pitch [DMD mirrors]")
    ax.set_ylabel("Number of spots")
    ax.set_title(f"{label_prefix}: nearest-neighbor pitch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(fwhm_major, bins=30, alpha=0.7, label="major")
    ax.hist(fwhm_minor, bins=30, alpha=0.7, label="minor")
    ax.set_xlabel("FWHM [DMD mirrors]")
    ax.set_ylabel("Number of spots")
    ax.set_title(f"{label_prefix}: DMD spot FWHM")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figs.append(fig)

    return figs


# =============================================================================
# Save using data_manager
# =============================================================================

def save_detected_spot_analysis_with_dm(
    img,
    proc,
    threshold_used,
    measured,
    summary,
    M_cam_to_dmd,
    zero_cam_xy,
    zero_dmd_xy,
    main_fig,
    extra_figs,
    label=None,
):
    if label is None:
        label = OUTPUT_LABEL

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, label)

    cam_pts = np.asarray([m["thorcam_centroid"] for m in measured], dtype=np.float32)
    dmd_pts = np.asarray([m["dmd_centroid"] for m in measured], dtype=np.float32)
    spot_sums = np.asarray([m["sum"] for m in measured], dtype=np.float32)
    peak_vals = np.asarray([m.get("peak", np.nan) for m in measured], dtype=np.float32)
    area_vals = np.asarray([m.get("area", -1) for m in measured], dtype=np.float32)
    fwhm_major_cam = np.asarray([m.get("fwhm_major_cam", np.nan) for m in measured], dtype=np.float32)
    fwhm_minor_cam = np.asarray([m.get("fwhm_minor_cam", np.nan) for m in measured], dtype=np.float32)
    sigma_major_cam = np.asarray([m.get("sigma_major_cam", np.nan) for m in measured], dtype=np.float32)
    sigma_minor_cam = np.asarray([m.get("sigma_minor_cam", np.nan) for m in measured], dtype=np.float32)

    raw_data = {
        "timestamp": timestamp,
        "experiment": "first_200_detected_spots_dmd_radius",
        "label": label,
        "num_measured": int(len(measured)),
        "threshold_used": float(threshold_used),
        "recommended_radius_dmd": float(summary["recommended_radius_dmd"]),
        "median_fwhm_major_dmd": float(np.nanmedian(summary["fwhm_major_dmd"])),
        "median_fwhm_minor_dmd": float(np.nanmedian(summary["fwhm_minor_dmd"])),
        "fwhm_major_90pct_dmd": float(np.nanpercentile(summary["fwhm_major_dmd"], 90)),
        "pitch_min_dmd": float(np.nanmin(summary["nn_pitch_dmd"])),
        "pitch_5pct_dmd": float(np.nanpercentile(summary["nn_pitch_dmd"], 5)),
        "pitch_median_dmd": float(np.nanmedian(summary["nn_pitch_dmd"])),
        "img": np.asarray(img),
        "proc": np.asarray(proc),
        "measured_thorcam": cam_pts,
        "measured_dmd": dmd_pts,
        "spot_sums": spot_sums,
        "peak_vals": peak_vals,
        "area_vals": area_vals,
        "fwhm_major_cam": fwhm_major_cam,
        "fwhm_minor_cam": fwhm_minor_cam,
        "sigma_major_cam": sigma_major_cam,
        "sigma_minor_cam": sigma_minor_cam,
        "fwhm_major_dmd": summary["fwhm_major_dmd"],
        "fwhm_minor_dmd": summary["fwhm_minor_dmd"],
        "sigma_major_dmd": summary["sigma_major_dmd"],
        "nn_pitch_dmd": summary["nn_pitch_dmd"],
        "radius_95_dmd": summary["radius_95_dmd"],
        "radius_99_dmd": summary["radius_99_dmd"],
        "M_cam_to_dmd": np.asarray(M_cam_to_dmd, dtype=np.float32),
        "zero_cam_xy": np.asarray(zero_cam_xy, dtype=np.float32),
        "zero_dmd_xy": np.asarray(zero_dmd_xy, dtype=np.float32),
    }

    dm.save_raw_data(
        raw_data,
        file_path,
        keys_to_compress=[
            "img",
            "proc",
            "measured_thorcam",
            "measured_dmd",
            "spot_sums",
            "peak_vals",
            "area_vals",
            "fwhm_major_cam",
            "fwhm_minor_cam",
            "sigma_major_cam",
            "sigma_minor_cam",
            "fwhm_major_dmd",
            "fwhm_minor_dmd",
            "sigma_major_dmd",
            "nn_pitch_dmd",
            "radius_95_dmd",
            "radius_99_dmd",
            "M_cam_to_dmd",
            "zero_cam_xy",
            "zero_dmd_xy",
        ],
    )

    dm.save_figure(main_fig, file_path)

    for ind, fig in enumerate(extra_figs):
        fig_path = dm.get_file_path(__file__, timestamp, f"{label}-scatter-{ind}")
        dm.save_figure(fig, fig_path)

    print("\nSaved detected-spot analysis with dm:")
    print(file_path)

    return raw_data, file_path


# =============================================================================
# Main analysis pipeline
# =============================================================================

def analyze_image(
    img,
    M_cam_to_dmd,
    zero_cam_xy,
    zero_dmd_xy,
    expected_n=None,
):
    if expected_n is None:
        expected_n = EXPECTED_N_SPOTS

    selected, labels, proc, threshold_used = detect_spots_from_image(
        img,
        expected_n=expected_n,
        min_area=2,
        max_area=5000,
        min_separation_px=6,
        zero_cam_xy=zero_cam_xy,
        zero_exclusion_radius=40,
    )

    measured = measure_component_widths(
        img=img,
        proc=proc,
        selected=selected,
        labels=labels,
        M_cam_to_dmd=M_cam_to_dmd,
    )

    measured = sorted(measured, key=lambda m: m["sum"], reverse=True)[:expected_n]

    summary = summarize_dmd_radius_and_pitch(measured)

    return measured, summary, proc, threshold_used


def run_take_new():
    img, centroid_xy, img_path = do_thorcam_hardware_roi_with_yellow(
        label="first-200-nv-spots-detect-from-image",
        exposure=EXPOSURE,
        yellow_channel=YELLOW_CHANNEL,
        yellow_amp=YELLOW_AMP,
        roi_xywh=ROI_XYWH,
        wait_before_cleanup=WAIT_BEFORE_CLEANUP,
    )

    M_cam_to_dmd, zero_cam_xy, zero_dmd_xy = load_dmd_affine_calibration()

    measured, summary, proc, threshold_used = analyze_image(
        img=img,
        M_cam_to_dmd=M_cam_to_dmd,
        zero_cam_xy=zero_cam_xy,
        zero_dmd_xy=zero_dmd_xy,
    )

    main_fig = make_main_detection_plot(img, measured, zero_cam_xy)

    extra_figs = make_extra_scatter_plots(
        measured=measured,
        summary=summary,
        zero_dmd_xy=zero_dmd_xy,
        label_prefix="first-200-detected-spots",
    )

    raw_data, file_path = save_detected_spot_analysis_with_dm(
        img=img,
        proc=proc,
        threshold_used=threshold_used,
        measured=measured,
        summary=summary,
        M_cam_to_dmd=M_cam_to_dmd,
        zero_cam_xy=zero_cam_xy,
        zero_dmd_xy=zero_dmd_xy,
        main_fig=main_fig,
        extra_figs=extra_figs,
        label=OUTPUT_LABEL,
    )

    return {
        "raw_data": raw_data,
        "file_path": file_path,
        "measured": measured,
        "summary": summary,
        "figs": [main_fig] + extra_figs,
    }


def run_load_saved():
    saved = load_saved_data_dm(INPUT_FILE_STEM)

    if "img" not in saved:
        raise KeyError("Saved file does not contain key 'img'.")

    img = np.asarray(saved["img"], dtype=np.float32)

    M_cam_to_dmd, zero_cam_xy, zero_dmd_xy = load_dmd_affine_calibration()

    if "measured_thorcam" in saved and "measured_dmd" in saved:
        print("\nUsing saved measured arrays.")
        measured = measured_from_saved_arrays(saved)

        if "proc" in saved:
            proc = np.asarray(saved["proc"], dtype=np.float32)
        else:
            proc = preprocess_for_spots(img)

        if "threshold_used" in saved:
            threshold_used = float(np.asarray(saved["threshold_used"]).item())
        else:
            threshold_used = np.nan

        summary = summarize_dmd_radius_and_pitch(measured)

    else:
        print("\nSaved file has no measured arrays. Re-detecting from image.")

        measured, summary, proc, threshold_used = analyze_image(
            img=img,
            M_cam_to_dmd=M_cam_to_dmd,
            zero_cam_xy=zero_cam_xy,
            zero_dmd_xy=zero_dmd_xy,
        )

    main_fig = make_main_detection_plot(img, measured, zero_cam_xy)

    extra_figs = make_extra_scatter_plots(
        measured=measured,
        summary=summary,
        zero_dmd_xy=zero_dmd_xy,
        label_prefix="first-200-detected-spots-reanalysis",
    )

    raw_data, file_path = save_detected_spot_analysis_with_dm(
        img=img,
        proc=proc,
        threshold_used=threshold_used,
        measured=measured,
        summary=summary,
        M_cam_to_dmd=M_cam_to_dmd,
        zero_cam_xy=zero_cam_xy,
        zero_dmd_xy=zero_dmd_xy,
        main_fig=main_fig,
        extra_figs=extra_figs,
        label=f"{OUTPUT_LABEL}-reanalysis",
    )

    return {
        "raw_data": raw_data,
        "file_path": file_path,
        "measured": measured,
        "summary": summary,
        "figs": [main_fig] + extra_figs,
    }


def expected_diffraction_spot_size(
    wavelength_um=0.589,
    lens_f_mm=180.0,
    beam_diameter_mm=5.0,
    dmd_pitch_um=7.56,
):
    """
    Estimate diffraction-limited spot size from lens focal length and beam diameter.

    Assumptions:
        beam_diameter_mm is approximately the 1/e^2 Gaussian beam diameter.
        wavelength_um is the optical wavelength.
        dmd_pitch_um is DMD mirror pitch.

    Returns sizes in microns and DMD mirror units.
    """
    lam_um = float(wavelength_um)
    f_um = float(lens_f_mm) * 1000.0
    D_um = float(beam_diameter_mm) * 1000.0

    NA = D_um / (2.0 * f_um)

    # Gaussian focused beam, 1/e^2 radius.
    gaussian_w0_radius_um = 2.0 * lam_um * f_um / (np.pi * D_um)

    # Intensity FWHM diameter for Gaussian I = I0 exp(-2 r^2 / w0^2).
    gaussian_fwhm_diameter_um = 1.17741 * gaussian_w0_radius_um

    # Airy estimates for circular aperture.
    airy_fwhm_diameter_um = 1.03 * lam_um * f_um / D_um
    airy_first_zero_diameter_um = 2.44 * lam_um * f_um / D_um

    rayleigh_range_um = np.pi * gaussian_w0_radius_um**2 / lam_um

    out = {
        "wavelength_um": lam_um,
        "lens_f_mm": float(lens_f_mm),
        "beam_diameter_mm": float(beam_diameter_mm),
        "NA": float(NA),

        "gaussian_w0_radius_um": float(gaussian_w0_radius_um),
        "gaussian_1e2_diameter_um": float(2.0 * gaussian_w0_radius_um),
        "gaussian_fwhm_diameter_um": float(gaussian_fwhm_diameter_um),

        "airy_fwhm_diameter_um": float(airy_fwhm_diameter_um),
        "airy_first_zero_diameter_um": float(airy_first_zero_diameter_um),

        "rayleigh_range_um": float(rayleigh_range_um),
        "confocal_parameter_um": float(2.0 * rayleigh_range_um),

        "gaussian_fwhm_dmd_mirrors": float(gaussian_fwhm_diameter_um / dmd_pitch_um),
        "airy_fwhm_dmd_mirrors": float(airy_fwhm_diameter_um / dmd_pitch_um),
        "airy_first_zero_dmd_mirrors": float(airy_first_zero_diameter_um / dmd_pitch_um),
    }

    print("\n=== Expected diffraction-limited spot size ===")
    print("wavelength [um]:", out["wavelength_um"])
    print("lens focal length [mm]:", out["lens_f_mm"])
    print("beam diameter [mm]:", out["beam_diameter_mm"])
    print("NA:", out["NA"])
    print("Gaussian 1/e^2 radius [um]:", out["gaussian_w0_radius_um"])
    print("Gaussian FWHM diameter [um]:", out["gaussian_fwhm_diameter_um"])
    print("Airy FWHM diameter [um]:", out["airy_fwhm_diameter_um"])
    print("Airy first-zero diameter [um]:", out["airy_first_zero_diameter_um"])
    print("Gaussian FWHM [DMD mirrors]:", out["gaussian_fwhm_dmd_mirrors"])
    print("Airy FWHM [DMD mirrors]:", out["airy_fwhm_dmd_mirrors"])
    print("Airy first-zero [DMD mirrors]:", out["airy_first_zero_dmd_mirrors"])
    print("Rayleigh range [um]:", out["rayleigh_range_um"])

    return out

import numpy as np


def dmd_spot_and_scaling_summary(
    # ----------------------------
    # Diffraction estimate inputs
    # ----------------------------
    wavelength_um=0.589,
    lens_focal_length_mm=180.0,
    beam_diameter_mm=4.0,
    dmd_mirror_pitch_um=7.56,

    # ----------------------------
    # Experimental measured values
    # ----------------------------
    measured_spots=198,
    fwhm_major_median_dmd=2.691753387451172,
    fwhm_minor_median_dmd=2.325435161590576,
    fwhm_major_90pct_dmd=2.978878974914551,
    nn_pitch_min_dmd=19.559860229492188,
    nn_pitch_5pct_dmd=19.771042346954346,
    nn_pitch_median_dmd=20.380401611328125,
    radius_95_dmd=2.800337791442871,
    radius_99_dmd=3.4632749557495117,
    radius_99_90pct_dmd=3.832697629928589,
    safe_radius_pitch_fraction=0.4,

    # ----------------------------
    # Diamond / microscope path
    # ----------------------------
    diamond_nv_pitch_um=2.6,
    objective_mag=60.0,
    relay_f1_mm=300.0,
    relay_f2_mm=500.0,
):
    """
    Summarize DMD spot size, DMD pitch, and effective magnification.

    Optical path considered:
        SLM/readout path
        -> 180 mm lens
        -> DMD at focal plane
        -> retroreflected through same path
        -> 300 mm lens
        -> 500 mm lens
        -> 60x, 0.95 NA objective
        -> diamond

    Notes:
        The measured DMD pitch gives the effective diamond-to-DMD scaling directly.
        The 300/500 relay is compared as a possible image relay, but the measured
        scaling determines the actual effective magnification.
    """

    # ------------------------------------------------------------------
    # 1. Diffraction-limited spot size at DMD
    # ------------------------------------------------------------------
    lens_focal_length_um = lens_focal_length_mm * 1000.0
    beam_diameter_um = beam_diameter_mm * 1000.0

    # Small-angle NA estimate for beam focused by the 180 mm lens.
    NA = beam_diameter_mm / (2.0 * lens_focal_length_mm)

    # Gaussian estimate.
    gaussian_w0_um = wavelength_um / (np.pi * NA)  # 1/e^2 radius
    gaussian_fwhm_diameter_um = 1.1774100225154747 * gaussian_w0_um

    # Airy estimates.
    airy_first_zero_radius_um = 1.22 * wavelength_um * lens_focal_length_um / beam_diameter_um
    airy_first_zero_diameter_um = 2.0 * airy_first_zero_radius_um
    airy_fwhm_diameter_um = 1.03 * wavelength_um * lens_focal_length_um / beam_diameter_um

    rayleigh_range_um = np.pi * gaussian_w0_um**2 / wavelength_um

    gaussian_fwhm_dmd = gaussian_fwhm_diameter_um / dmd_mirror_pitch_um
    airy_fwhm_dmd = airy_fwhm_diameter_um / dmd_mirror_pitch_um
    airy_first_zero_dmd = airy_first_zero_diameter_um / dmd_mirror_pitch_um

    # ------------------------------------------------------------------
    # 2. Experimental DMD pitch and spot size
    # ------------------------------------------------------------------
    measured_pitch_um = nn_pitch_median_dmd * dmd_mirror_pitch_um
    pitch_min_um = nn_pitch_min_dmd * dmd_mirror_pitch_um
    pitch_5pct_um = nn_pitch_5pct_dmd * dmd_mirror_pitch_um

    fwhm_major_median_um = fwhm_major_median_dmd * dmd_mirror_pitch_um
    fwhm_minor_median_um = fwhm_minor_median_dmd * dmd_mirror_pitch_um
    fwhm_major_90pct_um = fwhm_major_90pct_dmd * dmd_mirror_pitch_um

    # ------------------------------------------------------------------
    # 3. Effective magnification from measured DMD pitch
    # ------------------------------------------------------------------
    measured_mag = measured_pitch_um / diamond_nv_pitch_um
    inferred_relay_factor = measured_mag / objective_mag

    relay_mag_500_over_300 = relay_f2_mm / relay_f1_mm
    relay_mag_300_over_500 = relay_f1_mm / relay_f2_mm

    expected_mag_objective_only = objective_mag
    expected_mag_500_over_300 = objective_mag * relay_mag_500_over_300
    expected_mag_300_over_500 = objective_mag * relay_mag_300_over_500

    expected_pitch_objective_only_um = diamond_nv_pitch_um * expected_mag_objective_only
    expected_pitch_500_over_300_um = diamond_nv_pitch_um * expected_mag_500_over_300
    expected_pitch_300_over_500_um = diamond_nv_pitch_um * expected_mag_300_over_500

    expected_pitch_objective_only_dmd = expected_pitch_objective_only_um / dmd_mirror_pitch_um
    expected_pitch_500_over_300_dmd = expected_pitch_500_over_300_um / dmd_mirror_pitch_um
    expected_pitch_300_over_500_dmd = expected_pitch_300_over_500_um / dmd_mirror_pitch_um

    error_vs_objective_only_pct = 100.0 * (
        measured_mag - expected_mag_objective_only
    ) / expected_mag_objective_only

    error_vs_500_over_300_pct = 100.0 * (
        measured_mag - expected_mag_500_over_300
    ) / expected_mag_500_over_300

    error_vs_300_over_500_pct = 100.0 * (
        measured_mag - expected_mag_300_over_500
    ) / expected_mag_300_over_500

    # ------------------------------------------------------------------
    # 4. Map measured DMD spot sizes back to diamond plane
    # ------------------------------------------------------------------
    measured_fwhm_major_diamond_um = fwhm_major_median_um / measured_mag
    measured_fwhm_minor_diamond_um = fwhm_minor_median_um / measured_mag
    measured_fwhm_major_90pct_diamond_um = fwhm_major_90pct_um / measured_mag

    diffraction_gaussian_fwhm_diamond_um = gaussian_fwhm_diameter_um / measured_mag
    diffraction_airy_fwhm_diamond_um = airy_fwhm_diameter_um / measured_mag
    diffraction_airy_first_zero_diamond_um = airy_first_zero_diameter_um / measured_mag

    safe_radius_from_pitch_dmd = safe_radius_pitch_fraction * nn_pitch_5pct_dmd

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n=== Diffraction-limited DMD spot estimate ===")
    print(f"wavelength [um]: {wavelength_um:.4f}")
    print(f"180 mm lens focal length [mm]: {lens_focal_length_mm:.1f}")
    print(f"beam diameter on 180 mm lens [mm]: {beam_diameter_mm:.2f}")
    print(f"NA ≈ D/(2f): {NA:.5f}")
    print(f"Gaussian 1/e^2 radius [um]: {gaussian_w0_um:.3f}")
    print(f"Gaussian FWHM diameter [um]: {gaussian_fwhm_diameter_um:.3f}")
    print(f"Airy FWHM diameter [um]: {airy_fwhm_diameter_um:.3f}")
    print(f"Airy first-zero diameter [um]: {airy_first_zero_diameter_um:.3f}")
    print(f"Gaussian FWHM [DMD mirrors]: {gaussian_fwhm_dmd:.3f}")
    print(f"Airy FWHM [DMD mirrors]: {airy_fwhm_dmd:.3f}")
    print(f"Airy first-zero [DMD mirrors]: {airy_first_zero_dmd:.3f}")
    print(f"Rayleigh range [um]: {rayleigh_range_um:.1f}")

    print("\n=== Experimental DMD spot-size summary ===")
    print(f"Measured spots: {measured_spots}")
    print(f"DMD FWHM major median [mirrors]: {fwhm_major_median_dmd:.3f}")
    print(f"DMD FWHM minor median [mirrors]: {fwhm_minor_median_dmd:.3f}")
    print(f"DMD FWHM major 90% [mirrors]: {fwhm_major_90pct_dmd:.3f}")
    print(f"DMD FWHM major median [um]: {fwhm_major_median_um:.3f}")
    print(f"DMD FWHM minor median [um]: {fwhm_minor_median_um:.3f}")

    print("\n=== DMD pitch summary ===")
    print(f"Nearest-neighbor pitch min [mirrors]: {nn_pitch_min_dmd:.3f}")
    print(f"Nearest-neighbor pitch 5% [mirrors]: {nn_pitch_5pct_dmd:.3f}")
    print(f"Nearest-neighbor pitch median [mirrors]: {nn_pitch_median_dmd:.3f}")
    print(f"Nearest-neighbor pitch median [um]: {measured_pitch_um:.3f}")

    print("\n=== Diamond-to-DMD magnification check ===")
    print(f"Diamond NV pitch [um]: {diamond_nv_pitch_um:.3f}")
    print(f"Measured DMD pitch [um]: {measured_pitch_um:.3f}")
    print(f"Inferred diamond-to-DMD magnification: {measured_mag:.3f}x")
    print(f"Objective nominal magnification: {objective_mag:.1f}x")
    print(f"Inferred relay factor relative to 60x objective: {inferred_relay_factor:.3f}")
    print(f"Error vs objective-only 60x: {error_vs_objective_only_pct:.2f}%")

    print("\n=== Comparison with possible relay magnifications ===")
    print(
        f"Objective only expected pitch: "
        f"{expected_pitch_objective_only_um:.2f} um = "
        f"{expected_pitch_objective_only_dmd:.2f} mirrors"
    )
    print(
        f"Objective × 500/300 expected pitch: "
        f"{expected_pitch_500_over_300_um:.2f} um = "
        f"{expected_pitch_500_over_300_dmd:.2f} mirrors "
        f"(error {error_vs_500_over_300_pct:.1f}%)"
    )
    print(
        f"Objective × 300/500 expected pitch: "
        f"{expected_pitch_300_over_500_um:.2f} um = "
        f"{expected_pitch_300_over_500_dmd:.2f} mirrors "
        f"(error {error_vs_300_over_500_pct:.1f}%)"
    )

    print("\n=== DMD radius recommendation ===")
    print(f"Median radius for ~95% power [mirrors]: {radius_95_dmd:.3f}")
    print(f"Median radius for ~99% power [mirrors]: {radius_99_dmd:.3f}")
    print(f"90% radius for ~99% power [mirrors]: {radius_99_90pct_dmd:.3f}")
    print(
        f"Safe radius from pitch, {safe_radius_pitch_fraction:.1f} × pitch_5% "
        f"[mirrors]: {safe_radius_from_pitch_dmd:.3f}"
    )
    print(f"Recommended starting radius [mirrors]: {radius_99_90pct_dmd:.3f}")
    print(f"Recommended starting radius rounded: {int(np.ceil(radius_99_90pct_dmd))} mirrors")

    print("\n=== Spot size mapped back to diamond plane ===")
    print(f"Measured major FWHM median [um on diamond]: {measured_fwhm_major_diamond_um:.3f}")
    print(f"Measured minor FWHM median [um on diamond]: {measured_fwhm_minor_diamond_um:.3f}")
    print(f"Measured major FWHM 90% [um on diamond]: {measured_fwhm_major_90pct_diamond_um:.3f}")
    print(f"Diffraction Gaussian FWHM [um on diamond]: {diffraction_gaussian_fwhm_diamond_um:.3f}")
    print(f"Diffraction Airy FWHM [um on diamond]: {diffraction_airy_fwhm_diamond_um:.3f}")
    print(f"Diffraction Airy first-zero [um on diamond]: {diffraction_airy_first_zero_diamond_um:.3f}")

    return {
        "NA": NA,
        "gaussian_w0_um": gaussian_w0_um,
        "gaussian_fwhm_diameter_um": gaussian_fwhm_diameter_um,
        "airy_fwhm_diameter_um": airy_fwhm_diameter_um,
        "airy_first_zero_diameter_um": airy_first_zero_diameter_um,
        "gaussian_fwhm_dmd": gaussian_fwhm_dmd,
        "airy_fwhm_dmd": airy_fwhm_dmd,
        "airy_first_zero_dmd": airy_first_zero_dmd,
        "rayleigh_range_um": rayleigh_range_um,
        "measured_pitch_um": measured_pitch_um,
        "measured_mag": measured_mag,
        "inferred_relay_factor": inferred_relay_factor,
        "expected_pitch_objective_only_dmd": expected_pitch_objective_only_dmd,
        "expected_pitch_500_over_300_dmd": expected_pitch_500_over_300_dmd,
        "expected_pitch_300_over_500_dmd": expected_pitch_300_over_500_dmd,
        "error_vs_objective_only_pct": error_vs_objective_only_pct,
        "error_vs_500_over_300_pct": error_vs_500_over_300_pct,
        "error_vs_300_over_500_pct": error_vs_300_over_500_pct,
        "safe_radius_from_pitch_dmd": safe_radius_from_pitch_dmd,
        "recommended_starting_radius_dmd": radius_99_90pct_dmd,
        "recommended_starting_radius_rounded_dmd": int(np.ceil(radius_99_90pct_dmd)),
        "measured_fwhm_major_diamond_um": measured_fwhm_major_diamond_um,
        "measured_fwhm_minor_diamond_um": measured_fwhm_minor_diamond_um,
        "diffraction_gaussian_fwhm_diamond_um": diffraction_gaussian_fwhm_diamond_um,
        "diffraction_airy_fwhm_diamond_um": diffraction_airy_fwhm_diamond_um,
    }


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    # kpl.init_kplotlib()

    # try:
    #     if RUN_MODE == "take_new":
    #         result = run_take_new()

    #     elif RUN_MODE == "load_saved":
    #         result = run_load_saved()

    #     else:
    #         raise ValueError("RUN_MODE must be 'take_new' or 'load_saved'.")

    # except Exception:
    #     print("Script failed:")
    #     print(traceback.format_exc())
    #     raise
    
    
    expected_spot = expected_diffraction_spot_size(
        wavelength_um=0.589,
        lens_f_mm=180.0,
        beam_diameter_mm=4.0,
        dmd_pitch_um=7.56,
    )   
    # result = dmd_spot_and_scaling_summary()
    plt.show(block=True)