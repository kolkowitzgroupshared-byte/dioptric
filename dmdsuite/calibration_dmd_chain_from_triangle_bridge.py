# -*- coding: utf-8 -*-
"""
Generate final Nuvu -> ThorCam_DMD -> DMD chain using manual triangle order.

This script assumes:

1. Current SLM calibration is valid:
   slmsuite/calibration/nuvu_to_thorcam_slm.npz

2. DMD-side triangle calibration exists:
   dmdsuite/calibration/triangle_affine_onpass.npz

3. You manually know which DMD camera triangle point corresponds to each
   bridge/Nuvu triangle point.

Manual ordering convention:
    triangle_nuvu[i] corresponds to tri_cam_pts_raw[DMD_TRIANGLE_ORDER[i]]
"""

from __future__ import annotations
import sys
from pathlib import Path
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import common, kplotlib as kpl


# =============================================================================
# Paths
# =============================================================================

NV_COORDS_PATH = "slmsuite/nv_blob_detection/nv_blob_1274nvs_reordered.npz"

NUVU_TO_THORCAM_SLM_PATH = "slmsuite/calibration/nuvu_to_thorcam_slm.npz"

DMD_TRIANGLE_CALIB_PATH = "dmdsuite/calibration/triangle_affine_onpass.npz"

NUVU_TO_THORCAM_DMD_OUT = "dmdsuite/calibration/nuvu_to_thorcam_dmd.npz"

DMD_CHAIN_OUT = "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1274.npz"

DMD_WIDTH = 1920
DMD_HEIGHT = 1080


# =============================================================================
# Manual calibration inputs
# =============================================================================

# This is the OLD bridge triangle in ThorCam_SLM coordinates.
# This should match the SLM triangle you used to create/compare DMD spots.
BRIDGE_TRIANGLE_THORCAM_SLM = np.array(
    [
        [834.90381057, 595.0],
        [575.09618943, 595.0],
        [705.0,        370.0],
    ],
    dtype=np.float32,
)

# MANUAL ORDER.
# Meaning:
#   triangle_nuvu[0] -> tri_cam_pts_raw[DMD_TRIANGLE_ORDER[0]]
#   triangle_nuvu[1] -> tri_cam_pts_raw[DMD_TRIANGLE_ORDER[1]]
#   triangle_nuvu[2] -> tri_cam_pts_raw[DMD_TRIANGLE_ORDER[2]]
#
# Change this manually after checking which DMD camera point corresponds to which.
DMD_TRIANGLE_ORDER = [2, 0, 1]

MAX_ALLOWED_ZERO_ERROR_PX = 20.0


# =============================================================================
# Path helpers
# =============================================================================

def repo_path() -> Path:
    return Path(common.get_repo_path())


def resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_path() / p


def ensure_parent(path_like: str | Path) -> Path:
    p = resolve_path(path_like)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def print_dmd_triangle_calibration():
    dmd_tri = np.load(resolve_path(DMD_TRIANGLE_CALIB_PATH), allow_pickle=True)

    tri_cam_pts_raw = np.asarray(dmd_tri["tri_cam_pts"], dtype=np.float32).reshape(3, 2)
    tri_dmd_pts_raw = np.asarray(dmd_tri["tri_dmd_pts"], dtype=np.float32).reshape(3, 2)
    zero_cam_xy = np.asarray(dmd_tri["zero_cam_xy"], dtype=np.float32).reshape(2)
    zero_dmd_xy = np.asarray(dmd_tri["zero_dmd_xy"], dtype=np.float32).reshape(2)

    order = np.asarray(DMD_TRIANGLE_ORDER, dtype=np.int32)

    print("\n=== Raw DMD camera calibration points ===")
    for i, p in enumerate(tri_cam_pts_raw):
        print(f"raw cam {i}: {p}")

    print("\n=== Raw DMD command points ===")
    for i, p in enumerate(tri_dmd_pts_raw):
        print(f"raw dmd {i}: {p}")

    print("\n=== Manual ordered correspondence ===")
    print("DMD_TRIANGLE_ORDER:", order.tolist())
    for i, raw_i in enumerate(order):
        print(
            f"triangle_nuvu[{i}] -> "
            f"cam raw {raw_i}: {tri_cam_pts_raw[raw_i]} -> "
            f"dmd raw {raw_i}: {tri_dmd_pts_raw[raw_i]}"
        )

    print("\nzero_cam_xy:", zero_cam_xy)
    print("zero_dmd_xy:", zero_dmd_xy)
    
    
# print_dmd_triangle_calibration()    
# sys.exit()

# =============================================================================
# Affine helpers
# =============================================================================

def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)

    single_point = False
    if pts.ndim == 1:
        pts = pts[None, :]
        single_point = True

    pts = pts.reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    out = pts_h @ np.asarray(M, dtype=np.float32).T

    if single_point:
        return out[0]

    return out


def invert_affine(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(M, dtype=np.float32))


def affine_2x3_to_3x3(M: np.ndarray) -> np.ndarray:
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = np.asarray(M, dtype=np.float32)
    return H


def affine_3x3_to_2x3(H: np.ndarray) -> np.ndarray:
    return np.asarray(H, dtype=np.float32)[:2, :]


def compose_affines(M2: np.ndarray, M1: np.ndarray) -> np.ndarray:
    """
    Compose M2 after M1.

    If:
        p_mid = M1(p_in)
        p_out = M2(p_mid)

    then:
        p_out = compose_affines(M2, M1)(p_in)
    """
    H2 = affine_2x3_to_3x3(M2)
    H1 = affine_2x3_to_3x3(M1)
    return affine_3x3_to_2x3(H2 @ H1)


def fit_affine_3pt(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    src_pts = np.asarray(src_pts, dtype=np.float32).reshape(3, 2)
    dst_pts = np.asarray(dst_pts, dtype=np.float32).reshape(3, 2)
    return cv2.getAffineTransform(src_pts, dst_pts).astype(np.float32)


# =============================================================================
# Loaders
# =============================================================================

def load_nv_coords(path: str | Path):
    p = resolve_path(path)

    with np.load(p, allow_pickle=True) as data:
        if "nv_coordinates" not in data.files:
            raise KeyError(f"{p} missing key 'nv_coordinates'.")

        nv_coords = np.asarray(data["nv_coordinates"], dtype=np.float32)

        spot_weights = None
        if "updated_spot_weights" in data.files:
            spot_weights = np.asarray(data["updated_spot_weights"], dtype=np.float32)
        elif "spot_weights" in data.files:
            spot_weights = np.asarray(data["spot_weights"], dtype=np.float32)

        original_global_indices = None
        if "original_global_indices" in data.files:
            original_global_indices = np.asarray(
                data["original_global_indices"],
                dtype=np.int32,
            )

        meta = {
            "source_file": str(p),
            "keys": list(data.files),
        }

    return nv_coords, spot_weights, original_global_indices, meta


def compute_inside_dmd_mask(dmd_points, width=DMD_WIDTH, height=DMD_HEIGHT):
    dmd_points = np.asarray(dmd_points, dtype=np.float32).reshape(-1, 2)

    inside = (
        (dmd_points[:, 0] >= 0)
        & (dmd_points[:, 0] < width)
        & (dmd_points[:, 1] >= 0)
        & (dmd_points[:, 1] < height)
    )

    inside_indices = np.where(inside)[0].astype(np.int32)
    outside_indices = np.where(~inside)[0].astype(np.int32)

    return inside, inside_indices, outside_indices


def get_center_test_indices(dmd_points, n=10):
    dmd_points = np.asarray(dmd_points, dtype=np.float32).reshape(-1, 2)
    _, inside_indices, _ = compute_inside_dmd_mask(dmd_points)

    center = np.array([DMD_WIDTH / 2, DMD_HEIGHT / 2], dtype=np.float32)
    dist = np.linalg.norm(dmd_points[inside_indices] - center, axis=1)

    return inside_indices[np.argsort(dist)[:n]].astype(np.int32)


# =============================================================================
# Diagnostics
# =============================================================================

def report_manual_order(
    triangle_nuvu,
    tri_cam_pts_raw,
    tri_dmd_pts_raw,
    order,
    zero_nuvu,
    zero_cam_xy,
):
    order = np.asarray(order, dtype=np.int32)

    if sorted(order.tolist()) != [0, 1, 2]:
        raise ValueError(
            f"DMD_TRIANGLE_ORDER must be a permutation of [0, 1, 2], got {order}."
        )

    triangle_thorcam_dmd = tri_cam_pts_raw[order]
    triangle_dmd = tri_dmd_pts_raw[order]

    M_nuvu_to_thorcam_dmd = fit_affine_3pt(
        triangle_nuvu,
        triangle_thorcam_dmd,
    )

    pred_zero = apply_affine(M_nuvu_to_thorcam_dmd, zero_nuvu)
    zero_error_px = float(np.linalg.norm(pred_zero - zero_cam_xy))

    print("\n=== Manual DMD triangle order ===")
    print("DMD_TRIANGLE_ORDER:", order.tolist())

    print("\ntriangle_nuvu:")
    print(triangle_nuvu)

    print("\ntri_cam_pts_raw:")
    print(tri_cam_pts_raw)

    print("\ntriangle_thorcam_dmd after manual order:")
    print(triangle_thorcam_dmd)

    print("\ntri_dmd_pts_raw:")
    print(tri_dmd_pts_raw)

    print("\ntriangle_dmd after manual order:")
    print(triangle_dmd)

    print("\nZero check:")
    print("predicted zero_cam_xy:", pred_zero)
    print("actual zero_cam_xy:   ", zero_cam_xy)
    print(f"zero error [px]: {zero_error_px:.3f}")

    if zero_error_px > MAX_ALLOWED_ZERO_ERROR_PX:
        print("\nWARNING:")
        print(
            f"Zero error {zero_error_px:.2f} px is larger than "
            f"{MAX_ALLOWED_ZERO_ERROR_PX:.2f} px."
        )
        print("This may be okay if zero moved, but check manual triangle order carefully.")

    return {
        "order": order,
        "triangle_thorcam_dmd": triangle_thorcam_dmd,
        "triangle_dmd": triangle_dmd,
        "M_nuvu_to_thorcam_dmd": M_nuvu_to_thorcam_dmd,
        "pred_zero": pred_zero,
        "zero_error_px": zero_error_px,
    }


def plot_triangle_correspondence(
    triangle_nuvu,
    triangle_thorcam_slm,
    triangle_thorcam_dmd,
    triangle_dmd,
    zero_nuvu,
    zero_thorcam_slm,
    zero_thorcam_dmd,
    zero_dmd,
    save_path,
):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))

    packs = [
        (triangle_nuvu, zero_nuvu, "Nuvu"),
        (triangle_thorcam_slm, zero_thorcam_slm, "ThorCam_SLM"),
        (triangle_thorcam_dmd, zero_thorcam_dmd, "ThorCam_DMD"),
        (triangle_dmd, zero_dmd, "DMD"),
    ]

    for ax, (pts, zero, title) in zip(axes, packs):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)

        ax.plot(pts[:, 0], pts[:, 1], "o-")

        for i, (x, y) in enumerate(pts):
            ax.text(x + 3, y + 3, str(i))

        if zero is not None:
            zero = np.asarray(zero, dtype=np.float32).reshape(2)
            ax.plot(zero[0], zero[1], "rx", markersize=10)
            ax.text(zero[0] + 3, zero[1] + 3, "zero", color="red")

        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(resolve_path(save_path), dpi=200)
    print("Saved diagnostic plot:", resolve_path(save_path))

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== Loading SLM-side calibration ===")

    slm = np.load(resolve_path(NUVU_TO_THORCAM_SLM_PATH), allow_pickle=True)

    M_nuvu_to_thorcam_slm = np.asarray(
        slm["M_nuvu_to_thorcam_slm"],
        dtype=np.float32,
    )

    if "M_thorcam_slm_to_nuvu" in slm.files:
        M_thorcam_slm_to_nuvu = np.asarray(
            slm["M_thorcam_slm_to_nuvu"],
            dtype=np.float32,
        )
    else:
        M_thorcam_slm_to_nuvu = invert_affine(M_nuvu_to_thorcam_slm)

    zero_nuvu = np.asarray(slm["zero_nuvu"], dtype=np.float32).reshape(2)
    zero_thorcam_slm = np.asarray(
        slm["zero_thorcam_slm"],
        dtype=np.float32,
    ).reshape(2)

    triangle_thorcam_slm = BRIDGE_TRIANGLE_THORCAM_SLM
    triangle_nuvu = apply_affine(
        M_thorcam_slm_to_nuvu,
        triangle_thorcam_slm,
    )

    print("M_nuvu_to_thorcam_slm:")
    print(M_nuvu_to_thorcam_slm)

    print("\nUsing bridge triangle in ThorCam_SLM:")
    print(triangle_thorcam_slm)

    print("\nBridge triangle mapped to Nuvu:")
    print(triangle_nuvu)

    pred_back = apply_affine(M_nuvu_to_thorcam_slm, triangle_nuvu)
    slm_residuals = triangle_thorcam_slm - pred_back

    print("\nSLM bridge residuals [px]:")
    print(slm_residuals)

    print("\n=== Loading DMD-side triangle calibration ===")

    dmd_tri = np.load(resolve_path(DMD_TRIANGLE_CALIB_PATH), allow_pickle=True)

    M_cam_to_dmd = np.asarray(dmd_tri["M_cam_to_dmd"], dtype=np.float32)
    zero_cam_xy = np.asarray(dmd_tri["zero_cam_xy"], dtype=np.float32).reshape(2)
    zero_dmd_xy = np.asarray(dmd_tri["zero_dmd_xy"], dtype=np.float32).reshape(2)

    tri_cam_pts_raw = np.asarray(
        dmd_tri["tri_cam_pts"],
        dtype=np.float32,
    ).reshape(3, 2)

    tri_dmd_pts_raw = np.asarray(
        dmd_tri["tri_dmd_pts"],
        dtype=np.float32,
    ).reshape(3, 2)

    print("tri_cam_pts_raw:")
    print(tri_cam_pts_raw)

    print("tri_dmd_pts_raw:")
    print(tri_dmd_pts_raw)

    print("zero_cam_xy:", zero_cam_xy)
    print("zero_dmd_xy:", zero_dmd_xy)

    manual = report_manual_order(
        triangle_nuvu=triangle_nuvu,
        tri_cam_pts_raw=tri_cam_pts_raw,
        tri_dmd_pts_raw=tri_dmd_pts_raw,
        order=DMD_TRIANGLE_ORDER,
        zero_nuvu=zero_nuvu,
        zero_cam_xy=zero_cam_xy,
    )

    order = manual["order"]
    triangle_thorcam_dmd = manual["triangle_thorcam_dmd"]
    triangle_dmd = manual["triangle_dmd"]
    M_nuvu_to_thorcam_dmd = manual["M_nuvu_to_thorcam_dmd"]

    M_nuvu_to_dmd = compose_affines(
        M_cam_to_dmd,
        M_nuvu_to_thorcam_dmd,
    )

    print("\n=== Final matrices ===")
    print("M_nuvu_to_thorcam_dmd:")
    print(M_nuvu_to_thorcam_dmd)

    print("\nM_cam_to_dmd:")
    print(M_cam_to_dmd)

    print("\nM_nuvu_to_dmd:")
    print(M_nuvu_to_dmd)

    print("\n=== Saving Nuvu -> ThorCam_DMD calibration ===")

    out_nuvu_to_thorcam_dmd = ensure_parent(NUVU_TO_THORCAM_DMD_OUT)

    np.savez_compressed(
        out_nuvu_to_thorcam_dmd,
        M_nuvu_to_thorcam_dmd=M_nuvu_to_thorcam_dmd.astype(np.float32),
        M_thorcam_dmd_to_nuvu=invert_affine(M_nuvu_to_thorcam_dmd).astype(np.float32),
        M_nuvu_to_thorcam_slm=M_nuvu_to_thorcam_slm.astype(np.float32),
        M_thorcam_slm_to_nuvu=M_thorcam_slm_to_nuvu.astype(np.float32),
        triangle_nuvu=triangle_nuvu.astype(np.float32),
        triangle_thorcam_slm=triangle_thorcam_slm.astype(np.float32),
        triangle_thorcam_dmd=triangle_thorcam_dmd.astype(np.float32),
        triangle_dmd=triangle_dmd.astype(np.float32),
        raw_tri_cam_pts=tri_cam_pts_raw.astype(np.float32),
        raw_tri_dmd_pts=tri_dmd_pts_raw.astype(np.float32),
        dmd_triangle_order=order.astype(np.int32),
        zero_nuvu=zero_nuvu.astype(np.float32),
        zero_thorcam_slm=zero_thorcam_slm.astype(np.float32),
        zero_cam_xy=zero_cam_xy.astype(np.float32),
        zero_dmd_xy=zero_dmd_xy.astype(np.float32),
        predicted_zero_cam_xy=manual["pred_zero"].astype(np.float32),
        zero_error_px=np.array(manual["zero_error_px"], dtype=np.float32),
        source_nuvu_to_thorcam_slm_path=str(resolve_path(NUVU_TO_THORCAM_SLM_PATH)),
        source_dmd_triangle_calib_path=str(resolve_path(DMD_TRIANGLE_CALIB_PATH)),
        convention="NUVU_TO_THORCAM_DMD_MANUAL_TRIANGLE_ORDER",
    )

    print("Saved:", out_nuvu_to_thorcam_dmd)

    print("\n=== Generating final NV -> DMD chain ===")

    nv_coords_nuvu, spot_weights, original_global_indices, nv_meta = load_nv_coords(
        NV_COORDS_PATH
    )

    nv_coords_thorcam_slm = apply_affine(
        M_nuvu_to_thorcam_slm,
        nv_coords_nuvu,
    )

    nv_coords_thorcam_dmd = apply_affine(
        M_nuvu_to_thorcam_dmd,
        nv_coords_nuvu,
    )

    nv_coords_dmd = apply_affine(
        M_nuvu_to_dmd,
        nv_coords_nuvu,
    )

    inside_dmd_mask, inside_dmd_indices, outside_dmd_indices = compute_inside_dmd_mask(
        nv_coords_dmd
    )

    center_test_indices = get_center_test_indices(nv_coords_dmd, n=10)

    out_chain = ensure_parent(DMD_CHAIN_OUT)

    save_dict = {
        # DMD server compatibility keys
        "M_cam_to_dmd": M_cam_to_dmd.astype(np.float32),
        "zero_dmd_xy": zero_dmd_xy.astype(np.float32),
        "zero_cam_xy": zero_cam_xy.astype(np.float32),
        "dmd_points": nv_coords_dmd.astype(np.float32),
        "pattern_dmd_points": nv_coords_dmd.astype(np.float32),
        "pattern_camera_points": nv_coords_thorcam_dmd.astype(np.float32),
        "slm_camera_points": nv_coords_thorcam_dmd.astype(np.float32),

        # Full chain keys
        "M_nuvu_to_thorcam_slm": M_nuvu_to_thorcam_slm.astype(np.float32),
        "M_thorcam_slm_to_nuvu": M_thorcam_slm_to_nuvu.astype(np.float32),
        "M_nuvu_to_thorcam_dmd": M_nuvu_to_thorcam_dmd.astype(np.float32),
        "M_thorcam_dmd_to_dmd": M_cam_to_dmd.astype(np.float32),
        "M_nuvu_to_dmd": M_nuvu_to_dmd.astype(np.float32),

        "nv_coords_nuvu": nv_coords_nuvu.astype(np.float32),
        "nv_coords_thorcam_slm": nv_coords_thorcam_slm.astype(np.float32),
        "nv_coords_thorcam_dmd": nv_coords_thorcam_dmd.astype(np.float32),
        "nv_coords_dmd": nv_coords_dmd.astype(np.float32),

        "spot_weights": (
            np.asarray([], dtype=np.float32)
            if spot_weights is None
            else spot_weights.astype(np.float32)
        ),

        # Triangle diagnostics
        "triangle_nuvu": triangle_nuvu.astype(np.float32),
        "triangle_thorcam_slm": triangle_thorcam_slm.astype(np.float32),
        "triangle_thorcam_dmd": triangle_thorcam_dmd.astype(np.float32),
        "triangle_dmd": triangle_dmd.astype(np.float32),
        "dmd_triangle_order": order.astype(np.int32),

        "zero_nuvu": zero_nuvu.astype(np.float32),
        "zero_thorcam_slm": zero_thorcam_slm.astype(np.float32),
        "predicted_zero_cam_xy": manual["pred_zero"].astype(np.float32),
        "zero_error_px": np.array(manual["zero_error_px"], dtype=np.float32),

        # DMD coverage
        "dmd_width": np.array(DMD_WIDTH, dtype=np.int32),
        "dmd_height": np.array(DMD_HEIGHT, dtype=np.int32),
        "inside_dmd_mask": inside_dmd_mask.astype(bool),
        "inside_dmd_indices": inside_dmd_indices.astype(np.int32),
        "outside_dmd_indices": outside_dmd_indices.astype(np.int32),
        "center_test_indices": center_test_indices.astype(np.int32),

        # Provenance
        "source_nv_coords_path": nv_meta["source_file"],
        "source_nuvu_to_thorcam_slm_path": str(resolve_path(NUVU_TO_THORCAM_SLM_PATH)),
        "source_dmd_triangle_calib_path": str(resolve_path(DMD_TRIANGLE_CALIB_PATH)),
        "convention": "NUVU_TO_DMD_MANUAL_TRIANGLE_ORDER",
    }

    if original_global_indices is not None:
        save_dict["original_global_indices"] = original_global_indices.astype(np.int32)

    np.savez_compressed(out_chain, **save_dict)

    print("Saved:", out_chain)
    print("Number of NV DMD points:", len(nv_coords_dmd))
    print("inside DMD:", len(inside_dmd_indices), "/", len(nv_coords_dmd))
    print("outside DMD:", len(outside_dmd_indices), "/", len(nv_coords_dmd))
    print("x min/max:", np.min(nv_coords_dmd[:, 0]), np.max(nv_coords_dmd[:, 0]))
    print("y min/max:", np.min(nv_coords_dmd[:, 1]), np.max(nv_coords_dmd[:, 1]))
    print("center_test_indices:", center_test_indices.tolist())
    print("center_test_indices JSON:")
    print(json.dumps(center_test_indices.tolist()))

    diag_path = out_chain.with_suffix(".triangle_diagnostic.png")

    plot_triangle_correspondence(
        triangle_nuvu=triangle_nuvu,
        triangle_thorcam_slm=triangle_thorcam_slm,
        triangle_thorcam_dmd=triangle_thorcam_dmd,
        triangle_dmd=triangle_dmd,
        zero_nuvu=zero_nuvu,
        zero_thorcam_slm=zero_thorcam_slm,
        zero_thorcam_dmd=zero_cam_xy,
        zero_dmd=zero_dmd_xy,
        save_path=diag_path,
    )

    print("\nNext:")
    print(f'  dmd.load_calibration("{DMD_CHAIN_OUT}", True)')
    print("  dmd.pass_loaded_indices(<center_test_indices_json>, 80, 230)")

    return {
        "M_nuvu_to_thorcam_dmd": M_nuvu_to_thorcam_dmd,
        "M_nuvu_to_dmd": M_nuvu_to_dmd,
        "nv_coords_dmd": nv_coords_dmd,
        "inside_dmd_indices": inside_dmd_indices,
        "outside_dmd_indices": outside_dmd_indices,
        "center_test_indices": center_test_indices,
        "zero_error_px": manual["zero_error_px"],
        "dmd_triangle_order": order,
    }


if __name__ == "__main__":
    kpl.init_kplotlib()
    result = main()
    kpl.show(block=True)