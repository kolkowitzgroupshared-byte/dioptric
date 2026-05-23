# -*- coding: utf-8 -*-
"""
Generate final Nuvu -> ThorCam_DMD -> DMD chain using the OLD bridge triangle.

This script assumes:

1. Current SLM calibration is valid:
   slmsuite/calibration/nuvu_to_thorcam_slm.npz

2. DMD-side triangle calibration was generated using the OLD smaller SLM triangle:
   dmdsuite/calibration/triangle_affine_onpass.npz

3. The old bridge triangle in ThorCam_SLM coordinates is:
   [[779.2820323, 580.0],
    [640.7179677, 580.0],
    [710.0000000, 460.0]]

The script:
    old ThorCam_SLM triangle
        -> Nuvu triangle using inverse SLM calibration
        -> ThorCam_DMD triangle using DMD camera spots
        -> DMD coordinates using M_cam_to_dmd

Outputs:
    dmdsuite/calibration/nuvu_to_thorcam_dmd.npz
    dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1277.npz
"""

from __future__ import annotations

from pathlib import Path
import itertools
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import common, kplotlib as kpl


# =============================================================================
# Simple hard-coded paths
# =============================================================================

NV_COORDS_PATH = "slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz"

NUVU_TO_THORCAM_SLM_PATH = (
    "slmsuite/calibration/nuvu_to_thorcam_slm.npz"
)

DMD_TRIANGLE_CALIB_PATH = (
    "dmdsuite/calibration/triangle_affine_onpass.npz"
)

NUVU_TO_THORCAM_DMD_OUT = (
    "dmdsuite/calibration/nuvu_to_thorcam_dmd.npz"
)

DMD_CHAIN_OUT = (
    "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1277.npz"
)

DMD_WIDTH = 1920
DMD_HEIGHT = 1080

# This is the OLD bridge triangle that was actually used when
# triangle_affine_onpass.npz was generated.
BRIDGE_TRIANGLE_THORCAM_SLM = np.array(
    [
        [831.24355653, 610.0],
        [588.75644347, 610.0],
        [710.0,       400.0],
    ],
    dtype=np.float32,
)

MAX_ALLOWED_ZERO_ERROR_PX = 10.0


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


# =============================================================================
# Affine helpers
# =============================================================================

def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    return pts_h @ np.asarray(M, dtype=np.float32).T


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
    data = np.load(p, allow_pickle=True)

    if "nv_coordinates" not in data.files:
        raise KeyError(f"{p} missing key 'nv_coordinates'.")

    nv_coords = np.asarray(data["nv_coordinates"], dtype=np.float32)

    spot_weights = None
    if "updated_spot_weights" in data.files:
        spot_weights = np.asarray(data["updated_spot_weights"], dtype=np.float32)

    return nv_coords, spot_weights, {"source_file": str(p), "keys": list(data.files)}


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
# DMD triangle order finder
# =============================================================================

def find_best_dmd_triangle_order(
    triangle_nuvu,
    tri_cam_pts_raw,
    zero_nuvu,
    zero_cam_xy,
):
    """
    Try all permutations of DMD-side triangle points.

    Choose the one where Nuvu -> ThorCam_DMD affine predicts the zero-order
    camera position most accurately.
    """
    best = None

    for perm in itertools.permutations(range(3)):
        tri_cam = tri_cam_pts_raw[list(perm)]

        M = fit_affine_3pt(triangle_nuvu, tri_cam)
        pred_zero = apply_affine(M, zero_nuvu)[0]
        err = float(np.linalg.norm(pred_zero - zero_cam_xy))

        print(
            f"perm={perm}, zero error={err:.3f} px, "
            f"pred_zero={pred_zero}"
        )

        if best is None or err < best["zero_error_px"]:
            best = {
                "order": list(perm),
                "zero_error_px": err,
                "pred_zero": pred_zero,
                "M_nuvu_to_thorcam_dmd": M,
                "triangle_thorcam_dmd": tri_cam,
            }

    print("\nBEST DMD triangle order:")
    print("DMD_TRIANGLE_ORDER =", best["order"])
    print("zero error [px] =", best["zero_error_px"])
    print("predicted zero:", best["pred_zero"])
    print("actual zero:   ", zero_cam_xy)

    return best


# =============================================================================
# Diagnostics
# =============================================================================

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
        slm["M_nuvu_to_thorcam_slm"], dtype=np.float32
    )
    M_thorcam_slm_to_nuvu = invert_affine(M_nuvu_to_thorcam_slm)

    zero_nuvu = np.asarray(slm["zero_nuvu"], dtype=np.float32).reshape(2)
    zero_thorcam_slm = np.asarray(slm["zero_thorcam_slm"], dtype=np.float32).reshape(2)

    # This is the actual bridge:
    # old SLM ThorCam triangle -> Nuvu triangle
    triangle_thorcam_slm = BRIDGE_TRIANGLE_THORCAM_SLM
    triangle_nuvu = apply_affine(M_thorcam_slm_to_nuvu, triangle_thorcam_slm)

    print("M_nuvu_to_thorcam_slm:")
    print(M_nuvu_to_thorcam_slm)

    print("Using bridge triangle in ThorCam_SLM:")
    print(triangle_thorcam_slm)

    print("Bridge triangle mapped to Nuvu:")
    print(triangle_nuvu)

    # Check internal consistency.
    pred_back = apply_affine(M_nuvu_to_thorcam_slm, triangle_nuvu)
    slm_residuals = triangle_thorcam_slm - pred_back
    print("SLM bridge residuals [px]:")
    print(slm_residuals)

    print("\n=== Loading DMD-side triangle calibration ===")

    dmd_tri = np.load(resolve_path(DMD_TRIANGLE_CALIB_PATH), allow_pickle=True)

    M_cam_to_dmd = np.asarray(dmd_tri["M_cam_to_dmd"], dtype=np.float32)
    zero_cam_xy = np.asarray(dmd_tri["zero_cam_xy"], dtype=np.float32).reshape(2)
    zero_dmd_xy = np.asarray(dmd_tri["zero_dmd_xy"], dtype=np.float32).reshape(2)

    tri_cam_pts_raw = np.asarray(dmd_tri["tri_cam_pts"], dtype=np.float32).reshape(3, 2)
    tri_dmd_pts_raw = np.asarray(dmd_tri["tri_dmd_pts"], dtype=np.float32).reshape(3, 2)

    print("tri_cam_pts_raw:")
    print(tri_cam_pts_raw)
    print("tri_dmd_pts_raw:")
    print(tri_dmd_pts_raw)
    print("zero_cam_xy:", zero_cam_xy)
    print("zero_dmd_xy:", zero_dmd_xy)

    print("\n=== Finding DMD triangle order ===")

    best = find_best_dmd_triangle_order(
        triangle_nuvu=triangle_nuvu,
        tri_cam_pts_raw=tri_cam_pts_raw,
        zero_nuvu=zero_nuvu,
        zero_cam_xy=zero_cam_xy,
    )

    if best["zero_error_px"] > MAX_ALLOWED_ZERO_ERROR_PX:
        raise RuntimeError(
            f"Best zero-order error is {best['zero_error_px']:.2f} px, "
            f"which is larger than {MAX_ALLOWED_ZERO_ERROR_PX} px. "
            "This means the bridge triangle and DMD triangle calibration do not match."
        )

    order = np.asarray(best["order"], dtype=np.int32)
    triangle_thorcam_dmd = best["triangle_thorcam_dmd"]
    triangle_dmd = tri_dmd_pts_raw[order]
    M_nuvu_to_thorcam_dmd = best["M_nuvu_to_thorcam_dmd"]

    M_nuvu_to_dmd = compose_affines(M_cam_to_dmd, M_nuvu_to_thorcam_dmd)

    print("\n=== Final matrices ===")
    print("M_nuvu_to_thorcam_dmd:")
    print(M_nuvu_to_thorcam_dmd)
    print("M_cam_to_dmd:")
    print(M_cam_to_dmd)
    print("M_nuvu_to_dmd:")
    print(M_nuvu_to_dmd)

    print("\n=== Saving Nuvu -> ThorCam_DMD calibration ===")

    out_nuvu_to_thorcam_dmd = ensure_parent(NUVU_TO_THORCAM_DMD_OUT)

    np.savez_compressed(
        out_nuvu_to_thorcam_dmd,
        M_nuvu_to_thorcam_dmd=M_nuvu_to_thorcam_dmd,
        M_thorcam_dmd_to_nuvu=invert_affine(M_nuvu_to_thorcam_dmd),
        M_nuvu_to_thorcam_slm=M_nuvu_to_thorcam_slm,
        M_thorcam_slm_to_nuvu=M_thorcam_slm_to_nuvu,
        triangle_nuvu=triangle_nuvu,
        triangle_thorcam_slm=triangle_thorcam_slm,
        triangle_thorcam_dmd=triangle_thorcam_dmd,
        triangle_dmd=triangle_dmd,
        raw_tri_cam_pts=tri_cam_pts_raw,
        raw_tri_dmd_pts=tri_dmd_pts_raw,
        dmd_triangle_order=order,
        zero_nuvu=zero_nuvu,
        zero_thorcam_slm=zero_thorcam_slm,
        zero_cam_xy=zero_cam_xy,
        zero_dmd_xy=zero_dmd_xy,
        predicted_zero_cam_xy=best["pred_zero"],
        zero_error_px=np.array(best["zero_error_px"], dtype=np.float32),
        source_nuvu_to_thorcam_slm_path=str(resolve_path(NUVU_TO_THORCAM_SLM_PATH)),
        source_dmd_triangle_calib_path=str(resolve_path(DMD_TRIANGLE_CALIB_PATH)),
        convention="NUVU_TO_THORCAM_DMD_FROM_OLD_BRIDGE_TRIANGLE",
    )

    print("Saved:", out_nuvu_to_thorcam_dmd)

    print("\n=== Generating final NV -> DMD chain ===")

    nv_coords_nuvu, spot_weights, nv_meta = load_nv_coords(NV_COORDS_PATH)

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

    inside_dmd_mask, inside_dmd_indices, outside_dmd_indices = (
        compute_inside_dmd_mask(nv_coords_dmd)
    )

    center_test_indices = get_center_test_indices(nv_coords_dmd, n=10)

    out_chain = ensure_parent(DMD_CHAIN_OUT)

    np.savez_compressed(
        out_chain,
        # DMD server compatibility keys
        M_cam_to_dmd=M_cam_to_dmd,
        zero_dmd_xy=zero_dmd_xy,
        zero_cam_xy=zero_cam_xy,
        dmd_points=nv_coords_dmd,
        pattern_dmd_points=nv_coords_dmd,
        pattern_camera_points=nv_coords_thorcam_dmd,
        slm_camera_points=nv_coords_thorcam_dmd,
        # Full chain keys
        M_nuvu_to_thorcam_slm=M_nuvu_to_thorcam_slm,
        M_thorcam_slm_to_nuvu=M_thorcam_slm_to_nuvu,
        M_nuvu_to_thorcam_dmd=M_nuvu_to_thorcam_dmd,
        M_thorcam_dmd_to_dmd=M_cam_to_dmd,
        M_nuvu_to_dmd=M_nuvu_to_dmd,
        nv_coords_nuvu=nv_coords_nuvu,
        nv_coords_thorcam_slm=nv_coords_thorcam_slm,
        nv_coords_thorcam_dmd=nv_coords_thorcam_dmd,
        nv_coords_dmd=nv_coords_dmd,
        spot_weights=np.asarray([]) if spot_weights is None else spot_weights,
        # Triangle diagnostics
        triangle_nuvu=triangle_nuvu,
        triangle_thorcam_slm=triangle_thorcam_slm,
        triangle_thorcam_dmd=triangle_thorcam_dmd,
        triangle_dmd=triangle_dmd,
        dmd_triangle_order=order,
        zero_nuvu=zero_nuvu,
        zero_thorcam_slm=zero_thorcam_slm,
        predicted_zero_cam_xy=best["pred_zero"],
        zero_error_px=np.array(best["zero_error_px"], dtype=np.float32),
        # DMD coverage
        dmd_width=np.array(DMD_WIDTH, dtype=np.int32),
        dmd_height=np.array(DMD_HEIGHT, dtype=np.int32),
        inside_dmd_mask=inside_dmd_mask,
        inside_dmd_indices=inside_dmd_indices,
        outside_dmd_indices=outside_dmd_indices,
        center_test_indices=center_test_indices,
        # Provenance
        source_nv_coords_path=nv_meta["source_file"],
        source_nuvu_to_thorcam_slm_path=str(resolve_path(NUVU_TO_THORCAM_SLM_PATH)),
        source_dmd_triangle_calib_path=str(resolve_path(DMD_TRIANGLE_CALIB_PATH)),
        convention="NUVU_TO_DMD_FROM_OLD_BRIDGE_TRIANGLE",
    )

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
        "zero_error_px": best["zero_error_px"],
        "dmd_triangle_order": order,
    }


if __name__ == "__main__":
    kpl.init_kplotlib()
    main()
    kpl.show(block=True)