# -*- coding: utf-8 -*-
"""
Generate Nuvu -> ThorCam_DMD -> DMD calibration chain.

Uses:
    1. Known SLM bridge triangle:
       ThorCam_SLM <-> Nuvu

    2. DMD triangle calibration:
       ThorCam_DMD -> DMD
       from dmdsuite/calibration/triangle_affine_onpass.npz

Output:
    dmdsuite/calibration/nuvu_to_thorcam_dmd.npz
    dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1271.npz
"""

from pathlib import Path
import itertools
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import common
from utils import kplotlib as kpl


# =============================================================================
# Paths
# =============================================================================
config = common.get_config_dict()
# NV_COORDS_PATH = "slmsuite/nv_blob_detection/nv_blob_1176nvs_reordered_inside_dmd.npz"
NV_COORDS_PATH = config["SpatialCalibrations"]["active_nv_coords_path"]
NUVU_TO_THORCAM_SLM_PATH = "slmsuite/calibration/nuvu_to_thorcam_slm.npz"
DMD_TRIANGLE_CALIB_PATH = "dmdsuite/calibration/triangle_affine_onpass.npz"

NUVU_TO_THORCAM_DMD_OUT = "dmdsuite/calibration/nuvu_to_thorcam_dmd.npz"
DMD_CHAIN_OUT = "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd.npz"

DMD_WIDTH = 1920
DMD_HEIGHT = 1080

# If None, the code tests all 6 orders and chooses the best by zero check.
# If you want manual override, set e.g. DMD_TRIANGLE_ORDER = [2, 0, 1]
DMD_TRIANGLE_ORDER = None

MAX_ALLOWED_ZERO_ERROR_PX = 50.0


# =============================================================================
# Known bridge triangle
# =============================================================================

BRIDGE_TRIANGLE_THORCAM_SLM = np.array(
    [
        [839.90381057, 605.0],
        [580.09618943,  605.0],
        [710.0,        380.0],
    ],
    dtype=np.float32,
)

BRIDGE_TRIANGLE_NUVU = np.array(
    [[77.248, 53.84], [126.626, 318.882], [329.954, 145.427]], dtype=np.float32,
)


# =============================================================================
# Helpers
# =============================================================================

def repo_path():
    return Path(common.get_repo_path())


def resolve_path(path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_path() / p


def ensure_parent(path_like):
    p = resolve_path(path_like)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


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


def invert_affine(M):
    return cv2.invertAffineTransform(np.asarray(M, dtype=np.float32))


def affine_2x3_to_3x3(M):
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = np.asarray(M, dtype=np.float32)
    return H


def affine_3x3_to_2x3(H):
    return np.asarray(H, dtype=np.float32)[:2, :]


def compose_affines(M2, M1):
    """
    Return affine for M2(M1(x)).
    """
    return affine_3x3_to_2x3(
        affine_2x3_to_3x3(M2) @ affine_2x3_to_3x3(M1)
    )


def fit_affine_3pt(src_pts, dst_pts):
    return cv2.getAffineTransform(
        np.asarray(src_pts, dtype=np.float32).reshape(3, 2),
        np.asarray(dst_pts, dtype=np.float32).reshape(3, 2),
    ).astype(np.float32)


def load_nv_coords(path):
    p = resolve_path(path)

    with np.load(p, allow_pickle=True) as data:
        nv_coords = np.asarray(data["nv_coordinates"], dtype=np.float32)

        if "updated_spot_weights" in data.files:
            spot_weights = np.asarray(data["updated_spot_weights"], dtype=np.float32)
        elif "spot_weights" in data.files:
            spot_weights = np.asarray(data["spot_weights"], dtype=np.float32)
        else:
            spot_weights = None

        if "original_global_indices" in data.files:
            original_global_indices = np.asarray(
                data["original_global_indices"],
                dtype=np.int32,
            )
        else:
            original_global_indices = None

        meta = {
            "source_file": str(p),
            "keys": list(data.files),
        }

    return nv_coords, spot_weights, original_global_indices, meta


def compute_inside_dmd_mask(dmd_points):
    dmd_points = np.asarray(dmd_points, dtype=np.float32).reshape(-1, 2)

    inside = (
        (dmd_points[:, 0] >= 0)
        & (dmd_points[:, 0] < DMD_WIDTH)
        & (dmd_points[:, 1] >= 0)
        & (dmd_points[:, 1] < DMD_HEIGHT)
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
# Order testing
# =============================================================================

def test_all_orders(
    triangle_nuvu,
    tri_cam_pts_raw,
    tri_dmd_pts_raw,
    zero_nuvu,
    zero_cam_xy,
):
    print("\n=== Testing all DMD triangle orders ===")

    results = []

    for order_tuple in itertools.permutations([0, 1, 2]):
        order = np.asarray(order_tuple, dtype=np.int32)

        triangle_thorcam_dmd = tri_cam_pts_raw[order]
        triangle_dmd = tri_dmd_pts_raw[order]

        M_nuvu_to_thorcam_dmd = fit_affine_3pt(
            triangle_nuvu,
            triangle_thorcam_dmd,
        )

        pred_zero = apply_affine(M_nuvu_to_thorcam_dmd, zero_nuvu)
        zero_error_px = float(np.linalg.norm(pred_zero - zero_cam_xy))

        results.append(
            {
                "order": order,
                "triangle_thorcam_dmd": triangle_thorcam_dmd,
                "triangle_dmd": triangle_dmd,
                "M_nuvu_to_thorcam_dmd": M_nuvu_to_thorcam_dmd,
                "pred_zero": pred_zero,
                "zero_error_px": zero_error_px,
            }
        )

    results = sorted(results, key=lambda r: r["zero_error_px"])

    for r in results:
        print(
            f"order {r['order'].tolist()} | "
            f"zero error = {r['zero_error_px']:.3f} px | "
            f"pred zero = {r['pred_zero']}"
        )

    print("\nBest order:", results[0]["order"].tolist())
    print("Best zero error:", results[0]["zero_error_px"])

    return results


def choose_order(
    triangle_nuvu,
    tri_cam_pts_raw,
    tri_dmd_pts_raw,
    zero_nuvu,
    zero_cam_xy,
):
    results = test_all_orders(
        triangle_nuvu=triangle_nuvu,
        tri_cam_pts_raw=tri_cam_pts_raw,
        tri_dmd_pts_raw=tri_dmd_pts_raw,
        zero_nuvu=zero_nuvu,
        zero_cam_xy=zero_cam_xy,
    )

    if DMD_TRIANGLE_ORDER is None:
        chosen = results[0]
    else:
        manual_order = np.asarray(DMD_TRIANGLE_ORDER, dtype=np.int32)

        chosen = None
        for r in results:
            if np.array_equal(r["order"], manual_order):
                chosen = r
                break

        if chosen is None:
            raise ValueError(f"Invalid DMD_TRIANGLE_ORDER: {DMD_TRIANGLE_ORDER}")

    print("\n=== Chosen DMD triangle order ===")
    print("order:", chosen["order"].tolist())
    print("zero error [px]:", chosen["zero_error_px"])
    print("predicted zero:", chosen["pred_zero"])
    print("actual zero:", zero_cam_xy)

    if chosen["zero_error_px"] > MAX_ALLOWED_ZERO_ERROR_PX:
        print(
            "\nWARNING: zero error is large. "
            "Order may still be correct, but check zero/triangle consistency."
        )

    return chosen


# =============================================================================
# Plot
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
        pts = np.asarray(pts, dtype=np.float32)

        ax.plot(pts[:, 0], pts[:, 1], "o-")

        for i, p in enumerate(pts):
            ax.text(p[0] + 3, p[1] + 3, str(i))

        if zero is not None:
            zero = np.asarray(zero, dtype=np.float32)
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

    with np.load(resolve_path(NUVU_TO_THORCAM_SLM_PATH), allow_pickle=True) as slm:
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

        if "zero_nuvu" in slm.files:
            zero_nuvu = np.asarray(slm["zero_nuvu"], dtype=np.float32).reshape(2)
        else:
            zero_nuvu = np.array([186.796, 186.907], dtype=np.float32)

        if "zero_thorcam_slm" in slm.files:
            zero_thorcam_slm = np.asarray(
                slm["zero_thorcam_slm"],
                dtype=np.float32,
            ).reshape(2)
        else:
            zero_thorcam_slm = apply_affine(M_nuvu_to_thorcam_slm, zero_nuvu)

    triangle_nuvu = BRIDGE_TRIANGLE_NUVU
    triangle_thorcam_slm = BRIDGE_TRIANGLE_THORCAM_SLM

    # Diagnostic check only.
    pred_thorcam_slm = apply_affine(M_nuvu_to_thorcam_slm, triangle_nuvu)
    pred_nuvu = apply_affine(M_thorcam_slm_to_nuvu, triangle_thorcam_slm)

    print("M_nuvu_to_thorcam_slm:")
    print(M_nuvu_to_thorcam_slm)

    print("\nBridge triangle Nuvu:")
    print(triangle_nuvu)

    print("\nBridge triangle ThorCam_SLM:")
    print(triangle_thorcam_slm)

    print("\nNuvu -> ThorCam_SLM residuals [px]:")
    print(triangle_thorcam_slm - pred_thorcam_slm)

    print("\nThorCam_SLM -> Nuvu residuals [px]:")
    print(triangle_nuvu - pred_nuvu)

    print("\nzero_nuvu:", zero_nuvu)
    print("zero_thorcam_slm:", zero_thorcam_slm)

    print("\n=== Loading DMD-side triangle calibration ===")

    with np.load(resolve_path(DMD_TRIANGLE_CALIB_PATH), allow_pickle=True) as dmd_tri:
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

    chosen = choose_order(
        triangle_nuvu=triangle_nuvu,
        tri_cam_pts_raw=tri_cam_pts_raw,
        tri_dmd_pts_raw=tri_dmd_pts_raw,
        zero_nuvu=zero_nuvu,
        zero_cam_xy=zero_cam_xy,
    )

    order = chosen["order"]
    triangle_thorcam_dmd = chosen["triangle_thorcam_dmd"]
    triangle_dmd = chosen["triangle_dmd"]
    M_nuvu_to_thorcam_dmd = chosen["M_nuvu_to_thorcam_dmd"]

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

    print("\n=== Saving bridge calibration ===")

    out_bridge = ensure_parent(NUVU_TO_THORCAM_DMD_OUT)

    np.savez_compressed(
        out_bridge,
        M_nuvu_to_thorcam_dmd=M_nuvu_to_thorcam_dmd.astype(np.float32),
        M_thorcam_dmd_to_nuvu=invert_affine(M_nuvu_to_thorcam_dmd).astype(np.float32),
        M_cam_to_dmd=M_cam_to_dmd.astype(np.float32),
        M_nuvu_to_dmd=M_nuvu_to_dmd.astype(np.float32),
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
        predicted_zero_cam_xy=chosen["pred_zero"].astype(np.float32),
        zero_error_px=np.asarray(chosen["zero_error_px"], dtype=np.float32),
        source_nuvu_to_thorcam_slm_path=str(resolve_path(NUVU_TO_THORCAM_SLM_PATH)),
        source_dmd_triangle_calib_path=str(resolve_path(DMD_TRIANGLE_CALIB_PATH)),
        convention=np.asarray("NUVU_TO_THORCAM_DMD_FROM_KNOWN_TRIANGLE"),
    )

    print("Saved:", out_bridge)

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
        # DMD server compatibility
        "M_cam_to_dmd": M_cam_to_dmd.astype(np.float32),
        "zero_dmd_xy": zero_dmd_xy.astype(np.float32),
        "zero_cam_xy": zero_cam_xy.astype(np.float32),
        "dmd_points": nv_coords_dmd.astype(np.float32),
        "pattern_dmd_points": nv_coords_dmd.astype(np.float32),
        "pattern_camera_points": nv_coords_thorcam_dmd.astype(np.float32),
        "slm_camera_points": nv_coords_thorcam_dmd.astype(np.float32),

        # Full chain
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

        # Bridge diagnostics
        "triangle_nuvu": triangle_nuvu.astype(np.float32),
        "triangle_thorcam_slm": triangle_thorcam_slm.astype(np.float32),
        "triangle_thorcam_dmd": triangle_thorcam_dmd.astype(np.float32),
        "triangle_dmd": triangle_dmd.astype(np.float32),
        "raw_tri_cam_pts": tri_cam_pts_raw.astype(np.float32),
        "raw_tri_dmd_pts": tri_dmd_pts_raw.astype(np.float32),
        "dmd_triangle_order": order.astype(np.int32),

        "zero_nuvu": zero_nuvu.astype(np.float32),
        "zero_thorcam_slm": zero_thorcam_slm.astype(np.float32),
        "predicted_zero_cam_xy": chosen["pred_zero"].astype(np.float32),
        "zero_error_px": np.asarray(chosen["zero_error_px"], dtype=np.float32),

        # DMD coverage
        "dmd_width": np.asarray(DMD_WIDTH, dtype=np.int32),
        "dmd_height": np.asarray(DMD_HEIGHT, dtype=np.int32),
        "inside_dmd_mask": inside_dmd_mask.astype(bool),
        "inside_dmd_indices": inside_dmd_indices.astype(np.int32),
        "outside_dmd_indices": outside_dmd_indices.astype(np.int32),
        "center_test_indices": center_test_indices.astype(np.int32),

        # Provenance
        "source_nv_coords_path": nv_meta["source_file"],
        "source_nuvu_to_thorcam_slm_path": str(resolve_path(NUVU_TO_THORCAM_SLM_PATH)),
        "source_dmd_triangle_calib_path": str(resolve_path(DMD_TRIANGLE_CALIB_PATH)),
        "convention": np.asarray("NUVU_TO_DMD_FROM_KNOWN_TRIANGLE"),
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
    print(f'dmd.load_calibration("{DMD_CHAIN_OUT}", True)')
    print("dmd.pass_loaded_indices(<center_test_indices_json>, 50, 230)")

    return {
        "M_nuvu_to_thorcam_dmd": M_nuvu_to_thorcam_dmd,
        "M_nuvu_to_dmd": M_nuvu_to_dmd,
        "nv_coords_dmd": nv_coords_dmd,
        "inside_dmd_indices": inside_dmd_indices,
        "outside_dmd_indices": outside_dmd_indices,
        "center_test_indices": center_test_indices,
        "zero_error_px": chosen["zero_error_px"],
        "dmd_triangle_order": order,
    }


if __name__ == "__main__":
    kpl.init_kplotlib()
    result = main()
    kpl.show(block=True)