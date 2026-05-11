# -*- coding: utf-8 -*-

from pathlib import Path
import itertools
import json

import cv2
import numpy as np

from utils import common


NV_COORDS_PATH = "slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz"
SLM_CAL_PATH = "slmsuite/calibration/nuvu_to_thorcam_slm.npz"
DMD_TRIANGLE_PATH = "dmdsuite/calibration/triangle_affine_onpass.npz"
OUT_CHAIN_PATH = "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1277.npz"
OUT_NUVU_TO_THORCAM_DMD_PATH = "dmdsuite/calibration/nuvu_to_thorcam_dmd.npz"

DMD_WIDTH = 1920
DMD_HEIGHT = 1080

# OLD bridge triangle that was actually used when triangle_affine_onpass.npz was made
BRIDGE_TRIANGLE_THORCAM_SLM = np.array(
    [
        [779.2820323, 580.0],
        [640.7179677, 580.0],
        [710.0000000, 460.0],
    ],
    dtype=np.float32,
)


def repo_path():
    return Path(common.get_repo_path())


def resolve(path):
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_path() / p


def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return np.hstack([pts, ones]) @ np.asarray(M, dtype=np.float32).T


def affine_2x3_to_3x3(M):
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = np.asarray(M, dtype=np.float32)
    return H


def affine_3x3_to_2x3(H):
    return np.asarray(H, dtype=np.float32)[:2, :]


def compose_affines(M2, M1):
    """
    Return M2 after M1.

    p_mid = M1(p)
    p_out = M2(p_mid)
    """
    return affine_3x3_to_2x3(
        affine_2x3_to_3x3(M2) @ affine_2x3_to_3x3(M1)
    )


def invert_affine(M):
    return cv2.invertAffineTransform(np.asarray(M, dtype=np.float32))


def inside_dmd_mask(dmd_pts):
    dmd_pts = np.asarray(dmd_pts, dtype=np.float32).reshape(-1, 2)
    inside = (
        (dmd_pts[:, 0] >= 0)
        & (dmd_pts[:, 0] < DMD_WIDTH)
        & (dmd_pts[:, 1] >= 0)
        & (dmd_pts[:, 1] < DMD_HEIGHT)
    )
    return inside


def center_indices(dmd_pts, n=10):
    dmd_pts = np.asarray(dmd_pts, dtype=np.float32).reshape(-1, 2)
    inside = inside_dmd_mask(dmd_pts)
    inside_inds = np.where(inside)[0]

    center = np.array([DMD_WIDTH / 2, DMD_HEIGHT / 2], dtype=np.float32)
    dist = np.linalg.norm(dmd_pts[inside_inds] - center, axis=1)

    return inside_inds[np.argsort(dist)[:n]].astype(np.int32)


def main():
    # ---------------------------------------------------------------------
    # Load SLM-side matrix: Nuvu -> SLM ThorCam
    # ---------------------------------------------------------------------
    slm = np.load(resolve(SLM_CAL_PATH), allow_pickle=True)

    M_N_to_S = np.asarray(slm["M_nuvu_to_thorcam_slm"], dtype=np.float32)
    M_S_to_N = invert_affine(M_N_to_S)

    zero_N = np.asarray(slm["zero_nuvu"], dtype=np.float32).reshape(2)
    zero_S = np.asarray(slm["zero_thorcam_slm"], dtype=np.float32).reshape(2)

    print("\n=== SLM side ===")
    print("M_N_to_S:")
    print(M_N_to_S)
    print("zero_N:", zero_N)
    print("zero_S:", zero_S)
    print("bridge triangle SLM ThorCam:")
    print(BRIDGE_TRIANGLE_THORCAM_SLM)

    bridge_triangle_N = apply_affine(M_S_to_N, BRIDGE_TRIANGLE_THORCAM_SLM)
    print("bridge triangle mapped to Nuvu:")
    print(bridge_triangle_N)

    # ---------------------------------------------------------------------
    # Load DMD-side calibration: DMD ThorCam -> DMD chip
    # ---------------------------------------------------------------------
    dmd = np.load(resolve(DMD_TRIANGLE_PATH), allow_pickle=True)

    M_C_to_D = np.asarray(dmd["M_cam_to_dmd"], dtype=np.float32)
    zero_C = np.asarray(dmd["zero_cam_xy"], dtype=np.float32).reshape(2)
    zero_D = np.asarray(dmd["zero_dmd_xy"], dtype=np.float32).reshape(2)

    tri_C_raw = np.asarray(dmd["tri_cam_pts"], dtype=np.float32).reshape(3, 2)
    tri_D_raw = np.asarray(dmd["tri_dmd_pts"], dtype=np.float32).reshape(3, 2)

    print("\n=== DMD side ===")
    print("M_C_to_D:")
    print(M_C_to_D)
    print("zero_C:", zero_C)
    print("zero_D:", zero_D)
    print("tri_C_raw:")
    print(tri_C_raw)
    print("tri_D_raw:")
    print(tri_D_raw)

    # ---------------------------------------------------------------------
    # Find bridge order: SLM ThorCam -> DMD ThorCam
    # ---------------------------------------------------------------------
    print("\n=== Testing DMD triangle order using direct SLM-ThorCam bridge ===")

    best = None

    for perm in itertools.permutations(range(3)):
        tri_C = tri_C_raw[list(perm)]

        # Direct bridge matrix:
        # SLM ThorCam -> DMD ThorCam
        M_S_to_C = cv2.getAffineTransform(
            BRIDGE_TRIANGLE_THORCAM_SLM.astype(np.float32),
            tri_C.astype(np.float32),
        ).astype(np.float32)

        pred_zero_C_from_S = apply_affine(M_S_to_C, zero_S)[0]
        err_zero_C = np.linalg.norm(pred_zero_C_from_S - zero_C)

        # Also test zero from full Nuvu -> SLM -> DMD camera chain
        M_N_to_C = compose_affines(M_S_to_C, M_N_to_S)
        pred_zero_C_from_N = apply_affine(M_N_to_C, zero_N)[0]
        err_zero_C_from_N = np.linalg.norm(pred_zero_C_from_N - zero_C)

        print(
            f"perm={perm}, "
            f"err zero S->C={err_zero_C:.3f} px, "
            f"err zero N->C={err_zero_C_from_N:.3f} px, "
            f"pred_zero_C={pred_zero_C_from_S}"
        )

        if best is None or err_zero_C < best["err_zero_C"]:
            best = {
                "perm": perm,
                "M_S_to_C": M_S_to_C,
                "M_N_to_C": M_N_to_C,
                "tri_C": tri_C,
                "tri_D": tri_D_raw[list(perm)],
                "pred_zero_C": pred_zero_C_from_S,
                "err_zero_C": float(err_zero_C),
                "err_zero_C_from_N": float(err_zero_C_from_N),
            }

    print("\n=== BEST bridge ===")
    print("DMD_TRIANGLE_ORDER:", list(best["perm"]))
    print("zero error SLM ThorCam -> DMD ThorCam:", best["err_zero_C"])
    print("zero error Nuvu -> DMD ThorCam:", best["err_zero_C_from_N"])
    print("predicted zero_C:", best["pred_zero_C"])
    print("actual zero_C:   ", zero_C)

    if best["err_zero_C"] > 5:
        print("\nWARNING:")
        print("Best zero error is still > 5 px.")
        print("That means the bridge triangle and triangle_affine_onpass.npz may not match.")
        print("Do not trust final NV mapping until this is fixed.")

    # ---------------------------------------------------------------------
    # Compose final matrix: Nuvu -> DMD
    # ---------------------------------------------------------------------
    M_S_to_C = best["M_S_to_C"]
    M_N_to_C = best["M_N_to_C"]
    M_N_to_D = compose_affines(M_C_to_D, M_N_to_C)

    print("\n=== Final composed matrices ===")
    print("M_S_to_C:")
    print(M_S_to_C)
    print("M_N_to_C:")
    print(M_N_to_C)
    print("M_N_to_D:")
    print(M_N_to_D)

    # ---------------------------------------------------------------------
    # Generate all NV DMD points
    # ---------------------------------------------------------------------
    nv_data = np.load(resolve(NV_COORDS_PATH), allow_pickle=True)
    nv_coords_N = np.asarray(nv_data["nv_coordinates"], dtype=np.float32)

    spot_weights = None
    if "updated_spot_weights" in nv_data.files:
        spot_weights = np.asarray(nv_data["updated_spot_weights"], dtype=np.float32)

    nv_coords_S = apply_affine(M_N_to_S, nv_coords_N)
    nv_coords_C = apply_affine(M_N_to_C, nv_coords_N)
    nv_coords_D = apply_affine(M_N_to_D, nv_coords_N)

    inside = inside_dmd_mask(nv_coords_D)
    inside_indices = np.where(inside)[0].astype(np.int32)
    outside_indices = np.where(~inside)[0].astype(np.int32)
    center_test_indices = center_indices(nv_coords_D, n=10)

    print("\n=== DMD point report ===")
    print("num NVs:", len(nv_coords_D))
    print("inside DMD:", len(inside_indices), "/", len(nv_coords_D))
    print("outside DMD:", len(outside_indices))
    print("x min/max:", np.min(nv_coords_D[:, 0]), np.max(nv_coords_D[:, 0]))
    print("y min/max:", np.min(nv_coords_D[:, 1]), np.max(nv_coords_D[:, 1]))
    print("center_test_indices:", center_test_indices.tolist())
    print("center_test_indices JSON:")
    print(json.dumps(center_test_indices.tolist()))

    # ---------------------------------------------------------------------
    # Save Nuvu -> DMD camera bridge
    # ---------------------------------------------------------------------
    out_nuvu_to_dmdcam = resolve(OUT_NUVU_TO_THORCAM_DMD_PATH)
    out_nuvu_to_dmdcam.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_nuvu_to_dmdcam,
        M_nuvu_to_thorcam_dmd=M_N_to_C,
        M_thorcam_dmd_to_nuvu=invert_affine(M_N_to_C),
        M_nuvu_to_thorcam_slm=M_N_to_S,
        M_thorcam_slm_to_nuvu=M_S_to_N,
        M_thorcam_slm_to_thorcam_dmd=M_S_to_C,
        bridge_triangle_thorcam_slm=BRIDGE_TRIANGLE_THORCAM_SLM,
        bridge_triangle_nuvu=bridge_triangle_N,
        triangle_thorcam_dmd=best["tri_C"],
        triangle_dmd=best["tri_D"],
        raw_tri_cam_pts=tri_C_raw,
        raw_tri_dmd_pts=tri_D_raw,
        dmd_triangle_order=np.asarray(best["perm"], dtype=np.int32),
        zero_nuvu=zero_N,
        zero_thorcam_slm=zero_S,
        zero_cam_xy=zero_C,
        zero_dmd_xy=zero_D,
        predicted_zero_cam_xy=best["pred_zero_C"],
        zero_error_px=np.array(best["err_zero_C"], dtype=np.float32),
        convention="DIRECT_BRIDGE_NUVU_TO_THORCAM_DMD",
    )

    print("\nSaved:", out_nuvu_to_dmdcam)

    # ---------------------------------------------------------------------
    # Save final chain for DMD server
    # ---------------------------------------------------------------------
    out_chain = resolve(OUT_CHAIN_PATH)
    out_chain.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_chain,
        # DMD server compatibility keys
        M_cam_to_dmd=M_C_to_D,
        zero_dmd_xy=zero_D,
        zero_cam_xy=zero_C,
        dmd_points=nv_coords_D,
        pattern_dmd_points=nv_coords_D,
        pattern_camera_points=nv_coords_C,
        slm_camera_points=nv_coords_C,
        # Full chain keys
        M_nuvu_to_thorcam_slm=M_N_to_S,
        M_thorcam_slm_to_nuvu=M_S_to_N,
        M_thorcam_slm_to_thorcam_dmd=M_S_to_C,
        M_nuvu_to_thorcam_dmd=M_N_to_C,
        M_thorcam_dmd_to_dmd=M_C_to_D,
        M_nuvu_to_dmd=M_N_to_D,
        nv_coords_nuvu=nv_coords_N,
        nv_coords_thorcam_slm=nv_coords_S,
        nv_coords_thorcam_dmd=nv_coords_C,
        nv_coords_dmd=nv_coords_D,
        spot_weights=np.asarray([]) if spot_weights is None else spot_weights,
        # Diagnostics
        bridge_triangle_thorcam_slm=BRIDGE_TRIANGLE_THORCAM_SLM,
        bridge_triangle_nuvu=bridge_triangle_N,
        triangle_thorcam_dmd=best["tri_C"],
        triangle_dmd=best["tri_D"],
        dmd_triangle_order=np.asarray(best["perm"], dtype=np.int32),
        zero_nuvu=zero_N,
        zero_thorcam_slm=zero_S,
        predicted_zero_cam_xy=best["pred_zero_C"],
        zero_error_px=np.array(best["err_zero_C"], dtype=np.float32),
        # Coverage
        dmd_width=np.array(DMD_WIDTH, dtype=np.int32),
        dmd_height=np.array(DMD_HEIGHT, dtype=np.int32),
        inside_dmd_mask=inside,
        inside_dmd_indices=inside_indices,
        outside_dmd_indices=outside_indices,
        center_test_indices=center_test_indices,
        # Provenance
        source_nv_coords_path=str(resolve(NV_COORDS_PATH)),
        source_nuvu_to_thorcam_slm_path=str(resolve(SLM_CAL_PATH)),
        source_dmd_triangle_calib_path=str(resolve(DMD_TRIANGLE_PATH)),
        convention="DIRECT_BRIDGE_NUVU_TO_DMD",
    )

    print("Saved:", out_chain)

    print("\nNext LabRAD test:")
    print(f'dmd.load_calibration("{OUT_CHAIN_PATH}")')
    print("dmd.pass_loaded_indices(<center_test_indices_json>, 80, 230)")


if __name__ == "__main__":
    main()