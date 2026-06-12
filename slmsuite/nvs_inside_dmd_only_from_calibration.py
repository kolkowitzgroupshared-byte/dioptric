# -*- coding: utf-8 -*-

from pathlib import Path
import os
import numpy as np

from utils import common


NV_BLOB_PATH = "slmsuite/nv_blob_detection/nv_blob_1271nvs_reordered.npz"

CALIBRATION_PATH = "dmdsuite/calibration/nuvu_to_thorcam_dmd.npz"

DMD_WIDTH = 1920
DMD_HEIGHT = 1080


def resolve_path(path_like):
    p = Path(path_like)

    if p.is_absolute():
        return p

    return Path(common.get_repo_path()) / p


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


def compute_inside_dmd_indices(dmd_points):
    dmd_points = np.asarray(dmd_points, dtype=np.float32)

    inside = (
        (dmd_points[:, 0] >= 0)
        & (dmd_points[:, 0] < DMD_WIDTH)
        & (dmd_points[:, 1] >= 0)
        & (dmd_points[:, 1] < DMD_HEIGHT)
    )

    return np.where(inside)[0].astype(np.int32), inside


def save_nv_blob_inside_dmd_only_from_calibration(
    nv_blob_path=NV_BLOB_PATH,
    calibration_path=CALIBRATION_PATH,
):
    nv_blob_path = resolve_path(nv_blob_path)
    calibration_path = resolve_path(calibration_path)

    # ------------------------------------------------------------------
    # Load original NV blob file.
    # ------------------------------------------------------------------
    with np.load(nv_blob_path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}

    if "nv_coordinates" not in out:
        raise KeyError("Input file does not contain 'nv_coordinates'.")

    nv_coordinates = np.asarray(out["nv_coordinates"], dtype=np.float32)
    num_nvs_old = len(nv_coordinates)

    # ------------------------------------------------------------------
    # Load calibration matrix.
    # ------------------------------------------------------------------
    with np.load(calibration_path, allow_pickle=True) as calib:
        if "M_nuvu_to_dmd" not in calib.files:
            raise KeyError(
                f"{calibration_path} does not contain 'M_nuvu_to_dmd'."
            )

        M_nuvu_to_dmd = np.asarray(calib["M_nuvu_to_dmd"], dtype=np.float32)

    # ------------------------------------------------------------------
    # Transform original NV coordinates into DMD coordinates.
    # ------------------------------------------------------------------
    dmd_points = apply_affine(M_nuvu_to_dmd, nv_coordinates)

    inside_inds, inside_mask = compute_inside_dmd_indices(dmd_points)
    num_nvs_new = len(inside_inds)

    print("Original NVs:", num_nvs_old)
    print("Inside DMD:", num_nvs_new)
    print("Outside DMD:", num_nvs_old - num_nvs_new)
    print("DMD x min/max:", float(np.min(dmd_points[:, 0])), float(np.max(dmd_points[:, 0])))
    print("DMD y min/max:", float(np.min(dmd_points[:, 1])), float(np.max(dmd_points[:, 1])))

    # ------------------------------------------------------------------
    # Filter only per-NV arrays.
    # Keep all original keys.
    # ------------------------------------------------------------------
    for key in list(out.keys()):
        arr = np.asarray(out[key])

        if arr.ndim >= 1 and arr.shape[0] == num_nvs_old:
            out[key] = arr[inside_inds]

    # Optional useful extra keys.
    # Remove these lines if you want absolutely no extra keys.
    out["dmd_points"] = dmd_points[inside_inds].astype(np.float32)
    out["M_nuvu_to_dmd"] = M_nuvu_to_dmd.astype(np.float32)
    out["source_calibration_path"] = np.asarray(str(calibration_path))
    out["coordinate_convention"] = np.asarray(
        "ORIGINAL_NUVU_COORDINATES_FILTERED_BY_DMD_CALIBRATION"
    )

    # ------------------------------------------------------------------
    # Save with updated number of NVs in filename.
    # ------------------------------------------------------------------
    out_name = f"nv_blob_{num_nvs_new}nvs_reordered_inside_dmd.npz"
    out_path = nv_blob_path.with_name(out_name)

    tmp_path = out_path.with_name(out_path.stem + "_tmp.npz")

    np.savez_compressed(tmp_path, **out)
    os.replace(tmp_path, out_path)

    print("\nSaved:", out_path)
    print("Keys saved:")
    print(list(out.keys()))
    print("New nv_coordinates shape:", out["nv_coordinates"].shape)

    return out_path, out


if __name__ == "__main__":
    save_nv_blob_inside_dmd_only_from_calibration()