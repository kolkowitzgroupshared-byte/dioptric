# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return np.hstack([pts, ones]) @ np.asarray(M, dtype=np.float32).T


def fit_affine_least_squares(src, dst):
    """
    Fit affine transform src -> dst using all points.

    Returns 2x3 matrix M such that:
        dst = M @ [src_x, src_y, 1]
    """
    src = np.asarray(src, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float32).reshape(-1, 2)

    if len(src) < 3:
        raise ValueError("Need at least 3 points for affine fit.")

    X = np.column_stack([src, np.ones(len(src), dtype=np.float32)])
    B, *_ = np.linalg.lstsq(X, dst, rcond=None)

    return B.T.astype(np.float32)


def affine_residual_report(name, M, src, dst):
    pred = apply_affine(M, src)
    residuals = np.asarray(dst, dtype=np.float32) - pred
    err = np.linalg.norm(residuals, axis=1)

    print(f"\n=== {name} residual report ===")
    print("mean error [px]:", float(np.mean(err)))
    print("RMS error  [px]:", float(np.sqrt(np.mean(err**2))))
    print("max error  [px]:", float(np.max(err)))

    for i, (s, d, p, r, e) in enumerate(zip(src, dst, pred, residuals, err)):
        print(
            f"{i:02d}: src={s}, dst={d}, pred={p}, "
            f"res={r}, |res|={e:.4f}"
        )

    return pred, residuals, err


# =============================================================================
# Data
# =============================================================================

save_path = "slmsuite/calibration/nuvu_to_thorcam_slm.npz"

sets = []

sets.append({
    "name": "set1",
    "thorcam": np.array([
        [834.90381057, 595.0],
        [575.09618943,  595.0],
        [705.0,        370.0],
    ], dtype=np.float32),
    "nuvu": np.array([
[87.107, 69.497], [139.137, 333.599], [340.163, 157.953]
 ], dtype=np.float32),
})
#  [[88.528, 69.463], [140.67, 332.807], [341.79, 157.813], [189.794, 186.23]]

        # [156.115, 278.98], 
        # [110.64, 151.517], 
        # [246.473, 176.508],
# =============================================================================
# Combine all points
# =============================================================================

cal_coords_thorcam_slm = np.vstack([s["thorcam"] for s in sets]).astype(np.float32)
cal_coords_nuvu = np.vstack([s["nuvu"] for s in sets]).astype(np.float32)

set_labels = []
point_labels = []
for s in sets:
    for k in range(len(s["nuvu"])):
        set_labels.append(s["name"])
        point_labels.append(k)

set_labels = np.array(set_labels)
point_labels = np.array(point_labels)


# =============================================================================
# Zero-order check point
# =============================================================================

zero_thorcam_slm = np.array([705.05789009, 519.93369574], dtype=np.float32)
zero_nuvu = np.array([189.794, 186.23], dtype=np.float32)


# =============================================================================
# Fit Nuvu -> ThorCam_SLM
# =============================================================================

M_nuvu_to_thorcam_slm = fit_affine_least_squares(
    cal_coords_nuvu,
    cal_coords_thorcam_slm,
)

M_thorcam_slm_to_nuvu = cv2.invertAffineTransform(
    M_nuvu_to_thorcam_slm.astype(np.float32)
)

print("\nM_nuvu_to_thorcam_slm:")
print(M_nuvu_to_thorcam_slm)

print("\nM_thorcam_slm_to_nuvu:")
print(M_thorcam_slm_to_nuvu)

pred_thorcam, residuals_thorcam, err_thorcam = affine_residual_report(
    "Nuvu -> ThorCam_SLM",
    M_nuvu_to_thorcam_slm,
    cal_coords_nuvu,
    cal_coords_thorcam_slm,
)

pred_nuvu, residuals_nuvu, err_nuvu = affine_residual_report(
    "ThorCam_SLM -> Nuvu",
    M_thorcam_slm_to_nuvu,
    cal_coords_thorcam_slm,
    cal_coords_nuvu,
)


# =============================================================================
# Zero-order validation
# =============================================================================

pred_zero_thorcam_slm = apply_affine(M_nuvu_to_thorcam_slm, zero_nuvu)[0]
zero_error_thorcam_px = float(np.linalg.norm(pred_zero_thorcam_slm - zero_thorcam_slm))

pred_zero_nuvu = apply_affine(M_thorcam_slm_to_nuvu, zero_thorcam_slm)[0]
zero_error_nuvu_px = float(np.linalg.norm(pred_zero_nuvu - zero_nuvu))

print("\n=== Zero-order check ===")
print("zero_nuvu:", zero_nuvu)
print("pred_zero_thorcam_slm:", pred_zero_thorcam_slm)
print("actual zero_thorcam_slm:", zero_thorcam_slm)
print("zero error ThorCam px:", zero_error_thorcam_px)

print("\nzero_thorcam_slm:", zero_thorcam_slm)
print("pred_zero_nuvu:", pred_zero_nuvu)
print("actual zero_nuvu:", zero_nuvu)
print("zero error Nuvu px:", zero_error_nuvu_px)


# =============================================================================
# Save
# =============================================================================

os.makedirs(os.path.dirname(save_path), exist_ok=True)

np.savez_compressed(
    save_path,

    # Main transforms
    M_nuvu_to_thorcam_slm=M_nuvu_to_thorcam_slm.astype(np.float32),
    M_thorcam_slm_to_nuvu=M_thorcam_slm_to_nuvu.astype(np.float32),

    # Calibration data
    cal_coords_nuvu=cal_coords_nuvu.astype(np.float32),
    cal_coords_thorcam_slm=cal_coords_thorcam_slm.astype(np.float32),
    pred_coords_thorcam_slm=pred_thorcam.astype(np.float32),
    residuals_thorcam_slm=residuals_thorcam.astype(np.float32),
    err_thorcam_slm=err_thorcam.astype(np.float32),
    pred_coords_nuvu=pred_nuvu.astype(np.float32),
    residuals_nuvu=residuals_nuvu.astype(np.float32),
    err_nuvu=err_nuvu.astype(np.float32),

    # Zero-order check
    zero_nuvu=zero_nuvu.astype(np.float32),
    zero_thorcam_slm=zero_thorcam_slm.astype(np.float32),
    pred_zero_thorcam_slm=pred_zero_thorcam_slm.astype(np.float32),
    pred_zero_nuvu=pred_zero_nuvu.astype(np.float32),
    zero_error_thorcam_px=np.array(zero_error_thorcam_px, dtype=np.float32),
    zero_error_nuvu_px=np.array(zero_error_nuvu_px, dtype=np.float32),

    # Labels/provenance
    set_labels=set_labels,
    point_labels=point_labels.astype(np.int32),
    convention="NUVU_TO_THORCAM_SLM_GLOBAL_AFFINE_ALL_SETS",
)

print("\nSaved:", save_path)