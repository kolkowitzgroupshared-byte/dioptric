import os
import cv2
import numpy as np

def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    ones = np.ones((len(pts), 1), dtype=np.float32)
    return np.hstack([pts, ones]) @ M.T


save_path = "slmsuite/calibration/nuvu_to_thorcam_slm.npz"


cal_coords_thorcam_slm = np.array(
    [
        [848.56406461, 620.0],
        [571.43593539, 620.0],
        [710.0,        380.0],
    ],
    dtype=np.float32,
)

nuvu_raw = np.array(
[[82.355, 11.393], [91.011, 357.211], [336.384, 188.434]],

    dtype=np.float32,
)

NUVU_TRIANGLE_ORDER = [0, 1, 2]
cal_coords_nuvu = nuvu_raw[NUVU_TRIANGLE_ORDER]

zero_thorcam_slm = np.array([705.05789009, 519.93369574], dtype=np.float32)
zero_nuvu = np.array([190.694, 191.013], dtype=np.float32)

M_nuvu_to_thorcam_slm = cv2.getAffineTransform(
    cal_coords_nuvu.astype(np.float32),
    cal_coords_thorcam_slm.astype(np.float32),
)

pred_triangle = apply_affine(M_nuvu_to_thorcam_slm, cal_coords_nuvu)
triangle_residuals = cal_coords_thorcam_slm - pred_triangle

pred_zero = apply_affine(M_nuvu_to_thorcam_slm, zero_nuvu)[0]
zero_error_px = np.linalg.norm(pred_zero - zero_thorcam_slm)

print("M_nuvu_to_thorcam_slm:")
print(M_nuvu_to_thorcam_slm)

print("triangle residuals [px]:")
print(triangle_residuals)

print("predicted zero:", pred_zero)
print("actual zero:   ", zero_thorcam_slm)
print("zero error [px]:", zero_error_px)

os.makedirs(os.path.dirname(save_path), exist_ok=True)

np.savez_compressed(
    save_path,
    M_nuvu_to_thorcam_slm=M_nuvu_to_thorcam_slm.astype(np.float32),
    cal_coords_nuvu=cal_coords_nuvu.astype(np.float32),
    cal_coords_thorcam_slm=cal_coords_thorcam_slm.astype(np.float32),
    zero_nuvu=zero_nuvu.astype(np.float32),
    zero_thorcam_slm=zero_thorcam_slm.astype(np.float32),
    pred_zero_thorcam_slm=pred_zero.astype(np.float32),
    zero_error_px=np.array(zero_error_px, dtype=np.float32),
    nuvu_triangle_order=np.array(NUVU_TRIANGLE_ORDER, dtype=np.int32),
    convention="NUVU_TO_THORCAM_SLM",
)

print("Saved:", save_path)
