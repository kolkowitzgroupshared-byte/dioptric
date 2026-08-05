import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

from utils import data_manager as dm
from utils import kplotlib as kpl


def load_combined_image(file_stem):
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True)

    if "img_array" in data:
        img = np.asarray(data["img_array"], dtype=np.float32)
    elif "ref_img_array" in data:
        img = np.asarray(data["ref_img_array"], dtype=np.float32)
    elif "sig_img_array" in data:
        img = np.asarray(data["sig_img_array"], dtype=np.float32)
    else:
        raise KeyError(f"No recognized image key in {file_stem}. Keys: {data.keys()}")

    return img


def preprocess_for_registration(img, sigma_bg=20, sigma_smooth=1.0):
    """
    Prepare image for rotation/translation registration.

    1. Convert to float.
    2. Remove slow background.
    3. Clip extreme values.
    4. Normalize to 0-1.
    """
    img = np.asarray(img, dtype=np.float32)

    # Remove slow spatial background.
    bg = gaussian_filter(img, sigma=sigma_bg)
    proc = img - bg

    # Light smoothing helps ECC registration.
    if sigma_smooth is not None and sigma_smooth > 0:
        proc = gaussian_filter(proc, sigma=sigma_smooth)

    # Percentile clipping for robustness.
    lo, hi = np.percentile(proc, [1, 99.8])
    proc = np.clip(proc, lo, hi)

    proc = proc - np.min(proc)
    denom = np.max(proc)
    if denom > 0:
        proc = proc / denom

    return proc.astype(np.float32)


def estimate_rotation_ecc(
    img_before,
    img_after,
    motion_model=cv2.MOTION_EUCLIDEAN,
    num_iters=5000,
    eps=1e-7,
):
    """
    Estimate transform mapping img_before -> img_after.

    For MOTION_EUCLIDEAN:
        M = [[cos(theta), -sin(theta), tx],
             [sin(theta),  cos(theta), ty]]

    In image coordinates, y points downward.
    """
    before = preprocess_for_registration(img_before)
    after = preprocess_for_registration(img_after)

    if before.shape != after.shape:
        raise ValueError(f"Image shapes differ: {before.shape} vs {after.shape}")

    # Initial transform: identity.
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        num_iters,
        eps,
    )

    cc, warp_matrix = cv2.findTransformECC(
        templateImage=after,
        inputImage=before,
        warpMatrix=warp_matrix,
        motionType=motion_model,
        criteria=criteria,
        inputMask=None,
        gaussFiltSize=5,
    )

    a = warp_matrix[0, 0]
    b = warp_matrix[1, 0]

    theta_image_deg = np.degrees(np.arctan2(b, a))

    # In normal math coordinates y is upward, so sign flips.
    theta_math_deg = -theta_image_deg

    tx = warp_matrix[0, 2]
    ty = warp_matrix[1, 2]

    return {
        "cc": cc,
        "warp_matrix": warp_matrix,
        "theta_image_deg": theta_image_deg,
        "theta_math_deg": theta_math_deg,
        "tx_px": tx,
        "ty_px": ty,
        "before_proc": before,
        "after_proc": after,
    }


def apply_warp(img, warp_matrix):
    h, w = img.shape
    return cv2.warpAffine(
        img.astype(np.float32),
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )


def plot_registration_check(result):
    before = result["before_proc"]
    after = result["after_proc"]
    M = result["warp_matrix"]

    before_registered = apply_warp(before, M)

    diff_before = after - before
    diff_after = after - before_registered

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    axes[0].imshow(before, cmap="gray")
    axes[0].set_title("Before")

    axes[1].imshow(after, cmap="gray")
    axes[1].set_title("After")

    axes[2].imshow(diff_before, cmap="bwr")
    axes[2].set_title("After - before")

    axes[3].imshow(diff_after, cmap="bwr")
    axes[3].set_title("After - registered before")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Rotation = {result['theta_image_deg']:+.3f} deg image coords "
        f"({result['theta_math_deg']:+.3f} deg math coords), "
        f"tx={result['tx_px']:.2f}px, ty={result['ty_px']:.2f}px, "
        f"ECC={result['cc']:.4f}"
    )

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    kpl.init_kplotlib()

    before_file = "2026_05_21-20_51_21-combined_image_array"
    # after_file = "2026_05_22-19_34_19-combined_image_array"
    # after_file = "2026_05_22-20_52_15-combined_image_array"
    after_file = "2026_05_23-13_25_26-combined_image_array"
    

    img_before = load_combined_image(before_file)
    img_after = load_combined_image(after_file)

    result = estimate_rotation_ecc(img_before, img_after)

    print("\n=== Image rotation estimate ===")
    print("Maps BEFORE image -> AFTER image")
    print("ECC correlation:", result["cc"])
    print("warp_matrix:")
    print(result["warp_matrix"])
    print(f"rotation in image coordinates: {result['theta_image_deg']:+.4f} deg")
    print(f"rotation in math coordinates:  {result['theta_math_deg']:+.4f} deg")
    print(f"translation: tx={result['tx_px']:+.2f} px, ty={result['ty_px']:+.2f} px")

    fig = plot_registration_check(result)
    plt.show(block=True)