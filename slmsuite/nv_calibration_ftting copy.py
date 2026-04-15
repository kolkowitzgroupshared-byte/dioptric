import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl

kpl.init_kplotlib()

# ============================================================
# USER CONFIG
# ============================================================

NPZ_PATH = "slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz"

# Which channel do you want to align first?
CHANNEL = "green"   # "green" or "red"

# How to get new measured spot centers after each physical rotation?
# "manual" -> paste the fitted coordinates into terminal
# "image"  -> give a file_stem and fit spots from that image
MEASUREMENT_MODE = "manual"

# Save history here
HISTORY_JSON = "aod_rotation_history.json"

# Use 4 points, including center
PIXEL_COORDS_BASELINE = np.array([
    [209.693, 202.035],
    [355.855,  55.308],
    [220.425, 359.764],
    [ 25.893,  55.843],
], dtype=float)

GREEN_COORDS = np.array([
    [101.132, 100.716],
    [ 72.218, 124.960],
    [101.986,  72.343],
    [131.603, 130.079],
], dtype=float)

RED_COORDS = np.array([
    [66.259, 65.255],
    [41.505, 82.791],
    [68.562, 42.354],
    [89.161, 91.279],
], dtype=float)

# For auto-fit mode from image
AUTO_FIT_BOX_SIZE = 5

# Plot after each iteration
SHOW_PLOTS = True


# ============================================================
# 2D GAUSSIAN FIT
# ============================================================

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def fit_gaussian_2d_local(image, center, size=5):
    x0, y0 = center
    x_min = max(0, int(np.floor(x0 - size)))
    x_max = min(image.shape[1], int(np.ceil(x0 + size + 1)))
    y_min = max(0, int(np.floor(y0 - size)))
    y_max = min(image.shape[0], int(np.ceil(y0 + size + 1)))

    local = image[y_min:y_max, x_min:x_max]
    if local.size == 0:
        raise RuntimeError(f"Empty fit window around center {center}")

    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    initial_guess = (
        float(local.max() - local.min()),
        float(x0),
        float(y0),
        1.5,
        1.5,
        0.0,
        float(local.min()),
    )

    bounds = (
        [0, x_min, y_min, 0.3, 0.3, -np.pi/2, -np.inf],
        [np.inf, x_max, y_max, 8.0, 8.0,  np.pi/2,  np.inf],
    )

    popt, _ = curve_fit(
        gaussian_2d,
        (xx, yy),
        local.ravel(),
        p0=initial_guess,
        bounds=bounds,
        maxfev=20000,
    )

    amp, xo, yo, sigma_x, sigma_y, theta, offset = popt
    return np.array([xo, yo], dtype=float), popt


def fit_many_spots_from_image(image, initial_peaks, size=5):
    fitted = []
    params = []
    for peak in np.asarray(initial_peaks, dtype=float):
        xy, popt = fit_gaussian_2d_local(image, peak, size=size)
        fitted.append(xy)
        params.append(popt)
    return np.asarray(fitted, dtype=float), params


def plot_fit_overlay(image, initial_pts, fitted_pts, title="Spot fits"):
    fig, ax = plt.subplots()
    kpl.imshow(ax, image, cbar_label="Photons")
    initial_pts = np.asarray(initial_pts)
    fitted_pts = np.asarray(fitted_pts)
    ax.scatter(initial_pts[:, 0], initial_pts[:, 1], c="black", marker="x", label="Initial")
    ax.scatter(fitted_pts[:, 0], fitted_pts[:, 1], c="cyan", marker="o", label="Fitted")
    for i, p in enumerate(fitted_pts):
        ax.text(p[0] + 3, p[1] + 3, f"{i}", color="white", fontsize=10)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.show(block=False)


# ============================================================
# GEOMETRY / ALIGNMENT HELPERS
# ============================================================

def fit_affine(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    X = np.column_stack([src, np.ones(len(src))])
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)
    M = M.T
    A = M[:, :2]
    t = M[:, 2]
    return A, t


def apply_affine(src, A, t):
    src = np.asarray(src, dtype=float)
    return src @ A.T + t


def affine_rms_error(src, dst, A, t):
    pred = apply_affine(src, A, t)
    err = pred - np.asarray(dst, dtype=float)
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector")
    return v / n


def angle_deg(v):
    v = np.asarray(v, dtype=float)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def wrapped_angle_diff_deg(a, b):
    d = a - b
    while d > 90:
        d -= 180
    while d < -90:
        d += 180
    return float(d)


def inter_axis_angle_deg(basis):
    e1 = basis[:, 0]
    e2 = basis[:, 1]
    c = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def normalize_basis_cols(B):
    return np.column_stack([normalize(B[:, 0]), normalize(B[:, 1])])


def best_rigid_rotation_deg(from_basis, to_basis):
    """
    Best rigid rotation that maps from_basis -> to_basis, after normalizing columns.
    """
    F = normalize_basis_cols(np.asarray(from_basis, dtype=float))
    T = normalize_basis_cols(np.asarray(to_basis, dtype=float))

    U, _, Vt = np.linalg.svd(T @ F.T)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(theta)


def rotate_basis_90(basis):
    """
    Proper +90 degree rotation.
    """
    R = np.array([[0, -1],
                  [1,  0]], dtype=float)
    return R @ basis


def estimate_square_lattice_basis(points, n_neighbors=4):
    pts = np.asarray(points, dtype=float)
    N = len(pts)
    center = pts.mean(axis=0)

    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)

    nn_idx = np.argpartition(dist2, kth=n_neighbors-1, axis=1)[:, :n_neighbors]

    vecs = []
    weights = []
    for i in range(N):
        for j in nn_idx[i]:
            v = pts[j] - pts[i]
            d = np.linalg.norm(v)
            if d == 0:
                continue
            vecs.append(v)
            weights.append(1.0 / d)

    vecs = np.asarray(vecs, dtype=float)
    weights = np.asarray(weights, dtype=float)

    phi = np.arctan2(vecs[:, 1], vecs[:, 0])
    psi4 = np.sum(weights * np.exp(1j * 4 * phi)) / np.sum(weights)
    theta = np.angle(psi4) / 4.0

    e1 = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    e2 = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    basis = np.column_stack([e1, e2])
    return center, basis, theta


def choose_pillar_basis_orientation(pillar_basis, ref_basis):
    pillar_basis = np.asarray(pillar_basis, dtype=float).copy()
    ref_basis = np.asarray(ref_basis, dtype=float)
    if np.dot(pillar_basis[:, 0], ref_basis[:, 0]) < 0:
        pillar_basis *= -1
    return pillar_basis


def basis_metrics(A, pillar_basis):
    basis = np.asarray(A, dtype=float)
    axis1 = basis[:, 0]
    axis2 = basis[:, 1]
    p1 = pillar_basis[:, 0]
    p2 = pillar_basis[:, 1]

    axis1_angle = angle_deg(axis1)
    axis2_angle = angle_deg(axis2)
    pillar1_angle = angle_deg(p1)
    pillar2_angle = angle_deg(p2)

    return {
        "axis1_angle_deg": axis1_angle,
        "axis2_angle_deg": axis2_angle,
        "pillar1_angle_deg": pillar1_angle,
        "pillar2_angle_deg": pillar2_angle,
        "axis1_minus_pillar_deg": wrapped_angle_diff_deg(axis1_angle, pillar1_angle),
        "axis2_minus_pillar_deg": wrapped_angle_diff_deg(axis2_angle, pillar2_angle),
        "best_rotation_to_pillar_deg": best_rigid_rotation_deg(basis, pillar_basis),
        "inter_axis_angle_deg": inter_axis_angle_deg(basis),
        "orthogonality_error_deg": inter_axis_angle_deg(basis) - 90.0,
        "axis1_length": float(np.linalg.norm(axis1)),
        "axis2_length": float(np.linalg.norm(axis2)),
    }


def print_metrics(label, metrics, rms=None):
    print(f"\n=== {label} ===")
    print(f"axis 1 angle              : {metrics['axis1_angle_deg']:+.6f} deg")
    print(f"axis 2 angle              : {metrics['axis2_angle_deg']:+.6f} deg")
    print(f"axis 1 - pillar           : {metrics['axis1_minus_pillar_deg']:+.6f} deg")
    print(f"axis 2 - pillar           : {metrics['axis2_minus_pillar_deg']:+.6f} deg")
    print(f"best rigid rot to pillar  : {metrics['best_rotation_to_pillar_deg']:+.6f} deg")
    print(f"inter-axis angle          : {metrics['inter_axis_angle_deg']:+.6f} deg")
    print(f"orthogonality error       : {metrics['orthogonality_error_deg']:+.6f} deg")
    print(f"axis lengths              : {metrics['axis1_length']:.6f}, {metrics['axis2_length']:.6f}")
    if rms is not None:
        print(f"affine RMS error          : {rms:.6f} px")


# ============================================================
# VISUALIZATION
# ============================================================

def draw_basis(ax, origin, basis, label_prefix, axis_len=180, text_offset=8, color="k"):
    e1 = normalize(basis[:, 0]) * axis_len
    e2 = normalize(basis[:, 1]) * axis_len

    ax.arrow(origin[0], origin[1], e1[0], e1[1],
             length_includes_head=True, head_width=8, head_length=12,
             linewidth=2.2, color=color)
    ax.arrow(origin[0], origin[1], e2[0], e2[1],
             length_includes_head=True, head_width=8, head_length=12,
             linewidth=2.2, color=color)

    ax.text(origin[0] + e1[0] + text_offset, origin[1] + e1[1] + text_offset,
            f"{label_prefix}1", color=color, fontsize=11)
    ax.text(origin[0] + e2[0] + text_offset, origin[1] + e2[1] + text_offset,
            f"{label_prefix}2", color=color, fontsize=11)


def plot_alignment(nv_pixel, pillar_basis, src_coords, measured_pixels, A, channel_label, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    origin = nv_pixel.mean(axis=0)

    ax.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=8, alpha=0.30, color="gray", label="NVs")
    ax.scatter(measured_pixels[:, 0], measured_pixels[:, 1], s=80, marker="s",
               color="black", label=f"{channel_label} measured")

    pred = apply_affine(src_coords, A, np.zeros(2))  # not correct translation; skip for now

    color = "limegreen" if channel_label.lower() == "green" else "crimson"
    draw_basis(ax, origin, A, label_prefix=f"{channel_label[0]}_", axis_len=180, color=color)
    draw_basis(ax, origin, pillar_basis, label_prefix="p_", axis_len=180, color="royalblue")

    ax.set_title(f"{channel_label} vs pillar {title_suffix}")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)


# ============================================================
# INPUT / HISTORY
# ============================================================

def parse_coords_from_text(s):
    arr = np.array(json.loads(s), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected shape (N,2)")
    return arr


def save_history(path, history):
    Path(path).write_text(json.dumps(history, indent=2))


def load_history(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return []


def get_measured_pixels_manual(n_expected):
    print("\nPaste measured fitted coords as JSON-like list, e.g.")
    print('[[209.69, 202.03], [355.85, 55.31], [220.42, 359.76], [25.89, 55.84]]')
    s = input("measured_pixels = ").strip()
    arr = parse_coords_from_text(s)
    if len(arr) != n_expected:
        raise ValueError(f"Expected {n_expected} points, got {len(arr)}")
    return arr


def get_measured_pixels_from_image(initial_peaks, size=5):
    file_stem = input("file_stem = ").strip()
    data = dm.get_raw_data(file_stem=file_stem, load_npz=True)
    img = np.asarray(data["img_array"])
    fitted, _ = fit_many_spots_from_image(img, initial_peaks, size=size)
    print("Fitted coordinates:")
    print(np.round(fitted, 3).tolist())
    if SHOW_PLOTS:
        plot_fit_overlay(img, initial_peaks, fitted, title=f"Fitted spots: {file_stem}")
    return fitted


# ============================================================
# MAIN ALIGNMENT LOOP
# ============================================================

def run_alignment_loop(channel, src_coords, baseline_pixels, pillar_basis, nv_pixel):
    history = load_history(HISTORY_JSON)

    # Baseline fit
    A0, t0 = fit_affine(src_coords, baseline_pixels)
    rms0 = affine_rms_error(src_coords, baseline_pixels, A0, t0)
    m0 = basis_metrics(A0, pillar_basis)
    print_metrics(f"{channel.upper()} BASELINE", m0, rms=rms0)

    if SHOW_PLOTS:
        plot_alignment(nv_pixel, pillar_basis, src_coords, baseline_pixels, A0, channel, title_suffix="baseline")

    current_pixels = np.asarray(baseline_pixels, dtype=float)
    current_A = A0
    current_t = t0
    current_metrics = m0

    iteration = 0
    while True:
        print("\nOptions: [Enter] continue  |  q quit")
        cmd = input("choice = ").strip().lower()
        if cmd == "q":
            break

        mech_step_deg = float(input("Applied physical rotation step (signed deg) = ").strip())

        if MEASUREMENT_MODE == "manual":
            new_pixels = get_measured_pixels_manual(len(src_coords))
        elif MEASUREMENT_MODE == "image":
            # use previous measured spots as initial guesses
            new_pixels = get_measured_pixels_from_image(current_pixels, size=AUTO_FIT_BOX_SIZE)
        else:
            raise ValueError("MEASUREMENT_MODE must be 'manual' or 'image'")

        A_new, t_new = fit_affine(src_coords, new_pixels)
        rms_new = affine_rms_error(src_coords, new_pixels, A_new, t_new)
        m_new = basis_metrics(A_new, pillar_basis)

        print_metrics(f"{channel.upper()} AFTER STEP", m_new, rms=rms_new)

        prev_abs = abs(current_metrics["best_rotation_to_pillar_deg"])
        new_abs = abs(m_new["best_rotation_to_pillar_deg"])

        print(f"\nApplied mechanical step: {mech_step_deg:+.6f} deg")

        if new_abs < prev_abs:
            recommendation = "KEEP SAME mechanical direction"
        elif new_abs > prev_abs:
            recommendation = "REVERSE mechanical direction"
        else:
            recommendation = "NO CLEAR CHANGE"

        print(f"Recommendation: {recommendation}")
        print(f"Remaining best rigid rotation to pillar: {m_new['best_rotation_to_pillar_deg']:+.6f} deg")

        entry = {
            "iteration": iteration,
            "channel": channel,
            "mech_step_deg": mech_step_deg,
            "recommendation": recommendation,
            "measured_pixels": np.round(new_pixels, 6).tolist(),
            "metrics": m_new,
            "rms_px": rms_new,
        }
        history.append(entry)
        save_history(HISTORY_JSON, history)

        if SHOW_PLOTS:
            plot_alignment(nv_pixel, pillar_basis, src_coords, new_pixels, A_new, channel,
                           title_suffix=f"iter {iteration}")

        current_pixels = new_pixels
        current_A = A_new
        current_t = t_new
        current_metrics = m_new
        iteration += 1
 
        # optional stop hint
        if abs(current_metrics["best_rotation_to_pillar_deg"]) < 0.5:
            print("\nAlignment is already within ~0.5 deg of the pillar basis.")
            print("This is a good point to stop physical rotation and do final affine calibration.")

# ============================================================
# ENTRY POINT
# ============================================================
def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    nv_pixel = np.asarray(data["nv_coordinates"], dtype=float)
    print(f"Loaded {len(nv_pixel)} NVs")

    # use baseline affine from the selected channel only to choose pillar-basis sign
    if CHANNEL.lower() == "green":
        src_coords = GREEN_COORDS
    elif CHANNEL.lower() == "red":
        src_coords = RED_COORDS
    else:
        raise ValueError("CHANNEL must be 'green' or 'red'")

    baseline_pixels = PIXEL_COORDS_BASELINE

    A_ref, _ = fit_affine(src_coords, baseline_pixels)

    pillar_center, pillar_basis, pillar_theta = estimate_square_lattice_basis(
        nv_pixel, n_neighbors=4
    )
    pillar_basis = choose_pillar_basis_orientation(pillar_basis, A_ref)

    print("\nPillar basis:")
    print(pillar_basis)
    print(f"pillar axis1 angle = {angle_deg(pillar_basis[:, 0]):+.6f} deg")
    print(f"pillar axis2 angle = {angle_deg(pillar_basis[:, 1]):+.6f} deg")

    run_alignment_loop(
        channel=CHANNEL.lower(),
        src_coords=src_coords,
        baseline_pixels=baseline_pixels,
        pillar_basis=pillar_basis,
        nv_pixel=nv_pixel,
    )

if __name__ == "__main__":
    main()
    
