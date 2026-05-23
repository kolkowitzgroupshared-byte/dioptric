# -*- coding: utf-8 -*-
"""
AOD axis calculation relative to the rotated square pillar/NV array.

This script:
    1. Loads current after-rotation NV/pillar pixel coordinates.
    2. Fits green and red AOD affine maps into current camera pixel space.
    3. Checks affine fit residuals for green/red calibration.
    4. Estimates the pillar-array axes using a square-lattice constraint.
    5. Compares green/red AOD axes to the pillar axes.
    6. Suggests pillar-aligned 3-point calibration targets.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from utils import kplotlib as kpl

kpl.init_kplotlib()


# =============================================================================
# INPUTS
# =============================================================================

calibration_coords_pixel = np.array(
    [
        [199.693, 201.937],
        [342.930, 44.107],
        [204.188, 358.940],
        [10.992, 53.880],
    ],
    dtype=float,
)

calibration_coords_green = np.array(
    [
        [99.745, 99.794],
        [70.712, 125.372],
        [102.332, 71.682],
        [130.634, 130.331],
    ],
    dtype=float,
)

calibration_coords_red = np.array(
    [
        [65.590, 65.255],
        [42.505, 86.210],
        [67.262, 42.154],
        [90.161, 89.279],
    ],
    dtype=float,
)

# This should point to the chosen file produced by Script 1.
npz_path = "slmsuite/nv_blob_detection/nv_blob_1274nvs_reordered_after_sample_rotation.npz"

ROTATE_PILLAR_BY_90 = False
TARGET_STEP_IN_LATTICE_SPACINGS = 15
SHOW_PLOTS = True


# =============================================================================
# BASIC HELPERS
# =============================================================================

def fit_affine(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    if src.shape != dst.shape:
        raise ValueError(f"src and dst must have same shape, got {src.shape} and {dst.shape}")

    X = np.column_stack([src, np.ones(len(src))])
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)
    M = M.T

    A = M[:, :2]
    t = M[:, 2]
    return A, t


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def normalize_cols(B):
    B = np.asarray(B, dtype=float)
    return B / np.linalg.norm(B, axis=0, keepdims=True)


def angle_deg(v):
    v = np.asarray(v, dtype=float)
    return np.degrees(np.arctan2(v[1], v[0]))


def wrap_axis_angle_deg(angle):
    """
    Wrap angle to [-90, 90] because an axis is equivalent modulo 180 degrees.
    """
    return ((float(angle) + 90.0) % 180.0) - 90.0


def axis_delta_deg(a, b):
    """
    Difference a - b between two axis angles, modulo 180.
    """
    return ((float(a) - float(b) + 90.0) % 180.0) - 90.0


def rotate_basis_90(basis):
    R = np.array(
        [
            [0, -1],
            [1, 0],
        ],
        dtype=float,
    )
    return R @ basis


def axis_report(name, basis):
    e1 = basis[:, 0]
    e2 = basis[:, 1]

    print(f"\n=== {name} ===")
    print("axis 1 vector:", e1)
    print("axis 2 vector:", e2)
    print("axis 1 angle raw deg:", angle_deg(e1))
    print("axis 1 angle wrapped deg:", wrap_axis_angle_deg(angle_deg(e1)))
    print("axis 2 angle raw deg:", angle_deg(e2))
    print("axis 2 angle wrapped deg:", wrap_axis_angle_deg(angle_deg(e2)))
    print("axis 1 length:", np.linalg.norm(e1))
    print("axis 2 length:", np.linalg.norm(e2))

    cosang = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    cosang = np.clip(cosang, -1.0, 1.0)
    print("inter-axis angle deg:", np.degrees(np.arccos(cosang)))


def src_to_camera(src_pts, A, t):
    src_pts = np.asarray(src_pts, dtype=float)
    return src_pts @ A.T + t


def camera_to_src(pixel_pts, A, t):
    pixel_pts = np.asarray(pixel_pts, dtype=float)
    Ainv = np.linalg.inv(A)
    return (pixel_pts - t) @ Ainv.T


def report_affine_fit(name, src_pts, dst_pts, A, t):
    src_pts = np.asarray(src_pts, dtype=float)
    dst_pts = np.asarray(dst_pts, dtype=float)

    pred = src_to_camera(src_pts, A, t)
    residuals = dst_pts - pred
    err = np.linalg.norm(residuals, axis=1)

    rms = float(np.sqrt(np.mean(err**2)))
    max_err = float(np.max(err))

    print(f"\n=== {name} affine fit residuals ===")
    print("source coords:")
    print(src_pts)
    print("predicted pixel coords:")
    print(pred)
    print("actual pixel coords:")
    print(dst_pts)
    print("residuals [px]:")
    print(residuals)
    print("per-point error [px]:")
    print(err)
    print(f"RMS error [px]: {rms:.3f}")
    print(f"max error [px]: {max_err:.3f}")

    if rms > 5:
        print("WARNING: affine RMS error is large. Check point order or bad picked points.")

    return {
        "pred": pred,
        "residuals": residuals,
        "err": err,
        "rms": rms,
        "max": max_err,
    }


# =============================================================================
# PILLAR-LATTICE ESTIMATION
# =============================================================================

def estimate_square_lattice_basis(points, n_neighbors=4):
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)

    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist2, np.inf)

    nn_idx = np.argpartition(dist2, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]

    vecs = []
    weights = []

    for i in range(len(pts)):
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


def estimate_lattice_pitch(points):
    pts = np.asarray(points, dtype=float)

    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)

    nn1 = np.min(dist, axis=1)
    pitch = np.median(nn1)
    return float(pitch)


def choose_center_anchor(points):
    pts = np.asarray(points, dtype=float)
    ctr = pts.mean(axis=0)
    idx = int(np.argmin(np.sum((pts - ctr) ** 2, axis=1)))
    return pts[idx], idx


def fit_square_lattice_similarity(grid_ij, pixel_pts):
    """
    Fit:
        pixel = origin + i*u + j*v

    with:
        u = [a, b]
        v = [-b, a]
    """
    G = np.asarray(grid_ij, dtype=float)
    P = np.asarray(pixel_pts, dtype=float)

    if len(G) < 3:
        raise ValueError("Need at least 3 points to fit square lattice.")

    A_rows = []
    y_vals = []

    for (i, j), (x, y) in zip(G, P):
        A_rows.append([1.0, 0.0, i, -j])
        y_vals.append(x)

        A_rows.append([0.0, 1.0, j, i])
        y_vals.append(y)

    A_rows = np.asarray(A_rows, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    params, *_ = np.linalg.lstsq(A_rows, y_vals, rcond=None)
    ox, oy, a, b = params

    origin = np.array([ox, oy], dtype=float)
    basis = np.array(
        [
            [a, -b],
            [b, a],
        ],
        dtype=float,
    )

    pitch = float(np.sqrt(a**2 + b**2))
    pred = G @ basis.T + origin
    residuals = P - pred

    return basis, origin, pitch, residuals


def deduplicate_grid_assignments(grid_ij, pixel_pts, basis, origin):
    G = np.asarray(grid_ij, dtype=int)
    P = np.asarray(pixel_pts, dtype=float)

    pred = G @ basis.T + origin
    err = np.linalg.norm(P - pred, axis=1)

    best = {}

    for ind, g in enumerate(G):
        key = tuple(g.tolist())
        if key not in best or err[ind] < best[key][1]:
            best[key] = (ind, err[ind])

    keep_inds = sorted([val[0] for val in best.values()])
    return np.asarray(keep_inds, dtype=int)


def estimate_square_lattice_basis_constrained(
    points,
    n_neighbors=4,
    max_iter=10,
    residual_clip_frac=0.35,
    min_residual_clip_px=4.0,
    verbose=True,
):
    pts = np.asarray(points, dtype=float)
    num_pts = len(pts)

    _, basis0, _ = estimate_square_lattice_basis(pts, n_neighbors=n_neighbors)
    pitch0 = estimate_lattice_pitch(pts)
    anchor_pix, anchor_idx = choose_center_anchor(pts)

    coeff0 = (pts - anchor_pix) @ basis0 / pitch0
    grid_ij = np.rint(coeff0).astype(int)

    active = np.ones(num_pts, dtype=bool)

    basis = basis0 * pitch0
    origin = anchor_pix.copy()
    pitch = float(pitch0)

    best_state = None

    for it in range(max_iter):
        active_inds = np.where(active)[0]

        keep_local = deduplicate_grid_assignments(
            grid_ij[active_inds],
            pts[active_inds],
            basis,
            origin,
        )

        fit_inds = active_inds[keep_local]

        if len(fit_inds) < 3:
            raise RuntimeError("Too few inlier lattice points for square fit.")

        basis, origin, pitch, _ = fit_square_lattice_similarity(
            grid_ij[fit_inds],
            pts[fit_inds],
        )

        e1 = basis[:, 0] / np.linalg.norm(basis[:, 0])
        e2 = basis[:, 1] / np.linalg.norm(basis[:, 1])
        basis_unit = np.column_stack([e1, e2])

        coeff = (pts - origin) @ basis_unit / pitch
        new_grid_ij = np.rint(coeff).astype(int)

        pred_all = new_grid_ij @ basis.T + origin
        residuals = pts - pred_all
        resid_norm = np.linalg.norm(residuals, axis=1)

        clip_px = max(residual_clip_frac * pitch, min_residual_clip_px)
        new_active = resid_norm < clip_px

        rms = float(np.sqrt(np.nanmean(resid_norm[new_active] ** 2))) if np.sum(new_active) > 0 else np.inf

        if verbose:
            print(
                f"square fit iter {it}: "
                f"pitch={pitch:.3f} px, "
                f"inliers={np.sum(new_active)}/{num_pts}, "
                f"rms={rms:.3f} px, "
                f"clip={clip_px:.3f} px"
            )

        if best_state is None or rms < best_state["rms_residual_px"]:
            best_state = {
                "basis": basis.copy(),
                "origin": origin.copy(),
                "pitch": float(pitch),
                "basis_unit": basis_unit.copy(),
                "grid_ij": new_grid_ij.copy(),
                "active": new_active.copy(),
                "residuals": residuals.copy(),
                "resid_norm": resid_norm.copy(),
                "rms_residual_px": float(rms),
            }

        if np.array_equal(new_grid_ij, grid_ij) and np.array_equal(new_active, active):
            break

        grid_ij = new_grid_ij
        active = new_active

    if best_state is None:
        raise RuntimeError("Square-lattice fit failed.")

    basis = best_state["basis"]
    origin = best_state["origin"]
    pitch = best_state["pitch"]
    basis_unit = best_state["basis_unit"]
    grid_ij = best_state["grid_ij"]
    active = best_state["active"]

    residuals = best_state["residuals"]
    resid_norm = best_state["resid_norm"]
    rms = best_state["rms_residual_px"]

    center = pts[active].mean(axis=0) if np.sum(active) > 0 else pts.mean(axis=0)
    theta = np.arctan2(basis_unit[1, 0], basis_unit[0, 0])

    info = {
        "origin": origin,
        "pitch_px": pitch,
        "basis_px": basis,
        "basis_unit": basis_unit,
        "grid_ij": grid_ij,
        "active_mask": active,
        "residuals": residuals,
        "resid_norm": resid_norm,
        "rms_residual_px": rms,
        "anchor_idx": anchor_idx,
        "anchor_pix": anchor_pix,
        "initial_pitch_px": pitch0,
        "initial_basis": basis0,
    }

    return center, basis_unit, theta, info


def orient_square_basis_to_reference(basis, ref_vec):
    e1 = basis[:, 0]
    e2 = basis[:, 1]
    ref = normalize(ref_vec)

    variants = [
        np.column_stack([e1, e2]),
        np.column_stack([-e1, -e2]),
        np.column_stack([e2, -e1]),
        np.column_stack([-e2, e1]),
    ]

    scores = [np.dot(B[:, 0], ref) for B in variants]
    return variants[int(np.argmax(scores))]


# =============================================================================
# PLOTTING
# =============================================================================

def draw_basis(ax, origin, basis, label_prefix, axis_len=180, text_offset=10, color="k"):
    e1 = normalize(basis[:, 0]) * axis_len
    e2 = normalize(basis[:, 1]) * axis_len

    ax.arrow(
        origin[0], origin[1], e1[0], e1[1],
        length_includes_head=True,
        head_width=8,
        head_length=12,
        linewidth=2.5,
        color=color,
    )

    ax.arrow(
        origin[0], origin[1], e2[0], e2[1],
        length_includes_head=True,
        head_width=8,
        head_length=12,
        linewidth=2.5,
        color=color,
    )

    ax.text(origin[0] + e1[0] + text_offset, origin[1] + e1[1] + text_offset,
            f"{label_prefix} 1", fontsize=12, color=color)

    ax.text(origin[0] + e2[0] + text_offset, origin[1] + e2[1] + text_offset,
            f"{label_prefix} 2", fontsize=12, color=color)


def draw_single_square(ax, origin, basis, side=160, color="k"):
    u = normalize(basis[:, 0]) * side
    v = normalize(basis[:, 1]) * side

    p0 = origin
    p1 = origin + u
    p2 = origin + u + v
    p3 = origin + v

    sq = np.vstack([p0, p1, p2, p3, p0])
    ax.plot(sq[:, 0], sq[:, 1], linewidth=2.5, color=color)


def plot_square_lattice_fit(nv_pixel, pillar_info):
    pts = np.asarray(nv_pixel, dtype=float)
    active = pillar_info["active_mask"]
    grid_ij = pillar_info["grid_ij"]
    basis_px = pillar_info["basis_px"]
    origin = pillar_info["origin"]

    pred = grid_ij @ basis_px.T + origin

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(pts[~active, 0], pts[~active, 1], s=12, alpha=0.5, color="red", label="Rejected/outlier")
    ax.scatter(pts[active, 0], pts[active, 1], s=8, alpha=0.35, color="gray", label="NV/pillar positions")
    ax.scatter(pred[active, 0], pred[active, 1], s=10, alpha=0.6, color="royalblue", label="Square-lattice fit")

    active_inds = np.where(active)[0]
    stride = max(1, len(active_inds) // 150)
    for ind in active_inds[::stride]:
        ax.plot(
            [pred[ind, 0], pts[ind, 0]],
            [pred[ind, 1], pts[ind, 1]],
            color="orange",
            linewidth=0.7,
            alpha=0.6,
        )

    ax.set_title(
        f"Square-constrained pillar lattice fit\n"
        f"pitch={pillar_info['pitch_px']:.2f} px, "
        f"RMS residual={pillar_info['rms_residual_px']:.2f} px"
    )
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    plt.tight_layout()

    return fig


def plot_axes(nv_pixel, calibration_coords_pixel, common_origin, green_basis, red_basis, pillar_basis):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=8, alpha=0.35, color="gray", label="NVs")
    ax.scatter(calibration_coords_pixel[:, 0], calibration_coords_pixel[:, 1], s=90, marker="s", color="black", label="Calibration points")

    draw_basis(ax, common_origin, green_basis, "green", axis_len=200, color="limegreen")
    draw_basis(ax, common_origin, red_basis, "red", axis_len=200, color="crimson")
    draw_basis(ax, common_origin, pillar_basis, "pillar", axis_len=200, color="royalblue")

    ax.scatter([common_origin[0]], [common_origin[1]], s=120, marker="x", color="black", label="Common origin")

    ax.set_title("Green, red, and square-constrained pillar axes on camera")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    return fig


def plot_square_frames(nv_pixel, calibration_coords_pixel, common_origin, green_basis, red_basis, pillar_basis):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=8, alpha=0.35, color="gray")
    ax.scatter(calibration_coords_pixel[:, 0], calibration_coords_pixel[:, 1], s=90, marker="s", color="black")

    draw_single_square(ax, common_origin, green_basis, side=180, color="limegreen")
    draw_single_square(ax, common_origin, red_basis, side=180, color="crimson")
    draw_single_square(ax, common_origin, pillar_basis, side=180, color="royalblue")

    draw_basis(ax, common_origin, green_basis, "g", axis_len=120, color="limegreen")
    draw_basis(ax, common_origin, red_basis, "r", axis_len=120, color="crimson")
    draw_basis(ax, common_origin, pillar_basis, "p", axis_len=120, color="royalblue")

    ax.set_title("Green, red, and pillar frames")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.axis("equal")
    ax.invert_yaxis()
    plt.tight_layout()

    return fig


def plot_target_pixels(nv_pixel, calibration_coords_pixel, common_origin, green_basis, red_basis, pillar_basis, target_pixels_3, anchor_pix):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=8, alpha=0.35, color="gray")
    ax.scatter(calibration_coords_pixel[:, 0], calibration_coords_pixel[:, 1], s=90, marker="s", color="black", label="Current calib pixels")

    draw_basis(ax, common_origin, green_basis, "green", axis_len=180, color="limegreen")
    draw_basis(ax, common_origin, red_basis, "red", axis_len=180, color="crimson")
    draw_basis(ax, common_origin, pillar_basis, "pillar", axis_len=180, color="royalblue")

    ax.scatter(target_pixels_3[:, 0], target_pixels_3[:, 1], s=140, marker="+", linewidths=2.5, color="gold", label="3 target pixels")

    for k, p in enumerate(target_pixels_3):
        ax.text(p[0] + 6, p[1] + 6, f"T{k}", color="gold", fontsize=11)

    ax.scatter([anchor_pix[0]], [anchor_pix[1]], s=120, marker="x", color="cyan", label="Anchor pillar")

    ax.set_title("Pillar-aligned target pixels for AOD calibration")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    return fig


# =============================================================================
# ALIGNMENT
# =============================================================================

def best_rigid_rotation_deg(from_basis, to_basis):
    F = normalize_cols(from_basis)
    T = normalize_cols(to_basis)

    U, _, Vt = np.linalg.svd(T @ F.T)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    theta_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(theta_deg), R


def print_alignment_guidance(green_basis, red_basis, pillar_basis):
    green_rot_deg, _ = best_rigid_rotation_deg(green_basis, pillar_basis)
    red_rot_deg, _ = best_rigid_rotation_deg(red_basis, pillar_basis)

    print("\n=== Best rigid rotations into pillar basis ===")
    print(f"green -> pillar: {green_rot_deg:+.3f} deg")
    print(f"red   -> pillar: {red_rot_deg:+.3f} deg")

    print("\nInterpretation:")
    print("These are best-fit image-coordinate rotations that align each AOD basis to the pillar basis.")
    print("Mechanical sign may be opposite depending on stage convention.")
    print("Use magnitude first, then verify with calibration points.")

    if abs(green_rot_deg - red_rot_deg) > 2.0:
        print("\nWARNING:")
        print("Green and red require noticeably different rotations.")
        print("A common mechanical rotation will not perfectly align both.")

    return green_rot_deg, red_rot_deg


def make_triplet_targets(anchor_pix, pillar_basis, pitch_px, step=15):
    u = normalize(pillar_basis[:, 0]) * pitch_px * step
    v = normalize(pillar_basis[:, 1]) * pitch_px * step

    target_pixels = np.vstack(
        [
            anchor_pix,
            anchor_pix + u,
            anchor_pix + v,
        ]
    )

    return target_pixels, u, v


# =============================================================================
# MAIN
# =============================================================================

def main():
    with np.load(npz_path, allow_pickle=True) as data:
        nv_pixel = np.asarray(data["nv_coordinates"], dtype=float)

    print(f"Loaded {len(nv_pixel)} NVs")
    print("NPZ path:", npz_path)

    A_green, t_green = fit_affine(calibration_coords_green, calibration_coords_pixel)
    A_red, t_red = fit_affine(calibration_coords_red, calibration_coords_pixel)

    green_basis = A_green.copy()
    red_basis = A_red.copy()

    axis_report("GREEN basis on camera", green_basis)
    axis_report("RED basis on camera", red_basis)

    green_fit = report_affine_fit("GREEN", calibration_coords_green, calibration_coords_pixel, A_green, t_green)
    red_fit = report_affine_fit("RED", calibration_coords_red, calibration_coords_pixel, A_red, t_red)

    pillar_center, pillar_basis, pillar_theta, pillar_info = (
        estimate_square_lattice_basis_constrained(
            nv_pixel,
            n_neighbors=4,
            max_iter=10,
            residual_clip_frac=0.35,
            min_residual_clip_px=4.0,
            verbose=True,
        )
    )

    pillar_basis = orient_square_basis_to_reference(pillar_basis, red_basis[:, 0])

    if ROTATE_PILLAR_BY_90:
        pillar_basis = rotate_basis_90(pillar_basis)

    axis_report("PILLAR basis on camera square-constrained", pillar_basis)

    print("\n=== Square-constrained pillar lattice ===")
    print("pitch_px:", pillar_info["pitch_px"])
    print("rms residual px:", pillar_info["rms_residual_px"])
    print("num inliers:", np.sum(pillar_info["active_mask"]), "/", len(nv_pixel))

    if pillar_info["rms_residual_px"] > 5:
        print("\nWARNING:")
        print("Square-lattice RMS residual is larger than expected.")
        print("Check coordinate file, outliers, wrong rotation sign, or distortion.")

    green_ang1 = angle_deg(green_basis[:, 0])
    red_ang1 = angle_deg(red_basis[:, 0])
    pillar_ang1 = angle_deg(pillar_basis[:, 0])

    print("\n=== Misalignment summary using axis 1 ===")
    print("green axis 1 raw angle:", green_ang1)
    print("red axis 1 raw angle:", red_ang1)
    print("pillar axis 1 raw angle:", pillar_ang1)

    print("\n=== Wrapped angle summary ===")
    print("green wrapped:", wrap_axis_angle_deg(green_ang1))
    print("red wrapped:", wrap_axis_angle_deg(red_ang1))
    print("pillar wrapped:", wrap_axis_angle_deg(pillar_ang1))
    print("green - pillar wrapped:", axis_delta_deg(green_ang1, pillar_ang1))
    print("red   - pillar wrapped:", axis_delta_deg(red_ang1, pillar_ang1))
    print("green - red wrapped:", axis_delta_deg(green_ang1, red_ang1))

    green_rot_deg, red_rot_deg = print_alignment_guidance(
        green_basis,
        red_basis,
        pillar_basis,
    )

    pitch_px = pillar_info["pitch_px"]
    anchor_pix, anchor_idx = choose_center_anchor(nv_pixel)

    print(f"\nEstimated pillar pitch from constrained fit [px]: {pitch_px:.3f}")
    print(f"Chosen anchor index: {anchor_idx}")
    print(f"Chosen anchor pixel: {anchor_pix}")

    target_pixels_3, u_step, v_step = make_triplet_targets(
        anchor_pix=anchor_pix,
        pillar_basis=pillar_basis,
        pitch_px=pitch_px,
        step=TARGET_STEP_IN_LATTICE_SPACINGS,
    )

    print("\n=== Suggested 3-point target pixels ===")
    print(target_pixels_3)

    green_target_coords = camera_to_src(target_pixels_3, A_green, t_green)
    red_target_coords = camera_to_src(target_pixels_3, A_red, t_red)

    print("\n=== GREEN source coords for target pixels ===")
    print(green_target_coords)

    print("\n=== RED source coords for target pixels ===")
    print(red_target_coords)

    if SHOW_PLOTS:
        common_origin = nv_pixel.mean(axis=0)

        plot_square_lattice_fit(nv_pixel, pillar_info)
        plot_axes(nv_pixel, calibration_coords_pixel, common_origin, green_basis, red_basis, pillar_basis)
        plot_square_frames(nv_pixel, calibration_coords_pixel, common_origin, green_basis, red_basis, pillar_basis)
        plot_target_pixels(
            nv_pixel,
            calibration_coords_pixel,
            common_origin,
            green_basis,
            red_basis,
            pillar_basis,
            target_pixels_3,
            anchor_pix,
        )

        plt.show(block=True)

    return {
        "A_green": A_green,
        "t_green": t_green,
        "A_red": A_red,
        "t_red": t_red,
        "green_basis": green_basis,
        "red_basis": red_basis,
        "pillar_basis": pillar_basis,
        "pillar_info": pillar_info,
        "green_fit": green_fit,
        "red_fit": red_fit,
        "green_rot_deg": green_rot_deg,
        "red_rot_deg": red_rot_deg,
        "target_pixels_3": target_pixels_3,
        "green_target_coords": green_target_coords,
        "red_target_coords": red_target_coords,
    }


if __name__ == "__main__":
    main()