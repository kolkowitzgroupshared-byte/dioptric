# -*- coding: utf-8 -*-
"""
AOD axis calculation relative to rotated square NV/pillar array.

This version includes:
    - Raw affine fit for green/red AODs
    - 90-degree constrained AOD fit for green/red AODs
    - Square-lattice constrained pillar-axis estimate
    - Green/red/pillar angular mismatch
    - Suggested pillar-aligned 3-point calibration targets
    - One summary plot

Model for 90-constrained AOD fit:
    pixel = t + aod_x * u + aod_y * v

with:
    u = s1 * [cos(theta), sin(theta)]
    v = s2 * [-sin(theta), cos(theta)]

Therefore:
    angle(u, v) = 90 degrees exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import kplotlib as kpl

kpl.init_kplotlib()


# =============================================================================
# INPUTS
# =============================================================================

REF_PIXEL = np.array([186.627, 177.289], dtype=float)

calibration_coords_pixel = np.array(
    [
        [194.039, 189.963],
        [319.015, 83.106],
        [192.998, 353.981],
        [17.982, 41.943],
    ],
    dtype=float,
)

calibration_coords_green = np.array(
    [
        [97.763, 98.184],
        [73.727, 115.462],
        [100.738, 68.901],
        [126.958, 127.769],
    ],
    dtype=float,
)

calibration_coords_red = np.array(
    [
        [66.845, 67.629],
        [47.253, 81.361],
        [69.609, 44.003],
        [90.617, 92.506],
    ],
    dtype=float,
)

npz_path = "slmsuite/nv_blob_detection/nv_blob_1176nvs_reordered_inside_dmd.npz"

TARGET_STEP_IN_LATTICE_SPACINGS = 15
ROTATE_PILLAR_BY_90 = False
SHOW_PLOT = True

# If True, use 90-degree constrained AOD basis for final analysis.
# If False, use raw affine basis.
USE_90_CONSTRAINED_AOD = True


# =============================================================================
# BASIC HELPERS
# =============================================================================

def fit_affine(src, dst):
    """
    General affine fit:
        dst = src @ A.T + t
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    X = np.column_stack([src, np.ones(len(src))])
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)
    M = M.T

    return M[:, :2], M[:, 2]


def src_to_camera(src_pts, A, t):
    """
    Convert AOD/source coordinates to camera pixel coordinates.
    """
    src_pts = np.asarray(src_pts, dtype=float)
    return src_pts @ A.T + t


def camera_to_src(pixel_pts, A, t):
    """
    Convert camera pixel coordinates to AOD/source coordinates.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=float)
    return (pixel_pts - t) @ np.linalg.inv(A).T


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def normalize_cols(B):
    B = np.asarray(B, dtype=float)
    return B / np.linalg.norm(B, axis=0, keepdims=True)


def angle_deg(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def wrap_axis_angle_deg(angle):
    """
    Wrap an axis angle into [-90, +90) degrees.
    Axis has 180-degree equivalence.
    """
    return ((float(angle) + 90.0) % 180.0) - 90.0


def axis_delta_deg(a, b):
    """
    Difference between two axis angles, respecting 180-degree axis symmetry.
    """
    return ((float(a) - float(b) + 90.0) % 180.0) - 90.0


def rotate_basis_90(basis):
    return np.array([[0, -1], [1, 0]], dtype=float) @ basis


def choose_anchor_near_ref(points, ref_pixel):
    pts = np.asarray(points, dtype=float)
    idx = int(np.argmin(np.sum((pts - ref_pixel) ** 2, axis=1)))
    anchor = pts[idx]
    err = np.linalg.norm(anchor - ref_pixel)

    print("\n=== Reference anchor ===")
    print("requested REF_PIXEL:", ref_pixel)
    print("nearest saved NV index:", idx)
    print("nearest saved NV coord:", anchor)
    print(f"distance from REF_PIXEL: {err:.4f} px")

    return anchor, idx


def report_affine_fit(name, src_pts, dst_pts, A, t):
    pred = src_to_camera(src_pts, A, t)
    residuals = dst_pts - pred
    err = np.linalg.norm(residuals, axis=1)

    print(f"\n=== {name} affine fit ===")
    print("A matrix:")
    print(A)
    print("t vector:")
    print(t)
    print("RMS error [px]:", np.sqrt(np.mean(err**2)))
    print("max error [px]:", np.max(err))
    print("per-point error [px]:", err)

    return pred, residuals, err


def axis_report(name, basis):
    e1 = basis[:, 0]
    e2 = basis[:, 1]

    print(f"\n=== {name} ===")
    print("axis 1 vector:", e1)
    print("axis 2 vector:", e2)

    print("axis 1 angle raw:", angle_deg(e1))
    print("axis 1 angle wrapped:", wrap_axis_angle_deg(angle_deg(e1)))
    print("axis 2 angle raw:", angle_deg(e2))
    print("axis 2 angle wrapped:", wrap_axis_angle_deg(angle_deg(e2)))

    print("axis 1 length:", np.linalg.norm(e1))
    print("axis 2 length:", np.linalg.norm(e2))

    cosang = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    cosang = np.clip(cosang, -1.0, 1.0)
    inter_axis_angle = np.degrees(np.arccos(cosang))
    print("inter-axis angle:", inter_axis_angle)

    return inter_axis_angle


# =============================================================================
# 90-DEGREE CONSTRAINED AOD FIT
# =============================================================================

def _aod_90_objective_for_theta(theta, src_c, dst_c):
    """
    For a fixed theta, solve best s1 and s2 analytically.

    Model:
        dst_c = x * s1 * r + y * s2 * rp

    where:
        r  = [cos(theta), sin(theta)]
        rp = [-sin(theta), cos(theta)]

    Returns:
        objective, s1, s2
    """
    x = src_c[:, 0]
    y = src_c[:, 1]

    r = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    rp = np.array([-np.sin(theta), np.cos(theta)], dtype=float)

    dst_r = dst_c @ r
    dst_rp = dst_c @ rp

    denom_x = np.sum(x**2)
    denom_y = np.sum(y**2)

    if denom_x < 1e-15:
        s1 = 0.0
    else:
        s1 = np.sum(x * dst_r) / denom_x

    if denom_y < 1e-15:
        s2 = 0.0
    else:
        s2 = np.sum(y * dst_rp) / denom_y

    pred = np.outer(x * s1, r) + np.outer(y * s2, rp)
    resid = dst_c - pred
    obj = float(np.sum(resid**2))

    return obj, s1, s2


def _golden_section_minimize_periodic(func, a, b, n_iter=80):
    """
    Simple pure-numpy golden-section search.
    """
    gr = (np.sqrt(5.0) - 1.0) / 2.0

    c = b - gr * (b - a)
    d = a + gr * (b - a)

    fc = func(c)
    fd = func(d)

    for _ in range(n_iter):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = func(d)

    theta = 0.5 * (a + b)
    return theta, func(theta)


def fit_affine_90_constrained(src, dst, n_grid=720):
    """
    Fit an affine-like transform with constrained orthogonal columns.

    Model:
        dst = src @ A.T + t

    with:
        A[:, 0] = s1 * [cos(theta), sin(theta)]
        A[:, 1] = s2 * [-sin(theta), cos(theta)]

    Therefore the two AOD axes are exactly 90 degrees apart.

    This allows different magnifications along the two AOD axes,
    but no shear between them.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    def objective(theta):
        obj, _, _ = _aod_90_objective_for_theta(theta, src_c, dst_c)
        return obj

    # Coarse search over 0 to pi because axis direction has 180-degree symmetry.
    theta_grid = np.linspace(0.0, np.pi, n_grid, endpoint=False)
    obj_grid = np.array([objective(th) for th in theta_grid])
    best_idx = int(np.argmin(obj_grid))
    theta0 = theta_grid[best_idx]

    # Refine around the best coarse grid point.
    step = np.pi / n_grid
    a = theta0 - 2.0 * step
    b = theta0 + 2.0 * step

    theta_best, _ = _golden_section_minimize_periodic(objective, a, b)
    theta_best = theta_best % np.pi

    _, s1, s2 = _aod_90_objective_for_theta(theta_best, src_c, dst_c)

    r = np.array([np.cos(theta_best), np.sin(theta_best)], dtype=float)
    rp = np.array([-np.sin(theta_best), np.cos(theta_best)], dtype=float)

    A90 = np.column_stack([s1 * r, s2 * rp])

    # Translation after fixing A
    t90 = dst_mean - src_mean @ A90.T

    return A90, t90


def enforce_90_keep_axis1(A):
    """
    Alternative simple method:
    keep axis-1 direction fixed and rotate axis-2 to the nearest perpendicular.

    This is not the final method used by default, but useful for comparison.
    """
    e1 = normalize(A[:, 0])
    e2 = np.array([-e1[1], e1[0]], dtype=float)

    if np.dot(e2, A[:, 1]) < 0:
        e2 = -e2

    s1 = np.linalg.norm(A[:, 0])
    s2 = np.linalg.norm(A[:, 1])

    A90 = np.column_stack([s1 * e1, s2 * e2])
    return A90


def refit_translation_for_fixed_A(src, dst, A):
    """
    Best translation after fixing A.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    return np.mean(dst, axis=0) - np.mean(src, axis=0) @ A.T


def fit_affine_90_keep_axis1(src, dst):
    """
    Simple post-processing method:
        1. Fit raw affine.
        2. Force second axis to be perpendicular to first.
        3. Refit translation only.
    """
    A_raw, t_raw = fit_affine(src, dst)
    A90 = enforce_90_keep_axis1(A_raw)
    t90 = refit_translation_for_fixed_A(src, dst, A90)
    return A90, t90, A_raw, t_raw


# =============================================================================
# SQUARE-LATTICE ESTIMATION
# =============================================================================

def estimate_square_lattice_basis_initial(points, n_neighbors=4):
    pts = np.asarray(points, dtype=float)

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
            if d > 0:
                vecs.append(v)
                weights.append(1.0 / d)

    vecs = np.asarray(vecs, dtype=float)
    weights = np.asarray(weights, dtype=float)

    phi = np.arctan2(vecs[:, 1], vecs[:, 0])

    # Four-fold order parameter for square lattice.
    psi4 = np.sum(weights * np.exp(1j * 4 * phi)) / np.sum(weights)
    theta = np.angle(psi4) / 4.0

    e1 = np.array([np.cos(theta), np.sin(theta)])
    e2 = np.array([-np.sin(theta), np.cos(theta)])

    return np.column_stack([e1, e2])


def estimate_lattice_pitch(points):
    pts = np.asarray(points, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.median(np.min(dist, axis=1)))


def fit_square_lattice_similarity(grid_ij, pixel_pts):
    """
    Fit square model:
        pixel = origin + i*u + j*v

    with:
        u = [a, b]
        v = [-b, a]

    This forces:
        |u| = |v|
        angle(u, v) = 90 degrees
    """
    G = np.asarray(grid_ij, dtype=float)
    P = np.asarray(pixel_pts, dtype=float)

    A_rows = []
    y_vals = []

    for (i, j), (x, y) in zip(G, P):
        A_rows.append([1.0, 0.0, i, -j])
        y_vals.append(x)
        A_rows.append([0.0, 1.0, j, i])
        y_vals.append(y)

    params, *_ = np.linalg.lstsq(np.asarray(A_rows), np.asarray(y_vals), rcond=None)
    ox, oy, a, b = params

    origin = np.array([ox, oy])
    basis_px = np.array([[a, -b], [b, a]])
    pitch = float(np.sqrt(a**2 + b**2))

    pred = G @ basis_px.T + origin
    residuals = P - pred

    return basis_px, origin, pitch, residuals


def deduplicate_grid_assignments(grid_ij, pixel_pts, basis_px, origin):
    G = np.asarray(grid_ij, dtype=int)
    P = np.asarray(pixel_pts, dtype=float)

    pred = G @ basis_px.T + origin
    err = np.linalg.norm(P - pred, axis=1)

    best = {}
    for ind, g in enumerate(G):
        key = tuple(g.tolist())
        if key not in best or err[ind] < best[key][1]:
            best[key] = (ind, err[ind])

    return np.asarray(sorted(v[0] for v in best.values()), dtype=int)


def estimate_square_lattice_basis_constrained(
    points,
    ref_pixel,
    max_iter=10,
    residual_clip_frac=0.35,
    min_residual_clip_px=4.0,
):
    pts = np.asarray(points, dtype=float)
    n = len(pts)

    basis0 = estimate_square_lattice_basis_initial(pts)
    pitch0 = estimate_lattice_pitch(pts)
    anchor_pix, anchor_idx = choose_anchor_near_ref(pts, ref_pixel)

    grid_ij = np.rint((pts - anchor_pix) @ basis0 / pitch0).astype(int)
    active = np.ones(n, dtype=bool)

    basis_px = basis0 * pitch0
    origin = anchor_pix.copy()

    best = None

    for it in range(max_iter):
        active_inds = np.where(active)[0]

        keep_local = deduplicate_grid_assignments(
            grid_ij[active_inds],
            pts[active_inds],
            basis_px,
            origin,
        )
        fit_inds = active_inds[keep_local]

        basis_px, origin, pitch, _ = fit_square_lattice_similarity(
            grid_ij[fit_inds],
            pts[fit_inds],
        )

        basis_unit = normalize_cols(basis_px)

        grid_new = np.rint((pts - origin) @ basis_unit / pitch).astype(int)
        pred = grid_new @ basis_px.T + origin

        residuals = pts - pred
        resid_norm = np.linalg.norm(residuals, axis=1)

        clip_px = max(residual_clip_frac * pitch, min_residual_clip_px)
        active_new = resid_norm < clip_px

        if np.sum(active_new) > 0:
            rms = float(np.sqrt(np.mean(resid_norm[active_new] ** 2)))
        else:
            rms = np.inf

        print(
            f"square fit iter {it}: "
            f"pitch={pitch:.3f} px, "
            f"inliers={np.sum(active_new)}/{n}, "
            f"rms={rms:.3f} px"
        )

        if best is None or rms < best["rms_residual_px"]:
            best = {
                "basis_px": basis_px.copy(),
                "basis_unit": basis_unit.copy(),
                "origin": origin.copy(),
                "pitch_px": pitch,
                "grid_ij": grid_new.copy(),
                "active_mask": active_new.copy(),
                "residuals": residuals.copy(),
                "resid_norm": resid_norm.copy(),
                "rms_residual_px": rms,
                "anchor_pix": anchor_pix,
                "anchor_idx": anchor_idx,
            }

        if np.array_equal(grid_new, grid_ij) and np.array_equal(active_new, active):
            break

        grid_ij = grid_new
        active = active_new

    return best["basis_unit"], best


def orient_square_basis_to_reference(basis, ref_vec):
    """
    Choose among equivalent square-basis orientations so axis 1 is closest
    to a reference vector.
    """
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


def best_rigid_rotation_deg(from_basis, to_basis):
    """
    Best rotation angle that maps from_basis to to_basis.
    """
    F = normalize_cols(from_basis)
    T = normalize_cols(to_basis)

    U, _, Vt = np.linalg.svd(T @ F.T)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def make_triplet_targets(anchor_pix, pillar_basis, pitch_px, step):
    u = normalize(pillar_basis[:, 0]) * pitch_px * step
    v = normalize(pillar_basis[:, 1]) * pitch_px * step

    return np.vstack([anchor_pix, anchor_pix + u, anchor_pix + v])


# =============================================================================
# EXTRA DIAGNOSTICS
# =============================================================================

def compare_raw_and_constrained(name, src_pts, dst_pts, A_raw, t_raw, A90, t90):
    print(f"\n\n================ {name}: RAW VS 90-CONSTRAINED ================")

    _, _, err_raw = report_affine_fit(
        f"{name} RAW",
        src_pts,
        dst_pts,
        A_raw,
        t_raw,
    )

    _, _, err_90 = report_affine_fit(
        f"{name} 90-CONSTRAINED",
        src_pts,
        dst_pts,
        A90,
        t90,
    )

    print(f"\n{name} RMS increase from enforcing 90 deg [px]:")
    print(np.sqrt(np.mean(err_90**2)) - np.sqrt(np.mean(err_raw**2)))

    axis_report(f"{name} RAW basis", A_raw)
    axis_report(f"{name} 90-CONSTRAINED basis", A90)


def report_basis_mismatch(green_basis, red_basis, pillar_basis):
    green_ang = angle_deg(green_basis[:, 0])
    red_ang = angle_deg(red_basis[:, 0])
    pillar_ang = angle_deg(pillar_basis[:, 0])

    print("\n=== Wrapped angle summary ===")
    print("green wrapped:", wrap_axis_angle_deg(green_ang))
    print("red wrapped:", wrap_axis_angle_deg(red_ang))
    print("pillar wrapped:", wrap_axis_angle_deg(pillar_ang))

    print("\n=== Axis mismatch relative to pillar ===")
    print("green - pillar:", axis_delta_deg(green_ang, pillar_ang))
    print("red   - pillar:", axis_delta_deg(red_ang, pillar_ang))
    print("green - red:", axis_delta_deg(green_ang, red_ang))

    print("\n=== Best rigid rotations into pillar basis ===")
    print("green -> pillar:", best_rigid_rotation_deg(green_basis, pillar_basis))
    print("red   -> pillar:", best_rigid_rotation_deg(red_basis, pillar_basis))


# =============================================================================
# PLOTTING
# =============================================================================

def draw_basis(ax, origin, basis, label, color, axis_len=130):
    e1 = normalize(basis[:, 0]) * axis_len
    e2 = normalize(basis[:, 1]) * axis_len

    ax.arrow(
        origin[0],
        origin[1],
        e1[0],
        e1[1],
        length_includes_head=True,
        head_width=6,
        head_length=10,
        linewidth=2.2,
        color=color,
    )

    ax.arrow(
        origin[0],
        origin[1],
        e2[0],
        e2[1],
        length_includes_head=True,
        head_width=6,
        head_length=10,
        linewidth=2.2,
        color=color,
    )

    ax.text(origin[0] + e1[0] + 4, origin[1] + e1[1] + 4, f"{label} 1", color=color)
    ax.text(origin[0] + e2[0] + 4, origin[1] + e2[1] + 4, f"{label} 2", color=color)


def plot_summary(
    nv_pixel,
    common_origin,
    calib_pixels,
    green_basis,
    red_basis,
    pillar_basis,
    target_pixels,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        nv_pixel[:, 0],
        nv_pixel[:, 1],
        s=8,
        alpha=0.35,
        color="gray",
        label="NVs",
    )

    ax.scatter(
        calib_pixels[:, 0],
        calib_pixels[:, 1],
        s=20,
        marker="s",
        color="black",
        label="AOD calib pixels",
    )

    ax.scatter(
        [common_origin[0]],
        [common_origin[1]],
        s=20,
        marker="x",
        color="red",
        label="Reference NV",
    )

    draw_basis(ax, common_origin, green_basis, "green", "limegreen")
    draw_basis(ax, common_origin, red_basis, "red", "crimson")
    draw_basis(ax, common_origin, pillar_basis, "pillar", "royalblue")

    # ax.scatter(
    #     target_pixels[:, 0],
    #     target_pixels[:, 1],
    #     s=20,
    #     marker="+",
    #     linewidths=2.5,
    #     color="gold",
    #     label="Suggested targets",
    # )

    # for i, p in enumerate(target_pixels):
    #     ax.text(p[0] + 5, p[1] + 5, f"T{i}", color="gold")

    ax.set_title("AOD axes relative to pillar lattice")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    with np.load(npz_path, allow_pickle=True) as data:
        nv_pixel = np.asarray(data["nv_coordinates"], dtype=float)

    print(f"Loaded {len(nv_pixel)} NVs")
    print("NPZ path:", npz_path)

    common_origin, ref_idx = choose_anchor_near_ref(nv_pixel, REF_PIXEL)

    # -------------------------------------------------------------------------
    # Raw affine fits
    # -------------------------------------------------------------------------
    A_green_raw, t_green_raw = fit_affine(
        calibration_coords_green,
        calibration_coords_pixel,
    )

    A_red_raw, t_red_raw = fit_affine(
        calibration_coords_red,
        calibration_coords_pixel,
    )

    # -------------------------------------------------------------------------
    # 90-degree constrained AOD fits
    # -------------------------------------------------------------------------
    A_green_90, t_green_90 = fit_affine_90_constrained(
        calibration_coords_green,
        calibration_coords_pixel,
    )

    A_red_90, t_red_90 = fit_affine_90_constrained(
        calibration_coords_red,
        calibration_coords_pixel,
    )

    compare_raw_and_constrained(
        "GREEN",
        calibration_coords_green,
        calibration_coords_pixel,
        A_green_raw,
        t_green_raw,
        A_green_90,
        t_green_90,
    )

    compare_raw_and_constrained(
        "RED",
        calibration_coords_red,
        calibration_coords_pixel,
        A_red_raw,
        t_red_raw,
        A_red_90,
        t_red_90,
    )

    # Choose which AOD calibration to use downstream.
    if USE_90_CONSTRAINED_AOD:
        A_green, t_green = A_green_90, t_green_90
        A_red, t_red = A_red_90, t_red_90
        print("\nUsing 90-degree constrained AOD calibration downstream.")
    else:
        A_green, t_green = A_green_raw, t_green_raw
        A_red, t_red = A_red_raw, t_red_raw
        print("\nUsing raw affine AOD calibration downstream.")

    green_basis = A_green.copy()
    red_basis = A_red.copy()

    # -------------------------------------------------------------------------
    # Pillar square-lattice basis
    # -------------------------------------------------------------------------
    pillar_basis, pillar_info = estimate_square_lattice_basis_constrained(
        nv_pixel,
        ref_pixel=REF_PIXEL,
    )

    # Orient pillar axis 1 to be close to red AOD axis 1.
    pillar_basis = orient_square_basis_to_reference(pillar_basis, red_basis[:, 0])

    if ROTATE_PILLAR_BY_90:
        pillar_basis = rotate_basis_90(pillar_basis)

    # -------------------------------------------------------------------------
    # Axis reports
    # -------------------------------------------------------------------------
    axis_report("GREEN final basis", green_basis)
    axis_report("RED final basis", red_basis)
    axis_report("PILLAR basis", pillar_basis)

    report_basis_mismatch(green_basis, red_basis, pillar_basis)

    # -------------------------------------------------------------------------
    # Suggested pillar-aligned target pixels
    # -------------------------------------------------------------------------
    pitch_px = pillar_info["pitch_px"]
    target_pixels = make_triplet_targets(
        common_origin,
        pillar_basis,
        pitch_px,
        TARGET_STEP_IN_LATTICE_SPACINGS,
    )

    green_target_coords = camera_to_src(target_pixels, A_green, t_green)
    red_target_coords = camera_to_src(target_pixels, A_red, t_red)

    print("\n=== Square-lattice quality ===")
    print("pitch_px:", pitch_px)
    print("rms residual px:", pillar_info["rms_residual_px"])
    print("inliers:", np.sum(pillar_info["active_mask"]), "/", len(nv_pixel))

    print("\n=== Suggested target pixels ===")
    print(target_pixels)

    print("\n=== GREEN source coords for target pixels ===")
    print(green_target_coords)

    print("\n=== RED source coords for target pixels ===")
    print(red_target_coords)

    if SHOW_PLOT:
        plot_summary(
            nv_pixel,
            common_origin,
            calibration_coords_pixel,
            green_basis,
            red_basis,
            pillar_basis,
            target_pixels,
        )
        plt.show(block=True)

    return {
        "common_origin": common_origin,
        "ref_idx": ref_idx,

        "A_green_raw": A_green_raw,
        "t_green_raw": t_green_raw,
        "A_red_raw": A_red_raw,
        "t_red_raw": t_red_raw,

        "A_green_90": A_green_90,
        "t_green_90": t_green_90,
        "A_red_90": A_red_90,
        "t_red_90": t_red_90,

        "green_basis": green_basis,
        "red_basis": red_basis,
        "pillar_basis": pillar_basis,
        "pillar_info": pillar_info,

        "target_pixels": target_pixels,
        "green_target_coords": green_target_coords,
        "red_target_coords": red_target_coords,
    }


if __name__ == "__main__":
    results = main()