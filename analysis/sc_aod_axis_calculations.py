import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from utils import kplotlib as kpl
import matplotlib.lines as mlines
kpl.init_kplotlib()

# ============================================================
# INPUTS
# ============================================================

calibration_coords_pixel = np.array([
    [355.855,  55.308],
    [220.425, 359.764],
    [ 25.893,  55.843],
], dtype=float)

calibration_coords_green = np.array([
    [ 72.248, 124.933],
    [102.003,  72.355],
    [131.597, 130.110],
], dtype=float)

calibration_coords_red = np.array([
    [41.498, 82.808],
    [68.574, 42.356],
    [89.185, 91.264],
], dtype=float)

npz_path = "slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz"

# ============================================================
# HELPERS
# ============================================================

def fit_affine(src, dst):
    """
    Fit affine map:
        dst = A @ src + t
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    X = np.column_stack([src, np.ones(len(src))])   # (N,3)
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)     # (3,2)
    M = M.T                                         # (2,3)

    A = M[:, :2]
    t = M[:, 2]
    return A, t


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def angle_deg(v):
    v = np.asarray(v, dtype=float)
    return np.degrees(np.arctan2(v[1], v[0]))


def wrapped_angle_diff_deg(a, b):
    """
    Return difference wrapped to [-90, 90] because axes are equivalent modulo 180.
    """
    d = a - b
    while d > 90:
        d -= 180
    while d < -90:
        d += 180
    return d


def axis_report(name, basis):
    e1 = basis[:, 0]
    e2 = basis[:, 1]
    print(f"\n=== {name} ===")
    print("axis 1 vector:", e1)
    print("axis 2 vector:", e2)
    print("axis 1 angle (deg):", angle_deg(e1))
    print("axis 2 angle (deg):", angle_deg(e2))
    print("axis 1 length:", np.linalg.norm(e1))
    print("axis 2 length:", np.linalg.norm(e2))
    cosang = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    cosang = np.clip(cosang, -1.0, 1.0)
    print("inter-axis angle (deg):", np.degrees(np.arccos(cosang)))


def rotate_basis_90(basis):
    """
    Rotate the whole 2D basis by +90 degrees.
    """
    R = np.array([[1, 0],
                  [0,  1]], dtype=float)
    return R @ basis


# ============================================================
# PILLAR AXIS FROM SQUARE-LATTICE SYMMETRY, NOT PCA
# ============================================================

def estimate_square_lattice_basis(points, n_neighbors=4):
    """
    Estimate pillar-array axes from nearest-neighbor directions using 4-fold symmetry.
    Good for square arrays.

    Returns:
        center : mean point
        basis  : 2x2 orthonormal basis, columns = axis1, axis2
        theta  : axis1 angle in radians
    """
    pts = np.asarray(points, dtype=float)
    N = len(pts)
    center = pts.mean(axis=0)

    # Pairwise differences
    diff = pts[:, None, :] - pts[None, :, :]              # (N,N,2)
    dist2 = np.sum(diff**2, axis=2)                       # (N,N)

    # Ignore self-distance
    np.fill_diagonal(dist2, np.inf)

    # Take nearest neighbors for each point
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
            # weight nearer neighbors a bit more strongly
            weights.append(1.0 / d)

    vecs = np.asarray(vecs, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Angles of nearest-neighbor vectors
    phi = np.arctan2(vecs[:, 1], vecs[:, 0])

    # 4-fold orientational order parameter for square lattice
    psi4 = np.sum(weights * np.exp(1j * 4 * phi)) / np.sum(weights)
    theta = np.angle(psi4) / 4.0

    # Build orthonormal basis from theta
    e1 = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    e2 = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
    basis = np.column_stack([e1, e2])

    return center, basis, theta


# ============================================================
# PLOTTING
# ============================================================

def draw_basis(ax, origin, basis, label_prefix, axis_len=180, text_offset=10, color="k"):
    e1 = normalize(basis[:, 0]) * axis_len
    e2 = normalize(basis[:, 1]) * axis_len

    ax.arrow(
        origin[0], origin[1], e1[0], e1[1],
        length_includes_head=True, head_width=8, head_length=12,
        linewidth=2.5, color=color
    )
    ax.arrow(
        origin[0], origin[1], e2[0], e2[1],
        length_includes_head=True, head_width=8, head_length=12,
        linewidth=2.5, color=color
    )

    ax.text(
        origin[0] + e1[0] + text_offset,
        origin[1] + e1[1] + text_offset,
        f"{label_prefix} 1",
        fontsize=12, color=color
    )
    ax.text(
        origin[0] + e2[0] + text_offset,
        origin[1] + e2[1] + text_offset,
        f"{label_prefix} 2",
        fontsize=12, color=color
    )


def draw_single_square(ax, origin, basis, side=160, color="k"):
    u = normalize(basis[:, 0]) * side
    v = normalize(basis[:, 1]) * side

    p0 = origin
    p1 = origin + u
    p2 = origin + u + v
    p3 = origin + v

    sq = np.vstack([p0, p1, p2, p3, p0])
    ax.plot(sq[:, 0], sq[:, 1], linewidth=2.5, color=color)
# ============================================================
# LOAD NV DATA
# ============================================================

data = np.load(npz_path, allow_pickle=True)
nv_pixel = np.asarray(data["nv_coordinates"], dtype=float)

print(f"Loaded {len(nv_pixel)} NVs")


# ============================================================
# FIT GREEN / RED BASES IN CAMERA SPACE
# ============================================================

A_green, t_green = fit_affine(calibration_coords_green, calibration_coords_pixel)
A_red, t_red = fit_affine(calibration_coords_red, calibration_coords_pixel)

green_basis = A_green.copy()
red_basis = A_red.copy()

axis_report("GREEN basis on camera", green_basis)
axis_report("RED basis on camera", red_basis)


# ============================================================
# PILLAR BASIS FROM SQUARE-LATTICE ORIENTATION
# ============================================================

pillar_center, pillar_basis, pillar_theta = estimate_square_lattice_basis(
    nv_pixel,
    n_neighbors=4
)

# Make pillar axis 1 point in roughly the same direction as red axis 1
if np.dot(pillar_basis[:, 0], red_basis[:, 0]) < 0:
    pillar_basis = -pillar_basis
# If you prefer the 90-degree-rotated version, turn this on
ROTATE_PILLAR_BY_90 = False
if ROTATE_PILLAR_BY_90:
    pillar_basis = rotate_basis_90(pillar_basis)

axis_report("PILLAR basis on camera (square-lattice estimate)", pillar_basis)


# ============================================================
# COMMON ORIGIN FOR EVERYTHING
# ============================================================

common_origin = nv_pixel.mean(axis=0)

# ============================================================
# COLORS
# ============================================================

green_color = "limegreen"
red_color = "crimson"
pillar_color = "royalblue"
nv_color = "gray"
cal_color = "black"
origin_color = "black"


# ============================================================
# LEGEND HANDLES
# ============================================================

legend_handles_axes = [
    mlines.Line2D([], [], color=green_color, lw=3, label="Green AOD axes"),
    mlines.Line2D([], [], color=red_color, lw=3, label="Red AOD axes"),
    mlines.Line2D([], [], color=pillar_color, lw=3, label="Pillar axes"),
    mlines.Line2D([], [], color=nv_color, marker="o", linestyle="None",
                  markersize=6, label="NVs"),
    mlines.Line2D([], [], color=cal_color, marker="s", linestyle="None",
                  markersize=7, label="Calibration points"),
    mlines.Line2D([], [], color=origin_color, marker="x", linestyle="None",
                  markersize=8, markeredgewidth=2, label="Common origin"),
]

legend_handles_squares = [
    mlines.Line2D([], [], color=green_color, lw=3, label="Green AOD frame"),
    mlines.Line2D([], [], color=red_color, lw=3, label="Red AOD frame"),
    mlines.Line2D([], [], color=pillar_color, lw=3, label="Pillar frame"),
    mlines.Line2D([], [], color=nv_color, marker="o", linestyle="None",
                  markersize=6, label="NVs"),
    mlines.Line2D([], [], color=cal_color, marker="s", linestyle="None",
                  markersize=7, label="Calibration points"),
    mlines.Line2D([], [], color=origin_color, marker="x", linestyle="None",
                  markersize=8, markeredgewidth=2, label="Common origin"),
]


# ============================================================
# PLOT 1: ALL AXES FROM SAME POINT, VERY LARGE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    nv_pixel[:, 0], nv_pixel[:, 1],
    s=8, alpha=0.35, color=nv_color
)

ax.scatter(
    calibration_coords_pixel[:, 0], calibration_coords_pixel[:, 1],
    s=90, marker="s", color=cal_color
)

draw_basis(
    ax, common_origin, green_basis,
    label_prefix="green", axis_len=200, color=green_color
)
draw_basis(
    ax, common_origin, red_basis,
    label_prefix="red", axis_len=200, color=red_color
)
draw_basis(
    ax, common_origin, pillar_basis,
    label_prefix="pillar", axis_len=200, color=pillar_color
)

ax.scatter(
    [common_origin[0]], [common_origin[1]],
    s=120, marker="x", color=origin_color
)

ax.set_title("Green, red, and pillar axes on camera")
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")
ax.axis("equal")
ax.invert_yaxis()
ax.legend(
    handles=legend_handles_axes,
    loc="upper right",
    frameon=True,
    fontsize=11,
    title="Legend",
    title_fontsize=12
)
plt.tight_layout()
plt.show()


# ============================================================
# PLOT 2: ONE BIG SQUARE FOR EACH, SAME ORIGIN
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    nv_pixel[:, 0], nv_pixel[:, 1],
    s=8, alpha=0.35, color=nv_color
)

ax.scatter(
    calibration_coords_pixel[:, 0], calibration_coords_pixel[:, 1],
    s=90, marker="s", color=cal_color
)

draw_single_square(
    ax, common_origin, green_basis,
    side=180, color=green_color
)
draw_single_square(
    ax, common_origin, red_basis,
    side=180, color=red_color
)
draw_single_square(
    ax, common_origin, pillar_basis,
    side=180, color=pillar_color
)

draw_basis(
    ax, common_origin, green_basis,
    label_prefix="g", axis_len=120, color=green_color
)
draw_basis(
    ax, common_origin, red_basis,
    label_prefix="r", axis_len=120, color=red_color
)
draw_basis(
    ax, common_origin, pillar_basis,
    label_prefix="p", axis_len=120, color=pillar_color
)

ax.scatter(
    [common_origin[0]], [common_origin[1]],
    s=120, marker="x", color=origin_color
)

ax.set_title("One big square for green, red, and pillar bases")
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")
ax.axis("equal")
ax.invert_yaxis()
ax.legend(
    handles=legend_handles_squares,
    loc="upper right",
    frameon=True,
    fontsize=11,
    title="Legend",
    title_fontsize=12
)
plt.tight_layout()
plt.show()




# ============================================================
# ANGLE SUMMARY
# ============================================================

green_ang1 = angle_deg(green_basis[:, 0])
red_ang1 = angle_deg(red_basis[:, 0])
pillar_ang1 = angle_deg(pillar_basis[:, 0])

print("\n=== Misalignment summary using axis 1 ===")
print("green axis 1 angle:", green_ang1)
print("red axis 1 angle:", red_ang1)
print("pillar axis 1 angle:", pillar_ang1)

print("green - pillar (deg):", wrapped_angle_diff_deg(green_ang1, pillar_ang1))
print("red   - pillar (deg):", wrapped_angle_diff_deg(red_ang1, pillar_ang1))
print("green - red    (deg):", wrapped_angle_diff_deg(green_ang1, red_ang1))



import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FIX THIS BUG
# ============================================================

def rotate_basis_90(basis):
    """
    Rotate the whole 2D basis by +90 degrees.
    """
    R = np.array([[0, -1],
                  [1,  0]], dtype=float)
    return R @ basis


# ============================================================
# EXTRA HELPERS FOR ALIGNMENT
# ============================================================

def normalize_cols(B):
    B = np.asarray(B, dtype=float)
    return B / np.linalg.norm(B, axis=0, keepdims=True)


def best_rigid_rotation_deg(from_basis, to_basis):
    """
    Best rigid rotation angle that maps from_basis -> to_basis
    after column normalization.
    """
    F = normalize_cols(from_basis)
    T = normalize_cols(to_basis)

    U, _, Vt = np.linalg.svd(T @ F.T)
    R = U @ Vt

    # enforce proper rotation
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    theta_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return theta_deg, R


def estimate_lattice_pitch(points):
    """
    Estimate square-lattice pitch from nearest-neighbor distances.
    """
    pts = np.asarray(points, dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)

    nn1 = np.min(dist, axis=1)
    pitch = np.median(nn1)
    return pitch


def choose_center_anchor(points):
    """
    Pick the measured NV/pillar center closest to the cloud center.
    """
    pts = np.asarray(points, dtype=float)
    ctr = pts.mean(axis=0)
    idx = np.argmin(np.sum((pts - ctr) ** 2, axis=1))
    return pts[idx], idx


def make_triplet_targets(anchor_pix, pillar_basis, pitch_px, step=8):
    """
    3-point calibration target set aligned to the pillar lattice.
    """
    u = normalize(pillar_basis[:, 0]) * pitch_px * step
    v = normalize(pillar_basis[:, 1]) * pitch_px * step

    target_pixels = np.vstack([
        anchor_pix,
        anchor_pix + u,
        anchor_pix + v,
    ])
    return target_pixels, u, v


def make_grid_targets(anchor_pix, pillar_basis, pitch_px, half_span=2):
    """
    2D grid of pillar-aligned target positions in camera pixels.
    Good for a more robust multi-point calibration.
    """
    u = normalize(pillar_basis[:, 0]) * pitch_px
    v = normalize(pillar_basis[:, 1]) * pitch_px

    pts = []
    ij = []
    for i in range(-half_span, half_span + 1):
        for j in range(-half_span, half_span + 1):
            pts.append(anchor_pix + i * u + j * v)
            ij.append((i, j))
    return np.asarray(pts, dtype=float), ij


def src_to_camera(src_pts, A, t):
    src_pts = np.asarray(src_pts, dtype=float)
    return src_pts @ A.T + t


def camera_to_src(pixel_pts, A, t):
    """
    Convert desired camera pixel positions back into source coordinates
    (green or red command coordinates) using the current affine map.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=float)
    Ainv = np.linalg.inv(A)
    return (pixel_pts - t) @ Ainv.T


def angle_err_to_pillar_deg(basis, pillar_basis):
    theta, _ = best_rigid_rotation_deg(basis, pillar_basis)
    return theta


def print_alignment_guidance():
    green_rot_deg, _ = best_rigid_rotation_deg(green_basis, pillar_basis)
    red_rot_deg, _ = best_rigid_rotation_deg(red_basis, pillar_basis)

    print("\n=== Best rigid rotations into pillar basis ===")
    print(f"green -> pillar: {green_rot_deg:+.3f} deg")
    print(f"red   -> pillar: {red_rot_deg:+.3f} deg")

    print("\nInterpretation:")
    print("Negative means the printed AOD angle should DECREASE in your current angle convention.")
    print("For your current data, green should move by about -5.4 deg.")
    print("Red is already close, so a common rotation of both AODs is not ideal.")

    if abs(red_rot_deg) < 1.0 and abs(green_rot_deg) > 3.0:
        print("WARNING: red is already nearly aligned to the pillar axis.")
        print("If green and red rotate together mechanically, you cannot make both perfect at once.")


# ============================================================
# AFTER YOU COMPUTE green_basis, red_basis, pillar_basis
# ============================================================

pitch_px = estimate_lattice_pitch(nv_pixel)
anchor_pix, anchor_idx = choose_center_anchor(nv_pixel)

print(f"\nEstimated pillar pitch (pixels): {pitch_px:.3f}")
print(f"Chosen anchor index: {anchor_idx}")
print(f"Chosen anchor pixel: {anchor_pix}")

print_alignment_guidance()

# 3-point pillar-aligned targets on camera
target_pixels_3, u_step, v_step = make_triplet_targets(
    anchor_pix=anchor_pix,
    pillar_basis=pillar_basis,
    pitch_px=pitch_px,
    step=8,   # try 6, 8, or 10 depending on your FOV
)

print("\n=== Suggested 3-point target pixels for GREEN ===")
print(target_pixels_3)

# If you want the CURRENT green mapping to hit those camera pixels,
# these are the green coordinates you would command:
green_target_coords = camera_to_src(target_pixels_3, A_green, t_green)

print("\n=== GREEN source coords that correspond to those target pixels ===")
print(green_target_coords)

# You can do the same for red if needed:
red_target_coords = camera_to_src(target_pixels_3, A_red, t_red)

print("\n=== RED source coords that correspond to those target pixels ===")
print(red_target_coords)


# ============================================================
# OPTIONAL: plot the target pixels on top of your NV image
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=8, alpha=0.35, color="gray")
ax.scatter(calibration_coords_pixel[:, 0], calibration_coords_pixel[:, 1],
           s=90, marker="s", color="black", label="Current calib pixels")

# Existing basis overlays
draw_basis(ax, common_origin, green_basis,  "green",  axis_len=180, color="limegreen")
draw_basis(ax, common_origin, red_basis,    "red",    axis_len=180, color="crimson")
draw_basis(ax, common_origin, pillar_basis, "pillar", axis_len=180, color="royalblue")

# New target pixels
ax.scatter(target_pixels_3[:, 0], target_pixels_3[:, 1],
           s=140, marker="+", linewidths=2.5, color="gold", label="3 target pixels")

for k, p in enumerate(target_pixels_3):
    ax.text(p[0] + 6, p[1] + 6, f"T{k}", color="gold", fontsize=11)

ax.scatter([anchor_pix[0]], [anchor_pix[1]], s=120, marker="x", color="cyan", label="Anchor pillar")

ax.set_title("Pillar-aligned target pixels for green calibration")
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")
ax.axis("equal")
ax.invert_yaxis()
ax.legend(loc="upper right", frameon=True)
plt.tight_layout()

plt.show(block=True)