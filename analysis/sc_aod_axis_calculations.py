import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from utils import kplotlib as kpl
kpl.init_kplotlib()

# ============================================================
# 1. INPUTS
# ============================================================

# Your 3-point calibration
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

# Path to your uploaded NV coordinates
npz_path = "slmsuite/nv_blob_detection/nv_blob_1277nvs_reordered.npz"


# ============================================================
# 2. BASIC CALIBRATION FUNCTIONS
# ============================================================

def fit_affine(src, dst):
    """
    Fit affine map:
        dst = A @ src + t

    src: (N,2)
    dst: (N,2)

    Returns:
        A: (2,2)
        t: (2,)
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    X = np.column_stack([src, np.ones(len(src))])   # (N,3)
    M, *_ = np.linalg.lstsq(X, dst, rcond=None)     # (3,2)
    M = M.T                                         # (2,3)

    A = M[:, :2]
    t = M[:, 2]
    return A, t


def apply_affine(coords, A, t):
    """
    coords: (N,2)
    returns transformed coords: (N,2)
    """
    coords = np.asarray(coords, dtype=float)
    return (A @ coords.T).T + t


def invert_affine(coords, A, t):
    """
    Invert:
        pixel = A @ aod + t
    so:
        aod = A^{-1} @ (pixel - t)
    """
    coords = np.asarray(coords, dtype=float)
    Ainv = np.linalg.inv(A)
    return (Ainv @ (coords - t).T).T


def axis_info(A):
    """
    Returns basis vectors, angles, scales, and inter-axis angle.
    """
    a1 = A[:, 0]
    a2 = A[:, 1]

    len1 = np.linalg.norm(a1)
    len2 = np.linalg.norm(a2)

    ang1 = np.degrees(np.arctan2(a1[1], a1[0]))
    ang2 = np.degrees(np.arctan2(a2[1], a2[0]))

    cosang = np.dot(a1, a2) / (len1 * len2)
    cosang = np.clip(cosang, -1.0, 1.0)
    inter_axis = np.degrees(np.arccos(cosang))

    return {
        "axis1_vector": a1,
        "axis2_vector": a2,
        "axis1_scale": len1,
        "axis2_scale": len2,
        "axis1_angle_deg": ang1,
        "axis2_angle_deg": ang2,
        "inter_axis_angle_deg": inter_axis,
    }


# ============================================================
# 3. LOAD NV DATA
# ============================================================

data = np.load(npz_path, allow_pickle=True)

nv_pixel = np.asarray(data["nv_coordinates"], dtype=float)

if "updated_spot_weights" in data.files:
    nv_weights = np.asarray(data["updated_spot_weights"], dtype=float)
else:
    nv_weights = None

print(f"Loaded {len(nv_pixel)} NV coordinates")


# ============================================================
# 4. FIT GREEN / RED CALIBRATIONS
# ============================================================

A_green, t_green = fit_affine(calibration_coords_green, calibration_coords_pixel)
A_red, t_red = fit_affine(calibration_coords_red, calibration_coords_pixel)

print("\n=== GREEN calibration: pixel = A_green @ green + t_green ===")
print("A_green =\n", A_green)
print("t_green =", t_green)
print(axis_info(A_green))

print("\n=== RED calibration: pixel = A_red @ red + t_red ===")
print("A_red =\n", A_red)
print("t_red =", t_red)
print(axis_info(A_red))


# ============================================================
# 5. TRANSFORM NVs INTO AOD SPACE
# ============================================================

# Choose which AOD space you want to analyze.
# Usually green is the initialization / main addressing path.
# You can switch between "green" and "red".
ANALYSIS_SPACE = "green"

if ANALYSIS_SPACE == "green":
    A_use = A_green
    t_use = t_green
elif ANALYSIS_SPACE == "red":
    A_use = A_red
    t_use = t_red
else:
    raise ValueError("ANALYSIS_SPACE must be 'green' or 'red'")

nv_aod = invert_affine(nv_pixel, A_use, t_use)


# ============================================================
# 6. PLOTTING UTILITIES
# ============================================================

def plot_camera_axes(pixel_pts, A_green, A_red, title="AOD axes in camera pixel space"):
    plt.figure(figsize=(8, 7))
    plt.scatter(pixel_pts[:, 0], pixel_pts[:, 1], s=60, label="Calibration points")

    origin = np.mean(pixel_pts, axis=0)

    # Use normalized arrows for visibility
    g1 = A_green[:, 0]
    g2 = A_green[:, 1]
    r1 = A_red[:, 0]
    r2 = A_red[:, 1]

    scale = 25.0

    def draw_vec(vec, label):
        v = vec / np.linalg.norm(vec) * scale
        plt.arrow(
            origin[0], origin[1], v[0], v[1],
            length_includes_head=True, head_width=5, head_length=8
        )
        plt.text(origin[0] + v[0], origin[1] + v[1], label)

    draw_vec(g1, "green axis 1")
    draw_vec(g2, "green axis 2")
    draw_vec(r1, "red axis 1")
    draw_vec(r2, "red axis 2")

    plt.scatter([origin[0]], [origin[1]], s=80, marker="x", label="origin")
    plt.gca().invert_yaxis()  # camera-style if desired; comment out if not wanted
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.title(title)
    plt.legend()
    plt.axis("equal")

def plot_nv_camera_and_aod(nv_pixel, nv_aod, weights=None):
    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    if weights is None:
        ax1.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=8)
    else:
        ax1.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=10, c=weights)
        plt.colorbar(ax1.collections[0], ax=ax1, label="weight")
    ax1.set_title("NVs in camera pixel space")
    ax1.set_xlabel("Pixel X")
    ax1.set_ylabel("Pixel Y")
    ax1.axis("equal")
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(1, 2, 2)
    if weights is None:
        ax2.scatter(nv_aod[:, 0], nv_aod[:, 1], s=8)
    else:
        ax2.scatter(nv_aod[:, 0], nv_aod[:, 1], s=10, c=weights)
        plt.colorbar(ax2.collections[0], ax=ax2, label="weight")
    ax2.set_title(f"NVs in {ANALYSIS_SPACE} AOD coordinates")
    ax2.set_xlabel("u (AOD axis 1)")
    ax2.set_ylabel("v (AOD axis 2)")
    ax2.axis("equal")


def plot_example_groups(nv_pixel, groups, title, max_groups=10):
    """
    groups: list of lists of NV indices
    """
    plt.figure(figsize=(8, 7))
    plt.scatter(nv_pixel[:, 0], nv_pixel[:, 1], s=8, alpha=0.3)

    for i, g in enumerate(groups[:max_groups]):
        pts = nv_pixel[np.array(g)]
        plt.scatter(pts[:, 0], pts[:, 1], s=40)
        for j, idx in enumerate(g):
            plt.text(pts[j, 0], pts[j, 1], str(idx), fontsize=8)

        # Close loop for 4-point groups
        if len(g) == 4:
            ordered = pts[np.argsort(pts[:, 0] + 0.2 * pts[:, 1])]
            cx, cy = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
            cyc = pts[np.argsort(angles)]
            cyc = np.vstack([cyc, cyc[0]])
            plt.plot(cyc[:, 0], cyc[:, 1], linewidth=1)

        elif len(g) == 2:
            plt.plot(pts[:, 0], pts[:, 1], linewidth=1)

    plt.gca().invert_yaxis()
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.title(title)
    plt.axis("equal")

# ============================================================
# 7. BINNING / GROUPING IN AOD SPACE
# ============================================================

def quantize(values, tol):
    """
    Convert continuous values to integer bins.
    """
    return np.round(values / tol).astype(int)


def find_1x2_pairs(nv_aod, u_tol=0.5, v_min_sep=0.8, v_max_sep=20.0):
    """
    Find NV pairs that share nearly same u (same AOD axis 1 coordinate),
    but differ in v -> valid for 1x2 addressing.
    """
    u = nv_aod[:, 0]
    v = nv_aod[:, 1]

    u_bin = quantize(u, u_tol)
    groups = defaultdict(list)
    for i, b in enumerate(u_bin):
        groups[b].append(i)

    pairs = []
    for b, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            dv = abs(v[i] - v[j])
            if v_min_sep <= dv <= v_max_sep:
                pairs.append([i, j])

    return pairs


def find_2x1_pairs(nv_aod, v_tol=0.5, u_min_sep=0.8, u_max_sep=20.0):
    """
    Find NV pairs that share nearly same v,
    but differ in u -> valid for 2x1 addressing.
    """
    u = nv_aod[:, 0]
    v = nv_aod[:, 1]

    v_bin = quantize(v, v_tol)
    groups = defaultdict(list)
    for i, b in enumerate(v_bin):
        groups[b].append(i)

    pairs = []
    for b, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            du = abs(u[i] - u[j])
            if u_min_sep <= du <= u_max_sep:
                pairs.append([i, j])

    return pairs


def find_2x2_rectangles(nv_aod, u_tol=0.5, v_tol=0.5):
    """
    Find 2x2-compatible groups:
        (u1, v1), (u1, v2), (u2, v1), (u2, v2)

    Method:
    - bin NVs in AOD space
    - look for filled corners
    """
    u = nv_aod[:, 0]
    v = nv_aod[:, 1]

    u_bin = quantize(u, u_tol)
    v_bin = quantize(v, v_tol)

    # Map bin -> NV indices
    bin_to_indices = defaultdict(list)
    for i, (ub, vb) in enumerate(zip(u_bin, v_bin)):
        bin_to_indices[(ub, vb)].append(i)

    # Get occupied bins
    occupied = sorted(bin_to_indices.keys())

    # Group bins by u
    u_to_vs = defaultdict(set)
    for ub, vb in occupied:
        u_to_vs[ub].add(vb)

    rectangles = []
    used_bin_rects = set()

    u_keys = sorted(u_to_vs.keys())

    for u1, u2 in combinations(u_keys, 2):
        common_vs = sorted(u_to_vs[u1].intersection(u_to_vs[u2]))
        if len(common_vs) < 2:
            continue

        for v1, v2 in combinations(common_vs, 2):
            rect_bins = tuple(sorted([(u1, v1), (u1, v2), (u2, v1), (u2, v2)]))
            if rect_bins in used_bin_rects:
                continue

            # choose first NV from each occupied bin
            idxs = [
                bin_to_indices[(u1, v1)][0],
                bin_to_indices[(u1, v2)][0],
                bin_to_indices[(u2, v1)][0],
                bin_to_indices[(u2, v2)][0],
            ]
            rectangles.append(idxs)
            used_bin_rects.add(rect_bins)

    return rectangles


def remove_overlapping_groups(groups):
    """
    Greedy selection of non-overlapping groups.
    """
    selected = []
    used = set()

    # Prefer larger groups first if needed
    groups = sorted(groups, key=lambda g: (-len(g), tuple(g)))

    for g in groups:
        s = set(g)
        if used.intersection(s):
            continue
        selected.append(g)
        used.update(s)

    return selected


# ============================================================
# 8. RUN GROUP SEARCH
# ============================================================

# Tolerances: you may tune these.
# In AOD units, start modestly.
u_tol = 0.75
v_tol = 0.75

pairs_1x2 = find_1x2_pairs(nv_aod, u_tol=u_tol, v_min_sep=0.8, v_max_sep=25.0)
pairs_2x1 = find_2x1_pairs(nv_aod, v_tol=v_tol, u_min_sep=0.8, u_max_sep=25.0)
rectangles_2x2 = find_2x2_rectangles(nv_aod, u_tol=u_tol, v_tol=v_tol)

pairs_1x2_nonoverlap = remove_overlapping_groups(pairs_1x2)
pairs_2x1_nonoverlap = remove_overlapping_groups(pairs_2x1)
rectangles_2x2_nonoverlap = remove_overlapping_groups(rectangles_2x2)

print("\n=== GROUPING SUMMARY ===")
print(f"1x2 candidate pairs found: {len(pairs_1x2)}")
print(f"1x2 non-overlapping pairs: {len(pairs_1x2_nonoverlap)} "
      f"({2 * len(pairs_1x2_nonoverlap)} NVs)")

print(f"2x1 candidate pairs found: {len(pairs_2x1)}")
print(f"2x1 non-overlapping pairs: {len(pairs_2x1_nonoverlap)} "
      f"({2 * len(pairs_2x1_nonoverlap)} NVs)")

print(f"2x2 candidate rectangles found: {len(rectangles_2x2)}")
print(f"2x2 non-overlapping rectangles: {len(rectangles_2x2_nonoverlap)} "
      f"({4 * len(rectangles_2x2_nonoverlap)} NVs)")


# ============================================================
# 9. VISUALIZATIONS
# ============================================================

plot_camera_axes(
    calibration_coords_pixel,
    A_green,
    A_red,
    title="Green / red AOD axes expressed in camera pixel space"
)

plot_nv_camera_and_aod(nv_pixel, nv_aod, weights=nv_weights)

plot_example_groups(
    nv_pixel,
    pairs_1x2_nonoverlap,
    title=f"Example 1x2-compatible NV pairs ({ANALYSIS_SPACE} AOD space)",
    max_groups=12,
)

plot_example_groups(
    nv_pixel,
    pairs_2x1_nonoverlap,
    title=f"Example 2x1-compatible NV pairs ({ANALYSIS_SPACE} AOD space)",
    max_groups=12,
)

plot_example_groups(
    nv_pixel,
    rectangles_2x2_nonoverlap,
    title=f"Example 2x2-compatible NV groups ({ANALYSIS_SPACE} AOD space)",
    max_groups=12,
)




# ============================================================
# 10. OPTIONAL: PRINT SOME EXAMPLE GROUP DETAILS
# ============================================================

def group_details(nv_pixel, nv_aod, groups, nshow=5, name="groups"):
    print(f"\n=== Example {name} ===")
    for k, g in enumerate(groups[:nshow]):
        print(f"\n{name} #{k+1}: indices = {g}")
        for idx in g:
            print(
                f"  NV {idx:4d} | pixel = {nv_pixel[idx]} | aod = {nv_aod[idx]}"
            )

group_details(nv_pixel, nv_aod, pairs_1x2_nonoverlap, nshow=5, name="1x2 pairs")
group_details(nv_pixel, nv_aod, pairs_2x1_nonoverlap, nshow=5, name="2x1 pairs")
group_details(nv_pixel, nv_aod, rectangles_2x2_nonoverlap, nshow=5, name="2x2 rectangles")



import numpy as np
import matplotlib.pyplot as plt


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

# npz_path = "/mnt/data/48e7d17e-3cc2-4d77-98f9-3d0fc3f410ee.npz"


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
# PLOT 1: ALL AXES FROM SAME POINT, VERY LARGE
# ============================================================

import matplotlib.lines as mlines

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
plt.show(block=True)