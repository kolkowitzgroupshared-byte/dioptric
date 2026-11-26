# -*- coding: utf-8 -*-
"""
Check whether ThorCam_SLM -> Nuvu mapping is affine or field-dependent.

Input:
    Four SLM triangle calibration sets.

Output:
    1. Global affine ThorCam_SLM -> Nuvu.
    2. Residual vectors for each point.
    3. Per-set affine changes.
    4. Residual plots.

Interpretation:
    Small residuals for one global affine map => mostly affine mapping.
    Systematic residuals growing with position => possible aberration/distortion.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# INPUT DATA
# =============================================================================

sets = []

# set 1
sets.append(
    {
        "name": "set1",
        "thorcam": np.array(
            [
                [848.56406461, 620.0],
                [571.43593539, 620.0],
                [710.0,        380.0],
            ],
            dtype=np.float32,
        ),
        "nuvu": np.array(
            [
                [84.57, 7.014],
                [92.258, 353.467],
                [337.832, 184.63],
            ],
            dtype=np.float32,
        ),
    }
)

# set 2
sets.append(
    {
        "name": "set2",
        "thorcam": np.array(
            [
                [831.24355653, 610.0],
                [588.75644347, 610.0],
                [710.0,        400.0],
            ],
            dtype=np.float32,
        ),
        "nuvu": np.array(
            [
                [95.767, 28.179],
                [102.592, 331.285],
                [316.852, 183.808],
            ],
            dtype=np.float32,
        ),
    }
)

# set 3
sets.append(
    {
        "name": "set3",
        "thorcam": np.array(
            [
                [813.92304845, 600.0],
                [606.07695155, 600.0],
                [710.0,        420.0],
            ],
            dtype=np.float32,
        ),
        "nuvu": np.array(
            [
                [105.971, 49.68],
                [112.133, 309.714],
                [296.146, 183.337],
            ],
            dtype=np.float32,
        ),
    }
)

# set 4
sets.append(
    {
        "name": "set4",
        "thorcam": np.array(
            [
                [796.60254038, 590.0],
                [623.39745962, 590.0],
                [710.0,        440.0],
            ],
            dtype=np.float32,
        ),
        "nuvu": np.array(
            [
                [116.737, 71.334],
                [122.193, 288.122],
                [275.363, 182.454],
            ],
            dtype=np.float32,
        ),
    }
)

# set 5
sets.append(
    {
        "name": "set5",
        "thorcam": np.array(
            [
                [761.96152423, 570.0],
                [658.03847577, 570.0],
                [710.0,        480.0],
            ],
            dtype=np.float32,
        ),
        "nuvu": np.array(
            [
                [138.143, 114.477],
                [141.56, 245.128],
                [233.972, 181.738],
            ],
            dtype=np.float32,
        ),
    }
)
# =============================================================================
# HELPERS
# =============================================================================

def fit_affine(src, dst):
    """
    Fit affine map:
        dst = A @ src + t

    Returns:
        M : 2x3 matrix
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    X = np.column_stack([src, np.ones(len(src))])
    B, *_ = np.linalg.lstsq(X, dst, rcond=None)

    return B.T


def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=float)
    X = np.column_stack([pts, np.ones(len(pts))])
    return X @ M.T


def affine_report(M):
    """
    Report local scale/rotation/shear using SVD.
    """
    A = M[:, :2]

    U, svals, Vt = np.linalg.svd(A)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    rot_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    return {
        "A": A,
        "t": M[:, 2],
        "singular_values": svals,
        "rotation_deg": rot_deg,
        "det": np.linalg.det(A),
    }


def triangle_edge_report(thorcam, nuvu):
    pairs = [(0, 1), (0, 2), (1, 2)]

    out = []
    for i, j in pairs:
        d_thor = np.linalg.norm(thorcam[i] - thorcam[j])
        d_nuvu = np.linalg.norm(nuvu[i] - nuvu[j])
        out.append((i, j, d_thor, d_nuvu, d_nuvu / d_thor))

    return out


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    thor_all = np.vstack([s["thorcam"] for s in sets])
    nuvu_all = np.vstack([s["nuvu"] for s in sets])

    labels = []
    local_inds = []

    for s in sets:
        for k in range(3):
            labels.append(s["name"])
            local_inds.append(k)

    labels = np.asarray(labels)
    local_inds = np.asarray(local_inds)

    # -------------------------------------------------------------------------
    # 1. Fit one global affine.
    # -------------------------------------------------------------------------
    M_global = fit_affine(thor_all, nuvu_all)

    pred_all = apply_affine(M_global, thor_all)
    residuals = nuvu_all - pred_all
    err = np.linalg.norm(residuals, axis=1)

    print("\n=== Global affine ThorCam_SLM -> Nuvu ===")
    print(M_global)

    print("\n=== Global affine residuals ===")
    print(f"mean residual [Nuvu px]: {np.mean(err):.4f}")
    print(f"RMS residual  [Nuvu px]: {np.sqrt(np.mean(err**2)):.4f}")
    print(f"max residual  [Nuvu px]: {np.max(err):.4f}")

    print("\nPer-point residuals:")
    for i in range(len(thor_all)):
        print(
            f"{labels[i]} point {local_inds[i]}: "
            f"thor={thor_all[i]}, "
            f"nuvu={nuvu_all[i]}, "
            f"pred={pred_all[i]}, "
            f"res={residuals[i]}, "
            f"|res|={err[i]:.4f}"
        )

    rep = affine_report(M_global)
    print("\nGlobal affine decomposition:")
    print("rotation-like angle [deg]:", rep["rotation_deg"])
    print("singular values:", rep["singular_values"])
    print("determinant:", rep["det"])

    # -------------------------------------------------------------------------
    # 2. Per-set affine comparison.
    # -------------------------------------------------------------------------
    print("\n=== Per-set affine comparison ===")

    for s in sets:
        M_set = fit_affine(s["thorcam"], s["nuvu"])
        rep_set = affine_report(M_set)

        print(f"\n{s['name']}")
        print("M:")
        print(M_set)
        print("rotation-like angle [deg]:", rep_set["rotation_deg"])
        print("singular values:", rep_set["singular_values"])
        print("determinant:", rep_set["det"])

        print("triangle edge ratios Nuvu/ThorCam:")
        for i, j, d_thor, d_nuvu, ratio in triangle_edge_report(
            s["thorcam"],
            s["nuvu"],
        ):
            print(
                f"  edge {i}-{j}: "
                f"ThorCam={d_thor:.3f}, "
                f"Nuvu={d_nuvu:.3f}, "
                f"ratio={ratio:.6f}"
            )

    # -------------------------------------------------------------------------
    # 3. Plot residual vectors in ThorCam coordinates.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    for s in sets:
        thor = s["thorcam"]
        nuvu = s["nuvu"]
        pred = apply_affine(M_global, thor)
        res = nuvu - pred

        ax.scatter(thor[:, 0], thor[:, 1], s=60, label=s["name"])

        # residuals are in Nuvu pixels, but plotted at ThorCam positions.
        # scale visually for clarity.
        scale = 20.0
        ax.quiver(
            thor[:, 0],
            thor[:, 1],
            res[:, 0] * scale,
            res[:, 1] * scale,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
        )

        for k, p in enumerate(thor):
            ax.text(p[0] + 4, p[1] + 4, f"{s['name']}-{k}", fontsize=8)

    ax.set_title("Global-affine residual vectors plotted in ThorCam_SLM coordinates")
    ax.set_xlabel("ThorCam_SLM x [px]")
    ax.set_ylabel("ThorCam_SLM y [px]")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # 4. Plot residual vectors in Nuvu coordinates.
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    for s in sets:
        thor = s["thorcam"]
        nuvu = s["nuvu"]
        pred = apply_affine(M_global, thor)
        res = nuvu - pred

        ax.scatter(nuvu[:, 0], nuvu[:, 1], s=60, label=s["name"])

        scale = 20.0
        ax.quiver(
            nuvu[:, 0],
            nuvu[:, 1],
            res[:, 0] * scale,
            res[:, 1] * scale,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.004,
        )

        for k, p in enumerate(nuvu):
            ax.text(p[0] + 3, p[1] + 3, f"{s['name']}-{k}", fontsize=8)

    ax.set_title("Global-affine residual vectors plotted in Nuvu coordinates")
    ax.set_xlabel("Nuvu x [px]")
    ax.set_ylabel("Nuvu y [px]")
    ax.axis("equal")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()

    plt.show(block=True)


if __name__ == "__main__":
    main()