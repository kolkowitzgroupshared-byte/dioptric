
from pathlib import Path
import numpy as np
import pandas as pd
# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector.")
    return v / n


# ----------------------------------------------------------------------
# Read hyperfine table
# ----------------------------------------------------------------------
def read_hyperfine_table_safe(path: str | Path) -> pd.DataFrame:
    """
    Read NV-2 hyperfine table.

    Expected columns:
        index distance x y z Axx Ayy Azz Axy Axz Ayz

    The A tensor entries are assumed to be in MHz.
    Positions are assumed to be in Angstrom.
    """

    path = Path(path)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    def _is_int_start(s: str) -> bool:
        s = s.lstrip()
        if not s:
            return False
        token = s.split()[0]
        try:
            int(token)
            return True
        except Exception:
            return False

    try:
        data_start = next(i for i, line in enumerate(lines) if _is_int_start(line))
    except StopIteration:
        raise RuntimeError(f"Could not locate data start in hyperfine file: {path}")

    hf_cols = [
        "index",
        "distance",
        "x",
        "y",
        "z",
        "Axx",
        "Ayy",
        "Azz",
        "Axy",
        "Axz",
        "Ayz",
    ]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="#",
        header=None,
        names=hf_cols,
        usecols=list(range(11)),
        skiprows=data_start,
    )

    df = df.astype(
        {
            "index": int,
            "distance": float,
            "x": float,
            "y": float,
            "z": float,
            "Axx": float,
            "Ayy": float,
            "Azz": float,
            "Axy": float,
            "Axz": float,
            "Ayz": float,
        }
    )

    return df


def get_A_file_from_row(row):
    """
    Build the hyperfine tensor and position vector from one table row.

    The table tensor is treated as A_file, i.e. in the dataset local frame.

    For the NV-2 30.12.2023 dataset:
        z_file || <111>

    Returns
    -------
    A_file_Hz : 3x3 ndarray
        Hyperfine tensor in Hz.

    pos_file_A : 3-vector ndarray
        13C position in Angstrom, in the dataset local frame.
    """

    A_file_Hz = np.array(
        [
            [row.Axx, row.Axy, row.Axz],
            [row.Axy, row.Ayy, row.Ayz],
            [row.Axz, row.Ayz, row.Azz],
        ],
        dtype=float,
    ) * 1e6  # MHz -> Hz

    pos_file_A = np.array([row.x, row.y, row.z], dtype=float)

    return A_file_Hz, pos_file_A


# ----------------------------------------------------------------------
# Frame rotations
# ----------------------------------------------------------------------
def U_111_to_cubic():
    """
    Dataset local [111] frame -> cubic/lab frame.

    Convention:
        v_cubic = U @ v_file
        A_cubic = U @ A_file @ U.T

    For the NV-2 30.12.2023 dataset:
        z_file || <111>
    """

    ex = _unit([1.0, -1.0, 0.0])
    ez = _unit([1.0, 1.0, 1.0])
    ey = _unit(np.cross(ez, ex))

    return np.column_stack([ex, ey, ez])


def make_R_NV(nv_axis_crystal, x_hint_crystal=(1.0, -1.0, 0.0)):
    """
    Cubic/lab frame -> target NV frame.

    Returns R such that:
        v_NV = R @ v_cubic
        A_NV = R @ A_cubic @ R.T

    nv_axis_crystal:
        Target NV orientation in cubic/lab frame, e.g. (1,1,1).

    x_hint_crystal:
        Direction used to define the local x axis after projection
        perpendicular to the NV axis. For the [111] dataset, (1,-1,0)
        is a natural choice.
    """

    ez = _unit(nv_axis_crystal)

    x_hint = _unit(x_hint_crystal)
    ex = x_hint - np.dot(x_hint, ez) * ez

    # If x_hint is nearly parallel to the NV axis, choose another direction.
    if np.linalg.norm(ex) < 1e-9:
        x_hint = _unit([0.0, 1.0, -1.0])
        ex = x_hint - np.dot(x_hint, ez) * ez

    ex = _unit(ex)

    # Right-handed frame.
    ey = _unit(np.cross(ez, ex))

    # Rows are local basis vectors written in cubic coordinates.
    R = np.vstack([ex, ey, ez])

    return R


def compute_hyperfine_components(A_tensor_Hz, dir_hat):
    """
    Compute A_parallel and A_perp relative to dir_hat.

    dir_hat should be the magnetic-field direction in the same frame
    as A_tensor_Hz.
    """

    ez = _unit(dir_hat)

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, ez)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = tmp - np.dot(tmp, ez) * ez
    e1 = _unit(e1)

    e2 = _unit(np.cross(ez, e1))

    A_par = float(ez @ A_tensor_Hz @ ez)

    A_perp = float(
        np.sqrt(
            (e1 @ A_tensor_Hz @ ez) ** 2
            + (e2 @ A_tensor_Hz @ ez) ** 2
        )
    )

    return A_par, A_perp


def rotate_site_for_experiment(
    A_file_Hz,
    pos_file_A,
    B_lab_G,
    nv_axis_crystal=(1, 1, 1),
):
    """
    Full pipeline for one 13C site.

    A_file_Hz:
        Hyperfine tensor from the NV-2 30.12.2023 dataset.
        Dataset frame has z || <111>.

    pos_file_A:
        13C position from the dataset local frame, in Angstrom.

    B_lab_G:
        Experimentally extracted lab/sample-frame B vector in Gauss.

    nv_axis_crystal:
        Target NV orientation in cubic/lab frame.
    """

    B_lab_T = np.asarray(B_lab_G, dtype=float) * 1e-4

    # Dataset local [111] frame -> cubic/lab.
    U_file_to_cubic = U_111_to_cubic()

    A_cubic_Hz = U_file_to_cubic @ A_file_Hz @ U_file_to_cubic.T
    pos_cubic_A = U_file_to_cubic @ np.asarray(pos_file_A, dtype=float)

    # Cubic/lab -> target NV frame.
    R_NV = make_R_NV(nv_axis_crystal)

    B_NV_T = R_NV @ B_lab_T
    B_hat_NV = B_NV_T / np.linalg.norm(B_NV_T)

    A_NV_Hz = R_NV @ A_cubic_Hz @ R_NV.T
    pos_NV_A = R_NV @ pos_cubic_A

    A_par_Hz, A_perp_Hz = compute_hyperfine_components(
        A_NV_Hz,
        B_hat_NV,
    )

    return {
        "R_NV": R_NV,
        "B_NV_G": B_NV_T / 1e-4,
        "B_parallel_G": float((B_NV_T / 1e-4)[2]),
        "B_transverse_G": float(np.linalg.norm((B_NV_T / 1e-4)[:2])),
        "A_cubic_Hz": A_cubic_Hz,
        "A_NV_Hz": A_NV_Hz,
        "pos_cubic_A": pos_cubic_A,
        "pos_NV_A": pos_NV_A,
        "A_parallel_Hz": float(A_par_Hz),
        "A_perp_Hz": float(A_perp_Hz),
    }

if __name__ == "__main__":
    hyperfine_path = "analysis/nv_hyperfine_coupling/nv-2.txt"

    # Example measured B-field vector from ODMR, in lab/sample frame, Gauss.
    B_lab_G = np.array([-48.67047318, -32.07615947, 22.49657427])

    # Example target NV orientation.
    nv_axis_crystal = (1, 1, 1)

    df = read_hyperfine_table_safe(hyperfine_path)

    results = []

    for _, row in df.iterrows():
        A_file_Hz, pos_file_A = get_A_file_from_row(row)

        out = rotate_site_for_experiment(
            A_file_Hz=A_file_Hz,
            pos_file_A=pos_file_A,
            B_lab_G=B_lab_G,
            nv_axis_crystal=nv_axis_crystal,
        )

        results.append(
            {
                "site_index": int(row["index"]),
                "distance_A": float(row["distance"]),
                "B_parallel_G": out["B_parallel_G"],
                "B_transverse_G": out["B_transverse_G"],
                "A_parallel_Hz": out["A_parallel_Hz"],
                "A_perp_Hz": out["A_perp_Hz"],
            }
        )

    print(results[:5])