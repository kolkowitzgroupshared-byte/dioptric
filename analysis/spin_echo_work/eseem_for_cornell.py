from pathlib import Path
import numpy as np
import pandas as pd
import json

# ----------------------------------------------------------------------
# B Filed
# ----------------------------------------------------------------------
B_vec_G = np.array([-48.67047318, -32.07615947, 22.49657427], dtype=float)  ##62.48G
B_vec_T = B_vec_G * 1e-4

# ----------------------------------------------------------------------
# Load A tensor
# ----------------------------------------------------------------------
def read_hyperfine_table_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # find first data row that starts with an integer (skip headers/junk)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    def _is_int_start(s: str) -> bool:
        s = s.lstrip()
        if not s:
            return False
        t = s.split()[0]
        try:
            int(t)
            return True
        except Exception:
            return False

    try:
        data_start = next(i for i, line in enumerate(lines) if _is_int_start(line))
    except StopIteration:
        raise RuntimeError(f"Could not locate data start in hyperfine file: {path}")

    HF_COLS = [
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
    # primary path: pandas
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",  # robust whitespace split
            engine="python",
            comment="#",  # ignore commented tails
            header=None,
            names=HF_COLS,
            usecols=list(range(11)),  # ensure exactly 11 cols
            skiprows=data_start,
            na_filter=False,
        )
        # enforce dtypes
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
            },
            errors="ignore",
        )
        return df
    except Exception as e:
        # fallback: numpy → DataFrame
        arr = np.loadtxt(
            path,
            comments="#",
            dtype=float,
            ndmin=2,
        )
        if arr.shape[1] < 11:
            raise RuntimeError(
                f"Expected ≥11 columns, found {arr.shape[1]} in {path}"
            ) from e
        arr = arr[:, :11]
        df = pd.DataFrame(arr, columns=HF_COLS)
        # index is float now; coerce to int safely
        df["index"] = df["index"].round().astype(int)
        return df

# ----------------------------------------------------------------------
# Spin-1/2 operators (Pauli / 2)
# ----------------------------------------------------------------------
Sx = 0.5 * np.array([[0, 1],
                     [1, 0]], float)
Sy = 0.5 * np.array([[0,-1j],
                     [1j, 0]], complex)
Sz = 0.5 * np.array([[1, 0],
                     [0,-1]], float)

# ----------------------------------------------------------------------
# Geometry + Hamiltonian helpers
# ----------------------------------------------------------------------
def _build_U_from_orientation(orientation, phi_deg: float = 0.0):
    """
    Build a rotation matrix U that sends cubic axes to the NV frame for
    a given orientation (±1, ±1, ±1) and an optional in-plane twist phi_deg.
    """
    ez = np.asarray(orientation, float)
    ez /= np.linalg.norm(ez)

    # pick a trial x-axis not collinear with ez
    trial = np.array([1.0, -1.0, 0.0])
    if abs(np.dot(trial / np.linalg.norm(trial), ez)) > 0.95:
        trial = np.array([0.0, 1.0, -1.0])

    ex = trial - np.dot(trial, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)

    U0 = np.column_stack([ex, ey, ez])

    phi = np.deg2rad(phi_deg)
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0],
                   [np.sin(phi),  np.cos(phi), 0.0],
                   [0.0,          0.0,        1.0]])
    return U0 @ Rz, ez


def essem_lines_by_diag(
    A_file_Hz: np.ndarray,
    orientation=(1, 1, 1),
    B_lab_vec=None,
    gamma_n_Hz_per_T: float = 10.705e6,
    ms: int = -1,
    phi_deg: float = 0.0,
):
    """
    Diagonalize the nuclear Hamiltonian with and without hyperfine to get
    ESEEM frequencies f_- and f_+ for a single 13C site.

    Parameters
    ----------
    A_file_Hz : (3,3) array
        Hyperfine tensor (Hz) in the NV-(111) frame.
    orientation : tuple
        NV orientation in cubic coordinates (e.g. (1,1,1)).
    B_lab_vec : array-like, shape (3,)
        Lab-frame B-field vector. Units must match gamma_n_Hz_per_T.
    gamma_n_Hz_per_T : float
        Nuclear gyromagnetic ratio in Hz/T (13C: 10.705e6 Hz/T).
    ms : int
        Electron spin manifold (+/-1). For NV- ESEEM, typically -1.
    phi_deg : float
        Additional in-plane twist of hyperfine tensor about NV z.

    Returns
    -------
    f_minus, f_plus, fI_split, omega_ms_split, A_cubic, z_nv_cubic
        All in Hz (except A_cubic tensor and unit vector z_nv_cubic).
    """
    if B_lab_vec is None:
        raise ValueError("B_lab_vec must be provided.")

    # rotate A into cubic frame for this NV
    U, z_nv_cubic = _build_U_from_orientation(orientation, phi_deg=phi_deg)
    A_cubic = U @ A_file_Hz @ U.T

    B_lab = np.asarray(B_lab_vec, float)
    Bmag = float(np.linalg.norm(B_lab))
    if Bmag == 0.0:
        raise ValueError("B field magnitude is zero.")
    bx, by, bz = B_lab / Bmag

    # nuclear Zeeman
    fI_Hz = gamma_n_Hz_per_T * Bmag
    HZ = fI_Hz * (bx * Sx + by * Sy + bz * Sz)

    # hyperfine term projected along NV axis (ms-dependent)
    Aeff_vec = A_cubic @ z_nv_cubic
    Hhf = float(ms) * (Aeff_vec[0] * Sx + Aeff_vec[1] * Sy + Aeff_vec[2] * Sz)

    evals0 = np.linalg.eigvalsh(HZ)
    evalsms = np.linalg.eigvalsh(HZ + Hhf)

    fI_split = float(abs(evals0[1] - evals0[0]))
    omega_ms_split = float(abs(evalsms[1] - evalsms[0]))

    f_minus = abs(omega_ms_split - fI_split)
    f_plus = omega_ms_split + fI_split
    return f_minus, f_plus, fI_split, omega_ms_split, A_cubic, z_nv_cubic


# ----------------------------------------------------------------------
# 1) Build and save full ESEEM catalog
# ----------------------------------------------------------------------
def build_essem_catalog(
    hyperfine_path: str,
    B_lab_vec,
    orientations=((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)),
    distance_max_A: float = 22.0,
    gamma_n_Hz_per_T: float = 10.705e6,
    ms: int = -1,
    phi_deg: float = 0.0,
    out_json: str = "essem_freq_catalog.json",
    out_csv: str = "essem_freq_catalog.csv",
):
    """
    Reads the hyperfine table, computes (f-, f+) for all sites within distance_max_A
    and all NV orientations; saves JSON+CSV with handy fields for later fitting.

    Units:
      - Axx, Ayy, ... in your table are assumed MHz -> multiplied by 1e6 to Hz.
      - B_lab_vec units must match gamma_n_Hz_per_T (default assumes Tesla).
    """
    df = read_hyperfine_table_safe(hyperfine_path).copy()
    df = df[df["distance"] <= float(distance_max_A)].reset_index(drop=True)

    B_lab = np.asarray(B_lab_vec, float)
    Bmag = float(np.linalg.norm(B_lab))
    B_hat = B_lab / Bmag

    records = []
    for ori in orientations:
        ori_tuple = tuple(int(x) for x in ori)

        for i, row in df.iterrows():
            # A_file in Hz (NV-(111) frame)
            A_file_Hz = np.array(
                [
                    [row.Axx, row.Axy, row.Axz],
                    [row.Axy, row.Ayy, row.Ayz],
                    [row.Axz, row.Ayz, row.Azz],
                ],
                float,
            ) * 1e6  # MHz -> Hz

            (
                fm,
                fp,
                fI,
                wms,
                A_cubic,
                z_nv,
            ) = essem_lines_by_diag(
                A_file_Hz=A_file_Hz,
                orientation=ori,
                B_lab_vec=B_lab,
                gamma_n_Hz_per_T=gamma_n_Hz_per_T,
                ms=ms,
                phi_deg=phi_deg,
            )

            # amplitude proxy: ~ sin^2(theta)*(A_perp/omega)^2
            A_par = float(B_hat @ A_cubic @ B_hat)
            A_perp_vec = A_cubic @ B_hat - A_par * B_hat
            A_perp = float(np.linalg.norm(A_perp_vec))
            cos_th = float(np.clip(B_hat @ (z_nv / np.linalg.norm(z_nv)), -1, 1))
            sin2_th = 1.0 - cos_th**2
            amp_wt = (A_perp / max(wms, 1e-30)) ** 2 * sin2_th

            records.append(
                {
                    "orientation": ori_tuple,
                    "site_index": int(i),
                    "distance_A": float(row["distance"]),
                    "f_minus_Hz": float(fm),
                    "f_plus_Hz": float(fp),
                    "fI_Hz": float(fI),
                    "omega_ms_Hz": float(wms),
                    "A_par_Hz": float(A_par),
                    "A_perp_Hz": float(A_perp),
                    "amp_weight": float(amp_wt),
                }
            )

    # Save JSON
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)

    return records
