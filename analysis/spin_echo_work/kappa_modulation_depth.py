# =============================================================================
# Orientation-aware ESEEM catalog + filtering + plotting
# =============================================================================
from __future__ import annotations
import sys
import json, csv, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional
from analysis.sc_c13_hyperfine_sim_data_driven import read_hyperfine_table_safe, B_vec_T
from typing import Tuple, Optional, Sequence
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
from utils import kplotlib as kpl
from itertools import combinations, product

# ---------- You already have these imports/helpers in your env ----------
# from analysis.sc_c13_hyperfine_sim_data_driven import read_hyperfine_table_safe, B_vec_T
# If not available, provide your own read_hyperfine_table_safe

# ========= Exact-κ orientation-aware ESEEM catalog (Hz) ======================

# Pauli/2:
Sx = 0.5 * np.array([[0, 1], [1, 0]], float)
Sy = 0.5 * np.array([[0, -1j], [1j, 0]], complex)
Sz = 0.5 * np.array([[1, 0], [0, -1]], float)

GAMMA_C13_HZ_PER_T = 10.708e6  # 13C γ (non-angular), Hz/T


def _build_U_from_orientation(orientation, phi_deg=0.0):
    ez = np.asarray(orientation, float)
    ez /= np.linalg.norm(ez)
    trial = np.array([1.0, -1.0, 0.0])
    if abs(np.dot(trial / np.linalg.norm(trial), ez)) > 0.95:
        trial = np.array([0.0, 1.0, -1.0])
    ex = trial - np.dot(trial, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    U0 = np.column_stack([ex, ey, ez])
    phi = np.deg2rad(phi_deg)
    Rz = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return U0 @ Rz, ez


def _safe_norm(v, eps=1e-30):
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


def _kappa_and_fpm(
    A_file_Hz,
    orientation,
    B_lab_vec,
    gamma_hz_per_t=GAMMA_C13_HZ_PER_T,
    ms=-1,
    phi_deg=0.0,
):
    U, z_nv_cubic = _build_U_from_orientation(orientation, phi_deg=phi_deg)
    A_cubic = U @ A_file_Hz @ U.T

    B = np.asarray(B_lab_vec, float)
    Bmag = _safe_norm(B)
    Bhat = B / Bmag
    omegaI = float(gamma_hz_per_t * Bmag)  # |Ω0| in Hz
    a_vec = A_cubic @ z_nv_cubic  # hyperfine vector (Hz)

    Omega0 = omegaI * Bhat
    Omega_m = Omega0 + float(ms) * a_vec  # <-- honor ms (0, ±1)

    n0 = _safe_norm(Omega0)
    nm = _safe_norm(Omega_m)
    n0_hat = Omega0 / n0
    nm_hat = Omega_m / nm

    cross = np.cross(Omega0, Omega_m)
    kappa = float((cross @ cross) / (n0 * n0 * nm * nm))
    kappa = max(0.0, min(1.0, kappa))

    f_minus = abs(nm - n0)  # Hz
    f_plus = nm + n0  # Hz

    # diagnostics w.r.t. B̂
    A_par = float(a_vec @ Bhat)
    A_perp = float(_safe_norm(a_vec - A_par * Bhat))

    # optional: report tilt angle between manifolds
    cos_theta = float(np.clip((Omega0 @ Omega_m) / (n0 * nm), -1.0, 1.0))
    theta_deg = np.degrees(np.arccos(cos_theta))

    return (
        kappa,
        f_minus,
        f_plus,
        omegaI,
        nm,
        A_par,
        A_perp,
        theta_deg,
        n0_hat,
        nm_hat,
        Bhat,
        z_nv_cubic,
    )


# ---------- Build catalog with exact κ and per-line weights ----------
def build_essem_catalog_with_kappa(
    hyperfine_path,
    B_lab_vec,
    orientations=((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)),
    distance_max_A=22.0,
    gamma_n_Hz_per_T=GAMMA_C13_HZ_PER_T,
    p_occ=0.011,  # 13C abundance
    ms=-1,
    phi_deg=0.0,
    out_json="essem_freq_catalog_22A.json",
    out_csv="essem_freq_catalog_22A.csv",
    read_hf_table_fn=None,
):
    """
    Adds fields:
      • kappa:          exact modulation depth (0..1)
      • line_w_minus/+  = p_occ * (kappa/4)  (first-order ESEEM weight per line)
    """
    if read_hf_table_fn is None:
        from analysis.sc_c13_hyperfine_sim_data_driven import (
            read_hyperfine_table_safe as read_hf_table_fn,
        )

    df = read_hf_table_fn(hyperfine_path).copy()
    df = df[df["distance"] <= float(distance_max_A)].reset_index(drop=True)

    B = np.asarray(B_lab_vec, float)

    recs = []
    for ori in orientations:
        for i, row in df.iterrows():
            A_file_Hz = (
                np.array(
                    [
                        [row.Axx, row.Axy, row.Axz],
                        [row.Axy, row.Ayy, row.Ayz],
                        [row.Axz, row.Ayz, row.Azz],
                    ],
                    float,
                )
                * 1e6
            )
            (
                kappa,
                f_minus,
                f_plus,
                omegaI,
                nm,
                A_par,
                A_perp,
                theta_deg,
                n0_hat,
                nm_hat,
                Bhat,
                z_nv_cubic,
            ) = _kappa_and_fpm(
                A_file_Hz,
                ori,
                B,
                gamma_hz_per_t=gamma_n_Hz_per_T,
                ms=ms,
                phi_deg=phi_deg,
            )

            w_line = float(p_occ) * (kappa * 0.25)  # per-line first-order weight

            # handy scalar projections
            cos_B_n0 = float(Bhat @ n0_hat)
            cos_B_nm = float(Bhat @ nm_hat)
            cos_NV_n0 = float(z_nv_cubic @ n0_hat)
            cos_NV_nm = float(z_nv_cubic @ nm_hat)

            recs.append(
                {
                    "orientation": tuple(int(x) for x in ori),
                    "site_index": int(i),
                    "distance_A": float(row["distance"]),
                    "kappa": float(kappa),
                    "f_minus_Hz": float(f_minus),
                    "f_plus_Hz": float(f_plus),
                    "fI_Hz": float(omegaI),
                    "omega_ms_Hz": float(nm),
                    "A_par_Hz": float(A_par),
                    "A_perp_Hz": float(A_perp),
                    "theta_deg": float(theta_deg),
                    "line_w_minus": w_line,
                    "line_w_plus": w_line,
                    # NEW: coordinates from hyperfine table (if present)
                    "x_A": float(row.get("x_A", row.get("x", np.nan))),
                    "y_A": float(row.get("y_A", row.get("y", np.nan))),
                    "z_A": float(row.get("z_A", row.get("z", np.nan))),
                    # NEW: quantization directions & geometry (all JSON-friendly lists/scalars)
                    "n0_hat": [float(v) for v in n0_hat],  # ms = 0 quantization axis
                    "nm_hat": [
                        float(v) for v in nm_hat
                    ],  # ms manifold quantization axis
                    "B_hat": [float(v) for v in Bhat],  # direction of B
                    "z_nv_cubic": [
                        float(v) for v in z_nv_cubic
                    ],  # NV axis in cubic frame
                    "cos_B_n0": cos_B_n0,
                    "cos_B_nm": cos_B_nm,
                    "cos_NV_n0": cos_NV_n0,
                    "cos_NV_nm": cos_NV_nm,
                }
            )

    # Save
    with open(out_json, "w") as f:
        json.dump(recs, f, indent=2)

    keys = list(recs[0].keys()) if recs else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    return recs


# ---------------
def compute_dispersion_grid_for_site(
    hyperfine_path: str,
    orientation: Tuple[int, int, int],
    site_index: int,
    B_hat0: np.ndarray,
    Bmin_G: float = 20.0,
    Bmax_G: float = 80.0,
    n_B: int = 81,
    theta_min_deg: float = -60.0,
    theta_max_deg: float = 60.0,
    n_theta: int = 121,
    gamma_n_Hz_per_T: float = GAMMA_C13_HZ_PER_T,
    ms: int = -1,
    phi_deg: float = 0.0,
):
    """
    Compute f_-(B,theta) and f_+(B,theta) for a single lattice site on a 2D grid:
      - B magnitude in [Bmin_G, Bmax_G] along a cone around B_hat0
      - tilt angle theta in [theta_min_deg, theta_max_deg] in a plane containing B_hat0

    Returns: dict with keys:
      B_mags_G:   (n_B,) array
      thetas_deg: (n_theta,) array
      f_minus_kHz: (n_theta, n_B) array
      f_plus_kHz:  (n_theta, n_B) array
      kappa:       (n_theta, n_B) array
      theta_ms0_ms1_deg: (n_theta, n_B) array
    """
    # Normalize reference direction
    B_hat0 = np.asarray(B_hat0, float)
    B_hat0 /= np.linalg.norm(B_hat0)

    # Choose an axis perpendicular to B_hat0 to define the tilt plane
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, B_hat0)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    u_perp = trial - np.dot(trial, B_hat0) * B_hat0
    u_perp /= np.linalg.norm(u_perp)

    # Grids
    B_mags_G = np.linspace(Bmin_G, Bmax_G, n_B)
    thetas_deg = np.linspace(theta_min_deg, theta_max_deg, n_theta)
    B_mags_T = B_mags_G * 1e-4

    # Hyperfine tensor for this site
    df_hf = read_hyperfine_table_safe(hyperfine_path).copy()
    row = df_hf.iloc[int(site_index)]
    A_file_Hz = (
        np.array(
            [
                [row.Axx, row.Axy, row.Axz],
                [row.Axy, row.Ayy, row.Ayz],
                [row.Axz, row.Ayz, row.Azz],
            ],
            float,
        )
        * 1e6
    )

    # Allocate grids
    f_minus_grid = np.zeros((n_theta, n_B), float)
    f_plus_grid = np.zeros((n_theta, n_B), float)
    kappa_grid = np.zeros((n_theta, n_B), float)
    theta_ms0_ms1_grid = np.zeros((n_theta, n_B), float)

    for i_th, th_deg in enumerate(thetas_deg):
        th = np.deg2rad(th_deg)
        # B_hat in the plane spanned by B_hat0 and u_perp
        B_hat = np.cos(th) * B_hat0 + np.sin(th) * u_perp
        B_hat /= np.linalg.norm(B_hat)
        for j_B, B_T in enumerate(B_mags_T[:, None] * B_hat):
            (
                kappa,
                f_minus,
                f_plus,
                omegaI,
                nm,
                A_par,
                A_perp,
                theta_deg,
                n0_hat,
                nm_hat,
                Bhat_vec,
                z_nv_cubic,
            ) = _kappa_and_fpm(
                A_file_Hz,
                orientation,
                B_T,
                gamma_hz_per_t=gamma_n_Hz_per_T,
                ms=ms,
                phi_deg=phi_deg,
            )

            f_minus_grid[i_th, j_B] = f_minus / 1e3  # Hz -> kHz
            f_plus_grid[i_th, j_B] = f_plus / 1e3
            kappa_grid[i_th, j_B] = kappa
            theta_ms0_ms1_grid[i_th, j_B] = theta_deg

    return dict(
        B_mags_G=B_mags_G,
        thetas_deg=thetas_deg,
        f_minus_kHz=f_minus_grid,
        f_plus_kHz=f_plus_grid,
        kappa=kappa_grid,
        theta_ms0_ms1_deg=theta_ms0_ms1_grid,
    )


# def plot_dispersion_2D_for_site(
#     grid: dict, title_prefix: str = "", cmap: str = "viridis"
# ):
#     """
#     Make a 1×2 panel:
#       left:  f_-(B,theta) vs (|B|, tilt angle)
#       right: f_+(B,theta) vs (|B|, tilt angle)
#     """
#     B_mags_G = grid["B_mags_G"]
#     thetas_deg = grid["thetas_deg"]
#     f_minus = grid["f_minus_kHz"]
#     f_plus = grid["f_plus_kHz"]

#     fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

#     extent = [B_mags_G.min(), B_mags_G.max(), thetas_deg.min(), thetas_deg.max()]

#     im0 = axs[0].imshow(
#         f_minus,
#         origin="lower",
#         aspect="auto",
#         extent=extent,
#     )
#     axs[0].set_title(r"$f_-(|B|,\theta)$")
#     axs[0].set_xlabel(r"|B| [G]")
#     axs[0].set_ylabel(r"tilt angle $\theta$ [deg]")
#     fig.colorbar(im0, ax=axs[0], label="kHz")

#     im1 = axs[1].imshow(
#         f_plus,
#         origin="lower",
#         aspect="auto",
#         extent=extent,
#     )
#     axs[1].set_title(r"$f_+(|B|,\theta)$")
#     axs[1].set_xlabel(r"|B| [G]")
#     fig.colorbar(im1, ax=axs[1], label="kHz")

#     if title_prefix:
#         fig.suptitle(title_prefix, y=1.02)

#     # fig.tight_layout()
#     return fig, axs


def plot_dispersion_2D_for_site(
    grid: dict,
    title_prefix: str = "",
    cmap: str = "viridis",
    contour: bool = True,
    n_levels: int = 10,
    contour_color: str = "w",
    contour_lw: float = 0.7,
    add_kappa: bool = False,
    kappa_as: str = "contour",  # "contour" or "alpha"
    B_meas_G: float | None = None,
    show_theta0: bool = True,
):
    """
    1×2 panel:
      left:  f_-(|B|,θ)   right: f_+(|B|,θ)
    Adds contours + optional κ overlay + optional measured |B| marker.
    """
    B_mags_G = np.asarray(grid["B_mags_G"], float)
    thetas_deg = np.asarray(grid["thetas_deg"], float)
    f_minus = np.asarray(grid["f_minus_kHz"], float)
    f_plus = np.asarray(grid["f_plus_kHz"], float)
    kappa = np.asarray(grid.get("kappa", None), float) if "kappa" in grid else None

    # Mesh for contouring
    BB, TT = np.meshgrid(B_mags_G, thetas_deg)  # shapes (n_theta, n_B)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    extent = [B_mags_G.min(), B_mags_G.max(), thetas_deg.min(), thetas_deg.max()]

    def _panel(ax, Z, title):
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent)

        if contour:
            zmin, zmax = np.nanmin(Z), np.nanmax(Z)
            levels = np.linspace(zmin, zmax, n_levels)
            cs = ax.contour(
                BB, TT, Z, levels=levels, colors=contour_color, linewidths=contour_lw
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")  # label in kHz (rounded)

        # Optional κ overlay
        if add_kappa and (kappa is not None):
            if kappa_as == "contour":
                # κ contours (e.g. 0.1, 0.2, ... 0.9)
                k_levels = np.linspace(0.1, 0.9, 9)
                ck = ax.contour(
                    BB,
                    TT,
                    kappa,
                    levels=k_levels,
                    colors="k",
                    linewidths=0.6,
                    alpha=0.7,
                )
                ax.clabel(ck, inline=True, fontsize=6, fmt="κ=%.1f")
            elif kappa_as == "alpha":
                # make high-κ regions more opaque
                kk = np.clip(kappa, 0, 1)
                ax.imshow(
                    kk,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap="gray",
                    alpha=0.25,
                )

        if B_meas_G is not None:
            ax.axvline(float(B_meas_G), linestyle="--", linewidth=1.0, alpha=0.8)

        if show_theta0:
            ax.axhline(0.0, linestyle=":", linewidth=1.0, alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel(r"|B| [G]")
        ax.grid(False)
        return im

    im0 = _panel(axs[0], f_minus, r"$f_-(|B|,\theta)$  [kHz]")
    axs[0].set_ylabel(r"tilt angle $\theta$ [deg]")
    fig.colorbar(im0, ax=axs[0], label="kHz")

    im1 = _panel(axs[1], f_plus, r"$f_+(|B|,\theta)$  [kHz]")
    fig.colorbar(im1, ax=axs[1], label="kHz")

    if title_prefix:
        fig.suptitle(title_prefix, y=1.03)

    # fig.tight_layout()
    return fig, axs


def _orthonormal_basis_from_B0(B0_hat: np.ndarray):
    """Return (u1, u2, B0_hat) with u1,u2 ⟂ B0_hat."""
    B0_hat = np.asarray(B0_hat, float)
    B0_hat /= np.linalg.norm(B0_hat)

    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, B0_hat)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])

    u1 = trial - np.dot(trial, B0_hat) * B0_hat
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(B0_hat, u1)
    u2 /= np.linalg.norm(u2)
    return u1, u2, B0_hat


def compute_full_angle_map_lab_spherical_for_site(
    hyperfine_path: str,
    orientation: tuple,
    site_index: int,
    Bmag_G: float,
    n_theta: int = 91,
    n_phi: int = 181,
    ms: int = -1,
    phi_deg_nv: float = 120.0,  # rotation used inside _build_U_from_orientation (NV gauge), NOT lab azimuth
    phi_range: str = "0_360",
):
    thetas_deg = np.linspace(0.0, 180.0, n_theta)
    phis_deg = (
        np.linspace(0.0, 360.0, n_phi, endpoint=False)
        if phi_range == "0_360"
        else np.linspace(-180.0, 180.0, n_phi, endpoint=False)
    )

    th = np.deg2rad(thetas_deg)[:, None]
    ph = np.deg2rad(phis_deg)[None, :]

    # Absolute LAB spherical directions (independent of any measured B):
    Bhat_grid = np.stack(
        [
            np.sin(th) * np.cos(ph),
            np.sin(th) * np.sin(ph),
            np.cos(th) * np.ones_like(ph),
        ],
        axis=2,
    )  # (n_theta,n_phi,3)

    Bmag_T = float(Bmag_G) * 1e-4

    df_hf = read_hyperfine_table_safe(hyperfine_path).copy()
    row = df_hf.iloc[int(site_index)]
    A_file_Hz = (
        np.array(
            [
                [row.Axx, row.Axy, row.Axz],
                [row.Axy, row.Ayy, row.Ayz],
                [row.Axz, row.Ayz, row.Azz],
            ],
            float,
        )
        * 1e6
    )

    f_minus = np.zeros((n_theta, n_phi), float)
    f_plus = np.zeros((n_theta, n_phi), float)
    kappa = np.zeros((n_theta, n_phi), float)

    for i in range(n_theta):
        for j in range(n_phi):
            B_vec_T = Bmag_T * Bhat_grid[i, j, :]
            kap, f_m, f_p, *_ = _kappa_and_fpm(
                A_file_Hz, orientation, B_vec_T, ms=ms, phi_deg=phi_deg_nv
            )
            f_minus[i, j] = f_m / 1e3
            f_plus[i, j] = f_p / 1e3
            kappa[i, j] = kap

    return dict(
        basis="lab_spherical",
        phi_range=phi_range,
        thetas_deg=thetas_deg,
        phis_deg=phis_deg,
        Bhat_grid=Bhat_grid,
        Bmag_G=float(Bmag_G),
        f_minus_kHz=f_minus,
        f_plus_kHz=f_plus,
        kappa=kappa,
    )


def relative_theta_phi(B0_hat, Bhat_list, *, u1=None, u2=None):
    """
    Compute (theta,phi) in the SAME coordinate system as the map:
      Bhat = cosθ B0_hat + sinθ (cosφ u1 + sinφ u2)

    If u1/u2 are provided, we use them (this guarantees consistency with the grid).
    """
    B0_hat = np.asarray(B0_hat, float)
    B0_hat /= np.linalg.norm(B0_hat)

    if u1 is None or u2 is None:
        # build (u1,u2) orthonormal basis around B0_hat
        trial = np.array([1.0, 0.0, 0.0])
        if abs(trial @ B0_hat) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        u1 = trial - (trial @ B0_hat) * B0_hat
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(B0_hat, u1)
        u2 /= np.linalg.norm(u2)
    else:
        u1 = np.asarray(u1, float)
        u1 /= np.linalg.norm(u1)
        u2 = np.asarray(u2, float)
        u2 /= np.linalg.norm(u2)

    Bhat = np.asarray(Bhat_list, float)
    if Bhat.ndim == 1:
        Bhat = Bhat[None, :]

    dots = np.clip(Bhat @ B0_hat, -1.0, 1.0)
    theta = np.arccos(dots)

    x = Bhat @ u1
    y = Bhat @ u2
    phi = np.arctan2(y, x)

    return np.degrees(theta), np.degrees(phi)


def wrap_phi_to_grid(phi_deg, phis_grid):
    """
    Wrap phi_deg into the same convention/range as phis_grid.
    Supports grids like [0,360] or [-180,180].
    """
    phi = np.asarray(phi_deg, float)
    phis = np.asarray(phis_grid, float)
    pmin, pmax = float(np.nanmin(phis)), float(np.nanmax(phis))

    # Case 1: grid looks like 0..360 (or 0..359.xx)
    if pmin >= -1e-9 and pmax > 180:
        return np.mod(phi, 360.0)

    # Case 2: grid looks like -180..180
    if pmin < 0 and pmax <= 180 + 1e-9:
        return ((phi + 180.0) % 360.0) - 180.0

    # Fallback: map phi into [pmin,pmax] by shifting
    span = pmax - pmin
    if span <= 0:
        return phi
    return pmin + np.mod(phi - pmin, span)


def lab_theta_phi_from_Bhat(Bhat, phi_range="0_360"):
    """
    Absolute lab spherical angles:
      theta = arccos(z) in [0,180]
      phi   = atan2(y,x) in [-180,180] then mapped per phi_range
    """
    b = np.asarray(Bhat, float)
    b /= np.linalg.norm(b)

    theta = np.degrees(np.arccos(np.clip(b[2], -1.0, 1.0)))
    phi = np.degrees(np.arctan2(b[1], b[0]))  # [-180,180]

    if phi_range == "0_360":
        phi = phi % 360.0
    else:
        phi = ((phi + 180.0) % 360.0) - 180.0

    return theta, phi


def plot_full_angle_map_theta_phi_lab(
    angle_map: dict,
    title_prefix: str = "",
    contour: bool = True,
    n_levels: int = 12,
    add_kappa_contours: bool = False,
    overlay_points=None,  # [{"label": "...", "Bhat": np.array([x,y,z])}, ...]
):
    assert angle_map.get("basis") == "lab_spherical"

    thetas = np.asarray(angle_map["thetas_deg"], float)
    phis = np.asarray(angle_map["phis_deg"], float)
    f_minus = np.asarray(angle_map["f_minus_kHz"], float)
    f_plus = np.asarray(angle_map["f_plus_kHz"], float)
    kappa = np.asarray(angle_map.get("kappa", np.nan), float)
    phi_range = angle_map.get("phi_range", "0_360")

    PH, TH = np.meshgrid(phis, thetas)
    extent = [phis.min(), phis.max(), thetas.min(), thetas.max()]

    fig, axs = plt.subplots(
        1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True
    )

    def _one(ax, Z, title):
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent)
        if contour:
            levels = np.linspace(np.nanmin(Z), np.nanmax(Z), n_levels)
            cs = ax.contour(PH, TH, Z, levels=levels, colors="w", linewidths=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        if add_kappa_contours:
            ck = ax.contour(
                PH,
                TH,
                kappa,
                levels=np.linspace(0.1, 0.9, 9),
                colors="m",
                linewidths=0.6,
                alpha=0.7,
            )
            ax.clabel(ck, inline=True, fontsize=6, fmt="κ=%.1f")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(r"lab azimuth $\phi$ [deg]")
        ax.set_xlim(phis.min(), phis.max())
        ax.set_ylim(thetas.min(), thetas.max())
        return im

    im0 = _one(axs[0], f_minus, r"$f_-(\theta,\phi)$  [kHz]")
    im1 = _one(axs[1], f_plus, r"$f_+(\theta,\phi)$  [kHz]")
    axs[0].set_ylabel(r"lab polar $\theta$ [deg] from +z")

    # --- overlay experimental directions ---
    if overlay_points:
        th_list, ph_list, labels = [], [], []
        for p in overlay_points:
            b = np.asarray(p["Bhat"], float)
            b /= np.linalg.norm(b)
            th, ph = lab_theta_phi_from_Bhat(b, phi_range=phi_range)
            th_list.append(th)
            ph_list.append(ph)
            labels.append(str(p.get("label", "")))

        ph_arr = wrap_phi_to_grid(np.asarray(ph_list), phis)
        th_arr = np.asarray(th_list)

        for ax in axs:
            ax.scatter(ph_arr, th_arr, marker="o", s=60, edgecolors="k", linewidths=0.6)
            for ph, th, lab in zip(ph_arr, th_arr, labels):
                ax.text(ph + 2, th + 2, lab, fontsize=8)

    fig.colorbar(im0, ax=axs[0], label="kHz")
    fig.colorbar(im1, ax=axs[1], label="kHz")

    if title_prefix:
        Bmag_G = None
        sup = title_prefix + (f"  (|B|={Bmag_G:.1f} G)" if Bmag_G is not None else "")
        fig.suptitle(sup, y=1.00, fontsize=13)

    return fig, axs


def lab_theta_phi_from_Bvec(Bvec, *, phi_range="0_360"):
    """
    Convert a LAB-frame vector Bvec to LAB spherical angles:
      theta = arccos(z) in [0,180]
      phi   = atan2(y,x) in (-180,180] then wrapped to match phi_range
    """
    Bvec = np.asarray(Bvec, float)
    n = np.linalg.norm(Bvec)
    if n == 0:
        raise ValueError("Bvec has zero magnitude")
    bhat = Bvec / n

    theta = np.degrees(np.arccos(np.clip(bhat[2], -1.0, 1.0)))
    phi = np.degrees(np.arctan2(bhat[1], bhat[0]))

    if phi_range == "0_360":
        phi = phi % 360.0
    else:  # "-180_180"
        phi = ((phi + 180.0) % 360.0) - 180.0

    return float(theta), float(phi), bhat


def nearest_phi_index(phi_grid_deg, phi_deg):
    """
    Works for either 0..360 or -180..180 grids.
    Chooses nearest in circular sense.
    """
    ph = np.asarray(phi_grid_deg, float)
    # circular difference in (-180,180]
    d = (ph - float(phi_deg) + 180.0) % 360.0 - 180.0
    return int(np.argmin(np.abs(d)))


def validate_site72_direct_vs_labmap(
    hyperfine_path,
    orientation,
    site_index,
    Bvec_G,  # LAB vector in Gauss (or any units; direction matters)
    Bmag_G,  # the magnitude you used in the map (must match your direct sim)
    *,
    ms=-1,
    phi_deg_nv=120.0,  # IMPORTANT: must match what you used in fitting/catalog
    phi_range="0_360",
    n_theta=181,
    n_phi=360,
):
    # --- load A tensor for this site ---
    A_file_Hz, _ = load_A_file_Hz_by_site_index(hyperfine_path, site_index)

    # --- DIRECT at this exact B direction + magnitude ---
    _, _, bhat = lab_theta_phi_from_Bvec(Bvec_G, phi_range=phi_range)
    B_T = (float(Bmag_G) * 1e-4) * bhat

    kap0, f_m0, f_p0, *_ = _kappa_and_fpm(
        A_file_Hz, orientation, B_T, ms=ms, phi_deg=phi_deg_nv
    )
    fm0_kHz = f_m0 / 1e3
    fp0_kHz = f_p0 / 1e3

    # --- MAP over all lab directions at fixed |B| ---
    amap = compute_full_angle_map_lab_spherical_for_site(
        hyperfine_path=hyperfine_path,
        orientation=orientation,
        site_index=site_index,
        Bmag_G=Bmag_G,
        n_theta=n_theta,
        n_phi=n_phi,
        ms=ms,
        phi_deg_nv=phi_deg_nv,
        phi_range=phi_range,
    )

    # locate this B direction in (theta,phi)
    theta_deg, phi_deg, _ = lab_theta_phi_from_Bvec(Bvec_G, phi_range=phi_range)

    it = int(np.argmin(np.abs(amap["thetas_deg"] - theta_deg)))
    ip = nearest_phi_index(amap["phis_deg"], phi_deg)

    fm1_kHz = float(amap["f_minus_kHz"][it, ip])
    fp1_kHz = float(amap["f_plus_kHz"][it, ip])
    kap1 = float(amap["kappa"][it, ip])

    print(f"=== site (ori={orientation}, sid={site_index}) ===")
    print(
        f"target angles: theta={theta_deg:.3f} deg, phi={phi_deg:.3f} deg (grid idx it={it}, ip={ip})"
    )
    print(f"[direct]  f-={fm0_kHz:.9f} kHz, f+={fp0_kHz:.9f} kHz, kappa={kap0:.12f}")
    print(f"[map]     f-={fm1_kHz:.9f} kHz, f+={fp1_kHz:.9f} kHz, kappa={kap1:.12f}")
    print(
        f"Δ(direct-map): Δf-={fm0_kHz-fm1_kHz:.3e} kHz, Δf+={fp0_kHz-fp1_kHz:.3e} kHz, Δκ={kap0-kap1:.3e}"
    )


def load_A_file_Hz_by_site_index(hyperfine_path: str, site_index: int):
    df_hf = read_hyperfine_table_safe(hyperfine_path).copy()
    row = df_hf.iloc[int(site_index)]
    A_file_Hz = (
        np.array(
            [
                [row.Axx, row.Axy, row.Axz],
                [row.Axy, row.Ayy, row.Ayz],
                [row.Axz, row.Ayz, row.Azz],
            ],
            float,
        )
        * 1e6
    )
    return A_file_Hz, row


def simulate_fpm_vs_B_for_site_all_fields(
    hyperfine_path: str,
    site_key: tuple,  # (orientation_tuple, site_index)
    B_by_field: dict,  # {field_label: B_vec_G}
    Bmin_G: float = 20,
    Bmax_G: float = 80,
    n_B: int = 121,
    ms: int = -1,
    phi_deg: float = 0.0,
):
    ori, sid = site_key
    ori = tuple(ori)
    sid = int(sid)

    A_file_Hz, _ = load_A_file_Hz_by_site_index(hyperfine_path, sid)

    B_grid_G = np.linspace(Bmin_G, Bmax_G, n_B)

    recs = []
    for lab, Bvec_G in B_by_field.items():
        Bvec_G = np.asarray(Bvec_G, float)
        Bhat = Bvec_G / np.linalg.norm(Bvec_G)

        for Bg in B_grid_G:
            B_T = (Bg * 1e-4) * Bhat
            kappa, f_minus, f_plus, *_ = _kappa_and_fpm(
                A_file_Hz, ori, B_T, ms=ms, phi_deg=phi_deg
            )
            recs.append(
                dict(
                    field_label=lab,
                    B_G=float(Bg),
                    f_minus_kHz=float(f_minus / 1e3),
                    f_plus_kHz=float(f_plus / 1e3),
                    kappa=float(kappa),
                )
            )
    return pd.DataFrame(recs)


def plot_site_sim_overlay(site_key, sim_df, B_by_field, title_extra=""):
    ori, sid = site_key
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

    for lab, g in sim_df.groupby("field_label"):
        g = g.sort_values("B_G")
        axs[0].plot(g["B_G"], g["f_minus_kHz"], label=lab)
        axs[1].plot(g["B_G"], g["f_plus_kHz"], label=lab)

        # mark the actual measured magnitude for this field
        Bmag = float(np.linalg.norm(B_by_field[lab]))
        # nearest simulated point
        axs[0].axvline(Bmag, alpha=0.2)
        axs[1].axvline(Bmag, alpha=0.2)

    axs[0].set_title(r"Simulated $f_-(|B|)$")
    axs[1].set_title(r"Simulated $f_+(|B|)$")
    for ax in axs:
        ax.set_xlabel("|B| [G]")
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("kHz")
    axs[1].legend()

    fig.suptitle(f"Site {sid}, ori={ori} {title_extra}", y=1.02)
    fig.tight_layout()
    return fig, axs


# ---------- 0) IO ----------
def load_catalog(path_json: str) -> List[Dict]:
    """Load ESEEM catalog (with κ and f± in Hz) written by your builder."""
    with open(path_json, "r") as f:
        return json.load(f)


def _ori_tuple(o) -> Tuple[int, int, int]:
    return tuple(int(x) for x in o)


# ---------- 1) Filter records by orientation and frequency band ----------
def select_records(
    records: List[Dict],
    fmin_kHz: float = 150.0,
    fmax_kHz: float = 20000.0,
    orientations: Optional[Iterable[Tuple[int, int, int]]] = None,
    use_plus_and_minus: bool = True,
) -> List[Dict]:
    """
    Keep entries whose f_- or f_+ (Hz) lie in [fmin_kHz, fmax_kHz].
    If orientations is provided, keep only those NV orientations.
    """
    lo, hi = float(fmin_kHz) * 1e3, float(fmax_kHz) * 1e3
    allowed = None if orientations is None else {_ori_tuple(o) for o in orientations}
    out = []
    for r in records:
        if allowed is not None and _ori_tuple(r["orientation"]) not in allowed:
            continue
        fm, fp = float(r["f_minus_Hz"]), float(r["f_plus_Hz"])
        keep = (
            ((lo <= fm <= hi) or (lo <= fp <= hi))
            if use_plus_and_minus
            else (lo <= fm <= hi)
        )
        if keep:
            out.append(r)
    return out


# ---------- 2) Extract sticks (frequency, weight) ----------
def lines_from_recs(
    records: List[Dict],
    orientations: Optional[Iterable[Tuple[int, int, int]]] = None,
    fmin_kHz: float = 0.0,
    fmax_kHz: float = np.inf,
    *,
    weight_mode: str = "kappa",  # {"kappa", "per_line", "unit"}
    per_line_key_minus: str = "line_w_minus",
    per_line_key_plus: str = "line_w_plus",
    kappa_key: str = "kappa",
    per_line_scale: float = 1.0,  # extra multiplier if desired
) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """
    Convert selected records into arrays of (freq_kHz, weight) sticks.

    weight_mode:
      - "kappa"    → use raw κ for each site (same κ for both f±), scaled by per_line_scale
      - "per_line" → use the catalog’s per-line weights (e.g., p_occ·κ/4) saved as
                     `line_w_minus/line_w_plus`, then multiply by per_line_scale
      - "unit"     → all sticks weight=1 (then multiply by per_line_scale)

    Returns
    -------
    freqs_kHz : ndarray  (sorted)
    weights   : ndarray  (sorted like freqs_kHz)
    meta      : list of (orientation, site_index, "+/-tag") in the same order
    """
    lo, hi = float(fmin_kHz) * 1e3, float(fmax_kHz) * 1e3
    allowed = None if orientations is None else {_ori_tuple(o) for o in orientations}

    F, W, M = [], [], []
    for r in records:
        if allowed is not None and _ori_tuple(r["orientation"]) not in allowed:
            continue

        kappa_val = float(r.get(kappa_key, 0.0))
        for tag, wkey in (
            ("f_minus_Hz", per_line_key_minus),
            ("f_plus_Hz", per_line_key_plus),
        ):
            fHz = float(r[tag])
            if not (lo <= fHz <= hi):
                continue
            F.append(fHz * 1e-3)  # to kHz
            if weight_mode == "unit":
                w = 1.0
            elif weight_mode == "per_line":
                # e.g., p_occ·κ/4 already encoded per line
                w = float(r.get(wkey, 0.0))
            elif weight_mode == "kappa":
                w = kappa_val
            else:
                raise ValueError(f"Unknown weight_mode='{weight_mode}'")
            W.append(per_line_scale * w)
            M.append((_ori_tuple(r["orientation"]), int(r["site_index"]), tag))

    if not F:
        return np.array([]), np.array([]), []
    order = np.argsort(F)
    return np.asarray(F)[order], np.asarray(W)[order], [M[i] for i in order]


# ---------- 3A) Plot discrete sticks with explicit κ labels ----------
def plot_sticks_kappa(
    freqs_kHz: np.ndarray,
    weights: Optional[np.ndarray] = None,
    meta: Optional[List[Tuple]] = None,
    *,
    title: str = None,
    weight_caption: str = None,
    min_weight: float = 0.0,
):
    """
    Vertical-stick plot for ESEEM lines.

    If `meta` is provided (from lines_from_recs), it must be a list of
    (orientation, site_index, tag) where `tag` identifies f- vs f+.
    In that case, f₋ and f₊ are plotted in different colors with a legend.

    `weight_caption` describes what the weights mean (e.g., 'raw κ',
    or 'per-line weight = p_occ·κ/4').
    """
    if freqs_kHz.size == 0:
        print("[plot_sticks_kappa] No lines to plot.")
        return

    # Weights
    w = np.ones_like(freqs_kHz) if weights is None else np.asarray(weights, float)

    # Apply weight threshold
    m = w >= float(min_weight)
    f = freqs_kHz[m]
    w = w[m]
    if meta is not None:
        meta = [meta[i] for i, keep in enumerate(m) if keep]

    # Sort by frequency
    order = np.argsort(f)
    f = f[order]
    w = w[order]
    if meta is not None:
        meta = [meta[i] for i in order]

    if title is None:
        title = "Discrete ESEEM sticks (sorted by frequency)"
    if weight_caption is None:
        weight_caption = "Weight"

    plt.figure(figsize=(18, 6))

    # ---------- Case A: no meta → single-color fallback ----------
    if meta is None:
        for fk, wk in zip(f, w):
            plt.vlines(fk, 0.0, wk, linewidth=0.6)
    else:
        # Extract tags and classify f- vs f+ in a robust way
        tags_raw = [
            m_i[2] for m_i in meta
        ]  # third entry is tag ("f_minus_Hz" / "f_plus_Hz")
        tags_str = [str(t).lower() for t in tags_raw]

        is_minus = np.array(["minus" in t for t in tags_str], dtype=bool)
        is_plus = np.array(["plus" in t for t in tags_str], dtype=bool)

        print(
            f"[plot_sticks_kappa] N_minus={is_minus.sum()}, N_plus={is_plus.sum()}, N_total={len(tags_str)}"
        )

        # f− (difference branch)
        if np.any(is_minus):
            plt.vlines(
                f[is_minus],
                0.0,
                w[is_minus],
                linewidth=0.6,
                color="C0",
                label=r"$f_{-}$",
            )

        # f+ (sum branch)
        if np.any(is_plus):
            plt.vlines(
                f[is_plus],
                0.0,
                w[is_plus],
                linewidth=0.6,
                color="C1",
                label=r"$f_{+}$",
            )

        # If tags couldn't be interpreted, fall back
        if (not np.any(is_minus)) and (not np.any(is_plus)):
            print(
                "[plot_sticks_kappa] WARNING: could not identify f- vs f+ from meta tags; using single color."
            )
            for fk, wk in zip(f, w):
                plt.vlines(fk, 0.0, wk, linewidth=0.6)
        else:
            plt.legend(loc="upper right", fontsize=10)

    plt.title(title)
    plt.xlabel("Frequency (kHz)")
    plt.xscale("log")
    plt.ylabel(weight_caption)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


# ---------- 3B) Convolved spectrum (Gaussian/Lorentzian) with precise labels ----------
def convolved_expected_from_recs(
    records: List[Dict],
    *,
    orientations: Optional[Iterable[Tuple[int, int, int]]] = None,
    f_range_kHz: Tuple[float, float] = (150, 20000),
    npts: int = 2000,
    shape: str = "gauss",
    width_kHz: float = 8.0,
    # weighting options (match lines_from_recs)
    weight_mode: str = "kappa",  # {"kappa","per_line","unit"}
    per_line_key_minus: str = "line_w_minus",
    per_line_key_plus: str = "line_w_plus",
    kappa_key: str = "kappa",
    per_line_scale: float = 1.0,
    # Larmor marker
    larmor_kHz: Optional[float] = None,
    # labeling
    title_prefix: str = "Expected ESEEM spectrum",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a smooth expected spectrum by convolving each discrete line with a kernel.
    Branch-resolved: f_- and f_+ are plotted in different colors.

    If larmor_kHz is not None, draw a vertical line at the nuclear Larmor frequency ν_I.

    weight_mode:
      - "kappa"    → weight = κ (same for f± per site) × per_line_scale
      - "per_line" → weight = record['line_w_minus/plus'] × per_line_scale
      - "unit"     → weight = 1 × per_line_scale

    Returns
    -------
    f : ndarray
        Frequency axis (kHz, log-spaced).
    spec_total : ndarray
        Total spectrum = spec_minus + spec_plus (+ any untagged lines).
    """
    freqs_kHz, w, meta = lines_from_recs(
        records,
        orientations=orientations,
        fmin_kHz=f_range_kHz[0],
        fmax_kHz=f_range_kHz[1],
        weight_mode=weight_mode,
        per_line_key_minus=per_line_key_minus,
        per_line_key_plus=per_line_key_plus,
        kappa_key=kappa_key,
        per_line_scale=per_line_scale,
    )
    if freqs_kHz.size == 0:
        raise ValueError("No lines in requested range.")

    # Log-spaced frequency axis
    f = np.logspace(np.log10(f_range_kHz[0]), np.log10(f_range_kHz[1]), int(npts))

    # ---------- Fallback if no meta (no branch tags) ----------
    if not meta:
        spec = np.zeros_like(f, float)

        if shape.lower().startswith("gauss"):
            s = float(width_kHz)
            norm = s * np.sqrt(2 * np.pi)
            for f0, a in zip(freqs_kHz, w):
                spec += a * np.exp(-0.5 * ((f - f0) / s) ** 2) / norm
            kern_label = f"Gaussian (σ={width_kHz:.2f} kHz)"
        else:
            g = float(width_kHz)
            for f0, a in zip(freqs_kHz, w):
                spec += a * (g / np.pi) / ((f - f0) ** 2 + g**2)
            kern_label = f"Lorentzian (γ={width_kHz:.2f} kHz)"

        # y-label text
        if weight_mode == "kappa":
            wlabel = "κ"
        elif weight_mode == "per_line":
            wlabel = "per-line weight (e.g., p_occ·κ/4)"
        else:
            wlabel = "unit weight"

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.set_xscale("log")
        ax.plot(f, spec, lw=1.6)
        ax.set_xlim(*f_range_kHz)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel(f"Expected intensity (arb.)\nweights = {wlabel}")
        title = f"{title_prefix} • {kern_label}"
        if orientations is not None:
            title += f" • orientations={list(map(tuple, orientations))}"
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)

        # --- Larmor marker (if requested) ---
        if larmor_kHz is not None:
            ax.axvline(larmor_kHz, linestyle=":", linewidth=1.0)
            ymin, ymax = ax.get_ylim()
            ax.text(
                larmor_kHz,
                ymax,
                r"$\nu_I$",
                rotation=90,
                ha="right",
                va="top",
                fontsize=9,
            )

        plt.tight_layout()
        plt.show()
        return f, spec

    # ---------- Branch-resolved case: split into f- and f+ ----------
    tags_raw = [m_i[2] for m_i in meta]  # ("f_minus_Hz" / "f_plus_Hz")
    tags_str = [str(t).lower() for t in tags_raw]

    is_minus = np.array(["minus" in t for t in tags_str], dtype=bool)
    is_plus = np.array(["plus" in t for t in tags_str], dtype=bool)
    is_other = ~(is_minus | is_plus)

    # Helper: convolve for a specific branch
    def _convolve_branch(freqs_branch_kHz, weights_branch):
        spec_branch = np.zeros_like(f, float)
        if freqs_branch_kHz.size == 0:
            return spec_branch

        if shape.lower().startswith("gauss"):
            s = float(width_kHz)
            norm = s * np.sqrt(2 * np.pi)
            for f0, a in zip(freqs_branch_kHz, weights_branch):
                spec_branch += a * np.exp(-0.5 * ((f - f0) / s) ** 2) / norm
        else:
            g = float(width_kHz)
            for f0, a in zip(freqs_branch_kHz, weights_branch):
                spec_branch += a * (g / np.pi) / ((f - f0) ** 2 + g**2)
        return spec_branch

    freqs_minus = freqs_kHz[is_minus]
    w_minus = w[is_minus]
    freqs_plus = freqs_kHz[is_plus]
    w_plus = w[is_plus]
    freqs_other = freqs_kHz[is_other]
    w_other = w[is_other]

    spec_minus = _convolve_branch(freqs_minus, w_minus)
    spec_plus = _convolve_branch(freqs_plus, w_plus)
    spec_other = _convolve_branch(freqs_other, w_other)
    spec_total = spec_minus + spec_plus + spec_other

    if shape.lower().startswith("gauss"):
        kern_label = f"Gaussian (σ={width_kHz:.2f} kHz)"
    else:
        kern_label = f"Lorentzian (γ={width_kHz:.2f} kHz)"

    if weight_mode == "kappa":
        wlabel = "κ (per line)"
    elif weight_mode == "per_line":
        wlabel = "per-line weight (e.g., p_occ·κ/4)"
    else:
        wlabel = "unit weight"

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xscale("log")
    ax.set_xlim(*f_range_kHz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(f"Expected intensity (arb.)\nweights = {wlabel}")

    # Plot branches with different styles
    if np.any(is_minus):
        ax.plot(f, spec_minus, lw=1.4, label=r"$f_{-}$")
    if np.any(is_plus):
        ax.plot(f, spec_plus, lw=1.4, label=r"$f_{+}$")
    if np.any(is_other):
        ax.plot(f, spec_other, lw=1.0, label="other lines", linestyle=":")

    # Total as thin dashed curve
    ax.plot(f, spec_total, lw=1.0, linestyle="--", alpha=0.6, label="sum")

    title = f"{title_prefix} • {kern_label}"
    if orientations is not None:
        title += f" • orientations={list(map(tuple, orientations))}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)

    # --- Larmor marker (if requested) ---
    if larmor_kHz is not None:
        ax.axvline(larmor_kHz, linestyle=":", linewidth=1.0)
        ymin, ymax = ax.get_ylim()
        ax.text(
            larmor_kHz,
            ymax,
            r"$\nu_I$",
            rotation=90,
            ha="right",
            va="top",
            fontsize=9,
        )

    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()

    return f, spec_total


# Small helper if you don't already have it
def _ori_tuple(o) -> Tuple[int, int, int]:
    return tuple(int(x) for x in o)


def pair_axis_from_recs(
    records: List[Dict],
    orientations: Optional[Iterable[Tuple[int, int, int]]] = None,
    *,
    fmin_kHz: float = 0.0,
    fmax_kHz: float = np.inf,
    axis: str = "f_mid",  # {"f_mid", "delta_f"}
    weight_mode: str = "kappa",  # {"kappa", "per_pair", "unit"}
    kappa_key: str = "kappa",
    per_line_key_minus: str = "line_w_minus",
    per_line_key_plus: str = "line_w_plus",
    per_pair_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """
    Build a 1D 'pair spectrum' from catalog records.

    axis:
        'f_mid'   → use (f_+ + f_-)/2  (Larmor-like frequency)
        'delta_f' → use (f_+ - f_-)/2  (anisotropy / mixing scale)

    weight_mode:
        'kappa'   → weight = κ for that site
        'per_pair'→ weight = line_w_minus + line_w_plus (if present)
        'unit'    → weight = 1 for every pair

    Returns
    -------
    freq_axis_kHz : np.ndarray
        1D axis in kHz (mid or delta, depending on `axis`).
    weights       : np.ndarray
        Same length as freq_axis_kHz.
    meta          : list of tuples
        (orientation, site_index, f_plus_kHz, f_minus_kHz) for each stick.
    """
    lo_Hz = float(fmin_kHz) * 1e3
    hi_Hz = float(fmax_kHz) * 1e3

    allowed = None if orientations is None else {_ori_tuple(o) for o in orientations}

    F_axis = []
    W = []
    meta = []

    axis = axis.lower()
    if axis not in {"f_mid", "delta_f"}:
        raise ValueError(f"axis must be 'f_mid' or 'delta_f', got {axis!r}")

    for r in records:
        ori = _ori_tuple(r["orientation"])
        if allowed is not None and ori not in allowed:
            continue

        f_plus_Hz = float(r["f_plus_Hz"])
        f_minus_Hz = float(r["f_minus_Hz"])

        # Require both branches in the band (like your pair extractor)
        if not (lo_Hz <= f_plus_Hz <= hi_Hz and lo_Hz <= f_minus_Hz <= hi_Hz):
            continue

        # Convert to kHz
        f_plus_kHz = f_plus_Hz * 1e-3
        f_minus_kHz = f_minus_Hz * 1e-3

        # Pair axes
        f_mid_kHz = 0.5 * (f_plus_kHz + f_minus_kHz)
        delta_f_kHz = 0.5 * abs(f_plus_kHz - f_minus_kHz)

        if axis == "f_mid":
            f_axis = f_mid_kHz
        else:  # "delta_f"
            f_axis = delta_f_kHz

        # Weight choices
        if weight_mode == "unit":
            w = 1.0
        elif weight_mode == "kappa":
            w = float(r.get(kappa_key, 0.0))
        elif weight_mode == "per_pair":
            w_m = float(r.get(per_line_key_minus, 0.0))
            w_p = float(r.get(per_line_key_plus, 0.0))
            w = w_m + w_p
        else:
            raise ValueError(f"Unknown weight_mode='{weight_mode}'")

        w *= per_pair_scale

        F_axis.append(f_axis)
        W.append(w)
        meta.append((ori, int(r["site_index"]), f_plus_kHz, f_minus_kHz))

    if not F_axis:
        return np.array([]), np.array([]), []

    F_axis = np.asarray(F_axis, float)
    W = np.asarray(W, float)

    order = np.argsort(F_axis)
    return F_axis[order], W[order], [meta[i] for i in order]


def plot_pair_spectrum(
    freq_axis_kHz: np.ndarray,
    weights: Optional[np.ndarray] = None,
    *,
    axis: str = "f_mid",  # {"f_mid", "delta_f"}
    title: Optional[str] = None,
    min_weight: float = 0.0,
):
    """
    Plot a 1D stick spectrum of ESEEM pairs along either f_mid or Δf.

    freq_axis_kHz : output from pair_axis_from_recs(...).
    weights       : same length, or None → all ones.
    axis          : 'f_mid' or 'delta_f' (for axis label/title).
    """
    axis = axis.lower()
    if freq_axis_kHz.size == 0:
        print("[plot_pair_spectrum] No pairs to plot.")
        return

    w = np.ones_like(freq_axis_kHz) if weights is None else np.asarray(weights, float)
    m = w >= float(min_weight)
    f = freq_axis_kHz[m]
    w = w[m]

    order = np.argsort(f)
    f = f[order]
    w = w[order]

    if axis == "f_mid":
        xlabel = r"$f_{\mathrm{mid}} = (f_+ + f_-)/2$ (kHz)"
        default_title = "ESEEM pair spectrum vs mid-frequency"
    else:
        xlabel = r"$\Delta f = (f_+ - f_-)/2$ (kHz)"
        default_title = "ESEEM pair spectrum vs half-splitting"

    if title is None:
        title = default_title

    plt.figure(figsize=(10, 5))
    for fk, wk in zip(f, w):
        plt.vlines(fk, 0.0, wk, linewidth=0.7)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Weight")
    plt.xscale("log")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def expected_sticks_with_multiplicity_pair(
    records: List[Dict],
    *,
    orientations: Optional[List[Tuple[int, int, int]]] = None,
    fmin_kHz: float = 1.0,
    fmax_kHz: float = 20000.0,
    # first-order weighting
    use_first_order: bool = True,
    first_weight_mode: str = "kappa",  # {"kappa","per_line","unit"}
    per_line_key_minus: str = "line_w_minus",
    per_line_key_plus: str = "line_w_plus",
    kappa_key: str = "kappa",
    per_line_scale: float = 1.0,  # e.g. =1 or =p_occ/4 if you want that explicitly
    # pair (second-order) options
    use_pairs: bool = True,
    pair_scale: float = 1.0,  # global fudge factor for pair amplitude model
    p_occ: float = 0.011,  # Bernoulli prob per site (if per-line weights didn't already include it)
    top_k_by_kappa: Optional[int] = 200,  # limit to strongest sites to cap O(N^2)
):
    """
    Build an 'expected' stick list including:
      • First-order lines f_{i±} with chosen weights.
      • Optional pair lines at |f_i ± f_j| with weak weights ~ p_i p_j κ_i κ_j / 16 × pair_scale.

    Returns: freqs_kHz, weights, tags
    tags: list of tuples like ("1st", site_index, "+/-") or ("pair", i, j, "diff"/"sum")
    """

    # Helper: select by orientation & band
    def _ori_tuple(o):
        return tuple(int(x) for x in o)

    allowed = None if orientations is None else {_ori_tuple(o) for o in orientations}
    lo, hi = float(fmin_kHz) * 1e3, float(fmax_kHz) * 1e3

    # Pull per-site info
    site_entries = []
    for r in records:
        if allowed is not None and _ori_tuple(r["orientation"]) not in allowed:
            continue
        kappa = float(r.get(kappa_key, 0.0))
        f_minus = float(r["f_minus_Hz"])
        f_plus = float(r["f_plus_Hz"])
        if not np.isfinite(kappa):
            kappa = 0.0
        site_entries.append((int(r["site_index"]), kappa, f_minus, f_plus, r))

    if not site_entries:
        return np.array([]), np.array([]), []

    # Sort by kappa (desc) to optionally trim pairs
    site_entries.sort(key=lambda t: t[1], reverse=True)
    if top_k_by_kappa is not None:
        site_entries = site_entries[: int(top_k_by_kappa)]

    # First-order sticks
    F = []
    W = []
    TAGS = []
    if use_first_order:
        for site_idx, kappa, fmi, fpl, r in site_entries:
            for tag, fHz in (("-", fmi), ("+", fpl)):
                if not (lo <= fHz <= hi):
                    continue
                if first_weight_mode == "kappa":
                    w = per_line_scale * kappa
                elif first_weight_mode == "per_line":
                    key = per_line_key_minus if tag == "-" else per_line_key_plus
                    w = per_line_scale * float(r.get(key, 0.0))
                else:
                    w = per_line_scale * 1.0
                F.append(fHz * 1e-3)
                W.append(w)
                TAGS.append(("1st", site_idx, tag))

    # Pair (second-order) sticks
    if use_pairs:
        # Simple amplitude model:
        #   expected amplitude ~ (p_occ^2) * (κ_i κ_j / 16) * pair_scale
        # You can swap to a more precise expression later if desired.
        n = len(site_entries)
        for a in range(n):
            i, ki, fmi_i, fpl_i, _ri = site_entries[a]
            for b in range(a + 1, n):
                j, kj, fmi_j, fpl_j, _rj = site_entries[b]
                # Choose one representative per site (you can also include ± branches explicitly)
                # We'll include pair lines at |f_i - f_j| and f_i + f_j using the "dominant" branch per site.
                # For robustness we include both using f_plus for both sites (typ. close to ω_-1 + ω_I).
                # You can also mix (f_plus,f_minus) combos if you prefer denser modeling.
                for Fi, Fj, lab in ((fpl_i, fpl_j, "pp"), (fmi_i, fmi_j, "mm")):
                    # difference
                    f_diff = abs(Fi - Fj)
                    if lo <= f_diff <= hi:
                        w_pair = pair_scale * (p_occ**2) * ((ki * kj) / 16.0)
                        F.append(f_diff * 1e-3)
                        W.append(w_pair)
                        TAGS.append(("pair", i, j, "diff", lab))
                    # sum
                    f_sum = Fi + Fj
                    if lo <= f_sum <= hi:
                        w_pair = pair_scale * (p_occ**2) * ((ki * kj) / 16.0)
                        F.append(f_sum * 1e-3)
                        W.append(w_pair)
                        TAGS.append(("pair", i, j, "sum", lab))

    if not F:
        return np.array([]), np.array([]), []
    order = np.argsort(F)
    return np.asarray(F)[order], np.asarray(W)[order], [TAGS[k] for k in order]


def expected_sticks_general(
    records: List[Dict],
    *,
    # selection
    orientations: Optional[List[Tuple[int, int, int]]] = None,
    fmin_kHz: float = 1.0,
    fmax_kHz: float = 20000.0,
    kappa_key: str = "kappa",
    # 1st-order
    include_order1: bool = True,
    first_weight_mode: str = "kappa",  # {"kappa","per_line","unit"}
    per_line_key_minus: str = "line_w_minus",
    per_line_key_plus: str = "line_w_plus",
    scale_order1: float = 1.0,  # multiplies chosen first-order weight
    p_occ: float = 0.011,  # only used if first_weight_mode="kappa"
    # 2nd-order (pairs)
    include_order2: bool = True,
    pair_scale: float = 1.0,  # global factor for pair weights
    # branch policy for pairs/triples
    branch_mode: str = "pp_mm",  # {"pp_mm","plus_only","all_pairs_strict"}
    # 3rd-order (triples)
    include_order3: bool = False,
    triple_scale: float = 1.0,
    triple_branch_mode: Optional[str] = None,  # None→same as branch_mode
    # complexity control
    top_k_by_kappa: Optional[int] = 200,  # trim before O(K^2/3)
    # dedup
    dedup_tol_kHz: float = 0.5,
):
    """
    Build expected ESEEM sticks including 1st order (f_{i±}), pairs (|f_i ± f_j|),
    and optional triples. Designed to be *compatible* with your prior pair-only
    function when:
        include_order1=True, include_order2=True, include_order3=False,
        first_weight_mode="kappa", scale_order1=1.0,
        branch_mode="pp_mm", pair_scale=1.0, top_k_by_kappa=200,
        dedup_tol_kHz≈previous value.
    That setting reproduces: use (f+,+) and (f-, -) for pairs, both sum & diff,
    with pair weights ∝ (p_occ^2) * (kappa_i*kappa_j) / 16.
    """

    def _ori_tuple(o):
        return tuple(int(x) for x in o)

    def _branch_freq(fmi, fpl, s):  # s ∈ {+1,-1}
        return fpl if s > 0 else fmi

    # -------- selection by orientation + band limits (in Hz) --------
    allowed = None if orientations is None else {_ori_tuple(o) for o in orientations}
    lo_Hz, hi_Hz = float(fmin_kHz) * 1e3, float(fmax_kHz) * 1e3

    # -------- collect site entries: (site_idx, kappa, f_minus_Hz, f_plus_Hz, per-line dict) --------
    site_entries = []
    for r in records:
        if allowed is not None and _ori_tuple(r["orientation"]) not in allowed:
            continue
        kap = float(r.get(kappa_key, 0.0))
        fmi = float(r["f_minus_Hz"])
        fpl = float(r["f_plus_Hz"])
        site_entries.append(
            (int(r["site_index"]), (0.0 if not np.isfinite(kap) else kap), fmi, fpl, r)
        )

    if not site_entries:
        return np.array([]), np.array([]), []

    # stable sort by kappa (desc), then by site_index for determinism
    site_entries.sort(key=lambda t: (-t[1], t[0]))

    # optional trimming before combinatorics
    if top_k_by_kappa is not None:
        site_entries = site_entries[: int(top_k_by_kappa)]

    # -------- branch sets --------
    if branch_mode == "pp_mm":
        pair_branches = (("pp", (1, 1)), ("mm", (-1, -1)))
    elif branch_mode == "plus_only":
        pair_branches = (("pp", (1, 1)),)
    elif branch_mode == "all_pairs_strict":
        pair_branches = (
            ("pp", (1, 1)),
            ("pm", (1, -1)),
            ("mp", (-1, 1)),
            ("mm", (-1, -1)),
        )
    else:
        raise ValueError(
            "branch_mode must be one of {'pp_mm','plus_only','all_pairs_strict'}"
        )

    if triple_branch_mode is None:
        triple_branch_mode = branch_mode
    if triple_branch_mode == "pp_mm":
        triple_branches = (("ppp", (1, 1, 1)), ("mmm", (-1, -1, -1)))
    elif triple_branch_mode == "plus_only":
        triple_branches = (("ppp", (1, 1, 1)),)
    elif triple_branch_mode == "all_pairs_strict":
        triple_branches = tuple(
            ("".join("p" if s > 0 else "m" for s in bt), bt)
            for bt in product((1, -1), repeat=3)
        )
    else:
        raise ValueError(
            "triple_branch_mode must be one of {'pp_mm','plus_only','all_pairs_strict'}"
        )

    F, W, TAGS = [], [], []

    # -------- 1st order --------
    if include_order1:
        for i, kap, fmi, fpl, r in site_entries:
            for tag, fHz in (("+", fpl), ("-", fmi)):
                if lo_Hz <= fHz <= hi_Hz:
                    if first_weight_mode == "kappa":
                        w1 = scale_order1 * (p_occ * kap / 4.0)
                    elif first_weight_mode == "per_line":
                        key = per_line_key_plus if tag == "+" else per_line_key_minus
                        w1 = scale_order1 * float(r.get(key, 0.0))
                    elif first_weight_mode == "unit":
                        w1 = scale_order1 * 1.0
                    else:
                        raise ValueError(
                            "first_weight_mode must be {'kappa','per_line','unit'}"
                        )
                    F.append(fHz * 1e-3)
                    W.append(w1)
                    TAGS.append(("1st", i, tag))

    # -------- 2nd order (pairs) --------
    if include_order2:
        for (i, ki, fmi_i, fpl_i, _ri), (j, kj, fmi_j, fpl_j, _rj) in combinations(
            site_entries, 2
        ):
            for lab, (si, sj) in pair_branches:
                Fi = _branch_freq(fmi_i, fpl_i, si)
                Fj = _branch_freq(fmi_j, fpl_j, sj)
                # sum
                f_sum = Fi + Fj
                if lo_Hz <= f_sum <= hi_Hz:
                    w2 = pair_scale * ((p_occ**2) * (ki * kj) / (4.0**2))
                    F.append(f_sum * 1e-3)
                    W.append(w2)
                    TAGS.append(("pair", i, j, "sum", lab))
                # diff
                f_diff = abs(Fi - Fj)
                if lo_Hz <= f_diff <= hi_Hz:
                    w2 = pair_scale * ((p_occ**2) * (ki * kj) / (4.0**2))
                    F.append(f_diff * 1e-3)
                    W.append(w2)
                    TAGS.append(("pair", i, j, "diff", lab))

    # -------- 3rd order (triples) --------
    if include_order3:
        for (
            (i, ki, fmi_i, fpl_i, _ri),
            (j, kj, fmi_j, fpl_j, _rj),
            (k, kk, fmi_k, fpl_k, _rk),
        ) in combinations(site_entries, 3):
            for lab, bt in triple_branches:
                Fi = _branch_freq(fmi_i, fpl_i, bt[0])
                Fj = _branch_freq(fmi_j, fpl_j, bt[1])
                Fk = _branch_freq(fmi_k, fpl_k, bt[2])
                # two canonical triple sums (others are permutations/signs)
                for comb_name, signs in (
                    ("sum", (+1, +1, +1)),
                    ("sum-mix", (+1, +1, -1)),
                ):
                    fH = abs(signs[0] * Fi + signs[1] * Fj + signs[2] * Fk)
                    if lo_Hz <= fH <= hi_Hz:
                        w3 = triple_scale * ((p_occ**3) * (ki * kj * kk) / (4.0**3))
                        F.append(fH * 1e-3)
                        W.append(w3)
                        TAGS.append(("triple", i, j, k, comb_name, lab))

    if not F:
        return np.array([]), np.array([]), []

    # -------- sort & de-duplicate --------
    F = np.asarray(F, float)
    W = np.asarray(W, float)
    order = np.argsort(F)
    F, W = F[order], W[order]
    TAGS = [TAGS[t] for t in order]

    if dedup_tol_kHz is not None and dedup_tol_kHz > 0:
        f_out, w_out, tags_out = [F[0]], [W[0]], [TAGS[0]]
        for fk, wk, tg in zip(F[1:], W[1:], TAGS[1:]):
            if abs(fk - f_out[-1]) <= dedup_tol_kHz:
                w_out[-1] += wk
            else:
                f_out.append(fk)
                w_out.append(wk)
                tags_out.append(tg)
        F, W, TAGS = np.array(f_out), np.array(w_out), tags_out

    return F, W, TAGS


def convolved_exp_spectrum(
    freqs_kHz: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    *,
    f_range_kHz: Tuple[float, float] = (1.0, 20000.0),
    npts: int = 2400,
    shape: str = "gauss",  # {"gauss","lorentz"}
    width_kHz: float = 8.0,  # σ for Gauss, γ (HWHM) for Lorentz
    title_prefix: str = "Experimental ESEEM spectrum",
    weight_caption: str = "unit weight",
    log_x: bool = True,
    normalize_area: bool = False,  # if True, normalize area under spectrum to 1
) -> Tuple[np.ndarray, np.ndarray, plt.Figure, plt.Axes]:
    """
    Build a smooth spectrum by convolving discrete experimental lines with
    a Gaussian (σ) or Lorentzian (γ) kernel.

    Parameters
    ----------
    freqs_kHz : array
        Discrete line positions in kHz.
    weights : array or None
        Per-line amplitudes (same length as freqs_kHz). If None, all ones.
    f_range_kHz : (lo, hi)
        Frequency window [kHz] for evaluation and display.
    npts : int
        Number of sample points in the output spectrum.
    shape : {"gauss","lorentz"}
        Convolution kernel type.
    width_kHz : float
        Kernel width: σ for Gaussian; γ (=HWHM) for Lorentzian.
    title_prefix : str
        Figure title prefix.
    weight_caption : str
        Text shown in the y-label describing weights.
    log_x : bool
        If True, use a log-spaced frequency grid; else linear.
    normalize_area : bool
        If True, normalize the resulting spectrum to unit area
        (useful when comparing shapes independent of overall scale).

    Returns
    -------
    f : array (kHz)
        Frequency grid.
    spec : array (arb.)
        Convolved spectrum on the grid.
    fig, ax : matplotlib Figure and Axes
        The created plot objects (for further customization).
    """
    freqs_kHz = np.asarray(freqs_kHz, float)
    if freqs_kHz.size == 0:
        raise ValueError("No experimental lines provided.")

    if weights is None:
        w = np.ones_like(freqs_kHz, float)
    else:
        w = np.asarray(weights, float)
        if w.shape != freqs_kHz.shape:
            raise ValueError("weights must have the same shape as freqs_kHz")

    lo, hi = map(float, f_range_kHz)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError("f_range_kHz must be finite with hi > lo")

    # Keep only lines inside the plotting band and finite weights
    m = np.isfinite(freqs_kHz) & np.isfinite(w) & (freqs_kHz >= lo) & (freqs_kHz <= hi)
    f0 = freqs_kHz[m]
    a0 = w[m]
    if f0.size == 0:
        raise ValueError("No experimental lines in requested range.")

    # Output grid
    if log_x:
        f = np.logspace(np.log10(lo), np.log10(hi), int(npts))
    else:
        f = np.linspace(lo, hi, int(npts))

    spec = np.zeros_like(f, float)
    shape_l = shape.lower().strip()

    if shape_l.startswith("gauss"):
        s = float(width_kHz)
        if s <= 0:
            raise ValueError("Gaussian σ must be > 0")
        # Normalized Gaussian kernel: (1/(σ√(2π))) exp(-(Δf)^2/(2σ^2))
        norm = s * np.sqrt(2 * np.pi)
        for fk, ak in zip(f0, a0):
            spec += ak * np.exp(-0.5 * ((f - fk) / s) ** 2) / norm
        kern_label = f"Gaussian (σ={width_kHz:.2f} kHz)"
    elif shape_l.startswith("lorentz"):
        g = float(width_kHz)
        if g <= 0:
            raise ValueError("Lorentzian γ must be > 0")
        # Normalized Lorentzian kernel: (1/π) * (γ / ((Δf)^2 + γ^2))
        for fk, ak in zip(f0, a0):
            spec += ak * (g / np.pi) / ((f - fk) ** 2 + g**2)
        kern_label = f"Lorentzian (γ={width_kHz:.2f} kHz)"
    else:
        raise ValueError("shape must be 'gauss' or 'lorentz'")

    if normalize_area:
        # Trapezoidal area normalization on the chosen grid
        area = np.trapz(spec, f)
        if area > 0:
            spec /= area

    fig, ax = plt.subplots(figsize=(9, 5))
    if log_x:
        ax.set_xscale("log")
    ax.plot(f, spec, lw=1.6)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(f"Intensity (arb.)\nweights = {weight_caption}")
    ax.set_title(f"{title_prefix} • {kern_label}")
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    return f, spec, fig, ax


# ---------- small helper used by your panel plot ----------
def _mask_huge_errors(y, yerr, *, rel_cap=1.0, pct_cap=99):
    """
    Return a copy of yerr where obviously huge / broken errors are zeroed.
    - rel_cap: drop points where yerr > rel_cap * |y|
    - pct_cap: also drop errors above this percentile of the remaining ones.
    """
    if yerr is None:
        return None

    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float).copy()

    bad = ~np.isfinite(yerr)

    # only operate where both y and yerr are finite
    finite = np.isfinite(y) & np.isfinite(yerr)
    if np.any(finite):
        # 1) relative cap
        bad |= yerr > rel_cap * np.maximum(1e-12, np.abs(y))

        # 2) percentile cap – make percentile safe
        pct = float(pct_cap)
        if not np.isfinite(pct):
            pct = 100.0
        pct = min(max(pct, 0.0), 100.0)  # clamp to [0, 100]

        tail_mask = ~bad & finite
        if np.any(tail_mask):
            thr = np.nanpercentile(yerr[tail_mask], pct)
            bad |= yerr > thr

    # zero out bad errors
    yerr[bad] = 0.0
    return yerr


def plot_sorted_branches(
    f0_kHz,
    f1_kHz,
    sf0_kHz=None,
    sf1_kHz=None,
    *,
    title_prefix="Spin-echo",
    f_range_kHz=(15, 15000),
    label0=r"$f_+$ (lower branch)",
    label1=r"$f_-$ (upper branch)",
    A_rel_cap=0.7,
    A_pct_cap=200.0,
):
    """
    Generic version: plot two branches (f0, f1) sorted by magnitude.

    You can use it for experiment or theory by changing title_prefix/labels.
    """
    f0_kHz = np.asarray(f0_kHz, float)
    f1_kHz = np.asarray(f1_kHz, float)
    sf0_kHz = None if sf0_kHz is None else np.asarray(sf0_kHz, float)
    sf1_kHz = None if sf1_kHz is None else np.asarray(sf1_kHz, float)

    # --- Branch f0 ---
    valid0 = np.isfinite(f0_kHz) & (f0_kHz > 0)
    if np.any(valid0):
        idx0 = np.where(valid0)[0]
        order0 = idx0[np.argsort(f0_kHz[idx0])]
        x0 = np.arange(1, order0.size + 1)
        y0 = f0_kHz[order0]
        y0err_raw = sf0_kHz[order0] if sf0_kHz is not None else None
        y0err = _mask_huge_errors(y0, y0err_raw, rel_cap=A_rel_cap, pct_cap=A_pct_cap)
    else:
        order0 = np.array([], int)
        x0 = y0 = y0err = None

    # --- Branch f1 ---
    valid1 = np.isfinite(f1_kHz) & (f1_kHz > 0)
    if np.any(valid1):
        idx1 = np.where(valid1)[0]
        order1 = idx1[np.argsort(f1_kHz[idx1])]
        x1 = np.arange(1, order1.size + 1)
        y1 = f1_kHz[order1]
        y1err_raw = sf1_kHz[order1] if sf1_kHz is not None else None
        y1err = _mask_huge_errors(y1, y1err_raw, rel_cap=A_rel_cap, pct_cap=A_pct_cap)
    else:
        order1 = np.array([], int)
        x1 = y1 = y1err = None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    if x0 is not None:
        ax.errorbar(
            x0,
            y0,
            yerr=y0err,
            fmt="o",
            ms=1,
            lw=0.8,
            capsize=1,
            elinewidth=0.8,
            alpha=0.95,
            label=label0,
        )

    if x1 is not None:
        ax.errorbar(
            x1,
            y1,
            yerr=y1err,
            fmt="s",
            ms=1,
            lw=0.8,
            capsize=1,
            elinewidth=0.8,
            alpha=0.95,
            label=label1,
        )

    ax.set_yscale("log")
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("NV index (sorted within each branch)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"{title_prefix}: $f_0$ and $f_1$ (sorted)")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3)

    note0 = f"Excluded f0: {(~valid0).sum()}"
    note1 = f"Excluded f1: {(~valid1).sum()}"
    ax.text(0.01, 0.97, note0, transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.text(0.01, 0.92, note1, transform=ax.transAxes, ha="left", va="top", fontsize=8)

    ax.legend(framealpha=0.85)
    fig.tight_layout()
    plt.show()

    return fig, ax


def plot_theoretical_kappa_vs_distance(
    catalog_json: str,
    orientations=None,
    distance_max_A: float | None = None,
):
    """
    Visualize catalog kappa values:
      - κ vs distance (scatter)
      - κ sorted in descending order vs rank
    """
    # ---- load catalog ----
    recs = load_catalog(catalog_json)

    if orientations is not None:
        ori_set = {tuple(o) for o in orientations}
        ori_str = ", ".join([f"[{o[0]}, {o[1]}, {o[2]}]" for o in ori_set])
        ori_label = f"orientations: {ori_str}"
    else:
        ori_set = None
        ori_label = "all orientations"

    dist = []
    kap = []

    for r in recs:
        if ori_set is not None and tuple(r["orientation"]) not in ori_set:
            continue

        dA = float(r.get("distance_A", np.nan))
        k = float(r.get("kappa", np.nan))

        if not (np.isfinite(dA) and np.isfinite(k)):
            continue

        if (distance_max_A is not None) and (dA > distance_max_A):
            continue

        dist.append(dA)
        kap.append(k)

    dist = np.asarray(dist, float)
    kap = np.asarray(kap, float)

    if dist.size == 0:
        raise ValueError("No finite (distance_A, kappa) pairs found in catalog.")

    # ---- build figure: κ vs r and κ sorted ----
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Panel 1: κ vs distance
    ax0.scatter(dist, kap, s=12, alpha=0.7)
    ax0.set_xlabel("Distance (Å)")
    ax0.set_ylabel(r"Theoretical $\kappa$")
    ax0.set_title(rf"$\kappa$ vs distance ({ori_label})")
    ax0.grid(True, alpha=0.3)

    # Panel 2: κ sorted (descending) vs rank
    kap_sorted = np.sort(kap)[::-1]
    ranks = np.arange(1, kap_sorted.size + 1)

    ax1.plot(ranks, kap_sorted, "o-", ms=2, lw=1.0, alpha=0.9)
    ax1.set_xlabel("Site rank (sorted by κ, descending)")
    ax1.set_ylabel(r"Theoretical $\kappa$")
    ax1.set_title(r"$\kappa$ spectrum (descending)")
    ax1.grid(True, alpha=0.3)
    # --- add full κ definition + small-coupling approximation on the left panel ---
    kappa_text = (
        r"$\kappa = \frac{|\boldsymbol{\Omega}_0 \times \boldsymbol{\Omega}_m|^2}"
        r"{|\boldsymbol{\Omega}_0|^2\,|\boldsymbol{\Omega}_m|^2}"
        r" = \sin^2\theta$"
        "\n"
        r"$\boldsymbol{\Omega}_0 = \omega_I \hat{\mathbf{B}},\ "
        r"\boldsymbol{\Omega}_m = \boldsymbol{\Omega}_0 + m_s \mathbf{A}$"
        "\n"
        r"Small-coupling ($|A|\ll\omega_I$): "
        r"$\kappa \approx (A_{\perp}/\omega_I)^2$"
    )

    # put the box on panel 2 (ax1)
    ax1.text(
        0.97,
        0.8,
        kappa_text,
        transform=ax1.transAxes,  # <--- changed from ax0 to ax1
        ha="right",
        va="top",
        fontsize=13,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=6),
    )

    return fig, (ax0, ax1)


# ------------------------------ Example --------------------------------------
if __name__ == "__main__":
    kpl.init_kplotlib()
    # 0) (Optional) Build catalog once (then reuse JSON/CSV)
    build_essem_catalog_with_kappa(
        hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
        B_lab_vec=B_vec_T,
        orientations=((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)),
        distance_max_A=22.0,
        gamma_n_Hz_per_T=GAMMA_C13_HZ_PER_T,
        p_occ=0.011,  # 13C abundance
        ms=-1,
        phi_deg=0.0,
        out_json="essem_freq_kappa_catalog_22A_62G.json",
        out_csv="essem_freq_kappa_catalog_22A_62G.csv",
        read_hf_table_fn=None,
    )
    sys.exit()
    recs_all = load_catalog(
        "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_49G.json"
    )

    # Filter by orientation & frequency band (or set orientations=None for all)
    # ori_sel = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    ori_sel = [(1, 1, 1)]
    recs = select_records(
        recs_all,
        fmin_kHz=1,
        fmax_kHz=20000,
        orientations=ori_sel,
        use_plus_and_minus=True,
    )

    # fig, axes = plot_theoretical_kappa_vs_distance(
    #     catalog_json="analysis/spin_echo_work/essem_freq_kappa_catalog_22A_updated.json",
    #     orientations=[(1,1,1)],   # or None for all
    #     # orientations=None,   # or None for all
    #     distance_max_A=15.0,                 # match your catalog cutoff
    # )
    # plt.show(block=True)
    # sys.exit()
    # kpl.show(block=True)

    # A) Discrete sticks using *per-line* weights (p_occ·κ/4 stored per f±)
    freqs_kHz, weights, meta = lines_from_recs(
        recs,
        orientations=ori_sel,  # or None
        fmin_kHz=1.0,
        fmax_kHz=10000.0,
        weight_mode="kappa",  # or "per_line", "unit"
    )

    plot_sticks_kappa(
        freqs_kHz,
        weights,
        meta=meta,  # <-- IMPORTANT
        title=f"ESEEM discrete lines {ori_sel}",
        weight_caption="κ (dimensionless)",
        min_weight=0.0,
    )
    nu_I_kHz = 53.3
    # # B) Smooth spectrum with explicit kernel/weight labels
    f, spec = convolved_expected_from_recs(
        recs,
        orientations=ori_sel,  # or None
        f_range_kHz=(1, 10000),
        npts=2000,
        shape="gauss",
        width_kHz=8.0,
        weight_mode="kappa",
        larmor_kHz=nu_I_kHz,
    )

    plt.show(block=True)
    sys.exit()
    # 1) Load catalog you already built (has f± and kappa)
    # 1) Load full catalog
    recs_all = load_catalog("analysis/spin_echo_work/essem_freq_kappa_catalog_22A.json")

    # 2) Optionally restrict to some orientations and band
    ori_sel = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
    recs = select_records(
        recs_all,
        fmin_kHz=50.0,
        fmax_kHz=20000.0,
        orientations=ori_sel,
        use_plus_and_minus=True,
    )

    # 3) Extract f_- and f_+ separately, in kHz
    f_minus_kHz = np.array([float(r["f_minus_Hz"]) * 1e-3 for r in recs])
    f_plus_kHz = np.array([float(r["f_plus_Hz"]) * 1e-3 for r in recs])

    # (optional) also grab site indices / orientations if you want to track them
    site_idx = np.array([int(r["site_index"]) for r in recs])
    oris = np.array([tuple(r["orientation"]) for r in recs])
    kappa = np.array([float(r.get("kappa", 0.0)) for r in recs])

    # fig_th, ax_th = plot_sorted_branches(
    #     f_minus_kHz,
    #     f_plus_kHz,
    #     sf0_kHz=None,
    #     sf1_kHz=None,
    #     title_prefix="Theoretical ESEEM catalog",
    #     label0=r"Theory $f_-$",
    #     label1=r"Theory $f_+$",
    #     f_range_kHz=(10, 10000),
    # )

    # 3) Get expected sticks including pairs (tunable pair_scale)
    fk, w, tags = expected_sticks_with_multiplicity_pair(
        recs,
        orientations=None,  # already filtered above
        fmin_kHz=1,
        fmax_kHz=20000,
        use_first_order=True,  # keep dominant lines
        first_weight_mode="kappa",  # or "per_line" to use p_occ*κ/4 you stored
        per_line_scale=1.0,
        use_pairs=True,  # add |fi±fj| lines
        pair_scale=1.0,  # start at 1.0; decrease if pairs look too strong
        p_occ=0.011,
        top_k_by_kappa=600,  # trims O(N^2)
    )
    # # # If you want to convolve the (fk, w) that already include pairs, use your 'experimental' convolver:
    # convolved_exp_spectrum(
    #     fk,
    #     w,
    #     f_range_kHz=(1, 20000),
    #     npts=2400,
    #     shape="gauss",
    #     width_kHz=8.0,
    #     title_prefix="Expected ESEEM with multiplicity",
    #     weight_caption="first + pairs",
    # )
    # Build multiplicity sticks
    # fk, w, tags = expected_sticks_general(
    #     recs,
    #     orientations=None,
    #     fmin_kHz=1,
    #     fmax_kHz=20000,
    #     include_order1=True,
    #     first_weight_mode="kappa",  # or "per_line" if you stored p_occ*κ/4
    #     scale_order1=1.0,
    #     p_occ=0.011,
    #     include_order2=True,
    #     branch_mode="all_pairs_strict",  # <-- legacy behavior
    #     pair_scale=1.0,
    #     include_order3=False,  # <-- OFF to match pair-only
    #     top_k_by_kappa=600,
    #     dedup_tol_kHz=0.75,
    # )

    # Convolve to a smooth curve (use your existing util)
    # _f, _S, fig, ax = convolved_exp_spectrum(
    #     fk,
    #     w,
    #     f_range_kHz=(1, 20000),
    #     npts=2400,
    #     shape="gauss",
    #     width_kHz=8.0,
    #     title_prefix="Expected ESEEM (1st + pairs + triples)",
    #     weight_caption="p_occ-scaled κ^m / 4^m",
    #     log_x=True,
    # )
    plt.show()

    # 4) Plot sticks & convolved spectrum (you already have plotting helpers)
    # fk_mc, w_mc = mc_sticks_with_multiplicity(
    #     recs, orientations=None, p_occ=0.011, trials=1000
    # )
    # convolved_exp_spectrum(
    #     fk_mc,
    #     w_mc,
    #     f_range_kHz=(1, 20000),
    #     npts=2400,
    #     shape="gauss",
    #     width_kHz=8.0,
    #     title_prefix="MC Expected ESEEM",
    #     weight_caption="MC (first + pairs)",
    # )
