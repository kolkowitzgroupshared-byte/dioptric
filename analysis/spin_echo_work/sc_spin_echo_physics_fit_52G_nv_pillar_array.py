# -*- coding: utf-8 -*-
"""
Physics-informed wide-field spin-echo fitting pipeline.

Designed for the new QNami spin-echo scans:
- runs directly from ONE raw Dioptric file
- preserves the original widefield.process_counts() normalization
- uses the measured B-field vector to constrain the 13C revival period
- fits a compact collapse/revival envelope first
- optionally tests physically allowed single-13C ESEEM frequency pairs
  from the existing hyperfine table
- uses AICc to decide whether adding ESEEM structure is justified
- parallelizes across NVs
- saves CSV + summary PNG/PDF + full-trace and first-revival PDFs

Internal fit coordinate:
    tau = half the total Hahn-echo evolution time

Plots:
    total evolution time = 2*tau

B field:
    all-negative branch from the ODMR reconstruction:
    (-48.551229, -18.748242, -5.973533) G

@author: Saroj Chand
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math
import os
import json
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from scipy.signal import lombscargle, find_peaks
from threadpoolctl import threadpool_limits

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield


# =============================================================================
# USER SETTINGS
# =============================================================================

FILE_STEM = "2026_09_21-15_37_34-qnami_spin_echo_combined_52G"

APPLY_THRESHOLD = True

# -------------------------------------------------------------------------
# Magnetic field
# -------------------------------------------------------------------------

# ODMR solution, explicitly using the all-negative sign branch.
B_VECTOR_G = np.array(
    [-48.551229, -18.748242, -5.973533],
    dtype=float,
)

GAMMA_C13_KHZ_PER_G = 1.0705

# Allow the measured revival period to move slightly around the B-field prior.
REVIVAL_RELATIVE_TOL = 0.06

# -------------------------------------------------------------------------
# Core collapse/revival model
# -------------------------------------------------------------------------

T2_US_BOUNDS = (10.0, 5000.0)
T2_EXP_BOUNDS = (0.5, 4.0)
WIDTH_US_BOUNDS = (0.4, 10.0)
TAPER_BOUNDS = (0.0, 3.0)
BASELINE_BOUNDS = (0.0, 1.5)
CONTRAST_BOUNDS = (-1.2, 1.2)

CORE_MULTISTART = 8
ROBUST_LOSS = "soft_l1"
ERR_FLOOR = 1e-3

# -------------------------------------------------------------------------
# ESEEM / hyperfine-informed stage
# -------------------------------------------------------------------------

USE_ESEEM = True
USE_HYPERFINE_CATALOG = True

# Existing table used by the old spin-echo work.
HYPERFINE_PATH = r"analysis\nv_hyperfine_coupling\nv-2.txt"

# Reuse the exact-kappa catalog from the established spin_echo_work workflow.
# If it does not exist, this fitter can build it automatically.
CATALOG_JSON = Path(
    r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json"
)
CATALOG_CSV = Path(
    r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.csv"
)
AUTO_BUILD_CATALOG = True
P_C13 = 0.011
CATALOG_PHI_DEG = 0.0
CATALOG_MS = -1

# Maximum hyperfine-site radius included in the candidate catalog.
DISTANCE_MAX_A = 22.0

# Candidate frequency range. Actual upper bound is also clipped by the
# experimental Nyquist limit inferred from the dense tau spacing.
ESEEM_FREQ_RANGE_KHZ = (5.0, 3000.0)

# Keep only the strongest physically allowed sites per orientation before
# testing them against each NV. This is the main speed control.
MAX_CATALOG_SITES_PER_ORIENTATION = 180

# Accept the ESEEM-augmented model only if it improves AICc by this amount.
MIN_DELTA_AICC = 6.0

# Bound amplitudes of the four linear quadratures in the final joint refinement.
ESEEM_QUAD_BOUND = 0.5

# If catalog construction fails, use residual-spectrum peaks as a fallback.
ALLOW_SPECTRAL_FALLBACK = True
NUM_FALLBACK_PEAKS = 6

# -------------------------------------------------------------------------
# Parallelism / output
# -------------------------------------------------------------------------

# CPU acceleration.
# Purcell currently reports 14 physical / 20 logical CPU cores.
# For small independent nonlinear fits, one process per physical core is a
# good default; BLAS is limited to one thread inside each worker to prevent
# oversubscription.
CPU_COUNT = os.cpu_count() or 4
N_JOBS = min(14, max(1, CPU_COUNT - 2))
JOBLIB_BACKEND = "loky"

# GPU note:
# The heavy nonlinear optimization is SciPy least_squares and is CPU-based.
# Candidate ESEEM screening below is vectorized in NumPy, which is typically
# faster than launching many tiny GPU kernels for only ~94 time points.
USE_GPU_FOR_SCREENING = False

# None -> fit all NVs
# Example: [0, 1, 2, 20, 75]
NV_INDICES = None

PDF_COLS = 3
PDF_ROWS = 4

SAVE_CSV = True
SAVE_RESULTS = True
SAVE_CHECKPOINT_NPZ = True
SAVE_SUMMARY_PNG = True
SAVE_SUMMARY_PDF = True
SAVE_FULL_FIT_PDF = True
SAVE_FIRST_REVIVAL_PDF = True

SHOW_SUMMARY = True

OUTPUT_BASENAME = "spin_echo_physics_fit_52G"


# =============================================================================
# PHYSICAL CONSTANTS / DERIVED PRIORS
# =============================================================================

B_MAG_G = float(np.linalg.norm(B_VECTOR_G))
C13_LARMOR_KHZ = GAMMA_C13_KHZ_PER_G * B_MAG_G

# In this experiment tau is half of total evolution time.
# The revival is expected at tau ~= 1/f_C13.
REVIVAL_TAU_US_THEORY = 1000.0 / C13_LARMOR_KHZ
REVIVAL_TOTAL_US_THEORY = 2.0 * REVIVAL_TAU_US_THEORY


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CatalogRecord:
    orientation: tuple[int, int, int]
    site_index: int
    distance_A: float
    f_minus_kHz: float
    f_plus_kHz: float
    kappa: float
    amp_weight: float


@dataclass
class FitResult:
    nv_index: int
    status: str

    red_chi2: float
    aicc: float

    baseline: float
    contrast: float
    revival_tau_us: float
    fitted_B_G: float
    width_us: float
    T2_us: float
    T2_exp: float
    taper_alpha: float

    eseem_used: bool
    delta_aicc: float

    site_index: int
    orientation: tuple | None
    distance_A: float
    kappa: float

    f_minus_kHz: float
    f_plus_kHz: float

    amp_minus: float
    phase_minus_rad: float
    amp_plus: float
    phase_plus_rad: float

    fit_curve: np.ndarray | None = None
    core_curve: np.ndarray | None = None


# =============================================================================
# BASIC HELPERS
# =============================================================================

def safe_sigma(arr):
    arr = np.abs(np.asarray(arr, dtype=float))

    good = np.isfinite(arr) & (arr > 0)

    if np.any(good):
        fallback = float(np.nanmedian(arr[good]))
    else:
        fallback = ERR_FLOOR

    arr = np.where(good, arr, fallback)
    arr = np.maximum(arr, ERR_FLOOR)

    return arr


def resolve_nv_indices(num_nvs):
    if NV_INDICES is None:
        return np.arange(num_nvs, dtype=int)

    inds = np.asarray(NV_INDICES, dtype=int)
    inds = inds[(inds >= 0) & (inds < num_nvs)]

    return np.unique(inds)


def get_output_base():
    timestamp = dm.get_time_stamp()

    base = Path(
        dm.get_file_path(
            __file__,
            timestamp,
            OUTPUT_BASENAME,
        )
    ).with_suffix("")

    base.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return base


def calc_fit_stats(y, yerr, yfit, npar):
    y = np.asarray(y, dtype=float)
    yerr = safe_sigma(yerr)
    yfit = np.asarray(yfit, dtype=float)

    chi2 = float(
        np.sum(
            ((y - yfit) / yerr) ** 2
        )
    )

    dof = max(1, len(y) - int(npar))
    red = chi2 / dof

    k = int(npar)
    n = int(len(y))

    aic = chi2 + 2.0 * k

    if n > k + 1:
        aicc = (
            aic
            + 2.0 * k * (k + 1)
            / (n - k - 1)
        )
    else:
        aicc = np.inf

    return chi2, red, float(aicc)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_single_file(file_stem):
    """
    Load one raw or already-processed spin-echo file.

    If norm_counts are already stored, use them directly.
    Otherwise use the original widefield.process_counts() normalization with
    the rep axis untouched.
    """
    print("=" * 78)
    print("PHYSICS-INFORMED SPIN-ECHO FITTING")
    print("=" * 78)
    print(f"Loading: {file_stem}")

    data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
    )

    nv_list = data["nv_list"]

    if "taus" in data:
        taus_ns = np.asarray(
            data["taus"],
            dtype=float,
        ).ravel()

        tau_us = taus_ns / 1e3

    elif "total_evolution_times" in data:
        total_us = np.asarray(
            data["total_evolution_times"],
            dtype=float,
        ).ravel()

        tau_us = total_us / 2.0

    else:
        raise KeyError(
            "Data must contain either 'taus' or "
            "'total_evolution_times'."
        )

    if (
        "norm_counts" in data
        and "norm_counts_ste" in data
    ):
        print(
            "Using stored norm_counts / norm_counts_ste."
        )

        norm_counts = np.asarray(
            data["norm_counts"],
            dtype=float,
        )

        norm_counts_ste = np.asarray(
            data["norm_counts_ste"],
            dtype=float,
        )

    else:
        print(
            "Using original widefield.process_counts() "
            "normalization."
        )

        counts = np.asarray(
            data["counts"]
        )

        print(
            f"Raw counts shape: {counts.shape}"
        )

        sig_counts = np.asarray(
            counts[0],
            dtype=np.float32,
        )

        ref_counts = np.asarray(
            counts[1],
            dtype=np.float32,
        )

        norm_counts, norm_counts_ste = (
            widefield.process_counts(
                nv_list,
                sig_counts,
                ref_counts,
                threshold=APPLY_THRESHOLD,
            )
        )

        norm_counts = np.asarray(
            norm_counts,
            dtype=float,
        )

        norm_counts_ste = np.asarray(
            norm_counts_ste,
            dtype=float,
        )

    norm_counts_ste = safe_sigma(
        norm_counts_ste
    )

    if (
        norm_counts.shape
        != norm_counts_ste.shape
    ):
        raise ValueError(
            "norm_counts and norm_counts_ste "
            "shape mismatch."
        )

    if (
        norm_counts.shape[0]
        != len(nv_list)
    ):
        raise ValueError(
            "NV count does not match "
            "norm_counts rows."
        )

    if (
        norm_counts.shape[1]
        != tau_us.size
    ):
        raise ValueError(
            "Tau length does not match "
            "norm_counts columns."
        )

    # Sort by tau because the fit assumes ordered x for some diagnostics.
    order = np.argsort(tau_us)

    tau_us = tau_us[order]
    norm_counts = norm_counts[:, order]
    norm_counts_ste = norm_counts_ste[:, order]

    total_evolution_us = 2.0 * tau_us

    orientations = extract_nv_orientations(
        data,
        nv_list,
    )

    print()
    print(f"NVs:                  {len(nv_list)}")
    print(f"Tau points:           {len(tau_us)}")
    print(f"|B|:                  {B_MAG_G:.4f} G")
    print(
        f"B vector:             "
        f"{B_VECTOR_G.tolist()} G"
    )
    print(
        f"13C Larmor frequency: "
        f"{C13_LARMOR_KHZ:.4f} kHz"
    )
    print(
        f"Expected revival tau: "
        f"{REVIVAL_TAU_US_THEORY:.4f} us"
    )
    print(
        f"Expected total time:  "
        f"{REVIVAL_TOTAL_US_THEORY:.4f} us"
    )

    return (
        data,
        nv_list,
        tau_us,
        total_evolution_us,
        norm_counts,
        norm_counts_ste,
        orientations,
    )


def extract_nv_orientations(data, nv_list):
    """
    Return shape (N_NV, 3) integer orientations when available.

    Missing orientation -> row [0,0,0], which means "search all orientations".
    """
    num_nvs = len(nv_list)

    out = np.zeros(
        (num_nvs, 3),
        dtype=int,
    )

    raw = data.get(
        "orientations",
        None,
    )

    if raw is not None:
        arr = np.asarray(raw)

        if (
            arr.ndim == 2
            and arr.shape[0] >= num_nvs
            and arr.shape[1] == 3
        ):
            return arr[:num_nvs].astype(int)

    for i, nv in enumerate(nv_list):
        ori = None

        if hasattr(nv, "orientation"):
            ori = getattr(
                nv,
                "orientation",
            )

        elif isinstance(nv, dict):
            ori = nv.get(
                "orientation",
                None,
            )

        if ori is None:
            continue

        try:
            arr = np.asarray(
                ori,
                dtype=int,
            ).ravel()

            if arr.size == 3:
                out[i] = arr

        except Exception:
            pass

    return out


# =============================================================================
# CORE PHYSICS MODEL
# =============================================================================

def revival_comb(
    tau_us,
    revival_tau_us,
    width_us,
    taper_alpha,
):
    """
    Quartic revival comb adapted from the previous spin-echo model.

    Centers:
        tau_k = k * T_rev

    Amplitude:
        1/(1+k)^alpha
    """
    tau_us = np.asarray(
        tau_us,
        dtype=float,
    )

    revival_tau_us = max(
        float(revival_tau_us),
        1e-9,
    )

    width_us = max(
        float(width_us),
        1e-9,
    )

    tau_max = float(
        np.nanmax(tau_us)
    )

    n_rev = (
        int(
            np.ceil(
                tau_max
                / revival_tau_us
            )
        )
        + 2
    )

    comb = np.zeros_like(
        tau_us,
        dtype=float,
    )

    for k in range(n_rev):
        center = (
            k * revival_tau_us
        )

        amp = (
            1.0
            / (1.0 + k)
            ** float(taper_alpha)
        )

        x = (
            (tau_us - center)
            / width_us
        )

        comb += (
            amp
            * np.exp(
                -(x ** 4)
            )
        )

    return comb


def core_model(
    tau_us,
    baseline,
    contrast,
    revival_tau_us,
    width_us,
    T2_us,
    T2_exp,
    taper_alpha,
):
    """
    Compact B-informed collapse/revival model.

    y(tau) =
        baseline
        - contrast
        * exp[-(2*tau/T2)^p]
        * revival_comb(tau)

    T2 is written in TOTAL evolution time, hence 2*tau/T2.
    """
    tau_us = np.asarray(
        tau_us,
        dtype=float,
    )

    T2_us = max(
        float(T2_us),
        1e-9,
    )

    env = np.exp(
        -(
            2.0 * tau_us
            / T2_us
        )
        ** float(T2_exp)
    )

    comb = revival_comb(
        tau_us,
        revival_tau_us,
        width_us,
        taper_alpha,
    )

    return (
        float(baseline)
        - float(contrast)
        * env
        * comb
    )


def core_carrier(
    tau_us,
    revival_tau_us,
    width_us,
    T2_us,
    T2_exp,
    taper_alpha,
):
    """
    Positive envelope x revival-comb carrier used by ESEEM modulation.
    """
    T2_us = max(
        float(T2_us),
        1e-9,
    )

    env = np.exp(
        -(
            2.0 * np.asarray(
                tau_us,
                dtype=float,
            )
            / T2_us
        )
        ** float(T2_exp)
    )

    comb = revival_comb(
        tau_us,
        revival_tau_us,
        width_us,
        taper_alpha,
    )

    return env * comb


def initial_core_guess(
    tau_us,
    y,
):
    tau_us = np.asarray(
        tau_us,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    baseline = float(
        np.nanmedian(y)
    )

    near = (
        np.abs(
            tau_us
            - REVIVAL_TAU_US_THEORY
        )
        <= 0.25
        * REVIVAL_TAU_US_THEORY
    )

    if np.any(near):
        local = float(
            np.nanmedian(
                y[near]
            )
        )

        contrast = (
            baseline
            - local
        )
    else:
        contrast = (
            np.nanpercentile(
                y,
                80,
            )
            - np.nanpercentile(
                y,
                20,
            )
        )

    contrast = float(
        np.clip(
            contrast,
            -0.7,
            0.7,
        )
    )

    width = min(
        6.0,
        0.35
        * REVIVAL_TAU_US_THEORY,
    )

    T2_us = max(
        3.0
        * REVIVAL_TOTAL_US_THEORY,
        100.0,
    )

    return np.array(
        [
            baseline,
            contrast,
            REVIVAL_TAU_US_THEORY,
            width,
            T2_us,
            2.0,
            0.4,
        ],
        dtype=float,
    )


def core_bounds():
    rev_lo = (
        REVIVAL_TAU_US_THEORY
        * (1.0 - REVIVAL_RELATIVE_TOL)
    )

    rev_hi = (
        REVIVAL_TAU_US_THEORY
        * (1.0 + REVIVAL_RELATIVE_TOL)
    )

    width_hi = min(
        WIDTH_US_BOUNDS[1],
        0.48 * rev_lo,
    )

    lb = np.array(
        [
            BASELINE_BOUNDS[0],
            CONTRAST_BOUNDS[0],
            rev_lo,
            WIDTH_US_BOUNDS[0],
            T2_US_BOUNDS[0],
            T2_EXP_BOUNDS[0],
            TAPER_BOUNDS[0],
        ],
        dtype=float,
    )

    ub = np.array(
        [
            BASELINE_BOUNDS[1],
            CONTRAST_BOUNDS[1],
            rev_hi,
            width_hi,
            T2_US_BOUNDS[1],
            T2_EXP_BOUNDS[1],
            TAPER_BOUNDS[1],
        ],
        dtype=float,
    )

    return lb, ub


def fit_core(
    tau_us,
    y,
    yerr,
):
    tau_us = np.asarray(
        tau_us,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    yerr = safe_sigma(
        yerr
    )

    good = (
        np.isfinite(tau_us)
        & np.isfinite(y)
        & np.isfinite(yerr)
    )

    t = tau_us[good]
    yy = y[good]
    ee = yerr[good]

    if len(t) < 12:
        raise RuntimeError(
            "Too few valid points."
        )

    pbase = initial_core_guess(
        t,
        yy,
    )

    lb, ub = core_bounds()

    rng = np.random.default_rng(
        12345
    )

    starts = [
        pbase.copy()
    ]

    for _ in range(
        max(
            0,
            CORE_MULTISTART - 1,
        )
    ):
        p = pbase.copy()

        p[1] *= rng.uniform(
            0.6,
            1.4,
        )

        p[2] *= rng.uniform(
            0.985,
            1.015,
        )

        p[3] *= rng.uniform(
            0.65,
            1.4,
        )

        p[4] *= rng.uniform(
            0.5,
            2.0,
        )

        p[5] = rng.uniform(
            1.0,
            3.0,
        )

        p[6] = rng.uniform(
            0.0,
            1.2,
        )

        p = np.clip(
            p,
            lb + 1e-8,
            ub - 1e-8,
        )

        starts.append(p)

    best = None

    def residual(p):
        return (
            yy
            - core_model(
                t,
                *p,
            )
        ) / ee

    for p0 in starts:
        try:
            res = least_squares(
                residual,
                x0=p0,
                bounds=(lb, ub),
                loss=ROBUST_LOSS,
                f_scale=1.0,
                max_nfev=25_000,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )

            p = res.x

            yfit = core_model(
                t,
                *p,
            )

            _, red, aicc = (
                calc_fit_stats(
                    yy,
                    ee,
                    yfit,
                    len(p),
                )
            )

            score = (
                aicc,
                red,
            )

            if (
                best is None
                or score < best[0]
            ):
                best = (
                    score,
                    p,
                )

        except Exception:
            continue

    if best is None:
        raise RuntimeError(
            "Core fit failed."
        )

    p = best[1]

    full_curve = core_model(
        tau_us,
        *p,
    )

    _, red, aicc = calc_fit_stats(
        y,
        yerr,
        full_curve,
        len(p),
    )

    return p, full_curve, red, aicc


# =============================================================================
# HYPERFINE / ESEEM PHYSICS
# =============================================================================

Sx = 0.5 * np.array(
    [[0, 1], [1, 0]],
    dtype=complex,
)

Sy = 0.5 * np.array(
    [[0, -1j], [1j, 0]],
    dtype=complex,
)

Sz = 0.5 * np.array(
    [[1, 0], [0, -1]],
    dtype=complex,
)


def build_U_from_orientation(
    orientation,
):
    """
    Same cubic -> NV-frame construction used in the old ESEEM catalog code.
    """
    ez = np.asarray(
        orientation,
        dtype=float,
    )

    ez /= np.linalg.norm(
        ez
    )

    trial = np.array(
        [1.0, -1.0, 0.0],
        dtype=float,
    )

    trial /= np.linalg.norm(
        trial
    )

    if (
        abs(
            np.dot(
                trial,
                ez,
            )
        )
        > 0.95
    ):
        trial = np.array(
            [0.0, 1.0, -1.0],
            dtype=float,
        )

    ex = (
        trial
        - np.dot(
            trial,
            ez,
        )
        * ez
    )

    ex /= np.linalg.norm(
        ex
    )

    ey = np.cross(
        ez,
        ex,
    )

    ey /= np.linalg.norm(
        ey
    )

    U = np.column_stack(
        [ex, ey, ez]
    )

    return U, ez


def eseem_lines_by_diag(
    A_file_Hz,
    orientation,
    B_lab_T,
    ms=-1,
):
    """
    Nuclear Hamiltonian diagonalization copied conceptually from the old
    build_essem_catalog.py workflow.
    """
    U, z_nv = (
        build_U_from_orientation(
            orientation
        )
    )

    A_cubic = (
        U
        @ A_file_Hz
        @ U.T
    )

    B = np.asarray(
        B_lab_T,
        dtype=float,
    )

    Bmag = float(
        np.linalg.norm(B)
    )

    bhat = (
        B / Bmag
    )

    fI_Hz = (
        10.705e6
        * Bmag
    )

    HZ = (
        fI_Hz
        * (
            bhat[0] * Sx
            + bhat[1] * Sy
            + bhat[2] * Sz
        )
    )

    Aeff = (
        A_cubic
        @ z_nv
    )

    Hhf = (
        float(ms)
        * (
            Aeff[0] * Sx
            + Aeff[1] * Sy
            + Aeff[2] * Sz
        )
    )

    e0 = np.linalg.eigvalsh(
        HZ
    )

    ems = np.linalg.eigvalsh(
        HZ + Hhf
    )

    f0 = float(
        abs(
            e0[1] - e0[0]
        )
    )

    fms = float(
        abs(
            ems[1] - ems[0]
        )
    )

    f_minus = abs(
        fms - f0
    )

    f_plus = (
        fms + f0
    )

    Bhat = bhat

    A_par = float(
        np.real(
            Bhat
            @ A_cubic
            @ Bhat
        )
    )

    A_perp_vec = (
        A_cubic
        @ Bhat
        - A_par * Bhat
    )

    A_perp = float(
        np.linalg.norm(
            A_perp_vec
        )
    )

    cos_th = float(
        np.clip(
            Bhat
            @ (
                z_nv
                / np.linalg.norm(z_nv)
            ),
            -1,
            1,
        )
    )

    sin2_th = (
        1.0 - cos_th**2
    )

    amp_weight = (
        (
            A_perp
            / max(
                fms,
                1e-30,
            )
        )
        ** 2
        * sin2_th
    )

    return (
        f_minus,
        f_plus,
        amp_weight,
    )


def build_exact_kappa_catalog_files():
    """
    Build the 52 G catalog using the established exact-kappa implementation
    from kappa_modulation_depth.py.
    """
    from analysis.spin_echo_work.kappa_modulation_depth import (
        build_essem_catalog_with_kappa,
    )

    hyperfine_path = Path(HYPERFINE_PATH)

    if not hyperfine_path.exists():
        raise FileNotFoundError(
            f"Hyperfine table not found: {hyperfine_path}"
        )

    CATALOG_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print("BUILDING EXACT-KAPPA 52 G ESEEM CATALOG")
    print("=" * 78)
    print(f"B vector: {B_VECTOR_G.tolist()} G")
    print(f"|B|:      {B_MAG_G:.6f} G")
    print(f"JSON:     {CATALOG_JSON}")
    print(f"CSV:      {CATALOG_CSV}")

    records = build_essem_catalog_with_kappa(
        hyperfine_path=str(hyperfine_path),
        B_lab_vec=B_VECTOR_G * 1e-4,
        orientations=(
            (1, 1, 1),
            (1, 1, -1),
            (1, -1, 1),
            (-1, 1, 1),
        ),
        distance_max_A=DISTANCE_MAX_A,
        gamma_n_Hz_per_T=GAMMA_C13_KHZ_PER_G * 1e7,
        p_occ=P_C13,
        ms=CATALOG_MS,
        phi_deg=CATALOG_PHI_DEG,
        out_json=str(CATALOG_JSON),
        out_csv=str(CATALOG_CSV),
        read_hf_table_fn=None,
    )

    print(
        f"Built {len(records)} exact-kappa records."
    )

    return records


def load_exact_kappa_catalog():
    """
    Load the established JSON catalog. Auto-build it once when requested.
    """
    if not CATALOG_JSON.exists():
        if not AUTO_BUILD_CATALOG:
            raise FileNotFoundError(
                f"ESEEM catalog not found: {CATALOG_JSON}"
            )

        build_exact_kappa_catalog_files()

    with open(
        CATALOG_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        raw_records = json.load(f)

    if isinstance(raw_records, dict):
        raw_records = raw_records.get(
            "records",
            raw_records,
        )

    records = []

    for r in raw_records:
        try:
            rec = CatalogRecord(
                orientation=tuple(
                    int(x)
                    for x in r["orientation"]
                ),
                site_index=int(
                    r["site_index"]
                ),
                distance_A=float(
                    r["distance_A"]
                ),
                f_minus_kHz=float(
                    r["f_minus_Hz"]
                ) / 1e3,
                f_plus_kHz=float(
                    r["f_plus_Hz"]
                ) / 1e3,
                kappa=float(
                    r.get(
                        "kappa",
                        np.nan,
                    )
                ),
                # Use exact kappa as the physical ranking score.
                amp_weight=float(
                    r.get(
                        "kappa",
                        r.get(
                            "amp_weight",
                            0.0,
                        ),
                    )
                ),
            )

            if (
                np.isfinite(rec.f_minus_kHz)
                and np.isfinite(rec.f_plus_kHz)
            ):
                records.append(rec)

        except Exception:
            continue

    if not records:
        raise RuntimeError(
            f"No valid records loaded from {CATALOG_JSON}"
        )

    # Preserve strongest exact-kappa sites per orientation, as in the old
    # workflow where catalog strength was used to control the search budget.
    selected = []

    for ori in (
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
    ):
        local = [
            r
            for r in records
            if r.orientation == ori
        ]

        local.sort(
            key=lambda r: (
                -np.nan_to_num(
                    r.kappa,
                    nan=-np.inf,
                ),
                r.site_index,
            )
        )

        selected.extend(
            local[
                :MAX_CATALOG_SITES_PER_ORIENTATION
            ]
        )

    print()
    print("=" * 78)
    print("ESEEM CATALOG")
    print("=" * 78)
    print(f"Catalog file:        {CATALOG_JSON}")
    print(f"All valid records:   {len(records)}")
    print(f"Selected for search: {len(selected)}")
    print(
        "Selection: strongest exact-kappa sites "
        f"(max {MAX_CATALOG_SITES_PER_ORIENTATION}/orientation)"
    )
    print("=" * 78)

    return selected


def build_catalog():
    """
    Compatibility wrapper used by main().
    """
    if not USE_HYPERFINE_CATALOG:
        return []

    return load_exact_kappa_catalog()


# =============================================================================
# SPECTRAL FALLBACK
# =============================================================================

def infer_nyquist_kHz(
    tau_us,
):
    unique_t = np.unique(
        np.asarray(
            tau_us,
            dtype=float,
        )
    )

    if unique_t.size < 2:
        return np.inf

    dt = np.diff(
        unique_t
    )

    dt = dt[
        dt > 0
    ]

    if dt.size == 0:
        return np.inf

    min_dt_us = float(
        np.min(dt)
    )

    # cycles/us -> MHz -> kHz
    return (
        0.5
        / min_dt_us
        * 1000.0
    )


def residual_spectral_peaks(
    tau_us,
    residual,
    revival_tau_us,
    width_us,
):
    """
    Lomb-Scargle peak finder on the first dense revival region.

    Used only as a fallback / diagnostic; the preferred ESEEM frequencies are
    the Hamiltonian-generated catalog frequencies.
    """
    tau_us = np.asarray(
        tau_us,
        dtype=float,
    )

    residual = np.asarray(
        residual,
        dtype=float,
    )

    mask = (
        np.abs(
            tau_us
            - revival_tau_us
        )
        <= max(
            1.25 * width_us,
            0.30
            * revival_tau_us,
        )
    )

    t = tau_us[mask]
    r = residual[mask]

    if len(t) < 12:
        t = tau_us
        r = residual

    r = (
        r
        - np.nanmean(r)
    )

    fmin = max(
        1.0,
        ESEEM_FREQ_RANGE_KHZ[0],
    )

    fmax = min(
        ESEEM_FREQ_RANGE_KHZ[1],
        0.95
        * infer_nyquist_kHz(t),
    )

    if (
        not np.isfinite(fmax)
        or fmax <= fmin
    ):
        return []

    freq_kHz = np.linspace(
        fmin,
        fmax,
        1800,
    )

    # omega in radians per microsecond:
    # f_kHz / 1000 = cycles/us.
    omega = (
        2.0
        * np.pi
        * freq_kHz
        / 1000.0
    )

    try:
        power = lombscargle(
            t,
            r,
            omega,
            normalize=True,
            precenter=True,
        )
    except Exception:
        return []

    peaks, _ = find_peaks(
        power
    )

    if peaks.size == 0:
        return []

    order = peaks[
        np.argsort(
            power[peaks]
        )[::-1]
    ]

    selected = []

    min_sep_kHz = max(
        8.0,
        1.0
        / max(
            np.ptp(t),
            1e-6,
        )
        * 1000.0
        * 0.4,
    )

    for idx in order:
        f = float(
            freq_kHz[idx]
        )

        if all(
            abs(
                f - old
            )
            >= min_sep_kHz
            for old in selected
        ):
            selected.append(f)

        if (
            len(selected)
            >= NUM_FALLBACK_PEAKS
        ):
            break

    return selected


# =============================================================================
# ESEEM CANDIDATE FITTING
# =============================================================================

def orientation_matches(
    rec,
    nv_orientation,
):
    ori = np.asarray(
        nv_orientation,
        dtype=int,
    ).ravel()

    if (
        ori.size != 3
        or not np.any(ori)
    ):
        return True

    return (
        tuple(
            int(x)
            for x in ori
        )
        == rec.orientation
    )


def filter_catalog_for_trace(
    catalog,
    nv_orientation,
    tau_us,
):
    nyq = infer_nyquist_kHz(
        tau_us
    )

    lo = float(
        ESEEM_FREQ_RANGE_KHZ[0]
    )

    hi = min(
        float(
            ESEEM_FREQ_RANGE_KHZ[1]
        ),
        0.95 * nyq,
    )

    out = []

    for rec in catalog:
        if not orientation_matches(
            rec,
            nv_orientation,
        ):
            continue

        fm = rec.f_minus_kHz
        fp = rec.f_plus_kHz

        if not (
            lo <= fm <= hi
            and lo <= fp <= hi
        ):
            continue

        if (
            abs(fp - fm)
            < 1e-3
        ):
            continue

        out.append(rec)

    return out


def design_eseem_matrix(
    tau_us,
    carrier,
    f_minus_kHz,
    f_plus_kHz,
):
    t = np.asarray(
        tau_us,
        dtype=float,
    )

    c = np.asarray(
        carrier,
        dtype=float,
    )

    f0 = (
        float(f_minus_kHz)
        / 1000.0
    )

    f1 = (
        float(f_plus_kHz)
        / 1000.0
    )

    return np.column_stack(
        [
            c
            * np.cos(
                2.0 * np.pi * f0 * t
            ),
            c
            * np.sin(
                2.0 * np.pi * f0 * t
            ),
            c
            * np.cos(
                2.0 * np.pi * f1 * t
            ),
            c
            * np.sin(
                2.0 * np.pi * f1 * t
            ),
        ]
    )


def linear_eseem_fit(
    tau_us,
    y,
    yerr,
    core_curve,
    core_params,
    f_minus_kHz,
    f_plus_kHz,
):
    (
        baseline,
        contrast,
        revival_tau_us,
        width_us,
        T2_us,
        T2_exp,
        taper_alpha,
    ) = core_params

    carrier = core_carrier(
        tau_us,
        revival_tau_us,
        width_us,
        T2_us,
        T2_exp,
        taper_alpha,
    )

    X = design_eseem_matrix(
        tau_us,
        carrier,
        f_minus_kHz,
        f_plus_kHz,
    )

    ee = safe_sigma(
        yerr
    )

    target = (
        np.asarray(
            y,
            dtype=float,
        )
        - np.asarray(
            core_curve,
            dtype=float,
        )
    )

    Xw = (
        X / ee[:, None]
    )

    yw = (
        target / ee
    )

    try:
        coeff, *_ = np.linalg.lstsq(
            Xw,
            yw,
            rcond=None,
        )
    except Exception:
        return None

    coeff = np.clip(
        coeff,
        -ESEEM_QUAD_BOUND,
        ESEEM_QUAD_BOUND,
    )

    fit_curve = (
        core_curve
        + X @ coeff
    )

    _, red, aicc = calc_fit_stats(
        y,
        yerr,
        fit_curve,
        7 + 4,
    )

    return (
        coeff,
        fit_curve,
        red,
        aicc,
    )



def batch_screen_catalog(
    tau_us,
    y,
    yerr,
    core_curve,
    core_params,
    catalog_records,
):
    """
    Vectorized screening of all physical (f-, f+) candidates for one NV.

    This replaces hundreds of small np.linalg.lstsq calls with one batched
    normal-equation solve. For ~100–200 candidate sites and ~94 time points,
    this is substantially faster on CPU and avoids GPU launch overhead.

    Returns
    -------
    best_record, best_coeff, best_curve, best_red_chi2, best_aicc
    """
    if not catalog_records:
        return None

    t = np.asarray(tau_us, dtype=float)
    yy = np.asarray(y, dtype=float)
    ee = safe_sigma(yerr)
    core = np.asarray(core_curve, dtype=float)

    (
        _baseline,
        _contrast,
        revival_tau_us,
        width_us,
        T2_us,
        T2_exp,
        taper_alpha,
    ) = core_params

    carrier = core_carrier(
        t,
        revival_tau_us,
        width_us,
        T2_us,
        T2_exp,
        taper_alpha,
    )

    fm = np.asarray(
        [r.f_minus_kHz for r in catalog_records],
        dtype=float,
    )
    fp = np.asarray(
        [r.f_plus_kHz for r in catalog_records],
        dtype=float,
    )

    # shape: (M, N)
    phm = (
        2.0
        * np.pi
        * (fm[:, None] / 1000.0)
        * t[None, :]
    )
    php = (
        2.0
        * np.pi
        * (fp[:, None] / 1000.0)
        * t[None, :]
    )

    c = carrier[None, :]

    # X shape: (M candidates, N time points, 4 quadratures)
    X = np.stack(
        [
            c * np.cos(phm),
            c * np.sin(phm),
            c * np.cos(php),
            c * np.sin(php),
        ],
        axis=2,
    )

    target = yy - core
    w = 1.0 / np.maximum(ee, ERR_FLOOR) ** 2

    # Batched weighted normal equations:
    # A_m = X_m^T W X_m
    # b_m = X_m^T W r
    A = np.einsum(
        "mni,mnj,n->mij",
        X,
        X,
        w,
        optimize=True,
    )
    b = np.einsum(
        "mni,n,n->mi",
        X,
        target,
        w,
        optimize=True,
    )

    # Tiny ridge only for numerical stability when frequencies are nearly
    # degenerate. It is far below the experimental error scale.
    ridge = 1e-10
    A += ridge * np.eye(4)[None, :, :]

    try:
        # For stacked A with shape (M, 4, 4), explicitly give b a trailing
        # singleton RHS dimension. This is compatible across NumPy versions:
        #     (M,4,4) @ (M,4,1) -> (M,4,1)
        coeff = np.linalg.solve(
            A,
            b[..., None],
        )[..., 0]
    except np.linalg.LinAlgError:
        coeff = np.empty((len(catalog_records), 4), dtype=float)
        for i in range(len(catalog_records)):
            try:
                coeff[i] = np.linalg.lstsq(
                    A[i],
                    b[i],
                    rcond=None,
                )[0]
            except Exception:
                coeff[i] = np.nan

    coeff = np.clip(
        coeff,
        -ESEEM_QUAD_BOUND,
        ESEEM_QUAD_BOUND,
    )

    modulation = np.einsum(
        "mni,mi->mn",
        X,
        coeff,
        optimize=True,
    )

    resid = (
        target[None, :]
        - modulation
    ) / ee[None, :]

    chi2 = np.sum(
        resid * resid,
        axis=1,
    )

    n = len(t)
    k = 11
    dof = max(1, n - k)
    red = chi2 / dof

    aic = chi2 + 2.0 * k

    if n > k + 1:
        aicc = (
            aic
            + 2.0
            * k
            * (k + 1)
            / (n - k - 1)
        )
    else:
        aicc = np.full_like(
            aic,
            np.inf,
        )

    valid = (
        np.isfinite(aicc)
        & np.all(
            np.isfinite(coeff),
            axis=1,
        )
    )

    if not np.any(valid):
        return None

    masked = np.where(
        valid,
        aicc,
        np.inf,
    )

    best_ind = int(
        np.argmin(masked)
    )

    best_curve = (
        core
        + modulation[best_ind]
    )

    return (
        catalog_records[best_ind],
        coeff[best_ind],
        best_curve,
        float(red[best_ind]),
        float(aicc[best_ind]),
    )

def joint_model(
    tau_us,
    p,
    f_minus_kHz,
    f_plus_kHz,
):
    core_p = p[:7]
    quad = p[7:11]

    core = core_model(
        tau_us,
        *core_p,
    )

    carrier = core_carrier(
        tau_us,
        core_p[2],
        core_p[3],
        core_p[4],
        core_p[5],
        core_p[6],
    )

    X = design_eseem_matrix(
        tau_us,
        carrier,
        f_minus_kHz,
        f_plus_kHz,
    )

    return (
        core
        + X @ quad
    )


def refine_joint_fit(
    tau_us,
    y,
    yerr,
    core_params,
    quad,
    f_minus_kHz,
    f_plus_kHz,
):
    core_lb, core_ub = (
        core_bounds()
    )

    lb = np.concatenate(
        [
            core_lb,
            np.full(
                4,
                -ESEEM_QUAD_BOUND,
            ),
        ]
    )

    ub = np.concatenate(
        [
            core_ub,
            np.full(
                4,
                ESEEM_QUAD_BOUND,
            ),
        ]
    )

    p0 = np.concatenate(
        [
            core_params,
            quad,
        ]
    )

    p0 = np.clip(
        p0,
        lb + 1e-9,
        ub - 1e-9,
    )

    ee = safe_sigma(
        yerr
    )

    def residual(p):
        return (
            np.asarray(
                y,
                dtype=float,
            )
            - joint_model(
                tau_us,
                p,
                f_minus_kHz,
                f_plus_kHz,
            )
        ) / ee

    res = least_squares(
        residual,
        x0=p0,
        bounds=(lb, ub),
        loss=ROBUST_LOSS,
        f_scale=1.0,
        max_nfev=20_000,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    p = res.x

    curve = joint_model(
        tau_us,
        p,
        f_minus_kHz,
        f_plus_kHz,
    )

    _, red, aicc = calc_fit_stats(
        y,
        yerr,
        curve,
        len(p),
    )

    return (
        p,
        curve,
        red,
        aicc,
    )


def quadratures_to_amp_phase(
    c,
    s,
):
    amp = float(
        np.hypot(
            c,
            s,
        )
    )

    # c cos(wt) + s sin(wt)
    # = A cos(wt + phi)
    phi = float(
        np.arctan2(
            -s,
            c,
        )
    )

    return amp, phi


# =============================================================================
# SINGLE-NV FIT
# =============================================================================

def fit_one_nv(
    nv_index,
    tau_us,
    y,
    yerr,
    nv_orientation,
    catalog,
):
    try:
        y = np.asarray(
            y,
            dtype=float,
        )

        yerr = safe_sigma(
            yerr
        )

        core_p, core_curve, core_red, core_aicc = (
            fit_core(
                tau_us,
                y,
                yerr,
            )
        )

        (
            baseline,
            contrast,
            revival_tau_us,
            width_us,
            T2_us,
            T2_exp,
            taper_alpha,
        ) = core_p

        fitted_B_G = (
            1000.0
            / (
                GAMMA_C13_KHZ_PER_G
                * revival_tau_us
            )
        )

        best = None
        eseem_stage_error = None

        try:

            if USE_ESEEM:
                local_catalog = (
                    filter_catalog_for_trace(
                        catalog,
                        nv_orientation,
                        tau_us,
                    )
                    if catalog
                    else []
                )

                # Fast vectorized screening of every physically allowed
                # hyperfine candidate for this NV.
                screened = batch_screen_catalog(
                    tau_us,
                    y,
                    yerr,
                    core_curve,
                    core_p,
                    local_catalog,
                )

                if screened is not None:
                    (
                        rec,
                        quad,
                        curve,
                        red,
                        aicc,
                    ) = screened

                    best = (
                        (aicc, red),
                        rec,
                        quad,
                        curve,
                        red,
                        aicc,
                    )

                # Fallback: use strongest residual spectral peaks.
                if (
                    best is None
                    and ALLOW_SPECTRAL_FALLBACK
                ):
                    peaks = residual_spectral_peaks(
                        tau_us,
                        y - core_curve,
                        revival_tau_us,
                        width_us,
                    )

                    if len(peaks) >= 2:
                        pairs = []

                        for i in range(
                            len(peaks)
                        ):
                            for j in range(
                                i + 1,
                                len(peaks),
                            ):
                                pairs.append(
                                    (
                                        min(
                                            peaks[i],
                                            peaks[j],
                                        ),
                                        max(
                                            peaks[i],
                                            peaks[j],
                                        ),
                                    )
                                )

                        for fm, fp in pairs:
                            out = linear_eseem_fit(
                                tau_us,
                                y,
                                yerr,
                                core_curve,
                                core_p,
                                fm,
                                fp,
                            )

                            if out is None:
                                continue

                            quad, curve, red, aicc = out

                            rec = CatalogRecord(
                                orientation=(0, 0, 0),
                                site_index=-1,
                                distance_A=np.nan,
                                f_minus_kHz=float(fm),
                                f_plus_kHz=float(fp),
                                kappa=np.nan,
                                amp_weight=np.nan,
                            )

                            score = (
                                aicc,
                                red,
                            )

                            if (
                                best is None
                                or score < best[0]
                            ):
                                best = (
                                    score,
                                    rec,
                                    quad,
                                    curve,
                                    red,
                                    aicc,
                                )

        except Exception as exc:
            # The core collapse/revival fit is still scientifically useful.
            # Do not throw away the whole NV because the optional ESEEM stage
            # had a catalog/numerical problem.
            eseem_stage_error = str(exc)
            best = None

        use_eseem = False
        delta_aicc = 0.0

        if best is not None:
            (
                _score,
                rec,
                quad,
                _curve,
                _red,
                candidate_aicc,
            ) = best

            delta_aicc = (
                core_aicc
                - candidate_aicc
            )

            if (
                np.isfinite(
                    delta_aicc
                )
                and delta_aicc
                >= MIN_DELTA_AICC
            ):
                try:
                    (
                        joint_p,
                        joint_curve,
                        joint_red,
                        joint_aicc,
                    ) = refine_joint_fit(
                        tau_us,
                        y,
                        yerr,
                        core_p,
                        quad,
                        rec.f_minus_kHz,
                        rec.f_plus_kHz,
                    )

                    refined_delta = (
                        core_aicc
                        - joint_aicc
                    )

                    if (
                        np.isfinite(
                            refined_delta
                        )
                        and refined_delta
                        >= MIN_DELTA_AICC
                    ):
                        use_eseem = True
                        delta_aicc = refined_delta

                        core_p = (
                            joint_p[:7]
                        )

                        quad = (
                            joint_p[7:11]
                        )

                        fit_curve = (
                            joint_curve
                        )

                        red = (
                            joint_red
                        )

                        aicc = (
                            joint_aicc
                        )

                        (
                            baseline,
                            contrast,
                            revival_tau_us,
                            width_us,
                            T2_us,
                            T2_exp,
                            taper_alpha,
                        ) = core_p

                        core_curve = core_model(
                            tau_us,
                            *core_p,
                        )

                        fitted_B_G = (
                            1000.0
                            / (
                                GAMMA_C13_KHZ_PER_G
                                * revival_tau_us
                            )
                        )

                except Exception:
                    pass

        if not use_eseem:
            fit_curve = (
                core_curve
            )

            red = (
                core_red
            )

            aicc = (
                core_aicc
            )

            rec = CatalogRecord(
                orientation=(
                    tuple(
                        int(v)
                        for v in nv_orientation
                    )
                    if np.any(
                        nv_orientation
                    )
                    else (0, 0, 0)
                ),
                site_index=-1,
                distance_A=np.nan,
                f_minus_kHz=np.nan,
                f_plus_kHz=np.nan,
                kappa=np.nan,
                amp_weight=np.nan,
            )

            quad = np.zeros(
                4,
                dtype=float,
            )

            delta_aicc = max(
                0.0,
                float(
                    delta_aicc
                    if np.isfinite(
                        delta_aicc
                    )
                    else 0.0
                ),
            )

        amp_minus, phi_minus = (
            quadratures_to_amp_phase(
                quad[0],
                quad[1],
            )
        )

        amp_plus, phi_plus = (
            quadratures_to_amp_phase(
                quad[2],
                quad[3],
            )
        )

        ori_out = (
            rec.orientation
            if rec.orientation
            != (0, 0, 0)
            else None
        )

        if use_eseem:
            status = "ok_eseem"
        elif eseem_stage_error is not None:
            status = "ok_core_eseem_failed"
        else:
            status = "ok_core"

        return FitResult(
            nv_index=int(
                nv_index
            ),
            status=status,

            red_chi2=float(
                red
            ),
            aicc=float(
                aicc
            ),

            baseline=float(
                baseline
            ),
            contrast=float(
                contrast
            ),
            revival_tau_us=float(
                revival_tau_us
            ),
            fitted_B_G=float(
                fitted_B_G
            ),
            width_us=float(
                width_us
            ),
            T2_us=float(
                T2_us
            ),
            T2_exp=float(
                T2_exp
            ),
            taper_alpha=float(
                taper_alpha
            ),

            eseem_used=bool(
                use_eseem
            ),
            delta_aicc=float(
                delta_aicc
            ),

            site_index=int(
                rec.site_index
            ),
            orientation=ori_out,
            distance_A=float(
                rec.distance_A
            ),
            kappa=float(
                rec.kappa
            ),

            f_minus_kHz=float(
                rec.f_minus_kHz
            ),
            f_plus_kHz=float(
                rec.f_plus_kHz
            ),

            amp_minus=float(
                amp_minus
            ),
            phase_minus_rad=float(
                phi_minus
            ),
            amp_plus=float(
                amp_plus
            ),
            phase_plus_rad=float(
                phi_plus
            ),

            fit_curve=np.asarray(
                fit_curve,
                dtype=float,
            ),
            core_curve=np.asarray(
                core_curve,
                dtype=float,
            ),
        )

    except Exception as exc:
        print(
            f"[WARN] NV {nv_index} fit failed: {exc}"
        )

        return FitResult(
            nv_index=int(
                nv_index
            ),
            status="failed",

            red_chi2=np.nan,
            aicc=np.nan,

            baseline=np.nan,
            contrast=np.nan,
            revival_tau_us=np.nan,
            fitted_B_G=np.nan,
            width_us=np.nan,
            T2_us=np.nan,
            T2_exp=np.nan,
            taper_alpha=np.nan,

            eseem_used=False,
            delta_aicc=np.nan,

            site_index=-1,
            orientation=None,
            distance_A=np.nan,
            kappa=np.nan,

            f_minus_kHz=np.nan,
            f_plus_kHz=np.nan,

            amp_minus=np.nan,
            phase_minus_rad=np.nan,
            amp_plus=np.nan,
            phase_plus_rad=np.nan,

            fit_curve=None,
            core_curve=None,
        )


# =============================================================================
# FIT ALL NVs
# =============================================================================

def fit_dataset(
    nv_list,
    tau_us,
    norm_counts,
    norm_counts_ste,
    orientations,
    catalog,
):
    nv_indices = resolve_nv_indices(
        len(nv_list)
    )

    print()
    print(
        f"Fitting {len(nv_indices)} NVs "
        f"with {N_JOBS} CPU worker processes "
        f"(detected logical CPUs={CPU_COUNT})"
    )

    # Use process-level parallelism across independent NV fits.
    # Limit BLAS/OpenMP to one thread inside each worker, otherwise 14 workers
    # can each spawn many BLAS threads and make the analysis much slower.
    with threadpool_limits(limits=1):
        results = Parallel(
            n_jobs=N_JOBS,
            backend=JOBLIB_BACKEND,
            verbose=5,
            batch_size=1,
        )(
            delayed(
                fit_one_nv
            )(
                int(nv_ind),
                tau_us,
                norm_counts[nv_ind],
                norm_counts_ste[nv_ind],
                orientations[nv_ind],
                catalog,
            )
            for nv_ind in nv_indices
        )

    return results


# =============================================================================
# RESULTS TABLE
# =============================================================================

def results_to_dataframe(
    results,
):
    rows = []

    for r in results:
        row = asdict(r)

        row.pop(
            "fit_curve",
            None,
        )

        row.pop(
            "core_curve",
            None,
        )

        if row["orientation"] is not None:
            row["orientation"] = str(
                tuple(
                    row["orientation"]
                )
            )
        else:
            row["orientation"] = ""

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# =============================================================================
# SUMMARY FIGURE
# =============================================================================

def make_summary_figure(
    tau_us,
    norm_counts,
    norm_counts_ste,
    results,
):
    good_results = [
        r
        for r in results
        if (
            r.fit_curve is not None
            and np.all(
                np.isfinite(
                    r.fit_curve
                )
            )
        )
    ]

    if not good_results:
        raise RuntimeError(
            "No successful fits."
        )

    inds = np.array(
        [
            r.nv_index
            for r in good_results
        ],
        dtype=int,
    )

    data_median = np.nanmedian(
        norm_counts[inds],
        axis=0,
    )

    data_ste = np.nanmedian(
        norm_counts_ste[inds],
        axis=0,
    )

    fit_stack = np.vstack(
        [
            r.fit_curve
            for r in good_results
        ]
    )

    fit_median = np.nanmedian(
        fit_stack,
        axis=0,
    )

    first_mask = (
        np.abs(
            tau_us
            - REVIVAL_TAU_US_THEORY
        )
        <= 6.5
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    ax = axes[0, 0]

    ax.errorbar(
        2.0 * tau_us,
        data_median,
        yerr=data_ste,
        fmt="o",
        markersize=3.5,
        capsize=1.5,
        label="Median data",
    )

    ax.plot(
        2.0 * tau_us,
        fit_median,
        linewidth=1.8,
        label="Median fitted curve",
    )

    ax.axvline(
        REVIVAL_TOTAL_US_THEORY,
        linestyle="--",
        linewidth=1,
        label="52 G 13C revival",
    )

    ax.set_xlabel(
        "Total evolution time (us)"
    )

    ax.set_ylabel(
        r"Normalized NV$^{-}$ population"
    )

    ax.set_title(
        f"Median spin echo — {len(good_results)} fitted NVs"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    ax = axes[0, 1]

    ax.errorbar(
        2.0 * tau_us[first_mask],
        data_median[first_mask],
        yerr=data_ste[first_mask],
        fmt="o",
        markersize=3.5,
        capsize=1.5,
        label="Median data",
    )

    ax.plot(
        2.0 * tau_us[first_mask],
        fit_median[first_mask],
        linewidth=1.8,
        label="Median fit",
    )

    ax.axvline(
        REVIVAL_TOTAL_US_THEORY,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Total evolution time (us)"
    )

    ax.set_ylabel(
        r"Normalized NV$^{-}$ population"
    )

    ax.set_title(
        "Dense first revival"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    Bfits = np.array(
        [
            r.fitted_B_G
            for r in good_results
            if np.isfinite(
                r.fitted_B_G
            )
        ],
        dtype=float,
    )

    ax = axes[1, 0]

    ax.hist(
        Bfits,
        bins=min(
            30,
            max(
                8,
                int(
                    np.sqrt(
                        len(Bfits)
                    )
                ),
            ),
        ),
    )

    ax.axvline(
        B_MAG_G,
        linestyle="--",
        label=(
            f"ODMR |B| = "
            f"{B_MAG_G:.2f} G"
        ),
    )

    ax.set_xlabel(
        "B inferred from revival (G)"
    )

    ax.set_ylabel(
        "NV count"
    )

    ax.set_title(
        "Revival-derived field"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    ax = axes[1, 1]

    core_n = sum(
        not r.eseem_used
        for r in good_results
    )

    eseem_n = sum(
        r.eseem_used
        for r in good_results
    )

    ax.bar(
        ["Core only", "Core + ESEEM"],
        [core_n, eseem_n],
    )

    ax.set_ylabel(
        "NV count"
    )

    ax.set_title(
        f"ESEEM accepted if ΔAICc ≥ {MIN_DELTA_AICC:g}"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.suptitle(
        "Physics-informed spin-echo fit summary\n"
        f"B = {B_VECTOR_G.tolist()} G, "
        f"|B| = {B_MAG_G:.3f} G",
        fontsize=14,
    )

    fig.tight_layout()

    return fig


# =============================================================================
# MULTIPAGE PDF
# =============================================================================

def save_fit_pdf(
    path,
    tau_us,
    norm_counts,
    norm_counts_ste,
    results,
    zoom_first_revival=False,
):
    plots_per_page = (
        PDF_COLS
        * PDF_ROWS
    )

    with PdfPages(path) as pdf:
        for start in range(
            0,
            len(results),
            plots_per_page,
        ):
            page = results[
                start
                : start
                + plots_per_page
            ]

            fig, axes = plt.subplots(
                PDF_ROWS,
                PDF_COLS,
                figsize=(
                    5.0 * PDF_COLS,
                    3.5 * PDF_ROWS,
                ),
                squeeze=False,
            )

            axes = axes.ravel()

            for slot, r in enumerate(
                page
            ):
                ax = axes[slot]

                nv = r.nv_index

                if zoom_first_revival:
                    mask = (
                        np.abs(
                            tau_us
                            - REVIVAL_TAU_US_THEORY
                        )
                        <= 6.5
                    )
                else:
                    mask = np.ones(
                        tau_us.size,
                        dtype=bool,
                    )

                x = (
                    2.0
                    * tau_us[mask]
                )

                ax.errorbar(
                    x,
                    norm_counts[nv, mask],
                    yerr=norm_counts_ste[nv, mask],
                    fmt="o",
                    markersize=2.7,
                    capsize=1.0,
                    linewidth=0.6,
                    label="Data",
                )

                if r.core_curve is not None:
                    ax.plot(
                        x,
                        r.core_curve[mask],
                        linewidth=1.0,
                        linestyle="--",
                        label="Core",
                    )

                if r.fit_curve is not None:
                    ax.plot(
                        x,
                        r.fit_curve[mask],
                        linewidth=1.4,
                        label="Fit",
                    )

                ax.axvline(
                    REVIVAL_TOTAL_US_THEORY,
                    linestyle=":",
                    linewidth=0.8,
                )

                title = (
                    f"NV {nv} | "
                    f"{r.status}"
                )

                if np.isfinite(
                    r.red_chi2
                ):
                    title += (
                        f" | χ²r={r.red_chi2:.2f}"
                    )

                ax.set_title(
                    title,
                    fontsize=8.5,
                )

                ax.set_xlabel(
                    "Total evolution (us)",
                    fontsize=8,
                )

                ax.set_ylabel(
                    r"Norm. NV$^{-}$ pop.",
                    fontsize=8,
                )

                ax.tick_params(
                    labelsize=7,
                )

                ax.grid(
                    alpha=0.22
                )

                ax.legend(
                    fontsize=6,
                )

                if r.eseem_used:
                    text = (
                        f"f-={r.f_minus_kHz:.0f} kHz\n"
                        f"f+={r.f_plus_kHz:.0f} kHz\n"
                        f"ΔAICc={r.delta_aicc:.1f}"
                    )

                    ax.text(
                        0.02,
                        0.03,
                        text,
                        transform=ax.transAxes,
                        fontsize=6,
                        va="bottom",
                    )

            for slot in range(
                len(page),
                len(axes),
            ):
                axes[slot].axis(
                    "off"
                )

            label = (
                "First-revival zoom"
                if zoom_first_revival
                else "Full spin echo"
            )

            fig.suptitle(
                f"{label} — "
                f"{start + 1}–"
                f"{start + len(page)} "
                f"of {len(results)}",
                fontsize=14,
                y=0.995,
            )

            fig.tight_layout(
                rect=[
                    0,
                    0,
                    1,
                    0.975,
                ]
            )

            pdf.savefig(
                fig,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

    print(
        f"Saved: {path}"
    )


def save_fit_checkpoint_npz(
    output_base,
    tau_us,
    norm_counts,
    norm_counts_ste,
    results,
):
    """
    Save everything needed to replot the fit WITHOUT re-running optimization.

    This checkpoint is intentionally written before summary/PDF generation so
    a later plotting error does not lose an expensive fit.
    """
    if not SAVE_CHECKPOINT_NPZ:
        return None

    n_nv = len(results)
    n_t = len(tau_us)

    fit_curves = np.full(
        (n_nv, n_t),
        np.nan,
        dtype=float,
    )
    core_curves = np.full(
        (n_nv, n_t),
        np.nan,
        dtype=float,
    )

    nv_indices = np.empty(
        n_nv,
        dtype=int,
    )
    status = np.empty(
        n_nv,
        dtype="U64",
    )

    for i, r in enumerate(results):
        nv_indices[i] = int(r.nv_index)
        status[i] = str(r.status)

        if r.fit_curve is not None:
            arr = np.asarray(
                r.fit_curve,
                dtype=float,
            ).ravel()
            if arr.size == n_t:
                fit_curves[i] = arr

        if r.core_curve is not None:
            arr = np.asarray(
                r.core_curve,
                dtype=float,
            ).ravel()
            if arr.size == n_t:
                core_curves[i] = arr

    checkpoint_path = Path(
        str(output_base)
        + "_fit_checkpoint.npz"
    )

    np.savez_compressed(
        checkpoint_path,
        source_file_stem=np.asarray(
            [str(FILE_STEM)]
        ),
        B_vector_G=np.asarray(
            B_VECTOR_G,
            dtype=float,
        ),
        B_magnitude_G=np.asarray(
            [B_MAG_G],
            dtype=float,
        ),
        tau_us=np.asarray(
            tau_us,
            dtype=float,
        ),
        total_evolution_us=(
            2.0
            * np.asarray(
                tau_us,
                dtype=float,
            )
        ),
        norm_counts=np.asarray(
            norm_counts,
            dtype=float,
        ),
        norm_counts_ste=np.asarray(
            norm_counts_ste,
            dtype=float,
        ),
        nv_indices=nv_indices,
        status=status,
        fit_curves=fit_curves,
        core_curves=core_curves,
    )

    print(
        f"Saved fit checkpoint: "
        f"{checkpoint_path}"
    )

    return checkpoint_path


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_outputs(
    output_base,
    tau_us,
    norm_counts,
    norm_counts_ste,
    results,
    summary_fig,
):
    df = results_to_dataframe(
        results
    )

    if SAVE_CSV:
        csv_path = Path(
            str(output_base)
            + "_fit_results.csv"
        )

        df.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"Saved: {csv_path}"
        )

    if SAVE_RESULTS:
        serializable = {
            "source_file_stem":
                FILE_STEM,

            "B_vector_G":
                B_VECTOR_G.tolist(),

            "B_magnitude_G":
                B_MAG_G,

            "c13_larmor_kHz":
                C13_LARMOR_KHZ,

            "revival_tau_us_theory":
                REVIVAL_TAU_US_THEORY,

            "revival_total_us_theory":
                REVIVAL_TOTAL_US_THEORY,

            "tau_us":
                tau_us.tolist(),

            "fit_results":
                df.to_dict(
                    orient="records"
                ),
        }

        dm.save_raw_data(
            serializable,
            output_base,
        )

        print(
            f"Saved analysis data: "
            f"{output_base}"
        )

    if SAVE_SUMMARY_PNG:
        path = Path(
            str(output_base)
            + "_summary.png"
        )

        summary_fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved: {path}"
        )

    if SAVE_SUMMARY_PDF:
        path = Path(
            str(output_base)
            + "_summary.pdf"
        )

        summary_fig.savefig(
            path,
            bbox_inches="tight",
        )

        print(
            f"Saved: {path}"
        )

    if SAVE_FULL_FIT_PDF:
        save_fit_pdf(
            Path(
                str(output_base)
                + "_all_nv_full_fits.pdf"
            ),
            tau_us,
            norm_counts,
            norm_counts_ste,
            results,
            zoom_first_revival=False,
        )

    if SAVE_FIRST_REVIVAL_PDF:
        save_fit_pdf(
            Path(
                str(output_base)
                + "_all_nv_first_revival_fits.pdf"
            ),
            tau_us,
            norm_counts,
            norm_counts_ste,
            results,
            zoom_first_revival=True,
        )

    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    kpl.init_kplotlib()

    (
        data,
        nv_list,
        tau_us,
        total_evolution_us,
        norm_counts,
        norm_counts_ste,
        orientations,
    ) = load_single_file(
        FILE_STEM
    )

    print()
    print(
        "Building physics priors..."
    )

    catalog = build_catalog()

    if (
        USE_ESEEM
        and USE_HYPERFINE_CATALOG
        and not catalog
    ):
        print(
            "No hyperfine catalog available; "
            "spectral fallback will be used."
        )

    results = fit_dataset(
        nv_list,
        tau_us,
        norm_counts,
        norm_counts_ste,
        orientations,
        catalog,
    )

    successful = [
        r
        for r in results
        if r.status != "failed"
    ]

    eseem_count = sum(
        r.eseem_used
        for r in successful
    )

    print()
    print("=" * 78)
    print("FIT SUMMARY")
    print("=" * 78)
    print(
        f"Successful fits: "
        f"{len(successful)} / "
        f"{len(results)}"
    )
    print(
        f"ESEEM accepted:  "
        f"{eseem_count}"
    )

    Bvals = np.array(
        [
            r.fitted_B_G
            for r in successful
            if np.isfinite(
                r.fitted_B_G
            )
        ],
        dtype=float,
    )

    if Bvals.size:
        print(
            f"Median B from revival: "
            f"{np.nanmedian(Bvals):.3f} G"
        )

    T2vals = np.array(
        [
            r.T2_us
            for r in successful
            if np.isfinite(
                r.T2_us
            )
        ],
        dtype=float,
    )

    if T2vals.size:
        print(
            f"Median T2: "
            f"{np.nanmedian(T2vals):.2f} us"
        )

    print("=" * 78)

    # Create the output name immediately after fitting and save a complete
    # checkpoint BEFORE any plotting. This protects the expensive fit if a
    # later plotting/PDF step raises an exception.
    output_base = (
        get_output_base()
    )

    save_fit_checkpoint_npz(
        output_base,
        tau_us,
        norm_counts,
        norm_counts_ste,
        results,
    )

    summary_fig = (
        make_summary_figure(
            tau_us,
            norm_counts,
            norm_counts_ste,
            results,
        )
    )

    df = save_outputs(
        output_base,
        tau_us,
        norm_counts,
        norm_counts_ste,
        results,
        summary_fig,
    )

    if SHOW_SUMMARY:
        plt.show(
            block=True
        )
    else:
        plt.close(
            summary_fig
        )

    return df, results


if __name__ == "__main__":
    main()
