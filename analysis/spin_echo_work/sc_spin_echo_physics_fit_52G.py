# -*- coding: utf-8 -*-
"""
V7: orientation-locked, physically constrained 13C Hahn-echo fitter
==================================================================

Purpose
-------
Refit the 52 G QNami spin-echo data from scratch with a model that is much more
identifiable than the earlier phenomenological 14-parameter fit.

Core physical choices
---------------------
1. NV orientation is an EXTERNAL constraint.  For each NV, only catalog sites
   from that assigned orientation are ever considered.
2. The 13C revival period is fixed by gamma_13C * |B|, with only a small fitted
   global timing correction per trace.
3. No free revival chirp, no revival-amplitude taper, no revival-width slope,
   and no arbitrary independent f0/f1 phases.
4. A candidate site's modulation is determined by its catalog
       kappa, fI, omega_ms
   through the standard I=1/2 Hahn-echo ESEEM factor
       M(2tau) = 1 - 2*kappa*sin^2(omega_I*tau/2)
                            *sin^2(omega_ms*tau/2)
   where the experiment x-axis is total evolution time t = 2*tau.
5. T2 and beta are fitted with conservative bounds, but T2 is ALSO profiled.
   If the data do not close the upper confidence interval, V7 reports T2 as
   "lower_bound_only" rather than reporting the arbitrary search ceiling.
6. Every shortlisted carbon site is refit with the same nuisance model and
   multistart budget before AICc/Akaike ranking.

Outputs
-------
- *_v7_all_site_fits.csv
- *_v7_top10_sites.csv
- *_v7_nv_summary.csv
- *_v7_orientation_assignments.csv
- *_v7_t2_profiles.csv
- *_v7_dashboard.pdf
- *_v7_global_summary.png/.pdf
- *_v7_checkpoint.npz

Run from repository root:
    python analysis/spin_echo_work/sc_spin_echo_physics_fit_52G_v7.py
"""

from __future__ import annotations

import ast
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits


# =============================================================================
# DATA / PHYSICS
# =============================================================================

SEARCH_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master")
RESULT_TAG = "spin_echo_old_protocol_ranked_52G"

ALL_ATTEMPTS_PATH = None
CHECKPOINT_PATH = None
OUTPUT_DIR = None

CATALOG_PATH = Path(
    r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json"
)

# Independent orientation assignment produced by V6.  If None, newest matching
# file is discovered automatically.
ORIENTATION_ASSIGNMENTS_CSV = None

ALLOWED_ORIENTATIONS = (
    (1, 1, -1),
    (-1, 1, 1),
)

# 52 G field used to build the catalog.
B_VECTOR_G = np.array(
    [-48.551229, -18.748242, -5.973533],
    dtype=float,
)
GAMMA_C13_KHZ_PER_G = 1.0705

B_MAG_G = float(np.linalg.norm(B_VECTOR_G))
F_I_THEORY_KHZ = GAMMA_C13_KHZ_PER_G * B_MAG_G
REVIVAL_TOTAL_THEORY_US = 2000.0 / F_I_THEORY_KHZ

# Exclude the vacancy/zero-distance record and extremely remote candidates.
MIN_DISTANCE_A = 1.4
MAX_DISTANCE_A = 22.0
MIN_KAPPA = 1e-4

# Candidate spectral band.  The actual upper bound is also limited by the
# sampling Nyquist frequency.
MIN_ESEEM_LINE_KHZ = 5.0


# =============================================================================
# MODEL COMPLEXITY / BOUNDS
# =============================================================================

# bath revival model: quartic revival lobes at k*Trev.
REVIVAL_DELTA_BOUND_US = 0.80
REVIVAL_WIDTH_BOUNDS_US = (2.0, 10.0)

# T2 search bounds are computational, NOT a statement that the upper endpoint
# is a measured T2.  The profile-likelihood status determines reportability.
T2_SEARCH_BOUNDS_US = (5.0, 300.0)
BETA_BOUNDS = (0.70, 2.50)

BASELINE_BOUNDS = (0.0, 1.10)
CONTRAST_BOUNDS = (0.0, 0.95)

# robust fitting
LOSS = "soft_l1"
F_SCALE = 1.0

# Stage 1 core (no single-site ESEEM) multistarts
CORE_T2_STARTS_US = (20.0, 40.0, 80.0, 150.0)
CORE_BETA_STARTS = (1.0, 1.5, 2.0)

# Stage 2: all correct-orientation sites are screened with nuisance parameters
# frozen at the core fit.  Only the best candidates receive full nonlinear fits.
SCREEN_KEEP = 24

# Stage 3 equal-footing nonlinear refit
FULL_T2_MULTIPLIERS = (0.60, 1.0, 1.8)
FULL_BETA_STARTS = (1.0, 1.7)
FULL_MAX_NFEV = 20_000

# T2 profile for the winning site
PROFILE_T2_GRID_US = np.geomspace(
    T2_SEARCH_BOUNDS_US[0],
    T2_SEARCH_BOUNDS_US[1],
    30,
)
PROFILE_MAX_NFEV = 10_000
PROFILE_DELTA_CHI2_68 = 1.0
PROFILE_DELTA_CHI2_95 = 3.84

# "near boundary" is used only for diagnostics.
BOUNDARY_FRAC = 0.015

# parallelism
CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(1, min(18, CPU_COUNT - 2))
BLAS_THREADS_PER_WORKER = 1

# plotting
DENSE_POINTS = 3000
TOP_PLOT = 3
TOP_SAVE = 10
POSITION_TOP_N = 8
WEIGHT_TOP_N = 8
ZOOM_HALF_WIDTH_US = 12.5
SHOW_GLOBAL_SUMMARY = True

RANDOM_SEED = 20260923


# =============================================================================
# PATHS / I/O
# =============================================================================

@dataclass
class InputPaths:
    attempts: Path
    checkpoint: Path
    prefix: Path
    orientation_csv: Path


def newest_match(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} under {root}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def discover_paths() -> InputPaths:
    attempts = (
        Path(ALL_ATTEMPTS_PATH)
        if ALL_ATTEMPTS_PATH is not None
        else newest_match(
            SEARCH_ROOT,
            f"*{RESULT_TAG}_all_attempts.csv.gz",
        )
    )

    suffix = "_all_attempts.csv.gz"
    s = str(attempts)
    if not s.endswith(suffix):
        raise ValueError(f"Unexpected attempt filename: {attempts}")

    prefix = Path(s[: -len(suffix)])

    checkpoint = (
        Path(CHECKPOINT_PATH)
        if CHECKPOINT_PATH is not None
        else Path(str(prefix) + "_fit_checkpoint.npz")
    )

    orientation_csv = (
        Path(ORIENTATION_ASSIGNMENTS_CSV)
        if ORIENTATION_ASSIGNMENTS_CSV is not None
        else newest_match(
            SEARCH_ROOT,
            "*orientation_locked_confidence_v6_orientation_assignments.csv",
        )
    )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not orientation_csv.exists():
        raise FileNotFoundError(orientation_csv)
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(CATALOG_PATH)

    return InputPaths(
        attempts=attempts,
        checkpoint=checkpoint,
        prefix=prefix,
        orientation_csv=orientation_csv,
    )


def canonical_orientation(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    arr = np.asarray(value, int).ravel()
    if arr.size != 3:
        raise ValueError(f"Bad orientation: {value}")
    return tuple(int(v) for v in arr)


def load_inputs(paths: InputPaths):
    ck = np.load(paths.checkpoint, allow_pickle=True)

    t = np.asarray(ck["times_us"], float)
    y = np.asarray(ck["norm_counts"], float)
    e = np.asarray(ck["norm_counts_ste"], float)

    if y.ndim != 2 or e.shape != y.shape:
        raise ValueError(f"Unexpected y/e shapes: {y.shape}, {e.shape}")
    if y.shape[1] != len(t):
        raise ValueError("Time axis does not match data.")

    e = np.maximum(np.abs(e), 1e-4)

    ori_df = pd.read_csv(paths.orientation_csv)
    ori_map = {
        int(row.nv_index): canonical_orientation(row.orientation)
        for row in ori_df.itertuples()
    }

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw_catalog = json.load(f)

    allowed = set(ALLOWED_ORIENTATIONS)
    catalog = []
    for rec in raw_catalog:
        ori = canonical_orientation(rec["orientation"])
        if ori not in allowed:
            continue

        distance = float(rec.get("distance_A", np.nan))
        kappa = float(rec.get("kappa", np.nan))

        if not np.isfinite(distance) or not (MIN_DISTANCE_A <= distance <= MAX_DISTANCE_A):
            continue
        if not np.isfinite(kappa) or kappa < MIN_KAPPA:
            continue

        r = dict(rec)
        r["orientation_tuple"] = ori
        r["site_id"] = int(rec["site_index"])
        r["fI_kHz"] = float(rec["fI_Hz"]) / 1e3
        r["fm_kHz"] = float(rec["omega_ms_Hz"]) / 1e3
        r["fminus_kHz"] = float(rec["f_minus_Hz"]) / 1e3
        r["fplus_kHz"] = float(rec["f_plus_Hz"]) / 1e3
        catalog.append(r)

    return t, y, e, ori_df, ori_map, catalog


# =============================================================================
# PHYSICAL MODEL
# =============================================================================

def quartic_revival_comb(t_us, revival_total_us, width_us):
    """
    Unit-height quartic bath-revival lobes.

    The x-axis is total Hahn-echo evolution time 2*tau.
    """
    t = np.asarray(t_us, float)
    tmax = float(np.max(t)) if t.size else 0.0
    nrev = max(1, int(np.ceil(tmax / revival_total_us)) + 2)

    out = np.zeros_like(t)
    w4 = max(float(width_us), 1e-6) ** 4

    for k in range(nrev):
        mu = k * float(revival_total_us)
        out += np.exp(-((t - mu) ** 4) / w4)

    return out


def single_c13_hahn_factor(
    total_time_us,
    kappa,
    fI_kHz,
    fm_kHz,
):
    """
    Standard single-I=1/2 Hahn-echo ESEEM factor.

        V(2tau) = 1 - 2*kappa
                    sin^2(omega_I*tau/2)
                    sin^2(omega_m*tau/2)

    Input x-axis is total_time_us = 2*tau, therefore

        omega*tau/2 = pi*f*total_time/2.

    Frequencies are supplied in kHz.
    """
    t = np.asarray(total_time_us, float)

    fI_cyc_us = float(fI_kHz) / 1000.0
    fm_cyc_us = float(fm_kHz) / 1000.0

    a = np.sin(0.5 * np.pi * fI_cyc_us * t) ** 2
    b = np.sin(0.5 * np.pi * fm_cyc_us * t) ** 2

    return 1.0 - 2.0 * float(kappa) * a * b


def physical_model(
    t_us,
    baseline,
    contrast,
    revival_delta_us,
    width_us,
    T2_us,
    beta,
    *,
    site=None,
):
    """
    Reduced identifiable model.

    No taper, no chirp, no width growth, no arbitrary ESEEM phases.
    """
    t = np.asarray(t_us, float)

    revival = REVIVAL_TOTAL_THEORY_US + float(revival_delta_us)
    envelope = np.exp(
        -np.power(
            np.maximum(t, 0.0) / max(float(T2_us), 1e-9),
            float(beta),
        )
    )

    bath = quartic_revival_comb(
        t,
        revival_total_us=revival,
        width_us=width_us,
    )

    coherence = envelope * bath

    if site is not None:
        coherence = coherence * single_c13_hahn_factor(
            t,
            kappa=site["kappa"],
            fI_kHz=site["fI_kHz"],
            fm_kHz=site["fm_kHz"],
        )

    return float(baseline) - float(contrast) * coherence


# =============================================================================
# FITTING
# =============================================================================

PARAM_NAMES = (
    "baseline",
    "contrast",
    "revival_delta_us",
    "width_us",
    "T2_us",
    "beta",
)

LB = np.array(
    [
        BASELINE_BOUNDS[0],
        CONTRAST_BOUNDS[0],
        -REVIVAL_DELTA_BOUND_US,
        REVIVAL_WIDTH_BOUNDS_US[0],
        T2_SEARCH_BOUNDS_US[0],
        BETA_BOUNDS[0],
    ],
    float,
)

UB = np.array(
    [
        BASELINE_BOUNDS[1],
        CONTRAST_BOUNDS[1],
        +REVIVAL_DELTA_BOUND_US,
        REVIVAL_WIDTH_BOUNDS_US[1],
        T2_SEARCH_BOUNDS_US[1],
        BETA_BOUNDS[1],
    ],
    float,
)


def data_seed(yv):
    yv = np.asarray(yv, float)
    baseline = float(np.clip(np.nanpercentile(yv, 90), 0.25, 1.0))
    ymin = float(np.nanpercentile(yv, 5))
    contrast = float(np.clip(baseline - ymin, 0.03, 0.65))

    return np.array(
        [
            baseline,
            contrast,
            0.0,
            5.5,
            50.0,
            1.4,
        ],
        float,
    )


def ordinary_stats(y, e, pred, npar):
    resid = (np.asarray(y) - np.asarray(pred)) / np.maximum(np.asarray(e), 1e-12)
    chi2 = float(np.sum(resid**2))

    n = len(y)
    k = int(npar)
    dof = max(1, n - k)
    red = chi2 / dof

    aic = chi2 + 2 * k
    aicc = (
        aic + 2 * k * (k + 1) / (n - k - 1)
        if n > k + 1
        else np.inf
    )

    return chi2, red, float(aicc)


def fit_model(
    t,
    y,
    e,
    p0,
    *,
    site=None,
    fixed_T2_us=None,
    max_nfev=FULL_MAX_NFEV,
):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    e = np.maximum(np.asarray(e, float), 1e-12)

    if fixed_T2_us is None:
        lb = LB.copy()
        ub = UB.copy()
        x0 = np.clip(np.asarray(p0, float), lb + 1e-8, ub - 1e-8)

        def residual(p):
            return (
                y
                - physical_model(
                    t,
                    *p,
                    site=site,
                )
            ) / e

        res = least_squares(
            residual,
            x0=x0,
            bounds=(lb, ub),
            loss=LOSS,
            f_scale=F_SCALE,
            max_nfev=int(max_nfev),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )

        popt = res.x

    else:
        fixed_T2_us = float(fixed_T2_us)
        keep = [0, 1, 2, 3, 5]

        lb5 = LB[keep]
        ub5 = UB[keep]

        p0 = np.asarray(p0, float)
        x0 = np.clip(p0[keep], lb5 + 1e-8, ub5 - 1e-8)

        def assemble(q):
            p = np.empty(6, float)
            p[keep] = q
            p[4] = fixed_T2_us
            return p

        def residual(q):
            p = assemble(q)
            return (
                y
                - physical_model(
                    t,
                    *p,
                    site=site,
                )
            ) / e

        res = least_squares(
            residual,
            x0=x0,
            bounds=(lb5, ub5),
            loss=LOSS,
            f_scale=F_SCALE,
            max_nfev=int(max_nfev),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )

        popt = assemble(res.x)

    pred = physical_model(
        t,
        *popt,
        site=site,
    )

    chi2, red, aicc = ordinary_stats(
        y,
        e,
        pred,
        npar=5 if fixed_T2_us is not None else 6,
    )

    return {
        "popt": popt,
        "pred": pred,
        "chi2": chi2,
        "red_chi2": red,
        "aicc": aicc,
        "success": bool(res.success),
        "nfev": int(res.nfev),
    }


def fit_core_multistart(t, y, e):
    seed = data_seed(y)
    fits = []

    for t2 in CORE_T2_STARTS_US:
        for beta in CORE_BETA_STARTS:
            p0 = seed.copy()
            p0[4] = t2
            p0[5] = beta

            try:
                fit = fit_model(
                    t,
                    y,
                    e,
                    p0,
                    site=None,
                    max_nfev=12_000,
                )
                fits.append(fit)
            except Exception:
                pass

    if not fits:
        raise RuntimeError("Core fit failed for all starts.")

    return min(fits, key=lambda r: (r["chi2"], r["red_chi2"]))


def sampling_nyquist_khz(t):
    unique = np.unique(np.asarray(t, float))
    dt = np.diff(unique)
    dt = dt[dt > 0]

    if not len(dt):
        return np.inf

    return 500.0 / float(np.min(dt))


def catalog_for_nv(catalog, orientation, t):
    """
    Hard orientation filter + sampling filter.
    """
    orientation = tuple(orientation)
    nyq = sampling_nyquist_khz(t)

    out = []

    for rec in catalog:
        if tuple(rec["orientation_tuple"]) != orientation:
            continue

        # The exact modulation produces f+ and f- components.  Keep sites whose
        # informative lines are representable by the sampled trace.
        fm = float(rec["fminus_kHz"])
        fp = float(rec["fplus_kHz"])

        if fp > nyq * 0.98:
            continue
        if max(fm, fp) < MIN_ESEEM_LINE_KHZ:
            continue

        out.append(rec)

    return out


def screen_sites(t, y, e, core_popt, sites):
    """
    Fast deterministic screen: all nuisance parameters fixed to the core fit.
    No site-specific free parameter is introduced here.
    """
    rows = []

    for site in sites:
        pred = physical_model(
            t,
            *core_popt,
            site=site,
        )

        chi2, red, aicc = ordinary_stats(
            y,
            e,
            pred,
            npar=6,
        )

        rows.append(
            {
                "site": site,
                "screen_chi2": chi2,
                "screen_red_chi2": red,
                "screen_aicc": aicc,
            }
        )

    rows.sort(key=lambda r: (r["screen_chi2"], -float(r["site"]["kappa"])))
    return rows


def full_fit_site(t, y, e, core_popt, site):
    fits = []

    base_t2 = float(core_popt[4])

    for mult in FULL_T2_MULTIPLIERS:
        for beta in FULL_BETA_STARTS:
            p0 = np.asarray(core_popt, float).copy()
            p0[4] = np.clip(
                base_t2 * float(mult),
                T2_SEARCH_BOUNDS_US[0] + 1e-4,
                T2_SEARCH_BOUNDS_US[1] - 1e-4,
            )
            p0[5] = np.clip(
                beta,
                BETA_BOUNDS[0] + 1e-4,
                BETA_BOUNDS[1] - 1e-4,
            )

            try:
                fits.append(
                    fit_model(
                        t,
                        y,
                        e,
                        p0,
                        site=site,
                        max_nfev=FULL_MAX_NFEV,
                    )
                )
            except Exception:
                pass

    if not fits:
        return None

    return min(fits, key=lambda r: (r["chi2"], r["red_chi2"]))


def near_bound_flags(p):
    p = np.asarray(p, float)
    span = UB - LB

    near_lo = (p - LB) <= BOUNDARY_FRAC * span
    near_hi = (UB - p) <= BOUNDARY_FRAC * span

    return {
        f"{name}_at_bound": bool(lo or hi)
        for name, lo, hi in zip(PARAM_NAMES, near_lo, near_hi)
    }


def profile_t2(t, y, e, site, best_popt):
    rows = []
    warm = np.asarray(best_popt, float).copy()

    # Add the point estimate to the grid so delta=0 is represented exactly.
    grid = np.unique(
        np.concatenate(
            [
                np.asarray(PROFILE_T2_GRID_US, float),
                [float(best_popt[4])],
            ]
        )
    )

    for t2 in grid:
        try:
            fit = fit_model(
                t,
                y,
                e,
                warm,
                site=site,
                fixed_T2_us=float(t2),
                max_nfev=PROFILE_MAX_NFEV,
            )
            warm = fit["popt"].copy()

            rows.append(
                {
                    "T2_us": float(t2),
                    "chi2": float(fit["chi2"]),
                    "red_chi2": float(fit["red_chi2"]),
                    "popt": fit["popt"],
                }
            )
        except Exception:
            continue

    if not rows:
        return [], {
            "T2_profile_status": "profile_failed",
            "T2_profile_best_us": np.nan,
            "T2_68_low_us": np.nan,
            "T2_68_high_us": np.nan,
            "T2_95_low_us": np.nan,
            "T2_95_high_us": np.nan,
        }

    min_chi2 = min(r["chi2"] for r in rows)

    for r in rows:
        r["delta_chi2"] = r["chi2"] - min_chi2

    def interval(threshold):
        good = [r["T2_us"] for r in rows if r["delta_chi2"] <= threshold]

        if not good:
            return np.nan, np.nan, False, False

        lo = float(min(good))
        hi = float(max(good))

        touches_lo = np.isclose(
            lo,
            T2_SEARCH_BOUNDS_US[0],
            rtol=0,
            atol=0.02 * T2_SEARCH_BOUNDS_US[0],
        )

        touches_hi = hi >= 0.98 * T2_SEARCH_BOUNDS_US[1]

        return lo, hi, touches_lo, touches_hi

    lo68, hi68, touch_lo68, touch_hi68 = interval(PROFILE_DELTA_CHI2_68)
    lo95, hi95, touch_lo95, touch_hi95 = interval(PROFILE_DELTA_CHI2_95)

    best_row = min(rows, key=lambda r: r["chi2"])
    profile_best = float(best_row["T2_us"])

    if touch_hi95:
        status = "lower_bound_only"
        report = f"> {lo95:.1f} us (95% profile)"
    elif touch_lo95:
        status = "upper_bound_only"
        report = f"< {hi95:.1f} us (95% profile)"
    else:
        status = "bounded"
        report = f"{profile_best:.1f} us [{lo95:.1f}, {hi95:.1f}] 95% profile"

    summary = {
        "T2_profile_status": status,
        "T2_profile_report": report,
        "T2_profile_best_us": profile_best,
        "T2_68_low_us": lo68,
        "T2_68_high_us": hi68,
        "T2_95_low_us": lo95,
        "T2_95_high_us": hi95,
        "T2_profile_touches_upper_68": touch_hi68,
        "T2_profile_touches_upper_95": touch_hi95,
    }

    return rows, summary


# =============================================================================
# PER-NV FIT
# =============================================================================

def site_metadata(site):
    return {
        "orientation": str(tuple(site["orientation_tuple"])),
        "site_id": int(site["site_id"]),
        "distance_A": float(site["distance_A"]),
        "kappa": float(site["kappa"]),
        "fI_kHz": float(site["fI_kHz"]),
        "fm_kHz": float(site["fm_kHz"]),
        "fminus_kHz": float(site["fminus_kHz"]),
        "fplus_kHz": float(site["fplus_kHz"]),
        "A_par_kHz": float(site.get("A_par_Hz", np.nan)) / 1e3,
        "A_perp_kHz": float(site.get("A_perp_Hz", np.nan)) / 1e3,
        "theta_deg": float(site.get("theta_deg", np.nan)),
        "x_A": float(site.get("x_A", np.nan)),
        "y_A": float(site.get("y_A", np.nan)),
        "z_A": float(site.get("z_A", np.nan)),
    }


def fit_one_nv(nv, t, y, e, orientation, catalog):
    yv = np.asarray(y[nv], float)
    ev = np.asarray(e[nv], float)

    core = fit_core_multistart(t, yv, ev)
    core_p = core["popt"]

    sites = catalog_for_nv(
        catalog,
        orientation=orientation,
        t=t,
    )

    if not sites:
        raise RuntimeError(
            f"NV {nv}: no catalog sites survive for orientation {orientation}"
        )

    screened = screen_sites(
        t,
        yv,
        ev,
        core_p,
        sites,
    )

    shortlist = screened[: min(SCREEN_KEEP, len(screened))]

    full_rows = []

    for srow in shortlist:
        site = srow["site"]

        fit = full_fit_site(
            t,
            yv,
            ev,
            core_p,
            site,
        )

        if fit is None:
            continue

        p = fit["popt"]
        row = {
            "nv_index": int(nv),
            **site_metadata(site),
            "screen_chi2": float(srow["screen_chi2"]),
            "screen_red_chi2": float(srow["screen_red_chi2"]),
            "chi2": float(fit["chi2"]),
            "red_chi2": float(fit["red_chi2"]),
            "aicc": float(fit["aicc"]),
            "baseline": float(p[0]),
            "contrast": float(p[1]),
            "revival_delta_us": float(p[2]),
            "revival_total_us": float(REVIVAL_TOTAL_THEORY_US + p[2]),
            "width_us": float(p[3]),
            "T2_us": float(p[4]),
            "beta": float(p[5]),
            "nfev": int(fit["nfev"]),
            "popt_json": json.dumps(np.asarray(p, float).tolist()),
            **near_bound_flags(p),
        }

        full_rows.append(row)

    if not full_rows:
        raise RuntimeError(f"NV {nv}: all shortlisted site fits failed.")

    full_rows.sort(key=lambda r: (r["aicc"], r["chi2"]))

    aicc = np.asarray([r["aicc"] for r in full_rows], float)
    da = aicc - np.min(aicc)

    logw = -0.5 * da
    logw -= np.max(logw)
    weights = np.exp(logw)
    weights /= np.sum(weights)

    for rank, (row, delta, weight) in enumerate(
        zip(full_rows, da, weights),
        start=1,
    ):
        row["site_rank"] = int(rank)
        row["delta_aicc"] = float(delta)
        row["akaike_weight"] = float(weight)

    best = full_rows[0]
    best_site = next(
        s for s in sites
        if int(s["site_id"]) == int(best["site_id"])
    )

    best_p = np.asarray(json.loads(best["popt_json"]), float)

    prof_rows, prof_summary = profile_t2(
        t,
        yv,
        ev,
        best_site,
        best_p,
    )

    for r in prof_rows:
        r["nv_index"] = int(nv)
        r["orientation"] = str(tuple(orientation))
        r["site_id"] = int(best["site_id"])
        r.pop("popt", None)

    summary = {
        "nv_index": int(nv),
        "orientation": str(tuple(orientation)),
        "core_red_chi2": float(core["red_chi2"]),
        "core_T2_us": float(core_p[4]),
        "best_site_id": int(best["site_id"]),
        "best_kappa": float(best["kappa"]),
        "best_distance_A": float(best["distance_A"]),
        "best_fminus_kHz": float(best["fminus_kHz"]),
        "best_fplus_kHz": float(best["fplus_kHz"]),
        "best_red_chi2": float(best["red_chi2"]),
        "best_aicc": float(best["aicc"]),
        "best_weight": float(best["akaike_weight"]),
        "fit_T2_us": float(best["T2_us"]),
        "fit_beta": float(best["beta"]),
        "fit_revival_total_us": float(best["revival_total_us"]),
        "fit_width_us": float(best["width_us"]),
        "fit_baseline": float(best["baseline"]),
        "fit_contrast": float(best["contrast"]),
        **prof_summary,
    }

    print(
        f"[NV {nv:3d}] ori={orientation} "
        f"site={best['site_id']:4d} "
        f"redχ²={best['red_chi2']:.3g} "
        f"T2fit={best['T2_us']:.1f} us "
        f"profile={prof_summary['T2_profile_status']}"
    )

    return full_rows, prof_rows, summary


# =============================================================================
# PLOTTING
# =============================================================================

def parse_popt(row):
    return np.asarray(json.loads(row["popt_json"]), float)


def reconstruct_site(row):
    return {
        "kappa": float(row["kappa"]),
        "fI_kHz": float(row["fI_kHz"]),
        "fm_kHz": float(row["fm_kHz"]),
    }


def set_3d_equal(ax, xyz):
    xyz = np.asarray(xyz, float)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]

    if not len(xyz):
        return

    xyz = np.vstack([xyz, np.zeros((1, 3))])
    lo = np.min(xyz, axis=0)
    hi = np.max(xyz, axis=0)
    c = 0.5 * (lo + hi)
    r = max(1.0, 0.55 * np.max(hi - lo))

    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def plot_nv_dashboard(pdf, nv, t, y, e, site_df, profile_df, summary_row):
    sub = (
        site_df[site_df["nv_index"] == nv]
        .sort_values("site_rank")
        .copy()
    )

    prof = (
        profile_df[profile_df["nv_index"] == nv]
        .sort_values("T2_us")
        .copy()
    )

    if sub.empty:
        return

    colors = {
        rank: plt.get_cmap("tab10")((rank - 1) % 10)
        for rank in range(1, max(POSITION_TOP_N, WEIGHT_TOP_N) + 1)
    }

    fig = plt.figure(figsize=(20.5, 11.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.15, 1.05, 1.0],
        hspace=0.30,
        wspace=0.26,
    )

    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_weight = fig.add_subplot(gs[0, 2])
    ax_pos = fig.add_subplot(gs[1, 0], projection="3d")
    ax_prof = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[1, 2])

    # full + zoom
    ax_full.errorbar(
        t,
        y[nv],
        yerr=e[nv],
        fmt="o",
        ms=3.0,
        capsize=1,
        lw=0.55,
        label="data",
        zorder=10,
    )

    td = np.linspace(float(np.min(t)), float(np.max(t)), DENSE_POINTS)
    top3 = sub.head(TOP_PLOT)

    for _, row in top3.iterrows():
        rank = int(row.site_rank)
        p = parse_popt(row)
        site = reconstruct_site(row)

        ax_full.plot(
            td,
            physical_model(td, *p, site=site),
            color=colors[rank],
            lw=1.8 if rank == 1 else 1.25,
            label=(
                f"#{rank} site {int(row.site_id)} | "
                f"χ²r={row.red_chi2:.2f}, w={row.akaike_weight:.2f}"
            ),
        )

    ax_full.set_title("Full trace: physical V7 model")
    ax_full.set_xlabel("Total evolution time (µs)")
    ax_full.set_ylabel("Normalized signal")
    ax_full.grid(alpha=0.20)
    ax_full.legend(fontsize=7)

    center = REVIVAL_TOTAL_THEORY_US
    lo = center - ZOOM_HALF_WIDTH_US
    hi = center + ZOOM_HALF_WIDTH_US
    m = (t >= lo) & (t <= hi)

    ax_zoom.errorbar(
        t[m],
        y[nv, m],
        yerr=e[nv, m],
        fmt="o",
        ms=3.0,
        capsize=1,
        lw=0.55,
        zorder=10,
    )

    tz = np.linspace(lo, hi, DENSE_POINTS)

    for _, row in top3.iterrows():
        rank = int(row.site_rank)
        p = parse_popt(row)
        site = reconstruct_site(row)

        ax_zoom.plot(
            tz,
            physical_model(tz, *p, site=site),
            color=colors[rank],
            lw=1.8 if rank == 1 else 1.25,
            label=f"#{rank} site {int(row.site_id)}",
        )

    ax_zoom.axvline(
        REVIVAL_TOTAL_THEORY_US,
        ls="--",
        lw=0.9,
        alpha=0.6,
        label="13C theory revival",
    )
    ax_zoom.set_title("First-revival zoom")
    ax_zoom.set_xlabel("Total evolution time (µs)")
    ax_zoom.set_ylabel("Normalized signal")
    ax_zoom.grid(alpha=0.20)
    ax_zoom.legend(fontsize=7)

    # Akaike weights
    ws = sub.head(WEIGHT_TOP_N)
    x = np.arange(len(ws))
    bars = ax_weight.bar(
        x,
        ws["akaike_weight"].to_numpy(float),
        color=[colors[int(r)] for r in ws["site_rank"]],
    )
    ax_weight.set_xticks(x)
    ax_weight.set_xticklabels(
        [
            f"#{int(r.site_rank)}\nS{int(r.site_id)}"
            for r in ws.itertuples()
        ],
        fontsize=7,
    )
    ax_weight.set_ylabel("Akaike weight")
    ax_weight.set_title("Relative carbon-site support")
    ax_weight.grid(alpha=0.20, axis="y")

    for bar, w in zip(bars, ws["akaike_weight"]):
        ax_weight.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{w:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # positions
    ps = sub.head(POSITION_TOP_N)
    xyz = []

    ax_pos.scatter(
        [0], [0], [0],
        marker="*",
        s=200,
        c="black",
        label="NV",
        depthshade=False,
    )

    for _, row in ps.iterrows():
        rank = int(row.site_rank)
        point = np.array([row.x_A, row.y_A, row.z_A], float)

        if not np.all(np.isfinite(point)):
            continue

        xyz.append(point)

        ax_pos.plot(
            [0, point[0]],
            [0, point[1]],
            [0, point[2]],
            color=colors[rank],
            alpha=0.45,
            lw=0.8,
        )
        ax_pos.scatter(
            [point[0]],
            [point[1]],
            [point[2]],
            color=colors[rank],
            s=95 if rank <= 3 else 50,
            depthshade=False,
            label=f"#{rank} S{int(row.site_id)}",
        )
        ax_pos.text(
            point[0],
            point[1],
            point[2],
            f" #{rank}:S{int(row.site_id)}",
            fontsize=7,
        )

    if xyz:
        set_3d_equal(ax_pos, np.vstack(xyz))

    ax_pos.set_xlabel("x (Å)")
    ax_pos.set_ylabel("y (Å)")
    ax_pos.set_zlabel("z (Å)")
    ax_pos.set_title(
        f"$^{{13}}$C candidates\norientation {summary_row.orientation}"
    )
    ax_pos.view_init(elev=24, azim=38)
    ax_pos.legend(fontsize=6)

    # T2 profile
    if not prof.empty:
        ax_prof.plot(
            prof["T2_us"],
            prof["delta_chi2"],
            marker="o",
            ms=3.5,
            lw=1.2,
        )
        ax_prof.axhline(
            PROFILE_DELTA_CHI2_68,
            ls="--",
            lw=0.8,
            label="Δχ²=1",
        )
        ax_prof.axhline(
            PROFILE_DELTA_CHI2_95,
            ls=":",
            lw=1.0,
            label="Δχ²=3.84",
        )
        ax_prof.set_xscale("log")
        ax_prof.set_ylim(
            bottom=0,
            top=max(
                5.0,
                min(
                    25.0,
                    1.10 * float(np.nanmax(prof["delta_chi2"])),
                ),
            ),
        )
        ax_prof.legend(fontsize=7)

    ax_prof.set_xlabel("$T_2$ (µs)")
    ax_prof.set_ylabel("Profile Δχ²")
    ax_prof.set_title(
        f"$T_2$ identifiability: {summary_row.T2_profile_status}"
    )
    ax_prof.grid(alpha=0.20, which="both")

    # parameter text
    ax_text.axis("off")
    best = sub.iloc[0]

    text_lines = [
        f"NV {nv}",
        f"orientation = {summary_row.orientation}",
        "",
        f"BEST SITE = {int(best.site_id)}",
        f"distance = {best.distance_A:.2f} Å",
        f"kappa = {best.kappa:.4f}",
        f"f- / f+ = {best.fminus_kHz:.2f} / {best.fplus_kHz:.2f} kHz",
        f"A_parallel = {best.A_par_kHz:.2f} kHz",
        f"A_perp = {best.A_perp_kHz:.2f} kHz",
        "",
        f"red chi2 = {best.red_chi2:.3f}",
        f"Akaike weight = {best.akaike_weight:.3f}",
        f"delta AICc (#2) = "
        + (
            f"{sub.iloc[1].delta_aicc:.3f}"
            if len(sub) > 1
            else "n/a"
        ),
        "",
        f"T2 point fit = {best.T2_us:.1f} µs",
        f"T2 profile = {summary_row.T2_profile_report}",
        f"beta = {best.beta:.3f}",
        f"revival = {best.revival_total_us:.3f} µs",
        f"theory revival = {REVIVAL_TOTAL_THEORY_US:.3f} µs",
        f"width = {best.width_us:.3f} µs",
        f"baseline = {best.baseline:.4f}",
        f"contrast = {best.contrast:.4f}",
        "",
        "No taper / chirp / width-growth / free phases.",
    ]

    ax_text.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.3,
    )

    fig.suptitle(
        (
            f"V7 physical fit | NV {nv} | {summary_row.orientation} | "
            f"site {int(best.site_id)} | χ²r={best.red_chi2:.3f}"
        ),
        fontsize=13,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def global_summary_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(summary["fit_T2_us"], bins=35)
    ax.set_xlabel("Point-fit $T_2$ (µs)")
    ax.set_ylabel("NV count")
    ax.set_title("V7 point-fit T2")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    counts = summary["T2_profile_status"].value_counts()
    ax.bar(np.arange(len(counts)), counts.values)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_ylabel("NV count")
    ax.set_title("T2 identifiability")
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1, 0]
    ax.scatter(
        summary["best_red_chi2"],
        summary["best_weight"],
        s=16,
    )
    ax.set_xlabel("Best reduced χ²")
    ax.set_ylabel("Best-site Akaike weight")
    ax.set_title("Fit quality vs site support")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.hist(summary["fit_beta"], bins=30)
    ax.set_xlabel("Stretch exponent β")
    ax.set_ylabel("NV count")
    ax.set_title("V7 β distribution")
    ax.grid(alpha=0.2)

    fig.suptitle(
        f"V7 orientation-locked physical fit | {len(summary)} NVs",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(RANDOM_SEED)

    paths = discover_paths()
    t, y, e, ori_df, ori_map, catalog = load_inputs(paths)

    print("=" * 96)
    print("V7 ORIENTATION-LOCKED PHYSICALLY CONSTRAINED 13C SPIN-ECHO FIT")
    print("=" * 96)
    print(f"attempt source       : {paths.attempts}")
    print(f"checkpoint           : {paths.checkpoint}")
    print(f"orientation source   : {paths.orientation_csv}")
    print(f"catalog              : {CATALOG_PATH}")
    print(f"NVs                  : {y.shape[0]}")
    print(f"time points          : {len(t)}")
    print(f"time range           : {t.min():.3f}–{t.max():.3f} µs")
    print(f"|B|                  : {B_MAG_G:.6f} G")
    print(f"13C Larmor           : {F_I_THEORY_KHZ:.6f} kHz")
    print(f"theory total revival : {REVIVAL_TOTAL_THEORY_US:.6f} µs")
    print(f"catalog records kept : {len(catalog)}")
    print(f"workers              : {N_JOBS}")
    print("=" * 96)

    jobs = []

    for nv in range(y.shape[0]):
        ori = ori_map.get(nv)

        if ori not in set(ALLOWED_ORIENTATIONS):
            print(f"[NV {nv}] skipped: invalid/missing orientation {ori}")
            continue

        jobs.append((nv, ori))

    def task(nv, ori):
        return fit_one_nv(
            nv,
            t,
            y,
            e,
            ori,
            catalog,
        )

    with threadpool_limits(limits=BLAS_THREADS_PER_WORKER):
        results = Parallel(
            n_jobs=N_JOBS,
            backend="loky",
            batch_size=1,
            verbose=5,
        )(
            delayed(task)(nv, ori)
            for nv, ori in jobs
        )

    site_rows = []
    profile_rows = []
    summaries = []

    for sites, prof, summary in results:
        site_rows.extend(sites)
        profile_rows.extend(prof)
        summaries.append(summary)

    site_df = pd.DataFrame(site_rows)
    profile_df = pd.DataFrame(profile_rows)
    summary_df = pd.DataFrame(summaries).sort_values("nv_index")

    outdir = (
        Path(OUTPUT_DIR)
        if OUTPUT_DIR is not None
        else paths.prefix.parent
    )
    outdir.mkdir(parents=True, exist_ok=True)

    base = outdir / (paths.prefix.name + "_v7_physical")

    site_csv = Path(str(base) + "_all_site_fits.csv")
    top_csv = Path(str(base) + "_top10_sites.csv")
    summary_csv = Path(str(base) + "_nv_summary.csv")
    ori_csv = Path(str(base) + "_orientation_assignments.csv")
    profile_csv = Path(str(base) + "_t2_profiles.csv")
    dashboard_pdf = Path(str(base) + "_dashboard.pdf")
    global_png = Path(str(base) + "_global_summary.png")
    global_pdf = Path(str(base) + "_global_summary.pdf")
    checkpoint_out = Path(str(base) + "_checkpoint.npz")

    site_df.to_csv(site_csv, index=False)
    site_df[site_df["site_rank"] <= TOP_SAVE].to_csv(top_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    ori_df.to_csv(ori_csv, index=False)
    profile_df.to_csv(profile_csv, index=False)

    np.savez_compressed(
        checkpoint_out,
        times_us=t,
        norm_counts=y,
        norm_counts_ste=e,
        B_vector_G=B_VECTOR_G,
        B_mag_G=B_MAG_G,
        fI_theory_kHz=F_I_THEORY_KHZ,
        revival_total_theory_us=REVIVAL_TOTAL_THEORY_US,
    )

    with PdfPages(dashboard_pdf) as pdf:
        for row in summary_df.itertuples():
            plot_nv_dashboard(
                pdf,
                int(row.nv_index),
                t,
                y,
                e,
                site_df,
                profile_df,
                row,
            )

    fig = global_summary_figure(summary_df)
    fig.savefig(global_png, dpi=300, bbox_inches="tight")
    fig.savefig(global_pdf, bbox_inches="tight")

    print()
    print("=" * 96)
    print("V7 COMPLETE")
    print("=" * 96)
    print(f"NVs fitted                  : {len(summary_df)}")
    print(
        "T2 status counts            : "
        + str(summary_df["T2_profile_status"].value_counts().to_dict())
    )
    print(
        f"median point-fit T2          : "
        f"{summary_df['fit_T2_us'].median():.2f} µs"
    )
    print(
        f"median best reduced chi2     : "
        f"{summary_df['best_red_chi2'].median():.3f}"
    )
    print(f"site fits                    : {site_csv}")
    print(f"summary                      : {summary_csv}")
    print(f"T2 profiles                  : {profile_csv}")
    print(f"dashboard                    : {dashboard_pdf}")
    print("=" * 96)

    if SHOW_GLOBAL_SUMMARY:
        plt.show(block=True)
    else:
        plt.close(fig)

    return {
        "site_fits": site_df,
        "profiles": profile_df,
        "summary": summary_df,
        "orientation_assignments": ori_df,
    }


if __name__ == "__main__":
    main()
