# -*- coding: utf-8 -*-
"""
Spin-echo ranked confidence analysis V6 — orientation-locked physical model
=======================================

What this version fixes
-----------------------
1. This experiment uses exactly TWO allowed crystallographic NV orientations.
2. Each individual NV has ONE independently determined crystallographic orientation.
3. For each NV, ONLY the 13C catalog entries belonging to that NV's orientation
   are allowed into screening, equal-footing refitting, ranking, Akaike weights,
   bootstrap/CV, and plotting.
4. The C13 spin-echo fit is therefore NOT allowed to choose the NV orientation.
   Orientation is an external physical constraint, supplied by ESR/resonance
   assignment (or by a manual/CSV map).
5. The cheap screen-only candidates are not mixed directly with deep fits.
   Instead, the best candidates from the correct orientation pool are refit
   on equal footing with the same model, bounds, optimizer and budget.
4. A physical hypothesis is unique by:
       (NV index, known NV orientation, 13C site_id)
   so the same site cannot repeat in the top-3.
5. The main PDF is one NV per page:
       LEFT   = full spin-echo trace + dense top-3 fitted curves
       MIDDLE = first-revival zoom + same top-3 colors
       RIGHT  = spatial positions of the best 13C candidates, colored by rank
6. Fit curves are evaluated on a dense grid (default 3000 points).
7. Every equal-footing fit stores the old fitter's penalized score
   (score_primary), reduced chi-square, AICc, delta-AICc and Akaike weight.
8. Fitted physical/model parameters are expanded into explicit CSV columns:
   T2, revival time, revival width, stretch exponent, comb contrast, oscillation
   amplitude/frequencies/phases, taper, chirp, etc.
9. The main PDF is a six-information dashboard:
       top-left     full trace + dense top-3 fits
       top-middle   first-revival zoom
       top-right    Akaike-weight plot
       bottom-left  colored 13C candidate positions
       bottom-right top-3 detailed fit-parameter table
10. Optional bootstrap and held-out validation use the two experiment orientations.

This script consumes the saved exhaustive *_all_attempts.csv.gz and checkpoint;
it does NOT repeat the original 1500-site search.

For 52 G QNami:
    orientation is inferred from the latest resonance-analysis CSV by matching
    the measured doublet (f1,f2) to the four known ODMR orientation pairs.

For 49 G Johnson:
    orientation is read directly from the saved combined dataset's
    "orientations" array.

@author: Saroj Chand / analysis helper
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import lombscargle
from threadpoolctl import threadpool_limits

from analysis.spin_echo_work import fitter_module_for_spin_echo as oldfit

fine_decay = oldfit.fine_decay


# =============================================================================
# USER SETTINGS
# =============================================================================

# ---- Choose dataset -----------------------------------------------------------
RESULT_TAG = "spin_echo_old_protocol_ranked_52G"
# RESULT_TAG = "spin_echo_old_protocol_ranked_49G"

# The ONLY two NV orientations used in this experiment.
# Every NV is fit against BOTH corresponding 13C catalog pools.
ALLOWED_ORIENTATIONS = (
    (1, 1, -1),
    (-1, 1, 1),
)

SEARCH_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master")

ALL_ATTEMPTS_PATH = None
CHECKPOINT_PATH = None
OUTPUT_DIR = None

# ---- Catalog -----------------------------------------------------------------
if RESULT_TAG.endswith("52G"):
    CATALOG_PATH = Path(
        r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json"
    )
else:
    CATALOG_PATH = Path(
        r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_49G.json"
    )

# ---- Orientation: 52 G QNami -------------------------------------------------
AUTO_ASSIGN_52G_ORIENTATION_FROM_RESONANCE = True

RESONANCE_ANALYSIS_ROOT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master"
    r"\sc_resonance_analysis_optimized\resonance_analysis"
)

# The four measured ODMR pairs at the 52 G field.
# low transition, high transition in GHz.
ESR_ORIENTATION_TARGETS_GHZ = {
    (1, 1, -1): (2.7773, 2.9758),
    (-1, 1, 1): (2.8421, 2.9195),
}

# We still assign the nearest joint doublet if this is exceeded, but flag it.
ORIENTATION_WARN_RMS_MHZ = 10.0

# ---- Orientation: 49 G Johnson -----------------------------------------------
JOHNSON_49G_FILE_STEM = (
    "2025_11_15-14_11_49-johnson_204nv_s9-17d44b"
)

# ---- Manual override ----------------------------------------------------------
# Manual values take precedence over ESR/automatic assignments.
# Example:
# MANUAL_ORIENTATION_MAP = {16: (-1,1,1), 25: (1,1,-1)}
MANUAL_ORIENTATION_MAP = {}

# Hard rule requested: do not compare a site from the wrong NV orientation.
REQUIRE_KNOWN_ORIENTATION = True

# =============================================================================
# ORIENTATION-POOL REFIT
# =============================================================================

# Use the correct-orientation cheap screen to nominate sites, then refit them
# equally. Existing correct-orientation deep candidates are added to the pool.
POOL_TOP_SCREEN_SITES = 16
POOL_MAX_UNIQUE_SITES = 28

# Every candidate receives the SAME broad oscillation-amplitude bounds.
POOL_AMP_BOUNDS = (-2.0, 2.0)

# Equal optimizer budget for every candidate.
POOL_REFIT_MAX_NFEV = 60_000

# Each candidate gets two equal-footing starts:
#   (1) standardized model seed
#   (2) its best saved seed from the exhaustive run
POOL_USE_TWO_STARTS = True

# Parallelize by NV.
CPU_COUNT = os.cpu_count() or 4
POOL_N_JOBS = max(1, min(18, CPU_COUNT - 2))
BLAS_THREADS_PER_WORKER = 1

# =============================================================================
# RANKING / CONFIDENCE
# =============================================================================

TOP_UNIQUE_SAVE = 10
TOP_UNIQUE_PLOT = 3

# Spectroscopically similar family threshold, within the SAME orientation.
# Non-transitive leader grouping is used.
FAMILY_TOL_KHZ = 2.0

# =============================================================================
# PDF DESIGN
# =============================================================================

DENSE_CURVE_POINTS = 3000

FIRST_REVIVAL_HALF_WIDTH_US = 12.5

# How many candidate carbon positions to show in the right panel.
POSITION_TOP_N = 8
AKAIKE_PLOT_TOP_N = 8

# Main page table: only the top hypotheses, but with the detailed fit parameters.
FIT_TABLE_TOP_N = 3

SAVE_MAIN_THREE_PANEL_PDF = True
SAVE_CONFIDENCE_TABLES = True
SAVE_GLOBAL_SUMMARY = True
SHOW_GLOBAL_SUMMARY = True

# None -> all NVs
PLOT_NV_INDICES = None

# =============================================================================
# OPTIONAL BOOTSTRAP / CV
# =============================================================================

RUN_BOOTSTRAP = True
BOOTSTRAP_NV_INDICES = [16]
BOOTSTRAP_N = 300
BOOTSTRAP_TOP_SITES = 5
BOOTSTRAP_MAX_NFEV = 30_000

RUN_CROSS_VALIDATION = True
CV_NV_INDICES = [16]
CV_REPEATS = 50
CV_TOP_SITES = 3
CV_TRAIN_FRACTION = 0.80
CV_MAX_NFEV = 30_000

POST_N_JOBS = max(1, min(12, CPU_COUNT - 2))
RANDOM_SEED = 20260922

# =============================================================================
# RESIDUAL SPECTROSCOPY
# =============================================================================

RUN_RESIDUAL_SPECTROSCOPY = True
RESIDUAL_FREQ_MIN_KHZ = 5.0
RESIDUAL_FREQ_MAX_KHZ = None
RESIDUAL_GRID_POINTS = 5000
RESIDUAL_N_PEAKS = 5


PARAM_NAMES = [
    "baseline",
    "comb_contrast",
    "revival_time_us",
    "width0_us",
    "T2_ms",
    "T2_exp",
    "amp_taper_alpha",
    "width_slope",
    "revival_chirp",
    "osc_amp",
    "osc_f0",
    "osc_phi0",
    "osc_f1",
    "osc_phi1",
]


# =============================================================================
# PATH DISCOVERY
# =============================================================================

@dataclass
class ResultPaths:
    all_attempts: Path
    checkpoint: Path
    prefix: Path


def newest_match(root: Path, pattern: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Search root does not exist: {root}")
    matches = list(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files matching {pattern!r} under {root}"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def discover_paths() -> ResultPaths:
    if ALL_ATTEMPTS_PATH is None:
        all_path = newest_match(
            SEARCH_ROOT,
            f"*{RESULT_TAG}_all_attempts.csv.gz",
        )
    else:
        all_path = Path(ALL_ATTEMPTS_PATH)

    suffix = "_all_attempts.csv.gz"
    s = str(all_path)
    if not s.endswith(suffix):
        raise ValueError(
            f"Expected filename ending {suffix}; got {all_path}"
        )

    prefix = Path(s[: -len(suffix)])
    checkpoint = (
        Path(CHECKPOINT_PATH)
        if CHECKPOINT_PATH is not None
        else Path(str(prefix) + "_fit_checkpoint.npz")
    )

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    return ResultPaths(
        all_attempts=all_path,
        checkpoint=checkpoint,
        prefix=prefix,
    )


# =============================================================================
# BASIC HELPERS
# =============================================================================

def canonical_orientation(value):
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return None

    try:
        a = np.asarray(value, int).ravel()
    except Exception:
        return None

    if a.size != 3:
        return None

    return tuple(int(v) for v in a)


def orientation_str(value):
    ori = canonical_orientation(value)
    return str(ori) if ori is not None else ""


def parse_popt(s):
    try:
        p = np.asarray(json.loads(s), float)
        if (
            p.ndim == 1
            and len(p) == len(PARAM_NAMES)
            and np.all(np.isfinite(p))
        ):
            return p
    except Exception:
        pass
    return None


def fit_stats(y, e, pred, npar):
    y = np.asarray(y, float)
    e = np.maximum(np.asarray(e, float), 1e-12)
    pred = np.asarray(pred, float)

    chi2 = float(np.sum(((y - pred) / e) ** 2))

    n = len(y)
    k = int(npar)
    dof = max(1, n - k)

    red = chi2 / dof
    aic = chi2 + 2 * k

    if n > k + 1:
        aicc = (
            aic
            + 2 * k * (k + 1) / (n - k - 1)
        )
    else:
        aicc = np.inf

    return chi2, red, float(aicc)


def curve_from_popt(t, popt):
    return np.asarray(
        fine_decay(np.asarray(t, float), *np.asarray(popt, float)),
        float,
    )


def finite_success(df):
    out = df[df["status"].astype(str) == "ok"].copy()

    for col in ("aicc", "red_chi2", "score_primary"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[
        np.isfinite(out["aicc"])
        & np.isfinite(out["red_chi2"])
        & np.isfinite(out["score_primary"])
    ].copy()

    out["orientation_tuple"] = out["orientation"].map(
        canonical_orientation
    )
    out = out[out["orientation_tuple"].notna()].copy()

    out["site_id"] = pd.to_numeric(
        out["site_id"],
        errors="coerce",
    )
    out = out[np.isfinite(out["site_id"])].copy()
    out["site_id"] = out["site_id"].astype(int)

    return out


# =============================================================================
# LOAD EXHAUSTIVE RESULTS
# =============================================================================

def load_inputs(paths):
    print("=" * 96)
    print("SPIN-ECHO ORIENTATION-LOCKED CONFIDENCE ANALYSIS V6")
    print("=" * 96)
    print(f"result tag   : {RESULT_TAG}")
    print(f"all attempts : {paths.all_attempts}")
    print(f"checkpoint   : {paths.checkpoint}")

    attempts = pd.read_csv(paths.all_attempts)

    ckpt = np.load(
        paths.checkpoint,
        allow_pickle=True,
    )

    t = np.asarray(ckpt["times_us"], float)
    y = np.asarray(ckpt["norm_counts"], float)
    e = np.asarray(ckpt["norm_counts_ste"], float)

    expected_revival = np.nan

    if "expected_revival_total_us" in ckpt:
        arr = np.asarray(
            ckpt["expected_revival_total_us"],
            float,
        ).ravel()

        if arr.size:
            expected_revival = float(arr[0])

    print(f"attempt rows : {len(attempts):,}")
    print(f"data shape   : {y.shape}")

    return (
        attempts,
        ckpt,
        t,
        y,
        e,
        expected_revival,
    )


# =============================================================================
# KNOWN NV ORIENTATION
# =============================================================================

def assign_52g_orientations_from_resonance():
    """
    Infer each QNami NV orientation by JOINT matching of its measured ESR
    doublet (f1,f2) to the four known orientation doublets.

    This is much stronger than choosing f1 and f2 independently.
    """
    csv_path = newest_match(
        RESONANCE_ANALYSIS_ROOT,
        "fit_parameters_filtered.csv",
    )

    df = pd.read_csv(csv_path)

    if not {"nv_index", "f1", "f2"}.issubset(df.columns):
        raise ValueError(
            f"Resonance CSV lacks nv_index/f1/f2: {csv_path}"
        )

    target_items = list(
        ESR_ORIENTATION_TARGETS_GHZ.items()
    )

    mapping = {}
    rows = []

    for _, row in df.iterrows():
        nv = int(row["nv_index"])
        f1 = float(row["f1"])
        f2 = float(row["f2"])

        measured = np.sort([f1, f2])

        scored = []

        for ori, pair in target_items:
            target = np.sort(
                np.asarray(pair, float)
            )

            err_mhz = 1000.0 * (
                measured - target
            )

            rms = float(
                np.sqrt(
                    np.mean(
                        err_mhz**2
                    )
                )
            )

            scored.append(
                (
                    rms,
                    tuple(ori),
                    target,
                    err_mhz,
                )
            )

        scored.sort(
            key=lambda x: x[0]
        )

        rms, ori, target, err_mhz = (
            scored[0]
        )

        mapping[nv] = tuple(ori)

        rows.append(
            {
                "nv_index": nv,
                "orientation": str(
                    tuple(ori)
                ),
                "measured_f1_GHz": float(
                    measured[0]
                ),
                "measured_f2_GHz": float(
                    measured[1]
                ),
                "target_f1_GHz": float(
                    target[0]
                ),
                "target_f2_GHz": float(
                    target[1]
                ),
                "orientation_rms_error_MHz":
                    rms,
                "orientation_warning":
                    bool(
                        rms
                        > ORIENTATION_WARN_RMS_MHZ
                    ),
            }
        )

    quality = pd.DataFrame(rows)

    print(
        f"[orientation] QNami ESR mapping: "
        f"{len(mapping)} NVs from {csv_path}"
    )

    bad = int(
        quality[
            "orientation_warning"
        ].sum()
    )

    print(
        f"[orientation] assignments with RMS error "
        f">{ORIENTATION_WARN_RMS_MHZ:g} MHz: "
        f"{bad}"
    )

    return mapping, quality, csv_path


def load_49g_johnson_orientations():
    from utils import data_manager as dm

    data = dm.get_raw_data(
        file_stem=JOHNSON_49G_FILE_STEM,
        load_npz=True,
    )

    arr = np.asarray(
        data["orientations"],
        int,
    )

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            "49 G orientations array is not N x 3."
        )

    mapping = {}
    rows = []

    for nv in range(arr.shape[0]):
        ori = canonical_orientation(
            arr[nv]
        )

        if ori is None or ori == (0, 0, 0):
            continue

        mapping[int(nv)] = ori

        rows.append(
            {
                "nv_index": int(nv),
                "orientation": str(ori),
                "orientation_rms_error_MHz":
                    np.nan,
                "orientation_warning":
                    False,
            }
        )

    print(
        f"[orientation] Johnson saved orientation map: "
        f"{len(mapping)} NVs"
    )

    return mapping, pd.DataFrame(rows), None


def load_orientation_map():
    if RESULT_TAG.endswith("52G"):
        if not AUTO_ASSIGN_52G_ORIENTATION_FROM_RESONANCE:
            mapping = {}
            quality = pd.DataFrame()
            src = None
        else:
            mapping, quality, src = (
                assign_52g_orientations_from_resonance()
            )
    else:
        mapping, quality, src = (
            load_49g_johnson_orientations()
        )

    # Manual overrides win.
    for nv, ori in MANUAL_ORIENTATION_MAP.items():
        c = canonical_orientation(ori)

        if c is not None:
            mapping[int(nv)] = c

    return mapping, quality, src


# =============================================================================
# LOAD C13 CATALOG / POSITIONS
# =============================================================================

def load_catalog():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG_PATH}"
        )

    with open(
        CATALOG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        records = json.load(f)

    allowed = {tuple(int(v) for v in ori) for ori in ALLOWED_ORIENTATIONS}
    records = [
        rec for rec in records
        if canonical_orientation(rec.get("orientation")) in allowed
    ]

    lookup = {}

    for rec in records:
        ori = canonical_orientation(
            rec.get("orientation")
        )

        site = int(
            rec.get(
                "site_index",
                -1,
            )
        )

        if ori is None or site < 0:
            continue

        lookup[
            (
                ori,
                site,
            )
        ] = rec

    print(
        f"[catalog] records={len(records):,}, "
        f"lookup keys={len(lookup):,}"
    )

    return records, lookup


# =============================================================================
# ORIENTATION-SPECIFIC CANDIDATE POOL
# =============================================================================

def best_saved_row_per_site(
    nv_attempts,
):
    """
    One best saved attempt per physical C13 hypothesis:
        (orientation, site_id)
    restricted to the two allowed orientations.
    """
    if nv_attempts.empty:
        return pd.DataFrame()

    return (
        nv_attempts.sort_values(
            [
                "orientation_tuple",
                "site_id",
                "aicc",
                "red_chi2",
                "score_primary",
            ],
            kind="stable",
        )
        .groupby(
            ["orientation_tuple", "site_id"],
            as_index=False,
        )
        .first()
    )


def make_orientation_locked_candidate_pool(
    attempts,
    nv,
    orientation,
):
    good = finite_success(
        attempts[
            attempts["nv_index"]
            == int(nv)
        ]
    )

    orientation = tuple(int(v) for v in orientation)

    if orientation not in {tuple(int(v) for v in ori) for ori in ALLOWED_ORIENTATIONS}:
        raise ValueError(
            f"NV {nv}: orientation {orientation} is not one of "
            f"ALLOWED_ORIENTATIONS={ALLOWED_ORIENTATIONS}"
        )

    # HARD PHYSICAL CONSTRAINT:
    # only carbon sites from this NV's independently known orientation.
    good = good[
        good["orientation_tuple"] == orientation
    ].copy()

    if good.empty:
        return pd.DataFrame()

    screen = best_saved_row_per_site(
        good[
            good["stage"].astype(str)
            == "screen"
        ]
    )

    deep = best_saved_row_per_site(
        good[
            good["stage"].astype(str)
            .isin(
                [
                    "multistart",
                    "refine",
                ]
            )
        ]
    )

    # Nominate best screen candidates from the CORRECT orientation only.
    screen = screen.sort_values(
        [
            "aicc",
            "red_chi2",
        ]
    ).head(
        int(
            POOL_TOP_SCREEN_SITES
        )
    )

    # Union screen nominees + every already-deep candidate in correct axis.
    by_site = {}

    for _, row in screen.iterrows():
        key = (
            tuple(row["orientation_tuple"]),
            int(row["site_id"]),
        )
        by_site[key] = row

    for _, row in deep.iterrows():
        key = (
            tuple(row["orientation_tuple"]),
            int(row["site_id"]),
        )

        old = by_site.get(key)

        if (
            old is None
            or float(row["aicc"]) < float(old["aicc"])
        ):
            by_site[key] = row

    candidates = pd.DataFrame(
        [
            r.to_dict()
            for r in by_site.values()
        ]
    )

    if candidates.empty:
        return candidates

    candidates = (
        candidates
        .sort_values(
            [
                "aicc",
                "red_chi2",
            ]
        )
        .head(
            int(
                POOL_MAX_UNIQUE_SITES
            )
        )
        .reset_index(
            drop=True
        )
    )

    return candidates


# =============================================================================
# EQUAL-FOOTING REFIT
# =============================================================================

def standardized_vectors(
    t,
    y,
    row,
    expected_revival,
    warm=None,
):
    p0, lb, ub = (
        oldfit._initial_guess_and_bounds(
            np.asarray(t, float),
            np.asarray(y, float),
            enable_extras=True,
            fixed_rev_time=None,
        )
    )

    pmap = (
        oldfit._param_index_map(
            fine_decay
        )
    )

    oldfit._set_osc_amp_bounds(
        lb,
        ub,
        fine_decay,
        float(
            POOL_AMP_BOUNDS[0]
        ),
        float(
            POOL_AMP_BOUNDS[1]
        ),
    )

    # Use saved catalog frequencies, locked tightly.
    f0 = (
        float(
            row["f0_kHz"]
        )
        / 1000.0
    )

    f1 = (
        float(
            row["f1_kHz"]
        )
        / 1000.0
    )

    freq_eps = 1e-6

    for name, val in (
        ("osc_f0", f0),
        ("osc_f1", f1),
    ):
        ind = pmap[name]

        p0[ind] = val
        lb[ind] = (
            val - freq_eps
        )
        ub[ind] = (
            val + freq_eps
        )

    if (
        np.isfinite(
            expected_revival
        )
        and "revival_time"
        in pmap
    ):
        ind = pmap[
            "revival_time"
        ]

        p0[ind] = np.clip(
            float(
                expected_revival
            ),
            lb[ind]
            + 1e-6,
            ub[ind]
            - 1e-6,
        )

    if warm is not None:
        w = np.asarray(
            warm,
            float,
        )

        if w.shape == p0.shape:
            p0 = w.copy()

            p0[
                pmap["osc_f0"]
            ] = f0

            p0[
                pmap["osc_f1"]
            ] = f1

    p0, lb, ub = (
        oldfit._retie_contrast_to_baseline(
            p0,
            lb,
            ub,
            pmap,
            eps=0.01,
        )
    )

    eps = (
        1e-9
        * np.maximum(
            1.0,
            ub - lb,
        )
    )

    p0 = np.minimum(
        np.maximum(
            p0,
            lb + eps,
        ),
        ub - eps,
    )

    return (
        p0,
        lb,
        ub,
    )


def one_equal_fit(
    t,
    y,
    e,
    row,
    expected_revival,
    warm,
):
    p0, lb, ub = (
        standardized_vectors(
            t,
            y,
            row,
            expected_revival,
            warm=warm,
        )
    )

    popt, _pcov, _red = (
        oldfit._fit_least_squares(
            fine_decay,
            np.asarray(
                t,
                float,
            ),
            np.asarray(
                y,
                float,
            ),
            np.maximum(
                np.asarray(
                    e,
                    float,
                ),
                1e-12,
            ),
            p0,
            lb,
            ub,
            max_nfev=int(
                POOL_REFIT_MAX_NFEV
            ),
        )
    )

    pred = curve_from_popt(
        t,
        popt,
    )

    chi2, red, aicc = (
        fit_stats(
            y,
            e,
            pred,
            len(popt),
        )
    )

    # Preserve the old fitter's selection diagnostic as an explicit output.
    # This is NOT used as a probability; lower is better.
    pmap = oldfit._param_index_map(fine_decay)
    score_primary, score_amp_tie = oldfit._score_tuple(
        popt,
        red,
        lb,
        ub,
        pmap,
    )

    return (
        popt,
        pred,
        chi2,
        red,
        aicc,
        float(score_primary),
        float(score_amp_tie),
    )


def refit_candidate_equal_footing(
    nv,
    t,
    y,
    e,
    row,
    expected_revival,
):
    saved = parse_popt(
        row.get(
            "popt_json",
            "",
        )
    )

    starts = [
        (
            "standard",
            None,
        )
    ]

    if POOL_USE_TWO_STARTS:
        starts.append(
            (
                "saved",
                saved,
            )
        )

    results = []

    for start_name, warm in starts:
        try:
            (
                popt,
                pred,
                chi2,
                red,
                aicc,
                score_primary,
                score_amp_tie,
            ) = one_equal_fit(
                t,
                y,
                e,
                row,
                expected_revival,
                warm,
            )

            results.append(
                {
                    "start_name":
                        start_name,
                    "popt":
                        popt,
                    "pred":
                        pred,
                    "chi2":
                        chi2,
                    "red_chi2":
                        red,
                    "aicc":
                        aicc,
                    "score_primary":
                        score_primary,
                    "score_amp_tie":
                        score_amp_tie,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "start_name":
                        start_name,
                    "error":
                        str(exc),
                }
            )

    good = [
        r
        for r in results
        if "aicc" in r
        and np.isfinite(
            r["aicc"]
        )
    ]

    if not good:
        return {
            "nv_index":
                int(nv),
            "orientation":
                str(
                    row[
                        "orientation_tuple"
                    ]
                ),
            "site_id":
                int(
                    row[
                        "site_id"
                    ]
                ),
            "status":
                "fail",
        }

    good.sort(
        key=lambda r: (
            r["aicc"],
            r["red_chi2"],
        )
    )

    best = good[0]

    out = {
        "nv_index":
            int(nv),
        "orientation":
            str(
                row[
                    "orientation_tuple"
                ]
            ),
        "site_id":
            int(
                row[
                    "site_id"
                ]
            ),
        "status":
            "ok",
        "winning_start":
            best[
                "start_name"
            ],
        "f0_kHz":
            float(
                row[
                    "f0_kHz"
                ]
            ),
        "f1_kHz":
            float(
                row[
                    "f1_kHz"
                ]
            ),
        "kappa":
            float(
                row.get(
                    "kappa",
                    np.nan,
                )
            ),
        "distance_A":
            float(
                row.get(
                    "distance_A",
                    np.nan,
                )
            ),
        "chi2":
            float(
                best[
                    "chi2"
                ]
            ),
        "red_chi2":
            float(
                best[
                    "red_chi2"
                ]
            ),
        "aicc":
            float(
                best[
                    "aicc"
                ]
            ),
        "score_primary":
            float(
                best[
                    "score_primary"
                ]
            ),
        "score_amp_tie":
            float(
                best[
                    "score_amp_tie"
                ]
            ),
        "popt_json":
            json.dumps(
                np.asarray(
                    best[
                        "popt"
                    ],
                    float,
                ).tolist()
            ),
    }

    return out


def refit_one_nv_orientation_locked(
    nv,
    orientation,
    attempts,
    t,
    y,
    e,
    expected_revival,
):
    pool = (
        make_orientation_locked_candidate_pool(
            attempts,
            nv,
            orientation,
        )
    )

    if pool.empty:
        return {
            "nv_index": int(nv),
            "orientation": tuple(int(v) for v in orientation),
            "status": "no_candidates",
            "rows": [],
        }

    rows = []

    for _, row in pool.iterrows():
        rows.append(
            refit_candidate_equal_footing(
                nv,
                t,
                y[nv],
                e[nv],
                row,
                expected_revival,
            )
        )

    good = [
        r
        for r in rows
        if r.get(
            "status"
        )
        == "ok"
    ]

    good.sort(
        key=lambda r: (
            r["aicc"],
            r["red_chi2"],
        )
    )

    return {
        "nv_index": int(nv),
        "orientation": tuple(int(v) for v in orientation),
        "status": ("ok" if good else "failed"),
        "rows": rows,
    }


def run_orientation_locked_refits(
    attempts,
    orientation_map,
    t,
    y,
    e,
    expected_revival,
):
    """
    Equal-footing site refits, but with a hard per-NV orientation constraint.

    The orientation is NEVER inferred from the C13 fit itself.  For NV i, only
    catalog/attempt rows with orientation == orientation_map[i] are considered.
    """
    n_nv = y.shape[0]

    jobs = []
    missing = []

    for nv in range(n_nv):
        ori = orientation_map.get(int(nv))
        if ori is None:
            missing.append(int(nv))
            continue
        jobs.append((int(nv), tuple(int(v) for v in ori)))

    if missing and REQUIRE_KNOWN_ORIENTATION:
        print(
            f"[orientation] {len(missing)} NVs lack an orientation and will "
            f"be skipped: {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )

    print()
    print("=" * 96)
    print("EQUAL-FOOTING REFIT WITH PER-NV ORIENTATION LOCK")
    print("=" * 96)
    print(f"allowed experiment orientations : {ALLOWED_ORIENTATIONS}")
    print(f"NVs with assigned orientation   : {len(jobs)} / {n_nv}")
    print(f"top screen sites/orientation    : {POOL_TOP_SCREEN_SITES}")
    print(f"max sites/NV                    : {POOL_MAX_UNIQUE_SITES}")
    print(f"common amplitude bounds         : {POOL_AMP_BOUNDS}")
    print(f"two starts/candidate            : {POOL_USE_TWO_STARTS}")
    print(f"max_nfev/start                  : {POOL_REFIT_MAX_NFEV}")
    print(f"workers                          : {POOL_N_JOBS}")
    print("=" * 96)

    def task(nv, ori):
        result = refit_one_nv_orientation_locked(
            nv,
            ori,
            attempts,
            t,
            y,
            e,
            expected_revival,
        )

        good = [
            r for r in result["rows"]
            if r.get("status") == "ok"
        ]

        if good:
            good.sort(key=lambda r: (r["aicc"], r["red_chi2"]))
            b = good[0]

            # Safety assertion: wrong-orientation hypothesis must never survive.
            bad = [
                r for r in good
                if canonical_orientation(r["orientation"]) != tuple(ori)
            ]
            if bad:
                raise RuntimeError(
                    f"NV {nv}: wrong-orientation candidates leaked into fit: "
                    f"{[(x['orientation'], x['site_id']) for x in bad[:5]]}"
                )

            print(
                f"[NV {nv:3d}] locked_ori={ori}, "
                f"site={b['site_id']}, redchi={b['red_chi2']:.3g}, "
                f"sites={len(good)}"
            )

        return result

    with threadpool_limits(limits=BLAS_THREADS_PER_WORKER):
        results = Parallel(
            n_jobs=POOL_N_JOBS,
            backend="loky",
            batch_size=1,
            verbose=5,
        )(
            delayed(task)(nv, ori)
            for nv, ori in jobs
        )

    rows = []
    for result in results:
        expected_ori = tuple(result["orientation"])
        for row in result["rows"]:
            if row.get("status") != "ok":
                continue
            actual_ori = canonical_orientation(row["orientation"])
            if actual_ori != expected_ori:
                raise RuntimeError(
                    f"NV {result['nv_index']}: orientation mismatch "
                    f"{actual_ori} != {expected_ori}"
                )
            rows.append(row)

    if not rows:
        raise RuntimeError("No successful orientation-locked refits.")

    return pd.DataFrame(rows)


# =============================================================================
# RANK EQUAL-FOOTING UNIQUE SITES WITHIN THE LOCKED NV ORIENTATION
# =============================================================================

def rank_equal_footing_sites(
    refit_df,
):
    df = refit_df.copy()

    df["delta_aicc"] = np.nan
    df["delta_chi2"] = np.nan
    df["akaike_weight"] = np.nan
    df["site_rank"] = -1

    for nv, inds in df.groupby(
        "nv_index"
    ).groups.items():
        inds = np.asarray(
            list(inds),
            int,
        )

        aicc = df.loc[
            inds,
            "aicc",
        ].to_numpy(float)

        chi2 = df.loc[
            inds,
            "chi2",
        ].to_numpy(float)

        da = (
            aicc
            - np.min(
                aicc
            )
        )

        dc = (
            chi2
            - np.min(
                chi2
            )
        )

        logw = (
            -0.5
            * da
        )

        logw -= (
            np.max(
                logw
            )
        )

        w = np.exp(
            logw
        )

        w /= np.sum(
            w
        )

        order = np.argsort(
            aicc
        )

        rank = np.empty_like(
            order
        )

        rank[
            order
        ] = np.arange(
            1,
            len(order)
            + 1,
        )

        df.loc[
            inds,
            "delta_aicc",
        ] = da

        df.loc[
            inds,
            "delta_chi2",
        ] = dc

        df.loc[
            inds,
            "akaike_weight",
        ] = w

        df.loc[
            inds,
            "site_rank",
        ] = rank

    return df.sort_values(
        [
            "nv_index",
            "site_rank",
        ]
    ).reset_index(
        drop=True
    )


def expand_fit_parameter_columns(
    ranked,
):
    """
    Expand popt_json into human-readable columns.

    fine_decay parameter order:
      baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp,
      amp_taper_alpha, width_slope, revival_chirp, osc_amp,
      osc_f0, osc_phi0, osc_f1, osc_phi1
    """
    ranked = ranked.copy()

    cols = {
        "baseline": [],
        "comb_contrast": [],
        "revival_time_us": [],
        "width0_us": [],
        "T2_ms": [],
        "T2_us": [],
        "T2_exp": [],
        "amp_taper_alpha": [],
        "width_slope": [],
        "revival_chirp": [],
        "osc_amp": [],
        "fit_f0_kHz": [],
        "osc_phi0_rad": [],
        "fit_f1_kHz": [],
        "osc_phi1_rad": [],
    }

    for s in ranked["popt_json"]:
        p = parse_popt(s)

        if p is None:
            vals = [np.nan] * len(PARAM_NAMES)
        else:
            vals = list(map(float, p))

        cols["baseline"].append(vals[0])
        cols["comb_contrast"].append(vals[1])
        cols["revival_time_us"].append(vals[2])
        cols["width0_us"].append(vals[3])
        cols["T2_ms"].append(vals[4])
        cols["T2_us"].append(1000.0 * vals[4] if np.isfinite(vals[4]) else np.nan)
        cols["T2_exp"].append(vals[5])
        cols["amp_taper_alpha"].append(vals[6])
        cols["width_slope"].append(vals[7])
        cols["revival_chirp"].append(vals[8])
        cols["osc_amp"].append(vals[9])
        cols["fit_f0_kHz"].append(1000.0 * vals[10] if np.isfinite(vals[10]) else np.nan)
        cols["osc_phi0_rad"].append(vals[11])
        cols["fit_f1_kHz"].append(1000.0 * vals[12] if np.isfinite(vals[12]) else np.nan)
        cols["osc_phi1_rad"].append(vals[13])

    for name, values in cols.items():
        ranked[name] = values

    return ranked


# =============================================================================
# NON-TRANSITIVE SITE FAMILIES, SAME ORIENTATION ONLY
# =============================================================================

def assign_families(
    ranked,
):
    """
    Non-transitive spectral-family grouping within each NV AND orientation.
    Sites from the two different NV orientations are never placed in the same
    family.
    """
    ranked = ranked.copy()
    ranked["family_id"] = -1
    ranked["family_weight"] = np.nan
    ranked["family_rank"] = -1

    family_rows = []

    for nv, nvsub in ranked.groupby("nv_index", sort=True):
        next_id = 0
        nv_families = []

        for ori, sub in nvsub.groupby("orientation", sort=False):
            sub = sub.sort_values("site_rank")
            leaders = []

            for idx, row in sub.iterrows():
                f0 = float(row["f0_kHz"])
                f1 = float(row["f1_kHz"])

                chosen = None
                for fam in leaders:
                    if (
                        abs(f0 - fam["f0"]) <= FAMILY_TOL_KHZ
                        and abs(f1 - fam["f1"]) <= FAMILY_TOL_KHZ
                    ):
                        chosen = fam
                        break

                if chosen is None:
                    chosen = {
                        "family_id": next_id,
                        "f0": f0,
                        "f1": f1,
                        "leader_site_id": int(row["site_id"]),
                        "orientation": str(ori),
                        "members": [],
                    }
                    leaders.append(chosen)
                    next_id += 1

                chosen["members"].append(idx)
                ranked.loc[idx, "family_id"] = int(chosen["family_id"])

            for fam in leaders:
                members = ranked.loc[fam["members"]]
                nv_families.append(
                    {
                        "nv_index": int(nv),
                        "family_id": int(fam["family_id"]),
                        "orientation": fam["orientation"],
                        "leader_site_id": int(fam["leader_site_id"]),
                        "leader_f0_kHz": float(fam["f0"]),
                        "leader_f1_kHz": float(fam["f1"]),
                        "n_sites": int(len(members)),
                        "family_weight": float(members["akaike_weight"].sum()),
                        "best_delta_aicc": float(members["delta_aicc"].min()),
                    }
                )

        famdf = pd.DataFrame(nv_families).sort_values(
            ["family_weight", "best_delta_aicc"],
            ascending=[False, True],
        ).reset_index(drop=True)

        famdf["family_rank"] = np.arange(1, len(famdf) + 1)

        rank_map = dict(zip(famdf["family_id"], famdf["family_rank"]))
        weight_map = dict(zip(famdf["family_id"], famdf["family_weight"]))

        inds = nvsub.index
        ranked.loc[inds, "family_rank"] = [
            rank_map[int(x)] for x in ranked.loc[inds, "family_id"]
        ]
        ranked.loc[inds, "family_weight"] = [
            weight_map[int(x)] for x in ranked.loc[inds, "family_id"]
        ]

        family_rows.append(famdf)

    families = (
        pd.concat(family_rows, ignore_index=True)
        if family_rows
        else pd.DataFrame()
    )
    return ranked, families


# =============================================================================
# SUMMARY
# =============================================================================

def separation_label(
    delta_aicc,
):
    if not np.isfinite(
        delta_aicc
    ):
        return (
            "single_site"
        )

    if delta_aicc < 2:
        return (
            "near_degenerate"
        )

    if delta_aicc < 6:
        return (
            "some_separation"
        )

    if delta_aicc < 10:
        return (
            "substantial_separation"
        )

    return (
        "large_separation"
    )


def build_summary(
    ranked,
):
    rows = []

    for nv, sub in ranked.groupby(
        "nv_index",
        sort=True,
    ):
        sub = sub.sort_values(
            "site_rank"
        )

        first = sub.iloc[0]

        second = (
            sub.iloc[1]
            if len(sub) > 1
            else None
        )

        third = (
            sub.iloc[2]
            if len(sub) > 2
            else None
        )

        d2 = (
            float(
                second[
                    "delta_aicc"
                ]
            )
            if second
            is not None
            else np.nan
        )

        rows.append(
            {
                "nv_index":
                    int(nv),
                "orientation":
                    str(
                        first[
                            "orientation"
                        ]
                    ),
                "best_site_id":
                    int(
                        first[
                            "site_id"
                        ]
                    ),
                "best_red_chi2":
                    float(
                        first[
                            "red_chi2"
                        ]
                    ),
                "best_score_primary":
                    float(
                        first[
                            "score_primary"
                        ]
                    ),
                "best_aicc":
                    float(
                        first[
                            "aicc"
                        ]
                    ),
                "best_T2_us":
                    float(
                        first.get(
                            "T2_us",
                            np.nan,
                        )
                    ),
                "best_revival_time_us":
                    float(
                        first.get(
                            "revival_time_us",
                            np.nan,
                        )
                    ),
                "best_width0_us":
                    float(
                        first.get(
                            "width0_us",
                            np.nan,
                        )
                    ),
                "best_T2_exp":
                    float(
                        first.get(
                            "T2_exp",
                            np.nan,
                        )
                    ),
                "best_osc_amp":
                    float(
                        first.get(
                            "osc_amp",
                            np.nan,
                        )
                    ),
                "best_weight":
                    float(
                        first[
                            "akaike_weight"
                        ]
                    ),
                "best_family_weight":
                    float(
                        first[
                            "family_weight"
                        ]
                    ),
                "best_f0_kHz":
                    float(
                        first[
                            "f0_kHz"
                        ]
                    ),
                "best_f1_kHz":
                    float(
                        first[
                            "f1_kHz"
                        ]
                    ),
                "best_kappa":
                    float(
                        first[
                            "kappa"
                        ]
                    ),
                "best_distance_A":
                    float(
                        first[
                            "distance_A"
                        ]
                    ),
                "second_site_id":
                    (
                        int(
                            second[
                                "site_id"
                            ]
                        )
                        if second
                        is not None
                        else -1
                    ),
                "second_delta_aicc":
                    d2,
                "third_site_id":
                    (
                        int(
                            third[
                                "site_id"
                            ]
                        )
                        if third
                        is not None
                        else -1
                    ),
                "separation_label":
                    separation_label(
                        d2
                    ),
                "n_two_orientation_hypotheses":
                    int(
                        len(
                            sub
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# C13 POSITION LOOKUP / VISUALIZATION
# =============================================================================

def add_catalog_positions(
    ranked,
    catalog_lookup,
):
    ranked = ranked.copy()

    for col in (
        "x_A",
        "y_A",
        "z_A",
        "A_par_kHz",
        "A_perp_kHz",
        "theta_deg",
    ):
        ranked[col] = np.nan

    for idx, row in ranked.iterrows():
        ori = canonical_orientation(
            row[
                "orientation"
            ]
        )

        site = int(
            row[
                "site_id"
            ]
        )

        rec = catalog_lookup.get(
            (
                ori,
                site,
            )
        )

        if rec is None:
            continue

        ranked.loc[
            idx,
            "x_A",
        ] = float(
            rec.get(
                "x_A",
                np.nan,
            )
        )

        ranked.loc[
            idx,
            "y_A",
        ] = float(
            rec.get(
                "y_A",
                np.nan,
            )
        )

        ranked.loc[
            idx,
            "z_A",
        ] = float(
            rec.get(
                "z_A",
                np.nan,
            )
        )

        ranked.loc[
            idx,
            "A_par_kHz",
        ] = float(
            rec.get(
                "A_par_Hz",
                np.nan,
            )
        ) / 1000.0

        ranked.loc[
            idx,
            "A_perp_kHz",
        ] = float(
            rec.get(
                "A_perp_Hz",
                np.nan,
            )
        ) / 1000.0

        ranked.loc[
            idx,
            "theta_deg",
        ] = float(
            rec.get(
                "theta_deg",
                np.nan,
            )
        )

    return ranked


def set_3d_equal_limits(
    ax,
    xyz,
):
    xyz = np.asarray(
        xyz,
        float,
    )

    finite = np.all(
        np.isfinite(
            xyz
        ),
        axis=1,
    )

    xyz = xyz[
        finite
    ]

    if not len(
        xyz
    ):
        return

    xyz = np.vstack(
        [
            xyz,
            np.zeros(
                (
                    1,
                    3,
                )
            ),
        ]
    )

    mins = np.min(
        xyz,
        axis=0,
    )

    maxs = np.max(
        xyz,
        axis=0,
    )

    center = (
        0.5
        * (
            mins
            + maxs
        )
    )

    radius = (
        0.55
        * np.max(
            maxs
            - mins
        )
    )

    radius = max(
        radius,
        1.0,
    )

    ax.set_xlim(
        center[0]
        - radius,
        center[0]
        + radius,
    )

    ax.set_ylim(
        center[1]
        - radius,
        center[1]
        + radius,
    )

    ax.set_zlim(
        center[2]
        - radius,
        center[2]
        + radius,
    )


# =============================================================================
# MAIN ANALYSIS DASHBOARD PDF
# =============================================================================

def plot_one_nv_page(
    pdf,
    nv,
    t,
    y,
    e,
    ranked,
    expected_revival,
):
    """
    One-page analysis dashboard.

    Top row:
      full trace | first-revival zoom | Akaike weights

    Bottom row:
      3D C13 positions | detailed top-fit table spanning two columns
    """
    sub = (
        ranked[ranked["nv_index"] == int(nv)]
        .sort_values("site_rank")
        .copy()
    )

    if sub.empty:
        return

    cmap = plt.get_cmap("tab10")
    colors = {
        rank: cmap((rank - 1) % 10)
        for rank in range(1, max(POSITION_TOP_N, AKAIKE_PLOT_TOP_N) + 1)
    }

    # All candidates on this page belong to the same locked NV orientation.

    fig = plt.figure(figsize=(20.5, 11.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.08],
        width_ratios=[1.12, 1.06, 1.0],
        hspace=0.28,
        wspace=0.25,
    )

    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_weight = fig.add_subplot(gs[0, 2])
    ax_pos = fig.add_subplot(gs[1, 0], projection="3d")
    ax_table = fig.add_subplot(gs[1, 1:3])

    top3 = sub.head(TOP_UNIQUE_PLOT)

    # ------------------------------------------------------------------
    # Full trace
    # ------------------------------------------------------------------
    ax_full.errorbar(
        t,
        y[nv],
        yerr=e[nv],
        fmt="o",
        ms=3.0,
        capsize=1.0,
        lw=0.55,
        label="data",
        zorder=10,
    )

    td_full = np.linspace(float(np.min(t)), float(np.max(t)), DENSE_CURVE_POINTS)

    for _, row in top3.iterrows():
        rank = int(row["site_rank"])
        popt = parse_popt(row["popt_json"])
        if popt is None:
            continue

        curve = curve_from_popt(td_full, popt)

        ax_full.plot(
            td_full,
            curve,
            lw=1.9 if rank == 1 else 1.35,
            color=colors[rank],
            label=(
                f"#{rank} {row['orientation']}, site {int(row['site_id'])}\n"
                f"χ²r={row['red_chi2']:.2f}, score={row['score_primary']:.2f}"
            ),
        )

    ax_full.set_xlabel("Total evolution time (µs)")
    ax_full.set_ylabel("Normalized signal")
    ax_full.set_title("Full spin-echo trace")
    ax_full.grid(alpha=0.22)
    ax_full.legend(fontsize=7.0, loc="best")

    # ------------------------------------------------------------------
    # First-revival zoom
    # ------------------------------------------------------------------
    if np.isfinite(expected_revival):
        lo = expected_revival - FIRST_REVIVAL_HALF_WIDTH_US
        hi = expected_revival + FIRST_REVIVAL_HALF_WIDTH_US
    else:
        center = float(np.median(t))
        lo = center - FIRST_REVIVAL_HALF_WIDTH_US
        hi = center + FIRST_REVIVAL_HALF_WIDTH_US

    mask = (t >= lo) & (t <= hi)

    if np.sum(mask) < 3:
        mask = np.ones(len(t), dtype=bool)
        lo = float(np.min(t))
        hi = float(np.max(t))

    ax_zoom.errorbar(
        t[mask],
        y[nv, mask],
        yerr=e[nv, mask],
        fmt="o",
        ms=3.2,
        capsize=1.1,
        lw=0.6,
        zorder=10,
    )

    td_zoom = np.linspace(lo, hi, DENSE_CURVE_POINTS)

    for _, row in top3.iterrows():
        rank = int(row["site_rank"])
        popt = parse_popt(row["popt_json"])
        if popt is None:
            continue

        ax_zoom.plot(
            td_zoom,
            curve_from_popt(td_zoom, popt),
            lw=1.9 if rank == 1 else 1.35,
            color=colors[rank],
            label=f"#{rank} site {int(row['site_id'])}",
        )

    if np.isfinite(expected_revival):
        ax_zoom.axvline(
            expected_revival,
            ls="--",
            lw=0.8,
            alpha=0.6,
            label="13C revival",
        )

    ax_zoom.set_xlabel("Total evolution time (µs)")
    ax_zoom.set_ylabel("Normalized signal")
    ax_zoom.set_title("First-revival zoom")
    ax_zoom.grid(alpha=0.22)
    ax_zoom.legend(fontsize=7.0, loc="best")

    # ------------------------------------------------------------------
    # Akaike-weight plot
    # ------------------------------------------------------------------
    weight_sub = sub.head(AKAIKE_PLOT_TOP_N).copy()
    xpos = np.arange(len(weight_sub))

    bar_colors = [
        colors[int(r)]
        for r in weight_sub["site_rank"]
    ]

    bars = ax_weight.bar(
        xpos,
        weight_sub["akaike_weight"].to_numpy(float),
        color=bar_colors,
        alpha=0.82,
    )

    # Hatch distinguishes the two NV orientation pools.
    for bar, (_, row) in zip(bars, weight_sub.iterrows()):
        ori = canonical_orientation(row["orientation"])
        if ori == tuple(ALLOWED_ORIENTATIONS[1]):
            bar.set_hatch("//")

    labels = [
        (
            f"#{int(row['site_rank'])}\n"
            f"{tuple(canonical_orientation(row['orientation']))}\n"
            f"S{int(row['site_id'])}"
        )
        for _, row in weight_sub.iterrows()
    ]

    ax_weight.set_xticks(xpos)
    ax_weight.set_xticklabels(labels, fontsize=6.5)
    ax_weight.set_ylabel("Akaike weight")
    ax_weight.set_ylim(
        0.0,
        max(
            1.0,
            1.10 * float(weight_sub["akaike_weight"].max()),
        ),
    )
    ax_weight.set_title("Relative site support\n(within locked orientation)")
    ax_weight.grid(alpha=0.20, axis="y")

    for x, (_, row) in enumerate(weight_sub.iterrows()):
        w = float(row["akaike_weight"])
        ax_weight.text(
            x,
            min(0.98, w + 0.025),
            f"{w:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # ------------------------------------------------------------------
    # C13 candidate positions
    # ------------------------------------------------------------------
    pos = sub.head(POSITION_TOP_N).copy()
    xyz_all = []

    ax_pos.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="*",
        s=210,
        c="black",
        label="NV",
        depthshade=False,
    )

    for _, row in pos.iterrows():
        rank = int(row["site_rank"])
        ori = canonical_orientation(row["orientation"])

        xyz = np.asarray(
            [
                row.get("x_A", np.nan),
                row.get("y_A", np.nan),
                row.get("z_A", np.nan),
            ],
            float,
        )

        if not np.all(np.isfinite(xyz)):
            continue

        xyz_all.append(xyz)

        ax_pos.plot(
            [0.0, xyz[0]],
            [0.0, xyz[1]],
            [0.0, xyz[2]],
            lw=0.8,
            color=colors[rank],
            alpha=0.50,
        )

        marker = "o"

        ax_pos.scatter(
            [xyz[0]],
            [xyz[1]],
            [xyz[2]],
            s=105 if rank <= 3 else 55,
            color=colors[rank],
            marker=marker,
            depthshade=False,
            label=(
                f"#{rank} {ori}, site {int(row['site_id'])}"
            ),
        )

        ax_pos.text(
            xyz[0],
            xyz[1],
            xyz[2],
            f" #{rank}:S{int(row['site_id'])}",
            fontsize=7,
        )

    if xyz_all:
        set_3d_equal_limits(ax_pos, np.vstack(xyz_all))

    ax_pos.set_xlabel("x (Å)")
    ax_pos.set_ylabel("y (Å)")
    ax_pos.set_zlabel("z (Å)")
    locked_ori = canonical_orientation(sub.iloc[0]["orientation"])
    ax_pos.set_title(
        "Candidate $^{13}$C positions\n"
        f"locked NV orientation = {locked_ori}"
    )
    ax_pos.view_init(elev=24, azim=38)
    ax_pos.legend(fontsize=6.0, loc="upper left")

    # ------------------------------------------------------------------
    # Detailed top-fit table
    # ------------------------------------------------------------------
    ax_table.axis("off")

    detail = sub.head(FIT_TABLE_TOP_N)

    col_labels = [
        "Rank",
        "Orientation",
        "Site",
        "χ²r",
        "Score",
        "ΔAICc",
        "Weight",
        "T2 (µs)",
        "Revival (µs)",
        "Width (µs)",
        "β",
        "Osc amp",
        "f0 / f1 (kHz)",
        "κ",
        "r (Å)",
    ]

    cell_text = []

    for _, row in detail.iterrows():
        cell_text.append(
            [
                f"#{int(row['site_rank'])}",
                str(row["orientation"]),
                str(int(row["site_id"])),
                f"{float(row['red_chi2']):.3f}",
                f"{float(row['score_primary']):.3f}",
                f"{float(row['delta_aicc']):.2f}",
                f"{float(row['akaike_weight']):.3f}",
                f"{float(row.get('T2_us', np.nan)):.1f}",
                f"{float(row.get('revival_time_us', np.nan)):.3f}",
                f"{float(row.get('width0_us', np.nan)):.3f}",
                f"{float(row.get('T2_exp', np.nan)):.3f}",
                f"{float(row.get('osc_amp', np.nan)):.3f}",
                (
                    f"{float(row.get('fit_f0_kHz', np.nan)):.2f} / "
                    f"{float(row.get('fit_f1_kHz', np.nan)):.2f}"
                ),
                f"{float(row.get('kappa', np.nan)):.3g}",
                f"{float(row.get('distance_A', np.nan)):.2f}",
            ]
        )

    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.42, 1.0, 0.54],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.35)

    # Additional best-fit details beneath table.
    first = sub.iloc[0]

    extra_lines = [
        (
            f"Best-fit model details: baseline={first.get('baseline', np.nan):.4f}, "
            f"comb contrast={first.get('comb_contrast', np.nan):.4f}, "
            f"taper α={first.get('amp_taper_alpha', np.nan):.3f}, "
            f"width slope={first.get('width_slope', np.nan):.3f}, "
            f"revival chirp={first.get('revival_chirp', np.nan):.4f}"
        ),
        (
            f"phases: φ0={first.get('osc_phi0_rad', np.nan):.3f} rad, "
            f"φ1={first.get('osc_phi1_rad', np.nan):.3f} rad | "
            f"A∥={first.get('A_par_kHz', np.nan):.2f} kHz, "
            f"A⊥={first.get('A_perp_kHz', np.nan):.2f} kHz, "
            f"θ={first.get('theta_deg', np.nan):.1f}°"
        ),
        (
            "Score = old fitter's reduced-χ² + bound-wall penalty; "
            "Akaike weight is the relative model support across the equal-footing "
            "orientation-locked candidate set."
        ),
    ]

    ax_table.text(
        0.01,
        0.32,
        "\n".join(extra_lines),
        transform=ax_table.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        family="monospace",
    )

    fig.suptitle(
        (
            f"NV {nv} | best {first['orientation']} / site {int(first['site_id'])} | "
            f"χ²r={first['red_chi2']:.3f} | score={first['score_primary']:.3f} | "
            f"weight={first['akaike_weight']:.3f} | "
            f"T2={first.get('T2_us', np.nan):.1f} µs | "
            f"revival={first.get('revival_time_us', np.nan):.3f} µs"
        ),
        fontsize=13,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def save_dashboard_pdf(
    path,
    t,
    y,
    e,
    ranked,
    expected_revival,
):
    if PLOT_NV_INDICES is None:
        nvs = sorted(
            ranked[
                "nv_index"
            ].astype(
                int
            ).unique()
        )
    else:
        allowed = set(
            int(x)
            for x in PLOT_NV_INDICES
        )

        nvs = [
            nv
            for nv in sorted(
                ranked[
                    "nv_index"
                ].astype(
                    int
                ).unique()
            )
            if nv in allowed
        ]

    with PdfPages(
        path
    ) as pdf:
        for nv in nvs:
            plot_one_nv_page(
                pdf,
                nv,
                t,
                y,
                e,
                ranked,
                expected_revival,
            )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# RESIDUAL SPECTROSCOPY
# =============================================================================

def residual_peak_table(
    t,
    y,
    e,
    ranked,
):
    if not RUN_RESIDUAL_SPECTROSCOPY:
        return pd.DataFrame()

    dt = np.diff(
        np.unique(
            np.asarray(
                t,
                float,
            )
        )
    )

    dt = dt[
        dt > 0
    ]

    if not len(
        dt
    ):
        return pd.DataFrame()

    nyquist_khz = (
        500.0
        / float(
            np.min(
                dt
            )
        )
    )

    fmax = (
        min(
            float(
                RESIDUAL_FREQ_MAX_KHZ
            ),
            nyquist_khz,
        )
        if RESIDUAL_FREQ_MAX_KHZ
        is not None
        else nyquist_khz
    )

    if (
        fmax
        <= RESIDUAL_FREQ_MIN_KHZ
    ):
        return pd.DataFrame()

    freq = np.linspace(
        float(
            RESIDUAL_FREQ_MIN_KHZ
        ),
        fmax,
        int(
            RESIDUAL_GRID_POINTS
        ),
    )

    omega = (
        2.0
        * np.pi
        * freq
        / 1000.0
    )

    rows = []

    for nv, sub in ranked.groupby(
        "nv_index",
        sort=True,
    ):
        best = sub.sort_values(
            "site_rank"
        ).iloc[0]

        popt = parse_popt(
            best[
                "popt_json"
            ]
        )

        if popt is None:
            continue

        pred = curve_from_popt(
            t,
            popt,
        )

        rr = (
            (
                y[nv]
                - pred
            )
            / np.maximum(
                e[nv],
                1e-12,
            )
        )

        rr -= np.mean(
            rr
        )

        power = lombscargle(
            t,
            rr,
            omega,
            normalize=True,
        )

        order = np.argsort(
            power
        )[::-1]

        selected = []

        for ind in order:
            f = float(
                freq[
                    ind
                ]
            )

            if any(
                abs(
                    f
                    - oldf
                )
                <= FAMILY_TOL_KHZ
                for oldf, _p
                in selected
            ):
                continue

            selected.append(
                (
                    f,
                    float(
                        power[
                            ind
                        ]
                    ),
                )
            )

            if (
                len(
                    selected
                )
                >= RESIDUAL_N_PEAKS
            ):
                break

        for rank, (
            f,
            pwr,
        ) in enumerate(
            selected,
            start=1,
        ):
            rows.append(
                {
                    "nv_index":
                        int(nv),
                    "residual_peak_rank":
                        int(rank),
                    "freq_kHz":
                        f,
                    "power":
                        pwr,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# BOOTSTRAP / CROSS VALIDATION WITH RANKED ORIENTATION POOL
# =============================================================================

def row_refit_for_new_data(
    t,
    y,
    e,
    row,
    expected_revival,
    max_nfev,
):
    # Re-use equal-footing bound construction, but override budget locally.
    warm = parse_popt(
        row[
            "popt_json"
        ]
    )

    p0, lb, ub = (
        standardized_vectors(
            t,
            y,
            row,
            expected_revival,
            warm=warm,
        )
    )

    popt, _pcov, _red = (
        oldfit._fit_least_squares(
            fine_decay,
            np.asarray(
                t,
                float,
            ),
            np.asarray(
                y,
                float,
            ),
            np.maximum(
                np.asarray(
                    e,
                    float,
                ),
                1e-12,
            ),
            p0,
            lb,
            ub,
            max_nfev=int(
                max_nfev
            ),
        )
    )

    pred = curve_from_popt(
        t,
        popt,
    )

    chi2, red, aicc = (
        fit_stats(
            y,
            e,
            pred,
            len(
                popt
            ),
        )
    )

    return (
        popt,
        pred,
        chi2,
        red,
        aicc,
    )


def bootstrap_one_nv(
    nv,
    ranked,
    t,
    y,
    e,
    expected_revival,
):
    candidates = (
        ranked[
            ranked[
                "nv_index"
            ]
            == int(
                nv
            )
        ]
        .sort_values(
            "site_rank"
        )
        .head(
            BOOTSTRAP_TOP_SITES
        )
    )

    if len(
        candidates
    ) < 2:
        return pd.DataFrame()

    best = candidates.iloc[
        0
    ]

    best_popt = parse_popt(
        best[
            "popt_json"
        ]
    )

    truth = curve_from_popt(
        t,
        best_popt,
    )

    candidate_rows = [
        row
        for _, row
        in candidates.iterrows()
    ]

    def one_rep(rep):
        rng = np.random.default_rng(
            RANDOM_SEED
            + 100000
            * int(
                nv
            )
            + int(
                rep
            )
        )

        yb = (
            truth
            + rng.normal(
                0.0,
                e[nv],
            )
        )

        fits = []

        for row in candidate_rows:
            try:
                (
                    _popt,
                    _pred,
                    _chi2,
                    red,
                    aicc,
                ) = (
                    row_refit_for_new_data(
                        t,
                        yb,
                        e[nv],
                        row,
                        expected_revival,
                        BOOTSTRAP_MAX_NFEV,
                    )
                )

                fits.append(
                    (
                        aicc,
                        red,
                        int(
                            row[
                                "site_id"
                            ]
                        ),
                    )
                )

            except Exception:
                continue

        if not fits:
            return None

        fits.sort(
            key=lambda x: (
                x[0],
                x[1],
            )
        )

        return fits[
            0
        ]

    with threadpool_limits(
        limits=1
    ):
        reps = Parallel(
            n_jobs=POST_N_JOBS,
            backend="loky",
            batch_size=1,
        )(
            delayed(one_rep)(
                i
            )
            for i
            in range(
                int(
                    BOOTSTRAP_N
                )
            )
        )

    reps = [
        r
        for r in reps
        if r
        is not None
    ]

    if not reps:
        return pd.DataFrame()

    raw = pd.DataFrame(
        reps,
        columns=[
            "aicc",
            "red_chi2",
            "site_id",
        ],
    )

    out = (
        raw.groupby(
            "site_id",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "wins"
            }
        )
        .sort_values(
            "wins",
            ascending=False,
        )
    )

    out[
        "nv_index"
    ] = int(
        nv
    )

    out[
        "bootstrap_successful_reps"
    ] = len(
        raw
    )

    out[
        "bootstrap_win_fraction"
    ] = (
        out[
            "wins"
        ]
        / len(
            raw
        )
    )

    return out


def run_bootstrap(
    ranked,
    t,
    y,
    e,
    expected_revival,
):
    if not RUN_BOOTSTRAP:
        return pd.DataFrame()

    frames = []

    available = set(
        ranked[
            "nv_index"
        ].astype(
            int
        )
    )

    for nv in BOOTSTRAP_NV_INDICES:
        nv = int(
            nv
        )

        if nv not in available:
            continue

        print(
            f"[bootstrap] NV {nv}: "
            f"{BOOTSTRAP_N} replicas, "
            f"top {BOOTSTRAP_TOP_SITES} "
            "orientation+site hypotheses from the two experiment orientations"
        )

        df = bootstrap_one_nv(
            nv,
            ranked,
            t,
            y,
            e,
            expected_revival,
        )

        if not df.empty:
            frames.append(
                df
            )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        if frames
        else pd.DataFrame()
    )


def cross_validate_one_nv(
    nv,
    ranked,
    t,
    y,
    e,
    expected_revival,
):
    candidates = (
        ranked[
            ranked[
                "nv_index"
            ]
            == int(
                nv
            )
        ]
        .sort_values(
            "site_rank"
        )
        .head(
            CV_TOP_SITES
        )
    )

    if len(
        candidates
    ) < 2:
        return pd.DataFrame()

    rows = [
        row
        for _, row
        in candidates.iterrows()
    ]

    n = len(
        t
    )

    ntrain = int(
        round(
            CV_TRAIN_FRACTION
            * n
        )
    )

    ntrain = max(
        len(
            PARAM_NAMES
        )
        + 5,
        ntrain,
    )

    ntrain = min(
        ntrain,
        n - 3,
    )

    def one_rep(rep):
        rng = np.random.default_rng(
            RANDOM_SEED
            + 200000
            * int(
                nv
            )
            + int(
                rep
            )
        )

        perm = rng.permutation(
            n
        )

        train = np.sort(
            perm[
                :ntrain
            ]
        )

        test = np.sort(
            perm[
                ntrain:
            ]
        )

        out = []

        for row in rows:
            try:
                (
                    popt,
                    _pred,
                    _chi2,
                    _red,
                    _aicc,
                ) = (
                    row_refit_for_new_data(
                        t[
                            train
                        ],
                        y[
                            nv,
                            train,
                        ],
                        e[
                            nv,
                            train,
                        ],
                        row,
                        expected_revival,
                        CV_MAX_NFEV,
                    )
                )

                pred_test = curve_from_popt(
                    t[
                        test
                    ],
                    popt,
                )

                test_chi2 = float(
                    np.sum(
                        (
                            (
                                y[
                                    nv,
                                    test,
                                ]
                                - pred_test
                            )
                            / np.maximum(
                                e[
                                    nv,
                                    test,
                                ],
                                1e-12,
                            )
                        )
                        ** 2
                    )
                )

                out.append(
                    {
                        "rep":
                            int(
                                rep
                            ),
                        "site_id":
                            int(
                                row[
                                    "site_id"
                                ]
                            ),
                        "test_chi2":
                            test_chi2,
                    }
                )

            except Exception:
                continue

        return out

    with threadpool_limits(
        limits=1
    ):
        reps = Parallel(
            n_jobs=POST_N_JOBS,
            backend="loky",
            batch_size=1,
        )(
            delayed(one_rep)(
                i
            )
            for i in range(
                int(
                    CV_REPEATS
                )
            )
        )

    raw_rows = [
        x
        for rep in reps
        for x in rep
    ]

    if not raw_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(
        raw_rows
    )

    agg = (
        raw.groupby(
            "site_id",
            as_index=False,
        )
        .agg(
            mean_test_chi2=(
                "test_chi2",
                "mean",
            ),
            median_test_chi2=(
                "test_chi2",
                "median",
            ),
            std_test_chi2=(
                "test_chi2",
                "std",
            ),
            cv_successful_reps=(
                "rep",
                "nunique",
            ),
        )
        .sort_values(
            "mean_test_chi2"
        )
        .reset_index(
            drop=True
        )
    )

    agg[
        "nv_index"
    ] = int(
        nv
    )

    agg[
        "predictive_rank"
    ] = np.arange(
        1,
        len(
            agg
        )
        + 1,
    )

    return agg


def run_cross_validation(
    ranked,
    t,
    y,
    e,
    expected_revival,
):
    if not RUN_CROSS_VALIDATION:
        return pd.DataFrame()

    frames = []

    available = set(
        ranked[
            "nv_index"
        ].astype(
            int
        )
    )

    for nv in CV_NV_INDICES:
        nv = int(
            nv
        )

        if nv not in available:
            continue

        print(
            f"[CV] NV {nv}: "
            f"{CV_REPEATS} splits, "
            f"top {CV_TOP_SITES} "
            "orientation+site hypotheses from the two experiment orientations"
        )

        df = (
            cross_validate_one_nv(
                nv,
                ranked,
                t,
                y,
                e,
                expected_revival,
            )
        )

        if not df.empty:
            frames.append(
                df
            )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        if frames
        else pd.DataFrame()
    )


# =============================================================================
# GLOBAL SUMMARY
# =============================================================================

def make_global_summary(
    summary,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            13,
            9,
        ),
    )

    ax = axes[
        0,
        0,
    ]

    vals = summary[
        "second_delta_aicc"
    ].to_numpy(float)

    vals = vals[
        np.isfinite(
            vals
        )
    ]

    ax.hist(
        vals,
        bins=35,
    )

    ax.set_xlabel(
        "ΔAICc: second site − best site"
    )

    ax.set_ylabel(
        "NV count"
    )

    ax.set_title(
        "Same-orientation site separation"
    )

    ax.grid(
        alpha=0.2
    )

    ax = axes[
        0,
        1,
    ]

    ax.hist(
        summary[
            "best_weight"
        ].dropna(),
        bins=30,
    )

    ax.set_xlabel(
        "Best-site Akaike weight"
    )

    ax.set_ylabel(
        "NV count"
    )

    ax.set_title(
        "Confidence within the independently assigned NV orientation"
    )

    ax.grid(
        alpha=0.2
    )

    ax = axes[
        1,
        0,
    ]

    ax.scatter(
        summary[
            "best_red_chi2"
        ],
        summary[
            "second_delta_aicc"
        ],
        s=16,
    )

    ax.set_xlabel(
        "Best reduced χ²"
    )

    ax.set_ylabel(
        "ΔAICc to second site"
    )

    ax.set_title(
        "Fit quality vs site discrimination"
    )

    ax.grid(
        alpha=0.2
    )

    ax = axes[
        1,
        1,
    ]

    cats = [
        "near_degenerate",
        "some_separation",
        "substantial_separation",
        "large_separation",
    ]

    vals = [
        int(
            (
                summary[
                    "separation_label"
                ]
                == c
            ).sum()
        )
        for c in cats
    ]

    ax.bar(
        np.arange(
            4
        ),
        vals,
    )

    ax.set_xticks(
        np.arange(
            4
        )
    )

    ax.set_xticklabels(
        [
            "Δ<2",
            "2–6",
            "6–10",
            "≥10",
        ]
    )

    ax.set_ylabel(
        "NV count"
    )

    ax.set_title(
        "Confidence classes"
    )

    ax.grid(
        alpha=0.2,
        axis="y",
    )

    fig.suptitle(
        (
            f"{RESULT_TAG}: orientation-constrained "
            f"$^{{13}}$C site inference | "
            f"{len(summary)} NVs"
        ),
        fontsize=14,
    )

    fig.tight_layout()

    return fig


def save_table(
    df,
    path,
):
    if (
        df is None
        or df.empty
    ):
        return

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    paths = discover_paths()

    (
        attempts,
        ckpt,
        t,
        y,
        e,
        expected_revival,
    ) = load_inputs(
        paths
    )

    # Orientation is an independent physical constraint.  Determine it first,
    # then fit C13 sites ONLY from the corresponding orientation-specific pool.
    orientation_map, orientation_quality, orientation_source = load_orientation_map()

    print(
        "[orientation] per-NV orientation is locked before C13 fitting; "
        f"allowed experiment orientations={ALLOWED_ORIENTATIONS}"
    )

    catalog_records, catalog_lookup = load_catalog()

    outdir = (
        Path(
            OUTPUT_DIR
        )
        if OUTPUT_DIR
        is not None
        else paths.prefix.parent
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = outdir / (
        paths.prefix.name
        + "_orientation_locked_confidence_v6"
    )

    # Save the independent orientation assignment for auditability.
    if not orientation_quality.empty:
        save_table(
            orientation_quality,
            Path(str(base) + "_orientation_assignments.csv"),
        )

    # Equal-footing refit ONLY inside each NV's locked orientation pool.
    refit_df = run_orientation_locked_refits(
        attempts,
        orientation_map,
        t,
        y,
        e,
        expected_revival,
    )

    ranked = (
        rank_equal_footing_sites(
            refit_df
        )
    )

    ranked, families = (
        assign_families(
            ranked
        )
    )

    # Expand all fitted model parameters into explicit analysis columns.
    ranked = expand_fit_parameter_columns(
        ranked
    )

    ranked = (
        add_catalog_positions(
            ranked,
            catalog_lookup,
        )
    )

    summary = build_summary(
        ranked
    )

    top10 = ranked[
        ranked[
            "site_rank"
        ]
        <= TOP_UNIQUE_SAVE
    ].copy()

    residuals = (
        residual_peak_table(
            t,
            y,
            e,
            ranked,
        )
    )

    bootstrap = (
        run_bootstrap(
            ranked,
            t,
            y,
            e,
            expected_revival,
        )
    )

    cv = (
        run_cross_validation(
            ranked,
            t,
            y,
            e,
            expected_revival,
        )
    )

    if SAVE_CONFIDENCE_TABLES:
        save_table(
            ranked,
            Path(
                str(base)
                + "_all_equal_footing_sites.csv"
            ),
        )

        save_table(
            top10,
            Path(
                str(base)
                + "_top10_unique_sites.csv"
            ),
        )

        save_table(
            summary,
            Path(
                str(base)
                + "_nv_confidence_summary.csv"
            ),
        )

        save_table(
            families,
            Path(
                str(base)
                + "_site_families.csv"
            ),
        )

        save_table(
            residuals,
            Path(
                str(base)
                + "_residual_peaks.csv"
            ),
        )

        save_table(
            bootstrap,
            Path(
                str(base)
                + "_bootstrap_site_stability.csv"
            ),
        )

        save_table(
            cv,
            Path(
                str(base)
                + "_heldout_prediction.csv"
            ),
        )

    if SAVE_MAIN_THREE_PANEL_PDF:
        save_dashboard_pdf(
            Path(
                str(base)
                + "_dashboard_full_zoom_weights_positions_details.pdf"
            ),
            t,
            y,
            e,
            ranked,
            expected_revival,
        )

    fig = make_global_summary(
        summary
    )

    if SAVE_GLOBAL_SUMMARY:
        png = Path(
            str(base)
            + "_global_summary.png"
        )

        pdf = Path(
            str(base)
            + "_global_summary.pdf"
        )

        fig.savefig(
            png,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            pdf,
            bbox_inches="tight",
        )

        print(
            f"Saved: {png}"
        )

        print(
            f"Saved: {pdf}"
        )

    print()
    print("=" * 96)
    print("ORIENTATION-LOCKED CONFIDENCE V6 COMPLETE")
    print("=" * 96)
    print(
        f"NVs analyzed                   : "
        f"{len(summary)}"
    )
    print(
        f"experiment orientations        : "
        f"{ALLOWED_ORIENTATIONS}"
    )
    print(
        f"equal-footing site fits        : "
        f"{len(ranked)}"
    )
    print(
        f"top-3 identity                 : "
        f"unique site_id within each NV's locked orientation"
    )
    print(
        f"dense points/fit curve         : "
        f"{DENSE_CURVE_POINTS}"
    )
    print(
        f"13C positions shown/NV         : "
        f"{POSITION_TOP_N}"
    )
    print(
        f"Akaike hypotheses plotted/NV   : "
        f"{AKAIKE_PLOT_TOP_N}"
    )
    print(
        f"fit-detail rows shown/NV       : "
        f"{FIT_TABLE_TOP_N}"
    )
    print(
        "fit columns                    : score, AICc, Akaike weight, T2, "
        "revival, width, beta, amplitudes, phases, frequencies, hyperfine"
    )
    print(
        f"near-degenerate ΔAICc<2        : "
        f"{int((summary['separation_label'] == 'near_degenerate').sum())}"
    )
    print(
        f"large-separation ΔAICc>=10     : "
        f"{int((summary['separation_label'] == 'large_separation').sum())}"
    )
    print("=" * 96)

    if SHOW_GLOBAL_SUMMARY:
        plt.show(
            block=True
        )
    else:
        plt.close(
            fig
        )

    return {
        "allowed_orientations": ALLOWED_ORIENTATIONS,
        "orientation_map": orientation_map,
        "orientation_quality": orientation_quality,
        "ranked_sites":
            ranked,
        "families":
            families,
        "summary":
            summary,
        "residuals":
            residuals,
        "bootstrap":
            bootstrap,
        "cross_validation":
            cv,
    }


if __name__ == "__main__":
    main()
