# -*- coding: utf-8 -*-
"""
Spin-echo ranked-fit confidence analysis V2.

Designed for the saved exhaustive outputs from:
    spin_echo_old_protocol_ranked_52G
or
    spin_echo_old_protocol_ranked_49G

Key V2 changes
--------------
1) Confidence ranking uses ONLY equally deep attempts:
       stage in {"multistart", "refine"}
   Screen-only fits are not allowed to beat deeply optimized candidates.

2) A physical hypothesis is:
       (NV orientation, lattice site_id)
   Repeated optimizer solutions for the SAME orientation+site collapse to ONE
   best fit before ranking.

3) Top-3 fit PDFs show THREE UNIQUE orientation+site hypotheses.
   The same physical hypothesis cannot repeat.

4) Fit lines are evaluated on a DENSE time grid, so plotted curves are smooth.

5) Site-family clustering is orientation-aware and NON-TRANSITIVE.

6) Optional known per-NV orientations can be supplied and enforced.

7) Bootstrap and held-out validation refit shortlisted hypotheses using the
   same optimizer/budget for fair comparison.
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

RESULT_TAG = "spin_echo_old_protocol_ranked_52G"
# RESULT_TAG = "spin_echo_old_protocol_ranked_49G"

SEARCH_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master")

ALL_ATTEMPTS_PATH = None
CHECKPOINT_PATH = None
OUTPUT_DIR = None

PRIMARY_STAGES = ("multistart", "refine")

# ---------------- Known orientation information -------------------------------
# CSV option: columns nv_index, orientation where orientation looks like
# "(1, 1, -1)"
KNOWN_ORIENTATION_CSV = None

# Dict option, e.g. {0: (1,1,-1), 1: (-1,1,1)}
KNOWN_ORIENTATION_MAP = {}

# Auto-load "orientations" from Dioptric raw-data object if available.
AUTO_KNOWN_ORIENTATION_FILE_STEM = None
if RESULT_TAG.endswith("49G"):
    AUTO_KNOWN_ORIENTATION_FILE_STEM = (
        "2025_11_15-14_11_49-johnson_204nv_s9-17d44b"
    )

# If True, NVs without a known orientation are dropped.
REQUIRE_KNOWN_ORIENTATION = False

# ---------------- Ranking / site families -------------------------------------
TOP_UNIQUE_HYPOTHESES_SAVE = 10
TOP_UNIQUE_HYPOTHESES_PLOT = 3

# Practical spectral-family tolerance. This is intentionally much tighter than
# the old 12-kHz transitive clustering.
FAMILY_TOL_KHZ = 2.0

# ---------------- Dense fitted curves -----------------------------------------
DENSE_CURVE_POINTS = 3000
FIRST_REVIVAL_HALF_WIDTH_US = 12.5

# ---------------- Bootstrap ----------------------------------------------------
RUN_BOOTSTRAP = True
BOOTSTRAP_NV_INDICES = [16]
BOOTSTRAP_N = 300
BOOTSTRAP_TOP_HYPOTHESES = 5
BOOTSTRAP_MAX_NFEV = 30_000

# ---------------- Held-out validation -----------------------------------------
RUN_CROSS_VALIDATION = True
CV_NV_INDICES = [16]
CV_REPEATS = 50
CV_TOP_HYPOTHESES = 3
CV_TRAIN_FRACTION = 0.80
CV_MAX_NFEV = 30_000

CPU_COUNT = os.cpu_count() or 4
POST_N_JOBS = max(1, min(12, CPU_COUNT - 2))
RANDOM_SEED = 20260922

# ---------------- Residual spectroscopy ---------------------------------------
RUN_RESIDUAL_SPECTROSCOPY = True
RESIDUAL_FREQ_MIN_KHZ = 5.0
RESIDUAL_FREQ_MAX_KHZ = None
RESIDUAL_GRID_POINTS = 5000
RESIDUAL_N_PEAKS = 5

# ---------------- Plotting -----------------------------------------------------
SAVE_TOP3_DENSE_FULL_PDF = True
SAVE_TOP3_DENSE_FIRST_REVIVAL_PDF = True
SAVE_DIAGNOSTIC_PDF = True
SAVE_GLOBAL_SUMMARY = True

DIAGNOSTIC_NV_INDICES = None

PDF_COLS = 3
PDF_ROWS = 4
SHOW_GLOBAL_SUMMARY = True

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
    "osc_f0_cyc_per_us",
    "osc_phi0_rad",
    "osc_f1_cyc_per_us",
    "osc_phi1_rad",
]


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
        raise FileNotFoundError(f"No files matching {pattern!r} under {root}")
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
        raise ValueError(f"Unexpected all-attempt filename: {all_path}")

    prefix = Path(s[: -len(suffix)])
    ckpt = (
        Path(CHECKPOINT_PATH)
        if CHECKPOINT_PATH is not None
        else Path(str(prefix) + "_fit_checkpoint.npz")
    )

    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    return ResultPaths(all_attempts=all_path, checkpoint=ckpt, prefix=prefix)


def canonical_orientation(value):
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception:
            return None

    try:
        arr = np.asarray(value, int).ravel()
    except Exception:
        return None

    if arr.size != 3:
        return None

    return str(tuple(int(x) for x in arr))


def load_known_orientation_map():
    mapping = {}

    for nv, ori in KNOWN_ORIENTATION_MAP.items():
        c = canonical_orientation(ori)
        if c is not None:
            mapping[int(nv)] = c

    if KNOWN_ORIENTATION_CSV is not None:
        p = Path(KNOWN_ORIENTATION_CSV)
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)

        if not {"nv_index", "orientation"}.issubset(df.columns):
            raise ValueError(
                "KNOWN_ORIENTATION_CSV needs columns nv_index, orientation"
            )

        for _, row in df.iterrows():
            c = canonical_orientation(row["orientation"])
            if c is not None:
                mapping[int(row["nv_index"])] = c

    if AUTO_KNOWN_ORIENTATION_FILE_STEM:
        try:
            from utils import data_manager as dm

            data = dm.get_raw_data(
                file_stem=AUTO_KNOWN_ORIENTATION_FILE_STEM,
                load_npz=True,
            )
            arr = data.get("orientations", None)

            if arr is not None:
                arr = np.asarray(arr, int)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    for nv in range(arr.shape[0]):
                        c = canonical_orientation(arr[nv])
                        if c is not None:
                            mapping[int(nv)] = c

                    print(
                        f"[orientation] loaded {arr.shape[0]} known orientations "
                        f"from {AUTO_KNOWN_ORIENTATION_FILE_STEM}"
                    )

        except Exception as exc:
            print(
                "[orientation] auto-load failed; using orientation as part of "
                f"hypothesis identity only: {exc}"
            )

    return mapping


def load_inputs(paths):
    print("=" * 92)
    print("RANKED SPIN-ECHO CONFIDENCE ANALYSIS V2")
    print("=" * 92)
    print(f"result tag   : {RESULT_TAG}")
    print(f"all attempts : {paths.all_attempts}")
    print(f"checkpoint   : {paths.checkpoint}")

    attempts = pd.read_csv(paths.all_attempts)
    ckpt = np.load(paths.checkpoint, allow_pickle=True)

    t = np.asarray(ckpt["times_us"], float)
    y = np.asarray(ckpt["norm_counts"], float)
    e = np.asarray(ckpt["norm_counts_ste"], float)

    expected_revival = np.nan
    if "expected_revival_total_us" in ckpt:
        vals = np.asarray(ckpt["expected_revival_total_us"], float).ravel()
        if vals.size:
            expected_revival = float(vals[0])

    print(f"attempt rows : {len(attempts):,}")
    print(f"data shape   : {y.shape}")
    print(
        "stages       : "
        + ", ".join(
            f"{k}={v:,}"
            for k, v in attempts["stage"].value_counts().to_dict().items()
        )
    )

    return attempts, ckpt, t, y, e, expected_revival


def finite_success(df):
    out = df.copy()
    out = out[out["status"].astype(str) == "ok"].copy()

    for col in ("aicc", "red_chi2", "score_primary"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[
        np.isfinite(out["aicc"])
        & np.isfinite(out["red_chi2"])
        & np.isfinite(out["score_primary"])
    ].copy()

    out["orientation"] = out["orientation"].map(canonical_orientation)
    out = out[out["orientation"].notna()].copy()

    out["site_id"] = pd.to_numeric(out["site_id"], errors="coerce")
    out = out[np.isfinite(out["site_id"])].copy()
    out["site_id"] = out["site_id"].astype(int)

    return out


def parse_popt(s):
    try:
        p = np.asarray(json.loads(s), float)
        if p.ndim == 1 and len(p) == len(PARAM_NAMES) and np.all(np.isfinite(p)):
            return p
    except Exception:
        pass
    return None


def curve_from_row(t_eval, row):
    p = parse_popt(row["popt_json"])
    if p is None:
        return None
    return np.asarray(fine_decay(np.asarray(t_eval, float), *p), float)


def fit_stats(y, e, pred, k):
    y = np.asarray(y, float)
    e = np.maximum(np.asarray(e, float), 1e-12)
    pred = np.asarray(pred, float)

    chi2 = float(np.sum(((y - pred) / e) ** 2))
    n = len(y)
    dof = max(1, n - int(k))
    red = chi2 / dof
    aic = chi2 + 2 * k
    aicc = (
        aic + 2 * k * (k + 1) / (n - k - 1)
        if n > k + 1
        else np.inf
    )
    return chi2, red, float(aicc)


def filter_known_orientations(df, known_map):
    if not known_map:
        if REQUIRE_KNOWN_ORIENTATION:
            raise RuntimeError(
                "REQUIRE_KNOWN_ORIENTATION=True but no known orientation map loaded."
            )
        return df.copy()

    keep = np.ones(len(df), dtype=bool)
    nvs = df["nv_index"].to_numpy(int)
    oris = df["orientation"].astype(str).to_numpy()

    for i, (nv, ori) in enumerate(zip(nvs, oris)):
        known = known_map.get(int(nv))
        if known is None:
            keep[i] = not REQUIRE_KNOWN_ORIENTATION
        else:
            keep[i] = ori == known

    filtered = df.loc[keep].copy()
    print(
        f"[orientation] removed {len(df) - len(filtered):,} attempts "
        "inconsistent with known NV orientations"
    )
    return filtered


def build_deep_hypotheses(attempts, known_map, n_points):
    good = finite_success(attempts)
    deep = good[good["stage"].astype(str).isin(PRIMARY_STAGES)].copy()
    deep = filter_known_orientations(deep, known_map)

    if deep.empty:
        raise RuntimeError("No successful deep attempts remain after filtering.")

    # Physical hypothesis = NV + orientation + lattice site.
    group_cols = ["nv_index", "orientation", "site_id"]

    deep = deep.sort_values(
        group_cols + ["aicc", "red_chi2", "score_primary"],
        kind="stable",
    )

    hyp = (
        deep.groupby(group_cols, as_index=False, dropna=False)
        .first()
        .copy()
    )

    hyp["npar"] = hyp["popt_json"].map(
        lambda s: len(json.loads(s))
        if isinstance(s, str) and s
        else len(PARAM_NAMES)
    )
    hyp["dof"] = np.maximum(
        1,
        int(n_points) - hyp["npar"].to_numpy(int),
    )
    hyp["chi2"] = (
        hyp["red_chi2"].to_numpy(float)
        * hyp["dof"].to_numpy(float)
    )

    hyp["delta_aicc"] = np.nan
    hyp["delta_chi2"] = np.nan
    hyp["relative_likelihood"] = np.nan
    hyp["akaike_weight"] = np.nan
    hyp["hypothesis_rank"] = -1

    for nv, inds in hyp.groupby("nv_index").groups.items():
        inds = np.asarray(list(inds), int)
        aicc = hyp.loc[inds, "aicc"].to_numpy(float)
        chi2 = hyp.loc[inds, "chi2"].to_numpy(float)

        da = aicc - np.min(aicc)
        dc = chi2 - np.min(chi2)

        logw = -0.5 * da
        logw -= np.max(logw)
        w = np.exp(logw)
        w /= np.sum(w)

        order = np.argsort(aicc)
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)

        hyp.loc[inds, "delta_aicc"] = da
        hyp.loc[inds, "delta_chi2"] = dc
        hyp.loc[inds, "relative_likelihood"] = np.exp(-0.5 * da)
        hyp.loc[inds, "akaike_weight"] = w
        hyp.loc[inds, "hypothesis_rank"] = rank

    hyp = hyp.sort_values(
        ["nv_index", "hypothesis_rank"]
    ).reset_index(drop=True)

    print(
        f"[primary] {len(hyp):,} unique deeply optimized "
        "(orientation, site) hypotheses"
    )
    print(
        f"[primary] median hypotheses/NV = "
        f"{hyp.groupby('nv_index').size().median():.0f}"
    )

    return hyp


def build_screen_challengers(attempts, deep_hyp, known_map):
    good = finite_success(attempts)
    screen = good[good["stage"].astype(str) == "screen"].copy()
    screen = filter_known_orientations(screen, known_map)

    if screen.empty:
        return pd.DataFrame()

    screen = (
        screen.sort_values(
            ["nv_index", "orientation", "site_id", "aicc", "red_chi2"],
            kind="stable",
        )
        .groupby(
            ["nv_index", "orientation", "site_id"],
            as_index=False,
        )
        .first()
    )

    deep_keys = set(
        zip(
            deep_hyp["nv_index"].astype(int),
            deep_hyp["orientation"].astype(str),
            deep_hyp["site_id"].astype(int),
        )
    )

    mask = [
        (int(nv), str(ori), int(site)) not in deep_keys
        for nv, ori, site in zip(
            screen["nv_index"],
            screen["orientation"],
            screen["site_id"],
        )
    ]

    out = screen.loc[mask].copy()
    out["screen_rank"] = (
        out.groupby("nv_index")["aicc"]
        .rank(method="first")
        .astype(int)
    )
    return out.sort_values(["nv_index", "screen_rank"])


def assign_families(hyp):
    """
    Orientation-aware, non-transitive leader clustering.
    """
    hyp = hyp.copy()
    hyp["family_id"] = -1
    hyp["family_rank"] = -1
    hyp["family_weight"] = np.nan

    family_rows = []

    for nv, nvsub in hyp.groupby("nv_index", sort=True):
        nv_family_records = []
        next_family_id = 0

        for ori, sub in nvsub.groupby("orientation", sort=False):
            sub = sub.sort_values("hypothesis_rank")
            leaders = []

            for idx, row in sub.iterrows():
                f0 = float(row["f0_kHz"])
                f1 = float(row["f1_kHz"])

                chosen = None
                for fam in leaders:
                    if (
                        abs(f0 - fam["leader_f0"]) <= FAMILY_TOL_KHZ
                        and abs(f1 - fam["leader_f1"]) <= FAMILY_TOL_KHZ
                    ):
                        chosen = fam
                        break

                if chosen is None:
                    chosen = {
                        "family_id": next_family_id,
                        "leader_f0": f0,
                        "leader_f1": f1,
                        "leader_site_id": int(row["site_id"]),
                        "orientation": str(ori),
                        "members": [],
                    }
                    leaders.append(chosen)
                    next_family_id += 1

                chosen["members"].append(idx)
                hyp.loc[idx, "family_id"] = int(chosen["family_id"])

            for fam in leaders:
                members = hyp.loc[fam["members"]]
                nv_family_records.append(
                    {
                        "nv_index": int(nv),
                        "family_id": int(fam["family_id"]),
                        "orientation": str(fam["orientation"]),
                        "leader_site_id": int(fam["leader_site_id"]),
                        "leader_f0_kHz": float(fam["leader_f0"]),
                        "leader_f1_kHz": float(fam["leader_f1"]),
                        "n_sites": int(len(members)),
                        "family_weight": float(
                            members["akaike_weight"].sum()
                        ),
                        "best_delta_aicc": float(
                            members["delta_aicc"].min()
                        ),
                    }
                )

        famdf = pd.DataFrame(nv_family_records)
        if famdf.empty:
            continue

        famdf = famdf.sort_values(
            ["family_weight", "best_delta_aicc"],
            ascending=[False, True],
        ).reset_index(drop=True)
        famdf["family_rank"] = np.arange(1, len(famdf) + 1)

        rank_map = dict(zip(famdf["family_id"], famdf["family_rank"]))
        weight_map = dict(zip(famdf["family_id"], famdf["family_weight"]))

        inds = hyp.index[hyp["nv_index"] == nv]
        hyp.loc[inds, "family_rank"] = [
            rank_map[int(x)] for x in hyp.loc[inds, "family_id"]
        ]
        hyp.loc[inds, "family_weight"] = [
            weight_map[int(x)] for x in hyp.loc[inds, "family_id"]
        ]

        family_rows.append(famdf)

    families = (
        pd.concat(family_rows, ignore_index=True)
        if family_rows
        else pd.DataFrame()
    )
    return hyp, families


def separation_label(delta_aicc):
    if not np.isfinite(delta_aicc):
        return "single_hypothesis"
    if delta_aicc < 2:
        return "near_degenerate"
    if delta_aicc < 6:
        return "some_separation"
    if delta_aicc < 10:
        return "substantial_separation"
    return "large_separation"


def build_nv_summary(hyp, families, known_map):
    rows = []

    for nv, sub in hyp.groupby("nv_index", sort=True):
        sub = sub.sort_values("hypothesis_rank")
        best = sub.iloc[0]
        second = sub.iloc[1] if len(sub) > 1 else None
        third = sub.iloc[2] if len(sub) > 2 else None

        bf = families[
            (families["nv_index"] == nv)
            & (families["family_id"] == int(best["family_id"]))
        ]
        bf = bf.iloc[0] if len(bf) else None

        d2 = (
            float(second["delta_aicc"])
            if second is not None
            else np.nan
        )

        rows.append(
            {
                "nv_index": int(nv),
                "known_orientation": known_map.get(int(nv), ""),
                "best_site_id": int(best["site_id"]),
                "best_orientation": str(best["orientation"]),
                "best_f0_kHz": float(best["f0_kHz"]),
                "best_f1_kHz": float(best["f1_kHz"]),
                "best_kappa": float(best["kappa"]),
                "best_distance_A": float(best["distance_A"]),
                "best_red_chi2": float(best["red_chi2"]),
                "best_aicc": float(best["aicc"]),
                "best_hypothesis_weight": float(best["akaike_weight"]),
                "best_family_id": int(best["family_id"]),
                "best_family_weight": (
                    float(bf["family_weight"])
                    if bf is not None
                    else np.nan
                ),
                "best_family_n_sites": (
                    int(bf["n_sites"])
                    if bf is not None
                    else 1
                ),
                "second_site_id": (
                    int(second["site_id"])
                    if second is not None
                    else -1
                ),
                "second_orientation": (
                    str(second["orientation"])
                    if second is not None
                    else ""
                ),
                "second_delta_aicc": d2,
                "second_delta_chi2": (
                    float(second["delta_chi2"])
                    if second is not None
                    else np.nan
                ),
                "third_site_id": (
                    int(third["site_id"])
                    if third is not None
                    else -1
                ),
                "third_orientation": (
                    str(third["orientation"])
                    if third is not None
                    else ""
                ),
                "separation_label": separation_label(d2),
                "n_deep_hypotheses": int(len(sub)),
            }
        )

    return pd.DataFrame(rows)


def top_unique_table(hyp, n=TOP_UNIQUE_HYPOTHESES_SAVE):
    return (
        hyp[hyp["hypothesis_rank"] <= int(n)]
        .sort_values(["nv_index", "hypothesis_rank"])
        .copy()
    )


def dense_grid_for_mask(t, mask=None):
    t = np.asarray(t, float)
    if mask is None:
        lo, hi = float(np.min(t)), float(np.max(t))
    else:
        tt = t[np.asarray(mask, bool)]
        lo, hi = float(np.min(tt)), float(np.max(tt))
    return np.linspace(lo, hi, int(DENSE_CURVE_POINTS))


def save_top3_dense_pdf(
    path,
    t,
    y,
    e,
    hyp,
    expected_revival,
    first_revival_only=False,
):
    npp = PDF_COLS * PDF_ROWS
    nvs = sorted(hyp["nv_index"].astype(int).unique().tolist())

    with PdfPages(path) as pdf:
        for start in range(0, len(nvs), npp):
            page_nvs = nvs[start : start + npp]
            fig, axes = plt.subplots(
                PDF_ROWS,
                PDF_COLS,
                figsize=(5.2 * PDF_COLS, 3.7 * PDF_ROWS),
                squeeze=False,
            )
            axes = axes.ravel()

            for ax, nv in zip(axes, page_nvs):
                sub = (
                    hyp[hyp["nv_index"] == nv]
                    .sort_values("hypothesis_rank")
                    .head(TOP_UNIQUE_HYPOTHESES_PLOT)
                )

                if first_revival_only and np.isfinite(expected_revival):
                    mask = (
                        np.abs(t - expected_revival)
                        <= FIRST_REVIVAL_HALF_WIDTH_US
                    )
                else:
                    mask = np.ones(len(t), dtype=bool)

                td = dense_grid_for_mask(t, mask)

                ax.errorbar(
                    t[mask],
                    y[nv, mask],
                    yerr=e[nv, mask],
                    fmt="o",
                    ms=2.8,
                    capsize=1,
                    lw=0.55,
                    label="data",
                    zorder=4,
                )

                styles = ["-", "--", ":"]
                for j, (_, row) in enumerate(sub.iterrows()):
                    curve = curve_from_row(td, row)
                    if curve is None:
                        continue

                    rank = int(row["hypothesis_rank"])
                    ax.plot(
                        td,
                        curve,
                        linestyle=styles[min(j, len(styles) - 1)],
                        lw=1.6 if rank == 1 else 1.15,
                        label=(
                            f"#{rank} site {int(row['site_id'])} "
                            f"{row['orientation']} | "
                            f"χ²r={row['red_chi2']:.2f}, "
                            f"ΔAICc={row['delta_aicc']:.2f}"
                        ),
                        zorder=3 - j,
                    )

                if np.isfinite(expected_revival):
                    ax.axvline(
                        expected_revival,
                        ls="-.",
                        lw=0.7,
                        alpha=0.6,
                    )

                ax.set_title(f"NV {nv}", fontsize=9)
                ax.set_xlabel("Total evolution time (µs)", fontsize=8)
                ax.set_ylabel("Normalized signal", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.20)
                ax.legend(fontsize=5.4, loc="best")

            for ax in axes[len(page_nvs) :]:
                ax.axis("off")

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved: {path}")


def residual_peak_table(t, y, e, hyp):
    if not RUN_RESIDUAL_SPECTROSCOPY:
        return pd.DataFrame()

    dt = np.diff(np.unique(np.asarray(t, float)))
    dt = dt[dt > 0]
    if not len(dt):
        return pd.DataFrame()

    nyquist_khz = 500.0 / float(np.min(dt))
    fmax = (
        min(float(RESIDUAL_FREQ_MAX_KHZ), nyquist_khz)
        if RESIDUAL_FREQ_MAX_KHZ is not None
        else nyquist_khz
    )

    fmin = float(RESIDUAL_FREQ_MIN_KHZ)
    if fmax <= fmin:
        return pd.DataFrame()

    freq_khz = np.linspace(
        fmin,
        fmax,
        int(RESIDUAL_GRID_POINTS),
    )
    omega = 2 * np.pi * freq_khz / 1000.0
    rows = []

    for nv, sub in hyp.groupby("nv_index", sort=True):
        best = sub.sort_values("hypothesis_rank").iloc[0]
        curve = curve_from_row(t, best)
        if curve is None:
            continue

        rr = (
            (np.asarray(y[nv], float) - curve)
            / np.maximum(np.asarray(e[nv], float), 1e-12)
        )
        rr -= np.nanmean(rr)

        m = np.isfinite(t) & np.isfinite(rr)
        if np.sum(m) < 8:
            continue

        power = lombscargle(
            np.asarray(t)[m],
            rr[m],
            omega,
            normalize=True,
        )

        order = np.argsort(power)[::-1]
        chosen = []

        for ind in order:
            f = float(freq_khz[ind])
            if any(
                abs(f - oldf) <= FAMILY_TOL_KHZ
                for oldf, _ in chosen
            ):
                continue
            chosen.append((f, float(power[ind])))
            if len(chosen) >= RESIDUAL_N_PEAKS:
                break

        for rank, (freq, pwr) in enumerate(chosen, 1):
            rows.append(
                {
                    "nv_index": int(nv),
                    "residual_peak_rank": int(rank),
                    "freq_kHz": freq,
                    "power": pwr,
                }
            )

    return pd.DataFrame(rows)


def prepare_bounds_for_hypothesis(t, y, row, warm=None):
    p0, lb, ub = oldfit._initial_guess_and_bounds(
        np.asarray(t, float),
        np.asarray(y, float),
        enable_extras=True,
        fixed_rev_time=None,
    )
    pmap = oldfit._param_index_map(fine_decay)

    amp_min = float(row.get("amp_min", -1.0))
    amp_max = float(row.get("amp_max", 1.0))
    oldfit._set_osc_amp_bounds(
        lb,
        ub,
        fine_decay,
        amp_min,
        amp_max,
    )

    f0 = float(row["f0_kHz"]) / 1000.0
    f1 = float(row["f1_kHz"]) / 1000.0
    eps_f = 1e-6

    for name, val in (("osc_f0", f0), ("osc_f1", f1)):
        i = pmap[name]
        p0[i] = val
        lb[i] = val - eps_f
        ub[i] = val + eps_f

    if warm is not None:
        w = np.asarray(warm, float)
        if w.shape == p0.shape:
            p0 = w.copy()
            p0[pmap["osc_f0"]] = f0
            p0[pmap["osc_f1"]] = f1

    p0, lb, ub = oldfit._retie_contrast_to_baseline(
        p0,
        lb,
        ub,
        pmap,
        eps=0.01,
    )

    eps = 1e-9 * np.maximum(1.0, ub - lb)
    p0 = np.minimum(np.maximum(p0, lb + eps), ub - eps)

    return p0, lb, ub


def refit_hypothesis(t, y, e, row, max_nfev):
    warm = parse_popt(row["popt_json"])
    p0, lb, ub = prepare_bounds_for_hypothesis(
        t,
        y,
        row,
        warm=warm,
    )

    popt, _, _ = oldfit._fit_least_squares(
        fine_decay,
        np.asarray(t, float),
        np.asarray(y, float),
        np.maximum(np.asarray(e, float), 1e-12),
        p0,
        lb,
        ub,
        max_nfev=int(max_nfev),
    )

    pred = fine_decay(np.asarray(t, float), *popt)
    chi2, red, aicc = fit_stats(y, e, pred, len(popt))
    return popt, pred, chi2, red, aicc


def bootstrap_one_nv(nv, hyp, t, y, e):
    candidates = (
        hyp[hyp["nv_index"] == nv]
        .sort_values("hypothesis_rank")
        .head(BOOTSTRAP_TOP_HYPOTHESES)
        .copy()
    )

    if len(candidates) < 2:
        return pd.DataFrame()

    truth_curve = curve_from_row(t, candidates.iloc[0])
    if truth_curve is None:
        return pd.DataFrame()

    rows = [row for _, row in candidates.iterrows()]

    def one_rep(rep):
        rng = np.random.default_rng(
            RANDOM_SEED + 100000 * int(nv) + int(rep)
        )

        yb = truth_curve + rng.normal(
            0.0,
            np.asarray(e[nv], float),
        )

        fitted = []

        for row in rows:
            try:
                _, _, _, red, aicc = refit_hypothesis(
                    t,
                    yb,
                    e[nv],
                    row,
                    BOOTSTRAP_MAX_NFEV,
                )
                fitted.append(
                    (
                        float(aicc),
                        float(red),
                        int(row["site_id"]),
                        str(row["orientation"]),
                    )
                )
            except Exception:
                continue

        if not fitted:
            return None

        fitted.sort(key=lambda x: (x[0], x[1]))
        return fitted[0]

    with threadpool_limits(limits=1):
        reps = Parallel(
            n_jobs=POST_N_JOBS,
            backend="loky",
            batch_size=1,
        )(
            delayed(one_rep)(i)
            for i in range(int(BOOTSTRAP_N))
        )

    reps = [r for r in reps if r is not None]

    if not reps:
        return pd.DataFrame()

    raw = pd.DataFrame(
        reps,
        columns=["aicc", "red_chi2", "site_id", "orientation"],
    )

    out = (
        raw.groupby(
            ["site_id", "orientation"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "wins"})
        .sort_values("wins", ascending=False)
    )

    out["nv_index"] = int(nv)
    out["bootstrap_successful_reps"] = len(raw)
    out["bootstrap_win_fraction"] = (
        out["wins"] / len(raw)
    )
    out["bootstrap_n_requested"] = int(BOOTSTRAP_N)

    return out


def run_bootstrap(hyp, t, y, e):
    if not RUN_BOOTSTRAP:
        return pd.DataFrame()

    frames = []
    available = set(hyp["nv_index"].astype(int))

    for nv in BOOTSTRAP_NV_INDICES:
        nv = int(nv)

        if nv not in available:
            print(f"[bootstrap] NV {nv}: unavailable after filtering")
            continue

        print(
            f"[bootstrap] NV {nv}: {BOOTSTRAP_N} replicas, "
            f"top {BOOTSTRAP_TOP_HYPOTHESES} unique orientation+site hypotheses"
        )

        df = bootstrap_one_nv(
            nv,
            hyp,
            t,
            y,
            e,
        )

        if not df.empty:
            frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def cross_validate_one_nv(nv, hyp, t, y, e):
    candidates = (
        hyp[hyp["nv_index"] == nv]
        .sort_values("hypothesis_rank")
        .head(CV_TOP_HYPOTHESES)
        .copy()
    )

    if len(candidates) < 2:
        return pd.DataFrame()

    rows = [row for _, row in candidates.iterrows()]

    n = len(t)
    ntrain = max(
        len(PARAM_NAMES) + 5,
        int(round(CV_TRAIN_FRACTION * n)),
    )
    ntrain = min(ntrain, n - 3)

    def one_rep(rep):
        rng = np.random.default_rng(
            RANDOM_SEED + 200000 * int(nv) + int(rep)
        )

        perm = rng.permutation(n)
        train = np.sort(perm[:ntrain])
        test = np.sort(perm[ntrain:])

        fits = []

        for row in rows:
            try:
                popt, _, _, _, _ = refit_hypothesis(
                    t[train],
                    y[nv, train],
                    e[nv, train],
                    row,
                    CV_MAX_NFEV,
                )

                pred = fine_decay(
                    t[test],
                    *popt,
                )

                test_chi2 = float(
                    np.sum(
                        (
                            (y[nv, test] - pred)
                            / np.maximum(
                                e[nv, test],
                                1e-12,
                            )
                        )
                        ** 2
                    )
                )

                fits.append(
                    {
                        "rep": int(rep),
                        "site_id": int(row["site_id"]),
                        "orientation": str(row["orientation"]),
                        "test_chi2": test_chi2,
                        "n_test": int(len(test)),
                    }
                )

            except Exception:
                continue

        return fits

    with threadpool_limits(limits=1):
        reps = Parallel(
            n_jobs=POST_N_JOBS,
            backend="loky",
            batch_size=1,
        )(
            delayed(one_rep)(i)
            for i in range(int(CV_REPEATS))
        )

    rows_out = [
        x
        for rep in reps
        for x in rep
    ]

    if not rows_out:
        return pd.DataFrame()

    raw = pd.DataFrame(rows_out)

    agg = (
        raw.groupby(
            ["site_id", "orientation"],
            as_index=False,
        )
        .agg(
            mean_test_chi2=("test_chi2", "mean"),
            median_test_chi2=("test_chi2", "median"),
            std_test_chi2=("test_chi2", "std"),
            cv_successful_reps=("rep", "nunique"),
        )
        .sort_values("mean_test_chi2")
        .reset_index(drop=True)
    )

    agg["nv_index"] = int(nv)
    agg["predictive_rank"] = np.arange(
        1,
        len(agg) + 1,
    )

    return agg


def run_cross_validation(hyp, t, y, e):
    if not RUN_CROSS_VALIDATION:
        return pd.DataFrame()

    frames = []
    available = set(hyp["nv_index"].astype(int))

    for nv in CV_NV_INDICES:
        nv = int(nv)

        if nv not in available:
            print(f"[CV] NV {nv}: unavailable after filtering")
            continue

        print(
            f"[CV] NV {nv}: {CV_REPEATS} splits, "
            f"top {CV_TOP_HYPOTHESES} unique orientation+site hypotheses"
        )

        df = cross_validate_one_nv(
            nv,
            hyp,
            t,
            y,
            e,
        )

        if not df.empty:
            frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )


def parameter_landscape_spread(attempts, hyp):
    good = finite_success(attempts)
    good = good[
        good["stage"].astype(str).isin(PRIMARY_STAGES)
    ].copy()

    rows = []

    for nv, sub in hyp.groupby("nv_index", sort=True):
        best = sub.sort_values(
            "hypothesis_rank"
        ).iloc[0]

        same = good[
            (good["nv_index"] == nv)
            & (good["orientation"] == best["orientation"])
            & (good["site_id"] == best["site_id"])
        ].copy()

        if same.empty:
            continue

        delta = (
            same["aicc"].to_numpy(float)
            - float(same["aicc"].min())
        )

        same = same[delta <= 2.0]

        pars = []

        for s in same["popt_json"]:
            p = parse_popt(s)
            if p is not None:
                pars.append(p)

        if not pars:
            continue

        arr = np.vstack(pars)
        q16, q50, q84 = np.percentile(
            arr,
            [16, 50, 84],
            axis=0,
        )

        for j, name in enumerate(PARAM_NAMES):
            rows.append(
                {
                    "nv_index": int(nv),
                    "site_id": int(best["site_id"]),
                    "orientation": str(best["orientation"]),
                    "parameter": name,
                    "n_near_optimal_attempts": int(len(arr)),
                    "q16": float(q16[j]),
                    "median": float(q50[j]),
                    "q84": float(q84[j]),
                    "half_68pct_spread": float(
                        0.5 * (q84[j] - q16[j])
                    ),
                }
            )

    return pd.DataFrame(rows)


def residual_spectrum(t, yv, ev, row):
    curve = curve_from_row(t, row)

    if curve is None:
        return None, None

    dt = np.diff(np.unique(np.asarray(t, float)))
    dt = dt[dt > 0]

    if not len(dt):
        return None, None

    fmax = 500.0 / float(np.min(dt))

    if RESIDUAL_FREQ_MAX_KHZ is not None:
        fmax = min(
            fmax,
            float(RESIDUAL_FREQ_MAX_KHZ),
        )

    freq = np.linspace(
        RESIDUAL_FREQ_MIN_KHZ,
        fmax,
        RESIDUAL_GRID_POINTS,
    )

    rr = (
        (yv - curve)
        / np.maximum(ev, 1e-12)
    )
    rr -= np.nanmean(rr)

    power = lombscargle(
        t,
        rr,
        2 * np.pi * freq / 1000.0,
        normalize=True,
    )

    return freq, power


def save_diagnostic_pdf(
    path,
    t,
    y,
    e,
    hyp,
    nv_summary,
):
    if DIAGNOSTIC_NV_INDICES is None:
        nvs = nv_summary[
            "nv_index"
        ].astype(int).tolist()
    else:
        wanted = set(
            int(x)
            for x in DIAGNOSTIC_NV_INDICES
        )
        nvs = [
            int(x)
            for x in nv_summary["nv_index"]
            if int(x) in wanted
        ]

    td = np.linspace(
        float(np.min(t)),
        float(np.max(t)),
        DENSE_CURVE_POINTS,
    )

    with PdfPages(path) as pdf:
        for nv in nvs:
            sub = (
                hyp[hyp["nv_index"] == nv]
                .sort_values("hypothesis_rank")
                .head(8)
            )

            if sub.empty:
                continue

            fig, axes = plt.subplots(
                2,
                2,
                figsize=(13, 9),
            )

            ax = axes[0, 0]

            ax.errorbar(
                t,
                y[nv],
                yerr=e[nv],
                fmt="o",
                ms=3,
                capsize=1,
                lw=0.55,
                label="data",
            )

            for j, (_, row) in enumerate(
                sub.head(
                    TOP_UNIQUE_HYPOTHESES_PLOT
                ).iterrows()
            ):
                curve = curve_from_row(
                    td,
                    row,
                )

                if curve is None:
                    continue

                ax.plot(
                    td,
                    curve,
                    ls=["-", "--", ":"][min(j, 2)],
                    lw=1.5 if j == 0 else 1.0,
                    label=(
                        f"#{int(row['hypothesis_rank'])} "
                        f"site {int(row['site_id'])} "
                        f"{row['orientation']}"
                    ),
                )

            ax.set_xlabel(
                "Total evolution time (µs)"
            )
            ax.set_ylabel(
                "Normalized signal"
            )
            ax.set_title(
                f"NV {nv}: dense top-3 unique fits"
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)

            ax = axes[0, 1]

            show = sub.head(8)

            labels = [
                f"{int(r.site_id)}\n{r.orientation}"
                for r in show.itertuples()
            ]

            ax.bar(
                np.arange(len(show)),
                show["akaike_weight"].to_numpy(float),
            )

            ax.set_xticks(
                np.arange(len(show))
            )
            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right",
                fontsize=7,
            )
            ax.set_ylabel(
                "Akaike weight"
            )
            ax.set_title(
                "Deep unique orientation+site hypotheses"
            )
            ax.grid(
                alpha=0.2,
                axis="y",
            )

            ax = axes[1, 0]

            freq, power = residual_spectrum(
                t,
                y[nv],
                e[nv],
                sub.iloc[0],
            )

            if freq is not None:
                ax.plot(
                    freq,
                    power,
                    lw=1.0,
                )

            ax.set_xlabel(
                "Residual frequency (kHz)"
            )
            ax.set_ylabel(
                "Lomb-Scargle power"
            )
            ax.set_title(
                "Whitened residual spectrum"
            )
            ax.grid(alpha=0.2)

            ax = axes[1, 1]
            ax.axis("off")

            best = sub.iloc[0]
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

            lines = [
                f"NV {nv}",
                "",
                (
                    f"rank 1: site {int(best['site_id'])} "
                    f"{best['orientation']}"
                ),
                (
                    f"  f0/f1 = "
                    f"{best['f0_kHz']:.2f}, "
                    f"{best['f1_kHz']:.2f} kHz"
                ),
                f"  kappa = {best['kappa']:.4g}",
                f"  distance = {best['distance_A']:.3f} A",
                f"  red chi2 = {best['red_chi2']:.4g}",
                f"  weight = {best['akaike_weight']:.3f}",
                f"  family weight = {best['family_weight']:.3f}",
            ]

            if second is not None:
                lines += [
                    "",
                    (
                        f"rank 2: site {int(second['site_id'])} "
                        f"{second['orientation']}"
                    ),
                    f"  Delta AICc = {second['delta_aicc']:.3f}",
                    f"  weight = {second['akaike_weight']:.3f}",
                ]

            if third is not None:
                lines += [
                    "",
                    (
                        f"rank 3: site {int(third['site_id'])} "
                        f"{third['orientation']}"
                    ),
                    f"  Delta AICc = {third['delta_aicc']:.3f}",
                    f"  weight = {third['akaike_weight']:.3f}",
                ]

            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                va="top",
                ha="left",
                family="monospace",
                fontsize=9.5,
            )

            fig.tight_layout()
            pdf.savefig(
                fig,
                bbox_inches="tight",
            )
            plt.close(fig)

    print(f"Saved: {path}")


def make_global_summary(nv_summary):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    ax = axes[0, 0]
    vals = nv_summary[
        "second_delta_aicc"
    ].to_numpy(float)
    vals = vals[np.isfinite(vals)]

    ax.hist(
        vals,
        bins=35,
    )
    ax.set_xlabel(
        "ΔAICc: rank 2 - rank 1"
    )
    ax.set_ylabel(
        "NV count"
    )
    ax.set_title(
        "Unique-hypothesis separation"
    )
    ax.grid(alpha=0.2)

    ax = axes[0, 1]

    ax.hist(
        nv_summary[
            "best_hypothesis_weight"
        ].dropna(),
        bins=30,
        alpha=0.75,
        label="orientation+site",
    )

    ax.hist(
        nv_summary[
            "best_family_weight"
        ].dropna(),
        bins=30,
        alpha=0.55,
        label="spectral family",
    )

    ax.set_xlabel(
        "Model weight"
    )
    ax.set_ylabel(
        "NV count"
    )
    ax.legend()
    ax.set_title(
        "Hypothesis vs family concentration"
    )
    ax.grid(alpha=0.2)

    ax = axes[1, 0]

    ax.scatter(
        nv_summary["best_red_chi2"],
        nv_summary["second_delta_aicc"],
        s=14,
    )

    ax.set_xlabel(
        "Best reduced χ²"
    )
    ax.set_ylabel(
        "ΔAICc to rank 2"
    )
    ax.set_title(
        "Fit quality vs site separation"
    )
    ax.grid(alpha=0.2)

    ax = axes[1, 1]

    cats = [
        "near_degenerate",
        "some_separation",
        "substantial_separation",
        "large_separation",
    ]

    counts = [
        int(
            (
                nv_summary["separation_label"]
                == c
            ).sum()
        )
        for c in cats
    ]

    ax.bar(
        np.arange(len(cats)),
        counts,
    )

    ax.set_xticks(
        np.arange(len(cats))
    )
    ax.set_xticklabels(
        ["Δ<2", "2–6", "6–10", "≥10"]
    )
    ax.set_ylabel(
        "NV count"
    )
    ax.set_title(
        "ΔAICc confidence classes"
    )
    ax.grid(
        alpha=0.2,
        axis="y",
    )

    fig.suptitle(
        f"{RESULT_TAG}: fair deep-fit confidence | "
        f"{len(nv_summary)} NVs",
        fontsize=14,
    )

    fig.tight_layout()
    return fig


def save_table(df, path):
    if df is None or df.empty:
        return
    df.to_csv(
        path,
        index=False,
    )
    print(f"Saved: {path}")


def main():
    paths = discover_paths()

    (
        attempts,
        ckpt,
        t,
        y,
        e,
        expected_revival,
    ) = load_inputs(paths)

    known_map = load_known_orientation_map()

    print(
        f"[orientation] known NV orientations available: "
        f"{len(known_map)}"
    )

    outdir = (
        Path(OUTPUT_DIR)
        if OUTPUT_DIR is not None
        else paths.prefix.parent
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = outdir / (
        paths.prefix.name
        + "_confidence_v2"
    )

    hyp = build_deep_hypotheses(
        attempts,
        known_map,
        n_points=len(t),
    )

    screen_challengers = (
        build_screen_challengers(
            attempts,
            hyp,
            known_map,
        )
    )

    hyp, families = assign_families(
        hyp
    )

    nv_summary = build_nv_summary(
        hyp,
        families,
        known_map,
    )

    top_unique = top_unique_table(
        hyp
    )

    residuals = residual_peak_table(
        t,
        y,
        e,
        hyp,
    )

    landscape = (
        parameter_landscape_spread(
            attempts,
            hyp,
        )
    )

    bootstrap = run_bootstrap(
        hyp,
        t,
        y,
        e,
    )

    cv = run_cross_validation(
        hyp,
        t,
        y,
        e,
    )

    save_table(
        hyp,
        Path(
            str(base)
            + "_deep_unique_hypotheses.csv"
        ),
    )

    save_table(
        top_unique,
        Path(
            str(base)
            + "_top10_unique_hypotheses.csv"
        ),
    )

    save_table(
        nv_summary,
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
        screen_challengers,
        Path(
            str(base)
            + "_screen_only_challengers.csv"
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
        landscape,
        Path(
            str(base)
            + "_parameter_landscape_spread.csv"
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

    if SAVE_TOP3_DENSE_FULL_PDF:
        save_top3_dense_pdf(
            Path(
                str(base)
                + "_top3_unique_dense_fits.pdf"
            ),
            t,
            y,
            e,
            hyp,
            expected_revival,
            first_revival_only=False,
        )

    if SAVE_TOP3_DENSE_FIRST_REVIVAL_PDF:
        save_top3_dense_pdf(
            Path(
                str(base)
                + "_top3_unique_dense_first_revival.pdf"
            ),
            t,
            y,
            e,
            hyp,
            expected_revival,
            first_revival_only=True,
        )

    if SAVE_DIAGNOSTIC_PDF:
        save_diagnostic_pdf(
            Path(
                str(base)
                + "_nv_diagnostics.pdf"
            ),
            t,
            y,
            e,
            hyp,
            nv_summary,
        )

    fig = make_global_summary(
        nv_summary
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

        print(f"Saved: {png}")
        print(f"Saved: {pdf}")

    print()
    print("=" * 92)
    print("CONFIDENCE V2 COMPLETE")
    print("=" * 92)
    print(
        f"NVs analyzed                    : "
        f"{len(nv_summary)}"
    )
    print(
        f"deep unique hypotheses          : "
        f"{len(hyp)}"
    )
    print(
        f"family tolerance                : "
        f"{FAMILY_TOL_KHZ:.3f} kHz"
    )
    print(
        f"near-degenerate (ΔAICc < 2)     : "
        f"{int((nv_summary['separation_label'] == 'near_degenerate').sum())}"
    )
    print(
        f"large separation (ΔAICc >= 10)  : "
        f"{int((nv_summary['separation_label'] == 'large_separation').sum())}"
    )
    print(
        "top-3 PDF identity             : "
        "(orientation, site_id) unique"
    )
    print(
        f"dense fit points per curve      : "
        f"{DENSE_CURVE_POINTS}"
    )

    if len(known_map):
        print(
            f"known orientation constraints    : "
            f"{len(known_map)} NVs"
        )

    if not bootstrap.empty:
        print(
            "bootstrap                        : saved"
        )

    if not cv.empty:
        print(
            "held-out validation              : saved"
        )

    print("=" * 92)

    if SHOW_GLOBAL_SUMMARY:
        plt.show(
            block=True
        )
    else:
        plt.close(
            fig
        )

    return {
        "paths": paths,
        "deep_unique_hypotheses": hyp,
        "screen_challengers": screen_challengers,
        "families": families,
        "nv_summary": nv_summary,
        "top_unique": top_unique,
        "residual_peaks": residuals,
        "parameter_landscape": landscape,
        "bootstrap": bootstrap,
        "cross_validation": cv,
    }


if __name__ == "__main__":
    main()
