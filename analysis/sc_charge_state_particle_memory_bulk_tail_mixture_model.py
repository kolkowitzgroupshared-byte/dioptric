# -*- coding: utf-8 -*-
"""
Raw (non-truth-filtered) phenomenological analysis of NV charge-memory data.

This script deliberately separates TWO questions:

1) BULK / PEAK SHIFT
   What does the motion of the main NV- -> NV0 loss distribution with dark
   time tell us about ordinary dark charge dynamics in the diamond?

2) EXTREME UPPER TAIL
   How often do unusually large same-run loss coincidences occur, how does
   that rate scale with exposure time, and is that rate scale comparable to
   the geometric cosmic-muon crossing rate?

IMPORTANT
---------
* This script DOES NOT use the transition-truth / misclassification filter.
* It uses the original hard charge-state classification only.
* It DOES NOT load image pixels and DOES NOT perform spatial analysis.
* Cosmic muons are used only as an external RATE-SCALE comparison.  A rate
  agreement does not identify individual events as muons.

Definitions
-----------
For run r:
    K_r = number of hard-classified NV- -> NV0 losses
    N_r = number of evaluable initially-NV- centers
    f_r = K_r / N_r

A site is evaluable when it is initially NV- and the final charge state can
be classified as either NV- (retained) or NV0 (lost).

BULK MODEL
----------
The central count distribution is described phenomenologically by

    K_r ~ BetaBinomial(N_r, p_bulk(t), rho_bulk(t)).

p_bulk sets the center of the ordinary loss distribution; rho_bulk captures
extra run-to-run width beyond an independent binomial model.

The fitted central loss probability is summarized by

    p_bulk(t) = 1 - (1-p0) exp(-Gamma_dark t),

where p0 is the finite baseline at nominal zero dark wait.  With only a few
wait times, Gamma_dark and tau_dark=1/Gamma_dark should be called effective
kinetic parameters, not proof of a unique microscopic exponential process.

UPPER-TAIL MODEL
----------------
For each run, use the central bulk probability to define the local ordinary
Poisson expectation

    lambda_r = N_r p_bulk(t).

The exact upper-tail probability is

    p_tail,r = P[X >= K_r | X ~ Poisson(lambda_r)],

which is converted to a one-sided Gaussian-equivalent screening coordinate

    z_r = Phi^{-1}(1-p_tail,r).

The script reports >=3, >=4 and >=5 sigma event fractions.  These sigma
values are screening coordinates, not formal discovery significances.

For each sigma threshold s, the observed event counts M_s(t) are fitted with

    M_s(t) ~ Binomial(n_t, q_s(t)).

Models compared:

    constant background:
        q_s(T) = q_bg

    background + free exposure-dependent rate:
        q_s(T) = 1 - (1-q_bg) exp(-R_tail T)

    background + geometric cosmic rate fixed:
        q_s(T) = 1 - (1-q_bg) exp(-R_mu T)

    pure geometric cosmic crossing:
        q_s(T) = 1 - exp(-R_mu T)

The fitted ratio

    epsilon_eff = R_tail / R_mu

is reported as an EFFECTIVE RATE-SCALE RATIO, not automatically a detector
or muon efficiency.

OUTPUTS
-------
analysis_output/bulk_tail_phenomenology/
    raw_bulk_summary.csv
    bulk_trim_sensitivity.csv
    bulk_kinetic_fit_by_trim.csv
    tail_event_summary.csv
    tail_rate_fit_summary.csv
    tail_events.csv
    run_level_metrics.csv
    muon_geometry_summary.csv
    analysis_summary.txt

    bulk_peak_fit_0s.png
    bulk_peak_fit_30s.png
    bulk_peak_fit_60s.png
    bulk_peak_fit_90s.png
    bulk_probability_vs_dark_wait.png
    bulk_gamma_vs_tail_trim.png
    tail_poisson_sigma_0s.png
    tail_poisson_sigma_30s.png
    tail_poisson_sigma_60s.png
    tail_poisson_sigma_90s.png
    tail_rate_fit_3sigma.png
    tail_rate_fit_4sigma.png
    tail_rate_fit_5sigma.png
    tail_rate_scale_vs_sigma.png

Expected environment
--------------------
This uses the safe counts-only helpers and DATASETS from

    sc_charge_state_particle_memory_spatial_model.py

which are already used by the existing particle-memory analyses.

MULTIPLE FILES AT THE SAME WAIT
-------------------------------
Each physical wait condition may be represented by ONE OR MANY acquisition
files. The code accepts any of these base.DATASETS styles:

    "file_stem": "one_file"

    "file_stem": (
        "file_1",
        "file_2",
    )

    "file_stems": [
        "file_1",
        "file_2",
        "file_3",
    ]

It also supports several separate DATASETS entries that parse to the same wait.

Every physical source file is loaded, quality-screened, and hard-classified
using ITS OWN saved threshold vector. Only then are its valid K/N runs appended
to the other acquisitions at the same physical wait. One common wait-specific
beta-binomial bulk model p_bulk(t), rho_bulk(t) and one upper-tail analysis are
then fit to that combined population.

Different wait conditions are NEVER appended together.

Each combined run retains:
    original_run        unique appended/global raw-run index
    source_file_ind     physical acquisition index
    source_local_run    original run number within that source
    source_label
    source_file_stem
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize, minimize_scalar
from scipy.special import betaln, gammaln
from scipy.stats import binom, norm, poisson

import sc_charge_state_particle_memory_spatial_model as base
from utils import data_manager as dm


# =============================================================================
# USER CONFIGURATION
# =============================================================================

WANTED_WAITS_S = (0.0, 30.0, 60.0, 90.0)

# Measured wait durations from the experiment timing metadata.
ACTUAL_DARK_WAIT_S = {
    0.0: 0.000002,
    30.0: 30.005827,
    60.0: 60.007120,

    # Replace with the measured 90-s timing metadata if/when you have a more
    # precise value.  Using 90.0 s is preferable to inventing sub-ms timing.
    90.0: 90.0,
}

# Estimated additional center-to-center timing between the rep11 and rep12
# readout windows.  This is used ONLY for the particle-exposure comparison.
INTER_READOUT_CORRECTION_S = 0.63

# Saved charge-state reps.
REP_INITIAL = 11
REP_FINAL = 12

# Charge-state hard classification margin.  Keep 0.0 to reproduce the raw
# threshold analysis.  This is intentionally NOT the truth-test margin.
RAW_MARGIN_COUNTS = 0.0

# Quality rejection: same severe global-collapse removal used in the main
# analysis.  No additional truth/misclassification filtering is done.
REJECT_GLOBAL_DROP_RUNS = True

# -----------------------------------------------------------------------------
# Bulk / peak fitting
# -----------------------------------------------------------------------------

# Central bulk fit iteratively excludes only the HIGH-SIDE tail above this
# beta-binomial standardized residual.  This is used to estimate the peak/
# central distribution without letting a few upper-tail runs move it.
BULK_CORE_Z_CUT = 3.0
BULK_CORE_MAX_ITER = 10
BULK_CORE_MIN_FRACTION = 0.70

BULK_P_BOUNDS = (1e-5, 0.25)
BULK_RHO_BOUNDS = (1e-9, 0.05)

# Independent tail-removal sensitivity check.  Runs are ranked by f=K/N and
# the largest fractions are removed before refitting the bulk.
TRIM_FRACTIONS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)

# -----------------------------------------------------------------------------
# Upper-tail screening
# -----------------------------------------------------------------------------

SIGMA_THRESHOLDS = (3.0, 4.0, 5.0)
TOP_TAIL_RUNS_TO_PRINT = 20

# -----------------------------------------------------------------------------
# Tail-rate fit
# -----------------------------------------------------------------------------

QBG_BOUNDS = (1e-8, 0.20)
RTAIL_BOUNDS_S = (0.0, 5.0e-3)

# Parametric bootstrap of the fitted binomial tail-rate model.  This is very
# cheap because it refits only a few wait-level event counts, not the raw NV data.
TAIL_BOOTSTRAP_REPS = 400
TAIL_BOOTSTRAP_SEED = 260828

# -----------------------------------------------------------------------------
# Cosmic-muon geometric reference
# -----------------------------------------------------------------------------

# Total downward muon flux convention used in the directional cos^2(theta)
# treatment.  Units: s^-1 cm^-2.
MUON_FLUX_TOTAL = 0.0133
MUON_FLUX_TOTAL_SE = 0.0008

# Diamond dimensions in cm: 2 mm x 1 mm x 50 um.
DIAMOND_L_CM = 0.20
DIAMOND_W_CM = 0.10
DIAMOND_T_CM = 0.005

# For a horizontal rectangular slab and an angular distribution proportional
# to cos^2(theta), the angular-averaged effective cross section is
#
#   A_eff = 3/4 L W + 3/8 t (L+W).
#
# This is the primary geometric acceptance used below.

SAVE_OUTPUTS = False
SHOW_FIGURES = True
OUTPUT_DIR = Path("analysis_output") / "bulk_tail_phenomenology"

SCRIPT_VERSION = (
    "RAW_BULK_TAIL_V3_MULTI_FILE_PER_WAIT_APPEND_2026-08-29"
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def _write_csv(path, rows):
    if not SAVE_OUTPUTS or not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _actual_wait(wait_s):
    wait_s = float(wait_s)
    for key, value in ACTUAL_DARK_WAIT_S.items():
        if abs(float(key) - wait_s) < 1e-8:
            return float(value)
    return wait_s


def _effective_particle_exposure(wait_s):
    return _actual_wait(wait_s) + float(INTER_READOUT_CORRECTION_S)


def _parse_dark_wait_s(dataset, metadata=None):
    for src in (dataset, metadata or {}):
        for key in ("dark_wait_s", "wait_s", "dark_time_s"):
            if key in src:
                try:
                    return float(src[key])
                except Exception:
                    pass

    file_text = dataset.get(
        "file_stems",
        dataset.get("file_stem", ""),
    )
    text = f"{dataset.get('label', '')} {file_text}"

    for pat in (
        r"wait[_-]?(\d+(?:\.\d+)?)s",
        r"source[_-]?off[_-]?(\d+(?:\.\d+)?)s",
        r"dark[_-]?wait[_-]?(\d+(?:\.\d+)?)s",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return np.nan


def _safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den != 0))
    return out



# =============================================================================
# MULTI-FILE / SAME-WAIT CONFIGURATION HELPERS
# =============================================================================


def _cfg_file_stems(dataset):
    """
    Return all physical acquisition stems represented by one logical config.
    """
    if "file_stems" in dataset:
        stems = dataset.get("file_stems")
    else:
        stems = dataset.get("file_stem")

    if isinstance(stems, (list, tuple)):
        stems = list(stems)
    else:
        stems = [stems]

    stems = [
        str(stem)
        for stem in stems
        if stem is not None
    ]

    if not stems:
        raise ValueError(
            f"{dataset.get('label', '<unnamed>')}: no file stems configured."
        )

    return stems


def _cfg_npz_overrides(dataset, num_files):
    """
    Resolve optional NPZ overrides for one logical configuration.
    """
    if "npz_path_overrides" in dataset:
        overrides = dataset.get("npz_path_overrides")
    else:
        overrides = dataset.get("npz_path_override")

    if overrides is None:
        return [None] * int(num_files)

    if isinstance(overrides, (list, tuple)):
        overrides = list(overrides)
    else:
        if int(num_files) != 1:
            raise ValueError(
                f"{dataset.get('label', '<unnamed>')}: "
                f"{num_files} file stems require npz_path_overrides=[...] "
                "or no explicit override."
            )
        overrides = [overrides]

    if len(overrides) != int(num_files):
        raise ValueError(
            f"{dataset.get('label', '<unnamed>')}: "
            f"{len(overrides)} NPZ overrides for {num_files} file stems."
        )

    return overrides


def _expand_dataset_to_single_file_sources(
    dataset,
    wait_s=None,
):
    """
    Convert one logical config into one ordinary single-file config per source.

    The returned configs are safe to pass to _load_counts_thresholds() and
    _prepare_single_dataset().
    """
    stems = _cfg_file_stems(dataset)
    overrides = _cfg_npz_overrides(
        dataset,
        len(stems),
    )

    if wait_s is None:
        wait_s = _parse_dark_wait_s(dataset)

    base_label = str(
        dataset.get(
            "label",
            (
                f"dark_wait_{wait_s:g}s"
                if np.isfinite(wait_s)
                else "dataset"
            ),
        )
    )

    out = []

    for source_ind, (stem, override) in enumerate(
        zip(stems, overrides)
    ):
        cfg = dict(dataset)

        cfg.pop("file_stems", None)
        cfg.pop("npz_path_overrides", None)
        cfg.pop("source_cfgs", None)

        cfg["file_stem"] = str(stem)
        cfg["npz_path_override"] = override
        cfg["label"] = (
            base_label
            if len(stems) == 1
            else f"{base_label}_part{source_ind + 1}"
        )

        cfg["_source_file_ind"] = int(source_ind)
        cfg["_parent_label"] = base_label

        if np.isfinite(wait_s):
            cfg["dark_wait_s"] = float(wait_s)

        out.append(cfg)

    return out


def _source_identity_for_run(ds, run_ind):
    """
    Return exact acquisition identity for one valid appended run.
    """
    run_ind = int(run_ind)

    if "source_file_ind" not in ds:
        return {
            "source_file_ind": 0,
            "source_local_run": int(
                ds["original_run"][run_ind]
            ),
            "source_label": str(
                ds.get("label", "")
            ),
            "source_file_stem": str(
                ds.get("file_stem", "")
            ),
        }

    return {
        "source_file_ind": int(
            ds["source_file_ind"][run_ind]
        ),
        "source_local_run": int(
            ds["source_local_run"][run_ind]
        ),
        "source_label": str(
            ds["source_label"][run_ind]
        ),
        "source_file_stem": str(
            ds["source_file_stem"][run_ind]
        ),
    }


# =============================================================================
# SAFE COUNTS-ONLY LOADING
# =============================================================================


def _require_base_helper(name):
    fn = getattr(base, name, None)
    if not callable(fn):
        raise RuntimeError(
            f"Required helper base.{name} is unavailable.  "
            "Run this with the same sc_charge_state_particle_memory_spatial_model.py "
            "used by the current particle-memory analysis."
        )
    return fn


def _load_metadata_without_npz(file_stem):
    helper = getattr(base, "_try_metadata_without_npz", None)
    if callable(helper):
        try:
            metadata = helper(file_stem)
            if isinstance(metadata, dict):
                return metadata
        except Exception as exc:
            print(
                f"[metadata] base helper failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    try:
        try:
            metadata = dm.get_raw_data(
                file_stem=file_stem,
                load_npz=False,
                use_cache=False,
            )
        except TypeError:
            metadata = dm.get_raw_data(
                file_stem=file_stem,
                load_npz=False,
            )
    except Exception as exc:
        print(
            f"[metadata] fallback failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None

    return metadata if isinstance(metadata, dict) else None


def _load_counts_thresholds(dataset):
    label = str(dataset["label"])
    file_stem = str(dataset["file_stem"])

    print("\n" + "=" * 120)
    print(f"RAW COUNTS-ONLY DATASET: {label}")
    print("=" * 120)

    metadata = _load_metadata_without_npz(file_stem)

    discover = _require_base_helper("_discover_npz_path")
    npz_path = discover(
        file_stem=file_stem,
        npz_path_override=dataset.get("npz_path_override"),
        metadata=metadata,
    )
    print(f"[counts-only] using NPZ: {npz_path}")

    load_reps = _require_base_helper("_load_count_reps_streaming")
    c11, c12 = load_reps(
        npz_path,
        rep_initial=REP_INITIAL,
        rep_final=REP_FINAL,
    )
    c11 = np.asarray(c11, dtype=np.float32)
    c12 = np.asarray(c12, dtype=np.float32)

    n_nv, n_run = c11.shape
    thresholds = None
    dark_wait_npz = None

    # Access only small NPZ members.  img_arrays is never read/decompressed.
    with np.load(npz_path, allow_pickle=True) as archive:
        keys = set(archive.files)
        for key in ("analysis_thresholds", "thresholds"):
            if key in keys:
                thresholds = np.asarray(archive[key], dtype=np.float32).ravel()
                break
        if "dark_wait_s" in keys:
            try:
                dark_wait_npz = float(np.asarray(archive["dark_wait_s"]).item())
            except Exception:
                pass

    if thresholds is None and isinstance(metadata, dict):
        for key in ("analysis_thresholds", "thresholds"):
            if key in metadata:
                thresholds = np.asarray(metadata[key], dtype=np.float32).ravel()
                break

    if thresholds is None:
        raise ValueError("Could not find analysis_thresholds or thresholds.")
    if thresholds.shape != (n_nv,):
        raise ValueError(
            f"Threshold shape {thresholds.shape} does not match NV count {n_nv}."
        )

    wait_s = _parse_dark_wait_s(dataset, metadata)
    if not np.isfinite(wait_s) and dark_wait_npz is not None:
        wait_s = dark_wait_npz

    print(
        f"[counts-only] NVs={n_nv}, runs={n_run}, nominal dark wait={wait_s:g} s"
    )

    return {
        "label": label,
        "file_stem": file_stem,
        "npz_path": str(npz_path),
        "metadata": metadata,
        "c11": c11,
        "c12": c12,
        "thresholds": thresholds,
        "wait_s": float(wait_s),
    }


# =============================================================================
# RAW HARD CLASSIFICATION
# =============================================================================


def _classify_raw(c11, c12, thresholds, margin=0.0):
    c11 = np.asarray(c11, dtype=float)
    c12 = np.asarray(c12, dtype=float)
    thr = np.asarray(thresholds, dtype=float).ravel()[:, None]
    m = float(margin)

    finite11 = np.isfinite(c11) & np.isfinite(thr)
    finite12 = np.isfinite(c12) & np.isfinite(thr)

    initial_nvm = finite11 & (c11 > thr + m)
    initial_nv0 = finite11 & (c11 <= thr - m)
    final_nvm = finite12 & (c12 > thr + m)
    final_nv0 = finite12 & (c12 <= thr - m)

    loss = initial_nvm & final_nv0
    retained_nvm = initial_nvm & final_nvm
    gain = initial_nv0 & final_nvm
    retained_nv0 = initial_nv0 & final_nv0

    # The loss denominator contains only initially-NV- sites whose final charge
    # state is also confidently classifiable.
    loss_evaluable = loss | retained_nvm

    return {
        "initial_nvm": initial_nvm,
        "initial_nv0": initial_nv0,
        "final_nvm": final_nvm,
        "final_nv0": final_nv0,
        "loss": loss,
        "gain": gain,
        "retained_nvm": retained_nvm,
        "retained_nv0": retained_nv0,
        "loss_evaluable": loss_evaluable,
        "loss_count": np.sum(loss, axis=0).astype(int),
        "gain_count": np.sum(gain, axis=0).astype(int),
        "loss_evaluable_count": np.sum(loss_evaluable, axis=0).astype(int),
        "initial_nvm_count": np.sum(initial_nvm, axis=0).astype(int),
        "final_nvm_count": np.sum(final_nvm, axis=0).astype(int),
    }


def _good_run_mask(c11, c12):
    n_run = c11.shape[1]
    if not REJECT_GLOBAL_DROP_RUNS:
        return np.ones(n_run, dtype=bool), None

    detect = _require_base_helper("_detect_global_drop_runs")
    quality = detect(
        c11,
        c12,
        min_total_fraction=getattr(base, "MIN_RUN_TOTAL_FRACTION", 0.50),
        per_nv_collapse_fraction=getattr(base, "PER_NV_COLLAPSE_FRACTION", 0.25),
        max_collapsed_nv_fraction=getattr(base, "MAX_COLLAPSED_NV_FRACTION", 0.80),
    )
    good = np.asarray(quality["good_run_mask"], dtype=bool)
    return good, quality


def _prepare_single_dataset(dataset):
    """
    Original one-file preparation path.

    Each physical acquisition is independently:
      * loaded,
      * quality-screened,
      * classified with its own threshold vector.
    """
    small = _load_counts_thresholds(dataset)
    c11 = small["c11"]
    c12 = small["c12"]
    thresholds = small["thresholds"]

    good, quality = _good_run_mask(
        c11,
        c12,
    )
    raw = _classify_raw(
        c11,
        c12,
        thresholds,
        margin=RAW_MARGIN_COUNTS,
    )

    K_all = np.asarray(
        raw["loss_count"],
        dtype=int,
    )
    N_all = np.asarray(
        raw["loss_evaluable_count"],
        dtype=int,
    )
    frac_all = _safe_divide(
        K_all,
        N_all,
    )

    valid = (
        good
        & (N_all > 0)
        & np.isfinite(frac_all)
    )
    original_run = np.where(
        valid
    )[0].astype(int)

    source_file_ind = int(
        dataset.get(
            "_source_file_ind",
            0,
        )
    )

    ds = {
        **small,

        "quality": quality,
        "good_run_mask": good,
        "valid_run_mask": valid,

        "original_run": original_run,

        "K": K_all[valid],
        "N": N_all[valid],
        "loss_fraction": frac_all[valid],

        "initial_nvm_count": np.asarray(
            raw["initial_nvm_count"],
            dtype=int,
        )[valid],
        "final_nvm_count": np.asarray(
            raw["final_nvm_count"],
            dtype=int,
        )[valid],

        "actual_wait_s": _actual_wait(
            small["wait_s"]
        ),
        "particle_exposure_s": (
            _effective_particle_exposure(
                small["wait_s"]
            )
        ),

        # Source identity is present even for a one-file dataset.
        "source_file_ind": np.full(
            int(np.sum(valid)),
            source_file_ind,
            dtype=int,
        ),
        "source_local_run": original_run.copy(),
        "source_label": np.full(
            int(np.sum(valid)),
            str(small["label"]),
            dtype=object,
        ),
        "source_file_stem": np.full(
            int(np.sum(valid)),
            str(small["file_stem"]),
            dtype=object,
        ),

        "num_source_files": 1,
        "raw_runs_loaded": int(
            c11.shape[1]
        ),
    }

    print(
        f"[quality] good/evaluable runs = "
        f"{len(ds['K'])}/{c11.shape[1]}; "
        f"rejected global-drop = "
        f"{int(np.sum(~good))}"
    )
    print(
        f"[raw losses] mean K="
        f"{np.mean(ds['K']):.4f}, "
        f"pooled p="
        f"{100*np.sum(ds['K'])/np.sum(ds['N']):.4f}%"
    )

    return ds


def _append_prepared_same_wait(
    prepared_parts,
    wait_s,
    label,
):
    """
    Append valid runs from multiple independently prepared acquisitions.

    IMPORTANT:
    thresholding and severe quality rejection have ALREADY occurred inside each
    source file. Therefore this append does not mix threshold vectors between
    acquisitions.
    """
    if not prepared_parts:
        raise ValueError(
            f"No prepared datasets for wait={wait_s:g} s."
        )

    K_parts = []
    N_parts = []
    frac_parts = []
    init_parts = []
    final_parts = []

    global_run_parts = []
    source_file_ind_parts = []
    source_local_run_parts = []
    source_label_parts = []
    source_stem_parts = []

    good_mask_parts = []
    valid_mask_parts = []

    thresholds_by_source = []
    npz_paths = []
    file_stems = []
    source_summaries = []

    raw_offset = 0

    for source_ind, part in enumerate(
        prepared_parts
    ):
        part_wait = float(part["wait_s"])

        if abs(
            part_wait - float(wait_s)
        ) > 1e-6:
            raise ValueError(
                f"Cannot append source {source_ind}: "
                f"wait={part_wait:g} s differs from "
                f"group wait={wait_s:g} s."
            )

        K = np.asarray(
            part["K"],
            dtype=int,
        )
        N = np.asarray(
            part["N"],
            dtype=int,
        )
        local_run = np.asarray(
            part["original_run"],
            dtype=int,
        )

        if not (
            K.shape == N.shape == local_run.shape
        ):
            raise ValueError(
                f"Source {source_ind} has inconsistent valid-run arrays."
            )

        K_parts.append(K)
        N_parts.append(N)
        frac_parts.append(
            np.asarray(
                part["loss_fraction"],
                dtype=float,
            )
        )
        init_parts.append(
            np.asarray(
                part["initial_nvm_count"],
                dtype=int,
            )
        )
        final_parts.append(
            np.asarray(
                part["final_nvm_count"],
                dtype=int,
            )
        )

        global_run_parts.append(
            raw_offset + local_run
        )

        source_file_ind_parts.append(
            np.full(
                len(K),
                source_ind,
                dtype=int,
            )
        )
        source_local_run_parts.append(
            local_run
        )
        source_label_parts.append(
            np.full(
                len(K),
                str(part["label"]),
                dtype=object,
            )
        )
        source_stem_parts.append(
            np.full(
                len(K),
                str(part["file_stem"]),
                dtype=object,
            )
        )

        good_mask_parts.append(
            np.asarray(
                part["good_run_mask"],
                dtype=bool,
            )
        )
        valid_mask_parts.append(
            np.asarray(
                part["valid_run_mask"],
                dtype=bool,
            )
        )

        thresholds_by_source.append(
            np.asarray(
                part["thresholds"],
                dtype=np.float32,
            )
        )
        npz_paths.append(
            str(part["npz_path"])
        )
        file_stems.append(
            str(part["file_stem"])
        )

        source_summaries.append(
            {
                "source_file_ind": int(
                    source_ind
                ),
                "label": str(
                    part["label"]
                ),
                "file_stem": str(
                    part["file_stem"]
                ),
                "npz_path": str(
                    part["npz_path"]
                ),
                "raw_runs_loaded": int(
                    part.get(
                        "raw_runs_loaded",
                        len(part["good_run_mask"]),
                    )
                ),
                "valid_runs": int(
                    len(part["K"])
                ),
            }
        )

        raw_offset += int(
            part.get(
                "raw_runs_loaded",
                len(part["good_run_mask"]),
            )
        )

    combined = {
        "label": str(label),

        # Lists because this is a logical wait-level dataset.
        "file_stem": file_stems,
        "npz_path": npz_paths,

        "wait_s": float(wait_s),
        "actual_wait_s": float(
            np.median(
                [
                    part["actual_wait_s"]
                    for part in prepared_parts
                ]
            )
        ),
        "particle_exposure_s": float(
            np.median(
                [
                    part["particle_exposure_s"]
                    for part in prepared_parts
                ]
            )
        ),

        # Keep source-specific threshold vectors explicit.
        "thresholds": thresholds_by_source[0],
        "thresholds_by_source": thresholds_by_source,

        # Raw acquisition-level masks, appended only for bookkeeping.
        "good_run_mask": np.concatenate(
            good_mask_parts
        ),
        "valid_run_mask": np.concatenate(
            valid_mask_parts
        ),

        # Valid analysis population.
        "original_run": np.concatenate(
            global_run_parts
        ).astype(int),

        "K": np.concatenate(
            K_parts
        ).astype(int),
        "N": np.concatenate(
            N_parts
        ).astype(int),
        "loss_fraction": np.concatenate(
            frac_parts
        ).astype(float),

        "initial_nvm_count": np.concatenate(
            init_parts
        ).astype(int),
        "final_nvm_count": np.concatenate(
            final_parts
        ).astype(int),

        # Exact physical-source identity for every valid run.
        "source_file_ind": np.concatenate(
            source_file_ind_parts
        ).astype(int),
        "source_local_run": np.concatenate(
            source_local_run_parts
        ).astype(int),
        "source_label": np.concatenate(
            source_label_parts
        ),
        "source_file_stem": np.concatenate(
            source_stem_parts
        ),

        "num_source_files": int(
            len(prepared_parts)
        ),
        "raw_runs_loaded": int(
            sum(
                s["raw_runs_loaded"]
                for s in source_summaries
            )
        ),
        "source_summaries": source_summaries,
    }

    print(
        "\n"
        + "=" * 120
    )
    print(
        f"APPENDED SAME-WAIT DATASET: "
        f"{wait_s:g} s"
    )
    print("=" * 120)
    print(
        f"{combined['num_source_files']} source file(s), "
        f"{len(combined['K'])}/"
        f"{combined['raw_runs_loaded']} "
        f"good/evaluable combined runs"
    )

    for summary in source_summaries:
        print(
            f"  source "
            f"{summary['source_file_ind']}: "
            f"{summary['valid_runs']}/"
            f"{summary['raw_runs_loaded']} valid | "
            f"{summary['file_stem']}"
        )

    return combined


def _prepare_dataset(dataset):
    """
    Prepare one logical physical wait condition.

    Works for:
      * one scalar file_stem;
      * one tuple/list of same-wait stems;
      * a group generated by _select_datasets() from multiple same-wait
        base.DATASETS entries.
    """
    if "source_cfgs" in dataset:
        source_cfgs = list(
            dataset["source_cfgs"]
        )
        group_wait = float(
            dataset.get(
                "dark_wait_s",
                _parse_dark_wait_s(dataset),
            )
        )
        group_label = str(
            dataset.get(
                "label",
                (
                    f"dark_wait_{group_wait:g}s"
                    if np.isfinite(group_wait)
                    else "dataset"
                ),
            )
        )
    else:
        group_wait = _parse_dark_wait_s(
            dataset
        )
        source_cfgs = (
            _expand_dataset_to_single_file_sources(
                dataset,
                wait_s=group_wait,
            )
        )
        group_label = str(
            dataset.get(
                "label",
                (
                    f"dark_wait_{group_wait:g}s"
                    if np.isfinite(group_wait)
                    else "dataset"
                ),
            )
        )

    if len(source_cfgs) == 1:
        part = _prepare_single_dataset(
            source_cfgs[0]
        )

        # Preserve the logical parent label when source expansion generated
        # a suffix or when caller supplied an aggregate label.
        part["label"] = group_label
        return part

    prepared_parts = []

    for source_ind, cfg in enumerate(
        source_cfgs
    ):
        cfg = dict(cfg)
        cfg["_source_file_ind"] = int(
            source_ind
        )

        print(
            "\n"
            + "-" * 120
        )
        print(
            f"PREPARING SAME-WAIT SOURCE "
            f"{source_ind + 1}/"
            f"{len(source_cfgs)}"
        )
        print("-" * 120)
        print(cfg["file_stem"])

        prepared_parts.append(
            _prepare_single_dataset(cfg)
        )

    # All prepared parts should resolve to one wait.
    waits = np.asarray(
        [
            float(part["wait_s"])
            for part in prepared_parts
        ],
        dtype=float,
    )
    wait_ref = float(
        np.median(waits)
    )

    if np.max(
        np.abs(waits - wait_ref)
    ) > 1e-6:
        raise ValueError(
            "Refusing to append files with "
            f"different waits: {waits.tolist()}"
        )

    return _append_prepared_same_wait(
        prepared_parts,
        wait_s=wait_ref,
        label=group_label,
    )




# =============================================================================
# BETA-BINOMIAL BULK MODEL
# =============================================================================


def _bb_logpmf(k, n, p, rho):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    k, n = np.broadcast_arrays(k, n)

    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    rho = float(max(rho, 0.0))

    out = np.full(k.shape, -np.inf, dtype=float)
    valid = (
        np.isfinite(k)
        & np.isfinite(n)
        & (n >= 0)
        & (k >= 0)
        & (k <= n)
    )
    if not np.any(valid):
        return out

    kv = k[valid]
    nv = n[valid]

    if rho <= 1e-8:
        out[valid] = binom.logpmf(kv.astype(int), nv.astype(int), p)
        return out

    concentration = 1.0 / rho - 1.0
    alpha = p * concentration
    beta = (1.0 - p) * concentration

    log_choose = (
        gammaln(nv + 1.0)
        - gammaln(kv + 1.0)
        - gammaln(nv - kv + 1.0)
    )
    out[valid] = (
        log_choose
        + betaln(kv + alpha, nv - kv + beta)
        - betaln(alpha, beta)
    )
    return out


def _bb_mean_var(n, p, rho):
    n = np.asarray(n, dtype=float)
    p = float(p)
    rho = float(max(rho, 0.0))
    mean = n * p
    var = n * p * (1.0 - p) * (1.0 + (n - 1.0) * rho)
    return mean, var


def _fit_beta_binomial(k, n, p0=None, rho0=1e-3):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    valid = (
        np.isfinite(k)
        & np.isfinite(n)
        & (n > 0)
        & (k >= 0)
        & (k <= n)
    )
    k = k[valid]
    n = n[valid]

    if k.size < 20:
        raise RuntimeError("Too few runs for beta-binomial fit.")

    if p0 is None:
        p0 = float(np.sum(k) / np.sum(n))
    p0 = float(np.clip(p0, *BULK_P_BOUNDS))
    rho0 = float(np.clip(rho0, *BULK_RHO_BOUNDS))

    def objective(x):
        p, rho = map(float, x)
        ll = _bb_logpmf(k, n, p, rho)
        if not np.all(np.isfinite(ll)):
            return 1e100
        return -float(np.sum(ll))

    starts = [
        [p0, rho0],
        [p0, 1e-7],
        [p0, 5e-4],
        [p0, 1e-3],
        [p0, 3e-3],
        [p0, 1e-2],
    ]

    best = None
    for start in starts:
        opt = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[BULK_P_BOUNDS, BULK_RHO_BOUNDS],
        )
        if best is None or opt.fun < best.fun:
            best = opt

    if best is None:
        raise RuntimeError("Beta-binomial optimizer returned no result.")

    p, rho = map(float, best.x)
    return {
        "p": p,
        "rho": rho,
        "loglike": -float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
    }


def _fit_central_bulk(k, n):
    """Iteratively estimate the central distribution, excluding high-side tail."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    valid = (
        np.isfinite(k)
        & np.isfinite(n)
        & (n > 0)
        & (k >= 0)
        & (k <= n)
    )
    core = valid.copy()
    history = []

    for iteration in range(BULK_CORE_MAX_ITER):
        fit = _fit_beta_binomial(k[core], n[core])
        mu, var = _bb_mean_var(n, fit["p"], fit["rho"])
        z = np.full(k.shape, np.nan, dtype=float)
        goodz = valid & np.isfinite(var) & (var > 0)
        z[goodz] = (k[goodz] - mu[goodz]) / np.sqrt(var[goodz])

        proposed = valid & np.isfinite(z) & (z <= BULK_CORE_Z_CUT)
        min_core = int(math.ceil(BULK_CORE_MIN_FRACTION * np.sum(valid)))
        if np.sum(proposed) < min_core:
            proposed = core.copy()

        history.append(
            {
                "iteration": iteration,
                "p": fit["p"],
                "rho": fit["rho"],
                "core_runs": int(np.sum(core)),
                "next_core_runs": int(np.sum(proposed)),
            }
        )

        if np.array_equal(proposed, core):
            core = proposed
            break
        core = proposed

    fit = _fit_beta_binomial(k[core], n[core])
    mu, var = _bb_mean_var(n, fit["p"], fit["rho"])
    z = np.full(k.shape, np.nan, dtype=float)
    goodz = valid & np.isfinite(var) & (var > 0)
    z[goodz] = (k[goodz] - mu[goodz]) / np.sqrt(var[goodz])

    frac = _safe_divide(k, n)
    core_frac = frac[core & np.isfinite(frac)]
    p_se_emp = (
        float(np.std(core_frac, ddof=1) / np.sqrt(core_frac.size))
        if core_frac.size > 1
        else np.nan
    )

    return {
        **fit,
        "core_mask": core,
        "z_bb": z,
        "mu_run": mu,
        "var_run": var,
        "num_valid": int(np.sum(valid)),
        "num_core": int(np.sum(core)),
        "core_fraction": float(np.sum(core) / np.sum(valid)),
        "p_se_empirical": p_se_emp,
        "history": history,
    }


def _expected_bb_histogram(n_runs, p, rho, k_grid):
    n_runs = np.asarray(n_runs, dtype=int)
    out = np.zeros(len(k_grid), dtype=float)
    for j, kval in enumerate(k_grid):
        kvals = np.full(n_runs.shape, int(kval), dtype=float)
        out[j] = float(np.sum(np.exp(_bb_logpmf(kvals, n_runs, p, rho))))
    return out


def _bb_mode_for_n(n, p, rho):
    n = int(n)
    grid = np.arange(0, n + 1, dtype=int)
    pmf = np.exp(_bb_logpmf(grid, np.full(grid.shape, n), p, rho))
    return int(grid[np.nanargmax(pmf)])


# =============================================================================
# BULK DARK-KINETIC FIT
# =============================================================================


def _dark_probability(t, p0, gamma):
    t = np.asarray(t, dtype=float)
    return 1.0 - (1.0 - float(p0)) * np.exp(-float(gamma) * t)


def _fit_dark_kinetics(times_s, p_values, p_se=None):
    t = np.asarray(times_s, dtype=float)
    p = np.asarray(p_values, dtype=float)

    if p_se is None:
        sigma = np.ones_like(p)
        absolute_sigma = False
    else:
        sigma = np.asarray(p_se, dtype=float)
        good = np.isfinite(sigma) & (sigma > 0)
        fallback = float(np.nanmedian(sigma[good])) if np.any(good) else 1e-3
        sigma[~good] = fallback
        absolute_sigma = True

    p0_guess = float(np.clip(p[np.argmin(t)], 1e-6, 0.2))
    gamma_guess = 3e-4

    popt, pcov = curve_fit(
        _dark_probability,
        t,
        p,
        p0=[p0_guess, gamma_guess],
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        bounds=([1e-8, 0.0], [0.25, 0.05]),
        maxfev=20000,
    )

    p0, gamma = map(float, popt)
    err = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    p0_se = float(err[0]) if err.size > 0 else np.nan
    gamma_se = float(err[1]) if err.size > 1 else np.nan

    pred = _dark_probability(t, p0, gamma)
    residual = p - pred
    chi2 = float(np.sum((residual / sigma) ** 2)) if p_se is not None else np.nan
    dof = max(0, len(t) - 2)

    tau = 1.0 / gamma if gamma > 0 else np.inf
    tau_se = gamma_se / gamma**2 if gamma > 0 and np.isfinite(gamma_se) else np.nan

    return {
        "p0": p0,
        "p0_se": p0_se,
        "gamma_s_inv": gamma,
        "gamma_se_s_inv": gamma_se,
        "tau_s": tau,
        "tau_se_s": tau_se,
        "chi2": chi2,
        "dof": dof,
        "predicted": pred,
    }


def _trimmed_bulk_fit(ds, trim_fraction):
    frac = np.asarray(ds["loss_fraction"], dtype=float)
    K = np.asarray(ds["K"], dtype=int)
    N = np.asarray(ds["N"], dtype=int)

    valid = np.isfinite(frac) & (N > 0)
    inds = np.where(valid)[0]
    order = inds[np.argsort(frac[inds])]

    keep_n = int(np.floor((1.0 - float(trim_fraction)) * len(order)))
    keep_n = max(20, min(keep_n, len(order)))
    keep = order[:keep_n]

    fit = _fit_beta_binomial(K[keep], N[keep])
    frac_keep = frac[keep]
    p_se = float(np.std(frac_keep, ddof=1) / np.sqrt(len(frac_keep)))

    return {
        "trim_fraction": float(trim_fraction),
        "keep_indices": keep,
        "kept_runs": len(keep),
        "removed_runs": len(order) - len(keep),
        "mean_K": float(np.mean(K[keep])),
        "mean_loss_fraction": float(np.mean(frac_keep)),
        "pooled_p": float(np.sum(K[keep]) / np.sum(N[keep])),
        "p_fit": fit["p"],
        "rho_fit": fit["rho"],
        "p_se_empirical": p_se,
        "loglike": fit["loglike"],
    }


# =============================================================================
# POISSON-LOCAL UPPER-TAIL SCREEN
# =============================================================================


def _poisson_local_tail(K, N, p_bulk):
    K = np.asarray(K, dtype=int)
    N = np.asarray(N, dtype=float)
    lam = N * float(p_bulk)

    # Exact one-sided Poisson upper tail P(X >= K).
    p_tail = poisson.sf(K - 1, lam)
    p_tail = np.clip(p_tail, 1e-300, 1.0 - 1e-16)
    sigma = norm.isf(p_tail)

    return {
        "lambda": lam,
        "p_tail": p_tail,
        "sigma": sigma,
    }


def _expected_poisson_histogram(lam, k_grid):
    lam = np.asarray(lam, dtype=float)
    out = np.zeros(len(k_grid), dtype=float)
    for j, kval in enumerate(k_grid):
        out[j] = float(np.sum(poisson.pmf(int(kval), lam)))
    return out


# =============================================================================
# BINOMIAL TAIL-RATE PHENOMENOLOGY
# =============================================================================


def _binom_loglike(counts, totals, probs):
    counts = np.asarray(counts, dtype=int)
    totals = np.asarray(totals, dtype=int)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(np.sum(binom.logpmf(counts, totals, probs)))


def _q_exposure(T, q_bg, rate_s):
    T = np.asarray(T, dtype=float)
    return 1.0 - (1.0 - float(q_bg)) * np.exp(-float(rate_s) * T)


def _fit_constant_tail(counts, totals):
    q = float(np.sum(counts) / np.sum(totals))
    q = float(np.clip(q, *QBG_BOUNDS))
    ll = _binom_loglike(counts, totals, np.full(len(counts), q))
    return {
        "name": "constant_background",
        "q_bg": q,
        "rate_s_inv": 0.0,
        "loglike": ll,
        "num_params": 1,
    }


def _fit_exposure_tail(counts, totals, T):
    counts = np.asarray(counts, dtype=int)
    totals = np.asarray(totals, dtype=int)
    T = np.asarray(T, dtype=float)

    def objective(x):
        q_bg, rate_s = map(float, x)
        q = _q_exposure(T, q_bg, rate_s)
        return -_binom_loglike(counts, totals, q)

    starts = [
        [max(1e-5, counts[0] / totals[0] * 0.8), 1e-4],
        [max(1e-5, counts[0] / totals[0] * 0.5), 2e-4],
        [max(1e-5, counts[0] / totals[0] * 0.8), 5e-4],
        [1e-4, 1e-3],
    ]

    best = None
    for start in starts:
        opt = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[QBG_BOUNDS, RTAIL_BOUNDS_S],
        )
        if best is None or opt.fun < best.fun:
            best = opt

    q_bg, rate_s = map(float, best.x)
    return {
        "name": "background_plus_exposure",
        "q_bg": q_bg,
        "rate_s_inv": rate_s,
        "loglike": -float(best.fun),
        "num_params": 2,
        "success": bool(best.success),
        "message": str(best.message),
    }


def _fit_cosmic_fixed_tail(counts, totals, T, muon_rate_s):
    counts = np.asarray(counts, dtype=int)
    totals = np.asarray(totals, dtype=int)
    T = np.asarray(T, dtype=float)

    def objective(q_bg):
        q = _q_exposure(T, float(q_bg), float(muon_rate_s))
        return -_binom_loglike(counts, totals, q)

    opt = minimize_scalar(
        objective,
        bounds=QBG_BOUNDS,
        method="bounded",
    )
    q_bg = float(opt.x)
    return {
        "name": "background_plus_fixed_geometric_muon_rate",
        "q_bg": q_bg,
        "rate_s_inv": float(muon_rate_s),
        "loglike": -float(opt.fun),
        "num_params": 1,
        "success": bool(opt.success),
    }


def _pure_cosmic_tail(counts, totals, T, muon_rate_s):
    q = 1.0 - np.exp(-float(muon_rate_s) * np.asarray(T, dtype=float))
    ll = _binom_loglike(counts, totals, q)
    return {
        "name": "pure_geometric_muon",
        "q_bg": 0.0,
        "rate_s_inv": float(muon_rate_s),
        "loglike": ll,
        "num_params": 0,
    }


def _aic_bic(loglike, k, nobs):
    aic = 2.0 * k - 2.0 * loglike
    bic = math.log(float(nobs)) * k - 2.0 * loglike
    return float(aic), float(bic)


def _bootstrap_tail_fit(counts, totals, T, fitted, reps, rng):
    if reps <= 0:
        return []

    qfit = _q_exposure(T, fitted["q_bg"], fitted["rate_s_inv"])
    rows = []

    for b in range(int(reps)):
        sim_counts = rng.binomial(np.asarray(totals, dtype=int), qfit)
        try:
            fit = _fit_exposure_tail(sim_counts, totals, T)
            rows.append(
                {
                    "bootstrap": b,
                    "q_bg": fit["q_bg"],
                    "rate_s_inv": fit["rate_s_inv"],
                }
            )
        except Exception:
            continue
    return rows


# =============================================================================
# COSMIC-MUON GEOMETRY
# =============================================================================


def _muon_geometry():
    L = float(DIAMOND_L_CM)
    W = float(DIAMOND_W_CM)
    t = float(DIAMOND_T_CM)

    face = L * W
    a_eff = 0.75 * L * W + 0.375 * t * (L + W)
    r_uncorr = MUON_FLUX_TOTAL * face
    r_dir = MUON_FLUX_TOTAL * a_eff
    r_dir_se = MUON_FLUX_TOTAL_SE * a_eff

    return {
        "L_cm": L,
        "W_cm": W,
        "t_cm": t,
        "face_area_cm2": face,
        "direction_corrected_area_cm2": a_eff,
        "muon_flux_s_inv_cm2": MUON_FLUX_TOTAL,
        "muon_flux_se_s_inv_cm2": MUON_FLUX_TOTAL_SE,
        "uncorrected_rate_s_inv": r_uncorr,
        "direction_corrected_rate_s_inv": r_dir,
        "direction_corrected_rate_se_s_inv": r_dir_se,
        "direction_corrected_rate_per_day": 86400.0 * r_dir,
    }


# =============================================================================
# DATASET SELECTION
# =============================================================================


def _select_datasets():
    """
    Select ONE logical group per requested wait, retaining ALL physical files.

    This fixes the previous behavior where multiple matches at one wait emitted
    a warning and silently used only the first acquisition.
    """
    source = getattr(
        base,
        "DATASETS",
        None,
    )

    if not source:
        raise RuntimeError(
            "base.DATASETS is empty or unavailable."
        )

    selected = []

    for target in WANTED_WAITS_S:
        source_cfgs = []
        parent_labels = []

        for dataset in source:
            wait = _parse_dark_wait_s(
                dataset
            )

            if not (
                np.isfinite(wait)
                and abs(
                    float(wait)
                    - float(target)
                ) < 1e-8
            ):
                continue

            expanded = (
                _expand_dataset_to_single_file_sources(
                    dataset,
                    wait_s=float(target),
                )
            )

            for cfg in expanded:
                cfg["_source_file_ind"] = int(
                    len(source_cfgs)
                )
                source_cfgs.append(cfg)

            parent_labels.append(
                str(
                    dataset.get(
                        "label",
                        "",
                    )
                )
            )

        if not source_cfgs:
            raise RuntimeError(
                f"Could not find a base.DATASETS "
                f"entry at dark wait "
                f"{target:g} s."
            )

        group_label = next(
            (
                label
                for label in parent_labels
                if label
            ),
            f"dark_wait_{target:g}s",
        )

        print(
            f"[dataset selection] "
            f"wait={target:g} s: "
            f"{len(source_cfgs)} physical "
            f"acquisition file(s)."
        )

        for source_ind, cfg in enumerate(
            source_cfgs
        ):
            print(
                f"    source {source_ind}: "
                f"{cfg['file_stem']}"
            )

        selected.append(
            {
                "label": group_label,
                "dark_wait_s": float(
                    target
                ),
                "source_cfgs": source_cfgs,

                # Also expose stems in a conventional field so downstream
                # helpers can parse this logical group if needed.
                "file_stems": [
                    cfg["file_stem"]
                    for cfg in source_cfgs
                ],
            }
        )

    return selected




# =============================================================================
# BULK ANALYSIS OUTPUTS
# =============================================================================


def _analyze_bulk(datasets):
    bulk_rows = []
    trim_rows = []

    for ds in datasets:
        ds["bulk"] = _fit_central_bulk(ds["K"], ds["N"])
        b = ds["bulk"]

        n_ref = int(round(np.median(ds["N"])))
        fitted_mode = _bb_mode_for_n(n_ref, b["p"], b["rho"])
        obs_hist = np.bincount(ds["K"].astype(int))
        observed_mode = int(np.argmax(obs_hist))

        bulk_rows.append(
            {
                "dataset": ds["label"],
                "dark_wait_s": ds["wait_s"],
                "actual_dark_wait_s": ds["actual_wait_s"],
                "particle_exposure_s": ds["particle_exposure_s"],
                "num_source_files": int(ds.get("num_source_files", 1)),
                "raw_runs_loaded": int(
                    ds.get("raw_runs_loaded", len(ds["K"]))
                ),
                "good_runs": len(ds["K"]),
                "mean_K_all": float(np.mean(ds["K"])),
                "var_K_all": float(np.var(ds["K"], ddof=1)),
                "fano_K_all": float(np.var(ds["K"], ddof=1) / np.mean(ds["K"])),
                "mean_N": float(np.mean(ds["N"])),
                "pooled_p_all": float(np.sum(ds["K"]) / np.sum(ds["N"])),
                "central_p": b["p"],
                "central_p_se_empirical": b["p_se_empirical"],
                "central_rho": b["rho"],
                "central_core_runs": b["num_core"],
                "central_core_fraction": b["core_fraction"],
                "observed_mode_K": observed_mode,
                "reference_N_for_fitted_mode": n_ref,
                "fitted_beta_binomial_mode_K": fitted_mode,
            }
        )

        for trim in TRIM_FRACTIONS:
            tr = _trimmed_bulk_fit(ds, trim)
            trim_rows.append(
                {
                    "dataset": ds["label"],
                    "dark_wait_s": ds["wait_s"],
                    **{k: v for k, v in tr.items() if k != "keep_indices"},
                }
            )

        # Per-dataset peak-fit figure.
        kmax = int(max(np.max(ds["K"]) + 3, np.percentile(ds["K"], 99.9) + 5))
        grid = np.arange(0, kmax + 1, dtype=int)
        observed = np.bincount(ds["K"].astype(int), minlength=kmax + 1)[: kmax + 1]
        expected = _expected_bb_histogram(ds["N"], b["p"], b["rho"], grid)

        fig, ax = plt.subplots(figsize=(8.6, 6.2))
        ax.plot(grid, observed, marker="o", linestyle="none", markersize=4, label="observed raw loss counts")
        ax.plot(grid, expected, linewidth=1.7, label="central beta-binomial model")
        ax.axvline(np.mean(ds["K"]), linestyle="--", linewidth=1.0, label="observed mean")
        ax.axvline(fitted_mode, linestyle=":", linewidth=1.2, label="fitted central mode")
        ax.set_xlabel("NV- -> NV0 losses per run, K")
        ax.set_ylabel("Runs / count bin")
        ax.set_title(f"{ds['label']}: bulk/peak distribution")
        ax.legend(fontsize=8)
        fig.tight_layout()
        if SAVE_OUTPUTS:
            fig.savefig(
                OUTPUT_DIR / f"bulk_peak_fit_{ds['wait_s']:g}s.png",
                dpi=180,
                bbox_inches="tight",
            )

    _write_csv(OUTPUT_DIR / "raw_bulk_summary.csv", bulk_rows)
    _write_csv(OUTPUT_DIR / "bulk_trim_sensitivity.csv", trim_rows)

    # Fit kinetics to the central beta-binomial probabilities.
    waits = np.asarray([d["actual_wait_s"] for d in datasets], dtype=float)
    p = np.asarray([d["bulk"]["p"] for d in datasets], dtype=float)
    pse = np.asarray([d["bulk"]["p_se_empirical"] for d in datasets], dtype=float)
    kinetic = _fit_dark_kinetics(waits, p, pse)

    # Also refit the dark kinetic parameters after each explicit upper-tail trim.
    kinetic_trim_rows = []
    for trim in TRIM_FRACTIONS:
        rows = sorted(
            [r for r in trim_rows if abs(r["trim_fraction"] - trim) < 1e-12],
            key=lambda r: r["dark_wait_s"],
        )
        pp = np.asarray([r["p_fit"] for r in rows], dtype=float)
        ss = np.asarray([r["p_se_empirical"] for r in rows], dtype=float)
        fit = _fit_dark_kinetics(waits, pp, ss)
        kinetic_trim_rows.append(
            {
                "trim_fraction": trim,
                "trim_percent": 100.0 * trim,
                "p0": fit["p0"],
                "p0_se": fit["p0_se"],
                "gamma_dark_s_inv": fit["gamma_s_inv"],
                "gamma_dark_se_s_inv": fit["gamma_se_s_inv"],
                "tau_dark_s": fit["tau_s"],
                "tau_dark_min": fit["tau_s"] / 60.0,
                "tau_dark_se_s": fit["tau_se_s"],
            }
        )

    _write_csv(OUTPUT_DIR / "bulk_kinetic_fit_by_trim.csv", kinetic_trim_rows)

    # p_bulk versus dark wait.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.errorbar(
        waits,
        100.0 * p,
        yerr=100.0 * pse,
        marker="o",
        linestyle="none",
        capsize=3,
        label="central beta-binomial p",
    )
    tgrid = np.linspace(0.0, max(waits) * 1.05 + 1e-9, 300)
    ax.plot(
        tgrid,
        100.0 * _dark_probability(tgrid, kinetic["p0"], kinetic["gamma_s_inv"]),
        linewidth=1.7,
        label="effective dark-kinetic fit",
    )
    ax.set_xlabel("Actual dark wait (s)")
    ax.set_ylabel("Central NV- -> NV0 loss probability (%)")
    ax.set_title("Bulk charge-loss probability versus dark wait")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR / "bulk_probability_vs_dark_wait.png",
            dpi=180,
            bbox_inches="tight",
        )

    # Gamma versus removed upper-tail fraction.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    trims_pct = np.asarray([r["trim_percent"] for r in kinetic_trim_rows], dtype=float)
    gammas = np.asarray([r["gamma_dark_s_inv"] for r in kinetic_trim_rows], dtype=float)
    gamma_se = np.asarray([r["gamma_dark_se_s_inv"] for r in kinetic_trim_rows], dtype=float)
    ax.errorbar(
        trims_pct,
        1e4 * gammas,
        yerr=1e4 * gamma_se,
        marker="o",
        capsize=3,
    )
    ax.set_xlabel("Upper-tail runs removed (%)")
    ax.set_ylabel(r"Effective $\Gamma_{\rm dark}$ ($10^{-4}$ s$^{-1}$)")
    ax.set_title("Does the inferred dark rate depend on extreme runs?")
    fig.tight_layout()
    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR / "bulk_gamma_vs_tail_trim.png",
            dpi=180,
            bbox_inches="tight",
        )

    return bulk_rows, trim_rows, kinetic, kinetic_trim_rows


# =============================================================================
# UPPER-TAIL ANALYSIS OUTPUTS
# =============================================================================


def _analyze_upper_tail(datasets, muon):
    event_summary_rows = []
    event_rows = []
    run_rows = []

    # Compute per-run local Poisson screening coordinates from the independently
    # fitted central bulk probability at each wait.
    for ds in datasets:
        tail = _poisson_local_tail(ds["K"], ds["N"], ds["bulk"]["p"])
        ds["tail"] = tail

        for i in range(len(ds["K"])):
            run_rows.append(
                {
                    "dataset": ds["label"],
                    "dark_wait_s": ds["wait_s"],
                    "original_run": int(ds["original_run"][i]),
                    **_source_identity_for_run(ds, i),
                    "K_loss": int(ds["K"][i]),
                    "N_evaluable": int(ds["N"][i]),
                    "loss_fraction": float(ds["loss_fraction"][i]),
                    "central_p": ds["bulk"]["p"],
                    "poisson_lambda": float(tail["lambda"][i]),
                    "poisson_tail_p": float(tail["p_tail"][i]),
                    "poisson_sigma": float(tail["sigma"][i]),
                }
            )

        print("\n" + "-" * 115)
        print(f"POISSON-LOCAL UPPER TAIL: {ds['label']}")
        print("-" * 115)
        print(
            f"central p={100*ds['bulk']['p']:.5f}%  |  "
            f"N runs={len(ds['K'])}  |  exposure={ds['particle_exposure_s']:.6f} s"
        )

        for zcut in SIGMA_THRESHOLDS:
            mask = np.isfinite(tail["sigma"]) & (tail["sigma"] >= zcut)
            count = int(np.sum(mask))
            den = len(ds["K"])
            frac = count / den
            event_summary_rows.append(
                {
                    "dataset": ds["label"],
                    "dark_wait_s": ds["wait_s"],
                    "particle_exposure_s": ds["particle_exposure_s"],
                    "sigma_threshold": zcut,
                    "events": count,
                    "runs": den,
                    "event_fraction": frac,
                    "event_percent": 100.0 * frac,
                    "gaussian_one_sided_fraction": float(norm.sf(zcut)),
                }
            )
            print(
                f">={zcut:.0f} sigma: {count}/{den} = {100*frac:.5f}% "
                f"(~1 in {den/count:.1f})" if count else
                f">={zcut:.0f} sigma: 0/{den}"
            )

            for i in np.where(mask)[0]:
                event_rows.append(
                    {
                        "dataset": ds["label"],
                        "dark_wait_s": ds["wait_s"],
                        "sigma_threshold": zcut,
                        "original_run": int(ds["original_run"][i]),
                        **_source_identity_for_run(ds, i),
                        "K_loss": int(ds["K"][i]),
                        "N_evaluable": int(ds["N"][i]),
                        "loss_fraction": float(ds["loss_fraction"][i]),
                        "poisson_lambda": float(tail["lambda"][i]),
                        "poisson_tail_p": float(tail["p_tail"][i]),
                        "poisson_sigma": float(tail["sigma"][i]),
                    }
                )

        # Print strongest runs independent of a chosen threshold.
        order = np.argsort(tail["sigma"])[::-1]
        print("\nTop upper-tail runs:")
        print(
            "rank  globalR  src  localR    K    N    loss%   "
            "PoisSigma     p_tail"
        )
        shown = 0

        for i in order:
            if not np.isfinite(
                tail["sigma"][i]
            ):
                continue

            shown += 1
            identity = (
                _source_identity_for_run(
                    ds,
                    i,
                )
            )

            print(
                f"{shown:4d}  "
                f"{int(ds['original_run'][i]):7d}  "
                f"{identity['source_file_ind']:3d}  "
                f"{identity['source_local_run']:6d}  "
                f"{int(ds['K'][i]):4d} "
                f"{int(ds['N'][i]):4d}  "
                f"{100*ds['loss_fraction'][i]:7.3f}  "
                f"{tail['sigma'][i]:9.3f}  "
                f"{tail['p_tail'][i]:.3e}"
            )

            if shown >= TOP_TAIL_RUNS_TO_PRINT:
                break

        # Histogram with Poisson-local central reference on a log scale.
        kmax = int(max(np.max(ds["K"]) + 3, np.percentile(ds["K"], 99.9) + 8))
        grid = np.arange(0, kmax + 1, dtype=int)
        obs = np.bincount(ds["K"].astype(int), minlength=kmax + 1)[: kmax + 1]
        expected = _expected_poisson_histogram(tail["lambda"], grid)

        fig, ax = plt.subplots(figsize=(8.8, 6.4))
        ax.plot(grid, obs, marker="o", linestyle="none", markersize=4, label="observed")
        ax.plot(grid, expected, linewidth=1.6, label="local Poisson reference")
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(1e-4, 0.2 / len(ds["K"])))

        # Show approximate K locations where the median-N run first crosses each z.
        nref = int(round(np.median(ds["N"])))
        lamref = nref * ds["bulk"]["p"]
        for zcut in SIGMA_THRESHOLDS:
            target_p = norm.sf(zcut)
            kval = 0
            while poisson.sf(kval - 1, lamref) > target_p and kval < nref:
                kval += 1
            ax.axvline(kval, linestyle="--", linewidth=0.9, label=f"~{zcut:.0f} sigma at median N")

        ax.set_xlabel("NV- -> NV0 losses per run, K")
        ax.set_ylabel("Runs / count bin (log scale)")
        ax.set_title(f"{ds['label']}: raw upper-tail screening")
        ax.legend(fontsize=7)
        fig.tight_layout()
        if SAVE_OUTPUTS:
            fig.savefig(
                OUTPUT_DIR / f"tail_poisson_sigma_{ds['wait_s']:g}s.png",
                dpi=180,
                bbox_inches="tight",
            )

    _write_csv(OUTPUT_DIR / "tail_event_summary.csv", event_summary_rows)
    _write_csv(OUTPUT_DIR / "tail_events.csv", event_rows)
    _write_csv(OUTPUT_DIR / "run_level_metrics.csv", run_rows)

    # ------------------------------------------------------------------
    # Fit exposure-dependent tail rates separately for 3, 4, 5 sigma.
    # ------------------------------------------------------------------
    rate_rows = []
    rng = np.random.default_rng(TAIL_BOOTSTRAP_SEED)

    T = np.asarray([d["particle_exposure_s"] for d in datasets], dtype=float)
    n_runs = np.asarray([len(d["K"]) for d in datasets], dtype=int)
    wait_order = np.asarray([d["wait_s"] for d in datasets], dtype=float)
    Rmu = float(muon["direction_corrected_rate_s_inv"])

    fitted_by_sigma = {}

    for zcut in SIGMA_THRESHOLDS:
        counts = np.asarray(
            [
                next(
                    r["events"]
                    for r in event_summary_rows
                    if r["dark_wait_s"] == d["wait_s"]
                    and r["sigma_threshold"] == zcut
                )
                for d in datasets
            ],
            dtype=int,
        )

        constant = _fit_constant_tail(counts, n_runs)
        exposure = _fit_exposure_tail(counts, n_runs, T)
        cosmic_fixed = _fit_cosmic_fixed_tail(counts, n_runs, T, Rmu)
        pure_cosmic = _pure_cosmic_tail(counts, n_runs, T, Rmu)

        models = [constant, exposure, cosmic_fixed, pure_cosmic]
        nobs = int(np.sum(n_runs))
        for model in models:
            aic, bic = _aic_bic(model["loglike"], model["num_params"], nobs)
            model["AIC"] = aic
            model["BIC"] = bic
        min_aic = min(m["AIC"] for m in models)
        min_bic = min(m["BIC"] for m in models)
        for model in models:
            model["delta_AIC"] = model["AIC"] - min_aic
            model["delta_BIC"] = model["BIC"] - min_bic

        # Bootstrap the phenomenological exposure model.
        boot = _bootstrap_tail_fit(
            counts,
            n_runs,
            T,
            exposure,
            TAIL_BOOTSTRAP_REPS,
            rng,
        )
        if boot:
            qb = np.asarray([r["q_bg"] for r in boot], dtype=float)
            rb = np.asarray([r["rate_s_inv"] for r in boot], dtype=float)
            qlo, qmed, qhi = np.percentile(qb, [2.5, 50.0, 97.5])
            rlo, rmed, rhi = np.percentile(rb, [2.5, 50.0, 97.5])
        else:
            qlo = qmed = qhi = np.nan
            rlo = rmed = rhi = np.nan

        eps = exposure["rate_s_inv"] / Rmu if Rmu > 0 else np.nan
        eps_lo = rlo / Rmu if Rmu > 0 and np.isfinite(rlo) else np.nan
        eps_hi = rhi / Rmu if Rmu > 0 and np.isfinite(rhi) else np.nan

        fitted_by_sigma[zcut] = {
            "counts": counts,
            "constant": constant,
            "exposure": exposure,
            "cosmic_fixed": cosmic_fixed,
            "pure_cosmic": pure_cosmic,
            "bootstrap": boot,
            "epsilon_eff": eps,
            "epsilon_eff_lo": eps_lo,
            "epsilon_eff_hi": eps_hi,
        }

        for model in models:
            rate_rows.append(
                {
                    "sigma_threshold": zcut,
                    "model": model["name"],
                    "q_bg": model.get("q_bg", np.nan),
                    "rate_s_inv": model.get("rate_s_inv", np.nan),
                    "rate_per_day": 86400.0 * model.get("rate_s_inv", np.nan),
                    "rate_over_direction_corrected_muon_rate": (
                        model.get("rate_s_inv", np.nan) / Rmu
                        if Rmu > 0
                        else np.nan
                    ),
                    "loglike": model["loglike"],
                    "AIC": model["AIC"],
                    "delta_AIC": model["delta_AIC"],
                    "BIC": model["BIC"],
                    "delta_BIC": model["delta_BIC"],
                    "bootstrap_q_bg_2p5": qlo if model["name"] == "background_plus_exposure" else np.nan,
                    "bootstrap_q_bg_50": qmed if model["name"] == "background_plus_exposure" else np.nan,
                    "bootstrap_q_bg_97p5": qhi if model["name"] == "background_plus_exposure" else np.nan,
                    "bootstrap_rate_2p5_s_inv": rlo if model["name"] == "background_plus_exposure" else np.nan,
                    "bootstrap_rate_50_s_inv": rmed if model["name"] == "background_plus_exposure" else np.nan,
                    "bootstrap_rate_97p5_s_inv": rhi if model["name"] == "background_plus_exposure" else np.nan,
                    "epsilon_eff_Rtail_over_Rmu": eps if model["name"] == "background_plus_exposure" else np.nan,
                    "epsilon_eff_2p5": eps_lo if model["name"] == "background_plus_exposure" else np.nan,
                    "epsilon_eff_97p5": eps_hi if model["name"] == "background_plus_exposure" else np.nan,
                }
            )

        print("\n" + "=" * 115)
        print(f"TAIL-RATE PHENOMENOLOGY: >= {zcut:.0f} SIGMA")
        print("=" * 115)
        print("wait(s)   exposure(s)   events/runs    observed%")
        for w, tt, m, n in zip(wait_order, T, counts, n_runs):
            print(f"{w:7.1f}   {tt:11.6f}   {m:4d}/{n:<5d}      {100*m/n:8.5f}%")
        print(
            f"free exposure model: q_bg={100*exposure['q_bg']:.5f}%  "
            f"R_tail={exposure['rate_s_inv']:.6e} s^-1 "
            f"({86400*exposure['rate_s_inv']:.3f}/day)"
        )
        print(
            f"R_tail / R_mu(direction-corrected) = {eps:.3f}"
        )
        if np.isfinite(rlo):
            print(
                f"bootstrap 95% R_tail interval = [{rlo:.6e}, {rhi:.6e}] s^-1; "
                f"epsilon_eff interval = [{eps_lo:.3f}, {eps_hi:.3f}]"
            )
        print("model comparison (smaller AIC/BIC is preferred):")
        for model in sorted(models, key=lambda m: m["AIC"]):
            print(
                f"  {model['name']:<42s} "
                f"AIC={model['AIC']:.2f} dAIC={model['delta_AIC']:.2f} "
                f"BIC={model['BIC']:.2f}"
            )

        # Rate-vs-exposure figure for this threshold.
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        obs = counts / n_runs
        obs_se = np.sqrt(obs * (1.0 - obs) / n_runs)
        ax.errorbar(
            T,
            100.0 * obs,
            yerr=100.0 * obs_se,
            marker="o",
            linestyle="none",
            capsize=3,
            label="observed event fraction",
        )

        tgrid = np.linspace(0.0, max(T) * 1.05, 400)
        ax.plot(
            tgrid,
            100.0 * _q_exposure(tgrid, exposure["q_bg"], exposure["rate_s_inv"]),
            linewidth=1.7,
            label="background + fitted exposure rate",
        )
        ax.plot(
            tgrid,
            100.0 * _q_exposure(tgrid, cosmic_fixed["q_bg"], Rmu),
            linewidth=1.4,
            linestyle="--",
            label="background + fixed geometric muon rate",
        )
        ax.plot(
            tgrid,
            100.0 * (1.0 - np.exp(-Rmu * tgrid)),
            linewidth=1.2,
            linestyle=":",
            label="pure geometric muon crossing probability",
        )
        ax.set_xlabel("Estimated rep11-to-rep12 particle exposure (s)")
        ax.set_ylabel(f"Runs with >= {zcut:.0f} sigma event (%)")
        ax.set_title(f">= {zcut:.0f} sigma upper-tail rate versus exposure")
        ax.legend(fontsize=8)
        fig.tight_layout()
        if SAVE_OUTPUTS:
            fig.savefig(
                OUTPUT_DIR / f"tail_rate_fit_{zcut:.0f}sigma.png",
                dpi=180,
                bbox_inches="tight",
            )

    _write_csv(OUTPUT_DIR / "tail_rate_fit_summary.csv", rate_rows)

    # Summary of inferred exposure-rate scale versus sigma threshold.
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    zcuts = np.asarray(SIGMA_THRESHOLDS, dtype=float)
    eps = np.asarray([fitted_by_sigma[z]["epsilon_eff"] for z in zcuts], dtype=float)
    epslo = np.asarray([fitted_by_sigma[z]["epsilon_eff_lo"] for z in zcuts], dtype=float)
    epshi = np.asarray([fitted_by_sigma[z]["epsilon_eff_hi"] for z in zcuts], dtype=float)
    yerr = np.vstack([eps - epslo, epshi - eps])
    yerr[~np.isfinite(yerr)] = 0.0

    ax.errorbar(zcuts, eps, yerr=yerr, marker="o", capsize=3)
    ax.axhline(1.0, linestyle="--", linewidth=1.1, label="R_tail = geometric R_mu")
    ax.set_xlabel("Poisson-local screening threshold (sigma)")
    ax.set_ylabel(r"Effective rate ratio $R_{tail}/R_{\mu,geom}$")
    ax.set_title("Exposure-dependent rare-event rate scale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR / "tail_rate_scale_vs_sigma.png",
            dpi=180,
            bbox_inches="tight",
        )

    return event_summary_rows, rate_rows, fitted_by_sigma


# =============================================================================
# HUMAN-READABLE SUMMARY
# =============================================================================


def _build_summary(datasets, kinetic, kinetic_trim_rows, fitted_by_sigma, muon):
    lines = []
    lines.append("RAW BULK + UPPER-TAIL PHENOMENOLOGICAL ANALYSIS")
    lines.append("=" * 84)
    lines.append("No truth/misclassification filtering and no spatial analysis are used.")
    lines.append("")

    lines.append("1. BULK / PEAK SHIFT")
    lines.append("-" * 84)
    for ds in datasets:
        b = ds["bulk"]
        lines.append(
            f"{ds['wait_s']:g} s: central p={100*b['p']:.5f}%  "
            f"rho={b['rho']:.6g}  core={b['num_core']}/{b['num_valid']} "
            f"({100*b['core_fraction']:.2f}%)"
        )
    lines.append(
        f"Effective dark kinetics: Gamma_dark={kinetic['gamma_s_inv']:.6e} +/- "
        f"{kinetic['gamma_se_s_inv']:.3e} s^-1; "
        f"tau_dark={kinetic['tau_s']:.1f} s = {kinetic['tau_s']/60.0:.2f} min."
    )
    lines.append(
        "Interpretation: p_bulk describes the ordinary dark charge-loss probability; "
        "rho describes ordinary bulk overdispersion/common-mode heterogeneity."
    )
    lines.append("")

    lines.append("Tail-removal stability of the inferred dark rate:")
    for row in kinetic_trim_rows:
        lines.append(
            f"  remove top {row['trim_percent']:g}%: "
            f"Gamma={row['gamma_dark_s_inv']:.6e} s^-1, "
            f"tau={row['tau_dark_min']:.2f} min"
        )
    lines.append(
        "If Gamma_dark is stable as the top 1--5% of runs are removed, the peak/mean "
        "shift is a bulk diamond charge-relaxation effect rather than a consequence "
        "of the extreme tail."
    )
    lines.append("")

    lines.append("2. EXTREME UPPER TAIL")
    lines.append("-" * 84)
    Rmu = muon["direction_corrected_rate_s_inv"]
    lines.append(
        f"Direction-corrected geometric muon rate: R_mu={Rmu:.6e} s^-1 = "
        f"{86400*Rmu:.3f}/day; A_eff={muon['direction_corrected_area_cm2']:.7f} cm^2."
    )

    for zcut in SIGMA_THRESHOLDS:
        f = fitted_by_sigma[zcut]
        expfit = f["exposure"]
        lines.append("")
        lines.append(f">= {zcut:.0f} sigma:")
        lines.append(
            f"  q_bg={100*expfit['q_bg']:.5f}%"
        )
        lines.append(
            f"  R_tail={expfit['rate_s_inv']:.6e} s^-1 = "
            f"{86400*expfit['rate_s_inv']:.3f}/day"
        )
        lines.append(
            f"  effective rate ratio R_tail/R_mu={f['epsilon_eff']:.3f}"
        )
        lines.append(
            f"  DeltaAIC exposure-vs-best = {expfit['delta_AIC']:.2f}; "
            f"DeltaAIC fixed-muon-vs-best = {f['cosmic_fixed']['delta_AIC']:.2f}"
        )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 84)
    lines.append(
        "The bulk fit and the tail-rate fit are intentionally separate.  The bulk "
        "parameters characterize ordinary charge stability of the diamond.  The "
        "tail fit characterizes the frequency of unusually large same-run loss "
        "coincidences."
    )
    lines.append(
        "R_tail/R_mu near unity means only that the exposure-dependent rare-event "
        "rate is on the same geometric scale as direct cosmic-muon crossings.  It "
        "does not establish a cosmic origin or a microscopic detection efficiency."
    )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def main():
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 120)
    print("RAW BULK / PEAK + EXTREME UPPER-TAIL PHENOMENOLOGY")
    print("#" * 120)
    print("No truth filtering. No image pixels. No spatial statistics.")

    selected = _select_datasets()
    datasets = [
        _prepare_dataset(ds)
        for ds in selected
    ]
    datasets = sorted(
        datasets,
        key=lambda d: d["wait_s"],
    )

    print("\n" + "=" * 120)
    print("SAME-WAIT SOURCE FILE SUMMARY")
    print("=" * 120)

    for ds in datasets:
        print(
            f"{ds['wait_s']:g} s: "
            f"{int(ds.get('num_source_files', 1))} "
            f"source file(s), "
            f"{len(ds['K'])}/"
            f"{int(ds.get('raw_runs_loaded', len(ds['K'])))} "
            f"good/evaluable combined runs"
        )

    muon = _muon_geometry()
    _write_csv(OUTPUT_DIR / "muon_geometry_summary.csv", [muon])

    print("\n" + "=" * 120)
    print("DIRECTION-CORRECTED COSMIC-MUON GEOMETRY")
    print("=" * 120)
    print(f"face area             = {muon['face_area_cm2']:.7f} cm^2")
    print(f"effective area        = {muon['direction_corrected_area_cm2']:.7f} cm^2")
    print(f"uncorrected rate      = {muon['uncorrected_rate_s_inv']:.6e} s^-1")
    print(f"direction-corrected R = {muon['direction_corrected_rate_s_inv']:.6e} s^-1")
    print(f"                       = {muon['direction_corrected_rate_per_day']:.3f}/day")

    # 1) Bulk / peak analysis.
    bulk_rows, trim_rows, kinetic, kinetic_trim_rows = _analyze_bulk(datasets)

    print("\n" + "=" * 120)
    print("BULK / PEAK SUMMARY")
    print("=" * 120)
    for row in bulk_rows:
        print(
            f"{row['dark_wait_s']:5.1f} s: central p={100*row['central_p']:.5f}%  "
            f"rho={row['central_rho']:.6g}  "
            f"obs mode K={row['observed_mode_K']}  "
            f"fit mode K~{row['fitted_beta_binomial_mode_K']}"
        )
    print(
        f"Gamma_dark = {kinetic['gamma_s_inv']:.6e} +/- "
        f"{kinetic['gamma_se_s_inv']:.3e} s^-1"
    )
    print(
        f"tau_dark = {kinetic['tau_s']:.1f} s = {kinetic['tau_s']/60.0:.2f} min"
    )

    # 2) Upper-tail / sigma analysis and exposure-rate fit.
    event_summary, rate_rows, fitted_by_sigma = _analyze_upper_tail(datasets, muon)

    summary = _build_summary(
        datasets,
        kinetic,
        kinetic_trim_rows,
        fitted_by_sigma,
        muon,
    )

    print("\n")
    print(summary)

    if SAVE_OUTPUTS:
        with (OUTPUT_DIR / "analysis_summary.txt").open("w", encoding="utf-8") as f:
            f.write(summary)
            f.write("\n")

    print("\n" + "=" * 120)
    print("OUTPUT DIRECTORY")
    print("=" * 120)
    print(OUTPUT_DIR.resolve())

    if SHOW_FIGURES:
        plt.show(block=True)
    else:
        plt.close("all")

    return {
        "datasets": datasets,
        "muon": muon,
        "bulk_rows": bulk_rows,
        "trim_rows": trim_rows,
        "kinetic": kinetic,
        "kinetic_trim_rows": kinetic_trim_rows,
        "tail_event_summary": event_summary,
        "tail_rate_rows": rate_rows,
        "tail_fits": fitted_by_sigma,
    }


if __name__ == "__main__":
    from utils import kplotlib as kpl
    kpl.init_kplotlib()
    analysis = main()
    kpl.show(block=SHOW_FIGURES)

