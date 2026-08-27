# -*- coding: utf-8 -*-
"""
Counts-only V5 focused directionality diagnostic: V4 probabilistic analysis + high-confidence discrete-event test.

Purpose
-------
Separate two different failure modes in the particle-memory data:

1) apparent NV0 -> NV- "gains" caused by a dim/misclassified rep11;
2) apparent NV- -> NV0 "losses" caused by a dim/misclassified rep12.

The key idea is to ask, for every raw threshold crossing, whether the same
transition survives an independently estimated run-wise common-mode correction
and a stricter two-sided classification margin.

This script DOES NOT load img_arrays.
It also tolerates temporary metadata/nv_list lookup failures: non-spatial
truth/tail/Fano diagnostics continue, while spatial analysis is skipped only
when camera coordinates truly cannot be recovered.

It reuses the safe counts-only loader / quality rejection from
    sc_charge_state_particle_memory_spatial_model.py

Outputs
-------
For each dataset:
  * run-level CSV with raw, robust, likely-misclassified and ambiguous counts;
  * event-level CSV (one row per raw gain/loss transition);
  * top-run tables printed to terminal;
  * figures showing raw vs robust gains/losses and common-mode sensitivity;
  * robust loss-vs-gain tail/Fano comparison;
  * exact Poisson-binomial tails for top robust candidates;
  * conservative exact-Poisson-binomial + Benjamini-Hochberg FDR audit;
  * per-NV state-history diagnostic for residual rep11 gain misclassification;
  * state-noise-normalized threshold-depth diagnostic;
  * local rep11/rep12 brightness diagnostics around each robust transition;
  * per-run and per-event audit CSVs for the cleanest candidates;
  * optional weighted exact same-K spatial null on robust loss and gain masks.

Interpretation
--------------
Apparent NV0 -> NV-:
    likely rep11 misclassification
        raw:       NV0(rep11) -> NV-(rep12)
        corrected: NV-(rep11) -> NV-(rep12)

Apparent NV- -> NV0:
    likely rep12 misclassification
        raw:       NV-(rep11) -> NV0(rep12)
        corrected: NV-(rep11) -> NV-(rep12)

A "robust" transition must survive:
    * a strict raw two-sided margin,
    * cross-fitted additive correction,
    * cross-fitted multiplicative correction,
all with the same strict two-sided margin.

The common-mode correction is CROSS-FITTED across NVs: for a target NV in
fold f, the run-wise correction is estimated only from stable reference NVs
outside fold f.  Thus the target NV cannot influence its own correction.

V4 adds a cross-fitted probabilistic charge-state model. Hard threshold
assignments are converted to posterior state probabilities, and rare-event
significance is re-evaluated from continuous soft transition scores using an
independent-across-NV run-scramble null. This directly tests whether shallow
threshold crossings (especially apparent NV0 -> NV- gains) retain significance.

V5 is intentionally focused rather than adding another large stack of tests.
It answers the practical directionality question: when we use NV- -> NV0 as
the event channel, do those transitions remain discrete and high-confidence,
while reverse NV0 -> NV- spikes are much more sensitive to the dim rep11
initial-state classification?  V5 therefore adds only three diagnostics:
  (i) raw -> robust -> posterior-confident survival funnels;
  (ii) initial/final state-confidence asymmetry for each direction;
  (iii) top-spike composition and a high-confidence discrete-event run trace.
The reverse channel is treated as an internal control, not assumed to be
strictly impossible physically.

This is a diagnostic, not a proof of microscopic mechanism. A transition that
survives these controls is a much stronger candidate for a real charge change;
a transition that disappears is measurement-sensitive.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, norm

import sc_charge_state_particle_memory_spatial_model as base
from utils import kplotlib as kpl
from utils import data_manager as dm


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# None -> automatically use DATASETS from the main particle-memory script.
DATASETS = None

# Saved charge-state reps.
REP_INITIAL = 11
REP_FINAL = 12

# A transition is called "strong" only when BOTH sides are at least this far
# from threshold, in integrated-count units.
STRICT_MARGIN_COUNTS = 3.0

# Also report a margin sweep for global robustness.
MARGIN_SWEEP_COUNTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

# Stable NVs used to estimate common-mode count fluctuations.
REFERENCE_OCCUPANCY = 0.90
REFERENCE_BASELINE_MARGIN_COUNTS = 3.0
MIN_REFERENCE_NVS = 80

# Cross-fitting.  For a target NV in fold f, use stable references NOT in f.
# 5 folds leaves ~80% of the stable references for each correction estimate.
NUM_CROSSFIT_FOLDS = 5

# Strong evidence that a raw apparent gain is a rep11 misclassification:
# the same NV is normally NV- in >= this fraction of OTHER good runs.
GAIN_STABLE_NVM_OCCUPANCY = 0.95

# For labels only.  A low scale means the corresponding rep was globally dim.
BRIGHTNESS_Z_CUT = -2.0

TOP_RUNS_TO_PRINT = 20
SAVE_OUTPUTS = True
OUTPUT_DIR = Path("analysis_output") / "transition_truth_diagnostic"

# Save every raw transition to a detailed CSV.  Usually ~1e5 rows, which is
# manageable and very useful for auditing individual NVs/runs.
SAVE_EVENT_LEVEL_CSV = True

# Same severe global-collapse rejection as the main analysis.
REJECT_GLOBAL_DROP_RUNS = True


# =============================================================================
# NEXT-ANALYSIS CONFIGURATION: WHICH DIRECTION CARRIES REAL RARE EVENTS?
# =============================================================================

# These analyses use only the truth-tested consensus transition masks.
RUN_ROBUST_TAIL_ANALYSIS = True
ROBUST_TAIL_Z_THRESHOLDS = (3.0, 4.0, 5.0)

# Per-NV background probabilities are estimated from central runs.  An initial
# heterogeneous-null pass is used to exclude obvious tail runs before the final
# p_i values are estimated.
TAIL_BACKGROUND_MAX_ABS_Z = 3.0
TAIL_BACKGROUND_MIN_FRACTION = 0.50
TOP_EXACT_POIBIN_RUNS = 20

# Exact weighted same-K spatial null on the ROBUST masks.  This is deliberately
# lower than a publication run so this diagnostic does not take days.  If the
# robust-loss signal survives, change this to 250--500 for the final result.
# The 50-null spatial diagnostic has already been useful.  The next priority is
# history/local-brightness/FDR discrimination, so spatial reruns are OFF by
# default.  Set True after the new diagnostics identify clean candidates.
RUN_ROBUST_V23_SPATIAL = False
ROBUST_V23_NULL_DATASETS = 50
ROBUST_V23_RANDOM_SEED = 260826

# Run the spatial null for both directions.  The gain channel is the internal
# negative/control channel for the event-directionality claim.
ROBUST_V23_DIRECTIONS = ("loss", "gain")

# -----------------------------------------------------------------------------
# V3 discrimination analysis: residual gain artifact vs real charge transition
# -----------------------------------------------------------------------------
RUN_V3_DISCRIMINATION = True

# A robust gain from an NV that is NV- in almost every OTHER rep11 measurement
# remains suspicious even after global common-mode correction.
HISTORY_SUSPICIOUS_NVM_OCCUPANCY = 0.95

# Estimate state-conditioned readout noise from multiplicatively corrected
# rep11+rep12 counts.  Events are then reported in units of their own NV's
# state-conditioned sigma rather than only in raw counts from threshold.
STATE_NOISE_MIN_SAMPLES = 20

# Local optical/readout artifact test.  For each target NV, compare the median
# brightness of stable neighboring reference NVs with the global run brightness.
# A negative local z means the target neighborhood is dimmer than the array-wide
# common mode in that run.
LOCAL_BRIGHTNESS_RADII_UM = (10.0, 20.0, 30.0)
LOCAL_BRIGHTNESS_PRIMARY_RADIUS_UM = 20.0
LOCAL_BRIGHTNESS_MIN_REFERENCE_NVS = 3
LOCAL_DIM_Z_CUT = -2.0

# Exact Poisson-binomial FDR.  Exact PB tails are expensive, so screen with the
# heterogeneous-null Z first.  Unscreened runs are assigned p=1; therefore the
# resulting BH q-values are conservative.
EXACT_FDR_SCREEN_Z = 2.0
EXACT_FDR_LEVELS = (0.05, 0.01)
TOP_FDR_RUNS_TO_PRINT = 20
SAVE_V3_EVENT_AUDIT_CSV = True
SAVE_V3_NV_QUALITY_CSV = True

# -----------------------------------------------------------------------------
# V4 probabilistic / soft charge-state analysis
# -----------------------------------------------------------------------------
RUN_V4_SOFT_CLASSIFIER = True

# Cross-fit the state model across RUNS so a candidate run never contributes to
# its own NV-/NV0 likelihood parameters. This is separate from the NV-fold
# cross-fitting used for common-mode correction.
SOFT_NUM_RUN_FOLDS = 5
SOFT_MODEL_TRAIN_MARGIN_COUNTS = 3.0
SOFT_MODEL_MIN_STATE_SAMPLES = 20
SOFT_MODEL_MIN_FALLBACK_SAMPLES = 5
SOFT_PRIOR_ALPHA = 0.5
SOFT_PRIOR_MIN = 1.0e-3
SOFT_SIGMA_FLOOR_COUNTS = 0.50
SOFT_NV_MIN_GOOD_RUN_COVERAGE = 0.95
SOFT_RUN_MIN_VALID_NV_FRACTION = 0.95
SOFT_RANDOM_SEED = 260827

# Build a candidate-free-ish background by removing preliminary |z| >= 3 runs,
# then independently resample each NV's soft weight from background runs. This
# destroys same-run coincidences while preserving each NV's empirical soft
# weight distribution and readout ambiguity.
SOFT_BACKGROUND_MAX_ABS_Z = 3.0
SOFT_BACKGROUND_MIN_FRACTION = 0.50
SOFT_NULL_MONTE_CARLO_SAMPLES = 200000
SOFT_NULL_FINAL_RECOMMENDED_SAMPLES = 1000000
SOFT_FDR_LEVELS = (0.05, 0.01)
SOFT_TOP_RUNS_TO_PRINT = 20
SOFT_EVENT_WEIGHT_CUTS = (0.50, 0.80)
SAVE_V4_EVENT_AUDIT_CSV = True

# -----------------------------------------------------------------------------
# V5 focused directionality analysis (minimal, presentation-oriented)
# -----------------------------------------------------------------------------
RUN_V5_FOCUSED_DIRECTIONALITY = True

# A discrete transition used for the final event-directionality figure must be
# both truth-tested (raw/additive/multiplicative consensus) AND have a large
# posterior transition weight. 0.95 is intentionally stringent; 0.90 is kept
# as a sensitivity check.
V5_PRIMARY_POSTERIOR_WEIGHT = 0.95
V5_SENSITIVITY_POSTERIOR_WEIGHTS = (0.90, 0.95)
V5_TOP_SPIKES_TO_PRINT = 15
V5_TOP_DISCRETE_RUNS_TO_PRINT = 15

# Brightness guide only; no candidate is accepted/rejected solely by this cut.
V5_DIM_RELEVANT_REP_Z = -2.0
SAVE_V5_CSV = True


# =============================================================================
# HELPERS
# =============================================================================


def _robust_center_scale(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.nan, np.nan
    vals = x[finite]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
    return med, sigma


def _robust_z_from_good(x, good):
    x = np.asarray(x, dtype=float)
    good = np.asarray(good, dtype=bool)
    med, sigma = _robust_center_scale(x[good])
    out = np.full(x.shape, np.nan, dtype=float)
    if np.isfinite(sigma) and sigma > 0:
        out[:] = (x - med) / sigma
    return out, med, sigma


def _safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def _parse_dark_wait_s(dataset, metadata=None):
    # Try explicit fields first.
    for src in (dataset, metadata or {}):
        for key in ("dark_wait_s", "wait_s", "dark_time_s"):
            if key in src:
                try:
                    return float(src[key])
                except Exception:
                    pass

    text = f"{dataset.get('label', '')} {dataset.get('file_stem', '')}"
    patterns = [
        r"wait[_-]?(\d+(?:\.\d+)?)s",
        r"source[_-]?off[_-]?(\d+(?:\.\d+)?)s",
        r"dark[_-]?wait[_-]?(\d+(?:\.\d+)?)s",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return np.nan


def _classify(c11, c12, thresholds, margin=0.0):
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

    gain = initial_nv0 & final_nvm
    loss = initial_nvm & final_nv0

    return {
        "initial_nvm": initial_nvm,
        "initial_nv0": initial_nv0,
        "final_nvm": final_nvm,
        "final_nv0": final_nv0,
        "gain": gain,
        "loss": loss,
        "initial_nvm_count": np.sum(initial_nvm, axis=0).astype(int),
        "initial_nv0_count": np.sum(initial_nv0, axis=0).astype(int),
        "gain_count": np.sum(gain, axis=0).astype(int),
        "loss_count": np.sum(loss, axis=0).astype(int),
    }


def _loo_rep11_nvm_occupancy(c11, thresholds, good):
    """
    Leave-one-run-out probability that each NV is classified NV- at rep11.

    Returns shape (N_NV, N_runs).  For a candidate in run r this answers:
      "In the OTHER good runs, how often is this same NV normally NV-?"
    """
    c11 = np.asarray(c11, dtype=float)
    thr = np.asarray(thresholds, dtype=float).ravel()[:, None]
    good = np.asarray(good, dtype=bool)

    finite = np.isfinite(c11) & np.isfinite(thr)
    nvm = finite & (c11 > thr)

    good2 = good[None, :]
    nvm_good = nvm & good2
    finite_good = finite & good2

    total_nvm = np.sum(nvm_good, axis=1).astype(float)
    total_finite = np.sum(finite_good, axis=1).astype(float)

    numerator = total_nvm[:, None] - nvm_good.astype(float)
    denominator = total_finite[:, None] - finite_good.astype(float)

    return _safe_divide(numerator, denominator)


def _build_reference_mask(c11, thresholds, good, occupancy_all_good):
    """Stable bright NVs for run-wise common-mode estimation."""
    c11 = np.asarray(c11, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    good = np.asarray(good, dtype=bool)
    occ = np.asarray(occupancy_all_good, dtype=float)

    baseline11 = np.nanmedian(c11[:, good], axis=1)

    ref = (
        np.isfinite(baseline11)
        & np.isfinite(thresholds)
        & np.isfinite(occ)
        & (occ >= float(REFERENCE_OCCUPANCY))
        & (baseline11 > thresholds + float(REFERENCE_BASELINE_MARGIN_COUNTS))
    )

    if np.sum(ref) < MIN_REFERENCE_NVS:
        ref = (
            np.isfinite(baseline11)
            & np.isfinite(thresholds)
            & np.isfinite(occ)
            & (occ >= float(REFERENCE_OCCUPANCY))
        )

    if np.sum(ref) < MIN_REFERENCE_NVS:
        raise RuntimeError(
            f"Too few stable reference NVs: {np.sum(ref)} < {MIN_REFERENCE_NVS}"
        )

    return ref


def _crossfit_common_mode(c, good, reference_mask, num_folds=5):
    """
    Cross-fitted additive + multiplicative common-mode correction.

    NV i is assigned fold f = i mod num_folds.  For that NV, the correction
    is estimated from stable reference NVs outside f, so the target NV never
    contributes to its own run-wise correction.

    Returns full corrected arrays with same shape as c, plus the per-NV/run
    additive shift and multiplicative scale actually used.
    """
    c = np.asarray(c, dtype=float)
    good = np.asarray(good, dtype=bool)
    ref = np.asarray(reference_mask, dtype=bool)
    n_nv, n_run = c.shape

    baseline = np.nanmedian(c[:, good], axis=1)
    fold_id = np.arange(n_nv, dtype=int) % int(num_folds)

    corr_add = np.full_like(c, np.nan, dtype=float)
    corr_mult = np.full_like(c, np.nan, dtype=float)
    used_delta = np.full_like(c, np.nan, dtype=float)
    used_scale = np.full_like(c, np.nan, dtype=float)

    fold_delta = np.full((num_folds, n_run), np.nan, dtype=float)
    fold_scale = np.full((num_folds, n_run), np.nan, dtype=float)

    for f in range(num_folds):
        refs = ref & (fold_id != f) & np.isfinite(baseline)
        if np.sum(refs) < MIN_REFERENCE_NVS:
            raise RuntimeError(
                f"Too few cross-fit references for fold {f}: {np.sum(refs)}"
            )

        # Additive shift.
        residual = c[refs, :] - baseline[refs, None]
        delta = np.nanmedian(residual, axis=0)

        # Multiplicative scale.
        valid_ratio_ref = refs & (baseline > 0)
        ratio = _safe_divide(
            c[valid_ratio_ref, :],
            baseline[valid_ratio_ref, None],
        )
        scale = np.nanmedian(ratio, axis=0)
        bad = ~np.isfinite(scale) | (scale <= 0)
        scale[bad] = 1.0

        fold_delta[f, :] = delta
        fold_scale[f, :] = scale

        targets = fold_id == f
        corr_add[targets, :] = c[targets, :] - delta[None, :]
        corr_mult[targets, :] = c[targets, :] / scale[None, :]
        used_delta[targets, :] = delta[None, :]
        used_scale[targets, :] = scale[None, :]

    return {
        "baseline_per_nv": baseline,
        "fold_id": fold_id,
        "corrected_additive": corr_add,
        "corrected_multiplicative": corr_mult,
        "used_additive_shift": used_delta,
        "used_multiplicative_scale": used_scale,
        "fold_additive_shift": fold_delta,
        "fold_multiplicative_scale": fold_scale,
    }


def _event_verdict_masks(
    c11,
    c12,
    thresholds,
    raw0,
    raw_strict,
    add_strict,
    mult_strict,
    add0,
    mult0,
    occ_loo,
):
    """Build event-level robust/misclassification/ambiguous masks."""
    thr = np.asarray(thresholds, dtype=float).ravel()[:, None]
    m = float(STRICT_MARGIN_COUNTS)

    # -----------------------------
    # GAINS: raw NV0 -> NV-
    # -----------------------------
    raw_gain = raw0["gain"]

    # Strong real-candidate gain: confidently NV0 then NV- in raw AND both
    # cross-fitted correction models.
    robust_gain = (
        raw_gain
        & raw_strict["gain"]
        & add_strict["gain"]
        & mult_strict["gain"]
    )

    # Strong rep11 misclassification signature: raw says NV0 -> NV-, but after
    # BOTH corrections the NV is confidently NV- at both reps.  High LOO
    # occupancy makes this especially compelling.
    # Explicit strict conditions using corrected classifications.
    add_same_nvm_strict = add_strict["initial_nvm"] & add_strict["final_nvm"]
    mult_same_nvm_strict = mult_strict["initial_nvm"] & mult_strict["final_nvm"]

    rep11_misclass_gain = (
        raw_gain
        & add_same_nvm_strict
        & mult_same_nvm_strict
        & (occ_loo >= float(GAIN_STABLE_NVM_OCCUPANCY))
    )

    # Measurement-sensitive gain even if not strong enough for the strict
    # high-occupancy label: the raw gain disappears in either correction.
    gain_sensitive = raw_gain & ~(add0["gain"] & mult0["gain"])

    ambiguous_gain = raw_gain & ~robust_gain & ~rep11_misclass_gain

    # -----------------------------
    # LOSSES: raw NV- -> NV0
    # -----------------------------
    raw_loss = raw0["loss"]

    robust_loss = (
        raw_loss
        & raw_strict["loss"]
        & add_strict["loss"]
        & mult_strict["loss"]
    )

    # Strong rep12 misclassification signature: raw says NV- -> NV0, but after
    # BOTH corrections the NV is confidently NV- at both reps.
    add_same_nvm_loss = add_strict["initial_nvm"] & add_strict["final_nvm"]
    mult_same_nvm_loss = mult_strict["initial_nvm"] & mult_strict["final_nvm"]

    rep12_misclass_loss = raw_loss & add_same_nvm_loss & mult_same_nvm_loss

    # General measurement sensitivity: loss disappears under either correction.
    loss_sensitive = raw_loss & ~(add0["loss"] & mult0["loss"])

    ambiguous_loss = raw_loss & ~robust_loss & ~rep12_misclass_loss

    return {
        "robust_gain": robust_gain,
        "rep11_misclass_gain": rep11_misclass_gain,
        "gain_sensitive": gain_sensitive,
        "ambiguous_gain": ambiguous_gain,
        "robust_loss": robust_loss,
        "rep12_misclass_loss": rep12_misclass_loss,
        "loss_sensitive": loss_sensitive,
        "ambiguous_loss": ambiguous_loss,
    }


def _count(mask):
    return np.sum(np.asarray(mask, dtype=bool), axis=0).astype(int)


def _corr(x, y, good):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.asarray(good, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 4:
        return np.nan, np.nan
    r = spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue)


def _write_run_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_event_csv(
    path,
    label,
    good,
    c11,
    c12,
    thresholds,
    raw0,
    raw_strict,
    cm11,
    cm12,
    add0,
    mult0,
    occ_loo,
    verdict,
    b11_z,
    b12_z,
):
    """Write one row per raw gain/loss event without building a huge list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    thr = np.asarray(thresholds, dtype=float).ravel()
    good = np.asarray(good, dtype=bool)

    fields = [
        "dataset", "run", "nv", "direction", "verdict",
        "c11_raw", "c12_raw", "threshold",
        "d_initial_raw", "d_final_raw",
        "c11_add", "c12_add", "c11_mult", "c12_mult",
        "d_initial_add", "d_final_add",
        "d_initial_mult", "d_final_mult",
        "raw_strict_margin_survives",
        "loo_rep11_nvm_occupancy",
        "rep11_brightness_z", "rep12_brightness_z",
        "scale11_used", "scale12_used",
        "delta11_used", "delta12_used",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for direction in ("gain", "loss"):
            raw_mask = raw0[direction] & good[None, :]
            inds = np.argwhere(raw_mask)

            if direction == "gain":
                robust_mask = verdict["robust_gain"]
                mis_mask = verdict["rep11_misclass_gain"]
                amb_mask = verdict["ambiguous_gain"]
                sensitive_mask = verdict["gain_sensitive"]
            else:
                robust_mask = verdict["robust_loss"]
                mis_mask = verdict["rep12_misclass_loss"]
                amb_mask = verdict["ambiguous_loss"]
                sensitive_mask = verdict["loss_sensitive"]

            for nv, run in inds:
                nv = int(nv)
                run = int(run)

                if robust_mask[nv, run]:
                    lab = "robust_candidate"
                elif mis_mask[nv, run]:
                    lab = (
                        "likely_rep11_misclassification"
                        if direction == "gain"
                        else "likely_rep12_misclassification"
                    )
                elif sensitive_mask[nv, run]:
                    lab = "correction_sensitive"
                elif amb_mask[nv, run]:
                    lab = "ambiguous"
                else:
                    lab = "unclassified"

                t = thr[nv]

                if direction == "gain":
                    d_initial_raw = t - c11[nv, run]
                    d_final_raw = c12[nv, run] - t
                    d_initial_add = t - cm11["corrected_additive"][nv, run]
                    d_final_add = cm12["corrected_additive"][nv, run] - t
                    d_initial_mult = t - cm11["corrected_multiplicative"][nv, run]
                    d_final_mult = cm12["corrected_multiplicative"][nv, run] - t
                else:
                    d_initial_raw = c11[nv, run] - t
                    d_final_raw = t - c12[nv, run]
                    d_initial_add = cm11["corrected_additive"][nv, run] - t
                    d_final_add = t - cm12["corrected_additive"][nv, run]
                    d_initial_mult = cm11["corrected_multiplicative"][nv, run] - t
                    d_final_mult = t - cm12["corrected_multiplicative"][nv, run]

                row = {
                    "dataset": label,
                    "run": run,
                    "nv": nv,
                    "direction": direction,
                    "verdict": lab,
                    "c11_raw": f"{c11[nv, run]:.6g}",
                    "c12_raw": f"{c12[nv, run]:.6g}",
                    "threshold": f"{t:.6g}",
                    "d_initial_raw": f"{d_initial_raw:.6g}",
                    "d_final_raw": f"{d_final_raw:.6g}",
                    "c11_add": f"{cm11['corrected_additive'][nv, run]:.6g}",
                    "c12_add": f"{cm12['corrected_additive'][nv, run]:.6g}",
                    "c11_mult": f"{cm11['corrected_multiplicative'][nv, run]:.6g}",
                    "c12_mult": f"{cm12['corrected_multiplicative'][nv, run]:.6g}",
                    "d_initial_add": f"{d_initial_add:.6g}",
                    "d_final_add": f"{d_final_add:.6g}",
                    "d_initial_mult": f"{d_initial_mult:.6g}",
                    "d_final_mult": f"{d_final_mult:.6g}",
                    "raw_strict_margin_survives": bool(raw_strict[direction][nv, run]),
                    "loo_rep11_nvm_occupancy": f"{occ_loo[nv, run]:.6g}",
                    "rep11_brightness_z": f"{b11_z[run]:.6g}",
                    "rep12_brightness_z": f"{b12_z[run]:.6g}",
                    "scale11_used": f"{cm11['used_multiplicative_scale'][nv, run]:.6g}",
                    "scale12_used": f"{cm12['used_multiplicative_scale'][nv, run]:.6g}",
                    "delta11_used": f"{cm11['used_additive_shift'][nv, run]:.6g}",
                    "delta12_used": f"{cm12['used_additive_shift'][nv, run]:.6g}",
                }
                w.writerow(row)



# =============================================================================
# ROBUST EVENT-CHANNEL ANALYSIS HELPERS
# =============================================================================


def _truth_consensus_masks(raw_strict, add_strict, mult_strict, verdict):
    """
    Build switch/evaluable masks for the truth-tested event channels.

    A site is evaluable only when raw, additive-corrected and
    multiplicative-corrected classifiers all confidently agree on its initial
    charge state and on either "switched" or "retained" final state.

    This is intentionally stricter than the raw analysis.  It prevents
    ambiguous near-threshold sites from entering the rare-event or spatial null.
    """
    robust_loss = np.asarray(verdict["robust_loss"], dtype=bool)
    robust_gain = np.asarray(verdict["robust_gain"], dtype=bool)

    retained_nvm = (
        raw_strict["initial_nvm"] & raw_strict["final_nvm"]
        & add_strict["initial_nvm"] & add_strict["final_nvm"]
        & mult_strict["initial_nvm"] & mult_strict["final_nvm"]
    )
    retained_nv0 = (
        raw_strict["initial_nv0"] & raw_strict["final_nv0"]
        & add_strict["initial_nv0"] & add_strict["final_nv0"]
        & mult_strict["initial_nv0"] & mult_strict["final_nv0"]
    )

    loss_evaluable = robust_loss | retained_nvm
    gain_evaluable = robust_gain | retained_nv0

    return {
        "loss_switch": robust_loss,
        "loss_evaluable": loss_evaluable,
        "gain_switch": robust_gain,
        "gain_evaluable": gain_evaluable,
        "retained_nvm": retained_nvm,
        "retained_nv0": retained_nv0,
    }


def _smoothed_nv_probability(switch_mask, evaluable_mask, run_mask):
    """Jeffreys-smoothed per-NV transition probability."""
    run_mask = np.asarray(run_mask, dtype=bool)
    s = np.asarray(switch_mask, dtype=bool)[:, run_mask]
    e = np.asarray(evaluable_mask, dtype=bool)[:, run_mask]
    events = np.sum(s, axis=1).astype(float)
    trials = np.sum(e, axis=1).astype(float)
    p = (events + 0.5) / (trials + 1.0)
    p[trials <= 0] = np.nan
    return p, events, trials


def _heterogeneous_run_null(switch_mask, evaluable_mask, good):
    """
    Independent heterogeneous-NV null for one transition direction.

    Returns per-run expected mean/variance and standardized excess Z.  The final
    p_i values are estimated from central runs after one preliminary pass, so
    the largest candidate bursts do not define their own background.
    """
    s = np.asarray(switch_mask, dtype=bool)
    e = np.asarray(evaluable_mask, dtype=bool)
    good = np.asarray(good, dtype=bool)

    k = np.sum(s, axis=0).astype(float)

    def evaluate(p):
        pcol = np.nan_to_num(np.asarray(p, dtype=float), nan=0.0)[:, None]
        ef = e.astype(float, copy=False)
        mu = np.sum(ef * pcol, axis=0)
        var = np.sum(ef * pcol * (1.0 - pcol), axis=0)
        z = np.full(k.shape, np.nan)
        tail = np.full(k.shape, np.nan)
        valid = good & np.isfinite(mu) & np.isfinite(var) & (var > 0)
        z[valid] = (k[valid] - mu[valid]) / np.sqrt(var[valid])
        tail[valid] = norm.sf(
            (k[valid] - 0.5 - mu[valid]) / np.sqrt(var[valid])
        )
        return mu, var, z, tail, valid

    p0, _, _ = _smoothed_nv_probability(s, e, good)
    mu0, var0, z0, _, valid0 = evaluate(p0)

    background = (
        valid0
        & np.isfinite(z0)
        & (np.abs(z0) < float(TAIL_BACKGROUND_MAX_ABS_Z))
    )
    if np.sum(background) < float(TAIL_BACKGROUND_MIN_FRACTION) * np.sum(good):
        background = good.copy()

    p_bg, events_bg, trials_bg = _smoothed_nv_probability(s, e, background)
    mu, var, z, tail, valid = evaluate(p_bg)

    obs = k[valid]
    mu_good = mu[valid]
    var_good = var[valid]
    obs_mean = float(np.mean(obs)) if obs.size else np.nan
    obs_var = float(np.var(obs, ddof=1)) if obs.size > 1 else np.nan
    obs_fano = obs_var / obs_mean if obs_mean > 0 else np.nan

    # Law of total variance for a heterogeneous independent-switching null.
    expected_total_var = (
        float(np.mean(var_good))
        + (float(np.var(mu_good, ddof=1)) if mu_good.size > 1 else 0.0)
        if mu_good.size else np.nan
    )
    null_mean = float(np.mean(mu_good)) if mu_good.size else np.nan
    null_fano = (
        expected_total_var / null_mean
        if np.isfinite(expected_total_var) and null_mean > 0
        else np.nan
    )
    dispersion_ratio = (
        obs_var / expected_total_var
        if np.isfinite(obs_var) and np.isfinite(expected_total_var)
        and expected_total_var > 0
        else np.nan
    )

    threshold_rows = []
    for zcut in ROBUST_TAIL_Z_THRESHOLDS:
        n = int(np.sum(valid & (z >= float(zcut))))
        den = int(np.sum(valid))
        threshold_rows.append({
            "zcut": float(zcut),
            "count": n,
            "denominator": den,
            "fraction": n / den if den else np.nan,
            "normal_one_sided": float(norm.sf(float(zcut))),
        })

    return {
        "k": k,
        "p_background": p_bg,
        "background_events_by_nv": events_bg,
        "background_trials_by_nv": trials_bg,
        "background_run_mask": background,
        "mu": mu,
        "var": var,
        "z": z,
        "normal_tail_p": tail,
        "valid": valid,
        "observed_mean": obs_mean,
        "observed_variance": obs_var,
        "observed_fano": obs_fano,
        "null_mean": null_mean,
        "null_total_variance": expected_total_var,
        "null_fano": null_fano,
        "dispersion_ratio": dispersion_ratio,
        "threshold_rows": threshold_rows,
    }


def _exact_poibin_tail(probabilities, k_observed):
    """Exact Poisson-binomial upper tail; used only for top screened runs."""
    p = np.asarray(probabilities, dtype=float)
    p = p[np.isfinite(p)]
    p = np.clip(p, 0.0, 1.0)
    n = int(p.size)
    k = int(k_observed)
    if n == 0:
        return np.nan
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    # Prefer the tested implementation in the main spatial script when present.
    fn = getattr(base, "_v20_exact_poibin_tail", None)
    if callable(fn):
        return float(fn(p, k))

    pmf = np.zeros(n + 1, dtype=float)
    pmf[0] = 1.0
    highest = 0
    for pi in p:
        new = pmf.copy()
        new[1:highest + 2] = (
            pmf[1:highest + 2] * (1.0 - pi)
            + pmf[:highest + 1] * pi
        )
        new[0] = pmf[0] * (1.0 - pi)
        pmf = new
        highest += 1
    return float(np.sum(pmf[k:]))


def _top_exact_tail_rows(null_result, evaluable_mask, good, top_n):
    z = np.asarray(null_result["z"], dtype=float)
    k = np.asarray(null_result["k"], dtype=float)
    p_bg = np.asarray(null_result["p_background"], dtype=float)
    e = np.asarray(evaluable_mask, dtype=bool)
    good = np.asarray(good, dtype=bool)

    inds = np.where(good & np.isfinite(z))[0]
    inds = inds[np.argsort(z[inds])[::-1]]
    rows = []
    for r in inds[: min(int(top_n), inds.size)]:
        probs = p_bg[e[:, r]]
        p_exact = _exact_poibin_tail(probs, int(k[r]))
        sigma_exact = (
            float(norm.isf(np.clip(p_exact, 1e-300, 1.0 - 1e-16)))
            if np.isfinite(p_exact) and p_exact > 0
            else np.inf
        )
        rows.append({
            "run": int(r),
            "K": int(k[r]),
            "N_evaluable": int(np.sum(e[:, r])),
            "mu": float(null_result["mu"][r]),
            "z_approx": float(z[r]),
            "p_exact": float(p_exact),
            "sigma_exact": sigma_exact,
        })
    return rows


def _empirical_ccdf(values, grid):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    grid = np.asarray(grid, dtype=float)
    if values.size == 0:
        return np.full(grid.shape, np.nan)
    return np.asarray([np.mean(values >= x) for x in grid], dtype=float)


def _print_direction_tail_summary(direction, null_result, exact_rows):
    print(f"\nROBUST {direction.upper()} EVENT-CHANNEL STATISTICS")
    print("-" * 142)
    print(
        f"mean K={null_result['observed_mean']:.4f}; "
        f"variance={null_result['observed_variance']:.4f}; "
        f"Fano={null_result['observed_fano']:.4f}"
    )
    print(
        f"heterogeneous-null mean={null_result['null_mean']:.4f}; "
        f"null Fano={null_result['null_fano']:.4f}; "
        f"variance/null-variance={null_result['dispersion_ratio']:.4f}"
    )
    print("rare robust events:")
    for row in null_result["threshold_rows"]:
        print(
            f"  Z>={row['zcut']:.0f}: {row['count']}/{row['denominator']} "
            f"= {100.0*row['fraction']:.5f}% "
            f"(Gaussian one-sided {100.0*row['normal_one_sided']:.6f}%)"
        )
    print("top exact Poisson-binomial tails:")
    print("  run   K   N_eval    mu      Zapprox      p_exact      exact_sigma")
    for row in exact_rows[:10]:
        print(
            f"  {row['run']:4d} {row['K']:3d} {row['N_evaluable']:7d} "
            f"{row['mu']:7.3f} {row['z_approx']:10.3f} "
            f"{row['p_exact']:12.3e} {row['sigma_exact']:11.3f}"
        )


def _run_weighted_same_k_spatial(
    direction,
    switch_mask,
    evaluable_mask,
    good,
    p_background,
    small,
    dark_wait_s,
):
    """Run V23 on the truth-tested mask, without reading image pixels."""
    required = (
        "_coerce_img_coords",
        "_v23_global_weighted_same_k_null",
        "_v23_fit_weighted_k_spatial_length",
    )
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        return {
            "success": False,
            "reason": "main spatial script lacks: " + ", ".join(missing),
        }

    nv_list = small.get("nv_list", None)
    if nv_list is None:
        return {
            "success": False,
            "reason": (
                "nv_list/camera coordinates unavailable because metadata could "
                "not be loaded; non-spatial robust analyses are still valid"
            ),
        }

    coords_xy = base._coerce_img_coords(nv_list)
    um_per_pixel = float(getattr(base, "UM_PER_PIXEL", 1.0))
    coords_um = np.asarray(coords_xy, dtype=float) * um_per_pixel
    xspan = float(np.ptp(coords_um[:, 0]))
    yspan = float(np.ptp(coords_um[:, 1]))
    fov_diag = float(np.hypot(xspan, yspan))

    seed = (
        int(ROBUST_V23_RANDOM_SEED)
        + int(round(float(dark_wait_s) * 1000.0))
        + (0 if direction == "loss" else 1000003)
    )
    rng = np.random.default_rng(seed)

    old_num = getattr(base, "V23_NULL_DATASETS", None)
    try:
        if old_num is not None:
            base.V23_NULL_DATASETS = int(ROBUST_V23_NULL_DATASETS)
        v23_null = base._v23_global_weighted_same_k_null(
            coords_um,
            switch_mask,
            evaluable_mask,
            good,
            p_background,
            rng,
        )
        v23_fit = base._v23_fit_weighted_k_spatial_length(
            v23_null,
            fov_diag,
        )
    finally:
        if old_num is not None:
            base.V23_NULL_DATASETS = old_num

    return {
        "success": bool(v23_null.get("success", False)),
        "direction": direction,
        "coords_um": coords_um,
        "fov_diagonal_um": fov_diag,
        "null": v23_null,
        "fit": v23_fit,
    }


def _print_v23_summary(direction, spatial):
    print(f"\nROBUST {direction.upper()} WEIGHTED SAME-K SPATIAL NULL")
    print("-" * 142)
    if not spatial.get("success", False):
        print(f"unavailable: {spatial.get('reason', 'spatial null failed')}")
        return
    null = spatial["null"]
    fit = spatial["fit"]
    print(f"synthetic null data sets: {null.get('num_null_datasets', 0)}")
    if not fit.get("success", False):
        print(f"fit unavailable: {fit.get('reason', 'fit failed')}")
        return
    print(
        f"best model={fit['best_model']}; "
        f"DeltaAICc(decay vs constant)={fit['delta_aicc_vs_constant']:.2f}"
    )
    if fit.get("resolved", False):
        print(
            f"RESOLVED robust-{direction} xi_wK="
            f"{fit['xi_um']:.2f} +/- {fit['xi_se_um']:.2f} um"
        )
    elif fit.get("fov_limited", False):
        print(
            f"robust-{direction} spatial scale is FOV-limited: "
            f"xi_fit={fit['xi_um']:.1f} um; do not quote finite xi"
        )
    else:
        print("no finite distance-dependent excess survives the robust same-K null")



# =============================================================================
# V3 DISCRIMINATION HELPERS
# =============================================================================


def _bh_qvalues(p_values, tested_mask):
    """Benjamini-Hochberg q-values on the requested family of tests."""
    p = np.asarray(p_values, dtype=float)
    tested = np.asarray(tested_mask, dtype=bool) & np.isfinite(p)
    q = np.full(p.shape, np.nan, dtype=float)
    inds = np.where(tested)[0]
    if inds.size == 0:
        return q

    vals = np.clip(p[inds], 0.0, 1.0)
    order = np.argsort(vals)
    ranked = vals[order]
    m = float(ranked.size)
    ranks = np.arange(1, ranked.size + 1, dtype=float)
    qrank = ranked * m / ranks
    qrank = np.minimum.accumulate(qrank[::-1])[::-1]
    qrank = np.clip(qrank, 0.0, 1.0)

    tmp = np.empty_like(qrank)
    tmp[order] = qrank
    q[inds] = tmp
    return q


def _screened_exact_pb_fdr(null_result, evaluable_mask, good):
    """
    Conservative exact-Poisson-binomial + BH-FDR audit.

    Exact tails are calculated for runs whose approximate heterogeneous-null
    Z >= EXACT_FDR_SCREEN_Z.  All other good runs receive p=1.  This cannot
    create false discoveries relative to an all-run exact calculation; it can
    only make q-values more conservative.
    """
    z = np.asarray(null_result["z"], dtype=float)
    k = np.asarray(null_result["k"], dtype=float)
    p_bg = np.asarray(null_result["p_background"], dtype=float)
    e = np.asarray(evaluable_mask, dtype=bool)
    good = np.asarray(good, dtype=bool)

    p_exact = np.ones(k.shape, dtype=float)
    p_exact[~good] = np.nan
    screened = good & np.isfinite(z) & (z >= float(EXACT_FDR_SCREEN_Z))

    for r in np.where(screened)[0]:
        probs = p_bg[e[:, r]]
        p_exact[r] = _exact_poibin_tail(probs, int(k[r]))

    q_direction = _bh_qvalues(p_exact, good)
    sigma_exact = np.full(k.shape, np.nan, dtype=float)
    finite = good & np.isfinite(p_exact) & (p_exact > 0)
    sigma_exact[finite] = norm.isf(
        np.clip(p_exact[finite], 1e-300, 1.0 - 1e-16)
    )

    return {
        "p_exact_screened": p_exact,
        "q_direction": q_direction,
        "sigma_exact": sigma_exact,
        "screened_mask": screened,
        "num_screened": int(np.sum(screened)),
    }


def _combined_direction_fdr(fdr_loss, fdr_gain, good):
    """BH correction across loss + gain runs within one dataset."""
    good = np.asarray(good, dtype=bool)
    p_loss = np.asarray(fdr_loss["p_exact_screened"], dtype=float)
    p_gain = np.asarray(fdr_gain["p_exact_screened"], dtype=float)
    n = good.size

    combined = np.concatenate([p_loss, p_gain])
    tested = np.concatenate([good, good])
    q = _bh_qvalues(combined, tested)
    return q[:n], q[n:]


def _state_conditioned_noise_model(c11_corr, c12_corr, thresholds, good):
    """
    Robust per-NV readout widths for NV- and NV0.

    Uses multiplicatively corrected counts from BOTH reps, but only points that
    lie beyond STRICT_MARGIN_COUNTS on the corresponding side of threshold.
    The purpose is not to fit a microscopic readout model; it is to express an
    event's threshold depth in units of that NV's empirical state-conditioned
    count noise.
    """
    c11_corr = np.asarray(c11_corr, dtype=float)
    c12_corr = np.asarray(c12_corr, dtype=float)
    thr = np.asarray(thresholds, dtype=float).ravel()
    good = np.asarray(good, dtype=bool)
    n_nv = thr.size

    sigma_nvm = np.full(n_nv, np.nan, dtype=float)
    sigma_nv0 = np.full(n_nv, np.nan, dtype=float)
    center_nvm = np.full(n_nv, np.nan, dtype=float)
    center_nv0 = np.full(n_nv, np.nan, dtype=float)
    n_nvm = np.zeros(n_nv, dtype=int)
    n_nv0 = np.zeros(n_nv, dtype=int)

    for i in range(n_nv):
        vals = np.concatenate([c11_corr[i, good], c12_corr[i, good]])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or not np.isfinite(thr[i]):
            continue

        vm = vals[vals > thr[i] + float(STRICT_MARGIN_COUNTS)]
        v0 = vals[vals <= thr[i] - float(STRICT_MARGIN_COUNTS)]
        n_nvm[i] = vm.size
        n_nv0[i] = v0.size

        if vm.size >= int(STATE_NOISE_MIN_SAMPLES):
            center_nvm[i], sigma_nvm[i] = _robust_center_scale(vm)
        if v0.size >= int(STATE_NOISE_MIN_SAMPLES):
            center_nv0[i], sigma_nv0[i] = _robust_center_scale(v0)

    # Conservative fallback for NVs with too few samples in one state.
    good_sm = np.isfinite(sigma_nvm) & (sigma_nvm > 0)
    good_s0 = np.isfinite(sigma_nv0) & (sigma_nv0 > 0)
    fallback_m = float(np.nanmedian(sigma_nvm[good_sm])) if np.any(good_sm) else 1.0
    fallback_0 = float(np.nanmedian(sigma_nv0[good_s0])) if np.any(good_s0) else fallback_m

    sigma_nvm[~good_sm] = fallback_m
    sigma_nv0[~good_s0] = fallback_0

    return {
        "sigma_nvm": sigma_nvm,
        "sigma_nv0": sigma_nv0,
        "center_nvm": center_nvm,
        "center_nv0": center_nv0,
        "n_nvm_samples": n_nvm,
        "n_nv0_samples": n_nv0,
        "fallback_sigma_nvm": fallback_m,
        "fallback_sigma_nv0": fallback_0,
    }


def _event_depth_sigma(direction, switch_mask, c11_corr, c12_corr, thresholds, noise):
    """Noise-normalized distance from threshold on both sides of a transition."""
    switch = np.asarray(switch_mask, dtype=bool)
    c11_corr = np.asarray(c11_corr, dtype=float)
    c12_corr = np.asarray(c12_corr, dtype=float)
    thr = np.asarray(thresholds, dtype=float).ravel()[:, None]
    sm = np.asarray(noise["sigma_nvm"], dtype=float)[:, None]
    s0 = np.asarray(noise["sigma_nv0"], dtype=float)[:, None]

    if direction == "gain":
        d_initial = _safe_divide(thr - c11_corr, s0)
        d_final = _safe_divide(c12_corr - thr, sm)
    elif direction == "loss":
        d_initial = _safe_divide(c11_corr - thr, sm)
        d_final = _safe_divide(thr - c12_corr, s0)
    else:
        raise ValueError(direction)

    d_initial[~switch] = np.nan
    d_final[~switch] = np.nan
    d_min = np.fmin(d_initial, d_final)
    d_min[~switch] = np.nan
    return {
        "initial_sigma": d_initial,
        "final_sigma": d_final,
        "min_sigma": d_min,
    }


def _coords_um_optional(small):
    nv_list = small.get("nv_list", None)
    if nv_list is None or not hasattr(base, "_coerce_img_coords"):
        return None
    try:
        xy = np.asarray(base._coerce_img_coords(nv_list), dtype=float)
        return xy * float(getattr(base, "UM_PER_PIXEL", 1.0))
    except Exception as exc:
        print(
            f"[local-brightness] camera-coordinate conversion failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _local_reference_brightness(c, good, reference_mask, coords_um, global_scale, radius_um):
    """
    Local-minus-global brightness for every target NV/run.

    The local estimator uses ONLY stable reference NVs within radius_um and
    excludes the target itself.  The result is converted to a robust z score
    independently for each target location across good runs.  Thus a negative
    z means that location is unusually dim relative to the array-wide common
    mode for that same run.
    """
    c = np.asarray(c, dtype=float)
    good = np.asarray(good, dtype=bool)
    ref = np.asarray(reference_mask, dtype=bool)
    xy = np.asarray(coords_um, dtype=float)
    global_scale = np.asarray(global_scale, dtype=float)
    n_nv, n_run = c.shape

    baseline = np.nanmedian(c[:, good], axis=1)
    ratio = _safe_divide(c, baseline[:, None])

    local_scale = np.full((n_nv, n_run), np.nan, dtype=np.float32)
    local_resid = np.full((n_nv, n_run), np.nan, dtype=np.float32)
    local_z = np.full((n_nv, n_run), np.nan, dtype=np.float32)
    n_neighbors = np.zeros(n_nv, dtype=int)

    for i in range(n_nv):
        d = np.linalg.norm(xy - xy[i], axis=1)
        nbr = ref & np.isfinite(d) & (d > 0) & (d <= float(radius_um))
        n_neighbors[i] = int(np.sum(nbr))
        if n_neighbors[i] < int(LOCAL_BRIGHTNESS_MIN_REFERENCE_NVS):
            continue

        ls = np.nanmedian(ratio[nbr, :], axis=0)
        resid = ls - global_scale
        local_scale[i, :] = ls.astype(np.float32)
        local_resid[i, :] = resid.astype(np.float32)

        med, sig = _robust_center_scale(resid[good])
        if np.isfinite(sig) and sig > 0:
            local_z[i, :] = ((resid - med) / sig).astype(np.float32)

    return {
        "radius_um": float(radius_um),
        "local_scale": local_scale,
        "local_minus_global": local_resid,
        "z": local_z,
        "n_neighbors": n_neighbors,
    }


def _masked_values(arr, mask, good):
    arr = np.asarray(arr, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.asarray(good, dtype=bool)[None, :]
    vals = arr[mask]
    return vals[np.isfinite(vals)]


def _run_median_over_events(arr, event_mask):
    """Median event-level diagnostic in each run; NaN when no event."""
    arr = np.asarray(arr, dtype=float)
    event_mask = np.asarray(event_mask, dtype=bool)
    n_run = event_mask.shape[1]
    out = np.full(n_run, np.nan, dtype=float)
    for r in range(n_run):
        inds = event_mask[:, r] & np.isfinite(arr[:, r])
        if np.any(inds):
            out[r] = float(np.nanmedian(arr[inds, r]))
    return out


def _build_candidate_audit_rows(
    label,
    dark_wait_s,
    direction,
    switch_mask,
    null_result,
    fdr_result,
    q_combined,
    occ_loo,
    depth,
    local_by_radius,
    good,
):
    """Run-level table combining significance with artifact diagnostics."""
    switch = np.asarray(switch_mask, dtype=bool)
    good = np.asarray(good, dtype=bool)
    k = np.asarray(null_result["k"], dtype=float)
    z = np.asarray(null_result["z"], dtype=float)
    p = np.asarray(fdr_result["p_exact_screened"], dtype=float)
    qd = np.asarray(fdr_result["q_direction"], dtype=float)
    qc = np.asarray(q_combined, dtype=float)

    order_pool = np.where(good & np.isfinite(p))[0]
    order = order_pool[np.argsort(p[order_pool])]
    rows = []
    for r in order[: min(TOP_FDR_RUNS_TO_PRINT, order.size)]:
        ev = switch[:, r]
        if not np.any(ev):
            continue

        occ = occ_loo[ev, r]
        d0 = depth["initial_sigma"][ev, r]
        d1 = depth["final_sigma"][ev, r]
        dm = depth["min_sigma"][ev, r]

        row = {
            "dataset": label,
            "dark_wait_s": float(dark_wait_s),
            "direction": direction,
            "run": int(r),
            "K_robust": int(k[r]),
            "mu": float(null_result["mu"][r]),
            "z_approx": float(z[r]),
            "p_exact_screened": float(p[r]),
            "q_direction": float(qd[r]),
            "q_loss_gain_family": float(qc[r]),
            "median_loo_rep11_nvm_occupancy": float(np.nanmedian(occ)) if np.any(np.isfinite(occ)) else np.nan,
            "fraction_event_nvs_normally_nvm_ge95": (
                float(np.mean(occ[np.isfinite(occ)] >= HISTORY_SUSPICIOUS_NVM_OCCUPANCY))
                if np.any(np.isfinite(occ)) else np.nan
            ),
            "median_initial_depth_sigma": float(np.nanmedian(d0)) if np.any(np.isfinite(d0)) else np.nan,
            "median_final_depth_sigma": float(np.nanmedian(d1)) if np.any(np.isfinite(d1)) else np.nan,
            "median_min_depth_sigma": float(np.nanmedian(dm)) if np.any(np.isfinite(dm)) else np.nan,
        }

        for radius, reps in local_by_radius.items():
            key = "rep11" if direction == "gain" else "rep12"
            zz = reps[key]["z"][ev, r]
            goodz = np.isfinite(zz)
            tag = f"{float(radius):g}um"
            row[f"median_relevant_local_z_{tag}"] = (
                float(np.nanmedian(zz[goodz])) if np.any(goodz) else np.nan
            )
            row[f"fraction_relevant_local_z_le_m2_{tag}"] = (
                float(np.mean(zz[goodz] <= LOCAL_DIM_Z_CUT)) if np.any(goodz) else np.nan
            )

        rows.append(row)
    return rows


def _write_dict_rows_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_robust_event_audit_csv(
    path,
    label,
    dark_wait_s,
    truth_masks,
    good,
    occ_loo,
    depth_by_direction,
    local_by_radius,
    robust_tail,
    fdr_by_direction,
    q_combined_by_direction,
):
    """One row per truth-tested robust transition with V3 diagnostics."""
    fields = [
        "dataset", "dark_wait_s", "direction", "run", "nv",
        "run_K_robust", "run_mu", "run_z_approx",
        "run_p_exact_screened", "run_q_direction", "run_q_loss_gain_family",
        "loo_rep11_nvm_occupancy", "normally_nvm_ge95",
        "initial_depth_sigma", "final_depth_sigma", "min_depth_sigma",
    ]
    for radius in LOCAL_BRIGHTNESS_RADII_UM:
        tag = f"{float(radius):g}um"
        fields.extend([
            f"local_rep11_z_{tag}", f"local_rep12_z_{tag}",
            f"local_relevant_z_{tag}",
        ])

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for direction in ("loss", "gain"):
            mask = np.asarray(truth_masks[f"{direction}_switch"], dtype=bool)
            mask &= np.asarray(good, dtype=bool)[None, :]
            depth = depth_by_direction[direction]
            tail = robust_tail[direction]
            fdr = fdr_by_direction[direction]
            qcomb = q_combined_by_direction[direction]
            for nv, run in np.argwhere(mask):
                nv = int(nv); run = int(run)
                row = {
                    "dataset": label,
                    "dark_wait_s": float(dark_wait_s),
                    "direction": direction,
                    "run": run,
                    "nv": nv,
                    "run_K_robust": int(tail["k"][run]),
                    "run_mu": float(tail["mu"][run]),
                    "run_z_approx": float(tail["z"][run]),
                    "run_p_exact_screened": float(fdr["p_exact_screened"][run]),
                    "run_q_direction": float(fdr["q_direction"][run]),
                    "run_q_loss_gain_family": float(qcomb[run]),
                    "loo_rep11_nvm_occupancy": float(occ_loo[nv, run]),
                    "normally_nvm_ge95": bool(occ_loo[nv, run] >= HISTORY_SUSPICIOUS_NVM_OCCUPANCY),
                    "initial_depth_sigma": float(depth["initial_sigma"][nv, run]),
                    "final_depth_sigma": float(depth["final_sigma"][nv, run]),
                    "min_depth_sigma": float(depth["min_sigma"][nv, run]),
                }
                for radius in LOCAL_BRIGHTNESS_RADII_UM:
                    tag = f"{float(radius):g}um"
                    reps = local_by_radius.get(float(radius), None)
                    if reps is None:
                        z11 = z12 = np.nan
                    else:
                        z11 = float(reps["rep11"]["z"][nv, run])
                        z12 = float(reps["rep12"]["z"][nv, run])
                    row[f"local_rep11_z_{tag}"] = z11
                    row[f"local_rep12_z_{tag}"] = z12
                    row[f"local_relevant_z_{tag}"] = z11 if direction == "gain" else z12
                w.writerow(row)


def _print_v3_discrimination_summary(
    direction,
    switch_mask,
    good,
    occ_loo,
    depth,
    local_primary,
    fdr,
    q_combined,
):
    mask = np.asarray(switch_mask, dtype=bool) & np.asarray(good, dtype=bool)[None, :]
    occ = occ_loo[mask]
    mind = depth["min_sigma"][mask]
    relevant_key = "rep11" if direction == "gain" else "rep12"
    if local_primary is not None:
        lz = local_primary[relevant_key]["z"][mask]
    else:
        lz = np.asarray([], dtype=float)

    print(f"\nV3 {direction.upper()} DISCRIMINATION: HISTORY / DEPTH / LOCAL BRIGHTNESS / FDR")
    print("-" * 142)
    if occ.size:
        print(
            f"robust event-NV history: median P(NV- at OTHER rep11)="
            f"{np.nanmedian(occ):.4f}; fraction >= {HISTORY_SUSPICIOUS_NVM_OCCUPANCY:.2f}="
            f"{100*np.mean(occ[np.isfinite(occ)] >= HISTORY_SUSPICIOUS_NVM_OCCUPANCY):.2f}%"
        )
    finite_d = mind[np.isfinite(mind)]
    if finite_d.size:
        print(
            f"noise-normalized two-sided confidence: median min-depth="
            f"{np.nanmedian(finite_d):.3f} sigma; "
            f"fraction min-depth<2 sigma={100*np.mean(finite_d < 2.0):.2f}%"
        )
    finite_lz = lz[np.isfinite(lz)]
    if finite_lz.size:
        rep = 11 if direction == "gain" else 12
        print(
            f"local rep{rep} brightness at {LOCAL_BRIGHTNESS_PRIMARY_RADIUS_UM:g} um: "
            f"median z={np.nanmedian(finite_lz):+.3f}; "
            f"fraction z<={LOCAL_DIM_Z_CUT:g}={100*np.mean(finite_lz <= LOCAL_DIM_Z_CUT):.2f}%"
        )
    else:
        print("local brightness: unavailable (camera coordinates/nv_list missing)")

    qd = np.asarray(fdr["q_direction"], dtype=float)
    qc = np.asarray(q_combined, dtype=float)
    goodv = np.asarray(good, dtype=bool)
    print(
        f"exact-PB screen: {fdr['num_screened']}/{np.sum(goodv)} good runs "
        f"had approximate Z>={EXACT_FDR_SCREEN_Z:g} and received exact tails"
    )
    for alpha in EXACT_FDR_LEVELS:
        nd = int(np.sum(goodv & np.isfinite(qd) & (qd <= alpha)))
        nc = int(np.sum(goodv & np.isfinite(qc) & (qc <= alpha)))
        print(
            f"  q<={alpha:g}: {nd} run(s) within-direction; "
            f"{nc} run(s) after loss+gain family correction"
        )


def _per_nv_quality_rows(
    label,
    occ_all,
    noise,
    truth_masks,
    good,
    local_by_radius,
):
    good = np.asarray(good, dtype=bool)
    rg = np.sum(np.asarray(truth_masks["gain_switch"], dtype=bool)[:, good], axis=1)
    rl = np.sum(np.asarray(truth_masks["loss_switch"], dtype=bool)[:, good], axis=1)
    rows = []
    n_nv = len(occ_all)
    for i in range(n_nv):
        row = {
            "dataset": label,
            "nv": int(i),
            "rep11_nvm_occupancy_all_good": float(occ_all[i]),
            "sigma_nvm": float(noise["sigma_nvm"][i]),
            "sigma_nv0": float(noise["sigma_nv0"][i]),
            "n_nvm_noise_samples": int(noise["n_nvm_samples"][i]),
            "n_nv0_noise_samples": int(noise["n_nv0_samples"][i]),
            "robust_gain_event_count": int(rg[i]),
            "robust_loss_event_count": int(rl[i]),
        }
        for radius, reps in local_by_radius.items():
            row[f"local_reference_neighbors_{float(radius):g}um"] = int(
                reps["rep11"]["n_neighbors"][i]
            )
        rows.append(row)
    return rows


# =============================================================================
# ROBUST METADATA / SMALL-MEMBER LOADING
# =============================================================================


def _load_metadata_robust(file_stem):
    """Load metadata without touching the large NPZ.

    First use the main script helper.  If that fails (for example because a
    stale/corrupt local data-manager cache raises OSError [Errno 22]), retry
    with ``use_cache=False`` when supported.  This second attempt downloads
    only the small .txt metadata, never the linked NPZ.
    """
    metadata = base._try_metadata_without_npz(file_stem)
    if isinstance(metadata, dict):
        return metadata

    print(
        "[metadata-fallback] retrying data_manager with use_cache=False...",
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
            # Older data_manager versions may not expose use_cache.
            metadata = dm.get_raw_data(
                file_stem=file_stem,
                load_npz=False,
            )
    except SystemExit as exc:
        print(
            f"[metadata-fallback] metadata retry attempted SystemExit "
            f"(code={exc.code!r}); continuing without metadata.",
            flush=True,
        )
        return None
    except Exception as exc:
        print(
            f"[metadata-fallback] retry failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None

    if isinstance(metadata, dict):
        print(
            f"[metadata-fallback] metadata recovered: "
            f"{len(metadata)} top-level keys",
            flush=True,
        )
        return metadata

    print(
        f"[metadata-fallback] retry returned {type(metadata).__name__}; "
        "continuing without metadata.",
        flush=True,
    )
    return None


def _load_small_dataset_members_tolerant(npz_path, metadata=None):
    """Load counts + thresholds safely; make nv_list optional.

    The truth/tail/Fano analyses require only c11, c12 and thresholds.  Spatial
    analysis additionally requires nv_list for camera coordinates.  The main
    spatial loader historically raises immediately when nv_list is unavailable,
    which unnecessarily prevents all non-spatial diagnostics from running.

    This loader never reads ``img_arrays`` and never requests ``load_npz=True``.
    """
    print("[large-file] loading counts + small metadata only...", flush=True)

    c11, c12 = base._load_count_reps_streaming(
        npz_path,
        rep_initial=getattr(base, "REP_INITIAL", 11),
        rep_final=getattr(base, "REP_FINAL", 12),
    )
    num_nvs, num_runs = c11.shape

    thresholds = None
    nv_list = None
    dark_wait_s = None

    # Only small NPZ members are accessed.  Merely opening the archive/listing
    # names does not decompress counts or img_arrays; counts were streamed above.
    with np.load(npz_path, allow_pickle=True) as archive:
        keys = set(archive.files)

        for key in ("analysis_thresholds", "thresholds"):
            if key in keys:
                thresholds = np.asarray(archive[key], dtype=np.float32).ravel()
                break

        if "nv_list" in keys:
            value = archive["nv_list"]
            if isinstance(value, np.ndarray) and value.dtype == object:
                nv_list = value.tolist()
            else:
                nv_list = value

        if "dark_wait_s" in keys:
            try:
                dark_wait_s = float(np.asarray(archive["dark_wait_s"]).item())
            except Exception:
                dark_wait_s = None

    if thresholds is None and isinstance(metadata, dict):
        for key in ("analysis_thresholds", "thresholds"):
            if key in metadata:
                thresholds = np.asarray(metadata[key], dtype=np.float32).ravel()
                break

    if nv_list is None and isinstance(metadata, dict):
        nv_list = metadata.get("nv_list", None)

    if dark_wait_s is None and isinstance(metadata, dict):
        try:
            dark_wait_s = float(metadata.get("dark_wait_s", np.nan))
        except Exception:
            dark_wait_s = np.nan

    if thresholds is None:
        raise ValueError("Could not find analysis_thresholds or thresholds.")
    if thresholds.shape != (num_nvs,):
        raise ValueError(
            f"Threshold shape {thresholds.shape} does not match {num_nvs} NVs."
        )

    if nv_list is None:
        print(
            "[metadata-warning] nv_list unavailable. All truth/tail/Fano "
            "diagnostics will run; weighted spatial analysis will be skipped "
            "for this dataset unless metadata becomes available.",
            flush=True,
        )

    print(
        f"[large-file] counts ready: NVs={num_nvs}, runs={num_runs}, "
        f"dark_wait_s={dark_wait_s}",
        flush=True,
    )

    return {
        "c11": c11,
        "c12": c12,
        "thresholds": thresholds,
        "nv_list": nv_list,
        "dark_wait_s": dark_wait_s,
    }



# =============================================================================
# V5 FOCUSED DIRECTIONALITY HELPERS
# =============================================================================


def _v5_state_confidences(post, direction):
    """Return initial/final state confidence arrays for one transition direction."""
    p11 = np.asarray(post["p_nvm_rep11"], dtype=float)
    p12 = np.asarray(post["p_nvm_rep12"], dtype=float)
    if direction == "loss":
        # NV-(rep11) -> NV0(rep12)
        return p11, 1.0 - p12
    if direction == "gain":
        # NV0(rep11) -> NV-(rep12); the initial state is the dim state.
        return 1.0 - p11, p12
    raise ValueError(direction)


def _v5_event_conf_summary(post, weights, hard_switch, good, direction):
    init_conf, final_conf = _v5_state_confidences(post, direction)
    hard = np.asarray(hard_switch, dtype=bool) & np.asarray(good, dtype=bool)[None, :]
    w = np.asarray(weights, dtype=float)
    vals_i = init_conf[hard]
    vals_f = final_conf[hard]
    vals_w = w[hard]
    vals_i = vals_i[np.isfinite(vals_i)]
    vals_f = vals_f[np.isfinite(vals_f)]
    vals_w = vals_w[np.isfinite(vals_w)]
    return {
        "n": int(vals_w.size),
        "median_initial_conf": float(np.median(vals_i)) if vals_i.size else np.nan,
        "median_final_conf": float(np.median(vals_f)) if vals_f.size else np.nan,
        "median_weight": float(np.median(vals_w)) if vals_w.size else np.nan,
        "mean_weight": float(np.mean(vals_w)) if vals_w.size else np.nan,
        "fraction_weight_ge_0p90": float(np.mean(vals_w >= 0.90)) if vals_w.size else np.nan,
        "fraction_weight_ge_0p95": float(np.mean(vals_w >= 0.95)) if vals_w.size else np.nan,
    }


def _v5_run_median_event_value(values, event_mask):
    values = np.asarray(values, dtype=float)
    event_mask = np.asarray(event_mask, dtype=bool)
    out = np.full(values.shape[1], np.nan, dtype=float)
    for r in range(values.shape[1]):
        v = values[event_mask[:, r], r]
        v = v[np.isfinite(v)]
        if v.size:
            out[r] = float(np.median(v))
    return out


def _v5_robust_run_z(k, good):
    k = np.asarray(k, dtype=float)
    good = np.asarray(good, dtype=bool)
    med, sig = _robust_center_scale(k[good])
    z = np.full(k.shape, np.nan, dtype=float)
    if np.isfinite(sig) and sig > 0:
        z[good] = (k[good] - med) / sig
    return z, med, sig


def _run_v5_focused(label, dark_wait_s, post, weights_by_direction, truth_masks,
                    raw_gain, raw_loss, robust_gain, robust_loss,
                    scale11_z, scale12_z, good):
    """
    Minimal directionality audit.

    The primary event count is not the total soft score. It is the number of
    truth-tested hard transitions whose posterior transition weight is >=0.95.
    This intentionally favors discrete, individually resolved charge changes
    and suppresses distributed weak posterior shifts.
    """
    good = np.asarray(good, dtype=bool)
    out = {"directions": {}, "primary_weight": float(V5_PRIMARY_POSTERIOR_WEIGHT)}

    print("\n" + "=" * 142)
    print("V5 FOCUSED DIRECTIONALITY TEST: DISCRETE HIGH-CONFIDENCE TRANSITIONS")
    print("=" * 142)
    print(
        "Primary event definition: truth-tested hard transition AND posterior "
        f"transition weight >= {V5_PRIMARY_POSTERIOR_WEIGHT:.2f}."
    )
    print(
        "Purpose: show whether NV- -> NV0 remains a clean discrete-event channel "
        "while reverse spikes are more sensitive to the dim rep11 initial-state readout."
    )

    for direction in ("loss", "gain"):
        hard = np.asarray(truth_masks[f"{direction}_switch"], dtype=bool)
        w = np.asarray(weights_by_direction[direction], dtype=float)
        init_conf, final_conf = _v5_state_confidences(post, direction)
        raw_run = np.asarray(raw_loss if direction == "loss" else raw_gain, dtype=int)
        robust_run = np.asarray(robust_loss if direction == "loss" else robust_gain, dtype=int)
        relevant_z = np.asarray(scale12_z if direction == "loss" else scale11_z, dtype=float)

        high_by_cut = {}
        mask_by_cut = {}
        for cut in V5_SENSITIVITY_POSTERIOR_WEIGHTS:
            m = hard & np.isfinite(w) & (w >= float(cut))
            mask_by_cut[float(cut)] = m
            high_by_cut[float(cut)] = np.sum(m, axis=0).astype(int)

        primary_mask = hard & np.isfinite(w) & (w >= float(V5_PRIMARY_POSTERIOR_WEIGHT))
        primary_k = np.sum(primary_mask, axis=0).astype(int)
        z_primary, med_primary, sig_primary = _v5_robust_run_z(primary_k, good)
        conf = _v5_event_conf_summary(post, w, hard, good, direction)

        run_med_init = _v5_run_median_event_value(init_conf, hard)
        run_med_final = _v5_run_median_event_value(final_conf, hard)
        run_med_weight = _v5_run_median_event_value(w, hard)

        total_raw = int(np.sum(raw_run[good]))
        total_rob = int(np.sum(robust_run[good]))
        total_hi = int(np.sum(primary_k[good]))

        rho_raw, p_raw = _corr(raw_run, relevant_z, good)
        rho_rob, p_rob = _corr(robust_run, relevant_z, good)
        rho_hi, p_hi = _corr(primary_k, relevant_z, good)

        out["directions"][direction] = {
            "raw_run": raw_run,
            "robust_run": robust_run,
            "primary_k": primary_k,
            "primary_mask": primary_mask,
            "high_k_by_cut": high_by_cut,
            "high_mask_by_cut": mask_by_cut,
            "z_primary": z_primary,
            "primary_median": med_primary,
            "primary_robust_sigma": sig_primary,
            "relevant_brightness_z": relevant_z,
            "run_median_initial_conf": run_med_init,
            "run_median_final_conf": run_med_final,
            "run_median_weight": run_med_weight,
            "event_confidence": conf,
            "total_raw": total_raw,
            "total_robust": total_rob,
            "total_primary": total_hi,
            "rho_raw_brightness": rho_raw,
            "p_raw_brightness": p_raw,
            "rho_robust_brightness": rho_rob,
            "p_robust_brightness": p_rob,
            "rho_primary_brightness": rho_hi,
            "p_primary_brightness": p_hi,
        }

        arrow = "NV- -> NV0" if direction == "loss" else "NV0 -> NV-"
        relevant_rep = "rep12" if direction == "loss" else "rep11 (dim initial NV0 readout)"
        print(f"\n{arrow}")
        print("-" * 142)
        print(
            f"raw -> robust -> posterior>={V5_PRIMARY_POSTERIOR_WEIGHT:.2f}: "
            f"{total_raw} -> {total_rob} -> {total_hi} total transitions"
        )
        print(
            f"survival: robust/raw={100*total_rob/max(total_raw,1):.2f}% ; "
            f"high-conf/raw={100*total_hi/max(total_raw,1):.2f}% ; "
            f"high-conf/robust={100*total_hi/max(total_rob,1):.2f}%"
        )
        print(
            f"robust-event state confidence: initial median={conf['median_initial_conf']:.4f}, "
            f"final median={conf['median_final_conf']:.4f}, "
            f"joint-weight median={conf['median_weight']:.4f}"
        )
        print(
            f"fraction of robust transitions with joint posterior >=0.90: "
            f"{100*conf['fraction_weight_ge_0p90']:.2f}% ; >=0.95: "
            f"{100*conf['fraction_weight_ge_0p95']:.2f}%"
        )
        print(
            f"brightness coupling ({relevant_rep}): "
            f"rho raw={rho_raw:+.3f}, robust={rho_rob:+.3f}, "
            f"high-conf={rho_hi:+.3f}"
        )

        print(f"\nTop raw {direction} spikes: raw -> robust -> high-confidence")
        print(
            "run   raw  robust  K95  robust/raw  K95/raw  relevantZ  "
            "medInitConf medJointW"
        )
        good_inds = np.where(good)[0]
        order = good_inds[np.argsort(raw_run[good_inds])[::-1]]
        rows = []
        for r in order[: min(int(V5_TOP_SPIKES_TO_PRINT), order.size)]:
            row = {
                "dataset": label,
                "dark_wait_s": float(dark_wait_s),
                "direction": direction,
                "run": int(r),
                "raw_K": int(raw_run[r]),
                "robust_K": int(robust_run[r]),
                "high_conf_K": int(primary_k[r]),
                "robust_over_raw": float(robust_run[r] / raw_run[r]) if raw_run[r] > 0 else np.nan,
                "high_conf_over_raw": float(primary_k[r] / raw_run[r]) if raw_run[r] > 0 else np.nan,
                "relevant_brightness_z": float(relevant_z[r]),
                "median_initial_state_confidence": float(run_med_init[r]) if np.isfinite(run_med_init[r]) else np.nan,
                "median_final_state_confidence": float(run_med_final[r]) if np.isfinite(run_med_final[r]) else np.nan,
                "median_joint_transition_weight": float(run_med_weight[r]) if np.isfinite(run_med_weight[r]) else np.nan,
                "high_conf_run_z": float(z_primary[r]) if np.isfinite(z_primary[r]) else np.nan,
            }
            rows.append(row)
            print(
                f"{r:4d} {raw_run[r]:5d} {robust_run[r]:7d} {primary_k[r]:4d} "
                f"{row['robust_over_raw']:10.3f} {row['high_conf_over_raw']:8.3f} "
                f"{relevant_z[r]:9.2f} {run_med_init[r]:11.3f} {run_med_weight[r]:9.3f}"
            )
        out["directions"][direction]["top_raw_spike_rows"] = rows

        print(f"\nTop high-confidence discrete {direction} runs")
        print("run  K95   z95  relevantZ  robustK rawK medInitConf medJointW")
        order2 = good_inds[np.argsort(primary_k[good_inds])[::-1]]
        event_rows = []
        for r in order2[: min(int(V5_TOP_DISCRETE_RUNS_TO_PRINT), order2.size)]:
            row = {
                "dataset": label,
                "dark_wait_s": float(dark_wait_s),
                "direction": direction,
                "run": int(r),
                "high_conf_K": int(primary_k[r]),
                "high_conf_run_z": float(z_primary[r]) if np.isfinite(z_primary[r]) else np.nan,
                "relevant_brightness_z": float(relevant_z[r]),
                "robust_K": int(robust_run[r]),
                "raw_K": int(raw_run[r]),
                "median_initial_state_confidence": float(run_med_init[r]) if np.isfinite(run_med_init[r]) else np.nan,
                "median_joint_transition_weight": float(run_med_weight[r]) if np.isfinite(run_med_weight[r]) else np.nan,
            }
            event_rows.append(row)
            print(
                f"{r:4d} {primary_k[r]:4d} {z_primary[r]:6.2f} {relevant_z[r]:9.2f} "
                f"{robust_run[r]:7d} {raw_run[r]:4d} {run_med_init[r]:11.3f} {run_med_weight[r]:9.3f}"
            )
        out["directions"][direction]["top_discrete_rows"] = event_rows

    # Compact directional comparison used for the manuscript narrative.
    L = out["directions"]["loss"]
    G = out["directions"]["gain"]
    print("\nV5 DIRECTIONAL TAKEAWAY")
    print("-" * 142)
    print(
        f"Loss high-confidence/robust survival = "
        f"{100*L['total_primary']/max(L['total_robust'],1):.2f}% ; "
        f"Gain = {100*G['total_primary']/max(G['total_robust'],1):.2f}%"
    )
    print(
        f"Initial-state confidence: loss NV- median={L['event_confidence']['median_initial_conf']:.4f}; "
        f"gain initial NV0 median={G['event_confidence']['median_initial_conf']:.4f}."
    )
    print(
        "Use NV- -> NV0 as the primary event channel if it retains high posterior "
        "confidence and its top spikes survive the funnel. Treat NV0 -> NV- as a "
        "classification-sensitive control channel; do not claim every reverse "
        "transition is impossible or artificial."
    )
    return out

# =============================================================================
# CORE ANALYSIS
# =============================================================================



# =============================================================================
# V4 SOFT / PROBABILISTIC CHARGE-STATE HELPERS
# =============================================================================


def _log_gaussian_pdf(x, center, sigma):
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), float(SOFT_SIGMA_FLOOR_COUNTS))
    return -0.5 * ((x - float(center)) / sigma) ** 2 - np.log(sigma)


def _posterior_nvm_from_gaussians(x, mu_m, sig_m, mu_0, sig_0, prior_m):
    """Posterior P(NV-|count) from two state-conditioned Gaussian likelihoods."""
    x = np.asarray(x, dtype=float)
    prior_m = float(np.clip(prior_m, SOFT_PRIOR_MIN, 1.0 - SOFT_PRIOR_MIN))
    lm = _log_gaussian_pdf(x, mu_m, sig_m) + np.log(prior_m)
    l0 = _log_gaussian_pdf(x, mu_0, sig_0) + np.log1p(-prior_m)
    delta = np.clip(l0 - lm, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(delta))
    out[~np.isfinite(x)] = np.nan
    return out


def _crossfit_soft_state_posteriors(c11_corr, c12_corr, thresholds, good):
    """
    Cross-fitted P(NV-|count) for rep11 and rep12.

    For each held-out RUN fold and NV, fit NV-/NV0 centers and widths using only
    other good runs. Confident training samples are defined relative to the saved
    threshold with SOFT_MODEL_TRAIN_MARGIN_COUNTS. Rep-specific state priors are
    estimated on the training runs. Thus a rare candidate cannot sharpen its own
    state model or prior.
    """
    c11 = np.asarray(c11_corr, dtype=float)
    c12 = np.asarray(c12_corr, dtype=float)
    thr = np.asarray(thresholds, dtype=float).ravel()
    good = np.asarray(good, dtype=bool)
    n_nv, n_run = c11.shape

    p11 = np.full((n_nv, n_run), np.nan, dtype=np.float32)
    p12 = np.full((n_nv, n_run), np.nan, dtype=np.float32)
    fold_id = np.full(n_run, -1, dtype=int)

    rng = np.random.default_rng(int(SOFT_RANDOM_SEED))
    good_runs = np.where(good)[0]
    shuffled = good_runs.copy()
    rng.shuffle(shuffled)
    fold_id[shuffled] = np.arange(shuffled.size) % int(max(2, SOFT_NUM_RUN_FOLDS))

    # Diagnostics aggregated over fold fits.
    fit_valid = np.zeros((n_nv, int(max(2, SOFT_NUM_RUN_FOLDS))), dtype=bool)
    sep_sigma = np.full_like(fit_valid, np.nan, dtype=float)
    n_train_m = np.zeros_like(fit_valid, dtype=int)
    n_train_0 = np.zeros_like(fit_valid, dtype=int)

    margin = float(SOFT_MODEL_TRAIN_MARGIN_COUNTS)
    alpha = float(SOFT_PRIOR_ALPHA)

    for f in range(fit_valid.shape[1]):
        test = good & (fold_id == f)
        train = good & (fold_id != f)
        if not np.any(test) or not np.any(train):
            continue

        for i in range(n_nv):
            t = float(thr[i])
            if not np.isfinite(t):
                continue

            v11 = c11[i, train]
            v12 = c12[i, train]
            vm = np.concatenate([
                v11[np.isfinite(v11) & (v11 > t + margin)],
                v12[np.isfinite(v12) & (v12 > t + margin)],
            ])
            v0 = np.concatenate([
                v11[np.isfinite(v11) & (v11 <= t - margin)],
                v12[np.isfinite(v12) & (v12 <= t - margin)],
            ])
            n_train_m[i, f] = int(vm.size)
            n_train_0[i, f] = int(v0.size)

            # Robust centers/scales; permit a small-sample fallback but reject
            # truly unconstrained state models.
            if vm.size >= int(SOFT_MODEL_MIN_STATE_SAMPLES):
                mu_m, sig_m = _robust_center_scale(vm)
            elif vm.size >= int(SOFT_MODEL_MIN_FALLBACK_SAMPLES):
                mu_m = float(np.nanmedian(vm))
                sig_m = float(np.nanstd(vm, ddof=1)) if vm.size > 1 else np.nan
            else:
                continue

            if v0.size >= int(SOFT_MODEL_MIN_STATE_SAMPLES):
                mu_0, sig_0 = _robust_center_scale(v0)
            elif v0.size >= int(SOFT_MODEL_MIN_FALLBACK_SAMPLES):
                mu_0 = float(np.nanmedian(v0))
                sig_0 = float(np.nanstd(v0, ddof=1)) if v0.size > 1 else np.nan
            else:
                continue

            if not (np.isfinite(mu_m) and np.isfinite(mu_0) and np.isfinite(sig_m) and np.isfinite(sig_0)):
                continue
            sig_m = max(float(sig_m), float(SOFT_SIGMA_FLOOR_COUNTS))
            sig_0 = max(float(sig_0), float(SOFT_SIGMA_FLOOR_COUNTS))
            if mu_m <= mu_0:
                # Pathological/unresolved state model; do not invent a posterior.
                continue

            # Rep-specific priors from training data. Using all thresholded finite
            # samples here gives a stable prior while the likelihood parameters are
            # trained only on well-separated points.
            fin11 = np.isfinite(v11)
            fin12 = np.isfinite(v12)
            nm11 = int(np.sum(fin11 & (v11 > t)))
            n011 = int(np.sum(fin11 & (v11 <= t)))
            nm12 = int(np.sum(fin12 & (v12 > t)))
            n012 = int(np.sum(fin12 & (v12 <= t)))
            prior11 = (nm11 + alpha) / (nm11 + n011 + 2.0 * alpha) if (nm11 + n011) else 0.5
            prior12 = (nm12 + alpha) / (nm12 + n012 + 2.0 * alpha) if (nm12 + n012) else 0.5

            inds = np.where(test)[0]
            p11[i, inds] = _posterior_nvm_from_gaussians(
                c11[i, inds], mu_m, sig_m, mu_0, sig_0, prior11
            ).astype(np.float32)
            p12[i, inds] = _posterior_nvm_from_gaussians(
                c12[i, inds], mu_m, sig_m, mu_0, sig_0, prior12
            ).astype(np.float32)

            fit_valid[i, f] = True
            sep_sigma[i, f] = (mu_m - mu_0) / np.sqrt(0.5 * (sig_m**2 + sig_0**2))

    return {
        "p_nvm_rep11": p11,
        "p_nvm_rep12": p12,
        "fold_id": fold_id,
        "fit_valid": fit_valid,
        "state_separation_sigma": sep_sigma,
        "n_train_nvm": n_train_m,
        "n_train_nv0": n_train_0,
    }


def _soft_transition_weights(post, good):
    p11 = np.asarray(post["p_nvm_rep11"], dtype=float)
    p12 = np.asarray(post["p_nvm_rep12"], dtype=float)
    good = np.asarray(good, dtype=bool)
    finite = np.isfinite(p11) & np.isfinite(p12)

    coverage_nv = np.mean(finite[:, good], axis=1) if np.any(good) else np.zeros(p11.shape[0])
    valid_nv = coverage_nv >= float(SOFT_NV_MIN_GOOD_RUN_COVERAGE)
    finite &= valid_nv[:, None]

    loss = p11 * (1.0 - p12)
    gain = (1.0 - p11) * p12
    loss[~finite] = np.nan
    gain[~finite] = np.nan

    n_valid_nv = int(np.sum(valid_nv))
    valid_count_run = np.sum(finite, axis=0)
    if n_valid_nv > 0:
        coverage_run = valid_count_run / float(n_valid_nv)
    else:
        coverage_run = np.zeros(p11.shape[1], dtype=float)
    good_soft = good & (coverage_run >= float(SOFT_RUN_MIN_VALID_NV_FRACTION))

    return {
        "loss": loss.astype(np.float32),
        "gain": gain.astype(np.float32),
        "valid_nv_mask": valid_nv,
        "nv_good_run_coverage": coverage_nv,
        "run_valid_nv_fraction": coverage_run,
        "good_soft_run_mask": good_soft,
    }


def _soft_preliminary_background(weights, good):
    score = np.nansum(np.asarray(weights, dtype=float), axis=0)
    z, med, sig = _robust_z_from_good(score, good)
    bg = np.asarray(good, dtype=bool) & np.isfinite(z) & (np.abs(z) < float(SOFT_BACKGROUND_MAX_ABS_Z))
    if np.sum(bg) < float(SOFT_BACKGROUND_MIN_FRACTION) * np.sum(good):
        bg = np.asarray(good, dtype=bool).copy()
    return score, z, bg, med, sig


def _independent_nv_soft_null(weights, background, n_samples, seed):
    """
    Monte-Carlo null for the SUM of soft transition weights.

    Each NV independently draws one of its own background-run weights. This
    preserves each detector's empirical ambiguity/transition distribution while
    destroying same-run correlations across NVs.
    """
    w = np.asarray(weights, dtype=float)
    background = np.asarray(background, dtype=bool)
    rng = np.random.default_rng(int(seed))
    n_samples = int(n_samples)
    null_sum = np.zeros(n_samples, dtype=np.float64)
    used = 0
    for i in range(w.shape[0]):
        vals = w[i, background]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        inds = rng.integers(0, vals.size, size=n_samples)
        null_sum += vals[inds]
        used += 1
    return null_sum, used


def _empirical_upper_tail_p(values, null_samples):
    values = np.asarray(values, dtype=float)
    null_sorted = np.sort(np.asarray(null_samples, dtype=float))
    n = null_sorted.size
    p = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if n == 0 or not np.any(finite):
        return p
    pos = np.searchsorted(null_sorted, values[finite], side="left")
    ge = n - pos
    p[finite] = (ge + 1.0) / (n + 1.0)
    return p


def _soft_direction_null(weights, good, direction, seed_offset=0):
    score, prelim_z, background, _, _ = _soft_preliminary_background(weights, good)
    null_samples, n_nv_used = _independent_nv_soft_null(
        weights,
        background,
        SOFT_NULL_MONTE_CARLO_SAMPLES,
        int(SOFT_RANDOM_SEED) + int(seed_offset),
    )
    p = _empirical_upper_tail_p(score, null_samples)
    tested = np.asarray(good, dtype=bool) & np.isfinite(p)
    q = _bh_qvalues(p, tested)
    null_mean = float(np.mean(null_samples)) if null_samples.size else np.nan
    null_std = float(np.std(null_samples, ddof=1)) if null_samples.size > 1 else np.nan
    z = np.full(score.shape, np.nan, dtype=float)
    if np.isfinite(null_std) and null_std > 0:
        z[tested] = (score[tested] - null_mean) / null_std
    return {
        "direction": direction,
        "score": score,
        "preliminary_z": prelim_z,
        "background_run_mask": background,
        "null_samples": null_samples,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_mc": p,
        "q_direction": q,
        "z_vs_null": z,
        "n_nv_used": int(n_nv_used),
        "n_null_samples": int(null_samples.size),
    }


def _combined_soft_fdr(loss_result, gain_result, good):
    p_loss = np.asarray(loss_result["p_mc"], dtype=float)
    p_gain = np.asarray(gain_result["p_mc"], dtype=float)
    good = np.asarray(good, dtype=bool)
    n = good.size
    combined = np.concatenate([p_loss, p_gain])
    tested = np.concatenate([good & np.isfinite(p_loss), good & np.isfinite(p_gain)])
    q = _bh_qvalues(combined, tested)
    return q[:n], q[n:]


def _soft_event_weight_summary(weights, hard_switch, good):
    mask = np.asarray(hard_switch, dtype=bool) & np.asarray(good, dtype=bool)[None, :]
    vals = np.asarray(weights, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    out = {
        "n": int(vals.size),
        "median": float(np.median(vals)) if vals.size else np.nan,
        "mean": float(np.mean(vals)) if vals.size else np.nan,
    }
    for cut in SOFT_EVENT_WEIGHT_CUTS:
        out[f"fraction_ge_{cut:g}"] = float(np.mean(vals >= float(cut))) if vals.size else np.nan
    return out


def _build_v4_candidate_rows(label, dark_wait_s, direction, soft_result, q_combined,
                             weights, hard_switch, v3_fdr, good):
    good = np.asarray(good, dtype=bool)
    p = np.asarray(soft_result["p_mc"], dtype=float)
    order = np.where(good & np.isfinite(p))[0]
    order = order[np.argsort(p[order])]
    rows = []
    hard_switch = np.asarray(hard_switch, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    hard_q = np.asarray(v3_fdr.get("q_direction", np.full(good.size, np.nan)), dtype=float)
    for r in order[: min(int(SOFT_TOP_RUNS_TO_PRINT), order.size)]:
        ev = hard_switch[:, r]
        evw = weights[ev, r]
        evw = evw[np.isfinite(evw)]
        rows.append({
            "dataset": label,
            "dark_wait_s": float(dark_wait_s),
            "direction": direction,
            "run": int(r),
            "soft_score": float(soft_result["score"][r]),
            "soft_null_mean": float(soft_result["null_mean"]),
            "soft_null_std": float(soft_result["null_std"]),
            "soft_excess": float(soft_result["score"][r] - soft_result["null_mean"]),
            "soft_z_vs_null": float(soft_result["z_vs_null"][r]),
            "soft_p_mc": float(soft_result["p_mc"][r]),
            "soft_q_direction": float(soft_result["q_direction"][r]),
            "soft_q_loss_gain_family": float(q_combined[r]),
            "hard_robust_K": int(np.sum(ev)),
            "hard_q_direction_v3": float(hard_q[r]) if np.isfinite(hard_q[r]) else np.nan,
            "hard_event_weight_sum": float(np.sum(evw)) if evw.size else 0.0,
            "hard_event_weight_mean": float(np.mean(evw)) if evw.size else np.nan,
            "hard_event_weight_median": float(np.median(evw)) if evw.size else np.nan,
            "hard_event_fraction_weight_ge_0p5": float(np.mean(evw >= 0.5)) if evw.size else np.nan,
            "hard_event_fraction_weight_ge_0p8": float(np.mean(evw >= 0.8)) if evw.size else np.nan,
        })
    return rows


def _write_v4_event_audit(path, label, dark_wait_s, weights_by_direction,
                          truth_masks, soft_results, q_combined, good):
    fields = [
        "dataset", "dark_wait_s", "direction", "run", "nv",
        "soft_transition_weight", "run_soft_score", "run_soft_excess",
        "run_soft_z", "run_soft_p_mc", "run_soft_q_direction",
        "run_soft_q_loss_gain_family",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for direction in ("loss", "gain"):
            hard = np.asarray(truth_masks[f"{direction}_switch"], dtype=bool)
            hard &= np.asarray(good, dtype=bool)[None, :]
            sw = np.asarray(weights_by_direction[direction], dtype=float)
            sr = soft_results[direction]
            for nv, run in np.argwhere(hard):
                nv = int(nv); run = int(run)
                w.writerow({
                    "dataset": label,
                    "dark_wait_s": float(dark_wait_s),
                    "direction": direction,
                    "run": run,
                    "nv": nv,
                    "soft_transition_weight": float(sw[nv, run]),
                    "run_soft_score": float(sr["score"][run]),
                    "run_soft_excess": float(sr["score"][run] - sr["null_mean"]),
                    "run_soft_z": float(sr["z_vs_null"][run]),
                    "run_soft_p_mc": float(sr["p_mc"][run]),
                    "run_soft_q_direction": float(sr["q_direction"][run]),
                    "run_soft_q_loss_gain_family": float(q_combined[direction][run]),
                })


def _print_v4_summary(direction, soft_result, event_weight_summary, q_combined, good):
    good = np.asarray(good, dtype=bool)
    score = np.asarray(soft_result["score"], dtype=float)
    qd = np.asarray(soft_result["q_direction"], dtype=float)
    qc = np.asarray(q_combined, dtype=float)
    print(f"\nV4 SOFT {direction.upper()} CHANNEL")
    print("-" * 142)
    print(
        f"cross-fitted posterior score: mean={np.nanmean(score[good]):.4f}; "
        f"independent-NV null mean={soft_result['null_mean']:.4f}, "
        f"std={soft_result['null_std']:.4f}; "
        f"MC null samples={soft_result['n_null_samples']}, NVs used={soft_result['n_nv_used']}"
    )
    print(
        f"hard-robust event posterior weight: median={event_weight_summary['median']:.3f}, "
        f"mean={event_weight_summary['mean']:.3f}, "
        f"fraction >=0.5={100*event_weight_summary.get('fraction_ge_0.5', np.nan):.2f}%, "
        f">=0.8={100*event_weight_summary.get('fraction_ge_0.8', np.nan):.2f}%"
    )
    for alpha in SOFT_FDR_LEVELS:
        nd = int(np.sum(good & np.isfinite(qd) & (qd <= alpha)))
        nc = int(np.sum(good & np.isfinite(qc) & (qc <= alpha)))
        print(f"  soft q<={alpha:g}: {nd} within-direction; {nc} after loss+gain family correction")



def analyze_dataset(dataset):
    label = str(dataset["label"])
    file_stem = str(dataset["file_stem"])

    print("\n" + "=" * 142)
    print(f"TRANSITION TRUTH DIAGNOSTIC: {label}")
    print("=" * 142)

    metadata = _load_metadata_robust(file_stem)
    npz_path = base._discover_npz_path(
        file_stem=file_stem,
        npz_path_override=dataset.get("npz_path_override"),
        metadata=metadata,
    )

    print(f"[counts-only] using NPZ: {npz_path}")
    small = _load_small_dataset_members_tolerant(npz_path, metadata=metadata)

    c11 = np.asarray(small["c11"], dtype=np.float32)
    c12 = np.asarray(small["c12"], dtype=np.float32)
    thresholds = np.asarray(small["thresholds"], dtype=np.float32).ravel()

    n_nv, n_run = c11.shape
    runs = np.arange(n_run, dtype=int)

    if REJECT_GLOBAL_DROP_RUNS:
        quality = base._detect_global_drop_runs(
            c11,
            c12,
            min_total_fraction=base.MIN_RUN_TOTAL_FRACTION,
            per_nv_collapse_fraction=base.PER_NV_COLLAPSE_FRACTION,
            max_collapsed_nv_fraction=base.MAX_COLLAPSED_NV_FRACTION,
        )
        good = np.asarray(quality["good_run_mask"], dtype=bool)
    else:
        good = np.ones(n_run, dtype=bool)

    dark_wait_s = _parse_dark_wait_s(dataset, metadata)

    print(
        f"NVs={n_nv}, runs={n_run}, good={np.sum(good)}, "
        f"rejected={np.sum(~good)}, dark_wait_s={dark_wait_s:g}"
    )

    # ------------------------------------------------------------------
    # Raw classification and margin sweep.
    # ------------------------------------------------------------------
    raw0 = _classify(c11, c12, thresholds, margin=0.0)
    raw_strict = _classify(
        c11, c12, thresholds, margin=STRICT_MARGIN_COUNTS
    )

    by_margin = {
        float(m): _classify(c11, c12, thresholds, margin=float(m))
        for m in MARGIN_SWEEP_COUNTS
    }

    # ------------------------------------------------------------------
    # Leave-one-run-out "normally NV-" occupancy.
    # ------------------------------------------------------------------
    occ_loo = _loo_rep11_nvm_occupancy(c11, thresholds, good)

    # All-good occupancy only for fixed reference-NV selection.
    thr = thresholds[:, None]
    finite11 = np.isfinite(c11) & np.isfinite(thr)
    nvm11 = finite11 & (c11 > thr)
    occ_all = _safe_divide(
        np.sum(nvm11[:, good], axis=1),
        np.sum(finite11[:, good], axis=1),
    )

    reference_mask = _build_reference_mask(
        c11, thresholds, good, occ_all
    )
    print(f"stable common-mode references: {np.sum(reference_mask)} / {n_nv}")

    # ------------------------------------------------------------------
    # Cross-fitted run-level correction, separately for rep11 and rep12.
    # ------------------------------------------------------------------
    cm11 = _crossfit_common_mode(
        c11, good, reference_mask, num_folds=NUM_CROSSFIT_FOLDS
    )
    cm12 = _crossfit_common_mode(
        c12, good, reference_mask, num_folds=NUM_CROSSFIT_FOLDS
    )

    add0 = _classify(
        cm11["corrected_additive"],
        cm12["corrected_additive"],
        thresholds,
        margin=0.0,
    )
    add_strict = _classify(
        cm11["corrected_additive"],
        cm12["corrected_additive"],
        thresholds,
        margin=STRICT_MARGIN_COUNTS,
    )
    mult0 = _classify(
        cm11["corrected_multiplicative"],
        cm12["corrected_multiplicative"],
        thresholds,
        margin=0.0,
    )
    mult_strict = _classify(
        cm11["corrected_multiplicative"],
        cm12["corrected_multiplicative"],
        thresholds,
        margin=STRICT_MARGIN_COUNTS,
    )

    verdict = _event_verdict_masks(
        c11=c11,
        c12=c12,
        thresholds=thresholds,
        raw0=raw0,
        raw_strict=raw_strict,
        add_strict=add_strict,
        mult_strict=mult_strict,
        add0=add0,
        mult0=mult0,
        occ_loo=occ_loo,
    )

    raw_gain = raw0["gain_count"]
    raw_loss = raw0["loss_count"]
    robust_gain = _count(verdict["robust_gain"])
    mis_gain = _count(verdict["rep11_misclass_gain"])
    amb_gain = _count(verdict["ambiguous_gain"])
    sensitive_gain = _count(verdict["gain_sensitive"])

    robust_loss = _count(verdict["robust_loss"])
    mis_loss = _count(verdict["rep12_misclass_loss"])
    amb_loss = _count(verdict["ambiguous_loss"])
    sensitive_loss = _count(verdict["loss_sensitive"])

    # ------------------------------------------------------------------
    # Run-level brightness summaries.  The cross-fit correction varies very
    # slightly by target fold, so use the median across folds as a run-level
    # diagnostic only.
    # ------------------------------------------------------------------
    run_scale11 = np.nanmedian(cm11["fold_multiplicative_scale"], axis=0)
    run_scale12 = np.nanmedian(cm12["fold_multiplicative_scale"], axis=0)
    run_delta11 = np.nanmedian(cm11["fold_additive_shift"], axis=0)
    run_delta12 = np.nanmedian(cm12["fold_additive_shift"], axis=0)

    scale11_z, _, _ = _robust_z_from_good(run_scale11, good)
    scale12_z, _, _ = _robust_z_from_good(run_scale12, good)
    relscale_z, _, _ = _robust_z_from_good(run_scale12 - run_scale11, good)

    # ------------------------------------------------------------------
    # Summaries.
    # ------------------------------------------------------------------
    good_n = int(np.sum(good))
    total_raw_gain = int(np.sum(raw_gain[good]))
    total_raw_loss = int(np.sum(raw_loss[good]))
    total_robust_gain = int(np.sum(robust_gain[good]))
    total_mis_gain = int(np.sum(mis_gain[good]))
    total_robust_loss = int(np.sum(robust_loss[good]))
    total_mis_loss = int(np.sum(mis_loss[good]))

    print("\nGAIN TRUTH CHECK: apparent NV0 -> NV-")
    print("-" * 142)
    print(f"raw gains:                       {total_raw_gain}")
    print(
        f"robust gain candidates:          {total_robust_gain} "
        f"({100*total_robust_gain/max(total_raw_gain,1):.2f}% of raw gains)"
    )
    print(
        f"likely rep11 misclassification:  {total_mis_gain} "
        f"({100*total_mis_gain/max(total_raw_gain,1):.2f}% of raw gains)"
    )
    print(
        f"correction-sensitive gains:      {int(np.sum(sensitive_gain[good]))} "
        f"({100*np.sum(sensitive_gain[good])/max(total_raw_gain,1):.2f}% of raw gains)"
    )

    print("\nLOSS TRUTH CHECK: apparent NV- -> NV0")
    print("-" * 142)
    print(f"raw losses:                      {total_raw_loss}")
    print(
        f"robust loss candidates:          {total_robust_loss} "
        f"({100*total_robust_loss/max(total_raw_loss,1):.2f}% of raw losses)"
    )
    print(
        f"likely rep12 misclassification:  {total_mis_loss} "
        f"({100*total_mis_loss/max(total_raw_loss,1):.2f}% of raw losses)"
    )
    print(
        f"correction-sensitive losses:     {int(np.sum(sensitive_loss[good]))} "
        f"({100*np.sum(sensitive_loss[good])/max(total_raw_loss,1):.2f}% of raw losses)"
    )

    rho_rg, p_rg = _corr(raw_gain, scale11_z, good)
    rho_mg, p_mg = _corr(mis_gain, scale11_z, good)
    rho_rl, p_rl = _corr(raw_loss, scale12_z, good)
    rho_ml, p_ml = _corr(mis_loss, scale12_z, good)
    rho_rob_l, p_rob_l = _corr(robust_loss, scale12_z, good)

    print("\nBRIGHTNESS COUPLING")
    print("-" * 142)
    print(f"rho(raw gain, rep11 scale z)       = {rho_rg:+.4f}  p={p_rg:.3e}")
    print(f"rho(rep11-misclass gain, scale z)  = {rho_mg:+.4f}  p={p_mg:.3e}")
    print(f"rho(raw loss, rep12 scale z)       = {rho_rl:+.4f}  p={p_rl:.3e}")
    print(f"rho(rep12-misclass loss, scale z)  = {rho_ml:+.4f}  p={p_ml:.3e}")
    print(f"rho(ROBUST loss, rep12 scale z)    = {rho_rob_l:+.4f}  p={p_rob_l:.3e}")

    # ------------------------------------------------------------------
    # NEXT ANALYSIS: truth-tested tail statistics in BOTH directions.
    # ------------------------------------------------------------------
    truth_masks = _truth_consensus_masks(
        raw_strict,
        add_strict,
        mult_strict,
        verdict,
    )

    robust_tail = {}
    exact_tail_rows = {}
    if RUN_ROBUST_TAIL_ANALYSIS:
        for direction in ("loss", "gain"):
            switch = truth_masks[f"{direction}_switch"]
            evaluable = truth_masks[f"{direction}_evaluable"]
            null_result = _heterogeneous_run_null(switch, evaluable, good)
            exact_rows = _top_exact_tail_rows(
                null_result,
                evaluable,
                good,
                TOP_EXACT_POIBIN_RUNS,
            )
            robust_tail[direction] = null_result
            exact_tail_rows[direction] = exact_rows
            _print_direction_tail_summary(direction, null_result, exact_rows)

    # ------------------------------------------------------------------
    # V3: discriminate residual rep11/rep12 artifacts from real transitions.
    # ------------------------------------------------------------------
    v3 = {
        "noise": None,
        "depth": {},
        "local_by_radius": {},
        "fdr": {},
        "q_combined": {},
        "candidate_rows": [],
    }
    if RUN_V3_DISCRIMINATION and "loss" in robust_tail and "gain" in robust_tail:
        # State-conditioned noise model and threshold depth use the cross-fitted
        # multiplicative correction, which was the best common-mode normalizer
        # in the earlier diagnostic.
        noise = _state_conditioned_noise_model(
            cm11["corrected_multiplicative"],
            cm12["corrected_multiplicative"],
            thresholds,
            good,
        )
        v3["noise"] = noise

        for direction in ("loss", "gain"):
            v3["depth"][direction] = _event_depth_sigma(
                direction,
                truth_masks[f"{direction}_switch"],
                cm11["corrected_multiplicative"],
                cm12["corrected_multiplicative"],
                thresholds,
                noise,
            )
            v3["fdr"][direction] = _screened_exact_pb_fdr(
                robust_tail[direction],
                truth_masks[f"{direction}_evaluable"],
                good,
            )

        qloss_comb, qgain_comb = _combined_direction_fdr(
            v3["fdr"]["loss"], v3["fdr"]["gain"], good
        )
        v3["q_combined"] = {
            "loss": qloss_comb,
            "gain": qgain_comb,
        }

        # Local rep11/rep12 residual brightness around every target location.
        coords_um = _coords_um_optional(small)
        if coords_um is not None:
            for radius in LOCAL_BRIGHTNESS_RADII_UM:
                radius = float(radius)
                print(
                    f"[local-brightness] computing {radius:g} um neighborhood "
                    "diagnostic for rep11/rep12...",
                    flush=True,
                )
                v3["local_by_radius"][radius] = {
                    "rep11": _local_reference_brightness(
                        c11, good, reference_mask, coords_um, run_scale11, radius
                    ),
                    "rep12": _local_reference_brightness(
                        c12, good, reference_mask, coords_um, run_scale12, radius
                    ),
                }
        else:
            print(
                "[local-brightness] skipped because nv_list/camera coordinates "
                "are unavailable.",
                flush=True,
            )

        primary_local = v3["local_by_radius"].get(
            float(LOCAL_BRIGHTNESS_PRIMARY_RADIUS_UM), None
        )
        for direction in ("loss", "gain"):
            _print_v3_discrimination_summary(
                direction=direction,
                switch_mask=truth_masks[f"{direction}_switch"],
                good=good,
                occ_loo=occ_loo,
                depth=v3["depth"][direction],
                local_primary=primary_local,
                fdr=v3["fdr"][direction],
                q_combined=v3["q_combined"][direction],
            )
            rows = _build_candidate_audit_rows(
                label=label,
                dark_wait_s=dark_wait_s,
                direction=direction,
                switch_mask=truth_masks[f"{direction}_switch"],
                null_result=robust_tail[direction],
                fdr_result=v3["fdr"][direction],
                q_combined=v3["q_combined"][direction],
                occ_loo=occ_loo,
                depth=v3["depth"][direction],
                local_by_radius=v3["local_by_radius"],
                good=good,
            )
            v3["candidate_rows"].extend(rows)

        print("\nV3 TOP CANDIDATE RUNS AFTER EXACT-PB + FDR + ARTIFACT AUDIT")
        print("-" * 142)
        print(
            "dir   run   K    Z      p_exact      q_dir      q_both   "
            "medHist  fracHist95  minDepthSig  localZ20"
        )
        for row in sorted(v3["candidate_rows"], key=lambda x: x["p_exact_screened"])[
            : min(TOP_FDR_RUNS_TO_PRINT, len(v3["candidate_rows"]))
        ]:
            print(
                f"{row['direction']:>4s} {row['run']:5d} {row['K_robust']:3d} "
                f"{row['z_approx']:6.2f} {row['p_exact_screened']:12.3e} "
                f"{row['q_direction']:10.3e} {row['q_loss_gain_family']:10.3e} "
                f"{row['median_loo_rep11_nvm_occupancy']:8.3f} "
                f"{row['fraction_event_nvs_normally_nvm_ge95']:10.3f} "
                f"{row['median_min_depth_sigma']:11.3f} "
                f"{row.get('median_relevant_local_z_20um', np.nan):9.3f}"
            )

    # ------------------------------------------------------------------
    # V4: cross-fitted probabilistic charge-state classifier.
    # ------------------------------------------------------------------
    v4 = {
        "posteriors": None,
        "weights": {},
        "soft_results": {},
        "q_combined": {},
        "event_weight_summary": {},
        "candidate_rows": [],
    }
    if RUN_V4_SOFT_CLASSIFIER:
        print(
            "\n[V4-soft] fitting cross-fitted NV-/NV0 likelihoods on "
            "multiplicatively corrected counts...",
            flush=True,
        )
        post = _crossfit_soft_state_posteriors(
            cm11["corrected_multiplicative"],
            cm12["corrected_multiplicative"],
            thresholds,
            good,
        )
        weights = _soft_transition_weights(post, good)
        v4["posteriors"] = post
        v4["weights"] = {"loss": weights["loss"], "gain": weights["gain"]}
        v4["valid_nv_mask"] = weights["valid_nv_mask"]
        v4["nv_good_run_coverage"] = weights["nv_good_run_coverage"]
        v4["run_valid_nv_fraction"] = weights["run_valid_nv_fraction"]
        soft_good = weights["good_soft_run_mask"]
        v4["good_run_mask"] = soft_good
        print(
            f"[V4-soft] posterior coverage: {np.sum(weights['valid_nv_mask'])}/{n_nv} NVs "
            f"meet >= {100*SOFT_NV_MIN_GOOD_RUN_COVERAGE:.1f}% good-run coverage; "
            f"{np.sum(soft_good)}/{np.sum(good)} good runs meet >= "
            f"{100*SOFT_RUN_MIN_VALID_NV_FRACTION:.1f}% valid-NV coverage",
            flush=True,
        )

        soft_loss = _soft_direction_null(weights["loss"], soft_good, "loss", seed_offset=11)
        soft_gain = _soft_direction_null(weights["gain"], soft_good, "gain", seed_offset=29)
        v4["soft_results"] = {"loss": soft_loss, "gain": soft_gain}
        qloss_soft, qgain_soft = _combined_soft_fdr(soft_loss, soft_gain, soft_good)
        v4["q_combined"] = {"loss": qloss_soft, "gain": qgain_soft}

        for direction in ("loss", "gain"):
            es = _soft_event_weight_summary(
                weights[direction],
                truth_masks[f"{direction}_switch"],
                soft_good,
            )
            v4["event_weight_summary"][direction] = es
            _print_v4_summary(
                direction,
                v4["soft_results"][direction],
                es,
                v4["q_combined"][direction],
                soft_good,
            )
            v3fdr = v3.get("fdr", {}).get(direction, {})
            v4["candidate_rows"].extend(
                _build_v4_candidate_rows(
                    label,
                    dark_wait_s,
                    direction,
                    v4["soft_results"][direction],
                    v4["q_combined"][direction],
                    weights[direction],
                    truth_masks[f"{direction}_switch"],
                    v3fdr,
                    soft_good,
                )
            )

        print("\nV4 TOP RUNS: HARD CANDIDATE -> SOFT POSTERIOR AUDIT")
        print("-" * 142)
        print(
            "dir   run hardK  softS excess   zSoft     pMC        qBoth   "
            "hardWsum  medHardW  fracW>=.5"
        )
        for row in sorted(v4["candidate_rows"], key=lambda x: x["soft_p_mc"])[
            : min(int(SOFT_TOP_RUNS_TO_PRINT), len(v4["candidate_rows"]))
        ]:
            print(
                f"{row['direction']:>4s} {row['run']:5d} {row['hard_robust_K']:5d} "
                f"{row['soft_score']:6.2f} {row['soft_excess']:6.2f} "
                f"{row['soft_z_vs_null']:7.2f} {row['soft_p_mc']:10.3e} "
                f"{row['soft_q_loss_gain_family']:10.3e} "
                f"{row['hard_event_weight_sum']:8.2f} "
                f"{row['hard_event_weight_median']:9.3f} "
                f"{row['hard_event_fraction_weight_ge_0p5']:10.3f}"
            )

        if SAVE_OUTPUTS:
            safe_label_v4 = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            candidate_csv = OUTPUT_DIR / f"{safe_label_v4}_v4_soft_candidate_run_audit.csv"
            _write_dict_rows_csv(candidate_csv, v4["candidate_rows"])
            print(f"Saved V4 soft candidate audit: {candidate_csv}")
            if SAVE_V4_EVENT_AUDIT_CSV:
                event_csv = OUTPUT_DIR / f"{safe_label_v4}_v4_soft_robust_event_weights.csv"
                _write_v4_event_audit(
                    event_csv,
                    label,
                    dark_wait_s,
                    weights,
                    truth_masks,
                    v4["soft_results"],
                    v4["q_combined"],
                    soft_good,
                )
                print(f"Saved V4 robust-event soft weights: {event_csv}")

    # ------------------------------------------------------------------
    # V5 focused directionality: discrete high-confidence transitions only.
    # ------------------------------------------------------------------
    v5 = {}
    if RUN_V5_FOCUSED_DIRECTIONALITY and RUN_V4_SOFT_CLASSIFIER and v4.get("posteriors") is not None:
        v5 = _run_v5_focused(
            label=label,
            dark_wait_s=dark_wait_s,
            post=v4["posteriors"],
            weights_by_direction=v4["weights"],
            truth_masks=truth_masks,
            raw_gain=raw_gain,
            raw_loss=raw_loss,
            robust_gain=robust_gain,
            robust_loss=robust_loss,
            scale11_z=scale11_z,
            scale12_z=scale12_z,
            good=v4.get("good_run_mask", good),
        )
        if SAVE_OUTPUTS and SAVE_V5_CSV:
            safe_label_v5 = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
            for direction in ("loss", "gain"):
                spike_path = OUTPUT_DIR / f"{safe_label_v5}_v5_{direction}_top_raw_spike_funnel.csv"
                event_path = OUTPUT_DIR / f"{safe_label_v5}_v5_{direction}_top_high_conf_discrete_runs.csv"
                _write_dict_rows_csv(spike_path, v5["directions"][direction]["top_raw_spike_rows"])
                _write_dict_rows_csv(event_path, v5["directions"][direction]["top_discrete_rows"])
                print(f"Saved V5 {direction} spike funnel: {spike_path}")
                print(f"Saved V5 {direction} high-confidence runs: {event_path}")

    # ------------------------------------------------------------------
    # NEXT ANALYSIS: weighted exact same-K spatial null on ROBUST masks.
    # Run both directions so NV0 -> NV- is a simultaneous internal control.
    # ------------------------------------------------------------------
    robust_spatial = {}
    if RUN_ROBUST_V23_SPATIAL:
        print(
            f"\n[robust-v23] exploratory run with "
            f"{ROBUST_V23_NULL_DATASETS} null data sets/direction; "
            "use 250--500 only after the robust signal survives."
        )
        for direction in ROBUST_V23_DIRECTIONS:
            if direction not in ("loss", "gain"):
                continue
            if direction not in robust_tail:
                switch = truth_masks[f"{direction}_switch"]
                evaluable = truth_masks[f"{direction}_evaluable"]
                robust_tail[direction] = _heterogeneous_run_null(
                    switch, evaluable, good
                )
            spatial = _run_weighted_same_k_spatial(
                direction=direction,
                switch_mask=truth_masks[f"{direction}_switch"],
                evaluable_mask=truth_masks[f"{direction}_evaluable"],
                good=good,
                p_background=robust_tail[direction]["p_background"],
                small=small,
                dark_wait_s=dark_wait_s,
            )
            robust_spatial[direction] = spatial
            _print_v23_summary(direction, spatial)

    # ------------------------------------------------------------------
    # Top-run tables.
    # ------------------------------------------------------------------
    good_inds = np.where(good)[0]

    print("\nTOP RAW GAIN RUNS: IS THE LOW REP11 POPULATION JUST MISCLASSIFICATION?")
    print("-" * 142)
    print(
        "run  rawG  robustG  rep11MisG  ambG  sensitiveG  "
        "N-_init  rep11_z  rep12_z  dScale_z"
    )
    order = good_inds[np.argsort(raw_gain[good_inds])[::-1]]
    for r in order[: min(TOP_RUNS_TO_PRINT, order.size)]:
        print(
            f"{r:4d} {raw_gain[r]:5d} {robust_gain[r]:8d} {mis_gain[r]:10d} "
            f"{amb_gain[r]:5d} {sensitive_gain[r]:10d} "
            f"{raw0['initial_nvm_count'][r]:7d} "
            f"{scale11_z[r]:8.2f} {scale12_z[r]:8.2f} {relscale_z[r]:9.2f}"
        )

    print("\nTOP RAW LOSS RUNS: HOW MANY NV- -> NV0 TRANSITIONS SURVIVE?")
    print("-" * 142)
    print(
        "run  rawL  robustL  rep12MisL  ambL  sensitiveL  "
        "strictRawL  rep11_z  rep12_z  dScale_z"
    )
    order = good_inds[np.argsort(raw_loss[good_inds])[::-1]]
    for r in order[: min(TOP_RUNS_TO_PRINT, order.size)]:
        strict_raw_l = raw_strict["loss_count"][r]
        print(
            f"{r:4d} {raw_loss[r]:5d} {robust_loss[r]:8d} {mis_loss[r]:10d} "
            f"{amb_loss[r]:5d} {sensitive_loss[r]:10d} "
            f"{strict_raw_l:10d} "
            f"{scale11_z[r]:8.2f} {scale12_z[r]:8.2f} {relscale_z[r]:9.2f}"
        )

    # ------------------------------------------------------------------
    # Run-level CSV.
    # ------------------------------------------------------------------
    run_rows = []
    for r in range(n_run):
        run_rows.append(
            {
                "dataset": label,
                "dark_wait_s": dark_wait_s,
                "run": int(r),
                "good_run": bool(good[r]),
                "initial_nvm_raw": int(raw0["initial_nvm_count"][r]),
                "initial_nv0_raw": int(raw0["initial_nv0_count"][r]),
                "raw_gain": int(raw_gain[r]),
                "robust_gain": int(robust_gain[r]),
                "likely_rep11_misclass_gain": int(mis_gain[r]),
                "ambiguous_gain": int(amb_gain[r]),
                "correction_sensitive_gain": int(sensitive_gain[r]),
                "raw_loss": int(raw_loss[r]),
                "raw_strict_loss": int(raw_strict["loss_count"][r]),
                "robust_loss": int(robust_loss[r]),
                "likely_rep12_misclass_loss": int(mis_loss[r]),
                "ambiguous_loss": int(amb_loss[r]),
                "correction_sensitive_loss": int(sensitive_loss[r]),
                "rep11_scale": float(run_scale11[r]),
                "rep12_scale": float(run_scale12[r]),
                "rep11_scale_z": float(scale11_z[r]),
                "rep12_scale_z": float(scale12_z[r]),
                "relative_scale_z": float(relscale_z[r]),
                "rep11_additive_shift": float(run_delta11[r]),
                "rep12_additive_shift": float(run_delta12[r]),
            }
        )

    safe_label = label.replace(" ", "_").replace("/", "_")
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run_csv = OUTPUT_DIR / f"{safe_label}_run_truth_table.csv"
        _write_run_csv(run_csv, run_rows)
        print(f"\nSaved run truth table: {run_csv}")

        if SAVE_EVENT_LEVEL_CSV:
            event_csv = OUTPUT_DIR / f"{safe_label}_event_truth_table.csv"
            _write_event_csv(
                event_csv,
                label=label,
                good=good,
                c11=c11,
                c12=c12,
                thresholds=thresholds,
                raw0=raw0,
                raw_strict=raw_strict,
                cm11=cm11,
                cm12=cm12,
                add0=add0,
                mult0=mult0,
                occ_loo=occ_loo,
                verdict=verdict,
                b11_z=scale11_z,
                b12_z=scale12_z,
            )
            print(f"Saved event truth table: {event_csv}")

        if RUN_V3_DISCRIMINATION and v3["depth"]:
            candidate_csv = OUTPUT_DIR / f"{safe_label}_v3_candidate_run_audit.csv"
            _write_dict_rows_csv(candidate_csv, v3["candidate_rows"])
            print(f"Saved V3 candidate run audit: {candidate_csv}")

            if SAVE_V3_EVENT_AUDIT_CSV:
                event_audit_csv = OUTPUT_DIR / f"{safe_label}_v3_robust_event_audit.csv"
                _write_robust_event_audit_csv(
                    event_audit_csv,
                    label=label,
                    dark_wait_s=dark_wait_s,
                    truth_masks=truth_masks,
                    good=good,
                    occ_loo=occ_loo,
                    depth_by_direction=v3["depth"],
                    local_by_radius=v3["local_by_radius"],
                    robust_tail=robust_tail,
                    fdr_by_direction=v3["fdr"],
                    q_combined_by_direction=v3["q_combined"],
                )
                print(f"Saved V3 robust-event audit: {event_audit_csv}")

            if SAVE_V3_NV_QUALITY_CSV:
                nv_quality_csv = OUTPUT_DIR / f"{safe_label}_v3_per_nv_detector_quality.csv"
                _write_dict_rows_csv(
                    nv_quality_csv,
                    _per_nv_quality_rows(
                        label=label,
                        occ_all=occ_all,
                        noise=v3["noise"],
                        truth_masks=truth_masks,
                        good=good,
                        local_by_radius=v3["local_by_radius"],
                    ),
                )
                print(f"Saved V3 per-NV quality table: {nv_quality_csv}")

    # ------------------------------------------------------------------
    # FIGURES
    # ------------------------------------------------------------------
    figures = {}

    # V5-A. Compact survival funnel: the manuscript's simplest directionality plot.
    if RUN_V5_FOCUSED_DIRECTIONALITY and v5.get("directions"):
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        dirs = ("loss", "gain")
        x = np.arange(2, dtype=float)
        width = 0.24
        raw_norm = np.ones(2)
        robust_frac = np.asarray([
            v5["directions"][d]["total_robust"] / max(v5["directions"][d]["total_raw"], 1)
            for d in dirs
        ])
        high_frac = np.asarray([
            v5["directions"][d]["total_primary"] / max(v5["directions"][d]["total_raw"], 1)
            for d in dirs
        ])
        ax.bar(x - width, raw_norm, width=width, label="raw (normalized)")
        ax.bar(x, robust_frac, width=width, label="truth-tested / raw")
        ax.bar(x + width, high_frac, width=width,
               label=f"posterior >= {V5_PRIMARY_POSTERIOR_WEIGHT:.2f} / raw")
        ax.set_xticks(x)
        ax.set_xticklabels(["NV- -> NV0", "NV0 -> NV-"])
        ax.set_ylabel("Fraction of raw transitions")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label}: transition-confidence survival funnel")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v5_directional_survival_funnel"] = fig

        # V5-B. Initial-state confidence: directly tests the dim initial NV0 issue.
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        p11 = np.asarray(v4["posteriors"]["p_nvm_rep11"], dtype=float)
        init_loss = p11[truth_masks["loss_switch"] & good[None, :]]
        init_gain = (1.0 - p11)[truth_masks["gain_switch"] & good[None, :]]
        bins = np.linspace(0.0, 1.0, 61)
        init_loss = init_loss[np.isfinite(init_loss)]
        init_gain = init_gain[np.isfinite(init_gain)]
        if init_loss.size:
            ax.hist(init_loss, bins=bins, density=True, histtype="step", linewidth=1.8,
                    label="loss: P(NV- at rep11)")
        if init_gain.size:
            ax.hist(init_gain, bins=bins, density=True, histtype="step", linewidth=1.8,
                    label="gain: P(NV0 at rep11), dim initial state")
        ax.axvline(0.95, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Posterior confidence in the initial charge state")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}: initial-state confidence asymmetry")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v5_initial_state_confidence"] = fig

        # V5-C. Only high-confidence discrete transitions in acquisition order.
        fig, ax = plt.subplots(figsize=(11.0, 5.8))
        for direction in ("loss", "gain"):
            kk = np.asarray(v5["directions"][direction]["primary_k"], dtype=float)
            ax.plot(runs, kk, linewidth=0.9,
                    label=("NV- -> NV0" if direction == "loss" else "NV0 -> NV-"))
        ax.set_xlabel("Run index")
        ax.set_ylabel(f"Truth-tested transitions with posterior >= {V5_PRIMARY_POSTERIOR_WEIGHT:.2f}")
        ax.set_title(f"{label}: high-confidence discrete event-channel trace")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v5_high_conf_discrete_run_trace"] = fig

    # 1. Gains in acquisition order: raw vs robust vs rep11 misclassification.
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.plot(runs, raw_gain, linewidth=1.0, label="raw NV0 -> NV- gains")
    ax.plot(runs, robust_gain, linewidth=1.0, label="robust gain candidates")
    ax.plot(runs, mis_gain, linewidth=1.0, label="likely rep11 misclassification")
    ax.set_xlabel("Run index")
    ax.set_ylabel("NVs / run")
    ax.set_title(f"{label}: decompose apparent NV0 -> NV- gains")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["gain_decomposition"] = fig

    # 2. Losses in acquisition order.
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.plot(runs, raw_loss, linewidth=1.0, label="raw NV- -> NV0 losses")
    ax.plot(runs, robust_loss, linewidth=1.0, label="robust loss candidates")
    ax.plot(runs, mis_loss, linewidth=1.0, label="likely rep12 misclassification")
    ax.set_xlabel("Run index")
    ax.set_ylabel("NVs / run")
    ax.set_title(f"{label}: decompose apparent NV- -> NV0 losses")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["loss_decomposition"] = fig

    # 3. Rep11 dimming vs gain decomposition.
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(scale11_z[good], raw_gain[good], s=13, alpha=0.45, label="raw gain")
    ax.scatter(scale11_z[good], robust_gain[good], s=13, alpha=0.45, label="robust gain")
    ax.scatter(scale11_z[good], mis_gain[good], s=13, alpha=0.45, label="rep11 misclass")
    ax.axvline(BRIGHTNESS_Z_CUT, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Rep11 cross-fit brightness scale robust z")
    ax.set_ylabel("NV0 -> NV- count")
    ax.set_title(f"{label}: are gain spikes explained by dim rep11?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["gain_vs_rep11_brightness"] = fig

    # 4. Rep12 dimming vs loss decomposition.
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(scale12_z[good], raw_loss[good], s=13, alpha=0.45, label="raw loss")
    ax.scatter(scale12_z[good], robust_loss[good], s=13, alpha=0.45, label="robust loss")
    ax.scatter(scale12_z[good], mis_loss[good], s=13, alpha=0.45, label="rep12 misclass")
    ax.axvline(BRIGHTNESS_Z_CUT, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Rep12 cross-fit brightness scale robust z")
    ax.set_ylabel("NV- -> NV0 count")
    ax.set_title(f"{label}: do real-looking losses survive rep12 correction?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["loss_vs_rep12_brightness"] = fig

    # 5. Raw vs robust loss count per run.
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.scatter(raw_loss[good], robust_loss[good], s=15, alpha=0.5)
    max_k = max(float(np.max(raw_loss[good])), 1.0)
    ax.plot([0, max_k], [0, max_k], linestyle="--", linewidth=1.0)
    ax.set_xlabel("Raw apparent NV- -> NV0 losses")
    ax.set_ylabel("Robust losses after all controls")
    ax.set_title(f"{label}: event survival after misclassification controls")
    fig.tight_layout()
    figures["raw_vs_robust_loss"] = fig

    # 6. Margin sweep: mean raw gain/loss counts.
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    margins = np.asarray(sorted(by_margin), dtype=float)
    mean_gain_margin = np.asarray(
        [np.mean(by_margin[m]["gain_count"][good]) for m in margins]
    )
    mean_loss_margin = np.asarray(
        [np.mean(by_margin[m]["loss_count"][good]) for m in margins]
    )
    ax.plot(margins, mean_gain_margin, marker="o", label="NV0 -> NV-")
    ax.plot(margins, mean_loss_margin, marker="o", label="NV- -> NV0")
    ax.set_xlabel("Two-sided classification margin (counts)")
    ax.set_ylabel("Mean transitions / good run")
    ax.set_title(f"{label}: raw transition survival versus classification margin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["margin_sweep"] = fig

    # 7. Standardized robust tail CCDF: loss versus gain.
    if RUN_ROBUST_TAIL_ANALYSIS and "loss" in robust_tail and "gain" in robust_tail:
        z_loss = robust_tail["loss"]["z"]
        z_gain = robust_tail["gain"]["z"]
        valid_loss = robust_tail["loss"]["valid"]
        valid_gain = robust_tail["gain"]["valid"]
        finite_max = [6.0]
        if np.any(valid_loss & np.isfinite(z_loss)):
            finite_max.append(float(np.nanmax(z_loss[valid_loss])))
        if np.any(valid_gain & np.isfinite(z_gain)):
            finite_max.append(float(np.nanmax(z_gain[valid_gain])))
        zmax = min(max(finite_max), 12.0)
        grid = np.linspace(0.0, zmax, 121)

        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.semilogy(
            grid,
            _empirical_ccdf(z_loss[valid_loss], grid),
            linewidth=1.8,
            label="robust NV- -> NV0",
        )
        ax.semilogy(
            grid,
            _empirical_ccdf(z_gain[valid_gain], grid),
            linewidth=1.8,
            label="robust NV0 -> NV-",
        )
        ax.semilogy(
            grid,
            norm.sf(grid),
            linestyle="--",
            linewidth=1.2,
            label="unit-normal reference",
        )
        ax.set_xlabel("Heterogeneous-null anomaly Z")
        ax.set_ylabel("Empirical P(Z >= z)")
        ax.set_title(f"{label}: truth-tested rare-event tails")
        ax.set_ylim(bottom=max(0.5 / max(np.sum(good), 1), 1e-5), top=1.0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["robust_bidirectional_tail_ccdf"] = fig

        # 8. Rare-event rates at 3/4/5 sigma, side-by-side.
        fig, ax = plt.subplots(figsize=(8.4, 6.2))
        zcuts = np.asarray(ROBUST_TAIL_Z_THRESHOLDS, dtype=float)
        loss_rates = np.asarray([
            100.0 * row["fraction"]
            for row in robust_tail["loss"]["threshold_rows"]
        ])
        gain_rates = np.asarray([
            100.0 * row["fraction"]
            for row in robust_tail["gain"]["threshold_rows"]
        ])
        x = np.arange(len(zcuts), dtype=float)
        width = 0.36
        ax.bar(x - width/2, loss_rates, width=width, label="robust NV- -> NV0")
        ax.bar(x + width/2, gain_rates, width=width, label="robust NV0 -> NV-")
        ax.set_xticks(x)
        ax.set_xticklabels([f">={z:g} sigma" for z in zcuts])
        ax.set_ylabel("Observed good-run fraction (%)")
        ax.set_yscale("log")
        ax.set_title(f"{label}: which direction carries the robust rare-event tail?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["robust_bidirectional_rare_event_rates"] = fig

        # 9. Run-by-run robust loss anomaly versus robust gain anomaly.
        fig, ax = plt.subplots(figsize=(7.0, 6.5))
        both = good & np.isfinite(z_loss) & np.isfinite(z_gain)
        ax.scatter(z_gain[both], z_loss[both], s=15, alpha=0.5)
        ax.axhline(4.0, linestyle="--", linewidth=1.0)
        ax.axvline(4.0, linestyle="--", linewidth=1.0)
        ax.axhline(0.0, linestyle=":", linewidth=0.8)
        ax.axvline(0.0, linestyle=":", linewidth=0.8)
        ax.set_xlabel("Robust NV0 -> NV- anomaly Z")
        ax.set_ylabel("Robust NV- -> NV0 anomaly Z")
        ax.set_title(f"{label}: bidirectional robust-event map")
        fig.tight_layout()
        figures["robust_bidirectional_z_scatter"] = fig

        # 10. Fano/dispersion comparison against heterogeneous null.
        fig, ax = plt.subplots(figsize=(7.6, 6.0))
        names = ["NV- -> NV0", "NV0 -> NV-"]
        obs_fano = [
            robust_tail["loss"]["observed_fano"],
            robust_tail["gain"]["observed_fano"],
        ]
        null_fano = [
            robust_tail["loss"]["null_fano"],
            robust_tail["gain"]["null_fano"],
        ]
        x = np.arange(2)
        width = 0.36
        ax.bar(x - width/2, obs_fano, width=width, label="observed robust Fano")
        ax.bar(x + width/2, null_fano, width=width, label="heterogeneous-null Fano")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Variance / mean")
        ax.set_title(f"{label}: robust count dispersion by transition direction")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["robust_bidirectional_fano"] = fig

    # ------------------------------------------------------------------
    # V3 figures: history, state confidence, local brightness, exact-PB FDR.
    # ------------------------------------------------------------------
    if RUN_V3_DISCRIMINATION and v3["depth"]:
        robust_gain_mask = truth_masks["gain_switch"] & good[None, :]
        robust_loss_mask = truth_masks["loss_switch"] & good[None, :]
        retained_nv0_mask = truth_masks["retained_nv0"] & good[None, :]
        retained_nvm_mask = truth_masks["retained_nvm"] & good[None, :]

        # V3-A. Per-NV rep11 history: are robust gains coming from NVs that are
        # normally NV- almost all the time?
        gain_hist = occ_loo[robust_gain_mask]
        gain_ctrl = occ_loo[retained_nv0_mask]
        fig, ax = plt.subplots(figsize=(8.2, 6.1))
        bins = np.linspace(0.0, 1.0, 51)
        if gain_ctrl.size:
            ax.hist(gain_ctrl[np.isfinite(gain_ctrl)], bins=bins, density=True,
                    histtype="step", linewidth=1.5,
                    label="retained NV0 controls")
        if gain_hist.size:
            ax.hist(gain_hist[np.isfinite(gain_hist)], bins=bins, density=True,
                    histtype="step", linewidth=1.8,
                    label="robust NV0 -> NV- events")
        ax.axvline(HISTORY_SUSPICIOUS_NVM_OCCUPANCY, linestyle="--", linewidth=1.0,
                   label=f"suspicious history >= {HISTORY_SUSPICIOUS_NVM_OCCUPANCY:.2f}")
        ax.set_xlabel("LOO P(NV- at rep11 in other good runs)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}: state-history test for residual rep11 gain misclassification")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v3_gain_state_history"] = fig

        # V3-B. Noise-normalized two-sided threshold depth.
        fig, ax = plt.subplots(figsize=(8.2, 6.1))
        dg = v3["depth"]["gain"]["min_sigma"][robust_gain_mask]
        dl = v3["depth"]["loss"]["min_sigma"][robust_loss_mask]
        dg = dg[np.isfinite(dg)]; dl = dl[np.isfinite(dl)]
        upper = 10.0
        bins = np.linspace(0.0, upper, 61)
        if dl.size:
            ax.hist(np.clip(dl, 0, upper), bins=bins, density=True,
                    histtype="step", linewidth=1.8, label="robust NV- -> NV0")
        if dg.size:
            ax.hist(np.clip(dg, 0, upper), bins=bins, density=True,
                    histtype="step", linewidth=1.8, label="robust NV0 -> NV-")
        ax.axvline(2.0, linestyle="--", linewidth=1.0, label="2-sigma state-depth guide")
        ax.set_xlabel("Minimum two-sided threshold depth / state-conditioned sigma")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}: are surviving transitions deep inside both charge states?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v3_noise_normalized_threshold_depth"] = fig

        # V3-C. Local relevant-rep brightness at the primary radius.
        primary = v3["local_by_radius"].get(float(LOCAL_BRIGHTNESS_PRIMARY_RADIUS_UM))
        if primary is not None:
            fig, ax = plt.subplots(figsize=(8.3, 6.1))
            gain_local = primary["rep11"]["z"][robust_gain_mask]
            loss_local = primary["rep12"]["z"][robust_loss_mask]
            gain_control = primary["rep11"]["z"][retained_nv0_mask]
            loss_control = primary["rep12"]["z"][retained_nvm_mask]
            bins = np.linspace(-6.0, 6.0, 73)
            for vals, lab, lw in (
                (gain_control, "retained NV0 control: rep11", 1.2),
                (gain_local, "robust gain: rep11", 1.8),
                (loss_control, "retained NV- control: rep12", 1.2),
                (loss_local, "robust loss: rep12", 1.8),
            ):
                vals = np.asarray(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ax.hist(np.clip(vals, -6, 6), bins=bins, density=True,
                            histtype="step", linewidth=lw, label=lab)
            ax.axvline(LOCAL_DIM_Z_CUT, linestyle="--", linewidth=1.0,
                       label=f"local dimming z <= {LOCAL_DIM_Z_CUT:g}")
            ax.axvline(0.0, linestyle=":", linewidth=0.8)
            ax.set_xlabel(
                f"Local-minus-global brightness z ({LOCAL_BRIGHTNESS_PRIMARY_RADIUS_UM:g} um neighborhood)"
            )
            ax.set_ylabel("Density")
            ax.set_title(f"{label}: local artifact test at the physically relevant readout rep")
            ax.legend(fontsize=7)
            fig.tight_layout()
            figures["v3_local_brightness_event_vs_control"] = fig

            # V3-D. Candidate significance versus local dimming.
            fig, ax = plt.subplots(figsize=(7.4, 6.3))
            for direction, marker in (("loss", "o"), ("gain", "s")):
                mask = truth_masks[f"{direction}_switch"] & good[None, :]
                relevant = primary["rep12" if direction == "loss" else "rep11"]["z"]
                run_local = _run_median_over_events(relevant, mask)
                q = v3["q_combined"][direction]
                y = -np.log10(np.clip(q, 1e-300, 1.0))
                ok = good & np.isfinite(run_local) & np.isfinite(y)
                ax.scatter(run_local[ok], y[ok], s=18, alpha=0.55,
                           marker=marker,
                           label="NV- -> NV0" if direction == "loss" else "NV0 -> NV-")
            ax.axvline(LOCAL_DIM_Z_CUT, linestyle="--", linewidth=1.0)
            ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0,
                       label="combined FDR q=0.05")
            ax.set_xlabel("Median relevant local-brightness z among robust event NVs")
            ax.set_ylabel("-log10(combined exact-PB FDR q)")
            ax.set_title(f"{label}: are significant runs tied to local dimming?")
            ax.legend(fontsize=8)
            fig.tight_layout()
            figures["v3_fdr_vs_local_brightness"] = fig

        # V3-E. Exact-PB/BH q-values in acquisition order.
        fig, ax = plt.subplots(figsize=(12.0, 5.8))
        for direction in ("loss", "gain"):
            q = np.asarray(v3["q_combined"][direction], dtype=float)
            y = -np.log10(np.clip(q, 1e-300, 1.0))
            ok = good & np.isfinite(y)
            ax.scatter(runs[ok], y[ok], s=15, alpha=0.55,
                       label="NV- -> NV0" if direction == "loss" else "NV0 -> NV-")
        ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1.0,
                   label="loss+gain family FDR q=0.05")
        ax.axhline(-np.log10(0.01), linestyle=":", linewidth=1.0,
                   label="loss+gain family FDR q=0.01")
        ax.set_xlabel("Run index")
        ax.set_ylabel("-log10(exact-PB BH q)")
        ax.set_title(f"{label}: conservative exact-PB FDR audit")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v3_exact_pb_fdr_run_trace"] = fig

        # V3-F. Per-NV event participation versus normal rep11 occupancy.
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        rg_nv = np.sum(robust_gain_mask, axis=1)
        rl_nv = np.sum(robust_loss_mask, axis=1)
        ax.scatter(occ_all, rg_nv, s=18, alpha=0.55, label="robust gain participation")
        ax.scatter(occ_all, rl_nv, s=18, alpha=0.55, label="robust loss participation")
        ax.axvline(HISTORY_SUSPICIOUS_NVM_OCCUPANCY, linestyle="--", linewidth=1.0)
        ax.set_xlabel("P(NV- at rep11 over all good runs)")
        ax.set_ylabel("Number of robust event participations")
        ax.set_title(f"{label}: do gain events select NVs that are normally NV-?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v3_per_nv_participation_vs_history"] = fig

    # V4 figures: posterior event confidence and soft coincidence tails.
    if RUN_V4_SOFT_CLASSIFIER and v4.get("soft_results"):
        soft_good_plot = np.asarray(v4.get("good_run_mask", good), dtype=bool)
        # V4-A. Posterior transition weight of hard-robust events.
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        for direction in ("loss", "gain"):
            hard = np.asarray(truth_masks[f"{direction}_switch"], dtype=bool) & soft_good_plot[None, :]
            vals = np.asarray(v4["weights"][direction], dtype=float)[hard]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                bins = np.linspace(0.0, 1.0, 51)
                hist, edges = np.histogram(vals, bins=bins, density=True)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centers, hist, where="mid", linewidth=1.5, label=f"robust {direction}")
        ax.set_xlabel("Soft transition weight")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}: posterior confidence of hard-robust transitions")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v4_hard_event_soft_weight_distribution"] = fig

        # V4-B. Observed soft-score tail vs independently scrambled NV null.
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        for direction in ("loss", "gain"):
            sr = v4["soft_results"][direction]
            obs = np.asarray(sr["score"], dtype=float)[soft_good_plot]
            obs = np.sort(obs[np.isfinite(obs)])
            nul = np.sort(np.asarray(sr["null_samples"], dtype=float))
            if obs.size:
                y = np.arange(obs.size, 0, -1, dtype=float) / obs.size
                ax.step(obs, y, where="post", linewidth=1.5, label=f"observed {direction}")
            if nul.size:
                # Thin the null curve for plotting only.
                step = max(1, nul.size // 5000)
                xx = nul[::step]
                yy = (nul.size - np.arange(0, nul.size, step, dtype=float)) / nul.size
                ax.step(xx, yy, where="post", linewidth=1.0, linestyle="--", label=f"independent-NV null {direction}")
        ax.set_yscale("log")
        ax.set_xlabel("Soft transition score per run")
        ax.set_ylabel("Upper-tail probability")
        ax.set_title(f"{label}: probabilistic transition tail vs independent-NV null")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v4_soft_score_tail_vs_scramble_null"] = fig

        # V4-C. Acquisition-order soft anomaly trace.
        fig, ax = plt.subplots(figsize=(10.0, 5.6))
        for direction in ("loss", "gain"):
            zsoft = np.asarray(v4["soft_results"][direction]["z_vs_null"], dtype=float)
            ax.plot(runs, zsoft, linewidth=0.8, alpha=0.75, label=f"soft {direction}")
        ax.axhline(0.0, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Run")
        ax.set_ylabel("Soft score z vs independent-NV null")
        ax.set_title(f"{label}: probabilistic event-channel trace")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v4_soft_z_run_trace"] = fig

        # V4-D. Hard robust K versus soft excess, directly exposing attenuation.
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        for direction in ("loss", "gain"):
            hardk = np.sum(truth_masks[f"{direction}_switch"], axis=0).astype(float)
            sr = v4["soft_results"][direction]
            excess = np.asarray(sr["score"], dtype=float) - float(sr["null_mean"])
            m = soft_good_plot & np.isfinite(excess)
            ax.scatter(hardk[m], excess[m], s=12, alpha=0.45, label=direction)
        ax.axhline(0.0, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Hard robust transition count K")
        ax.set_ylabel("Soft score - independent-null mean")
        ax.set_title(f"{label}: do hard coincidence bursts survive state uncertainty?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v4_hard_K_vs_soft_excess"] = fig

        # V4-E. Soft p/q map: candidate directionality after state uncertainty.
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        for direction in ("loss", "gain"):
            sr = v4["soft_results"][direction]
            qc = np.asarray(v4["q_combined"][direction], dtype=float)
            zz = np.asarray(sr["z_vs_null"], dtype=float)
            m = soft_good_plot & np.isfinite(qc) & np.isfinite(zz)
            y = -np.log10(np.clip(qc[m], 1e-12, 1.0))
            ax.scatter(zz[m], y, s=13, alpha=0.55, label=direction)
        ax.axhline(-np.log10(0.05), linewidth=1.0, linestyle="--", label="q=0.05")
        ax.set_xlabel("Soft z vs independent-NV null")
        ax.set_ylabel("-log10(q), loss+gain family")
        ax.set_title(f"{label}: soft event significance after state uncertainty")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["v4_soft_significance_map"] = fig

    # 11/12. V23 weighted same-K spatial curves for robust loss and gain.
    for direction, spatial in robust_spatial.items():
        if not spatial.get("success", False):
            continue
        null = spatial["null"]
        fit = spatial["fit"]
        centers = np.asarray(null["centers_um"], dtype=float)
        gvals = np.asarray(null["g_weighted"], dtype=float)
        valid = np.asarray(null["valid_bin_mask"], dtype=bool)
        null_mean = np.asarray(null["null_mean_pair_counts"], dtype=float)
        null_std = np.asarray(null["null_std_pair_counts"], dtype=float)
        yerr = np.full(gvals.shape, np.nan)
        ok = valid & np.isfinite(null_mean) & (null_mean > 0)
        yerr[ok] = null_std[ok] / null_mean[ok]

        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.errorbar(
            centers[ok],
            gvals[ok],
            yerr=yerr[ok],
            marker="o",
            linestyle="none",
            capsize=2,
            label=f"robust {direction} / weighted same-K null",
        )
        ax.axhline(1.0, linestyle="--", linewidth=1.0, label="null expectation")

        if fit.get("success", False):
            best_name = fit.get("best_model")
            best = fit.get("fits", {}).get(best_name, {})
            xfit = np.asarray(fit.get("x_used", []), dtype=float)
            yfit = np.asarray(best.get("pred", []), dtype=float)
            if xfit.size and yfit.size == xfit.size:
                ax.plot(xfit, yfit, linewidth=1.5, label=f"{best_name} fit")

        ax.set_xlabel("NV-pair separation (um)")
        ax.set_ylabel("Observed / weighted same-K null")
        ax.set_title(f"{label}: robust {direction} spatial correlation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures[f"robust_{direction}_weighted_same_k_spatial"] = fig

    if SAVE_OUTPUTS:
        for name, fig in figures.items():
            fig.savefig(
                OUTPUT_DIR / f"{safe_label}_{name}.png",
                dpi=180,
                bbox_inches="tight",
            )
        print(f"Saved figures under: {OUTPUT_DIR.resolve()}")

    return {
        "dataset_label": label,
        "dark_wait_s": dark_wait_s,
        "good_run_mask": good,
        "raw_gain": raw_gain,
        "raw_loss": raw_loss,
        "robust_gain": robust_gain,
        "mis_gain": mis_gain,
        "amb_gain": amb_gain,
        "robust_loss": robust_loss,
        "mis_loss": mis_loss,
        "amb_loss": amb_loss,
        "scale11_z": scale11_z,
        "scale12_z": scale12_z,
        "by_margin": by_margin,
        "truth_masks": truth_masks,
        "robust_tail": robust_tail,
        "exact_tail_rows": exact_tail_rows,
        "robust_spatial": robust_spatial,
        "v3": v3,
        "v4": v4,
        "v5": v5,
        "figures": figures,
        "total_raw_gain": total_raw_gain,
        "total_robust_gain": total_robust_gain,
        "total_mis_gain": total_mis_gain,
        "total_raw_loss": total_raw_loss,
        "total_robust_loss": total_robust_loss,
        "total_mis_loss": total_mis_loss,
    }


# =============================================================================
# CROSS-DATASET SUMMARY
# =============================================================================


def compare_datasets(results):
    if not results:
        return None

    print("\n" + "=" * 150)
    print("TRANSITION TRUTH SUMMARY ACROSS DATASETS")
    print("=" * 150)
    print(
        "wait(s)  goodRuns  rawG/run  robustG/run  rep11MisG/run  "
        "rawL/run  robustL/run  rep12MisL/run  robustL/rawL"
    )
    print("-" * 150)

    rows = []
    for r in sorted(results, key=lambda x: x["dark_wait_s"]):
        good = r["good_run_mask"]
        ng = int(np.sum(good))

        raw_g = float(np.mean(r["raw_gain"][good]))
        rob_g = float(np.mean(r["robust_gain"][good]))
        mis_g = float(np.mean(r["mis_gain"][good]))
        raw_l = float(np.mean(r["raw_loss"][good]))
        rob_l = float(np.mean(r["robust_loss"][good]))
        mis_l = float(np.mean(r["mis_loss"][good]))
        surv_l = rob_l / raw_l if raw_l > 0 else np.nan

        print(
            f"{r['dark_wait_s']:7.1f} {ng:9d} "
            f"{raw_g:9.4f} {rob_g:12.4f} {mis_g:14.4f} "
            f"{raw_l:9.4f} {rob_l:12.4f} {mis_l:14.4f} {surv_l:13.4f}"
        )

        rows.append(
            {
                "dataset": r["dataset_label"],
                "dark_wait_s": r["dark_wait_s"],
                "good_runs": ng,
                "raw_gain_per_run": raw_g,
                "robust_gain_per_run": rob_g,
                "likely_rep11_misclass_gain_per_run": mis_g,
                "raw_loss_per_run": raw_l,
                "robust_loss_per_run": rob_l,
                "likely_rep12_misclass_loss_per_run": mis_l,
                "robust_loss_over_raw_loss": surv_l,
            }
        )

    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_run_csv(OUTPUT_DIR / "dataset_truth_summary.csv", rows)

    # Main comparison figure: raw and robust losses versus dark wait.
    waits = np.asarray([x["dark_wait_s"] for x in sorted(results, key=lambda x: x["dark_wait_s"])])
    raw_l = np.asarray([
        np.mean(x["raw_loss"][x["good_run_mask"]])
        for x in sorted(results, key=lambda x: x["dark_wait_s"])
    ])
    rob_l = np.asarray([
        np.mean(x["robust_loss"][x["good_run_mask"]])
        for x in sorted(results, key=lambda x: x["dark_wait_s"])
    ])
    mis_l = np.asarray([
        np.mean(x["mis_loss"][x["good_run_mask"]])
        for x in sorted(results, key=lambda x: x["dark_wait_s"])
    ])

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.plot(waits, raw_l, marker="o", linewidth=1.5, label="raw NV- -> NV0")
    ax.plot(waits, rob_l, marker="o", linewidth=1.5, label="robust NV- -> NV0")
    ax.plot(waits, mis_l, marker="o", linewidth=1.5, label="likely rep12 misclassification")
    ax.set_xlabel("Dark wait (s)")
    ax.set_ylabel("Mean transitions / good run")
    ax.set_title("Does the NV- -> NV0 dark-time increase survive misclassification controls?")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR / "comparison_raw_robust_loss_vs_dark_wait.png",
            dpi=180,
            bbox_inches="tight",
        )

    # ------------------------------------------------------------------
    # Cross-dataset robust event-channel comparison.
    # ------------------------------------------------------------------
    if all("loss" in r.get("robust_tail", {}) and "gain" in r.get("robust_tail", {}) for r in results):
        print("\nROBUST EVENT-CHANNEL SUMMARY ACROSS DATASETS")
        print("-" * 150)
        print(
            "wait(s) direction   meanK   Fano(obs/null)  VarRatio  "
            "Z>=3%    Z>=4%    Z>=5%"
        )
        channel_rows = []
        sorted_results = sorted(results, key=lambda x: x["dark_wait_s"])
        for r in sorted_results:
            for direction in ("loss", "gain"):
                tr = r["robust_tail"][direction]
                rates = {row["zcut"]: 100.0*row["fraction"] for row in tr["threshold_rows"]}
                print(
                    f"{r['dark_wait_s']:7.1f} {direction:>9s} "
                    f"{tr['observed_mean']:7.3f} "
                    f"{tr['observed_fano']:6.3f}/{tr['null_fano']:.3f} "
                    f"{tr['dispersion_ratio']:8.3f} "
                    f"{rates.get(3.0, np.nan):7.3f} "
                    f"{rates.get(4.0, np.nan):7.3f} "
                    f"{rates.get(5.0, np.nan):7.3f}"
                )
                channel_rows.append({
                    "dataset": r["dataset_label"],
                    "dark_wait_s": r["dark_wait_s"],
                    "direction": direction,
                    "mean_k": tr["observed_mean"],
                    "variance_k": tr["observed_variance"],
                    "fano_observed": tr["observed_fano"],
                    "fano_heterogeneous_null": tr["null_fano"],
                    "variance_ratio_to_null": tr["dispersion_ratio"],
                    "rare_z3_percent": rates.get(3.0, np.nan),
                    "rare_z4_percent": rates.get(4.0, np.nan),
                    "rare_z5_percent": rates.get(5.0, np.nan),
                })
        if SAVE_OUTPUTS:
            _write_run_csv(OUTPUT_DIR / "robust_event_channel_summary.csv", channel_rows)

        # Cross-dataset rare-event-rate plot at the primary 4-sigma cut.
        waits2 = np.asarray([r["dark_wait_s"] for r in sorted_results], dtype=float)
        loss_rate4 = np.asarray([
            100.0 * next(row["fraction"] for row in r["robust_tail"]["loss"]["threshold_rows"] if row["zcut"] == 4.0)
            for r in sorted_results
        ])
        gain_rate4 = np.asarray([
            100.0 * next(row["fraction"] for row in r["robust_tail"]["gain"]["threshold_rows"] if row["zcut"] == 4.0)
            for r in sorted_results
        ])

        fig2, ax2 = plt.subplots(figsize=(8.2, 6.2))
        ax2.plot(waits2, loss_rate4, marker="o", linewidth=1.5, label="robust NV- -> NV0")
        ax2.plot(waits2, gain_rate4, marker="o", linewidth=1.5, label="robust NV0 -> NV-")
        ax2.set_xlabel("Dark wait (s)")
        ax2.set_ylabel("Good runs with Z >= 4 (%)")
        ax2.set_title("Truth-tested rare-event rate versus dark exposure")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        if SAVE_OUTPUTS:
            fig2.savefig(
                OUTPUT_DIR / "comparison_robust_z4_event_rate_vs_dark_wait.png",
                dpi=180,
                bbox_inches="tight",
            )

    # ------------------------------------------------------------------
    # V4 cross-dataset soft-channel summary.
    # ------------------------------------------------------------------
    if all(r.get("v4", {}).get("soft_results") for r in results):
        print("\nV4 SOFT EVENT-CHANNEL SUMMARY ACROSS DATASETS")
        print("-" * 150)
        print(
            "wait(s) direction meanSoft nullMean nullStd  medHardW  "
            "q<=.05(dir) q<=.05(both)"
        )
        v4_rows = []
        for r in sorted(results, key=lambda x: x["dark_wait_s"]):
            good = np.asarray(r["v4"].get("good_run_mask", r["good_run_mask"]), dtype=bool)
            for direction in ("loss", "gain"):
                sr = r["v4"]["soft_results"][direction]
                es = r["v4"]["event_weight_summary"][direction]
                qd = np.asarray(sr["q_direction"], dtype=float)
                qb = np.asarray(r["v4"]["q_combined"][direction], dtype=float)
                nqd = int(np.sum(good & np.isfinite(qd) & (qd <= 0.05)))
                nqb = int(np.sum(good & np.isfinite(qb) & (qb <= 0.05)))
                mean_soft = float(np.nanmean(np.asarray(sr["score"], dtype=float)[good]))
                print(
                    f"{r['dark_wait_s']:7.1f} {direction:>9s} "
                    f"{mean_soft:8.3f} {sr['null_mean']:8.3f} {sr['null_std']:7.3f} "
                    f"{es['median']:9.3f} {nqd:11d} {nqb:12d}"
                )
                v4_rows.append({
                    "dataset": r["dataset_label"],
                    "dark_wait_s": r["dark_wait_s"],
                    "direction": direction,
                    "mean_soft_score": mean_soft,
                    "null_mean": sr["null_mean"],
                    "null_std": sr["null_std"],
                    "median_hard_event_soft_weight": es["median"],
                    "q05_within_direction_count": nqd,
                    "q05_loss_gain_family_count": nqb,
                })
        if SAVE_OUTPUTS:
            _write_run_csv(OUTPUT_DIR / "v4_soft_event_channel_summary.csv", v4_rows)

        # Mean soft score and q<0.05 event fraction vs dark wait.
        sorted_results = sorted(results, key=lambda x: x["dark_wait_s"])
        waits4 = np.asarray([r["dark_wait_s"] for r in sorted_results], dtype=float)
        fig4, ax4 = plt.subplots(figsize=(8.2, 6.2))
        for direction in ("loss", "gain"):
            means = []
            for r in sorted_results:
                g = np.asarray(r["v4"].get("good_run_mask", r["good_run_mask"]), dtype=bool)
                means.append(np.nanmean(np.asarray(r["v4"]["soft_results"][direction]["score"], dtype=float)[g]))
            ax4.plot(waits4, means, marker="o", linewidth=1.5, label=f"soft {direction}")
        ax4.set_xlabel("Dark wait (s)")
        ax4.set_ylabel("Mean soft transition score / run")
        ax4.set_title("Probabilistic charge-transition score versus dark wait")
        ax4.legend(fontsize=8)
        fig4.tight_layout()
        if SAVE_OUTPUTS:
            fig4.savefig(OUTPUT_DIR / "comparison_v4_mean_soft_score_vs_dark_wait.png", dpi=180, bbox_inches="tight")

    # Focused V5 cross-dataset table for the final directionality claim.
    if RUN_V5_FOCUSED_DIRECTIONALITY and all(r.get("v5", {}).get("directions") for r in results):
        print("\nV5 FOCUSED HIGH-CONFIDENCE DIRECTIONAL SUMMARY")
        print("-" * 150)
        print("wait(s) direction raw/run robust/run K95/run robust/raw K95/robust medInitConf medJointW")
        for r in sorted(results, key=lambda x: x["dark_wait_s"]):
            good_r = np.asarray(r["good_run_mask"], dtype=bool)
            ng = max(int(np.sum(good_r)), 1)
            for direction in ("loss", "gain"):
                d = r["v5"]["directions"][direction]
                conf = d["event_confidence"]
                print(
                    f"{r['dark_wait_s']:7.1f} {direction:>9s} "
                    f"{d['total_raw']/ng:7.3f} {d['total_robust']/ng:10.3f} "
                    f"{d['total_primary']/ng:8.3f} "
                    f"{d['total_robust']/max(d['total_raw'],1):10.3f} "
                    f"{d['total_primary']/max(d['total_robust'],1):11.3f} "
                    f"{conf['median_initial_conf']:11.3f} {conf['median_weight']:9.3f}"
                )

    return fig


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    kpl.init_kplotlib()
    datasets = base.DATASETS if DATASETS is None else DATASETS

    results = []
    for dataset in datasets:
        results.append(analyze_dataset(dataset))

    comparison_figure = compare_datasets(results)

    analyses = results
    analysis = results[-1] if results else None

    kpl.show(block=True)
