# -*- coding: utf-8 -*-
"""
Spatial carrier-event analysis v5B with multi-file-per-wait appending and full-FOV spatial-support diagnostics.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
Keep the count-only carrier-capture inference and the spatial assumptions
separate.  This script reuses the same RAW NV- -> NV0 classification, but then
adds spatial statistics only after a fixed physical carrier-hazard threshold
Lambda_h has selected candidate events.

It intentionally does NOT use:
    - truth/misclassification filtering
    - image pixels / img_arrays
    - a pre-assumed diffusion length
    - Poisson sigma as the primary spatial event definition

PRIMARY QUESTIONS
-----------------
1. Does same-run NV- -> NV0 switching show excess spatial correlation versus
   a geometry-preserving / time-series-preserving null?

2. For fixed Lambda_h event classes, is the spatial pattern better described
   by:
       U: a spatially uniform/common-mode carrier burst,
       P: a localized point-like exponential transport kernel,
       L: a line-like exponential transport kernel?

3. Only if a localized model beats the uniform model should its fitted
   L_eff be discussed as an *effective spatial transport scale*.  L_eff is
   NOT automatically a microscopic diffusion length.

4. With passive initial/final readout, D and carrier lifetime tau cannot
   generally be separated.  A controlled source + variable delay is needed
   for D and tau separately.

DATA SELECTION
--------------
Default waits:
    0, 30, 60, 90 s

Each wait may be represented by ONE OR MANY physical acquisition files.
All files at the same wait are classified with their own saved thresholds,
quality-screened independently, and then appended before p_bulk, Lambda_h,
spatial nulls, event models, jackknife, and trajectory analysis.

If a requested wait is not present in base.DATASETS, it is skipped gracefully.

For consistency with the primary dark-time comparison, the 0-s dataset uses
ONLY the first 2000 runs before quality rejection.  This should leave 1992
good runs when runs 628--635 are rejected.

SPATIAL EVENT VARIABLE
----------------------
For every good run

    Lambda_h = max[
        0,
        -ln( (1-K/N)/(1-p_bulk) )
    ]

where p_bulk is the fitted central beta-binomial loss probability at that wait.

Fixed event thresholds:
    Lambda_h >= 0.04, 0.05, 0.07

The default primary spatial threshold is Lambda_h >= 0.05.

PER-NV DARK BASELINE
--------------------
For each wait and NV, estimate p_i^dark from runs below
BASELINE_EXCLUDE_LAMBDA_H, with beta-prior shrinkage toward p_bulk:

    p_i = (k_i + s p_bulk)/(n_i + s)

where s = BASELINE_PRIOR_STRENGTH.

This prevents intrinsically unstable NVs from automatically creating apparent
spatial clustering.

PAIR-CORRELATION STATISTIC
--------------------------
Define

    Y_ir = X_ir - p_i^dark

for evaluable NVs, where X_ir=1 for NV- -> NV0.

For each distance bin d,

    C(d) =
      sum_selected_runs sum_pairs_in_bin Y_i Y_j
      ------------------------------------------------
      number of evaluable NV pairs contributing

The spatial null is conditioned on the observed event magnitude.

For each selected run we first fit a spatially uniform event amplitude A_r,

    p_ir^U = 1 - (1-p_i^dark) exp(-A_r),

and form uniform-event residuals

    R_ir = X_ir - p_ir^U.

The observed distance-binned correlation is built from R_ir R_jr.

For the null, every event keeps:
    - exactly the same evaluable NV set,
    - exactly the same observed number K_r of losses,
    - the same per-NV dark probabilities,
    - the same fitted uniform event amplitude.

The identities of the K_r lost NVs are resampled from the exact conditional
Bernoulli distribution implied by p_ir^U, given sum_i X_ir = K_r.

This tests spatial structure *beyond the global event amplitude* and avoids
the failure mode where a large same-run loss burst automatically creates
positive correlations at all NV-NV separations.

SPATIAL MODEL COMPARISON
------------------------
For a selected event, the Bernoulli probability is

    p_i = 1 - (1-p_i^dark) exp(-Lambda_i).

Models:

Uniform/common-mode:
    Lambda_i = A

Point-like:
    Lambda_i = A exp(-r_i/L_eff)

Line-like proxy:
    Lambda_i = A exp(-d_perp_i/L_eff)

Models are fit by maximum Bernoulli likelihood and compared by AIC/BIC.

IMPORTANT:
The line model is a 2D projected screening model, NOT an exact 3D muon-track
Green-function solution.  It answers only whether a line-like pattern predicts
the binary NV charge map better than a uniform or point-like pattern.

OUTPUTS
-------
analysis_output/spatial_carrier_analysis/

    spatial_dataset_summary.csv
    spatial_pair_correlation.csv
    spatial_event_model_fits.csv
    spatial_event_summary.csv
    spatial_pair_correlation_jackknife.csv
    spatial_trajectory_summary.csv
    spatial_trajectory_bootstrap.csv
    spatial_trajectory_nv_coordinates.csv
    spatial_analysis_summary.txt

    pair_correlation_<wait>_<cut>.png
    pair_correlation_combined_<cut>.png
    model_preference_<cut>.png
    jackknife_pair_correlation_<wait>_<cut>.png

    event_maps/
        event_<wait>_run_<...>.png

    trajectory/
        trajectory_transverse_<wait>_run_<...>.png

RECOMMENDED INTERPRETATION
--------------------------
Do NOT report a diffusion length unless:
    - the event class has a reproducible non-null spatial correlation, AND
    - a localized model is preferred over uniform/common-mode by a meaningful
      information-criterion margin (default Delta AIC >= 6).

Even then call L_eff an "effective spatial transport scale" until a controlled
localized carrier-injection experiment calibrates the transport kernel.
"""

from __future__ import annotations

import csv
import importlib
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import coo_matrix
from scipy.special import logsumexp
from scipy.spatial import ConvexHull


# =============================================================================
# USER CONFIGURATION
# =============================================================================

WANTED_WAITS_S = (0.0, 30.0, 60.0, 90.0)
ALLOW_MISSING_WAITS = True

# Apples-to-apples primary data selection.
MAX_RUNS_BY_WAIT = {
    0.0: 2008,
    30.0: None,
    60.0: None,
    90.0: None,
}

# Fixed physical event thresholds.
LAMBDA_H_CUTS = (0.04, 0.05, 0.07)
PRIMARY_LAMBDA_H_CUT = 0.05

# Per-NV dark baseline excludes candidate-like large runs.
BASELINE_EXCLUDE_LAMBDA_H = 0.04

# Beta-prior pseudo-count strength for per-NV dark probability.
# With ~1000-2000 runs per wait this is deliberately weak shrinkage.
BASELINE_PRIOR_STRENGTH = 50.0

# ---------------------------------------------------------------------------
# Spatial calibration
# ---------------------------------------------------------------------------

# Existing charge-memory spatial analyses used 0.43 um/camera pixel.
# Keep this configurable and CHECK it against the known array pitch.
UM_PER_PIXEL = 0.43

# Qnami array pitch used as a calibration sanity check only.
# The code prints the um/pixel that the median nearest-neighbor spacing would
# imply if this pitch is correct.  It does NOT silently rescale coordinates.
EXPECTED_ARRAY_PITCH_UM = 2.6
SCALE_WARNING_FRACTION = 0.15

# Camera frame used only for the field-of-view diagnostic.  The raw files in
# this experiment use 375 x 375 pixel images.  This does NOT load img_arrays.
CAMERA_FRAME_SHAPE_PX = (375, 375)  # (height, width)

# Pair-correlation distance bins.
PAIR_BIN_WIDTH_UM = 5.0

# None = extend the plotted/counted pair correlation all the way to the
# largest NV-NV separation actually supported by the sensor coordinates.
PAIR_MAX_DISTANCE_UM = None

# Preserve the previous 0-100 um family-wise distance-search range for the
# quoted primary global p-value.  Distances beyond this are diagnostic only.
PAIR_PRIMARY_GLOBAL_TEST_MAX_DISTANCE_UM = 100.0

# Fixed-K, uniform-event conditional null.
#
# Each null event has the same K and evaluable-NV set as the observed event.
# 500 gives a minimum empirical p-value of ~0.002.  Increase to 2000+ for a
# final figure after the analysis structure is stable.
PAIR_NULL_SCRAMBLES = 500
PAIR_NULL_SEED = 260829

# Save-and-close individual event maps immediately so a large event set does
# not accumulate dozens of open matplotlib figures.
CLOSE_EVENT_MAPS_IMMEDIATELY = True

# Minimum selected runs needed to attempt class-level pair correlation.
MIN_EVENTS_FOR_PAIR_CORRELATION = 3

# ---------------------------------------------------------------------------
# Per-event model fitting
# ---------------------------------------------------------------------------

FIT_UNIFORM_MODEL = True
FIT_POINT_MODEL = True
FIT_LINE_MODEL = True

# Effective scale bounds.  These are intentionally broad screening bounds.
L_EFF_MIN_UM = 0.5
L_EFF_MAX_UM = 200.0

# Event-burst hazard amplitude bound.
A_MIN = 1e-7
A_MAX = 5.0

# Number of optimization start points.
POINT_RANDOM_STARTS = 8
LINE_RANDOM_STARTS = 10
MODEL_FIT_SEED = 260830

# Require this AIC improvement over uniform before labeling an event localized.
LOCALIZED_DELTA_AIC_THRESHOLD = 6.0

# ---------------------------------------------------------------------------
# Leave-one-event-out robustness of the CLASS-LEVEL spatial correlation.
# ---------------------------------------------------------------------------

RUN_SPATIAL_JACKKNIFE = True

# The primary carrier-hazard class is the cleanest one to jackknife.
# Add 0.04 or 0.07 here if desired.
JACKKNIFE_LAMBDA_H_CUTS = (0.05,)

# ---------------------------------------------------------------------------
# Projected trajectory inference for decisively line-like events.
# ---------------------------------------------------------------------------

RUN_TRAJECTORY_ANALYSIS = True
TRAJECTORY_LAMBDA_H_CUT = 0.05

# Fixed-K parametric bootstrap under the fitted line model.
# 200 is a useful exploratory value.  For final quoted intervals, increase
# to ~500-1000 after the model/selection choices are frozen.
TRAJECTORY_BOOTSTRAP_REPS = 200
TRAJECTORY_BOOTSTRAP_SEED = 260831

# Number of local optimization starts used for each bootstrap refit.
TRAJECTORY_BOOTSTRAP_LOCAL_STARTS = 5

# A trajectory is only called "angle resolved" if the central 95% bootstrap
# width is narrower than this.  The line may still be preferred even if the
# orientation is poorly resolved.
MAX_RESOLVED_TRACK_ANGLE_CI_WIDTH_DEG = 60.0

# Save transverse-coordinate diagnostic plots.
SAVE_TRAJECTORY_TRANSVERSE_PROFILES = True

# Save 2-panel spatial plots that overlay the fitted projected trajectory and
# the corresponding line-model probability field.
SAVE_TRAJECTORY_SPATIAL_PREDICTION_PLOTS = True

# Geometry-aware fixed-K null for fitted trajectory angles.
# For each real line-like event, synthetic events are generated with the same K,
# the same evaluable NV set, and the same fitted uniform-event baseline.  The
# same spatial-model selection is then rerun on each synthetic event.
RUN_TRAJECTORY_ANGLE_NULL = True
TRAJECTORY_NULL_REPS_PER_EVENT = 250
TRAJECTORY_NULL_SEED = 260832

# Population-level axial-clustering null: sample one null line-like angle from
# each event-specific null pool and compare the axial resultant length R.
TRAJECTORY_CLUSTER_NULL_REPS = 2000

# Fit at most this many highest-Lambda events for each wait/cut to keep runtime
# bounded.  Set None to fit every selected event.
MAX_MODEL_EVENTS_PER_WAIT_PER_CUT = 50

# Save top event maps for each wait at the PRIMARY cut.
MAX_EVENT_MAPS_PER_WAIT = 12

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

SAVE_OUTPUTS = True
SHOW_FIGURES = True
OUTPUT_DIR = Path("analysis_output") / "spatial_carrier_analysis"
EVENT_MAP_DIR = OUTPUT_DIR / "event_maps"
TRAJECTORY_DIR = OUTPUT_DIR / "trajectory"


# =============================================================================
# GENERIC HELPERS
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


def _ensure_output_dirs():
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        EVENT_MAP_DIR.mkdir(parents=True, exist_ok=True)
        TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


def _safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float)
    np.divide(num, den, out=out, where=np.asarray(den) != 0)
    return out


def _percentile_summary(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p16": np.nan,
            "p84": np.nan,
            "p025": np.nan,
            "p975": np.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
        "p025": float(np.percentile(arr, 2.5)),
        "p975": float(np.percentile(arr, 97.5)),
    }


def _cut_token(cut):
    return f"lambda_{float(cut):.3f}".replace(".", "p")


# =============================================================================
# IMPORT RAW PHENOMENOLOGY
# =============================================================================

def _import_phenomenology():
    names = (
        "sc_charge_state_particle_memory_bulk_and_tail_phenomenology",
        "sc_charge_state_particle_memory_bulk_tail_mixture_model",
    )

    errors = []
    for name in names:
        try:
            mod = importlib.import_module(name)
            required = (
                "_load_counts_thresholds",
                "_good_run_mask",
                "_classify_raw",
                "_fit_central_bulk",
                "_parse_dark_wait_s",
            )
            if all(hasattr(mod, attr) for attr in required):
                print(f"[import] using raw phenomenology module: {name}")
                return mod
            errors.append(f"{name}: required helper(s) missing")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "Could not import raw/no-truth phenomenology module:\n"
        + "\n".join(errors)
    )


# =============================================================================
# DATASET SELECTION -- ONE OR MANY FILES PER WAIT CONDITION
# =============================================================================

def _cfg_file_stems(cfg):
    """
    Return all physical acquisition stems represented by one base.DATASETS entry.

    Supported base configuration styles:

        {"file_stem": "one-file"}
        {"file_stem": ("file-1", "file-2")}
        {"file_stems": ["file-1", "file-2"]}

    The spatial script expands every physical file before loading it.
    """
    if "file_stems" in cfg:
        stems = cfg.get("file_stems")
    else:
        stems = cfg.get("file_stem")

    if isinstance(stems, (list, tuple)):
        stems = list(stems)
    else:
        stems = [stems]

    stems = [str(stem) for stem in stems if stem is not None]

    if not stems:
        raise ValueError(
            f"Dataset config {cfg.get('label', '<unnamed>')} contains no file stems."
        )

    return stems


def _cfg_npz_overrides(cfg, num_files):
    """
    Resolve optional NPZ overrides for a logical base.DATASETS entry.
    """
    if "npz_path_overrides" in cfg:
        overrides = cfg.get("npz_path_overrides")
    else:
        overrides = cfg.get("npz_path_override")

    if overrides is None:
        return [None] * int(num_files)

    if isinstance(overrides, (list, tuple)):
        overrides = list(overrides)
    else:
        if int(num_files) != 1:
            raise ValueError(
                f"{cfg.get('label', '<unnamed>')}: {num_files} file stems "
                "require one npz_path_override per source file, or None."
            )
        overrides = [overrides]

    if len(overrides) != int(num_files):
        raise ValueError(
            f"{cfg.get('label', '<unnamed>')}: "
            f"{len(overrides)} NPZ overrides for {num_files} file stems."
        )

    return overrides


def _wait_from_text(text):
    """
    Parse a dark wait from common particle-memory stem/label forms.
    """
    text = str(text)

    patterns = (
        r"wait[_-](\d+(?:\.\d+)?)s",
        r"dark[_-]?wait[_-]?(\d+(?:\.\d+)?)s",
        r"source[_-]?off[_-]?(\d+(?:\.\d+)?)s",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    return np.nan


def _infer_wait_for_cfg(phen, cfg):
    """
    Infer the physical wait even when cfg contains multiple file stems and the
    imported phenomenology parser only understands a single file_stem.
    """
    try:
        wait = float(phen._parse_dark_wait_s(cfg))
        if np.isfinite(wait):
            return wait
    except Exception:
        pass

    candidates = []

    for stem in _cfg_file_stems(cfg):
        wait = _wait_from_text(stem)
        if np.isfinite(wait):
            candidates.append(float(wait))

    label_wait = _wait_from_text(cfg.get("label", ""))
    if np.isfinite(label_wait):
        candidates.append(float(label_wait))

    if not candidates:
        return np.nan

    ref = float(np.median(candidates))
    if np.max(np.abs(np.asarray(candidates) - ref)) > 1e-8:
        raise ValueError(
            f"One logical dataset config mixes different waits: {candidates}"
        )

    return ref


def _expand_cfg_to_physical_sources(cfg, wait_s):
    """
    Convert one logical config into one normal single-file config per physical
    acquisition.  Each returned config is acceptable to
    phen._load_counts_thresholds().
    """
    stems = _cfg_file_stems(cfg)
    overrides = _cfg_npz_overrides(cfg, len(stems))

    base_label = str(cfg.get("label", f"wait_{wait_s:g}s"))

    sources = []

    for source_ind, (stem, override) in enumerate(
        zip(stems, overrides)
    ):
        source_cfg = dict(cfg)

        source_cfg.pop("file_stems", None)
        source_cfg.pop("npz_path_overrides", None)

        source_cfg["file_stem"] = str(stem)
        source_cfg["npz_path_override"] = override
        source_cfg["label"] = (
            base_label
            if len(stems) == 1
            else f"{base_label}_part{source_ind + 1}"
        )

        # Bookkeeping fields used only by this spatial analysis.
        source_cfg["_spatial_source_ind"] = int(source_ind)
        source_cfg["_spatial_parent_label"] = base_label
        source_cfg["_spatial_wait_s"] = float(wait_s)

        sources.append(source_cfg)

    return sources


def _select_dataset_configs(phen):
    """
    Return ONE logical analysis group per requested wait.

    Every matching physical acquisition is retained.  This supports both:
      1. one base.DATASETS entry containing multiple file stems, and
      2. multiple separate base.DATASETS entries at the same wait.

    No longer uses "first match only".
    """
    source = getattr(getattr(phen, "base", None), "DATASETS", None)
    if not source:
        raise RuntimeError("phen.base.DATASETS is empty or unavailable.")

    groups = []

    for target in WANTED_WAITS_S:
        physical_sources = []
        parent_labels = []

        for cfg in source:
            try:
                wait = _infer_wait_for_cfg(phen, cfg)
            except Exception as exc:
                print(
                    f"[dataset selection] could not parse one config: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if not (
                np.isfinite(wait)
                and abs(float(wait) - float(target)) < 1e-8
            ):
                continue

            expanded = _expand_cfg_to_physical_sources(
                cfg,
                wait_s=float(target),
            )

            # Reindex sources across ALL configs at this same wait.
            for source_cfg in expanded:
                source_cfg["_spatial_source_ind"] = len(physical_sources)
                physical_sources.append(source_cfg)

            parent_labels.append(str(cfg.get("label", "")))

        if not physical_sources:
            message = f"No dataset found at wait={target:g} s."
            if ALLOW_MISSING_WAITS:
                print(f"[dataset selection] {message} Skipping.")
                continue
            raise RuntimeError(message)

        group_label = (
            next(
                (
                    label
                    for label in parent_labels
                    if label
                ),
                f"dark_wait_{target:g}s",
            )
        )

        print(
            f"[dataset selection] wait={target:g} s: "
            f"{len(physical_sources)} physical acquisition file(s)."
        )

        for source_ind, cfg in enumerate(physical_sources):
            print(
                f"    source {source_ind}: {cfg['file_stem']}"
            )

        groups.append(
            {
                "wait_s": float(target),
                "label": group_label,
                "source_cfgs": physical_sources,
            }
        )

    if not groups:
        raise RuntimeError("No requested datasets were found.")

    return groups


# =============================================================================
# NV COORDINATES -- SMALL METADATA ONLY
# =============================================================================

def _try_get_nv_img_xy(nv):
    for attr in (
        "pixel_coords",
        "img_coords",
        "image_coords",
        "camera_coords",
    ):
        value = getattr(nv, attr, None)
        if value is not None:
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                return float(arr[0]), float(arr[1])

    coords = getattr(nv, "coords", None)
    if isinstance(coords, dict):
        for key, value in coords.items():
            key_name = getattr(key, "name", None)
            key_text = str(key).upper()
            if (
                key_name == "PIXEL"
                or key_text == "PIXEL"
                or key_text.endswith(".PIXEL")
            ):
                arr = np.asarray(value, dtype=float).ravel()
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    return float(arr[0]), float(arr[1])

        for key in (
            "pixel",
            "pixels",
            "pixel_coords",
            "img",
            "image",
            "camera",
            "camera_coords",
        ):
            if key in coords:
                arr = np.asarray(coords[key], dtype=float).ravel()
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    return float(arr[0]), float(arr[1])

    return None


def _load_nv_list(npz_path, metadata, num_nvs):
    nv_list = None

    # Access ONLY nv_list.  img_arrays is never indexed/decompressed.
    with np.load(npz_path, allow_pickle=True) as archive:
        if "nv_list" in archive.files:
            value = archive["nv_list"]
            if isinstance(value, np.ndarray) and value.dtype == object:
                nv_list = value.tolist()
            else:
                nv_list = value

    if nv_list is None and isinstance(metadata, dict):
        nv_list = metadata.get("nv_list", None)

    if nv_list is None:
        raise ValueError(
            "Could not load nv_list; camera coordinates are unavailable."
        )

    if len(nv_list) != int(num_nvs):
        raise ValueError(
            f"nv_list length {len(nv_list)} != num_nvs {num_nvs}."
        )

    return nv_list


def _coerce_img_coords(nv_list):
    coords = []
    for nv_ind, nv in enumerate(nv_list):
        xy = _try_get_nv_img_xy(nv)
        if xy is None:
            raise ValueError(
                f"Could not obtain camera PIXEL coordinates for NV {nv_ind}."
            )
        coords.append(xy)
    return np.asarray(coords, dtype=float)


def _nearest_neighbor_distances(coords):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        delta = coords - coords[i]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        dist[i] = np.inf
        out[i] = np.min(dist)

    return out


def _coordinate_calibration_report(coords_px):
    nn_px = _nearest_neighbor_distances(coords_px)
    med_nn_px = float(np.nanmedian(nn_px))
    configured_pitch_um = med_nn_px * float(UM_PER_PIXEL)

    inferred_um_per_pixel = (
        float(EXPECTED_ARRAY_PITCH_UM) / med_nn_px
        if med_nn_px > 0
        else np.nan
    )

    print(
        f"[coordinates] median nearest-neighbor = {med_nn_px:.4f} px "
        f"= {configured_pitch_um:.4f} um with UM_PER_PIXEL={UM_PER_PIXEL:g}"
    )
    print(
        f"[coordinates] if true pitch={EXPECTED_ARRAY_PITCH_UM:g} um, "
        f"inferred scale would be {inferred_um_per_pixel:.6f} um/px"
    )

    if (
        np.isfinite(inferred_um_per_pixel)
        and inferred_um_per_pixel > 0
        and abs(UM_PER_PIXEL - inferred_um_per_pixel)
        / inferred_um_per_pixel
        > SCALE_WARNING_FRACTION
    ):
        print(
            "[coordinates] WARNING: configured um/pixel differs substantially "
            "from the scale inferred from the expected array pitch. "
            "Verify calibration before interpreting L_eff in micrometers."
        )

    coords_um = np.asarray(coords_px, dtype=float) * float(UM_PER_PIXEL)
    x_um = coords_um[:, 0]
    y_um = coords_um[:, 1]

    bbox_width_um = float(np.nanmax(x_um) - np.nanmin(x_um))
    bbox_height_um = float(np.nanmax(y_um) - np.nanmin(y_um))
    bbox_diagonal_um = float(np.hypot(bbox_width_um, bbox_height_um))

    tri_i, tri_j = np.triu_indices(len(coords_um), k=1)
    pair_delta = coords_um[tri_i] - coords_um[tri_j]
    pair_dist = np.sqrt(np.sum(pair_delta * pair_delta, axis=1))
    max_pair_distance_um = (
        float(np.nanmax(pair_dist))
        if pair_dist.size
        else np.nan
    )

    hull_area_um2 = np.nan
    hull_perimeter_um = np.nan
    hull_vertex_indices = np.asarray([], dtype=int)
    if len(coords_um) >= 3:
        try:
            hull = ConvexHull(coords_um)
            # In 2D scipy ConvexHull.volume is area and .area is perimeter.
            hull_area_um2 = float(hull.volume)
            hull_perimeter_um = float(hull.area)
            hull_vertex_indices = np.asarray(hull.vertices, dtype=int)
        except Exception:
            pass

    camera_h_px, camera_w_px = CAMERA_FRAME_SHAPE_PX
    camera_width_um = float(camera_w_px) * float(UM_PER_PIXEL)
    camera_height_um = float(camera_h_px) * float(UM_PER_PIXEL)
    camera_diagonal_um = float(
        np.hypot(camera_width_um, camera_height_um)
    )

    print(
        f"[footprint] NV bbox = {bbox_width_um:.2f} x "
        f"{bbox_height_um:.2f} um; bbox diagonal={bbox_diagonal_um:.2f} um; "
        f"max NV-NV separation={max_pair_distance_um:.2f} um"
    )
    print(
        f"[footprint] camera FOV = {camera_width_um:.2f} x "
        f"{camera_height_um:.2f} um ({camera_w_px} x {camera_h_px} px); "
        f"hull area={hull_area_um2:.1f} um^2"
    )

    return {
        "median_nn_px": med_nn_px,
        "median_nn_um_configured": configured_pitch_um,
        "pitch_inferred_um_per_pixel": inferred_um_per_pixel,
        "bbox_width_um": bbox_width_um,
        "bbox_height_um": bbox_height_um,
        "bbox_diagonal_um": bbox_diagonal_um,
        "max_pair_distance_um": max_pair_distance_um,
        "hull_area_um2": hull_area_um2,
        "hull_perimeter_um": hull_perimeter_um,
        "hull_vertex_indices": hull_vertex_indices,
        "camera_width_um": camera_width_um,
        "camera_height_um": camera_height_um,
        "camera_diagonal_um": camera_diagonal_um,
    }


# =============================================================================
# RAW SPATIAL DATASET PREPARATION
# =============================================================================

def _event_hazard(K, N, p_dark):
    K = np.asarray(K, dtype=float)
    N = np.asarray(N, dtype=float)

    p_obs = _safe_divide(K, N)
    dark_survival = max(1.0 - float(p_dark), 1e-12)
    observed_survival = np.clip(1.0 - p_obs, 1e-12, 1.0)

    ratio = observed_survival / dark_survival
    ratio = np.clip(ratio, 1e-12, 1.0)

    lam = -np.log(ratio)
    lam[~np.isfinite(p_obs)] = np.nan
    lam[p_obs <= float(p_dark)] = 0.0
    return lam


def _prepare_spatial_dataset(phen, dataset_group):
    """
    Prepare ONE combined spatial dataset for one physical wait condition.

    Multiple source acquisitions are:
      * loaded independently,
      * classified with their OWN saved NV thresholds,
      * quality-screened independently,
      * coordinate-validated against a common NV geometry,
      * concatenated along the run axis,
      * then given ONE combined p_bulk / Lambda_h / per-NV baseline.

    This avoids pretending that different files share identical thresholds,
    while still giving one statistical population per wait condition.
    """
    source_cfgs = list(dataset_group["source_cfgs"])
    requested_wait_s = float(dataset_group["wait_s"])
    group_label = str(dataset_group["label"])

    max_runs_total = None
    for key, value in MAX_RUNS_BY_WAIT.items():
        if abs(float(key) - requested_wait_s) < 1e-8:
            max_runs_total = value
            break

    if max_runs_total is not None:
        max_runs_total = int(max_runs_total)
        if max_runs_total <= 0:
            raise ValueError(
                f"MAX_RUNS_BY_WAIT[{requested_wait_s:g}] must be positive or None."
            )

    reference_coords_px = None
    reference_num_nvs = None
    reference_nv_list = None

    parts = []
    total_selected_raw_runs = 0

    for source_ind, source_cfg in enumerate(source_cfgs):
        # A total wait-level cap is consumed in acquisition order.
        if (
            max_runs_total is not None
            and total_selected_raw_runs >= max_runs_total
        ):
            print(
                f"[run selection] wait={requested_wait_s:g} s: "
                f"total cap {max_runs_total} already reached; "
                f"skipping remaining source {source_ind}."
            )
            break

        print(
            f"\n[dataset load] wait={requested_wait_s:g} s "
            f"source {source_ind + 1}/{len(source_cfgs)}"
        )
        print(f"    {source_cfg['file_stem']}")

        small = phen._load_counts_thresholds(source_cfg)

        c11 = np.asarray(small["c11"], dtype=np.float32)
        c12 = np.asarray(small["c12"], dtype=np.float32)
        thresholds = np.asarray(
            small["thresholds"],
            dtype=np.float32,
        )

        wait_s = float(small["wait_s"])
        if abs(wait_s - requested_wait_s) > 1e-6:
            raise ValueError(
                f"Source file reports wait={wait_s:g} s but was grouped "
                f"under wait={requested_wait_s:g} s."
            )

        n_nv, n_runs_original = c11.shape

        if reference_num_nvs is None:
            reference_num_nvs = int(n_nv)
        elif int(n_nv) != int(reference_num_nvs):
            raise ValueError(
                f"NV count mismatch at wait={wait_s:g} s: "
                f"reference={reference_num_nvs}, source={n_nv}."
            )

        # Apply the WAIT-LEVEL cap across the appended acquisitions, not once
        # per source file.
        n_take = int(n_runs_original)

        if max_runs_total is not None:
            remaining = max_runs_total - total_selected_raw_runs
            n_take = min(n_take, int(remaining))

        if n_take < n_runs_original:
            print(
                f"[run selection] wait={wait_s:g} s source {source_ind}: "
                f"using first {n_take}/{n_runs_original} runs."
            )

        if n_take <= 0:
            continue

        c11 = c11[:, :n_take]
        c12 = c12[:, :n_take]

        # Load only the small NV metadata/coordinate object.
        nv_list = _load_nv_list(
            small["npz_path"],
            small.get("metadata"),
            n_nv,
        )
        coords_px_this = _coerce_img_coords(nv_list)

        if reference_coords_px is None:
            reference_coords_px = coords_px_this.copy()
            reference_nv_list = nv_list
        else:
            # Same physical array may have a global camera translation between
            # acquisitions.  Remove that translation before checking topology.
            delta = coords_px_this - reference_coords_px
            global_shift = np.nanmedian(delta, axis=0)
            residual = delta - global_shift[None, :]
            residual_rms = float(
                np.sqrt(
                    np.nanmean(
                        np.sum(residual * residual, axis=1)
                    )
                )
            )

            print(
                f"[coordinates] source {source_ind}: "
                f"global shift=({global_shift[0]:.3f}, "
                f"{global_shift[1]:.3f}) px; "
                f"topology RMS={residual_rms:.4f} px"
            )

            if residual_rms > 2.0:
                raise ValueError(
                    f"NV ordering/geometry differs too much between "
                    f"same-wait acquisitions (RMS={residual_rms:.3f} px)."
                )

        # IMPORTANT: classify every source with its OWN saved thresholds.
        good, quality = phen._good_run_mask(c11, c12)

        raw = phen._classify_raw(
            c11,
            c12,
            thresholds,
            margin=getattr(
                phen,
                "RAW_MARGIN_COUNTS",
                0.0,
            ),
        )

        K_all = np.asarray(raw["loss_count"], dtype=int)
        N_all = np.asarray(
            raw["loss_evaluable_count"],
            dtype=int,
        )
        frac_all = _safe_divide(K_all, N_all)

        valid = (
            good
            & (N_all > 0)
            & np.isfinite(frac_all)
        )

        global_run_start = int(total_selected_raw_runs)
        global_run_all = (
            global_run_start
            + np.arange(n_take, dtype=int)
        )

        parts.append(
            {
                "source_ind": int(source_ind),
                "label": str(
                    source_cfg.get(
                        "label",
                        f"{group_label}_part{source_ind + 1}",
                    )
                ),
                "file_stem": str(small["file_stem"]),
                "npz_path": str(small["npz_path"]),
                "metadata": small.get("metadata"),

                "thresholds": thresholds,
                "quality": quality,

                "n_runs_original": int(n_runs_original),
                "n_runs_selected": int(n_take),

                "good": np.asarray(good, dtype=bool),
                "valid": np.asarray(valid, dtype=bool),

                "global_run_all": global_run_all,
                "local_run_all": np.arange(
                    n_take,
                    dtype=int,
                ),

                "K_all": K_all,
                "N_all": N_all,
                "frac_all": frac_all,

                "loss_all": np.asarray(
                    raw["loss"],
                    dtype=bool,
                ),
                "evaluable_all": np.asarray(
                    raw["loss_evaluable"],
                    dtype=bool,
                ),
            }
        )

        total_selected_raw_runs += int(n_take)

    if not parts:
        raise RuntimeError(
            f"No runs were loaded for wait={requested_wait_s:g} s."
        )

    # Reference geometry comes from the first source; only relative NV
    # separations are used by the spatial analysis.
    coord_report = _coordinate_calibration_report(
        reference_coords_px
    )
    coords_um = (
        reference_coords_px * float(UM_PER_PIXEL)
    )

    # ------------------------------------------------------------------
    # APPEND VALID RUNS FROM ALL ACQUISITIONS
    # ------------------------------------------------------------------
    K = np.concatenate(
        [p["K_all"][p["valid"]] for p in parts]
    )
    N = np.concatenate(
        [p["N_all"][p["valid"]] for p in parts]
    )

    loss = np.concatenate(
        [
            p["loss_all"][:, p["valid"]]
            for p in parts
        ],
        axis=1,
    )
    evaluable = np.concatenate(
        [
            p["evaluable_all"][:, p["valid"]]
            for p in parts
        ],
        axis=1,
    )

    loss_fraction = np.concatenate(
        [
            p["frac_all"][p["valid"]]
            for p in parts
        ]
    )

    # Unique global appended-run number plus original source/local identity.
    original_run = np.concatenate(
        [
            p["global_run_all"][p["valid"]]
            for p in parts
        ]
    ).astype(int)

    source_file_ind = np.concatenate(
        [
            np.full(
                int(np.sum(p["valid"])),
                p["source_ind"],
                dtype=int,
            )
            for p in parts
        ]
    )

    source_local_run = np.concatenate(
        [
            p["local_run_all"][p["valid"]]
            for p in parts
        ]
    ).astype(int)

    source_label = np.concatenate(
        [
            np.full(
                int(np.sum(p["valid"])),
                p["label"],
                dtype=object,
            )
            for p in parts
        ]
    )

    source_file_stem = np.concatenate(
        [
            np.full(
                int(np.sum(p["valid"])),
                p["file_stem"],
                dtype=object,
            )
            for p in parts
        ]
    )

    # Raw-run masks retained for bookkeeping.
    good_run_mask = np.concatenate(
        [p["good"] for p in parts]
    )
    valid_run_mask = np.concatenate(
        [p["valid"] for p in parts]
    )

    # One central bulk model for the combined physical wait condition.
    bulk = phen._fit_central_bulk(K, N)
    p_bulk = float(bulk["p"])

    lambda_h = _event_hazard(
        K,
        N,
        p_bulk,
    )
    k_excess = (
        K.astype(float)
        - N.astype(float) * p_bulk
    )

    if hasattr(
        phen,
        "_effective_particle_exposure",
    ):
        particle_exposure_s = float(
            phen._effective_particle_exposure(
                requested_wait_s
            )
        )
    else:
        particle_exposure_s = (
            requested_wait_s + 0.63
        )

    ds = {
        "label": group_label,
        "file_stem": [
            p["file_stem"] for p in parts
        ],
        "npz_path": [
            p["npz_path"] for p in parts
        ],
        "source_parts": parts,
        "num_source_files": len(parts),

        "wait_s": requested_wait_s,
        "particle_exposure_s": particle_exposure_s,

        "coords_px": reference_coords_px,
        "coords_um": coords_um,
        "coordinate_report": coord_report,

        # Thresholds remain source-specific.
        "thresholds": parts[0]["thresholds"],
        "thresholds_by_source": [
            p["thresholds"] for p in parts
        ],

        "good_run_mask": good_run_mask,
        "valid_run_mask": valid_run_mask,

        # Global appended raw-run index.
        "original_run": original_run,

        # Source identity for every VALID analysis run.
        "source_file_ind": source_file_ind,
        "source_local_run": source_local_run,
        "source_label": source_label,
        "source_file_stem": source_file_stem,

        "K": K,
        "N": N,
        "loss_fraction": loss_fraction,
        "loss": loss,
        "evaluable": evaluable,

        "bulk": bulk,
        "p_bulk": p_bulk,
        "lambda_h": lambda_h,
        "k_excess": k_excess,

        "n_runs_loaded": int(
            sum(p["n_runs_selected"] for p in parts)
        ),
    }

    print(
        f"\n[dataset combined] wait={requested_wait_s:g} s: "
        f"{len(parts)} source file(s), "
        f"good/evaluable={len(K)}/{ds['n_runs_loaded']}; "
        f"central p={100*p_bulk:.5f}%; "
        f"Lambda>=0.05: "
        f"{np.sum(lambda_h >= 0.05)} runs."
    )

    for p in parts:
        print(
            f"    source {p['source_ind']}: "
            f"selected={p['n_runs_selected']}, "
            f"valid={int(np.sum(p['valid']))}, "
            f"{p['file_stem']}"
        )

    return ds


def _source_identity_for_run(ds, run_ind):
    """
    Human/output bookkeeping for one VALID run in a combined wait dataset.
    """
    run_ind = int(run_ind)

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
# PER-NV DARK BASELINE
# =============================================================================

def _fit_per_nv_dark_baseline(ds):
    baseline_runs = np.asarray(
        ds["lambda_h"] < float(BASELINE_EXCLUDE_LAMBDA_H),
        dtype=bool,
    )

    if np.sum(baseline_runs) < 20:
        raise RuntimeError(
            f"Too few baseline runs at wait={ds['wait_s']:g} s."
        )

    loss = ds["loss"][:, baseline_runs]
    evaluable = ds["evaluable"][:, baseline_runs]

    k_i = np.sum(loss, axis=1).astype(float)
    n_i = np.sum(evaluable, axis=1).astype(float)

    p0 = float(ds["p_bulk"])
    strength = float(BASELINE_PRIOR_STRENGTH)

    p_i = (k_i + strength * p0) / (n_i + strength)
    p_i = np.clip(p_i, 1e-6, 1.0 - 1e-6)

    ds["p_i_dark"] = p_i
    ds["baseline_run_mask"] = baseline_runs
    ds["baseline_k_i"] = k_i
    ds["baseline_n_i"] = n_i

    print(
        f"[per-NV baseline] wait={ds['wait_s']:g} s: "
        f"{np.sum(baseline_runs)} runs; "
        f"median p_i={100*np.median(p_i):.4f}%."
    )

    return ds


# =============================================================================
# PAIR GEOMETRY + SPARSE ADJACENCY
# =============================================================================

def _build_pair_geometry(coords_um):
    """
    Build pair-distance bins to the full coordinate-supported NV-NV separation
    unless PAIR_MAX_DISTANCE_UM explicitly clips the diagnostic range.

    The quoted family-wise global p-value remains restricted to
    PAIR_PRIMARY_GLOBAL_TEST_MAX_DISTANCE_UM.
    """
    coords_um = np.asarray(coords_um, dtype=float)
    n_nv = len(coords_um)

    tri_i_all, tri_j_all = np.triu_indices(n_nv, k=1)
    delta_all = coords_um[tri_i_all] - coords_um[tri_j_all]
    dist_all = np.sqrt(np.sum(delta_all * delta_all, axis=1))

    finite_dist = dist_all[np.isfinite(dist_all)]
    if finite_dist.size == 0:
        raise ValueError("No finite NV-NV distances are available.")

    max_accessible_um = float(np.max(finite_dist))

    rounded_support_um = (
        math.ceil(max_accessible_um / float(PAIR_BIN_WIDTH_UM))
        * float(PAIR_BIN_WIDTH_UM)
    )

    if PAIR_MAX_DISTANCE_UM is None:
        max_analyzed_um = rounded_support_um
    else:
        max_analyzed_um = min(
            float(PAIR_MAX_DISTANCE_UM),
            rounded_support_um,
        )

    max_analyzed_um = max(
        max_analyzed_um,
        float(PAIR_BIN_WIDTH_UM),
    )

    edges = np.arange(
        0.0,
        max_analyzed_um + float(PAIR_BIN_WIDTH_UM) + 1e-12,
        float(PAIR_BIN_WIDTH_UM),
    )

    bin_index_all = np.digitize(dist_all, edges) - 1
    valid = (
        (bin_index_all >= 0)
        & (bin_index_all < len(edges) - 1)
        & np.isfinite(dist_all)
    )

    tri_i = tri_i_all[valid]
    tri_j = tri_j_all[valid]
    bin_index = bin_index_all[valid]

    adjacency = []
    pair_counts = []

    for b in range(len(edges) - 1):
        m = bin_index == b
        i = tri_i[m]
        j = tri_j[m]

        rows = np.concatenate([i, j])
        cols = np.concatenate([j, i])
        data = np.ones(len(rows), dtype=float)

        A = coo_matrix(
            (data, (rows, cols)),
            shape=(n_nv, n_nv),
        ).tocsr()

        adjacency.append(A)
        pair_counts.append(int(np.sum(m)))

    pair_counts = np.asarray(pair_counts, dtype=int)
    centers = 0.5 * (edges[:-1] + edges[1:])

    primary_test_mask = (
        (edges[1:] <= float(PAIR_PRIMARY_GLOBAL_TEST_MAX_DISTANCE_UM) + 1e-12)
        & (pair_counts > 0)
    )

    total_possible_pairs = int(n_nv * (n_nv - 1) // 2)

    return {
        "edges_um": edges,
        "centers_um": centers,
        "pair_counts_geometry": pair_counts,
        "adjacency": adjacency,
        "max_accessible_distance_um": max_accessible_um,
        "max_analyzed_distance_um": float(edges[-1]),
        "primary_test_mask": primary_test_mask,
        "primary_global_test_max_distance_um": float(
            PAIR_PRIMARY_GLOBAL_TEST_MAX_DISTANCE_UM
        ),
        "total_possible_pairs": total_possible_pairs,
        "fraction_of_all_pairs_by_bin": (
            pair_counts.astype(float) / max(total_possible_pairs, 1)
        ),
    }


# =============================================================================
# DISTANCE-BINNED RESIDUAL CORRELATION
# =============================================================================

def _pair_correlation_from_matrices(
    residual_matrix,
    evaluable_matrix,
    pair_geometry,
):
    """
    Correlate event-by-event residuals in distance bins.

    residual_matrix: shape (n_nv, n_events)
    evaluable_matrix: shape (n_nv, n_events), bool/0-1

    Non-evaluable sites must have zero residual.
    """
    R = np.asarray(residual_matrix, dtype=float)
    E = np.asarray(evaluable_matrix, dtype=float)

    numerators = []
    denominators = []

    for A in pair_geometry["adjacency"]:
        # A is symmetric, so every unordered pair appears twice.
        num = 0.5 * float(np.sum(R * (A @ R)))
        den = 0.5 * float(np.sum(E * (A @ E)))

        numerators.append(num)
        denominators.append(den)

    numerators = np.asarray(numerators, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    C = _safe_divide(numerators, denominators)

    return {
        "C": C,
        "numerator": numerators,
        "denominator_pairs": denominators,
    }


def _conditional_bernoulli_log_dp(probabilities, k):
    """
    Dynamic-programming table for exact conditional Bernoulli sampling.

    Independent Bernoulli probabilities p_i conditioned on exactly K=k
    successes imply

        P(S | |S|=k) proportional to prod_{i in S} w_i,

    with odds w_i = p_i/(1-p_i).

    The DP stores log elementary-symmetric sums, allowing exact backward
    sampling without rejection.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-10, 1.0 - 1e-10)
    n = len(p)
    k = int(k)

    if k < 0 or k > n:
        raise ValueError(f"Invalid fixed-K request: k={k}, n={n}.")

    logw = np.log(p) - np.log1p(-p)

    dp = np.full((n + 1, k + 1), -np.inf, dtype=float)
    dp[:, 0] = 0.0

    for j in range(1, n + 1):
        upper = min(j, k)
        lw = logw[j - 1]

        for m in range(1, upper + 1):
            dp[j, m] = np.logaddexp(
                dp[j - 1, m],
                lw + dp[j - 1, m - 1],
            )

    if not np.isfinite(dp[n, k]):
        raise RuntimeError(
            "Conditional Bernoulli DP became non-finite for "
            f"n={n}, k={k}."
        )

    return logw, dp


def _sample_conditional_bernoulli_fixed_k(
    logw,
    dp,
    k,
    rng,
):
    """
    Exact sample from independent Bernoulli variables conditioned on sum=K.
    """
    n = len(logw)
    remaining = int(k)
    selected = np.zeros(n, dtype=bool)

    for j in range(n, 0, -1):
        if remaining == 0:
            break

        if remaining == j:
            selected[:j] = True
            remaining = 0
            break

        log_num = logw[j - 1] + dp[j - 1, remaining - 1]
        log_den = dp[j, remaining]

        if not np.isfinite(log_num):
            p_include = 0.0
        else:
            p_include = math.exp(min(0.0, log_num - log_den))
            p_include = float(np.clip(p_include, 0.0, 1.0))

        if rng.random() < p_include:
            selected[j - 1] = True
            remaining -= 1

    if remaining != 0:
        raise RuntimeError(
            "Conditional Bernoulli sampler did not place the requested "
            f"number of successes; remaining={remaining}."
        )

    return selected


def _prepare_uniform_event_payload(ds, run_ind):
    """
    Fit the global/uniform event amplitude for one selected run and return
    the residual vector used by the spatial correlation test.
    """
    eligible = np.asarray(ds["evaluable"][:, run_ind], dtype=bool)
    y_full = np.asarray(ds["loss"][:, run_ind], dtype=bool)

    inds = np.where(eligible)[0]
    if inds.size < 2:
        return None

    y = y_full[inds].astype(float)
    p_dark = np.asarray(ds["p_i_dark"][inds], dtype=float)

    # _fit_uniform_model ignores coords, but pass a correctly shaped dummy.
    dummy_coords = np.zeros((inds.size, 2), dtype=float)
    fit = _fit_uniform_model(dummy_coords, y, p_dark)

    A = float(fit["A"])
    lambda_uniform = np.full(inds.size, A, dtype=float)
    p_uniform = _event_probability_from_lambda(
        p_dark,
        lambda_uniform,
    )

    residual_full = np.zeros(len(ds["p_i_dark"]), dtype=float)
    eval_full = np.zeros(len(ds["p_i_dark"]), dtype=float)

    residual_full[inds] = y - p_uniform
    eval_full[inds] = 1.0

    k_obs = int(np.sum(y))
    logw, dp = _conditional_bernoulli_log_dp(
        p_uniform,
        k_obs,
    )

    return {
        "run_ind": int(run_ind),
        "original_run": int(ds["original_run"][run_ind]),
        **_source_identity_for_run(ds, run_ind),
        "eligible_indices": inds,
        "K_observed": k_obs,
        "A_uniform": A,
        "p_uniform": p_uniform,
        "observed_residual_full": residual_full,
        "evaluable_full": eval_full,
        "conditional_logw": logw,
        "conditional_dp": dp,
    }


def _pair_correlation_with_null(ds, cut, pair_geometry):
    """
    Uniform-event-residual spatial correlation with an exact fixed-K null.

    This is the primary spatial-correlation test.

    It deliberately removes the run's global event amplitude before looking
    for distance-dependent structure.
    """
    selected = np.where(ds["lambda_h"] >= float(cut))[0]

    if selected.size < MIN_EVENTS_FOR_PAIR_CORRELATION:
        return {
            "selected_run_inds": selected,
            "event_payloads": [],
            "observed": None,
            "null_type": "fixed_K_conditional_uniform_event",
        }

    payloads = []
    for run_ind in selected:
        payload = _prepare_uniform_event_payload(
            ds,
            int(run_ind),
        )
        if payload is not None:
            payloads.append(payload)

    if len(payloads) < MIN_EVENTS_FOR_PAIR_CORRELATION:
        return {
            "selected_run_inds": selected,
            "event_payloads": payloads,
            "observed": None,
            "null_type": "fixed_K_conditional_uniform_event",
        }

    residual_obs = np.column_stack(
        [p["observed_residual_full"] for p in payloads]
    )
    eval_matrix = np.column_stack(
        [p["evaluable_full"] for p in payloads]
    )

    observed = _pair_correlation_from_matrices(
        residual_obs,
        eval_matrix,
        pair_geometry,
    )

    rng = np.random.default_rng(
        PAIR_NULL_SEED
        + int(round(ds["wait_s"] * 100))
        + int(round(cut * 1e5))
    )

    n_bins = len(pair_geometry["centers_um"])
    null_C = np.full(
        (PAIR_NULL_SCRAMBLES, n_bins),
        np.nan,
        dtype=float,
    )

    n_nv = len(ds["p_i_dark"])
    n_events = len(payloads)

    for rep in range(PAIR_NULL_SCRAMBLES):
        residual_null = np.zeros(
            (n_nv, n_events),
            dtype=float,
        )

        for e, payload in enumerate(payloads):
            chosen = _sample_conditional_bernoulli_fixed_k(
                payload["conditional_logw"],
                payload["conditional_dp"],
                payload["K_observed"],
                rng,
            )

            inds = payload["eligible_indices"]
            p_uniform = payload["p_uniform"]

            residual_null[inds, e] = (
                chosen.astype(float) - p_uniform
            )

        result = _pair_correlation_from_matrices(
            residual_null,
            eval_matrix,
            pair_geometry,
        )
        null_C[rep, :] = result["C"]

    null_mean = np.nanmean(null_C, axis=0)
    null_std = np.nanstd(null_C, axis=0, ddof=1)
    null_lo = np.nanpercentile(null_C, 2.5, axis=0)
    null_hi = np.nanpercentile(null_C, 97.5, axis=0)

    z = _safe_divide(
        observed["C"] - null_mean,
        null_std,
    )

    # Per-bin empirical upper-tail p.
    p_upper = np.full(n_bins, np.nan, dtype=float)
    for b in range(n_bins):
        vals = null_C[:, b]
        vals = vals[np.isfinite(vals)]

        if vals.size and np.isfinite(observed["C"][b]):
            p_upper[b] = (
                1.0 + np.sum(vals >= observed["C"][b])
            ) / (1.0 + vals.size)

    # Approximate family-wise correction for having searched multiple distance
    # bins: compare the largest observed standardized excess with the largest
    # standardized excess in each null replicate.
    valid_bins_all = (
        np.isfinite(z)
        & np.isfinite(null_std)
        & (null_std > 0)
        & (pair_geometry["pair_counts_geometry"] > 0)
    )

    if np.any(valid_bins_all):
        all_inds = np.where(valid_bins_all)[0]
        full_range_max_bin = int(
            all_inds[np.nanargmax(z[valid_bins_all])]
        )
        full_range_max_z_observed = float(
            z[full_range_max_bin]
        )
    else:
        full_range_max_bin = -1
        full_range_max_z_observed = np.nan

    # Primary significance remains on the previously used distance-search
    # range; the extra long-distance bins are diagnostic only.
    valid_bins = (
        valid_bins_all
        & np.asarray(
            pair_geometry["primary_test_mask"],
            dtype=bool,
        )
    )

    if np.any(valid_bins):
        max_z_observed = float(np.nanmax(z[valid_bins]))

        null_z = (
            null_C[:, valid_bins]
            - null_mean[valid_bins][None, :]
        ) / null_std[valid_bins][None, :]

        null_max_z = np.nanmax(null_z, axis=1)

        global_max_p = float(
            (
                1.0
                + np.sum(null_max_z >= max_z_observed)
            )
            / (1.0 + len(null_max_z))
        )
    else:
        max_z_observed = np.nan
        null_max_z = np.asarray([], dtype=float)
        global_max_p = np.nan

    return {
        "selected_run_inds": selected,
        "event_payloads": payloads,
        "observed": observed,

        "null_type": "fixed_K_conditional_uniform_event",
        "null_C": null_C,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_lo": null_lo,
        "null_hi": null_hi,

        "z": z,
        "p_upper": p_upper,

        "max_z_observed": max_z_observed,
        "null_max_z": null_max_z,
        "global_max_p": global_max_p,
        "full_range_max_bin": full_range_max_bin,
        "full_range_max_z_observed": full_range_max_z_observed,
    }


# =============================================================================
# BERNOULLI SPATIAL MODEL FITS
# =============================================================================

def _event_probability_from_lambda(p_dark_i, lambda_i):
    p_dark_i = np.asarray(p_dark_i, dtype=float)
    lambda_i = np.asarray(lambda_i, dtype=float)

    prob = 1.0 - (1.0 - p_dark_i) * np.exp(-np.clip(lambda_i, 0.0, 100.0))
    return np.clip(prob, 1e-10, 1.0 - 1e-10)


def _bernoulli_loglike(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1.0 - 1e-10)

    return float(
        np.sum(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        )
    )


def _aic_bic(loglike, num_params, n_obs):
    aic = 2.0 * num_params - 2.0 * float(loglike)
    bic = math.log(max(int(n_obs), 1)) * num_params - 2.0 * float(loglike)
    return float(aic), float(bic)


def _fit_uniform_model(coords, y, p_dark):
    del coords

    def objective(logA):
        A = math.exp(float(logA))
        lam = np.full(len(y), A, dtype=float)
        p = _event_probability_from_lambda(p_dark, lam)
        return -_bernoulli_loglike(y, p)

    opt = minimize_scalar(
        objective,
        bounds=(math.log(A_MIN), math.log(A_MAX)),
        method="bounded",
    )

    A = math.exp(float(opt.x))
    ll = -float(opt.fun)
    aic, bic = _aic_bic(ll, 1, len(y))

    return {
        "model": "uniform",
        "loglike": ll,
        "num_params": 1,
        "A": A,
        "L_eff_um": np.nan,
        "x0_um": np.nan,
        "y0_um": np.nan,
        "theta_rad": np.nan,
        "offset_um": np.nan,
        "AIC": aic,
        "BIC": bic,
        "success": bool(opt.success),
    }


def _point_lambda(coords, A, x0, y0, L):
    d = np.sqrt(
        (coords[:, 0] - float(x0)) ** 2
        + (coords[:, 1] - float(y0)) ** 2
    )
    return float(A) * np.exp(-d / float(L))


def _fit_point_model(coords, y, p_dark, rng):
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)

    span = max(xmax - xmin, ymax - ymin)
    pad = 0.20 * span

    loss_coords = coords[np.asarray(y, dtype=bool)]
    if len(loss_coords):
        centroid = np.mean(loss_coords, axis=0)
    else:
        centroid = np.mean(coords, axis=0)

    starts = []
    for L0 in (3.0, 10.0, 30.0, 80.0):
        starts.append(
            [
                math.log(0.08),
                centroid[0],
                centroid[1],
                math.log(np.clip(L0, L_EFF_MIN_UM, L_EFF_MAX_UM)),
            ]
        )

    for _ in range(POINT_RANDOM_STARTS):
        starts.append(
            [
                math.log(rng.uniform(0.02, 0.20)),
                rng.uniform(xmin, xmax),
                rng.uniform(ymin, ymax),
                math.log(
                    math.exp(
                        rng.uniform(
                            math.log(L_EFF_MIN_UM),
                            math.log(L_EFF_MAX_UM),
                        )
                    )
                ),
            ]
        )

    bounds = [
        (math.log(A_MIN), math.log(A_MAX)),
        (xmin - pad, xmax + pad),
        (ymin - pad, ymax + pad),
        (math.log(L_EFF_MIN_UM), math.log(L_EFF_MAX_UM)),
    ]

    def objective(x):
        logA, x0, y0, logL = map(float, x)
        A = math.exp(logA)
        L = math.exp(logL)
        lam = _point_lambda(coords, A, x0, y0, L)
        p = _event_probability_from_lambda(p_dark, lam)
        return -_bernoulli_loglike(y, p)

    best = None
    for start in starts:
        opt = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or opt.fun < best.fun:
            best = opt

    logA, x0, y0, logL = map(float, best.x)
    A = math.exp(logA)
    L = math.exp(logL)
    ll = -float(best.fun)
    aic, bic = _aic_bic(ll, 4, len(y))

    return {
        "model": "point_exp",
        "loglike": ll,
        "num_params": 4,
        "A": A,
        "L_eff_um": L,
        "x0_um": x0,
        "y0_um": y0,
        "theta_rad": np.nan,
        "offset_um": np.nan,
        "AIC": aic,
        "BIC": bic,
        "success": bool(best.success),
    }


def _line_distance(coords, theta, offset):
    center = np.mean(coords, axis=0)
    centered = coords - center

    nx = math.cos(float(theta))
    ny = math.sin(float(theta))

    projection = centered[:, 0] * nx + centered[:, 1] * ny
    return np.abs(projection - float(offset))


def _line_lambda(coords, A, L, theta, offset):
    d = _line_distance(coords, theta, offset)
    return float(A) * np.exp(-d / float(L))


def _fit_line_model(coords, y, p_dark, rng):
    center = np.mean(coords, axis=0)
    centered = coords - center
    diag = float(
        np.sqrt(
            (np.max(coords[:, 0]) - np.min(coords[:, 0])) ** 2
            + (np.max(coords[:, 1]) - np.min(coords[:, 1])) ** 2
        )
    )

    starts = []

    for theta0 in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
        starts.append(
            [
                math.log(0.08),
                math.log(20.0),
                theta0,
                0.0,
            ]
        )

    for _ in range(LINE_RANDOM_STARTS):
        starts.append(
            [
                math.log(rng.uniform(0.02, 0.20)),
                rng.uniform(
                    math.log(L_EFF_MIN_UM),
                    math.log(L_EFF_MAX_UM),
                ),
                rng.uniform(0.0, math.pi),
                rng.uniform(-0.5 * diag, 0.5 * diag),
            ]
        )

    bounds = [
        (math.log(A_MIN), math.log(A_MAX)),
        (math.log(L_EFF_MIN_UM), math.log(L_EFF_MAX_UM)),
        (0.0, math.pi),
        (-diag, diag),
    ]

    def objective(x):
        logA, logL, theta, offset = map(float, x)
        A = math.exp(logA)
        L = math.exp(logL)

        lam = _line_lambda(
            coords,
            A,
            L,
            theta,
            offset,
        )
        p = _event_probability_from_lambda(p_dark, lam)
        return -_bernoulli_loglike(y, p)

    best = None
    for start in starts:
        opt = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or opt.fun < best.fun:
            best = opt

    logA, logL, theta, offset = map(float, best.x)
    A = math.exp(logA)
    L = math.exp(logL)
    ll = -float(best.fun)
    aic, bic = _aic_bic(ll, 4, len(y))

    return {
        "model": "line_exp_proxy",
        "loglike": ll,
        "num_params": 4,
        "A": A,
        "L_eff_um": L,
        "x0_um": np.nan,
        "y0_um": np.nan,
        "theta_rad": theta,
        "offset_um": offset,
        "AIC": aic,
        "BIC": bic,
        "success": bool(best.success),
    }


def _fit_event_spatial_models(ds, run_ind, cut, rng):
    eligible = ds["evaluable"][:, run_ind]
    if np.sum(eligible) < 20:
        return None

    coords = ds["coords_um"][eligible]
    y = ds["loss"][:, run_ind][eligible].astype(float)
    p_dark = ds["p_i_dark"][eligible]

    models = []

    if FIT_UNIFORM_MODEL:
        models.append(_fit_uniform_model(coords, y, p_dark))
    if FIT_POINT_MODEL:
        models.append(_fit_point_model(coords, y, p_dark, rng))
    if FIT_LINE_MODEL:
        models.append(_fit_line_model(coords, y, p_dark, rng))

    models = sorted(models, key=lambda m: m["AIC"])
    best_aic = models[0]["AIC"]
    best_bic = min(m["BIC"] for m in models)

    uniform = next(
        (m for m in models if m["model"] == "uniform"),
        None,
    )

    rows = []
    for m in models:
        row = {
            "dataset": ds["label"],
            "dark_wait_s": float(ds["wait_s"]),
            "particle_exposure_s": float(ds["particle_exposure_s"]),
            "lambda_cut": float(cut),
            "valid_run_ind": int(run_ind),
            "original_run": int(ds["original_run"][run_ind]),
            **_source_identity_for_run(ds, run_ind),
            "K_loss": int(ds["K"][run_ind]),
            "N_evaluable": int(ds["N"][run_ind]),
            "K_excess": float(ds["k_excess"][run_ind]),
            "Lambda_h": float(ds["lambda_h"][run_ind]),

            **m,

            "delta_AIC": float(m["AIC"] - best_aic),
            "delta_BIC": float(m["BIC"] - best_bic),
        }

        if uniform is not None:
            row["AIC_improvement_vs_uniform"] = float(
                uniform["AIC"] - m["AIC"]
            )
        else:
            row["AIC_improvement_vs_uniform"] = np.nan

        rows.append(row)

    best = models[0]
    if best["model"] == "uniform":
        classification = "uniform_or_unresolved"
    else:
        improvement = (
            uniform["AIC"] - best["AIC"]
            if uniform is not None
            else np.nan
        )
        if (
            np.isfinite(improvement)
            and improvement >= LOCALIZED_DELTA_AIC_THRESHOLD
        ):
            classification = (
                "point_localized_preferred"
                if best["model"] == "point_exp"
                else "line_like_preferred"
            )
        else:
            classification = "localized_not_decisive"

    summary = {
        "dataset": ds["label"],
        "dark_wait_s": float(ds["wait_s"]),
        "lambda_cut": float(cut),
        "valid_run_ind": int(run_ind),
        "original_run": int(ds["original_run"][run_ind]),
        **_source_identity_for_run(ds, run_ind),
        "K_loss": int(ds["K"][run_ind]),
        "N_evaluable": int(ds["N"][run_ind]),
        "K_excess": float(ds["k_excess"][run_ind]),
        "Lambda_h": float(ds["lambda_h"][run_ind]),

        "best_model": best["model"],
        "best_AIC": float(best["AIC"]),
        "best_BIC": float(best["BIC"]),
        "best_L_eff_um": float(best["L_eff_um"])
        if np.isfinite(best["L_eff_um"])
        else np.nan,

        "classification": classification,

        "uniform_AIC": float(uniform["AIC"])
        if uniform is not None
        else np.nan,

        "best_AIC_improvement_vs_uniform": (
            float(uniform["AIC"] - best["AIC"])
            if uniform is not None
            else np.nan
        ),
    }

    return rows, summary


# =============================================================================
# EVENT MAPS
# =============================================================================

def _model_probability_map(ds, run_ind, model_row):
    eligible = ds["evaluable"][:, run_ind]
    coords = ds["coords_um"][eligible]
    p_dark = ds["p_i_dark"][eligible]

    model = model_row["model"]

    if model == "uniform":
        lam = np.full(
            len(coords),
            float(model_row["A"]),
            dtype=float,
        )
    elif model == "point_exp":
        lam = _point_lambda(
            coords,
            model_row["A"],
            model_row["x0_um"],
            model_row["y0_um"],
            model_row["L_eff_um"],
        )
    elif model == "line_exp_proxy":
        lam = _line_lambda(
            coords,
            model_row["A"],
            model_row["L_eff_um"],
            model_row["theta_rad"],
            model_row["offset_um"],
        )
    else:
        return eligible, np.full(len(coords), np.nan)

    p = _event_probability_from_lambda(p_dark, lam)
    return eligible, p


def _plot_event_map(ds, run_ind, summary, model_rows):
    eligible = ds["evaluable"][:, run_ind]
    lost = ds["loss"][:, run_ind]
    coords = ds["coords_um"]

    best_row = next(
        row for row in model_rows
        if row["model"] == summary["best_model"]
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.7))

    ax.scatter(
        coords[eligible, 0],
        coords[eligible, 1],
        s=16,
        alpha=0.28,
        label="evaluable NVs",
    )

    ax.scatter(
        coords[lost, 0],
        coords[lost, 1],
        s=44,
        marker="o",
        facecolors="none",
        linewidths=1.4,
        label="NV- -> NV0",
    )

    if summary["best_model"] == "point_exp":
        ax.scatter(
            [best_row["x0_um"]],
            [best_row["y0_um"]],
            marker="x",
            s=90,
            linewidths=2.0,
            label="best point source",
        )

    elif summary["best_model"] == "line_exp_proxy":
        theta = float(best_row["theta_rad"])
        offset = float(best_row["offset_um"])

        center = np.mean(coords[eligible], axis=0)
        tangent = np.array([-math.sin(theta), math.cos(theta)])
        normal = np.array([math.cos(theta), math.sin(theta)])
        line_center = center + offset * normal

        span = max(
            np.ptp(coords[eligible, 0]),
            np.ptp(coords[eligible, 1]),
        )
        p1 = line_center - 1.2 * span * tangent
        p2 = line_center + 1.2 * span * tangent

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            linestyle="--",
            linewidth=1.4,
            label="best line proxy",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")

    identity = _source_identity_for_run(ds, run_ind)

    title = (
        f"{ds['wait_s']:g} s | global run {ds['original_run'][run_ind]} | "
        f"source {identity['source_file_ind']} local run "
        f"{identity['source_local_run']} | "
        f"K={ds['K'][run_ind]} | Lambda={ds['lambda_h'][run_ind]:.4f}\n"
        f"best={summary['best_model']} | "
        f"dAIC vs uniform={summary['best_AIC_improvement_vs_uniform']:.2f}"
    )
    ax.set_title(title)
    ax.legend(fontsize=8)

    if SAVE_OUTPUTS:
        path = (
            EVENT_MAP_DIR
            / (
                f"event_{ds['wait_s']:g}s_run_"
                f"{int(ds['original_run'][run_ind]):04d}.png"
            )
        )
        fig.savefig(path, dpi=190, bbox_inches="tight")

    if CLOSE_EVENT_MAPS_IMMEDIATELY:
        plt.close(fig)
        return None

    return fig



# =============================================================================
# LEAVE-ONE-EVENT-OUT SPATIAL-CORRELATION JACKKNIFE
# =============================================================================

def _jackknife_pair_correlation(ds, cut, result, pair_geometry):
    """
    Recompute the observed class-level uniform-event-residual C_U(d) after
    removing each selected event one at a time.

    The expensive fixed-K null is NOT rerun for every leave-one-out sample.
    Instead, each jackknife curve is expressed relative to the full-sample
    null mean/std.  This is a robustness diagnostic, not a new formal p-value.

    The main question is:
        Does the spatial feature survive removal of every single event?
    """
    if (
        result is None
        or result.get("observed") is None
        or len(result.get("event_payloads", [])) < 3
    ):
        return {
            "rows": [],
            "curves": np.empty((0, len(pair_geometry["centers_um"]))),
            "peak_bin": None,
        }

    payloads = result["event_payloads"]
    full_C = np.asarray(result["observed"]["C"], dtype=float)
    full_z = np.asarray(result["z"], dtype=float)
    null_mean = np.asarray(result["null_mean"], dtype=float)
    null_std = np.asarray(result["null_std"], dtype=float)

    valid = (
        np.isfinite(full_z)
        & np.isfinite(full_C)
        & np.isfinite(null_std)
        & (null_std > 0)
    )
    if not np.any(valid):
        return {
            "rows": [],
            "curves": np.empty((0, len(full_C))),
            "peak_bin": None,
        }

    valid_bins = np.where(valid)[0]
    peak_bin = int(valid_bins[np.argmax(full_z[valid_bins])])

    full_peak_C = float(full_C[peak_bin])
    full_peak_z = float(full_z[peak_bin])

    rows = []
    curves = []

    for omit_index, omitted in enumerate(payloads):
        kept = [
            payload
            for j, payload in enumerate(payloads)
            if j != omit_index
        ]
        if len(kept) < 2:
            continue

        R = np.column_stack(
            [payload["observed_residual_full"] for payload in kept]
        )
        E = np.column_stack(
            [payload["evaluable_full"] for payload in kept]
        )

        jk = _pair_correlation_from_matrices(
            R,
            E,
            pair_geometry,
        )
        C = np.asarray(jk["C"], dtype=float)
        curves.append(C)

        z_vs_full_null = _safe_divide(
            C - null_mean,
            null_std,
        )

        valid_jk = np.isfinite(z_vs_full_null) & valid
        if np.any(valid_jk):
            jk_valid_bins = np.where(valid_jk)[0]
            max_bin = int(
                jk_valid_bins[
                    np.argmax(z_vs_full_null[jk_valid_bins])
                ]
            )
            max_z = float(z_vs_full_null[max_bin])
        else:
            max_bin = -1
            max_z = np.nan

        peak_C = float(C[peak_bin])
        peak_z = float(z_vs_full_null[peak_bin])

        rows.append({
            "dataset": ds["label"],
            "dark_wait_s": float(ds["wait_s"]),
            "lambda_cut": float(cut),

            "num_events_full": len(payloads),
            "num_events_after_omit": len(kept),

            "omitted_original_run": int(
                omitted["original_run"]
            ),
            "omitted_source_file_ind": int(
                omitted.get("source_file_ind", -1)
            ),
            "omitted_source_local_run": int(
                omitted.get("source_local_run", -1)
            ),
            "omitted_source_label": str(
                omitted.get("source_label", "")
            ),
            "omitted_valid_run_ind": int(
                omitted["run_ind"]
            ),
            "omitted_K": int(
                omitted["K_observed"]
            ),
            "omitted_Lambda_h": float(
                ds["lambda_h"][omitted["run_ind"]]
            ),

            "full_peak_distance_low_um": float(
                pair_geometry["edges_um"][peak_bin]
            ),
            "full_peak_distance_high_um": float(
                pair_geometry["edges_um"][peak_bin + 1]
            ),
            "full_peak_C": full_peak_C,
            "full_peak_z": full_peak_z,

            "jackknife_peak_C_same_bin": peak_C,
            "jackknife_peak_z_vs_full_null_same_bin": peak_z,

            "jackknife_peak_C_fraction_of_full": (
                peak_C / full_peak_C
                if np.isfinite(full_peak_C)
                and abs(full_peak_C) > 1e-15
                else np.nan
            ),

            "jackknife_max_z_vs_full_null": max_z,
            "jackknife_max_bin_low_um": (
                float(pair_geometry["edges_um"][max_bin])
                if max_bin >= 0
                else np.nan
            ),
            "jackknife_max_bin_high_um": (
                float(pair_geometry["edges_um"][max_bin + 1])
                if max_bin >= 0
                else np.nan
            ),
        })

    curves = (
        np.asarray(curves, dtype=float)
        if curves
        else np.empty((0, len(full_C)))
    )

    return {
        "rows": rows,
        "curves": curves,
        "peak_bin": peak_bin,
        "full_C": full_C,
        "full_z": full_z,
        "null_mean": null_mean,
        "null_std": null_std,
    }


def _plot_jackknife_pair_correlation(
    ds,
    cut,
    result,
    pair_geometry,
    jackknife,
):
    if not jackknife["rows"]:
        return None

    x = np.asarray(pair_geometry["centers_um"], dtype=float)
    curves = np.asarray(jackknife["curves"], dtype=float)

    lo = np.nanmin(curves, axis=0)
    hi = np.nanmax(curves, axis=0)
    med = np.nanmedian(curves, axis=0)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    ax.fill_between(
        x,
        lo,
        hi,
        alpha=0.20,
        label="leave-one-event-out envelope",
    )
    ax.plot(
        x,
        med,
        linestyle="--",
        linewidth=1.3,
        label="jackknife median",
    )
    ax.plot(
        x,
        jackknife["full_C"],
        marker="o",
        linewidth=1.5,
        label="all selected events",
    )

    ax.axhline(0.0, linewidth=0.9)
    ax.set_xlabel("NV-NV separation (um)")
    ax.set_ylabel(
        r"Uniform-event residual correlation $C_U(d)$"
    )
    ax.set_title(
        f"Leave-one-event-out robustness | "
        f"{ds['wait_s']:g} s | Lambda >= {cut:.3f}"
    )
    ax.legend(fontsize=8)

    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR
            / (
                f"jackknife_pair_correlation_{ds['wait_s']:g}s_"
                f"{_cut_token(cut)}.png"
            ),
            dpi=190,
            bbox_inches="tight",
        )

    return fig


# =============================================================================
# PROJECTED LINE-TRAJECTORY INFERENCE
# =============================================================================

def _wrap_period_pi(angle):
    """
    Wrap an unoriented line angle into [0, pi).
    """
    return float(angle % math.pi)


def _signed_period_pi_difference(angle, reference):
    """
    Small signed angular difference for an unoriented line, in [-pi/2, pi/2).

    Returns:
        delta_angle, offset_sign

    offset_sign tells how a line offset must be transformed when the normal
    direction is flipped by pi to align with the reference orientation.
    """
    raw = float(angle) - float(reference)
    sign = 1.0

    while raw >= math.pi / 2.0:
        raw -= math.pi
        sign *= -1.0

    while raw < -math.pi / 2.0:
        raw += math.pi
        sign *= -1.0

    return float(raw), float(sign)


def _line_track_angle_from_normal(theta_normal):
    """
    _fit_line_model parameterizes the line by its NORMAL angle theta.
    The projected trajectory/tangent direction is theta + pi/2 modulo pi.
    """
    return _wrap_period_pi(
        float(theta_normal) + math.pi / 2.0
    )


def _line_center_xy(coords, theta_normal, offset):
    """
    Return one canonical point on the fitted line.

    offset is defined relative to the mean coordinate of the evaluable NV set.
    """
    center = np.mean(coords, axis=0)
    normal = np.array(
        [
            math.cos(float(theta_normal)),
            math.sin(float(theta_normal)),
        ],
        dtype=float,
    )
    return center + float(offset) * normal


def _track_coordinates(
    coords,
    theta_normal,
    offset,
):
    """
    Rotate coordinates into:
        s_parallel    : coordinate along the projected trajectory,
        d_perp_signed: signed distance from the fitted line.
    """
    coords = np.asarray(coords, dtype=float)
    center = np.mean(coords, axis=0)
    centered = coords - center

    normal = np.array(
        [
            math.cos(float(theta_normal)),
            math.sin(float(theta_normal)),
        ],
        dtype=float,
    )
    tangent = np.array(
        [
            -math.sin(float(theta_normal)),
            math.cos(float(theta_normal)),
        ],
        dtype=float,
    )

    d_perp_signed = centered @ normal - float(offset)
    s_parallel = centered @ tangent

    return s_parallel, d_perp_signed


def _line_fit_local_bootstrap(
    coords,
    y,
    p_dark,
    best_row,
    rng,
):
    """
    Fast local line-model refit for one bootstrap sample.

    Uses the original best fit plus small perturbations rather than the much
    larger global multi-start search used for the original event.
    """
    center = np.mean(coords, axis=0)
    del center

    diag = float(
        np.sqrt(
            np.ptp(coords[:, 0]) ** 2
            + np.ptp(coords[:, 1]) ** 2
        )
    )

    base_A = float(best_row["A"])
    base_L = float(best_row["L_eff_um"])
    base_theta = float(best_row["theta_rad"])
    base_offset = float(best_row["offset_um"])

    starts = [
        [
            math.log(np.clip(base_A, A_MIN, A_MAX)),
            math.log(
                np.clip(
                    base_L,
                    L_EFF_MIN_UM,
                    L_EFF_MAX_UM,
                )
            ),
            _wrap_period_pi(base_theta),
            np.clip(base_offset, -diag, diag),
        ]
    ]

    for _ in range(
        max(0, int(TRAJECTORY_BOOTSTRAP_LOCAL_STARTS) - 1)
    ):
        starts.append([
            math.log(
                np.clip(
                    base_A
                    * math.exp(rng.normal(0.0, 0.25)),
                    A_MIN,
                    A_MAX,
                )
            ),
            math.log(
                np.clip(
                    base_L
                    * math.exp(rng.normal(0.0, 0.30)),
                    L_EFF_MIN_UM,
                    L_EFF_MAX_UM,
                )
            ),
            _wrap_period_pi(
                base_theta + rng.normal(0.0, 0.15)
            ),
            np.clip(
                base_offset + rng.normal(0.0, 5.0),
                -diag,
                diag,
            ),
        ])

    bounds = [
        (math.log(A_MIN), math.log(A_MAX)),
        (
            math.log(L_EFF_MIN_UM),
            math.log(L_EFF_MAX_UM),
        ),
        (0.0, math.pi),
        (-diag, diag),
    ]

    def objective(x):
        logA, logL, theta, offset = map(float, x)

        lam = _line_lambda(
            coords,
            math.exp(logA),
            math.exp(logL),
            theta,
            offset,
        )
        p = _event_probability_from_lambda(
            p_dark,
            lam,
        )
        return -_bernoulli_loglike(y, p)

    best = None
    for start in starts:
        opt = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or opt.fun < best.fun:
            best = opt

    logA, logL, theta, offset = map(float, best.x)

    return {
        "A": math.exp(logA),
        "L_eff_um": math.exp(logL),
        "theta_rad": _wrap_period_pi(theta),
        "offset_um": offset,
        "loglike": -float(best.fun),
        "success": bool(best.success),
    }


def _trajectory_bootstrap_for_event(
    ds,
    run_ind,
    line_fit_row,
    rng,
):
    """
    Fixed-K bootstrap of a decisively line-like event.

    The fitted line probabilities are conditioned on exactly the observed K
    losses.  Each bootstrap map is therefore a spatial resampling at fixed
    event magnitude.
    """
    eligible = np.asarray(
        ds["evaluable"][:, run_ind],
        dtype=bool,
    )
    inds = np.where(eligible)[0]

    coords = np.asarray(
        ds["coords_um"][inds],
        dtype=float,
    )
    y_obs = np.asarray(
        ds["loss"][:, run_ind][inds],
        dtype=float,
    )
    p_dark = np.asarray(
        ds["p_i_dark"][inds],
        dtype=float,
    )

    K = int(np.sum(y_obs))

    lam = _line_lambda(
        coords,
        line_fit_row["A"],
        line_fit_row["L_eff_um"],
        line_fit_row["theta_rad"],
        line_fit_row["offset_um"],
    )
    p_line = _event_probability_from_lambda(
        p_dark,
        lam,
    )

    logw, dp = _conditional_bernoulli_log_dp(
        p_line,
        K,
    )

    theta_hat = float(line_fit_row["theta_rad"])
    offset_hat = float(line_fit_row["offset_um"])

    boot_rows = []

    for rep in range(int(TRAJECTORY_BOOTSTRAP_REPS)):
        selected = _sample_conditional_bernoulli_fixed_k(
            logw,
            dp,
            K,
            rng,
        )
        y_boot = selected.astype(float)

        fit = _line_fit_local_bootstrap(
            coords,
            y_boot,
            p_dark,
            line_fit_row,
            rng,
        )

        delta_theta, offset_sign = (
            _signed_period_pi_difference(
                fit["theta_rad"],
                theta_hat,
            )
        )

        aligned_offset = (
            offset_sign * float(fit["offset_um"])
        )

        boot_rows.append({
            "dataset": ds["label"],
            "dark_wait_s": float(ds["wait_s"]),
            "valid_run_ind": int(run_ind),
            "original_run": int(
                ds["original_run"][run_ind]
            ),
            **_source_identity_for_run(ds, run_ind),
            "bootstrap_rep": int(rep),

            "A": float(fit["A"]),
            "L_eff_um": float(fit["L_eff_um"]),

            "theta_normal_rad_raw": float(
                fit["theta_rad"]
            ),
            "theta_normal_delta_deg_aligned": float(
                math.degrees(delta_theta)
            ),

            "offset_um_raw": float(
                fit["offset_um"]
            ),
            "offset_um_aligned": float(
                aligned_offset
            ),

            "track_angle_deg_raw": float(
                math.degrees(
                    _line_track_angle_from_normal(
                        fit["theta_rad"]
                    )
                )
            ),

            "loglike": float(fit["loglike"]),
            "success": bool(fit["success"]),
        })

    return boot_rows


def _trajectory_summary_from_bootstrap(
    ds,
    run_ind,
    line_fit_row,
    boot_rows,
):
    theta_normal_hat = float(line_fit_row["theta_rad"])
    track_angle_hat = math.degrees(
        _line_track_angle_from_normal(
            theta_normal_hat
        )
    )

    eligible = np.asarray(
        ds["evaluable"][:, run_ind],
        dtype=bool,
    )
    coords = np.asarray(
        ds["coords_um"][eligible],
        dtype=float,
    )

    line_xy = _line_center_xy(
        coords,
        theta_normal_hat,
        line_fit_row["offset_um"],
    )

    theta_delta = np.asarray(
        [
            r["theta_normal_delta_deg_aligned"]
            for r in boot_rows
            if np.isfinite(
                r["theta_normal_delta_deg_aligned"]
            )
        ],
        dtype=float,
    )
    offsets = np.asarray(
        [
            r["offset_um_aligned"]
            for r in boot_rows
            if np.isfinite(r["offset_um_aligned"])
        ],
        dtype=float,
    )
    Lvals = np.asarray(
        [
            r["L_eff_um"]
            for r in boot_rows
            if np.isfinite(r["L_eff_um"])
        ],
        dtype=float,
    )
    Avals = np.asarray(
        [
            r["A"]
            for r in boot_rows
            if np.isfinite(r["A"])
        ],
        dtype=float,
    )

    def q(arr, percent):
        return (
            float(np.percentile(arr, percent))
            if arr.size
            else np.nan
        )

    delta_lo = q(theta_delta, 2.5)
    delta_hi = q(theta_delta, 97.5)
    angle_width = (
        delta_hi - delta_lo
        if np.isfinite(delta_lo)
        and np.isfinite(delta_hi)
        else np.nan
    )

    angle_resolved = (
        np.isfinite(angle_width)
        and angle_width
        <= MAX_RESOLVED_TRACK_ANGLE_CI_WIDTH_DEG
    )

    return {
        "dataset": ds["label"],
        "dark_wait_s": float(ds["wait_s"]),
        "particle_exposure_s": float(
            ds["particle_exposure_s"]
        ),
        "lambda_cut": float(
            TRAJECTORY_LAMBDA_H_CUT
        ),

        "valid_run_ind": int(run_ind),
        "original_run": int(
            ds["original_run"][run_ind]
        ),
        **_source_identity_for_run(ds, run_ind),
        "K_loss": int(ds["K"][run_ind]),
        "N_evaluable": int(ds["N"][run_ind]),
        "K_excess": float(ds["k_excess"][run_ind]),
        "Lambda_h": float(ds["lambda_h"][run_ind]),

        "A_hat": float(line_fit_row["A"]),
        "L_eff_hat_um": float(
            line_fit_row["L_eff_um"]
        ),

        "theta_normal_hat_deg": float(
            math.degrees(theta_normal_hat)
        ),
        "track_angle_hat_deg_mod180": float(
            track_angle_hat
        ),

        "track_angle_delta_95_low_deg": delta_lo,
        "track_angle_delta_95_high_deg": delta_hi,
        "track_angle_95_width_deg": angle_width,
        "track_angle_resolved": bool(
            angle_resolved
        ),

        "offset_hat_um": float(
            line_fit_row["offset_um"]
        ),
        "offset_bootstrap_2p5_um": q(
            offsets,
            2.5,
        ),
        "offset_bootstrap_median_um": q(
            offsets,
            50.0,
        ),
        "offset_bootstrap_97p5_um": q(
            offsets,
            97.5,
        ),

        "L_eff_bootstrap_2p5_um": q(
            Lvals,
            2.5,
        ),
        "L_eff_bootstrap_median_um": q(
            Lvals,
            50.0,
        ),
        "L_eff_bootstrap_97p5_um": q(
            Lvals,
            97.5,
        ),

        "A_bootstrap_2p5": q(Avals, 2.5),
        "A_bootstrap_median": q(Avals, 50.0),
        "A_bootstrap_97p5": q(Avals, 97.5),

        "line_anchor_x_um": float(line_xy[0]),
        "line_anchor_y_um": float(line_xy[1]),

        "bootstrap_reps": len(boot_rows),

        "trajectory_note": (
            "Projected unoriented 2D trajectory. "
            "The data do not determine direction of travel "
            "or a unique 3D track."
        ),
    }


def _trajectory_nv_rows(
    ds,
    run_ind,
    line_fit_row,
):
    eligible = np.asarray(
        ds["evaluable"][:, run_ind],
        dtype=bool,
    )
    inds = np.where(eligible)[0]

    coords = np.asarray(
        ds["coords_um"][inds],
        dtype=float,
    )
    y = np.asarray(
        ds["loss"][:, run_ind][inds],
        dtype=bool,
    )
    p_dark = np.asarray(
        ds["p_i_dark"][inds],
        dtype=float,
    )

    s_parallel, d_perp_signed = (
        _track_coordinates(
            coords,
            line_fit_row["theta_rad"],
            line_fit_row["offset_um"],
        )
    )

    lam = _line_lambda(
        coords,
        line_fit_row["A"],
        line_fit_row["L_eff_um"],
        line_fit_row["theta_rad"],
        line_fit_row["offset_um"],
    )
    p_model = _event_probability_from_lambda(
        p_dark,
        lam,
    )

    rows = []
    for j, nv_ind in enumerate(inds):
        rows.append({
            "dataset": ds["label"],
            "dark_wait_s": float(ds["wait_s"]),
            "original_run": int(
                ds["original_run"][run_ind]
            ),
            **_source_identity_for_run(ds, run_ind),
            "nv_index": int(nv_ind),

            "x_um": float(coords[j, 0]),
            "y_um": float(coords[j, 1]),

            "s_parallel_um": float(
                s_parallel[j]
            ),
            "d_perp_signed_um": float(
                d_perp_signed[j]
            ),
            "d_perp_abs_um": float(
                abs(d_perp_signed[j])
            ),

            "loss_observed": int(y[j]),
            "p_dark": float(p_dark[j]),
            "p_line_model": float(
                p_model[j]
            ),
            "lambda_line_model": float(
                lam[j]
            ),
        })

    return rows


def _plot_trajectory_transverse_profile(
    ds,
    run_ind,
    line_fit_row,
):
    eligible = np.asarray(
        ds["evaluable"][:, run_ind],
        dtype=bool,
    )
    inds = np.where(eligible)[0]

    coords = np.asarray(
        ds["coords_um"][inds],
        dtype=float,
    )
    y = np.asarray(
        ds["loss"][:, run_ind][inds],
        dtype=float,
    )
    p_dark = np.asarray(
        ds["p_i_dark"][inds],
        dtype=float,
    )

    _, d_perp_signed = _track_coordinates(
        coords,
        line_fit_row["theta_rad"],
        line_fit_row["offset_um"],
    )

    lam = _line_lambda(
        coords,
        line_fit_row["A"],
        line_fit_row["L_eff_um"],
        line_fit_row["theta_rad"],
        line_fit_row["offset_um"],
    )
    p_model = _event_probability_from_lambda(
        p_dark,
        lam,
    )

    d_abs = np.abs(d_perp_signed)

    max_d = float(np.nanmax(d_abs))
    bin_width = max(
        2.5,
        min(10.0, max_d / 10.0)
    )
    edges = np.arange(
        0.0,
        max_d + bin_width + 1e-12,
        bin_width,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    obs = []
    model = []
    counts = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_abs >= lo) & (d_abs < hi)
        counts.append(int(np.sum(m)))

        if np.sum(m):
            obs.append(float(np.mean(y[m])))
            model.append(float(np.mean(p_model[m])))
        else:
            obs.append(np.nan)
            model.append(np.nan)

    obs = np.asarray(obs, dtype=float)
    model = np.asarray(model, dtype=float)
    counts = np.asarray(counts, dtype=int)

    fig, ax = plt.subplots(figsize=(8.4, 6.1))

    m = counts > 0
    ax.plot(
        centers[m],
        obs[m],
        marker="o",
        linewidth=1.2,
        label="observed loss fraction",
    )
    ax.plot(
        centers[m],
        model[m],
        marker="s",
        linestyle="--",
        linewidth=1.2,
        label="fitted line-model probability",
    )

    ax.set_xlabel(
        "Absolute transverse distance to fitted track (um)"
    )
    ax.set_ylabel(
        r"$NV^- \rightarrow NV^0$ probability"
    )

    track_angle = math.degrees(
        _line_track_angle_from_normal(
            line_fit_row["theta_rad"]
        )
    )

    ax.set_title(
        f"{ds['wait_s']:g} s | run "
        f"{ds['original_run'][run_ind]} | "
        f"track angle={track_angle:.1f} deg | "
        f"L_eff={line_fit_row['L_eff_um']:.1f} um"
    )
    ax.legend(fontsize=8)

    if SAVE_OUTPUTS:
        fig.savefig(
            TRAJECTORY_DIR
            / (
                f"trajectory_transverse_{ds['wait_s']:g}s_"
                f"run_{int(ds['original_run'][run_ind]):04d}.png"
            ),
            dpi=190,
            bbox_inches="tight",
        )

    plt.close(fig)
    return None



def _plot_trajectory_spatial_prediction(
    ds,
    run_ind,
    line_fit_row,
):
    """
    Two-panel spatial plot for an event that preferred the line model.

    Left: observed lost-NV pattern with the fitted projected trajectory overlay.
    Right: fitted line-model per-NV event probabilities on the same coordinates.
    """
    eligible = np.asarray(ds["evaluable"][:, run_ind], dtype=bool)
    inds = np.where(eligible)[0]
    if inds.size < 2:
        return None

    coords = np.asarray(ds["coords_um"][inds], dtype=float)
    lost = np.asarray(ds["loss"][:, run_ind][inds], dtype=bool)
    p_dark = np.asarray(ds["p_i_dark"][inds], dtype=float)

    theta = float(line_fit_row["theta_rad"])
    offset = float(line_fit_row["offset_um"])
    lam = _line_lambda(
        coords,
        float(line_fit_row["A"]),
        float(line_fit_row["L_eff_um"]),
        theta,
        offset,
    )
    p_model = _event_probability_from_lambda(p_dark, lam)

    center = np.mean(coords, axis=0)
    tangent = np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    normal = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    line_center = center + offset * normal

    span = 1.15 * max(
        float(np.ptp(coords[:, 0])),
        float(np.ptp(coords[:, 1])),
        25.0,
    )
    p1 = line_center - span * tangent
    p2 = line_center + span * tangent

    track_angle = math.degrees(
        _line_track_angle_from_normal(theta)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9), sharex=True, sharey=True)

    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.25, label='evaluable NVs')
    ax.scatter(
        coords[lost, 0],
        coords[lost, 1],
        s=60,
        marker='o',
        facecolors='none',
        linewidths=1.4,
        label='observed NV- -> NV0',
    )
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', linewidth=1.6, label='fitted trajectory axis')
    ax.scatter([line_center[0]], [line_center[1]], marker='x', s=70, linewidths=1.6, label='line anchor')
    ax.set_title('Observed event pattern')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=8, loc='best')

    ax = axes[1]
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=p_model,
        s=26,
        cmap='viridis',
        vmin=float(np.nanmin(p_model)),
        vmax=float(np.nanmax(p_model)) if np.nanmax(p_model) > np.nanmin(p_model) else float(np.nanmin(p_model) + 1e-12),
    )
    ax.scatter(
        coords[lost, 0],
        coords[lost, 1],
        s=62,
        marker='o',
        facecolors='none',
        edgecolors='white',
        linewidths=1.2,
    )
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', linewidth=1.6)
    ax.scatter([line_center[0]], [line_center[1]], marker='x', s=70, linewidths=1.6)
    ax.set_title('Predicted trajectory-weighted probability')
    ax.set_xlabel('x (um)')
    ax.set_aspect('equal', adjustable='box')
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('P(event | fitted line model)')

    fig.suptitle(
        f"{ds['wait_s']:g} s | original run {int(ds['original_run'][run_ind])} | "
        f"Lambda={ds['lambda_h'][run_ind]:.4f} | track angle={track_angle:.1f} deg | "
        f"L_eff={float(line_fit_row['L_eff_um']):.1f} um"
    )

    if SAVE_OUTPUTS:
        fig.savefig(
            TRAJECTORY_DIR / (
                f"trajectory_spatial_prediction_{ds['wait_s']:g}s_"
                f"run_{int(ds['original_run'][run_ind]):04d}.png"
            ),
            dpi=190,
            bbox_inches='tight',
        )

    plt.close(fig)
    return None


def _trajectory_angle_null_for_event(
    ds,
    run_ind,
    observed_summary_row,
    rng,
):
    """
    Geometry-aware fixed-K null for a single line-like event.

    Synthetic events are drawn from the fitted uniform-event conditional null,
    preserving the observed K and evaluable NV set.  The same uniform / point /
    line spatial fits are rerun on each synthetic event.
    """
    payload = _prepare_uniform_event_payload(ds, run_ind)
    if payload is None:
        return []

    inds = np.asarray(payload['eligible_indices'], dtype=int)
    coords = np.asarray(ds['coords_um'][inds], dtype=float)
    p_dark = np.asarray(ds['p_i_dark'][inds], dtype=float)
    k_obs = int(payload['K_observed'])

    rows = []
    for rep in range(int(TRAJECTORY_NULL_REPS_PER_EVENT)):
        selected = _sample_conditional_bernoulli_fixed_k(
            payload['conditional_logw'],
            payload['conditional_dp'],
            k_obs,
            rng,
        )
        y = np.asarray(selected, dtype=float)

        models = []
        if FIT_UNIFORM_MODEL:
            models.append(_fit_uniform_model(coords, y, p_dark))
        if FIT_POINT_MODEL:
            models.append(_fit_point_model(coords, y, p_dark, rng))
        if FIT_LINE_MODEL:
            models.append(_fit_line_model(coords, y, p_dark, rng))

        models = sorted(models, key=lambda m: m['AIC'])
        best = models[0]
        uniform = next((m for m in models if m['model'] == 'uniform'), None)
        improvement = (float(uniform['AIC'] - best['AIC']) if uniform is not None else np.nan)

        if best['model'] == 'uniform':
            classification = 'uniform_or_unresolved'
        else:
            if np.isfinite(improvement) and improvement >= LOCALIZED_DELTA_AIC_THRESHOLD:
                classification = ('point_localized_preferred' if best['model'] == 'point_exp' else 'line_like_preferred')
            else:
                classification = 'localized_not_decisive'

        track_angle = (
            math.degrees(_line_track_angle_from_normal(best['theta_rad']))
            if best['model'] == 'line_exp_proxy' and np.isfinite(best['theta_rad'])
            else np.nan
        )

        rows.append({
            'dataset': ds['label'],
            'dark_wait_s': float(ds['wait_s']),
            'lambda_cut': float(TRAJECTORY_LAMBDA_H_CUT),
            'valid_run_ind': int(run_ind),
            'original_run': int(ds['original_run'][run_ind]),
            **_source_identity_for_run(ds, run_ind),
            'Lambda_h': float(ds['lambda_h'][run_ind]),
            'observed_track_angle_deg_mod180': float(observed_summary_row['track_angle_hat_deg_mod180']),
            'observed_L_eff_um': float(observed_summary_row['L_eff_hat_um']),
            'null_rep': int(rep),
            'K_loss': int(k_obs),
            'best_model': best['model'],
            'classification': classification,
            'best_AIC': float(best['AIC']),
            'uniform_AIC': float(uniform['AIC']) if uniform is not None else np.nan,
            'AIC_improvement_vs_uniform': improvement,
            'line_like_preferred': int(classification == 'line_like_preferred'),
            'track_angle_deg_mod180': float(track_angle) if np.isfinite(track_angle) else np.nan,
            'L_eff_um': float(best['L_eff_um']) if np.isfinite(best['L_eff_um']) else np.nan,
        })

    return rows


def _plot_trajectory_angle_null_hist(
    ds,
    run_ind,
    observed_summary_row,
    null_rows,
):
    line_angles = np.asarray(
        [
            r['track_angle_deg_mod180']
            for r in null_rows
            if int(r['line_like_preferred']) == 1
            and np.isfinite(r['track_angle_deg_mod180'])
        ],
        dtype=float,
    )

    if line_angles.size == 0:
        return None

    observed_angle = float(observed_summary_row['track_angle_hat_deg_mod180'])
    line_fraction = 100.0 * line_angles.size / max(1, len(null_rows))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bins = np.arange(0.0, 180.0 + 10.0, 10.0)
    ax.hist(line_angles, bins=bins, alpha=0.75)
    ax.axvline(observed_angle, linestyle='--', linewidth=1.8, label='observed trajectory angle')
    ax.set_xlabel('Track angle modulo 180 deg')
    ax.set_ylabel('Null line-like count')
    ax.set_title(
        f"{ds['wait_s']:g} s | run {int(ds['original_run'][run_ind])} | "
        f"null line-like fraction={line_fraction:.2f}%"
    )
    ax.legend(fontsize=8)

    if SAVE_OUTPUTS:
        fig.savefig(
            TRAJECTORY_DIR / (
                f"trajectory_angle_null_{ds['wait_s']:g}s_"
                f"run_{int(ds['original_run'][run_ind]):04d}.png"
            ),
            dpi=190,
            bbox_inches='tight',
        )

    plt.close(fig)
    return None


def _axial_resultant_from_angles_deg(angles_deg):
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if angles_deg.size == 0:
        return np.nan, np.nan

    phi = np.deg2rad(2.0 * angles_deg)
    z = np.mean(np.exp(1j * phi))
    R = float(np.abs(z))
    mean_axis = 0.5 * np.angle(z)
    mean_axis_deg = math.degrees(_wrap_period_pi(mean_axis))
    return R, mean_axis_deg


def _trajectory_population_orientation_summary(
    trajectory_summary_rows,
    trajectory_null_rows,
):
    if not trajectory_summary_rows or not trajectory_null_rows:
        return []

    rng = np.random.default_rng(TRAJECTORY_NULL_SEED + 100003)
    out = []

    keys = sorted({
        (float(r['dark_wait_s']), float(r.get('lambda_cut', TRAJECTORY_LAMBDA_H_CUT)))
        for r in trajectory_summary_rows
    })

    for wait_s, lambda_cut in keys:
        obs_rows = [
            r for r in trajectory_summary_rows
            if abs(float(r['dark_wait_s']) - wait_s) < 1e-12
            and abs(float(r.get('lambda_cut', TRAJECTORY_LAMBDA_H_CUT)) - lambda_cut) < 1e-12
        ]
        if len(obs_rows) < 2:
            continue

        obs_angles = np.asarray([r['track_angle_hat_deg_mod180'] for r in obs_rows], dtype=float)
        obs_R, obs_mean_axis = _axial_resultant_from_angles_deg(obs_angles)

        pools = []
        event_keys = []
        pooled_angles = []
        for r in obs_rows:
            key = (int(r['original_run']), float(r['dark_wait_s']))
            event_keys.append(key)
            pool = np.asarray([
                nr['track_angle_deg_mod180']
                for nr in trajectory_null_rows
                if abs(float(nr['dark_wait_s']) - wait_s) < 1e-12
                and int(nr['original_run']) == int(r['original_run'])
                and int(nr['line_like_preferred']) == 1
                and np.isfinite(nr['track_angle_deg_mod180'])
            ], dtype=float)
            pool = pool[np.isfinite(pool)]
            pools.append(pool)
            if pool.size:
                pooled_angles.extend(pool.tolist())

        if not all(pool.size > 0 for pool in pools):
            out.append({
                'dark_wait_s': float(wait_s),
                'lambda_cut': float(lambda_cut),
                'n_observed_line_events': int(len(obs_rows)),
                'observed_axial_R': float(obs_R),
                'observed_mean_axis_deg_mod180': float(obs_mean_axis),
                'null_reps': 0,
                'null_axial_R_median': np.nan,
                'null_axial_R_p95': np.nan,
                'cluster_empirical_p': np.nan,
                'cluster_note': 'At least one event had no line-like null fits; no conditional angle-clustering p-value.',
            })
            continue

        null_R = np.full(int(TRAJECTORY_CLUSTER_NULL_REPS), np.nan, dtype=float)
        for rep in range(int(TRAJECTORY_CLUSTER_NULL_REPS)):
            sample = [pool[rng.integers(0, len(pool))] for pool in pools]
            null_R[rep], _ = _axial_resultant_from_angles_deg(sample)

        p_emp = (1.0 + np.sum(null_R >= obs_R)) / (1.0 + np.sum(np.isfinite(null_R)))

        out.append({
            'dark_wait_s': float(wait_s),
            'lambda_cut': float(lambda_cut),
            'n_observed_line_events': int(len(obs_rows)),
            'observed_axial_R': float(obs_R),
            'observed_mean_axis_deg_mod180': float(obs_mean_axis),
            'null_reps': int(np.sum(np.isfinite(null_R))),
            'null_axial_R_median': float(np.nanmedian(null_R)),
            'null_axial_R_p95': float(np.nanpercentile(null_R, 95.0)),
            'cluster_empirical_p': float(p_emp),
            'cluster_note': 'Conditional-on-line-like null using event-specific fixed-K angle pools.',
        })

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.hist(null_R[np.isfinite(null_R)], bins=25, alpha=0.75)
        ax.axvline(obs_R, linestyle='--', linewidth=1.8, label='observed axial R')
        ax.set_xlabel('Axial resultant R')
        ax.set_ylabel('Null count')
        ax.set_title(
            f"{wait_s:g} s | n={len(obs_rows)} line-like events | "
            f"p={p_emp:.4g}"
        )
        ax.legend(fontsize=8)
        if SAVE_OUTPUTS:
            fig.savefig(
                TRAJECTORY_DIR / f"trajectory_population_axial_null_{wait_s:g}s.png",
                dpi=190,
                bbox_inches='tight',
            )
        plt.close(fig)

    return out


def _run_trajectory_analysis(
    datasets,
    model_fit_rows,
    event_summary_rows,
):
    if not RUN_TRAJECTORY_ANALYSIS:
        return {
            "summary_rows": [],
            "bootstrap_rows": [],
            "nv_rows": [],
        }

    rng = np.random.default_rng(
        TRAJECTORY_BOOTSTRAP_SEED
    )

    trajectory_summary_rows = []
    trajectory_bootstrap_rows = []
    trajectory_nv_rows = []
    trajectory_null_rows = []

    for ds in datasets:
        candidates = [
            r
            for r in event_summary_rows
            if abs(
                r["dark_wait_s"] - ds["wait_s"]
            ) < 1e-12
            and abs(
                r["lambda_cut"]
                - TRAJECTORY_LAMBDA_H_CUT
            ) < 1e-12
            and r["classification"]
            == "line_like_preferred"
        ]

        for summary in candidates:
            run_ind = int(
                summary["valid_run_ind"]
            )

            line_rows = [
                r
                for r in model_fit_rows
                if abs(
                    r["dark_wait_s"] - ds["wait_s"]
                ) < 1e-12
                and abs(
                    r["lambda_cut"]
                    - TRAJECTORY_LAMBDA_H_CUT
                ) < 1e-12
                and int(r["valid_run_ind"])
                == run_ind
                and r["model"]
                == "line_exp_proxy"
            ]

            if not line_rows:
                continue

            line_fit_row = line_rows[0]

            print(
                f"[trajectory] wait={ds['wait_s']:g}s "
                f"run={ds['original_run'][run_ind]} "
                f"Lambda={ds['lambda_h'][run_ind]:.4f} "
                f"bootstrap={TRAJECTORY_BOOTSTRAP_REPS}",
                flush=True,
            )

            boot_rows = (
                _trajectory_bootstrap_for_event(
                    ds,
                    run_ind,
                    line_fit_row,
                    rng,
                )
            )

            traj_summary = (
                _trajectory_summary_from_bootstrap(
                    ds,
                    run_ind,
                    line_fit_row,
                    boot_rows,
                )
            )

            nv_rows = _trajectory_nv_rows(
                ds,
                run_ind,
                line_fit_row,
            )

            trajectory_bootstrap_rows.extend(
                boot_rows
            )
            trajectory_summary_rows.append(
                traj_summary
            )
            trajectory_nv_rows.extend(
                nv_rows
            )

            if SAVE_TRAJECTORY_TRANSVERSE_PROFILES:
                _plot_trajectory_transverse_profile(
                    ds,
                    run_ind,
                    line_fit_row,
                )

            if SAVE_TRAJECTORY_SPATIAL_PREDICTION_PLOTS:
                _plot_trajectory_spatial_prediction(
                    ds,
                    run_ind,
                    line_fit_row,
                )

            if RUN_TRAJECTORY_ANGLE_NULL:
                null_rng = np.random.default_rng(
                    TRAJECTORY_NULL_SEED
                    + int(round(ds['wait_s'] * 1000.0))
                    + int(ds['original_run'][run_ind])
                )
                null_rows = _trajectory_angle_null_for_event(
                    ds,
                    run_ind,
                    traj_summary,
                    null_rng,
                )
                trajectory_null_rows.extend(null_rows)
                _plot_trajectory_angle_null_hist(
                    ds,
                    run_ind,
                    traj_summary,
                    null_rows,
                )

    trajectory_cluster_summary_rows = _trajectory_population_orientation_summary(
        trajectory_summary_rows,
        trajectory_null_rows,
    )

    return {
        "summary_rows": trajectory_summary_rows,
        "bootstrap_rows": trajectory_bootstrap_rows,
        "nv_rows": trajectory_nv_rows,
        "null_rows": trajectory_null_rows,
        "cluster_summary_rows": trajectory_cluster_summary_rows,
    }



# =============================================================================
# FIELD-OF-VIEW / SENSOR-FOOTPRINT DIAGNOSTICS
# =============================================================================

def _footprint_summary_row(ds):
    cr = ds["coordinate_report"]
    return {
        "dataset": ds["label"],
        "dark_wait_s": float(ds["wait_s"]),
        "camera_width_um": float(cr["camera_width_um"]),
        "camera_height_um": float(cr["camera_height_um"]),
        "camera_diagonal_um": float(cr["camera_diagonal_um"]),
        "nv_bbox_width_um": float(cr["bbox_width_um"]),
        "nv_bbox_height_um": float(cr["bbox_height_um"]),
        "nv_bbox_diagonal_um": float(cr["bbox_diagonal_um"]),
        "nv_hull_area_um2": float(cr["hull_area_um2"]),
        "nv_hull_perimeter_um": float(cr["hull_perimeter_um"]),
        "max_nv_nv_separation_um": float(cr["max_pair_distance_um"]),
        "bbox_width_fraction_of_camera": float(
            cr["bbox_width_um"] / max(cr["camera_width_um"], 1e-12)
        ),
        "bbox_height_fraction_of_camera": float(
            cr["bbox_height_um"] / max(cr["camera_height_um"], 1e-12)
        ),
    }


def _plot_sensor_footprint(ds):
    coords_um = np.asarray(ds["coords_um"], dtype=float)
    cr = ds["coordinate_report"]

    fig, ax = plt.subplots(figsize=(7.2, 7.0))

    ax.scatter(
        coords_um[:, 0],
        coords_um[:, 1],
        s=12,
        label=f"NV coordinates (n={len(coords_um)})",
    )

    hull_inds = np.asarray(
        cr.get("hull_vertex_indices", []),
        dtype=int,
    )
    if hull_inds.size >= 3:
        hull_xy = coords_um[hull_inds]
        hull_xy = np.vstack([hull_xy, hull_xy[0]])
        ax.plot(
            hull_xy[:, 0],
            hull_xy[:, 1],
            linewidth=1.4,
            label="NV convex hull",
        )

    fw = float(cr["camera_width_um"])
    fh = float(cr["camera_height_um"])
    frame_x = [0.0, fw, fw, 0.0, 0.0]
    frame_y = [0.0, 0.0, fh, fh, 0.0]
    ax.plot(
        frame_x,
        frame_y,
        linestyle="--",
        linewidth=1.1,
        label="camera FOV",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("camera x (um)")
    ax.set_ylabel("camera y (um)")
    ax.set_title(
        f"{ds['wait_s']:g} s sensor footprint\n"
        f"NV bbox={cr['bbox_width_um']:.1f} x "
        f"{cr['bbox_height_um']:.1f} um; "
        f"max NV-NV={cr['max_pair_distance_um']:.1f} um; "
        f"camera={fw:.1f} x {fh:.1f} um"
    )
    ax.legend(fontsize=8)

    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR / f"sensor_footprint_{ds['wait_s']:g}s.png",
            dpi=190,
            bbox_inches="tight",
        )

    return fig


# =============================================================================
# CLASS-LEVEL PLOTS
# =============================================================================

def _plot_pair_correlation(ds, cut, result, pair_geometry):
    if result["observed"] is None:
        return None

    x = np.asarray(pair_geometry["centers_um"], dtype=float)
    obs = np.asarray(result["observed"]["C"], dtype=float)

    fig, (ax, ax_support) = plt.subplots(
        2,
        1,
        figsize=(9.0, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
    )

    ax.plot(
        x,
        obs,
        marker="o",
        linewidth=1.4,
        label="observed uniform-event residuals",
    )
    ax.plot(
        x,
        result["null_mean"],
        linestyle="--",
        linewidth=1.2,
        label="fixed-K uniform-event null mean",
    )
    ax.fill_between(
        x,
        result["null_lo"],
        result["null_hi"],
        alpha=0.20,
        label="null 95% interval",
    )
    ax.axhline(0.0, linewidth=0.9)

    primary_max = float(
        pair_geometry["primary_global_test_max_distance_um"]
    )
    if pair_geometry["max_accessible_distance_um"] > primary_max:
        ax.axvline(
            primary_max,
            linestyle=":",
            linewidth=1.1,
            label=f"primary global-test limit ({primary_max:g} um)",
        )
        ax_support.axvline(
            primary_max,
            linestyle=":",
            linewidth=1.1,
        )

    ax.set_ylabel(
        r"Uniform-event residual correlation $C_U(d)$"
    )
    ax.set_title(
        f"{ds['wait_s']:g} s | Lambda >= {cut:.3f} | "
        f"{len(result['event_payloads'])} events | "
        f"primary global p={result['global_max_p']:.3g}\n"
        f"full NV support to "
        f"{pair_geometry['max_accessible_distance_um']:.1f} um"
    )
    ax.legend(fontsize=8)

    geom_pairs = np.asarray(
        pair_geometry["pair_counts_geometry"],
        dtype=float,
    )
    evaluable_per_event = (
        np.asarray(
            result["observed"]["denominator_pairs"],
            dtype=float,
        )
        / max(len(result["event_payloads"]), 1)
    )

    ax_support.plot(
        x,
        geom_pairs,
        marker="o",
        linewidth=1.2,
        label="all NV pairs in geometry",
    )
    ax_support.plot(
        x,
        evaluable_per_event,
        marker=".",
        linewidth=1.0,
        label="mean evaluable pairs / event",
    )
    ax_support.set_yscale("log")
    ax_support.set_xlabel("NV-NV separation (um)")
    ax_support.set_ylabel("pair support")
    ax_support.legend(fontsize=7)

    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR
            / (
                f"pair_correlation_full_support_"
                f"{ds['wait_s']:g}s_{_cut_token(cut)}.png"
            ),
            dpi=190,
            bbox_inches="tight",
        )

    return fig


def _plot_combined_pair_correlation(all_results, cut):
    entries = [
        (ds, result, geometry)
        for ds, this_cut, result, geometry in all_results
        if abs(this_cut - float(cut)) < 1e-12
        and result["observed"] is not None
    ]

    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(8.7, 6.2))

    for ds, result, geometry in entries:
        ax.plot(
            geometry["centers_um"],
            result["observed"]["C"],
            marker="o",
            linewidth=1.2,
            label=f"{ds['wait_s']:g} s",
        )

    ax.axhline(0.0, linewidth=0.9)
    ax.set_xlabel("NV-NV separation (um)")
    ax.set_ylabel(r"Uniform-event residual correlation $C_U(d)$")
    ax.set_title(
        f"Fixed physical event class: Lambda >= {cut:.3f}"
    )
    ax.legend()

    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR
            / f"pair_correlation_combined_{_cut_token(cut)}.png",
            dpi=190,
            bbox_inches="tight",
        )

    return fig


def _plot_model_preference(event_summary_rows, cut):
    rows = [
        r for r in event_summary_rows
        if abs(r["lambda_cut"] - float(cut)) < 1e-12
    ]
    if not rows:
        return None

    waits = sorted({r["dark_wait_s"] for r in rows})
    categories = (
        "uniform_or_unresolved",
        "localized_not_decisive",
        "point_localized_preferred",
        "line_like_preferred",
    )

    x = np.arange(len(waits), dtype=float)
    bottom = np.zeros(len(waits), dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    for category in categories:
        vals = []
        for wait in waits:
            sub = [r for r in rows if r["dark_wait_s"] == wait]
            if sub:
                vals.append(
                    100.0
                    * sum(r["classification"] == category for r in sub)
                    / len(sub)
                )
            else:
                vals.append(0.0)

        vals = np.asarray(vals, dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=category.replace("_", " "),
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w:g} s" for w in waits])
    ax.set_ylabel("Selected events (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Per-event spatial model preference | Lambda >= {cut:.3f}"
    )
    ax.legend(fontsize=8)

    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR
            / f"model_preference_{_cut_token(cut)}.png",
            dpi=190,
            bbox_inches="tight",
        )

    return fig


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def _dataset_summary_row(ds):
    return {
        "dataset": ds["label"],
        "dark_wait_s": ds["wait_s"],
        "particle_exposure_s": ds["particle_exposure_s"],
        "num_source_files": int(ds.get("num_source_files", 1)),
        "raw_runs_loaded": int(ds["n_runs_loaded"]),
        "good_runs": len(ds["K"]),
        "central_p_bulk": ds["p_bulk"],
        "central_rho_bulk": ds["bulk"]["rho"],
        "mean_K": float(np.mean(ds["K"])),
        "mean_N": float(np.mean(ds["N"])),
        "median_nn_px": ds["coordinate_report"]["median_nn_px"],
        "median_nn_um_configured": ds["coordinate_report"][
            "median_nn_um_configured"
        ],
        "pitch_inferred_um_per_pixel": ds["coordinate_report"][
            "pitch_inferred_um_per_pixel"
        ],
        "events_lambda_ge_0p04": int(
            np.sum(ds["lambda_h"] >= 0.04)
        ),
        "events_lambda_ge_0p05": int(
            np.sum(ds["lambda_h"] >= 0.05)
        ),
        "events_lambda_ge_0p07": int(
            np.sum(ds["lambda_h"] >= 0.07)
        ),
    }


def _pair_rows_from_result(ds, cut, result, geometry):
    rows = []

    if result["observed"] is None:
        return rows

    edges = geometry["edges_um"]
    centers = geometry["centers_um"]

    for b in range(len(centers)):
        rows.append({
            "dataset": ds["label"],
            "dark_wait_s": ds["wait_s"],
            "lambda_cut": float(cut),
            "selected_events": len(result["selected_run_inds"]),

            "distance_low_um": float(edges[b]),
            "distance_high_um": float(edges[b + 1]),
            "distance_center_um": float(centers[b]),

            "geometry_pair_count": int(
                geometry["pair_counts_geometry"][b]
            ),
            "geometry_pair_fraction_of_all_pairs": float(
                geometry["fraction_of_all_pairs_by_bin"][b]
            ),
            "evaluable_pair_observations": float(
                result["observed"]["denominator_pairs"][b]
            ),
            "inside_primary_global_test_range": bool(
                geometry["primary_test_mask"][b]
            ),
            "max_accessible_nv_pair_distance_um": float(
                geometry["max_accessible_distance_um"]
            ),
            "distance_center_fraction_of_max_accessible": float(
                centers[b]
                / max(geometry["max_accessible_distance_um"], 1e-12)
            ),

            "C_observed": float(result["observed"]["C"][b]),
            "C_null_mean": float(result["null_mean"][b]),
            "C_null_std": float(result["null_std"][b]),
            "C_null_2p5": float(result["null_lo"][b]),
            "C_null_97p5": float(result["null_hi"][b]),
            "C_z_vs_null": float(result["z"][b]),
            "C_empirical_upper_p": float(result["p_upper"][b]),

            "null_type": result["null_type"],
            "global_max_z_observed": float(
                result["max_z_observed"]
            ),
            "global_max_p_corrected_over_distance_bins": float(
                result["global_max_p"]
            ),
        })

    return rows


def _fit_models_for_cut(ds, cut, rng):
    selected = np.where(ds["lambda_h"] >= float(cut))[0]

    if selected.size == 0:
        return [], []

    # Highest-Lambda events first.
    selected = selected[
        np.argsort(ds["lambda_h"][selected])[::-1]
    ]

    if MAX_MODEL_EVENTS_PER_WAIT_PER_CUT is not None:
        selected = selected[: int(MAX_MODEL_EVENTS_PER_WAIT_PER_CUT)]

    fit_rows = []
    summary_rows = []

    for idx, run_ind in enumerate(selected, start=1):
        identity = _source_identity_for_run(
            ds,
            run_ind,
        )

        print(
            f"[model fit] wait={ds['wait_s']:g}s cut={cut:.3f} "
            f"event {idx}/{len(selected)} "
            f"global_run={ds['original_run'][run_ind]} "
            f"source={identity['source_file_ind']} "
            f"local_run={identity['source_local_run']} "
            f"Lambda={ds['lambda_h'][run_ind]:.4f}",
            flush=True,
        )

        result = _fit_event_spatial_models(
            ds,
            int(run_ind),
            cut,
            rng,
        )
        if result is None:
            continue

        rows, summary = result
        fit_rows.extend(rows)
        summary_rows.append(summary)

    return fit_rows, summary_rows


def _summary_text(
    datasets,
    pair_rows,
    event_summary_rows,
    jackknife_rows,
    trajectory_summary_rows,
    trajectory_null_rows,
    trajectory_cluster_summary_rows,
):
    lines = []
    lines.append("SPATIAL CARRIER-EVENT ANALYSIS V5B-FOV")
    lines.append("=" * 92)
    lines.append("")
    lines.append(
        "RAW NV- -> NV0 classification; no truth filtering; no image pixels."
    )
    lines.append(
        "Primary event variable is fixed Lambda_h, not wait-dependent sigma."
    )
    lines.append("")

    lines.append("DATASETS")
    lines.append("-" * 92)
    for ds in datasets:
        lines.append(
            f"{ds['wait_s']:g} s: good={len(ds['K'])}; "
            f"p_bulk={100*ds['p_bulk']:.4f}%; "
            f"events Lambda>=0.05={np.sum(ds['lambda_h'] >= 0.05)}."
        )
    lines.append("")

    lines.append("COORDINATE CALIBRATION")
    lines.append("-" * 92)
    for ds in datasets:
        cr = ds["coordinate_report"]
        lines.append(
            f"{ds['wait_s']:g} s: median NN={cr['median_nn_px']:.4f} px "
            f"= {cr['median_nn_um_configured']:.4f} um with configured "
            f"{UM_PER_PIXEL:g} um/px; pitch-implied scale="
            f"{cr['pitch_inferred_um_per_pixel']:.6f} um/px."
        )
        lines.append(
            f"  NV bbox={cr['bbox_width_um']:.2f} x "
            f"{cr['bbox_height_um']:.2f} um; "
            f"max NV-NV separation={cr['max_pair_distance_um']:.2f} um; "
            f"camera FOV={cr['camera_width_um']:.2f} x "
            f"{cr['camera_height_um']:.2f} um; "
            f"convex-hull area={cr['hull_area_um2']:.1f} um^2."
        )
    lines.append("")

    lines.append("FIELD-OF-VIEW / PAIR-SUPPORT RULE")
    lines.append("-" * 92)
    lines.append(
        f"Correlation curves are evaluated to the largest coordinate-supported "
        f"NV-NV separation. The quoted distance-search-corrected global p-value "
        f"remains restricted to <= "
        f"{PAIR_PRIMARY_GLOBAL_TEST_MAX_DISTANCE_UM:g} um to preserve the "
        f"previous primary test. Distances beyond that are diagnostic."
    )
    lines.append(
        "Each pair-correlation figure now includes a lower panel with the "
        "number of available NV pairs and the mean evaluable pairs per event."
    )
    lines.append("")

    lines.append("UNIFORM-EVENT-RESIDUAL SPATIAL CORRELATION")
    lines.append("-" * 92)

    for cut in LAMBDA_H_CUTS:
        lines.append(f"Lambda >= {cut:.3f}")

        for ds in datasets:
            sub = [
                r for r in pair_rows
                if abs(r["dark_wait_s"] - ds["wait_s"]) < 1e-12
                and abs(r["lambda_cut"] - cut) < 1e-12
                and np.isfinite(r["C_z_vs_null"])
            ]

            if not sub:
                lines.append(
                    f"  {ds['wait_s']:g} s: insufficient events / no pair result."
                )
                continue

            primary_sub = [
                r for r in sub
                if bool(
                    r.get(
                        "inside_primary_global_test_range",
                        False,
                    )
                )
            ]
            if not primary_sub:
                primary_sub = sub

            best = max(
                primary_sub,
                key=lambda r: r["C_z_vs_null"],
            )
            full_best = max(
                sub,
                key=lambda r: r["C_z_vs_null"],
            )

            lines.append(
                f"  {ds['wait_s']:g} s: PRIMARY max positive "
                f"uniform-residual bin "
                f"{best['distance_low_um']:.1f}-"
                f"{best['distance_high_um']:.1f} um; "
                f"z={best['C_z_vs_null']:.2f}; bin p="
                f"{best['C_empirical_upper_p']:.4g}; "
                f"distance-search-corrected max-statistic p="
                f"{best['global_max_p_corrected_over_distance_bins']:.4g}."
            )

            if (
                full_best["distance_low_um"] != best["distance_low_um"]
                or full_best["distance_high_um"] != best["distance_high_um"]
            ):
                lines.append(
                    f"    full-range diagnostic max is "
                    f"{full_best['distance_low_um']:.1f}-"
                    f"{full_best['distance_high_um']:.1f} um with "
                    f"z={full_best['C_z_vs_null']:.2f}; this does not "
                    "redefine the primary global p if outside the primary "
                    "distance-search range."
                )
        lines.append("")

    lines.append("SPATIAL MODEL PREFERENCE")
    lines.append("-" * 92)

    for cut in LAMBDA_H_CUTS:
        lines.append(f"Lambda >= {cut:.3f}")
        for ds in datasets:
            sub = [
                r for r in event_summary_rows
                if abs(r["dark_wait_s"] - ds["wait_s"]) < 1e-12
                and abs(r["lambda_cut"] - cut) < 1e-12
            ]

            if not sub:
                lines.append(f"  {ds['wait_s']:g} s: no fitted events.")
                continue

            counts = {}
            for r in sub:
                counts[r["classification"]] = (
                    counts.get(r["classification"], 0) + 1
                )

            localized = [
                r for r in sub
                if r["classification"] in (
                    "point_localized_preferred",
                    "line_like_preferred",
                )
                and np.isfinite(r["best_L_eff_um"])
            ]
            lstats = _percentile_summary(
                [r["best_L_eff_um"] for r in localized]
            )

            lines.append(
                f"  {ds['wait_s']:g} s: n={len(sub)}; classes={counts}; "
                f"localized L_eff median="
                f"{lstats['median']:.3g} um"
                if localized
                else
                f"  {ds['wait_s']:g} s: n={len(sub)}; classes={counts}; "
                "no decisively localized L_eff values."
            )
        lines.append("")

    lines.append("LEAVE-ONE-EVENT-OUT ROBUSTNESS")
    lines.append("-" * 92)

    if jackknife_rows:
        keys = sorted({
            (float(r["dark_wait_s"]), float(r["lambda_cut"]))
            for r in jackknife_rows
        })

        for wait, cut in keys:
            sub = [
                r for r in jackknife_rows
                if abs(r["dark_wait_s"] - wait) < 1e-12
                and abs(r["lambda_cut"] - cut) < 1e-12
            ]

            peak_z = np.asarray(
                [
                    r["jackknife_peak_z_vs_full_null_same_bin"]
                    for r in sub
                    if np.isfinite(
                        r["jackknife_peak_z_vs_full_null_same_bin"]
                    )
                ],
                dtype=float,
            )
            frac = np.asarray(
                [
                    r["jackknife_peak_C_fraction_of_full"]
                    for r in sub
                    if np.isfinite(
                        r["jackknife_peak_C_fraction_of_full"]
                    )
                ],
                dtype=float,
            )

            if peak_z.size:
                lines.append(
                    f"{wait:g} s, Lambda>={cut:.3f}: "
                    f"leave-one-out peak-bin z range="
                    f"[{np.min(peak_z):.2f}, {np.max(peak_z):.2f}], "
                    f"median={np.median(peak_z):.2f}; "
                    f"C/C_full range="
                    f"[{np.min(frac):.3f}, {np.max(frac):.3f}]."
                )
    else:
        lines.append("No jackknife results were generated.")

    lines.append("")

    lines.append("PROJECTED LINE-TRAJECTORY INFERENCE")
    lines.append("-" * 92)

    if trajectory_summary_rows:
        for r in trajectory_summary_rows:
            lines.append(
                f"{r['dark_wait_s']:g} s run {r['original_run']}: "
                f"Lambda={r['Lambda_h']:.4f}; "
                f"projected angle={r['track_angle_hat_deg_mod180']:.2f} deg "
                f"with aligned bootstrap delta95="
                f"[{r['track_angle_delta_95_low_deg']:.2f}, "
                f"{r['track_angle_delta_95_high_deg']:.2f}] deg; "
                f"L_eff={r['L_eff_hat_um']:.2f} um "
                f"(bootstrap95 "
                f"{r['L_eff_bootstrap_2p5_um']:.2f}-"
                f"{r['L_eff_bootstrap_97p5_um']:.2f} um); "
                f"angle_resolved={r['track_angle_resolved']}."
            )
    else:
        lines.append(
            "No events at the trajectory-analysis cut decisively "
            "preferred the line-like model."
        )

    lines.append("")
    lines.append(
        "Trajectory angles are unoriented 2D projected axes modulo 180 deg. "
        "Initial/final charge maps do not determine direction of travel or "
        "a unique 3D trajectory."
    )
    lines.append("")

    lines.append("TRAJECTORY-ANGLE NULL / AXIAL CLUSTERING")
    lines.append("-" * 92)
    if trajectory_null_rows:
        grouped = sorted({
            (float(r['dark_wait_s']), int(r['original_run']))
            for r in trajectory_null_rows
        })
        for wait, original_run in grouped:
            sub = [
                r for r in trajectory_null_rows
                if abs(float(r['dark_wait_s']) - wait) < 1e-12
                and int(r['original_run']) == original_run
            ]
            line_sub = [r for r in sub if int(r['line_like_preferred']) == 1 and np.isfinite(r['track_angle_deg_mod180'])]
            frac = 100.0 * len(line_sub) / max(1, len(sub))
            lines.append(
                f"{wait:g} s run {original_run}: null line-like fraction="
                f"{frac:.2f}% ({len(line_sub)}/{len(sub)})."
            )
    else:
        lines.append("No trajectory-angle null results were generated.")

    if trajectory_cluster_summary_rows:
        for r in trajectory_cluster_summary_rows:
            lines.append(
                f"{r['dark_wait_s']:g} s: n_line={r['n_observed_line_events']}; "
                f"observed axial R={r['observed_axial_R']:.3f}; "
                f"mean axis={r['observed_mean_axis_deg_mod180']:.2f} deg; "
                f"null median R={r['null_axial_R_median']:.3f}; "
                f"null 95th percentile={r['null_axial_R_p95']:.3f}; "
                f"cluster p={r['cluster_empirical_p']:.4g}."
            )
    lines.append("")

    lines.append("INTERPRETATION RULE")
    lines.append("-" * 92)
    lines.append(
        f"Do not call L_eff a diffusion length unless localized models beat "
        f"uniform/common-mode by Delta AIC >= "
        f"{LOCALIZED_DELTA_AIC_THRESHOLD:g} AND the fixed-K "
        "uniform-event-residual correlation shows reproducible non-null "
        "structure."
    )
    lines.append(
        "With passive initial/final readout, D and tau are not separately "
        "identified; a controlled localized source with variable delay is "
        "needed for D, tau, and L=sqrt(D tau)."
    )

    return "\n".join(lines)


def main():
    _ensure_output_dirs()

    print("\n" + "#" * 118)
    print("SPATIAL CARRIER-EVENT ANALYSIS V5B-FOV")
    print("#" * 118)
    print(
        "Raw charge transitions + NV coordinates only. "
        "No truth filtering. No img_arrays."
    )

    phen = _import_phenomenology()
    dataset_groups = _select_dataset_configs(phen)

    datasets = []
    for group in dataset_groups:
        ds = _prepare_spatial_dataset(
            phen,
            group,
        )
        ds = _fit_per_nv_dark_baseline(ds)
        datasets.append(ds)

    datasets = sorted(datasets, key=lambda d: d["wait_s"])

    dataset_rows = [_dataset_summary_row(ds) for ds in datasets]
    _write_csv(
        OUTPUT_DIR / "spatial_dataset_summary.csv",
        dataset_rows,
    )

    footprint_rows = [
        _footprint_summary_row(ds)
        for ds in datasets
    ]
    _write_csv(
        OUTPUT_DIR / "spatial_field_of_view_summary.csv",
        footprint_rows,
    )

    # ---------------------------------------------------------------------
    # Pair-correlation analysis.
    # ---------------------------------------------------------------------
    pair_rows = []
    pair_results = []
    figures = []

    for ds in datasets:
        footprint_fig = _plot_sensor_footprint(ds)
        if footprint_fig is not None:
            figures.append(footprint_fig)

    for ds in datasets:
        geometry = _build_pair_geometry(ds["coords_um"])

        for cut in LAMBDA_H_CUTS:
            print(
                f"[fixed-K spatial null] wait={ds['wait_s']:g}s "
                f"Lambda>={cut:.3f}",
                flush=True,
            )

            result = _pair_correlation_with_null(
                ds,
                cut,
                geometry,
            )

            pair_results.append(
                (ds, cut, result, geometry)
            )

            pair_rows.extend(
                _pair_rows_from_result(
                    ds,
                    cut,
                    result,
                    geometry,
                )
            )

            fig = _plot_pair_correlation(
                ds,
                cut,
                result,
                geometry,
            )
            if fig is not None:
                figures.append(fig)

    _write_csv(
        OUTPUT_DIR / "spatial_uniform_residual_correlation.csv",
        pair_rows,
    )

    # Backward-compatible filename; contents are now the corrected
    # uniform-event-residual / fixed-K null statistic.
    _write_csv(
        OUTPUT_DIR / "spatial_pair_correlation.csv",
        pair_rows,
    )

    for cut in LAMBDA_H_CUTS:
        fig = _plot_combined_pair_correlation(
            pair_results,
            cut,
        )
        if fig is not None:
            figures.append(fig)

    # ---------------------------------------------------------------------
    # Leave-one-event-out robustness of the class-level correlation.
    # ---------------------------------------------------------------------
    jackknife_rows = []

    if RUN_SPATIAL_JACKKNIFE:
        for ds, cut, result, geometry in pair_results:
            if not any(
                abs(float(cut) - float(target)) < 1e-12
                for target in JACKKNIFE_LAMBDA_H_CUTS
            ):
                continue

            jackknife = _jackknife_pair_correlation(
                ds,
                cut,
                result,
                geometry,
            )

            jackknife_rows.extend(
                jackknife["rows"]
            )

            fig = _plot_jackknife_pair_correlation(
                ds,
                cut,
                result,
                geometry,
                jackknife,
            )
            if fig is not None:
                figures.append(fig)

    _write_csv(
        OUTPUT_DIR
        / "spatial_pair_correlation_jackknife.csv",
        jackknife_rows,
    )

    # ---------------------------------------------------------------------
    # Per-event spatial model comparison.
    # ---------------------------------------------------------------------
    rng = np.random.default_rng(MODEL_FIT_SEED)

    model_fit_rows = []
    event_summary_rows = []

    for ds in datasets:
        for cut in LAMBDA_H_CUTS:
            fit_rows, summary_rows = _fit_models_for_cut(
                ds,
                cut,
                rng,
            )
            model_fit_rows.extend(fit_rows)
            event_summary_rows.extend(summary_rows)

    _write_csv(
        OUTPUT_DIR / "spatial_event_model_fits.csv",
        model_fit_rows,
    )
    _write_csv(
        OUTPUT_DIR / "spatial_event_summary.csv",
        event_summary_rows,
    )

    for cut in LAMBDA_H_CUTS:
        fig = _plot_model_preference(
            event_summary_rows,
            cut,
        )
        if fig is not None:
            figures.append(fig)

    # ---------------------------------------------------------------------
    # Projected trajectory inference for decisively line-like events.
    # ---------------------------------------------------------------------
    trajectory = _run_trajectory_analysis(
        datasets,
        model_fit_rows,
        event_summary_rows,
    )

    trajectory_summary_rows = trajectory["summary_rows"]
    trajectory_bootstrap_rows = trajectory["bootstrap_rows"]
    trajectory_nv_rows = trajectory["nv_rows"]
    trajectory_null_rows = trajectory["null_rows"]
    trajectory_cluster_summary_rows = trajectory["cluster_summary_rows"]

    _write_csv(
        OUTPUT_DIR / "spatial_trajectory_summary.csv",
        trajectory_summary_rows,
    )
    _write_csv(
        OUTPUT_DIR / "spatial_trajectory_bootstrap.csv",
        trajectory_bootstrap_rows,
    )
    _write_csv(
        OUTPUT_DIR / "spatial_trajectory_nv_coordinates.csv",
        trajectory_nv_rows,
    )
    _write_csv(
        OUTPUT_DIR / "spatial_trajectory_angle_null.csv",
        trajectory_null_rows,
    )
    _write_csv(
        OUTPUT_DIR / "spatial_trajectory_population_orientation.csv",
        trajectory_cluster_summary_rows,
    )

    # ---------------------------------------------------------------------
    # Save event maps only for strongest events at the primary cut.
    # ---------------------------------------------------------------------
    for ds in datasets:
        sub = [
            r for r in event_summary_rows
            if abs(r["dark_wait_s"] - ds["wait_s"]) < 1e-12
            and abs(
                r["lambda_cut"] - PRIMARY_LAMBDA_H_CUT
            ) < 1e-12
        ]

        sub = sorted(
            sub,
            key=lambda r: r["Lambda_h"],
            reverse=True,
        )[:MAX_EVENT_MAPS_PER_WAIT]

        for summary in sub:
            run_ind = int(summary["valid_run_ind"])
            fit_rows = [
                r for r in model_fit_rows
                if abs(r["dark_wait_s"] - ds["wait_s"]) < 1e-12
                and abs(
                    r["lambda_cut"] - PRIMARY_LAMBDA_H_CUT
                ) < 1e-12
                and int(r["valid_run_ind"]) == run_ind
            ]

            if not fit_rows:
                continue

            fig = _plot_event_map(
                ds,
                run_ind,
                summary,
                fit_rows,
            )
            if fig is not None:
                figures.append(fig)

    # ---------------------------------------------------------------------
    # Human-readable summary.
    # ---------------------------------------------------------------------
    summary = _summary_text(
        datasets,
        pair_rows,
        event_summary_rows,
        jackknife_rows,
        trajectory_summary_rows,
        trajectory_null_rows,
        trajectory_cluster_summary_rows,
    )

    print("\n")
    print(summary)

    if SAVE_OUTPUTS:
        with (
            OUTPUT_DIR / "spatial_analysis_summary.txt"
        ).open("w", encoding="utf-8") as f:
            f.write(summary)
            f.write("\n")

    print("\n" + "=" * 118)
    print("OUTPUT DIRECTORY")
    print("=" * 118)
    print(OUTPUT_DIR.resolve())

    if SHOW_FIGURES:
        plt.show(block=True)
    else:
        plt.close("all")

    return {
        "datasets": datasets,
        "dataset_rows": dataset_rows,
        "footprint_rows": footprint_rows,
        "pair_rows": pair_rows,
        "pair_results": pair_results,
        "model_fit_rows": model_fit_rows,
        "event_summary_rows": event_summary_rows,
        "jackknife_rows": jackknife_rows,
        "trajectory_summary_rows": trajectory_summary_rows,
        "trajectory_bootstrap_rows": trajectory_bootstrap_rows,
        "trajectory_nv_rows": trajectory_nv_rows,
        "trajectory_null_rows": trajectory_null_rows,
        "trajectory_cluster_summary_rows": trajectory_cluster_summary_rows,
        "figures": figures,
    }


if __name__ == "__main__":
    from utils import kplotlib as kpl
    kpl.init_kplotlib()
    analysis = main()
