# -*- coding: utf-8 -*-
"""
Count-only carrier-capture inference for raw NV charge-memory data, including fixed physical event thresholds.

PURPOSE
-------
Use the existing RAW bulk + upper-tail phenomenology analysis to convert the
measured size of each high-loss run into physically interpretable carrier-
capture quantities.

This script DOES NOT use:
    - truth/misclassification filtering
    - image pixels
    - spatial fitting

It reuses the raw hard-classified NV- -> NV0 loss counts and the fitted central
bulk probability p_bulk(t).

For each candidate upper-tail run:

    observed total loss fraction:
        p_obs = K / N

    ordinary dark loss probability:
        p_dark = p_bulk(t)

Assume that a transient hole burst acts only on NV- centers that would
otherwise have survived the ordinary dark process:

    1 - p_obs = (1 - p_dark) exp(-Lambda_h)

Therefore the event-integrated HOLE CAPTURE HAZARD is

    Lambda_h = -ln[(1-p_obs)/(1-p_dark)].

This is the most useful parameter that can be extracted directly from
count-only data without knowing the microscopic carrier density.

If a literature hole-capture coefficient C_h is adopted,

    Lambda_h = C_h * integral n_h(t) dt

so

    H = integral n_h(t) dt = Lambda_h / C_h
        [cm^-3 s].

If instead an effective hole-capture cross section sigma_h,eff is adopted,

    Lambda_h = sigma_h,eff * F_h

so

    F_h = Lambda_h / sigma_h,eff
        [cm^-2],

where F_h is an effective hole fluence.

IMPORTANT:
----------
Do NOT interpret H or F_h as separately measured microscopic quantities.
They inherit the assumed literature C_h or sigma_h,eff.

The script also reports:

    K_excess = K - N*p_dark

the excess number of losses above the central dark expectation.

For a reference ionizing-particle track with

    N_eh = 36 pairs/um * track_length_um,

it reports

    K_excess / N_eh

as the fraction of nominally generated holes represented by the monitored-NV
excess losses.  This is NOT a detector efficiency; it ignores carriers lost
to other traps, recombination, surfaces, track geometry, etc.

Literature priors used by default
---------------------------------
C_h(NV- -> NV0*) = 1.8e-7 cm^3/s at 300 K
C_e(NV0 -> NV-*) = 2.1e-9 cm^3/s at 300 K

    Vishwakarma et al., arXiv:2605.24768 (2026).

Effective single-defect hole-capture cross sections can be much larger than
microscopic thermal cross sections because of Coulomb/cascade capture.
A useful phenomenological scale is

    sigma_h,eff ~ 3e-11 cm^2 = 3e-3 um^2

motivated by single-NV carrier-capture experiments such as Lozovoi et al.,
Nano Lett. 23, 4495 (2023), DOI: 10.1021/acs.nanolett.3c00860.

The coefficient model and cross-section model are two alternative
parameterizations.  Do not multiply them together or treat both as independent
microscopic measurements.

EXPECTED INPUT
--------------
Place this script in the same analysis directory as your raw phenomenology
script. It tries these imports in order:

  1. sc_charge_state_particle_memory_bulk_and_tail_phenomenology
  2. sc_charge_state_particle_memory_bulk_tail_mixture_model

The second name is included because your current local run log shows that
filename while producing the raw/no-truth phenomenology output.

MULTIPLE FILES AT THE SAME WAIT
-------------------------------
This version supports one or many physical acquisition files for each dark
wait. It accepts either

    "file_stem": "one_file"

or

    "file_stem": ("file_1", "file_2", ...)

or

    "file_stems": ["file_1", "file_2", ...]

in the imported phenomenology/base DATASETS configuration.

Every physical file is prepared independently first, so its own saved charge
thresholds and quality rejection are respected. Prepared K/N runs are then
appended within the same physical wait. One combined bulk p_dark(t), one
Lambda_h mapping, and one fixed-threshold rate analysis are then performed for
that wait. Different waits are never appended together.

Every output event retains:
    source_file_ind
    source_local_run
    source_label
    source_file_stem
in addition to a unique appended original_run index.

OUTPUTS
-------
analysis_output/carrier_capture_inference_raw/

    carrier_event_parameters.csv
    carrier_event_summary_by_wait_sigma.csv
    carrier_model_constants.csv
    event_size_wait_comparison.csv
    analysis_summary.txt

    carrier_hazard_vs_wait_3sigma.png
    carrier_hazard_vs_wait_4sigma.png
    carrier_hazard_vs_wait_5sigma.png

    excess_losses_vs_wait_3sigma.png
    excess_losses_vs_wait_4sigma.png
    excess_losses_vs_wait_5sigma.png

    integrated_hole_exposure_vs_wait_4sigma.png
    effective_collection_radius_vs_wait_4sigma.png

Interpretation
--------------
The direct data-derived quantities are:

    p_dark(t)
    K
    N
    p_obs
    K_excess
    Lambda_h

The following require an assumed literature/model parameter:

    integral n_h dt  requires C_h
    hole fluence     requires sigma_h,eff
    effective collection radius requires sigma_h,eff + assumed N_eh
    effective mean hole density requires an assumed carrier transient duration
"""

from __future__ import annotations

import csv
import importlib
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, ks_2samp, mannwhitneyu


# =============================================================================
# CONFIGURATION
# =============================================================================

# Candidate upper-tail thresholds.
SIGMA_CUTS = (3.0, 4.0, 5.0)

PRIMARY_SIGMA_CUT = 4.0

# ---------------------------------------------------------------------------
# NEW: fixed PHYSICAL event thresholds.
#
# These cuts are applied to ALL good runs, not only to sigma-selected runs.
# This removes the main caveat of comparing fixed-sigma event populations
# when the bulk distribution itself changes with dark wait.
#
# Lambda_h is the preferred event-amplitude variable because it removes the
# wait-dependent central dark-loss probability:
#
#   Lambda_h = -ln[(1-K/N)/(1-p_dark)].
#
# For reference:
#   Lambda=0.04 -> conditional extra conversion probability 3.92%
#   Lambda=0.05 -> conditional extra conversion probability 4.88%
#   Lambda=0.07 -> conditional extra conversion probability 6.76%
#
LAMBDA_H_CUTS = (0.04, 0.05, 0.07)

# A second, more directly count-based control analysis.
K_EXCESS_CUTS = (20.0, 30.0, 40.0)

PRIMARY_LAMBDA_H_CUT = 0.05
PRIMARY_K_EXCESS_CUT = 30.0

# Direction-corrected geometric cosmic-muon crossing rate used only as a
# rate-scale comparison.  This is not imposed on the data-derived fit.
R_MU_GEOM_S_INV = 2.069813e-4

# Parametric/nonparametric binomial bootstrap for the exposure-rate fit.
# This remains fast because there are only a few wait/exposure points.
PHYSICAL_RATE_BOOTSTRAP = 500
PHYSICAL_RATE_BOOTSTRAP_SEED = 260829

# Broad rate bound for the free exposure model.
PHYSICAL_R_EVENT_MAX_S_INV = 5.0e-3

# ---------------------------------------------------------------------------
# Literature/model priors
# ---------------------------------------------------------------------------

# 2026 first-principles value:
# NV- + h+ -> NV0* -> NV0
C_H_CM3_S = 1.8e-7

# 2026 first-principles value:
# NV0 + e- -> NV-* -> NV-
# Used only for reporting the expected capture-coefficient asymmetry.
C_E_CM3_S = 2.1e-9

# Effective long-range single-defect hole-capture cross-section scale.
# This is deliberately labeled "effective".
SIGMA_H_EFF_CM2 = 3.0e-11

# Approximate electron-hole pair yield for a minimum-ionizing particle
# in diamond.
EH_PAIRS_PER_UM = 36.0

# Track-length sensitivities.
#
# 50 um:
#     normal crossing of the thin dimension.
#
# 66.7 um:
#     useful order-of-magnitude oblique-crossing reference
#     (~4/3 times the thickness).
#
# 100 um:
#     longer oblique/secondary reference.
REFERENCE_TRACK_LENGTHS_UM = (50.0, 66.7, 100.0)

# Optional assumed carrier-transient durations.
# These are NOT measured by this analysis.
# They are used only to convert integral n_h dt into an "if tau = ..." density.
ASSUMED_TRANSIENTS_S = {
    "1ns": 1e-9,
    "10ns": 1e-8,
    "100ns": 1e-7,
    "1us": 1e-6,
}

# Require the measured loss fraction to exceed the fitted bulk probability.
# This should naturally be true for upper-tail runs.
REQUIRE_POSITIVE_EXCESS = True

# Minimum numerical headroom.
EPS = 1e-12

SAVE_OUTPUTS = False
SHOW_FIGURES = True
OUTPUT_DIR = Path("analysis_output") / "carrier_capture_inference_raw"

SCRIPT_VERSION = (
    "CARRIER_CAPTURE_V3B_MULTI_FILE_PER_WAIT_BULK_COMPAT_2026-08-29"
)


# =============================================================================
# IMPORT EXISTING RAW PHENOMENOLOGY
# =============================================================================

def _import_phenomenology():
    module_names = (
        "sc_charge_state_particle_memory_bulk_and_tail_phenomenology",
        "sc_charge_state_particle_memory_bulk_tail_mixture_model",
    )

    errors = []
    for name in module_names:
        try:
            mod = importlib.import_module(name)

            required = (
                "_select_datasets",
                "_prepare_dataset",
                "_analyze_bulk",
                "_poisson_local_tail",
            )
            if all(hasattr(mod, attr) for attr in required):
                print(f"[import] using raw phenomenology module: {name}")
                return mod

            errors.append(
                f"{name}: imported, but required raw-phenomenology helpers "
                f"are missing."
            )
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "Could not import the raw/no-truth phenomenology analysis.\n"
        + "\n".join(errors)
    )


# =============================================================================
# IO
# =============================================================================

def _ensure_output_dir():
    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def _finite_percentiles(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "median": np.nan,
            "p16": np.nan,
            "p84": np.nan,
            "p025": np.nan,
            "p975": np.nan,
            "mean": np.nan,
        }

    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
        "p025": float(np.percentile(arr, 2.5)),
        "p975": float(np.percentile(arr, 97.5)),
        "mean": float(np.mean(arr)),
    }


# =============================================================================
# PHYSICS CONVERSIONS
# =============================================================================

def _event_hazard(K, N, p_dark):
    """
    Solve

        1-p_obs = (1-p_dark) exp(-Lambda)

    for Lambda.

    This removes the central dark-loss probability before assigning any
    additional loss to the candidate carrier burst.
    """
    K = float(K)
    N = float(N)
    p_dark = float(p_dark)

    if N <= 0:
        return np.nan

    p_obs = K / N

    if REQUIRE_POSITIVE_EXCESS and p_obs <= p_dark:
        return np.nan

    dark_survival = max(1.0 - p_dark, EPS)
    observed_survival = max(1.0 - p_obs, EPS)

    ratio = observed_survival / dark_survival

    # If p_obs > p_dark, ratio should be <= 1.
    ratio = min(max(ratio, EPS), 1.0)

    return float(-math.log(ratio))


def _particle_conversion_probability_from_hazard(lam_h):
    """
    Conditional probability that a center surviving ordinary dark relaxation
    is converted by the additional event.
    """
    if not np.isfinite(lam_h):
        return np.nan
    return float(1.0 - math.exp(-max(float(lam_h), 0.0)))


def _integrated_hole_density(lam_h):
    """
    H = integral n_h dt [cm^-3 s], assuming C_h.
    """
    if not np.isfinite(lam_h) or C_H_CM3_S <= 0:
        return np.nan
    return float(lam_h / C_H_CM3_S)


def _effective_hole_fluence(lam_h):
    """
    F_h [cm^-2], assuming an effective capture cross section.
    """
    if not np.isfinite(lam_h) or SIGMA_H_EFF_CM2 <= 0:
        return np.nan
    return float(lam_h / SIGMA_H_EFF_CM2)


def _generated_pairs(track_length_um):
    return float(EH_PAIRS_PER_UM * float(track_length_um))


def _uniform_fluence_collection_area_cm2(lam_h, track_length_um):
    """
    Very simple diagnostic model:

        Lambda = sigma_eff * F
        F = N_h / A_eff

    so

        A_eff = sigma_eff * N_h / Lambda.

    This is a "uniform-fluence equivalent collection area", NOT a measured
    diffusion area or physical cross section.
    """
    if not np.isfinite(lam_h) or lam_h <= 0:
        return np.nan

    n_holes = _generated_pairs(track_length_um)
    return float(SIGMA_H_EFF_CM2 * n_holes / lam_h)


def _equivalent_radius_um(area_cm2):
    if not np.isfinite(area_cm2) or area_cm2 <= 0:
        return np.nan
    radius_cm = math.sqrt(area_cm2 / math.pi)
    return float(radius_cm * 1e4)


# =============================================================================
# LOAD RAW DATA + BULK MODEL
# =============================================================================

def _cfg_file_stems(cfg):
    """
    Return every physical file stem represented by one logical DATASETS config.

    Supported:
        file_stem="one"
        file_stem=("one", "two")
        file_stems=["one", "two"]
    """
    if "file_stems" in cfg:
        stems = cfg.get("file_stems")
    else:
        stems = cfg.get("file_stem")

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
            f"{cfg.get('label', '<unnamed>')}: no file stems configured."
        )

    return stems


def _cfg_npz_overrides(cfg, num_files):
    """
    Resolve optional NPZ overrides for a logical config.
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
                f"{cfg.get('label', '<unnamed>')}: "
                f"{num_files} file stems require npz_path_overrides=[...] "
                "or no explicit override."
            )
        overrides = [overrides]

    if len(overrides) != int(num_files):
        raise ValueError(
            f"{cfg.get('label', '<unnamed>')}: "
            f"{len(overrides)} NPZ overrides for {num_files} file stems."
        )

    return overrides


def _wait_from_text(value):
    """
    Parse wait time from common particle-memory labels/stems.
    """
    value = str(value)

    patterns = (
        r"wait[_-](\d+(?:\.\d+)?)s",
        r"dark[_-]?wait[_-]?(\d+(?:\.\d+)?)s",
        r"source[_-]?off[_-]?(\d+(?:\.\d+)?)s",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            value,
            flags=re.IGNORECASE,
        )
        if match:
            return float(match.group(1))

    return np.nan


def _infer_wait_for_cfg(phen, cfg):
    """
    Infer a logical config's physical wait even when its file_stem is a list
    and the imported phenomenology parser only understands scalar stems.
    """
    parser = getattr(
        phen,
        "_parse_dark_wait_s",
        None,
    )

    if parser is not None:
        try:
            wait = float(parser(cfg))
            if np.isfinite(wait):
                return wait
        except Exception:
            pass

    waits = []

    for stem in _cfg_file_stems(cfg):
        wait = _wait_from_text(stem)
        if np.isfinite(wait):
            waits.append(float(wait))

    label_wait = _wait_from_text(
        cfg.get("label", "")
    )
    if np.isfinite(label_wait):
        waits.append(float(label_wait))

    if not waits:
        return np.nan

    ref = float(np.median(waits))

    if np.max(
        np.abs(np.asarray(waits) - ref)
    ) > 1e-8:
        raise ValueError(
            "One logical DATASETS entry mixes different waits: "
            + ", ".join(f"{w:g}" for w in waits)
        )

    return ref


def _expand_cfg_to_single_file_sources(
    cfg,
    wait_s,
):
    """
    Expand one logical same-wait config into normal one-file configs suitable
    for phen._prepare_dataset().
    """
    stems = _cfg_file_stems(cfg)
    overrides = _cfg_npz_overrides(
        cfg,
        len(stems),
    )

    base_label = str(
        cfg.get(
            "label",
            f"dark_wait_{wait_s:g}s",
        )
    )

    out = []

    for source_ind, (stem, override) in enumerate(
        zip(stems, overrides)
    ):
        source_cfg = dict(cfg)

        source_cfg.pop("file_stems", None)
        source_cfg.pop(
            "npz_path_overrides",
            None,
        )

        source_cfg["file_stem"] = str(stem)
        source_cfg["npz_path_override"] = override
        source_cfg["label"] = (
            base_label
            if len(stems) == 1
            else f"{base_label}_part{source_ind + 1}"
        )

        source_cfg[
            "_carrier_parent_label"
        ] = base_label
        source_cfg[
            "_carrier_source_ind"
        ] = int(source_ind)
        source_cfg[
            "_carrier_wait_s"
        ] = float(wait_s)

        out.append(source_cfg)

    return out


def _selected_waits_from_phen(phen):
    """
    Determine which physical waits the imported phenomenology intended to use.

    We use phen._select_datasets() only to determine the requested WAIT SET.
    We then rescan phen.base.DATASETS so additional same-wait acquisitions are
    not silently dropped by an older 'first match only' selector.
    """
    selected = phen._select_datasets()

    waits = []

    for cfg in selected:
        wait = _infer_wait_for_cfg(
            phen,
            cfg,
        )
        if np.isfinite(wait):
            waits.append(float(wait))

    # Fallback: if the old selector cannot parse a multi-file logical config,
    # use all finite waits found in base.DATASETS.
    if not waits:
        source = getattr(
            getattr(phen, "base", None),
            "DATASETS",
            None,
        )

        if source:
            for cfg in source:
                wait = _infer_wait_for_cfg(
                    phen,
                    cfg,
                )
                if np.isfinite(wait):
                    waits.append(float(wait))

    waits = sorted(
        {
            float(w)
            for w in waits
            if np.isfinite(w)
        }
    )

    if not waits:
        raise RuntimeError(
            "Could not determine requested dark waits from "
            "the imported phenomenology analysis."
        )

    return waits


def _collect_physical_source_cfgs_by_wait(
    phen,
):
    """
    Collect ALL physical acquisitions at each requested wait.

    Supports:
      * one config containing a tuple/list of file stems;
      * several separate DATASETS entries with the same wait.
    """
    source = getattr(
        getattr(phen, "base", None),
        "DATASETS",
        None,
    )

    if not source:
        # Fall back to the phenomenology selector itself if base.DATASETS is
        # not exposed.
        source = phen._select_datasets()

    wanted_waits = _selected_waits_from_phen(
        phen
    )

    groups = []

    for target in wanted_waits:
        source_cfgs = []
        parent_labels = []

        for cfg in source:
            wait = _infer_wait_for_cfg(
                phen,
                cfg,
            )

            if not (
                np.isfinite(wait)
                and abs(wait - target) < 1e-8
            ):
                continue

            expanded = (
                _expand_cfg_to_single_file_sources(
                    cfg,
                    wait_s=target,
                )
            )

            for one in expanded:
                one[
                    "_carrier_source_ind"
                ] = len(source_cfgs)
                source_cfgs.append(one)

            parent_labels.append(
                str(cfg.get("label", ""))
            )

        if not source_cfgs:
            continue

        group_label = next(
            (
                lab
                for lab in parent_labels
                if lab
            ),
            f"dark_wait_{target:g}s",
        )

        groups.append(
            {
                "wait_s": float(target),
                "label": group_label,
                "source_cfgs": source_cfgs,
            }
        )

    if not groups:
        raise RuntimeError(
            "No physical source acquisitions were found."
        )

    return groups


def _prepared_raw_run_span(ds):
    """
    Estimate the raw run-axis span represented by a prepared source.

    This is used only to make unique appended global run IDs.  The actual
    source-local run ID is stored separately and remains the authoritative
    identity inside each acquisition.
    """
    original = np.asarray(
        ds.get("original_run", []),
        dtype=int,
    )

    candidates = [
        ds.get("n_runs_loaded", None),
        ds.get("num_runs_loaded", None),
        ds.get("n_runs", None),
        ds.get("num_runs", None),
    ]

    for value in candidates:
        if value is None:
            continue
        try:
            value = int(value)
        except Exception:
            continue

        if value > 0:
            return value

    if original.size:
        return int(np.max(original)) + 1

    return 0


def _append_prepared_same_wait(
    prepared_parts,
    wait_s,
    label,
):
    """
    Append PREPARED count-only datasets belonging to one physical wait.

    Each source has already gone through its own:
      * file loading,
      * saved threshold application,
      * global-drop / quality rejection,
      * raw NV- -> NV0 classification.

    We append only K/N and the per-run source identity here.  The central bulk
    model is fit later to this combined wait-level population.
    """
    if not prepared_parts:
        raise ValueError(
            f"No prepared source files for wait={wait_s:g} s."
        )

    K_parts = []
    N_parts = []
    loss_fraction_parts = []
    initial_nvm_count_parts = []
    final_nvm_count_parts = []

    global_run_parts = []
    source_file_ind_parts = []
    source_local_run_parts = []
    source_label_parts = []
    source_stem_parts = []

    particle_exposures = []
    actual_waits = []

    raw_offset = 0

    for source_ind, ds in enumerate(
        prepared_parts
    ):
        part_wait = float(ds["wait_s"])

        if abs(part_wait - float(wait_s)) > 1e-6:
            raise ValueError(
                f"Source {source_ind} reports wait={part_wait:g} s "
                f"but belongs to wait={wait_s:g} s."
            )

        K = np.asarray(ds["K"], dtype=int)
        N = np.asarray(ds["N"], dtype=int)
        local_run = np.asarray(
            ds["original_run"],
            dtype=int,
        )

        if not (
            K.shape == N.shape == local_run.shape
        ):
            raise ValueError(
                f"Prepared source {source_ind} has inconsistent run arrays."
            )

        K_parts.append(K)
        N_parts.append(N)

        # phen._analyze_bulk() uses loss_fraction for the explicit upper-tail
        # trim sensitivity. Preserve the prepared value when available;
        # otherwise K/N is exactly the raw count-only definition.
        loss_fraction_parts.append(
            np.asarray(
                ds.get(
                    "loss_fraction",
                    np.divide(
                        K,
                        N,
                        out=np.full(K.shape, np.nan, dtype=float),
                        where=(N > 0),
                    ),
                ),
                dtype=float,
            )
        )

        # Retain these prepared count diagnostics when available.
        if "initial_nvm_count" in ds:
            initial_nvm_count_parts.append(
                np.asarray(ds["initial_nvm_count"], dtype=int)
            )
        if "final_nvm_count" in ds:
            final_nvm_count_parts.append(
                np.asarray(ds["final_nvm_count"], dtype=int)
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
                str(ds.get("label", "")),
                dtype=object,
            )
        )

        stem_value = ds.get(
            "file_stem",
            "",
        )
        source_stem_parts.append(
            np.full(
                len(K),
                str(stem_value),
                dtype=object,
            )
        )

        if np.isfinite(
            float(ds["particle_exposure_s"])
        ):
            particle_exposures.append(
                float(ds["particle_exposure_s"])
            )

        # The imported raw phenomenology bulk model explicitly requires
        # ds["actual_wait_s"].
        actual_wait_value = ds.get(
            "actual_wait_s",
            ds.get(
                "actual_dark_wait_s",
                ds.get("wait_s", wait_s),
            ),
        )
        try:
            actual_wait_value = float(actual_wait_value)
        except Exception:
            actual_wait_value = np.nan

        if np.isfinite(actual_wait_value):
            actual_waits.append(actual_wait_value)

        raw_offset += _prepared_raw_run_span(ds)

    K = np.concatenate(K_parts)
    N = np.concatenate(N_parts)
    loss_fraction = np.concatenate(
        loss_fraction_parts
    ).astype(float)

    if particle_exposures:
        particle_exposure_s = float(
            np.median(particle_exposures)
        )
    else:
        particle_exposure_s = (
            float(wait_s) + 0.63
        )

    if actual_waits:
        actual_wait_s = float(
            np.median(actual_waits)
        )
    else:
        actual_wait_s = float(wait_s)

    if not (
        K.shape
        == N.shape
        == loss_fraction.shape
    ):
        raise ValueError(
            "Same-wait append produced misaligned "
            "K/N/loss_fraction arrays."
        )

    combined = {
        "label": str(label),
        "wait_s": float(wait_s),
        "actual_wait_s": actual_wait_s,
        # Compatibility alias for downstream tables.
        "actual_dark_wait_s": actual_wait_s,
        "particle_exposure_s": particle_exposure_s,

        "K": K,
        "N": N,
        "loss_fraction": loss_fraction,

        # Unique appended raw-run ID.
        "original_run": np.concatenate(
            global_run_parts
        ).astype(int),

        # Exact physical-source identity.
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
        "raw_runs_loaded": int(raw_offset),
        "source_parts": prepared_parts,
    }

    if (
        len(initial_nvm_count_parts)
        == len(prepared_parts)
        and initial_nvm_count_parts
    ):
        combined["initial_nvm_count"] = np.concatenate(
            initial_nvm_count_parts
        ).astype(int)

    if (
        len(final_nvm_count_parts)
        == len(prepared_parts)
        and final_nvm_count_parts
    ):
        combined["final_nvm_count"] = np.concatenate(
            final_nvm_count_parts
        ).astype(int)

    print(
        f"[same-wait append] wait={wait_s:g} s: "
        f"{len(prepared_parts)} source file(s), "
        f"{len(K)} good/evaluable runs; "
        f"actual_wait={actual_wait_s:.6f} s."
    )

    for source_ind, ds in enumerate(
        prepared_parts
    ):
        print(
            f"    source {source_ind}: "
            f"{len(ds['K'])} good/evaluable runs | "
            f"{ds.get('file_stem', ds.get('label', ''))}"
        )

    return combined


def _source_identity_for_run(
    ds,
    run_ind,
):
    """
    Source-file identity for one combined valid run.
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


def _load_raw_analysis(phen):
    # Suppress imported script saving / final blocking display.
    if hasattr(phen, "SAVE_OUTPUTS"):
        phen.SAVE_OUTPUTS = False
    if hasattr(phen, "SHOW_FIGURES"):
        phen.SHOW_FIGURES = False

    print("\n" + "=" * 118)
    print("LOADING RAW / NO-TRUTH CHARGE-MEMORY DATA")
    print("=" * 118)

    groups = (
        _collect_physical_source_cfgs_by_wait(
            phen
        )
    )

    datasets = []

    for group in groups:
        wait_s = float(group["wait_s"])
        source_cfgs = list(
            group["source_cfgs"]
        )

        print(
            "\n"
            + "-" * 118
        )
        print(
            f"WAIT {wait_s:g} s: "
            f"{len(source_cfgs)} physical acquisition file(s)"
        )
        print("-" * 118)

        prepared_parts = []

        for source_ind, cfg in enumerate(
            source_cfgs
        ):
            print(
                f"[prepare] wait={wait_s:g} s "
                f"source {source_ind + 1}/{len(source_cfgs)}"
            )
            print(
                f"          {cfg['file_stem']}"
            )

            part = phen._prepare_dataset(cfg)
            prepared_parts.append(part)

        combined = _append_prepared_same_wait(
            prepared_parts,
            wait_s=wait_s,
            label=group["label"],
        )

        datasets.append(combined)

    datasets = sorted(
        datasets,
        key=lambda d: d["wait_s"],
    )

    # Fit ONE central bulk model per physical wait AFTER same-wait appending.
    bulk_rows, trim_rows, kinetic, kinetic_trim_rows = phen._analyze_bulk(
        datasets
    )

    # We only need the data, not figures produced by imported analysis.
    # plt.close("all")

    for ds in datasets:
        ds["tail"] = phen._poisson_local_tail(
            ds["K"],
            ds["N"],
            ds["bulk"]["p"],
        )

    return {
        "datasets": datasets,
        "bulk_rows": bulk_rows,
        "trim_rows": trim_rows,
        "kinetic": kinetic,
        "kinetic_trim_rows": kinetic_trim_rows,
    }


# =============================================================================
# EVENT-BY-EVENT CARRIER INFERENCE
# =============================================================================

def _infer_event_rows(datasets):
    rows = []

    for ds in datasets:
        K = np.asarray(ds["K"], dtype=int)
        N = np.asarray(ds["N"], dtype=int)
        sigma = np.asarray(ds["tail"]["sigma"], dtype=float)
        p_tail = np.asarray(ds["tail"]["p_tail"], dtype=float)
        p_dark = float(ds["bulk"]["p"])

        for zcut in SIGMA_CUTS:
            mask = np.isfinite(sigma) & (sigma >= float(zcut))

            for i in np.where(mask)[0]:
                k = int(K[i])
                n = int(N[i])
                p_obs = float(k / n)

                k_dark_expected = float(n * p_dark)
                k_excess = float(k - k_dark_expected)

                lam_h = _event_hazard(k, n, p_dark)
                p_event_cond = _particle_conversion_probability_from_hazard(lam_h)
                H = _integrated_hole_density(lam_h)
                fluence = _effective_hole_fluence(lam_h)

                row = {
                    "dataset": ds["label"],
                    "dark_wait_s": float(ds["wait_s"]),
                    "particle_exposure_s": float(ds["particle_exposure_s"]),
                    "sigma_cut": float(zcut),
                    "original_run": int(ds["original_run"][i]),
                    **_source_identity_for_run(ds, i),

                    # Direct observables / central model
                    "K_loss": k,
                    "N_evaluable": n,
                    "p_observed": p_obs,
                    "p_dark_bulk": p_dark,
                    "bulk_expected_K": k_dark_expected,
                    "K_excess_over_bulk": k_excess,
                    "poisson_sigma": float(sigma[i]),
                    "poisson_tail_p": float(p_tail[i]),

                    # Direct count-only carrier-burst parameter
                    "Lambda_h_event": lam_h,
                    "P_event_conditional_on_dark_survival": p_event_cond,

                    # Literature-dependent conversions
                    "integral_nh_dt_cm-3_s_using_Ch": H,
                    "effective_hole_fluence_cm-2_using_sigmaeff": fluence,
                }

                # Sensitivity to assumed transient duration.
                for tag, tau_s in ASSUMED_TRANSIENTS_S.items():
                    row[f"mean_nh_cm-3_if_{tag}"] = (
                        H / tau_s if np.isfinite(H) and tau_s > 0 else np.nan
                    )

                # Sensitivity to reference particle track length.
                for track_um in REFERENCE_TRACK_LENGTHS_UM:
                    tag = f"{track_um:g}um"
                    neh = _generated_pairs(track_um)

                    row[f"Neh_reference_{tag}"] = neh
                    row[f"Kexcess_over_Neh_{tag}"] = (
                        k_excess / neh if neh > 0 else np.nan
                    )

                    area_cm2 = _uniform_fluence_collection_area_cm2(
                        lam_h,
                        track_um,
                    )
                    row[f"uniform_fluence_Aeff_cm2_{tag}"] = area_cm2
                    row[f"uniform_fluence_radius_um_{tag}"] = (
                        _equivalent_radius_um(area_cm2)
                    )

                rows.append(row)

    return rows



# =============================================================================
# NEW: ALL-RUN PHYSICAL EVENT VARIABLE + FIXED-THRESHOLD ANALYSIS
# =============================================================================

def _infer_all_run_rows(datasets):
    """
    Build one physical carrier row for EVERY good/evaluable run.

    This is the key new layer.  Fixed Lambda_h or K_excess cuts are then
    applied to the same physical quantity at every dark wait.
    """
    rows = []

    for ds in datasets:
        K = np.asarray(ds["K"], dtype=int)
        N = np.asarray(ds["N"], dtype=int)
        sigma = np.asarray(ds["tail"]["sigma"], dtype=float)
        p_tail = np.asarray(ds["tail"]["p_tail"], dtype=float)
        p_dark = float(ds["bulk"]["p"])

        for i in range(len(K)):
            k = int(K[i])
            n = int(N[i])
            if n <= 0:
                continue

            p_obs = float(k / n)
            k_dark_expected = float(n * p_dark)
            k_excess = float(k - k_dark_expected)

            # For ordinary low-loss runs p_obs can be <= p_dark.  For the
            # physical upper-tail variable we define the additional positive
            # carrier hazard as zero in that case.
            lam_raw = _event_hazard(k, n, p_dark)
            lam_h = 0.0 if not np.isfinite(lam_raw) else float(max(lam_raw, 0.0))

            H = _integrated_hole_density(lam_h)
            fluence = _effective_hole_fluence(lam_h)

            rows.append({
                "dataset": ds["label"],
                "dark_wait_s": float(ds["wait_s"]),
                "particle_exposure_s": float(ds["particle_exposure_s"]),
                "original_run": int(ds["original_run"][i]),
                **_source_identity_for_run(ds, i),

                "K_loss": k,
                "N_evaluable": n,
                "p_observed": p_obs,
                "p_dark_bulk": p_dark,
                "bulk_expected_K": k_dark_expected,
                "K_excess_over_bulk": k_excess,

                "poisson_sigma": float(sigma[i]) if np.isfinite(sigma[i]) else np.nan,
                "poisson_tail_p": float(p_tail[i]) if np.isfinite(p_tail[i]) else np.nan,

                "Lambda_h_event": lam_h,
                "P_event_conditional_on_dark_survival": (
                    _particle_conversion_probability_from_hazard(lam_h)
                ),
                "integral_nh_dt_cm-3_s_using_Ch": H,
                "effective_hole_fluence_cm-2_using_sigmaeff": fluence,
            })

    return rows


def _q_exposure(q_bg, rate_s_inv, exposure_s):
    exposure_s = np.asarray(exposure_s, dtype=float)
    return 1.0 - (1.0 - float(q_bg)) * np.exp(
        -float(rate_s_inv) * exposure_s
    )


def _binomial_loglike_counts(events, runs, probs):
    events = np.asarray(events, dtype=int)
    runs = np.asarray(runs, dtype=int)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(np.sum(binom.logpmf(events, runs, probs)))


def _fit_constant_event_rate(events, runs, exposures):
    total_e = int(np.sum(events))
    total_n = int(np.sum(runs))
    q = total_e / total_n if total_n > 0 else np.nan
    probs = np.full(len(events), q, dtype=float)
    ll = _binomial_loglike_counts(events, runs, probs)

    return {
        "model": "constant_background",
        "loglike": ll,
        "num_params": 1,
        "q_bg": float(q),
        "R_event_s_inv": 0.0,
        "probs": probs,
    }


def _fit_free_exposure_event_rate(events, runs, exposures):
    events = np.asarray(events, dtype=int)
    runs = np.asarray(runs, dtype=int)
    exposures = np.asarray(exposures, dtype=float)

    def objective(x):
        q_bg, rate = map(float, x)
        probs = _q_exposure(q_bg, rate, exposures)
        return -_binomial_loglike_counts(events, runs, probs)

    observed = np.divide(
        events,
        runs,
        out=np.zeros_like(events, dtype=float),
        where=runs > 0,
    )
    q0 = float(max(1e-6, min(0.10, observed[0] if observed.size else 0.001)))

    starts = [
        (q0, 5e-5),
        (q0, 1e-4),
        (q0, R_MU_GEOM_S_INV),
        (q0, 5e-4),
        (max(q0 / 2.0, 1e-6), 1e-4),
    ]

    best = None
    for start in starts:
        opt = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[
                (1e-10, 0.20),
                (0.0, PHYSICAL_R_EVENT_MAX_S_INV),
            ],
        )
        if best is None or opt.fun < best.fun:
            best = opt

    if best is None:
        raise RuntimeError("Free exposure-rate fit failed.")

    q_bg, rate = map(float, best.x)
    probs = _q_exposure(q_bg, rate, exposures)

    return {
        "model": "background_plus_exposure",
        "loglike": -float(best.fun),
        "num_params": 2,
        "q_bg": q_bg,
        "R_event_s_inv": rate,
        "probs": probs,
        "success": bool(best.success),
        "message": str(best.message),
    }


def _fit_fixed_muon_event_rate(events, runs, exposures):
    events = np.asarray(events, dtype=int)
    runs = np.asarray(runs, dtype=int)
    exposures = np.asarray(exposures, dtype=float)

    def objective(x):
        q_bg = float(x[0])
        probs = _q_exposure(q_bg, R_MU_GEOM_S_INV, exposures)
        return -_binomial_loglike_counts(events, runs, probs)

    observed = np.divide(
        events,
        runs,
        out=np.zeros_like(events, dtype=float),
        where=runs > 0,
    )
    q0 = float(max(1e-6, min(0.10, observed[0] if observed.size else 0.001)))

    opt = minimize(
        objective,
        x0=np.asarray([q0], dtype=float),
        method="L-BFGS-B",
        bounds=[(1e-10, 0.20)],
    )

    q_bg = float(opt.x[0])
    probs = _q_exposure(q_bg, R_MU_GEOM_S_INV, exposures)

    return {
        "model": "background_plus_fixed_geometric_muon_rate",
        "loglike": -float(opt.fun),
        "num_params": 1,
        "q_bg": q_bg,
        "R_event_s_inv": float(R_MU_GEOM_S_INV),
        "probs": probs,
        "success": bool(opt.success),
        "message": str(opt.message),
    }


def _fit_pure_muon_event_rate(events, runs, exposures):
    events = np.asarray(events, dtype=int)
    runs = np.asarray(runs, dtype=int)
    exposures = np.asarray(exposures, dtype=float)
    probs = 1.0 - np.exp(-R_MU_GEOM_S_INV * exposures)
    ll = _binomial_loglike_counts(events, runs, probs)

    return {
        "model": "pure_geometric_muon",
        "loglike": ll,
        "num_params": 0,
        "q_bg": 0.0,
        "R_event_s_inv": float(R_MU_GEOM_S_INV),
        "probs": probs,
    }


def _add_information_criteria(models, total_runs):
    for model in models:
        k = int(model["num_params"])
        ll = float(model["loglike"])
        model["AIC"] = 2.0 * k - 2.0 * ll
        model["BIC"] = (
            math.log(max(int(total_runs), 1)) * k - 2.0 * ll
        )

    amin = min(m["AIC"] for m in models)
    bmin = min(m["BIC"] for m in models)

    for model in models:
        model["delta_AIC"] = model["AIC"] - amin
        model["delta_BIC"] = model["BIC"] - bmin

    return sorted(models, key=lambda m: m["AIC"])


def _bootstrap_physical_rate(events, runs, exposures, n_boot):
    """
    Binomial bootstrap around the observed fractions.

    This propagates finite event-count uncertainty into R_event.  It does not
    include uncertainty in p_dark or in the Lambda_h mapping.
    """
    if int(n_boot) <= 0:
        return np.asarray([], dtype=float)

    events = np.asarray(events, dtype=int)
    runs = np.asarray(runs, dtype=int)
    exposures = np.asarray(exposures, dtype=float)

    observed_q = np.divide(
        events,
        runs,
        out=np.zeros_like(events, dtype=float),
        where=runs > 0,
    )

    rng = np.random.default_rng(PHYSICAL_RATE_BOOTSTRAP_SEED)
    rates = []

    for _ in range(int(n_boot)):
        eboot = rng.binomial(runs, np.clip(observed_q, 0.0, 1.0))
        try:
            fit = _fit_free_exposure_event_rate(
                eboot,
                runs,
                exposures,
            )
            rates.append(float(fit["R_event_s_inv"]))
        except Exception:
            continue

    return np.asarray(rates, dtype=float)


def _wilson_interval(events, runs, z=1.959963984540054):
    if runs <= 0:
        return np.nan, np.nan

    n = float(runs)
    phat = float(events) / n
    denom = 1.0 + z * z / n

    center = (
        phat + z * z / (2.0 * n)
    ) / denom

    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / n
            + z * z / (4.0 * n * n)
        )
        / denom
    )

    return max(0.0, center - half), min(1.0, center + half)


def _threshold_value(row, threshold_type):
    if threshold_type == "Lambda_h":
        return float(row["Lambda_h_event"])
    if threshold_type == "K_excess":
        return float(row["K_excess_over_bulk"])
    raise ValueError(f"Unknown threshold type: {threshold_type}")


def _analyze_one_physical_threshold(all_run_rows, threshold_type, cut):
    waits = sorted({float(r["dark_wait_s"]) for r in all_run_rows})

    rate_rows = []
    selected_by_wait = {}

    for wait in waits:
        sub = [
            r for r in all_run_rows
            if abs(float(r["dark_wait_s"]) - wait) < 1e-12
        ]

        selected = [
            r for r in sub
            if np.isfinite(_threshold_value(r, threshold_type))
            and _threshold_value(r, threshold_type) >= float(cut)
        ]
        selected_by_wait[wait] = selected

        events = len(selected)
        runs = len(sub)
        frac = events / runs if runs else np.nan
        lo, hi = _wilson_interval(events, runs)

        exposure = (
            float(np.median([r["particle_exposure_s"] for r in sub]))
            if sub
            else np.nan
        )

        lambda_stats = _finite_percentiles(
            [r["Lambda_h_event"] for r in selected]
        )
        kex_stats = _finite_percentiles(
            [r["K_excess_over_bulk"] for r in selected]
        )

        rate_rows.append({
            "threshold_type": threshold_type,
            "threshold_value": float(cut),
            "dark_wait_s": wait,
            "particle_exposure_s": exposure,
            "events": events,
            "runs": runs,
            "observed_fraction": frac,
            "observed_percent": 100.0 * frac if np.isfinite(frac) else np.nan,
            "wilson95_low": lo,
            "wilson95_high": hi,
            "wilson95_low_percent": 100.0 * lo,
            "wilson95_high_percent": 100.0 * hi,
            "conditional_median_Lambda_h": lambda_stats["median"],
            "conditional_Lambda_h_p16": lambda_stats["p16"],
            "conditional_Lambda_h_p84": lambda_stats["p84"],
            "conditional_median_K_excess": kex_stats["median"],
            "conditional_K_excess_p16": kex_stats["p16"],
            "conditional_K_excess_p84": kex_stats["p84"],
        })

    rate_rows = sorted(rate_rows, key=lambda r: r["dark_wait_s"])

    events = np.asarray([r["events"] for r in rate_rows], dtype=int)
    runs = np.asarray([r["runs"] for r in rate_rows], dtype=int)
    exposures = np.asarray(
        [r["particle_exposure_s"] for r in rate_rows],
        dtype=float,
    )

    models = [
        _fit_constant_event_rate(events, runs, exposures),
        _fit_free_exposure_event_rate(events, runs, exposures),
        _fit_fixed_muon_event_rate(events, runs, exposures),
        _fit_pure_muon_event_rate(events, runs, exposures),
    ]
    models = _add_information_criteria(models, int(np.sum(runs)))

    boot_rates = _bootstrap_physical_rate(
        events,
        runs,
        exposures,
        PHYSICAL_RATE_BOOTSTRAP,
    )
    if boot_rates.size:
        rlo, rmed, rhi = np.percentile(boot_rates, [2.5, 50.0, 97.5])
    else:
        rlo = rmed = rhi = np.nan

    model_rows = []
    for model in models:
        row = {
            "threshold_type": threshold_type,
            "threshold_value": float(cut),
            "model": model["model"],
            "loglike": model["loglike"],
            "num_params": model["num_params"],
            "AIC": model["AIC"],
            "delta_AIC": model["delta_AIC"],
            "BIC": model["BIC"],
            "delta_BIC": model["delta_BIC"],
            "q_bg": model["q_bg"],
            "R_event_s_inv": model["R_event_s_inv"],
            "R_event_per_day": 86400.0 * model["R_event_s_inv"],
            "R_event_over_R_mu": (
                model["R_event_s_inv"] / R_MU_GEOM_S_INV
                if R_MU_GEOM_S_INV > 0
                else np.nan
            ),
        }

        if model["model"] == "background_plus_exposure":
            row["bootstrap_R_event_2p5_s_inv"] = rlo
            row["bootstrap_R_event_median_s_inv"] = rmed
            row["bootstrap_R_event_97p5_s_inv"] = rhi
            row["bootstrap_ratio_2p5"] = (
                rlo / R_MU_GEOM_S_INV
                if np.isfinite(rlo)
                else np.nan
            )
            row["bootstrap_ratio_97p5"] = (
                rhi / R_MU_GEOM_S_INV
                if np.isfinite(rhi)
                else np.nan
            )

        model_rows.append(row)

    # Conditional event-size comparison at fixed PHYSICAL threshold.
    x = np.asarray(
        [
            r["Lambda_h_event"]
            for r in selected_by_wait.get(30.0, [])
            if np.isfinite(r["Lambda_h_event"])
        ],
        dtype=float,
    )
    y = np.asarray(
        [
            r["Lambda_h_event"]
            for r in selected_by_wait.get(60.0, [])
            if np.isfinite(r["Lambda_h_event"])
        ],
        dtype=float,
    )

    if x.size and y.size:
        mw = mannwhitneyu(x, y, alternative="two-sided")
        ks = ks_2samp(x, y, alternative="two-sided", method="auto")
        mw_u = float(mw.statistic)
        mw_p = float(mw.pvalue)
        ks_stat = float(ks.statistic)
        ks_p = float(ks.pvalue)
    else:
        mw_u = mw_p = ks_stat = ks_p = np.nan

    comparison_row = {
        "threshold_type": threshold_type,
        "threshold_value": float(cut),
        "n_30s": int(x.size),
        "n_60s": int(y.size),
        "median_Lambda_30s": float(np.median(x)) if x.size else np.nan,
        "median_Lambda_60s": float(np.median(y)) if y.size else np.nan,
        "median_ratio_60_over_30": (
            float(np.median(y) / np.median(x))
            if x.size and y.size and np.median(x) > 0
            else np.nan
        ),
        "mannwhitney_U": mw_u,
        "mannwhitney_p": mw_p,
        "KS_statistic": ks_stat,
        "KS_p": ks_p,
        "selection_note": (
            "Fixed physical threshold across waits; cleaner than fixed sigma "
            "for testing conditional event-size stationarity."
        ),
    }

    return {
        "threshold_type": threshold_type,
        "threshold_value": float(cut),
        "rate_rows": rate_rows,
        "model_rows": model_rows,
        "comparison_row": comparison_row,
        "selected_by_wait": selected_by_wait,
    }


def _run_physical_threshold_analysis(all_run_rows):
    analyses = []

    for cut in LAMBDA_H_CUTS:
        analyses.append(
            _analyze_one_physical_threshold(
                all_run_rows,
                "Lambda_h",
                cut,
            )
        )

    for cut in K_EXCESS_CUTS:
        analyses.append(
            _analyze_one_physical_threshold(
                all_run_rows,
                "K_excess",
                cut,
            )
        )

    rate_rows = []
    model_rows = []
    comparison_rows = []

    for ana in analyses:
        rate_rows.extend(ana["rate_rows"])
        model_rows.extend(ana["model_rows"])
        comparison_rows.append(ana["comparison_row"])

    return {
        "analyses": analyses,
        "rate_rows": rate_rows,
        "model_rows": model_rows,
        "comparison_rows": comparison_rows,
    }


def _threshold_filename_token(threshold_type, cut):
    if threshold_type == "Lambda_h":
        return f"lambda_{float(cut):.3f}".replace(".", "p")
    return f"kexcess_{float(cut):g}".replace(".", "p")


def _plot_physical_rate_analysis(analysis):
    threshold_type = analysis["threshold_type"]
    cut = analysis["threshold_value"]
    rows = analysis["rate_rows"]

    T = np.asarray([r["particle_exposure_s"] for r in rows], dtype=float)
    q = np.asarray([r["observed_fraction"] for r in rows], dtype=float)
    lo = np.asarray([r["wilson95_low"] for r in rows], dtype=float)
    hi = np.asarray([r["wilson95_high"] for r in rows], dtype=float)

    yerr = np.vstack((q - lo, hi - q))

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.errorbar(
        T,
        100.0 * q,
        yerr=100.0 * yerr,
        marker="o",
        linestyle="none",
        capsize=4,
        label="observed (Wilson 95% CI)",
    )

    free = next(
        m for m in analysis["model_rows"]
        if m["model"] == "background_plus_exposure"
    )
    fixed = next(
        m for m in analysis["model_rows"]
        if m["model"] == "background_plus_fixed_geometric_muon_rate"
    )

    tgrid = np.linspace(0.0, max(T) * 1.05, 400)

    qfree = _q_exposure(
        free["q_bg"],
        free["R_event_s_inv"],
        tgrid,
    )
    qfixed = _q_exposure(
        fixed["q_bg"],
        R_MU_GEOM_S_INV,
        tgrid,
    )
    qpure = 1.0 - np.exp(-R_MU_GEOM_S_INV * tgrid)

    ax.plot(
        tgrid,
        100.0 * qfree,
        linewidth=1.6,
        label="background + fitted exposure rate",
    )
    ax.plot(
        tgrid,
        100.0 * qfixed,
        linestyle="--",
        linewidth=1.4,
        label="background + fixed geometric muon rate",
    )
    ax.plot(
        tgrid,
        100.0 * qpure,
        linestyle=":",
        linewidth=1.2,
        label="pure geometric muon crossing probability",
    )

    if threshold_type == "Lambda_h":
        title_cut = rf"$\Lambda_h \geq {cut:.3f}$"
    else:
        title_cut = rf"$K_{{\rm excess}} \geq {cut:g}$"

    ax.set_xlabel("Estimated rep11-to-rep12 exposure (s)")
    ax.set_ylabel("Runs above fixed physical threshold (%)")
    ax.set_title(f"Physical-threshold event rate: {title_cut}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if SAVE_OUTPUTS:
        token = _threshold_filename_token(threshold_type, cut)
        fig.savefig(
            OUTPUT_DIR / f"physical_event_rate_{token}.png",
            dpi=180,
            bbox_inches="tight",
        )

    return fig


def _plot_fixed_threshold_conditional_lambda(analysis):
    threshold_type = analysis["threshold_type"]
    cut = analysis["threshold_value"]

    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    for wait in sorted(analysis["selected_by_wait"]):
        sub = analysis["selected_by_wait"][wait]
        vals = np.asarray(
            [
                r["Lambda_h_event"]
                for r in sub
                if np.isfinite(r["Lambda_h_event"])
            ],
            dtype=float,
        )
        if vals.size == 0:
            continue

        jitter = (
            np.asarray([0.0])
            if vals.size == 1
            else np.linspace(-1.0, 1.0, vals.size)
        )

        ax.scatter(
            wait + jitter,
            vals,
            s=24,
            alpha=0.65,
        )

        med = float(np.median(vals))
        p16, p84 = np.percentile(vals, [16, 84])
        ax.errorbar(
            [wait],
            [med],
            yerr=[[med - p16], [p84 - med]],
            marker="D",
            capsize=4,
            linewidth=1.4,
        )

    if threshold_type == "Lambda_h":
        title_cut = rf"$\Lambda_h \geq {cut:.3f}$"
    else:
        title_cut = rf"$K_{{\rm excess}} \geq {cut:g}$"

    ax.set_xlabel("Dark wait (s)")
    ax.set_ylabel(r"Conditional event hazard $\Lambda_h$")
    ax.set_title(
        "Conditional event-size distribution at fixed physical threshold\n"
        + title_cut
    )
    fig.tight_layout()

    if SAVE_OUTPUTS:
        token = _threshold_filename_token(threshold_type, cut)
        fig.savefig(
            OUTPUT_DIR / f"conditional_lambda_{token}.png",
            dpi=180,
            bbox_inches="tight",
        )

    return fig


def _make_physical_threshold_figures(physical):
    figs = []

    for analysis in physical["analyses"]:
        figs.append(_plot_physical_rate_analysis(analysis))

        # Conditional Lambda plot is especially useful for the Lambda cuts and
        # for the primary K_excess control.
        if (
            analysis["threshold_type"] == "Lambda_h"
            or abs(
                analysis["threshold_value"] - PRIMARY_K_EXCESS_CUT
            ) < 1e-12
        ):
            figs.append(
                _plot_fixed_threshold_conditional_lambda(analysis)
            )

    # Combined Lambda-cut occurrence-rate plot.
    lambda_analyses = [
        a for a in physical["analyses"]
        if a["threshold_type"] == "Lambda_h"
    ]

    if lambda_analyses:
        fig, ax = plt.subplots(figsize=(8.7, 6.3))

        for analysis in lambda_analyses:
            rows = analysis["rate_rows"]
            T = np.asarray(
                [r["particle_exposure_s"] for r in rows],
                dtype=float,
            )
            q = np.asarray(
                [r["observed_fraction"] for r in rows],
                dtype=float,
            )

            ax.plot(
                T,
                100.0 * q,
                marker="o",
                linewidth=1.3,
                label=rf"$\Lambda_h\geq{analysis['threshold_value']:.2f}$",
            )

        ax.set_xlabel("Estimated rep11-to-rep12 exposure (s)")
        ax.set_ylabel("Runs above threshold (%)")
        ax.set_title(
            "Rare-event occurrence versus exposure at fixed carrier-hazard cuts"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()

        if SAVE_OUTPUTS:
            fig.savefig(
                OUTPUT_DIR / "physical_lambda_threshold_rates_combined.png",
                dpi=180,
                bbox_inches="tight",
            )
        figs.append(fig)

    return figs


def _physical_threshold_summary_text(physical):
    lines = []
    lines.append("")
    lines.append("FIXED PHYSICAL-THRESHOLD ANALYSIS")
    lines.append("=" * 86)
    lines.append(
        "Unlike the sigma comparison, these event cuts are applied to the same "
        "Lambda_h or K_excess value at every dark wait."
    )
    lines.append("")

    for analysis in physical["analyses"]:
        ttype = analysis["threshold_type"]
        cut = analysis["threshold_value"]

        if ttype == "Lambda_h":
            pcond = 100.0 * (1.0 - math.exp(-cut))
            lines.append(
                f"Lambda_h >= {cut:.3f} "
                f"(>= {pcond:.2f}% conditional extra conversion)"
            )
        else:
            lines.append(f"K_excess >= {cut:g}")

        lines.append("-" * 86)

        for row in analysis["rate_rows"]:
            lines.append(
                f"{row['dark_wait_s']:g} s: "
                f"{row['events']}/{row['runs']} = "
                f"{row['observed_percent']:.5f}% "
                f"[95% CI {row['wilson95_low_percent']:.5f}, "
                f"{row['wilson95_high_percent']:.5f}]; "
                f"median Lambda={row['conditional_median_Lambda_h']:.5g}."
            )

        free = next(
            r for r in analysis["model_rows"]
            if r["model"] == "background_plus_exposure"
        )
        fixed = next(
            r for r in analysis["model_rows"]
            if r["model"] == "background_plus_fixed_geometric_muon_rate"
        )
        const = next(
            r for r in analysis["model_rows"]
            if r["model"] == "constant_background"
        )

        lines.append(
            f"free exposure fit: q_bg={100*free['q_bg']:.5f}%, "
            f"R_event={free['R_event_s_inv']:.6e} s^-1 "
            f"({free['R_event_per_day']:.3f}/day), "
            f"R_event/R_mu={free['R_event_over_R_mu']:.3f}."
        )

        if np.isfinite(free.get("bootstrap_R_event_2p5_s_inv", np.nan)):
            lines.append(
                f"bootstrap 95% R_event/R_mu = "
                f"[{free['bootstrap_ratio_2p5']:.3f}, "
                f"{free['bootstrap_ratio_97p5']:.3f}]."
            )

        lines.append(
            f"model comparison: dAIC(constant)={const['delta_AIC']:.2f}; "
            f"dAIC(fixed-muon)={fixed['delta_AIC']:.2f}; "
            f"dAIC(free exposure)={free['delta_AIC']:.2f}."
        )

        cmp = analysis["comparison_row"]
        lines.append(
            f"30 vs 60 s conditional Lambda: "
            f"median {cmp['median_Lambda_30s']:.5g} vs "
            f"{cmp['median_Lambda_60s']:.5g}; "
            f"ratio={cmp['median_ratio_60_over_30']:.3f}; "
            f"Mann-Whitney p={cmp['mannwhitney_p']:.3g}; "
            f"KS p={cmp['KS_p']:.3g}."
        )
        lines.append("")

    lines.append("PHYSICAL INTERPRETATION TEST")
    lines.append("-" * 86)
    lines.append(
        "A stationary external-event picture is most naturally supported when:"
    )
    lines.append(
        "  (i) occurrence probability above a FIXED Lambda_h threshold rises "
        "with exposure, while"
    )
    lines.append(
        "  (ii) the conditional Lambda_h distribution of selected events is "
        "approximately unchanged between 30 and 60 s."
    )
    lines.append(
        "This is cleaner than the fixed-sigma comparison because the physical "
        "event-amplitude threshold no longer moves with the bulk distribution."
    )

    return "\n".join(lines)


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def _make_summary_rows(event_rows):
    rows = []

    waits = sorted({float(r["dark_wait_s"]) for r in event_rows})
    zcuts = sorted({float(r["sigma_cut"]) for r in event_rows})

    for zcut in zcuts:
        for wait in waits:
            sub = [
                r for r in event_rows
                if abs(r["sigma_cut"] - zcut) < 1e-12
                and abs(r["dark_wait_s"] - wait) < 1e-12
            ]
            if not sub:
                continue

            hazard = _finite_percentiles([r["Lambda_h_event"] for r in sub])
            kex = _finite_percentiles([r["K_excess_over_bulk"] for r in sub])
            H = _finite_percentiles([
                r["integral_nh_dt_cm-3_s_using_Ch"] for r in sub
            ])
            F = _finite_percentiles([
                r["effective_hole_fluence_cm-2_using_sigmaeff"] for r in sub
            ])

            row = {
                "sigma_cut": zcut,
                "dark_wait_s": wait,
                "events": len(sub),

                "Lambda_h_median": hazard["median"],
                "Lambda_h_p16": hazard["p16"],
                "Lambda_h_p84": hazard["p84"],
                "Lambda_h_p025": hazard["p025"],
                "Lambda_h_p975": hazard["p975"],

                "K_excess_median": kex["median"],
                "K_excess_p16": kex["p16"],
                "K_excess_p84": kex["p84"],

                "integral_nh_dt_median_cm-3_s": H["median"],
                "integral_nh_dt_p16_cm-3_s": H["p16"],
                "integral_nh_dt_p84_cm-3_s": H["p84"],

                "fluence_median_cm-2": F["median"],
                "fluence_p16_cm-2": F["p16"],
                "fluence_p84_cm-2": F["p84"],
            }

            for track_um in REFERENCE_TRACK_LENGTHS_UM:
                tag = f"{track_um:g}um"

                frac_stats = _finite_percentiles([
                    r[f"Kexcess_over_Neh_{tag}"] for r in sub
                ])
                rad_stats = _finite_percentiles([
                    r[f"uniform_fluence_radius_um_{tag}"] for r in sub
                ])

                row[f"Kexcess_over_Neh_median_{tag}"] = frac_stats["median"]
                row[f"uniform_radius_median_um_{tag}"] = rad_stats["median"]
                row[f"uniform_radius_p16_um_{tag}"] = rad_stats["p16"]
                row[f"uniform_radius_p84_um_{tag}"] = rad_stats["p84"]

            rows.append(row)

    return rows


def _compare_event_size_30_60(event_rows):
    """
    Diagnostic only.

    Compare the inferred carrier-burst hazard at 30 s and 60 s.
    If a stationary external particle population produces the same physical
    event-size distribution at each wait, Lambda_h might be broadly similar.

    BUT selection at a fixed sigma threshold is itself wait dependent because
    the central distribution changes.  Therefore this is not a clean causal
    test; it is a descriptive diagnostic.
    """
    rows = []

    for zcut in SIGMA_CUTS:
        x = np.asarray([
            r["Lambda_h_event"]
            for r in event_rows
            if abs(r["sigma_cut"] - zcut) < 1e-12
            and abs(r["dark_wait_s"] - 30.0) < 1e-9
            and np.isfinite(r["Lambda_h_event"])
        ], dtype=float)

        y = np.asarray([
            r["Lambda_h_event"]
            for r in event_rows
            if abs(r["sigma_cut"] - zcut) < 1e-12
            and abs(r["dark_wait_s"] - 60.0) < 1e-9
            and np.isfinite(r["Lambda_h_event"])
        ], dtype=float)

        if x.size and y.size:
            test = mannwhitneyu(x, y, alternative="two-sided")
            U = float(test.statistic)
            p = float(test.pvalue)
        else:
            U = np.nan
            p = np.nan

        rows.append({
            "sigma_cut": float(zcut),
            "n_30s": int(x.size),
            "n_60s": int(y.size),
            "median_Lambda_30s": float(np.median(x)) if x.size else np.nan,
            "median_Lambda_60s": float(np.median(y)) if y.size else np.nan,
            "median_ratio_60_over_30": (
                float(np.median(y) / np.median(x))
                if x.size and y.size and np.median(x) > 0
                else np.nan
            ),
            "mannwhitney_U": U,
            "mannwhitney_p": p,
            "interpretation_caution": (
                "Descriptive only: sigma-based selection threshold changes "
                "with the wait-dependent bulk distribution."
            ),
        })

    return rows


# =============================================================================
# FIGURES
# =============================================================================

def _plot_scatter_by_wait(event_rows, zcut, ykey, ylabel, filename, title):
    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    waits = sorted({
        float(r["dark_wait_s"])
        for r in event_rows
        if abs(r["sigma_cut"] - zcut) < 1e-12
    })

    for wait in waits:
        sub = [
            r for r in event_rows
            if abs(r["sigma_cut"] - zcut) < 1e-12
            and abs(r["dark_wait_s"] - wait) < 1e-12
            and np.isfinite(r[ykey])
        ]
        if not sub:
            continue

        vals = np.asarray([r[ykey] for r in sub], dtype=float)

        # Deterministic horizontal jitter for visibility.
        if vals.size == 1:
            jitter = np.asarray([0.0])
        else:
            jitter = np.linspace(-1.0, 1.0, vals.size)

        ax.scatter(
            wait + jitter,
            vals,
            s=24,
            alpha=0.65,
            label=f"{wait:g} s events" if wait == waits[0] else None,
        )

        med = float(np.median(vals))
        p16, p84 = np.percentile(vals, [16, 84])

        ax.errorbar(
            [wait],
            [med],
            yerr=[[med - p16], [p84 - med]],
            marker="D",
            capsize=4,
            linewidth=1.4,
        )

    ax.set_xlabel("Dark wait (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    if SAVE_OUTPUTS:
        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=180,
            bbox_inches="tight",
        )

    return fig


def _make_figures(event_rows):
    figs = []

    for zcut in SIGMA_CUTS:
        figs.append(
            _plot_scatter_by_wait(
                event_rows,
                zcut,
                "Lambda_h_event",
                r"Additional hole-capture hazard $\Lambda_h$",
                f"carrier_hazard_vs_wait_{zcut:g}sigma.png",
                f"Count-inferred carrier hazard for >= {zcut:g} sigma candidates",
            )
        )

        figs.append(
            _plot_scatter_by_wait(
                event_rows,
                zcut,
                "K_excess_over_bulk",
                "Excess losses above bulk expectation",
                f"excess_losses_vs_wait_{zcut:g}sigma.png",
                f"Event-size excess for >= {zcut:g} sigma candidates",
            )
        )

    # Primary 4-sigma physical-conversion plots.
    figs.append(
        _plot_scatter_by_wait(
            event_rows,
            PRIMARY_SIGMA_CUT,
            "integral_nh_dt_cm-3_s_using_Ch",
            r"$\int n_h(t)\,dt$ (cm$^{-3}$ s), assuming $C_h$",
            f"integrated_hole_exposure_vs_wait_{PRIMARY_SIGMA_CUT:g}sigma.png",
            "Literature-coefficient conversion of candidate event size",
        )
    )

    track_um = 66.7
    tag = f"{track_um:g}um"
    figs.append(
        _plot_scatter_by_wait(
            event_rows,
            PRIMARY_SIGMA_CUT,
            f"uniform_fluence_radius_um_{tag}",
            r"Uniform-fluence equivalent radius ($\mu$m)",
            f"effective_collection_radius_vs_wait_{PRIMARY_SIGMA_CUT:g}sigma.png",
            (
                "Model-dependent collection-radius scale "
                f"(reference track {track_um:g} um)"
            ),
        )
    )

    return figs


# =============================================================================
# HUMAN-READABLE SUMMARY
# =============================================================================

def _summary_lookup(summary_rows, zcut, wait):
    matches = [
        r for r in summary_rows
        if abs(r["sigma_cut"] - float(zcut)) < 1e-12
        and abs(r["dark_wait_s"] - float(wait)) < 1e-9
    ]
    return matches[0] if matches else None


def _build_summary(raw_analysis, summary_rows, comparison_rows):
    kinetic = raw_analysis["kinetic"]

    lines = []
    lines.append("COUNT-ONLY CARRIER-CAPTURE INFERENCE FROM RAW UPPER-TAIL EVENTS")
    lines.append("=" * 86)
    lines.append("")
    lines.append("No truth filtering, no image pixels, no spatial fit.")
    lines.append("")

    lines.append("LITERATURE / MODEL PRIORS")
    lines.append("-" * 86)
    lines.append(f"C_h = {C_H_CM3_S:.3e} cm^3/s")
    lines.append(f"C_e = {C_E_CM3_S:.3e} cm^3/s")
    lines.append(f"C_h/C_e = {C_H_CM3_S/C_E_CM3_S:.2f}")
    lines.append(f"sigma_h,eff = {SIGMA_H_EFF_CM2:.3e} cm^2")
    lines.append(f"pair yield = {EH_PAIRS_PER_UM:.2f} e-h pairs/um")
    lines.append("")

    lines.append("BULK DARK PROCESS")
    lines.append("-" * 86)
    lines.append(
        f"Gamma_dark = {kinetic['gamma_s_inv']:.6e} s^-1; "
        f"tau_dark = {kinetic['tau_s']/60.0:.2f} min."
    )
    lines.append(
        "The carrier-burst hazard below is calculated AFTER subtracting this "
        "central dark-loss probability."
    )
    lines.append("")

    lines.append("EVENT HAZARD DEFINITION")
    lines.append("-" * 86)
    lines.append(
        "Lambda_h = -ln[(1-K/N)/(1-p_dark)]."
    )
    lines.append(
        "Lambda_h is the main count-only data-derived carrier parameter."
    )
    lines.append(
        "H = integral n_h dt = Lambda_h/C_h and "
        "F_h = Lambda_h/sigma_h,eff depend on the adopted literature priors."
    )
    lines.append("")

    for zcut in SIGMA_CUTS:
        lines.append(f">= {zcut:g} SIGMA CANDIDATES")
        lines.append("-" * 86)

        for wait in (0.0, 30.0, 60.0):
            row = _summary_lookup(summary_rows, zcut, wait)
            if row is None:
                continue

            lines.append(
                f"{wait:g} s: n={row['events']}; "
                f"median Lambda_h={row['Lambda_h_median']:.5g} "
                f"[16-84%: {row['Lambda_h_p16']:.5g}, "
                f"{row['Lambda_h_p84']:.5g}]; "
                f"median excess K={row['K_excess_median']:.3f}."
            )

        lines.append("")

    lines.append("30 s VS 60 s EVENT-SIZE DIAGNOSTIC")
    lines.append("-" * 86)
    for row in comparison_rows:
        lines.append(
            f"{row['sigma_cut']:g} sigma: "
            f"median Lambda 30s={row['median_Lambda_30s']:.5g}, "
            f"60s={row['median_Lambda_60s']:.5g}, "
            f"ratio={row['median_ratio_60_over_30']:.3f}, "
            f"Mann-Whitney p={row['mannwhitney_p']:.3g}."
        )
    lines.append(
        "Caution: this comparison is descriptive because a fixed sigma cut "
        "corresponds to a wait-dependent absolute K threshold."
    )
    lines.append("")

    lines.append("WHAT YOU CAN CLAIM FROM COUNTS ALONE")
    lines.append("-" * 86)
    lines.append(
        "Directly inferable: p_dark(t), K_excess, and Lambda_h."
    )
    lines.append(
        "Inferable only after adopting a literature/model prior: "
        "integrated hole density, hole fluence, and equivalent collection-area "
        "scales."
    )
    lines.append(
        "Not separately identifiable from counts alone: C_h, D_h, tau_h, "
        "true diffusion length, track position, and microscopic carrier density."
    )
    lines.append("")
    lines.append(
        "A controlled alpha/beta source or synchronized particle tag is the "
        "cleanest way to calibrate E_dep -> N_eh -> Lambda_h -> K."
    )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    _ensure_output_dir()

    print("\n" + "#" * 118)
    print("RAW COUNT-ONLY CARRIER-CAPTURE INFERENCE")
    print("#" * 118)
    print("No truth filtering. No image pixels. No spatial fitting.")

    phen = _import_phenomenology()
    raw_analysis = _load_raw_analysis(phen)
    datasets = raw_analysis["datasets"]

    print("\n" + "=" * 118)
    print("SAME-WAIT SOURCE FILE SUMMARY")
    print("=" * 118)
    for ds in datasets:
        print(
            f"{ds['wait_s']:g} s: "
            f"{int(ds.get('num_source_files', 1))} source file(s), "
            f"{len(ds['K'])} good/evaluable combined runs."
        )

    event_rows = _infer_event_rows(datasets)
    summary_rows = _make_summary_rows(event_rows)
    comparison_rows = _compare_event_size_30_60(event_rows)

    # NEW: infer Lambda_h and K_excess for EVERY good run, then apply fixed
    # physical event-amplitude cuts across all waits.
    all_run_rows = _infer_all_run_rows(datasets)
    physical = _run_physical_threshold_analysis(all_run_rows)

    constants_rows = [{
        "C_h_cm3_s": C_H_CM3_S,
        "C_e_cm3_s": C_E_CM3_S,
        "Ch_over_Ce": C_H_CM3_S / C_E_CM3_S,
        "sigma_h_eff_cm2": SIGMA_H_EFF_CM2,
        "eh_pairs_per_um": EH_PAIRS_PER_UM,
        "reference_track_lengths_um": ";".join(
            f"{x:g}" for x in REFERENCE_TRACK_LENGTHS_UM
        ),
        "primary_sigma_cut": PRIMARY_SIGMA_CUT,
    }]

    _write_csv(OUTPUT_DIR / "carrier_event_parameters.csv", event_rows)
    _write_csv(
        OUTPUT_DIR / "all_run_carrier_parameters.csv",
        all_run_rows,
    )
    _write_csv(
        OUTPUT_DIR / "physical_threshold_rate_summary.csv",
        physical["rate_rows"],
    )
    _write_csv(
        OUTPUT_DIR / "physical_threshold_model_comparison.csv",
        physical["model_rows"],
    )
    _write_csv(
        OUTPUT_DIR / "physical_threshold_event_size_comparison.csv",
        physical["comparison_rows"],
    )
    _write_csv(
        OUTPUT_DIR / "carrier_event_summary_by_wait_sigma.csv",
        summary_rows,
    )
    _write_csv(
        OUTPUT_DIR / "event_size_wait_comparison.csv",
        comparison_rows,
    )
    _write_csv(
        OUTPUT_DIR / "carrier_model_constants.csv",
        constants_rows,
    )

    figs = _make_figures(event_rows)
    figs.extend(_make_physical_threshold_figures(physical))

    summary = _build_summary(
        raw_analysis,
        summary_rows,
        comparison_rows,
    )
    summary += "\n\n" + _physical_threshold_summary_text(physical)

    print("\n")
    print(summary)

    if SAVE_OUTPUTS:
        with (OUTPUT_DIR / "analysis_summary.txt").open(
            "w",
            encoding="utf-8",
        ) as f:
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
        "raw_analysis": raw_analysis,
        "event_rows": event_rows,
        "all_run_rows": all_run_rows,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "physical_threshold_analysis": physical,
        "figures": figs,
    }


if __name__ == "__main__":
    from utils import kplotlib as kpl
    kpl.init_kplotlib()
    analysis = main()
    kpl.show(block=SHOW_FIGURES)
