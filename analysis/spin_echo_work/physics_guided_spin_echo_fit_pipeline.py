"""
Spin-echo fitting pipeline (physics-guided priors + FFT seeds)
----------------------------------------------------------------
Drop this file next to your analysis code and run the entry-point
function `fit_spin_echo_dataset(...)`.

It assumes you already have:
  • norm_counts:   shape (N_NV, N_time)
  • norm_counts_ste: same shape (uncertainties)
  • total_evolution_times: 1D array of τ in microseconds
  • nv_list:       list of NV labels (ints or strings)
  • file_stems:    list of dataset source IDs (for provenance)
  • your fitter:   run_with_amp_and_freq_sweeps(...) imported/defined
  • your model:    fine_decay / fine_decay_fixed_revival

Optional: A catalog JSON with per-site expected ESEEM lines (f±, κ).
If provided, we build a small, diversified set of (f0,f1) candidates
(cycles/μs) to pass as priors to the fitter.

This module is careful to:
  • Bound frequency search by sampling-derived Nyquist limits
  • Use conservative amp windows first, then expand as needed
  • Serialize results with a stable schema (JSON)
  • Plot quick sanity visuals per NV (optional)
"""

from __future__ import annotations
import os, re, json, hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict
from utils import data_manager as dm
import sys
import numpy as np
import matplotlib.pyplot as plt

# --------- Imports expected to exist in your environment ---------------------
# - fine_decay / fine_decay_fixed_revival
# - run_with_amp_and_freq_sweeps
# - eseem_utils: optional physics-guided candidate builders

try:
    from eseem_utils import (
        load_catalog,
        select_records,
        build_feasible_freqs_from_recs,
    )
except Exception:
    pass
# We also expect your fitter to be importable from your analysis code
try:
    from fitter_module_for_spin_echo import (
        run_with_amp_and_freq_sweeps,
    )  # rename to your path
except Exception:
    # If already in the global namespace (e.g., Jupyter), this import may be unnecessary.
    pass


@dataclass
class FitConfig:
    # ---------------- Model toggles ----------------
    use_fixed_revival: bool = False
    enable_extras: bool = True
    # fixed_rev_time_us: float = 37.6
    fixed_rev_time_us: float = 28.6

    # ---------------- Amplitude sweep ----------------
    amp_bound_grid: Tuple[Tuple[float, float], ...] = ((-0.6, 0.6),)

    # ---------------- Frequency seeding/boxes ----------------
    freq_seed_band: Optional[Tuple[float, float]] = None
    freq_bound_boxes: Optional[Dict[str, Tuple[float, float]]] = None
    freq_seed_n_peaks: int = 16
    prior_pairs_topK: int = 8
    seed_include_harmonics: bool = True

    # ---------------- Catalog / allowed-lines ----------------
    catalog_path: Optional[str] = None
    orientations: Optional[List[Tuple[int, int, int]]] = None
    p_occ: float = 0.011
    f_range_kHz: Tuple[float, float] = (1.0, 6000.0)
    n_keep_each: int = 16
    min_sep_cyc_per_us: float = 0.01
    prior_weight_mode: str = "kappa"
    prior_per_line_scale: float = 1.0
    allowed_tol_kHz: float = 8.0
    allowed_weight_mode: str = "kappa"

    # These were added dynamically — include them explicitly:
    allowed_records: Optional[List[Dict]] = None
    allowed_orientations: Optional[List[Tuple[int, int, int]]] = None

    # ---------------- Prior pass & early stop ----------------
    prior_enable: bool = True
    early_stop_redchi: Optional[float] = None

    # ---------------- Budgets & screening ----------------
    coarse_K: int = 8
    small_maxfev: int = 40_000
    small_max_nfev: int = 60_000
    big_maxfev: int = 120_000
    big_max_nfev: int = 180_000
    refine_target_red: float = 1.05

    # ---------------- Misc ----------------
    verbose: bool = True


# ---------- tiny helpers you referenced ----------
def _infer_sampling_band(
    times_us: np.ndarray, margin: float = 0.05
) -> Tuple[float, float]:
    """
    Nyquist-safe band from sampling: fmin ~ 1/span, fmax ~ 1/(2*dt_min), then +/- margin.
    Returns (fmin, fmax) in cycles/µs.
    """
    t = np.asarray(times_us, float)
    if t.size < 2:
        return (0.0, 1.0)
    span = max(1e-9, float(t.max() - t.min()))
    dt_min = max(1e-9, float(np.diff(np.unique(t)).min()))
    fmin = (1.0 / span) * (1.0 - margin)
    fmax = (0.5 / dt_min) * (1.0 + margin)
    fmin = max(0.0, fmin)
    return (float(fmin), float(fmax))


def _load_catalog_records_if_needed(cfg: FitConfig) -> Optional[List[Dict]]:
    # Priority: explicit cfg.allowed_records > cfg.catalog_path > None
    if getattr(cfg, "allowed_records", None) is not None:
        return cfg.allowed_records
    if cfg.catalog_path:
        if not os.path.isfile(cfg.catalog_path):
            raise FileNotFoundError(f"Catalog not found: {cfg.catalog_path}")
        with open(cfg.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def filter_allowed_records(
    records: List[Dict],
    *,
    band_cyc_per_us: Tuple[float, float],
    orientations: Optional[List[Tuple[int, int, int]]] = None,
    kmin_kHz: float = 10.0,
    kmax_kHz: float = 10000.0,
    tol_kHz: float = 0.0,
    topK: int = 16,
    weight_field: str = "kappa",
) -> List[Dict]:
    """
    Filter + score allowed-line records so the fitter has a compact, high-value set.
    """
    lo, hi = band_cyc_per_us
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= max(lo, 1e-15):
        raise ValueError(f"Bad band_cyc_per_us={band_cyc_per_us}")

    kept: List[Tuple[float, Dict]] = []  # (score, record)
    ori_set = set(map(tuple, orientations)) if orientations else None

    for r in records:
        # Orientation filter
        if ori_set is not None:
            ori = r.get("orientation")
            if ori is None or tuple(ori) not in ori_set:
                continue

        # Collect candidate frequencies (kHz) within kHz span
        fk = []
        for key in ("f_plus_Hz", "f_minus_Hz"):
            val = r.get(key, None)
            if val is None:
                continue
            f_kHz = float(val) / 1e3
            if kmin_kHz <= f_kHz <= kmax_kHz:
                fk.append(f_kHz)
        if not fk:
            continue

        # Intersect with Nyquist band (in cycles/µs == MHz)
        cyc = [f / 1000.0 for f in fk]
        cyc = [
            f
            for f in cyc
            if (lo - tol_kHz / 1000.0) <= f <= (min(hi * 0.95, hi) + tol_kHz / 1000.0)
        ]
        if not cyc:
            continue

        # Weight: prefer larger kappa (or 1 if absent)
        wt = float(r.get(weight_field, 1.0))
        mid = 0.5 * (lo + hi)
        dist = min(abs(f - mid) for f in cyc)
        score = wt / (1e-9 + (1.0 + dist))
        kept.append((score, r))

    if not kept:
        return []

    kept.sort(key=lambda x: x[0], reverse=True)
    trimmed = [rec for _, rec in kept[: max(1, topK)]]

    # De-dup using (orientation, repr_kHz)
    seen = set()
    out = []
    for r in trimmed:
        repr_kHz = None
        for key in ("f_plus_Hz", "f_minus_Hz"):
            if r.get(key) is not None:
                repr_kHz = round(float(r[key]) / 1e3, 2)
                break
        key_ = (tuple(r.get("orientation", (0, 0, 0))), repr_kHz)
        if key_ in seen:
            continue
        seen.add(key_)
        out.append(r)
    return out


# --------------------------- Main entry-point -------------------------------
def fit_spin_echo_dataset(
    nv_list: List[int],
    norm_counts: np.ndarray,
    norm_counts_ste: np.ndarray,
    total_evolution_times: np.ndarray,
    file_stems: List[str],
    *,
    default_rev_us: float = 37.2,
    nv_inds: Optional[List[int]] = None,
    cfg: Optional[FitConfig] = None,
    save_dir: Optional[str] = None,  # (unused by dm; kept for API parity)
    make_plots: bool = False,
    nv_orientations=None,  # <--- this array you loaded from data
):
    """Run the full fit (amp+freq sweeps), optionally with physics-guided allowed-lines.

    Returns:
        (FitOutputs dataclass, saved_json_path)
    """
    if cfg is None:
        cfg = FitConfig()

    # ---------- inputs & band ----------
    norm_counts = np.asarray(norm_counts, float)
    norm_counts_ste = np.asarray(norm_counts_ste, float)
    total_evolution_times = np.asarray(total_evolution_times, float)

    if nv_inds is None:
        nv_inds = list(range(len(nv_list)))

    band = cfg.freq_seed_band
    if band is None:
        band = _infer_sampling_band(total_evolution_times, margin=0.05)
        # print(band)
        # band = (0.001, 6.0)  ## manual
        if cfg.verbose:
            print(
                f"[band] inferred from sampling: {band[0]:.6g}–{band[1]:.6g} cycles/μs"
            )

    # Default frequency boxes: respect Nyquist
    freq_boxes = cfg.freq_bound_boxes or {
        "osc_f0": (max(0.001, band[0]), band[1]),
        "osc_f1": (0.0, band[1]),
    }

    # ---------- allowed-lines preparation ----------
    allowed_records = _load_catalog_records_if_needed(cfg)

    # ---------- run the fitter ----------
    print("=== Spin-echo fits starting ===")
    (
        popts,
        pcovs,
        chis,
        fit_fns,
        fit_nv_labels,
        chosen_amp_bounds,
        chosen_overrides,
        site_ids,
    ) = run_with_amp_and_freq_sweeps(
        nv_list,
        norm_counts,
        norm_counts_ste,
        total_evolution_times,
        nv_inds=nv_inds,
        # amplitude sweep
        amp_bound_grid=cfg.amp_bound_grid,
        # frequency
        freq_bound_boxes=freq_boxes,
        freq_seed_band=band,
        freq_seed_n_peaks=cfg.freq_seed_n_peaks,
        seed_include_harmonics=cfg.seed_include_harmonics,
        # prior/extra overrides (phases, etc.) — pass None here; we rely on allowed-lines path
        extra_overrides_grid=None,
        # model toggles
        use_fixed_revival=cfg.use_fixed_revival,
        enable_extras=cfg.enable_extras,
        fixed_rev_time=cfg.fixed_rev_time_us,
        # allowed-lines (catalog) controls
        allowed_records=(
            allowed_records
            if allowed_records is not None
            else getattr(cfg, "allowed_records", None)
        ),
        allowed_orientations=getattr(cfg, "allowed_orientations", None)
        or cfg.orientations,
        allowed_tol_kHz=getattr(cfg, "allowed_tol_kHz", 8.0),
        allowed_weight_mode=getattr(cfg, "allowed_weight_mode", "kappa"),
        p_occ=getattr(cfg, "p_occ", 0.011),
        # prior / early-stop and budgets threaded (so CLI/CFG has one source of truth)
        prior_enable=getattr(cfg, "prior_enable", True),
        prior_pairs_topK=getattr(
            cfg, "prior_pairs_topK", 6
        ),  # reuse as small pair budget
        prior_min_sep=getattr(cfg, "min_sep_cyc_per_us", 0.01),
        early_stop_redchi=getattr(cfg, "early_stop_redchi", None),
        small_maxfev=getattr(cfg, "small_maxfev", 40_000),
        small_max_nfev=getattr(cfg, "small_max_nfev", 60_000),
        big_maxfev=getattr(cfg, "big_maxfev", 120_000),
        big_max_nfev=getattr(cfg, "big_max_nfev", 180_000),
        refine_target_red=getattr(cfg, "refine_target_red", 1.05),
        coarse_K=getattr(cfg, "coarse_K", 4),
        coarse_max_nfev=getattr(cfg, "coarse_max_nfev", 200_000),
        err_floor=1e-3,
        verbose=cfg.verbose,
        nv_orientations=nv_orientations,
    )

    # Package outputs (stable schema)
    unified_keys = [
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
        "osc_f1",
        "osc_phi0",
        "osc_phi1",
    ]

    out = dict(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        dataset_ids=list(map(str, file_stems)),
        default_rev_us=float(default_rev_us),
        nv_labels=list(map(int, fit_nv_labels)),
        times_us=np.asarray(total_evolution_times, float).tolist(),
        popts=[(p.tolist() if p is not None else None) for p in popts],
        pcovs=[(c.tolist() if c is not None else None) for c in pcovs],
        red_chi2=[(float(x) if x is not None else None) for x in chis],
        fit_fn_names=[(fn.__name__ if fn is not None else None) for fn in fit_fns],
        unified_keys=unified_keys,
        orientations=np.asarray(nv_orientations, int).tolist(),
        site_id=[int(s) for s in site_ids],  # <-- here
    )

    # Save JSON
    tokens = []
    for s in file_stems:
        m = re.search(r"-([A-Za-z0-9]+)-nv", s)  # e.g. "...-johnson-nv0_..."
        if m:
            tokens.append(m.group(1))
    sample = max(set(tokens), key=tokens.count) if tokens else "sample"
    srcsig = f"s{len(file_stems)}-{hashlib.sha1('|'.join(file_stems).encode()).hexdigest()[:6]}"
    # --- tiny signature of the source list ---
    name = f"{sample}_{len(fit_nv_labels)}nv_{srcsig}"
    # name   = f"{sample}_{len(fit_nv_labels)}nv_{date}_{rev}_{model}_{sweep}_{srcsig}"
    # print(name)
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, name)
    dm.save_raw_data(out, file_path)
    print(f"[SAVE] {file_path}")

    return out, file_path


if __name__ == "__main__":

    # --- 1) Load raw dataset ---
    # file_stem = "2025_11_11-01_15_45-johnson_204nv_s6-6d8f5c"   # dataset2 + dataset3
    # file_stem = "2025_11_15-14_11_49-johnson_204nv_s9-17d44b"  # dataset2 + dataset3 (more data and orientain)
    # file_stem = "2025_11_28-16_39_32-johnson_204nv_s6-902522"  # new B 65G field
    # file_stem = "2025_12_04-19_50_15-johnson_204nv_s9-2c83ab"  # new B 59G field
    # file_stem = "2025_12_24-19_56_03-johnson_204nv_s6-ff5e17"  # new B 62G field
    # file_stem = "2026_01_07-17_30_14-johnson_204nv_s12-06bcdd"  # new B 62G field
    file_stem = "2026_01_07-17_48_21-johnson_204nv_s10-34f8b7"  # new B 65G field

    data = dm.get_raw_data(file_stem=file_stem)
    nv_list = data["nv_list"]
    norm_counts = np.asarray(data["norm_counts"], float)
    norm_counts_ste = np.asarray(data["norm_counts_ste"], float)
    total_evolution_times = np.asarray(data["total_evolution_times"], float)
    nv_orientations = np.asarray(data["orientations"], dtype=int)

    # --- 2) Load the frequency catalog → allowed_records ---
    target_orientations = [
        (1, 1, -1),
        (-1, 1, 1),
    ]
    # NOTE: raw string for Windows backslashes
    catalog_path = r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_65G.json"
    with open(catalog_path, "r") as f:
        catalog_json = json.load(f)

    allowed_records = catalog_json

    # --- 3) Build config ---
    cfg = FitConfig(
        # Model toggles
        use_fixed_revival=False,
        enable_extras=True,
        fixed_rev_time_us=37.6,
        # Amplitude sweep (tight for speed; expand later if needed)
        amp_bound_grid=((-0.6, 0.6), (-1.0, 1.0)),
        # ---------------- Frequency seeding/boxes ----------------
        # freq_seed_band = (1.0, 6000),
        prior_pairs_topK=1500,
        # Catalog / allowed-lines (we're directly passing allowed_records below)
        catalog_path=catalog_path,  # <- set None to avoid double-loading via helper
        orientations=None,  # or e.g. [(1,1,1), (1,-1,1), ...]
        p_occ=0.011,
        f_range_kHz=(1, 6000),
        n_keep_each=1500,
        min_sep_cyc_per_us=0.03,  # coarser de-dup in band than 0.01
        prior_weight_mode="kappa",
        prior_per_line_scale=1.0,
        allowed_tol_kHz=2.0,
        allowed_weight_mode="kappa",
        # Hand the allowed-lines JSON directly:
        allowed_records=allowed_records,
        allowed_orientations=None,
        # Prior + budgets
        prior_enable=True,
        early_stop_redchi=None,  # e.g. 1.10 to stop early on good fits
        coarse_K=16,
        small_maxfev=200_000,
        small_max_nfev=200_000,
        big_maxfev=220_000,
        big_max_nfev=280_000,
        refine_target_red=1.05,
        verbose=True,
    )

    # --- 4) (Optional) choose a subset of NVs to fit (faster dev runs) ---
    # nv_inds = list(range(0, 20))
    nv_inds = None

    # --- 5) Run fits ---
    out, saved_path = fit_spin_echo_dataset(
        nv_list=nv_list,
        norm_counts=norm_counts,
        norm_counts_ste=norm_counts_ste,
        total_evolution_times=total_evolution_times,
        file_stems=[file_stem],
        # default_rev_us=37.2,
        default_rev_us=28.6,
        nv_inds=nv_inds,
        cfg=cfg,
        make_plots=True,
        nv_orientations=nv_orientations,  # <--- this array you loaded from data
    )

    # --- 6) Inspect results quickly ---
    print(f"\nSaved results: {saved_path}")
    # print(f"NVs fitted: {len(out.popts)}")
