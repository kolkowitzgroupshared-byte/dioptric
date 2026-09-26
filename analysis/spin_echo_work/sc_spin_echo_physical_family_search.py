# -*- coding: utf-8 -*-
"""V5 physical-family multi-13C search for the 49 G / 52 G spin-echo data."""
from __future__ import annotations
import argparse, ast, importlib, json, sys
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

VERSION = "v5b_band"

def load_backend(field):
    tag = str(field).upper()
    if tag == "49G":
        name = "analysis.spin_echo_work.sc_spin_echo_old_protocol_multic13_ranked_49G"
    elif tag == "52G":
        name = "analysis.spin_echo_work.sc_spin_echo_old_protocol_multic13_ranked_52G"
    else:
        raise ValueError("field must be 49G or 52G")
    return importlib.import_module(name)
def load_assigned_orientations(field, n_nv):
    """Return explicit per-NV orientation assignments when stored with the dataset."""
    tag = str(field).upper()
    if tag == "49G":
        try:
            mod = importlib.import_module(
                "analysis.spin_echo_work.sc_spin_echo_old_protocol_ranked_49G")
            _, nv_list, _, _, _, ori = mod.load_data()
            if ori is not None and len(ori) >= int(n_nv):
                return {i: tuple(int(v) for v in ori[i]) for i in range(int(n_nv))}
        except Exception as exc:
            print(f"[WARN] could not load 49G assigned orientations: {exc}")
    elif tag == "52G":
        p = Path(
            r"G:\nvdata\pc_Purcell\branch_master\resonance\2026_09"
            r"\2026_09_22-52G_orientation_map_212_from_resonance.json"
        )
        if p.exists():
            try:
                obj = json.load(open(p, "r", encoding="utf-8"))
                out = {int(k): tuple(int(x) for x in v) for k, v in obj.items()}
                if len(out) >= int(n_nv):
                    return out
            except Exception as exc:
                print(f"[WARN] could not load 52G resonance orientation map: {exc}")
    return {}

def load_orientation_map(path):
    """Load nv_index -> orientation from CSV or JSON."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out = {}
    if p.suffix.lower() == ".json":
        obj = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(obj, dict):
            for k, v in obj.items():
                out[int(k)] = tuple(int(x) for x in v)
        else:
            for r in obj:
                out[int(r["nv_index"])] = tuple(int(x) for x in r["orientation"])
    else:
        df = pd.read_csv(p)
        if "nv_index" not in df.columns:
            raise ValueError("orientation-map CSV needs nv_index")
        if "orientation" in df.columns:
            for r in df.itertuples():
                out[int(r.nv_index)] = tuple(int(x) for x in ast.literal_eval(str(r.orientation)))
        elif all(c in df.columns for c in ("ox","oy","oz")):
            for r in df.itertuples():
                out[int(r.nv_index)] = (int(r.ox), int(r.oy), int(r.oz))
        else:
            raise ValueError("orientation-map CSV needs orientation or ox,oy,oz columns")
    return out

def infer_orientation_from_fit_quality(base, attempt_rows):
    """Choose one orientation from the legacy single-C13 fit quality."""
    d = attempt_rows.copy()
    if d.empty:
        return None
    score = "score_primary" if "score_primary" in d.columns else "red_chi2"
    # Best successful single-site fit within each orientation, then best orientation.
    d = d.sort_values([score, "red_chi2", "aicc"])
    best = d.groupby("orientation", sort=False, as_index=False).first()
    best = best.sort_values([score, "red_chi2", "aicc"])
    if best.empty:
        return None
    return base.parse_orientation(best.iloc[0].orientation)

def choose_orientations(base, nv, attempt_rows, mode, explicit_map):
    """Return ([orientations], source) for this NV."""
    all_oris = sorted({base.parse_orientation(x) for x in attempt_rows.orientation.unique()})
    if mode == "all":
        return all_oris, "all"
    if mode in ("auto", "assigned") and int(nv) in explicit_map:
        return [tuple(explicit_map[int(nv)])], "assigned"
    if mode == "assigned":
        raise RuntimeError(f"NV {nv}: no explicit assigned orientation available")
    ori = infer_orientation_from_fit_quality(base, attempt_rows)
    if ori is None:
        raise RuntimeError(f"NV {nv}: could not infer orientation from fit quality")
    return [tuple(ori)], "fit-quality"

def load_catalog(base):
    d = pd.DataFrame(json.load(open(base.CATALOG_PATH, "r", encoding="utf-8")))
    d["ori"] = d["orientation"].apply(base.parse_orientation)
    d["site_id"] = d["site_index"].astype(int)
    d["f0_kHz"] = d["f_plus_Hz"].astype(float) / 1e3
    d["f1_kHz"] = d["f_minus_Hz"].astype(float) / 1e3
    return d

def screen_background(base, rows, t):
    vals = []
    for r in rows[:10]:
        try:
            vals.append(base.old_popt_to_theta(json.loads(r.popt_json))[:9])
        except Exception:
            pass
    if vals:
        bg = np.nanmedian(np.asarray(vals, float), axis=0)
    else:
        p0, _, _ = base.oldfit._initial_guess_and_bounds(t, np.ones_like(t), True, None)
        bg = np.asarray(p0[:9], float)
    bg = np.clip(bg, base.BG_LB + 1e-8, base.BG_UB - 1e-8)
    bg[4] = min(bg[4], base.t2_upper_us(t) / 1000.0)
    return bg

def experimental_frequency_band_mhz(t):
    """Resolvable search band from the sampled total-evolution-time grid.

    Mirrors the legacy old-protocol fitter:
      low  = max(1 kHz, 0.5 / total time span)
      high = 0.49 / minimum positive time step

    Times are in microseconds, so returned frequencies are in MHz.
    The upper edge stays slightly below the formal Nyquist limit.
    """
    tt = np.unique(np.asarray(t, float))
    span = max(float(np.ptp(tt)), 1e-9)
    dt = np.diff(tt)
    dt = dt[dt > 0]
    dt_min = float(np.min(dt)) if dt.size else span
    return max(0.001, 0.5 / span), 0.49 / dt_min


def physical_scores(base, t, e, bg, cat):
    _, contrast, carrier = base.carrier_from_bg(t, bg)
    trow = np.asarray(t, float)[None, :]
    kappa = cat["kappa"].to_numpy(float)
    fI = cat["fI_Hz"].to_numpy(float) / 1e6
    fm = cat["omega_ms_Hz"].to_numpy(float) / 1e6
    a = np.sin(0.5 * np.pi * fI[:, None] * trow) ** 2
    b = np.sin(0.5 * np.pi * fm[:, None] * trow) ** 2
    z = np.abs((2.0 * float(contrast) * kappa[:, None]) * carrier[None, :] * a * b)
    z /= base.safe_err(e)[None, :]
    return np.sqrt(np.sum(z * z, axis=1))

def linear_screen(base, t, y, e, bg, row):
    pred0 = base.model(t, np.asarray(bg, float), [])
    resid = np.asarray(y, float) - pred0
    carrier = base.carrier_from_bg(t, bg)[2]
    f0 = float(row.f0_kHz) / 1000.0
    f1 = float(row.f1_kHz) / 1000.0
    tt = np.asarray(t, float)
    X = np.column_stack([
        carrier * np.cos(2*np.pi*f0*tt), carrier * np.sin(2*np.pi*f0*tt),
        carrier * np.cos(2*np.pi*f1*tt), carrier * np.sin(2*np.pi*f1*tt),
    ])
    w = 1.0 / base.safe_err(e)
    Xw, rw = X * w[:, None], resid * w
    try:
        coef, *_ = np.linalg.lstsq(Xw, rw, rcond=None)
        fit = X @ coef
        dchi = float(np.sum((rw)**2) - np.sum(((resid-fit)*w)**2))
    except Exception:
        coef = np.zeros(4, float); dchi = 0.0
    a0, a1 = np.hypot(coef[0], coef[1]), np.hypot(coef[2], coef[3])
    amp = float(np.clip(0.5*(a0+a1), 0.02, 1.5))
    phi0 = float(np.arctan2(-coef[1], coef[0]))
    phi1 = float(np.arctan2(-coef[3], coef[2]))
    return max(0.0, dchi), amp, phi0, phi1
def frequency_families(df, tol_khz):
    """Anchored (non-transitive) frequency families; avoids chain-merging."""
    if df.empty:
        return df.copy(), []
    work = df.sort_values(
        ["matched_delta_chi2", "physical_snr"], ascending=False).copy()
    anchors, labels = [], {}
    tol = float(tol_khz)
    for idx, r in work.iterrows():
        best_f, best_d = None, np.inf
        for fi, (f0, f1) in enumerate(anchors):
            dist = max(abs(float(r.f0_kHz)-f0), abs(float(r.f1_kHz)-f1))
            if dist <= tol and dist < best_d:
                best_f, best_d = fi, dist
        if best_f is None:
            best_f = len(anchors)
            anchors.append((float(r.f0_kHz), float(r.f1_kHz)))
        labels[idx] = best_f
    outdf = df.copy()
    outdf["family_index"] = [labels[i] for i in outdf.index]
    summaries = []
    for fi, g in outdf.groupby("family_index", sort=False):
        g = g.sort_values(["matched_delta_chi2", "physical_snr"], ascending=False)
        rep = g.iloc[0].copy()
        rep["family_index"] = int(fi)
        rep["member_site_ids"] = json.dumps(sorted(g.site_id.astype(int).tolist()))
        rep["family_size"] = int(len(g))
        rep["family_f0_min_kHz"] = float(g.f0_kHz.min())
        rep["family_f0_max_kHz"] = float(g.f0_kHz.max())
        rep["family_f1_min_kHz"] = float(g.f1_kHz.min())
        rep["family_f1_max_kHz"] = float(g.f1_kHz.max())
        rep["family_max_physical_snr"] = float(g.physical_snr.max())
        rep["family_max_matched_delta_chi2"] = float(g.matched_delta_chi2.max())
        summaries.append(rep)
    return outdf, summaries
def site_dict(r, family_id=None):
    return dict(
        site_id=int(r.site_id), orientation=tuple(r.ori),
        kappa=float(r.kappa), distance_A=float(r.distance_A),
        f0_kHz=float(r.f0_kHz), f1_kHz=float(r.f1_kHz),
        family_id=family_id or str(getattr(r, "family_id", "")),
        member_site_ids=str(getattr(r, "member_site_ids", "[]")),
        physical_snr=float(getattr(r, "physical_snr", np.nan)),
        matched_delta_chi2=float(getattr(r, "matched_delta_chi2", np.nan)),
    )

def attempt_lookup_rows(base, attempts, nv):
    d = attempts[(attempts.nv_index == nv) & (attempts.status == "ok")].copy()
    if d.empty:
        return d, {}
    score = "score_primary" if "score_primary" in d.columns else "red_chi2"
    d = d.sort_values([score, "red_chi2", "aicc"])
    d = d.drop_duplicates(["orientation", "site_id"], keep="first")
    lookup = {}
    for r in d.itertuples():
        lookup[(base.parse_orientation(r.orientation), int(r.site_id))] = r
    return d, lookup

def load_incumbents(prefix):
    parent = prefix.parent
    pats = ["*v2*_candidate_fits.csv", "*v4*topK*_candidate_fits.csv"]
    files = []
    for pat in pats:
        files += [p for p in parent.glob(pat)
                  if "_subset_" not in p.name and "_quick" not in p.name]
    best, protected = {}, {}
    for p in sorted(set(files)):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        for r in d.itertuples():
            order = int(r.model_order)
            ids = tuple(int(getattr(r, f"c13_{j}_site_id"))
                        for j in range(1, order+1))
            try:
                ori = tuple(ast.literal_eval(str(r.orientation))) if order else ()
            except Exception:
                ori = ()
            key = (int(r.nv_index), order, ori, str(r.site_key))
            if key not in best or float(r.chi2) < best[key]["chi2"]:
                best[key] = dict(chi2=float(r.chi2),
                                 theta=np.asarray(json.loads(r.theta_json), float),
                                 site_ids=ids, orientation=ori, source=str(p))
            nv = int(r.nv_index)
            if nv not in protected or float(r.bic) < protected[nv]["bic"]:
                protected[nv] = dict(bic=float(r.bic), site_ids=ids,
                                     orientation=ori, source=str(p))
    return best, files, protected

def aligned_incumbent_seed(inc, sites):
    if inc is None:
        return None
    th = np.asarray(inc["theta"], float)
    blocks = {sid: th[9+3*i:12+3*i] for i, sid in enumerate(inc["site_ids"])}
    ids = [int(s["site_id"]) for s in sites]
    if not all(sid in blocks for sid in ids):
        return None
    return np.r_[th[:9], *[blocks[sid] for sid in ids]]

def incumbent_for(incumbents, nv, sites):
    sitekey = str(tuple(sorted(int(s["site_id"]) for s in sites)))
    ori = tuple(sites[0]["orientation"]) if sites else ()
    return incumbents.get((int(nv), len(sites), ori, sitekey))

def fit_model_preserve(base, t, y, e, sites, seeds, incumbent_seed=None):
    """Run the normal optimizer but never discard an already-better incumbent."""
    f = base.fit_model(t, y, e, sites, seeds)
    if incumbent_seed is None:
        return f
    th = np.asarray(incumbent_seed, float)
    try:
        pred = base.model(t, th, sites)
        st = base.calc_stats(y, e, pred, len(th))
        incfit = dict(theta=th, pred=pred, sites=tuple(sites),
                      site_key=base.site_key(sites),
                      t2_limit_us=float(base.t2_upper_us(t)), **st)
        if f is None or incfit["chi2"] < f["chi2"]:
            return incfit
    except Exception:
        pass
    return f

def augment_row(row, fit):
    for j, s in enumerate(fit["sites"], 1):
        row[f"c13_{j}_family_id"] = s.get("family_id", "")
        row[f"c13_{j}_family_members"] = s.get("member_site_ids", "[]")
        row[f"c13_{j}_physical_snr"] = s.get("physical_snr", np.nan)
        row[f"c13_{j}_matched_delta_chi2"] = s.get("matched_delta_chi2", np.nan)
    return row
def fit_single(base, t, y, e, bgfit, s, attempt_row, inc):
    seeds = []
    if attempt_row is not None:
        try:
            seeds.append(base.old_popt_to_theta(json.loads(attempt_row.popt_json)))
        except Exception:
            pass
    amp = float(s.get("amp_seed", 0.10))
    ph0 = float(s.get("phi0_seed", 0.0)); ph1 = float(s.get("phi1_seed", 0.0))
    seeds.append(np.r_[np.asarray(bgfit["theta"], float)[:9], amp, ph0, ph1])
    iseed = aligned_incumbent_seed(inc, [s])
    if iseed is not None:
        seeds.insert(0, iseed)
    return fit_model_preserve(base, t, y, e, [s], seeds, iseed)

def fit_pair(base, t, y, e, a, b, inc):
    sites = [a["sites"][0], b["sites"][0]]
    ta, tb = np.asarray(a["theta"], float), np.asarray(b["theta"], float)
    seeds = [
        np.r_[ta[:9], ta[9:12], tb[9:12]],
        np.r_[tb[:9], ta[9:12], tb[9:12]],
    ]
    iseed = aligned_incumbent_seed(inc, sites)
    if iseed is not None:
        seeds.insert(0, iseed)
    return fit_model_preserve(base, t, y, e, sites, seeds, iseed)

def fit_triple(base, t, y, e, singles_by_id, pair_map, ids, inc):
    pair_parents = []
    for pair_ids in combinations(ids, 2):
        p = pair_map.get(tuple(sorted(pair_ids)))
        if p is None:
            continue
        rem = next(x for x in ids if x not in pair_ids)
        pair_parents.append((p, singles_by_id[rem]))
    if not pair_parents:
        return None
    pair_parents.sort(key=lambda ps: (ps[0]["bic"], ps[0]["red_chi2"]))
    bp, bs = pair_parents[0]
    sites = list(bp["sites"]) + [bs["sites"][0]]
    seeds = [base.triple_seed_aligned(p, s, sites) for p, s in pair_parents]
    iseed = aligned_incumbent_seed(inc, sites)
    if iseed is not None:
        seeds.insert(0, iseed)
    return fit_model_preserve(base, t, y, e, sites, seeds, iseed)
def prepare_families(base, nv, t, y, e, attempt_rows, catalog, snr_min,
                     tol_khz, max_sites, max_per_family, protected_info,
                     orientations, orientation_source):
    rows_nt = list(attempt_rows.itertuples())
    bgfit = base.background_fit(t, y, e, rows_nt)
    if bgfit is None:
        raise RuntimeError(f"NV {nv}: background fit failed")
    # Use the fitted N=0 background for physical detectability.  Median
    # single-site backgrounds can collapse contrast toward zero on difficult NVs.
    bg_screen = np.asarray(bgfit["theta"], float)[:9]
    oris = [tuple(o) for o in orientations]
    band_lo_mhz, band_hi_mhz = experimental_frequency_band_mhz(t)
    site_rows, family_rows, selected = [], [], []
    for oi, ori in enumerate(oris):
        protected_sites = set()
        if tuple(protected_info.get("orientation", ())) == tuple(ori):
            protected_sites = set(int(x) for x in protected_info.get("site_ids", ()))
        g = catalog[catalog.ori == ori].copy()
        if g.empty:
            continue
        # Hard experimental bandwidth constraint.  Both catalog frequencies
        # must be directly resolvable on this Hahn-echo time grid; unlike the
        # SNR prescreen, protected incumbents do NOT bypass this constraint.
        f0_mhz = g["f0_kHz"].to_numpy(float) / 1000.0
        f1_mhz = g["f1_kHz"].to_numpy(float) / 1000.0
        in_band = (
            (f0_mhz >= band_lo_mhz) & (f0_mhz <= band_hi_mhz)
            & (f1_mhz >= band_lo_mhz) & (f1_mhz <= band_hi_mhz)
        )
        g = g.loc[in_band].copy()
        if g.empty:
            continue
        g["physical_snr"] = physical_scores(base, t, e, bg_screen, g)
        g["protected_incumbent_site"] = g.site_id.isin(protected_sites)
        g["passes_physical_snr"] = g.physical_snr >= float(snr_min)
        # Never throw away a previously best V2/V4 site solely because the
        # approximate physical prescreen rates it below threshold.
        g = g[g.passes_physical_snr | g.protected_incumbent_site].copy()
        if g.empty:
            continue
        vals = [linear_screen(base, t, y, e, bg_screen, r) for r in g.itertuples()]
        g["matched_delta_chi2"] = [v[0] for v in vals]
        g["amp_seed"] = [v[1] for v in vals]
        g["phi0_seed"] = [v[2] for v in vals]
        g["phi1_seed"] = [v[3] for v in vals]
        g, fams = frequency_families(g, tol_khz)
        fams.sort(key=lambda r: (float(r["family_max_matched_delta_chi2"]),
                                 float(r["family_max_physical_snr"])), reverse=True)
        fam_name = {int(r["family_index"]): f"o{oi}_f{fi:03d}"
                    for fi, r in enumerate(fams)}
        g["family_id"] = [fam_name[int(x)] for x in g.family_index]
        member_json = {
            fid: json.dumps(sorted(g[g.family_id == fid].site_id.astype(int).tolist()))
            for fid in set(g.family_id)
        }
        g["member_site_ids"] = [member_json[str(fid)] for fid in g.family_id]
        selected_idx, counts = [], {}
        # Protect previously best V2/V4 sites if they pass the physical screen.
        for idx, r in g[g.site_id.isin(protected_sites)].sort_values(
                "matched_delta_chi2", ascending=False).iterrows():
            fid = str(r.family_id)
            if counts.get(fid, 0) < int(max_per_family):
                selected_idx.append(idx); counts[fid] = counts.get(fid, 0) + 1
        # Fill remaining slots by matched score, limiting aliases per family.
        for idx, r in g.sort_values(
                ["matched_delta_chi2","physical_snr"], ascending=False).iterrows():
            if idx in selected_idx:
                continue
            fid = str(r.family_id)
            if counts.get(fid, 0) >= int(max_per_family):
                continue
            if len(selected_idx) >= int(max_sites):
                break
            selected_idx.append(idx); counts[fid] = counts.get(fid, 0) + 1
        selected_set = set(selected_idx)
        for r in g.itertuples():
            site_rows.append(dict(
                nv_index=int(nv), orientation=str(ori),
                orientation_source=str(orientation_source), site_id=int(r.site_id),
                family_id=str(r.family_id), selected_for_search=r.Index in selected_set,
                passes_physical_snr=bool(r.passes_physical_snr),
                protected_incumbent_site=bool(r.protected_incumbent_site),
                kappa=float(r.kappa), A_par_kHz=float(r.A_par_Hz)/1e3,
                A_perp_kHz=float(r.A_perp_Hz)/1e3, distance_A=float(r.distance_A),
                f0_kHz=float(r.f0_kHz), f1_kHz=float(r.f1_kHz),
                frequency_band_low_kHz=1000.0*band_lo_mhz,
                frequency_band_high_kHz=1000.0*band_hi_mhz,
                physical_snr=float(r.physical_snr),
                matched_delta_chi2=float(r.matched_delta_chi2)))
        for fi, r in enumerate(fams):
            fid = fam_name[int(r["family_index"])]
            members = g[g.family_id == fid]
            family_rows.append(dict(
                nv_index=int(nv), orientation=str(ori),
                orientation_source=str(orientation_source), family_id=fid,
                representative_site_id=int(r.site_id), family_size=int(r.family_size),
                member_site_ids=str(r.member_site_ids),
                selected_site_ids=json.dumps(sorted(
                    members[members.index.isin(selected_set)].site_id.astype(int).tolist())),
                f0_kHz=float(r.f0_kHz), f1_kHz=float(r.f1_kHz),
                frequency_band_low_kHz=1000.0*band_lo_mhz,
                frequency_band_high_kHz=1000.0*band_hi_mhz,
                family_max_physical_snr=float(r.family_max_physical_snr),
                family_max_matched_delta_chi2=float(r.family_max_matched_delta_chi2)))
        selected.extend([g.loc[idx].copy() for idx in selected_idx])
    return bgfit, selected, site_rows, family_rows

def fit_one_nv(base, nv, t, y, e, attempts, catalog, incumbents, protected,
               max_spins, snr_min, tol_khz, max_sites, max_per_family,
               orientation_mode, explicit_orientation_map):
    ad, lookup = attempt_lookup_rows(base, attempts, nv)
    if ad.empty:
        raise RuntimeError(f"NV {nv}: no successful legacy attempts")
    oris, orientation_source = choose_orientations(
        base, nv, ad, orientation_mode, explicit_orientation_map)
    protected_info = protected.get(int(nv), {})
    bgfit, fams, site_rows, family_rows = prepare_families(
        base, nv, t, y, e, ad, catalog, snr_min, tol_khz,
        max_sites, max_per_family, protected_info, oris, orientation_source)
    fits0 = [bgfit]
    singles = []
    for fr in fams:
        s = site_dict(fr, str(fr["family_id"]))
        s["amp_seed"] = float(fr["amp_seed"]); s["phi0_seed"] = float(fr["phi0_seed"])
        s["phi1_seed"] = float(fr["phi1_seed"])
        ar = lookup.get((tuple(s["orientation"]), int(s["site_id"])))
        inc = incumbent_for(incumbents, nv, [s])
        f = fit_single(base, t, y, e, bgfit, s, ar, inc)
        if f is not None:
            singles.append(f)
    singles.sort(key=lambda z: (z["bic"], z["red_chi2"]))
    pairs = []
    pair_map = {}
    if max_spins >= 2:
        byori = {}
        for s in singles:
            byori.setdefault(tuple(s["sites"][0]["orientation"]), []).append(s)
        for ori, fs in byori.items():
            for a, b in combinations(fs, 2):
                sites = [a["sites"][0], b["sites"][0]]
                inc = incumbent_for(incumbents, nv, sites)
                f = fit_pair(base, t, y, e, a, b, inc)
                if f is not None:
                    pairs.append(f)
                    pair_map[tuple(sorted(int(x["site_id"]) for x in f["sites"]))] = f
        pairs.sort(key=lambda z: (z["bic"], z["red_chi2"]))
    triples = []
    if max_spins >= 3:
        byori_ids = {}
        singles_by_id = {}
        for s in singles:
            sid = int(s["sites"][0]["site_id"]); singles_by_id[sid] = s
            byori_ids.setdefault(tuple(s["sites"][0]["orientation"]), []).append(sid)
        for ori, ids0 in byori_ids.items():
            for ids in combinations(ids0, 3):
                sites0 = [singles_by_id[i]["sites"][0] for i in ids]
                inc = incumbent_for(incumbents, nv, sites0)
                f = fit_triple(base, t, y, e, singles_by_id, pair_map, ids, inc)
                if f is not None:
                    triples.append(f)
        triples.sort(key=lambda z: (z["bic"], z["red_chi2"]))
    allfits = {0: fits0, 1: singles, 2: pairs, 3: triples}
    recs = []
    for order in range(max_spins + 1):
        for rank, f in enumerate(allfits.get(order, []), 1):
            row = augment_row(base.public_row(nv, order, rank, f), f)
            row["orientation_source"] = orientation_source
            row["orientation_candidates"] = str(tuple(oris))
            recs.append(row)
    if not recs:
        raise RuntimeError(f"NV {nv}: no fitted candidates")
    bic_idx = sorted(range(len(recs)), key=lambda i: (recs[i]["bic"], recs[i]["red_chi2"]))
    aic_idx = sorted(range(len(recs)), key=lambda i: (recs[i]["aicc"], recs[i]["bic"]))
    red_idx = sorted(range(len(recs)), key=lambda i: (recs[i]["red_chi2"], recs[i]["bic"]))
    for rank, i in enumerate(bic_idx, 1): recs[i]["rank_global_bic"] = rank
    for rank, i in enumerate(aic_idx, 1): recs[i]["rank_global_aicc"] = rank
    for rank, i in enumerate(red_idx, 1): recs[i]["rank_global_redchi"] = rank
    best = recs[bic_idx[0]]
    print(f"[NV {nv:3d}] ori={','.join(map(str,oris))} ({orientation_source}) "
          f"sites={len(fams)} BIC N={best['model_order']} "
          f"{best['site_key']} chi2r={best['red_chi2']:.3f}")
    return recs, site_rows, family_rows

def parse_nv_list(s):
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]

def safe_tag(x):
    return str(x).replace(".", "p").replace("-", "m")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=("49G", "52G"), required=True)
    ap.add_argument("--nv", type=str, default=None)
    ap.add_argument("--max-spins", type=int, default=3, choices=(1,2,3))
    ap.add_argument("--physical-snr-min", type=float, default=0.5)
    ap.add_argument("--family-tol-khz", type=float, default=12.0)
    ap.add_argument("--max-sites", type=int, default=15,
                    help="Maximum fitted sites per orientation after physical/family screening.")
    ap.add_argument("--max-per-family", type=int, default=3,
                    help="Maximum near-degenerate sites retained from one frequency family.")
    ap.add_argument("--orientation-mode", choices=("auto","assigned","fit-quality","all"),
                    default="auto",
                    help="auto=assigned if available else fit-quality; assigned=require map; fit-quality=infer from legacy single fits; all=diagnostic.")
    ap.add_argument("--orientation-map", type=str, default=None,
                    help="Optional CSV/JSON nv_index->orientation map; overrides embedded assignments in auto/assigned mode.")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    base = load_backend(args.field)
    _, checkpoint, _, prefix = base.discover_paths()
    attempts_path = Path(str(prefix) + "_all_attempts.csv.gz")
    print("="*100)
    print(f"{args.field} V5 PHYSICAL-FAMILY MULTI-13C SEARCH")
    print("="*100)
    print("attempts:", attempts_path)
    print("checkpoint:", checkpoint)
    t, y, e = base.load_data(checkpoint)
    usecols = ["nv_index","status","site_id","orientation","kappa","distance_A",
               "f0_kHz","f1_kHz","red_chi2","aicc","score_primary",
               "score_amp_tie","popt_json"]
    attempts = pd.read_csv(attempts_path, usecols=lambda c: c in usecols)
    catalog = load_catalog(base)
    incumbents, incfiles, protected = load_incumbents(prefix)
    embedded_map = load_assigned_orientations(args.field, y.shape[0])
    external_map = load_orientation_map(args.orientation_map)
    explicit_orientation_map = dict(embedded_map)
    explicit_orientation_map.update(external_map)
    print(f"incumbent candidates loaded: {len(incumbents)} from {len(incfiles)} full V2/V4 files")
    print(f"orientation mode={args.orientation_mode} | embedded assignments={len(embedded_map)} "
          f"| external assignments={len(external_map)}")
    req = parse_nv_list(args.nv)
    nvs = list(range(y.shape[0])) if req is None else [i for i in req if 0 <= i < y.shape[0]]
    band_lo_mhz, band_hi_mhz = experimental_frequency_band_mhz(t)
    print(f"NVs={len(nvs)} max_spins={args.max_spins} workers={args.workers}")
    print(f"experimental C13 band={1000*band_lo_mhz:.2f}-{1000*band_hi_mhz:.1f} kHz "
          f"(0.5/Tspan to 0.49/dt_min)")
    print(f"physical SNR >= {args.physical_snr_min:g} | family tol={args.family_tol_khz:g} kHz "
          f"| max sites/orientation={args.max_sites} | max/family={args.max_per_family}")
    def task(i):
        return fit_one_nv(base, i, t, y[i], e[i], attempts, catalog,
                          incumbents, protected, args.max_spins,
                          args.physical_snr_min, args.family_tol_khz,
                          args.max_sites, args.max_per_family,
                          args.orientation_mode, explicit_orientation_map)
    with threadpool_limits(limits=base.BLAS_THREADS_PER_WORKER):
        results = Parallel(n_jobs=max(1,args.workers), backend="loky",
                           batch_size=1, verbose=5)(delayed(task)(i) for i in nvs)
    rows = [r for block, _, _ in results for r in block]
    site_rows = [r for _, block, _ in results for r in block]
    fam_rows = [r for _, _, block in results for r in block]
    cdf = pd.DataFrame(rows).sort_values(["nv_index","rank_global_bic"])
    cdf["frequency_band_low_kHz"] = 1000.0 * band_lo_mhz
    cdf["frequency_band_high_kHz"] = 1000.0 * band_hi_mhz
    cdf = base.enrich_candidates_with_catalog(cdf, base.load_catalog_lookup())
    outdir = Path(base.OUTPUT_DIR) if base.OUTPUT_DIR else prefix.parent
    outdir.mkdir(parents=True, exist_ok=True)
    subset = "" if req is None else "_subset_" + "-".join(map(str,nvs))
    oritag = args.orientation_mode.replace("-", "")
    tag = (f"_snr{safe_tag(args.physical_snr_min)}_tol{safe_tag(args.family_tol_khz)}kHz"
           f"_topS{args.max_sites}_perF{args.max_per_family}_ori{oritag}"
           f"_maxC{args.max_spins}{subset}")
    outbase = outdir / (prefix.name + "_" + VERSION + tag)
    pd.DataFrame(site_rows).to_csv(Path(str(outbase)+"_physical_candidate_sites.csv"), index=False)
    pd.DataFrame(fam_rows).to_csv(Path(str(outbase)+"_frequency_families.csv"), index=False)
    cand = Path(str(outbase)+"_candidate_fits.csv"); cdf.to_csv(cand, index=False)
    best_aicc = cdf[cdf.rank_global_aicc == 1].sort_values("nv_index")
    best_red = cdf[cdf.rank_global_redchi == 1].sort_values("nv_index")
    evidence_rows, conservative_rows = [], []
    for nv, g in cdf.groupby("nv_index"):
        selected, raw, conservative, path, ev = base.select_bic_with_evidence(g)
        row = selected.copy()
        for k, v in ev.items(): row[k] = v
        row["selection_rule"] = "global_minimum_BIC"
        row["selection_path_json"] = json.dumps(path)
        row["candidate_pipeline"] = VERSION
        row["physical_snr_min"] = args.physical_snr_min
        row["family_tol_khz"] = args.family_tol_khz
        row["max_sites_per_orientation"] = args.max_sites
        row["max_sites_per_family"] = args.max_per_family
        row["orientation_mode_requested"] = args.orientation_mode
        evidence_rows.append(row)
        crow = conservative.copy()
        crow["selection_delta_bic_threshold"] = base.ORDER_ACCEPT_DELTA_BIC
        crow["selection_path_json"] = json.dumps(path)
        conservative_rows.append(crow)
    best_bic = pd.DataFrame(evidence_rows).sort_values("nv_index")
    best_cons = pd.DataFrame(conservative_rows).sort_values("nv_index")
    best_bic.to_csv(Path(str(outbase)+"_best_by_bic.csv"), index=False)
    best_bic.to_csv(Path(str(outbase)+"_best_by_bic_evidence.csv"), index=False)
    best_aicc.to_csv(Path(str(outbase)+"_best_by_aicc.csv"), index=False)
    best_red.to_csv(Path(str(outbase)+"_best_by_redchi.csv"), index=False)
    best_cons.to_csv(Path(str(outbase)+"_best_by_bic_delta10.csv"), index=False)
    if not args.no_dashboard:
        with PdfPages(Path(str(outbase)+"_dashboard.pdf")) as pdf:
            base.plot_global_summary(pdf, cdf, best_cons, args.max_spins)
            for nv in nvs:
                base.plot_nv(pdf, nv, t, y, e, cdf)
    print("="*100); print("COMPLETE")
    print("best-BIC orders:", best_bic.model_order.value_counts().sort_index().to_dict())
    print(f"median best-BIC chi2r={best_bic.red_chi2.median():.3f}")
    print("candidates:", cand)
    print("families:", Path(str(outbase)+"_frequency_families.csv"))
    if not args.no_dashboard:
        print("dashboard:", Path(str(outbase)+"_dashboard.pdf"))
    print("="*100)
    if not args.no_show:
        plt.show(block=True)
    return dict(candidates=cdf, best_by_bic=best_bic,
                best_by_aicc=best_aicc, best_by_redchi=best_red)

if __name__ == "__main__":
    main()
