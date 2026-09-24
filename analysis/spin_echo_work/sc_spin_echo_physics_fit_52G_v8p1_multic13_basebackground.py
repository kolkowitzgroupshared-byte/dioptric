# -*- coding: utf-8 -*-
"""
V8.1: orientation-locked single-/multi-13C fit using the proven current background.

Keeps the empirically useful stretched-envelope + quartic-revival background
from the current fitter, but replaces free additive oscillations with a
catalog-constrained product of single-13C Hahn-echo coherence factors.

For independent 13C spins coupled to the NV electron:
    L_C13(t) = prod_j [1 - eta_j q_j(t)]
    q_j(t) = 2*kappa_j*sin^2(pi*fI_j*(t+dt)/2)*sin^2(pi*fm_j*(t+dt)/2)
where t is total Hahn-echo evolution time (2*tau).

Signal:
    S(t) = baseline - contrast * E(t) * R(t) * L_C13(t)
    E(t) = exp[-(t/T2)^beta]

Important:
- NV orientation is fixed externally before the 13C search.
- Only sites from that orientation are considered.
- Catalog kappa/fI/fm set the ideal site response.
- eta_j is constrained to [0,1]: measured modulation may be weaker than ideal,
  but cannot exceed the ideal catalog prediction.
- One small common timing offset dt is allowed; no arbitrary phases/frequencies.
- The proven background taper/width/chirp terms are retained but weakly regularized; the free oscillator is removed.
- V6 is used only to seed nuisance/background parameters.
- Final comparisons use ordinary weighted least squares, so chi2/AICc/BIC and
  T2 profile likelihood are statistically consistent.
- Single, pair, and optional triple 13C models are searched with forward,
  residual-directed candidate generation rather than brute-force N^3 search.

Quick test:
  python analysis/spin_echo_work/sc_spin_echo_physics_fit_52G_v8p1_multic13.py \
      --nv 0,16,28,54,93 --max-spins 2 --quick --no-show
"""

from __future__ import annotations

import argparse
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
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

SEARCH_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master")
RESULT_TAG = "spin_echo_old_protocol_ranked_52G"
ALL_ATTEMPTS_PATH = None
CHECKPOINT_PATH = None
OUTPUT_DIR = None
CATALOG_PATH = Path(r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json")
V6_ALL_SITE_FITS = None
ORIENTATION_ASSIGNMENTS_CSV = None

ALLOWED_ORIENTATIONS = ((1, 1, -1), (-1, 1, 1))
MIN_DISTANCE_A = 1.4
MAX_DISTANCE_A = 22.0
MIN_KAPPA = 1e-5
MIN_ESEEM_LINE_KHZ = 4.0

# Background is the successful current phenomenological core, but WITHOUT the
# arbitrary additive oscillator.  Parameters:
# baseline, contrast, revival_time, width0, T2, beta, amplitude taper, width growth, chirp.
BG_NAMES = (
    "baseline", "contrast", "revival_time_us", "width0_us",
    "T2_us", "beta", "amp_taper_alpha", "width_slope", "revival_chirp",
)
BG_LB = np.array([0.00, 0.00, 30.0, 1.0, 5.0, 0.60, 0.00, 0.00, -0.03], float)
BG_UB = np.array([1.10, 0.95, 40.0, 15.0, 300.0, 3.00, 3.00, 0.80,  0.03], float)

# Weak physical regularization used only during optimization.  Final chi2/AIC/BIC
# are always computed from the experimental residuals only.
REVIVAL_PRIOR_US = 35.6631
REVIVAL_PRIOR_SIGMA_US = 1.2
CHIRP_PRIOR_SIGMA = 0.015
ALPHA_PRIOR_CENTER = 0.5
ALPHA_PRIOR_SIGMA = 1.2

ETA_BOUNDS = (0.0, 1.0)
DT_BOUNDS_US = (-0.25, 0.25)

BG_T2_STARTS_US = (20.0, 40.0, 80.0, 140.0)
BG_BETA_STARTS = (1.0, 1.5, 2.0)
BG_ROBUST_MAX_NFEV = 15000
BG_FINAL_MAX_NFEV = 25000
FULL_MAX_NFEV = 30000
FULL_ETA_STARTS = (0.20, 0.55, 0.90)
FULL_DT_STARTS_US = (0.0, -0.08, 0.08)

SINGLE_SCREEN_KEEP = 80
SINGLE_FULL_KEEP = 36
PAIR_SEED_SINGLES = 6
PAIR_ADD_KEEP_PER_SEED = 18
PAIR_FULL_MAX = 90
TRIPLE_SEED_PAIRS = 4
TRIPLE_ADD_KEEP_PER_SEED = 12
TRIPLE_FULL_MAX = 48
ORDER_ACCEPT_DELTA_BIC = 10.0

RUN_T2_PROFILE = True
PROFILE_T2_GRID_US = np.geomspace(BG_LB[4], BG_UB[4], 26)
PROFILE_DELTA_CHI2_68 = 1.0
PROFILE_DELTA_CHI2_95 = 3.84
PROFILE_MAX_NFEV = 12000

CPU_COUNT = os.cpu_count() or 4
DEFAULT_N_JOBS = max(1, min(18, CPU_COUNT - 2))
BLAS_THREADS_PER_WORKER = 1
DENSE_POINTS = 3000
ZOOM_HALF_WIDTH_US = 12.5
RANDOM_SEED = 20260923


@dataclass
class InputPaths:
    attempts: Path
    checkpoint: Path
    prefix: Path
    orientation_csv: Path
    v6_site_fits: Path


def newest_match(root, pattern):
    matches = list(Path(root).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} under {root}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def discover_paths():
    attempts = Path(ALL_ATTEMPTS_PATH) if ALL_ATTEMPTS_PATH else newest_match(
        SEARCH_ROOT, f"*{RESULT_TAG}_all_attempts.csv.gz"
    )
    suffix = "_all_attempts.csv.gz"
    s = str(attempts)
    if not s.endswith(suffix):
        raise ValueError(f"Unexpected attempts filename: {attempts}")
    prefix = Path(s[:-len(suffix)])
    checkpoint = Path(CHECKPOINT_PATH) if CHECKPOINT_PATH else Path(str(prefix) + "_fit_checkpoint.npz")
    orientation_csv = Path(ORIENTATION_ASSIGNMENTS_CSV) if ORIENTATION_ASSIGNMENTS_CSV else newest_match(
        SEARCH_ROOT, "*orientation_locked_confidence_v6_orientation_assignments.csv"
    )
    v6_site_fits = Path(V6_ALL_SITE_FITS) if V6_ALL_SITE_FITS else newest_match(
        SEARCH_ROOT, "*orientation_locked_confidence_v6_all_equal_footing_sites.csv"
    )
    for p in (checkpoint, orientation_csv, v6_site_fits, CATALOG_PATH):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    return InputPaths(attempts, checkpoint, prefix, orientation_csv, v6_site_fits)


def canonical_orientation(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    a = np.asarray(value, int).ravel()
    if a.size != 3:
        raise ValueError(f"Bad orientation {value}")
    return tuple(int(v) for v in a)


def load_inputs(paths):
    ck = np.load(paths.checkpoint, allow_pickle=True)
    t = np.asarray(ck["times_us"], float)
    y = np.asarray(ck["norm_counts"], float)
    e = np.maximum(np.abs(np.asarray(ck["norm_counts_ste"], float)), 1e-4)
    if y.ndim != 2 or y.shape != e.shape or y.shape[1] != len(t):
        raise ValueError(f"Bad checkpoint shapes t={t.shape}, y={y.shape}, e={e.shape}")

    odf = pd.read_csv(paths.orientation_csv)
    odf["orientation_tuple"] = odf["orientation"].map(canonical_orientation)
    ori_map = {int(r.nv_index): tuple(r.orientation_tuple) for r in odf.itertuples()}
    ori_quality = {
        int(r.nv_index): {
            "rms_mhz": float(getattr(r, "orientation_rms_error_MHz", np.nan)),
            "warning": bool(getattr(r, "orientation_warning", False)),
        }
        for r in odf.itertuples()
    }

    v6 = pd.read_csv(paths.v6_site_fits)
    v6_best = v6[v6["site_rank"] == 1].copy()
    v6_seed_map = {int(r.nv_index): r for r in v6_best.itertuples()}

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    catalog = []
    allowed = set(ALLOWED_ORIENTATIONS)
    for rec in raw:
        ori = canonical_orientation(rec["orientation"])
        if ori not in allowed:
            continue
        d = float(rec.get("distance_A", np.nan))
        k = float(rec.get("kappa", np.nan))
        if not np.isfinite(d) or not (MIN_DISTANCE_A <= d <= MAX_DISTANCE_A):
            continue
        if not np.isfinite(k) or k < MIN_KAPPA:
            continue
        r = dict(rec)
        r["orientation_tuple"] = ori
        r["site_id"] = int(rec["site_index"])
        r["fI_kHz"] = float(rec["fI_Hz"]) / 1e3
        r["fm_kHz"] = float(rec["omega_ms_Hz"]) / 1e3
        r["fminus_kHz"] = float(rec["f_minus_Hz"]) / 1e3
        r["fplus_kHz"] = float(rec["f_plus_Hz"]) / 1e3
        catalog.append(r)
    return t, y, e, odf, ori_map, ori_quality, v6_seed_map, catalog


# ------------------------------- MODEL ---------------------------------------

def quartic_revival_comb(t_us, revival_time_us, width0_us, amp_taper_alpha, width_slope, revival_chirp):
    """Same quartic revival comb structure as the current fitter."""
    t = np.asarray(t_us, float)
    trev = max(float(revival_time_us), 1e-9)
    nrev = max(1, min(16, int(np.ceil(float(t.max()) / trev)) + 2))
    out = np.zeros_like(t)
    for k in range(nrev):
        mu = k * trev * (1.0 + k * float(revival_chirp))
        w = max(0.25, float(width0_us) * (1.0 + k * float(width_slope)))
        amp = 1.0 / ((1.0 + k) ** float(amp_taper_alpha))
        out += amp * np.exp(-((t - mu) / w) ** 4)
    return out


def background_carrier(t_us, bg):
    baseline, contrast, trev, width0, T2, beta, alpha, width_slope, chirp = np.asarray(bg, float)
    t = np.asarray(t_us, float)
    env = np.exp(-np.power(np.maximum(t, 0.0) / max(float(T2), 1e-9), float(beta)))
    comb = quartic_revival_comb(t, trev, width0, alpha, width_slope, chirp)
    return float(baseline), float(contrast), env * comb


def background_prior_residual(bg):
    """Weak MAP regularization; does not enter reported data chi2."""
    bg = np.asarray(bg, float)
    return np.array([
        (bg[2] - REVIVAL_PRIOR_US) / REVIVAL_PRIOR_SIGMA_US,
        (bg[8] - 0.0) / CHIRP_PRIOR_SIGMA,
        (bg[6] - ALPHA_PRIOR_CENTER) / ALPHA_PRIOR_SIGMA,
    ], float)


def site_q(t_us, site, dt_us=0.0):
    # t is total evolution 2*tau, so omega*tau/2 = pi*f*t/2.
    t = np.asarray(t_us, float) + float(dt_us)
    fI = float(site["fI_kHz"]) / 1000.0
    fm = float(site["fm_kHz"]) / 1000.0
    return 2.0 * float(site["kappa"]) * (
        np.sin(0.5 * np.pi * fI * t) ** 2
    ) * (
        np.sin(0.5 * np.pi * fm * t) ** 2
    )


def c13_coherence(t_us, sites, etas, dt_us):
    L = np.ones_like(np.asarray(t_us, float))
    for site, eta in zip(sites, etas):
        L *= 1.0 - float(eta) * site_q(t_us, site, dt_us)
    return L


def model_from_parts(t_us, bg, sites=(), etas=(), dt_us=0.0):
    baseline, contrast, carrier = background_carrier(t_us, bg)
    return baseline - contrast * carrier * c13_coherence(t_us, sites, etas, dt_us)


def fit_stats(y, e, pred, npar):
    e = np.maximum(np.asarray(e, float), 1e-12)
    chi2 = float(np.sum(((np.asarray(y, float) - np.asarray(pred, float)) / e) ** 2))
    n = len(y); k = int(npar); dof = max(1, n-k)
    red = chi2 / dof
    aic = chi2 + 2*k
    aicc = aic + 2*k*(k+1)/(n-k-1) if n > k+1 else np.inf
    bic = chi2 + k*np.log(max(n, 2))
    return dict(chi2=chi2, red_chi2=float(red), aicc=float(aicc), bic=float(bic), npar=k)


def site_key(sites):
    return tuple(sorted(int(s["site_id"]) for s in sites))


# -------------------------- BACKGROUND FIT -----------------------------------

def data_seed(yv):
    yv = np.asarray(yv, float)
    b = float(np.clip(np.nanpercentile(yv, 90), BG_LB[0]+.01, BG_UB[0]-.01))
    c = float(np.clip(b - np.nanpercentile(yv, 5), .03, .70))
    return np.array([b, c, 35.66, 5.5, 45., 1.5, 0.5, .15, 0.0], float)


def v6_seed(row, yv):
    p = data_seed(yv)
    if row is None:
        return p
    for name, idx in (
        ("baseline",0), ("comb_contrast",1), ("revival_time_us",2),
        ("width0_us",3), ("T2_us",4), ("T2_exp",5), ("amp_taper_alpha",6), ("width_slope",7), ("revival_chirp",8),
    ):
        v = getattr(row, name, np.nan)
        if np.isfinite(v):
            p[idx] = float(v)
    if p[4] > 180:
        p[4] = 80.0
    return np.clip(p, BG_LB+1e-6, BG_UB-1e-6)


def fit_background(t, y, e, seed):
    e = np.maximum(np.asarray(e, float), 1e-12)
    def resid_data(p):
        return (np.asarray(y) - model_from_parts(t, p)) / e
    def resid(p):
        return np.concatenate([resid_data(p), background_prior_residual(p)])
    starts = []
    for t2 in BG_T2_STARTS_US:
        for beta in BG_BETA_STARTS:
            p = np.asarray(seed, float).copy(); p[4]=t2; p[5]=beta; starts.append(p)
    starts.append(np.asarray(seed, float))
    fits = []
    for p0 in starts:
        p0 = np.clip(p0, BG_LB+1e-8, BG_UB-1e-8)
        try:
            rr = least_squares(resid, p0, bounds=(BG_LB,BG_UB), loss="soft_l1",
                               f_scale=1., max_nfev=BG_ROBUST_MAX_NFEV, x_scale="jac")
            res = least_squares(resid, rr.x, bounds=(BG_LB,BG_UB), loss="linear",
                                max_nfev=BG_FINAL_MAX_NFEV, ftol=1e-9, xtol=1e-9,
                                gtol=1e-9, x_scale="jac")
            pred = model_from_parts(t, res.x)
            fits.append(dict(bg=res.x, pred=pred, success=res.success,
                             **fit_stats(y,e,pred,len(BG_NAMES))))
        except Exception:
            pass
    if not fits:
        raise RuntimeError("all background starts failed")
    return min(fits, key=lambda r:(r["chi2"],r["bic"]))


def sampling_nyquist_khz(t):
    u = np.unique(np.asarray(t,float)); d=np.diff(u); d=d[d>0]
    return np.inf if not len(d) else 500.0/float(d.min())


def catalog_for_nv(catalog, orientation, t):
    nyq = sampling_nyquist_khz(t); orientation=tuple(orientation); out=[]
    for r in catalog:
        if tuple(r["orientation_tuple"]) != orientation:
            continue
        if float(r["fplus_kHz"]) > .98*nyq:
            continue
        if max(float(r["fminus_kHz"]), float(r["fplus_kHz"])) < MIN_ESEEM_LINE_KHZ:
            continue
        out.append(r)
    return out


# ---------------------------- FAST SCREEN ------------------------------------

def analytic_eta_screen(t,y,e,bg,existing_sites,existing_etas,candidate_site,dt_us=0.):
    baseline, contrast, carrier = background_carrier(t,bg)
    L0 = c13_coherence(t,existing_sites,existing_etas,dt_us)
    pred0 = baseline - contrast*carrier*L0
    basis = contrast*carrier*L0*site_q(t,candidate_site,dt_us)
    w = 1.0/np.maximum(np.asarray(e,float),1e-12)**2
    den=float(np.sum(w*basis*basis)); num=float(np.sum(w*basis*(np.asarray(y)-pred0)))
    eta = 0.0 if den<=1e-18 else float(np.clip(num/den,*ETA_BOUNDS))
    pred = pred0 + eta*basis
    nsite=len(existing_sites)+1
    return dict(site=candidate_site, eta_screen=eta, pred=pred,
                **fit_stats(y,e,pred,len(BG_NAMES)+nsite+1))


def screen_single_sites(t,y,e,bg,sites):
    rows=[analytic_eta_screen(t,y,e,bg,(),(),s,0.) for s in sites]
    rows.sort(key=lambda r:(r["chi2"],r["bic"]))
    return rows


# -------------------------- DYNAMIC SITE FIT ---------------------------------

def unpack_theta(theta,nsite):
    theta=np.asarray(theta,float); bg=theta[:len(BG_NAMES)]
    if nsite==0:
        return bg,np.array([],float),0.
    i=len(BG_NAMES); return bg,theta[i:i+nsite],float(theta[i+nsite])


def theta_bounds(nsite):
    if nsite==0:
        return BG_LB.copy(),BG_UB.copy()
    return (
        np.concatenate([BG_LB,np.full(nsite,ETA_BOUNDS[0]),[DT_BOUNDS_US[0]]]),
        np.concatenate([BG_UB,np.full(nsite,ETA_BOUNDS[1]),[DT_BOUNDS_US[1]]]),
    )


def make_theta_seed(bg,nsite,eta=.5,dt=0.):
    if nsite==0:
        return np.asarray(bg,float).copy()
    return np.concatenate([np.asarray(bg,float),np.full(nsite,float(eta)),[float(dt)]])


def expand_previous_theta(prev_theta,new_nsite,new_eta=.35):
    bg,etas,dt=unpack_theta(prev_theta,new_nsite-1)
    return np.concatenate([bg,etas,[float(new_eta)],[float(dt)]])


def fit_site_set(t,y,e,sites,initial_thetas):
    nsite=len(sites); lb,ub=theta_bounds(nsite); e=np.maximum(np.asarray(e,float),1e-12)
    def pred(th):
        bg,etas,dt=unpack_theta(th,nsite); return model_from_parts(t,bg,sites,etas,dt)
    def resid_data(th): return (np.asarray(y)-pred(th))/e
    def resid(th):
        bg,_,_=unpack_theta(th,nsite)
        return np.concatenate([resid_data(th), background_prior_residual(bg)])
    fits=[]
    for seed in initial_thetas:
        seed=np.clip(np.asarray(seed,float),lb+1e-8,ub-1e-8)
        try:
            rr=least_squares(resid,seed,bounds=(lb,ub),loss="soft_l1",f_scale=1.,
                             max_nfev=max(5000,FULL_MAX_NFEV//3),x_scale="jac")
            res=least_squares(resid,rr.x,bounds=(lb,ub),loss="linear",max_nfev=FULL_MAX_NFEV,
                              ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale="jac")
            p=pred(res.x); fits.append(dict(theta=res.x,pred=p,success=res.success,nfev=res.nfev,
                                            **fit_stats(y,e,p,len(res.x))))
        except Exception:
            pass
    if not fits:
        return None
    b=min(fits,key=lambda r:(r["chi2"],r["bic"])); b["sites"]=tuple(sites); b["site_key"]=site_key(sites)
    return b


def single_initial_thetas(bg,eta_screen):
    etas=sorted(set([float(np.clip(eta_screen,.02,.98)),*FULL_ETA_STARTS]))
    return [make_theta_seed(bg,1,eta,dt) for eta in etas for dt in FULL_DT_STARTS_US]


def refit_singles(t,y,e,bg,screen_rows,keep):
    out=[]
    for r in screen_rows[:keep]:
        f=fit_site_set(t,y,e,[r["site"]],single_initial_thetas(bg,r["eta_screen"]))
        if f is not None: out.append(f)
    out.sort(key=lambda r:(r["bic"],r["chi2"])); return out


def incremental_screen_from_fit(t,y,e,parent_fit,candidate_sites):
    ps=list(parent_fit["sites"]); bg,etas,dt=unpack_theta(parent_fit["theta"],len(ps)); used=set(site_key(ps)); rows=[]
    for s in candidate_sites:
        if int(s["site_id"]) in used: continue
        rows.append(analytic_eta_screen(t,y,e,bg,ps,etas,s,dt))
    rows.sort(key=lambda r:(r["chi2"],r["bic"])); return rows


def fit_child_candidate(t,y,e,parent_fit,new_site,eta_screen):
    sites=list(parent_fit["sites"])+[new_site]
    s1=expand_previous_theta(parent_fit["theta"],len(sites),np.clip(eta_screen,.05,.95))
    s2=expand_previous_theta(parent_fit["theta"],len(sites),.50)
    return fit_site_set(t,y,e,sites,[s1,s2])


def search_next_order(t,y,e,parent_fits,all_sites,parent_keep,add_keep,full_max):
    specs={}
    for parent in parent_fits[:parent_keep]:
        for r in incremental_screen_from_fit(t,y,e,parent,all_sites)[:add_keep]:
            s=r["site"]; key=tuple(sorted(list(parent["site_key"])+[int(s["site_id"])]))
            spec=dict(parent=parent,new_site=s,eta_screen=float(r["eta_screen"]),screen_chi2=float(r["chi2"]))
            if key not in specs or spec["screen_chi2"]<specs[key]["screen_chi2"]:
                specs[key]=spec
    ordered=sorted(specs.values(),key=lambda x:x["screen_chi2"])[:full_max]
    fits=[]
    for spec in ordered:
        f=fit_child_candidate(t,y,e,spec["parent"],spec["new_site"],spec["eta_screen"])
        if f is not None: fits.append(f)
    fits.sort(key=lambda r:(r["bic"],r["chi2"])); return fits


# ------------------------- MODEL ORDER / T2 PROFILE --------------------------

def make_order0_record(bg_fit):
    return dict(theta=np.asarray(bg_fit["bg"],float),pred=np.asarray(bg_fit["pred"],float),sites=tuple(),
                site_key=tuple(),chi2=float(bg_fit["chi2"]),red_chi2=float(bg_fit["red_chi2"]),
                aicc=float(bg_fit["aicc"]),bic=float(bg_fit["bic"]),npar=int(bg_fit["npar"]),success=True)


def choose_model_order(best_by_order):
    order=0; selected=best_by_order[0]; decisions=[]
    for candidate_order in range(1,max(best_by_order)+1):
        cand=best_by_order.get(candidate_order)
        if cand is None: continue
        improvement=float(selected["bic"]-cand["bic"]); accepted=improvement>=ORDER_ACCEPT_DELTA_BIC
        decisions.append(dict(from_order=order,candidate_order=candidate_order,
                              delta_bic_improvement=improvement,accepted=accepted))
        if accepted: order=candidate_order; selected=cand
    return order,selected,decisions


def akaike_weights(fits):
    if not fits: return np.array([])
    a=np.array([f["aicc"] for f in fits],float); d=a-np.nanmin(a); w=np.exp(-.5*np.clip(d,0,140)); s=w.sum()
    return w/s if s>0 else np.full(len(fits),1/len(fits))


def profile_selected_t2(t,y,e,selected_fit):
    if not RUN_T2_PROFILE: return [],{}
    sites=list(selected_fit["sites"]); nsite=len(sites); theta0=np.asarray(selected_fit["theta"],float)
    lb,ub=theta_bounds(nsite); free=[i for i in range(len(theta0)) if i!=4]; rows=[]
    grid=np.unique(np.concatenate([PROFILE_T2_GRID_US,[theta0[4]]]))
    for t2 in grid:
        q0=theta0[free].copy(); qlb=lb[free]; qub=ub[free]
        def assemble(q):
            th=theta0.copy(); th[free]=q; th[4]=float(t2); return th
        def resid(q):
            th=assemble(q); bg,etas,dt=unpack_theta(th,nsite)
            data_r=(np.asarray(y)-model_from_parts(t,bg,sites,etas,dt))/np.maximum(np.asarray(e),1e-12)
            return np.concatenate([data_r, background_prior_residual(bg)])
        try:
            res=least_squares(resid,np.clip(q0,qlb+1e-8,qub-1e-8),bounds=(qlb,qub),loss="linear",
                              max_nfev=PROFILE_MAX_NFEV,x_scale="jac")
            th=assemble(res.x); bg,etas,dt=unpack_theta(th,nsite); pred=model_from_parts(t,bg,sites,etas,dt)
            st=fit_stats(y,e,pred,len(th)-1); rows.append(dict(T2_us=float(t2),chi2=st["chi2"],red_chi2=st["red_chi2"]))
        except Exception: pass
    if not rows: return [],dict(T2_profile_status="profile_failed")
    mn=min(r["chi2"] for r in rows)
    for r in rows: r["delta_chi2"]=float(r["chi2"]-mn)
    def interval(th):
        vals=[r["T2_us"] for r in rows if r["delta_chi2"]<=th]
        if not vals:return np.nan,np.nan,False,False
        lo=float(min(vals)); hi=float(max(vals)); return lo,hi,lo<=1.02*BG_LB[4],hi>=.98*BG_UB[4]
    lo68,hi68,_,hi_touch68=interval(PROFILE_DELTA_CHI2_68); lo95,hi95,lo_touch95,hi_touch95=interval(PROFILE_DELTA_CHI2_95)
    best=min(rows,key=lambda r:r["chi2"])["T2_us"]
    if hi_touch95: status="lower_bound_only"; report=f"> {lo95:.1f} us (95% profile)"
    elif lo_touch95: status="upper_bound_only"; report=f"< {hi95:.1f} us (95% profile)"
    else: status="bounded"; report=f"{best:.1f} us [{lo95:.1f}, {hi95:.1f}] 95% profile"
    return rows,dict(T2_profile_status=status,T2_profile_report=report,T2_profile_best_us=float(best),
                     T2_68_low_us=lo68,T2_68_high_us=hi68,T2_95_low_us=lo95,T2_95_high_us=hi95,
                     T2_profile_touches_upper_68=bool(hi_touch68),T2_profile_touches_upper_95=bool(hi_touch95))


# ------------------------- SERIALIZATION / ONE NV ----------------------------

def site_metadata(site,prefix):
    return {
        f"{prefix}_site_id":int(site["site_id"]), f"{prefix}_distance_A":float(site["distance_A"]),
        f"{prefix}_kappa":float(site["kappa"]), f"{prefix}_fI_kHz":float(site["fI_kHz"]),
        f"{prefix}_fm_kHz":float(site["fm_kHz"]), f"{prefix}_fminus_kHz":float(site["fminus_kHz"]),
        f"{prefix}_fplus_kHz":float(site["fplus_kHz"]),
        f"{prefix}_A_par_kHz":float(site.get("A_par_Hz",np.nan))/1e3,
        f"{prefix}_A_perp_kHz":float(site.get("A_perp_Hz",np.nan))/1e3,
        f"{prefix}_theta_deg":float(site.get("theta_deg",np.nan)),
        f"{prefix}_x_A":float(site.get("x_A",np.nan)), f"{prefix}_y_A":float(site.get("y_A",np.nan)),
        f"{prefix}_z_A":float(site.get("z_A",np.nan)),
    }


def fit_to_row(nv,orientation,order,rank,fit,weight=np.nan):
    bg,etas,dt=unpack_theta(fit["theta"],order)
    row=dict(nv_index=int(nv),orientation=str(tuple(orientation)),model_order=int(order),rank_within_order=int(rank),
             site_key=str(tuple(fit["site_key"])),chi2=float(fit["chi2"]),red_chi2=float(fit["red_chi2"]),
             aicc=float(fit["aicc"]),bic=float(fit["bic"]),akaike_weight_within_order=float(weight),
             baseline=float(bg[0]),contrast=float(bg[1]),revival_time_us=float(bg[2]),width0_us=float(bg[3]),
             T2_us=float(bg[4]),beta=float(bg[5]),amp_taper_alpha=float(bg[6]),width_slope=float(bg[7]),
             revival_chirp=float(bg[8]),dt_us=float(dt),
             theta_json=json.dumps(np.asarray(fit["theta"],float).tolist()))
    for j,s in enumerate(fit["sites"],1):
        row[f"eta{j}"]=float(etas[j-1]); row.update(site_metadata(s,f"c13_{j}"))
    return row


def fit_one_nv(nv,t,y,e,orientation,orientation_quality,v6_seed_row,catalog,max_spins,quick):
    yv=np.asarray(y[nv],float); ev=np.asarray(e[nv],float); sites=catalog_for_nv(catalog,orientation,t)
    if not sites: raise RuntimeError(f"NV {nv}: no sites for {orientation}")
    bg=fit_background(t,yv,ev,v6_seed(v6_seed_row,yv)); order0=make_order0_record(bg)
    screen=screen_single_sites(t,yv,ev,bg["bg"],sites)
    singles=refit_singles(t,yv,ev,bg["bg"],screen[:SINGLE_SCREEN_KEEP],min(16,SINGLE_FULL_KEEP) if quick else SINGLE_FULL_KEEP)
    best={0:order0}; allfits={0:[order0]}
    if singles: best[1]=singles[0]; allfits[1]=singles
    pairs=[]
    if max_spins>=2 and singles:
        pairs=search_next_order(t,yv,ev,singles,sites,min(3,PAIR_SEED_SINGLES) if quick else PAIR_SEED_SINGLES,
                                min(8,PAIR_ADD_KEEP_PER_SEED) if quick else PAIR_ADD_KEEP_PER_SEED,
                                min(20,PAIR_FULL_MAX) if quick else PAIR_FULL_MAX)
        if pairs: best[2]=pairs[0]; allfits[2]=pairs
    triples=[]
    if max_spins>=3 and pairs:
        triples=search_next_order(t,yv,ev,pairs,sites,min(2,TRIPLE_SEED_PAIRS) if quick else TRIPLE_SEED_PAIRS,
                                  min(6,TRIPLE_ADD_KEEP_PER_SEED) if quick else TRIPLE_ADD_KEEP_PER_SEED,
                                  min(12,TRIPLE_FULL_MAX) if quick else TRIPLE_FULL_MAX)
        if triples: best[3]=triples[0]; allfits[3]=triples
    selected_order,selected,decisions=choose_model_order(best)
    profile_rows,profile_summary=profile_selected_t2(t,yv,ev,selected)
    for r in profile_rows: r.update(nv_index=int(nv),model_order=int(selected_order))
    candidate_rows=[]
    for order,fits in allfits.items():
        for rank,(f,w) in enumerate(zip(fits,akaike_weights(fits)),1):
            candidate_rows.append(fit_to_row(nv,orientation,order,rank,f,w))
    bgsel,etasel,dtsel=unpack_theta(selected["theta"],selected_order)
    summary=dict(nv_index=int(nv),orientation=str(tuple(orientation)),
                 orientation_rms_error_MHz=float(orientation_quality.get("rms_mhz",np.nan)),
                 orientation_warning=bool(orientation_quality.get("warning",False)),
                 num_catalog_sites_considered=len(sites),selected_order=int(selected_order),
                 selected_site_key=str(tuple(selected["site_key"])),selected_red_chi2=float(selected["red_chi2"]),
                 selected_aicc=float(selected["aicc"]),selected_bic=float(selected["bic"]),
                 background_red_chi2=float(order0["red_chi2"]),background_bic=float(order0["bic"]),
                 T2_us=float(bgsel[4]),beta=float(bgsel[5]),revival_time_us=float(bgsel[2]),
                 width0_us=float(bgsel[3]),amp_taper_alpha=float(bgsel[6]),width_slope=float(bgsel[7]),
                 revival_chirp=float(bgsel[8]),baseline=float(bgsel[0]),contrast=float(bgsel[1]),
                 dt_us=float(dtsel),decision_json=json.dumps(decisions),**profile_summary)
    for j,s in enumerate(selected["sites"],1):
        summary[f"eta{j}"]=float(etasel[j-1]); summary.update(site_metadata(s,f"c13_{j}"))
    order_rows=[dict(nv_index=int(nv),model_order=int(o),site_key=str(tuple(f["site_key"])),chi2=float(f["chi2"]),
                     red_chi2=float(f["red_chi2"]),aicc=float(f["aicc"]),bic=float(f["bic"]),
                     delta_bic_vs_order0=float(f["bic"]-order0["bic"]),selected=bool(o==selected_order))
                for o,f in sorted(best.items())]
    site_txt=",".join(str(s["site_id"]) for s in selected["sites"]) or "none"
    print(f"[NV {nv:3d}] ori={orientation} N={selected_order} sites={site_txt} redchi={selected['red_chi2']:.3g} T2={bgsel[4]:.1f}us {profile_summary.get('T2_profile_status','')}")
    return dict(candidate_rows=candidate_rows,order_rows=order_rows,profile_rows=profile_rows,summary=summary)


# ------------------------------- PLOTS ---------------------------------------

def set_3d_equal(ax,xyz):
    xyz=np.asarray(xyz,float); xyz=xyz[np.all(np.isfinite(xyz),axis=1)]
    if not len(xyz): return
    xyz=np.vstack([xyz,np.zeros((1,3))]); lo=xyz.min(0); hi=xyz.max(0); c=.5*(lo+hi); r=max(1.,.55*float(np.max(hi-lo)))
    ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)


def plot_dashboard(pdf,nv,t,y,e,summary_df,candidate_df,order_df,profile_df,catalog):
    srow=summary_df[summary_df.nv_index==nv].iloc[0]; order=int(srow.selected_order)
    cdf=candidate_df[candidate_df.nv_index==nv]; odf=order_df[order_df.nv_index==nv].sort_values("model_order")
    pdfp=profile_df[profile_df.nv_index==nv].sort_values("T2_us") if not profile_df.empty else pd.DataFrame()
    ori=canonical_orientation(srow.orientation); recmap={(tuple(r["orientation_tuple"]),int(r["site_id"])):r for r in catalog}
    selected=cdf[(cdf.model_order==order)&(cdf.rank_within_order==1)].iloc[0]
    theta=np.asarray(json.loads(selected.theta_json),float); bg,etas,dt=unpack_theta(theta,order)
    sites=[recmap[(ori,int(selected[f"c13_{j}_site_id"]))] for j in range(1,order+1)]

    fig=plt.figure(figsize=(20.5,11.2)); gs=fig.add_gridspec(2,3,width_ratios=[1.15,1.05,1.0],hspace=.30,wspace=.27)
    ax1=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1]); ax3=fig.add_subplot(gs[0,2]); ax4=fig.add_subplot(gs[1,0],projection="3d"); ax5=fig.add_subplot(gs[1,1]); ax6=fig.add_subplot(gs[1,2])
    ax1.errorbar(t,y[nv],yerr=e[nv],fmt="o",ms=3,capsize=1,lw=.5,label="data",zorder=10); td=np.linspace(t.min(),t.max(),DENSE_POINTS); cmap=plt.get_cmap("tab10")
    for _,rr in odf.iterrows():
        o=int(rr.model_order); cr=cdf[(cdf.model_order==o)&(cdf.rank_within_order==1)].iloc[0]; th=np.asarray(json.loads(cr.theta_json),float); bgo,eo,dto=unpack_theta(th,o)
        so=[recmap[(ori,int(cr[f"c13_{j}_site_id"]))] for j in range(1,o+1)]; label="background" if o==0 else "+".join(str(s["site_id"]) for s in so)
        ax1.plot(td,model_from_parts(td,bgo,so,eo,dto),color=cmap(o%10),lw=2 if o==order else 1.1,alpha=1 if o==order else .65,label=f"N={o} {label} | chi2r={rr.red_chi2:.2f}"+(" SELECTED" if o==order else ""))
    ax1.set(title="Full trace: model-order comparison",xlabel="Total evolution time (us)",ylabel="Normalized signal"); ax1.grid(alpha=.2); ax1.legend(fontsize=7)
    lo=bg[2]-ZOOM_HALF_WIDTH_US; hi=bg[2]+ZOOM_HALF_WIDTH_US; m=(t>=lo)&(t<=hi); tz=np.linspace(lo,hi,DENSE_POINTS)
    ax2.errorbar(t[m],y[nv,m],yerr=e[nv,m],fmt="o",ms=3,capsize=1,lw=.5); ax2.plot(tz,model_from_parts(tz,bg,sites,etas,dt),lw=2,label="selected model"); ax2.axvline(bg[2],ls="--",lw=.8,alpha=.5,label="bath revival")
    ax2.set(title="First-revival zoom",xlabel="Total evolution time (us)",ylabel="Normalized signal"); ax2.grid(alpha=.2); ax2.legend(fontsize=7)
    x=np.arange(len(odf)); ax3.bar(x,odf.bic-odf.bic.min()); ax3.set_xticks(x); ax3.set_xticklabels([f"N={int(v)}" for v in odf.model_order]); ax3.set(title=f"Model order | selected N={order}",ylabel="Delta BIC from best tested order"); ax3.grid(alpha=.2,axis="y")
    ax4.scatter([0],[0],[0],marker="*",s=220,c="black",label="NV",depthshade=False); xyz=[]
    for j,s in enumerate(sites,1):
        pt=np.array([s.get("x_A",np.nan),s.get("y_A",np.nan),s.get("z_A",np.nan)],float)
        if np.all(np.isfinite(pt)):
            xyz.append(pt); ax4.plot([0,pt[0]],[0,pt[1]],[0,pt[2]],lw=1.1,alpha=.55); ax4.scatter([pt[0]],[pt[1]],[pt[2]],s=130,depthshade=False,label=f"C{j}: site {s['site_id']} eta={etas[j-1]:.2f}"); ax4.text(*pt,f" C{j}:S{s['site_id']}",fontsize=8)
    if xyz:set_3d_equal(ax4,np.vstack(xyz))
    ax4.set(xlabel="x (A)",ylabel="y (A)",zlabel="z (A)",title=f"Selected coherent 13C\n{srow.orientation}"); ax4.view_init(elev=24,azim=38); ax4.legend(fontsize=7)
    if not pdfp.empty:
        ax5.plot(pdfp.T2_us,pdfp.delta_chi2,marker="o",ms=3,lw=1.2); ax5.axhline(1,ls="--",lw=.8,label="68% Delta chi2=1"); ax5.axhline(3.84,ls=":",lw=1,label="95% Delta chi2=3.84"); ax5.set_xscale("log"); ax5.legend(fontsize=7)
    ax5.set(title=f"T2 identifiability: {srow.get('T2_profile_status','')}",xlabel="T2 (us)",ylabel="Profile Delta chi2"); ax5.grid(alpha=.2,which="both")
    ax6.axis("off"); lines=[f"NV {nv}",f"orientation = {srow.orientation}",f"orientation ESR RMS = {srow.orientation_rms_error_MHz:.2f} MHz",f"orientation warning = {bool(srow.orientation_warning)}","",f"SELECTED N_C13 = {order}",f"site key = {srow.selected_site_key}",f"reduced chi2 = {srow.selected_red_chi2:.3f}",f"BIC = {srow.selected_bic:.2f}","",f"T2 point = {srow.T2_us:.1f} us",f"T2 profile = {srow.get('T2_profile_report','')}",f"beta = {srow.beta:.3f}",f"revival = {srow.revival_time_us:.3f} us",f"width0 = {srow.width0_us:.3f} us",f"amp taper = {srow.amp_taper_alpha:.3f}",f"width slope = {srow.width_slope:.3f}",f"chirp = {srow.revival_chirp:.4f}",f"dt = {srow.dt_us:.4f} us",f"baseline = {srow.baseline:.4f}",f"contrast = {srow.contrast:.4f}",""]
    for j in range(1,order+1):
        lines += [f"C{j}: site {int(srow[f'c13_{j}_site_id'])}",f"  eta = {srow.get(f'eta{j}',np.nan):.3f}",f"  kappa = {srow[f'c13_{j}_kappa']:.4f}",f"  f-/f+ = {srow[f'c13_{j}_fminus_kHz']:.2f}/{srow[f'c13_{j}_fplus_kHz']:.2f} kHz",f"  r = {srow[f'c13_{j}_distance_A']:.2f} A"]
    ax6.text(.02,.98,"\n".join(lines),va="top",ha="left",family="monospace",fontsize=8.8)
    fig.suptitle(f"V8.1 multi-13C physical fit | NV {nv} | N={order} | {srow.selected_site_key}",fontsize=13); fig.tight_layout(rect=[0,0,1,.96]); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def global_summary_figure(summary):
    fig,axs=plt.subplots(2,2,figsize=(13,9)); c=summary.selected_order.value_counts().sort_index(); axs[0,0].bar(c.index.astype(str),c.values); axs[0,0].set(title="Selected model order",xlabel="Number of coherent 13C",ylabel="NV count"); axs[0,1].hist(summary.selected_red_chi2,bins=35); axs[0,1].set(title="Final fit quality",xlabel="Selected reduced chi2",ylabel="NV count"); axs[1,0].hist(summary.T2_us,bins=35); axs[1,0].set(title="T2 distribution",xlabel="T2 point estimate (us)",ylabel="NV count"); axs[1,1].hist(summary.background_bic-summary.selected_bic,bins=35); axs[1,1].set(title="Evidence for resolved C13",xlabel="BIC improvement over background",ylabel="NV count")
    for ax in axs.flat: ax.grid(alpha=.2)
    fig.suptitle(f"V8.1 orientation-locked multi-C13 fit | {len(summary)} NVs",fontsize=14); fig.tight_layout(); return fig


def parse_nv_list(s):
    if s is None or not str(s).strip(): return None
    return [int(v.strip()) for v in str(s).split(",") if v.strip()]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--nv",type=str,default=None); parser.add_argument("--max-spins",type=int,default=3,choices=(1,2,3)); parser.add_argument("--quick",action="store_true"); parser.add_argument("--workers",type=int,default=DEFAULT_N_JOBS); parser.add_argument("--no-profile",action="store_true"); parser.add_argument("--no-show",action="store_true"); args=parser.parse_args()
    global RUN_T2_PROFILE
    if args.no_profile: RUN_T2_PROFILE=False
    np.random.seed(RANDOM_SEED); paths=discover_paths(); t,y,e,ori_df,ori_map,ori_quality,v6_seed_map,catalog=load_inputs(paths)
    req=parse_nv_list(args.nv); nvs=list(range(y.shape[0])) if req is None else [nv for nv in req if 0<=nv<y.shape[0]]
    print("="*100); print("V8.1 ORIENTATION-LOCKED MULTI-13C FIT + CURRENT BACKGROUND"); print("="*100); print(f"checkpoint: {paths.checkpoint}"); print(f"orientation source: {paths.orientation_csv}"); print(f"V6 seed source: {paths.v6_site_fits}"); print(f"catalog: {CATALOG_PATH}"); print(f"NVs: {len(nvs)} | max spins: {args.max_spins} | quick: {args.quick} | workers: {args.workers}"); print(f"BIC add-spin threshold: {ORDER_ACCEPT_DELTA_BIC}"); print("="*100)
    jobs=[]
    for nv in nvs:
        ori=ori_map.get(nv)
        if ori not in set(ALLOWED_ORIENTATIONS): print(f"[NV {nv}] skipped orientation {ori}"); continue
        jobs.append((nv,ori,ori_quality.get(nv,{}),v6_seed_map.get(nv)))
    def task(nv,ori,oq,seed): return fit_one_nv(nv,t,y,e,ori,oq,seed,catalog,args.max_spins,args.quick)
    with threadpool_limits(limits=BLAS_THREADS_PER_WORKER):
        results=Parallel(n_jobs=max(1,args.workers),backend="loky",batch_size=1,verbose=5)(delayed(task)(*j) for j in jobs)
    candidates=[]; orders=[]; profiles=[]; summaries=[]
    for r in results: candidates+=r["candidate_rows"]; orders+=r["order_rows"]; profiles+=r["profile_rows"]; summaries.append(r["summary"])
    cdf=pd.DataFrame(candidates); odf=pd.DataFrame(orders); pdfp=pd.DataFrame(profiles); sdf=pd.DataFrame(summaries).sort_values("nv_index")
    outdir=Path(OUTPUT_DIR) if OUTPUT_DIR else paths.prefix.parent; outdir.mkdir(parents=True,exist_ok=True); subset="" if req is None else "_subset_"+"-".join(map(str,nvs)); quick="_quick" if args.quick else ""; base=outdir/(paths.prefix.name+"_v8p1_multic13"+subset+quick)
    candidate_csv=Path(str(base)+"_candidate_fits.csv"); order_csv=Path(str(base)+"_model_orders.csv"); profile_csv=Path(str(base)+"_t2_profiles.csv"); summary_csv=Path(str(base)+"_nv_summary.csv"); dashboard_pdf=Path(str(base)+"_dashboard.pdf"); global_png=Path(str(base)+"_global_summary.png"); global_pdf=Path(str(base)+"_global_summary.pdf")
    cdf.to_csv(candidate_csv,index=False); odf.to_csv(order_csv,index=False); pdfp.to_csv(profile_csv,index=False); sdf.to_csv(summary_csv,index=False)
    with PdfPages(dashboard_pdf) as pdf:
        for nv in sdf.nv_index.astype(int): plot_dashboard(pdf,int(nv),t,y,e,sdf,cdf,odf,pdfp,catalog)
    fig=global_summary_figure(sdf); fig.savefig(global_png,dpi=300,bbox_inches="tight"); fig.savefig(global_pdf,bbox_inches="tight")
    print("\n"+"="*100); print("V8.1 COMPLETE"); print("="*100); print(f"NVs fit: {len(sdf)}"); print("selected orders:",sdf.selected_order.value_counts().sort_index().to_dict()); print(f"median reduced chi2: {sdf.selected_red_chi2.median():.3f}"); print(f"median T2: {sdf.T2_us.median():.2f} us"); print("summary:",summary_csv); print("dashboard:",dashboard_pdf); print("="*100)
    if args.no_show: plt.close(fig)
    else: plt.show(block=True)
    return dict(candidate_fits=cdf,model_orders=odf,profiles=pdfp,summary=sdf)


if __name__ == "__main__":
    main()
