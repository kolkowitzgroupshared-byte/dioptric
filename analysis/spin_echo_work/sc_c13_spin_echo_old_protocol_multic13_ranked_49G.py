# -*- coding: utf-8 -*-
"""
49 G old-protocol multi-13C ranked fitter.

Extends the successful legacy additive ESEEM protocol to multiple 13C sites.
Each site keeps its catalog-locked (f+, f-) frequency pair but receives its own
signed amplitude and two phases. The old revival background is retained:
power-law revival taper, width growth, and revival chirp are all fitted.

The existing single-site old-protocol search is used to build the candidate
pool. Multi-site candidates must share one NV orientation, but all four
orientation hypotheses can compete. Final candidates are ranked both by raw
reduced chi-square and by complexity-aware BIC/AICc.

Quick test:
  python analysis/spin_echo_work/sc_c13_spin_echo_old_protocol_multic13_ranked_49G.py --nv 28 --max-spins 2 --quick --no-show
"""
from __future__ import annotations
import argparse, ast, json, os, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from analysis.spin_echo_work import fitter_module_for_spin_echo as oldfit

SEARCH_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo")
RESULT_TAG = "spin_echo_old_protocol_ranked_49G"
OUTPUT_DIR = None
CATALOG_PATH = REPO_ROOT / "analysis" / "spin_echo_work" / "essem_freq_kappa_catalog_22A_49G.json"
BG_LB = np.array([0.0,0.0,15.0,1.0,0.001,0.6,0.0,0.0,-0.06],float)
BG_UB = np.array([1.05,0.95,40.0,20.0,0.300,4.0,4.0,0.8,0.06],float)
AMP_BOUNDS = (-2.0,2.0)
PHASE_BOUNDS = (-np.pi,np.pi)
CPU_COUNT = os.cpu_count() or 4
DEFAULT_N_JOBS = max(1,min(14,CPU_COUNT-2))
BLAS_THREADS_PER_WORKER = 1
QUICK_SITES_PER_ORIENTATION = 3
FULL_SITES_PER_ORIENTATION = 6
QUICK_PAIR_CAP = 20
FULL_PAIR_CAP = 80
QUICK_TRIPLE_CAP = 10
FULL_TRIPLE_CAP = 30
ROBUST_MAX_NFEV = 2500
FINAL_MAX_NFEV = 9000
ORDER_ACCEPT_DELTA_BIC = 10.0  # retained only as a conservative secondary classification
T2_MAX_US_HARD = 300.0
T2_WINDOW_MULTIPLIER = 3.0
T2_BOUND_FRACTION = 0.98
DENSE_POINTS = 2500
TOP_PLOT = 4
TOP_RANK_TABLE = 10
ANALYSIS_VERSION = "v4_exhaustive_alignedseeds_bicprimary_t2window"

def newest_match(pattern):
    hits=list(SEARCH_ROOT.rglob(pattern))
    if not hits: raise FileNotFoundError(f"No {pattern!r} under {SEARCH_ROOT}")
    return max(hits,key=lambda x:x.stat().st_mtime)

def discover_paths():
    attempts=newest_match(f"*{RESULT_TAG}_all_attempts.csv.gz")
    suffix="_all_attempts.csv.gz"
    prefix=Path(str(attempts)[:-len(suffix)])
    checkpoint=Path(str(prefix)+"_fit_checkpoint.npz")
    top=Path(str(prefix)+"_top_fits.csv")
    pool=Path(str(prefix)+"_site_pool_top30.csv")
    for q in (checkpoint,top,pool):
        if not q.exists(): raise FileNotFoundError(q)
    return pool,checkpoint,top,prefix

def parse_orientation(x):
    if isinstance(x,str): x=ast.literal_eval(x)
    a=np.asarray(x,int).ravel()
    return tuple(int(v) for v in a)
def safe_err(e):
    e=np.abs(np.asarray(e,float)); good=np.isfinite(e)&(e>0)
    fallback=float(np.nanmedian(e[good])) if np.any(good) else 1e-3
    return np.maximum(np.where(good,e,fallback),1e-3)

def load_data(checkpoint):
    ck=np.load(checkpoint,allow_pickle=True)
    t=np.asarray(ck["times_us"],float)
    y=np.asarray(ck["norm_counts"],float)
    e=safe_err(ck["norm_counts_ste"])
    if y.ndim!=2 or e.shape!=y.shape or y.shape[1]!=len(t):
        raise ValueError(f"bad checkpoint shapes: t={t.shape}, y={y.shape}, e={e.shape}")
    return t,y,e

def calc_stats(y,e,pred,npar):
    r=(np.asarray(y)-np.asarray(pred))/safe_err(e)
    chi2=float(np.sum(r*r)); n=len(r); k=int(npar); dof=max(1,n-k)
    aic=chi2+2*k
    aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf
    bic=chi2+k*np.log(max(n,2))
    return dict(chi2=chi2,red_chi2=chi2/dof,aicc=float(aicc),bic=float(bic),npar=k)

def carrier_from_bg(t,bg):
    baseline,contrast,trev,width0,T2_ms,beta,taper,slope,chirp=np.asarray(bg,float)
    tt=np.asarray(t,float); T2_us=max(1e-9,1000.0*T2_ms)
    env=np.exp(-np.power(np.maximum(tt,0.0)/T2_us,beta))
    nrev=max(1,min(64,int(np.ceil(1.2*float(tt.max())/max(trev,1e-9)))+1))
    comb=oldfit._comb_quartic_powerlaw(tt,float(trev),float(width0),float(taper),float(slope),float(chirp),nrev)
    return float(baseline),float(contrast),env*comb

def model(t,theta,sites):
    n=len(sites); th=np.asarray(theta,float); bg=th[:9]
    baseline,contrast,carrier=carrier_from_bg(t,bg)
    osc=np.zeros_like(np.asarray(t,float))
    j=9
    for s in sites:
        amp,phi0,phi1=th[j:j+3]; j+=3
        f0=float(s["f0_kHz"])/1000.0; f1=float(s["f1_kHz"])/1000.0
        osc += amp*(np.cos(2*np.pi*f0*np.asarray(t)+phi0)+np.cos(2*np.pi*f1*np.asarray(t)+phi1))
    return baseline-contrast*carrier+carrier*osc
def t2_upper_us(t):
    """Upper T2 limit set by both a hard ceiling and the measured time window."""
    t=np.asarray(t,float)
    tmax=float(np.nanmax(t)) if t.size else 100.0
    return float(min(T2_MAX_US_HARD,max(120.0,T2_WINDOW_MULTIPLIER*tmax)))

def theta_bounds(nsite,baseline_seed=0.7,t2_max_us=T2_MAX_US_HARD):
    lb=list(BG_LB); ub=list(BG_UB)
    ub[1]=min(ub[1],max(0.05,float(baseline_seed)-0.01))
    ub[4]=min(float(ub[4]),float(t2_max_us)/1000.0)
    for _ in range(nsite):
        lb.extend([AMP_BOUNDS[0],PHASE_BOUNDS[0],PHASE_BOUNDS[0]])
        ub.extend([AMP_BOUNDS[1],PHASE_BOUNDS[1],PHASE_BOUNDS[1]])
    return np.asarray(lb,float),np.asarray(ub,float)

def old_popt_to_theta(popt):
    p=np.asarray(popt,float)
    if p.size<14: raise ValueError("legacy popt has fewer than 14 parameters")
    return np.r_[p[:9],p[9],p[11],p[13]]

def site_from_row(r):
    return dict(site_id=int(r.site_id),orientation=parse_orientation(r.orientation),
                kappa=float(getattr(r,"kappa",np.nan)),distance_A=float(getattr(r,"distance_A",np.nan)),
                f0_kHz=float(r.f0_kHz),f1_kHz=float(r.f1_kHz))

def site_key(sites):
    return tuple(sorted(int(s["site_id"]) for s in sites))

def fit_model(t,y,e,sites,seeds,robust_nfev=ROBUST_MAX_NFEV,final_nfev=FINAL_MAX_NFEV):
    if not seeds: return None
    baseline_seed=float(np.nanmedian([np.asarray(s,float)[0] for s in seeds]))
    t2_limit=t2_upper_us(t)
    lb,ub=theta_bounds(len(sites),baseline_seed,t2_limit)
    ee=safe_err(e)
    def resid(th): return (np.asarray(y)-model(t,th,sites))/ee
    robust=[]
    for seed in seeds:
        s=np.clip(np.asarray(seed,float),lb+1e-8,ub-1e-8)
        try:
            rr=least_squares(resid,s,bounds=(lb,ub),loss="soft_l1",f_scale=1.0,
                             max_nfev=robust_nfev,x_scale="jac")
            pred=model(t,rr.x,sites)
            robust.append((calc_stats(y,ee,pred,len(rr.x))["chi2"],rr.x))
        except Exception:
            pass
    if not robust: return None
    robust.sort(key=lambda z:z[0]); finals=[]
    for _,seed in robust[:min(3,len(robust))]:
        try:
            rr=least_squares(resid,seed,bounds=(lb,ub),loss="linear",max_nfev=final_nfev,
                             ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac")
            pred=model(t,rr.x,sites); st=calc_stats(y,ee,pred,len(rr.x))
            finals.append(dict(theta=rr.x,pred=pred,sites=tuple(sites),site_key=site_key(sites),
                               t2_limit_us=float(t2_limit),**st))
        except Exception:
            pass
    return min(finals,key=lambda z:(z["chi2"],z["bic"])) if finals else None
def legacy_candidate_pool(attempts,nv,quick,sites_per_orientation):
    d=attempts[(attempts.nv_index==nv)&(attempts.status=="ok")].copy()
    if d.empty: return []
    score="score_primary" if "score_primary" in d.columns else "red_chi2"
    d=d.sort_values([score,"red_chi2","aicc"])
    d=d.drop_duplicates(["orientation","site_id"],keep="first")
    keep=QUICK_SITES_PER_ORIENTATION if quick else int(sites_per_orientation)
    out=[]
    for ori,g in d.groupby("orientation",sort=False):
        out.extend([r for r in g.head(keep).itertuples()])
    out.sort(key=lambda r:(float(getattr(r,score)),float(r.red_chi2)))
    return out

def legacy_single_fit(t,y,e,row):
    """Reuse a legacy single-site solution unless its T2 lies outside the resolvable window."""
    s=site_from_row(row)
    th=old_popt_to_theta(json.loads(row.popt_json))
    t2_limit=t2_upper_us(t)
    if float(th[4])*1000.0 >= T2_BOUND_FRACTION*t2_limit:
        refit=fit_model(t,y,e,[s],[th],
                        robust_nfev=min(1500,ROBUST_MAX_NFEV),
                        final_nfev=min(6000,FINAL_MAX_NFEV))
        if refit is not None:
            refit["legacy_red_chi2"]=float(row.red_chi2)
            refit["legacy_refit_for_t2_bound"]=True
            return refit
    pred=model(t,th,[s])
    st=calc_stats(y,e,pred,len(th))
    return dict(theta=th,pred=pred,sites=(s,),site_key=site_key([s]),
                t2_limit_us=float(t2_limit),legacy_red_chi2=float(row.red_chi2),
                legacy_refit_for_t2_bound=False,**st)

def refit_singles(t,y,e,rows):
    fits=[legacy_single_fit(t,y,e,r) for r in rows]
    fits.sort(key=lambda z:(z["bic"],z["red_chi2"]))
    return fits

def background_fit(t,y,e,rows):
    seeds=[]
    for r in rows[:4]:
        try: seeds.append(np.asarray(json.loads(r.popt_json),float)[:9])
        except Exception: pass
    if not seeds:
        p0,_,_=oldfit._initial_guess_and_bounds(t,y,True,None); seeds=[p0[:9]]
    return fit_model(t,y,e,[],seeds)

def pair_seed(a,b):
    ta=np.asarray(a["theta"],float); tb=np.asarray(b["theta"],float)
    return np.r_[ta[:9],ta[9:12],tb[9:12]]

def fit_pairs(t,y,e,singles,quick,sites_per_orientation):
    byori={}
    for f in singles:
        ori=tuple(f["sites"][0]["orientation"]); byori.setdefault(ori,[]).append(f)
    specs=[]
    for ori,fs in byori.items():
        fs=sorted(fs,key=lambda z:(z["bic"],z["red_chi2"]))
        lim=QUICK_SITES_PER_ORIENTATION if quick else int(sites_per_orientation)
        fs=fs[:lim]
        for i in range(len(fs)):
            for j in range(i+1,len(fs)):
                specs.append((fs[i],fs[j]))
    specs=sorted(specs,key=lambda ab:(ab[0]["bic"]+ab[1]["bic"]))
    if quick:
        specs=specs[:QUICK_PAIR_CAP]
    out=[]
    for a,b in specs:
        sites=[a["sites"][0],b["sites"][0]]
        ta=np.asarray(a["theta"],float); tb=np.asarray(b["theta"],float)
        # Keep the C13 parameter blocks aligned with sites=[a,b] for both
        # background initializations.
        s1=np.r_[ta[:9],ta[9:12],tb[9:12]]
        s2=np.r_[tb[:9],ta[9:12],tb[9:12]]
        f=fit_model(t,y,e,sites,[s1,s2])
        if f is not None: out.append(f)
    out.sort(key=lambda z:(z["bic"],z["red_chi2"]))
    return out
def triple_seed(pair,single):
    tp=np.asarray(pair["theta"],float); ts=np.asarray(single["theta"],float)
    return np.r_[tp,ts[9:12]]

def triple_seed_aligned(pair,single,target_sites):
    """Build a pair+single seed whose 3-parameter blocks match target_sites order."""
    tp=np.asarray(pair["theta"],float); ts=np.asarray(single["theta"],float)
    blocks={}
    for idx,s in enumerate(pair["sites"]):
        blocks[int(s["site_id"])]=tp[9+3*idx:12+3*idx]
    blocks[int(single["sites"][0]["site_id"])]=ts[9:12]
    return np.concatenate([tp[:9]] + [blocks[int(s["site_id"])] for s in target_sites])

def fit_triples(t,y,e,pairs,singles,quick,sites_per_orientation):
    if not pairs: return []

    # quick=True retains the pruned beam-style search for fast screening.
    if quick:
        single_map={}
        for s in singles:
            ori=tuple(s["sites"][0]["orientation"])
            single_map.setdefault(ori,[]).append(s)
        specs={}
        parent_lim=4
        add_lim=5
        for p in pairs[:parent_lim]:
            ori=tuple(p["sites"][0]["orientation"])
            used=set(site_key(p["sites"]))
            for s in single_map.get(ori,[])[:add_lim]:
                sid=int(s["sites"][0]["site_id"])
                if sid in used: continue
                key=tuple(sorted((*used,sid)))
                specs.setdefault(key,(p,s))
        out=[]
        for p,s in list(specs.values())[:QUICK_TRIPLE_CAP]:
            sites=list(p["sites"])+[s["sites"][0]]
            f=fit_model(t,y,e,sites,[triple_seed(p,s)])
            if f is not None: out.append(f)
        out.sort(key=lambda z:(z["bic"],z["red_chi2"]))
        return out

    # Full mode: exhaustive triples among the same top-K singles/orientation
    # used for the pair search. This removes parent-pair pruning.
    byori={}
    for s in singles:
        ori=tuple(s["sites"][0]["orientation"])
        byori.setdefault(ori,[]).append(s)

    pair_map={}
    for p in pairs:
        pair_map[(tuple(p["sites"][0]["orientation"]),site_key(p["sites"]))]=p

    out=[]
    for ori,fs in byori.items():
        fs=sorted(fs,key=lambda z:(z["bic"],z["red_chi2"]))[:int(sites_per_orientation)]
        n=len(fs)
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1,n):
                    pair_parents=[]
                    for a,b,c in ((i,j,k),(i,k,j),(j,k,i)):
                        key=(ori,tuple(sorted((int(fs[a]["sites"][0]["site_id"]),
                                              int(fs[b]["sites"][0]["site_id"])))))
                        p=pair_map.get(key)
                        if p is not None:
                            pair_parents.append((p,fs[c]))

                    if pair_parents:
                        pair_parents.sort(key=lambda ps:(ps[0]["bic"],ps[0]["red_chi2"]))
                        best_pair,best_single=pair_parents[0]
                        sites=list(best_pair["sites"])+[best_single["sites"][0]]
                        seeds=[triple_seed_aligned(p,s,sites) for p,s in pair_parents]
                    else:
                        sites=[fs[i]["sites"][0],fs[j]["sites"][0],fs[k]["sites"][0]]
                        seeds=[np.r_[np.asarray(fs[i]["theta"],float)[:9],
                                     np.asarray(fs[i]["theta"],float)[9:12],
                                     np.asarray(fs[j]["theta"],float)[9:12],
                                     np.asarray(fs[k]["theta"],float)[9:12]]]

                    f=fit_model(t,y,e,sites,seeds)
                    if f is not None: out.append(f)
    out.sort(key=lambda z:(z["bic"],z["red_chi2"]))
    return out

def public_row(nv,order,rank,f):
    bg=np.asarray(f["theta"],float)[:9]
    ori=tuple(f["sites"][0]["orientation"]) if order else ()
    t2_limit=float(f.get("t2_limit_us",T2_MAX_US_HARD))
    t2_value=1000.0*float(bg[4])
    row=dict(nv_index=int(nv),model_order=int(order),rank_within_order=int(rank),
             orientation=str(ori),site_key=str(tuple(f["site_key"])),
             chi2=float(f["chi2"]),red_chi2=float(f["red_chi2"]),
             aicc=float(f["aicc"]),bic=float(f["bic"]),npar=int(f["npar"]),
             baseline=bg[0],contrast=bg[1],revival_time_us=bg[2],width0_us=bg[3],
             T2_us=t2_value,T2_limit_us=t2_limit,
             T2_bound_hit=bool(t2_value>=T2_BOUND_FRACTION*t2_limit),
             beta=bg[5],amp_taper_alpha=bg[6],
             width_slope=bg[7],revival_chirp=bg[8],
             theta_json=json.dumps(np.asarray(f["theta"],float).tolist()))
    if "legacy_red_chi2" in f: row["legacy_red_chi2"]=f["legacy_red_chi2"]
    if "legacy_refit_for_t2_bound" in f:
        row["legacy_refit_for_t2_bound"]=bool(f["legacy_refit_for_t2_bound"])
    for j,s in enumerate(f["sites"],1):
        row.update({f"c13_{j}_site_id":int(s["site_id"]),f"c13_{j}_f0_kHz":float(s["f0_kHz"]),
                    f"c13_{j}_f1_kHz":float(s["f1_kHz"]),f"c13_{j}_kappa":float(s["kappa"]),
                    f"c13_{j}_distance_A":float(s["distance_A"])})
        k=9+3*(j-1); row[f"c13_{j}_amp"]=float(f["theta"][k])
        row[f"c13_{j}_phi0"]=float(f["theta"][k+1]); row[f"c13_{j}_phi1"]=float(f["theta"][k+2])
    return row
def fit_one_nv(nv,t,y,e,attempts,max_spins,quick,sites_per_orientation):
    rows=legacy_candidate_pool(attempts,nv,quick,sites_per_orientation)
    if not rows: raise RuntimeError(f"NV {nv}: no successful legacy candidates")
    bg=background_fit(t,y,e,rows)
    singles=refit_singles(t,y,e,rows)
    allfits={0:[bg] if bg else [],1:singles}
    pairs=fit_pairs(t,y,e,singles,quick,sites_per_orientation) if max_spins>=2 else []
    if pairs: allfits[2]=pairs
    triples=fit_triples(t,y,e,pairs,singles,quick,sites_per_orientation) if max_spins>=3 else []
    if triples: allfits[3]=triples
    recs=[]
    for order,fits in allfits.items():
        for rank,f in enumerate(fits,1): recs.append(public_row(nv,order,rank,f))
    bic_idx=sorted(range(len(recs)),key=lambda i:(recs[i]["bic"],recs[i]["red_chi2"]))
    aicc_idx=sorted(range(len(recs)),key=lambda i:(recs[i]["aicc"],recs[i]["bic"]))
    red_idx=sorted(range(len(recs)),key=lambda i:(recs[i]["red_chi2"],recs[i]["bic"]))
    for rank,i in enumerate(bic_idx,1): recs[i]["rank_global_bic"]=rank
    for rank,i in enumerate(aicc_idx,1): recs[i]["rank_global_aicc"]=rank
    for rank,i in enumerate(red_idx,1): recs[i]["rank_global_redchi"]=rank
    best_bic=recs[bic_idx[0]]; best_aicc=recs[aicc_idx[0]]; best_red=recs[red_idx[0]]
    print(f"[NV {nv:3d}] BIC: N={best_bic['model_order']} {best_bic['site_key']} chi2r={best_bic['red_chi2']:.3f} | AICc: N={best_aicc['model_order']} {best_aicc['site_key']} | raw: N={best_red['model_order']} {best_red['site_key']} chi2r={best_red['red_chi2']:.3f}")
    return recs

def sites_from_row(row):
    ori=parse_orientation(row.orientation) if int(row.model_order)>0 else ()
    out=[]
    for j in range(1,int(row.model_order)+1):
        out.append(dict(site_id=int(row[f"c13_{j}_site_id"]),orientation=ori,
                        f0_kHz=float(row[f"c13_{j}_f0_kHz"]),f1_kHz=float(row[f"c13_{j}_f1_kHz"]),
                        kappa=float(row.get(f"c13_{j}_kappa",np.nan)),
                        distance_A=float(row.get(f"c13_{j}_distance_A",np.nan))))
    return out

def prediction_from_row(t,row):
    return model(t,np.asarray(json.loads(row.theta_json),float),sites_from_row(row))

def load_catalog_lookup():
    with open(CATALOG_PATH,"r",encoding="utf-8") as f:
        raw=json.load(f)
    out={}
    for rec in raw:
        try:
            key=(parse_orientation(rec["orientation"]),int(rec["site_index"]))
        except Exception:
            continue
        out[key]=rec
    return out

def enrich_candidates_with_catalog(cdf,catalog_lookup):
    cdf=cdf.copy()
    fields=("x_A","y_A","z_A","A_par_Hz","A_perp_Hz","theta_deg","f_minus_Hz","f_plus_Hz")
    for j in range(1,4):
        sid_col=f"c13_{j}_site_id"
        if sid_col not in cdf.columns: continue
        for field in fields:
            outfield=f"c13_{j}_{field}"
            if outfield not in cdf.columns: cdf[outfield]=np.nan
        valid=cdf[sid_col].notna()
        for idx,row in cdf.loc[valid].iterrows():
            try:
                ori=parse_orientation(row.orientation); sid=int(row[sid_col])
                rec=catalog_lookup.get((ori,sid))
            except Exception:
                rec=None
            if rec is None: continue
            for field in fields:
                val=rec.get(field,np.nan)
                if field in ("A_par_Hz","A_perp_Hz","f_minus_Hz","f_plus_Hz") and np.isfinite(val):
                    val=float(val)/1e3
                    outfield=f"c13_{j}_{field.replace('_Hz','_kHz')}"
                    if outfield not in cdf.columns: cdf[outfield]=np.nan
                    cdf.at[idx,outfield]=val
                else:
                    cdf.at[idx,f"c13_{j}_{field}"]=val
    return cdf

def select_delta10_for_nv(d,threshold=ORDER_ACCEPT_DELTA_BIC):
    by_order={int(o):gg.sort_values(["bic","red_chi2"]).iloc[0]
              for o,gg in d.groupby("model_order")}
    selected=by_order[min(by_order)]
    path=[]
    for o in sorted(k for k in by_order if k>0):
        cand=by_order[o]
        delta=float(selected.bic-cand.bic)
        accepted=bool(delta>=threshold)
        path.append(dict(candidate_order=int(o),delta_bic=delta,accepted=accepted))
        if accepted: selected=cand
    return selected,path

def _evidence_label(delta_bic):
    if not np.isfinite(delta_bic): return "n/a"
    if delta_bic < 2.0: return "ambiguous"
    if delta_bic < 6.0: return "positive"
    if delta_bic < 10.0: return "strong"
    return "very strong"

def select_bic_with_evidence(d):
    g=d.sort_values(["bic","red_chi2"]).copy()
    selected=g.iloc[0]
    raw=g.sort_values(["red_chi2","bic"]).iloc[0]
    runner_gap=float(g.iloc[1].bic-selected.bic) if len(g)>1 else np.nan
    other_order=g[g.model_order!=selected.model_order]
    order_gap=float(other_order.bic.min()-selected.bic) if len(other_order) else np.nan
    same_order=g[(g.model_order==selected.model_order)&(g.site_key!=selected.site_key)]
    site_gap=float(same_order.bic.min()-selected.bic) if len(same_order) else np.nan
    conservative,path=select_delta10_for_nv(d)
    aicc_best=d.sort_values(["aicc","bic"]).iloc[0]
    raw_gain=float(selected.red_chi2-raw.red_chi2)
    raw_bic_cost=float(raw.bic-selected.bic)
    review=[]
    if np.isfinite(order_gap) and order_gap<2.0: review.append("model order ambiguous")
    if int(selected.model_order)>0 and np.isfinite(site_gap) and site_gap<2.0:
        review.append("site assignment ambiguous")
    if raw_gain>0.10: review.append("raw fit materially better")
    if float(selected.red_chi2)>2.5: review.append("poor absolute fit")
    if bool(selected.get("T2_bound_hit",False)): review.append("T2 unresolved")
    ev=dict(
        delta_bic_runner_up=runner_gap,
        delta_bic_other_order=order_gap,
        delta_bic_same_order_site=site_gap,
        model_order_evidence=_evidence_label(order_gap),
        site_assignment_evidence=_evidence_label(site_gap),
        raw_best_delta_chi2r=raw_gain,
        raw_best_delta_bic=raw_bic_cost,
        aicc_best_order=int(aicc_best.model_order),
        aicc_best_site_key=str(aicc_best.site_key),
        aicc_best_red_chi2=float(aicc_best.red_chi2),
        conservative_order=int(conservative.model_order),
        conservative_site_key=str(conservative.site_key),
        review_flags="; ".join(review) if review else "none",
    )
    return selected,raw,conservative,path,ev

def _row_id(row):
    return (int(row.model_order),str(row.site_key))

def _fmt(v,digits=3):
    try:
        v=float(v)
        if not np.isfinite(v): return "nan"
        return f"{v:.{digits}f}"
    except Exception:
        return str(v)

def _set_equal_3d(ax,pts):
    pts=np.asarray(pts,float)
    if pts.size==0: return
    pts=pts[np.all(np.isfinite(pts),axis=1)]
    if len(pts)==0:return
    pts=np.vstack([pts,np.zeros((1,3))])
    lo=pts.min(0); hi=pts.max(0); center=.5*(lo+hi)
    radius=max(1.0,.58*float(np.max(hi-lo)))
    ax.set_xlim(center[0]-radius,center[0]+radius)
    ax.set_ylim(center[1]-radius,center[1]+radius)
    ax.set_zlim(center[2]-radius,center[2]+radius)

def _boundary_flags(row):
    flags=[]
    if bool(row.get("T2_bound_hit",False)):
        flags.append(f"T2 unresolved at >= {_fmt(row.get('T2_limit_us',np.nan),1)} us")
    if abs(float(row.beta)-BG_LB[5]) < 0.01 or abs(float(row.beta)-BG_UB[5]) < 0.01:
        flags.append("beta at bound")
    if abs(float(row.width_slope)-BG_LB[7]) < 0.01 or abs(float(row.width_slope)-BG_UB[7]) < 0.01:
        flags.append("width_slope at bound")
    if abs(float(row.revival_chirp)-BG_LB[8]) < 0.002 or abs(float(row.revival_chirp)-BG_UB[8]) < 0.002:
        flags.append("revival_chirp at bound")
    for j in range(1,int(row.model_order)+1):
        amp=float(row.get(f"c13_{j}_amp",0.0))
        if abs(amp) >= 0.98*max(abs(AMP_BOUNDS[0]),abs(AMP_BOUNDS[1])):
            flags.append(f"C13#{j} amplitude at bound")
    return flags

def _background_text(row,evidence,conservative,path):
    if bool(row.get("T2_bound_hit",False)):
        t2txt=f">= {_fmt(row.get('T2_limit_us',np.nan),1)} us  [UNRESOLVED]"
    else:
        t2txt=f"{_fmt(row.T2_us,3)} us"
    lines=[
        "PRIMARY SELECTION: GLOBAL MINIMUM BIC",
        f"N_C13 = {int(row.model_order)}    sites = {row.site_key}",
        f"orientation = {row.orientation}",
        f"reduced chi2 = {_fmt(row.red_chi2,3)}",
        f"BIC = {_fmt(row.bic,2)}    AICc = {_fmt(row.aicc,2)}",
        f"model-order evidence = {evidence['model_order_evidence']}  (DeltaBIC={_fmt(evidence['delta_bic_other_order'],2)})",
        f"site evidence        = {evidence['site_assignment_evidence']}  (DeltaBIC={_fmt(evidence['delta_bic_same_order_site'],2)})",
        f"runner-up DeltaBIC   = {_fmt(evidence['delta_bic_runner_up'],2)}",
        f"raw-best gain chi2r  = {_fmt(evidence['raw_best_delta_chi2r'],3)}",
        f"raw-best BIC cost     = {_fmt(evidence['raw_best_delta_bic'],2)}",
        f"review flags          = {evidence['review_flags']}",
        f"AICc-best N          = {evidence['aicc_best_order']} {evidence['aicc_best_site_key']} (chi2r={_fmt(evidence['aicc_best_red_chi2'],3)})",
        f"conservative N       = {int(conservative.model_order)} {conservative.site_key}",
        "",
        "BACKGROUND PARAMETERS",
        f"baseline          = {_fmt(row.baseline,6)}",
        f"contrast          = {_fmt(row.contrast,6)}",
        f"revival_time_us   = {_fmt(row.revival_time_us,6)}",
        f"width0_us         = {_fmt(row.width0_us,6)}",
        f"T2                = {t2txt}",
        f"beta              = {_fmt(row.beta,6)}",
        f"amp_taper_alpha   = {_fmt(row.amp_taper_alpha,6)}",
        f"width_slope       = {_fmt(row.width_slope,6)}",
        f"revival_chirp     = {_fmt(row.revival_chirp,6)}",
    ]
    flags=_boundary_flags(row)
    if flags:
        lines+=["","BOUNDARY / IDENTIFIABILITY FLAGS"]+[f"- {x}" for x in flags]
    if path:
        lines+=["","CONSERVATIVE DeltaBIC>=10 PATH"]
        for q in path:
            lines.append(f"N={q['candidate_order']}: DeltaBIC={q['delta_bic']:.2f} -> {'accept' if q['accepted'] else 'hold simpler'}")
    return "\n".join(lines)

def _c13_text(row,j):
    if j>int(row.model_order): return ""
    vals=[
        ("site",row.get(f"c13_{j}_site_id",np.nan),0),
        ("distance_A",row.get(f"c13_{j}_distance_A",np.nan),3),
        ("x_A",row.get(f"c13_{j}_x_A",np.nan),3),
        ("y_A",row.get(f"c13_{j}_y_A",np.nan),3),
        ("z_A",row.get(f"c13_{j}_z_A",np.nan),3),
        ("kappa",row.get(f"c13_{j}_kappa",np.nan),6),
        ("f0_kHz",row.get(f"c13_{j}_f0_kHz",np.nan),3),
        ("f1_kHz",row.get(f"c13_{j}_f1_kHz",np.nan),3),
        ("Apar_kHz",row.get(f"c13_{j}_A_par_kHz",np.nan),3),
        ("Aperp_kHz",row.get(f"c13_{j}_A_perp_kHz",np.nan),3),
        ("theta_deg",row.get(f"c13_{j}_theta_deg",np.nan),2),
        ("amp",row.get(f"c13_{j}_amp",np.nan),6),
        ("phi0_rad",row.get(f"c13_{j}_phi0",np.nan),5),
        ("phi1_rad",row.get(f"c13_{j}_phi1",np.nan),5),
    ]
    return "\n".join([f"C13 #{j}"]+[f"{name:13s} = {_fmt(val,nd)}" for name,val,nd in vals])

def _model_equation_text():
    return (
        "MODEL USED\n"
        "S(t) = b - C M(t) + M(t) * Sum_j a_j [cos(2*pi*f0_j*t + phi0_j)\n"
        "                              + cos(2*pi*f1_j*t + phi1_j)]\n"
        "M(t) = exp[-(t/T2)^beta] * R(t)\n"
        "R(t) = Sum_k (1+k)^(-alpha) exp[-((t-mu_k)/w_k)^4]\n"
        "mu_k = k*Trev*(1 + k*chirp),   w_k = w0*(1 + k*width_slope)\n"
        "f0_j, f1_j are locked to the field-specific C13 lattice catalog."
    )

def plot_global_summary(pdf,cdf,best_delta10,max_spins):
    best_bic=cdf[cdf.rank_global_bic==1].sort_values("nv_index")
    best_aicc=cdf[cdf.rank_global_aicc==1].sort_values("nv_index")
    best_raw=cdf[cdf.rank_global_redchi==1].sort_values("nv_index")
    field_label="49 G" if "49G" in RESULT_TAG else "52 G"

    evidence_rows=[]
    for nv,g in cdf.groupby("nv_index"):
        sel,raw,cons,path,ev=select_bic_with_evidence(g)
        evidence_rows.append(dict(nv_index=int(nv),**ev))
    evdf=pd.DataFrame(evidence_rows)

    fig,axs=plt.subplots(2,3,figsize=(20,13.5))
    orders=np.arange(max_spins+1)
    counts=best_bic.model_order.value_counts().reindex(orders,fill_value=0)
    axs[0,0].bar(counts.index.astype(str),counts.values)
    axs[0,0].set(title="Primary model order: global minimum BIC",
                 xlabel="Number of resolved C13",ylabel="NV count")
    for x,v in zip(counts.index,counts.values):
        axs[0,0].text(x,v+1,str(int(v)),ha="center",fontsize=9)

    vals=np.asarray(best_bic.red_chi2,float)
    finite=vals[np.isfinite(vals)]
    hi=max(2.5,min(10.0,float(np.nanpercentile(finite,97)))) if len(finite) else 3.0
    axs[0,1].hist(np.clip(finite,0,hi),bins=35)
    axs[0,1].axvline(np.nanmedian(finite),ls="--",lw=1.2,
                     label=f"median={np.nanmedian(finite):.2f}")
    axs[0,1].set(title="Primary fit quality",
                 xlabel="Reduced chi-square (clipped at 97th pct)",ylabel="NV count")
    axs[0,1].legend(fontsize=8)

    gaps=np.asarray(evdf.delta_bic_other_order,float)
    gaps=gaps[np.isfinite(gaps)]
    axs[0,2].hist(np.clip(gaps,0,20),bins=30)
    for x,lab in ((2,"2"),(6,"6"),(10,"10")):
        axs[0,2].axvline(x,ls="--",lw=.8,label=f"DeltaBIC {lab}")
    axs[0,2].set(title="Model-order evidence",
                 xlabel="Delta BIC to best competing order (clipped at 20)",ylabel="NV count")
    axs[0,2].legend(fontsize=7)
    t2hit=(best_bic.T2_bound_hit.astype(bool)
           if "T2_bound_hit" in best_bic else pd.Series(False,index=best_bic.index))
    for hit,g in best_bic.groupby(t2hit):
        lab="T2 unresolved/bound" if bool(hit) else "T2 resolved"
        axs[1,0].scatter(g.T2_us,g.red_chi2,s=20,alpha=.75,label=lab)
    axs[1,0].set(title="T2 identifiability vs fit quality",
                 xlabel="T2 (us)",ylabel="Reduced chi-square")
    axs[1,0].legend(fontsize=8)

    w=.19
    cb=best_bic.model_order.value_counts().reindex(orders,fill_value=0)
    ca=best_aicc.model_order.value_counts().reindex(orders,fill_value=0)
    cd=best_delta10.model_order.value_counts().reindex(orders,fill_value=0)
    cr=best_raw.model_order.value_counts().reindex(orders,fill_value=0)
    axs[1,1].bar(orders-1.5*w,cb.values,width=w,label="global BIC")
    axs[1,1].bar(orders-.5*w,ca.values,width=w,label="AICc")
    axs[1,1].bar(orders+.5*w,cd.values,width=w,label="conservative DeltaBIC>=10")
    axs[1,1].bar(orders+1.5*w,cr.values,width=w,label="raw chi2")
    axs[1,1].set_xticks(orders)
    axs[1,1].set(title="Model-order comparison",xlabel="Model order",ylabel="NV count")
    axs[1,1].legend(fontsize=7)

    axs[1,2].axis("off")
    ambiguous=int((evdf.delta_bic_other_order<2).sum())
    t2count=int(t2hit.sum())
    good_multi=best_bic[(best_bic.model_order>=2)&(best_bic.red_chi2<2)]
    good3=best_bic[(best_bic.model_order>=3)&(best_bic.red_chi2<2)]
    t2lim=float(best_bic.T2_limit_us.median()) if "T2_limit_us" in best_bic else np.nan
    summary=[
        "RUN SUMMARY",
        f"dataset                  : {field_label}",
        f"NVs fit                  : {len(best_bic)}",
        f"max C13 tested           : {max_spins}",
        "primary selection        : global minimum BIC",
        "DeltaBIC>=10             : secondary/conservative label only",
        f"median primary chi2r     : {best_bic.red_chi2.median():.3f}",
        f"median AICc-best chi2r   : {best_aicc.red_chi2.median():.3f}",
        f"median raw-best chi2r    : {best_raw.red_chi2.median():.3f}",
        f"ambiguous model order <2 : {ambiguous}",
        f"T2 unresolved at bound   : {t2count}",
        f"effective T2 ceiling     : {t2lim:.1f} us",
        f"primary N>=2, chi2r<2    : {len(good_multi)}",
        f"primary N>=3, chi2r<2    : {len(good3)}",
        "",
        "Per-NV pages: global-BIC primary + AICc/raw/conservative alternatives,",
        "residuals, rankings, model-order evidence, C13 geometry,",
        "all parameters, identifiability flags, and the fitted model equation.",
    ]
    axs[1,2].text(.02,.98,"\n".join(summary),va="top",ha="left",
                  family="monospace",fontsize=9)
    for ax in axs.flat:
        if ax is not axs[1,2]: ax.grid(alpha=.16)
    fig.suptitle(f"{field_label} old-protocol multi-C13 ranked fit: global diagnostics",
                 fontsize=16)
    fig.subplots_adjust(left=.06,right=.98,bottom=.06,top=.94,hspace=.28,wspace=.25)
    pdf.savefig(fig)
    plt.close(fig)

def plot_nv(pdf,nv,t,y,e,cdf):
    d=cdf[cdf.nv_index==nv].copy()
    selected,raw_best,conservative,path,evidence=select_bic_with_evidence(d)
    aicc_best=d.sort_values(["aicc","bic"]).iloc[0]
    top_bic=d.sort_values(["bic","red_chi2"]).head(TOP_RANK_TABLE)
    field_label="49 G" if "49G" in RESULT_TAG else "52 G"

    fig=plt.figure(figsize=(20,13.5))
    # Four-row layout keeps the model equation physically separate from all plots.
    gs=fig.add_gridspec(4,3,height_ratios=[1.0,1.0,1.12,.34],
                        width_ratios=[1.2,1.05,1.0],hspace=.34,wspace=.27)
    ax_full=fig.add_subplot(gs[0,:2]); ax_rank=fig.add_subplot(gs[0,2])
    ax_zoom=fig.add_subplot(gs[1,0]); ax_res=fig.add_subplot(gs[1,1])
    ax3d=fig.add_subplot(gs[1,2],projection="3d")
    ax_order=fig.add_subplot(gs[2,0])
    ax_info=fig.add_subplot(gs[2,1:])
    ax_model=fig.add_subplot(gs[3,:])
    yy=np.asarray(y[nv],float); ee=safe_err(e[nv])
    td=np.linspace(float(t.min()),float(t.max()),DENSE_POINTS)
    ax_full.errorbar(t,yy,yerr=ee,fmt="o",ms=3,capsize=1,lw=.45,
                     label="data",zorder=10)

    plotted=set()
    def draw_row(row,label,ls="-",lw=2.0,alpha=1.0):
        rid=_row_id(row)
        if rid in plotted: return
        plotted.add(rid)
        ax_full.plot(td,prediction_from_row(td,row),ls=ls,lw=lw,alpha=alpha,label=label)

    draw_row(selected,
             f"PRIMARY BIC #1 | N={int(selected.model_order)} {selected.site_key} | chi2r={selected.red_chi2:.2f}",
             lw=2.5)
    if _row_id(aicc_best)!=_row_id(selected):
        draw_row(aicc_best,
                 f"AICc #1 | N={int(aicc_best.model_order)} {aicc_best.site_key} | chi2r={aicc_best.red_chi2:.2f}",
                 ls="-.",lw=1.7)
    if _row_id(raw_best)!=_row_id(selected):
        draw_row(raw_best,
                 f"raw chi2 #1 | N={int(raw_best.model_order)} {raw_best.site_key} | chi2r={raw_best.red_chi2:.2f}",
                 ls=":",lw=1.8)
    if _row_id(conservative) not in plotted:
        draw_row(conservative,
                 f"conservative DeltaBIC>=10 | N={int(conservative.model_order)} {conservative.site_key}",
                 ls="--",lw=1.6)
    for _,r in top_bic.head(4).iterrows():
        if _row_id(r) not in plotted:
            draw_row(r,f"BIC rank {int(r.rank_global_bic)} | N={int(r.model_order)} {r.site_key}",
                     lw=.9,alpha=.45)
    ax_full.set(xlabel="Total evolution time (us)",ylabel="Normalized signal",
                title=f"NV {nv}: full trace and ranked alternatives")
    ax_full.grid(alpha=.2); ax_full.legend(fontsize=7,ncol=2,loc="best")

    center=float(selected.revival_time_us); m=np.abs(t-center)<=12.5
    tz=np.linspace(center-12.5,center+12.5,DENSE_POINTS)
    ax_zoom.errorbar(t[m],yy[m],yerr=ee[m],fmt="o",ms=3,capsize=1,lw=.45,label="data")
    ax_zoom.plot(tz,prediction_from_row(tz,selected),lw=2.2,label="BIC #1")
    if _row_id(aicc_best)!=_row_id(selected):
        ax_zoom.plot(tz,prediction_from_row(tz,aicc_best),ls="-.",lw=1.25,label="AICc #1")
    if _row_id(raw_best)!=_row_id(selected):
        ax_zoom.plot(tz,prediction_from_row(tz,raw_best),ls=":",lw=1.3,label="raw best")
    if _row_id(conservative) not in {_row_id(selected),_row_id(aicc_best),_row_id(raw_best)}:
        ax_zoom.plot(tz,prediction_from_row(tz,conservative),ls="--",lw=1.2,label="conservative")
    ax_zoom.axvline(center,ls=":",lw=.8,alpha=.7)
    ax_zoom.set(xlabel="Total evolution time (us)",ylabel="Normalized signal",
                title="First-revival zoom")
    ax_zoom.grid(alpha=.2); ax_zoom.legend(fontsize=7)

    pred=prediction_from_row(t,selected)
    resid=(yy-pred)/ee
    ax_res.axhline(0,lw=1)
    for q,ls in ((2,"--"),(-2,"--"),(3,":"),(-3,":")):
        ax_res.axhline(q,ls=ls,lw=.7,alpha=.55)
    ax_res.plot(t,resid,"o",ms=3)
    ax_res.set(xlabel="Total evolution time (us)",
               ylabel="Normalized residual (data-fit)/STE",
               title=f"BIC #1 residuals | RMS={np.sqrt(np.mean(resid**2)):.2f}")
    ax_res.grid(alpha=.2)

    ranks=top_bic.copy()
    db=np.asarray(ranks.bic,float)-float(np.nanmin(d.bic))
    labels=[f"#{int(r.rank_global_bic)} N{int(r.model_order)} {r.site_key}"
            for _,r in ranks.iterrows()]
    ypos=np.arange(len(ranks))
    ax_rank.barh(ypos,db)
    ax_rank.set_yticks(ypos); ax_rank.set_yticklabels(labels,fontsize=7)
    ax_rank.invert_yaxis()
    ax_rank.set(xlabel="Delta BIC from best candidate",
                title=f"Top {len(ranks)} BIC-ranked candidates")
    for yi,(_,r) in zip(ypos,ranks.iterrows()):
        ax_rank.text(db[yi]+.2,yi,f"chi2r={r.red_chi2:.2f}",va="center",fontsize=6.5)
    ax_rank.grid(alpha=.15,axis="x")

    by_order=(d.sort_values(["bic","red_chi2"])
                .groupby("model_order",as_index=False).first()
                .sort_values("model_order"))
    minbic=float(by_order.bic.min()); dob=np.asarray(by_order.bic,float)-minbic
    x=np.arange(len(by_order))
    ax_order.bar(x,dob)
    ax_order.set_xticks(x); ax_order.set_xticklabels([f"N={int(o)}" for o in by_order.model_order])
    ax_order.set(xlabel="Model order",ylabel="Delta BIC from best tested order",
                 title=f"Best candidate per order | primary N={int(selected.model_order)}")
    for xi,(_,r) in zip(x,by_order.iterrows()):
        ax_order.text(xi,dob[xi]+.3,f"chi2r={r.red_chi2:.2f}\n{r.site_key}",
                      ha="center",va="bottom",fontsize=7)
    ax_order.grid(alpha=.15,axis="y")

    ax_model.axis("off")
    ax_model.text(.012,.96,_model_equation_text(),va="top",ha="left",
                  family="monospace",fontsize=6.0,linespacing=1.12,
                  transform=ax_model.transAxes)

    geom=selected
    geom_note="primary BIC model"
    if int(selected.model_order)==0:
        nz=d[d.model_order>0].sort_values(["bic","red_chi2"])
        if len(nz):
            geom=nz.iloc[0]; geom_note="best nonzero-C13 alternative"
    pts=[]
    ax3d.scatter([0],[0],[0],marker="*",s=180,label="NV",depthshade=False)
    for j in range(1,int(geom.model_order)+1):
        xyz=np.array([geom.get(f"c13_{j}_x_A",np.nan),
                      geom.get(f"c13_{j}_y_A",np.nan),
                      geom.get(f"c13_{j}_z_A",np.nan)],float)
        if not np.all(np.isfinite(xyz)): continue
        pts.append(xyz)
        ax3d.plot([0,xyz[0]],[0,xyz[1]],[0,xyz[2]],lw=1.0,alpha=.65)
        ax3d.scatter([xyz[0]],[xyz[1]],[xyz[2]],s=95,depthshade=False,
                     label=f"C{j}: site {int(geom[f'c13_{j}_site_id'])}")
        ax3d.text(xyz[0],xyz[1],xyz[2],
                  f" S{int(geom[f'c13_{j}_site_id'])}",fontsize=7)
    _set_equal_3d(ax3d,pts)
    ax3d.set(xlabel="x (A)",ylabel="y (A)",zlabel="z (A)",
             title=f"C13 positions: {geom_note}\norientation {geom.orientation}")
    ax3d.view_init(elev=23,azim=38)
    if int(geom.model_order)>0: ax3d.legend(fontsize=6,loc="best")

    ax_info.axis("off")
    ax_info.text(.00,.99,_background_text(selected,evidence,conservative,path),
                 va="top",ha="left",family="monospace",fontsize=6.45)
    for j,xpos in ((1,.43),(2,.63),(3,.82)):
        txt=_c13_text(selected,j)
        if txt:
            ax_info.text(xpos,.99,txt,va="top",ha="left",
                         family="monospace",fontsize=5.85)

    flags=_boundary_flags(selected)
    flagtxt=(" | "+", ".join(flags[:2])) if flags else ""
    fig.suptitle(
        f"{field_label} NV {nv} | PRIMARY global-BIC N={int(selected.model_order)} {selected.site_key} | "
        f"chi2r={selected.red_chi2:.3f} | order evidence={evidence['model_order_evidence']}{flagtxt}",
        fontsize=13.5
    )
    fig.subplots_adjust(left=.05,right=.98,bottom=.04,top=.94,hspace=.34,wspace=.28)
    pdf.savefig(fig)
    plt.close(fig)

def parse_nv_list(s):
    if s is None or not str(s).strip(): return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nv",type=str,default=None)
    ap.add_argument("--max-spins",type=int,default=2,choices=(1,2,3))
    ap.add_argument("--quick",action="store_true")
    ap.add_argument("--workers",type=int,default=DEFAULT_N_JOBS)
    ap.add_argument("--sites-per-orientation",type=int,default=FULL_SITES_PER_ORIENTATION,
                    choices=range(3,31),metavar="K",
                    help="Full-mode exhaustive candidate pool size per orientation (3-30; default 6).")
    ap.add_argument("--no-show",action="store_true")
    args=ap.parse_args()
    pool_path,checkpoint,top_path,prefix=discover_paths()
    print("="*100); print("49 G OLD-PROTOCOL + MULTI-13C RANKED FIT"); print("="*100)
    print("legacy per-site pool:",pool_path); print("checkpoint:",checkpoint)
    t,y,e=load_data(checkpoint)
    usecols=["nv_index","status","site_id","orientation","kappa","distance_A",
             "f0_kHz","f1_kHz","red_chi2","aicc","score_primary","score_amp_tie","popt_json"]
    attempts=pd.read_csv(pool_path,usecols=lambda c:c in usecols)
    req=parse_nv_list(args.nv); nvs=list(range(y.shape[0])) if req is None else [i for i in req if 0<=i<y.shape[0]]
    print(f"NVs={len(nvs)} max_spins={args.max_spins} quick={args.quick} workers={args.workers}")
    print(f"analysis={ANALYSIS_VERSION} | sites/orientation={args.sites_per_orientation} | primary=global min BIC | conservative DeltaBIC>={ORDER_ACCEPT_DELTA_BIC:g}")
    print(f"T2 ceiling={t2_upper_us(t):.2f} us (min of {T2_MAX_US_HARD:g} us hard cap and {T2_WINDOW_MULTIPLIER:g}x data window)")
    def task(i): return fit_one_nv(i,t,y[i],e[i],attempts,args.max_spins,args.quick,args.sites_per_orientation)
    with threadpool_limits(limits=BLAS_THREADS_PER_WORKER):
        results=Parallel(n_jobs=max(1,args.workers),backend="loky",batch_size=1,verbose=5)(delayed(task)(i) for i in nvs)
    rows=[r for block in results for r in block]
    cdf=pd.DataFrame(rows).sort_values(["nv_index","rank_global_bic"])
    catalog_lookup=load_catalog_lookup()
    cdf=enrich_candidates_with_catalog(cdf,catalog_lookup)
    outdir=Path(OUTPUT_DIR) if OUTPUT_DIR else prefix.parent; outdir.mkdir(parents=True,exist_ok=True)
    subset="" if req is None else "_subset_"+"-".join(map(str,nvs)); quick="_quick" if args.quick else ""
    maxc=f"_maxC{args.max_spins}"
    pooltag="" if args.quick else f"_topK{args.sites_per_orientation}"
    base=outdir/(prefix.name+"_oldproto_multic13_ranked_"+ANALYSIS_VERSION+maxc+pooltag+subset+quick)
    cand=Path(str(base)+"_candidate_fits.csv"); cdf.to_csv(cand,index=False)
    best_aicc=cdf[cdf.rank_global_aicc==1].sort_values("nv_index")
    best_red=cdf[cdf.rank_global_redchi==1].sort_values("nv_index")

    # Primary selection is the global BIC minimum.  Evidence strength is
    # reported separately rather than used as another hard penalty.
    evidence_rows=[]
    threshold_rows=[]
    for nv,g in cdf.groupby("nv_index"):
        selected,raw,conservative,path,ev=select_bic_with_evidence(g)
        row=selected.copy()
        for k,v in ev.items(): row[k]=v
        row["selection_rule"]="global_minimum_BIC"
        row["selection_path_json"]=json.dumps(path)
        evidence_rows.append(row)

        crow=conservative.copy()
        crow["selection_delta_bic_threshold"]=ORDER_ACCEPT_DELTA_BIC
        crow["selection_path_json"]=json.dumps(path)
        threshold_rows.append(crow)

    best_bic=pd.DataFrame(evidence_rows).sort_values("nv_index")
    best_delta10=pd.DataFrame(threshold_rows).sort_values("nv_index")
    best_bic.to_csv(Path(str(base)+"_best_by_bic.csv"),index=False)
    best_bic.to_csv(Path(str(base)+"_best_by_bic_evidence.csv"),index=False)
    best_aicc.to_csv(Path(str(base)+"_best_by_aicc.csv"),index=False)
    best_red.to_csv(Path(str(base)+"_best_by_redchi.csv"),index=False)
    best_delta10.to_csv(Path(str(base)+"_best_by_bic_delta10.csv"),index=False)

    with PdfPages(Path(str(base)+"_dashboard.pdf")) as pdf:
        plot_global_summary(pdf,cdf,best_delta10,args.max_spins)
        for nv in nvs: plot_nv(pdf,nv,t,y,e,cdf)
    print("="*100); print("COMPLETE")
    print("best-BIC orders:",best_bic.model_order.value_counts().sort_index().to_dict())
    print("deltaBIC>=10 orders:",best_delta10.model_order.value_counts().sort_index().to_dict())
    print(f"median best-BIC chi2r={best_bic.red_chi2.median():.3f}")
    print(f"median raw-best chi2r={best_red.red_chi2.median():.3f}")
    print("candidates:",cand); print("dashboard:",Path(str(base)+"_dashboard.pdf")); print("="*100)
    if not args.no_show: plt.show(block=True)
    return dict(candidates=cdf,best_by_bic=best_bic,best_by_bic_delta10=best_delta10,best_by_redchi=best_red)

if __name__=="__main__":
    main()
