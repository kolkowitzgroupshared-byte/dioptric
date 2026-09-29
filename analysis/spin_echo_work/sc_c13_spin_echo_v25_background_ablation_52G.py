"""V25 background-model ablation for irreducibly bad V24 52G NVs.

Keeps each V24 winning discrete 13C assignment fixed and compares:
  reduced: beta=2, taper=0          (V22/V24 production model)
  beta_free: beta free, taper=0
  taper_free: beta=2, taper free
  full_bg: beta and taper both free

Purpose: determine whether poor reduced-chi2 is caused by the reduced
background rather than missing 13C lattice candidates.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

V24=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v24_targeted_heavy_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v25_background_ablation_52G\2026_09")
MODELS={
    "reduced": {5:2.0,6:0.0},
    "beta_free": {6:0.0},
    "taper_free": {5:2.0},
    "full_bg": {},
}
def fit_model(base,t,y,e,row,sites,fixed,max_nfev=5000):
    v6.VISIBILITY_SCALE_MAX=30.0
    seed=np.asarray(json.loads(str(row.theta_json)),float)
    n=int(row.model_order)
    lb,ub=v6.v6_theta_bounds(base,n,float(row.baseline),float(base.t2_upper_us(t)))
    ee=base.safe_err(e)
    free=[i for i in range(len(seed)) if i not in fixed]

    def expand(x,template):
        th=template.copy()
        th[free]=x
        for i,val in fixed.items(): th[int(i)]=float(val)
        return th
    def resid(x,template):
        th=expand(x,template)
        return (np.asarray(y,float)-v6.v6_model(base,t,th,sites))/ee

    seeds=[]
    beta_vals=[float(seed[5])] if 5 in fixed else [1.0,1.5,2.0,3.0]
    taper_vals=[float(seed[6])] if 6 in fixed else [0.0,0.5,1.0,2.0]
    for beta in beta_vals:
        for taper in taper_vals:
            s=seed.copy(); s[5]=beta; s[6]=taper
            for i,val in fixed.items(): s[int(i)]=float(val)
            seeds.append(np.clip(s,lb+1e-8,ub-1e-8))

    trials=[]
    for s in seeds:
        x0=s[free]
        try:
            rr=least_squares(lambda x:resid(x,s),x0,bounds=(lb[free],ub[free]),
                             loss="soft_l1",f_scale=1.0,max_nfev=max_nfev,
                             x_scale="jac")
            th=expand(rr.x,s)
            trials.append((float(np.sum(resid(rr.x,s)**2)),th))
        except Exception:
            pass
    if not trials:
        raise RuntimeError("all robust seeds failed")
    trials.sort(key=lambda z:z[0])
    best=None
    for _,s in trials[:4]:
        x0=s[free]
        try:
            rr=least_squares(lambda x:resid(x,s),x0,bounds=(lb[free],ub[free]),
                             loss="linear",max_nfev=max_nfev,
                             ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac")
            th=expand(rr.x,s)
            pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            q=dict(theta=th,**st)
            if best is None or q["chi2"]<best["chi2"]: best=q
        except Exception:
            pass
    if best is None:
        raise RuntimeError("all final fits failed")
    return best


def one_nv(nv,row,t,y,e):
    base=v6.load_backend("52G")
    sites=v14.sites_from_record(row)
    out={"nv_index":int(nv),"model_order":int(row.model_order),"site_key":str(row.site_key)}
    for name,fixed in MODELS.items():
        q=fit_model(base,t,y,e,row,sites,fixed)
        th=np.asarray(q["theta"],float)
        out.update({
            f"{name}_chi2":q["chi2"],
            f"{name}_redchi2":q["red_chi2"],
            f"{name}_bic":q["bic"],
            f"{name}_aicc":q["aicc"],
            f"{name}_npar":q["npar"],
            f"{name}_beta":th[5],
            f"{name}_taper":th[6],
            f"{name}_scale":th[9] if int(row.model_order)>0 else np.nan,
            f"{name}_theta_json":json.dumps([float(x) for x in th]),
        })
    out["dBIC_beta_free_vs_reduced"]=out["beta_free_bic"]-out["reduced_bic"]
    out["dBIC_taper_free_vs_reduced"]=out["taper_free_bic"]-out["reduced_bic"]
    out["dBIC_full_bg_vs_reduced"]=out["full_bg_bic"]-out["reduced_bic"]
    return out
def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    qc=pd.read_csv(V24/"v22_qc_summary.csv").fillna({"qc_flags":""})
    win=pd.read_csv(V24/"v22_winners.csv")
    bad=qc[qc.qc_flags.astype(str).map(lambda s:"high_redchi2" in s.split(";"))]
    ids=sorted(bad.nv_index.astype(int).tolist())
    rows=win[win.nv_index.isin(ids)].set_index("nv_index")

    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck)

    print(f"V25 background ablation: {len(ids)} bad-fit NVs")
    res=Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
        delayed(one_nv)(nv,rows.loc[nv],t,Y[nv],E[nv]) for nv in ids
    )
    d=pd.DataFrame(res).sort_values("nv_index")
    d.to_csv(OUT/"v25_background_ablation.csv",index=False)

    fav_beta=int((d.dBIC_beta_free_vs_reduced<=-6).sum())
    fav_taper=int((d.dBIC_taper_free_vs_reduced<=-6).sum())
    fav_full=int((d.dBIC_full_bg_vs_reduced<=-6).sum())
    print("\nV25 COMPLETE")
    print("beta-free strongly favored:",fav_beta)
    print("taper-free strongly favored:",fav_taper)
    print("full background strongly favored:",fav_full)
    print("median reduced redchi2:",float(d.reduced_redchi2.median()))
    print("median full-bg redchi2:",float(d.full_bg_redchi2.median()))
    print("output:",OUT)
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=10)
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
