"""V20b: NV37 sideband weights with the V19 production background frozen.

This isolates the equal-f+/f- amplitude assumption from background/contrast
degeneracy. Sites and frequencies stay fixed at (31,24,78).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v20_nv37_sideband_weight_diagnostic as v20

V19_FILE=v20.V19_FILE
DEFAULT_OUT=Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v20b_nv37_sideband_frozen_bg\2026_09"
)
RANDOM_STARTS=30
ROBUST_NFEV=18000
FINAL_NFEV=40000
def fit_kind(base,t,y,e,sites,kind,bg,seed_scale,seed_phases):
    nsite=len(sites); ee=base.safe_err(e)
    al,au=v20.amp_bounds(kind,nsite)
    lb=np.r_[al,np.full(2*nsite,-np.pi)]
    ub=np.r_[au,np.full(2*nsite, np.pi)]

    def seed_amp():
        if kind=="M0_equal": return np.array([seed_scale])
        if kind=="M1_global_imbalance": return np.array([seed_scale,0.0])
        if kind=="M2_site_imbalance": return np.r_[seed_scale,np.zeros(nsite)]
        if kind=="Msite_free_equal": return np.full(nsite,seed_scale)
        if kind=="M3_free_lines": return np.full(2*nsite,seed_scale)
        raise ValueError(kind)

    na=len(al)
    template=np.r_[seed_amp(),np.asarray(seed_phases,float).ravel()]

    def splitx(x):
        amp=np.asarray(x[:na],float)
        ph=np.asarray(x[na:],float).reshape(nsite,2)
        return amp,ph

    def predx(x):
        amp,ph=splitx(x)
        qs=v20.model_parts(kind,amp,nsite)
        return v20.model(base,t,bg,sites,ph,qs)

    def resid(x):
        return (np.asarray(y,float)-predx(x))/ee
    starts=[np.clip(template,lb+1e-8,ub-1e-8)]
    rng=np.random.default_rng(20260926+len(kind))
    for i in range(RANDOM_STARTS):
        q=template.copy()
        if kind=="M0_equal":
            q[0]=rng.uniform(2,25)
        elif kind=="M1_global_imbalance":
            q[0]=rng.uniform(2,25); q[1]=rng.uniform(-1.2,1.2)
        elif kind=="M2_site_imbalance":
            q[0]=rng.uniform(2,25); q[1:1+nsite]=rng.uniform(-1.2,1.2,nsite)
        elif kind=="Msite_free_equal":
            q[:nsite]=rng.uniform(0.2,30,nsite)
        elif kind=="M3_free_lines":
            q[:2*nsite]=rng.uniform(0.2,30,2*nsite)
        q[-2*nsite:]=rng.uniform(-np.pi,np.pi,2*nsite)
        starts.append(np.clip(q,lb+1e-8,ub-1e-8))

    candidates=[]; robust=[]
    for x0 in starts:
        try:
            pp=predx(x0); st=v20.stats(y,ee,pp,len(x0))
            candidates.append(dict(x=x0,pred=pp,source="seed",**st))
        except Exception: pass
        try:
            rr=least_squares(
                resid,x0,bounds=(lb,ub),loss="soft_l1",f_scale=1.0,
                max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((float(np.sum(resid(rr.x)**2)),rr.x))
        except Exception: pass
    robust.sort(key=lambda z:z[0])
    for _,x0 in robust[:10]:
        try:
            rr=least_squares(
                resid,x0,bounds=(lb,ub),loss="linear",
                max_nfev=FINAL_NFEV,ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale="jac")
            pp=predx(rr.x); st=v20.stats(y,ee,pp,len(rr.x))
            candidates.append(dict(x=rr.x,pred=pp,source="polish",**st))
        except Exception: pass
    if not candidates:
        raise RuntimeError(f"{kind}: all fits failed")
    return min(candidates,key=lambda q:(q["chi2"],q["bic"]))


def record(kind,fit,sites,bg):
    n=len(sites); al,_=v20.amp_bounds(kind,n); na=len(al)
    amp=fit["x"][:na]; ph=fit["x"][na:].reshape(n,2)
    qs=v20.model_parts(kind,amp,n)
    rec=dict(
        model=kind,chi2=fit["chi2"],red_chi2=fit["red_chi2"],
        aicc=fit["aicc"],bic=fit["bic"],npar=fit["npar"],source=fit["source"],
        frozen_baseline=float(bg[0]),frozen_contrast=float(bg[1]),
        frozen_revival_time_us=float(bg[2]),frozen_width0_us=float(bg[3]),
        frozen_T2_us=float(1000*bg[4]),frozen_width_slope=float(bg[7]),
        frozen_revival_chirp=float(bg[8]),
    )
    for j,s in enumerate(sites):
        sid=int(s["site_id"]); qp=float(qs[2*j]); qm=float(qs[2*j+1])
        rec[f"site{sid}_qplus"]=qp; rec[f"site{sid}_qminus"]=qm
        rec[f"site{sid}_ratio_plus_minus"]=qp/qm if qm>1e-12 else np.inf
        rec[f"site{sid}_phi_plus"]=float(ph[j,0])
        rec[f"site{sid}_phi_minus"]=float(ph[j,1])
    return rec


def run(args):
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)

    d=pd.read_csv(V19_FILE)
    q=d[(d.nv_index==v20.NV)&(d.hypothesis=="H0")].iloc[0]
    th=np.asarray(json.loads(q.theta_json),float)
    bg=th[:9]; seed_scale=float(q.shared_scale); seed_phases=th[10:].reshape(3,2)

    v14root=Path(
        r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v14_beta2_taper0_rerank\2026_09"
    )
    vf=max(v14root.glob("*52G*v14_beta2_taper0_candidate_fits.csv.gz"),key=lambda p:p.stat().st_mtime)
    vd=pd.read_csv(vf)
    row=vd[(vd.rank_global_bic==1)&(vd.nv_index==v20.NV)].iloc[0]
    sites=v20.load_catalog_sites(base,base.parse_orientation(row.orientation))

    kinds=["M0_equal","M1_global_imbalance","M2_site_imbalance","Msite_free_equal","M3_free_lines"]
    records=[]; preds={}
    with threadpool_limits(limits=1):
        for kind in kinds:
            fit=fit_kind(base,t,Y[v20.NV],E[v20.NV],sites,kind,bg,seed_scale,seed_phases)
            rec=record(kind,fit,sites,bg); records.append(rec); preds[kind]=fit["pred"]
            print(f"{kind}: chi2={rec['chi2']:.3f}, BIC={rec['bic']:.3f}, redchi2={rec['red_chi2']:.3f}")
    tab=pd.DataFrame(records)
    b0=float(tab.loc[tab.model=="M0_equal","bic"].iloc[0])
    c0=float(tab.loc[tab.model=="M0_equal","chi2"].iloc[0])
    tab["delta_bic_vs_M0"]=tab.bic-b0
    tab["chi2_gain_vs_M0"]=c0-tab.chi2
    tab.to_csv(outdir/"v20b_nv37_frozen_bg_models.csv",index=False)

    lines=[]
    for rec in records:
        for sid in v20.SITE_IDS:
            lines.append(dict(
                model=rec["model"],site_id=sid,
                qplus=rec[f"site{sid}_qplus"],qminus=rec[f"site{sid}_qminus"],
                ratio_plus_minus=rec[f"site{sid}_ratio_plus_minus"],
            ))
    pd.DataFrame(lines).to_csv(outdir/"v20b_nv37_frozen_bg_line_weights.csv",index=False)

    with PdfPages(outdir/"v20b_nv37_frozen_bg_diagnostic.pdf") as pdf:
        v20.plot_results(pdf,t,Y[v20.NV],base.safe_err(E[v20.NV]),records,preds)

    print("\nFROZEN-BACKGROUND COMPARISON")
    print(tab[["model","npar","chi2","red_chi2","bic","delta_bic_vs_M0","chi2_gain_vs_M0"]].to_string(index=False))
    print("\nLINE WEIGHTS")
    print(pd.DataFrame(lines).to_string(index=False))
    print("\nSaved:",outdir)
    return tab


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=ap.parse_args(); run(args)


if __name__=="__main__":
    main()
