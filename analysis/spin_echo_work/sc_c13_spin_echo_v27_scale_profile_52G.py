"""V27 profile-likelihood diagnostic for high-scale QNami 52G fits.

For every V24 winner flagged high_scale_gt10, scale_bound, or
line_amp_gt2contrast, hold its discrete C13 assignment fixed and profile the
shared visibility scale s_NV from 0..30 while reoptimizing all remaining
continuous parameters under the production beta=2, taper=0 background.

The scale is judged by profile Delta-chi2 (one parameter), not by raw optimizer
location.  A 95% profile interval uses Delta-chi2 <= 3.84.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

V24=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v24_targeted_heavy_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v27_scale_profile_52G\2026_09")
GRID=np.arange(0.0,30.0001,1.0)
def fit_at_scale(base,t,y,e,sites,seed,scale,max_nfev=3000):
    n=len(sites)
    if n==0:
        raise ValueError("scale profile requires at least one site")
    v6.VISIBILITY_SCALE_MAX=30.0
    lb,ub=v6.v6_theta_bounds(base,n,float(seed[0]),float(base.t2_upper_us(t)))
    fixed={5:2.0,6:0.0,9:float(scale)}
    free=[i for i in range(len(seed)) if i not in fixed]
    ee=base.safe_err(e)
    s=np.clip(np.asarray(seed,float),lb+1e-8,ub-1e-8)
    for i,val in fixed.items(): s[i]=val

    def expand(x):
        th=s.copy(); th[free]=x
        for i,val in fixed.items(): th[i]=val
        return th
    def resid(x):
        return (np.asarray(y,float)-v6.v6_model(base,t,expand(x),sites))/ee

    best=None
    # Robust step followed by ordinary least squares.
    try:
        rr=least_squares(resid,s[free],bounds=(lb[free],ub[free]),
                         loss="soft_l1",f_scale=1,max_nfev=max_nfev,x_scale="jac")
        rr2=least_squares(resid,rr.x,bounds=(lb[free],ub[free]),
                          loss="linear",max_nfev=max_nfev,
                          ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale="jac")
        th=expand(rr2.x); pred=v6.v6_model(base,t,th,sites)
        st=base.calc_stats(y,ee,pred,len(free))
        best=dict(theta=th,**st)
    except Exception:
        pass
    return best


def profile_nv(nv,row,t,y,e):
    base=v6.load_backend("52G")
    sites=v14.sites_from_record(row)
    seed0=np.asarray(json.loads(str(row.theta_json)),float)
    s0=float(seed0[9])
    values={}
    # Center-out continuation improves stability and speed.
    start=int(np.argmin(np.abs(GRID-s0)))
    order=[start]
    for d in range(1,len(GRID)):
        if start-d>=0: order.append(start-d)
        if start+d<len(GRID): order.append(start+d)
        if len(set(order))==len(GRID): break
    fitted={}
    with threadpool_limits(limits=1):
        for gi in order:
            s=float(GRID[gi])
            if fitted:
                nearest=min(fitted,key=lambda j:abs(GRID[j]-s))
                seed=fitted[nearest]["theta"]
            else:
                seed=seed0
            fit=fit_at_scale(base,t,y,e,sites,seed,s)
            if fit is None:
                continue
            fitted[gi]=fit
            values[s]=fit
    rows=[]
    for s in GRID:
        q=values.get(float(s))
        if q is None:
            continue
        rows.append(dict(nv_index=int(nv),scale=float(s),chi2=float(q["chi2"]),
                         redchi2=float(q["red_chi2"]),bic=float(q["bic"]),
                         theta_json=json.dumps([float(x) for x in q["theta"]])))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    qc=pd.read_csv(V24/"v22_qc_summary.csv").fillna({"qc_flags":""})
    winners=pd.read_csv(V24/"v22_winners.csv")
    def high(s):
        f=set(str(s).split(";"))
        return bool({"high_scale_gt10","scale_bound","line_amp_gt2contrast"} & f)
    ids=qc[qc.qc_flags.map(high)].nv_index.astype(int).tolist()
    w=winners[winners.nv_index.isin(ids)].set_index("nv_index")
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)

    print(f"V27 scale profiles: {len(ids)} NVs")
    parts=Parallel(n_jobs=10,backend="loky",verbose=10)(
        delayed(profile_nv)(nv,w.loc[nv],t,Y[nv],E[nv]) for nv in ids
    )
    prof=pd.DataFrame([r for part in parts for r in part])
    prof["chi2_min"]=prof.groupby("nv_index").chi2.transform("min")
    prof["delta_chi2"]=prof.chi2-prof.chi2_min
    prof.to_csv(OUT/"v27_scale_profiles.csv",index=False)
    summary=[]
    for nv,g in prof.groupby("nv_index"):
        g=g.sort_values("scale")
        win=w.loc[int(nv)]
        kmax=max([
            float(getattr(win,f"c13_{j}_kappa"))
            for j in (1,2,3)
            if f"c13_{j}_kappa" in win.index and pd.notna(getattr(win,f"c13_{j}_kappa"))
        ] or [0.0])
        ok=g[g.delta_chi2<=3.84]
        s_best=float(g.loc[g.chi2.idxmin(),"scale"])
        lo=float(ok.scale.min()) if len(ok) else np.nan
        hi=float(ok.scale.max()) if len(ok) else np.nan
        g10=g[g.scale<=10]
        d10=float(g10.delta_chi2.min()) if len(g10) else np.nan
        phys_limit=(4.0/kmax) if kmax>0 else np.inf
        gp=g[g.scale<=phys_limit]
        dphys=float(gp.delta_chi2.min()) if len(gp) else np.nan
        summary.append(dict(
            nv_index=int(nv),site_key=str(win.site_key),
            free_fit_scale=float(win.shared_scale) if "shared_scale" in win.index else float(win.visibility_scale),
            profile_best_scale=s_best,scale95_low=lo,scale95_high=hi,
            scale95_high_censored=bool(hi>=GRID.max()),
            delta_chi2_best_s_le10=d10,
            high_scale_required_95=bool(np.isfinite(lo) and lo>10),
            max_kappa=kmax,scale_for_line_over_contrast_1=phys_limit,
            delta_chi2_best_line_le_contrast=dphys,
            line_gt_contrast_required_95=bool(np.isfinite(dphys) and dphys>3.84),
        ))
    s=pd.DataFrame(summary).sort_values("nv_index")
    s.to_csv(OUT/"v27_scale_profile_summary.csv",index=False)

    with PdfPages(OUT/"v27_scale_profiles.pdf") as pdf:
        for q in s.itertuples():
            g=prof[prof.nv_index==q.nv_index].sort_values("scale")
            fig,ax=plt.subplots(figsize=(8.5,5.5))
            ax.plot(g.scale,g.delta_chi2,".-")
            ax.axhline(3.84,ls="--",lw=.8,label="95% profile threshold")
            ax.axvline(10,ls=":",lw=.8,label="s=10")
            if np.isfinite(q.scale_for_line_over_contrast_1):
                ax.axvline(q.scale_for_line_over_contrast_1,ls="-.",lw=.8,
                           label="max line/contrast = 1")
            ax.set(xlabel="fixed shared scale s_NV",ylabel="profile Delta chi2",
                   title=f"NV {q.nv_index} | sites {q.site_key}")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("\nV27 COMPLETE")
    print("profiled NVs:",len(s))
    print("s>10 required at 95%:",s[s.high_scale_required_95].nv_index.astype(int).tolist())
    print("count:",int(s.high_scale_required_95.sum()))
    print("line/contrast>1 required at 95%:",
          s[s.line_gt_contrast_required_95].nv_index.astype(int).tolist())
    print("count:",int(s.line_gt_contrast_required_95.sum()))
    print("output:",OUT)


if __name__=="__main__":
    main()
