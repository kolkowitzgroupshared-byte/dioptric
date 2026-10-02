"""V28b profile the shared scale under the V28 cross-term model for NV1/NV87."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14
import sc_c13_spin_echo_v28_nonlinear_eSEEM_52G as v28

V24=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v24_targeted_heavy_52G\2026_09")
V28=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v28_nonlinear_eSEEM_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v28b_cross_scale_profile_52G\2026_09")
GRID=np.arange(0.,30.0001,1.)
FIXED_BASE={5:2.,6:0.}
def fit_fixed_scale(base,t,y,e,sites,seed,scale):
    n=len(sites); v6.VISIBILITY_SCALE_MAX=30.
    lb,ub=v6.v6_theta_bounds(base,n,float(seed[0]),float(base.t2_upper_us(t)))
    fixed={5:2.,6:0.,9:float(scale)}
    free=[i for i in range(len(seed)) if i not in fixed]
    ee=base.safe_err(e)
    s=np.clip(np.asarray(seed,float),lb+1e-8,ub-1e-8)
    for i,v in fixed.items(): s[i]=v
    def expand(x):
        th=s.copy(); th[free]=x
        for i,v in fixed.items(): th[i]=v
        return th
    def resid(x):
        return (np.asarray(y,float)-v28.cross_model(base,t,expand(x),sites))/ee
    try:
        rr=least_squares(resid,s[free],bounds=(lb[free],ub[free]),loss="soft_l1",
                         f_scale=1,max_nfev=4000,x_scale="jac")
        ff=least_squares(resid,rr.x,bounds=(lb[free],ub[free]),loss="linear",
                         max_nfev=4000,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale="jac")
        th=expand(ff.x); pred=v28.cross_model(base,t,th,sites)
        st=base.calc_stats(y,ee,pred,len(free))
        return th,st
    except Exception:
        return None,None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    targets=pd.read_csv(V28/"v28_nonlinear_eSEEM_summary.csv")
    targets=targets[targets.cross_strongly_favored.astype(bool)]
    winners=pd.read_csv(V24/"v22_winners.csv").set_index("nv_index")
    base=v6.load_backend("52G"); _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck)
    rows=[]
    for q in targets.itertuples():
        nv=int(q.nv_index); win=winners.loc[nv]
        sites=v14.sites_from_record(win)
        additive_seed=np.asarray(json.loads(str(win.theta_json)),float)
        # Reconstruct the good free cross-term basin first; profiling from the
        # additive basin can get trapped in a completely different minimum.
        freefit=v28.fit_cross(base,t,Y[nv],E[nv],sites,additive_seed)
        if freefit is None:
            raise RuntimeError(f"NV{nv}: could not reconstruct free cross fit")
        seed=np.asarray(freefit["theta"],float)
        s0=float(seed[9])
        start=int(np.argmin(np.abs(GRID-s0)))
        order=[start]
        for dist in range(1,len(GRID)):
            if start-dist>=0: order.append(start-dist)
            if start+dist<len(GRID): order.append(start+dist)
            if len(set(order))==len(GRID): break
        fitted={}
        for gi in order:
            s=float(GRID[gi])
            if fitted:
                nearest=min(fitted,key=lambda j:abs(GRID[j]-s))
                use_seed=fitted[nearest][0]
            else:
                use_seed=seed
            th,st=fit_fixed_scale(base,t,Y[nv],E[nv],sites,use_seed,s)
            if th is None: continue
            fitted[gi]=(th,st)
            rows.append(dict(nv_index=nv,scale=s,chi2=st["chi2"],
                             redchi2=st["red_chi2"],
                             theta_json=json.dumps([float(x) for x in th])))
    d=pd.DataFrame(rows)
    d["chi2_min"]=d.groupby("nv_index").chi2.transform("min")
    d["delta_chi2"]=d.chi2-d.chi2_min
    d.to_csv(OUT/"v28b_cross_scale_profiles.csv",index=False)
    sums=[]
    for nv,g in d.groupby("nv_index"):
        ok=g[g.delta_chi2<=3.84]
        sums.append(dict(
            nv_index=int(nv),
            profile_best_scale=float(g.loc[g.chi2.idxmin(),"scale"]),
            scale95_low=float(ok.scale.min()),
            scale95_high=float(ok.scale.max()),
            s_gt10_required_95=bool(ok.scale.min()>10),
            delta_chi2_best_s_le10=float(g[g.scale<=10].delta_chi2.min()),
        ))
    s=pd.DataFrame(sums)
    s.to_csv(OUT/"v28b_cross_scale_summary.csv",index=False)
    print("V28b COMPLETE")
    print(s.to_string(index=False))
    print("output:",OUT)


if __name__=="__main__":
    main()
