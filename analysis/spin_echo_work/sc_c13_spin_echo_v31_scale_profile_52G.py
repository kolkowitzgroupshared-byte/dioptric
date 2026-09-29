"""V31 profile-likelihood diagnostic for shared 13C visibility scale.

Uses the conservative V30 state. For every NV with V30 scale > 10:
* hold the discrete site set/background model fixed,
* scan fixed shared scale s over a broad grid,
* reoptimize all other continuous parameters,
* report whether s<=10 lies inside the profile-likelihood confidence region.

This diagnoses scale/contrast identifiability without imposing an arbitrary
hard s<=10 production bound.
"""
from __future__ import annotations
import ast, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v27_residual_driven_add_site_52G as v27
import sc_c13_spin_echo_v28_parsimony_rerank_52G as v28

ROOT30=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v30_consolidated_state_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v31_scale_profile_52G\2026_09")
S_GRID=np.array([0.25,0.5,0.75,1,1.5,2,3,4,5,6,8,10,12,15,20,25,30],float)
def sites_from_ids(catalog,ori,site_key):
    ids=tuple(int(x) for x in ast.literal_eval(str(site_key)))
    sites=[]
    for sid in ids:
        sites.append(v28.get_catalog_site(catalog,ori,sid))
    return sites


def background_fixed(bg):
    return v27.fixed_for_bg(bg)


def fit_fixed_scale(base,t,y,e,sites,theta_seed,bg,scale,max_nfev=3500):
    n=len(sites)
    if n==0:
        return None
    v6.VISIBILITY_SCALE_MAX=30.0
    lb,ub=v6.v6_theta_bounds(
        base,n,float(theta_seed[0]),float(base.t2_upper_us(t))
    )
    fixed=dict(background_fixed(bg))
    fixed[9]=float(scale)
    seed=np.asarray(theta_seed,float).copy()
    seed[9]=float(scale)
    seed=np.clip(seed,lb+1e-8,ub-1e-8)
    free=[i for i in range(len(seed)) if i not in fixed]
    ee=base.safe_err(e)

    def expand(x):
        th=seed.copy(); th[free]=x
        for i,val in fixed.items(): th[int(i)]=float(val)
        return th
    def resid(x):
        return (np.asarray(y,float)-v6.v6_model(base,t,expand(x),sites))/ee

    try:
        rr=least_squares(
            resid,seed[free],bounds=(lb[free],ub[free]),
            loss="soft_l1",f_scale=1.0,max_nfev=max_nfev,x_scale="jac"
        )
        s2=expand(rr.x)
        free2=free
        def expand2(x):
            th=s2.copy(); th[free2]=x
            for i,val in fixed.items(): th[int(i)]=float(val)
            return th
        rr2=least_squares(
            lambda x:(np.asarray(y)-v6.v6_model(base,t,expand2(x),sites))/ee,
            s2[free2],bounds=(lb[free2],ub[free2]),loss="linear",
            max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac"
        )
        th=expand2(rr2.x); pred=v6.v6_model(base,t,th,sites)
        st=base.calc_stats(y,ee,pred,len(free2))
        return th,st
    except Exception:
        return None
def one_nv(nv,row,t,y,e,catalog,omap):
    base=v6.load_backend("52G")
    ori=tuple(omap[int(nv)])
    sites=sites_from_ids(catalog,ori,row.site_key)
    theta=np.asarray(json.loads(str(row.theta_json)),float)
    bg=str(row.background_model)
    results=[]

    with threadpool_limits(limits=1):
        # Warm start nearest to the current best scale first, then sweep.
        order=np.argsort(np.abs(S_GRID-float(row.visibility_scale)))
        last=theta.copy()
        temp={}
        for ind in order:
            s=float(S_GRID[ind])
            q=fit_fixed_scale(base,t,y,e,sites,last,bg,s)
            if q is None:
                continue
            th,st=q; last=th
            maxk=max(float(x["kappa"]) for x in sites)
            temp[s]=dict(
                nv_index=int(nv),scale=s,chi2=float(st["chi2"]),
                red_chi2=float(st["red_chi2"]),bic_fixed=float(st["bic"]),
                contrast=float(th[1]),
                max_line_to_contrast=float(s*maxk/4.0),
                theta_json=json.dumps([float(x) for x in th]),
            )
        results=[temp[s] for s in sorted(temp)]

    if not results:
        return [],dict(nv_index=int(nv),status="failed")
    d=pd.DataFrame(results)
    best=d.loc[d.chi2.idxmin()]
    dchi=d.chi2-float(best.chi2)
    d["delta_chi2"]=dchi

    within95=d[d.delta_chi2<=3.841459]
    within1=d[d.delta_chi2<=1.0]
    at10=d.iloc[np.argmin(np.abs(d.scale-10.0))]
    support10=float(at10.delta_chi2)
    identifiable_gt10=bool(support10>3.841459)
    bound_min=bool(float(best.scale)>=30.0-1e-9)
    summary=dict(
        nv_index=int(nv),status="ok",site_key=str(row.site_key),
        model_order=int(row.model_order),background_model=bg,
        v30_scale=float(row.visibility_scale),
        profile_best_scale=float(best.scale),
        profile_best_chi2=float(best.chi2),
        delta_chi2_at_s10=support10,
        s10_within_95pct=bool(support10<=3.841459),
        scale_gt10_identifiable_95pct=identifiable_gt10,
        profile_min_at_upper_bound=bound_min,
        scale_95_low=float(within95.scale.min()),
        scale_95_high=float(within95.scale.max()),
        scale_1sigma_low=float(within1.scale.min()),
        scale_1sigma_high=float(within1.scale.max()),
        contrast_at_profile_best=float(best.contrast),
        max_line_to_contrast_at_best=float(best.max_line_to_contrast),
    )
    return d.to_dict("records"),summary
def run(workers=10):
    OUT.mkdir(parents=True,exist_ok=True)
    state=pd.read_csv(ROOT30/"v30_consolidated_52G_state.csv")
    sel=state[np.isfinite(state.visibility_scale)&(state.visibility_scale>10)].copy()
    ids=sorted(sel.nv_index.astype(int).tolist())

    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    catalog=v6.load_catalog(base)
    omap=v6.load_assigned_orientations("52G",Y.shape[0])
    rows={int(r.nv_index):r for r in sel.itertuples()}

    print(f"V31 scale profiles: {len(ids)} V30 high-scale NVs")
    res=Parallel(n_jobs=workers,backend="loky",verbose=10)(
        delayed(one_nv)(nv,rows[nv],t,Y[nv],E[nv],catalog,omap)
        for nv in ids
    )
    prof=[]; sums=[]
    for rr,ss in res:
        prof.extend(rr); sums.append(ss)
    p=pd.DataFrame(prof)
    s=pd.DataFrame(sums).sort_values("nv_index")
    p.to_csv(OUT/"v31_scale_profiles.csv.gz",index=False,compression="gzip")
    s.to_csv(OUT/"v31_scale_profile_summary.csv",index=False)

    ok=s[s.status=="ok"]
    print("\nV31 COMPLETE")
    print("s=10 within 95% profile region:",
          int(ok.s10_within_95pct.sum()),"/",len(ok))
    print("large s identifiable at 95%:",
          int(ok.scale_gt10_identifiable_95pct.sum()))
    print("profile minimum at s=30 boundary:",
          int(ok.profile_min_at_upper_bound.sum()))
    print("output:",OUT)
    print(ok[[
        "nv_index","site_key","v30_scale","profile_best_scale",
        "delta_chi2_at_s10","s10_within_95pct","scale_95_low","scale_95_high",
        "max_line_to_contrast_at_best"
    ]].sort_values("delta_chi2_at_s10",ascending=False).to_string(index=False))


if __name__=="__main__":
    run()
