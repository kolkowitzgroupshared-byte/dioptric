"""V28 nonlinear-ESEEM diagnostics for V27 genuinely high-scale QNami NVs.

Compares the V24 additive shared-scale model against:
  1) phase-preserving multiplicative cross-term model with identical parameter
     count and identical catalog frequencies/phases structure;
  2) exact single-spin Hahn product using catalog fI/omega_ms, with one shared
     scale and a common timing offset.

Only NVs whose V27 95% profile lower bound is >10 are tested.
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
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

V24=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v24_targeted_heavy_52G\2026_09")
V27=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v27_scale_profile_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v28_nonlinear_eSEEM_52G\2026_09")
FIXED={5:2.0,6:0.0}
def parse_orientation(x):
    if isinstance(x,(tuple,list,np.ndarray)):
        return tuple(int(v) for v in x)
    return tuple(int(v) for v in ast.literal_eval(str(x)))


def cross_model(base,t,theta,sites):
    th=np.asarray(theta,float); bg=th[:9]
    baseline,contrast,carrier=base.carrier_from_bg(t,bg)
    tt=np.asarray(t,float); L=np.ones_like(tt)
    if sites:
        scale=float(th[9]); j=10
        for s in sites:
            p0,p1=th[j:j+2]; j+=2
            f0=float(s["f0_kHz"])/1000; f1=float(s["f1_kHz"])/1000
            h=np.cos(2*np.pi*f0*tt+p0)+np.cos(2*np.pi*f1*tt+p1)
            u=scale*float(s["kappa"])*h/4.0
            L*=1.0-u
    return baseline-contrast*carrier*L


def fit_cross(base,t,y,e,sites,seed,max_nfev=8000):
    n=len(sites); v6.VISIBILITY_SCALE_MAX=30.0
    lb,ub=v6.v6_theta_bounds(base,n,float(seed[0]),float(base.t2_upper_us(t)))
    free=[i for i in range(len(seed)) if i not in FIXED]
    ee=base.safe_err(e)
    starts=[]
    for s0 in [float(seed[9]),5.0,10.0,15.0,20.0,30.0]:
        q=np.asarray(seed,float).copy(); q[5]=2; q[6]=0; q[9]=s0
        starts.append(np.clip(q,lb+1e-8,ub-1e-8))
    fits=[]
    for s in starts:
        def expand(x):
            th=s.copy(); th[free]=x; th[5]=2; th[6]=0
            return th
        def resid(x):
            return (np.asarray(y,float)-cross_model(base,t,expand(x),sites))/ee
        try:
            rr=least_squares(resid,s[free],bounds=(lb[free],ub[free]),loss="soft_l1",
                             f_scale=1,max_nfev=max_nfev,x_scale="jac")
            ff=least_squares(resid,rr.x,bounds=(lb[free],ub[free]),loss="linear",
                             max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10,
                             x_scale="jac")
            th=expand(ff.x); pred=cross_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            fits.append(dict(theta=th,pred=pred,**st))
        except Exception:
            pass
    return min(fits,key=lambda q:q["chi2"]) if fits else None
def catalog_exact_sites(cat,orientation,sites):
    ori=parse_orientation(orientation)
    cc=cat[cat.ori.map(tuple)==ori].set_index("site_id")
    out=[]
    for s in sites:
        q=cc.loc[int(s["site_id"])]
        out.append(dict(
            site_id=int(s["site_id"]),kappa=float(q.kappa),
            fI_MHz=float(q.fI_Hz)/1e6,fm_MHz=float(q.omega_ms_Hz)/1e6,
        ))
    return out


def exact_product_model(base,t,bg,sites,scale,dt):
    baseline,contrast,carrier=base.carrier_from_bg(t,bg)
    tt=np.asarray(t,float)+float(dt)
    L=np.ones_like(tt)
    for s in sites:
        q=2*float(s["kappa"])*(
            np.sin(.5*np.pi*float(s["fI_MHz"])*tt)**2
        )*(
            np.sin(.5*np.pi*float(s["fm_MHz"])*tt)**2
        )
        L*=1-float(scale)*q
    return baseline-contrast*carrier*L


def fit_exact(base,t,y,e,sites,bgseed,max_nfev=10000):
    # theta = 9 background + exact scale + common dt.
    seed=np.r_[np.asarray(bgseed,float),1.0,0.0]
    lb=np.r_[np.asarray(base.BG_LB,float),0.0,-0.5]
    ub=np.r_[np.asarray(base.BG_UB,float),6.0,0.5]
    ub[1]=min(ub[1],max(0.05,float(seed[0])-0.01))
    ub[4]=min(ub[4],float(base.t2_upper_us(t))/1000)
    fixed={5:2.0,6:0.0}
    free=[i for i in range(len(seed)) if i not in fixed]
    ee=base.safe_err(e)
    starts=[]
    for scale in (0.25,0.5,1.0,2.0,3.0,5.0):
        for dt in (-0.25,0.0,0.25):
            q=seed.copy(); q[5]=2; q[6]=0; q[9]=scale; q[10]=dt
            starts.append(np.clip(q,lb+1e-8,ub-1e-8))
    fits=[]
    for s in starts:
        def expand(x):
            th=s.copy(); th[free]=x; th[5]=2; th[6]=0
            return th
        def resid(x):
            th=expand(x)
            pred=exact_product_model(base,t,th[:9],sites,th[9],th[10])
            return (np.asarray(y,float)-pred)/ee
        try:
            rr=least_squares(resid,s[free],bounds=(lb[free],ub[free]),loss="soft_l1",
                             f_scale=1,max_nfev=max_nfev,x_scale="jac")
            ff=least_squares(resid,rr.x,bounds=(lb[free],ub[free]),loss="linear",
                             max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10,
                             x_scale="jac")
            th=expand(ff.x)
            pred=exact_product_model(base,t,th[:9],sites,th[9],th[10])
            st=base.calc_stats(y,ee,pred,len(free))
            # Minimum coherence factor reached over measured times.
            tt=np.asarray(t,float)+th[10]; mins=[]
            for site in sites:
                q=2*site["kappa"]*(np.sin(.5*np.pi*site["fI_MHz"]*tt)**2)*(
                    np.sin(.5*np.pi*site["fm_MHz"]*tt)**2)
                mins.append(float(np.min(1-th[9]*q)))
            fits.append(dict(theta=th,pred=pred,min_single_factor=min(mins),**st))
        except Exception:
            pass
    return min(fits,key=lambda q:q["chi2"]) if fits else None


def analyze_nv(nv,row,t,y,e,cat):
    base=v6.load_backend("52G")
    sites=v14.sites_from_record(row)
    theta=np.asarray(json.loads(str(row.theta_json)),float)
    ee=base.safe_err(e)
    addpred=v6.v6_model(base,t,theta,sites)
    addst=base.calc_stats(y,ee,addpred,len(theta)-len(FIXED))
    cross=fit_cross(base,t,y,e,sites,theta)
    exactsites=catalog_exact_sites(cat,row.orientation,sites)
    exact=fit_exact(base,t,y,e,exactsites,theta[:9])
    out=dict(
        nv_index=int(nv),site_key=str(row.site_key),model_order=int(row.model_order),
        additive_scale=float(theta[9]),additive_bic=float(addst["bic"]),
        additive_redchi2=float(addst["red_chi2"]),
    )
    if cross:
        cth=np.asarray(cross["theta"])
        out.update(
            cross_scale=float(cth[9]),cross_bic=float(cross["bic"]),
            cross_redchi2=float(cross["red_chi2"]),
            delta_bic_cross_vs_add=float(cross["bic"]-addst["bic"]),
        )
    if exact:
        eth=np.asarray(exact["theta"])
        out.update(
            exact_scale=float(eth[9]),exact_dt_us=float(eth[10]),
            exact_bic=float(exact["bic"]),exact_redchi2=float(exact["red_chi2"]),
            exact_min_single_factor=float(exact["min_single_factor"]),
            delta_bic_exact_vs_add=float(exact["bic"]-addst["bic"]),
        )
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s27=pd.read_csv(V27/"v27_scale_profile_summary.csv")
    ids=s27[s27.high_scale_required_95.astype(bool)].nv_index.astype(int).tolist()
    winners=pd.read_csv(V24/"v22_winners.csv").set_index("nv_index")
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    cat=v6.load_catalog(base)
    print("V28 nonlinear-ESEEM targets:",ids)
    res=Parallel(n_jobs=10,backend="loky",verbose=10)(
        delayed(analyze_nv)(nv,winners.loc[nv],t,Y[nv],E[nv],cat)
        for nv in ids
    )
    d=pd.DataFrame(res).sort_values("nv_index")
    d["cross_strongly_favored"]=d.delta_bic_cross_vs_add<=-6
    d["exact_strongly_favored"]=d.delta_bic_exact_vs_add<=-6
    d["exact_physical_factor_nonnegative"]=d.exact_min_single_factor>=0
    d.to_csv(OUT/"v28_nonlinear_eSEEM_summary.csv",index=False)
    print("\nV28 COMPLETE")
    print("cross-term strongly favored:",
          d[d.cross_strongly_favored].nv_index.astype(int).tolist())
    print("exact-product strongly favored:",
          d[d.exact_strongly_favored].nv_index.astype(int).tolist())
    print("exact favored + nonnegative factors:",
          d[d.exact_strongly_favored & d.exact_physical_factor_nonnegative]
          .nv_index.astype(int).tolist())
    print(d[["nv_index","additive_redchi2","cross_redchi2",
             "delta_bic_cross_vs_add","exact_redchi2",
             "delta_bic_exact_vs_add","exact_min_single_factor"]].to_string(index=False))
    print("output:",OUT)


if __name__=="__main__":
    main()
