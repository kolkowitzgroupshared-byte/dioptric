"""V27 residual-driven global add-one-13C test for QNami 52 G.

For each unresolved bad-fit NV:
1) keep the current V24/V25 best model,
2) screen the ENTIRE assigned-orientation 13C catalog against its residual,
3) fully refit the best new-site candidates under the current background model.

This deliberately escapes the V6/V22 local-family shortlist. Existing N=3
models are allowed to become N=4 only when the residual supports it.
"""
from __future__ import annotations

import json, sys
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
import sc_c13_spin_echo_v26_residual_error_diagnostics_52G as v26

OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v27_residual_driven_add_site_52G\2026_09")
TOP_SCREEN=16
def fixed_for_bg(bg):
    if bg=="reduced": return {5:2.0,6:0.0}
    if bg=="beta_free": return {6:0.0}
    if bg=="taper_free": return {5:2.0}
    if bg=="full_bg": return {}
    raise ValueError(bg)


def row_sites(row):
    return v14.sites_from_record(row)


def site_from_catalog(r):
    return dict(
        site_id=int(r.site_id),orientation=tuple(r.ori),
        kappa=float(r.kappa),distance_A=float(r.distance_A),
        f0_kHz=float(r.f0_kHz),f1_kHz=float(r.f1_kHz),
    )


def baseline_stats(base,t,y,e,row,theta,bg):
    pred=v6.v6_model(base,t,theta,row_sites(row))
    fixed=fixed_for_bg(bg)
    k=len(theta)-len(fixed)
    return base.calc_stats(y,base.safe_err(e),pred,k),pred


def screen_sites(base,t,y,e,row,theta,bg,catalog,orientation):
    sites0=row_sites(row)
    used={int(s["site_id"]) for s in sites0}
    st0,pred=baseline_stats(base,t,y,e,row,theta,bg)
    resid=np.asarray(y,float)-pred
    ee=base.safe_err(e)
    w=1.0/ee
    _,contrast,carrier=base.carrier_from_bg(t,theta[:9])
    tt=np.asarray(t,float)
    lo,hi=v6.experimental_frequency_band_mhz(t)
    g=catalog[catalog.ori.apply(lambda x: tuple(x)==tuple(orientation))].copy()
    g=g[
        g.f0_kHz.between(lo*1000,hi*1000)
        & g.f1_kHz.between(lo*1000,hi*1000)
        & ~g.site_id.isin(used)
    ]
    rw=resid*w
    out=[]
    for r in g.itertuples():
        f0=float(r.f0_kHz)/1000.0
        f1=float(r.f1_kHz)/1000.0
        X=np.column_stack([
            carrier*np.cos(2*np.pi*f0*tt),
            carrier*np.sin(2*np.pi*f0*tt),
            carrier*np.cos(2*np.pi*f1*tt),
            carrier*np.sin(2*np.pi*f1*tt),
        ])
        Xw=X*w[:,None]
        try:
            coef,*_=np.linalg.lstsq(Xw,rw,rcond=None)
            rr=rw-Xw@coef
            dchi=float(st0["chi2"]-np.sum(rr*rr))
            a0=float(np.hypot(coef[0],coef[1]))
            a1=float(np.hypot(coef[2],coef[3]))
            phi0=float(np.arctan2(-coef[1],coef[0]))
            phi1=float(np.arctan2(-coef[3],coef[2]))
            amp=.5*(a0+a1)
            denom=max(abs(float(contrast))*float(r.kappa)/4.0,1e-9)
            scale_est=float(np.clip(amp/denom,0.0,30.0))
            out.append((dchi,int(r.site_id),phi0,phi1,scale_est,
                        float(r.f0_kHz),float(r.f1_kHz),float(r.kappa)))
        except Exception:
            continue
    out.sort(reverse=True,key=lambda z:z[0])
    # Deduplicate nearly identical frequency pairs so the nonlinear stage
    # samples distinct residual families, not symmetry duplicates.
    picked=[]
    for rec in out:
        if any(abs(rec[5]-q[5])<2.0 and abs(rec[6]-q[6])<2.0 for q in picked):
            continue
        picked.append(rec)
        if len(picked)>=TOP_SCREEN:
            break
    return st0,picked


def make_seed(theta0,n0,phi0,phi1,scale_est,scale):
    th=np.asarray(theta0,float)
    if n0==0:
        return np.r_[th[:9],float(scale),float(phi0),float(phi1)]
    return np.r_[th,float(phi0),float(phi1)]


def fit_added(base,t,y,e,sites,seed,bg,max_nfev=4500):
    n=len(sites); fixed=fixed_for_bg(bg)
    v6.VISIBILITY_SCALE_MAX=30.0
    lb,ub=v6.v6_theta_bounds(
        base,n,float(seed[0]),float(base.t2_upper_us(t))
    )
    seed=np.clip(np.asarray(seed,float),lb+1e-8,ub-1e-8)
    free=[i for i in range(len(seed)) if i not in fixed]
    ee=base.safe_err(e)

    def expand(x,template):
        th=template.copy(); th[free]=x
        for i,val in fixed.items(): th[int(i)]=float(val)
        return th
    def resid(x,template):
        return (np.asarray(y,float)-v6.v6_model(base,t,expand(x,template),sites))/ee

    robust=[]
    for s in [seed]:
        try:
            rr=least_squares(lambda x:resid(x,s),s[free],
                bounds=(lb[free],ub[free]),loss="soft_l1",f_scale=1.0,
                max_nfev=max_nfev,x_scale="jac")
            th=expand(rr.x,s)
            chi=float(np.sum(((np.asarray(y)-v6.v6_model(base,t,th,sites))/ee)**2))
            robust.append((chi,th))
        except Exception:
            pass
    if not robust: return None

    best=None
    for _,s in sorted(robust,key=lambda z:z[0])[:2]:
        try:
            rr=least_squares(lambda x:resid(x,s),s[free],
                bounds=(lb[free],ub[free]),loss="linear",
                max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac")
            th=expand(rr.x,s); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            q=dict(theta=th,**st)
            if best is None or q["chi2"]<best["chi2"]: best=q
        except Exception:
            pass
    return best
def one_nv(nv,row,v25row,t,y,e,catalog,omap):
    base=v6.load_backend("52G")
    theta,source,bg,bg_dbic=v26.current_theta(row,v25row)
    ori=tuple(omap[int(nv)])
    st0,screen=screen_sites(base,t,y,e,row,theta,bg,catalog,ori)
    sites0=row_sites(row); n0=len(sites0)
    fits=[]

    with threadpool_limits(limits=1):
        for rank,rec in enumerate(screen,1):
            dchi,sid,phi0,phi1,scale_est,f0,f1,kappa=rec
            hit=catalog[
                (catalog.site_id.astype(int)==int(sid))
                & catalog.ori.apply(lambda x: tuple(x)==tuple(ori))
            ]
            if hit.empty:
                continue
            cr=hit.iloc[0]
            site=site_from_catalog(cr)
            sites=sites0+[site]
            scales=[scale_est]
            if n0:
                scales += [float(theta[9]),0.5*(float(theta[9])+scale_est)]
            else:
                scales += [1.0,3.0]
            best=None
            for sc in dict.fromkeys(float(np.clip(s,0.05,30)) for s in scales):
                seed=make_seed(theta,n0,phi0,phi1,scale_est,sc)
                q=fit_added(base,t,y,e,sites,seed,bg)
                if q is not None and (best is None or q["chi2"]<best["chi2"]):
                    best=q
            if best is None: continue
            th=np.asarray(best["theta"],float)
            fits.append(dict(
                nv_index=int(nv),baseline_order=n0,new_order=n0+1,
                baseline_site_key=str(row.site_key),added_site_id=int(sid),
                new_site_key=str(tuple(sorted(
                    [int(s["site_id"]) for s in sites]
                ))),
                orientation=str(ori),background_model=bg,
                screen_rank=rank,screen_delta_chi2=float(dchi),
                added_f0_kHz=f0,added_f1_kHz=f1,added_kappa=kappa,
                baseline_chi2=float(st0["chi2"]),baseline_bic=float(st0["bic"]),
                baseline_redchi2=float(st0["red_chi2"]),
                new_chi2=float(best["chi2"]),new_bic=float(best["bic"]),
                new_redchi2=float(best["red_chi2"]),
                delta_bic=float(best["bic"]-st0["bic"]),
                delta_chi2=float(best["chi2"]-st0["chi2"]),
                visibility_scale=float(th[9]) if n0+1>0 else np.nan,
                theta_json=json.dumps([float(x) for x in th]),
            ))
    fits.sort(key=lambda r:r["new_bic"])
    if not fits:
        return [],dict(nv_index=int(nv),status="no_fit")
    b=fits[0]
    summary=dict(
        nv_index=int(nv),status="ok",baseline_order=n0,
        baseline_site_key=str(row.site_key),background_model=bg,
        best_added_site_id=b["added_site_id"],best_new_site_key=b["new_site_key"],
        best_added_f0_kHz=b["added_f0_kHz"],best_added_f1_kHz=b["added_f1_kHz"],
        best_added_kappa=b["added_kappa"],screen_rank=b["screen_rank"],
        delta_bic=b["delta_bic"],delta_chi2=b["delta_chi2"],
        baseline_redchi2=b["baseline_redchi2"],new_redchi2=b["new_redchi2"],
        new_scale=b["visibility_scale"],
    )
    return fits,summary
def run(workers=10):
    OUT.mkdir(parents=True,exist_ok=True)
    cur,v25=v26.load_current_rows()
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    catalog=v6.load_catalog(base)
    omap=v6.load_assigned_orientations("52G",Y.shape[0])
    bad_ids=sorted(v25.index.astype(int).tolist())

    print(f"V27 global residual-driven add-site test: {len(bad_ids)} NVs")
    res=Parallel(n_jobs=workers,backend="loky",verbose=10)(
        delayed(one_nv)(
            nv,cur.loc[nv],v25.loc[nv],t,Y[nv],E[nv],catalog,omap
        ) for nv in bad_ids
    )
    allfits=[]; summaries=[]
    for fits,s in res:
        allfits.extend(fits); summaries.append(s)
    f=pd.DataFrame(allfits)
    s=pd.DataFrame(summaries).sort_values("nv_index")
    f.to_csv(OUT/"v27_add_site_candidate_fits.csv.gz",index=False,compression="gzip")
    s.to_csv(OUT/"v27_add_site_summary.csv",index=False)

    ok=s[s.status=="ok"].copy()
    print("\nV27 COMPLETE")
    print("strong add-site support dBIC<=-6:",int((ok.delta_bic<=-6).sum()))
    print("very strong dBIC<=-10:",int((ok.delta_bic<=-10).sum()))
    print("new redchi2 <3:",int((ok.new_redchi2<3).sum()))
    print("new redchi2 <2:",int((ok.new_redchi2<2).sum()))
    print("N=3 -> N=4 strongly supported:",
          int(((ok.baseline_order==3)&(ok.delta_bic<=-6)).sum()))
    print("output:",OUT)
    print(ok.sort_values("delta_bic").to_string(index=False))


if __name__=="__main__":
    run()
