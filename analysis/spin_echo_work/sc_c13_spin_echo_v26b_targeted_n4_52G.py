"""V26b targeted N=4 validation for V26-supported QNami 52G NVs.

Only NVs with V26 catalog-pair residual support are tested.  For each:
  * keep the V24 N=3 winner as incumbent
  * use the V25-selected background freedom
  * screen unused same-orientation catalog sites on the incumbent residual
  * fully nonlinear-fit N=4 for the strongest residual sites
  * compare N=4 vs incumbent N=3 with BIC

This does not launch a global N=4 combinatorial search.
"""
from __future__ import annotations

import ast, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

V24=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v24_targeted_heavy_52G\2026_09")
V25=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v25_background_ablation_52G\2026_09")
V26=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v26_residual_error_diagnostics_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v26b_targeted_n4_52G\2026_09")
FIXED_BY_BG={
    "reduced":{5:2.0,6:0.0},
    "beta_free":{6:0.0},
    "taper_free":{5:2.0},
    "full_bg":{},
}


def parse_orientation(x):
    if isinstance(x,(tuple,list,np.ndarray)):
        return tuple(int(v) for v in x)
    return tuple(int(v) for v in ast.literal_eval(str(x)))


def residual_screen(base,cat,ori,t,y,e,theta,sites,top_n=8):
    cc=cat[cat.ori.map(tuple)==tuple(ori)].copy()
    lo,hi=v6.experimental_frequency_band_mhz(t)
    cc=cc[
        cc.f0_kHz.between(1000*lo,1000*hi)
        & cc.f1_kHz.between(1000*lo,1000*hi)
    ]
    used={int(s["site_id"]) for s in sites}
    cc=cc[~cc.site_id.astype(int).isin(used)]
    pred=v6.v6_model(base,t,theta,sites)
    ee=base.safe_err(e); rw=(np.asarray(y,float)-pred)/ee
    carrier=base.carrier_from_bg(t,theta[:9])[2]
    tt=np.asarray(t,float)
    rows=[]
    for q in cc.itertuples():
        f0=float(q.f0_kHz)/1000; f1=float(q.f1_kHz)/1000
        X=np.column_stack([
            carrier*np.cos(2*np.pi*f0*tt)/ee,
            carrier*np.sin(2*np.pi*f0*tt)/ee,
            carrier*np.cos(2*np.pi*f1*tt)/ee,
            carrier*np.sin(2*np.pi*f1*tt)/ee,
        ])
        try:
            c,*_=np.linalg.lstsq(X,rw,rcond=None)
            rr=rw-X@c
            gain=float(np.sum(rw*rw)-np.sum(rr*rr))
        except Exception:
            continue
        rows.append((gain,int(q.site_id),q))
    rows.sort(key=lambda z:z[0],reverse=True)
    return rows[:top_n]
def site_from_catalog(q):
    return dict(
        site_id=int(q.site_id),orientation=tuple(q.ori),
        kappa=float(q.kappa),distance_A=float(q.distance_A),
        f0_kHz=float(q.f0_kHz),f1_kHz=float(q.f1_kHz),
    )


def fit_fixed_bg(base,t,y,e,sites,seeds,fixed,max_nfev=8000):
    v6.VISIBILITY_SCALE_MAX=30.0
    n=len(sites)
    expected=9 if n==0 else 10+2*n
    seeds=[np.asarray(s,float) for s in seeds if len(s)==expected]
    baseline=float(np.nanmedian([s[0] for s in seeds]))
    lb,ub=v6.v6_theta_bounds(base,n,baseline,float(base.t2_upper_us(t)))
    free=[i for i in range(expected) if i not in fixed]
    ee=base.safe_err(e)

    def expand(x,s):
        th=s.copy(); th[free]=x
        for i,val in fixed.items(): th[int(i)]=float(val)
        return th
    def resid(x,s):
        return (np.asarray(y,float)-v6.v6_model(base,t,expand(x,s),sites))/ee

    robust=[]
    for seed in seeds:
        s=np.clip(seed,lb+1e-8,ub-1e-8)
        for i,val in fixed.items(): s[int(i)]=float(val)
        try:
            rr=least_squares(lambda x:resid(x,s),s[free],bounds=(lb[free],ub[free]),
                             loss="soft_l1",f_scale=1,max_nfev=max_nfev,x_scale="jac")
            th=expand(rr.x,s); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            robust.append((st["chi2"],th))
        except Exception:
            pass
    robust.sort(key=lambda z:z[0])
    finals=[]
    for _,s in robust[:5]:
        try:
            rr=least_squares(lambda x:resid(x,s),s[free],bounds=(lb[free],ub[free]),
                             loss="linear",max_nfev=max_nfev,
                             ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac")
            th=expand(rr.x,s); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            finals.append(dict(theta=th,pred=pred,**st))
        except Exception:
            pass
    if not finals:
        raise RuntimeError("fit failed")
    return min(finals,key=lambda z:z["chi2"])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s26=pd.read_csv(V26/"v26_residual_summary.csv")
    ids=s26[s26.n4_screen_support.astype(bool)].nv_index.astype(int).tolist()
    winners=pd.read_csv(V24/"v22_winners.csv").set_index("nv_index")
    v25=pd.read_csv(V25/"v25_background_ablation.csv").set_index("nv_index")
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    cat=v6.load_catalog(base)

    rows=[]
    print("V26b N=4 targets:",ids)
    for nv in ids:
        win=winners.loc[nv]; q=v25.loc[nv]
        bg=min(("reduced","beta_free","taper_free","full_bg"),
               key=lambda m:float(q[f"{m}_bic"]))
        fixed=FIXED_BY_BG[bg]
        theta3=np.asarray(json.loads(str(q[f"{bg}_theta_json"])),float)
        sites3=v14.sites_from_record(win)
        ori=parse_orientation(win.orientation)
        screen=residual_screen(base,cat,ori,t,Y[nv],E[nv],theta3,sites3,top_n=8)
        incumbent_bic=float(q[f"{bg}_bic"])
        incumbent_red=float(q[f"{bg}_redchi2"])
        for rank,(upper,site_id,qr) in enumerate(screen,1):
            newsite=site_from_catalog(qr)
            sites4=list(sites3)+[newsite]
            phase_seeds=[
                (0,0),(np.pi/2,0),(0,np.pi/2),(np.pi/2,np.pi/2),
                (-np.pi/2,0),(0,-np.pi/2),(-np.pi/2,-np.pi/2),
                (np.pi,0),(0,np.pi),
            ]
            seeds=[np.r_[theta3,p0,p1] for p0,p1 in phase_seeds]
            fit=fit_fixed_bg(base,t,Y[nv],E[nv],sites4,seeds,fixed)
            th=fit["theta"]
            rows.append(dict(
                nv_index=nv,background=bg,incumbent_site_key=str(win.site_key),
                incumbent_bic=incumbent_bic,incumbent_redchi2=incumbent_red,
                residual_rank=rank,residual_dchi2_upper=upper,
                added_site_id=site_id,added_f0_kHz=float(qr.f0_kHz),
                added_f1_kHz=float(qr.f1_kHz),added_kappa=float(qr.kappa),
                added_distance_A=float(qr.distance_A),
                n4_bic=float(fit["bic"]),n4_redchi2=float(fit["red_chi2"]),
                delta_bic_n4_vs_n3=float(fit["bic"]-incumbent_bic),
                chi2_gain=float(q[f"{bg}_chi2"]-fit["chi2"]),
                visibility_scale=float(th[9]),theta_json=json.dumps([float(x) for x in th]),
            ))
            print(nv,site_id,"dBIC",rows[-1]["delta_bic_n4_vs_n3"],
                  "red",rows[-1]["n4_redchi2"],flush=True)

    d=pd.DataFrame(rows)
    d["rank_n4_bic"]=d.groupby("nv_index").n4_bic.rank(method="first")
    d.to_csv(OUT/"v26b_targeted_n4_candidates.csv",index=False)
    best=d.sort_values(["nv_index","n4_bic"]).groupby("nv_index",as_index=False).first()
    best["n4_strongly_supported"]=best.delta_bic_n4_vs_n3<=-6
    best.to_csv(OUT/"v26b_targeted_n4_best.csv",index=False)
    print("\nV26b COMPLETE")
    print(best[["nv_index","added_site_id","delta_bic_n4_vs_n3",
                "n4_redchi2","n4_strongly_supported"]].to_string(index=False))
    print("output:",OUT)


if __name__=="__main__":
    main()
