"""V28 parsimony rerank for strong V27 multi-spin cases.

For V27 cases with baseline N>=2 and deltaBIC<=-6:
* test drop-one lower-order models,
* test same-order global replacements using top V27 residual-driven sites,
* compare them with the best V27 add-one model.

Goal: decide whether apparent N=4 evidence really requires a fourth carbon
or whether a globally discovered site simply replaces a wrong old assignment.
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
import sc_c13_spin_echo_v27_residual_driven_add_site_52G as v27

ROOT27=v27.OUT
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v28_parsimony_rerank_52G\2026_09")
TOP_NEW=8
def get_catalog_site(catalog,ori,sid):
    hit=catalog[
        (catalog.site_id.astype(int)==int(sid))
        & catalog.ori.apply(lambda x: tuple(x)==tuple(ori))
    ]
    if hit.empty:
        raise KeyError((ori,sid))
    return v27.site_from_catalog(hit.iloc[0])


def phase_map_from_theta(sites,theta):
    out={}
    if not sites:
        return out
    j=10
    for s in sites:
        out[int(s["site_id"])]=(float(theta[j]),float(theta[j+1]))
        j+=2
    return out


def build_seed(theta0,base_sites,cand_sites,new_phase_map,scale):
    bg=np.asarray(theta0[:9],float)
    if not cand_sites:
        return bg.copy()
    oldph=phase_map_from_theta(base_sites,theta0)
    vals=list(bg)+[float(scale)]
    for s in cand_sites:
        sid=int(s["site_id"])
        ph=oldph.get(sid,new_phase_map.get(sid,(0.0,0.0)))
        vals.extend([float(ph[0]),float(ph[1])])
    return np.asarray(vals,float)


def fit_sites(base,t,y,e,sites,seeds,bg,max_nfev=4500):
    fixed=v27.fixed_for_bg(bg)
    v6.VISIBILITY_SCALE_MAX=30.0
    n=len(sites)
    baseline_seed=float(np.median([s[0] for s in seeds]))
    lb,ub=v6.v6_theta_bounds(
        base,n,baseline_seed,float(base.t2_upper_us(t))
    )
    ee=base.safe_err(e)
    robust=[]
    for seed in seeds:
        seed=np.clip(np.asarray(seed,float),lb+1e-8,ub-1e-8)
        free=[i for i in range(len(seed)) if i not in fixed]
        def expand(x):
            th=seed.copy(); th[free]=x
            for i,val in fixed.items(): th[int(i)]=float(val)
            return th
        def resid(x):
            return (np.asarray(y,float)-v6.v6_model(base,t,expand(x),sites))/ee
        try:
            rr=least_squares(resid,seed[free],bounds=(lb[free],ub[free]),
                loss="soft_l1",f_scale=1.0,max_nfev=max_nfev,x_scale="jac")
            th=expand(rr.x)
            chi=float(np.sum(((np.asarray(y)-v6.v6_model(base,t,th,sites))/ee)**2))
            robust.append((chi,th))
        except Exception:
            pass
    if not robust:
        return None
    best=None
    for _,seed in sorted(robust,key=lambda z:z[0])[:3]:
        free=[i for i in range(len(seed)) if i not in fixed]
        def expand(x):
            th=seed.copy(); th[free]=x
            for i,val in fixed.items(): th[int(i)]=float(val)
            return th
        try:
            rr=least_squares(
                lambda x:(np.asarray(y)-v6.v6_model(base,t,expand(x),sites))/ee,
                seed[free],bounds=(lb[free],ub[free]),loss="linear",
                max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac")
            th=expand(rr.x); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            q=dict(theta=th,**st)
            if best is None or q["chi2"]<best["chi2"]: best=q
        except Exception:
            pass
    return best
def one_nv(nv,row,v25row,v27fits,v27summary,t,y,e,catalog,omap):
    base=v6.load_backend("52G")
    theta,source,bg,bg_dbic=v26.current_theta(row,v25row)
    base_sites=v14.sites_from_record(row)
    n0=len(base_sites); ori=tuple(omap[int(nv)])
    base_ids=[int(s["site_id"]) for s in base_sites]

    vf=v27fits[v27fits.nv_index==nv].sort_values("new_bic")
    top_ids=[]
    new_phase_map={}
    new_scale_map={}
    for r in vf.itertuples():
        sid=int(r.added_site_id)
        if sid not in top_ids:
            top_ids.append(sid)
            th=np.asarray(json.loads(str(r.theta_json)),float)
            new_phase_map[sid]=(float(th[-2]),float(th[-1]))
            new_scale_map[sid]=float(r.visibility_scale)
        if len(top_ids)>=TOP_NEW: break

    candidates=[]
    # Drop one old site.
    for drop in base_ids:
        ids=[x for x in base_ids if x!=drop]
        candidates.append(("drop_one",tuple(ids),None,drop))
    # Same-order replacement with globally discovered sites.
    for drop in base_ids:
        for sid in top_ids:
            ids=[x for x in base_ids if x!=drop]
            if sid in ids: continue
            candidates.append(("replace_one",tuple(ids+[sid]),sid,drop))

    # Canonical dedup.
    uniq={}
    for kind,ids,added,dropped in candidates:
        key=tuple(sorted(ids))
        if key not in uniq: uniq[key]=(kind,ids,added,dropped)
    candidates=list(uniq.values())
    rows=[]
    with threadpool_limits(limits=1):
        for kind,ids,added,dropped in candidates:
            # Preserve old-site order, then append any genuinely new site.
            sites=[]
            for s in base_sites:
                if int(s["site_id"]) in ids:
                    sites.append(s)
            for sid in ids:
                if sid not in base_ids:
                    sites.append(get_catalog_site(catalog,ori,sid))

            scales=[]
            if sites:
                scales.append(float(theta[9]))
                if added is not None:
                    scales.append(new_scale_map.get(int(added),float(theta[9])))
                scales += [1.0,3.0,10.0]
            else:
                scales=[1.0]

            seeds=[]
            if sites:
                for sc in dict.fromkeys(float(np.clip(x,0.05,30)) for x in scales):
                    seeds.append(build_seed(theta,base_sites,sites,new_phase_map,sc))
            else:
                seeds=[np.asarray(theta[:9],float)]

            fit=fit_sites(base,t,y,e,sites,seeds,bg)
            if fit is None: continue
            th=np.asarray(fit["theta"],float)
            rows.append(dict(
                nv_index=int(nv),candidate_kind=kind,
                model_order=len(sites),site_key=str(tuple(sorted(ids))),
                added_site_id=added,dropped_site_id=dropped,
                bic=float(fit["bic"]),chi2=float(fit["chi2"]),
                redchi2=float(fit["red_chi2"]),
                visibility_scale=float(th[9]) if sites else np.nan,
                theta_json=json.dumps([float(x) for x in th]),
            ))

    vs=v27summary.loc[int(nv)]
    v27_bic=float(vf.iloc[0].new_bic)
    v27_order=int(vs.baseline_order)+1
    v27_key=str(vs.best_new_site_key)
    if rows:
        best=min(rows,key=lambda r:r["bic"])
        lower_bic=float(best["bic"])
        delta_lower_vs_add=lower_bic-v27_bic
        if delta_lower_vs_add<=2:
            cls="lower_order_competitive"
        elif delta_lower_vs_add>=6:
            cls="add_one_order_strongly_preferred"
        else:
            cls="order_ambiguous"
    else:
        best={}; lower_bic=np.nan; delta_lower_vs_add=np.nan; cls="no_lower_fit"
    summary=dict(
        nv_index=int(nv),baseline_order=n0,baseline_site_key=str(row.site_key),
        v27_add_order=v27_order,v27_add_site_key=v27_key,v27_add_bic=v27_bic,
        best_lower_kind=best.get("candidate_kind",""),
        best_lower_order=best.get("model_order",np.nan),
        best_lower_site_key=best.get("site_key",""),
        best_lower_bic=lower_bic,
        deltaBIC_lower_minus_add=delta_lower_vs_add,
        classification=cls,
    )
    return rows,summary


def run(workers=10):
    OUT.mkdir(parents=True,exist_ok=True)
    cur,v25=v26.load_current_rows()
    s27=pd.read_csv(ROOT27/"v27_add_site_summary.csv").set_index("nv_index")
    f27=pd.read_csv(ROOT27/"v27_add_site_candidate_fits.csv.gz")
    strong=s27[(s27.status=="ok")&(s27.delta_bic<=-6)&(s27.baseline_order>=2)]
    ids=sorted(strong.index.astype(int).tolist())

    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    cat=v6.load_catalog(base); omap=v6.load_assigned_orientations("52G",Y.shape[0])

    print(f"V28 parsimony rerank: {len(ids)} strong multi-spin V27 cases")
    res=Parallel(n_jobs=workers,backend="loky",verbose=10)(
        delayed(one_nv)(
            nv,cur.loc[nv],v25.loc[nv],f27,strong,t,Y[nv],E[nv],cat,omap
        ) for nv in ids
    )
    rows=[]; summaries=[]
    for rr,ss in res:
        rows.extend(rr); summaries.append(ss)
    d=pd.DataFrame(rows); s=pd.DataFrame(summaries).sort_values("nv_index")
    d.to_csv(OUT/"v28_lower_order_candidate_fits.csv.gz",index=False,compression="gzip")
    s.to_csv(OUT/"v28_parsimony_summary.csv",index=False)
    print("\nV28 COMPLETE")
    print(s.classification.value_counts().to_dict())
    print("N4 strong after lower-order challenge:",
          int(((s.v27_add_order==4)&
               (s.classification=="add_one_order_strongly_preferred")).sum()))
    print("N4 with competitive N3/lower:",
          int(((s.v27_add_order==4)&
               (s.classification=="lower_order_competitive")).sum()))
    print("output:",OUT)
    print(s.to_string(index=False))


if __name__=="__main__":
    run()
