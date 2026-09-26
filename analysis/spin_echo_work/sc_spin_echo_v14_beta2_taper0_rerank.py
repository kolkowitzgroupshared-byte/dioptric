"""V14 production rerank: beta=2, revival taper=0 on every saved V6 candidate.

This does NOT repeat the lattice-site search.  It takes the complete V6
candidate table (all saved N=0/1/2/3 site combinations), refits each candidate
under the V13 reduced background, robustly polishes the candidates that can
matter for selection, and then recomputes all BIC/AICc/red-chi2 rankings.

Design:
  Stage 1 (all candidates):
    direct least-squares refit from that candidate's saved V6 solution with
    beta=2 and taper=0.  This is a fast screen over ~100k candidates/field.

  Stage 2 (selection-relevant candidates):
    robust multi-seed refit using the V12 optimizer.  The polish set is the
    union of top candidates by Stage-1 BIC and original V6 BIC within each
    NV/order, plus candidates close to the Stage-1 within-order minimum.

Both stages checkpoint per NV and are resumable.  Original V6 files are never
overwritten.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v7_diagnostic as v7
import sc_spin_echo_v10_lattice_bath_diagnostic as v10
import sc_spin_echo_v12_v6_background_ablation as v12

FIXED={5:2.0,6:0.0}
MODEL_TAG="v14_beta2_taper0"
FIT_COLS=[
    "chi2","red_chi2","aicc","bic","npar","baseline","contrast",
    "revival_time_us","width0_us","T2_us","T2_limit_us","T2_bound_hit",
    "beta","amp_taper_alpha","width_slope","revival_chirp","theta_json",
    "visibility_scale","visibility_scale_max","visibility_scale_bound_hit",
]
SITE_BASE_COLS=[
    "site_id","f0_kHz","f1_kHz","kappa","distance_A","family_id",
    "family_members","physical_snr","matched_delta_chi2",
    "x_A","y_A","z_A","A_par_Hz","A_perp_Hz","theta_deg",
    "f_minus_Hz","f_plus_Hz","A_par_kHz","A_perp_kHz",
    "f_minus_kHz","f_plus_kHz",
]

def source_candidate(field):
    base=v6.load_backend(field)
    return v7.latest_v6_candidate(base,field)


def compact_columns(df):
    cols=["_row_id","nv_index","model_order","theta_json","baseline",
          "T2_limit_us","orientation","site_key","bic"]
    for j in (1,2,3):
        for c in ["site_id","f0_kHz","f1_kHz","kappa"]:
            k=f"c13_{j}_{c}"
            if k in df.columns: cols.append(k)
    return [c for c in cols if c in df.columns]


def sites_from_record(rec):
    n=int(rec["model_order"])
    out=[]
    ori=rec.get("orientation",None)
    for j in range(1,n+1):
        out.append(dict(
            site_id=int(rec[f"c13_{j}_site_id"]),
            f0_kHz=float(rec[f"c13_{j}_f0_kHz"]),
            f1_kHz=float(rec[f"c13_{j}_f1_kHz"]),
            kappa=float(rec[f"c13_{j}_kappa"]),
            orientation=ori,
        ))
    return out


def fit_to_updates(base,t,row,sites,fit,stage):
    th=np.asarray(fit["theta"],float)
    n=int(row["model_order"])
    contrast=float(th[1])
    out=dict(
        _row_id=int(row["_row_id"]),
        chi2=float(fit["chi2"]),red_chi2=float(fit["red_chi2"]),
        aicc=float(fit["aicc"]),bic=float(fit["bic"]),npar=int(fit["npar"]),
        baseline=float(th[0]),contrast=contrast,
        revival_time_us=float(th[2]),width0_us=float(th[3]),
        T2_us=float(1000.0*th[4]),
        T2_limit_us=float(base.t2_upper_us(t)),
        T2_bound_hit=bool(abs(1000.0*th[4]-base.t2_upper_us(t)) <=
                          max(1e-3,1e-3*base.t2_upper_us(t))),
        beta=float(th[5]),amp_taper_alpha=float(th[6]),
        width_slope=float(th[7]),revival_chirp=float(th[8]),
        theta_json=json.dumps([float(x) for x in th]),
        visibility_scale=(float(th[9]) if n else np.nan),
        visibility_scale_max=(float(v6.VISIBILITY_SCALE_MAX) if n else np.nan),
        visibility_scale_bound_hit=(
            bool(th[9] >= v6.VISIBILITY_SCALE_MAX-0.06) if n else False),
        v14_fit_stage=str(stage),
    )
    if n:
        scale=float(th[9]); k=10
        for j,s in enumerate(sites,1):
            phi0,phi1=float(th[k]),float(th[k+1]);k+=2
            amp1=contrast*float(s["kappa"])/4.0
            out[f"c13_{j}_amp_expected_scale1"]=amp1
            out[f"c13_{j}_amp"]=scale*amp1
            out[f"c13_{j}_amp_scale"]=scale
            out[f"c13_{j}_phi0"]=phi0
            out[f"c13_{j}_phi1"]=phi1
    return out

def projected_seed_fit(base,t,y,e,row,sites,max_nfev):
    """Fast Stage-1 linear fit from the saved V6 candidate."""
    seed=np.asarray(json.loads(row["theta_json"]),float)
    n=int(row["model_order"])
    lb,ub=v6.v6_theta_bounds(
        base,n,float(row["baseline"]),float(base.t2_upper_us(t)))
    seed[5]=2.0; seed[6]=0.0
    free=[i for i in range(len(seed)) if i not in FIXED]
    x0=np.clip(seed[free],lb[free]+1e-9,ub[free]-1e-9)
    ee=base.safe_err(e)

    def expand(x):
        th=seed.copy();th[free]=x;th[5]=2.0;th[6]=0.0
        return th
    def resid(x):
        return (np.asarray(y,float)-v6.v6_model(base,t,expand(x),sites))/ee

    # Seed itself is always a valid fallback.
    th0=expand(x0)
    p0=v6.v6_model(base,t,th0,sites)
    st0=base.calc_stats(y,ee,p0,len(free))
    best=dict(theta=th0,pred=p0,**st0)
    try:
        rr=least_squares(
            resid,x0,bounds=(lb[free],ub[free]),loss="linear",
            max_nfev=int(max_nfev),ftol=3e-9,xtol=3e-9,gtol=3e-9,
            x_scale="jac")
        th=expand(rr.x);pred=v6.v6_model(base,t,th,sites)
        st=base.calc_stats(y,ee,pred,len(free))
        q=dict(theta=th,pred=pred,**st)
        if q["chi2"] < best["chi2"]: best=q
    except Exception:
        pass
    return best


def stage1_nv(field,nv,records,t,y,e,checkpoint,max_nfev):
    checkpoint=Path(checkpoint)
    if checkpoint.exists():
        return str(checkpoint)
    base=v6.load_backend(field)
    out=[]
    with threadpool_limits(limits=1):
        for rec in records:
            sites=sites_from_record(rec)
            fit=projected_seed_fit(base,t,y,e,rec,sites,max_nfev)
            out.append(fit_to_updates(base,t,rec,sites,fit,"stage1"))
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(out).to_csv(checkpoint,index=False,compression="gzip")
    return str(checkpoint)


def strong_fit(base,t,y,e,row,sites,stage1_theta):
    """Stage-2 robust fit; Stage-1 solution is supplied as an extra seed."""
    # v12.fit_ablation keeps seed solutions as candidates and performs robust
    # + final least-squares polishing.
    pseudo=pd.Series(row)
    q=v12.fit_ablation(
        base,t,y,e,pseudo,sites,FIXED,
        extra_full_seeds=[np.asarray(stage1_theta,float)])
    return q

def choose_polish_rows(df,per_order,delta_bic,cap_per_order):
    chosen=set()
    for (nv,order),g in df.groupby(["nv_index","model_order"],sort=False):
        g=g.sort_values("bic")
        # Top under the reduced model.
        chosen.update(g.head(int(per_order))._row_id.astype(int).tolist())
        # Original V6 top candidates protect against Stage-1 local minima.
        chosen.update(g.sort_values("v6_source_bic").head(int(per_order))
                      ._row_id.astype(int).tolist())
        b=float(g.bic.min())
        near=g[g.bic <= b+float(delta_bic)].head(int(cap_per_order))
        chosen.update(near._row_id.astype(int).tolist())
    # Also protect the globally best reduced candidates for every NV.
    for nv,g in df.groupby("nv_index",sort=False):
        chosen.update(g.sort_values("bic").head(max(3,int(per_order)))
                      ._row_id.astype(int).tolist())
    return chosen


def stage2_nv(field,nv,records,t,y,e,checkpoint):
    checkpoint=Path(checkpoint)
    if checkpoint.exists():
        return str(checkpoint)
    base=v6.load_backend(field)
    out=[]
    with threadpool_limits(limits=1):
        for rec in records:
            sites=sites_from_record(rec)
            stheta=json.loads(rec["stage1_theta_json"])
            q=strong_fit(base,t,y,e,rec,sites,stheta)
            if q is None:
                continue
            out.append(fit_to_updates(base,t,rec,sites,q,"polish"))
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(out).to_csv(checkpoint,index=False,compression="gzip")
    return str(checkpoint)


def apply_updates(df,updates):
    u=updates.set_index("_row_id")
    d=df.set_index("_row_id",drop=False).copy()
    for c in u.columns:
        if c=="_row_id":
            continue
        vals=u[c]
        if c not in d.columns:
            if pd.api.types.is_numeric_dtype(vals.dtype):
                d[c]=np.nan
            elif pd.api.types.is_bool_dtype(vals.dtype):
                d[c]=False
            else:
                d[c]=pd.Series([None]*len(d),index=d.index,dtype=object)
        # Pandas 2.x forbids assigning strings into a float placeholder column.
        if (pd.api.types.is_object_dtype(vals.dtype) or
            pd.api.types.is_string_dtype(vals.dtype)):
            d[c]=d[c].astype(object)
        d.loc[u.index,c]=vals.to_numpy()
    return d.reset_index(drop=True)

def rerank(df):
    d=df.copy()
    d["rank_within_order"]=(
        d.groupby(["nv_index","model_order"])["bic"]
         .rank(method="first").astype(int))
    d["rank_global_bic"]=(
        d.groupby("nv_index")["bic"].rank(method="first").astype(int))
    d["rank_global_aicc"]=(
        d.groupby("nv_index")["aicc"].rank(method="first").astype(int))
    d["rank_global_redchi"]=(
        d.groupby("nv_index")["red_chi2"].rank(method="first").astype(int))
    return d


def summarize(field,src,final):
    old=src[src.rank_global_bic==1][["nv_index","model_order","site_key","bic"]].copy()
    old.columns=["nv_index","v6_order","v6_site_key","v6_bic"]
    new=final[final.rank_global_bic==1][["nv_index","model_order","site_key","bic","red_chi2",
                                         "visibility_scale","visibility_scale_bound_hit"]].copy()
    new.columns=["nv_index","v14_order","v14_site_key","v14_bic","v14_red_chi2",
                 "v14_visibility_scale","v14_visibility_scale_bound_hit"]
    tr=old.merge(new,on="nv_index",how="outer")
    tr["order_changed"]=tr.v6_order!=tr.v14_order
    tr["site_changed"]=tr.v6_site_key.astype(str)!=tr.v14_site_key.astype(str)
    tr["delta_bic_newwinner_vs_oldwinner_v6bic"]=tr.v14_bic-tr.v6_bic
    counts=(new.v14_order.value_counts().reindex([0,1,2,3],fill_value=0)
            .rename_axis("model_order").reset_index(name="count"))
    return tr,counts


def run_field(field,args):
    t0=time.time()
    src_path=source_candidate(field)
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck)
    src=pd.read_csv(src_path)
    src["_row_id"]=np.arange(len(src),dtype=int)
    src["v6_source_chi2"]=src.chi2
    src["v6_source_red_chi2"]=src.red_chi2
    src["v6_source_bic"]=src.bic
    src["v6_source_aicc"]=src.aicc
    src["v6_source_npar"]=src.npar
    src["v6_source_theta_json"]=src.theta_json

    if args.nv:
        wanted={int(x) for x in args.nv.split(",") if x.strip()}
        src=src[src.nv_index.isin(wanted)].copy()
    nvs=sorted(src.nv_index.unique().astype(int).tolist())

    outdir=Path(args.output_dir) if args.output_dir else src_path.parent/"v14_beta2_taper0_rerank"
    outdir.mkdir(parents=True,exist_ok=True)
    ck1=outdir/f"checkpoint_stage1_{field}"
    ck2=outdir/f"checkpoint_stage2_{field}"

    compact=src[compact_columns(src)].copy()
    records_by_nv={int(nv):compact[compact.nv_index==nv].to_dict("records") for nv in nvs}
    jobs=[]
    for nv in nvs:
        cp=ck1/f"nv_{nv:04d}.csv.gz"
        if not cp.exists():
            jobs.append((nv,records_by_nv[nv],cp))
    print(f"\n{field} Stage 1: {len(src):,} candidates, {len(nvs)} NVs; "
          f"{len(jobs)} NV checkpoints remaining; workers={args.workers}")
    if jobs:
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(stage1_nv)(
                field,nv,recs,t,Y[nv],E[nv],str(cp),args.stage1_max_nfev)
            for nv,recs,cp in jobs)
    stage1=pd.concat(
        [pd.read_csv(ck1/f"nv_{nv:04d}.csv.gz") for nv in nvs],
        ignore_index=True)
    work=apply_updates(src,stage1)
    work["v14_background_model"]="beta2_taper0"
    work["amplitude_model"]="shared_scale_times_contrast_kappa_over_4"

    # Stage-2 candidate selection uses Stage-1 BIC plus original V6 BIC.
    chosen=choose_polish_rows(
        work,args.polish_per_order,args.polish_delta_bic,args.polish_cap_per_order)
    polish=work[work._row_id.isin(chosen)].copy()
    print(f"{field} Stage 2: polishing {len(polish):,}/{len(work):,} candidates "
          f"({100*len(polish)/max(1,len(work)):.2f}%)")

    fitneed=["_row_id","nv_index","model_order","theta_json","baseline",
             "orientation","site_key","v6_source_bic"]
    for j in (1,2,3):
        for c in ["site_id","f0_kHz","f1_kHz","kappa"]:
            fitneed.append(f"c13_{j}_{c}")
    pcompact=polish[[c for c in fitneed if c in polish.columns]].copy()
    # Stage-1 theta must remain available separately because theta_json is already
    # the Stage-1 fitted theta after apply_updates.
    pcompact["stage1_theta_json"]=pcompact["theta_json"]
    # v12.fit_ablation expects the candidate's theta_json as a seed. Stage-1 is
    # the best seed for the reduced model, so this is intentional.
    precs={int(nv):pcompact[pcompact.nv_index==nv].to_dict("records")
           for nv in sorted(pcompact.nv_index.unique())}
    jobs=[]
    for nv,recs in precs.items():
        cp=ck2/f"nv_{nv:04d}.csv.gz"
        if not cp.exists(): jobs.append((nv,recs,cp))
    if jobs:
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(stage2_nv)(field,nv,recs,t,Y[nv],E[nv],str(cp))
            for nv,recs,cp in jobs)
    stage2files=[ck2/f"nv_{nv:04d}.csv.gz" for nv in precs]
    stage2=pd.concat([pd.read_csv(p) for p in stage2files if p.exists()],
                     ignore_index=True) if stage2files else pd.DataFrame()
    if len(stage2):
        work=apply_updates(work,stage2)

    final=rerank(work)
    tag=f"{src_path.stem.replace('_candidate_fits','')}_{MODEL_TAG}"
    outcsv=outdir/f"{tag}_candidate_fits.csv.gz"
    final.to_csv(outcsv,index=False,compression="gzip")
    transitions,counts=summarize(field,src,final)
    transitions.to_csv(outdir/f"{tag}_winner_transitions.csv",index=False)
    counts.to_csv(outdir/f"{tag}_winner_order_counts.csv",index=False)

    meta=dict(
        field=field,source_candidate=str(src_path),output_candidate=str(outcsv),
        model="beta2_taper0",workers=int(args.workers),
        stage1_max_nfev=int(args.stage1_max_nfev),
        polish_per_order=int(args.polish_per_order),
        polish_delta_bic=float(args.polish_delta_bic),
        polish_cap_per_order=int(args.polish_cap_per_order),
        n_candidates=int(len(final)),n_polished=int(len(stage2)),
        n_nvs=int(final.nv_index.nunique()),
        elapsed_minutes=float((time.time()-t0)/60.0),
    )
    with open(outdir/f"{tag}_run_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print(f"\n{field} COMPLETE in {meta['elapsed_minutes']:.1f} min")
    print("winner orders:",counts.set_index("model_order")["count"].to_dict())
    print("order changes:",int(transitions.order_changed.sum()),
          "site changes:",int(transitions.site_changed.sum()))
    print("output:",outcsv)
    return meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--workers",type=int,default=12)
    ap.add_argument("--stage1-max-nfev",type=int,default=1200)
    ap.add_argument("--polish-per-order",type=int,default=3)
    ap.add_argument("--polish-delta-bic",type=float,default=12.0)
    ap.add_argument("--polish-cap-per-order",type=int,default=8)
    ap.add_argument("--nv",type=str,default=None,
                    help="optional comma-separated NV indices for smoke tests")
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()

    fields=["49G","52G"] if args.field=="both" else [args.field]
    metas=[run_field(f,args) for f in fields]
    print("\nALL REQUESTED FIELDS COMPLETE")
    for m in metas:
        print(m["field"],m["output_candidate"])


if __name__=="__main__":
    main()

