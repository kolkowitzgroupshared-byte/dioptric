"""V22 full production spin-echo analysis after V14-V21 troubleshooting.

This is the consolidated production pipeline.

Physics kept:
  * V14 reduced background: beta=2, revival-amplitude taper=0
  * additive physical 13C sidebands
  * one shared amplitude scale: A_j = s_NV * contrast * kappa_j / 4
  * fixed catalog frequencies; two fitted phases per selected 13C site

Changes motivated by diagnostics:
  1. Relax the old s_NV<=3 ceiling (default numerical ceiling 30).
  2. Refit/rerank every saved N=0/1/2/3 candidate.
  3. Around the best saved candidates, add local near-degenerate catalog-site
     substitutions so physically similar frequencies with different kappa are
     not missed (the V14 failure mode seen for NV85/168/171).
  4. Model order is reranked globally, allowing cases like NV37 N=3 -> N=2.
  5. Emit per-NV QC, runner-up margins, transitions, top candidates and PDF.

Original V14/V6 files are never overwritten. Checkpoints are resumable.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

DEFAULT_ROOT=Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v22_full_physical_rerank\2026_09"
)

def find_v14_file(field):
    root=Path(
        r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v14_beta2_taper0_rerank\2026_09"
    )
    fs=[p for p in root.glob(f"*{field}*v14_beta2_taper0_candidate_fits.csv.gz")
        if "smoke" not in str(p).lower()]
    if not fs:
        raise FileNotFoundError(f"No production V14 {field} file under {root}")
    return max(fs,key=lambda p:p.stat().st_mtime)


def config_tag(args):
    def f(x): return f"{float(x):g}".replace(".","p")
    return (
        f"smax{f(args.scale_max)}_tol{f(args.freq_tol_khz)}_"
        f"pool{args.local_pool_size}_sub{args.max_local_substitutions}_"
        f"topO{args.augment_top_per_order}_topG{args.augment_top_global}_"
        f"cap{args.max_augmented_per_base}"
    )
def patch_scale_max(scale_max):
    # v14/v12 call v6.v6_theta_bounds dynamically, so this patch propagates
    # through both screen and robust polish without changing historical files.
    v6.VISIBILITY_SCALE_MAX=float(scale_max)


def stage1_nv_relaxed(field,nv,records,t,y,e,checkpoint,max_nfev,scale_max):
    patch_scale_max(scale_max)
    return v14.stage1_nv(field,nv,records,t,y,e,checkpoint,max_nfev)


def stage2_nv_relaxed(field,nv,records,t,y,e,checkpoint,scale_max):
    patch_scale_max(scale_max)
    return v14.stage2_nv(field,nv,records,t,y,e,checkpoint)


def source_prepare(src):
    d=src.copy()
    if "_row_id" not in d.columns:
        d["_row_id"]=np.arange(len(d),dtype=int)
    d["source_kind"]="saved_v14"
    d["source_parent_site_key"]=d["site_key"].astype(str)
    d["source_baseline_seed"]=d["baseline"].astype(float)
    d["v6_source_bic"]=d["bic"].astype(float)
    d["v6_source_chi2"]=d["chi2"].astype(float)
    d["v6_source_red_chi2"]=d["red_chi2"].astype(float)
    d["v6_source_aicc"]=d["aicc"].astype(float)
    d["v6_source_npar"]=d["npar"].astype(int)
    d["v6_source_theta_json"]=d["theta_json"].astype(str)
    return d


def run_saved_stage(field,src,t,Y,E,args,outdir):
    ck1=outdir/"checkpoint_saved_stage1"
    ck2=outdir/"checkpoint_saved_stage2"
    cols=v14.compact_columns(src)
    compact=src[cols].copy()
    nvs=sorted(src.nv_index.unique().astype(int))
    recs={nv:compact[compact.nv_index==nv].to_dict("records") for nv in nvs}
    jobs=[]
    for nv in nvs:
        cp=ck1/f"nv_{nv:04d}.csv.gz"
        if not cp.exists(): jobs.append((nv,recs[nv],cp))
    print(
        f"\nSaved Stage 1: {len(src):,} candidates / {len(nvs)} NVs; "
        f"{len(jobs)} checkpoints remaining"
    )
    if jobs:
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(stage1_nv_relaxed)(
                field,nv,rr,t,Y[nv],E[nv],str(cp),args.stage1_max_nfev,
                args.scale_max,
            ) for nv,rr,cp in jobs
        )
    stage1=pd.concat(
        [pd.read_csv(ck1/f"nv_{nv:04d}.csv.gz") for nv in nvs],
        ignore_index=True,
    )
    work=v14.apply_updates(src,stage1)
    chosen=v14.choose_polish_rows(
        work,args.polish_per_order,args.polish_delta_bic,args.polish_cap_per_order
    )
    polish=work[work._row_id.isin(chosen)].copy()
    print(f"Saved Stage 2: robust polish {len(polish):,}/{len(work):,}")

    need=["_row_id","nv_index","model_order","theta_json","baseline",
          "orientation","site_key","v6_source_bic"]
    for j in (1,2,3):
        for c in ("site_id","f0_kHz","f1_kHz","kappa"):
            need.append(f"c13_{j}_{c}")
    pc=polish[[c for c in need if c in polish.columns]].copy()
    pc["stage1_theta_json"]=pc["theta_json"]
    precs={int(nv):pc[pc.nv_index==nv].to_dict("records")
           for nv in sorted(pc.nv_index.unique())}
    jobs=[]
    for nv,rr in precs.items():
        cp=ck2/f"nv_{nv:04d}.csv.gz"
        if not cp.exists(): jobs.append((nv,rr,cp))
    if jobs:
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(stage2_nv_relaxed)(
                field,nv,rr,t,Y[nv],E[nv],str(cp),args.scale_max
            ) for nv,rr,cp in jobs
        )
    fs=[ck2/f"nv_{nv:04d}.csv.gz" for nv in precs]
    s2=pd.concat([pd.read_csv(p) for p in fs if p.exists()],
                 ignore_index=True) if fs else pd.DataFrame()
    if len(s2): work=v14.apply_updates(work,s2)
    work["source_kind"]="saved_v14"
    return v14.rerank(work)


def local_pool(catalog,ori,site,freq_tol,pool_size):
    g=catalog[catalog.ori.map(tuple)==tuple(ori)].copy()
    g["df0"]=np.abs(g.f0_kHz.astype(float)-float(site["f0_kHz"]))
    g["df1"]=np.abs(g.f1_kHz.astype(float)-float(site["f1_kHz"]))
    g["freq_max_kHz"]=np.maximum(g.df0,g.df1)
    g["freq_rms_kHz"]=np.sqrt(0.5*(g.df0**2+g.df1**2))
    g=g[g.freq_max_kHz<=float(freq_tol)].drop_duplicates("site_id").copy()
    incumbent=int(site["site_id"])
    ids=[]

    def add(frame):
        for idx in frame.index:
            if idx not in ids: ids.append(idx)
            if len(ids)>=int(pool_size): return

    add(g[g.site_id.astype(int)==incumbent])
    near=g.sort_values(["freq_rms_kHz","freq_max_kHz"])
    add(near.head(max(2,int(pool_size)//2)))
    add(g.sort_values("kappa",ascending=False).head(max(2,int(pool_size)//3)))
    add(g.sort_values("kappa").head(max(2,int(pool_size)//3)))
    add(near.head(int(pool_size)))
    g=g.loc[ids[:int(pool_size)]].copy()
    out=[]
    for r in g.itertuples():
        out.append(dict(
            site_id=int(r.site_id),f0_kHz=float(r.f0_kHz),
            f1_kHz=float(r.f1_kHz),kappa=float(r.kappa),
            distance_A=float(getattr(r,"distance_A",np.nan)),
            orientation=tuple(ori),
            freq_rms_kHz=float(r.freq_rms_kHz),
            incumbent=bool(int(r.site_id)==incumbent),
        ))
    return out


def row_sites(row):
    out=[]
    for j in range(1,int(row.model_order)+1):
        out.append(dict(
            site_id=int(row[f"c13_{j}_site_id"]),
            f0_kHz=float(row[f"c13_{j}_f0_kHz"]),
            f1_kHz=float(row[f"c13_{j}_f1_kHz"]),
            kappa=float(row[f"c13_{j}_kappa"]),
            orientation=row.orientation,
        ))
    return out


def select_augment_bases(saved,args):
    chosen=[]
    for nv,g in saved.groupby("nv_index",sort=False):
        gg=g.sort_values("bic")
        chosen.extend(gg.head(args.augment_top_global)._row_id.astype(int))
        for order,h in gg[gg.model_order>0].groupby("model_order"):
            chosen.extend(h.head(args.augment_top_per_order)._row_id.astype(int))
    return saved[saved._row_id.isin(set(chosen)) & (saved.model_order>0)].copy()


def enumerate_local_candidates(base_row,catalog,args):
    n=int(base_row.model_order)
    ori=v6.load_backend(args.field).parse_orientation(base_row.orientation)
    inc=row_sites(base_row)
    pools=[local_pool(catalog,ori,s,args.freq_tol_khz,args.local_pool_size)
           for s in inc]
    out=[]; seen=set()
    for choice in itertools.product(*pools):
        ids=tuple(int(s["site_id"]) for s in choice)
        if len(set(ids))!=len(ids): continue
        nsub=sum(not s["incumbent"] for s in choice)
        if nsub<1 or nsub>args.max_local_substitutions: continue
        if ids in seen: continue
        seen.add(ids)
        out.append((ids,[dict(s) for s in choice],nsub))
    out.sort(key=lambda q:(
        q[2],
        np.sqrt(np.mean([s["freq_rms_kHz"]**2 for s in q[1]])),
    ))
    if args.max_augmented_per_base>0:
        out=out[:args.max_augmented_per_base]
    return out


def augmented_record(base_row,sites,new_id,nsub):
    r=base_row.copy()
    r["_row_id"]=int(new_id)
    r["site_key"]=str(tuple(int(s["site_id"]) for s in sites))
    r["source_kind"]="local_substitution"
    r["source_parent_site_key"]=str(base_row.site_key)
    r["local_substitution_count"]=int(nsub)
    r["source_baseline_seed"]=float(base_row.get("source_baseline_seed",base_row.baseline))
    for j in (1,2,3):
        if j<=len(sites):
            s=sites[j-1]
            r[f"c13_{j}_site_id"]=int(s["site_id"])
            r[f"c13_{j}_f0_kHz"]=float(s["f0_kHz"])
            r[f"c13_{j}_f1_kHz"]=float(s["f1_kHz"])
            r[f"c13_{j}_kappa"]=float(s["kappa"])
            if f"c13_{j}_distance_A" in r.index:
                r[f"c13_{j}_distance_A"]=float(s.get("distance_A",np.nan))
        else:
            for c in ("site_id","f0_kHz","f1_kHz","kappa","distance_A"):
                k=f"c13_{j}_{c}"
                if k in r.index: r[k]=np.nan
    return r
def fit_augmented_nv(field,nv,base_rows,t,y,e,catalog,args,checkpoint,start_id):
    patch_scale_max(args.scale_max)
    checkpoint=Path(checkpoint)
    if checkpoint.exists(): return str(checkpoint)
    base=v6.load_backend(field)
    rows=[]; seen=set(); next_id=int(start_id)
    with threadpool_limits(limits=1):
        for _,br in base_rows.sort_values("bic").iterrows():
            for ids,sites,nsub in enumerate_local_candidates(br,catalog,args):
                key=(int(br.model_order),ids)
                if key in seen: continue
                seen.add(key)
                rr=augmented_record(br,sites,next_id,nsub); next_id+=1
                try:
                    # First relax from the parent solution under the new frequencies.
                    q1=v14.projected_seed_fit(
                        base,t,y,e,rr,sites,args.augment_stage1_max_nfev
                    )
                    q2=v14.strong_fit(base,t,y,e,rr,sites,q1["theta"])
                    fit=q2 if q2 is not None else q1
                    upd=v14.fit_to_updates(base,t,rr,sites,fit,"v22_local_polish")
                    rec=rr.to_dict(); rec.update(upd)
                    rows.append(rec)
                except Exception as exc:
                    rec=rr.to_dict()
                    rec.update(fit_error=str(exc),bic=np.inf,chi2=np.inf,aicc=np.inf)
                    rows.append(rec)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(checkpoint,index=False,compression="gzip")
    return str(checkpoint)


def run_augmentation(field,saved,t,Y,E,args,outdir):
    bases=select_augment_bases(saved,args)
    catalog=v6.load_catalog(v6.load_backend(field))
    nvs=sorted(saved.nv_index.unique().astype(int))
    bynv={nv:bases[bases.nv_index==nv].copy() for nv in nvs}
    ck=outdir/"checkpoint_local_augmentation"
    stride=1_000_000
    jobs=[]
    for nv in nvs:
        cp=ck/f"nv_{nv:04d}.csv.gz"
        if not cp.exists():
            jobs.append((nv,bynv[nv],cp,10_000_000+nv*stride))
    print(
        f"\nLocal augmentation: {len(bases):,} parent candidates; "
        f"{len(jobs)} NV checkpoints remaining"
    )
    if jobs:
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(fit_augmented_nv)(
                field,nv,br,t,Y[nv],E[nv],catalog,args,str(cp),sid
            ) for nv,br,cp,sid in jobs
        )
    fs=[ck/f"nv_{nv:04d}.csv.gz" for nv in nvs]
    parts=[]
    empty_checkpoints=0
    unreadable_checkpoints=[]
    for p in fs:
        if not p.exists():
            continue
        try:
            q=pd.read_csv(p)
        except pd.errors.EmptyDataError:
            # A valid outcome: this NV had no admissible local substitutions.
            empty_checkpoints += 1
            continue
        except Exception as exc:
            unreadable_checkpoints.append((str(p),repr(exc)))
            continue
        if len(q):
            parts.append(q)
        else:
            empty_checkpoints += 1
    if unreadable_checkpoints:
        msg="; ".join(f"{p}: {err}" for p,err in unreadable_checkpoints[:5])
        raise RuntimeError(
            f"{len(unreadable_checkpoints)} augmentation checkpoints are unreadable: {msg}"
        )
    if empty_checkpoints:
        print(f"Local augmentation: {empty_checkpoints} empty NV checkpoints (no admissible substitutions)")
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def canonical_site_key(value):
    if value is None or (isinstance(value,float) and np.isnan(value)):
        return ()
    try:
        x=ast.literal_eval(str(value))
    except Exception:
        x=value
    if isinstance(x,(list,tuple,np.ndarray)):
        return tuple(sorted(int(v) for v in x))
    s=str(value).strip()
    if s in ("","()","nan","None"): return ()
    return tuple(sorted(int(v) for v in s.strip("()[] ").split(",") if str(v).strip()))


def dedup_and_rerank(saved,aug):
    d=pd.concat([saved,aug],ignore_index=True,sort=False) if len(aug) else saved.copy()
    d=d.copy()
    d["site_key_canonical"]=d.site_key.map(lambda x:str(canonical_site_key(x)))
    d["_dedup_key"]=(
        d.nv_index.astype(str)+"|"+d.model_order.astype(str)+"|"+
        d.site_key_canonical.astype(str)+"|"+d.orientation.astype(str)
    )
    d=d.sort_values(["_dedup_key","bic"]).drop_duplicates("_dedup_key",keep="first")
    d=d.drop(columns="_dedup_key")
    return v14.rerank(d.reset_index(drop=True))





def conditional_stats(y,e,pred,k):
    yy=np.asarray(y,float); ee=np.maximum(np.asarray(e,float),1e-12)
    chi2=float(np.sum(((yy-np.asarray(pred,float))/ee)**2))
    return chi2, float(chi2+int(k)*np.log(max(len(yy),2)))


def amplitude_diagnostics(final,t,Y,E,base):
    summary=[]; site_rows=[]
    winners=final[final.rank_global_bic==1].sort_values("nv_index")
    for _,rec in winners.iterrows():
        nv=int(rec.nv_index); n=int(rec.model_order)
        if n==0:
            summary.append(dict(
                nv_index=nv,amp_diag_delta_bic=np.nan,
                amp_diag_chi2_gain=np.nan,conditional_shared_scale=np.nan,
                free_site_scale_cv=np.nan,max_line_to_contrast=np.nan,
            ))
            continue
        th=np.asarray(json.loads(rec.theta_json),float)
        bg=th[:9]
        baseline,contrast,carrier=base.carrier_from_bg(t,bg)
        core=baseline-contrast*carrier
        sites=v14.sites_from_record(rec)
        ph=np.asarray(th[10:10+2*n],float).reshape(n,2)
        tt=np.asarray(t,float)
        X=[]
        for s,pair in zip(sites,ph):
            f0=float(s["f0_kHz"])/1000.0
            f1=float(s["f1_kHz"])/1000.0
            X.append(carrier*(
                np.cos(2*np.pi*f0*tt+pair[0])+
                np.cos(2*np.pi*f1*tt+pair[1])
            ))
        X=np.column_stack(X)
        expected=float(contrast)*np.asarray([float(s["kappa"]) for s in sites])/4.0
        target=np.asarray(Y[nv],float)-core
        ee=base.safe_err(E[nv]); wt=1.0/ee
        g=X@expected
        den=float(np.sum((g*wt)**2))
        shared=max(0.0,float(np.sum((g*wt)*(target*wt)))/den) if den>0 else 0.0
        pred_s=core+shared*g
        free=lsq_linear(X*wt[:,None],target*wt,bounds=(0.0,np.inf))
        amps=np.asarray(free.x,float)
        pred_f=core+X@amps
        chi_s,bic_s=conditional_stats(Y[nv],ee,pred_s,1)
        chi_f,bic_f=conditional_stats(Y[nv],ee,pred_f,n)
        scales=np.divide(amps,expected,out=np.full(n,np.nan),where=expected>1e-12)
        good=scales[np.isfinite(scales)&(scales>0)]
        cv=float(np.std(good)/np.mean(good)) if len(good)>1 and np.mean(good)>0 else 0.0
        max_ratio=float(np.max(float(rec.visibility_scale)*np.asarray(
            [float(s["kappa"]) for s in sites]
        )/4.0))
        summary.append(dict(
            nv_index=nv,amp_diag_delta_bic=float(bic_f-bic_s),
            amp_diag_chi2_gain=float(chi_s-chi_f),
            conditional_shared_scale=float(shared),free_site_scale_cv=cv,
            max_line_to_contrast=max_ratio,
        ))
        for j,(s,a,sc) in enumerate(zip(sites,amps,scales),1):
            site_rows.append(dict(
                nv_index=nv,site_slot=j,site_id=int(s["site_id"]),
                kappa=float(s["kappa"]),expected_amp_scale1=float(expected[j-1]),
                free_amp=float(a),free_effective_scale=float(sc),
                conditional_shared_scale=float(shared),
            ))
    return pd.DataFrame(summary),pd.DataFrame(site_rows)


def order_evidence(final):
    rows=[]
    for nv,g in final.groupby("nv_index",sort=True):
        winner=float(g.bic.min())
        for order,h in g.groupby("model_order",sort=True):
            q=h.sort_values("bic").iloc[0]
            rows.append(dict(
                nv_index=int(nv),model_order=int(order),
                best_site_key=str(q.site_key),
                best_site_key_canonical=str(canonical_site_key(q.site_key)),
                source_kind=str(q.get("source_kind","unknown")),
                bic=float(q.bic),delta_bic_vs_global_best=float(q.bic-winner),
                red_chi2=float(q.red_chi2),
                shared_scale=float(q.visibility_scale) if int(order)>0 else np.nan,
            ))
    return pd.DataFrame(rows)


def qc_summary(final,old,base,args,amp_diag):
    rows=[]
    amap=amp_diag.set_index("nv_index") if len(amp_diag) else pd.DataFrame()
    for nv,g in final.groupby("nv_index",sort=True):
        g=g.sort_values("bic")
        w=g.iloc[0]; r=g.iloc[1] if len(g)>1 else None
        oldw=old[(old.nv_index==nv)&(old.rank_global_bic==1)].iloc[0]
        n=int(w.model_order)
        scale=float(w.visibility_scale) if n>0 and pd.notna(w.visibility_scale) else np.nan
        seed=float(w.get("source_baseline_seed",w.baseline))
        contrast_ub=min(float(base.BG_UB[1]),max(0.05,seed-0.01))
        ad=amap.loc[int(nv)] if len(amap) and int(nv) in amap.index else None
        amp_dbic=float(ad.amp_diag_delta_bic) if ad is not None and pd.notna(ad.amp_diag_delta_bic) else np.nan
        amp_cv=float(ad.free_site_scale_cv) if ad is not None and pd.notna(ad.free_site_scale_cv) else np.nan
        max_line=float(ad.max_line_to_contrast) if ad is not None and pd.notna(ad.max_line_to_contrast) else np.nan
        cshared=float(ad.conditional_shared_scale) if ad is not None and pd.notna(ad.conditional_shared_scale) else np.nan

        flags=[]
        if float(w.red_chi2)>3.0: flags.append("high_redchi2")
        if np.isfinite(scale) and scale>10: flags.append("high_scale_gt10")
        elif np.isfinite(scale) and scale>3: flags.append("scale_gt3")
        if np.isfinite(scale) and scale>=args.scale_max-0.2: flags.append("scale_bound")
        if np.isfinite(max_line) and max_line>2: flags.append("line_amp_gt2contrast")
        elif np.isfinite(max_line) and max_line>1: flags.append("line_amp_gt_contrast")
        if np.isfinite(amp_dbic) and amp_dbic<=-6: flags.append("relative_kappa_mismatch")
        if abs(float(w.contrast)-contrast_ub)<2e-3: flags.append("contrast_upper")
        if abs(float(w.width_slope)-float(base.BG_UB[7]))<2e-3: flags.append("slope_upper")
        if abs(float(w.width_slope)-float(base.BG_LB[7]))<2e-3: flags.append("slope_lower")
        if min(abs(float(w.revival_chirp)-float(base.BG_LB[8])),
               abs(float(w.revival_chirp)-float(base.BG_UB[8])))<2e-3:
            flags.append("chirp_bound")
        margin=float(r.bic-w.bic) if r is not None else np.inf
        if margin<2: flags.append("bic_ambiguous")

        newcanon=str(canonical_site_key(w.site_key))
        oldcanon=str(canonical_site_key(oldw.site_key))
        rows.append(dict(
            nv_index=int(nv),model_order=n,site_key=str(w.site_key),
            site_key_canonical=newcanon,
            source_kind=str(w.get("source_kind","unknown")),
            bic=float(w.bic),red_chi2=float(w.red_chi2),
            shared_scale=scale,contrast=float(w.contrast),
            scale_times_contrast=(scale*float(w.contrast) if np.isfinite(scale) else np.nan),
            max_line_to_contrast=max_line,
            amp_diag_delta_bic=amp_dbic,
            free_site_scale_cv=amp_cv,
            conditional_shared_scale=cshared,
            delta_bic_to_second=margin,
            changed_order=bool(n!=int(oldw.model_order)),
            changed_site=bool(newcanon!=oldcanon),
            old_order=int(oldw.model_order),old_site_key=str(oldw.site_key),
            old_site_key_canonical=oldcanon,
            delta_bic_vs_old_v14=float(w.bic-oldw.bic),
            qc_flags=";".join(flags),
        ))
    return pd.DataFrame(rows)
def make_report(path,field,t,Y,E,final,qc,base,args):
    with PdfPages(path) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(11,8.5))
        q=qc
        axes[0,0].bar([0,1,2,3],
            q.model_order.value_counts().reindex([0,1,2,3],fill_value=0))
        axes[0,0].set(title=f"{field} winner model orders",xlabel="N",ylabel="NV count")
        axes[0,1].hist(q.shared_scale[np.isfinite(q.shared_scale)],bins=30)
        axes[0,1].axvline(3,ls="--",lw=.8); axes[0,1].axvline(10,ls=":",lw=.8)
        axes[0,1].set(title="Winner shared scale",xlabel="s_NV")
        axes[1,0].hist(q.delta_bic_to_second.replace(np.inf,np.nan).dropna(),bins=30)
        axes[1,0].set(title="BIC margin to runner-up",xlabel="Delta BIC")
        labels=["order changed","site changed","local winner","QC flagged"]
        vals=[
            int(q.changed_order.sum()),int(q.changed_site.sum()),
            int((q.source_kind=="local_substitution").sum()),
            int((q.qc_flags!="").sum()),
        ]
        axes[1,1].bar(np.arange(4),vals)
        axes[1,1].set_xticks(np.arange(4),labels,rotation=25,ha="right")
        axes[1,1].set(title=f"V22 summary ({len(q)} NVs)",ylabel="count")
        fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

        # One page per NV: winner, runner-up, residual, concise QC text.
        for nv in sorted(q.nv_index.astype(int)):
            g=final[final.nv_index==nv].sort_values("bic")
            w=g.iloc[0]; r=g.iloc[1] if len(g)>1 else None
            fig,axes=plt.subplots(2,1,figsize=(10,7.5),height_ratios=[2.2,1.0])
            ax=axes[0]
            yy=np.asarray(Y[nv],float); ee=base.safe_err(E[nv])
            ax.errorbar(t,yy,yerr=ee,fmt="o",ms=3,capsize=1,label="data")
            for rec,label,lw in [(w,"winner",1.5),(r,"runner-up",1.0)]:
                if rec is None: continue
                sites=v14.sites_from_record(rec)
                th=np.asarray(json.loads(rec.theta_json),float)
                pred=v6.v6_model(base,t,th,sites)
                ax.plot(t,pred,lw=lw,label=f"{label}: N{int(rec.model_order)} {rec.site_key}")
            ax.grid(alpha=.2); ax.legend(fontsize=7)
            ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="Normalized signal")
            qq=q[q.nv_index==nv].iloc[0]
            ax.set_title(
                f"NV{nv} | BIC={w.bic:.1f} | redchi2={w.red_chi2:.2f} | "
                f"s={w.visibility_scale:.2f} | margin={qq.delta_bic_to_second:.1f}"
            )
            ax=axes[1]
            sites=v14.sites_from_record(w)
            th=np.asarray(json.loads(w.theta_json),float)
            pred=v6.v6_model(base,t,th,sites)
            ax.plot(t,(yy-pred)/ee,"o-",ms=3,lw=.8)
            ax.axhline(0,ls="--",lw=.8); ax.grid(alpha=.2)
            ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="winner residual / sigma")
            ax.text(.01,.04,
                f"source={qq.source_kind}; old={qq.old_site_key}; flags={qq.qc_flags or 'none'}",
                transform=ax.transAxes,fontsize=8)
            fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def run(args):
    t0=time.time()
    patch_scale_max(args.scale_max)
    field=args.field
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    src_path=find_v14_file(field)
    old=pd.read_csv(src_path)
    src=source_prepare(old)
    if args.nv:
        wanted={int(x) for x in args.nv.split(",") if x.strip()}
        src=src[src.nv_index.isin(wanted)].copy()
        old=old[old.nv_index.isin(wanted)].copy()

    outdir=Path(args.output_dir) if args.output_dir else DEFAULT_ROOT/field/config_tag(args)
    outdir.mkdir(parents=True,exist_ok=True)

    saved=run_saved_stage(field,src,t,Y,E,args,outdir)
    aug=run_augmentation(field,saved,t,Y,E,args,outdir)
    final=dedup_and_rerank(saved,aug)
    amp_diag,amp_sites=amplitude_diagnostics(final,t,Y,E,base)
    order_tab=order_evidence(final)
    qc=qc_summary(final,old,base,args,amp_diag)

    winners=final[final.rank_global_bic==1].sort_values("nv_index").copy()
    top=final.sort_values(["nv_index","bic"]).groupby("nv_index").head(args.save_top_n)
    final.to_csv(outdir/"v22_candidate_fits.csv.gz",index=False,compression="gzip")
    winners.to_csv(outdir/"v22_winners.csv",index=False)
    top.to_csv(outdir/"v22_top_candidates.csv",index=False)
    order_tab.to_csv(outdir/"v22_order_evidence.csv",index=False)
    amp_diag.to_csv(outdir/"v22_amplitude_diagnostics.csv",index=False)
    amp_sites.to_csv(outdir/"v22_amplitude_diagnostic_sites.csv",index=False)
    qc.to_csv(outdir/"v22_qc_summary.csv",index=False)

    make_report(
        outdir/"v22_full_analysis_report.pdf",field,t,Y,E,final,qc,base,args
    )

    meta=dict(
        field=field,source_v14=str(src_path),scale_max=float(args.scale_max),
        freq_tol_khz=float(args.freq_tol_khz),local_pool_size=int(args.local_pool_size),
        max_local_substitutions=int(args.max_local_substitutions),
        augment_top_per_order=int(args.augment_top_per_order),
        augment_top_global=int(args.augment_top_global),
        n_saved_candidates=int(len(saved)),n_augmented_candidates=int(len(aug)),
        n_final_candidates=int(len(final)),n_nvs=int(qc.nv_index.nunique()),
        n_local_winners=int((qc.source_kind=="local_substitution").sum()),
        n_order_changes=int(qc.changed_order.sum()),
        n_site_changes=int(qc.changed_site.sum()),
        n_relative_kappa_mismatch=int((qc.amp_diag_delta_bic<=-6).fillna(False).sum()),
        n_line_amp_gt_contrast=int((qc.max_line_to_contrast>1).fillna(False).sum()),
        n_line_amp_gt2contrast=int((qc.max_line_to_contrast>2).fillna(False).sum()),
        elapsed_minutes=float((time.time()-t0)/60),
        amplitude_model="shared_scale_times_contrast_kappa_over_4",
        background_model="beta2_taper0",
    )
    with open(outdir/"v22_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)

    print("\nV22 COMPLETE")
    print("winner orders:",qc.model_order.value_counts().sort_index().to_dict())
    print("order changes:",meta["n_order_changes"],
          "site changes:",meta["n_site_changes"],
          "local-substitution winners:",meta["n_local_winners"])
    print("scale >3:",int((qc.shared_scale>3).sum()),
          "scale >10:",int((qc.shared_scale>10).sum()))
    print("relative-kappa mismatches:",meta["n_relative_kappa_mismatch"],
          "line amp > contrast:",meta["n_line_amp_gt_contrast"])
    print("QC flagged:",int((qc.qc_flags!="").sum()))
    print("output:",outdir)
    return final,qc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G"],default="52G")
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--scale-max",type=float,default=30.0)
    ap.add_argument("--stage1-max-nfev",type=int,default=1200)
    ap.add_argument("--polish-per-order",type=int,default=3)
    ap.add_argument("--polish-delta-bic",type=float,default=12.0)
    ap.add_argument("--polish-cap-per-order",type=int,default=8)
    ap.add_argument("--freq-tol-khz",type=float,default=12.0)
    ap.add_argument("--local-pool-size",type=int,default=8)
    ap.add_argument("--max-local-substitutions",type=int,choices=[1,2],default=1)
    ap.add_argument("--max-augmented-per-base",type=int,default=40)
    ap.add_argument("--augment-top-per-order",type=int,default=2)
    ap.add_argument("--augment-top-global",type=int,default=4)
    ap.add_argument("--augment-stage1-max-nfev",type=int,default=1800)
    ap.add_argument("--save-top-n",type=int,default=10)
    ap.add_argument("--nv",type=str,default=None,
                    help="optional comma-separated NV indices for smoke tests")
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
