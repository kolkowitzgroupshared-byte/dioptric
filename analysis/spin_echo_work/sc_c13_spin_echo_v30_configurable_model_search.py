"""V30 configurable physics-constrained multi-13C spin-echo search.

Designed as a reusable successor to V6/V7 after V22-V29 troubleshooting.

Key controls:
  --max-spins N               maximum resolved 13C spins (not hard-limited to 3)
  --max-sites K               screened lattice sites retained per orientation
  --max-per-family K          aliases retained per near-degenerate frequency family
  --beam-width K              parents retained when building N>=2 models
  --order-candidate-cap K      max discrete models actually fitted at each N
  --physical-snr-min X        physical detectability screen
  --family-tol-khz X          frequency-family span
  --scale-max X               shared visibility-scale ceiling
  --background MODEL          reduced/full/beta-free/taper-free
  --orientation-mode MODE     auto/assigned/fit-quality/all

For small candidate pools, N=2 can be exhaustive.  For large pools (e.g. 100
sites) higher orders use progressive beam search rather than combinatorial
brute force.  Every NV is checkpointed independently.
"""
from __future__ import annotations

import argparse,ast,json,itertools,time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

ROOT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo")
OUTROOT=ROOT/r"spin_echo_v30_configurable_search\2026_09"
V22ROOT=ROOT/r"c13_spin_echo_v22_full_physical_rerank\2026_09"
V29=ROOT/r"spin_echo_v29_final_consensus_52G\2026_09"
CFG="smax30_tol12_pool8_sub1_topO2_topG4_cap40"
def safe_tag(x):
    return str(x).replace(".","p").replace("-","m")


def fixed_background(mode):
    return {
        "reduced":{5:2.0,6:0.0},
        "full":{},
        "beta-free":{6:0.0},
        "taper-free":{5:2.0},
    }[mode]


def parse_nv_arg(text,nmax):
    if not text:
        return list(range(nmax))
    out=[]
    for token in str(text).split(","):
        token=token.strip()
        if not token: continue
        if "-" in token:
            a,b=map(int,token.split("-",1)); out.extend(range(a,b+1))
        else:
            out.append(int(token))
    return sorted({x for x in out if 0<=x<nmax})


def load_production_protected(field):
    out={}
    try:
        if field=="52G":
            d=pd.read_csv(V29/"v29_final_consensus_212NV.csv")
            for r in d.itertuples():
                ids=tuple(int(x) for x in ast.literal_eval(str(r.final_site_key)))
                ori=tuple(int(x) for x in ast.literal_eval(str(r.orientation)))
                out[int(r.nv_index)]={"orientation":ori,"site_ids":ids}
        else:
            p=V22ROOT/"49G"/CFG/"v22_winners.csv"
            d=pd.read_csv(p)
            for r in d.itertuples():
                ids=tuple(int(x) for x in ast.literal_eval(str(r.site_key)))
                ori=tuple(int(x) for x in ast.literal_eval(str(r.orientation)))
                out[int(r.nv_index)]={"orientation":ori,"site_ids":ids}
    except Exception as exc:
        print("[WARN] production protection unavailable:",exc)
    return out
def fit_model(base,t,y,e,sites,seeds,fixed,scale_max,robust_nfev,final_nfev):
    v6.VISIBILITY_SCALE_MAX=float(scale_max)
    n=len(sites)
    expected=9 if n==0 else 10+2*n
    good=[np.asarray(s,float) for s in seeds if np.asarray(s).size==expected]
    if not good:
        return None
    baseline_seed=float(np.nanmedian([s[0] for s in good]))
    lb,ub=v6.v6_theta_bounds(base,n,baseline_seed,float(base.t2_upper_us(t)))
    ee=base.safe_err(e)
    free=[i for i in range(expected) if i not in fixed]

    def project(seed):
        s=np.asarray(seed,float).copy()
        for i,val in fixed.items(): s[int(i)]=float(val)
        return np.clip(s,lb+1e-9,ub-1e-9)

    def expand(x,template):
        th=template.copy(); th[free]=x
        for i,val in fixed.items(): th[int(i)]=float(val)
        return th

    robust=[]
    for seed in good:
        s=project(seed)
        def resid(x):
            th=expand(x,s)
            return (np.asarray(y,float)-v6.v6_model(base,t,th,sites))/ee
        try:
            rr=least_squares(resid,s[free],bounds=(lb[free],ub[free]),
                             loss="soft_l1",f_scale=1.0,max_nfev=int(robust_nfev),
                             x_scale="jac")
            th=expand(rr.x,s); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            robust.append((float(st["chi2"]),th))
        except Exception:
            pass
    if not robust:
        return None
    robust.sort(key=lambda z:z[0])
    finals=[]
    for _,seed in robust[:min(4,len(robust))]:
        s=project(seed)
        def resid(x):
            th=expand(x,s)
            return (np.asarray(y,float)-v6.v6_model(base,t,th,sites))/ee
        try:
            rr=least_squares(resid,s[free],bounds=(lb[free],ub[free]),
                             loss="linear",max_nfev=int(final_nfev),
                             ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac")
            th=expand(rr.x,s); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            finals.append(dict(theta=th,pred=pred,sites=tuple(sites),
                               site_key=tuple(sorted(int(q["site_id"]) for q in sites)),
                               t2_limit_us=float(base.t2_upper_us(t)),**st))
        except Exception:
            pass
    return min(finals,key=lambda q:(q["bic"],q["chi2"])) if finals else None


def screened_sites(base,nv,t,y,e,attempts,catalog,args,protected,orientation_map):
    ad,_=v6.attempt_lookup_rows(base,attempts,nv)
    if ad.empty:
        raise RuntimeError(f"NV {nv}: no successful legacy attempts")
    oris,osrc=v6.choose_orientations(
        base,nv,ad,args.orientation_mode,orientation_map)
    bgfit,selected,site_rows,fam_rows=v6.prepare_families(
        base,nv,t,y,e,ad,catalog,args.physical_snr_min,args.family_tol_khz,
        args.max_sites,args.max_per_family,protected.get(int(nv),{}),oris,osrc)
    sites=[]
    for fr in selected:
        s=v6.site_dict(fr,str(fr.get("family_id","")))
        for k in ("amp_seed","phi0_seed","phi1_seed","physical_snr","matched_delta_chi2"):
            if k in fr: s[k]=float(fr[k])
        sites.append(s)
    return bgfit,sites,site_rows,fam_rows,oris,osrc
def bg_seed(bgfit,fixed):
    bg=np.asarray(bgfit["theta"],float)[:9].copy()
    for i,val in fixed.items(): bg[int(i)]=float(val)
    return bg


def single_seeds(bg,site,scale_max):
    contrast=max(abs(float(bg[1])),1e-8)
    denom=contrast*max(float(site["kappa"]),0.0)/4.0
    amp=abs(float(site.get("amp_seed",0.05)))
    est=amp/denom if denom>1e-12 else 1.0
    scales=[est,1.0,3.0,10.0,min(20.0,float(scale_max))]
    out=[]
    for s in scales:
        s=float(np.clip(s,1e-6,float(scale_max)-1e-6))
        out.append(np.r_[bg,s,float(site.get("phi0_seed",0.0)),
                         float(site.get("phi1_seed",0.0))])
    return out


def phase_map(fit):
    th=np.asarray(fit["theta"],float); out={}; j=10
    for s in fit["sites"]:
        out[int(s["site_id"])]=(float(th[j]),float(th[j+1])); j+=2
    return out


def expand_parent_seed(parent,sites,new_site,scale_override=None):
    th=np.asarray(parent["theta"],float)
    pm=phase_map(parent)
    phases=[]
    for s in sites:
        sid=int(s["site_id"])
        phases.extend(pm.get(sid,(
            float(s.get("phi0_seed",0.0)),float(s.get("phi1_seed",0.0)))))
    scale=float(th[9]) if scale_override is None else float(scale_override)
    return np.r_[th[:9],scale,phases]


def canonical_key(sites):
    ori=tuple(int(x) for x in sites[0]["orientation"]) if sites else ()
    ids=tuple(sorted(int(s["site_id"]) for s in sites))
    return ori,ids
def fit_one_nv(field,nv,t,y,e,attempts,catalog,args,protected,orientation_map,ckdir):
    cp=ckdir/f"nv_{nv:04d}_candidates.csv.gz"
    sp=ckdir/f"nv_{nv:04d}_sites.csv.gz"
    fp=ckdir/f"nv_{nv:04d}_families.csv.gz"
    if cp.exists() and sp.exists() and fp.exists():
        return str(cp),str(sp),str(fp)

    base=v6.load_backend(field)
    v6.VISIBILITY_SCALE_MAX=float(args.scale_max)
    fixed=fixed_background(args.background)
    bgfit,sites,site_rows,fam_rows,oris,osrc=screened_sites(
        base,nv,t,y,e,attempts,catalog,args,protected,orientation_map)
    bg=bg_seed(bgfit,fixed)

    fits_by_order={}
    f0=fit_model(base,t,y,e,[],[bg],fixed,args.scale_max,
                 args.robust_max_nfev,args.final_max_nfev)
    if f0 is None:
        raise RuntimeError(f"NV {nv}: N0 fit failed")
    fits_by_order[0]=[f0]

    singles=[]
    for s in sites:
        f=fit_model(base,t,y,e,[s],single_seeds(bg,s,args.scale_max),fixed,
                    args.scale_max,args.robust_max_nfev,args.final_max_nfev)
        if f is not None: singles.append(f)
    singles.sort(key=lambda q:(q["bic"],q["red_chi2"]))
    fits_by_order[1]=singles
    print(f"[NV {nv:3d}] screened={len(sites)} singles={len(singles)}")

    for order in range(2,args.max_spins+1):
        prev=fits_by_order.get(order-1,[])
        if not prev: break
        byori={}
        for f in prev:
            ori=tuple(f["sites"][0]["orientation"])
            byori.setdefault(ori,[]).append(f)
        single_by_ori={}
        for f in singles:
            ori=tuple(f["sites"][0]["orientation"])
            single_by_ori.setdefault(ori,[]).append(f)

        proposals={}
        for ori,parents0 in byori.items():
            parents=sorted(parents0,key=lambda q:q["bic"])[:args.beam_width]
            pool=single_by_ori.get(ori,[])
            if args.expand_sites>0: pool=pool[:args.expand_sites]

            exhaustive=(order==2 and (
                args.pair_mode=="exhaustive" or
                (args.pair_mode=="auto" and len(pool)<=args.exhaustive_pair_limit)
            ))
            if exhaustive:
                for a,b in itertools.combinations(pool,2):
                    ss=[a["sites"][0],b["sites"][0]]
                    key=canonical_key(ss)
                    proposals[key]=(float(a["bic"]+b["bic"]),None,ss)
            else:
                for parent in parents:
                    have={int(s["site_id"]) for s in parent["sites"]}
                    for sf in pool:
                        s=sf["sites"][0]
                        if int(s["site_id"]) in have: continue
                        ss=list(parent["sites"])+[s]
                        key=canonical_key(ss)
                        score=float(parent["bic"]+sf["bic"])
                        old=proposals.get(key)
                        if old is None or score<old[0]:
                            proposals[key]=(score,parent,ss)
        prop=sorted(proposals.values(),key=lambda z:z[0])
        prop=prop[:args.order_candidate_cap]
        cur=[]
        for _,parent,ss in prop:
            if parent is None:
                # Exhaustive N2: use the corresponding single fits as parents.
                ids={int(s["site_id"]) for s in ss}
                ps=[f for f in singles if int(f["sites"][0]["site_id"]) in ids]
                bestp=min(ps,key=lambda q:q["bic"])
                seeds=[v6.seed_from_parent_fits(ss,ps)]
            else:
                seeds=[expand_parent_seed(parent,ss,ss[-1])]
            seeds=[s for s in seeds if s is not None]
            # Alternative scale starts help avoid the large-scale/background basin.
            if seeds:
                s0=np.asarray(seeds[0],float)
                for sv in (3.0,10.0,min(20.0,args.scale_max)):
                    q=s0.copy(); q[9]=np.clip(sv,1e-6,args.scale_max-1e-6)
                    seeds.append(q)
            f=fit_model(base,t,y,e,ss,seeds,fixed,args.scale_max,
                        args.robust_max_nfev,args.final_max_nfev)
            if f is not None: cur.append(f)
        cur.sort(key=lambda q:(q["bic"],q["red_chi2"]))
        fits_by_order[order]=cur
        print(f"[NV {nv:3d}] N{order}: proposals={len(prop)} fits={len(cur)}")
        if not cur: break

    rows=[]
    for order,fits in fits_by_order.items():
        for rank,f in enumerate(sorted(fits,key=lambda q:q["bic"]),1):
            row=v6.v6_public_row(base,nv,order,rank,f)
            row=v6.augment_row(row,f)
            row["orientation_source"]=osrc
            row["orientation_candidates"]=str(tuple(oris))
            row["background_model_requested"]=args.background
            row["beam_width"]=args.beam_width
            row["max_sites_requested"]=args.max_sites
            row["max_spins_requested"]=args.max_spins
            rows.append(row)
    d=pd.DataFrame(rows)
    if d.empty: raise RuntimeError(f"NV {nv}: no candidate fits")
    d["rank_global_bic"]=d.bic.rank(method="first").astype(int)
    d["rank_global_aicc"]=d.aicc.rank(method="first").astype(int)
    d["rank_global_redchi"]=d.red_chi2.rank(method="first").astype(int)

    cp.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(cp,index=False,compression="gzip")
    pd.DataFrame(site_rows).to_csv(sp,index=False,compression="gzip")
    pd.DataFrame(fam_rows).to_csv(fp,index=False,compression="gzip")
    best=d.sort_values("bic").iloc[0]
    print(f"[NV {nv:3d}] WIN N{int(best.model_order)} {best.site_key} "
          f"BIC={best.bic:.2f} redchi2={best.red_chi2:.2f}")
    return str(cp),str(sp),str(fp)
def plot_nv(pdf,field,nv,t,y,e,candidates,site_rows):
    base=v6.load_backend(field)
    q=candidates[candidates.nv_index==nv].copy()
    bests=q.sort_values("bic").groupby("model_order",as_index=False).first()
    dense=np.linspace(float(t.min()),float(t.max()),1600)
    fig,ax=plt.subplots(2,2,figsize=(11,8.5))
    ax[0,0].errorbar(t,y,yerr=e,fmt=".",ms=2.7,lw=.5,alpha=.55,label="data")
    for _,r in bests.sort_values("model_order").iterrows():
        sites=v14.sites_from_record(r)
        th=np.asarray(json.loads(str(r.theta_json)),float)
        pred=v6.v6_model(base,dense,th,sites)
        ax[0,0].plot(dense,pred,lw=1,label=f"N{int(r.model_order)} {r.site_key}")
    ax[0,0].legend(fontsize=6.5,ncol=2)
    ax[0,0].set(xlabel="evolution time (us)",ylabel="normalized signal",
                title=f"{field} NV{nv}: best fit per model order")

    db=bests.bic-bests.bic.min()
    ax[0,1].plot(bests.model_order,db,"o-")
    ax[0,1].axhline(2,ls="--",lw=.7); ax[0,1].axhline(6,ls="--",lw=.7)
    ax[0,1].set(xlabel="N",ylabel="Delta BIC",title="Model-order evidence")

    ax[1,0].plot(bests.model_order,bests.red_chi2,"o-")
    ax[1,0].axhline(2,ls="--",lw=.7); ax[1,0].axhline(3,ls="--",lw=.7)
    ax[1,0].set(xlabel="N",ylabel="reduced chi-square",title="Fit quality")
    s=site_rows[site_rows.nv_index==nv].copy()
    if len(s):
        ax[1,1].scatter(s.f1_kHz,s.f0_kHz,
                        s=8+25*np.nan_to_num(s.kappa),alpha=.18)
    winner=q.sort_values("bic").iloc[0]
    for sid in ast.literal_eval(str(winner.site_key)):
        hit=s[s.site_id.astype(int)==int(sid)]
        if len(hit):
            r=hit.iloc[0]
            ax[1,1].scatter([r.f1_kHz],[r.f0_kHz],s=70,marker="*")
            ax[1,1].annotate(str(int(sid)),(r.f1_kHz,r.f0_kHz),fontsize=8,
                             xytext=(3,3),textcoords="offset points")
    ax[1,1].set(xlabel="f_minus (kHz)",ylabel="f_plus (kHz)",
                title="Screened C13 sites; winner highlighted")
    fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def config_tag(args):
    return (
        f"{args.field}_{args.background}_N{args.max_spins}_S{args.max_sites}_"
        f"F{args.max_per_family}_beam{args.beam_width}_cap{args.order_candidate_cap}_"
        f"snr{safe_tag(args.physical_snr_min)}_tol{safe_tag(args.family_tol_khz)}_"
        f"smax{safe_tag(args.scale_max)}_ori{args.orientation_mode.replace('-','')}"
    )


def run(args):
    base=v6.load_backend(args.field)
    _,checkpoint,_,prefix=base.discover_paths()
    t,Y,E=base.load_data(checkpoint)
    attempts_path=Path(str(prefix)+"_all_attempts.csv.gz")
    usecols=["nv_index","status","site_id","orientation","kappa","distance_A",
             "f0_kHz","f1_kHz","red_chi2","aicc","score_primary",
             "score_amp_tie","popt_json"]
    attempts=pd.read_csv(attempts_path,usecols=lambda c:c in usecols)
    catalog=v6.load_catalog(base)
    _,_,legacy_protected=v6.load_incumbents(prefix)
    protected=dict(legacy_protected)
    if args.protect_production_sites:
        protected.update(load_production_protected(args.field))
    embedded=v6.load_assigned_orientations(args.field,Y.shape[0])
    external=v6.load_orientation_map(args.orientation_map)
    orientation_map=dict(embedded); orientation_map.update(external)
    nvs=parse_nv_arg(args.nv,Y.shape[0])

    outdir=Path(args.output_dir)/config_tag(args)
    ckdir=outdir/"checkpoints"; outdir.mkdir(parents=True,exist_ok=True)
    print("="*100)
    print("V30 CONFIGURABLE SEARCH",config_tag(args))
    print("NVs:",len(nvs),"max spins:",args.max_spins,"max sites:",args.max_sites)
    print("background:",args.background,"scale max:",args.scale_max)
    print("beam:",args.beam_width,"order cap:",args.order_candidate_cap,
          "expand sites:",args.expand_sites or "all")
    def task(nv):
        return fit_one_nv(args.field,nv,t,Y[nv],E[nv],attempts,catalog,args,
                          protected,orientation_map,ckdir)
    with threadpool_limits(limits=1):
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(task)(nv) for nv in nvs)

    cparts=[]; sparts=[]; fparts=[]
    empty_site_checkpoints=0
    empty_family_checkpoints=0
    for nv in nvs:
        cp=ckdir/f"nv_{nv:04d}_candidates.csv.gz"
        sp=ckdir/f"nv_{nv:04d}_sites.csv.gz"
        fp=ckdir/f"nv_{nv:04d}_families.csv.gz"

        # Candidate checkpoints must always contain at least the N=0 fit.
        cparts.append(pd.read_csv(cp))

        # Zero screened sites/families is a valid physical outcome. Pandas raises
        # EmptyDataError on the headerless files produced by DataFrame([]).to_csv().
        try:
            q=pd.read_csv(sp)
            if len(q): sparts.append(q)
            else: empty_site_checkpoints += 1
        except pd.errors.EmptyDataError:
            empty_site_checkpoints += 1

        try:
            q=pd.read_csv(fp)
            if len(q): fparts.append(q)
            else: empty_family_checkpoints += 1
        except pd.errors.EmptyDataError:
            empty_family_checkpoints += 1

    if empty_site_checkpoints:
        print(f"Aggregation: {empty_site_checkpoints} empty site checkpoints (valid zero-site NVs)")
    if empty_family_checkpoints:
        print(f"Aggregation: {empty_family_checkpoints} empty family checkpoints (valid zero-family NVs)")

    cand=pd.concat(cparts,ignore_index=True)
    sites=pd.concat(sparts,ignore_index=True) if sparts else pd.DataFrame()
    fams=pd.concat(fparts,ignore_index=True) if fparts else pd.DataFrame()
    cand=base.enrich_candidates_with_catalog(cand,base.load_catalog_lookup())
    cand.to_csv(outdir/"v30_candidate_fits.csv.gz",index=False,compression="gzip")
    sites.to_csv(outdir/"v30_screened_sites.csv.gz",index=False,compression="gzip")
    fams.to_csv(outdir/"v30_frequency_families.csv.gz",index=False,compression="gzip")

    winners=cand.sort_values("bic").groupby("nv_index",as_index=False).first()
    orderbest=(cand.sort_values("bic")
               .groupby(["nv_index","model_order"],as_index=False).first())
    winners.to_csv(outdir/"v30_winners.csv",index=False)
    orderbest.to_csv(outdir/"v30_best_by_order.csv",index=False)

    with PdfPages(outdir/"v30_order_comparison_report.pdf") as pdf:
        fig,ax=plt.subplots(1,2,figsize=(11,5))
        oc=winners.model_order.value_counts().sort_index()
        ax[0].bar(oc.index.astype(str),oc.values)
        ax[0].set(title="Global BIC winner order",xlabel="N",ylabel="NV count")
        ax[1].hist(winners.red_chi2,bins=30)
        ax[1].axvline(2,ls="--",lw=.7); ax[1].axvline(3,ls="--",lw=.7)
        ax[1].set(title="Winner fit quality",xlabel="reduced chi-square",ylabel="NV count")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for nv in nvs:
            plot_nv(pdf,args.field,nv,t,Y[nv],E[nv],cand,sites)

    meta=vars(args).copy()
    meta.update(
        n_nv=len(nvs),
        winner_order_counts={str(k):int(v) for k,v in oc.items()},
        note="Beam search is used to control combinatorics; max-sites is the screened pool size, not an instruction to brute-force all combinations.",
    )
    with open(outdir/"v30_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print("V30 COMPLETE")
    print("winner orders:",meta["winner_order_counts"])
    print("output:",outdir)
    return cand,winners
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=("49G","52G"),required=True)
    ap.add_argument("--nv",default=None,help="e.g. 0,1,5-12")
    ap.add_argument("--max-spins",type=int,default=3,
                    help="Maximum resolved 13C model order; e.g. 3,4,5.")
    ap.add_argument("--max-sites",type=int,default=15,
                    help="Screened candidate sites retained per orientation; e.g. 15 or 100.")
    ap.add_argument("--max-per-family",type=int,default=3)
    ap.add_argument("--physical-snr-min",type=float,default=0.5)
    ap.add_argument("--family-tol-khz",type=float,default=12.0)
    ap.add_argument("--orientation-mode",
                    choices=("auto","assigned","fit-quality","all"),default="auto")
    ap.add_argument("--orientation-map",default=None)
    ap.add_argument("--scale-max",type=float,default=30.0)
    ap.add_argument("--background",
                    choices=("reduced","full","beta-free","taper-free"),default="reduced")
    ap.add_argument("--beam-width",type=int,default=30,
                    help="Number of best N-1 parents retained for expansion.")
    ap.add_argument("--expand-sites",type=int,default=0,
                    help="Top N single-site fits allowed for expansions; 0=all screened sites.")
    ap.add_argument("--order-candidate-cap",type=int,default=3000,
                    help="Maximum discrete models fitted at each N for each NV.")
    ap.add_argument("--pair-mode",choices=("auto","exhaustive","beam"),default="auto")
    ap.add_argument("--exhaustive-pair-limit",type=int,default=25)
    ap.add_argument("--robust-max-nfev",type=int,default=1800)
    ap.add_argument("--final-max-nfev",type=int,default=3000)
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--protect-production-sites",action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--output-dir",default=str(OUTROOT))
    args=ap.parse_args()
    if args.max_spins<1:
        ap.error("--max-spins must be >=1")
    if args.max_sites<1 or args.beam_width<1 or args.order_candidate_cap<1:
        ap.error("site/search limits must be positive")
    run(args)


if __name__=="__main__":
    main()
