"""V32 joint multi-field spin-echo analysis for the 204-NV Johnson array.

Fits the SAME physical 13C lattice-site IDs simultaneously to four fields:
49.70 G, 59.69 G, 62.48 G, and 65.14 G.

Shared across fields:
  * NV identity and orientation
  * discrete lattice-site IDs
  * model order N

Independent for each field:
  * echo background / contrast / T2
  * shared visibility scale s_NV for that field
  * the two phases for every selected 13C site

The field-specific f_plus, f_minus, and kappa are read from the corresponding
catalog for the same physical site IDs.

This is a new joint inference; old per-field fits are used only as seeds and
candidate hints.  NVs are checkpointed independently.
"""
from __future__ import annotations

import argparse,ast,json,itertools,math
from pathlib import Path
import sys

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from utils import data_manager as dm
import sc_c13_spin_echo_physical_family_search_v6 as v6

ROOT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo")
OUTROOT=ROOT/r"c13_spin_echo_v32_joint_multifield_johnson\2026_09"
V22_49=ROOT/r"c13_spin_echo_v22_full_physical_rerank\2026_09\49G\smax30_tol12_pool8_sub1_topO2_topG4_cap40"
FIELDS=[
 dict(label="49G", B_G=np.array([-46.19581364,-17.44900422,-5.57935388]),
      counts="2025_11_15-14_11_49-johnson_204nv_s9-17d44b",
      seedfit="2025_11_19-14_19_23-sample_204nv_s1-fcc605",
      catalog="essem_freq_kappa_catalog_22A_49G.json"),
 dict(label="59G", B_G=np.array([-41.57848995,-32.77145194,-27.5799348]),
      counts="2025_12_04-19_50_15-johnson_204nv_s9-2c83ab",
      seedfit="2025_12_05-07_51_13-sample_204nv_s1-4cf818",
      catalog="essem_freq_kappa_catalog_22A_59G.json"),
 dict(label="62G", B_G=np.array([-48.67047318,-32.07615947,22.49657427]),
      counts="2026_01_07-17_30_14-johnson_204nv_s12-06bcdd",
      seedfit="2026_01_07-22_29_55-sample_204nv_s1-c9580a",
      catalog="essem_freq_kappa_catalog_22A_62G.json"),
 dict(label="65G", B_G=np.array([-31.61263115,-56.58135644,-6.5512002]),
      counts="2026_01_07-17_48_21-johnson_204nv_s10-34f8b7",
      seedfit="2026_01_07-23_43_15-sample_204nv_s1-27e4dc",
      catalog="essem_freq_kappa_catalog_22A_65G.json"),
]
CATDIR=Path(__file__).resolve().parent


def parse_nv_arg(text,nmax):
    if not text: return list(range(nmax))
    out=[]
    for tok in str(text).split(","):
        tok=tok.strip()
        if not tok: continue
        if "-" in tok:
            a,b=map(int,tok.split("-",1)); out.extend(range(a,b+1))
        else: out.append(int(tok))
    return sorted({i for i in out if 0<=i<nmax})


def fixed_background(mode):
    return {
        "reduced":{5:2.0,6:0.0},
        "full":{},
        "beta-free":{6:0.0},
        "taper-free":{5:2.0},
    }[mode]
def load_counts(stem):
    d=dm.get_raw_data(file_stem=stem,load_npz=True)
    names=[str(n.name) for n in d["nv_list"]]
    y=np.asarray(d["norm_counts"],float)
    e=np.asarray(d["norm_counts_ste"],float)
    t=np.asarray(d["total_evolution_times"],float).ravel()
    order=np.argsort(t); t=t[order]; y=y[:,order]; e=e[:,order]
    ori=np.asarray(d.get("orientations",[]),int)
    inds=np.asarray(d.get("nv_indices_global",np.arange(len(names))),int)
    return dict(raw=d,names=names,t=t,y=y,e=e,ori=ori,indices=inds)


def load_seedfit(stem):
    d=dm.get_raw_data(file_stem=stem,load_npz=True)
    return dict(
        popts=d.get("popts",[]),
        site_id=d.get("site_id",[]),
        red_chi2=d.get("red_chi2",[]),
        fit_fn=d.get("fit_fn_names",[]),
        orientations=np.asarray(d.get("orientations",[]),int),
    )


def load_catalog(filename):
    d=pd.DataFrame(json.load(open(CATDIR/filename,"r",encoding="utf-8")))
    d["ori"]=d.orientation.map(lambda x:tuple(int(v) for v in x))
    d["site_id"]=d.site_index.astype(int)
    d["f0_kHz"]=d.f_plus_Hz.astype(float)/1e3
    d["f1_kHz"]=d.f_minus_Hz.astype(float)/1e3
    return d


def load_all():
    data={}; seeds={}; cats={}
    for cfg in FIELDS:
        lab=cfg["label"]
        data[lab]=load_counts(cfg["counts"])
        seeds[lab]=load_seedfit(cfg["seedfit"])
        cats[lab]=load_catalog(cfg["catalog"])

    ref=data["49G"]
    for lab in data:
        if data[lab]["names"]!=ref["names"]:
            raise RuntimeError(f"{lab}: NV names/order differ from 49G")
        if not np.array_equal(data[lab]["indices"],ref["indices"]):
            raise RuntimeError(f"{lab}: nv_indices_global differs from 49G")
        if data[lab]["ori"].shape!=(204,3):
            raise RuntimeError(f"{lab}: expected stored 204x3 orientations")
        if not np.array_equal(data[lab]["ori"],ref["ori"]):
            raise RuntimeError(f"{lab}: orientation map differs from 49G")
    return data,seeds,cats
def generic_bg_seed(t,y,Bmag):
    yy=np.asarray(y,float)
    baseline=float(np.nanmedian(yy))
    contrast=float(np.clip(np.nanpercentile(yy,90)-np.nanpercentile(yy,10),0.03,0.8))
    trev=float(2000.0/(1.0705*Bmag))
    width=float(np.clip(0.15*trev,1.5,12.0))
    t2_ms=float(np.clip(2.0*np.nanmax(t)/1000.0,0.03,0.25))
    return np.array([baseline,contrast,trev,width,t2_ms,2.0,0.0,0.02,0.0],float)


def old_bg_seed(base,seedfit,nv,t,y,Bmag):
    try:
        p=np.asarray(seedfit["popts"][nv],float)
        if p.size>=14 and np.all(np.isfinite(p[:9])):
            th=base.old_popt_to_theta(p)
            return np.asarray(th[:9],float)
    except Exception:
        pass
    return generic_bg_seed(t,y,Bmag)


def observed_pair(seedfit,nv):
    try:
        p=np.asarray(seedfit["popts"][nv],float)
        if p.size>=14 and np.isfinite(p[10]) and np.isfinite(p[12]):
            vals=sorted([abs(float(p[10]))*1000.0,abs(float(p[12]))*1000.0],reverse=True)
            return vals[0],vals[1]
    except Exception:
        pass
    return None


def catalog_orientation(cat,ori):
    return cat[cat.ori.map(tuple)==tuple(ori)].copy()


def site_record(cats,label,ori,sid):
    q=catalog_orientation(cats[label],ori)
    q=q[q.site_id.astype(int)==int(sid)]
    if q.empty: raise KeyError((label,ori,sid))
    r=q.iloc[0]
    return dict(site_id=int(sid),orientation=tuple(ori),
                f0_kHz=float(r.f0_kHz),f1_kHz=float(r.f1_kHz),
                kappa=float(r.kappa),distance_A=float(r.distance_A),
                x_A=float(r.x_A),y_A=float(r.y_A),z_A=float(r.z_A))
def candidate_pool(nv,ori,seeds,cats,args,v22=None):
    ids=set(); detail=[]
    # Per-field nearest sites to the historically fitted frequency pair.
    for cfg in FIELDS:
        lab=cfg["label"]; obs=observed_pair(seeds[lab],nv)
        q=catalog_orientation(cats[lab],ori)
        if obs is not None:
            d=np.hypot(q.f0_kHz.to_numpy(float)-obs[0],
                       q.f1_kHz.to_numpy(float)-obs[1])
            take=np.argsort(d)[:args.freq_candidates_per_field]
            for ii in take:
                sid=int(q.iloc[ii].site_id); ids.add(sid)
                detail.append(dict(nv_index=nv,site_id=sid,hint_field=lab,
                                   hint_kind="frequency_nearest",
                                   frequency_distance_kHz=float(d[ii])))
        try:
            sid=int(seeds[lab]["site_id"][nv])
            if sid>=0: ids.add(sid)
        except Exception: pass

    # Bring forward strong/multisite 49G V22 candidates as additional hints.
    if v22 is not None:
        q=v22[v22.nv_index==nv].sort_values("bic").head(args.v22_top_rows)
        for _,r in q.iterrows():
            for j in range(1,int(r.model_order)+1):
                c=f"c13_{j}_site_id"
                if c in r and pd.notna(r[c]):
                    ids.add(int(r[c]))

    # A small set of high-kappa in-band sites guards against missed weak old fits.
    for cfg in FIELDS:
        lab=cfg["label"]; q=catalog_orientation(cats[lab],ori).copy()
        tmax=float(args._data[lab]["t"].max())
        tt=np.unique(args._data[lab]["t"]); dt=np.diff(tt); dt=dt[dt>0]
        lo=1000*0.5/max(np.ptp(tt),1e-9)
        hi=1000*0.49/max(float(dt.min()) if len(dt) else np.ptp(tt),1e-9)
        q=q[(q.f0_kHz>=lo)&(q.f0_kHz<=hi)&(q.f1_kHz>=lo)&(q.f1_kHz<=hi)]
        for sid in q.sort_values("kappa",ascending=False).head(args.kappa_guard).site_id:
            ids.add(int(sid))

    # If the union is too large, rank by minimum frequency distance to any field hint,
    # while always protecting historical and V22 sites.
    protected=set()
    for lab in seeds:
        try:
            sid=int(seeds[lab]["site_id"][nv])
            if sid>=0: protected.add(sid)
        except Exception: pass
    if v22 is not None:
        q=v22[v22.nv_index==nv].sort_values("bic").head(args.v22_top_rows)
        for _,r in q.iterrows():
            for j in range(1,int(r.model_order)+1):
                c=f"c13_{j}_site_id"
                if c in r and pd.notna(r[c]): protected.add(int(r[c]))
    if len(ids)>args.pool_max:
        scores=[]
        for sid in ids:
            best=np.inf
            for cfg in FIELDS:
                lab=cfg["label"]; obs=observed_pair(seeds[lab],nv)
                if obs is None: continue
                sr=site_record(cats,lab,ori,sid)
                best=min(best,math.hypot(sr["f0_kHz"]-obs[0],sr["f1_kHz"]-obs[1]))
            scores.append((0 if sid in protected else 1,best,sid))
        scores.sort()
        ids={sid for _,_,sid in scores[:args.pool_max]}
    return sorted(ids),pd.DataFrame(detail)
def fit_field(base,t,y,e,sites,seeds_theta,fixed,scale_max,robust_nfev,final_nfev):
    v6.VISIBILITY_SCALE_MAX=float(scale_max)
    n=len(sites); expected=9 if n==0 else 10+2*n
    seeds_theta=[np.asarray(s,float) for s in seeds_theta if np.asarray(s).size==expected]
    if not seeds_theta: return None
    baseline=float(np.nanmedian([s[0] for s in seeds_theta]))
    lb,ub=v6.v6_theta_bounds(base,n,baseline,float(base.t2_upper_us(t)))
    ee=base.safe_err(e); free=[i for i in range(expected) if i not in fixed]

    def expand(x,tmp):
        th=tmp.copy(); th[free]=x
        for i,val in fixed.items(): th[int(i)]=float(val)
        return th

    robust=[]
    for seed in seeds_theta:
        tmp=np.asarray(seed,float).copy()
        for i,val in fixed.items(): tmp[int(i)]=float(val)
        tmp=np.clip(tmp,lb+1e-8,ub-1e-8)
        def resid(x):
            return (np.asarray(y,float)-v6.v6_model(base,t,expand(x,tmp),sites))/ee
        try:
            rr=least_squares(resid,tmp[free],bounds=(lb[free],ub[free]),
                             loss="soft_l1",f_scale=1,max_nfev=robust_nfev,x_scale="jac")
            th=expand(rr.x,tmp); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            robust.append((float(st["chi2"]),th))
        except Exception: pass
    if not robust: return None
    robust.sort(key=lambda z:z[0]); finals=[]
    for _,seed in robust[:min(3,len(robust))]:
        tmp=np.asarray(seed,float)
        def resid(x):
            return (np.asarray(y,float)-v6.v6_model(base,t,expand(x,tmp),sites))/ee
        try:
            rr=least_squares(resid,tmp[free],bounds=(lb[free],ub[free]),
                             loss="linear",max_nfev=final_nfev,
                             ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale="jac")
            th=expand(rr.x,tmp); pred=v6.v6_model(base,t,th,sites)
            st=base.calc_stats(y,ee,pred,len(free))
            finals.append(dict(theta=th,pred=pred,**st))
        except Exception: pass
    return min(finals,key=lambda z:(z["bic"],z["chi2"])) if finals else None
def phase_map(theta,site_ids):
    if not site_ids: return {}
    th=np.asarray(theta,float); out={}; j=10
    for sid in site_ids:
        out[int(sid)]=(float(th[j]),float(th[j+1])); j+=2
    return out


def seeds_for_field(bgtheta,site_ids,parent_fit=None,parent_ids=(),scale_max=30.0):
    if not site_ids: return [np.asarray(bgtheta,float)]
    phases=[]
    pm={}
    parent_scale=None
    if parent_fit is not None:
        parent_scale=float(np.asarray(parent_fit["theta"],float)[9])
        pm=phase_map(parent_fit["theta"],parent_ids)
    for sid in site_ids:
        phases.extend(pm.get(int(sid),(0.0,0.0)))
    scales=[]
    if parent_scale is not None: scales.append(parent_scale)
    scales.extend([1.0,3.0,10.0,min(20.0,float(scale_max))])
    out=[]
    for s in scales:
        s=float(np.clip(s,1e-6,float(scale_max)-1e-6))
        out.append(np.r_[np.asarray(bgtheta,float)[:9],s,phases])
    return out


def fit_joint_candidate(base,nv,ori,site_ids,data,cats,n0fits,args,parent=None):
    fixed=fixed_background(args.background)
    fieldfits={}; sum_bic=0.; sum_chi2=0.; dof=0
    for cfg in FIELDS:
        lab=cfg["label"]; dd=data[lab]
        sites=[site_record(cats,lab,ori,sid) for sid in site_ids]
        parfit=None; parids=()
        if parent is not None:
            parfit=parent["fieldfits"].get(lab); parids=parent["site_ids"]
        bgtheta=n0fits[lab]["theta"]
        seeds_theta=seeds_for_field(bgtheta,site_ids,parfit,parids,args.scale_max)
        ff=fit_field(base,dd["t"],dd["y"][nv],dd["e"][nv],sites,seeds_theta,
                     fixed,args.scale_max,args.robust_max_nfev,args.final_max_nfev)
        if ff is None: return None
        fieldfits[lab]=ff
        sum_bic+=float(ff["bic"]); sum_chi2+=float(ff["chi2"])
        dof+=max(1,len(dd["t"])-int(ff["npar"]))
    return dict(site_ids=tuple(sorted(int(x) for x in site_ids)),
                fieldfits=fieldfits,joint_bic_sum=sum_bic,
                joint_chi2=sum_chi2,joint_redchi2=sum_chi2/max(1,dof))
def fit_one_nv(nv,data,seeds,cats,args,v22,ckdir):
    cp=ckdir/f"nv_{nv:04d}.csv.gz"
    if cp.exists(): return str(cp)
    base=v6.load_backend("49G")
    ori=tuple(int(v) for v in data["49G"]["ori"][nv])
    fixed=fixed_background(args.background)

    # N0 per field: fit the field-specific background first.
    n0fits={}
    for cfg in FIELDS:
        lab=cfg["label"]; dd=data[lab]
        bg=old_bg_seed(base,seeds[lab],nv,dd["t"],dd["y"][nv],np.linalg.norm(cfg["B_G"]))
        ff=fit_field(base,dd["t"],dd["y"][nv],dd["e"][nv],[],[bg],fixed,
                     args.scale_max,args.robust_max_nfev,args.final_max_nfev)
        if ff is None: raise RuntimeError(f"NV{nv} {lab}: N0 failed")
        n0fits[lab]=ff

    pool,pooldetail=candidate_pool(nv,ori,seeds,cats,args,v22)
    if not pool: raise RuntimeError(f"NV{nv}: empty joint site pool")
    print(f"[NV {nv:3d}] ori={ori} joint pool={len(pool)}")

    fits_by_order={0:[fit_joint_candidate(base,nv,ori,(),data,cats,n0fits,args)]}
    # Replace N0 with the already fitted field backgrounds exactly.
    n0=dict(site_ids=(),fieldfits=n0fits,
            joint_bic_sum=sum(float(f["bic"]) for f in n0fits.values()),
            joint_chi2=sum(float(f["chi2"]) for f in n0fits.values()))
    dof=sum(max(1,len(data[l]["t"])-int(n0fits[l]["npar"])) for l in n0fits)
    n0["joint_redchi2"]=n0["joint_chi2"]/dof; fits_by_order[0]=[n0]

    singles=[]
    for sid in pool:
        q=fit_joint_candidate(base,nv,ori,(sid,),data,cats,n0fits,args)
        if q is not None: singles.append(q)
    singles.sort(key=lambda z:z["joint_bic_sum"]); fits_by_order[1]=singles
    print(f"[NV {nv:3d}] N1 fits={len(singles)}")

    for order in range(2,args.max_spins+1):
        prev=fits_by_order.get(order-1,[])
        if not prev: break
        parents=prev[:args.beam_width]
        props={}
        for par in parents:
            have=set(par["site_ids"])
            for sid in pool:
                if sid in have: continue
                ids=tuple(sorted((*have,int(sid))))
                props[ids]=par
        keys=list(props.keys())
        # Heuristic ordering: expand better parents first.
        keys.sort(key=lambda ids:props[ids]["joint_bic_sum"])
        keys=keys[:args.order_candidate_cap]
        cur=[]
        for ids in keys:
            q=fit_joint_candidate(base,nv,ori,ids,data,cats,n0fits,args,parent=props[ids])
            if q is not None: cur.append(q)
        cur.sort(key=lambda z:z["joint_bic_sum"]); fits_by_order[order]=cur
        print(f"[NV {nv:3d}] N{order} proposals={len(keys)} fits={len(cur)}")
        if not cur: break

    rows=[]
    for order,flist in fits_by_order.items():
        for rank,q in enumerate(flist,1):
            row=dict(nv_index=nv,orientation=str(ori),model_order=order,
                     site_key=str(q["site_ids"]),rank_within_order=rank,
                     joint_bic_sum=q["joint_bic_sum"],joint_chi2=q["joint_chi2"],
                     joint_redchi2=q["joint_redchi2"],pool_size=len(pool))
            for lab,ff in q["fieldfits"].items():
                th=np.asarray(ff["theta"],float)
                row[f"{lab}_bic"]=float(ff["bic"])
                row[f"{lab}_redchi2"]=float(ff["red_chi2"])
                row[f"{lab}_scale"]=float(th[9]) if order else np.nan
                row[f"{lab}_theta_json"]=json.dumps(th.tolist())
            rows.append(row)
    d=pd.DataFrame(rows).sort_values("joint_bic_sum")
    d["rank_global_joint_bic"]=np.arange(1,len(d)+1)
    ckdir.mkdir(parents=True,exist_ok=True)
    d.to_csv(cp,index=False,compression="gzip")
    pooldetail.to_csv(ckdir/f"nv_{nv:04d}_pool_hints.csv",index=False)
    pd.DataFrame({"site_id":pool}).to_csv(ckdir/f"nv_{nv:04d}_pool.csv",index=False)
    w=d.iloc[0]
    print(f"[NV {nv:3d}] WIN N{int(w.model_order)} {w.site_key} "
          f"jointBIC={w.joint_bic_sum:.1f} redchi={w.joint_redchi2:.2f}")
    return str(cp)
def plot_nv(pdf,nv,row,data,cats):
    ori=tuple(int(v) for v in ast.literal_eval(str(row.orientation)))
    ids=tuple(int(v) for v in ast.literal_eval(str(row.site_key)))
    fig,ax=plt.subplots(2,3,figsize=(14,8.5))
    base=v6.load_backend("49G")
    for k,cfg in enumerate(FIELDS):
        lab=cfg["label"]; dd=data[lab]
        a=ax[k//2,k%2]
        th=np.asarray(json.loads(row[f"{lab}_theta_json"]),float)
        sites=[site_record(cats,lab,ori,sid) for sid in ids]
        dense=np.linspace(float(dd["t"].min()),float(dd["t"].max()),1200)
        pred=v6.v6_model(base,dense,th,sites)
        a.errorbar(dd["t"],dd["y"][nv],yerr=dd["e"][nv],fmt=".",ms=2.5,lw=.5,alpha=.55)
        a.plot(dense,pred,lw=1.2)
        a.set(title=f"{lab} | redchi2={row[f'{lab}_redchi2']:.2f}",
              xlabel="evolution time (us)",ylabel="normalized signal")

    # Field trajectory for every accepted site.
    a=ax[0,2]
    bm=[np.linalg.norm(c["B_G"]) for c in FIELDS]
    for sid in ids:
        fm=[]; fp=[]
        for cfg in FIELDS:
            s=site_record(cats,cfg["label"],ori,sid)
            fm.append(s["f1_kHz"]); fp.append(s["f0_kHz"])
        a.plot(bm,fm,"o-",label=f"{sid} f-")
        a.plot(bm,fp,"o--",label=f"{sid} f+")
    a.set(xlabel="|B| (G)",ylabel="catalog frequency (kHz)",
          title="Same-site frequency trajectories")
    a.legend(fontsize=6,ncol=2)

    # Real-space lattice projection.
    a=ax[1,2]; cc=catalog_orientation(cats["49G"],ori)
    for sid in ids:
        q=cc[cc.site_id.astype(int)==sid].iloc[0]
        a.scatter([q.x_A],[q.y_A],s=80,marker="*")
        a.annotate(str(sid),(q.x_A,q.y_A),xytext=(3,3),textcoords="offset points")
    a.scatter([0],[0],marker="x",s=50)
    a.set_aspect("equal",adjustable="datalim")
    a.set(xlabel="x (A)",ylabel="y (A)",title="Jointly selected C13 sites")
    fig.suptitle(f"Johnson NV{nv}: joint N{int(row.model_order)} {ids} | "
                 f"joint redchi2={row.joint_redchi2:.2f}")
    fig.tight_layout(rect=[0,0,.99,.95]); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
def run(args):
    data,seeds,cats=load_all(); args._data=data
    nvs=parse_nv_arg(args.nv,204)
    v22=None
    p=V22_49/"v22_candidate_fits.csv.gz"
    if p.exists(): v22=pd.read_csv(p)

    tag=(f"N{args.max_spins}_pool{args.pool_max}_freq{args.freq_candidates_per_field}_"
         f"beam{args.beam_width}_cap{args.order_candidate_cap}_"
         f"{args.background}_smax{str(args.scale_max).replace('.','p')}")
    outdir=Path(args.output_dir)/tag; ckdir=outdir/"checkpoints"
    outdir.mkdir(parents=True,exist_ok=True)

    print("="*100)
    print("V32 JOHNSON FOUR-FIELD JOINT ANALYSIS")
    for cfg in FIELDS:
        print(cfg["label"],f"|B|={np.linalg.norm(cfg['B_G']):.4f} G",cfg["counts"])
    print("NVs",len(nvs),"max spins",args.max_spins,"pool max",args.pool_max)

    def task(nv): return fit_one_nv(nv,data,seeds,cats,args,v22,ckdir)
    with threadpool_limits(limits=1):
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(task)(nv) for nv in nvs)

    parts=[pd.read_csv(ckdir/f"nv_{nv:04d}.csv.gz") for nv in nvs]
    cand=pd.concat(parts,ignore_index=True)
    cand.to_csv(outdir/"v32_joint_candidates.csv.gz",index=False,compression="gzip")
    winners=(cand.sort_values("joint_bic_sum")
             .groupby("nv_index",as_index=False).first())
    orderbest=(cand.sort_values("joint_bic_sum")
               .groupby(["nv_index","model_order"],as_index=False).first())
    winners.to_csv(outdir/"v32_joint_winners.csv",index=False)
    orderbest.to_csv(outdir/"v32_joint_best_by_order.csv",index=False)

    with PdfPages(outdir/"v32_joint_multifield_report.pdf") as pdf:
        fig,ax=plt.subplots(1,2,figsize=(11,5))
        oc=winners.model_order.value_counts().sort_index()
        ax[0].bar(oc.index.astype(str),oc.values)
        ax[0].set(title="Joint BIC winner order",xlabel="N",ylabel="NV count")
        ax[1].hist(winners.joint_redchi2,bins=30)
        ax[1].axvline(2,ls="--",lw=.7); ax[1].axvline(3,ls="--",lw=.7)
        ax[1].set(title="Four-field joint fit quality",xlabel="joint reduced chi-square")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for nv in nvs:
            row=winners[winners.nv_index==nv].iloc[0]
            plot_nv(pdf,nv,row,data,cats)

    meta=vars(args).copy(); meta.pop("_data",None)
    meta["fields"]=[dict(label=c["label"],B_G=c["B_G"].tolist(),
                         B_mag_G=float(np.linalg.norm(c["B_G"])),
                         counts=c["counts"],seedfit=c["seedfit"]) for c in FIELDS]
    meta["winner_order_counts"]={str(k):int(v) for k,v in oc.items()}
    meta["identity_check"]="same 204 names, global indices, and orientations across all four fields"
    with open(outdir/"v32_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print("V32 COMPLETE",outdir)
    print("winner orders",meta["winner_order_counts"])
    return cand,winners
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nv",default=None,help="e.g. 0,1,10-20")
    ap.add_argument("--max-spins",type=int,default=3)
    ap.add_argument("--freq-candidates-per-field",type=int,default=12)
    ap.add_argument("--v22-top-rows",type=int,default=20)
    ap.add_argument("--kappa-guard",type=int,default=4)
    ap.add_argument("--pool-max",type=int,default=50)
    ap.add_argument("--beam-width",type=int,default=15)
    ap.add_argument("--order-candidate-cap",type=int,default=500)
    ap.add_argument("--background",choices=("reduced","full","beta-free","taper-free"),
                    default="reduced")
    ap.add_argument("--scale-max",type=float,default=30.0)
    ap.add_argument("--robust-max-nfev",type=int,default=1200)
    ap.add_argument("--final-max-nfev",type=int,default=2200)
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--output-dir",default=str(OUTROOT))
    args=ap.parse_args()
    if args.max_spins<1: ap.error("--max-spins must be >=1")
    run(args)


if __name__=="__main__":
    main()
