"""V19 production validation of parsimonious V18 site replacements.

For each diagnostic NV, compare the V14 incumbent site set H0 with one
parsimonious V18 alternative H1 while jointly refitting the V14 reduced model:

  beta = 2 fixed
  revival amplitude taper = 0 fixed
  all other V14 background parameters free
  one shared physical amplitude scale s_NV
  two phases per selected 13C site
  catalog frequencies fixed by the discrete lattice-site assignment

H0 and H1 have identical parameter counts, so delta BIC equals delta chi2.
The old s<=3 production ceiling is relaxed to s<=10 for both hypotheses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import sc_spin_echo_physical_family_search_v6 as v6
V14_ROOT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v14_beta2_taper0_rerank\2026_09"
)
V18_ROOT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v18_local_site_rerank\2026_09"
)
DEFAULT_OUT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v19_production_site_validation\2026_09"
)

HYPOTHESES = {
    37:  {"H0": (31,24,78),  "H1": (31,24,78)},
    85:  {"H0": (31,220,24), "H1": (31,206,24)},
    168: {"H0": (50,310,26), "H1": (50,202,26)},
    171: {"H0": (102,20,136),"H1": (102,20,104)},
}

FIXED = {5: 2.0, 6: 0.0}
SCALE_MAX = 10.0
ROBUST_NFEV = 12000
FINAL_NFEV = 30000
RANDOM_PHASE_STARTS = 10


def find_v14_file():
    fs=[p for p in V14_ROOT.glob("*52G*v14_beta2_taper0_candidate_fits.csv.gz")
        if "smoke" not in str(p).lower()]
    if not fs:
        raise FileNotFoundError(f"No production V14 52G table under {V14_ROOT}")
    return max(fs,key=lambda p:p.stat().st_mtime)
def load_catalog_index(base):
    d=v6.load_catalog(base)
    out={}
    for r in d.itertuples():
        out[(tuple(r.ori),int(r.site_id))]=dict(
            site_id=int(r.site_id),
            f0_kHz=float(r.f0_kHz),
            f1_kHz=float(r.f1_kHz),
            kappa=float(r.kappa),
            distance_A=float(getattr(r,"distance_A",np.nan)),
            orientation=tuple(r.ori),
        )
    return out


def fit_stats(y,e,pred,k):
    yy=np.asarray(y,float)
    ee=np.maximum(np.asarray(e,float),1e-12)
    pp=np.asarray(pred,float)
    chi2=float(np.sum(((yy-pp)/ee)**2))
    n=int(yy.size); k=int(k)
    red=chi2/max(1,n-k)
    aic=chi2+2*k
    aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf
    bic=chi2+k*np.log(max(n,2))
    return dict(chi2=chi2,red_chi2=red,aicc=float(aicc),bic=float(bic),npar=k)


def bounds(base,t,baseline_seed,nsite):
    lb=list(np.asarray(base.BG_LB,float))
    ub=list(np.asarray(base.BG_UB,float))
    ub[1]=min(ub[1],max(0.05,float(baseline_seed)-0.01))
    ub[4]=min(float(ub[4]),float(base.t2_upper_us(t))/1000.0)
    if nsite:
        lb.append(0.0); ub.append(SCALE_MAX)
        for _ in range(nsite):
            lb.extend([-np.pi,-np.pi])
            ub.extend([ np.pi, np.pi])
    return np.asarray(lb,float),np.asarray(ub,float)


def expand_free(x,template,free_idx):
    th=np.asarray(template,float).copy()
    th[np.asarray(free_idx,int)]=np.asarray(x,float)
    for i,val in FIXED.items():
        th[int(i)]=float(val)
    return th


def build_seed(bg,scale,phases):
    bg=np.asarray(bg,float).copy()
    bg[5]=2.0; bg[6]=0.0
    return np.r_[bg,float(scale),np.asarray(phases,float).ravel()]


def v18_seed_for(nv,site_key,v14_bg):
    fs=list(V18_ROOT.glob(f"ranked_nv_{nv:04d}_tol25_pool10_sub2_smax10.csv"))
    if not fs:
        return None
    d=pd.read_csv(fs[0])
    q=d[d.site_key.astype(str)==str(tuple(site_key))]
    if q.empty:
        return None
    q=q.sort_values("bic").iloc[0]
    phases=np.asarray(json.loads(q.phases_json),float)
    return build_seed(v14_bg,float(q.shared_scale),phases)
def v14_seed_from_row(row,site_ids,v14_bg):
    inc_ids=[int(row[f"c13_{j}_site_id"]) for j in range(1,int(row.model_order)+1)]
    phase_map={
        int(row[f"c13_{j}_site_id"]):(
            float(row[f"c13_{j}_phi0"]),float(row[f"c13_{j}_phi1"])
        )
        for j in range(1,int(row.model_order)+1)
    }
    phases=[]
    for sid in site_ids:
        phases.append(phase_map.get(int(sid),(0.0,0.0)))
    scale=float(row.visibility_scale)
    return build_seed(v14_bg,min(scale,SCALE_MAX-1e-6),phases)


def make_sites(cat,ori,ids):
    return [dict(cat[(tuple(ori),int(sid))]) for sid in ids]


def seed_bank(base,t,row,sites,site_ids,v18_seed):
    n=len(sites)
    bg=np.asarray(json.loads(row.theta_json),float)[:9]
    bg[5]=2.0; bg[6]=0.0
    out=[]

    own=v14_seed_from_row(row,site_ids,bg)
    out.append(own)
    if v18_seed is not None:
        out.append(np.asarray(v18_seed,float))

    # Deterministic background variants preserve a physically informed phase seed.
    phase_ref=(np.asarray(v18_seed,float)[10:] if v18_seed is not None else own[10:])
    scale_ref=(float(np.asarray(v18_seed,float)[9]) if v18_seed is not None else float(own[9]))
    bg_variants=[
        (1.0,1.0,0.0,0.0),
        (0.75,0.8,0.0,0.0),
        (1.30,1.25,0.0,0.0),
        (1.0,1.0,-0.35,-0.004),
        (1.0,1.0,+0.35,+0.004),
    ]
    for fT2,fW,dTrev,dchirp in bg_variants:
        b=bg.copy()
        b[4]*=fT2
        b[3]*=fW
        b[2]+=dTrev
        b[8]+=dchirp
        out.append(build_seed(b,scale_ref,phase_ref.reshape(n,2)))

    # Random phase starts protect against the strongly multimodal phase landscape.
    rng=np.random.default_rng(20260925+int(row.name)+sum((i+1)*s for i,s in enumerate(site_ids)))
    for i in range(RANDOM_PHASE_STARTS):
        b=bg.copy()
        b[4]*=rng.uniform(0.7,1.35)
        b[3]*=rng.uniform(0.75,1.3)
        b[2]+=rng.uniform(-0.5,0.5)
        b[8]+=rng.uniform(-0.006,0.006)
        sc=[1.5,2.5,3.5,4.5,6.0][i%5]
        ph=rng.uniform(-np.pi,np.pi,(n,2))
        out.append(build_seed(b,sc,ph))
    return out
def fit_hypothesis(base,t,y,e,row,sites,site_ids,v18_seed):
    n=len(sites)
    ee=base.safe_err(e)
    seeds=seed_bank(base,t,row,sites,site_ids,v18_seed)
    baseline_seed=float(row.baseline)
    lb,ub=bounds(base,t,baseline_seed,n)
    template=np.asarray(seeds[0],float).copy()
    free_idx=[i for i in range(len(template)) if i not in FIXED]
    lbf=lb[free_idx]; ubf=ub[free_idx]

    def pred_from_x(x):
        th=expand_free(x,template,free_idx)
        return v6.v6_model(base,t,th,sites)

    def resid(x):
        return (np.asarray(y,float)-pred_from_x(x))/ee

    # Keep raw seeds as candidates, then robustly optimize all.
    finals=[]
    robust=[]
    for full in seeds:
        full=np.clip(np.asarray(full,float),lb+1e-9,ub-1e-9)
        full[5]=2.0; full[6]=0.0
        x0=full[free_idx]
        try:
            p0=v6.v6_model(base,t,full,sites)
            st0=fit_stats(y,ee,p0,len(free_idx))
            finals.append(dict(theta=full,pred=p0,source="seed",**st0))
        except Exception:
            pass
        try:
            rr=least_squares(
                resid,x0,bounds=(lbf,ubf),loss="soft_l1",f_scale=1.0,
                max_nfev=ROBUST_NFEV,x_scale="jac",
            )
            robust.append((float(np.sum(resid(rr.x)**2)),np.asarray(rr.x,float)))
        except Exception:
            pass

    robust.sort(key=lambda z:z[0])
    for _,x0 in robust[:6]:
        try:
            ff=least_squares(
                resid,x0,bounds=(lbf,ubf),loss="linear",
                max_nfev=FINAL_NFEV,ftol=1e-11,xtol=1e-11,gtol=1e-11,
                x_scale="jac",
            )
            th=expand_free(ff.x,template,free_idx)
            pp=v6.v6_model(base,t,th,sites)
            st=fit_stats(y,ee,pp,len(free_idx))
            finals.append(dict(theta=th,pred=pp,source="polish",**st))
        except Exception:
            pass
    if not finals:
        raise RuntimeError(f"all fits failed for sites {site_ids}")

    best=min(finals,key=lambda q:(q["chi2"],q["bic"]))
    # Closure: use best solution as one final linear polish.
    x0=np.clip(best["theta"][free_idx],lbf+1e-9,ubf-1e-9)
    try:
        ff=least_squares(
            resid,x0,bounds=(lbf,ubf),loss="linear",
            max_nfev=FINAL_NFEV,ftol=3e-12,xtol=3e-12,gtol=3e-12,
            x_scale="jac",
        )
        th=expand_free(ff.x,template,free_idx)
        pp=v6.v6_model(base,t,th,sites)
        st=fit_stats(y,ee,pp,len(free_idx))
        q=dict(theta=th,pred=pp,source="closure",**st)
        if q["chi2"]<best["chi2"]:
            best=q
    except Exception:
        pass
    return best
def result_row(nv,label,site_ids,fit,base,t):
    th=np.asarray(fit["theta"],float)
    return dict(
        nv_index=int(nv),hypothesis=str(label),site_key=str(tuple(site_ids)),
        chi2=float(fit["chi2"]),red_chi2=float(fit["red_chi2"]),
        aicc=float(fit["aicc"]),bic=float(fit["bic"]),npar=int(fit["npar"]),
        baseline=float(th[0]),contrast=float(th[1]),
        revival_time_us=float(th[2]),width0_us=float(th[3]),
        T2_us=float(1000*th[4]),beta=float(th[5]),
        amp_taper_alpha=float(th[6]),width_slope=float(th[7]),
        revival_chirp=float(th[8]),shared_scale=float(th[9]),
        scale_bound_hit=bool(th[9]>=SCALE_MAX-0.1),
        T2_limit_us=float(base.t2_upper_us(t)),
        theta_json=json.dumps([float(x) for x in th]),
        optimizer_source=str(fit["source"]),
    )


def plot_nv(pdf,t,y,e,nv,h0,h1,p0,p1):
    fig,axes=plt.subplots(2,1,figsize=(10,7.5),height_ratios=[2.2,1.0])
    ax=axes[0]
    ax.errorbar(t,y,yerr=e,fmt="o",ms=3,capsize=1,label="data")
    ax.plot(t,p0,lw=1.2,label=f"H0 {h0.site_key}")
    ax.plot(t,p1,lw=1.5,label=f"H1 {h1.site_key}")
    ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="Normalized signal")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax.set_title(
        f"NV {nv}: dBIC(H1-H0)={h1.bic-h0.bic:+.2f}, "
        f"s0={h0.shared_scale:.2f}, s1={h1.shared_scale:.2f}"
    )
    ax=axes[1]
    r0=(np.asarray(y)-np.asarray(p0))/np.asarray(e)
    r1=(np.asarray(y)-np.asarray(p1))/np.asarray(e)
    ax.plot(t,r0,"o-",ms=3,lw=.8,label="H0 normalized residual")
    ax.plot(t,r1,"o-",ms=3,lw=.8,label="H1 normalized residual")
    ax.axhline(0,ls="--",lw=.8)
    ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="Residual / sigma")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig,bbox_inches="tight")
    plt.close(fig)


def run(args):
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck)

    v14_path=find_v14_file()
    d=pd.read_csv(v14_path)
    winners=d[d.rank_global_bic==1].set_index("nv_index")
    cat=load_catalog_index(base)

    targets=[int(x) for x in args.nv.split(",") if x.strip()]
    results=[]; curves={}
    with threadpool_limits(limits=1):
        for nv in targets:
            row=winners.loc[nv]
            ori=base.parse_orientation(row.orientation)
            hmap=HYPOTHESES[nv]
            curves[nv]={}
            for label in ("H0","H1"):
                ids=tuple(hmap[label])
                sites=make_sites(cat,ori,ids)
                bg=np.asarray(json.loads(row.theta_json),float)[:9]
                v18seed=v18_seed_for(nv,ids,bg)
                fit=fit_hypothesis(base,t,Y[nv],E[nv],row,sites,ids,v18seed)
                rec=result_row(nv,label,ids,fit,base,t)
                results.append(rec)
                curves[nv][label]=fit["pred"]
                print(
                    f"NV{nv} {label} {ids}: chi2={rec['chi2']:.3f}, "
                    f"BIC={rec['bic']:.3f}, redchi2={rec['red_chi2']:.3f}, "
                    f"s={rec['shared_scale']:.3f}"
                )

    tab=pd.DataFrame(results)
    comps=[]
    for nv in targets:
        q=tab[tab.nv_index==nv].set_index("hypothesis")
        h0=q.loc["H0"]; h1=q.loc["H1"]
        comps.append(dict(
            nv_index=nv,
            H0_site_key=h0.site_key,H1_site_key=h1.site_key,
            H0_bic=float(h0.bic),H1_bic=float(h1.bic),
            delta_bic_H1_minus_H0=float(h1.bic-h0.bic),
            delta_chi2_H1_minus_H0=float(h1.chi2-h0.chi2),
            H0_red_chi2=float(h0.red_chi2),H1_red_chi2=float(h1.red_chi2),
            H0_scale=float(h0.shared_scale),H1_scale=float(h1.shared_scale),
            H0_scale_bound=bool(h0.scale_bound_hit),H1_scale_bound=bool(h1.scale_bound_hit),
            prefer_H1_BIC=bool(h1.bic<h0.bic),
            strong_H1=bool(h1.bic-h0.bic<=-6),
            decisive_H1=bool(h1.bic-h0.bic<=-10),
        ))
    comp=pd.DataFrame(comps).sort_values("nv_index")
    tab.to_csv(outdir/"v19_52G_hypothesis_fits.csv",index=False)
    comp.to_csv(outdir/"v19_52G_comparison.csv",index=False)

    with PdfPages(outdir/"v19_52G_production_site_validation.pdf") as pdf:
        for nv in targets:
            q=tab[tab.nv_index==nv].set_index("hypothesis")
            plot_nv(
                pdf,t,Y[nv],base.safe_err(E[nv]),nv,
                q.loc["H0"],q.loc["H1"],
                curves[nv]["H0"],curves[nv]["H1"],
            )

    meta=dict(
        source_v14=str(v14_path),
        hypotheses={str(k):{a:list(b) for a,b in v.items()} for k,v in HYPOTHESES.items()},
        fixed_background={"beta":2.0,"amp_taper_alpha":0.0},
        scale_max=SCALE_MAX,
        robust_nfev=ROBUST_NFEV,
        final_nfev=FINAL_NFEV,
        random_phase_starts=RANDOM_PHASE_STARTS,
        note="H0/H1 have identical parameter counts; dBIC equals dchi2.",
    )
    with open(outdir/"v19_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)

    print("\nV19 COMPARISON")
    print(comp.to_string(index=False))
    print("\nSaved:",outdir)
    return comp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nv",default="37,85,168,171")
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
