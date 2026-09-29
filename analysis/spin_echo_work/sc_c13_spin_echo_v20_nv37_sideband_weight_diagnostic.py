"""V20: NV37 sideband-weight diagnostic on fixed sites (31,24,78).

Compare nested amplitude structures while jointly refitting the same V14
reduced background (beta=2, revival taper=0) and all six sideband phases.

M0 equal:
    A+_j = A-_j = s * C*kappa_j/4

M1 global imbalance:
    A+_j = s * C*kappa_j/4 * exp(+d)
    A-_j = s * C*kappa_j/4 * exp(-d)

M2 per-site imbalance:
    A+_j = s * C*kappa_j/4 * exp(+d_j)
    A-_j = s * C*kappa_j/4 * exp(-d_j)

M3 free lines (upper bound):
    A+/-_j = q+/-_j * C*kappa_j/4, q>=0

Frequencies and lattice sites never move.
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

import sc_c13_spin_echo_physical_family_search_v6 as v6

V19_FILE = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v19_production_site_validation"
    r"\2026_09\robust_smax20\v19_52G_hypothesis_fits.csv"
)
DEFAULT_OUT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v20_nv37_sideband_weights\2026_09"
)

NV = 37
SITE_IDS = (31,24,78)
FIXED = {5:2.0, 6:0.0}
SHARED_SCALE_MAX = 30.0
IMBALANCE_MAX = 2.5
LINE_SCALE_MAX = 40.0
ROBUST_NFEV = 18000
FINAL_NFEV = 40000
RANDOM_STARTS = 20


def stats(y,e,pred,k):
    yy=np.asarray(y,float); ee=np.maximum(np.asarray(e,float),1e-12)
    pp=np.asarray(pred,float)
    chi2=float(np.sum(((yy-pp)/ee)**2))
    n=len(yy); k=int(k)
    red=chi2/max(1,n-k)
    aic=chi2+2*k
    aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf
    bic=chi2+k*np.log(max(n,2))
    return dict(chi2=chi2,red_chi2=red,aicc=float(aicc),bic=float(bic),npar=k)


def load_catalog_sites(base,orientation):
    d=v6.load_catalog(base)
    idx={(tuple(r.ori),int(r.site_id)):r for r in d.itertuples()}
    out=[]
    for sid in SITE_IDS:
        r=idx[(tuple(orientation),int(sid))]
        out.append(dict(
            site_id=int(sid),f0_kHz=float(r.f0_kHz),f1_kHz=float(r.f1_kHz),
            kappa=float(r.kappa),distance_A=float(getattr(r,"distance_A",np.nan)),
        ))
    return out


def background_carrier(base,t,bg):
    return base.carrier_from_bg(t,np.asarray(bg,float))


def model(base,t,bg,sites,phases,line_scales):
    baseline,contrast,carrier=background_carrier(base,t,bg)
    tt=np.asarray(t,float)
    osc=np.zeros_like(tt)
    for j,s in enumerate(sites):
        a0=contrast*float(s["kappa"])/4.0
        qplus,qminus=float(line_scales[2*j]),float(line_scales[2*j+1])
        pplus,pminus=float(phases[j,0]),float(phases[j,1])
        fplus=float(s["f0_kHz"])/1000.0
        fminus=float(s["f1_kHz"])/1000.0
        osc += a0*(qplus*np.cos(2*np.pi*fplus*tt+pplus)
                   +qminus*np.cos(2*np.pi*fminus*tt+pminus))
    return baseline-contrast*carrier+carrier*osc
def model_parts(kind,amp_pars,nsite):
    a=np.asarray(amp_pars,float)
    if kind=="M0_equal":
        return np.full(2*nsite,float(a[0]))
    if kind=="M1_global_imbalance":
        s,d=float(a[0]),float(a[1])
        q=[]
        for _ in range(nsite):
            q.extend([s*np.exp(d),s*np.exp(-d)])
        return np.asarray(q,float)
    if kind=="M2_site_imbalance":
        s=float(a[0]); ds=a[1:1+nsite]
        q=[]
        for d in ds:
            q.extend([s*np.exp(float(d)),s*np.exp(-float(d))])
        return np.asarray(q,float)
    if kind=="Msite_free_equal":
        q=[]
        for s in a[:nsite]:
            q.extend([float(s),float(s)])
        return np.asarray(q,float)
    if kind=="M3_free_lines":
        return np.asarray(a[:2*nsite],float)
    raise ValueError(kind)


def amp_bounds(kind,nsite):
    if kind=="M0_equal":
        return np.array([0.0]),np.array([SHARED_SCALE_MAX])
    if kind=="M1_global_imbalance":
        return np.array([0.0,-IMBALANCE_MAX]),np.array([SHARED_SCALE_MAX,IMBALANCE_MAX])
    if kind=="M2_site_imbalance":
        return np.r_[0.0,np.full(nsite,-IMBALANCE_MAX)],np.r_[SHARED_SCALE_MAX,np.full(nsite,IMBALANCE_MAX)]
    if kind=="Msite_free_equal":
        return np.zeros(nsite),np.full(nsite,LINE_SCALE_MAX)
    if kind=="M3_free_lines":
        return np.zeros(2*nsite),np.full(2*nsite,LINE_SCALE_MAX)
    raise ValueError(kind)
def bg_bounds(base,t,baseline_seed):
    lb=np.asarray(base.BG_LB,float).copy()
    ub=np.asarray(base.BG_UB,float).copy()
    ub[1]=min(ub[1],max(0.05,float(baseline_seed)-0.01))
    ub[4]=min(float(ub[4]),float(base.t2_upper_us(t))/1000.0)
    return lb,ub


def encode(bg,amp,phases):
    return np.r_[np.asarray(bg,float),np.asarray(amp,float),np.asarray(phases,float).ravel()]


def split(theta,kind,nsite):
    th=np.asarray(theta,float)
    bg=th[:9]
    na=len(amp_bounds(kind,nsite)[0])
    amp=th[9:9+na]
    phases=th[9+na:].reshape(nsite,2)
    return bg,amp,phases


def fit_kind(base,t,y,e,sites,kind,seed_bg,seed_scale,seed_phases,
             baseline_bound_seed=None):
    nsite=len(sites); ee=base.safe_err(e)
    if baseline_bound_seed is None:
        baseline_bound_seed=seed_bg[0]
    bgl,bgu=bg_bounds(base,t,baseline_bound_seed)
    al,au=amp_bounds(kind,nsite)
    pl=np.full(2*nsite,-np.pi); pu=np.full(2*nsite,np.pi)
    lb=np.r_[bgl,al,pl]; ub=np.r_[bgu,au,pu]
    fixed=set(FIXED)
    free=[i for i in range(len(lb)) if i not in fixed]

    def seed_amp():
        if kind=="M0_equal": return np.array([seed_scale])
        if kind=="M1_global_imbalance": return np.array([seed_scale,0.0])
        if kind=="M2_site_imbalance": return np.r_[seed_scale,np.zeros(nsite)]
        if kind=="Msite_free_equal": return np.full(nsite,seed_scale)
        if kind=="M3_free_lines": return np.full(2*nsite,seed_scale)
    template=encode(seed_bg,seed_amp(),seed_phases)
    template[5]=2.0; template[6]=0.0
    lbf=lb[free]; ubf=ub[free]

    def expand(x):
        th=template.copy(); th[free]=x; th[5]=2.0; th[6]=0.0
        return th

    def predx(x):
        th=expand(x)
        bg,amp,ph=split(th,kind,nsite)
        qs=model_parts(kind,amp,nsite)
        return model(base,t,bg,sites,ph,qs)

    def resid(x):
        return (np.asarray(y,float)-predx(x))/ee

    starts=[]
    s0=np.clip(template,lb+1e-8,ub-1e-8)
    starts.append(s0[free])
    rng=np.random.default_rng(20260925+sum(SITE_IDS)+len(kind))

    for i in range(RANDOM_STARTS):
        q=template.copy()
        q[3]*=rng.uniform(.65,1.4)
        q[4]*=rng.uniform(.6,1.7)
        q[2]+=rng.uniform(-1.0,1.0)
        q[7]=rng.uniform(0.0,min(.8,bgu[7]))
        q[8]=rng.uniform(max(-.06,bgl[8]),min(.06,bgu[8]))
        if kind=="M0_equal":
            q[9]=rng.uniform(2,20)
        elif kind=="M1_global_imbalance":
            q[9]=rng.uniform(2,20); q[10]=rng.uniform(-1.2,1.2)
        elif kind=="M2_site_imbalance":
            q[9]=rng.uniform(2,20); q[10:10+nsite]=rng.uniform(-1.2,1.2,nsite)
        elif kind=="Msite_free_equal":
            q[9:9+nsite]=rng.uniform(1,22,nsite)
        elif kind=="M3_free_lines":
            q[9:9+2*nsite]=rng.uniform(1,22,2*nsite)
        q[-2*nsite:]=rng.uniform(-np.pi,np.pi,2*nsite)
        q=np.clip(q,lb+1e-8,ub-1e-8)
        q[5]=2.0; q[6]=0.0
        starts.append(q[free])

    candidates=[]
    robust=[]
    for x0 in starts:
        try:
            pp=predx(x0); st=stats(y,ee,pp,len(free))
            candidates.append(dict(theta=expand(x0),pred=pp,source="seed",**st))
        except Exception: pass
        try:
            rr=least_squares(
                resid,x0,bounds=(lbf,ubf),loss="soft_l1",f_scale=1.0,
                max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((float(np.sum(resid(rr.x)**2)),rr.x))
        except Exception: pass

    robust.sort(key=lambda z:z[0])
    for _,x0 in robust[:8]:
        try:
            rr=least_squares(
                resid,x0,bounds=(lbf,ubf),loss="linear",
                max_nfev=FINAL_NFEV,ftol=1e-11,xtol=1e-11,gtol=1e-11,x_scale="jac")
            th=expand(rr.x); pp=predx(rr.x); st=stats(y,ee,pp,len(free))
            candidates.append(dict(theta=th,pred=pp,source="polish",**st))
        except Exception: pass
    if not candidates:
        raise RuntimeError(f"{kind}: all fits failed")
    return min(candidates,key=lambda q:(q["chi2"],q["bic"]))


def result_record(base,t,kind,fit,sites):
    bg,amp,ph=split(fit["theta"],kind,len(sites))
    qs=model_parts(kind,amp,len(sites))
    rec=dict(
        model=kind,chi2=fit["chi2"],red_chi2=fit["red_chi2"],
        aicc=fit["aicc"],bic=fit["bic"],npar=fit["npar"],source=fit["source"],
        baseline=bg[0],contrast=bg[1],revival_time_us=bg[2],width0_us=bg[3],
        T2_us=1000*bg[4],beta=bg[5],amp_taper_alpha=bg[6],
        width_slope=bg[7],revival_chirp=bg[8],
        theta_json=json.dumps([float(x) for x in fit["theta"]]),
    )
    for j,s in enumerate(sites):
        qp,qm=float(qs[2*j]),float(qs[2*j+1])
        rec[f"site{int(s['site_id'])}_qplus"]=qp
        rec[f"site{int(s['site_id'])}_qminus"]=qm
        rec[f"site{int(s['site_id'])}_ratio_plus_minus"]=qp/qm if qm>1e-12 else np.inf
        rec[f"site{int(s['site_id'])}_phi_plus"]=float(ph[j,0])
        rec[f"site{int(s['site_id'])}_phi_minus"]=float(ph[j,1])
    return rec
def plot_results(pdf,t,y,e,records,preds):
    fig,axes=plt.subplots(2,1,figsize=(10,7.5),height_ratios=[2.2,1.0])
    ax=axes[0]
    ax.errorbar(t,y,yerr=e,fmt="o",ms=3,capsize=1,label="data")
    for rec in records:
        ax.plot(t,preds[rec["model"]],lw=1.2,label=f"{rec['model']} BIC={rec['bic']:.1f}")
    ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="Normalized signal")
    ax.grid(alpha=.2); ax.legend(fontsize=7,ncol=2)
    ax.set_title("NV37 fixed sites (31,24,78): sideband-weight diagnostic")

    ax=axes[1]
    base=records[0]["bic"]
    vals=[r["bic"]-base for r in records]
    labs=[r["model"] for r in records]
    ax.bar(np.arange(len(vals)),vals)
    ax.axhline(0,ls="--",lw=.8)
    ax.set_xticks(np.arange(len(vals)),labs,rotation=15,ha="right")
    ax.set(ylabel="delta BIC vs M0",xlabel="Amplitude model")
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

    # One page with fitted line multipliers.
    fig,ax=plt.subplots(figsize=(10,5.5))
    x=np.arange(6); width=.18
    labels=[]
    for sid in SITE_IDS: labels.extend([f"{sid} f+",f"{sid} f-"])
    for i,rec in enumerate(records):
        vals=[]
        for sid in SITE_IDS:
            vals.extend([rec[f"site{sid}_qplus"],rec[f"site{sid}_qminus"]])
        ax.bar(x+(i-1.5)*width,vals,width,label=rec["model"])
    ax.set_xticks(x,labels,rotation=35,ha="right")
    ax.set_ylabel("Line multiplier q relative to C*kappa/4")
    ax.grid(axis="y",alpha=.2); ax.legend(fontsize=8)
    ax.set_title("NV37 fitted f+/f- line strengths")
    fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def run(args):
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    v19=pd.read_csv(V19_FILE)
    q=v19[(v19.nv_index==NV)&(v19.hypothesis=="H0")].iloc[0]
    th0=np.asarray(json.loads(q.theta_json),float)
    seed_bg=th0[:9]; seed_scale=float(q.shared_scale)
    seed_phases=th0[10:].reshape(3,2)

    # Use V19 orientation from its V14 source via the production table.
    v14files=list(Path(
        r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v14_beta2_taper0_rerank\2026_09"
    ).glob("*52G*v14_beta2_taper0_candidate_fits.csv.gz"))
    v14=pd.read_csv(max(v14files,key=lambda p:p.stat().st_mtime))
    row=v14[(v14.rank_global_bic==1)&(v14.nv_index==NV)].iloc[0]
    ori=base.parse_orientation(row.orientation)
    sites=load_catalog_sites(base,ori)

    kinds=["M0_equal","M1_global_imbalance","M2_site_imbalance","Msite_free_equal","M3_free_lines"]
    records=[]; preds={}
    with threadpool_limits(limits=1):
        for kind in kinds:
            fit=fit_kind(base,t,Y[NV],E[NV],sites,kind,seed_bg,seed_scale,seed_phases)
            rec=result_record(base,t,kind,fit,sites)
            records.append(rec); preds[kind]=fit["pred"]
            print(
                f"{kind}: chi2={rec['chi2']:.3f}, BIC={rec['bic']:.3f}, "
                f"redchi2={rec['red_chi2']:.3f}, npar={rec['npar']}"
            )

    tab=pd.DataFrame(records)
    b0=float(tab.loc[tab.model=="M0_equal","bic"].iloc[0])
    c0=float(tab.loc[tab.model=="M0_equal","chi2"].iloc[0])
    tab["delta_bic_vs_M0"]=tab.bic-b0
    tab["chi2_gain_vs_M0"]=c0-tab.chi2
    tab.to_csv(outdir/"v20_nv37_sideband_models.csv",index=False)

    line_rows=[]
    for rec in records:
        for sid in SITE_IDS:
            qp=rec[f"site{sid}_qplus"]; qm=rec[f"site{sid}_qminus"]
            line_rows.append(dict(
                model=rec["model"],site_id=sid,qplus=qp,qminus=qm,
                ratio_plus_minus=rec[f"site{sid}_ratio_plus_minus"],
                geom_mean=np.sqrt(max(qp,0)*max(qm,0)),
            ))
    pd.DataFrame(line_rows).to_csv(outdir/"v20_nv37_line_weights.csv",index=False)

    with PdfPages(outdir/"v20_nv37_sideband_weight_diagnostic.pdf") as pdf:
        plot_results(pdf,t,Y[NV],base.safe_err(E[NV]),records,preds)

    print("\nMODEL COMPARISON")
    print(tab[["model","npar","chi2","red_chi2","bic","delta_bic_vs_M0","chi2_gain_vs_M0"]].to_string(index=False))
    print("\nLINE WEIGHTS")
    print(pd.DataFrame(line_rows).to_string(index=False))
    print("\nSaved:",outdir)
    return tab


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
