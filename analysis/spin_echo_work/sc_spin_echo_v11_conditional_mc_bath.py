"""V11: conditional Monte-Carlo 13C lattice-bath diagnostic.

Each measured NV has one fixed isotope realization, not an isotope-disorder
average.  This script samples actual occupied 13C configurations from the same
orientation-specific 22-A lattice catalog used by V6.

Default conditional model:
  * V6-selected explicit C13 sites are conditioned occupied and handled by
    the existing V6 discrete term (therefore excluded from stochastic bath).
  * catalog sites with V6 physical SNR >= --snr-cut are conditioned unoccupied
    unless explicitly selected.
  * every remaining carbon site is independently occupied with p=--p-occ.
  * for realization r, L_bath^r(t) = product_{i occupied} L_i(t).

Many realizations are screened cheaply; only the best few are fully refit.
The best-of-M BIC is diagnostic only (configuration selection adds a
look-elsewhere effect); the output reports the full screening distribution.
"""
from __future__ import annotations

import argparse, json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v7_diagnostic as v7
import sc_spin_echo_v10_lattice_bath_diagnostic as v10

P_OCC_DEFAULT=0.011
SNR_CUT_DEFAULT=0.5
N_REALIZATIONS_DEFAULT=500
TOP_REFIT_DEFAULT=10
SEED_DEFAULT=20260924


def single_spin_matrix(cat,t):
    """Exact Hahn factor L_i(t) for total evolution time t=2*tau."""
    tt=np.asarray(t,float)[None,:]
    kap=cat.kappa.to_numpy(float)[:,None]
    fI=cat.fI_Hz.to_numpy(float)[:,None]/1e6
    fm=cat.omega_ms_Hz.to_numpy(float)[:,None]/1e6
    return (1.0 - 2.0*kap
            * np.sin(0.5*np.pi*fI*tt)**2
            * np.sin(0.5*np.pi*fm*tt)**2)


def draw_realizations(L,p_occ,n_realizations,rng):
    """Draw Bernoulli isotope configurations and return baths + occupancy."""
    nsite,ntime=L.shape
    baths=np.ones((n_realizations,ntime),float)
    occupied=[]
    nocc=np.zeros(n_realizations,int)
    for r in range(n_realizations):
        idx=np.flatnonzero(rng.random(nsite)<float(p_occ))
        occupied.append(idx)
        nocc[r]=len(idx)
        if len(idx):
            baths[r]=np.prod(L[idx],axis=0)
    return baths,occupied,nocc

def discrete_shape(t,row,sites):
    """V6 discrete term divided by contrast: scale/4 * sum kappa*cosines."""
    tt=np.asarray(t,float); out=np.zeros_like(tt)
    if not sites:
        return out
    th=np.asarray(json.loads(row.theta_json),float)
    scale=float(th[9]);j=10
    for s in sites:
        phi0,phi1=th[j:j+2];j+=2
        f0=float(s["f0_kHz"])/1000.0
        f1=float(s["f1_kHz"])/1000.0
        out += scale*float(s["kappa"])/4.0 * (
            np.cos(2*np.pi*f0*tt+phi0)+np.cos(2*np.pi*f1*tt+phi1))
    return out


def fast_screen(base,t,y,e,row,sites,baths):
    """Screen baths with V6 T2/beta/scale/phases; fit baseline/contrast analytically."""
    th=np.asarray(json.loads(row.theta_json),float)
    T2_us=max(1e-9,1000.0*float(th[4])); beta=float(th[5])
    env=np.exp(-np.power(np.maximum(np.asarray(t,float),0)/T2_us,beta))
    g=-1.0+discrete_shape(t,row,sites)
    X=baths*(env*g)[None,:]
    yy=np.asarray(y,float); ee=base.safe_err(e)
    w=1.0/(ee*ee)
    S0=float(np.sum(w)); Sy=float(np.sum(w*yy))
    Sx=np.sum(X*w[None,:],axis=1)
    Sxx=np.sum(X*X*w[None,:],axis=1)
    Sxy=np.sum(X*(w*yy)[None,:],axis=1)
    det=S0*Sxx-Sx*Sx

    det=np.where(np.abs(det)<1e-15,np.nan,det)
    baseline=(Sy*Sxx-Sx*Sxy)/det
    contrast=(S0*Sxy-Sx*Sy)/det
    baseline=np.clip(np.nan_to_num(baseline,nan=float(th[0])),0.0,1.05)
    contrast=np.clip(np.nan_to_num(contrast,nan=float(th[1])),0.0,0.95)
    pred=baseline[:,None]+contrast[:,None]*X
    chi2=np.sum(((yy[None,:]-pred)/ee[None,:])**2,axis=1)
    return chi2,baseline,contrast


def field_seed(global_seed,field,nv):
    fcode=49 if field=="49G" else 52
    return np.random.SeedSequence([int(global_seed),fcode,int(nv)])


def candidate_catalog(base,t,e,row,catalog,orientation,p_occ,snr_cut,mode):
    full,weak,meta=v10.make_bath_catalogs(
        base,t,e,row,catalog,p_occ,snr_cut,orientation)
    if mode=="weak":
        return weak,meta
    if mode=="full":
        return full,meta
    raise ValueError(mode)


def ids_for_indices(cat,idx):
    if not len(idx): return []
    return cat.iloc[np.asarray(idx,int)].site_id.astype(int).tolist()

def fit_nv(base,field,t,y,e,row,sites,cat,role,n_realizations,top_refit,
           p_occ,snr_cut,mode,global_seed):
    rng=np.random.default_rng(field_seed(global_seed,field,int(row.nv_index)))
    L=single_spin_matrix(cat,t)
    baths,occ,nocc=draw_realizations(L,p_occ,n_realizations,rng)
    screen_chi2,screen_b,screen_c=fast_screen(base,t,y,e,row,sites,baths)
    order=np.argsort(screen_chi2)
    nkeep=min(int(top_refit),len(order))
    fitted=[]
    for rank,ri in enumerate(order[:nkeep],1):
        q=v10.fit_one(base,t,y,e,row,sites,baths[int(ri)])
        if q is None: continue
        fitted.append(dict(realization=int(ri),screen_rank=rank,
                           screen_chi2=float(screen_chi2[ri]),fit=q))
    if not fitted:
        raise RuntimeError(f"{field} NV{int(row.nv_index)}: all MC refits failed")
    fitted.sort(key=lambda z:z["fit"]["bic"])
    best=fitted[0]
    th6=np.asarray(json.loads(row.theta_json),float)
    p6=v6.v6_model(base,t,th6,sites)
    s6=v7.stats(y,base.safe_err(e),p6,int(row.npar))
    explicit=[int(row[f"c13_{j}_site_id"]) for j in range(1,int(row.model_order)+1)]
    best_idx=occ[best["realization"]]
    rec=dict(
        field=field,nv_index=int(row.nv_index),role=role,
        model_order=int(row.model_order),site_key=str(row.site_key),
        bath_mode=mode,p_occ=float(p_occ),snr_cut=float(snr_cut),
        n_realizations=int(n_realizations),top_refit=int(nkeep),
        candidate_sites=len(cat),expected_occupied=float(p_occ*len(cat)),
        best_realization=int(best["realization"]),
        best_n_occupied=int(nocc[best["realization"]]),
        best_site_ids=json.dumps(ids_for_indices(cat,best_idx)),
        explicit_site_ids=json.dumps(explicit),

        screen_chi2_min=float(np.min(screen_chi2)),
        screen_chi2_median=float(np.median(screen_chi2)),
        screen_chi2_p10=float(np.quantile(screen_chi2,.10)),
        screen_chi2_p90=float(np.quantile(screen_chi2,.90)),
        v6_redchi=float(s6["red_chi2"]),v6_bic=float(s6["bic"]),
        mc_redchi=float(best["fit"]["red_chi2"]),mc_bic=float(best["fit"]["bic"]),
        dbic_mc_v6=float(best["fit"]["bic"]-s6["bic"]),
        mc_T2_us=float(1000*best["fit"]["theta"][2]),
        mc_beta=float(best["fit"]["theta"][3]),
        mc_scale=(float(best["fit"]["theta"][4]) if sites else np.nan),
    )
    return rec,dict(row=row,sites=sites,cat=cat,baths=baths,occ=occ,nocc=nocc,
                    screen_chi2=screen_chi2,screen_b=screen_b,screen_c=screen_c,
                    fitted=fitted,best=best,p6=p6,s6=s6,
                    y=np.asarray(y,float),e=base.safe_err(e),role=role)


def evaluate(field,outdir,n_each,n_realizations,top_refit,p_occ,snr_cut,mode,seed):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths();t,y,e=base.load_data(ck)
    cand=v7.latest_v6_candidate(base,field)
    cdf=pd.read_csv(cand);bestdf=cdf[cdf.rank_global_bic==1].sort_values("nv_index")
    nvs,roles=v7.choose_subset(bestdf,field,n_each)
    catalog=v6.load_catalog(base)
    omap=v6.load_assigned_orientations(field,y.shape[0])
    rows=[];details=[]
    print(f"\n{field}: MC mode={mode}, M={n_realizations}, top_refit={top_refit}, p={p_occ:g}")

    for nv in nvs:
        row=bestdf[bestdf.nv_index==nv].iloc[0]
        if int(nv) not in omap: raise RuntimeError(f"No orientation for {field} NV{nv}")
        sites=v10.sites_from_row(row)
        cat,meta=candidate_catalog(base,t,e[nv],row,catalog,tuple(omap[int(nv)]),
                                   p_occ,snr_cut,mode)
        rec,det=fit_nv(base,field,t,y[nv],e[nv],row,sites,cat,roles[nv],
                       n_realizations,top_refit,p_occ,snr_cut,mode,seed)
        rec.update(n_full_sites=meta["n_full"],n_weak_sites=meta["n_weak"],
                   n_detectable_conditioned_unoccupied=meta["n_detectable_removed"],
                   assigned_orientation=meta["orientation"])
        rows.append(rec);details.append(det)
        print(f"  NV{nv:3d} {roles[nv]:12s} N={int(row.model_order)} "
              f"candidate={len(cat)} occ={rec['best_n_occupied']:3d} "
              f"chi2r {rec['v6_redchi']:.3f}->{rec['mc_redchi']:.3f} "
              f"dBIC(best-M)={rec['dbic_mc_v6']:+.1f}")
    tab=pd.DataFrame(rows)
    summary=pd.DataFrame([dict(
        field=field,n_nv=len(tab),bath_mode=mode,n_realizations=n_realizations,
        top_refit=top_refit,p_occ=p_occ,snr_cut=snr_cut,
        bestM_better_v6=int((tab.dbic_mc_v6<0).sum()),
        bestM_strong_better_v6=int((tab.dbic_mc_v6<=-6).sum()),
        median_dbic_bestM_v6=float(tab.dbic_mc_v6.median()),
        median_v6_redchi=float(tab.v6_redchi.median()),
        median_mc_redchi=float(tab.mc_redchi.median()),
        median_candidate_sites=float(tab.candidate_sites.median()),
        median_best_n_occupied=float(tab.best_n_occupied.median()))])
    stem=f"v11_conditional_mc_{mode}_{field}_M{n_realizations}"
    tab.to_csv(outdir/f"{stem}_per_nv.csv",index=False)
    summary.to_csv(outdir/f"{stem}_summary.csv",index=False)
    top_rows=[]

    configs={}
    for rec,det in zip(rows,details):
        nv=int(rec["nv_index"]); configs[str(nv)]={}
        for z in det["fitted"]:
            ri=int(z["realization"])
            top_rows.append(dict(field=field,nv_index=nv,role=rec["role"],
                realization=ri,screen_rank=int(z["screen_rank"]),
                screen_chi2=float(z["screen_chi2"]),
                fit_chi2=float(z["fit"]["chi2"]),fit_redchi=float(z["fit"]["red_chi2"]),
                fit_bic=float(z["fit"]["bic"]),
                dbic_v6=float(z["fit"]["bic"]-rec["v6_bic"]),
                n_occupied=int(det["nocc"][ri]),
                site_ids=json.dumps(ids_for_indices(det["cat"],det["occ"][ri]))))
        bri=int(det["best"]["realization"])
        configs[str(nv)]=dict(
            explicit_site_ids=json.loads(rec["explicit_site_ids"]),
            stochastic_occupied_site_ids=ids_for_indices(det["cat"],det["occ"][bri]),
            assigned_orientation=rec["assigned_orientation"],
            candidate_mode=mode,p_occ=p_occ,seed=seed,realization=bri)
    pd.DataFrame(top_rows).to_csv(outdir/f"{stem}_top_refits.csv",index=False)
    with open(outdir/f"{stem}_best_configurations.json","w",encoding="utf-8") as f:
        json.dump(configs,f,indent=2)
    pdf=outdir/f"{stem}.pdf"
    make_pdf(pdf,field,t,tab,details,mode,n_realizations,p_occ,snr_cut)
    return dict(field=field,table=tab,summary=summary,pdf=pdf,stem=stem)


def make_pdf(path,field,t,tab,details,mode,M,p_occ,snr_cut):
    with PdfPages(path) as pdf:
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(f"{field}: conditional Monte-Carlo lattice bath ({mode})",fontsize=16)
        x=np.arange(len(tab))
        axs[0,0].bar(x,tab.dbic_mc_v6)
        axs[0,0].axhline(0,lw=.8);axs[0,0].axhline(-6,ls="--",lw=.8)

        axs[0,0].set_xticks(x);axs[0,0].set_xticklabels(tab.nv_index.astype(str),rotation=60)
        axs[0,0].set(ylabel="Best-of-M BIC(MC)-BIC(V6)",xlabel="NV")
        axs[0,0].grid(alpha=.15,axis="y")
        axs[0,1].scatter(tab.v6_redchi,tab.mc_redchi,c=tab.model_order,s=55)
        mx=max(tab.v6_redchi.max(),tab.mc_redchi.max(),1)
        axs[0,1].plot([0,mx],[0,mx],ls="--",lw=1)
        axs[0,1].set(xlabel="V6 reduced chi-square",ylabel="Best MC reduced chi-square")
        axs[0,1].grid(alpha=.15)
        axs[1,0].scatter(tab.candidate_sites,tab.best_n_occupied,c=tab.model_order,s=55)
        xx=np.linspace(tab.candidate_sites.min(),tab.candidate_sites.max(),50)
        axs[1,0].plot(xx,p_occ*xx,ls="--",lw=1,label="p * N sites")
        axs[1,0].set(xlabel="Conditional candidate sites",ylabel="Occupied in best realization")
        axs[1,0].legend();axs[1,0].grid(alpha=.15)
        axs[1,1].axis("off")
        textlines=[
            f"bath mode = {mode}",f"realizations per NV = {M}",
            f"natural abundance p = {p_occ:g}",f"detectability cutoff = {snr_cut:g}",
            "", "Important:",
            "Best-of-M BIC is a diagnostic ranking, not formal evidence.",
            "Choosing the best random configuration creates a look-elsewhere effect.",
            "Use the screen distribution + reproducibility before population inference.",
        ]
        axs[1,1].text(.02,.98,"\n".join(textlines),va="top",
                      family="monospace",fontsize=9)
        fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);plt.close(fig)

        td=np.linspace(float(t.min()),float(t.max()),1200)
        for rec,det in zip(tab.to_dict("records"),details):
            row=det["row"];best=det["best"];ri=int(best["realization"])
            bath=det["baths"][ri]
            fig,axs=plt.subplots(2,2,figsize=(11,8.5))
            fig.suptitle(f"{field} NV {rec['nv_index']} | {rec['role']} | "
                         f"N={rec['model_order']} {rec['site_key']}",fontsize=14)
            axs[0,0].errorbar(t,det["y"],yerr=det["e"],fmt="o",ms=3,lw=.4,label="data")

            th6=np.asarray(json.loads(row.theta_json),float)
            axs[0,0].plot(td,v6.v6_model(v6.load_backend(field),td,th6,det["sites"]),
                          label="V6",lw=1.4)
            # Dense bath must be rebuilt from the winning occupied site list.
            occ=det["occ"][ri]
            Ld=single_spin_matrix(det["cat"],td)
            bd=np.prod(Ld[occ],axis=0) if len(occ) else np.ones_like(td)
            axs[0,0].plot(td,v10.model(td,best["fit"]["theta"],det["sites"],bd),
                          label="best MC bath",lw=1.4)
            axs[0,0].set(xlabel="Time (us)",ylabel="Signal")
            axs[0,0].legend(fontsize=7);axs[0,0].grid(alpha=.15)

            rr6=(det["y"]-det["p6"])/det["e"]
            rrm=(det["y"]-best["fit"]["pred"])/det["e"]
            axs[0,1].plot(t,rr6,"o-",ms=2,lw=.7,label="V6")
            axs[0,1].plot(t,rrm,"o-",ms=2,lw=.7,label="best MC")
            axs[0,1].axhline(0,lw=.7);axs[0,1].set(
                xlabel="Time (us)",ylabel="Normalized residual")
            axs[0,1].legend(fontsize=7);axs[0,1].grid(alpha=.15)

            vals=det["screen_chi2"]
            axs[1,0].hist(vals,bins=35)
            axs[1,0].axvline(vals[ri],ls="--",label="winning config")
            axs[1,0].set(xlabel="Fast-screen chi-square",ylabel="Realizations",
                         title=f"M={M} prior draws")
            axs[1,0].legend();axs[1,0].grid(alpha=.15)

            axs[1,1].axis("off")
            ids=ids_for_indices(det["cat"],det["occ"][ri])
            lines=[f"candidate sites = {rec['candidate_sites']}",
                   f"E[N13C] = {rec['expected_occupied']:.1f}",
                   f"winning N13C = {rec['best_n_occupied']}",
                   f"winning realization = {ri}",
                   f"V6 chi2r = {rec['v6_redchi']:.3f}",
                   f"MC chi2r = {rec['mc_redchi']:.3f}",
                   f"best-of-M dBIC = {rec['dbic_mc_v6']:+.2f}",
                   f"T2 = {rec['mc_T2_us']:.2f} us",f"beta = {rec['mc_beta']:.3f}",
                   "",f"occupied site IDs ({len(ids)}):",
                   ", ".join(map(str,ids[:45]))+(" ..." if len(ids)>45 else "")]
            axs[1,1].text(.02,.98,"\n".join(lines),va="top",
                          family="monospace",fontsize=8.1)
            fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--n-each",type=int,default=10)
    ap.add_argument("--bath-mode",choices=["weak","full"],default="weak")
    ap.add_argument("--realizations",type=int,default=N_REALIZATIONS_DEFAULT)
    ap.add_argument("--top-refit",type=int,default=TOP_REFIT_DEFAULT)
    ap.add_argument("--p-occ",type=float,default=P_OCC_DEFAULT)
    ap.add_argument("--snr-cut",type=float,default=SNR_CUT_DEFAULT)
    ap.add_argument("--seed",type=int,default=SEED_DEFAULT)
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    if args.realizations<1 or args.top_refit<1:
        raise ValueError("realizations and top-refit must be >= 1")
    ym=datetime.now().strftime("%Y_%m")
    outdir=(Path(args.output_dir) if args.output_dir else
            Path(r"G:\nvdata\pc_NVOffice\branch_master")/
            "spin_echo_v11_conditional_mc_bath"/ym)
    outdir.mkdir(parents=True,exist_ok=True)
    fields=["49G","52G"] if args.field=="both" else [args.field]
    res=[evaluate(f,outdir,args.n_each,args.realizations,args.top_refit,
                  args.p_occ,args.snr_cut,args.bath_mode,args.seed) for f in fields]
    if len(res)>1:
        tag=f"{args.bath_mode}_M{args.realizations}"
        pd.concat([x["table"] for x in res],ignore_index=True).to_csv(
            outdir/f"v11_conditional_mc_{tag}_both_per_nv.csv",index=False)
        pd.concat([x["summary"] for x in res],ignore_index=True).to_csv(
            outdir/f"v11_conditional_mc_{tag}_both_summary.csv",index=False)
    print("\nCOMPLETE")
    for x in res:
        print(x["pdf"]);print(x["summary"].to_string(index=False))


if __name__=="__main__":
    main()

