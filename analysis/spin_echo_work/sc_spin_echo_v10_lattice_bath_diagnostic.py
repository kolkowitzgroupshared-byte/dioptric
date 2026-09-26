"""V10 diagnostic: lattice-averaged natural-abundance 13C bath.

For every actual carbon site i in the assigned-orientation lattice catalog,
the exact single-spin Hahn factor for total evolution t=2*tau is

    L_i(t) = 1 - 2*kappa_i
                  sin^2(pi*fI_i*t/2) sin^2(pi*fm_i*t/2).

With independent 13C occupation probability p, an unresolved site's
configuration-averaged factor is
    (1-p) + p*L_i = 1 - 2*p*kappa_i*sin^2(...)*sin^2(...).

The bath is the product of these factors.  Explicit V6 C13 sites are removed
from the bath to avoid double counting and remain in the V6 additive discrete
term.  Two bath definitions are tested:
  FULL : all physical catalog carbon sites except explicit sites.
  WEAK : additionally remove sites whose V6 physical detectability SNR >= 0.5.

The zero-distance defect-center row is never treated as a carbon bath site.
No lattice-site search is performed in this diagnostic.
"""
from __future__ import annotations

import argparse, ast, json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v7_diagnostic as v7

P_OCC_DEFAULT=0.011
SNR_CUT_DEFAULT=0.5
ROBUST_NFEV=1400
FINAL_NFEV=4000

def sites_from_row(r):
    out=[]
    ori=ast.literal_eval(str(r.orientation)) if str(r.orientation)!="nan" else ()
    for j in range(1,int(r.model_order)+1):
        out.append(dict(site_id=int(r[f"c13_{j}_site_id"]),
                        f0_kHz=float(r[f"c13_{j}_f0_kHz"]),
                        f1_kHz=float(r[f"c13_{j}_f1_kHz"]),
                        kappa=float(r[f"c13_{j}_kappa"]),
                        orientation=ori))
    return out


def occupancy_factor(cat,t,p_occ,chunk=1000):
    """Return product_i [(1-p)+p L_i(t)] using log-space chunking."""
    tt=np.asarray(t,float)[None,:]
    acc=np.zeros(tt.shape[1],float)
    for start in range(0,len(cat),chunk):
        q=cat.iloc[start:start+chunk]
        kap=q.kappa.to_numpy(float)[:,None]
        fI=q.fI_Hz.to_numpy(float)[:,None]/1e6
        fm=q.omega_ms_Hz.to_numpy(float)[:,None]/1e6
        x=(2.0*float(p_occ)*kap
           * np.sin(0.5*np.pi*fI*tt)**2
           * np.sin(0.5*np.pi*fm*tt)**2)
        x=np.clip(x,0.0,1.0-1e-12)
        acc += np.sum(np.log1p(-x),axis=0)
    return np.exp(acc)


def make_bath_catalogs(base,t,e,row,catalog,p_occ,snr_cut,orientation):
    ori=tuple(orientation)
    g=catalog[(catalog.ori==ori)&(catalog.distance_A>1e-6)].copy()
    explicit={int(row[f"c13_{j}_site_id"])
              for j in range(1,int(row.model_order)+1)}
    g=g[~g.site_id.isin(explicit)].copy()
    th=np.asarray(json.loads(row.theta_json),float)
    snr=v6.physical_scores(base,t,e,th[:9],g)
    full=g.copy()
    weak=g.loc[snr<float(snr_cut)].copy()
    meta=dict(orientation=str(ori),n_catalog_physical=len(g)+len(explicit),
              n_explicit=len(explicit),n_full=len(full),n_weak=len(weak),
              n_detectable_removed=int(np.sum(snr>=float(snr_cut))))
    return full,weak,meta

def model(t,theta,sites,bath):
    """Lattice bath carrier + V6 additive discrete-site term."""
    th=np.asarray(theta,float)
    baseline,contrast,T2_ms,beta=th[:4]
    tt=np.asarray(t,float)
    T2_us=max(1e-9,1000.0*T2_ms)
    env=np.exp(-np.power(np.maximum(tt,0)/T2_us,beta))
    carrier=env*np.asarray(bath,float)
    osc=np.zeros_like(tt)
    if sites:
        scale=float(th[4]);j=5
        for s in sites:
            phi0,phi1=th[j:j+2];j+=2
            amp=scale*contrast*float(s["kappa"])/4.0
            f0=float(s["f0_kHz"])/1000.0
            f1=float(s["f1_kHz"])/1000.0
            osc += amp*(np.cos(2*np.pi*f0*tt+phi0)+
                        np.cos(2*np.pi*f1*tt+phi1))
    return baseline-contrast*carrier+carrier*osc


def seed_from_v6(row):
    th=np.asarray(json.loads(row.theta_json),float)
    bg=np.array([th[0],th[1],th[4],th[5]],float)
    if int(row.model_order):
        return np.r_[bg,th[9:]]
    return bg


def bounds(base,t,row):
    n=int(row.model_order);t2max=base.t2_upper_us(t)/1000.0
    lb=[0.0,0.0,0.001,0.4];ub=[1.05,0.95,t2max,4.0]
    if n:
        lb += [0.0]+[-np.pi,-np.pi]*n
        ub += [3.0]+[ np.pi, np.pi]*n
    return np.asarray(lb,float),np.asarray(ub,float)


def fit_one(base,t,y,e,row,sites,bath):
    seed=seed_from_v6(row);lb,ub=bounds(base,t,row)
    seed=np.clip(seed,lb+1e-8,ub-1e-8);ee=base.safe_err(e)
    fun=lambda th:(np.asarray(y,float)-model(t,th,sites,bath))/ee
    seeds=[seed]
    if sites:
        for sc in [0.5,1.0,2.0,2.9]:
            s=seed.copy();s[4]=sc;seeds.append(s)
    for beta in [1.0,2.0,3.5]:
        s=seed.copy();s[3]=beta;seeds.append(s)
    robust=[]
    for s in seeds:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="soft_l1",f_scale=1,
                            max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((float(np.sum(fun(q.x)**2)),q.x))
        except Exception: pass
    if not robust:return None
    robust.sort(key=lambda z:z[0]); finals=[]

    for _,s in robust[:3]:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="linear",
                            max_nfev=FINAL_NFEV,ftol=1e-10,xtol=1e-10,
                            gtol=1e-10,x_scale="jac")
            pred=model(t,q.x,sites,bath)
            st=v7.stats(y,ee,pred,len(q.x))
            finals.append(dict(theta=q.x,pred=pred,**st))
        except Exception: pass
    return min(finals,key=lambda z:z["chi2"]) if finals else None


def evaluate(field,outdir,n_each=10,p_occ=P_OCC_DEFAULT,snr_cut=SNR_CUT_DEFAULT):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths();t,y,e=base.load_data(ck)
    cand=v7.latest_v6_candidate(base,field)
    cdf=pd.read_csv(cand);best=cdf[cdf.rank_global_bic==1].sort_values("nv_index")
    nvs,roles=v7.choose_subset(best,field,n_each)
    catalog=v6.load_catalog(base)
    orientation_map=v6.load_assigned_orientations(field,y.shape[0])
    rows=[];details=[]
    print(f"\n{field}: p_occ={p_occ:g}, unresolved SNR cutoff={snr_cut:g}")
    for nv in nvs:
        r=best[best.nv_index==nv].iloc[0]
        sites=sites_from_row(r); yy=np.asarray(y[nv],float); ee=base.safe_err(e[nv])
        if int(nv) not in orientation_map:
            raise RuntimeError(f"No assigned orientation for {field} NV{nv}")
        assigned_ori=tuple(orientation_map[int(nv)])
        fullcat,weakcat,meta=make_bath_catalogs(
            base,t,e[nv],r,catalog,p_occ,snr_cut,assigned_ori)
        bath_full=occupancy_factor(fullcat,t,p_occ)
        bath_weak=occupancy_factor(weakcat,t,p_occ)
        th6=np.asarray(json.loads(r.theta_json),float)
        p6=v6.v6_model(base,t,th6,sites);s6=v7.stats(yy,ee,p6,int(r.npar))
        qf=fit_one(base,t,yy,ee,r,sites,bath_full)
        qw=fit_one(base,t,yy,ee,r,sites,bath_weak)
        if qf is None or qw is None: raise RuntimeError(f"fit failed {field} NV{nv}")
        rows.append(dict(field=field,nv_index=nv,role=roles[nv],
            model_order=int(r.model_order),site_key=str(r.site_key),
            p_occ=p_occ,snr_cut=snr_cut,**meta,
            bath_full_min=float(bath_full.min()),bath_weak_min=float(bath_weak.min()),
            v6_redchi=s6["red_chi2"],v6_bic=s6["bic"],
            full_redchi=qf["red_chi2"],full_bic=qf["bic"],
            weak_redchi=qw["red_chi2"],weak_bic=qw["bic"],
            dbic_full_v6=qf["bic"]-s6["bic"],
            dbic_weak_v6=qw["bic"]-s6["bic"],
            dbic_weak_full=qw["bic"]-qf["bic"],
            full_T2_us=1000*qf["theta"][2],full_beta=qf["theta"][3],
            weak_T2_us=1000*qw["theta"][2],weak_beta=qw["theta"][3],
            full_scale=(qf["theta"][4] if int(r.model_order)>0 else np.nan),
            weak_scale=(qw["theta"][4] if int(r.model_order)>0 else np.nan)))
        details.append(dict(nv=nv,role=roles[nv],row=r,sites=sites,y=yy,e=ee,
                            p6=p6,s6=s6,qf=qf,qw=qw,fullcat=fullcat,weakcat=weakcat,
                            bath_full=bath_full,bath_weak=bath_weak,meta=meta))
        print(f"  NV{nv:3d} {roles[nv]:12s} N={int(r.model_order)} "
              f"dBIC full={qf['bic']-s6['bic']:+.1f} weak={qw['bic']-s6['bic']:+.1f} "
              f"bath min {bath_full.min():.3g}/{bath_weak.min():.3g} "
              f"sites {len(fullcat)}/{len(weakcat)}")

    tab=pd.DataFrame(rows)
    summary=pd.DataFrame([dict(field=field,n_nv=len(tab),p_occ=p_occ,snr_cut=snr_cut,
        full_better_v6=int((tab.dbic_full_v6<0).sum()),
        full_strong_better_v6=int((tab.dbic_full_v6<=-6).sum()),
        weak_better_v6=int((tab.dbic_weak_v6<0).sum()),
        weak_strong_better_v6=int((tab.dbic_weak_v6<=-6).sum()),
        weak_better_full=int((tab.dbic_weak_full<0).sum()),
        median_dbic_full_v6=float(tab.dbic_full_v6.median()),
        median_dbic_weak_v6=float(tab.dbic_weak_v6.median()),
        median_dbic_weak_full=float(tab.dbic_weak_full.median()),
        median_v6_redchi=float(tab.v6_redchi.median()),
        median_full_redchi=float(tab.full_redchi.median()),
        median_weak_redchi=float(tab.weak_redchi.median()),
        median_full_bath_min=float(tab.bath_full_min.median()),
        median_weak_bath_min=float(tab.bath_weak_min.median()),
        median_detectable_removed=float(tab.n_detectable_removed.median()))])
    tab.to_csv(outdir/f"v10_lattice_bath_{field}_per_nv.csv",index=False)
    summary.to_csv(outdir/f"v10_lattice_bath_{field}_summary.csv",index=False)
    pdf=outdir/f"v10_lattice_bath_{field}.pdf"
    make_pdf(pdf,field,t,tab,details,p_occ,snr_cut)
    return dict(field=field,table=tab,summary=summary,pdf=pdf)


def make_pdf(path,field,t,tab,details,p_occ,snr_cut):
    with PdfPages(path) as pdf:
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(f"{field}: lattice-averaged natural-abundance 13C bath",fontsize=16)
        x=np.arange(len(tab))
        axs[0,0].bar(x-.18,tab.dbic_full_v6,width=.36,label="FULL - V6")
        axs[0,0].bar(x+.18,tab.dbic_weak_v6,width=.36,label="WEAK - V6")
        axs[0,0].axhline(0,lw=.8);axs[0,0].axhline(-6,ls="--",lw=.8)
        axs[0,0].set_xticks(x);axs[0,0].set_xticklabels(tab.nv_index.astype(str),rotation=60)
        axs[0,0].set_ylabel("Delta BIC");axs[0,0].legend();axs[0,0].grid(alpha=.15,axis="y")

        axs[0,1].scatter(tab.v6_redchi,tab.full_redchi,label="FULL",s=45)
        axs[0,1].scatter(tab.v6_redchi,tab.weak_redchi,label="WEAK",s=45,marker="x")
        mx=max(tab.v6_redchi.max(),tab.full_redchi.max(),tab.weak_redchi.max(),1)
        axs[0,1].plot([0,mx],[0,mx],ls="--",lw=1)
        axs[0,1].set(xlabel="V6 reduced chi-square",ylabel="Lattice-bath reduced chi-square")
        axs[0,1].legend();axs[0,1].grid(alpha=.15)

        axs[1,0].scatter(tab.n_detectable_removed,tab.dbic_weak_full,
                         c=tab.model_order,s=55)
        axs[1,0].axhline(0,lw=.8)
        axs[1,0].set(xlabel=f"Sites removed as detectable (SNR >= {snr_cut:g})",
                     ylabel="BIC(WEAK)-BIC(FULL)",title="Does conditioning help?")
        axs[1,0].grid(alpha=.15)

        R6=[];RF=[];RW=[]
        for z in details:
            R6.append((z["y"]-z["p6"])/z["e"])
            RF.append((z["y"]-z["qf"]["pred"])/z["e"])
            RW.append((z["y"]-z["qw"]["pred"])/z["e"])
        for name,R in [("V6",R6),("lattice FULL",RF),("lattice WEAK",RW)]:
            axs[1,1].plot(t,np.nanmedian(np.asarray(R),axis=0),label=name)
        axs[1,1].axhline(0,lw=.7);axs[1,1].set(
            xlabel="Total evolution time (us)",ylabel="Median normalized residual",
            title="Common residual across subset")
        axs[1,1].legend();axs[1,1].grid(alpha=.15)
        fig.text(.5,.012,
            f"Natural abundance p={p_occ:g}. FULL uses all physical carbon sites; "
            f"WEAK removes sites with V6 physical SNR >= {snr_cut:g}. "
            "Explicit V6 sites are excluded from both baths.",
            ha="center",fontsize=8.5)
        fig.tight_layout(rect=[0,.035,1,.95]);pdf.savefig(fig);plt.close(fig)

        td=np.linspace(float(t.min()),float(t.max()),1500)
        for z in details:
            r=z["row"]
            bf=occupancy_factor(z["fullcat"],td,p_occ)
            bw=occupancy_factor(z["weakcat"],td,p_occ)
            fig,axs=plt.subplots(2,2,figsize=(11,8.5))
            fig.suptitle(f"{field} NV {z['nv']} | {z['role']} | N={int(r.model_order)} {r.site_key}",
                         fontsize=14)
            axs[0,0].errorbar(t,z["y"],yerr=z["e"],fmt="o",ms=3,lw=.4,label="data")
            th6=np.asarray(json.loads(r.theta_json),float)
            axs[0,0].plot(td,v6.v6_model(v6.load_backend(field),td,th6,z["sites"]),
                          label="V6",lw=1.4)
            axs[0,0].plot(td,model(td,z["qf"]["theta"],z["sites"],bf),
                          label="lattice FULL",lw=1.3)
            axs[0,0].plot(td,model(td,z["qw"]["theta"],z["sites"],bw),
                          label="lattice WEAK",lw=1.3)
            axs[0,0].set(xlabel="Time (us)",ylabel="Signal");axs[0,0].legend(fontsize=7)
            axs[0,0].grid(alpha=.15)

            for name,pred in [("V6",z["p6"]),("FULL",z["qf"]["pred"]),("WEAK",z["qw"]["pred"])]:
                axs[0,1].plot(t,(z["y"]-pred)/z["e"],"o-",ms=2,lw=.7,label=name)
            axs[0,1].axhline(0,lw=.7);axs[0,1].set(
                xlabel="Time (us)",ylabel="Normalized residual")
            axs[0,1].legend(fontsize=7);axs[0,1].grid(alpha=.15)

            axs[1,0].plot(td,bf,label=f"FULL ({len(z['fullcat'])} sites)")
            axs[1,0].plot(td,bw,label=f"WEAK ({len(z['weakcat'])} sites)")
            axs[1,0].set(xlabel="Time (us)",ylabel="L_bath(t)",title="Predicted lattice bath")
            axs[1,0].legend(fontsize=7);axs[1,0].grid(alpha=.15)

            axs[1,1].axis("off")
            qf=z["qf"];qw=z["qw"];m=z["meta"]
            lines=[f"orientation = {m['orientation']}",
                   f"p(13C) = {p_occ:g}",f"explicit C13 removed = {m['n_explicit']}",
                   f"FULL bath sites = {m['n_full']}",f"WEAK bath sites = {m['n_weak']}",
                   f"detectable sites removed = {m['n_detectable_removed']}",
                   "",f"V6 chi2r = {z['s6']['red_chi2']:.3f}",
                   f"FULL chi2r = {qf['red_chi2']:.3f}",
                   f"WEAK chi2r = {qw['red_chi2']:.3f}",
                   f"dBIC FULL-V6 = {qf['bic']-z['s6']['bic']:+.2f}",
                   f"dBIC WEAK-V6 = {qw['bic']-z['s6']['bic']:+.2f}",
                   f"dBIC WEAK-FULL = {qw['bic']-qf['bic']:+.2f}"]
            if int(r.model_order):
                lines += ["",f"V6 scale = {float(r.visibility_scale):.3f}",
                          f"FULL scale = {qf['theta'][4]:.3f}",
                          f"WEAK scale = {qw['theta'][4]:.3f}"]
            axs[1,1].text(.02,.98,"\n".join(lines),va="top",family="monospace",fontsize=8.8)
            fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--n-each",type=int,default=10)
    ap.add_argument("--p-occ",type=float,default=P_OCC_DEFAULT)
    ap.add_argument("--snr-cut",type=float,default=SNR_CUT_DEFAULT)
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    ym=datetime.now().strftime("%Y_%m")
    outdir=(Path(args.output_dir) if args.output_dir else
            Path(r"G:\nvdata\pc_NVOffice\branch_master")/"spin_echo_v10_lattice_bath"/ym)
    outdir.mkdir(parents=True,exist_ok=True)
    fields=["49G","52G"] if args.field=="both" else [args.field]
    res=[evaluate(f,outdir,args.n_each,args.p_occ,args.snr_cut) for f in fields]
    if len(res)>1:
        pd.concat([x["table"] for x in res],ignore_index=True).to_csv(
            outdir/"v10_lattice_bath_both_per_nv.csv",index=False)
        pd.concat([x["summary"] for x in res],ignore_index=True).to_csv(
            outdir/"v10_lattice_bath_both_summary.csv",index=False)
    print("\nCOMPLETE")
    for x in res:
        print(x["pdf"]);print(x["summary"].to_string(index=False))


if __name__=="__main__":
    main()
