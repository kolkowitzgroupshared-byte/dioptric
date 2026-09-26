"""V9 diagnostic: physical 13C bath with hyperfine-frequency spread.

Extends V8 by one physically meaningful parameter sigma_f:
  chi_bath(t) = Lambda sin^2(pi fL t/2)
                * [1 - cos(pi fL t) exp(-(pi sigma_f t)^2/2)]/2
  L_bath(t) = exp[-(t/T2)^beta] exp[-chi_bath(t)]

fL is fixed from the field-specific lattice catalog.  The V6 discrete-site
assignment and additive discrete-C13 form are kept fixed to isolate bath physics.
V9 -> V8 continuously as sigma_f -> 0.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v7_diagnostic as v7
import sc_spin_echo_v8_physical_bath_diagnostic as v8

ROBUST_NFEV=1800
FINAL_NFEV=5000
LOGSIGMA_BOUNDS=(np.log(1e-3),np.log(100.0))  # sigma_f in kHz

def bath_carrier(t,bg6,fL_kHz):
    baseline,contrast,T2_ms,beta,loglam,logsigma=np.asarray(bg6,float)
    tt=np.asarray(t,float)
    T2_us=max(1e-9,1000.0*T2_ms)
    fL=float(fL_kHz)/1000.0
    sigma=np.exp(logsigma)/1000.0  # MHz
    x=np.pi*fL*tt
    avg_second=0.5*(1.0-np.cos(x)*np.exp(-0.5*(np.pi*sigma*tt)**2))
    chi=np.exp(loglam)*np.sin(x/2.0)**2*avg_second
    env=np.exp(-np.power(np.maximum(tt,0)/T2_us,beta))
    return float(baseline),float(contrast),env*np.exp(-chi)


def model(t,theta,sites,fL_kHz):
    th=np.asarray(theta,float)
    baseline,contrast,carrier=bath_carrier(t,th[:6],fL_kHz)
    tt=np.asarray(t,float);osc=np.zeros_like(tt)
    if sites:
        scale=float(th[6]);j=7
        for s in sites:
            phi0,phi1=th[j:j+2];j+=2
            amp=scale*contrast*float(s["kappa"])/4.0
            f0=float(s["f0_kHz"])/1000.0
            f1=float(s["f1_kHz"])/1000.0
            osc += amp*(np.cos(2*np.pi*f0*tt+phi0)+
                        np.cos(2*np.pi*f1*tt+phi1))
    return baseline-contrast*carrier+carrier*osc


def seed_from_v8(row,fL_kHz,sigma_kHz=3.0):
    s8=v8.seed_from_v6(row,fL_kHz)
    return np.r_[s8[:5],np.log(float(sigma_kHz)),s8[5:]]

def bounds(base,t,row):
    n=int(row.model_order)
    t2max=base.t2_upper_us(t)/1000.0
    lb=[0.0,0.0,0.001,0.4,v8.LOGLAMBDA_BOUNDS[0],LOGSIGMA_BOUNDS[0]]
    ub=[1.05,0.95,t2max,8.0,v8.LOGLAMBDA_BOUNDS[1],LOGSIGMA_BOUNDS[1]]
    if n:
        lb += [0.0]+[-np.pi,-np.pi]*n
        ub += [3.0]+[ np.pi, np.pi]*n
    return np.asarray(lb,float),np.asarray(ub,float)


def fit_one(base,t,y,e,row,sites,fL_kHz):
    lb,ub=bounds(base,t,row);ee=base.safe_err(e)
    fun=lambda th:(np.asarray(y,float)-model(t,th,sites,fL_kHz))/ee
    seeds=[]
    for sig in [0.03,0.3,1.0,3.0,10.0,30.0]:
        s=seed_from_v8(row,fL_kHz,sig)
        s=np.clip(s,lb+1e-8,ub-1e-8)
        for mult in [0.5,1.0,2.0]:
            z=s.copy()
            lam=np.clip(np.exp(z[4])*mult,np.exp(lb[4]+1e-8),np.exp(ub[4]-1e-8))
            z[4]=np.log(lam);seeds.append(z)
    robust=[]
    for s in seeds:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="soft_l1",f_scale=1,
                            max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((float(np.sum(fun(q.x)**2)),q.x))
        except Exception: pass
    if not robust:return None
    robust.sort(key=lambda z:z[0]);finals=[]

    for _,s in robust[:4]:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="linear",
                            max_nfev=FINAL_NFEV,ftol=1e-10,xtol=1e-10,
                            gtol=1e-10,x_scale="jac")
            pred=model(t,q.x,sites,fL_kHz)
            st=v7.stats(y,ee,pred,len(q.x))
            finals.append(dict(theta=q.x,pred=pred,**st))
        except Exception: pass
    return min(finals,key=lambda z:z["chi2"]) if finals else None


def evaluate(field,outdir,n_each=10):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths();t,y,e=base.load_data(ck)
    cand=v7.latest_v6_candidate(base,field)
    cdf=pd.read_csv(cand);best=cdf[cdf.rank_global_bic==1].sort_values("nv_index")
    nvs,roles=v7.choose_subset(best,field,n_each)
    fL=v8.catalog_larmor_khz(base);trev=2000.0/fL
    print(f"\n{field}: fL={fL:.6f} kHz, revival={trev:.4f} us")
    rows=[];details=[]
    for nv in nvs:
        r=best[best.nv_index==nv].iloc[0];sites=v8.sites_from_row(r)
        yy=np.asarray(y[nv],float);ee=base.safe_err(e[nv])
        th6=np.asarray(json.loads(r.theta_json),float)
        p6=v6.v6_model(base,t,th6,sites);s6=v7.stats(yy,ee,p6,int(r.npar))
        q8=v8.fit_one(base,t,yy,ee,r,sites,fL)
        q9=fit_one(base,t,yy,ee,r,sites,fL)
        if q8 is None or q9 is None:raise RuntimeError(f"fit failed {field} NV{nv}")
        th9=q9["theta"];sig=np.exp(th9[5]);lam=np.exp(th9[4])
        rows.append(dict(field=field,nv_index=nv,role=roles[nv],
            model_order=int(r.model_order),site_key=str(r.site_key),
            fL_kHz=fL,physical_revival_us=trev,
            v6_redchi=s6["red_chi2"],v6_bic=s6["bic"],
            v8_redchi=q8["red_chi2"],v8_bic=q8["bic"],
            v9_redchi=q9["red_chi2"],v9_bic=q9["bic"],
            delta_bic_v9_minus_v6=q9["bic"]-s6["bic"],
            delta_bic_v9_minus_v8=q9["bic"]-q8["bic"],
            Lambda_bath=lam,sigma_f_kHz=sig,T2_us=1000*th9[2],beta=th9[3],
            bath_scale=(th9[6] if int(r.model_order)>0 else np.nan)))
        details.append(dict(nv=nv,role=roles[nv],row=r,sites=sites,y=yy,e=ee,
                            p6=p6,s6=s6,q8=q8,q9=q9))
        print(f"  NV{nv:3d} {roles[nv]:12s} N={int(r.model_order)} "
              f"BIC V9-V6={q9['bic']-s6['bic']:+.1f}, V9-V8={q9['bic']-q8['bic']:+.1f}, "
              f"sigma={sig:.3f} kHz")

    tab=pd.DataFrame(rows)
    summary=pd.DataFrame([dict(field=field,n_nv=len(tab),fL_kHz=fL,
        physical_revival_us=trev,
        v9_better_v6_count=int((tab.delta_bic_v9_minus_v6<0).sum()),
        v9_strong_better_v6_count=int((tab.delta_bic_v9_minus_v6<=-6).sum()),
        v9_better_v8_count=int((tab.delta_bic_v9_minus_v8<0).sum()),
        v9_strong_better_v8_count=int((tab.delta_bic_v9_minus_v8<=-6).sum()),
        median_dbic_v9_v6=float(tab.delta_bic_v9_minus_v6.median()),
        median_dbic_v9_v8=float(tab.delta_bic_v9_minus_v8.median()),
        median_v6_redchi=float(tab.v6_redchi.median()),
        median_v8_redchi=float(tab.v8_redchi.median()),
        median_v9_redchi=float(tab.v9_redchi.median()),
        median_sigma_f_kHz=float(tab.sigma_f_kHz.median()),
        median_Lambda=float(tab.Lambda_bath.median()))])
    tab.to_csv(outdir/f"v9_physical_bath_spread_{field}_per_nv.csv",index=False)
    summary.to_csv(outdir/f"v9_physical_bath_spread_{field}_summary.csv",index=False)
    pdf=outdir/f"v9_physical_bath_spread_{field}.pdf"
    make_pdf(pdf,field,t,tab,details,fL,trev)
    return dict(field=field,table=tab,summary=summary,pdf=pdf)


def make_pdf(path,field,t,tab,details,fL,trev):
    with PdfPages(path) as pdf:
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(f"{field}: physical 13C bath with frequency spread",fontsize=16)
        x=np.arange(len(tab))
        axs[0,0].bar(x-.18,tab.delta_bic_v9_minus_v6,width=.36,label="V9-V6")
        axs[0,0].bar(x+.18,tab.delta_bic_v9_minus_v8,width=.36,label="V9-V8")
        axs[0,0].axhline(0,lw=.8);axs[0,0].axhline(-6,ls="--",lw=.8)
        axs[0,0].set_xticks(x);axs[0,0].set_xticklabels(tab.nv_index.astype(str),rotation=60)
        axs[0,0].set_ylabel("Delta BIC");axs[0,0].legend();axs[0,0].grid(alpha=.15,axis="y")

        axs[0,1].scatter(tab.sigma_f_kHz,tab.Lambda_bath,c=tab.model_order,s=55)
        axs[0,1].set(xscale="log",yscale="log",xlabel="sigma_f (kHz)",
                     ylabel="Lambda",title="Bath parameters")
        axs[0,1].grid(alpha=.15)

        axs[1,0].scatter(tab.v6_redchi,tab.v9_redchi,c=tab.model_order,s=55)
        mx=max(tab.v6_redchi.max(),tab.v9_redchi.max(),1)
        axs[1,0].plot([0,mx],[0,mx],ls="--",lw=1)
        axs[1,0].set(xlabel="V6 reduced chi-square",ylabel="V9 reduced chi-square")
        axs[1,0].grid(alpha=.15)

        R6=[];R8=[];R9=[]
        for z in details:
            R6.append((z["y"]-z["p6"])/z["e"])
            R8.append((z["y"]-z["q8"]["pred"])/z["e"])
            R9.append((z["y"]-z["q9"]["pred"])/z["e"])
        for name,R in [("V6",R6),("V8 sigma=0",R8),("V9 spread",R9)]:
            axs[1,1].plot(t,np.nanmedian(np.asarray(R),axis=0),label=name)
        axs[1,1].axhline(0,lw=.7)
        for n in [1,2]:
            xx=n*trev
            if xx<=t.max():axs[1,1].axvline(xx,ls=":",lw=.7)
        axs[1,1].set(xlabel="Total evolution time (us)",ylabel="Median normalized residual")
        axs[1,1].legend();axs[1,1].grid(alpha=.15)
        fig.text(.5,.012,
            f"fL fixed={fL:.6f} kHz, 2/fL={trev:.4f} us. "
            r"$\chi=\Lambda\sin^2(x/2)[1-\cos x\,e^{-(\pi\sigma_f t)^2/2}]/2$",
            ha="center",fontsize=8.5)
        fig.tight_layout(rect=[0,.035,1,.95]);pdf.savefig(fig);plt.close(fig)

        td=np.linspace(float(t.min()),float(t.max()),1500)
        for z in details:
            r=z["row"];fig,axs=plt.subplots(2,2,figsize=(11,8.5))
            fig.suptitle(f"{field} NV {z['nv']} | {z['role']} | N={int(r.model_order)} {r.site_key}",
                         fontsize=14)
            axs[0,0].errorbar(t,z["y"],yerr=z["e"],fmt="o",ms=3,lw=.4,label="data")
            th6=np.asarray(json.loads(r.theta_json),float)
            axs[0,0].plot(td,v6.v6_model(v6.load_backend(field),td,th6,z["sites"]),
                          label="V6",lw=1.4)
            axs[0,0].plot(td,v8.model(td,z["q8"]["theta"],z["sites"],fL),
                          label="V8 sigma=0",lw=1.3)
            axs[0,0].plot(td,model(td,z["q9"]["theta"],z["sites"],fL),
                          label="V9 spread",lw=1.4)
            axs[0,0].set(xlabel="Time (us)",ylabel="Signal");axs[0,0].legend(fontsize=7)
            axs[0,0].grid(alpha=.15)
            for name,pred in [("V6",z["p6"]),("V8",z["q8"]["pred"]),("V9",z["q9"]["pred"])]:
                axs[0,1].plot(t,(z["y"]-pred)/z["e"],"o-",ms=2,lw=.7,label=name)
            axs[0,1].axhline(0,lw=.7);axs[0,1].legend(fontsize=7)
            axs[0,1].set(xlabel="Time (us)",ylabel="Normalized residual");axs[0,1].grid(alpha=.15)
            vals=[z["s6"]["red_chi2"],z["q8"]["red_chi2"],z["q9"]["red_chi2"]]
            axs[1,0].bar(range(3),vals);axs[1,0].set_xticks(range(3))
            axs[1,0].set_xticklabels(["V6","V8","V9"]);axs[1,0].set_ylabel("Reduced chi-square")
            axs[1,0].grid(alpha=.15,axis="y")
            th=z["q9"]["theta"];axs[1,1].axis("off")
            lines=[f"fL fixed = {fL:.6f} kHz",f"2/fL = {trev:.4f} us",
                   f"Lambda = {np.exp(th[4]):.4g}",f"sigma_f = {np.exp(th[5]):.4g} kHz",
                   f"T2 = {1000*th[2]:.2f} us",f"beta = {th[3]:.3f}",
                   f"dBIC V9-V6 = {z['q9']['bic']-z['s6']['bic']:+.2f}",
                   f"dBIC V9-V8 = {z['q9']['bic']-z['q8']['bic']:+.2f}"]
            axs[1,1].text(.02,.98,"\n".join(lines),va="top",family="monospace",fontsize=9)
            fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--n-each",type=int,default=10)
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    ym=datetime.now().strftime("%Y_%m")
    outdir=(Path(args.output_dir) if args.output_dir else
            Path(r"G:\nvdata\pc_NVOffice\branch_master")/"spin_echo_v9_physical_bath_spread"/ym)
    outdir.mkdir(parents=True,exist_ok=True)
    fields=["49G","52G"] if args.field=="both" else [args.field]
    res=[evaluate(f,outdir,args.n_each) for f in fields]
    if len(res)>1:
        pd.concat([x["table"] for x in res],ignore_index=True).to_csv(
            outdir/"v9_physical_bath_spread_both_per_nv.csv",index=False)
        pd.concat([x["summary"] for x in res],ignore_index=True).to_csv(
            outdir/"v9_physical_bath_spread_both_summary.csv",index=False)
    print("\nCOMPLETE")
    for x in res:
        print(x["pdf"]);print(x["summary"].to_string(index=False))


if __name__=="__main__":
    main()
