"""V8 diagnostic: field-anchored physical 13C bath + V6 discrete C13 terms.

For many weakly coupled bath 13C spins,
  prod_j [1 - 2*kappa_j sin^2(wI*tau/2) sin^2(wmj*tau/2)]
with wmj ~= wI gives the weak-coupling bath approximation
  L_bath(t) ~= exp[-Lambda * sin^4(pi*fL*t/2)]
for total evolution time t=2*tau.

The nuclear Larmor frequency fL is fixed to the field-specific value stored
in the same lattice catalog used by V6.  No arbitrary revival period, chirp,
revival width, taper, or revival-count comb is fitted.

Discrete strongly coupled C13 terms are kept exactly in the V6 additive form
for this first diagnostic, so this script isolates the bath model.

No lattice-site search is performed; V6 best-BIC site assignments are fixed.
"""
from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v7_diagnostic as v7

ROBUST_NFEV=1800
FINAL_NFEV=5000
LOGLAMBDA_BOUNDS=(np.log(1e-3),np.log(1e5))


def catalog_larmor_khz(base):
    cat=v6.load_catalog(base)
    if "fI_Hz" in cat.columns:
        vals=pd.to_numeric(cat.fI_Hz,errors="coerce").dropna().to_numpy(float)
        if len(vals): return float(np.median(vals)/1000.0)
    # f1-f0 = 2*fI for the catalog branches used here.
    vals=0.5*np.abs(pd.to_numeric(cat.f1_kHz,errors="coerce")-
                    pd.to_numeric(cat.f0_kHz,errors="coerce"))
    vals=vals[np.isfinite(vals)]
    if not len(vals): raise RuntimeError("Could not determine catalog 13C Larmor frequency")
    return float(np.median(vals))


def sites_from_row(r):
    out=[]
    ori=ast.literal_eval(str(r.orientation)) if int(r.model_order)>0 else ()
    for j in range(1,int(r.model_order)+1):
        out.append(dict(site_id=int(r[f"c13_{j}_site_id"]),
                        f0_kHz=float(r[f"c13_{j}_f0_kHz"]),
                        f1_kHz=float(r[f"c13_{j}_f1_kHz"]),
                        kappa=float(r[f"c13_{j}_kappa"]),
                        orientation=ori))
    return out

def physical_bath_carrier(t,bg5,fL_kHz):
    baseline,contrast,T2_ms,beta,loglam=np.asarray(bg5,float)
    tt=np.asarray(t,float)
    T2_us=max(1e-9,1000.0*T2_ms)
    env=np.exp(-np.power(np.maximum(tt,0)/T2_us,beta))
    fL_mhz=float(fL_kHz)/1000.0
    phase=np.pi*fL_mhz*tt/2.0
    lam=np.exp(loglam)
    bath=np.exp(-lam*np.sin(phase)**4)
    return float(baseline),float(contrast),env*bath


def model(t,theta,sites,fL_kHz):
    th=np.asarray(theta,float)
    baseline,contrast,carrier=physical_bath_carrier(t,th[:5],fL_kHz)
    tt=np.asarray(t,float)
    osc=np.zeros_like(tt)
    if sites:
        scale=float(th[5]);j=6
        for s in sites:
            phi0,phi1=th[j:j+2];j+=2
            amp=scale*contrast*float(s["kappa"])/4.0
            f0=float(s["f0_kHz"])/1000.0
            f1=float(s["f1_kHz"])/1000.0
            osc += amp*(np.cos(2*np.pi*f0*tt+phi0)+
                        np.cos(2*np.pi*f1*tt+phi1))
    return baseline-contrast*carrier+carrier*osc


def seed_from_v6(row,fL_kHz):
    th=np.asarray(json.loads(row.theta_json),float)
    width=max(float(row.width0_us),0.5)
    fL_mhz=float(fL_kHz)/1000.0
    # Match exp[-(delta/width)^4] near a physical revival:
    # Lambda*(pi*fL*delta/2)^4 ~= (delta/width)^4.
    lam=(2.0/(np.pi*fL_mhz*width))**4
    lam=float(np.clip(lam,np.exp(LOGLAMBDA_BOUNDS[0])*1.01,
                      np.exp(LOGLAMBDA_BOUNDS[1])/1.01))
    bg=np.array([th[0],th[1],th[4],th[5],np.log(lam)],float)
    if int(row.model_order):
        return np.r_[bg,th[9:]]
    return bg


def bounds(base,t,row):
    n=int(row.model_order)
    t2max=base.t2_upper_us(t)/1000.0
    lb=[0.0,0.0,0.001,0.4,LOGLAMBDA_BOUNDS[0]]
    ub=[1.05,0.95,t2max,8.0,LOGLAMBDA_BOUNDS[1]]
    if n:
        lb += [0.0] + [-np.pi,-np.pi]*n
        ub += [3.0] + [ np.pi, np.pi]*n
    return np.asarray(lb,float),np.asarray(ub,float)


def fit_one(base,t,y,e,row,sites,fL_kHz):
    seed=seed_from_v6(row,fL_kHz)
    lb,ub=bounds(base,t,row)
    seed=np.clip(seed,lb+1e-8,ub-1e-8)
    ee=base.safe_err(e)
    fun=lambda th:(np.asarray(y,float)-model(t,th,sites,fL_kHz))/ee
    seeds=[]
    # Broad width/bath-strength coverage plus beta alternatives.
    for mult in [0.25,0.6,1.0,2.0,5.0]:
        for beta in [max(.5,min(8.0,float(seed[3]))),2.0,4.0]:
            s=seed.copy()
            lam=np.clip(np.exp(seed[4])*mult,np.exp(lb[4]+1e-8),np.exp(ub[4]-1e-8))
            s[4]=np.log(lam);s[3]=np.clip(beta,lb[3]+1e-8,ub[3]-1e-8)
            seeds.append(s)
    robust=[]
    for s in seeds:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="soft_l1",f_scale=1,
                            max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((float(np.sum(fun(q.x)**2)),q.x))
        except Exception:
            pass
    if not robust:return None
    robust.sort(key=lambda z:z[0])
    finals=[]
    for _,s in robust[:4]:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="linear",
                            max_nfev=FINAL_NFEV,ftol=1e-10,xtol=1e-10,
                            gtol=1e-10,x_scale="jac")
            pred=model(t,q.x,sites,fL_kHz)
            st=v7.stats(y,ee,pred,len(q.x))
            finals.append(dict(theta=q.x,pred=pred,**st))
        except Exception:
            pass
    return min(finals,key=lambda z:z["chi2"]) if finals else None

def evaluate(field,outdir,n_each=10):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths()
    t,y,e=base.load_data(ck)
    cand=v7.latest_v6_candidate(base,field)
    cdf=pd.read_csv(cand)
    best=cdf[cdf.rank_global_bic==1].sort_values("nv_index").copy()
    nvs,roles=v7.choose_subset(best,field,n_each)
    fL=catalog_larmor_khz(base)
    trev=2000.0/fL
    print(f"\n{field}: fL={fL:.6f} kHz -> physical total-evolution revival {trev:.4f} us")
    rows=[];results=[]
    for ii,nv in enumerate(nvs,1):
        r=best[best.nv_index==nv].iloc[0]
        sites=sites_from_row(r)
        yy=np.asarray(y[nv],float);ee=base.safe_err(e[nv])
        th6=np.asarray(json.loads(r.theta_json),float)
        pred6=v6.v6_model(base,t,th6,sites)
        st6=v7.stats(yy,ee,pred6,int(r.npar))
        q=fit_one(base,t,yy,ee,r,sites,fL)
        if q is None:raise RuntimeError(f"physical bath fit failed: {field} NV{nv}")
        # BIC is compared per NV because all physical-bath parameters here are local.
        delta=float(q["bic"]-st6["bic"])
        th=q["theta"]
        lam=float(np.exp(th[4]))
        rows.append(dict(
            field=field,nv_index=nv,role=roles[nv],model_order=int(r.model_order),
            site_key=str(r.site_key),fL_kHz=fL,physical_revival_us=trev,
            v6_chi2=st6["chi2"],v6_redchi=st6["red_chi2"],v6_bic=st6["bic"],
            phys_chi2=q["chi2"],phys_redchi=q["red_chi2"],phys_bic=q["bic"],
            delta_bic_phys_minus_v6=delta,
            baseline=th[0],contrast=th[1],T2_us=1000*th[2],beta=th[3],
            Lambda_bath=lam,
            bath_scale=(th[5] if int(r.model_order)>0 else np.nan),
            residual_rms_v6=float(np.sqrt(np.mean(st6["residual"]**2))),
            residual_rms_phys=float(np.sqrt(np.mean(q["residual"]**2))),
        ))
        results.append(dict(nv=nv,role=roles[nv],row=r,sites=sites,
                            y=yy,e=ee,pred6=pred6,st6=st6,fit=q))
        print(f"  NV{nv:3d} {roles[nv]:12s} N={int(r.model_order)} "
              f"chi2r {st6['red_chi2']:.3f}->{q['red_chi2']:.3f} "
              f"dBIC={delta:+.1f} Lambda={lam:.2f}")

    tab=pd.DataFrame(rows)
    csv=outdir/f"v8_physical_bath_{field}_per_nv.csv"
    tab.to_csv(csv,index=False)
    summary=summarize(tab,field,fL,trev)
    scsv=outdir/f"v8_physical_bath_{field}_summary.csv"
    summary.to_csv(scsv,index=False)
    pdf=outdir/f"v8_physical_bath_{field}.pdf"
    make_pdf(pdf,field,t,tab,results,fL,trev)
    return dict(field=field,table=tab,summary=summary,pdf=pdf,csv=csv,summary_csv=scsv)


def summarize(tab,field,fL,trev):
    d=tab.delta_bic_phys_minus_v6
    return pd.DataFrame([dict(
        field=field,n_nv=len(tab),fL_kHz=fL,physical_revival_us=trev,
        phys_bic_better_count=int((d<0).sum()),
        phys_strong_better_count=int((d<=-6).sum()),
        v6_bic_better_count=int((d>0).sum()),
        median_delta_bic=float(d.median()),
        mean_delta_bic=float(d.mean()),
        sum_delta_bic=float(d.sum()),
        median_v6_redchi=float(tab.v6_redchi.median()),
        median_phys_redchi=float(tab.phys_redchi.median()),
        median_Lambda=float(tab.Lambda_bath.median()),
        median_beta=float(tab.beta.median()),
        median_T2_us=float(tab.T2_us.median()),
    )])

def make_pdf(path,field,t,tab,results,fL,trev):
    with PdfPages(path) as pdf:
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(f"{field}: field-anchored physical 13C bath diagnostic",fontsize=16)
        order=np.arange(len(tab))
        axs[0,0].bar(order,tab.delta_bic_phys_minus_v6)
        axs[0,0].axhline(0,lw=.8);axs[0,0].axhline(-6,ls="--",lw=.8)
        axs[0,0].set_xticks(order);axs[0,0].set_xticklabels(tab.nv_index.astype(str),rotation=60)
        axs[0,0].set(ylabel="BIC(physical bath) - BIC(V6)",xlabel="NV",
                     title="Per-NV model evidence");axs[0,0].grid(alpha=.15,axis="y")

        axs[0,1].scatter(tab.v6_redchi,tab.phys_redchi,c=tab.model_order,s=50)
        mx=max(tab.v6_redchi.max(),tab.phys_redchi.max(),1.0)
        axs[0,1].plot([0,mx],[0,mx],ls="--",lw=1)
        axs[0,1].set(xlabel="V6 reduced chi-square",ylabel="Physical-bath reduced chi-square",
                     title="Fit quality");axs[0,1].grid(alpha=.15)

        axs[1,0].scatter(tab.Lambda_bath,tab.beta,c=tab.model_order,s=50)
        axs[1,0].set(xscale="log",xlabel="Bath strength Lambda",ylabel="beta",
                     title="Physical-bath fitted parameters");axs[1,0].grid(alpha=.15)

        # Common residual comparison across subset.
        R6=[];RP=[]
        for z in results:
            R6.append((z["y"]-z["pred6"])/z["e"])
            RP.append((z["y"]-z["fit"]["pred"])/z["e"])
        axs[1,1].plot(t,np.nanmedian(np.asarray(R6),axis=0),label="V6")
        axs[1,1].plot(t,np.nanmedian(np.asarray(RP),axis=0),label="physical bath")
        axs[1,1].axhline(0,lw=.7)
        for n in [1,2]:
            x=n*trev
            if x<=t.max():axs[1,1].axvline(x,ls=":",lw=.7)
        axs[1,1].set(xlabel="Total evolution time (us)",ylabel="Median normalized residual",
                     title="Common residual structure");axs[1,1].legend();axs[1,1].grid(alpha=.15)
        fig.text(.5,.01,
                 f"fL fixed from lattice catalog = {fL:.6f} kHz; "
                 f"revivals fixed at 2/fL = {trev:.4f} us.  "
                 r"$L_{bath}=e^{-(t/T_2)^\beta}e^{-\Lambda\sin^4(\pi f_L t/2)}$",
                 ha="center",fontsize=9)
        fig.tight_layout(rect=[0,.035,1,.95]);pdf.savefig(fig);plt.close(fig)

        td=np.linspace(float(t.min()),float(t.max()),1500)
        for z in results:
            r=z["row"];q=z["fit"];sites=z["sites"]
            fig,axs=plt.subplots(2,2,figsize=(11,8.5))
            fig.suptitle(f"{field} NV {z['nv']} | {z['role']} | V6 N={int(r.model_order)} {r.site_key}",
                         fontsize=14)
            axs[0,0].errorbar(t,z["y"],yerr=z["e"],fmt="o",ms=3,lw=.4,label="data")
            th6=np.asarray(json.loads(r.theta_json),float)
            axs[0,0].plot(td,v6.v6_model(v6.load_backend(field),td,th6,sites),
                          label="V6",lw=1.5)
            axs[0,0].plot(td,model(td,q["theta"],sites,fL),label="physical bath",lw=1.5)
            axs[0,0].set(xlabel="Time (us)",ylabel="Signal",title="Full trace")
            axs[0,0].legend();axs[0,0].grid(alpha=.15)

            rr6=(z["y"]-z["pred6"])/z["e"];rrp=(z["y"]-q["pred"])/z["e"]
            axs[0,1].plot(t,rr6,"o-",ms=2,lw=.7,label="V6")
            axs[0,1].plot(t,rrp,"o-",ms=2,lw=.7,label="physical bath")
            axs[0,1].axhline(0,lw=.7)
            for n in [1,2]:
                x=n*trev
                if x<=t.max():axs[0,1].axvline(x,ls=":",lw=.7)
            axs[0,1].set(xlabel="Time (us)",ylabel="Normalized residual",
                         title="Residuals");axs[0,1].legend();axs[0,1].grid(alpha=.15)

            vals=[z["st6"]["red_chi2"],q["red_chi2"]]
            axs[1,0].bar([0,1],vals);axs[1,0].set_xticks([0,1])
            axs[1,0].set_xticklabels(["V6","physical bath"])
            axs[1,0].set_ylabel("Reduced chi-square")
            axs[1,0].set_title(f"Delta BIC = {q['bic']-z['st6']['bic']:+.1f}")
            axs[1,0].grid(alpha=.15,axis="y")

            axs[1,1].axis("off")
            th=q["theta"]
            lines=[
                f"fL (fixed) = {fL:.6f} kHz",
                f"physical revival 2/fL = {trev:.4f} us",
                f"Lambda = {np.exp(th[4]):.4g}",
                f"T2 = {1000*th[2]:.2f} us",
                f"beta = {th[3]:.3f}",
                f"baseline = {th[0]:.6f}",
                f"contrast = {th[1]:.6f}",
                f"V6 chi2r = {z['st6']['red_chi2']:.3f}",
                f"physical chi2r = {q['red_chi2']:.3f}",
                f"Delta BIC phys-V6 = {q['bic']-z['st6']['bic']:+.2f}",
            ]
            if int(r.model_order):
                lines += [f"V6 scale = {float(r.visibility_scale):.3f}",
                          f"physical-bath scale = {th[5]:.3f}"]
            axs[1,1].text(.02,.98,"\n".join(lines),va="top",
                          family="monospace",fontsize=9)
            fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--n-each",type=int,default=10)
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    ym=datetime.now().strftime("%Y_%m")
    outdir=(Path(args.output_dir) if args.output_dir else
            Path(r"G:\nvdata\pc_NVOffice\branch_master")/"spin_echo_v8_physical_bath"/ym)
    outdir.mkdir(parents=True,exist_ok=True)
    fields=["49G","52G"] if args.field=="both" else [args.field]
    res=[evaluate(f,outdir,args.n_each) for f in fields]
    if len(res)>1:
        pd.concat([x["table"] for x in res],ignore_index=True).to_csv(
            outdir/"v8_physical_bath_both_per_nv.csv",index=False)
        pd.concat([x["summary"] for x in res],ignore_index=True).to_csv(
            outdir/"v8_physical_bath_both_summary.csv",index=False)
    print("\nCOMPLETE")
    for x in res:
        print(x["pdf"])
        print(x["summary"].to_string(index=False))


if __name__=="__main__":
    main()
