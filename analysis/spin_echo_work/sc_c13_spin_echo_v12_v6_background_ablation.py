"""V12 diagnostic: ablate the phenomenological V6 background, one term at a time.

Discrete-C13 site assignments and the V6 discrete architecture are unchanged.
For every ablation, all remaining free V6 parameters are re-optimized.

Background theta indices:
  0 baseline, 1 contrast, 2 Trev, 3 width0, 4 T2_ms,
  5 beta, 6 taper, 7 width_slope, 8 revival_chirp.

Trev_phys is fixed from the same field-specific lattice catalog:
  Trev_phys(total evolution) = 2 / fL.

Outputs report raw delta-chi2 AND delta-BIC relative to a same-optimizer
full-V6 refit.  This cleanly separates fit loss from parameter-count savings.
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

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v7_diagnostic as v7
import sc_c13_spin_echo_v8_physical_bath_diagnostic as v8
import sc_c13_spin_echo_v10_lattice_bath_diagnostic as v10

ROBUST_NFEV=1800
FINAL_NFEV=5000

BG_NAMES=["baseline","contrast","Trev","width0","T2","beta","taper","slope","chirp"]

def model_specs(trev_phys):
    """Ordered nested/single-parameter ablation suite."""
    return [
        ("M0_full",{}, "full V6"),
        ("M1_TrevPhys",{2:trev_phys},"Trev = 2/fL"),
        ("M2_chirp0",{8:0.0},"chirp = 0"),
        ("M3_slope0",{7:0.0},"width slope = 0"),
        ("M4_taper0",{6:0.0},"revival taper = 0"),
        ("M5_beta1",{5:1.0},"beta = 1"),
        ("M6_beta2",{5:2.0},"beta = 2"),
        ("M7_physTiming",{2:trev_phys,8:0.0},"Trev = 2/fL; chirp = 0"),
        ("M8_noAmpWidthEvol",{6:0.0,7:0.0},"taper = 0; slope = 0"),
        ("M9_simpleComb",{6:0.0,7:0.0,8:0.0},"taper=slope=chirp=0"),
        ("M10_physSimple",{2:trev_phys,6:0.0,7:0.0,8:0.0},
         "physical Trev; taper=slope=chirp=0"),
        ("M11_physSimpleBeta2",{2:trev_phys,5:2.0,6:0.0,7:0.0,8:0.0},
         "M10 + beta=2"),
    ]


def full_bounds(base,t,row):
    n=int(row.model_order)
    baseline_seed=float(row.baseline)
    return v6.v6_theta_bounds(base,n,baseline_seed,base.t2_upper_us(t))

def project_free(theta,free_idx):
    return np.asarray(theta,float)[np.asarray(free_idx,int)]


def expand_free(x,template,free_idx,fixed):
    th=np.asarray(template,float).copy()
    th[np.asarray(free_idx,int)]=np.asarray(x,float)
    for i,val in fixed.items():
        th[int(i)]=float(val)
    return th


def fit_ablation(base,t,y,e,row,sites,fixed,extra_full_seeds=None):
    seed=np.asarray(json.loads(row.theta_json),float)
    lb,ub=full_bounds(base,t,row)
    fixed={int(i):float(v) for i,v in fixed.items()}
    for i,val in fixed.items():
        if val < lb[i]-1e-12 or val > ub[i]+1e-12:
            raise ValueError(f"fixed {BG_NAMES[i]}={val} outside [{lb[i]},{ub[i]}]")
        seed[i]=val
    free_idx=[i for i in range(len(seed)) if i not in fixed]
    lbf=lb[free_idx];ubf=ub[free_idx]
    x0=np.clip(seed[free_idx],lbf+1e-9,ubf-1e-9)
    ee=base.safe_err(e)
    fun=lambda x:(np.asarray(y,float)-v6.v6_model(
        base,t,expand_free(x,seed,free_idx,fixed),sites))/ee

    seeds=[x0]
    # Promote solutions from other nested models into this parameterization.
    # This is crucial because the phenomenological comb has local minima.
    for full in (extra_full_seeds or []):
        s=np.asarray(full,float).copy()
        if s.size != seed.size:
            continue
        for i,val in fixed.items():
            s[int(i)]=float(val)
        s=np.clip(s,lb+1e-9,ub-1e-9)
        seeds.append(s[free_idx])
    # Deterministic alternatives mainly perturb width/T2 while preserving phases.
    for fT2,fW in [(0.7,0.8),(1.35,1.25)]:
        s=seed.copy()
        if 4 not in fixed:s[4]=np.clip(s[4]*fT2,lb[4]+1e-8,ub[4]-1e-8)
        if 3 not in fixed:s[3]=np.clip(s[3]*fW,lb[3]+1e-8,ub[3]-1e-8)
        seeds.append(np.clip(s[free_idx],lbf+1e-9,ubf-1e-9))
    # Keep every seed itself as a valid candidate.  This guarantees a more
    # general model can never be reported worse than a nested solution supplied
    # as one of its seeds, even if least_squares walks into another basin.
    finals=[]
    for s in seeds:
        try:
            th0=expand_free(s,seed,free_idx,fixed)
            pred0=v6.v6_model(base,t,th0,sites)
            st0=v7.stats(y,ee,pred0,len(free_idx))
            finals.append(dict(theta=th0,pred=pred0,free_idx=free_idx,
                               fixed=fixed,**st0))
        except Exception:
            pass
    robust=[]
    for s in seeds:
        try:
            q=least_squares(fun,s,bounds=(lbf,ubf),loss="soft_l1",f_scale=1,
                            max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((float(np.sum(fun(q.x)**2)),q.x))
        except Exception:
            pass
    if not robust and not finals:return None
    robust.sort(key=lambda z:z[0])

    for _,s in robust[:2]:
        try:
            q=least_squares(fun,s,bounds=(lbf,ubf),loss="linear",
                            max_nfev=FINAL_NFEV,ftol=1e-10,xtol=1e-10,
                            gtol=1e-10,x_scale="jac")
            th=expand_free(q.x,seed,free_idx,fixed)
            pred=v6.v6_model(base,t,th,sites)
            st=v7.stats(y,ee,pred,len(free_idx))
            finals.append(dict(theta=th,pred=pred,free_idx=free_idx,
                               fixed=fixed,**st))
        except Exception:
            pass
    return min(finals,key=lambda z:z["chi2"]) if finals else None


def evaluate(field,outdir,n_each=10):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths();t,y,e=base.load_data(ck)
    cand=v7.latest_v6_candidate(base,field)
    cdf=pd.read_csv(cand);bestdf=cdf[cdf.rank_global_bic==1].sort_values("nv_index")
    nvs,roles=v7.choose_subset(bestdf,field,n_each)
    fL=v8.catalog_larmor_khz(base);trev_phys=2000.0/fL
    specs=model_specs(trev_phys)
    rows=[];details={}
    print(f"\n{field}: V6 background ablation, fL={fL:.6f} kHz, "
          f"Trev_phys={trev_phys:.4f} us")
    for nv in nvs:
        row=bestdf[bestdf.nv_index==nv].iloc[0]
        sites=v10.sites_from_row(row)
        yy=np.asarray(y[nv],float);ee=base.safe_err(e[nv])
        # Pass 1: each nested model from the saved V6 optimum.
        prelim={}
        for name,fixed,desc in specs:
            q=fit_ablation(base,t,yy,ee,row,sites,fixed)
            if q is None:raise RuntimeError(f"{field} NV{nv} {name} preliminary fit failed")
            prelim[name]=q
        # A constrained fit can sometimes escape a local minimum that trapped V6.
        # Polish the unrestricted model from *all* nested solutions; mathematically
        # the unrestricted optimum must have chi2 <= every constrained optimum.
        full_seeds=[q["theta"] for q in prelim.values()]
        ref=fit_ablation(base,t,yy,ee,row,sites,{},extra_full_seeds=full_seeds)
        if ref is None:raise RuntimeError(f"{field} NV{nv} full polish failed")
        # Pass 2: refit every ablation from both its preliminary solution and the
        # polished unrestricted solution, removing seed-quality bias.
        fitmap={"M0_full":ref}
        for name,fixed,desc in specs[1:]:
            q=fit_ablation(base,t,yy,ee,row,sites,fixed,
                           extra_full_seeds=[ref["theta"],prelim[name]["theta"]])
            if q is None:raise RuntimeError(f"{field} NV{nv} {name} polished fit failed")
            fitmap[name]=q
        # Closure polish: second-pass nested fits can themselves discover a new
        # basin. Seed the unrestricted model from every final nested solution.
        ref2=fit_ablation(base,t,yy,ee,row,sites,{},
                          extra_full_seeds=[q["theta"] for q in fitmap.values()])
        if ref2 is None:raise RuntimeError(f"{field} NV{nv} closure polish failed")
        ref=ref2
        fitmap["M0_full"]=ref
        saved_th=np.asarray(json.loads(row.theta_json),float)
        saved_pred=v6.v6_model(base,t,saved_th,sites)
        saved=v7.stats(yy,ee,saved_pred,int(row.npar))
        print(f"  NV{nv:3d} {roles[nv]:12s} N={int(row.model_order)} "
              f"saved/refit dchi2={ref['chi2']-saved['chi2']:+.3g}")

        for name,fixed,desc in specs:
            q=fitmap[name]
            rec=dict(field=field,nv_index=int(nv),role=roles[nv],
                     model_order=int(row.model_order),site_key=str(row.site_key),
                     model=name,description=desc,n_fixed=len(fixed),
                     npar=q["npar"],chi2=q["chi2"],red_chi2=q["red_chi2"],
                     bic=q["bic"],aicc=q["aicc"],
                     delta_chi2=q["chi2"]-ref["chi2"],
                     delta_bic=q["bic"]-ref["bic"],
                     saved_v6_chi2=saved["chi2"],
                     full_refit_minus_saved_chi2=ref["chi2"]-saved["chi2"],
                     fL_kHz=fL,Trev_phys_us=trev_phys)
            for i,val in fixed.items():
                rec[f"fixed_{BG_NAMES[i]}"]=val
            # fitted background values for interpretation
            th=q["theta"]
            for i,bn in enumerate(BG_NAMES):
                rec[f"fit_{bn}"]=th[i]
            if int(row.model_order)>0:
                rec["visibility_scale"]=th[9]
            rows.append(rec)
        details[int(nv)]=dict(row=row,sites=sites,y=yy,e=ee,
                              fitmap=fitmap,saved=saved,role=roles[nv])
    tab=pd.DataFrame(rows)
    summary=summarize(tab,field,specs)
    stem=f"v12_v6_background_ablation_{field}"
    tab.to_csv(outdir/f"{stem}_per_nv_model.csv",index=False)
    summary.to_csv(outdir/f"{stem}_summary.csv",index=False)
    make_pdf(outdir/f"{stem}.pdf",field,t,tab,summary,details,specs)
    return dict(field=field,table=tab,summary=summary)


def summarize(tab,field,specs):
    out=[]
    for name,fixed,desc in specs:
        g=tab[tab.model==name]
        out.append(dict(field=field,model=name,description=desc,n_fixed=len(fixed),
            n_nv=len(g),median_delta_chi2=float(g.delta_chi2.median()),
            sum_delta_chi2=float(g.delta_chi2.sum()),
            median_delta_bic=float(g.delta_bic.median()),
            sum_delta_bic=float(g.delta_bic.sum()),
            bic_better_count=int((g.delta_bic<0).sum()),
            bic_strong_better_count=int((g.delta_bic<=-6).sum()),
            bic_strong_worse_count=int((g.delta_bic>=6).sum()),
            median_redchi=float(g.red_chi2.median())))
    return pd.DataFrame(out)

def heatmap(ax,pivot,title,cbar_label):
    a=pivot.to_numpy(float)
    im=ax.imshow(a,aspect="auto",interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)));ax.set_xticklabels(
        [c.replace("M","M",1) for c in pivot.columns],rotation=65,ha="right",fontsize=7)
    ax.set_yticks(range(len(pivot.index)));ax.set_yticklabels(pivot.index.astype(str),fontsize=8)
    ax.set_title(title);ax.set_xlabel("Ablation");ax.set_ylabel("NV")
    cb=ax.figure.colorbar(im,ax=ax,fraction=.025,pad=.02);cb.set_label(cbar_label)


def make_pdf(path,field,t,tab,summary,details,specs):
    names=[x[0] for x in specs]
    ablnames=names[1:]
    with PdfPages(path) as pdf:
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(f"{field}: which V6 background degrees of freedom matter?",fontsize=16)
        s=summary.set_index("model").loc[ablnames]
        x=np.arange(len(s))
        axs[0,0].bar(x,s.median_delta_bic)
        axs[0,0].axhline(0,lw=.8);axs[0,0].axhline(6,ls="--",lw=.8)
        axs[0,0].set_xticks(x);axs[0,0].set_xticklabels(ablnames,rotation=65,ha="right",fontsize=7)
        axs[0,0].set_ylabel("Median delta BIC vs full V6 refit");axs[0,0].grid(alpha=.15,axis="y")
        axs[0,1].bar(x,s.median_delta_chi2)
        axs[0,1].set_xticks(x);axs[0,1].set_xticklabels(ablnames,rotation=65,ha="right",fontsize=7)
        axs[0,1].set_ylabel("Median raw delta chi-square");axs[0,1].grid(alpha=.15,axis="y")
        axs[1,0].bar(x,s.bic_better_count,label="BIC better")
        axs[1,0].bar(x,s.bic_strong_worse_count,bottom=s.bic_better_count,
                     label="strongly worse")
        axs[1,0].set_xticks(x);axs[1,0].set_xticklabels(ablnames,rotation=65,ha="right",fontsize=7)
        axs[1,0].set_ylabel("NV count");axs[1,0].legend(fontsize=8)
        axs[1,0].grid(alpha=.15,axis="y")
        axs[1,1].axis("off")
        lines=["Ablation definitions:"]
        for _,r in summary.iterrows():
            lines.append(f"{r.model}: {r.description}")
        axs[1,1].text(.01,.99,"\n".join(lines),va="top",family="monospace",fontsize=7.4)
        fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);plt.close(fig)

        piv=tab[tab.model!="M0_full"].pivot(index="nv_index",columns="model",values="delta_bic")
        piv=piv.reindex(columns=ablnames)
        fig,ax=plt.subplots(figsize=(11,8.5));heatmap(ax,piv,"Per-NV delta BIC","delta BIC")
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)

        piv=tab[tab.model!="M0_full"].pivot(index="nv_index",columns="model",values="delta_chi2")
        piv=piv.reindex(columns=ablnames)
        fig,ax=plt.subplots(figsize=(11,8.5));heatmap(ax,piv,"Per-NV raw fit loss","delta chi-square")
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)

        # Common residual comparison for a compact set of informative constraints.
        key=["M0_full","M1_TrevPhys","M2_chirp0","M3_slope0","M4_taper0",
             "M7_physTiming","M9_simpleComb","M10_physSimple"]
        fig,ax=plt.subplots(figsize=(11,8.5))
        for name in key:
            R=[]
            for nv,z in details.items():
                q=z["fitmap"][name]
                R.append((z["y"]-q["pred"])/z["e"])
            ax.plot(t,np.nanmedian(np.asarray(R),axis=0),label=name,lw=1.2)
        ax.axhline(0,lw=.7);ax.set(xlabel="Total evolution time (us)",
            ylabel="Median normalized residual",title=f"{field}: common residual after each ablation")
        ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15)
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)

        for nv,z in details.items():
            g=tab[tab.nv_index==nv].set_index("model").loc[names]
            fig,axs=plt.subplots(2,2,figsize=(11,8.5))
            row=z["row"]
            fig.suptitle(f"{field} NV {nv} | {z['role']} | N={int(row.model_order)} {row.site_key}",
                         fontsize=14)
            axs[0,0].errorbar(t,z["y"],yerr=z["e"],fmt="o",ms=3,lw=.4,label="data")
            for name,ls in [("M0_full","-"),("M1_TrevPhys","--"),
                            ("M7_physTiming",":"),("M10_physSimple","-.")]:
                axs[0,0].plot(t,z["fitmap"][name]["pred"],ls=ls,lw=1.2,label=name)
            axs[0,0].set(xlabel="Time (us)",ylabel="Signal")
            axs[0,0].legend(fontsize=6.7);axs[0,0].grid(alpha=.15)

            for name in ["M0_full","M1_TrevPhys","M2_chirp0","M3_slope0",
                         "M4_taper0","M10_physSimple"]:
                q=z["fitmap"][name]
                axs[0,1].plot(t,(z["y"]-q["pred"])/z["e"],lw=.8,label=name)
            axs[0,1].axhline(0,lw=.7);axs[0,1].set(
                xlabel="Time (us)",ylabel="Normalized residual")
            axs[0,1].legend(fontsize=6.2,ncol=2);axs[0,1].grid(alpha=.15)

            xx=np.arange(len(ablnames))
            axs[1,0].bar(xx,g.loc[ablnames].delta_bic)
            axs[1,0].axhline(0,lw=.7);axs[1,0].axhline(6,ls="--",lw=.7)
            axs[1,0].set_xticks(xx);axs[1,0].set_xticklabels(ablnames,rotation=65,ha="right",fontsize=6)
            axs[1,0].set_ylabel("delta BIC vs M0")
            axs[1,0].grid(alpha=.15,axis="y")

            axs[1,1].axis("off")
            bestabl=g.loc[ablnames].sort_values("delta_bic").head(5)
            base=z["fitmap"]["M0_full"]["theta"]
            lines=[f"M0 chi2r={z['fitmap']['M0_full']['red_chi2']:.3f}",
                   f"saved-to-M0 dchi2={g.iloc[0].full_refit_minus_saved_chi2:+.3g}",
                   "M0 background:",
                   f"Trev={base[2]:.3f} us, width0={base[3]:.3f} us",
                   f"T2={1000*base[4]:.2f} us, beta={base[5]:.3f}",
                   f"taper={base[6]:.3f}, slope={base[7]:.3f}, chirp={base[8]:+.5f}",
                   "", "Best constrained variants:"]
            for name,r in bestabl.iterrows():
                lines.append(f"{name}: dchi2={r.delta_chi2:+.1f}, dBIC={r.delta_bic:+.1f}")
            axs[1,1].text(.01,.99,"\n".join(lines),va="top",family="monospace",fontsize=8.2)
            fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--n-each",type=int,default=10)
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    ym=datetime.now().strftime("%Y_%m")
    outdir=(Path(args.output_dir) if args.output_dir else
            Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo")/
            "spin_echo_v12_v6_background_ablation"/ym)
    outdir.mkdir(parents=True,exist_ok=True)
    fields=["49G","52G"] if args.field=="both" else [args.field]
    res=[evaluate(f,outdir,args.n_each) for f in fields]
    if len(res)>1:
        pd.concat([x["table"] for x in res],ignore_index=True).to_csv(
            outdir/"v12_v6_background_ablation_both_per_nv_model.csv",index=False)
        pd.concat([x["summary"] for x in res],ignore_index=True).to_csv(
            outdir/"v12_v6_background_ablation_both_summary.csv",index=False)
    print("\nCOMPLETE")
    for x in res:
        print("\n",x["field"])
        print(x["summary"].to_string(index=False))


if __name__=="__main__":
    main()
