# -*- coding: utf-8 -*-
"""Scientific diagnostic plots for random-pi_X / XY8 SCC 13C spectroscopy.

Direct run: python analysis/sc_rqs_c13_random_x_diagnostics.py
Selects latest Purcell random-X acquisition. Does not edit raw data.
Outputs diagnostic PNG and vector PDF figures through dm to the NAS,
with a UNC fallback when NVOffice does not have G: mapped.

These figures test model adequacy, not confirm 13C identity.
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import warnings
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import nnls
from analysis.sc_rqs_c13_random_x_estimator import (
    _latest_source,_load,_pool_different_masks,filter_design,noise_fit
)
from analysis.sc_rqs_c13_analysis import _filter_power
warnings.filterwarnings("ignore",category=RuntimeWarning)


def pdf_save(fig,base,name):
    from utils import data_manager as dm
    target=base.parent/(base.stem+"-"+name)
    dm.save_figure(fig,target)
    buffer=io.BytesIO()
    fig.savefig(buffer,format="pdf",bbox_inches="tight")
    dm.cloud.upload(target.with_suffix(".pdf"),buffer.getvalue())
    print("Saved",target.with_suffix(".pdf"))


def output_base(stem):
    from utils import data_manager as dm
    base=dm.get_file_path(__file__,dm.get_time_stamp(),
                          "random_pi_x_diagnostics",subfolder=stem)
    g=Path("G:/nvdata")
    if not g.is_dir():
        base=Path(r"\\192.168.0.197\G\nvdata")/base.relative_to(g)
    return base


def fit_with_cv(c,F,D,group):
    """Out-of-sample 4-fold split; compare 8-bin PSD vs duration-only null.

    Each fold holds out 1/4 of the random masks from EACH duration.
    If insufficient valid points are left, return NaNs (do not extrapolate).
    """
    valid=np.isfinite(c)&(c>.03)&(c<.999)
    if valid.sum()<40:
        return np.nan,np.nan,0
    chi=-np.log(np.clip(c,0.03,.999))
    predictions=[]; observed=[]; baseline=[]
    A=np.column_stack([F,D])
    nparam=A.shape[1]
    for fold in range(4):
        test=valid & ((np.arange(len(c))%16)%4==fold)
        train=valid & ~test
        if train.sum()<max(30,2*nparam) or test.sum()<2:
            continue
        M=A[train]; y=chi[train]
        scale=np.linalg.norm(M,axis=0)
        if np.any(scale<1e-10):
            continue
        norm=M/scale
        if np.linalg.matrix_rank(norm)<nparam:
            continue
        penalty=np.zeros((F.shape[1]-1,nparam))
        for k in range(F.shape[1]-1):
            penalty[k,k]=.025;penalty[k,k+1]=-.025
        coef,_=nnls(np.vstack([norm,penalty]),
                    np.r_[y,np.zeros(len(penalty))],maxiter=250*nparam)
        pred=A[test]@(coef/scale)
        duration_baseline=np.array([
            np.nanmean(y[group[train]==t])
            if np.any(group[train]==t) else np.nan
            for t in group[test]
        ])
        predictions.extend(pred.tolist())
        observed.extend(chi[test].tolist())
        baseline.extend(duration_baseline.tolist())
    if len(observed)<8:
        return np.nan,np.nan,len(observed)
    y=np.asarray(observed)
    pred=np.asarray(predictions)
    null=np.asarray(baseline)
    keep=np.isfinite(y)&np.isfinite(pred)&np.isfinite(null)
    if keep.sum()<8:return np.nan,np.nan,int(keep.sum())
    rmse=np.sqrt(np.mean((y[keep]-pred[keep])**2))
    null_rmse=np.sqrt(np.mean((y[keep]-null[keep])**2))
    return float(rmse),float(null_rmse),int(keep.sum())


def get_data(sources,reference_cut=2.5):
    from analysis.sc_rqs_c13_random_x_estimator import _resolve_source
    data=[_load(_resolve_source(x),min_ref=reference_cut) for x in sources]
    if len(data)==1:return data[0]
    return _pool_different_masks(data)


def generate(data,all_nvs=False, max_nv_pdfs=12):
    from utils import data_manager as dm
    p=data["proto"];C=data["C"];steps=data["steps"];nv_names=data["names"]
    ir=np.array([i for i,s in enumerate(steps)
                 if s["kind"]=="random_x" and s["final_pi2_phase_deg"]==0])
    ix=np.array([i for i,s in enumerate(steps)
                 if s["kind"]=="periodic_xy8"])
    if len(ir)<4 or len(ix)<4:raise ValueError("Missing random-X / XY8 steps")
    fxy=np.array([steps[i]["target_f_hz"] for i in ix])
    Txy=np.array([steps[i]["T_ns"]/1e3 for i in ix])
    isort=np.argsort(fxy)
    Tr=np.array([steps[i]["T_ns"]/1e3 for i in ir])
    times=np.array([int(steps[i]["T_ns"]) for i in ir])
    dur=np.unique(times)
    D=(times[:,None]==dur[None,:]).astype(float)
    edges=np.linspace(fxy.min(),fxy.max(),9)
    center=(edges[:-1]+edges[1:])/2/1e3
    F=filter_design([steps[i] for i in ir],edges)
    n=len(C);nbin=8
    coefficients=np.full((n,nbin),np.nan)
    rms=np.full(n,np.nan);usable=np.zeros(n,int)
    cv=np.full(n,np.nan);cvnull=np.full(n,np.nan);cvn=np.zeros(n,int)
    all_fits=[]
    for j in range(n):
        fit=noise_fit(C[j,ir],[steps[k] for k in ir],edges)
        all_fits.append(fit)
        usable[j]=fit.get("usable",0)
        if fit["status"]=="ok":
            coefficients[j]=fit["spectrum"]
            rms[j]=fit["rms"]
        cv[j],cvnull[j],cvn[j]=fit_with_cv(C[j,ir],F,D,times)
    improvement=1-cv/cvnull
    base=output_base(data["stem"])
    def save(fig,name):
        pdf_save(fig,base,name)
        plt.close(fig)
    # 1. XY8 spectral trace vs sensing time and NV distribution.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,8),layout="constrained")
    sel=C[:,ix][:,isort]; freq=fxy[isort]/1e3
    median=np.nanmedian(sel,axis=0)
    lo=np.nanpercentile(sel,25,axis=0);hi=np.nanpercentile(sel,75,axis=0)
    ax.fill_between(freq,lo,hi,alpha=.2,label="NV interquartile range")
    ax.plot(freq,median,"o-",label="Median 119-NV coherence")
    ax.axhline(1,color="grey",lw=.9,ls=":")
    ax.set(xlabel="XY8 nominal frequency (kHz)",ylabel="Normalized coherence",
           title="XY8 control: frequency changes together with sensing duration")
    ax.legend();ax.grid(alpha=.2)
    ax2.plot(freq,Txy[isort],"o-",color="tab:orange")
    ax2.set(xlabel="XY8 nominal frequency (kHz)",
            ylabel="Interrogation time (microseconds)",
            title="High frequencies have shorter sequences: this is a confound")
    ax2.grid(alpha=.2)
    save(fig,"01_xy8_response_and_duration")

    # 2. Per-NV filter response, repeated masks grouped by T.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(12,9),layout="constrained")
    order=np.lexsort((np.arange(len(ir)),Tr))
    cm=C[:,ir][:,order]
    image=ax.imshow(cm,origin="lower",aspect="auto",interpolation="nearest",
                    vmin=-1,vmax=1,cmap="coolwarm")
    fig.colorbar(image,ax=ax,label="SCC-normalized C (display range -1..1)")
    sorted_t=Tr[order];last=None
    for j,t in enumerate(sorted_t):
        if last is not None and t!=last:ax.axvline(j-.5,color="black",lw=1)
        last=t
    ax.set(xlabel="Random mask (sorted by sensing time)",ylabel="NV index",
           title="Each column is a DIFFERENT pulse-timing mask; blank = poor SCC")
    mask_means=np.nanmedian(C[:,ir],axis=0)
    for t in sorted(np.unique(Tr)):
        m=Tr==t
        ax2.scatter(np.full(m.sum(),t),mask_means[m],alpha=.7)
    ax2.plot(sorted(np.unique(Tr)),
             [np.nanmedian(mask_means[Tr==t]) for t in sorted(np.unique(Tr))],
             "k--",lw=1.5,label="Median over random masks")
    ax2.set(xlabel="Random sensing duration (microseconds)",
            ylabel="Median NV coherence per mask",
            title="Distinguish duration decay from mask-to-mask fluctuations")
    ax2.legend();ax2.grid(alpha=.2)
    save(fig,"02_random_mask_coherence")

    # 3. Filter-function coverage / conditioning
    freqs=np.linspace(fxy.min(),fxy.max(),450)
    powers=np.stack([_filter_power(steps[i],freqs) for i in ir])
    rownorm=powers/np.maximum(np.max(powers,axis=1,keepdims=True),1e-30)
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(12,9),layout="constrained")
    image=ax.imshow(rownorm,origin="lower",aspect="auto",cmap="viridis",
              extent=[freqs[0]/1e3,freqs[-1]/1e3,-.5,len(ir)-.5])
    fig.colorbar(image,ax=ax,label="Ideal |Y(f)|², normalized per mask")
    ax.set(xlabel="Frequency (kHz)",ylabel="Random mask index",
           title="Which frequencies does EACH randomized mask interrogate?")
    ax2.plot(freqs/1e3,np.mean(powers,axis=0),label="Mean filter power")
    ax2.set(xlabel="Frequency (kHz)",ylabel="Mean ideal |Y|² (s²)",
            title="Ensemble filter sensitivity (not an experimental spectrum)")
    ax2.grid(alpha=.2);ax2.legend()
    save(fig,"03_random_filter_coverage")

    # 4. Spectral coefficients + model quality.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(12,9),layout="constrained")
    display=np.log10(1+np.maximum(coefficients,0))
    if np.isfinite(display).any():
        im=ax.imshow(display,origin="lower",aspect="auto",interpolation="nearest",
                   extent=[edges[0]/1e3,edges[-1]/1e3,-.5,n-.5],cmap="viridis")
        fig.colorbar(im,ax=ax,label="log10(1 + fitted nonnegative PSD proxy)")
    ax.set(xlabel="Spectral bin (kHz)",ylabel="NV index",
           title="Regularized fit coefficients; peaks are NOT yet confirmed 13C")
    good=np.isfinite(rms)
    ax2.hist(rms[good],bins=22,label="Fit RMS in -ln(C)")
    ax2.set(xlabel="In-sample residual RMS (-ln C)",ylabel="NV count",
            title=f"{good.sum()}/{n} fit with existing 8-bin model")
    ax2.grid(alpha=.2);ax2.legend()
    save(fig,"04_spectral_bins_and_in_sample_fit")

    # 5. Predictive model versus duration-only null and SCC QC.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,9),layout="constrained")
    pred=np.isfinite(cv)&np.isfinite(cvnull)&(cvnull>1e-9)
    ax.scatter(cvnull[pred],cv[pred],c=usable[pred],cmap="viridis",s=35)
    if pred.any():
        ma=max(np.nanmax(cv[pred]),np.nanmax(cvnull[pred]))*1.05
        ax.plot([0,ma],[0,ma],"k--",label="Equal held-out error")
    ax.set(xlabel="Duration-only held-out RMSE (-ln C)",
           ylabel="8-bin model held-out RMSE (-ln C)",
           title=f"Held-out masks: {pred.sum()} NVs testable (lower is better)")
    ax.grid(alpha=.2);ax.legend()
    ax2.scatter(usable,1-cv/cvnull,s=27,alpha=.8)
    ax2.axhline(0,color="black",ls="--",lw=1)
    ax2.set(xlabel="Usable random-mask observations / 64",
            ylabel="1 - (spectral RMSE / duration-only RMSE)",
            title="Positive score means better prediction than ordinary T-dependent decay")
    ax2.grid(alpha=.2)
    save(fig,"05_heldout_model_comparison")

    # Per-NV diagnostics: show measured vs model predicted, plus XY8.
    valid=np.flatnonzero(np.isfinite(rms))
    priority=sorted(valid,key=lambda j:(
        0 if np.isfinite(improvement[j]) and improvement[j]>0 else 1,
        -improvement[j] if np.isfinite(improvement[j]) else 9e9,
        rms[j]))
    indices=list(range(n)) if all_nvs else priority[:max_nv_pdfs]
    book=io.BytesIO()
    with PdfPages(book) as pages:
        for j in indices:
            fig,axes=plt.subplots(2,2,figsize=(11,8.5),layout="constrained")
            fsort=np.argsort(fxy)
            axes[0,0].plot(fxy[fsort]/1e3,C[j,ix][fsort],"o-")
            axes[0,0].set(xlabel="XY8 frequency (kHz)",ylabel="Measured coherence",
                          title="Periodic XY8 reference")
            axes[0,0].grid(alpha=.2)
            m=F@coefficients[j]+D@all_fits[j]["baseline"] if np.isfinite(rms[j]) else np.full(len(ir),np.nan)
            obs=C[j,ir];predc=np.exp(-m)
            for t in sorted(np.unique(Tr)):
                a=(Tr==t)&np.isfinite(obs)
                axes[0,1].scatter(predc[a],obs[a],label=f"T={t:.0f} us",s=24)
            axes[0,1].plot([0,1],[0,1],"k--",lw=1)
            axes[0,1].set(xlabel="Model-predicted C (in sample)",
                          ylabel="Measured C",
                          title="Does the inferred spectrum reproduce coherence?")
            axes[0,1].legend(fontsize=7);axes[0,1].grid(alpha=.2)
            for t in sorted(np.unique(Tr)):
                a=(Tr==t)
                axes[1,0].scatter(np.flatnonzero(a),obs[a],label=f"T={t:.0f} us",s=22)
            axes[1,0].set(xlabel="Random mask index / 64",
                          ylabel="Measured coherence",title="Random-mask raw response")
            axes[1,0].grid(alpha=.2)
            axes[1,1].bar(center,coefficients[j],
                          width=(edges[1]-edges[0])/1000*.85)
            axes[1,1].set(xlabel="Spectral frequency bin center (kHz)",
                          ylabel="Nonnegative coefficient (a.u.)",
                          title="Fit-derived spectral proxy, NOT nuclear coupling")
            axes[1,1].grid(alpha=.2)
            qualifier=(f"held-out improvement={improvement[j]:.2f}"
                       if np.isfinite(improvement[j]) else "no held-out validation")
            fig.suptitle(f"NV {j}: {nv_names[j]} | {usable[j]}/64 masks | "
                         f"chi RMS={rms[j]:.2f} | {qualifier}",fontsize=11)
            pages.savefig(fig,bbox_inches="tight")
            if not all_nvs:
                pdf_save(fig,base,f"NV_{j:03d}_details")
            plt.close(fig)
    dm.cloud.upload(base.parent/(base.stem+"-06_per_nv_diagnostics.pdf"),
                    book.getvalue())
    summary={
        "source":data["stem"],"num_nv":n,"number_random_masks":len(ir),
        "number_xy8":len(ix),"num_spectral_fits":int(np.isfinite(rms).sum()),
        "num_nv_with_cross_validation":int(np.isfinite(improvement).sum()),
        "num_nv_with_cv_improvement":int(np.sum(improvement>0)),
        "median_fit_chi_RMSE":float(np.nanmedian(rms)),
        "median_cv_improvement":float(np.nanmedian(improvement))
            if np.isfinite(improvement).any() else None,
        "reference_cut":2.5,"notes":"Model fits are under Gaussian dephasing assumption; cannot assign 13C without controls."
    }
    dm.save_text(json.dumps(summary,indent=2),base,suffix=".json")
    dm.save_npz(base,compressed=True,spectrum=coefficients,
                coherence=C,fit_rmse=rms,cv_rmse=cv,cv_null_rmse=cvnull,
                cv_improvement=improvement,usable_masks=usable,
                freq_edges_hz=edges)
    print("DIAGNOSTICS SUMMARY",json.dumps(summary,indent=2))
    print("OUTPUT",base.parent)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files",nargs="+")
    ap.add_argument("--all-nvs",action="store_true",
                    help="Create all NV pages; default selected best-quality 12")
    ap.add_argument("--per-nv",type=int,default=12)
    opt=ap.parse_args()
    data=get_data(opt.files or [_latest_source()])
    generate(data,all_nvs=opt.all_nvs,max_nv_pdfs=opt.per_nv)


if __name__=="__main__":
    main()
