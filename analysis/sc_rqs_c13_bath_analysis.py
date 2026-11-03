# -*- coding: utf-8 -*-
"""Analyze Purcell fixed-T random-multisine NV/13C quantum-bath experiment.

This is NOT the phase-locked linear RQS estimator. SCC dual-rail X coherence
feeds a nonnegative weak-noise PSD-proxy with background and pulse-count
terms; held-out masks test whether frequency filters add predictive power.
Optional single-unpolarized-13C conditional nuclear-spin model is explicitly
illustrative and NOT a definitive hyperfine identification.

Runs without args: latest completed Purcell RQS 13C bath acquisition.
Use --files STEM1 STEM2 for matching T/field datasets with different masks.
All final PNG, PDF, CSV, JSON and NPZ saved with dm and UNC NAS fallback.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import sys
import warnings
from pathlib import Path
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from analysis.sc_rqs_c13_analysis import _load_file
from analysis.rqs_c13_bath_models import (
    filter_bins,toggling_filter,compare_models,heldout_cv,
    quantum_c13_coherence,quantum_fit_candidate)

SOURCE_ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_Purcell\branch_master\rqs_c13_bath")


def _latest():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError("RQS bath NAS directory unavailable: "+str(SOURCE_ROOT))
    candidates=[p for p in SOURCE_ROOT.glob("20??_??/*.txt")
                if p.with_suffix(".npz").is_file()]
    if not candidates:
        raise FileNotFoundError("No RQS bath .txt/.npz data yet; acquire first on Purcell")
    return max(candidates,key=lambda p:p.name)


def _resolve(source):
    p=Path(source)
    if p.is_file() and p.suffix==".txt":return p
    if p.suffix in (".txt",".npz") and len(p.parts)>1:return p.with_suffix(".txt")
    stem=p.stem if p.suffix else p.name
    if len(stem)>=7 and stem[4]=="_":
        candidate=SOURCE_ROOT/stem[:7]/(stem+".txt")
        if candidate.is_file() and candidate.with_suffix(".npz").is_file():
            return candidate
    raise FileNotFoundError("Cannot find paired Purcell RQS 13C bath file: "+str(source))


def load_file(source,reference_min=2.5):
    path=_resolve(source)
    _,raw=_load_file(str(path))
    proto=raw.get("rqs_c13_bath_protocol",{})
    if proto.get("experiment")!="rqs_c13_bath_fixed_T":
        raise ValueError(str(path)+": not the new nuclear-bath RQS experiment")
    if raw.get("rqs_complete") is not True:
        raise ValueError(str(path)+": acquisition marked incomplete")
    steps=proto["steps"]
    counts=np.asarray(raw["counts"],float)
    if counts.ndim!=5 or counts.shape[0]!=2 or counts.shape[3]!=len(steps):
        raise ValueError("Expected counts[signal/ref, NV, runs, steps, repetitions]")
    if not np.isfinite(counts).all():
        raise ValueError("Incomplete/nonfinite SCC counts")
    nv_num,runs,ns,reps=counts.shape[1:]
    if reps<2 or reps%2:
        raise ValueError("Even reps required for alternating SCC reference")
    names=[n["name"] if isinstance(n,dict) else n.name for n in raw["nv_list"]]
    if len(names)!=nv_num or len(set(names))!=len(names):
        raise ValueError("Missing/duplicate NV names")
    sig=counts[0]
    ref=counts[1]
    rb=ref[:,:,:,::2];rd=ref[:,:,:,1::2]
    def mean_sem(v):
        mu=np.mean(v,axis=(1,3))
        N=v.shape[1]*v.shape[3]
        if N>1:
            sem=np.sqrt(np.var(v,axis=(1,3),ddof=1)/N)
        else:
            sem=np.full_like(mu,np.nan)
        return mu,sem
    sm,ss=mean_sem(sig)
    bright,sb=mean_sem(rb)
    dark,sd=mean_sem(rd)
    diff=bright-dark
    with np.errstate(invalid="ignore",divide="ignore"):
        c=(2*sm-bright-dark)/diff
        cs=np.sqrt((2*ss/diff)**2+
                   ((1+c)*sb/diff)**2+
                   ((1-c)*sd/diff)**2)
    valid=np.isfinite(c)&np.isfinite(cs)&(np.abs(diff)>=reference_min)
    c[~valid]=np.nan;cs[~valid]=np.nan
    return {"source":path,"stem":path.stem,"protocol":proto,
            "steps":steps,"names":names,"coherence":c,
            "coherence_sem":cs,"reference_delta":diff,
            "shot_count":int(runs*reps),
            "field_g":float(proto["field_g_expected"]),
            "T_us":float(proto["T_us"]),"reference_min":float(reference_min)}


def pool_files(records):
    """For identical physical filters, inverse-variance average *normalized*
    responses; different mask seeds retain distinct filter designs.
    """
    if len(records)==1:return records[0]
    base=records[0]
    for d in records[1:]:
        a,b=base["protocol"],d["protocol"]
        if abs(base["T_us"]-d["T_us"])>.001:
            raise ValueError("Do not pool runs with different interrogation time")
        if abs(base["field_g"]-d["field_g"])>.3:
            raise ValueError("Do not pool runs with different magnetic field")
        if a["pi_ns"]!=b["pi_ns"] or a["pi2_ns"]!=b["pi2_ns"]:
            raise ValueError("Do not pool different microwave calibrations")
        if a["frequency_edges_hz"]!=b["frequency_edges_hz"]:
            raise ValueError("Do not pool different reconstruction bins")
    names=[n for n in base["names"] if all(n in d["names"] for d in records)]
    if not names:raise ValueError("No matching NV names")
    buckets={}
    for d in records:
        ix=[d["names"].index(name) for name in names]
        for j,s in enumerate(d["steps"]):
            key=(s["kind"],tuple(s["switch_ticks"]),s["phase_mode"],s["T_ns"])
            buckets.setdefault(key,[]).append((s,d["coherence"][ix,j],
                                            d["coherence_sem"][ix,j],
                                            d["reference_delta"][ix,j]))
    steps=[];values=[];errors=[];contrasts=[]
    for key,entries in buckets.items():
        x=np.stack([v for _,v,_,_ in entries],axis=1)
        e=np.stack([v for _,_,v,_ in entries],axis=1)
        contrast=np.stack([v for _,_,_,v in entries],axis=1)
        keep=np.isfinite(x)&np.isfinite(e)&(e>0)
        weights=np.where(keep,1/np.maximum(e,1e-5)**2,0.)
        divisor=np.sum(weights,axis=1)
        avg=np.divide(np.sum(np.where(keep,x*weights,0),axis=1),
                      divisor,out=np.full(len(names),np.nan),where=divisor>0)
        sem=np.divide(1,np.sqrt(divisor),
                      out=np.full(len(names),np.nan),where=divisor>0)
        steps.append(entries[0][0])
        values.append(avg);errors.append(sem)
        contrasts.append(np.nanmean(contrast,axis=1))
    out=dict(base)
    out.update(stem="combined_"+str(len(records))+"_runs",
        names=names,steps=steps,
        coherence=np.stack(values,axis=1),
        coherence_sem=np.stack(errors,axis=1),
        reference_delta=np.stack(contrasts,axis=1),
        shot_count=int(sum(d["shot_count"] for d in records)))
    return out


def _output_base(stem):
    from utils import data_manager as dm
    base=dm.get_file_path(__file__,dm.get_time_stamp(),
            "rqs_c13_bath_analysis",subfolder=stem)
    g=Path("G:/nvdata")
    if not g.is_dir():
        nas=Path(r"\\192.168.0.197\G\nvdata")
        if not nas.is_dir():
            raise FileNotFoundError("Neither mapped G:/nvdata nor NAS UNC root accessible")
        base=nas/base.relative_to(g)
    return base


def analyze(d, *, outdir=None, per_nv=12, quantum_nv=None):
    """Produces diagnostic figures, CSV/NPZ summaries and optional quantum fit."""
    from utils import data_manager as dm
    steps=d["steps"]
    rand=[i for i,s in enumerate(steps) if s["kind"]=="random_multisine"]
    xy8=[i for i,s in enumerate(steps) if s["kind"]=="periodic_xy8"]
    xy4=[i for i,s in enumerate(steps) if s["kind"]=="periodic_xy4"]
    echo=[i for i,s in enumerate(steps) if s["kind"]=="spin_echo"]
    if len(rand)<8 or len(xy8)<1 or len(xy4)<1 or len(echo)<1:
        raise ValueError("Needs random multisine and both same-T controls")
    random_steps=[steps[i] for i in rand]
    C=d["coherence"];S=d["coherence_sem"]
    cr=C[:,rand];sr=S[:,rand]
    F=filter_bins(random_steps,d["protocol"]["frequency_edges_hz"])
    freq_edges=np.array(d["protocol"]["frequency_edges_hz"])
    freq_center=np.asarray(d["protocol"]["frequency_centers_hz"])
    npi=np.asarray([s["n_pi"] for s in random_steps],float)
    nt=len(freq_center); nv=len(d["names"])
    fitvals=np.full((nv,nt),np.nan)
    fit_rms=np.full(nv,np.nan)
    cv_score=np.full(nv,np.nan)
    cv_full=np.full(nv,np.nan)
    cv_null=np.full(nv,np.nan)
    valid_count=np.zeros(nv,int)
    refs=d["reference_delta"]
    rows=[]
    allfits=[]
    for i,name in enumerate(d["names"]):
        fit,null=compare_models(F,npi,cr[i],sr[i])
        cv=heldout_cv(F,npi,cr[i],sr[i])
        allfits.append(fit)
        if fit["status"]=="ok":
            fitvals[i]=fit["coef"][2:]
            fit_rms[i]=fit["rms_chi"]
        if cv["status"]=="ok":
            cv_score[i]=cv["improvement"]
            cv_full[i]=cv["spectral_rmse"]
            cv_null[i]=cv["baseline_rmse"]
        valid_count[i]=sum(np.isfinite(cr[i])&(cr[i]>.04)&(cr[i]<.99))
        rows.append({
            "nv_index":i,"nv_name":name,"fit_status":fit["status"],
            "cv_status":cv["status"],"n_valid_masks":int(valid_count[i]),
            "fit_rms_chi":float(fit_rms[i]),
            "cv_spectral_rmse":float(cv_full[i]),
            "cv_pulsecount_null_rmse":float(cv_null[i]),
            "cv_improvement":float(cv_score[i]),
            "reference_median_photons":float(np.nanmedian(abs(refs[i]))),
            "xy8_C":float(C[i,xy8[0]]),
            "xy4_C":float(C[i,xy4[0]]),
            "echo_C":float(C[i,echo[0]]),
        })
    passed=np.isfinite(fit_rms)
    verified=np.isfinite(cv_score)
    improved=verified&(cv_score>0)
    summary={
       "source":str(d["stem"]),
       "num_NV":nv,"T_us":d["T_us"],"field_g_expected":d["field_g"],
       "13C_larmor_khz_expected":d["protocol"]["c13_larmor_hz_expected"]/1000,
       "num_random_masks":len(rand),"num_XY8_controls":len(xy8),
       "num_XY4_controls":len(xy4),
       "num_echo_controls":len(echo),
       "signal_shots_per_setting":d["shot_count"],
       "frequency_centers_khz":(freq_center/1000).tolist(),
       "frequency_edges_khz":(freq_edges/1000).tolist(),
       "fit_count":int(passed.sum()),
       "CV_testable_NV":int(verified.sum()),
       "CV_spectral_better_NV":int(improved.sum()),
       "median_CV_improvement":float(np.nanmedian(cv_score)) if verified.any() else None,
       "median_RMS_chi":float(np.nanmedian(fit_rms)) if passed.any() else None,
       "fraction_valid_C":float(np.isfinite(C).mean()),
       "fraction_unphysical_absC_gt1_among_valid":float(
           np.mean(abs(C[np.isfinite(C)])>1)) if np.isfinite(C).any() else None,
       "warning":"A spectral proxy is NOT a 13C identification; model assumes "
         "approximately weak Gaussian dephasing plus pulse-count-dependent "
         "errors. Conditional electron-nuclear dynamics may invalidate PSD."
    }
    if quantum_nv is not None:
        if not 0<=quantum_nv<nv:
            raise ValueError("quantum_nv must index a measured NV")
        qfit=quantum_fit_candidate(random_steps,cr[quantum_nv],sr[quantum_nv],
                   field_g=d["field_g"])
        summary["quantum_candidate_nv_index"]=quantum_nv
        summary["quantum_candidate_result"]=qfit
    figures={}
    # 1. Pulse timing filter sensitivity, which is NOT a measured spectrum.
    scan=np.linspace(freq_edges[0],freq_edges[-1],500)
    pulses=np.stack([abs(toggling_filter(s["switch_ticks"],4,s["T_ns"],scan))**2
                     for s in random_steps])
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,8),layout="constrained")
    arr=pulses/np.maximum(pulses.max(axis=1,keepdims=True),1e-20)
    m=ax.imshow(arr,aspect="auto",origin="lower",vmin=0,vmax=1,cmap="viridis",
                extent=[scan[0]/1e3,scan[-1]/1e3,-.5,len(rand)-.5])
    fig.colorbar(m,ax=ax,label="Filter power, per-mask normalized")
    ax.axvline(summary["13C_larmor_khz_expected"],color="white",
               linestyle="--",label="bare 13C Larmor")
    ax.set(xlabel="Frequency (kHz)",ylabel="Random mask index",
           title="Designed ideal filters |Y(f)|², not measured 13C spectrum")
    ax.legend()
    ax2.plot(scan/1e3,np.mean(pulses,axis=0))
    ax2.axvline(summary["13C_larmor_khz_expected"],color="gray",ls="--")
    ax2.set(xlabel="Frequency (kHz)",ylabel="Ensemble mean |Y|² (s²)",
            title="Average filter coverage")
    ax2.grid(alpha=.2)
    figures["01_filter_bank"]=fig
    # 2. Matrix of measured coherence in physically meaningful pulse order.
    order=np.lexsort((np.arange(len(rand)),npi))
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(12,8),layout="constrained")
    im=ax.imshow(cr[:,order],origin="lower",aspect="auto",
                 cmap="coolwarm",interpolation="nearest",vmin=-1,vmax=1)
    fig.colorbar(im,ax=ax,label="Measured normalized X coherence")
    ax.set(xlabel="Random mask sorted by pulse count",ylabel="NV index",
           title="Measured 13C bath response; blank = weak SCC reference")
    top=cr[:,order]
    x=np.asarray(npi[order])
    ax2.scatter(x,np.nanmedian(top,axis=0),alpha=.6)
    ax2.set(xlabel="Number of refocusing pi pulses / mask",
            ylabel="Median coherence across NVs",
            title="Pulse-count error is a competing explanation")
    ax2.grid(alpha=.2)
    figures["02_measured_coherence"]=fig
    # 3. Does coherence correlate with 33 kHz filter sensitivity?
    idx=int(np.argmin(abs(freq_center-d["protocol"]["c13_larmor_hz_expected"])))
    x=F[:,idx]/1e-9
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,8),layout="constrained")
    med=np.nanmedian(cr,axis=0)
    ax.scatter(x,med,c=npi,cmap="plasma")
    ax.set(xlabel="33 kHz-neighbor bin mean |Y|² / 1e-9 s²",
           ylabel="Median measured X coherence",
           title="Does a more 13C-sensitive mask produce lower coherence?")
    ax.grid(alpha=.2)
    # Pulse-count-only residual correlation reduces false attribution to bath.
    corrected=[]
    for i in range(nv):
        _,nul=compare_models(F,npi,cr[i],sr[i])
        if nul["status"]=="ok":
            with np.errstate(invalid="ignore",divide="ignore"):
                v=-np.log(cr[i])-nul["predicted_chi"]
            corrected.append(v)
    if corrected:
        residue=np.nanmedian(np.asarray(corrected),axis=0)
        ax2.scatter(x,residue,c=npi,cmap="plasma")
    ax2.set(xlabel="33 kHz-neighbor filter weight / 1e-9 s²",
            ylabel="Median residual decoherence -ln(C)",
            title="Filter dependence AFTER fitting pulse-count-only background")
    ax2.grid(alpha=.2)
    figures["03_filter_weight_correlation"]=fig

    # 4. Nonnegative spectral coefficient HEATMAP (do not label as hyperfine).
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,8),layout="constrained")
    plot=np.log10(1+np.maximum(fitvals,0))
    if np.isfinite(plot).any():
        im=ax.imshow(plot,origin="lower",aspect="auto",cmap="viridis",
                   interpolation="nearest",
                   extent=[freq_edges[0]/1000,freq_edges[-1]/1000,-.5,nv-.5])
        fig.colorbar(im,ax=ax,label="log10(1 + spectral-weight proxy)")
    else:
        ax.text(.5,.5,"No sufficiently constrained fits",
                transform=ax.transAxes,ha="center")
    ax.axvline(summary["13C_larmor_khz_expected"],color="white",ls="--")
    ax.set(xlabel="Frequency bin (kHz)",ylabel="NV index",
           title="Exploratory weak-Gaussian-noise spectral power (NOT hyperfine)")
    if passed.any():
        ax2.hist(fit_rms[passed],bins=24)
    ax2.set(xlabel="In-sample residual RMS in -ln(C)",ylabel="Number of NVs",
            title=f"Fit quality: {int(passed.sum())}/{nv} usable")
    ax2.grid(alpha=.2)
    figures["04_spectral_proxy_and_residuals"]=fig
    # 5. Essential falsification: held-out masks vs pulse-count-only model.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,8),layout="constrained")
    if verified.any():
        color=ax.scatter(cv_null[verified],cv_full[verified],
               c=valid_count[verified],cmap="viridis",s=30)
        fig.colorbar(color,ax=ax,label="Usable masks")
        cap=1.04*float(np.nanmax(np.r_[cv_null[verified],cv_full[verified]]))
        ax.plot([0,cap],[0,cap],"k--",label="Equal error")
        ax.legend()
        ax2.scatter(valid_count[verified],cv_score[verified],s=25)
    ax.axvline(0,color="gray",ls=":",lw=.8)
    ax.set(xlabel="Pulse-count baseline: held-out RMSE in -ln(C)",
           ylabel="Spectral + baseline: held-out RMSE",
           title=f"Held-out random masks; improvement for {int(improved.sum())}/{int(verified.sum())} testable NVs")
    ax.grid(alpha=.2)
    ax2.axhline(0,color="k",ls="--",lw=.9)
    ax2.set(xlabel="Valid random filters per NV",
            ylabel="1 - spectral_RMSE / baseline_RMSE",
            title="Positive values indicate reproducible spectral information")
    ax2.grid(alpha=.2)
    figures["05_heldout_validation"]=fig
    # 6. Control distributions; XY8 and echo are at same T, not extra frequency sweeps.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,8),layout="constrained")
    observations=[np.nanmedian(cr,axis=1),C[:,xy4[0]],
                  C[:,xy8[0]],C[:,echo[0]]]
    ax.boxplot([v[np.isfinite(v)] for v in observations])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Random masks median", "XY4 (~2/T)",
                        "XY8 (~4/T)", "Hahn echo"])
    ax.set(ylabel="Measured SCC-normalized coherence",
           title=f"Same-T controls at {d['T_us']:.0f} us")
    ax.grid(alpha=.2)
    median_contrast=np.nanmedian(abs(refs),axis=1)
    ax2.hist(median_contrast[np.isfinite(median_contrast)],bins=25)
    ax2.axvline(d["reference_min"],color="red",ls="--",label="SCC threshold")
    ax2.set(xlabel="Median bright-minus-dark SCC photons",
            ylabel="Number of NVs",title="Dual-rail reference quality")
    ax2.legend();ax2.grid(alpha=.2)
    figures["06_controls_and_SCC"]=fig

    if quantum_nv is not None and summary["quantum_candidate_result"]["status"]=="ok":
        q=summary["quantum_candidate_result"]
        sim=quantum_c13_coherence(random_steps,field_g=d["field_g"],
                 a_parallel_hz=q["a_parallel_khz"]*1000,
                 a_perp_hz=q["a_perp_khz"]*1000)
        prediction=q["contrast"]*np.exp(-q["pulse_loss"]*npi)*sim
        fig,ax=plt.subplots(figsize=(9,6),layout="constrained")
        ax.errorbar(prediction,cr[quantum_nv],yerr=sr[quantum_nv],
                    fmt="o",markersize=4,alpha=.75)
        ax.plot([-1,1],[-1,1],"k--")
        ax.set(xlabel="Single-13C conditional-spin prediction",
               ylabel="Measured coherence",title=f"Illustrative NV {quantum_nv} fit: "
                    "NOT a confirmed hyperfine assignment")
        ax.grid(alpha=.2)
        figures["07_quantum_candidate"]=fig

    if outdir is None:
        base=_output_base(d["stem"])
        out=base.parent
        def save_pdf(fig,name):
            target=base.with_name(base.stem+"-"+name)
            dm.save_figure(fig,target)
            buf=io.BytesIO()
            fig.savefig(buf,format="pdf",bbox_inches="tight")
            dm.cloud.upload(target.with_suffix(".pdf"),buf.getvalue())
    else:
        out=Path(outdir)
        out.mkdir(parents=True,exist_ok=True)
        base=out/"rqs_bath"
        def save_pdf(fig,name):
            fig.savefig(out/(name+".png"),dpi=180,bbox_inches="tight")
            fig.savefig(out/(name+".pdf"),bbox_inches="tight")
    for label,fig in figures.items():
        save_pdf(fig,label)
        plt.close(fig)

    # Detailed pages for interesting NVs. Score ranking is only used to
    # guide QC; no claimed evidence of nuclear-spin peaks from ranking.
    finite=np.flatnonzero(verified)
    ranked=sorted(finite,key=lambda j:cv_score[j],reverse=True)
    other=[j for j in range(nv) if j not in ranked]
    chosen=(ranked+other)[:per_nv] if per_nv>0 else list(range(nv))
    pages=io.BytesIO()
    with PdfPages(pages) as pdf:
        for j in chosen:
            fig,axs=plt.subplots(2,2,figsize=(11,8),layout="constrained")
            sns=cr[j];ses=sr[j]
            good=np.isfinite(sns)&np.isfinite(ses)
            axs[0,0].errorbar(np.arange(len(rand))[good],sns[good],
                             yerr=ses[good],fmt=".",alpha=.65)
            axs[0,0].set(xlabel="Random-mask index",ylabel="C ± SEM",
                         title="Measured coherence (not fitted)")
            axs[0,0].grid(alpha=.2)
            if allfits[j]["status"]=="ok":
                f=allfits[j]
                model_C=np.exp(-f["predicted_chi"])
                axs[0,1].scatter(model_C[good],sns[good],c=npi[good],
                                 cmap="plasma",s=25)
                axs[0,1].plot([0,1],[0,1],"k--")
            axs[0,1].set(xlabel="In-sample predicted C",ylabel="Observed C",
                         title="Check whether spectral fit captures each mask")
            axs[0,1].grid(alpha=.2)
            axs[1,0].scatter(npi,sns,c=F[:,idx],cmap="viridis")
            axs[1,0].set(xlabel="Refocusing pi pulses per mask",
                         ylabel="Coherence",title="Pulse-error confound")
            axs[1,0].grid(alpha=.2)
            if np.isfinite(fitvals[j]).all():
                axs[1,1].bar(freq_center/1000,fitvals[j],
                            width=np.diff(freq_edges).min()/1000*.6)
            axs[1,1].axvline(summary["13C_larmor_khz_expected"],c="k",ls="--")
            axs[1,1].set(xlabel="Frequency bin center (kHz)",
                         ylabel="Nonnegative spectral proxy",
                         title="Weak-Gaussian model only")
            axs[1,1].grid(alpha=.2)
            fig.suptitle(f"NV {j}: {d['names'][j]} | "
                         f"{valid_count[j]}/{len(rand)} good masks | "
                         f"held-out improvement={cv_score[j]:.3f}",fontsize=10)
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)
    if outdir is None:
        dm.cloud.upload(base.with_name(base.stem+"-07_NV_diagnostics.pdf"),
                        pages.getvalue())
    else:
        (out/"07_NV_diagnostics.pdf").write_bytes(pages.getvalue())
    buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0].keys()))
    writer.writeheader();writer.writerows(rows)
    if outdir is None:
        dm.save_text(json.dumps(summary,indent=2),base,suffix=".json")
        dm.save_text(buffer.getvalue(),base,suffix=".csv")
        dm.save_npz(base,compressed=True,coherence=C,coherence_sem=S,
                    spectral_proxy=fitvals,frequencies_hz=freq_center,
                    bin_edges_hz=freq_edges,cv_improvement=cv_score,
                    cv_spectral_rmse=cv_full,cv_null_rmse=cv_null,
                    fit_rms_chi=fit_rms,reference_delta=refs)
    else:
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
        (out/"per_nv.csv").write_text(buffer.getvalue())
        np.savez_compressed(out/"arrays.npz",coherence=C,coherence_sem=S,
             spectral_proxy=fitvals,frequencies_hz=freq_center,
             bin_edges_hz=freq_edges,cv_improvement=cv_score,
             cv_spectral_rmse=cv_full,cv_null_rmse=cv_null,
             fit_rms_chi=fit_rms,reference_delta=refs)
    print("BATH_RQS_SUMMARY",json.dumps(summary,indent=2))
    print("SAVED",out)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files",nargs="+",help="Bath raw file stems or .txt paths")
    ap.add_argument("--outdir",help="Override NAS dm output")
    ap.add_argument("--reference-min",type=float,default=0.0)
    ap.add_argument("--per-nv",type=int,default=12,
                    help="Detailed NV pages; use 0 for all NVs")
    ap.add_argument("--quantum-nv",type=int,
                    help="Fit illustrative conditional single 13C to this NV index")
    ap.add_argument("--one-figure-only",action="store_true",
                    help="Create easy-to-read real-data mean/median overview only")
    args=ap.parse_args()
    files=args.files or [str(_latest())]
    print("Selected bath acquisition:",files)
    if args.one_figure_only:
        if len(files)!=1:
            ap.error("--one-figure-only requires exactly one input dataset")
        from analysis.sc_rqs_c13_bath_one_figure import create_figure
        create_figure(files[0],min_reference=max(2.5,args.reference_min),
                      outdir=args.outdir)
        return
    d=pool_files([load_file(f,args.reference_min) for f in files])
    analyze(d,outdir=args.outdir,per_nv=args.per_nv,quantum_nv=args.quantum_nv)
    if len(files)==1:
        from analysis.sc_rqs_c13_bath_one_figure import create_figure
        create_figure(files[0],min_reference=max(2.5,args.reference_min),
                      outdir=args.outdir)
    else:
        print("Mean/median one-figure overview is available separately for "
              "each input with --one-figure-only --files STEM")


if __name__=="__main__":
    main()
