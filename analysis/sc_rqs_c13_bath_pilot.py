# -*- coding: utf-8 -*-
"""Pilot QC for 16-mask fixed-T random-multisine 13C bath acquisitions.

Uses real raw SCC signal / alternating reference counts. Splits runs into
independent halves to test reproducibility of mask-to-mask coherence.
DOES NOT claim a nuclear frequency or fit PSD with too few masks.

Direct run in NVOffice: python analysis/sc_rqs_c13_bath_pilot.py
Optional: --file FULL_SOURCE_TXT --min-reference 2.5 --outdir PATH
Default output: dm to NAS, individual PDF/PNG + combined report + CSV/JSON.
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
from analysis.sc_rqs_c13_bath_analysis import _latest,load_file
from analysis.sc_rqs_c13_analysis import _load_file
from analysis.rqs_c13_bath_models import filter_bins


def run_normalized(arr,ref_min):
    """Compute coherence and approximate SEM, separately for run split."""
    sig=arr[0];ref=arr[1]
    b=ref[:,:,:,::2];d=ref[:,:,:,1::2]
    def stat(z):
        mean=np.mean(z,axis=(1,3))
        n=z.shape[1]*z.shape[3]
        sem=np.sqrt(np.var(z,axis=(1,3),ddof=1)/n)
        return mean,sem
    ms,ss=stat(sig)
    mb,sb=stat(b)
    md,sd=stat(d)
    delta=mb-md
    with np.errstate(invalid="ignore",divide="ignore"):
        coh=(2*ms-mb-md)/delta
        se=np.sqrt((2*ss/delta)**2+
            ((1+coh)*sb/delta)**2+
            ((1-coh)*sd/delta)**2)
    bad=(np.abs(delta)<ref_min) | ~np.isfinite(coh) | ~np.isfinite(se)
    coh[bad]=np.nan
    se[bad]=np.nan
    return coh,se,delta


def plot_pilot(file=None,ref_min=2.5,outdir=None):
    file=str(file or _latest())
    d=load_file(file,reference_min=ref_min)
    _,raw=_load_file(file)
    counts=np.asarray(raw["counts"],float)
    if counts.ndim!=5 or counts.shape[0]!=2:
        raise ValueError("Expected SCC counts[2,nv,runs,step,reps]")
    n_run=counts.shape[2]
    if n_run<4:
        raise ValueError("Need >=4 completed runs for independent halves")
    # Interleaved split avoids monotonic drift bias from first vs second half.
    ca,ea,_=run_normalized(counts[:,:,::2],ref_min)
    cb,eb,_=run_normalized(counts[:,:,1::2],ref_min)
    full,err,delta=run_normalized(counts,ref_min)
    steps=d["steps"]
    ri=[i for i,s in enumerate(steps) if s["kind"]=="random_multisine"]
    controls={s["kind"]:i for i,s in enumerate(steps) if s["kind"]!="random_multisine"}
    R=full[:,ri];E=err[:,ri]
    A=ca[:,ri];B=cb[:,ri]
    names=d["names"]
    nm=len(ri);nv=len(names)
    pcount=np.asarray([steps[i]["n_pi"] for i in ri])
    filterweights=filter_bins([steps[i] for i in ri],
         d["protocol"]["frequency_edges_hz"])/1e-9
    center_hz=np.asarray(d["protocol"]["frequency_centers_hz"])
    k=int(np.argmin(np.abs(center_hz-d["protocol"]["c13_larmor_hz_expected"])))
    sensitivity=filterweights[:,k]
    valid=(np.isfinite(A)&np.isfinite(B))
    rsplit=np.full(nv,np.nan)
    npaired=np.sum(valid,axis=1)
    for j in range(nv):
        okay=valid[j]
        if okay.sum()>=8 and np.std(A[j,okay])>0 and np.std(B[j,okay])>0:
            rsplit[j]=np.corrcoef(A[j,okay],B[j,okay])[0,1]
    goodcorr=np.isfinite(rsplit)
    # Each NV's offset is removed before computing pooled correlation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        wa=A-np.nanmean(A,axis=1,keepdims=True)
        wb=B-np.nanmean(B,axis=1,keepdims=True)
        poolgood=valid&np.isfinite(wa)&np.isfinite(wb)
        pool_corr=float(np.corrcoef(wa[poolgood],wb[poolgood])[0,1])
        median_mask=np.nanmedian(R,axis=0)
        median_A=np.nanmedian(A,axis=0)
        median_B=np.nanmedian(B,axis=0)
        median_control={n:float(np.nanmedian(full[:,i])) for n,i in controls.items()}
    summary={
      "source":d["stem"],"T_us":float(d["T_us"]),
      "expected_field_g":float(d["field_g"]),
      "expected_13c_larmor_khz":float(d["protocol"]["c13_larmor_hz_expected"]/1e3),
      "n_nvs":int(nv),"n_random_masks":int(nm),"num_runs":int(n_run),
      "num_reps":int(counts.shape[-1]),
      "signal_shots_per_setting":int(d["shot_count"]),
      "SCC_reference_cut_photons":float(ref_min),
      "fraction_valid_coherence":float(np.isfinite(full).mean()),
      "fraction_absC_gt1_of_finite":float(np.mean(np.abs(full[np.isfinite(full)])>1)),
      "median_coherence_random":float(np.nanmedian(R)),
      "median_SCC_reference_delta_photons":float(np.nanmedian(abs(delta))),
      "median_coherence_sem":float(np.nanmedian(E)),
      "control_coherence_medians":median_control,
      "n_NV_for_split_correlation":int(goodcorr.sum()),
      "median_NV_split_half_correlation":float(np.nanmedian(rsplit)),
      "NV_centered_pooled_split_half_correlation":pool_corr,
      "n_NV_corr_gt_0p5":int(np.sum(rsplit>.5)),
      "n_NV_corr_lt_minus_0p5":int(np.sum(rsplit<-.5)),
      "all_masks_usable_NV":int(np.sum(np.isfinite(R).all(axis=1))),
      "warning":"This is pilot quality analysis, not a 13C identification. "
                "Coherence sign calibration may be needed; negative C is not "
                "automatically a real quantum-bath oscillation."
    }
    rows=[]
    for j in range(nv):
        row={"nv_index":j,"name":names[j],
             "n_valid_masks":int(np.isfinite(R[j]).sum()),
             "median_coherence":float(np.nanmedian(R[j])),
             "median_sem":float(np.nanmedian(E[j])),
             "split_half_corr":float(rsplit[j]),
             "split_half_pairs":int(npaired[j]),
             "median_ref_delta":float(np.nanmedian(abs(delta[j])))}
        for name,index in controls.items():
            row[name+"_coherence"]=float(full[j,index])
        rows.append(row)
    figures=[]
    def add(fig,name):
        figures.append((fig,name))
    # 1. All-NV x mask response and shot uncertainty.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,9),layout="constrained")
    im=ax.imshow(R,origin="lower",aspect="auto",interpolation="nearest",
                 vmin=-1,vmax=1,cmap="coolwarm")
    fig.colorbar(im,ax=ax,label="Raw normalized X coherence")
    ax.set(xlabel="Random mask index (0-based)",ylabel="NV index",
        title="Measured 60-us random-mask coherence (white = low SCC contrast)")
    ax2.errorbar(np.arange(nm),median_mask,
                 yerr=np.nanmedian(E,axis=0),
                 fmt="o-",capsize=2,label="NV median ± median single-NV SEM")
    ax2.axhline(0,color="gray",lw=1,ls=":")
    ax2.set(xlabel="Random mask index",
        ylabel="Median SCC-normalized coherence",
        title="Error bars indicate typical measurement uncertainty, not NV spread")
    ax2.grid(alpha=.2);ax2.legend()
    add(fig,"01_measured_coherence_and_uncertainty")
    # 2. Same-T controls and reference quality.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,9),layout="constrained")
    keys=["random median","periodic_xy4","periodic_xy8","spin_echo"]
    ensembles=[np.nanmedian(R,axis=1)]+[full[:,controls[k]] for k in keys[1:]]
    ax.boxplot([v[np.isfinite(v)] for v in ensembles])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Random median", "XY4 (~33.3 kHz)",
                        "XY8 (~66.7 kHz)", "Echo"])
    ax.set(ylabel="Normalized X coherence",
           title="Same-duration controls (negative sign may be readout convention)")
    ax.axhline(0,color="grey",lw=.8)
    ax2.hist(np.nanmedian(np.abs(delta),axis=1),bins=25)
    ax2.axvline(ref_min,color="red",ls="--",label="SCC cutoff")
    ax2.set(xlabel="Median bright-dark difference per NV (photons)",
            ylabel="NV count",title="SCC contrast determines precision")
    ax2.legend();ax2.grid(alpha=.2)
    add(fig,"02_controls_and_reference_quality")
    # 3. Split-half mask reproducibility.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,9),layout="constrained")
    ax.scatter(median_A,median_B,c=pcount,cmap="viridis",s=65,
               label="One point per random mask")
    lo=float(np.nanmin(np.r_[median_A,median_B]))-.07
    hi=float(np.nanmax(np.r_[median_A,median_B]))+.07
    ax.plot([lo,hi],[lo,hi],"k--")
    ax.set(xlabel="Median NV coherence — even-numbered runs",
           ylabel="Median NV coherence — odd-numbered runs",
           title=f"Independent run splits: pooled NV-centered r = {pool_corr:.3f}")
    ax.grid(alpha=.2)
    if goodcorr.any():
        ax2.hist(rsplit[goodcorr],bins=20)
    ax2.axvline(0,color="black",ls="--")
    ax2.set(xlabel="Per-NV mask-pattern split-half correlation",
            ylabel="NV count",
            title=f"Median r={np.nanmedian(rsplit):.3f}; {goodcorr.sum()} NVs eligible")
    ax2.grid(alpha=.2)
    add(fig,"03_split_half_reproducibility")
    # 4. Filter sensitivity and pulse-count confounding.
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,9),layout="constrained")
    scatter=ax.scatter(sensitivity,median_mask,c=pcount,cmap="plasma",s=70)
    fig.colorbar(scatter,ax=ax,label="Number of pi pulses")
    ax.set(xlabel="Ideal sensitivity in 33-kHz bin |Y|² / 1e-9 s²",
           ylabel="Median measured NV coherence",
           title="Screening correlation (not proof of nuclear spectroscopy)")
    ax.grid(alpha=.2)
    xs=np.arange(nm)
    ax2.plot(xs,median_A,"o-",label="Run half A")
    ax2.plot(xs,median_B,"s-",label="Run half B")
    ax2.set(xlabel="Random mask index",ylabel="Median NV coherence",
            title="Compare patterns across independent run splits")
    ax2.legend();ax2.grid(alpha=.2)
    add(fig,"04_filter_response_and_repeatability")
    # 5. A small set of quality-ranked example NVs.
    ranked=sorted(np.flatnonzero(goodcorr),key=lambda j:rsplit[j],reverse=True)
    chosen=ranked[:6]
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout="constrained")
    for ax,j in zip(axes.flat,chosen):
        ix=np.arange(nm)
        ax.errorbar(ix,A[j],yerr=ea[j,ri],
                    fmt="o-",ms=3,alpha=.7,label="Half A")
        ax.errorbar(ix,B[j],yerr=eb[j,ri],
                    fmt="s-",ms=3,alpha=.7,label="Half B")
        ax.set(title=f"NV {j}: r={rsplit[j]:.2f}; {npaired[j]} pairs",
               xlabel="Random mask",ylabel="Normalized C")
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Exploratory NVs ranked by split-half correlation; selection bias!",fontsize=13)
    add(fig,"05_example_NV_split_halves")

    from utils import data_manager as dm
    if outdir is None:
        base=dm.get_file_path(__file__,dm.get_time_stamp(),
              "rqs_c13_bath_pilot",subfolder=d["stem"])
        root=Path("G:/nvdata")
        if not root.is_dir():
            base=Path(r"\\192.168.0.197\G\nvdata")/base.relative_to(root)
        def put(binary,target):
            dm.cloud.upload(target,binary)
    else:
        out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
        base=out/"rqs_c13_bath_pilot"
        def put(binary,target):
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(binary)
    together=io.BytesIO()
    with PdfPages(together) as pdf:
        for fig,name in figures:
            outfile=base.with_name(base.stem+"-"+name)
            pdf.savefig(fig,bbox_inches="tight")
            one=io.BytesIO()
            fig.savefig(one,format="pdf",bbox_inches="tight")
            put(one.getvalue(),outfile.with_suffix(".pdf"))
            png=io.BytesIO()
            fig.savefig(png,format="png",dpi=170,bbox_inches="tight")
            put(png.getvalue(),outfile.with_suffix(".png"))
            plt.close(fig)
    put(together.getvalue(),base.with_name(base.stem+"-combined.pdf"))
    put(json.dumps(summary,indent=2).encode(),base.with_suffix(".json"))
    s=io.StringIO(newline="")
    wr=csv.DictWriter(s,fieldnames=list(rows[0]))
    wr.writeheader();wr.writerows(rows)
    put(s.getvalue().encode(),base.with_suffix(".csv"))
    print("PILOT SUMMARY",json.dumps(summary,indent=2))
    print("PILOT OUTPUT",base.parent)
    return summary


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file")
    ap.add_argument("--min-reference",type=float,default=2.5)
    ap.add_argument("--outdir")
    opt=ap.parse_args()
    plot_pilot(opt.file,opt.min_reference,opt.outdir)
