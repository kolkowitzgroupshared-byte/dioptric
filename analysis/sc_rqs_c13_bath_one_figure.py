# -*- coding: utf-8 -*-
"""One-figure, simulation-style plot of REAL Purcell 13C-bath RQS data.

Run without args to use the latest completed experiment:
    python analysis/sc_rqs_c13_bath_one_figure.py

Also supports --file STEM_OR_TXT, --outdir DIR, --min-reference 2.5,
--min-valid-fraction 0.75.

This is a non-parametric summary of raw SCC-normalized X responses.
DO NOT interpret negative X response as a physical negative bath
coherence until the microwave tomography/readout polarity is calibrated.
It does NOT claim a 13C resonance or fit an underpowered noise spectrum.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analysis.sc_rqs_c13_analysis import _load_file
from analysis.sc_rqs_c13_bath_analysis import _latest, load_file, _resolve
from analysis.sc_rqs_c13_bath_pilot import run_normalized
from analysis.rqs_c13_bath_models import filter_bins


def _median(arr, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(arr, axis=axis)


def _mean(arr, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def _percentile(arr, p, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(arr, p, axis=axis)


def create_figure(source=None, *, min_reference=2.5,
                  min_valid_fraction=.75, outdir=None):
    file=_resolve(source) if source else _latest()
    d=load_file(str(file), reference_min=min_reference)
    _,raw=_load_file(str(file))
    counts=np.asarray(raw["counts"], dtype=float)
    if counts.ndim!=5 or counts.shape[0]!=2:
        raise ValueError("Expected counts[experiment, NV, run, step, repetition]")
    steps=d["steps"]
    rand=np.asarray([i for i,s in enumerate(steps)
                     if s["kind"]=="random_multisine"],dtype=int)
    if len(rand)<4:
        raise ValueError("Not enough random masks")
    indices={k:next((i for i,s in enumerate(steps) if s["kind"]==k),None)
             for k in ("periodic_xy4","periodic_xy8","spin_echo")}
    if any(v is None for v in indices.values()):
        raise ValueError("Missing one of XY4, XY8, spin echo controls")
    C,SE,ref=run_normalized(counts, min_reference)
    Ca,Sa,_=run_normalized(counts[:,:,::2],min_reference)
    Cb,Sb,_=run_normalized(counts[:,:,1::2],min_reference)
    M=C[:,rand];A=Ca[:,rand];B=Cb[:,rand]
    n_masks=len(rand);n_NV=len(d["names"])
    good_counts=np.isfinite(M).sum(axis=1)
    eligible=good_counts>=max(4,int(np.ceil(min_valid_fraction*n_masks)))
    if eligible.sum()<5:
        raise ValueError(
            "Fewer than five NVs have sufficient valid mask measurements. "
            "Try --min-valid-fraction 0.5, or inspect SCC reference quality.")
    r=M[eligible]; a=A[eligible]; b=B[eligible]
    median=_median(r,axis=0)
    mean=_mean(r,axis=0)
    q25,q75=_percentile(r,[25,75],axis=0)
    median_a=_median(a,axis=0)
    median_b=_median(b,axis=0)
    n_pi=np.asarray([steps[k]["n_pi"] for k in rand],dtype=int)
    F=filter_bins([steps[k] for k in rand],
                  d["protocol"]["frequency_edges_hz"])
    k=int(np.argmin(abs(np.asarray(d["protocol"]["frequency_centers_hz"])-
                       d["protocol"]["c13_larmor_hz_expected"])))
    x=F[:,k]/1e-9
    # For each NV, remove a fitted constant and linear pulse-count effect.
    # This is an explanatory nuisance correction, not a physical PSD fit.
    residual=np.full_like(r,np.nan)
    for j,values in enumerate(r):
        valid=np.isfinite(values)
        if valid.sum()<8:continue
        design=np.column_stack([np.ones(valid.sum()),n_pi[valid]])
        p=np.linalg.lstsq(design,values[valid],rcond=None)[0]
        residual[j,valid]=values[valid]-design@p
    corrected=_median(residual,axis=0)
    valid_c=np.isfinite(x)&np.isfinite(corrected)
    rho=float(spearmanr(x[valid_c],corrected[valid_c]).statistic) if valid_c.sum()>5 else np.nan
    # Stability across masks: A & B independently estimate responses.
    valid_ab=np.isfinite(median_a)&np.isfinite(median_b)
    mask_halves_r=float(np.corrcoef(median_a[valid_ab],
                        median_b[valid_ab])[0,1]) if valid_ab.sum()>4 else np.nan
    # More conservative: correlations of NV-centered individual responses
    # across the two halves of acquisition (used in earlier pilot report).
    center_a=a-_mean(a,axis=1)[:,None]
    center_b=b-_mean(b,axis=1)[:,None]
    pairing=np.isfinite(center_a)&np.isfinite(center_b)
    pooled_r=float(np.corrcoef(center_a[pairing],center_b[pairing])[0,1]) \
        if pairing.sum()>5 else np.nan
    controls=[_median(r,axis=1)]+[C[eligible,indices[key]]
         for key in ("periodic_xy4","periodic_xy8","spin_echo")]
    control_medians=[float(_median(v)) for v in controls]
    effective_n=[int(np.isfinite(v).sum()) for v in controls]

    fig,axes=plt.subplots(2,2,figsize=(14,9.4))
    ax,ax2,ax3,ax4=axes.flat
    mask_index=np.arange(1,n_masks+1)
    ax.fill_between(mask_index,q25,q75,alpha=.20,color="tab:blue",
                    label="NV interquartile range (25-75%)")
    ax.plot(mask_index,median,"o-",ms=3.5,lw=1.7,
            color="tab:blue",label="Median across NVs")
    ax.plot(mask_index,mean,"--",lw=1.2,color="tab:orange",
            label="Mean across NVs")
    ax.axhline(0,color="0.5",lw=.8,ls=":")
    ax.set(title="A  Actual randomized-mask response",
           xlabel="Random pulse mask (same sensing duration)",
           ylabel="Raw SCC-normalized X response")
    ax.legend(loc="best",fontsize=9)
    ax.grid(alpha=.20)
    ax.set_xlim(.5,n_masks+.5)
    ax.text(.02,.02,f"{int(eligible.sum())}/{n_NV} NVs pass mask-quality cut",
            transform=ax.transAxes,fontsize=9,
            bbox=dict(boxstyle="round,pad=.3",facecolor="white",alpha=.82,lw=.5))

    ax2.plot(mask_index,median_a,".-",lw=1.4,ms=5,
             color="tab:green",label="Even runs (independent)")
    ax2.plot(mask_index,median_b,".-",lw=1.4,ms=5,
             color="tab:red",label="Odd runs (independent)")
    ax2.axhline(0,color="0.5",lw=.8,ls=":")
    ax2.set(title="B  Does the pattern repeat?",
        xlabel="Random pulse mask",ylabel="Median X response across NVs")
    ax2.set_xlim(.5,n_masks+.5)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=.20)
    ax2.text(.02,.02,
        f"Across-mask median correlation: r={mask_halves_r:.2f}\n"
        f"NV-centered pooled correlation: r={pooled_r:.3f}",
        transform=ax2.transAxes,fontsize=9,
        bbox=dict(boxstyle="round,pad=.3",facecolor="white",alpha=.88,lw=.5))

    color=ax3.scatter(x,corrected,c=n_pi,cmap="viridis",
                      s=42,edgecolors="white",linewidth=.3)
    fig.colorbar(color,ax=ax3,label="Number of refocusing pi pulses",
                 fraction=.045,pad=.03)
    ax3.axhline(0,color="0.5",lw=.8,ls=":")
    ax3.set(title="C  Is there a 33-kHz-dependent response?",
           xlabel="Calculated sensitivity in 33-kHz bin (10⁻⁹ s²)",
           ylabel="Median residual response after NV offset\nand pulse-count correction")
    ax3.grid(alpha=.20)
    ax3.text(.02,.03,f"Across-mask Spearman rho = {rho:.2f}\n"
             "Exploratory association; not a spectral peak",
             transform=ax3.transAxes,fontsize=9,
             bbox=dict(boxstyle="round,pad=.3",facecolor="white",alpha=.85,lw=.5))

    cleaned=[v[np.isfinite(v)] for v in controls]
    ax4.boxplot(cleaned,showfliers=False,whis=(10,90),
                medianprops={"color":"tab:orange","linewidth":2})
    ax4.set_xticks([1,2,3,4])
    ax4.set_xticklabels(["Random masks\nper-NV median",
                         "XY4\n(~33.3 kHz at 60 us)",
                         "XY8\n(~66.7 kHz at 60 us)",
                         "Hahn echo"],fontsize=9)
    for j,mid in enumerate(control_medians,start=1):
        ax4.plot(j,mid,marker="D",color="tab:orange",ms=5)
        ax4.text(j,mid+.045,f"{mid:+.2f}",ha="center",
                 va="bottom",fontsize=8,color="tab:orange")
    ax4.axhline(0,color="0.5",lw=.8,ls=":")
    ax4.set(title="D  Same-duration controls",
            ylabel="Raw SCC-normalized X response")
    ax4.grid(alpha=.20,axis="y")
    ax4.text(.02,.02,"Boxes: 25-75% of NVs; whiskers: 10-90%\n"
             "Orange diamonds: NV medians",
             transform=ax4.transAxes,fontsize=8.5,
             bbox=dict(boxstyle="round,pad=.3",facecolor="white",alpha=.85,lw=.5))

    duration=float(d["T_us"])
    fig.suptitle(f"Measured NV-13C randomized sensing | "
                 f"T={duration:g} us | {n_masks} masks | "
                 f"{n_NV} NVs | {int(d['shot_count']):,} shots/setting",
                 fontsize=16,fontweight="bold",y=.99)
    fig.text(.5,.012,
        "Real Purcell SCC data, NOT a simulation. "
        "Signed response is UNCALIBRATED (do not flip negative values). "
        "No evidence of a 13C line without independent reproducibility/field controls.",
        ha="center",fontsize=9,color="#444444")
    fig.tight_layout(rect=[0,.038,1,.951],h_pad=3.6,w_pad=2.5)

    if outdir:
        dest=Path(outdir)
    else:
        dest=(Path(r"\\192.168.0.197\G\nvdata\pc_Purcell\branch_master")
            /"sc_rqs_c13_bath_analysis"/d["stem"][:7]/d["stem"])
    dest.mkdir(parents=True,exist_ok=True)
    prefix=dest/"RQS_C13_ONE_FIGURE"
    fig.savefig(str(prefix)+".png",dpi=190,bbox_inches="tight")
    fig.savefig(str(prefix)+".pdf",bbox_inches="tight")
    plt.close(fig)

    summary={
        "source":d["stem"],"n_nv_total":n_NV,
        "n_nv_quality_selected":int(eligible.sum()),
        "min_valid_fraction":float(min_valid_fraction),
        "n_random_masks":n_masks,"fixed_T_us":duration,
        "signal_shots_per_setting":int(d["shot_count"]),
        "median_random_signed_X":float(_median(r)),
        "median_full_data_sem_per_NV_mask":float(_median(d["coherence_sem"][:,rand])),
        "mask_medians_split_half_correlation":mask_halves_r,
        "NV_centered_pooled_split_half_correlation":pooled_r,
        "mask_median_spearman_33kHz_after_npi":rho,
        "control_medians":{"random":control_medians[0],
          "XY4":control_medians[1],"XY8":control_medians[2],
          "Hahn_echo":control_medians[3]},
        "control_NV_counts":effective_n,
        "warning":"Raw X sign uncalibrated; simple pulse-count correction "
                  "does not establish a 13C bath spectrum."
    }
    (dest/"RQS_C13_ONE_FIGURE_summary.json").write_text(
        json.dumps(summary,indent=2),encoding="utf-8")
    with (dest/"RQS_C13_ONE_FIGURE_mask_data.csv").open(
         "w",newline="",encoding="utf-8") as fp:
        writer=csv.writer(fp)
        writer.writerow(["mask","num_pi","C_median","C_mean",
                         "C_25pct","C_75pct",
                         "median_even_runs","median_odd_runs",
                         "ideal_33kHz_filter_weight_1e9_s2",
                         "NV_and_pi_adjusted_median_residual","NV_count"])
        for i in range(n_masks):
            writer.writerow([i+1,int(n_pi[i]),median[i],mean[i],
                q25[i],q75[i],median_a[i],median_b[i],x[i],
                corrected[i],int(np.isfinite(r[:,i]).sum())])
    print("RQS_C13_OVERVIEW",json.dumps(summary,indent=2))
    print("PLOT_PNG",str(prefix)+".png")
    print("PLOT_PDF",str(prefix)+".pdf")
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file",help="Data stem or complete path, default latest")
    ap.add_argument("--outdir",help="Optional output folder")
    ap.add_argument("--min-reference",type=float,default=2.5)
    ap.add_argument("--min-valid-fraction",type=float,default=.75)
    opt=ap.parse_args()
    if not 0<opt.min_valid_fraction<=1:
        ap.error("--min-valid-fraction must lie in (0,1]")
    create_figure(opt.file,min_reference=opt.min_reference,
                  min_valid_fraction=opt.min_valid_fraction,
                  outdir=opt.outdir)


if __name__=="__main__":
    main()
