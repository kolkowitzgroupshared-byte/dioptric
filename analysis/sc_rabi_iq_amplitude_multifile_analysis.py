# -*- coding: utf-8 -*-
"""Combine IQ AMPLITUDE Rabi batches, show each NV in ONE multipage PDF.

Edit FILE_STEMS and run from the dioptric root:
    python -m analysis.sc_rabi_iq_amplitude_multifile_analysis

Or:
    python -m analysis.sc_rabi_iq_amplitude_multifile_analysis --files STEM1 STEM2 ...
    python -m analysis.sc_rabi_iq_amplitude_multifile_analysis --files STEM1 STEM2 --outdir C:\\temp\\rabi

For valid comparisons, batches must have equal IQ amplitude grids, pulse
duration, IQ phase, microwave chain, and saved SG394 frequency/power when
available. NVs are matched by NAME, not position. Each batch is independently
normalized using its own per-step dual-rail SCC reference BEFORE shot-weighted
averaging; raw counts are NEVER concatenated across batches.

Outputs on Purcell G:/NAS via data_manager:
  - one PDF with overview page + one page per common NV: +/-1 SEM and fit
  - averaged Rabi plot, NV heatmap, per-NV fit CSV, response CSV,
    source-files CSV, summary JSON, combined compressed NPZ.
The free-period cosine fit can be underconstrained when the amplitude span
contains less than one full oscillation (e.g., 96 ns at calibrated max power).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from analysis.sc_rabi_iq_amplitude_analysis import (
    cosine_model, fit_trace, load_data, process,
)
from utils import data_manager as dm

# EDIT THIS LIST to add independent timestamped acquisitions.
FILE_STEMS = [
    "2026_10_27-18_08_16-johnson-nv0_2026_10_02",
]
DATA_DIR = None            # None = the configured Purcell G: / NAS folder.
OUTDIR = None              # None = auto-save analysis to G: / NAS.
MIN_REFERENCE_CONTRAST = 0.0  # photons; set 0.0 to remove quality cutoff
MIN_REFERENCE_SNR = 0.0       # set 0.0 to remove quality cutoff
NAS_UNC_ROOT = r"\\192.168.0.197\G"


def _stem_name(arg):
    p=Path(str(arg))
    if p.suffix.lower() in (".txt", ".npz"):
        p=p.with_suffix("")
    return p.name


def _name(nv):
    value=(nv.get("name") if isinstance(nv, dict)
           else getattr(nv, "name", None))
    if not value:
        raise ValueError("Saved NV has no stable name: cannot align runs")
    return str(value)


def _load_one(arg, data_dir, min_contrast, min_snr):
    raw, source = load_data(arg, data_dir=data_dir)
    names=[_name(nv) for nv in raw["nv_list"]]
    if len(set(names))!=len(names):
        raise ValueError(f"{arg}: NV names are not unique")
    arr=np.asarray(raw["counts"])
    if arr.ndim != 5 or arr.shape[0] != 2 or arr.shape[1] != len(names):
        raise ValueError(f"{arg}: wrong IQ Rabi count shape {arr.shape}")
    if arr.shape[2] != int(raw["num_runs"]) or arr.shape[4] != int(raw["num_reps"]):
        raise ValueError(f"{arg}: incomplete runs/repetitions in saved counts")
    if len(raw.get("step_ind_master_list",[])) not in (0,int(raw["num_runs"])):
        raise ValueError(f"{arg}: step order/run metadata is incomplete")
    crash=raw.get("crash_counter",[])
    if crash and (len(crash) != int(raw["num_runs"])
                  or any(x is None for x in crash)):
        raise ValueError(f"{arg}: incomplete acquisition; refusing batch average")
    if raw.get("num_exps_per_step",2) != 2:
        raise ValueError(f"{arg}: not the two-experiment amplitude-Rabi protocol")
    processed=process(raw,min_contrast=min_contrast,min_snr=min_snr)
    if np.any(np.diff(processed["amplitudes_v"])<=0):
        raise ValueError(f"{arg}: amplitudes must increase strictly")
    return dict(
        stem=_stem_name(arg), source=source, raw=raw, result=processed,
        names=names, index={n:i for i,n in enumerate(names)},
        shots=int(raw["num_reps"])*int(raw["num_runs"]),
        microwaves=list(raw.get("uwave_ind_list",raw.get("uwave_ind",[]))),
        freq=raw.get("uwave_freq"),
        power=raw.get("uwave_power_dbm"),
    )


def _equal_optional(a,b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return bool(np.array_equal(np.asarray(a),np.asarray(b)))


def combine(file_stems=FILE_STEMS,*,data_dir=DATA_DIR,
            min_ref_contrast=MIN_REFERENCE_CONTRAST,
            min_ref_snr=MIN_REFERENCE_SNR):
    file_stems=[str(s) for s in file_stems]
    if not file_stems:
        raise ValueError("FILE_STEMS empty: supply one or more amplitude-Rabi runs")
    if len({_stem_name(s) for s in file_stems})!=len(file_stems):
        raise ValueError("Duplicate file stem; do not count one batch twice")
    entries=[_load_one(s,data_dir,min_ref_contrast,min_ref_snr)
             for s in file_stems]
    base=entries[0]
    axis=base["result"]["amplitudes_v"]
    t_ns=base["result"]["pulse_duration_ns"]
    phase=base["result"]["phase_deg"]
    for e in entries[1:]:
        p=e["result"]
        if (p["amplitudes_v"].shape!=axis.shape or
            not np.allclose(p["amplitudes_v"],axis,atol=1e-12,rtol=0)):
            raise ValueError(f"{e['stem']}: IQ amplitude grid differs")
        if p["pulse_duration_ns"]!=t_ns:
            raise ValueError(f"{e['stem']}: pulse duration differs; do not average")
        if p["phase_deg"]!=phase:
            raise ValueError(f"{e['stem']}: IQ phase differs; do not average")
        if e["microwaves"]!=base["microwaves"]:
            raise ValueError(f"{e['stem']}: microwave chain differs")
        for key in ("freq","power"):
            a,b=base[key],e[key]
            if not _equal_optional(a,b):
                if a is None or b is None:
                    warnings.warn(
                        f"{e['stem']}: {key} metadata missing from a batch. "
                        "Check SG394 settings were identical.",
                    )
                else:
                    raise ValueError(f"{e['stem']}: microwave {key} differs")
    if base["freq"] is None:
        warnings.warn(
            "Saved amplitude-Rabi file has no microwave frequency metadata; "
            "verify the SG394 settings manually before combining batches."
        )
    common=set(base["names"])
    for e in entries[1:]:
        common.intersection_update(e["names"])
    names=[n for n in base["names"] if n in common]
    if not names:
        raise ValueError("No matching NV names across the selected batches")

    obs=[]; errs=[]; masks=[]; shots=[]
    for e in entries:
        inds=[e["index"][n] for n in names]
        res=e["result"]
        obs.append(res["p1"][inds])
        errs.append(res["p1_sem"][inds])
        masks.append(res["valid"][inds])
        shots.append(e["shots"])
    obs=np.array(obs,dtype=float)             # [batch, nv, amp step]
    errs=np.array(errs,dtype=float)
    masks=np.array(masks,dtype=bool)
    masks &= np.isfinite(obs)&np.isfinite(errs)&(errs>=0)
    weights=np.where(masks,np.array(shots,dtype=float)[:,None,None],0.)
    W=np.sum(weights,axis=0)
    valid=W>0
    shape=W.shape
    with np.errstate(divide="ignore",invalid="ignore"):
        y=np.divide(np.sum(np.where(masks,obs,0.)*weights,axis=0),
                    W,out=np.full(shape,np.nan),where=valid)
        # Independent per-batch measurement uncertainties
        within=np.divide(
            np.sqrt(np.sum(np.where(masks,errs,0.)**2*weights**2,axis=0)),
            W,out=np.full(shape,np.nan),where=valid
        )
        weight2=np.sum(weights**2,axis=0)
        n_eff=np.divide(W**2,weight2,out=np.zeros_like(W),where=weight2>0)
        variance_denom=W-np.divide(
            weight2,W,out=np.zeros_like(W),where=valid
        )
        scatter=np.sum(weights*np.where(masks,obs-y[None,:,:],0.)**2,
                       axis=0)
        between=np.sqrt(np.divide(
            scatter,variance_denom*n_eff,
            out=np.zeros_like(W),where=(variance_denom*n_eff)>0
        ))
        yerr=np.where(valid,np.maximum(within,between),np.nan)

    return dict(
        amplitudes_v=axis, p1=y,p1_sem=yerr, sigma_z=1-2*y,
        valid=valid, n_files_contributing=np.sum(masks,axis=0),
        total_shots_contributing=W, nv_names=names,
        nv_indices=np.arange(len(names)),
        pulse_duration_ns=t_ns,phase_deg=phase,
        microwaves=base["microwaves"],
        entries=entries, num_files=len(entries),
        min_ref_contrast=min_ref_contrast,min_ref_snr=min_ref_snr,
    )


def _fit_all(result):
    """Return only finite fit results; do not pretend half an oscillation is robust."""
    x=result["amplitudes_v"]
    fits=[]
    for y,err in zip(result["p1"],result["p1_sem"]):
        valid=np.isfinite(y)&np.isfinite(err)&(err>0)
        fits.append(fit_trace(x[valid],y[valid],err[valid])
                    if np.sum(valid)>=8 else None)
    return fits


def _overview(result, fits):
    x=result["amplitudes_v"]
    y=np.where(result["valid"],result["p1"],np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        mu=np.nanmedian(y,axis=0)
        q25=np.nanpercentile(y,25,axis=0)
        q75=np.nanpercentile(y,75,axis=0)
    fig,ax=plt.subplots(figsize=(10,5.8),layout="constrained")
    ax.errorbar(x,mu,yerr=[
        np.maximum(0,mu-q25),np.maximum(0,q75-mu)],
        fmt="o",ms=3,capsize=2,label="NV median (25-75% spread)")
    ax.set(
        xlabel="IQ vector amplitude at OPX DAC (V)",
        ylabel=r"SCC normalized $P(m_s=-1)$",
        title=(f"IQ-amplitude Rabi | {result['pulse_duration_ns']} ns | "
               f"phase {result['phase_deg']}? | {result['num_files']} batch(es)"),
    )
    ax.grid(alpha=.22);ax.legend()
    return fig


def _heatmap(result):
    fig,ax=plt.subplots(figsize=(9.2,6),layout="constrained")
    x=result["amplitudes_v"]
    data=np.where(result["valid"],result["p1"],np.nan)
    im=ax.imshow(data,origin="lower",aspect="auto",interpolation="nearest",
                 extent=(x[0],x[-1],-.5,len(data)-.5),vmin=0,vmax=1)
    ax.set(xlabel="IQ amplitude (V)",ylabel="Common NV index",
           title="Batch-averaged Rabi P1 (unaccepted points masked)")
    fig.colorbar(im,ax=ax,label="P(ms=-1)")
    return fig


def _pdf(result,fits,outpath):
    x=result["amplitudes_v"]
    outpath.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(outpath,metadata={
        "Title":"Purcell IQ amplitude Rabi: combined batches, all NVs",
        "Subject":"One page per NV with raw normalized P1, 1-SEM and fit"
    }) as pdf:
        overview=_overview(result,fits)
        pdf.savefig(overview,bbox_inches="tight")
        plt.close(overview)
        for i,(name,fit) in enumerate(zip(result["nv_names"],fits)):
            y=result["p1"][i]
            err=result["p1_sem"][i]
            accepted=result["valid"][i]&np.isfinite(y)&np.isfinite(err)
            fig,ax=plt.subplots(figsize=(9,6),layout="constrained")
            if accepted.any():
                ax.errorbar(x[accepted],y[accepted],yerr=err[accepted],
                            fmt="o",markersize=4.4,capsize=2.2,elinewidth=1.1,
                            label="Combined SCC P1 (1 SEM)",zorder=3)
            rejected=~accepted
            if rejected.any():
                ax.plot(x[rejected],np.full(rejected.sum(),-.16),"rx",
                        label="Reference rejected")
            ax.axhline(0,color="gray",lw=.6)
            ax.axhline(1,color="gray",lw=.6)
            fit_txt="No identifiable period / insufficient usable points"
            if fit:
                fit_x=np.linspace(float(x[0]),float(x[-1]),700)
                ax.plot(fit_x,cosine_model(fit_x,*fit["fit_params"]),
                        lw=2,label="Cosine fit")
                fit_txt=(f"Period={fit['period_v']:.4g} V | "
                         f"first fitted pi={fit['pi_amplitude_v']:.4g} V | "
                         f"contrast={fit['contrast']:.3g}")
            if accepted.any():
                r=max(1.16,float(np.nanmax(y[accepted]+err[accepted]))+.13)
                lo=min(-.23,float(np.nanmin(y[accepted]-err[accepted]))-.14)
                ax.set_ylim(max(lo,-5),min(r,5))
            else:
                ax.set_ylim(-.22,1.25)
            ax.set(
                xlabel="IQ vector amplitude (V)",
                ylabel=r"$P(m_s=-1)$ from SCC",
                title=(f"IQ-amplitude Rabi | NV {i+1}/{len(fits)}: {name}\n"
                       f"fixed MW duration {result['pulse_duration_ns']} ns; "
                       f"phase {result['phase_deg']}?"),
            )
            ax.grid(alpha=.18)
            ax.legend(loc="best",fontsize=9)
            fig.text(.13,.025,
                     f"{fit_txt}   |   accepted {accepted.sum()}/{len(x)} "
                     f"amplitudes   |   {result['num_files']} source batch(es)",
                     fontsize=8.5,ha="left")
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)
    print(f"Single PDF ({len(fits)+1} pages): {outpath}")
    return outpath


def _output_folder(outdir):
    if outdir is not None:
        return Path(outdir),"iq_amplitude_rabi_multifile"
    now=dm.get_time_stamp()
    base=Path(dm.get_file_path(__file__,now,"iq_amplitude_rabi_multifile"))
    if str(base).upper().startswith("G:") and not Path("G:/").is_dir():
        base=Path(NAS_UNC_ROOT+str(base)[2:])
    if not base.parent.is_dir():
        base.parent.mkdir(parents=True,exist_ok=True)
    return base.parent,base.stem


def analyze_many(file_stems=FILE_STEMS,*,data_dir=DATA_DIR,outdir=OUTDIR,
                 min_ref_contrast=MIN_REFERENCE_CONTRAST,
                 min_ref_snr=MIN_REFERENCE_SNR):
    res=combine(file_stems,data_dir=data_dir,
                min_ref_contrast=min_ref_contrast,min_ref_snr=min_ref_snr)
    folder,basename=_output_folder(outdir)
    folder.mkdir(parents=True,exist_ok=True)
    fits=_fit_all(res)
    pdf=_pdf(res,fits,folder/(basename+"-all_individual_nvs.pdf"))
    fig=_overview(res,fits)
    fig.savefig(folder/(basename+"-overview.png"),dpi=180)
    plt.close(fig)
    fig=_heatmap(res)
    fig.savefig(folder/(basename+"-heatmap.png"),dpi=180)
    plt.close(fig)
    with (folder/(basename+"-per_nv_fits.csv")).open(
        "w",newline="",encoding="utf8"
    ) as f:
        writer=csv.writer(f)
        writer.writerow(["nv_name","n_accepted_amps","period_v","period_sem_v",
                         "first_pi_amplitude_v","contrast","chi2"])
        for i,(name,fit) in enumerate(zip(res["nv_names"],fits)):
            writer.writerow([name,int(np.sum(res["valid"][i]))]+(
                [fit["period_v"],fit["period_ste_v"],fit["pi_amplitude_v"],
                 fit["contrast"],fit["chi2"]]
                if fit else [""]*5))
    with (folder/(basename+"-normalized_response.csv")).open(
        "w",newline="",encoding="utf8"
    ) as f:
        writer=csv.writer(f)
        writer.writerow(["nv_name","amplitude_v","p1","p1_sem",
                         "sigma_z","valid","num_batches","total_shots"])
        for i,name in enumerate(res["nv_names"]):
            for j,amp in enumerate(res["amplitudes_v"]):
                writer.writerow([
                    name,float(amp),res["p1"][i,j],res["p1_sem"][i,j],
                    res["sigma_z"][i,j],bool(res["valid"][i,j]),
                    int(res["n_files_contributing"][i,j]),
                    int(res["total_shots_contributing"][i,j]),
                ])
    with (folder/(basename+"-source_files.csv")).open(
        "w",newline="",encoding="utf8"
    ) as f:
        writer=csv.writer(f)
        writer.writerow(["file_stem","nv_count","num_runs","num_reps",
                         "shots_per_step","valid_nv_step_fraction"])
        for e in res["entries"]:
            writer.writerow([
                e["stem"],len(e["names"]),e["raw"]["num_runs"],
                e["raw"]["num_reps"],e["shots"],
                float(np.mean(e["result"]["valid"]))
            ])
    np.savez_compressed(
        folder/(basename+"-combined.npz"),
        nv_names=np.asarray(res["nv_names"],dtype=str),
        iq_amplitudes_v=res["amplitudes_v"],p1=res["p1"],
        p1_sem=res["p1_sem"],sigma_z=res["sigma_z"],
        valid=res["valid"],n_files_contributing=res["n_files_contributing"],
        total_shots_contributing=res["total_shots_contributing"],
    )
    valid=res["valid"]
    summary={
        "experiment":"Fixed duration IQ-amplitude Rabi, multifile",
        "num_batches":res["num_files"],
        "file_stems":[e["stem"] for e in res["entries"]],
        "common_nvs":len(res["nv_names"]),
        "num_amplitudes":len(res["amplitudes_v"]),
        "amp_min_v":float(res["amplitudes_v"][0]),
        "amp_max_v":float(res["amplitudes_v"][-1]),
        "pulse_duration_ns":res["pulse_duration_ns"],
        "phase_deg":res["phase_deg"],
        "microwave_chains":res["microwaves"],
        "min_reference_contrast_photons":min_ref_contrast,
        "min_reference_snr":min_ref_snr,
        "accepted_nv_step_pairs":int(valid.sum()),
        "total_nv_step_pairs":int(valid.size),
        "n_nv_period_fits":sum(f is not None for f in fits),
        "pdf_path":str(pdf),
        "normalization":"per-file, per-NV and amplitude-step dual-rail reference",
        "aggregation":"shot-weighted normalized responses, SEM=max(within-file propagated, between-batch dispersion)",
        "warning":"A 96 ns amplitude sweep reaching only pi at 0.5 V spans half a population oscillation; fitted periods may be weakly constrained. Data are not gate fidelities.",
    }
    (folder/(basename+"-summary.json")).write_text(
        json.dumps(summary,indent=2),encoding="utf8"
    )
    print(f"Combined {len(res['entries'])} files; {len(res['nv_names'])} common NVs")
    print(f"Accepted NV/amplitude pairs: {valid.sum()}/{valid.size}")
    print("Saved results:",folder)
    return {"combined":res,"summary":summary,"pdf":pdf,"folder":folder}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files",nargs="+",default=None,
                    help="Saved timestamp-NV stems; overrides FILE_STEMS")
    ap.add_argument("--data-dir",default=DATA_DIR)
    ap.add_argument("--outdir",default=OUTDIR)
    ap.add_argument("--min-contrast",type=float,default=MIN_REFERENCE_CONTRAST)
    ap.add_argument("--min-snr",type=float,default=MIN_REFERENCE_SNR)
    args=ap.parse_args()
    return analyze_many(FILE_STEMS if args.files is None else args.files,
                        data_dir=args.data_dir,outdir=args.outdir,
                        min_ref_contrast=args.min_contrast,
                        min_ref_snr=args.min_snr)


if __name__=="__main__":
    main()
