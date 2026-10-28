# -*- coding: utf-8 -*-
"""Reconstruct phase-locked coherent AC tones using fixed-T multisine RQS.

Loads Purcell's RQS multisine SCC raw .txt/.npz and companion -bank.npz.
Uses the uploaded toolkit's signed Y response covariance inversion and
Bessel radial amplitude correction, ported to rqs_multisine_core.py.
Only valid for shot-by-shot phase-locked AC fields, NOT stochastic 13C.
Run with --latest from NVOffice; requires calibrated readout polarity.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import sys
from pathlib import Path
if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.rqs_multisine_core import Bank,estimate
from analysis.sc_rqs_c13_analysis import _load_file

SOURCE_ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_Purcell\branch_master\rqs_multisine")


def latest_source():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError("NAS RQS multisine folder unavailable")
    files=[p for p in SOURCE_ROOT.glob("20??_??/*.txt")
           if p.with_suffix(".npz").exists()
           and p.with_name(p.stem+"-bank.npz").exists()]
    if not files:
        raise FileNotFoundError("No completed multisine RQS data and bank found")
    return max(files,key=lambda p:p.name)


def load(path):
    path=Path(path)
    if path.suffix!=".txt":path=path.with_suffix(".txt")
    stem,raw=_load_file(str(path))
    proto=raw.get("rqs_multisine_protocol",{})
    if proto.get("experiment")!="rqs_fixed_T_multisine":
        raise ValueError("Not a fixed-duration multisine RQS acquisition")
    if proto.get("field_mode")!="phase_locked":
        raise ValueError("Missing shot-by-shot phase-lock assumption")
    if raw.get("rqs_complete") is not True:
        raise ValueError("Saved run is incomplete")
    bank_file=path.with_name(path.stem+"-bank.npz")
    with np.load(bank_file,allow_pickle=False) as z:
        lengths=[row[row>=0].tolist() for row in z["pulse_center_ticks"]]
        bank=Bank(frequencies_hz=z["frequencies_hz"].copy(),
            duration_s=float(z["duration_s"]),
            coupling_hz=float(z["coupling_hz"]),
            pulses_ticks=[tuple(int(v) for v in row) for row in lengths],
            initial_sign=z["initial_sign"].copy(),
            responses=z["response_vectors"].copy(),
            sigma=z["covariance"].copy(),
            condition=float(z["condition"]),
            clock_s=float(z["clock_s"]),
            seed=int(proto["seed"]),
            max_pulses=int(z["pulse_center_ticks"].shape[1]))
    steps=proto["rows"]
    counts=np.asarray(raw["counts"],dtype=float)
    if counts.ndim!=5 or counts.shape[0]!=2 or counts.shape[3]!=len(steps):
        raise ValueError("Invalid SCC signal/reference shape")
    if counts.shape[-1]%2:
        raise ValueError("Reference parity requires even repetitions")
    names=[nv["name"] for nv in raw["nv_list"]]
    signal=counts[0].mean(axis=(1,3))
    refplus=counts[1,:,:,:,::2].mean(axis=(1,3))
    refminus=counts[1,:,:,:,1::2].mean(axis=(1,3))
    delta=refplus-refminus
    with np.errstate(divide="ignore",invalid="ignore"):
        physical=(2*signal-refplus-refminus)/delta
    return bank,steps,names,physical,delta,proto


def analyze(bank,rows,names,physical,ref_delta,proto,
            *,x_polarity=1,y_polarity=1,min_ref=2.5,
            visibility_floor=.02,bound=1.,outdir=None,
            readout_calibrated=False):
    if x_polarity not in (-1,1) or y_polarity not in (-1,1):
        raise ValueError("Readout polarity must be +1 or -1")
    if not 0<min_ref or not 0<bound:
        raise ValueError("Invalid reference threshold or amplitude bound")
    n=len(names); nt=len(bank.frequencies_hz)
    m=physical.copy()
    for k,row in enumerate(rows):
        m[:,k]*=x_polarity if row["basis"]=="X" else y_polarity
    invalid=(~np.isfinite(m)) | (np.abs(m)>1) | (np.abs(ref_delta)<min_ref)
    m[invalid]=np.nan
    amplitudes=np.full((n,2*nt),np.nan)
    mu=np.full(n,np.nan)
    result_rows=[]
    for j,name in enumerate(names):
        fit=estimate(bank,rows,m[j],contrast_floor=visibility_floor,
                     bound=bound)
        if fit["status"]=="ok":
            amplitudes[j]=fit["amplitudes"]
            mu[j]=fit["mu"]
        result_rows.append({
            "nv":name,"nv_index":j,"status":fit["status"],
            "x_mean":fit.get("mu",np.nan),
            "num_valid_x":fit.get("n_x",fit.get("x",0)),
            "num_valid_y":fit.get("n_y",fit.get("y",0)),
            "contrast_floor_triggered":fit.get("floor_active",False),
            "median_reference_delta":float(np.nanmedian(np.abs(ref_delta[j])))
        })
    magnitude=np.hypot(amplitudes[:,:nt],amplitudes[:,nt:])
    status_count=sum(r["status"]=="ok" for r in result_rows)
    freq=bank.frequencies_hz/1e3
    charts={}
    fig,ax=plt.subplots(figsize=(11,6),layout="constrained")
    if np.isfinite(magnitude).any():
        img=ax.imshow(magnitude,origin="lower",aspect="auto",cmap="viridis",
                      extent=[freq.min(),freq.max(),-.5,n-.5])
        fig.colorbar(img,ax=ax,label="Coherent AC amplitude / assumed B0")
    ax.set(xlabel="Reconstruction grid (kHz)",ylabel="NV index",
           title="RQS phase-locked reconstructed Fourier amplitudes (pilot)")
    charts["01_amplitude_map"]=fig
    fig,ax=plt.subplots(figsize=(11,5),layout="constrained")
    ax.plot(freq,np.nanmedian(magnitude,axis=0),"o-",label="Median across fitted NVs")
    ax.set(xlabel="Frequency (kHz)",ylabel="Reconstructed amplitude / B0",
           title="Multisine RQS spectrum (requires phase calibration)")
    ax.legend();ax.grid(alpha=.25)
    charts["02_amplitude_summary"]=fig
    fig,axs=plt.subplots(2,1,figsize=(10,8),layout="constrained")
    axs[0].hist(mu[np.isfinite(mu)],bins=20)
    axs[0].set(xlabel="Mean calibrated X readout",ylabel="NV count",
               title="Contrast/attenuation observable used in RQS denominator")
    axs[1].hist(np.nanmedian(np.abs(ref_delta),axis=1),bins=20)
    axs[1].axvline(min_ref,color="red",linestyle="--",label="SCC reference cutoff")
    axs[1].set(xlabel="Median bright-dark SCC difference (photons)",
               ylabel="NV count",title="SCC data quality")
    axs[1].legend()
    charts["03_readout_QC"]=fig
    summary={"n_nvs":n,"fit_count":status_count,
             "n_frequencies":nt,"frequencies_hz":bank.frequencies_hz.tolist(),
             "duration_us":bank.duration_s*1e6,
             "bank_covariance_condition":bank.condition,
             "min_reference_delta":min_ref,
             "readout_calibrated":bool(readout_calibrated),
             "phase_locked_assumed":True,
             "warning":"Coherent AC Fourier amplitudes only; NOT stochastic nuclear PSD."}
    buf=io.StringIO(newline="")
    writer=csv.DictWriter(buf,fieldnames=list(result_rows[0]))
    writer.writeheader();writer.writerows(result_rows)

    if outdir is None:
        from utils import data_manager as dm
        base=dm.get_file_path(__file__,dm.get_time_stamp(),
                              "rqs_multisine_reconstruction",
                              subfolder=str(proto["fixed_T_us"])+"us")
        g=Path("G:/nvdata")
        if not g.is_dir():
            base=Path(r"\\192.168.0.197\G\nvdata")/base.relative_to(g)
        for title,fig in charts.items():
            path=base.with_name(base.stem+"-"+title)
            dm.save_figure(fig,path)
            pdf=io.BytesIO();fig.savefig(pdf,format="pdf",bbox_inches="tight")
            dm.cloud.upload(path.with_suffix(".pdf"),pdf.getvalue())
        dm.save_text(buf.getvalue(),base,suffix=".csv")
        dm.save_text(json.dumps(summary,indent=2),base,suffix=".json")
        dm.save_npz(base,compressed=True,amplitudes=amplitudes,
                    magnitudes=magnitude,mu=mu,
                    frequencies_hz=bank.frequencies_hz)
        print("Saved reconstruction via dm:",base)
    else:
        out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
        for title,fig in charts.items():
            fig.savefig(out/(title+".pdf"))
            fig.savefig(out/(title+".png"),dpi=150)
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
        (out/"per_nv.csv").write_text(buf.getvalue())
        np.savez_compressed(out/"arrays.npz",amplitudes=amplitudes,
                 magnitudes=magnitude,mu=mu,
                 frequencies_hz=bank.frequencies_hz)
    for fig in charts.values():plt.close(fig)
    print("RQS reconstruction:",status_count,"/",n,"NVs")
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file",help="Full .txt path (default: latest complete RQS)")
    ap.add_argument("--outdir")
    ap.add_argument("--x-polarity",type=int,choices=(-1,1),default=1)
    ap.add_argument("--y-polarity",type=int,choices=(-1,1),default=1)
    ap.add_argument("--ref-min",type=float,default=2.5)
    ap.add_argument("--readout-calibrated",action="store_true")
    args=ap.parse_args()
    path=args.file or latest_source()
    print("Loading",path)
    bank,rows,names,mean,delta,proto=load(path)
    analyze(bank,rows,names,mean,delta,proto,
            x_polarity=args.x_polarity,y_polarity=args.y_polarity,
            min_ref=args.ref_min,readout_calibrated=args.readout_calibrated,
            outdir=args.outdir)


if __name__=="__main__":
    main()
