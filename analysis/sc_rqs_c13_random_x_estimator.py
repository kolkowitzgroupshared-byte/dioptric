# -*- coding: utf-8 -*-
"""Random pi_X RQS pilot estimator with SCC and XY8 reference.

noise: reconstruct NONNEGATIVE spectral-weight PROXY from -ln(coherence)
       and exact randomized filter overlaps (weak Gaussian dephasing model).
phase: reconstruct calibrated coherent-AC Fourier quadratures from cosine
       and sine readouts (requires readout='both', known readout-phase sign).
These are experimental estimators, NOT Kannan's complete optimal RQS theorem.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
from analysis import sc_rqs_c13_analysis as legacy

GAMMA_NV_RAD_PER_S_PER_NT = 2*np.pi*28.02495

# Optional: put one or more exact acquisition stems here if you want to
# select runs by editing this file. Leave [] to use the newest RQS pi-X run.
FILE_STEMS = []

# NVOffice's UNC path to Purcell's saved experiments. No G: drive mapping
# is needed. --data-dir overrides this when working elsewhere.
SOURCE_ROOT = Path(
    r"\\192.168.0.197\G\nvdata\pc_Purcell\branch_master\rqs_c13_random_x"
)


def _latest_source():
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            "Cannot access Purcell data folder: " + str(SOURCE_ROOT) +
            ". Check network connectivity or use --files with a full .txt path."
        )
    candidates = [
        p for p in SOURCE_ROOT.glob("20??_??/*.txt")
        if p.with_suffix(".npz").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No complete .txt/.npz pair found in " + str(SOURCE_ROOT)
        )
    latest = max(candidates, key=lambda p: p.name)
    return str(latest)


def _resolve_source(source, data_dir=None):
    """Choose an explicit .txt pair or automatically find its NAS location."""
    source = str(source)
    path = Path(source)
    if path.suffix.lower() in (".txt", ".npz"):
        return source
    if data_dir is not None:
        return source  # existing --data-dir handling in legacy loader
    name = path.name
    # Stems begin 2026_10_24-..., with YYYY_MM as the monthly directory.
    if len(name) >= 7 and name[4:5] == "_" and name[0:4].isdigit():
        guess = SOURCE_ROOT / name[:7] / (name + ".txt")
        if guess.is_file() and guess.with_suffix(".npz").is_file():
            return str(guess)
    return source  # legacy dm fallback for other environments



def _load(source, data_dir=None, min_ref=2.5):
    stem,raw=legacy._load_file(source,data_dir=data_dir)
    p=raw.get("rqs_c13_protocol",{})
    if not p.get("experiment","").startswith("rqs_c13_random_pi_x"):
        raise ValueError(stem+": not a random pi_X acquisition")
    ct=np.asarray(raw["counts"],dtype=float)
    if ct.ndim!=5 or ct.shape[0]!=2 or ct.shape[3]!=len(p["steps"]):
        raise ValueError("Unexpected counts/step shape: "+str(ct.shape))
    if ct.shape[-1]%2 or raw.get("rqs_complete") is not True:
        raise ValueError("Odd repetitions or incomplete acquisition")
    sig=ct[0].mean(axis=(1,3))
    rb=ct[1,:,:,:,::2].mean(axis=(1,3))
    rd=ct[1,:,:,:,1::2].mean(axis=(1,3))
    delta=rb-rd
    with np.errstate(divide="ignore",invalid="ignore"):
        C=np.where(np.abs(delta)>=min_ref,
                   (2*sig-rb-rd)/delta,np.nan)
    steps=p["steps"]
    short=[i for i,s in enumerate(steps)
           if s["kind"]=="random_x" and s["final_pi2_phase_deg"]==0]
    short=sorted(short,key=lambda i:steps[i]["T_ns"])[:15]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        sign=np.sign(np.nanmedian(C[:,short],axis=1))
    sign[~np.isfinite(sign)|(sign==0)]=1
    C*=sign[:,None]
    names=[n["name"] if isinstance(n,dict) else n.name for n in raw["nv_list"]]
    return {"stem":stem,"steps":steps,"proto":p,"C":C,
            "names":names,"shot_count":ct.shape[2]*ct.shape[4]}


def _combine(records):
    if len(records)==1:
        return records[0]
    a=records[0]
    arrays=[]; weights=[]
    for d in records:
        if set(a["names"])!=set(d["names"]):
            raise ValueError("Different NV lists; use intersection-selected NVs")
        if len(a["steps"])!=len(d["steps"]) or any(
            x["waits_cc"]!=y["waits_cc"] or
            x["final_pi2_phase_deg"]!=y["final_pi2_phase_deg"] or
            x["phase_deg"]!=y["phase_deg"]
            for x,y in zip(a["steps"],d["steps"])
        ):
            raise ValueError("Different masks; do not average unmatched filters")
        if abs(a["proto"]["field_g_expected"]-d["proto"]["field_g_expected"])>.3:
            raise ValueError("Field changed; analyze separately")
        arrays.append(d["C"][[d["names"].index(n) for n in a["names"]]])
        weights.append(d["shot_count"])
    x=np.stack(arrays,axis=1)
    w=np.asarray(weights)[None,:,None]
    good=np.isfinite(x)
    den=np.sum(w*good,axis=1)
    mean=np.divide(np.sum(np.where(good,x*w,0),axis=1),den,
                   out=np.full_like(arrays[0],np.nan),where=den>0)
    out=dict(a)
    out.update(C=mean,stem="combined_"+str(len(records))+"_files",
               shot_count=sum(weights))
    return out


def filter_design(steps, edges):
    """Ideal |Y|^2 averaged within each frequency bin, scaled by 1e-9 sÂ²."""
    matrix=[]
    integrate=np.trapezoid
    for s in steps:
        row=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            freq=np.linspace(lo,hi,65)
            row.append(integrate(legacy._filter_power(s,freq),freq)/(hi-lo)/1e-9)
        matrix.append(row)
    return np.asarray(matrix)


def noise_fit(C, steps, edges):
    """chi=-ln(C)=constant(duration)+sum nonnegative bin weights*filter."""
    k=len(edges)-1
    t=np.asarray([int(s["T_ns"]) for s in steps])
    durations=np.unique(t)
    nparams=k+len(durations)
    good=np.isfinite(C)&(C>0.03)&(C<0.999)
    count=int(good.sum())
    if count<3*nparams:
        return {"status":"underidentified","usable":count,"required":3*nparams}
    F=filter_design(steps,edges)
    baselines=(t[:,None]==durations[None,:]).astype(float)
    A=np.column_stack([F[good],baselines[good]])
    y=-np.log(C[good])
    scale=np.linalg.norm(A,axis=0)
    if np.any(scale<1e-10):
        return {"status":"zero_filter_column","usable":count}
    Aunit=A/scale
    rank=np.linalg.matrix_rank(Aunit,tol=1e-8)
    if rank<nparams:
        return {"status":"rank_deficient","usable":count,"rank":int(rank)}
    # Mild regularization on adjacent spectral bins prevents spiky inversion.
    penalty=np.zeros((k-1,nparams))
    for i in range(k-1):
        penalty[i,i]=.025
        penalty[i,i+1]=-.025
    solution,_=nnls(np.vstack((Aunit,penalty)),
                    np.r_[y,np.zeros(k-1)],maxiter=200*nparams)
    coeff=solution/scale
    predicted=A@coeff
    return {"status":"ok","usable":count,"rank":int(rank),
            "spectrum":coeff[:k],
            "baseline":coeff[k:],
            "rms":float(np.sqrt(np.mean((y-predicted)**2)))}


def phase_fit(C, steps, edges, phase_sign=1, nfreq=6):
    """Calibrated coherent-AC Fourier inversion; NOT for stochastic 13C baths."""
    pairs={}
    for j,s in enumerate(steps):
        if s["kind"]=="random_x":
            pairs.setdefault(int(s["mask_id"]),{})[s["final_pi2_phase_deg"]]=j
    paired=[(steps[v[0]],C[v[0]],C[v[90]])
            for v in pairs.values() if 0 in v and 90 in v]
    paired=[(s,cx,cy) for s,cx,cy in paired
            if np.isfinite(cx) and np.isfinite(cy) and cx>.06]
    if len(paired)<max(36,3*2*nfreq):
        return {"status":"insufficient_phase_pairs","usable":len(paired)}
    freq=np.linspace(edges[0],edges[-1],nfreq)
    phi=np.asarray([np.arctan2(phase_sign*cy,cx) for s,cx,cy in paired])
    M=[]
    for s,_,_ in paired:
        ts=np.r_[0.,np.asarray(s["pi_centers_ns"])*1e-9,
                 float(s["T_ns"])*1e-9]
        signs=(-1.)**np.arange(len(ts)-1)
        omega=2*np.pi*freq
        Y=np.sum(signs[None,:]*(
            np.exp(1j*omega[:,None]*ts[1:])-
            np.exp(1j*omega[:,None]*ts[:-1])
        ),axis=1)/(1j*omega)
        M.append(GAMMA_NV_RAD_PER_S_PER_NT*np.r_[Y.real,Y.imag])
    M=np.asarray(M)
    ridge=.03*max(np.linalg.norm(M,ord=2),1e-12)
    coeff=np.linalg.solve(M.T@M+ridge**2*np.eye(2*nfreq),M.T@phi)
    return {"status":"ok","usable":len(paired),
            "frequencies_hz":freq,
            "cos_amplitude_nT_proxy":coeff[:nfreq],
            "sin_amplitude_nT_proxy":coeff[nfreq:],
            "rms":float(np.sqrt(np.mean((phi-M@coeff)**2)))}


def analyze(data, mode="noise", nbin=8, outdir=None, phase_sign=1):
    steps=data["steps"]
    ci=[i for i,s in enumerate(steps)
        if s["kind"]=="random_x" and s["final_pi2_phase_deg"]==0]
    xy=[i for i,s in enumerate(steps) if s["kind"]=="periodic_xy8"]
    if not ci or not xy:
        raise ValueError("Acquisition needs random-X and periodic XY8 controls")
    fxy=np.asarray([steps[i]["target_f_hz"] for i in xy])
    edges=np.linspace(np.min(fxy),np.max(fxy),nbin+1)
    spectra=np.full((len(data["names"]),nbin),np.nan)
    rows=[]
    for i,name in enumerate(data["names"]):
        fit=(noise_fit(data["C"][i,ci],[steps[j] for j in ci],edges)
             if mode=="noise" else phase_fit(data["C"][i],steps,edges,
                                              phase_sign=phase_sign,
                                              nfreq=min(nbin,6)))
        if fit["status"]=="ok":
            key="spectrum" if mode=="noise" else "cos_amplitude_nT_proxy"
            spectra[i,:len(fit[key])]=fit[key]
        rows.append({"index":i,"name":name,"status":fit["status"],
                     "usable_masks":fit.get("usable",0),
                     "fit_rms":fit.get("rms",np.nan)})
    summary={"source":data["stem"],"mode":mode,
             "n_nvs":len(data["names"]),"random_masks":len(ci),
             "xy8_points":len(xy),"f_edges_hz":edges.tolist(),
             "fit_count":sum(r["status"]=="ok" for r in rows),
             "note":"PSD proxy: Gaussian/stationary weak-noise assumption only; NOT hyperfine coupling."}
    fig,axs=plt.subplots(2,1,figsize=(11,9),layout="constrained")
    if np.isfinite(spectra).any():
        im=axs[0].imshow(spectra,origin="lower",aspect="auto",
                extent=[edges[0]/1e3,edges[-1]/1e3,-.5,len(data["names"])-.5],
                cmap="viridis")
        fig.colorbar(im,ax=axs[0],label="PSD proxy (arb.)" if mode=="noise"
                     else "calibration-dependent signed coefficient (nT)")
    else:
        axs[0].text(.5,.5,"Insufficient independent masks for trustworthy fit",
                    transform=axs[0].transAxes,ha="center",va="center")
    axs[0].set(xlabel="Frequency (kHz)",ylabel="NV index",
               title=mode+" estimator: "+str(summary["fit_count"])+" fitted NVs")
    order=np.argsort(fxy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        c=np.nanmedian(data["C"][:,xy],axis=0)
    axs[1].plot(fxy[order]/1e3,c[order],"o-")
    axs[1].set(xlabel="Periodic XY8 frequency (kHz)",
               ylabel="Median SCC-normalized coherence",
               title="Independent frequency-selective XY8 control")
    axs[1].grid(alpha=.2)
    output=io.StringIO(newline="")
    w=csv.DictWriter(output,fieldnames=["index","name","status",
                                        "usable_masks","fit_rms"])
    w.writeheader();w.writerows(rows)
    if outdir:
        out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
        fig.savefig(out/"rqs_pi_x_estimator.pdf")
        fig.savefig(out/"rqs_pi_x_estimator.png",dpi=180)
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
        (out/"per_nv.csv").write_text(output.getvalue())
        np.savez_compressed(out/"fit_arrays.npz",
                            spectral_proxy=spectra,f_edges_hz=edges,
                            coherence=data["C"])
        print("Saved estimator files:",out)
    else:
        from utils import data_manager as dm
        base=dm.get_file_path(__file__,dm.get_time_stamp(),
                              "rqs_pi_x_"+mode,subfolder=data["stem"])
        # Preserve the lab data_manager naming and saving API, but use the
        # UNC NAS location when NVOffice has no mapped G: drive.
        g_root=Path("G:/nvdata")
        if not g_root.is_dir():
            try:
                suffix=base.relative_to(g_root)
            except ValueError as exc:
                raise RuntimeError(
                    "Cannot resolve dm output under G:/nvdata: "+str(base)
                ) from exc
            base=Path(r"\\192.168.0.197\G\nvdata")/suffix
            if not Path(r"\\192.168.0.197\G\nvdata").is_dir():
                raise FileNotFoundError(
                    "Neither mapped G: nor NAS UNC root is accessible."
                )
            print("G: unavailable; writing dm files via NAS UNC:",base)
        dm.save_figure(fig,base)
        dm.save_text(json.dumps(summary,indent=2),base,suffix=".json")
        dm.save_text(output.getvalue(),base,suffix=".csv")
        dm.save_npz(base,compressed=True,spectral_proxy=spectra,
                    f_edges_hz=edges,coherence=data["C"])
        print("Saved estimator files with dm:",base)
    plt.close(fig)
    print("Fitted",summary["fit_count"],"of",summary["n_nvs"],
          "NVs; independent random masks:",len(ci))
    return summary


def self_test():
    from majorroutines.widefield.rqs_c13 import build_protocol
    p=build_protocol(band="high",masks_per_duration=16,pi_ns=96,pi2_ns=48)
    steps=[s for s in p["steps"] if s["kind"]=="random"]
    edges=np.linspace(150e3,1.5e6,9)
    F=filter_design(steps,edges)
    t=np.asarray([s["T_ns"] for s in steps])
    D=(t[:,None]==np.unique(t)[None,:]).astype(float)
    true=np.asarray([.01,.01,.02,.08,.02,.01,.01,.01])
    C=np.exp(-F@true-D@np.full(4,.04))
    fit=noise_fit(C,steps,edges)
    assert fit["status"]=="ok",fit
    assert noise_fit(C[:8],steps[:8],edges)["status"]=="underidentified"
    print("PASS synthetic nonnegative noise estimator and rank guard")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",help="Saved RQS pi-X stems or .txt paths")
    p.add_argument("--latest",action="store_true",
                   help="Select newest complete Purcell RQS pi-X run")
    p.add_argument("--data-dir",default=None)
    p.add_argument("--outdir",default=None)
    p.add_argument("--mode",choices=("noise","phase"),default="noise")
    p.add_argument("--bins",type=int,default=8)
    p.add_argument("--ref-min",type=float,default=2.5)
    p.add_argument("--phase-sign",type=int,choices=(-1,1),default=1)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test();return
    if not 2<=args.bins<=20:
        p.error("--bins must be 2..20")
    if args.files:
        selected = args.files
    elif args.latest or not FILE_STEMS:
        selected = [_latest_source()]
    else:
        selected = FILE_STEMS
    sources = [_resolve_source(x,args.data_dir) for x in selected]
    print("Selected RQS pi-X input(s):")
    for source in sources:
        print("  ",source)
    raw=[_load(x,args.data_dir,args.ref_min) for x in sources]
    d=_pool_different_masks(raw)
    analyze(d,args.mode,args.bins,args.outdir,args.phase_sign)






def _pool_different_masks(records):
    """Aggregate identical filters, but RETAIN new masks from distinct seeds.

    Align NVs by stable name. A mask's identity comes from its exact pulse
    schedule rather than the file-local mask_id number.
    """
    if len(records)==1:
        return records[0]
    ref=records[0]["proto"]
    for d in records[1:]:
        p=d["proto"]
        if p["band"]!=ref["band"] or abs(p["field_g_expected"]-ref["field_g_expected"])>.3:
            raise ValueError("Different field/band cannot be combined")
        for k in ("pi_ns","pi2_ns","number_of_pi_pulses"):
            if p[k]!=ref[k]:
                raise ValueError("Different control calibration: "+k)
    names=[x for x in records[0]["names"]
           if all(x in f["names"] for f in records)]
    if not names:
        raise ValueError("No common NVs between datasets")
    values={}; weights={}; templates={}
    for d in records:
        arr=d["C"][[d["names"].index(name) for name in names]]
        for j,s in enumerate(d["steps"]):
            key=(s["kind"],tuple(s["waits_cc"]),tuple(s["phase_deg"]),
                 int(s["final_pi2_phase_deg"]))
            templates[key]=s
            values.setdefault(key,[]).append(arr[:,j])
            weights.setdefault(key,[]).append(d["shot_count"])
    result_C=[]; result_steps=[]
    mid_map={}
    for key,s in templates.items():
        rows=np.stack(values[key],axis=1)
        w=np.asarray(weights[key])[None,:]
        good=np.isfinite(rows)
        denom=np.sum(good*w,axis=1)
        C=np.divide(np.sum(np.where(good,rows*w,0),axis=1),denom,
                    out=np.full(len(names),np.nan),where=denom>0)
        result_C.append(C)
        step=dict(s)
        mask_key=(s["kind"],tuple(s["waits_cc"]))
        if mask_key not in mid_map:
            mid_map[mask_key]=len(mid_map)
        step["mask_id"]=mid_map[mask_key]
        result_steps.append(step)
    out=dict(records[0])
    out.update(names=names,steps=result_steps,
               C=np.stack(result_C,axis=1),
               stem="pooled_"+str(len(records))+"_sources",
               shot_count=sum(f["shot_count"] for f in records))
    print("Pooled",len(records),"files into",len(result_steps),
          "unique filter/basis settings (unmatched masks retained)")
    return out




if __name__=="__main__":
    main()

