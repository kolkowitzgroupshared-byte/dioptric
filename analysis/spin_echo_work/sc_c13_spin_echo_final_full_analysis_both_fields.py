"""Generate publication-style full spin-echo analysis reports for 49G and 52G.

49G: Johnson sample, completed V22 result.
52G: QNami sample, refined V29 consensus (V22->V28b evidence incorporated).

This script does NOT repeat the expensive lattice search.  It reconstructs the
accepted fit for every NV, exports a detailed per-NV table, and writes one
multi-page PDF per field with summary pages plus one detailed page per NV.

IMPORTANT: 49G and 52G are different physical samples and must not be matched
by nv_index.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import lombscargle

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14
import sc_c13_spin_echo_v28_nonlinear_eSEEM_52G as v28

ROOT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo")
OUTROOT=ROOT/r"spin_echo_final_full_analysis\2026_09"
CFG="smax30_tol12_pool8_sub1_topO2_topG4_cap40"
V22ROOT=ROOT/r"c13_spin_echo_v22_full_physical_rerank\2026_09"
V24=ROOT/r"spin_echo_v24_targeted_heavy_52G\2026_09"
V25=ROOT/r"spin_echo_v25_background_ablation_52G\2026_09"
V26B=ROOT/r"spin_echo_v26b_targeted_n4_52G\2026_09"
V29=ROOT/r"spin_echo_v29_final_consensus_52G\2026_09"

SAMPLES={"49G":"Johnson","52G":"QNami"}


def canon(x):
    try:
        a=ast.literal_eval(str(x))
        if isinstance(a,(tuple,list,np.ndarray)):
            return tuple(int(v) for v in a)
    except Exception:
        pass
    return ()


def parse_nv_arg(text,nmax):
    if not text:
        return list(range(nmax))
    out=[]
    for token in str(text).split(","):
        token=token.strip()
        if not token:
            continue
        if "-" in token:
            a,b=map(int,token.split("-",1))
            out.extend(range(a,b+1))
        else:
            out.append(int(token))
    return sorted({x for x in out if 0<=x<nmax})
def load_measurements(field):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck)
    cat=v6.load_catalog(base)

    # Names are read from the original field-specific dataset only for labels.
    rawmod=importlib.import_module(
        f"analysis.spin_echo_work.sc_c13_spin_echo_old_protocol_ranked_{field}"
    )
    try:
        _,nv_list,_,_,_,_=rawmod.load_data()
        names=[str(nv.name) for nv in nv_list]
    except Exception:
        names=[f"nv{i}" for i in range(Y.shape[0])]
    if len(names)!=Y.shape[0]:
        names=[f"nv{i}" for i in range(Y.shape[0])]
    return base,np.asarray(t,float),np.asarray(Y,float),np.asarray(E,float),cat,names


def catalog_sites(cat,orientation,site_ids):
    ori=tuple(int(v) for v in canon(orientation))
    c=cat.copy()
    if "ori" not in c:
        c["ori"]=c.orientation.map(canon)
    rows=[]
    for sid in site_ids:
        q=c[(c.site_id.astype(int)==int(sid)) & (c.ori.map(tuple)==ori)]
        if q.empty:
            q=c[c.site_id.astype(int)==int(sid)]
        if q.empty:
            raise KeyError(f"site {sid} not found for orientation {ori}")
        r=q.iloc[0]
        rows.append(dict(
            site_id=int(sid),f0_kHz=float(r.f0_kHz),f1_kHz=float(r.f1_kHz),
            kappa=float(r.kappa),distance_A=float(r.distance_A),
        ))
    return rows
def load_sources(field):
    v22=V22ROOT/field/CFG
    src=dict(
        w22=pd.read_csv(v22/"v22_winners.csv").set_index("nv_index"),
        q22=pd.read_csv(v22/"v22_qc_summary.csv").set_index("nv_index"),
        c22=pd.read_csv(v22/"v22_candidate_fits.csv.gz"),
    )
    if field=="52G":
        src.update(
            final=pd.read_csv(V29/"v29_final_consensus_212NV.csv").set_index("nv_index"),
            w24=pd.read_csv(V24/"v22_winners.csv").set_index("nv_index"),
            q24=pd.read_csv(V24/"v22_qc_summary.csv").set_index("nv_index"),
            c24=pd.read_csv(V24/"v22_candidate_fits.csv.gz"),
            v25=pd.read_csv(V25/"v25_background_ablation.csv").set_index("nv_index"),
            n4=pd.read_csv(V26B/"v26b_targeted_n4_best.csv").set_index("nv_index"),
            n4all=pd.read_csv(V26B/"v26b_targeted_n4_candidates.csv"),
        )
    return src


def provisional_class_49(row,qc):
    flags=set(str(qc.qc_flags).split(";")) if pd.notna(qc.qc_flags) else set()
    if float(row.red_chi2)>3:
        return "D_model_inadequate"
    if int(row.model_order)==0:
        return "E_no_resolved_C13"
    if flags & {"high_scale_gt10","scale_bound","line_amp_gt2contrast"}:
        return "B_amplitude_limited"
    if float(qc.delta_bic_to_second)<2:
        return "C_assignment_ambiguous"
    return "A_robust_V22"
def base_row_52(nv,src):
    if nv in src["w24"].index:
        return src["w24"].loc[nv],src["q24"].loc[nv],"V24_targeted_heavy"
    return src["w22"].loc[nv],src["q22"].loc[nv],"V22_full"


def reconstruct(field,nv,base,t,Y,E,cat,src):
    if field=="49G":
        row=src["w22"].loc[nv]; qc=src["q22"].loc[nv]
        sites=v14.sites_from_record(row)
        theta=np.asarray(json.loads(str(row.theta_json)),float)
        pred=lambda x:v6.v6_model(base,x,theta,sites)
        return dict(
            row=row,qc=qc,sites=sites,theta=theta,predict=pred,
            model_form="additive_shared",background="beta2_taper0",
            source="V22_full",order=int(row.model_order),site_key=canon(row.site_key),
            orientation=canon(row.orientation),bic=float(row.bic),
            redchi2=float(row.red_chi2),
            scale=float(row.visibility_scale) if pd.notna(row.visibility_scale) else np.nan,
            margin=float(qc.delta_bic_to_second),
            confidence=provisional_class_49(row,qc),
            flags=str(qc.qc_flags) if pd.notna(qc.qc_flags) else "",
            note="49G V22 result; later QNami-only refinements not applied",
        )

    fin=src["final"].loc[nv]
    row,qc,base_source=base_row_52(nv,src)
    source=str(fin.final_source)
    orientation=canon(fin.orientation)
    note=str(fin.special_note) if pd.notna(fin.special_note) else ""
    if source=="V26b_targeted_N4":
        q=src["n4"].loc[nv]
        ids=list(canon(q.incumbent_site_key))+[int(q.added_site_id)]
        sites=catalog_sites(cat,orientation,ids)
        theta=np.asarray(json.loads(str(q.theta_json)),float)
        pred=lambda x:v6.v6_model(base,x,theta,sites)
    elif source=="V25_background":
        sites=v14.sites_from_record(row)
        bg=str(fin.final_background)
        q=src["v25"].loc[nv]
        theta=np.asarray(json.loads(str(q[f"{bg}_theta_json"])),float)
        pred=lambda x:v6.v6_model(base,x,theta,sites)
    elif source=="V28_cross_term":
        sites=v14.sites_from_record(row)
        seed=np.asarray(json.loads(str(row.theta_json)),float)
        fit=v28.fit_cross(base,t,Y[nv],E[nv],sites,seed)
        if fit is None:
            raise RuntimeError(f"NV{nv}: could not reconstruct V28 cross model")
        theta=np.asarray(fit["theta"],float)
        pred=lambda x:v28.cross_model(base,x,theta,sites)
    else:
        sites=v14.sites_from_record(row)
        theta=np.asarray(json.loads(str(row.theta_json)),float)
        pred=lambda x:v6.v6_model(base,x,theta,sites)

    return dict(
        row=row,qc=qc,sites=sites,theta=theta,predict=pred,
        model_form=str(fin.final_model_form),background=str(fin.final_background),
        source=source,order=int(fin.final_model_order),
        site_key=canon(fin.final_site_key),orientation=orientation,
        bic=float(fin.final_bic),redchi2=float(fin.final_redchi2),
        scale=float(fin.final_scale) if pd.notna(fin.final_scale) else np.nan,
        margin=float(fin.candidate_bic_margin),
        confidence=str(fin.confidence_class),
        flags=str(fin.original_qc_flags) if pd.notna(fin.original_qc_flags) else "",
        note=note,
    )
def candidate_rows(field,nv,src,final_bic):
    if field=="52G" and nv in src["n4"].index and bool(src["n4"].loc[nv].n4_strongly_supported):
        q=src["n4all"][src["n4all"].nv_index==nv].copy().sort_values("n4_bic")
        if len(q):
            best=float(q.n4_bic.min()); out=[]
            for _,r in q.head(5).iterrows():
                key=tuple(list(canon(r.incumbent_site_key))+[int(r.added_site_id)])
                out.append((4,str(key),float(r.n4_bic-best),float(r.n4_redchi2)))
            return out

    c=src["c24"] if field=="52G" and nv in src["w24"].index else src["c22"]
    q=c[c.nv_index==nv].copy()
    if q.empty:
        return []
    if "site_key_canonical" in q:
        q=q.sort_values("bic").drop_duplicates(["model_order","site_key_canonical"])
    else:
        q=q.sort_values("bic").drop_duplicates(["model_order","site_key"])
    best=float(q.bic.min())
    return [
        (int(r.model_order),str(canon(r.site_key)),float(r.bic-best),float(r.red_chi2))
        for _,r in q.head(5).iterrows()
    ]


def residual_periodogram(t,z):
    tt=np.asarray(t,float); zz=np.asarray(z,float)-np.nanmean(z)
    dt=np.diff(np.sort(np.unique(tt)))
    med=float(np.nanmedian(dt)) if len(dt) else 1.0
    fmax=min(2.0,0.5/max(med,1e-6))
    f=np.linspace(0.002,max(0.01,fmax),1200) # cycles/us = MHz
    p=lombscargle(tt,zz,2*np.pi*f,normalize=True)
    return f,p
def build_summary_row(field,nv,name,fit):
    out=dict(
        field=field,sample=SAMPLES[field],nv_index=int(nv),nv_name=name,
        source=fit["source"],model_form=fit["model_form"],
        background=fit["background"],model_order=fit["order"],
        site_key=str(fit["site_key"]),orientation=str(fit["orientation"]),
        bic=fit["bic"],red_chi2=fit["redchi2"],visibility_scale=fit["scale"],
        candidate_bic_margin=fit["margin"],confidence_class=fit["confidence"],
        qc_flags=fit["flags"],note=fit["note"],
    )
    for j,s in enumerate(fit["sites"],1):
        out[f"c13_{j}_site_id"]=int(s["site_id"])
        out[f"c13_{j}_f0_kHz"]=float(s["f0_kHz"])
        out[f"c13_{j}_f1_kHz"]=float(s["f1_kHz"])
        out[f"c13_{j}_kappa"]=float(s["kappa"])
        out[f"c13_{j}_distance_A"]=float(s.get("distance_A",np.nan))
    return out


def plot_nv_page(pdf,field,nv,name,t,y,e,fit,top):
    dense=np.linspace(float(np.min(t)),float(np.max(t)),2200)
    pred=np.asarray(fit["predict"](t),float)
    pdense=np.asarray(fit["predict"](dense),float)
    z=(np.asarray(y,float)-pred)/np.maximum(np.abs(e),1e-9)
    f,p=residual_periodogram(t,z)

    fig,ax=plt.subplots(2,2,figsize=(11,8.5))
    ax[0,0].errorbar(t,y,yerr=e,fmt=".",ms=3,lw=.6,alpha=.75,label="data")
    ax[0,0].plot(dense,pdense,lw=1.3,label="final fit")
    ax[0,0].set(xlabel="total evolution time (us)",ylabel="normalized signal",
                title=f"{field} NV{nv}: {name}")
    ax[0,0].legend(fontsize=8)

    ax[1,0].axhline(0,lw=.8)
    ax[1,0].plot(t,z,".",ms=4)
    ax[1,0].set(xlabel="total evolution time (us)",ylabel="normalized residual",
                title=f"Residuals | RMS={np.sqrt(np.mean(z*z)):.2f}")
    ax[0,1].plot(1000*f,p,lw=1)
    ax[0,1].set(xlabel="residual frequency (kHz)",ylabel="Lomb-Scargle power",
                title="Residual spectrum")

    ax[1,1].axis("off")
    lines=[
        f"sample: {SAMPLES[field]}",
        f"source: {fit['source']}",
        f"model: {fit['model_form']} | background: {fit['background']}",
        f"N = {fit['order']} | sites = {fit['site_key']}",
        f"orientation = {fit['orientation']}",
        f"red chi2 = {fit['redchi2']:.3f} | BIC = {fit['bic']:.2f}",
        f"scale = {fit['scale']:.3f}" if np.isfinite(fit["scale"]) else "scale = n/a",
        f"candidate BIC margin = {fit['margin']:.3f}" if np.isfinite(fit["margin"]) else "candidate BIC margin = inf",
        f"class = {fit['confidence']}",
    ]
    if fit["flags"]:
        lines.append("QC: "+fit["flags"])
    if fit["note"] and fit["note"]!="nan":
        lines.append("note: "+fit["note"])
    lines.append("")
    lines.append("13C site details:")
    for s in fit["sites"]:
        lines.append(
            f"  {s['site_id']}: f+= {s['f0_kHz']:.2f} kHz, "
            f"f-= {s['f1_kHz']:.2f} kHz, k={s['kappa']:.3f}, "
            f"r={float(s.get('distance_A',np.nan)):.2f} A"
        )
    lines.append("")
    lines.append("Top discrete candidates (source search):")
    for n,key,db,red in top:
        lines.append(f"  N{n} {key}: dBIC={db:.2f}, redchi2={red:.2f}")
    ax[1,1].text(0,1,"\n".join(lines),va="top",ha="left",fontsize=8.2,
                 family="monospace",transform=ax[1,1].transAxes)
    fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
def summary_pages(pdf,field,df):
    fig=plt.figure(figsize=(11,8.5)); fig.patch.set_facecolor("white")
    plt.axis("off")
    sample=SAMPLES[field]
    text=[
        f"FULL SPIN-ECHO ANALYSIS — {field} — {sample}",
        "",
        f"NV count: {len(df)}",
        "49G and 52G are different physical samples; do not match by NV index.",
        "",
        "49G source: V22 relaxed-scale + local-substitution production rerank.",
        "52G source: V29 QNami consensus incorporating V24-V28b diagnostics.",
        "",
        "Each following page contains:",
        "  measured trace + reconstructed accepted model",
        "  normalized residuals",
        "  residual Lomb-Scargle spectrum",
        "  carbon-site frequencies/kappa/distance and top discrete competitors",
    ]
    plt.text(.06,.94,"\n".join(text),va="top",fontsize=14)
    pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(2,2,figsize=(11,8.5))
    oc=df.model_order.value_counts().sort_index()
    ax[0,0].bar(oc.index.astype(str),oc.values)
    ax[0,0].set(title="Resolved 13C model order",xlabel="N",ylabel="NV count")
    ax[0,1].hist(df.red_chi2,bins=35)
    ax[0,1].axvline(2,ls="--",lw=.8); ax[0,1].axvline(3,ls="--",lw=.8)
    ax[0,1].set(title="Reduced chi-square",xlabel="red chi2",ylabel="NV count")
    cc=df.confidence_class.value_counts()
    ax[1,0].bar(np.arange(len(cc)),cc.values)
    ax[1,0].set_xticks(np.arange(len(cc)),cc.index,rotation=28,ha="right")
    ax[1,0].set(title="Confidence/QC classification",ylabel="NV count")
    vals=df.visibility_scale.replace([np.inf,-np.inf],np.nan).dropna()
    ax[1,1].hist(vals,bins=30)
    ax[1,1].set(title="Shared visibility scale",xlabel="s_NV",ylabel="NV count")
    fig.suptitle(f"{field} {SAMPLES[field]} population summary")
    fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
def run_field(field,args):
    base,t,Y,E,cat,names=load_measurements(field)
    src=load_sources(field)
    ids=parse_nv_arg(args.nv,Y.shape[0])
    outdir=Path(args.output_dir)/field
    outdir.mkdir(parents=True,exist_ok=True)
    sample=SAMPLES[field]
    pdfpath=outdir/f"spin_echo_full_analysis_{field}_{sample}.pdf"

    records=[]
    with PdfPages(pdfpath) as pdf:
        # Build first so summary pages use exactly the requested subset.
        built=[]
        for nv in ids:
            fit=reconstruct(field,nv,base,t,Y,E,cat,src)
            top=candidate_rows(field,nv,src,fit["bic"])
            rec=build_summary_row(field,nv,names[nv],fit)
            records.append(rec); built.append((nv,fit,top))
        df=pd.DataFrame(records)
        summary_pages(pdf,field,df)
        for nv,fit,top in built:
            plot_nv_page(pdf,field,nv,names[nv],t,Y[nv],E[nv],fit,top)

    df=pd.DataFrame(records)
    df.to_csv(outdir/f"spin_echo_full_analysis_{field}_{sample}.csv",index=False)
    meta=dict(
        field=field,sample=sample,n_nv=int(len(df)),
        pdf=str(pdfpath),
        source=("V29 consensus" if field=="52G" else "V22 production"),
        warning="49G Johnson and 52G QNami are different physical samples.",
        model_order_counts={str(k):int(v) for k,v in df.model_order.value_counts().sort_index().items()},
        confidence_counts={str(k):int(v) for k,v in df.confidence_class.value_counts().items()},
    )
    with open(outdir/f"spin_echo_full_analysis_{field}_{sample}_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print(f"\n{field} FULL ANALYSIS COMPLETE")
    print("sample:",sample,"NVs:",len(df))
    print("orders:",meta["model_order_counts"])
    print("classes:",meta["confidence_counts"])
    print("PDF:",pdfpath)
    return df
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=("49G","52G","both"),default="both")
    ap.add_argument("--nv",default=None,
                    help="Optional subset, e.g. 0,1,5-10. Applied independently per field.")
    ap.add_argument("--output-dir",default=str(OUTROOT))
    args=ap.parse_args()

    fields=["49G","52G"] if args.field=="both" else [args.field]
    for field in fields:
        run_field(field,args)

    if len(fields)==2:
        print("\nIMPORTANT: reports are independent.")
        print("49G = Johnson sample; 52G = QNami sample.")
        print("Do not compare NV indices across fields.")


if __name__=="__main__":
    main()
