"""V26 residual + uncertainty diagnostics for QNami 52 G spin echo.

Uses the best V25 background model for each V24 high-redchi2 NV, then:
  * standardized residual diagnostics and Lomb-Scargle spectra
  * catalog-pair matched residual screening for a possible additional 13C
  * cross-NV common-mode residual PCA/correlation
  * file-to-file reproducibility from all 9 raw source acquisitions

No lattice assignment is changed here.  This is a diagnostic/triage stage.
"""
from __future__ import annotations

import ast, json, sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks, lombscargle

from utils import data_manager as dm
from utils import widefield
import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14

V24=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v24_targeted_heavy_52G\2026_09")
V25=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v25_background_ablation_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v26_residual_error_diagnostics_52G\2026_09")
MODELS=("reduced","beta_free","taper_free","full_bg")


def best_background(row):
    vals={m:float(getattr(row,f"{m}_bic")) for m in MODELS}
    return min(vals,key=vals.get)


def parse_orientation(x):
    if isinstance(x,(tuple,list,np.ndarray)):
        return tuple(int(v) for v in x)
    try:
        a=ast.literal_eval(str(x))
        return tuple(int(v) for v in a)
    except Exception:
        return ()


def top_periodogram(t,r):
    lo,hi=v6.experimental_frequency_band_mhz(t)
    f=np.linspace(max(lo,0.002),hi,6000)
    z=np.asarray(r,float)-np.nanmean(r)
    p=lombscargle(np.asarray(t,float),z,2*np.pi*f,precenter=False,normalize=True)
    pk,_=find_peaks(p,distance=10)
    if len(pk)==0:
        pk=np.arange(len(p))
    order=pk[np.argsort(p[pk])[::-1]]
    order=order[:10]
    return f,p,f[order],p[order]


def catalog_residual_screen(base,cat,orientation,t,y,e,theta,sites):
    ori=parse_orientation(orientation)
    if not ori:
        return {}
    cc=cat[cat.ori.map(tuple)==tuple(ori)].copy()
    lo,hi=v6.experimental_frequency_band_mhz(t)
    cc=cc[
        cc.f0_kHz.between(1000*lo,1000*hi)
        & cc.f1_kHz.between(1000*lo,1000*hi)
    ]
    used={int(s["site_id"]) for s in sites}
    if used:
        cc=cc[~cc.site_id.astype(int).isin(used)]
    pred=v6.v6_model(base,t,theta,sites)
    ee=base.safe_err(e)
    rw=(np.asarray(y,float)-pred)/ee
    carrier=base.carrier_from_bg(t,np.asarray(theta,float)[:9])[2]
    tt=np.asarray(t,float)
    rows=[]
    for q in cc.itertuples():
        f0=float(q.f0_kHz)/1000.0
        f1=float(q.f1_kHz)/1000.0
        X=np.column_stack([
            carrier*np.cos(2*np.pi*f0*tt)/ee,
            carrier*np.sin(2*np.pi*f0*tt)/ee,
            carrier*np.cos(2*np.pi*f1*tt)/ee,
            carrier*np.sin(2*np.pi*f1*tt)/ee,
        ])
        # Exact weighted linear upper-bound screen for this frequency pair.
        try:
            coef,*_=np.linalg.lstsq(X,rw,rcond=None)
            rr=rw-X@coef
            gain=float(np.sum(rw*rw)-np.sum(rr*rr))
        except Exception:
            gain=np.nan
        rows.append((gain,int(q.site_id),f0,f1,float(q.kappa),float(q.distance_A)))
    rows=[x for x in rows if np.isfinite(x[0])]
    rows.sort(reverse=True,key=lambda z:z[0])
    if not rows:
        return {}
    g,site,f0,f1,kappa,dist=rows[0]
    return dict(
        residual_best_site=site,
        residual_pair_dchi2_upper=g,
        residual_site_f0_MHz=f0,
        residual_site_f1_MHz=f1,
        residual_site_kappa=kappa,
        residual_site_distance_A=dist,
    )


def load_file_level_curves(source_stems,nv_count,nstep):
    cache=OUT/"v26_file_level_curves.npz"
    if cache.exists():
        z=np.load(cache,allow_pickle=True)
        return z["norm"],z["ste"],list(z["stems"].astype(str))
    norms=[]; stes=[]; good=[]
    for stem in source_stems:
        print("file-level normalization:",stem,flush=True)
        raw=dm.get_raw_data(file_stem=str(stem),load_npz=True,use_cache=False)
        counts=np.asarray(raw["counts"])
        sig=np.asarray(counts[0],dtype=np.float32)
        ref=np.asarray(counts[1],dtype=np.float32)
        n,s=widefield.process_counts(raw["nv_list"],sig,ref,threshold=True)
        n=np.asarray(n,float); s=np.asarray(s,float)
        if n.shape!=(nv_count,nstep):
            raise ValueError(f"{stem}: unexpected normalized shape {n.shape}")
        norms.append(n); stes.append(s); good.append(str(stem))
    norm=np.stack(norms); ste=np.stack(stes)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,norm=norm,ste=ste,stems=np.asarray(good,dtype=object))
    return norm,ste,good


def file_scatter_stats(norm,ste,nvs):
    out=[]
    for nv in nvs:
        for step in range(norm.shape[2]):
            x=norm[:,nv,step]; s=np.maximum(np.abs(ste[:,nv,step]),1e-6)
            good=np.isfinite(x)&np.isfinite(s)&(s>0)
            if good.sum()<3:
                continue
            x=x[good]; s=s[good]; w=1/(s*s)
            mu=float(np.sum(w*x)/np.sum(w))
            chi=float(np.sum(((x-mu)/s)**2))
            dof=max(1,len(x)-1)
            out.append(dict(nv_index=nv,step=step,file_chi2=chi,
                            file_redchi2=chi/dof,birge_ratio=np.sqrt(chi/dof),
                            file_weighted_mean=mu,n_files=len(x)))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    v25=pd.read_csv(V25/"v25_background_ablation.csv")
    winners=pd.read_csv(V24/"v22_winners.csv").set_index("nv_index")
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck)
    cat=v6.load_catalog(base)
    assigned=v6.load_assigned_orientations("52G",Y.shape[0])
    records=[]; residual_rows=[]; spectra={}
    for q in v25.itertuples():
        nv=int(q.nv_index); win=winners.loc[nv]
        bg=best_background(q)
        theta=np.asarray(json.loads(getattr(q,f"{bg}_theta_json")),float)
        sites=v14.sites_from_record(win)
        pred=v6.v6_model(base,t,theta,sites)
        ee=base.safe_err(E[nv])
        resid=np.asarray(Y[nv],float)-pred
        z=resid/ee
        f,p,peaks,pp=top_periodogram(t,z)
        ori=parse_orientation(win.orientation)
        if not ori:
            ori=tuple(assigned.get(nv,()))
        screen=catalog_residual_screen(base,cat,ori,t,Y[nv],E[nv],theta,sites)
        pair_match=False
        if screen and len(peaks):
            tol=0.012
            pair_match=(
                np.min(np.abs(peaks-screen["residual_site_f0_MHz"]))<=tol
                and np.min(np.abs(peaks-screen["residual_site_f1_MHz"]))<=tol
            )
        rec=dict(
            nv_index=nv,best_background=bg,model_order=int(win.model_order),
            site_key=str(win.site_key),orientation=str(ori),
            redchi2=float(getattr(q,f"{bg}_redchi2")),
            residual_rms_sigma=float(np.sqrt(np.mean(z*z))),
            residual_max_abs_sigma=float(np.max(np.abs(z))),
            residual_lag1_corr=float(np.corrcoef(z[:-1],z[1:])[0,1]),
            periodogram_peak1_MHz=float(peaks[0]) if len(peaks) else np.nan,
            periodogram_peak1_power=float(pp[0]) if len(pp) else np.nan,
            periodogram_top_MHz=json.dumps([float(x) for x in peaks[:10]]),
            residual_pair_peak_match=bool(pair_match),
            **screen,
        )
        rec["n4_screen_support"]=bool(
            screen
            and screen.get("residual_pair_dchi2_upper",0)>=15.0
            and pair_match
        )
        records.append(rec)
        spectra[nv]=(f,p,peaks,pp,pred,z)
        for i,(tt,rr,zz) in enumerate(zip(t,resid,z)):
            residual_rows.append(dict(nv_index=nv,step=i,time_us=tt,
                                      residual=rr,residual_sigma=zz))
    summary=pd.DataFrame(records).sort_values("nv_index")
    residual_df=pd.DataFrame(residual_rows)
    summary.to_csv(OUT/"v26_residual_summary.csv",index=False)
    residual_df.to_csv(OUT/"v26_residuals_long.csv",index=False)

    # Cross-NV residual common-mode structure.
    nvs=summary.nv_index.astype(int).tolist()
    R=np.vstack([
        residual_df[residual_df.nv_index==nv].sort_values("step").residual_sigma
        for nv in nvs
    ]).astype(float)
    Rc=R-R.mean(axis=1,keepdims=True)
    U,S,Vt=np.linalg.svd(Rc,full_matrices=False)
    frac=(S*S)/np.sum(S*S)
    corr=np.corrcoef(Rc)
    off=corr[np.triu_indices_from(corr,k=1)]
    common=pd.DataFrame({
        "metric":["pc1_variance_fraction","pc2_variance_fraction",
                  "median_pairwise_residual_corr","mean_pairwise_residual_corr"],
        "value":[frac[0],frac[1] if len(frac)>1 else np.nan,
                 np.nanmedian(off),np.nanmean(off)],
    })
    common.to_csv(OUT/"v26_common_mode_summary.csv",index=False)
    pd.DataFrame({"time_us":t,"pc1":Vt[0],"mean_z":np.mean(R,axis=0),
                  "median_z":np.median(R,axis=0)}).to_csv(
        OUT/"v26_common_mode_trace.csv",index=False)

    # File-to-file uncertainty calibration.
    raw=dm.get_raw_data(
        file_stem="2026_09_21-15_37_34-qnami_spin_echo_combined_52G",
        load_npz=True,
    )
    norm,ste,stems=load_file_level_curves(
        raw["source_file_stems"],Y.shape[0],Y.shape[1]
    )
    fs=file_scatter_stats(norm,ste,nvs)
    fs.to_csv(OUT/"v26_file_scatter_by_step.csv",index=False)
    fsum=fs.groupby("nv_index").agg(
        median_birge=("birge_ratio","median"),
        mean_file_redchi2=("file_redchi2","mean"),
        frac_steps_birge_gt1p5=("birge_ratio",lambda x:float(np.mean(x>1.5))),
        frac_steps_birge_gt2=("birge_ratio",lambda x:float(np.mean(x>2.0))),
    ).reset_index()
    fsum.to_csv(OUT/"v26_file_scatter_summary.csv",index=False)
    summary=summary.merge(fsum,on="nv_index",how="left")
    summary.to_csv(OUT/"v26_residual_summary.csv",index=False)
    # Diagnostic PDF.
    with PdfPages(OUT/"v26_residual_diagnostics.pdf") as pdf:
        for rec in summary.itertuples():
            nv=int(rec.nv_index); f,p,peaks,pp,pred,z=spectra[nv]
            fig,ax=plt.subplots(2,2,figsize=(11,8.5))
            ax[0,0].errorbar(t,Y[nv],yerr=E[nv],fmt=".",ms=3,label="data")
            ax[0,0].plot(t,pred,lw=1.2,label="best V25 model")
            ax[0,0].set(title=f"NV {nv}: {rec.best_background} | {rec.site_key}",
                        xlabel="total evolution (us)",ylabel="normalized signal")
            ax[0,0].legend(fontsize=7)
            ax[0,1].axhline(0,lw=.8)
            ax[0,1].plot(t,z,".-",ms=3,lw=.7)
            ax[0,1].set(title=f"standardized residual | redchi2={rec.redchi2:.2f}",
                        xlabel="total evolution (us)",ylabel="residual / STE")
            ax[1,0].plot(f*1e3,p,lw=.9)
            if np.isfinite(getattr(rec,"residual_site_f0_MHz",np.nan)):
                ax[1,0].axvline(rec.residual_site_f0_MHz*1e3,ls="--",lw=.8)
                ax[1,0].axvline(rec.residual_site_f1_MHz*1e3,ls="--",lw=.8)
            ax[1,0].set(title="Residual Lomb-Scargle + best catalog pair",
                        xlabel="frequency (kHz)",ylabel="normalized power")
            ax[1,1].axis("off")
            txt=(
                f"best residual site: {getattr(rec,'residual_best_site',np.nan)}\n"
                f"pair dchi2 upper screen: {getattr(rec,'residual_pair_dchi2_upper',np.nan):.1f}\n"
                f"pair/periodogram match: {rec.residual_pair_peak_match}\n"
                f"N=4 screen support: {rec.n4_screen_support}\n"
                f"lag-1 residual corr: {rec.residual_lag1_corr:.2f}\n"
                f"median file Birge ratio: {rec.median_birge:.2f}\n"
                f"max |residual|: {rec.residual_max_abs_sigma:.1f} sigma"
            )
            ax[1,1].text(.02,.98,txt,va="top",family="monospace")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig,ax=plt.subplots(2,1,figsize=(11,8.5))
        ax[0].plot(t,Vt[0],label=f"PC1 ({100*frac[0]:.1f}% residual variance)")
        ax[0].plot(t,np.mean(R,axis=0),label="mean standardized residual",alpha=.8)
        ax[0].legend(); ax[0].set(xlabel="time (us)",title="Common-mode residual structure")
        im=ax[1].imshow(corr,aspect="auto",vmin=-1,vmax=1)
        ax[1].set(title="Residual correlation matrix",xlabel="NV index in bad-fit subset",
                  ylabel="NV index in bad-fit subset")
        fig.colorbar(im,ax=ax[1],label="correlation")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print("\nV26 COMPLETE")
    print("bad-fit NVs:",len(summary))
    print("N=4 residual-screen support:",summary[summary.n4_screen_support].nv_index.astype(int).tolist())
    print("PC1 residual variance fraction:",float(frac[0]))
    print("median pairwise residual correlation:",float(np.nanmedian(off)))
    print("median file Birge ratio across NVs:",float(summary.median_birge.median()))
    print("file Birge >1.5 median-NV count:",int((summary.median_birge>1.5).sum()))
    print("output:",OUT)


if __name__=="__main__":
    main()
