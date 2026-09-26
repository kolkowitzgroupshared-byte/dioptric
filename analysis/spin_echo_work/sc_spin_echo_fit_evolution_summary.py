# -*- coding: utf-8 -*-
"""Comprehensive post-processing of multi-13C spin-echo fitting evolution.

Discovers canonical full V1/V2/V3/V4/V5 results (and optionally V6),
builds standardized per-NV and per-carbon tables, and writes a single
multi-page PDF summarizing how population-level conclusions evolve.

This script never refits data.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOTS = {
    "49G": Path(r"G:\nvdata\pc_NVOffice\branch_master\sc_spin_echo_old_protocol_ranked_49G\2026_09"),
    "52G": Path(r"G:\nvdata\pc_NVOffice\branch_master\sc_spin_echo_physics_fit_52G_nv_pillar_array\2026_09"),
}
ORDER = ["V1", "V2", "V3", "V4", "V5", "V6"]
POOR_REDCHI = 2.5
STRONG_DBIC = 6.0
FREQ_BANDS_KHZ = {"49G": (5.643, 1331.522), "52G": (6.030, 1331.522)}

def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.18)

def page_title(fig, title, subtitle=""):
    fig.suptitle(title, fontsize=17, y=0.985)
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=9)

def no_subset_quick(p: Path) -> bool:
    n = p.name.lower()
    return "_subset_" not in n and "_quick" not in n

def choose_one(paths):
    paths = [p for p in paths if no_subset_quick(p)]
    if not paths:
        return None
    paths.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return paths[-1]

def discover_versions(root: Path, include_v6=False):
    found = {}
    patterns = {
        "V1": "*oldproto_multic13_ranked_maxC3_best_by_bic.csv",
        "V2": "*v2_bicprimary_t2window_maxC3_best_by_bic_evidence.csv",
        "V3": "*v3_top6_exhaustive_bicprimary_t2window_maxC3_best_by_bic_evidence.csv",
        "V4": "*v4_exhaustive_alignedseeds_bicprimary_t2window_maxC3_topK*_best_by_bic_evidence.csv",
        "V5": "*v5_physical_family_search_snr0p5_tol12p0kHz_topS15_perF3_oriassigned_maxC3_best_by_bic_evidence.csv",
    }
    if include_v6:
        patterns["V6"] = "*v6_physamp_snr0p5_tol12p0kHz_smax3p0_topS15_perF3_oriassigned_maxC3_best_by_bic_evidence.csv"
    for label, pat in patterns.items():
        p = choose_one(list(root.glob(pat)))
        if p is not None:
            found[label] = p
    return found

def safe_num(s, col, default=np.nan):
    if col not in s:
        return default
    try:
        v = float(s[col])
        return v if np.isfinite(v) else default
    except Exception:
        return default

def parse_family_size(x):
    try:
        v = ast.literal_eval(str(x))
        if isinstance(v, (list, tuple)):
            return len(v)
    except Exception:
        pass
    return np.nan

def standardize_best(df, field, version, source):
    x = df.copy()
    x["field"] = field
    x["version"] = version
    x["source_file"] = str(source)
    for col in ["conservative_order", "delta_bic_other_order",
                "delta_bic_same_order_site", "delta_bic_runner_up"]:
        if col not in x:
            x[col] = np.nan
    if "T2_bound_hit" not in x:
        x["T2_bound_hit"] = False
    return x

def components_from_best(df):
    rows = []
    for _, r in df.iterrows():
        order = int(r.get("model_order", 0))
        for j in range(1, order + 1):
            sid = r.get(f"c13_{j}_site_id", np.nan)
            if pd.isna(sid):
                continue
            f0 = safe_num(r, f"c13_{j}_f0_kHz")
            f1 = safe_num(r, f"c13_{j}_f1_kHz")
            flo, fhi = min(f0, f1), max(f0, f1)
            kappa = safe_num(r, f"c13_{j}_kappa")
            contrast = safe_num(r, "contrast")
            amp = safe_num(r, f"c13_{j}_amp")
            expected = contrast*kappa/4 if np.isfinite(contrast) and np.isfinite(kappa) else np.nan
            ratio = abs(amp)/expected if np.isfinite(amp) and np.isfinite(expected) and expected > 0 else np.nan
            rows.append(dict(
                field=r["field"], version=r["version"], nv_index=int(r["nv_index"]),
                model_order=order, orientation=str(r.get("orientation", "")),
                component=j, site_id=int(sid),
                f_low_kHz=flo, f_high_kHz=fhi,
                f_center_kHz=0.5*(flo+fhi), splitting_kHz=fhi-flo,
                kappa=kappa, distance_A=safe_num(r, f"c13_{j}_distance_A"),
                x_A=safe_num(r, f"c13_{j}_x_A"),
                y_A=safe_num(r, f"c13_{j}_y_A"),
                z_A=safe_num(r, f"c13_{j}_z_A"),
                A_par_kHz=safe_num(r, f"c13_{j}_A_par_kHz"),
                A_perp_kHz=safe_num(r, f"c13_{j}_A_perp_kHz"),
                theta_deg=safe_num(r, f"c13_{j}_theta_deg"),
                amp=amp, amp_expected_scale1=expected, amp_ratio=ratio,
                physical_snr=safe_num(r, f"c13_{j}_physical_snr"),
                matched_delta_chi2=safe_num(r, f"c13_{j}_matched_delta_chi2"),
                family_id=str(r.get(f"c13_{j}_family_id", "")),
                family_size=parse_family_size(r.get(f"c13_{j}_family_members", np.nan)),
                red_chi2=safe_num(r, "red_chi2"),
                T2_us=safe_num(r, "T2_us"),
            ))
    return pd.DataFrame(rows)

def version_summary(best):
    rows = []
    for (field, version), g in best.groupby(["field", "version"], sort=False):
        orders = g["model_order"].value_counts().to_dict()
        db = pd.to_numeric(g.delta_bic_other_order, errors="coerce")
        rows.append(dict(
            field=field, version=version, n_nv=len(g),
            n0=int(orders.get(0,0)), n1=int(orders.get(1,0)),
            n2=int(orders.get(2,0)), n3=int(orders.get(3,0)),
            frac_nonzero=float((g.model_order>0).mean()),
            median_redchi=float(g.red_chi2.median()),
            frac_redchi_lt2=float((g.red_chi2<2).mean()),
            frac_redchi_gt2p5=float((g.red_chi2>POOR_REDCHI).mean()),
            median_T2_us=float(g.T2_us.median()) if "T2_us" in g else np.nan,
            frac_T2_bound=float(pd.to_numeric(g.T2_bound_hit, errors="coerce").fillna(False).astype(bool).mean()),
            median_dbic_other=float(db.median()),
            frac_dbic_other_ge6=float((db>=STRONG_DBIC).mean()),
        ))
    return pd.DataFrame(rows)

def adjacent_transition_table(best_field):
    out = []
    versions = [v for v in ORDER if v in set(best_field.version)]
    maps = {v:best_field[best_field.version==v].set_index("nv_index") for v in versions}
    for va, vb in zip(versions[:-1], versions[1:]):
        common = maps[va].index.intersection(maps[vb].index)
        if not len(common):
            continue
        a, b = maps[va].loc[common], maps[vb].loc[common]
        changed = a.model_order.to_numpy() != b.model_order.to_numpy()
        out.append(dict(
            from_version=va, to_version=vb, n_common=len(common),
            n_changed=int(changed.sum()), frac_changed=float(changed.mean()),
            mean_order_from=float(a.model_order.mean()),
            mean_order_to=float(b.model_order.mean()),
            median_redchi_from=float(a.red_chi2.median()),
            median_redchi_to=float(b.red_chi2.median()),
        ))
    return pd.DataFrame(out)

def add_summary_page(pdf, field, best, comp, summ, sources):
    versions = [v for v in ORDER if v in set(best.version)]
    fig = plt.figure(figsize=(11,8.5))
    page_title(fig, f"{field}: fit-evolution overview",
               "Canonical full runs only; subset/quick outputs excluded")
    ax = fig.add_axes([.05,.08,.9,.80]); ax.axis("off")
    lines = ["Versions included: "+", ".join(versions), ""]
    for v in versions:
        s=summ[(summ.field==field)&(summ.version==v)].iloc[0]
        lines.append(
            f"{v:>3s}  NV={int(s.n_nv):3d}  orders 0/1/2/3="
            f"{int(s.n0):3d}/{int(s.n1):3d}/{int(s.n2):3d}/{int(s.n3):3d}  "
            f"nonzero={100*s.frac_nonzero:5.1f}%  median chi2r={s.median_redchi:5.3f}  "
            f"poor>{POOR_REDCHI:g}={100*s.frac_redchi_gt2p5:4.1f}%"
        )
    latest=versions[-1]
    gl=best[best.version==latest]; cl=comp[comp.version==latest]
    lines += ["", f"Latest included version: {latest}",
              f"Selected C13 components: {len(cl)} across {(gl.model_order>0).sum()} NVs"]
    if len(cl):
        lines.append(
            f"Frequency center median={cl.f_center_kHz.median():.1f} kHz; "
            f"splitting median={cl.splitting_kHz.median():.1f} kHz; "
            f"kappa median={cl.kappa.median():.3f}; "
            f"distance median={cl.distance_A.median():.2f} A")
    lines += ["", "Source files:"]
    lines += [f"  {v}: {sources[v].name}" for v in versions]
    ax.text(.01,.99,"\n".join(lines),va="top",family="monospace",fontsize=9)
    pdf.savefig(fig); plt.close(fig)

def add_order_pages(pdf, field, best):
    versions=[v for v in ORDER if v in set(best.version)]
    fig,axs=plt.subplots(2,1,figsize=(11,8.5))
    page_title(fig,f"{field}: model-order evolution")
    counts=pd.crosstab(best.version,best.model_order).reindex(versions).fillna(0)
    bottom=np.zeros(len(versions))
    for order in [0,1,2,3]:
        vals=counts.get(order,pd.Series(0,index=versions)).to_numpy()
        axs[0].bar(versions,vals,bottom=bottom,label=f"N={order}"); bottom+=vals
    axs[0].set_ylabel("NV count");axs[0].legend(ncol=4);clean_axes(axs[0])
    frac=counts.div(counts.sum(axis=1),axis=0); bottom=np.zeros(len(versions))
    for order in [0,1,2,3]:
        vals=frac.get(order,pd.Series(0,index=versions)).to_numpy()
        axs[1].bar(versions,vals,bottom=bottom,label=f"N={order}");bottom+=vals
    axs[1].set_ylabel("Fraction");axs[1].set_ylim(0,1);axs[1].set_xlabel("Analysis version");clean_axes(axs[1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_fit_quality_page(pdf, field, best):
    versions=[v for v in ORDER if v in set(best.version)]
    fig,axs=plt.subplots(2,2,figsize=(11,8.5))
    page_title(fig,f"{field}: fit quality and evidence")
    data=[best[best.version==v].red_chi2.dropna().to_numpy() for v in versions]
    axs[0,0].boxplot(data,tick_labels=versions,showfliers=False)
    axs[0,0].axhline(1,ls="--",lw=1);axs[0,0].axhline(POOR_REDCHI,ls=":",lw=1)
    axs[0,0].set_ylabel("Reduced chi-square");clean_axes(axs[0,0])
    for v in versions:
        z=np.sort(best[best.version==v].red_chi2.dropna().to_numpy())
        if len(z):axs[0,1].plot(z,np.arange(1,len(z)+1)/len(z),label=v)
    axs[0,1].set_xlabel("Reduced chi-square");axs[0,1].set_ylabel("CDF");axs[0,1].legend();clean_axes(axs[0,1])
    for v in versions:
        z=pd.to_numeric(best[best.version==v].delta_bic_other_order,errors="coerce").dropna()
        if len(z):axs[1,0].hist(np.clip(z,-20,50),bins=35,histtype="step",density=True,label=v)
    for x,ls in [(2,":"),(6,"--"),(10,"-.")]:axs[1,0].axvline(x,ls=ls,lw=1)
    axs[1,0].set_xlabel("Delta BIC to competing model order");axs[1,0].set_ylabel("Density");axs[1,0].legend();clean_axes(axs[1,0])
    for v in versions:
        z=best[best.version==v].T2_us.dropna().to_numpy()
        if len(z):axs[1,1].hist(z,bins=30,histtype="step",density=True,label=v)
    axs[1,1].set_xlabel("T2 (us)");axs[1,1].set_ylabel("Density");axs[1,1].legend();clean_axes(axs[1,1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_nv_evolution_page(pdf, field, best):
    versions=[v for v in ORDER if v in set(best.version)]
    piv=best.pivot_table(index="version",columns="nv_index",values="model_order",aggfunc="first").reindex(versions)
    fig,axs=plt.subplots(2,1,figsize=(11,8.5),gridspec_kw={"height_ratios":[2.2,1]})
    page_title(fig,f"{field}: per-NV evolution")
    im=axs[0].imshow(piv.to_numpy(),aspect="auto",interpolation="nearest",vmin=0,vmax=3)
    axs[0].set_yticks(range(len(versions)));axs[0].set_yticklabels(versions)
    axs[0].set_xlabel("NV index (ordered columns)");axs[0].set_ylabel("Version")
    fig.colorbar(im,ax=axs[0],pad=.01,label="Best-BIC model order")
    tr=adjacent_transition_table(best)
    if len(tr):
        x=np.arange(len(tr));vals=100*tr.frac_changed
        axs[1].bar(x,vals);axs[1].set_xticks(x)
        axs[1].set_xticklabels([f"{a}->{b}" for a,b in zip(tr.from_version,tr.to_version)])
        axs[1].set_ylabel("NVs changing order (%)");clean_axes(axs[1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_frequency_pages(pdf, field, comp):
    versions=[v for v in ORDER if v in set(comp.version)]
    if not versions:return
    latest=versions[-1];cl=comp[comp.version==latest]
    fig,axs=plt.subplots(2,2,figsize=(11,8.5))
    page_title(fig,f"{field}: selected C13 frequency distributions")
    for v in versions:
        g=comp[comp.version==v]
        if len(g):
            axs[0,0].hist(g.f_center_kHz.dropna(),bins=40,histtype="step",density=True,label=v)
            axs[0,1].hist(g.splitting_kHz.dropna(),bins=40,histtype="step",density=True,label=v)
    axs[0,0].set_xlabel("Frequency center (kHz)");axs[0,0].set_ylabel("Density");axs[0,0].legend();clean_axes(axs[0,0])
    axs[0,1].set_xlabel("|f_high-f_low| (kHz)");axs[0,1].set_ylabel("Density");axs[0,1].legend();clean_axes(axs[0,1])
    if len(cl):
        axs[1,0].scatter(cl.f_low_kHz,cl.f_high_kHz,s=14,alpha=.65)
        m=np.nanmax([cl.f_low_kHz.max(),cl.f_high_kHz.max()])
        axs[1,0].plot([0,m],[0,m],ls=":",lw=1)
        axs[1,0].set_xlabel("f_low (kHz)");axs[1,0].set_ylabel("f_high (kHz)")
        axs[1,0].set_title(f"{latest} frequency-pair plane");clean_axes(axs[1,0])
        sc=axs[1,1].scatter(cl.f_center_kHz,cl.splitting_kHz,c=cl.kappa,s=18,alpha=.75)
        axs[1,1].set_xlabel("Frequency center (kHz)");axs[1,1].set_ylabel("Splitting (kHz)")
        axs[1,1].set_title(f"{latest}: color = kappa");fig.colorbar(sc,ax=axs[1,1],pad=.01,label="kappa");clean_axes(axs[1,1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_geometry_pages(pdf, field, comp):
    versions=[v for v in ORDER if v in set(comp.version)]
    if not versions:return
    latest=versions[-1];cl=comp[comp.version==latest].copy()
    fig,axs=plt.subplots(2,2,figsize=(11,8.5))
    page_title(fig,f"{field}: C13 coupling and lattice geometry")
    for v in versions:
        g=comp[comp.version==v]
        if len(g):axs[0,0].hist(g.distance_A.dropna(),bins=35,histtype="step",density=True,label=v)
    axs[0,0].set_xlabel("NV-C13 distance (A)");axs[0,0].set_ylabel("Density");axs[0,0].legend();clean_axes(axs[0,0])
    g=cl.dropna(subset=["distance_A","kappa"])
    if len(g):
        sc=axs[0,1].scatter(g.distance_A,g.kappa,c=g.f_center_kHz,s=18,alpha=.7)
        axs[0,1].set_xlabel("Distance (A)");axs[0,1].set_ylabel("kappa")
        fig.colorbar(sc,ax=axs[0,1],pad=.01,label="Frequency center (kHz)");clean_axes(axs[0,1])
    g=cl.dropna(subset=["A_par_kHz","A_perp_kHz"])
    if len(g):
        sc=axs[1,0].scatter(g.A_par_kHz,g.A_perp_kHz,c=g.kappa,s=18,alpha=.7)
        axs[1,0].set_xlabel("A_parallel (kHz)");axs[1,0].set_ylabel("A_perp (kHz)")
        fig.colorbar(sc,ax=axs[1,0],pad=.01,label="kappa");clean_axes(axs[1,0])
    g=cl.dropna(subset=["x_A","y_A"])
    if len(g):
        sc=axs[1,1].scatter(g.x_A,g.y_A,c=g.distance_A,s=18,alpha=.7)
        axs[1,1].axhline(0,lw=.5);axs[1,1].axvline(0,lw=.5)
        axs[1,1].set_aspect("equal",adjustable="datalim")
        axs[1,1].set_xlabel("x (A)");axs[1,1].set_ylabel("y (A)")
        axs[1,1].set_title(f"{latest} selected lattice positions")
        fig.colorbar(sc,ax=axs[1,1],pad=.01,label="Distance (A)");clean_axes(axs[1,1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,4.2));page_title(fig,f"{field}: spatial projections ({latest})")
    for ax,a,b in [(axs[0],"x_A","y_A"),(axs[1],"x_A","z_A"),(axs[2],"y_A","z_A")]:
        g=cl.dropna(subset=[a,b])
        if len(g):
            ax.scatter(g[a],g[b],s=15,alpha=.65);ax.axhline(0,lw=.5);ax.axvline(0,lw=.5)
            ax.set_aspect("equal",adjustable="datalim")
        ax.set_xlabel(a.replace("_A"," (A)"));ax.set_ylabel(b.replace("_A"," (A)"));clean_axes(ax)
    fig.tight_layout(rect=[0,0,1,.90]);pdf.savefig(fig);plt.close(fig)

def add_amplitude_page(pdf,field,comp):
    versions=[v for v in ORDER if v in set(comp.version)]
    valid=comp.dropna(subset=["amp","amp_expected_scale1","amp_ratio"]).copy()
    if valid.empty:return
    fig,axs=plt.subplots(2,2,figsize=(11,8.5));page_title(fig,f"{field}: amplitude physicality")
    for v in versions:
        g=valid[(valid.version==v)&(valid.amp_ratio>0)]
        if len(g):axs[0,0].hist(np.log10(g.amp_ratio.clip(1e-4,1e6)),bins=40,histtype="step",density=True,label=v)
    axs[0,0].axvline(0,ls="--",lw=1);axs[0,0].axvline(np.log10(3),ls=":",lw=1);axs[0,0].axvline(1,ls="-.",lw=1)
    axs[0,0].set_xlabel("log10(|amp|/(contrast*kappa/4))");axs[0,0].set_ylabel("Density");axs[0,0].legend();clean_axes(axs[0,0])
    for v in versions:
        g=valid[valid.version==v]
        if len(g):axs[0,1].scatter(g.amp_expected_scale1.abs(),g.amp.abs(),s=10,alpha=.35,label=v)
    lo=1e-4;hi=max(valid.amp.abs().max(),valid.amp_expected_scale1.abs().max(),1e-2)
    axs[0,1].plot([lo,hi],[lo,hi],ls="--",lw=1);axs[0,1].plot([lo,hi],[3*lo,3*hi],ls=":",lw=1)
    axs[0,1].set_xscale("log");axs[0,1].set_yscale("log")
    axs[0,1].set_xlabel("contrast*kappa/4");axs[0,1].set_ylabel("|fitted amplitude|")
    axs[0,1].legend(fontsize=7);clean_axes(axs[0,1])
    latest=versions[-1];g=valid[valid.version==latest]
    sc=axs[1,0].scatter(g.kappa,g.amp_ratio,c=g.distance_A,s=16,alpha=.65)
    axs[1,0].set_yscale("log");axs[1,0].set_xlabel("kappa");axs[1,0].set_ylabel("Amplitude ratio")
    fig.colorbar(sc,ax=axs[1,0],pad=.01,label="Distance (A)");clean_axes(axs[1,0])
    q=[]
    for v in versions:
        g=valid[valid.version==v]
        if len(g):q.append([v,np.nanmedian(g.amp_ratio),np.nanpercentile(g.amp_ratio,90),np.sum(g.amp_ratio>3),np.sum(g.amp_ratio>10)])
    axs[1,1].axis("off")
    txt="version   median   p90   >3x   >10x\n"+"\n".join(
        f"{v:>6s} {m:8.2f} {p:7.2f} {a:5d} {b:6d}" for v,m,p,a,b in q)
    axs[1,1].text(.02,.98,txt,va="top",family="monospace",fontsize=10)
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_physical_v5_page(pdf,field,comp):
    g=comp[comp.version=="V5"].copy()
    if g.empty or g.physical_snr.notna().sum()==0:return
    fig,axs=plt.subplots(2,2,figsize=(11,8.5));page_title(fig,f"{field}: V5 physical-screen diagnostics")
    x=g.dropna(subset=["physical_snr","matched_delta_chi2"])
    axs[0,0].scatter(x.physical_snr,x.matched_delta_chi2,s=14,alpha=.6)
    axs[0,0].set_xlabel("Physical SNR");axs[0,0].set_ylabel("Matched delta chi2");clean_axes(axs[0,0])
    axs[0,1].hist(g.physical_snr.dropna(),bins=35,histtype="stepfilled",alpha=.6)
    axs[0,1].axvline(.5,ls="--",lw=1);axs[0,1].set_xlabel("Physical SNR")
    axs[0,1].set_ylabel("Selected components");clean_axes(axs[0,1])
    if g.family_size.notna().any():
        vc=g.family_size.dropna().astype(int).value_counts().sort_index()
        axs[1,0].bar(vc.index,vc.values);axs[1,0].set_xlabel("Frequency-family size")
        axs[1,0].set_ylabel("Selected components");clean_axes(axs[1,0])
    x=g.dropna(subset=["f_center_kHz","physical_snr"])
    sc=axs[1,1].scatter(x.f_center_kHz,x.physical_snr,c=x.kappa,s=14,alpha=.65)
    axs[1,1].set_xlabel("Frequency center (kHz)");axs[1,1].set_ylabel("Physical SNR")
    fig.colorbar(sc,ax=axs[1,1],pad=.01,label="kappa");clean_axes(axs[1,1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_site_stability_page(pdf,field,comp):
    versions=[v for v in ORDER if v in set(comp.version)]
    if len(versions)<2:return
    fig,axs=plt.subplots(2,1,figsize=(11,8.5));page_title(fig,f"{field}: site-selection stability")
    top=[]
    for v in versions:
        vc=comp[comp.version==v].site_id.value_counts().head(15)
        for sid,n in vc.items():top.append((v,int(sid),int(n)))
    topdf=pd.DataFrame(top,columns=["version","site_id","count"])
    sites=topdf.groupby("site_id")["count"].sum().sort_values(ascending=False).head(20).index
    mat=pd.DataFrame(0,index=versions,columns=sites,dtype=float)
    for v in versions:
        vc=comp[comp.version==v].site_id.value_counts()
        for sid in sites:mat.loc[v,sid]=vc.get(sid,0)
    im=axs[0].imshow(mat.to_numpy(),aspect="auto",interpolation="nearest")
    axs[0].set_yticks(range(len(versions)));axs[0].set_yticklabels(versions)
    axs[0].set_xticks(range(len(sites)));axs[0].set_xticklabels([str(s) for s in sites],rotation=90,fontsize=7)
    axs[0].set_xlabel("Site ID (top recurrent)");axs[0].set_ylabel("Version")
    fig.colorbar(im,ax=axs[0],pad=.01,label="Selected count")
    stable=[]
    for va,vb in zip(versions[:-1],versions[1:]):
        a=comp[comp.version==va].groupby("nv_index").site_id.apply(set)
        b=comp[comp.version==vb].groupby("nv_index").site_id.apply(set)
        common=a.index.intersection(b.index);vals=[]
        for nv in common:
            u=a[nv]|b[nv];vals.append(len(a[nv]&b[nv])/len(u) if u else 1)
        stable.append((f"{va}->{vb}",np.mean(vals) if vals else np.nan))
    if stable:
        axs[1].bar([x[0] for x in stable],[x[1] for x in stable]);axs[1].set_ylim(0,1)
        axs[1].set_ylabel("Mean exact-site Jaccard");clean_axes(axs[1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)


def add_flag_page(pdf,field,best,comp):
    versions=[v for v in ORDER if v in set(best.version)]
    latest=versions[-1];g=best[best.version==latest].copy()
    amp=comp[comp.version==latest]
    ampflag=set(amp.loc[amp.amp_ratio>10,"nv_index"].astype(int)) if len(amp) else set()
    bad=[]
    for _,r in g.iterrows():
        flags=[]
        if r.red_chi2>POOR_REDCHI:flags.append(f"chi2r>{POOR_REDCHI:g}")
        if bool(r.get("T2_bound_hit",False)):flags.append("T2-bound")
        db=safe_num(r,"delta_bic_other_order")
        if np.isfinite(db) and db<2:flags.append("order-ambiguous")
        if int(r.nv_index) in ampflag:flags.append("amp>10x-phys")
        if flags:bad.append((int(r.nv_index),int(r.model_order),float(r.red_chi2),db,", ".join(flags)))
    bad=sorted(bad,key=lambda z:(-z[2],z[0]))
    fig=plt.figure(figsize=(11,8.5))
    page_title(fig,f"{field}: latest-version review flags",
               f"{latest}; diagnostic only, not exclusion rules")
    ax=fig.add_axes([.05,.07,.9,.84]);ax.axis("off")
    lines=[" NV   N   chi2r   dBIC(order)   flags","-"*78]
    for x in bad[:55]:
        lines.append(f"{x[0]:3d}  {x[1]:1d}   {x[2]:6.3f}   {x[3]:10.2f}   {x[4]}")
    if len(bad)>55:lines.append(f"... plus {len(bad)-55} additional flagged NVs")
    ax.text(.01,.99,"\n".join(lines),va="top",family="monospace",fontsize=8.2)
    pdf.savefig(fig);plt.close(fig)

def add_cross_field_pages(pdf,best,comp):
    if set(best.field)!={"49G","52G"}:return
    latest={}
    for fld in ["49G","52G"]:
        vv=[v for v in ORDER if v in set(best[best.field==fld].version)]
        latest[fld]=vv[-1]
    fig,axs=plt.subplots(2,2,figsize=(11,8.5))
    page_title(fig,"49 G vs 52 G: latest included population comparison",
               f"49G={latest['49G']}, 52G={latest['52G']}")
    for fld in ["49G","52G"]:
        g=best[(best.field==fld)&(best.version==latest[fld])]
        vc=g.model_order.value_counts().reindex([0,1,2,3],fill_value=0)/len(g)
        axs[0,0].plot([0,1,2,3],vc.values,marker="o",label=fld)
        axs[0,1].hist(g.red_chi2,bins=35,histtype="step",density=True,label=fld)
        c=comp[(comp.field==fld)&(comp.version==latest[fld])]
        axs[1,0].hist(c.f_center_kHz.dropna(),bins=40,histtype="step",density=True,label=fld)
        axs[1,1].hist(c.distance_A.dropna(),bins=35,histtype="step",density=True,label=fld)
    axs[0,0].set_xlabel("Model order");axs[0,0].set_ylabel("Fraction of NVs")
    axs[0,0].set_xticks([0,1,2,3]);axs[0,0].legend();clean_axes(axs[0,0])
    axs[0,1].set_xlabel("Reduced chi-square");axs[0,1].set_ylabel("Density")
    axs[0,1].legend();clean_axes(axs[0,1])
    axs[1,0].set_xlabel("Selected C13 frequency center (kHz)")
    axs[1,0].set_ylabel("Density");axs[1,0].legend();clean_axes(axs[1,0])
    axs[1,1].set_xlabel("Selected C13 distance (A)")
    axs[1,1].set_ylabel("Density");axs[1,1].legend();clean_axes(axs[1,1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_measurement_band_page(pdf, field, comp):
    versions=[v for v in ORDER if v in set(comp.version)]
    if not versions:return
    flo,fhi=FREQ_BANDS_KHZ[field]
    fig,axs=plt.subplots(2,2,figsize=(11,8.5))
    page_title(fig,f"{field}: measurement-band evolution",
               f"Directly sampled band: {flo:.2f}-{fhi:.1f} kHz; both f_low and f_high must lie inside")
    frac=[];counts=[]
    for v in versions:
        g=comp[comp.version==v].copy()
        if g.empty:
            frac.append(np.nan);counts.append((0,0));continue
        inside=(g.f_low_kHz>=flo)&(g.f_high_kHz<=fhi)
        frac.append(100*(~inside).mean())
        counts.append((int(inside.sum()),int((~inside).sum())))
        z=g.loc[inside,"f_center_kHz"].dropna()
        if len(z):axs[0,1].hist(z,bins=35,histtype="step",density=True,label=v)
    axs[0,0].bar(versions,frac)
    axs[0,0].set_ylabel("Selected C13 outside band (%)");clean_axes(axs[0,0])
    for i,(inn,out) in enumerate(counts):
        axs[0,0].text(i,frac[i]+1 if np.isfinite(frac[i]) else 0,f"{out}/{inn+out}",ha="center",fontsize=8)
    axs[0,1].set_xlim(0,fhi*1.03);axs[0,1].set_xlabel("In-band frequency center (kHz)")
    axs[0,1].set_ylabel("Density");axs[0,1].legend();clean_axes(axs[0,1])
    latest=versions[-1];g=comp[comp.version==latest].copy()
    inside=(g.f_low_kHz>=flo)&(g.f_high_kHz<=fhi)
    if len(g):
        axs[1,0].scatter(g.f_low_kHz,g.f_high_kHz,s=13,alpha=.6)
        axs[1,0].axvline(fhi,ls="--",lw=1);axs[1,0].axhline(fhi,ls="--",lw=1)
        axs[1,0].set_xlabel("f_low (kHz)");axs[1,0].set_ylabel("f_high (kHz)")
        axs[1,0].set_title(f"{latest}: full selected-frequency plane");clean_axes(axs[1,0])
    gi=g[inside]
    if len(gi):
        sc=axs[1,1].scatter(gi.f_center_kHz,gi.splitting_kHz,c=gi.kappa,s=16,alpha=.7)
        axs[1,1].set_xlabel("In-band frequency center (kHz)");axs[1,1].set_ylabel("Splitting (kHz)")
        fig.colorbar(sc,ax=axs[1,1],pad=.01,label="kappa");clean_axes(axs[1,1])
    fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

def add_transition_matrix_page(pdf, field, best):
    versions=[v for v in ORDER if v in set(best.version)]
    if len(versions)<2:return
    va,vb=versions[-2],versions[-1]
    a=best[best.version==va].set_index("nv_index")
    b=best[best.version==vb].set_index("nv_index")
    common=a.index.intersection(b.index)
    if not len(common):return
    mat=np.zeros((4,4),int)
    for nv in common:
        mat[int(a.loc[nv,"model_order"]),int(b.loc[nv,"model_order"])]+=1
    fig,axs=plt.subplots(1,2,figsize=(11,4.7))
    page_title(fig,f"{field}: model-order transition {va} -> {vb}")
    im=axs[0].imshow(mat,vmin=0)
    axs[0].set_xticks(range(4));axs[0].set_yticks(range(4))
    axs[0].set_xlabel(f"{vb} order");axs[0].set_ylabel(f"{va} order")
    for i in range(4):
        for j in range(4):
            axs[0].text(j,i,str(mat[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=axs[0],pad=.01,label="NV count")
    delta=b.loc[common,"model_order"].to_numpy()-a.loc[common,"model_order"].to_numpy()
    vals=pd.Series(delta).value_counts().sort_index()
    axs[1].bar(vals.index.astype(str),vals.values)
    axs[1].set_xlabel("Change in best-BIC order");axs[1].set_ylabel("NV count");clean_axes(axs[1])
    fig.tight_layout(rect=[0,0,1,.90]);pdf.savefig(fig);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--include-v6",action="store_true",
                    help="Include only a completed full V6 run if present.")
    ap.add_argument("--include-drift-nv",action="store_true",
                    help="Include 52G NV0 in population statistics (default: exclude drift-reference NV0).")
    ap.add_argument("--output",type=str,default=None)
    args=ap.parse_args()
    fields=["49G","52G"] if args.field=="both" else [args.field]
    best_blocks=[];comp_blocks=[];sources_by_field={}
    for fld in fields:
        sources=discover_versions(ROOTS[fld],include_v6=args.include_v6)
        if not sources:
            raise RuntimeError(f"No canonical full results found for {fld}")
        sources_by_field[fld]=sources
        print(f"\n{fld}:")
        for v in ORDER:
            if v not in sources:continue
            p=sources[v];d=pd.read_csv(p)
            if fld=="52G" and not args.include_drift_nv:
                d=d[d.nv_index!=0].copy()
            d=standardize_best(d,fld,v,p);best_blocks.append(d)
            c=components_from_best(d)
            if len(c):comp_blocks.append(c)
            print(f"  {v}: {p.name}  NV={len(d)} C13={len(c)}")
    best=pd.concat(best_blocks,ignore_index=True,sort=False)
    comp=pd.concat(comp_blocks,ignore_index=True,sort=False) if comp_blocks else pd.DataFrame()
    summ=version_summary(best)
    if args.output:
        out=Path(args.output)
    else:
        ym=datetime.now().strftime("%Y_%m")
        outdir=(Path(r"G:\nvdata\pc_NVOffice\branch_master")
                /"spin_echo_fit_evolution_summary"/ym)
        out=outdir/f"spin_echo_fit_evolution_summary_{args.field}.pdf"
    out.parent.mkdir(parents=True,exist_ok=True)
    stem=out.with_suffix("")
    summ.to_csv(Path(str(stem)+"_version_summary.csv"),index=False)
    best.to_csv(Path(str(stem)+"_best_long.csv"),index=False)
    comp.to_csv(Path(str(stem)+"_components_long.csv"),index=False)
    trans=[]
    for fld in fields:
        t=adjacent_transition_table(best[best.field==fld])
        if len(t):
            t.insert(0,"field",fld);trans.append(t)
    if trans:
        pd.concat(trans,ignore_index=True).to_csv(
            Path(str(stem)+"_transitions.csv"),index=False)
    with PdfPages(out) as pdf:
        for fld in fields:
            bf=best[best.field==fld]
            cf=comp[comp.field==fld] if len(comp) else comp
            add_summary_page(pdf,fld,bf,cf,summ,sources_by_field[fld])
            add_order_pages(pdf,fld,bf)
            add_fit_quality_page(pdf,fld,bf)
            add_nv_evolution_page(pdf,fld,bf)
            add_frequency_pages(pdf,fld,cf)
            add_measurement_band_page(pdf,fld,cf)
            add_transition_matrix_page(pdf,fld,bf)
            add_geometry_pages(pdf,fld,cf)
            add_amplitude_page(pdf,fld,cf)
            add_physical_v5_page(pdf,fld,cf)
            add_site_stability_page(pdf,fld,cf)
            add_flag_page(pdf,fld,bf,cf)
        add_cross_field_pages(pdf,best,comp)
    print("\nWROTE",out)
    print("WROTE",Path(str(stem)+"_version_summary.csv"))
    print("WROTE",Path(str(stem)+"_components_long.csv"))

if __name__=="__main__":
    main()

