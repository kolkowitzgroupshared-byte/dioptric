"""V15: population dashboard for completed V14 beta=2, taper=0 rerank.

No fitting is performed.

For each field this script:
  * loads the original full V6 candidate table and the completed V14 rerank,
  * compares V6 -> V14 winner order/site transitions,
  * quantifies N3-vs-N2 BIC evidence,
  * diagnoses visibility-scale saturation and effective s*kappa,
  * summarizes remaining background-parameter/bound behavior,
  * reconstructs V6 and V14 population residuals from the measured data,
  * visualizes common residuals and a per-NV residual heatmap,
  * writes compact population/per-NV/component CSV tables and a PDF dashboard.

52G NV0 is retained in raw inputs but excluded from population statistics.
"""
from __future__ import annotations

import argparse, ast, json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v7_diagnostic as v7
import sc_spin_echo_v8_physical_bath_diagnostic as v8

V14_ROOT=Path(r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v14_beta2_taper0_rerank\2026_09")
OUT_ROOT=Path(r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v15_v14_population_dashboard\2026_09")
POOR_REDCHI=2.5
SCALE_BOUND=2.94


def discover_v14(field):
    pats=list(V14_ROOT.glob(f"*{field}*v14_beta2_taper0_candidate_fits.csv.gz"))
    if not pats:
        raise FileNotFoundError(f"No V14 candidate table for {field} in {V14_ROOT}")
    return max(pats,key=lambda p:p.stat().st_mtime)


def parse_site_key(x):
    try:
        z=ast.literal_eval(str(x))
        return tuple(int(v) for v in z) if isinstance(z,(tuple,list)) else ()
    except Exception:
        return ()


def row_sites(row):
    out=[]
    ori=()
    try: ori=ast.literal_eval(str(row.orientation))
    except Exception: pass
    for j in range(1,int(row.model_order)+1):
        out.append(dict(site_id=int(row[f"c13_{j}_site_id"]),
                        f0_kHz=float(row[f"c13_{j}_f0_kHz"]),
                        f1_kHz=float(row[f"c13_{j}_f1_kHz"]),
                        kappa=float(row[f"c13_{j}_kappa"]),
                        orientation=ori))
    return out


def prediction(base,t,row):
    th=np.asarray(json.loads(row.theta_json),float)
    return v6.v6_model(base,np.asarray(t,float),th,row_sites(row))


def safe_float(x):
    try:
        z=float(x); return z if np.isfinite(z) else np.nan
    except Exception:
        return np.nan


def components(best,field):
    rows=[]
    for _,r in best.iterrows():
        s=safe_float(r.get("visibility_scale",np.nan))
        for j in range(1,int(r.model_order)+1):
            k=safe_float(r.get(f"c13_{j}_kappa",np.nan))
            rows.append(dict(
                field=field,nv_index=int(r.nv_index),model_order=int(r.model_order),
                component=j,site_id=int(r[f"c13_{j}_site_id"]),
                f0_kHz=safe_float(r.get(f"c13_{j}_f0_kHz")),
                f1_kHz=safe_float(r.get(f"c13_{j}_f1_kHz")),
                kappa=k,visibility_scale=s,effective_kappa=s*k,
                distance_A=safe_float(r.get(f"c13_{j}_distance_A")),
                physical_snr=safe_float(r.get(f"c13_{j}_physical_snr")),
                amp=safe_float(r.get(f"c13_{j}_amp")),
            ))
    return pd.DataFrame(rows)


def n3_n2_gap_table(cdf,pop_nvs):
    rows=[]
    for nv in pop_nvs:
        g=cdf[cdf.nv_index==nv]
        q2=g[g.model_order==2];q3=g[g.model_order==3]
        if not len(q2) or not len(q3): continue
        r2=q2.loc[q2.bic.idxmin()];r3=q3.loc[q3.bic.idxmin()]
        rows.append(dict(
            nv_index=int(nv),bic_N2=float(r2.bic),bic_N3=float(r3.bic),
            delta_bic_N2_minus_N3=float(r2.bic-r3.bic),
            site_N2=str(r2.site_key),site_N3=str(r3.site_key),
        ))
    return pd.DataFrame(rows)


def transition_table(v6best,v14best):
    a=v6best[["nv_index","model_order","site_key","bic","red_chi2",
              "visibility_scale","visibility_scale_bound_hit"]].copy()
    a.columns=["nv_index","v6_order","v6_site_key","v6_bic","v6_redchi",
               "v6_scale","v6_scale_bound"]
    b=v14best[["nv_index","model_order","site_key","bic","red_chi2",
               "visibility_scale","visibility_scale_bound_hit"]].copy()
    b.columns=["nv_index","v14_order","v14_site_key","v14_bic","v14_redchi",
               "v14_scale","v14_scale_bound"]
    x=a.merge(b,on="nv_index",how="inner")
    x["order_changed"]=x.v6_order!=x.v14_order
    x["site_changed"]=x.v6_site_key.astype(str)!=x.v14_site_key.astype(str)
    x["order_delta"]=x.v14_order-x.v6_order
    x["delta_bic_v14_vs_v6"]=x.v14_bic-x.v6_bic
    overlaps=[];jacc=[]
    for _,r in x.iterrows():
        A=set(parse_site_key(r.v6_site_key)); B=set(parse_site_key(r.v14_site_key))
        overlaps.append(len(A&B))
        jacc.append(len(A&B)/len(A|B) if A|B else 1.0)
    x["site_overlap_count"]=overlaps;x["site_jaccard"]=jacc
    return x


def build_residuals(base,t,Y,E,best,nvs):
    rows=[]; mat=[]; rms=[]
    bynv=best.set_index("nv_index")
    for nv in nvs:
        r=bynv.loc[nv]
        ee=base.safe_err(E[nv]); yy=np.asarray(Y[nv],float)
        pred=prediction(base,t,r)
        z=(yy-pred)/ee
        mat.append(z);rms.append(np.sqrt(np.mean(z*z)))
        for ti,zz in zip(t,z):
            rows.append((nv,float(ti),float(zz)))
    return np.asarray(mat),np.asarray(rms),pd.DataFrame(rows,columns=["nv_index","time_us","residual_sigma"])


def page_title(fig,title,subtitle=""):
    fig.suptitle(title,fontsize=16,y=.985)
    if subtitle: fig.text(.5,.955,subtitle,ha="center",va="top",fontsize=9)


def clean(ax):
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.grid(alpha=.16)


def evidence_label(g):
    if g < 2: return "<2 ambiguous"
    if g < 6: return "2-6 positive"
    if g < 10:return "6-10 strong"
    return ">=10 very strong"


def make_dashboard(path,field,base,t,pop_v6,pop_v14,trans,gaps,comp,
                   rv6,rv14,rms6,rms14,summary):
    nvs=pop_v14.nv_index.astype(int).tolist()
    with PdfPages(path) as pdf:
        # PAGE 1 overview
        fig,axs=plt.subplots(2,3,figsize=(14,9))
        page_title(fig,f"{field} V14 population dashboard",
                   "V14 = V6 discrete-C13 architecture with beta=2 and revival taper=0")
        orders=np.arange(4)
        c6=pop_v6.model_order.value_counts().reindex(orders,fill_value=0)
        c14=pop_v14.model_order.value_counts().reindex(orders,fill_value=0)
        axs[0,0].bar(orders-.18,c6,width=.36,label="V6")
        axs[0,0].bar(orders+.18,c14,width=.36,label="V14")
        axs[0,0].set_xticks(orders);axs[0,0].set(xlabel="Number of C13",ylabel="NV count",
            title="Winner model-order population");axs[0,0].legend();clean(axs[0,0])

        bins=np.linspace(0,min(8,max(3,float(np.nanpercentile(pop_v14.red_chi2,97)))),32)
        axs[0,1].hist(pop_v6.red_chi2,bins=bins,histtype="step",lw=1.6,label="V6")
        axs[0,1].hist(pop_v14.red_chi2,bins=bins,histtype="step",lw=1.6,label="V14")
        axs[0,1].axvline(POOR_REDCHI,ls="--",lw=.9)
        axs[0,1].set(xlabel="Reduced chi-square",ylabel="NV count",title="Fit quality")
        axs[0,1].legend();clean(axs[0,1])

        db=trans.delta_bic_v14_vs_v6.to_numpy(float)
        axs[0,2].hist(np.clip(db,-20,30),bins=35)
        axs[0,2].axvline(0,lw=.8);axs[0,2].axvline(-6,ls="--",lw=.8)
        axs[0,2].axvline(6,ls="--",lw=.8)
        axs[0,2].set(xlabel="BIC(V14 winner) - BIC(V6 winner)",ylabel="NV count",
                     title="Does simplification pay?");clean(axs[0,2])

        s=pop_v14.loc[pop_v14.model_order>0,"visibility_scale"].dropna()
        axs[1,0].hist(s,bins=30)
        axs[1,0].axvline(3,ls="--",lw=.9)
        axs[1,0].set(xlabel="Shared visibility scale",ylabel="NV count",
                     title=f"V14 scale | bound hits={(s>=SCALE_BOUND).sum()}/{len(s)}")
        clean(axs[1,0])

        if len(gaps):
            g=gaps.delta_bic_N2_minus_N3.to_numpy(float)
            axs[1,1].hist(np.clip(g,-20,30),bins=40)
            for x in [0,2,6,10]:axs[1,1].axvline(x,ls="--" if x else "-",lw=.8)
            axs[1,1].set(xlabel="BIC(N2)-BIC(N3)",ylabel="NV count",
                         title="N3 vs N2 evidence (all comparable NVs)")
            clean(axs[1,1])

        axs[1,2].axis("off")
        lines=[
            f"Population NVs       : {len(pop_v14)}",
            f"V6 order counts      : {c6.tolist()}",
            f"V14 order counts     : {c14.tolist()}",
            f"Order changes        : {int(trans.order_changed.sum())}",
            f"Site changes         : {int(trans.site_changed.sum())}",
            f"Same-order site swaps: {int((~trans.order_changed & trans.site_changed).sum())}",
            f"Median chi2r V6/V14  : {pop_v6.red_chi2.median():.3f} / {pop_v14.red_chi2.median():.3f}",
            f"Median dBIC V14-V6   : {np.median(db):+.3f}",
            f"V14 BIC better       : {int((db<0).sum())}/{len(db)}",
            f"V14 strongly better  : {int((db<=-6).sum())}/{len(db)}",
            f"V6 strongly better   : {int((db>=6).sum())}/{len(db)}",
            f"Scale bound hits      : {int(pop_v14.visibility_scale_bound_hit.fillna(False).sum())}",
            "",
            "52G population excludes NV0." if field=="52G" else "",
        ]
        axs[1,2].text(.02,.98,"\n".join(lines),va="top",family="monospace",fontsize=9)
        fig.tight_layout(rect=[0,.01,1,.94]);pdf.savefig(fig);plt.close(fig)

        # PAGE 2 transitions
        fig,axs=plt.subplots(1,2,figsize=(14,6.5))
        page_title(fig,f"{field}: V6 -> V14 winner transitions")
        ct=pd.crosstab(trans.v6_order,trans.v14_order).reindex(index=orders,columns=orders,fill_value=0)
        im=axs[0].imshow(ct.to_numpy(),aspect="equal")
        axs[0].set_xticks(orders);axs[0].set_yticks(orders)
        axs[0].set(xlabel="V14 order",ylabel="V6 order",title="Model-order transition matrix")
        for i in range(4):
            for j in range(4):axs[0].text(j,i,str(int(ct.iloc[i,j])),ha="center",va="center")
        fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04)

        same=trans[~trans.order_changed]
        vals=[int((~same.site_changed).sum()),int(same.site_changed.sum()),
              int((trans.order_delta>0).sum()),int((trans.order_delta<0).sum())]
        labs=["same order+site","same order, site swap","order increased","order decreased"]
        axs[1].bar(np.arange(4),vals)
        axs[1].set_xticks(np.arange(4));axs[1].set_xticklabels(labs,rotation=25,ha="right")
        axs[1].set_ylabel("NV count");axs[1].set_title("What changed?");clean(axs[1])
        fig.tight_layout(rect=[0,.02,1,.93]);pdf.savefig(fig);plt.close(fig)

        # PAGE 3 N3 evidence
        fig,axs=plt.subplots(2,2,figsize=(12,8.5))
        page_title(fig,f"{field}: how strong is the N=3 assignment?")
        n3=pop_v14[pop_v14.model_order==3]
        ng=gaps[gaps.nv_index.isin(n3.nv_index)].copy()
        if len(ng):
            g=ng.delta_bic_N2_minus_N3
            axs[0,0].hist(g,bins=np.linspace(min(-2,g.min()),max(30,g.max()),38))
            for x in [2,6,10]:axs[0,0].axvline(x,ls="--",lw=.8)
            axs[0,0].set(xlabel="BIC(N2)-BIC(N3)",ylabel="N3 winners",title="N3 winner evidence")
            clean(axs[0,0])
            cats=pd.cut(g,[-np.inf,2,6,10,np.inf],right=False,
                        labels=["<2","2-6","6-10",">=10"])
            cc=cats.value_counts().reindex(["<2","2-6","6-10",">=10"])
            axs[0,1].bar(cc.index.astype(str),cc.values)
            axs[0,1].set(ylabel="N3 winners",title="Evidence categories");clean(axs[0,1])
            mm=n3.merge(ng,on="nv_index")
            axs[1,0].scatter(mm.delta_bic_N2_minus_N3,mm.red_chi2,
                             c=mm.visibility_scale,s=28)
            axs[1,0].axvline(6,ls="--",lw=.8)
            axs[1,0].set(xlabel="BIC(N2)-BIC(N3)",ylabel="V14 reduced chi-square",
                         title="Evidence vs fit quality");clean(axs[1,0])
            axs[1,1].scatter(mm.delta_bic_N2_minus_N3,mm.visibility_scale,s=28)
            axs[1,1].axhline(3,ls="--",lw=.8);axs[1,1].axvline(6,ls="--",lw=.8)
            axs[1,1].set(xlabel="BIC(N2)-BIC(N3)",ylabel="Visibility scale",
                         title="Evidence vs amplitude saturation");clean(axs[1,1])
        fig.tight_layout(rect=[0,.02,1,.94]);pdf.savefig(fig);plt.close(fig)

        # PAGE 4 amplitude / component diagnostics
        fig,axs=plt.subplots(2,2,figsize=(12,8.5))
        page_title(fig,f"{field}: discrete-C13 amplitude diagnostics")
        if len(comp):
            axs[0,0].scatter(comp.kappa,comp.effective_kappa,c=comp.visibility_scale,s=18,alpha=.7)
            axs[0,0].plot([0,1],[0,1],ls="--",lw=.8)
            axs[0,0].axhline(1,ls=":",lw=.8)
            axs[0,0].set(xlabel="Catalog kappa",ylabel="s_NV * kappa",
                         title="Effective modulation strength");clean(axs[0,0])
            axs[0,1].hist(comp.effective_kappa.dropna(),bins=35)
            axs[0,1].axvline(1,ls="--",lw=.8)
            axs[0,1].set(xlabel="s_NV * kappa",ylabel="Components",
                         title=f"Components with s*kappa>1: {(comp.effective_kappa>1).sum()}/{len(comp)}")
            clean(axs[0,1])
            axs[1,0].scatter(comp.distance_A,comp.kappa,c=comp.visibility_scale,s=18,alpha=.7)
            axs[1,0].set(xlabel="Distance (A)",ylabel="kappa",title="Selected lattice sites");clean(axs[1,0])
            fc=.5*(comp.f0_kHz+comp.f1_kHz)
            sp=np.abs(comp.f1_kHz-comp.f0_kHz)
            axs[1,1].scatter(fc,sp,c=comp.kappa,s=18,alpha=.7)
            axs[1,1].set(xlabel="Frequency center (kHz)",ylabel="Splitting (kHz)",
                         title="Selected C13 spectral distribution");clean(axs[1,1])
        fig.tight_layout(rect=[0,.02,1,.94]);pdf.savefig(fig);plt.close(fig)

        # PAGE 5 remaining background parameter behavior
        fig,axs=plt.subplots(2,3,figsize=(14,8.5))
        page_title(fig,f"{field}: remaining V14 background parameters")
        pars=[("revival_time_us","Trev (us)",40),("width0_us","width0 (us)",None),
              ("T2_us","T2 (us)",None),("width_slope","width slope",None),
              ("revival_chirp","chirp",None)]
        for ax,(col,lab,bound) in zip(axs.ravel()[:5],pars):
            z=pop_v14[col].dropna()
            ax.hist(z,bins=32)
            if bound is not None:ax.axvline(bound,ls="--",lw=.8)
            ax.set(xlabel=lab,ylabel="NV count");clean(ax)
        axs[1,2].axis("off")
        Tphys=2000.0/v8.catalog_larmor_khz(base)
        mu1=pop_v14.revival_time_us*(1+pop_v14.revival_chirp)
        mu2=2*pop_v14.revival_time_us*(1+2*pop_v14.revival_chirp)
        lines=[
            f"Trev physical = {Tphys:.3f} us",
            f"median Trev   = {pop_v14.revival_time_us.median():.3f} us",
            f"Trev=40 hits  = {int(np.isclose(pop_v14.revival_time_us,40,atol=1e-4).sum())}",
            f"slope=0 hits  = {int(np.isclose(pop_v14.width_slope,0,atol=1e-6).sum())}",
            f"slope=.8 hits = {int(np.isclose(pop_v14.width_slope,.8,atol=1e-5).sum())}",
            f"|chirp|=.06   = {int(np.isclose(abs(pop_v14.revival_chirp),.06,atol=1e-5).sum())}",
            f"T2 upper hits = {int(pop_v14.T2_bound_hit.fillna(False).sum())}",
            "",
            f"median mu1 - Tphys   = {np.median(mu1-Tphys):+.3f} us",
            f"median mu2 - 2Tphys  = {np.median(mu2-2*Tphys):+.3f} us",
        ]
        axs[1,2].text(.02,.98,"\n".join(lines),va="top",family="monospace",fontsize=9)
        fig.tight_layout(rect=[0,.02,1,.94]);pdf.savefig(fig);plt.close(fig)

        # PAGE 6 residual comparison
        fig,axs=plt.subplots(2,2,figsize=(12,8.5))
        page_title(fig,f"{field}: population residual structure")
        med6=np.nanmedian(rv6,axis=0);med14=np.nanmedian(rv14,axis=0)
        axs[0,0].plot(t,med6,label="V6")
        axs[0,0].plot(t,med14,label="V14")
        axs[0,0].axhline(0,lw=.7)
        axs[0,0].set(xlabel="Total evolution time (us)",ylabel="Median residual (sigma)",
                     title="Common normalized residual");axs[0,0].legend();clean(axs[0,0])

        for order in range(4):
            inds=np.flatnonzero(pop_v14.model_order.to_numpy()==order)
            if len(inds):
                axs[0,1].plot(t,np.nanmedian(rv14[inds],axis=0),label=f"N={order} ({len(inds)})")
        axs[0,1].axhline(0,lw=.7)
        axs[0,1].set(xlabel="Total evolution time (us)",ylabel="Median residual (sigma)",
                     title="V14 common residual by winning order")
        axs[0,1].legend(fontsize=8);clean(axs[0,1])

        axs[1,0].scatter(rms6,rms14,c=pop_v14.model_order,s=22)
        lim=max(np.nanpercentile(rms6,98),np.nanpercentile(rms14,98),1)
        axs[1,0].plot([0,lim],[0,lim],ls="--",lw=.8)
        axs[1,0].set(xlabel="V6 residual RMS (sigma)",ylabel="V14 residual RMS (sigma)",
                     title="Per-NV residual RMS");clean(axs[1,0])

        # residual heatmap sorted by order then RMS
        ordidx=np.lexsort((rms14,pop_v14.model_order.to_numpy()))
        im=axs[1,1].imshow(np.clip(rv14[ordidx],-3,3),aspect="auto",interpolation="nearest",
                           extent=[t.min(),t.max(),len(ordidx),0],
                           norm=TwoSlopeNorm(vmin=-3,vcenter=0,vmax=3))
        axs[1,1].set(xlabel="Total evolution time (us)",ylabel="NV (sorted)",
                     title="V14 residual heatmap, clipped +/-3 sigma")
        fig.colorbar(im,ax=axs[1,1],fraction=.046,pad=.04)
        fig.tight_layout(rect=[0,.02,1,.94]);pdf.savefig(fig);plt.close(fig)

        # PAGE 7 change / stability diagnostics
        fig,axs=plt.subplots(2,2,figsize=(12,8.5))
        page_title(fig,f"{field}: site-assignment stability")
        changed=trans[trans.site_changed]
        axs[0,0].hist(changed.site_jaccard,bins=np.linspace(0,1,11))
        axs[0,0].set(xlabel="Jaccard overlap of selected site sets",ylabel="Changed winners",
                     title="How different are changed site assignments?");clean(axs[0,0])
        axs[0,1].scatter(trans.v6_bic,trans.v14_bic,c=trans.order_delta,s=24)
        lo=min(trans.v6_bic.min(),trans.v14_bic.min());hi=max(trans.v6_bic.max(),trans.v14_bic.max())
        axs[0,1].plot([lo,hi],[lo,hi],ls="--",lw=.8)
        axs[0,1].set(xlabel="V6 winner BIC",ylabel="V14 winner BIC",
                     title="Winner BIC before/after simplification");clean(axs[0,1])
        axs[1,0].scatter(trans.v6_redchi,trans.v14_redchi,c=trans.order_delta,s=24)
        lim=max(np.nanpercentile(trans.v6_redchi,97),np.nanpercentile(trans.v14_redchi,97),1)
        axs[1,0].plot([0,lim],[0,lim],ls="--",lw=.8)
        axs[1,0].set(xlabel="V6 chi2r",ylabel="V14 chi2r",title="Winner fit quality");clean(axs[1,0])
        axs[1,1].axis("off")
        up=trans[trans.order_delta>0];down=trans[trans.order_delta<0]
        lines=[
            f"Order unchanged : {(trans.order_delta==0).sum()}",
            f"Order increased : {len(up)}",
            f"Order decreased : {len(down)}",
            f"Site changed    : {trans.site_changed.sum()}",
            f"same-order swaps: {((trans.order_delta==0)&trans.site_changed).sum()}",
            "",
            f"Median dBIC, order up   : {up.delta_bic_v14_vs_v6.median():+.2f}" if len(up) else "",
            f"Median dBIC, order down : {down.delta_bic_v14_vs_v6.median():+.2f}" if len(down) else "",
            f"Median Jaccard changed  : {changed.site_jaccard.median():.3f}" if len(changed) else "",
        ]
        axs[1,1].text(.02,.98,"\n".join(lines),va="top",family="monospace",fontsize=9)
        fig.tight_layout(rect=[0,.02,1,.94]);pdf.savefig(fig);plt.close(fig)

        # PAGE 8 problem classes / priority list
        fig,ax=plt.subplots(figsize=(12,8.5));ax.axis("off")
        page_title(fig,f"{field}: V14 diagnostic priority list",
                   "Flags identify where the current model still needs attention")
        gapmap=gaps.set_index("nv_index").delta_bic_N2_minus_N3.to_dict()
        rows=[]
        for _,r in pop_v14.iterrows():
            nv=int(r.nv_index);flags=[]
            if r.red_chi2>POOR_REDCHI:flags.append("poor_fit")
            if int(r.model_order)>0 and safe_float(r.visibility_scale)>=SCALE_BOUND:flags.append("scale_bound")
            if int(r.model_order)==3 and gapmap.get(nv,np.inf)<2:flags.append("N3_ambiguous")
            if np.isclose(r.revival_time_us,40,atol=1e-4):flags.append("Trev_bound")
            if np.isclose(abs(r.revival_chirp),.06,atol=1e-5):flags.append("chirp_bound")
            if flags:
                rows.append((nv,int(r.model_order),float(r.red_chi2),
                             safe_float(r.visibility_scale),gapmap.get(nv,np.nan),
                             ",".join(flags)))
        q=pd.DataFrame(rows,columns=["NV","N","chi2r","scale","dBIC_N2-N3","flags"])
        q["nflags"]=q["flags"].str.count(",")+1
        q=q.sort_values(["nflags","chi2r"],ascending=[False,False]).head(34)
        lines=[" NV   N  chi2r  scale  dBIC2-3  flags",
               "----  -  -----  -----  -------  -----------------------------------------"]
        for _,r in q.iterrows():
            db="   --  " if not np.isfinite(r["dBIC_N2-N3"]) else f"{r['dBIC_N2-N3']:7.2f}"
            sc=" --  " if not np.isfinite(r["scale"]) else f"{r['scale']:5.2f}"
            lines.append(f"{int(r.NV):4d}  {int(r.N):1d}  {r.chi2r:5.2f}  {sc}  {db}  {r.flags}")
        ax.text(.02,.97,"\n".join(lines),va="top",family="monospace",fontsize=8.5)
        pdf.savefig(fig);plt.close(fig)


def analyze_field(field,outdir):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths();t,Y,E=base.load_data(ck)
    p6=v7.latest_v6_candidate(base,field);p14=discover_v14(field)
    c6=pd.read_csv(p6);c14=pd.read_csv(p14)
    b6=c6[c6.rank_global_bic==1].sort_values("nv_index").copy()
    b14=c14[c14.rank_global_bic==1].sort_values("nv_index").copy()

    pop_nvs=sorted(set(b14.nv_index.astype(int)))
    if field=="52G" and 0 in pop_nvs:pop_nvs.remove(0)
    pop6=b6[b6.nv_index.isin(pop_nvs)].sort_values("nv_index").reset_index(drop=True)
    pop14=b14[b14.nv_index.isin(pop_nvs)].sort_values("nv_index").reset_index(drop=True)
    trans=transition_table(pop6,pop14)
    gaps=n3_n2_gap_table(c14,pop_nvs)
    comp=components(pop14,field)

    rv6,rms6,_=build_residuals(base,t,Y,E,pop6,pop_nvs)
    rv14,rms14,reslong=build_residuals(base,t,Y,E,pop14,pop_nvs)

    # per-NV diagnostic table
    per=trans.merge(
        pop14[["nv_index","model_order","site_key","red_chi2","visibility_scale",
               "visibility_scale_bound_hit","revival_time_us","width0_us","T2_us",
               "width_slope","revival_chirp","T2_bound_hit"]],
        on="nv_index",how="left")
    per=per.merge(gaps[["nv_index","delta_bic_N2_minus_N3"]],on="nv_index",how="left")
    per["v6_resid_rms"]=rms6;per["v14_resid_rms"]=rms14
    per["v14_resid_rms_minus_v6"]=rms14-rms6
    per["Trev_bound_hit"]=np.isclose(per.revival_time_us,40,atol=1e-4)
    per["slope_zero_hit"]=np.isclose(per.width_slope,0,atol=1e-6)
    per["slope_upper_hit"]=np.isclose(per.width_slope,.8,atol=1e-5)
    per["chirp_bound_hit"]=np.isclose(abs(per.revival_chirp),.06,atol=1e-5)

    c6o=pop6.model_order.value_counts().reindex([0,1,2,3],fill_value=0)
    c14o=pop14.model_order.value_counts().reindex([0,1,2,3],fill_value=0)
    n3=pop14[pop14.model_order==3]
    ng=gaps[gaps.nv_index.isin(n3.nv_index)].delta_bic_N2_minus_N3
    db=trans.delta_bic_v14_vs_v6
    summary=pd.DataFrame([dict(
        field=field,n_population=len(pop14),
        v6_n0=int(c6o[0]),v6_n1=int(c6o[1]),v6_n2=int(c6o[2]),v6_n3=int(c6o[3]),
        v14_n0=int(c14o[0]),v14_n1=int(c14o[1]),v14_n2=int(c14o[2]),v14_n3=int(c14o[3]),
        order_changes=int(trans.order_changed.sum()),
        site_changes=int(trans.site_changed.sum()),
        same_order_site_changes=int((~trans.order_changed & trans.site_changed).sum()),
        order_increases=int((trans.order_delta>0).sum()),
        order_decreases=int((trans.order_delta<0).sum()),
        median_delta_bic_v14_v6=float(db.median()),
        v14_bic_better=int((db<0).sum()),v14_bic_strong_better=int((db<=-6).sum()),
        v6_bic_strong_better=int((db>=6).sum()),
        median_v6_redchi=float(pop6.red_chi2.median()),
        median_v14_redchi=float(pop14.red_chi2.median()),
        v14_poor_fit_count=int((pop14.red_chi2>POOR_REDCHI).sum()),
        nonzero_count=int((pop14.model_order>0).sum()),
        scale_bound_count=int(pop14.visibility_scale_bound_hit.fillna(False).sum()),
        median_nonzero_scale=float(pop14.loc[pop14.model_order>0,"visibility_scale"].median()),
        selected_components=len(comp),
        components_effective_kappa_gt1=int((comp.effective_kappa>1).sum()) if len(comp) else 0,
        n3_winners=len(n3),n3_gap_median=float(ng.median()) if len(ng) else np.nan,
        n3_gap_lt2=int((ng<2).sum()),n3_gap_2_6=int(((ng>=2)&(ng<6)).sum()),
        n3_gap_6_10=int(((ng>=6)&(ng<10)).sum()),n3_gap_ge10=int((ng>=10).sum()),
        Trev40_hits=int(np.isclose(pop14.revival_time_us,40,atol=1e-4).sum()),
        slope0_hits=int(np.isclose(pop14.width_slope,0,atol=1e-6).sum()),
        slope_upper_hits=int(np.isclose(pop14.width_slope,.8,atol=1e-5).sum()),
        chirp_bound_hits=int(np.isclose(abs(pop14.revival_chirp),.06,atol=1e-5).sum()),
        T2_upper_hits=int(pop14.T2_bound_hit.fillna(False).sum()),
        common_resid_rms_v6=float(np.sqrt(np.mean(np.nanmedian(rv6,axis=0)**2))),
        common_resid_rms_v14=float(np.sqrt(np.mean(np.nanmedian(rv14,axis=0)**2))),
    )])

    outdir.mkdir(parents=True,exist_ok=True)
    prefix=f"v15_v14_population_{field}"
    summary.to_csv(outdir/f"{prefix}_summary.csv",index=False)
    per.to_csv(outdir/f"{prefix}_per_nv.csv",index=False)
    trans.to_csv(outdir/f"{prefix}_transitions.csv",index=False)
    gaps.to_csv(outdir/f"{prefix}_n3_n2_gaps.csv",index=False)
    comp.to_csv(outdir/f"{prefix}_components.csv",index=False)
    reslong.to_csv(outdir/f"{prefix}_v14_residuals_long.csv",index=False)
    make_dashboard(outdir/f"{prefix}.pdf",field,base,t,pop6,pop14,trans,gaps,comp,
                   rv6,rv14,rms6,rms14,summary)
    print("\n",field)
    print(summary.to_string(index=False))
    print("PDF:",outdir/f"{prefix}.pdf")
    return summary,per


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--output-dir",default=str(OUT_ROOT))
    args=ap.parse_args()
    out=Path(args.output_dir)
    fields=["49G","52G"] if args.field=="both" else [args.field]
    summaries=[];pers=[]
    for f in fields:
        s,p=analyze_field(f,out);summaries.append(s);pers.append(p)
    if len(summaries)>1:
        pd.concat(summaries,ignore_index=True).to_csv(out/"v15_v14_population_both_summary.csv",index=False)
        pp=pd.concat([p.assign(field=f) for p,f in zip(pers,fields)],ignore_index=True)
        pp.to_csv(out/"v15_v14_population_both_per_nv.csv",index=False)
    print("\nCOMPLETE")


if __name__=="__main__":
    main()
