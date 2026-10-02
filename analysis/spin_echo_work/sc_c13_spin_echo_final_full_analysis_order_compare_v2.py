"""Order-comparison report for the frozen 49G/52G spin-echo analyses.

For every requested NV:
  * overlays the best N=0,1,2,3 source-search fits
  * adds the refined final model (including N=4/cross-term at 52G)
  * plots DeltaBIC and reduced chi2 versus model order
  * plots the C13 candidate sites in (f_minus, f_plus) frequency space
  * labels the selected lattice-site IDs

This is visualization only; it does not rerun the lattice search.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14
import sc_c13_spin_echo_final_full_analysis_both_fields as full

OUTROOT=full.ROOT/r"spin_echo_final_order_compare\2026_09"
def source_candidates(field,nv,src):
    if field=="52G" and nv in src["w24"].index:
        c=src["c24"]
    else:
        c=src["c22"]
    q=c[c.nv_index==nv].copy()
    if q.empty:
        return q
    key="site_key_canonical" if "site_key_canonical" in q else "site_key"
    return q.sort_values("bic").drop_duplicates(["model_order",key])


def best_by_order(field,nv,src):
    q=source_candidates(field,nv,src)
    out={}
    for order,g in q.groupby("model_order"):
        out[int(order)]=g.sort_values("bic").iloc[0]
    return out


def row_model(base,row,t):
    sites=v14.sites_from_record(row)
    theta=np.asarray(json.loads(str(row.theta_json)),float)
    return sites,theta,lambda x:v6.v6_model(base,np.asarray(x,float),theta,sites)


def unique_candidate_sites(q):
    rows={}
    for _,r in q.iterrows():
        n=int(r.model_order)
        for j in range(1,n+1):
            sid=r.get(f"c13_{j}_site_id",np.nan)
            if pd.isna(sid):
                continue
            sid=int(sid)
            rows[sid]=dict(
                site_id=sid,
                f0=float(r.get(f"c13_{j}_f0_kHz",np.nan)),
                f1=float(r.get(f"c13_{j}_f1_kHz",np.nan)),
                kappa=float(r.get(f"c13_{j}_kappa",np.nan)),
            )
    return list(rows.values())
def plot_page(pdf,field,nv,name,base,t,y,e,cat,src):
    orders=best_by_order(field,nv,src)
    final=full.reconstruct(field,nv,base,t,y,e,cat,src)
    dense=np.linspace(float(t.min()),float(t.max()),1800)
    fig,ax=plt.subplots(2,3,figsize=(14,8.5))

    ax[0,0].errorbar(t,y,yerr=e,fmt=".",ms=2.8,lw=.5,alpha=.55,label="data")
    order_info=[]
    for order in sorted(orders):
        row=orders[order]
        sites,theta,predict=row_model(base,row,t)
        label=f"N{order} {full.canon(row.site_key) if order else '()'}"
        ax[0,0].plot(dense,predict(dense),lw=1.0,label=label)
        order_info.append(dict(
            order=order,bic=float(row.bic),red=float(row.red_chi2),
            sites=str(full.canon(row.site_key)),
        ))
    # Final refined 52G result can be outside / different from source N0-N3.
    if field=="52G" and (
        final["order"] not in orders or final["source"] not in ("V22_full","V24_targeted_heavy")
    ):
        ax[0,0].plot(dense,final["predict"](dense),lw=2.0,ls="--",
                     label=f"FINAL N{final['order']} {final['site_key']}")
    ax[0,0].set(xlabel="total evolution time (us)",ylabel="normalized signal",
                title=f"{field} NV{nv} {name}: best fit at each N")
    ax[0,0].legend(fontsize=6.5,ncol=2)
    oi=pd.DataFrame(order_info).sort_values("order")
    if len(oi):
        db=oi.bic-oi.bic.min()
        ax[0,1].plot(oi.order,db,"o-")
        ax[0,1].axhline(2,ls="--",lw=.7)
        ax[0,1].axhline(6,ls="--",lw=.7)
        ax[0,1].set(xticks=oi.order,xlabel="model order N",ylabel="Delta BIC",
                    title="Model-order evidence")

        ax[0,2].plot(oi.order,oi.red,"o-")
        ax[0,2].axhline(2,ls="--",lw=.7)
        ax[0,2].axhline(3,ls="--",lw=.7)
        ax[0,2].set(xticks=oi.order,xlabel="model order N",
                    ylabel="reduced chi-square",title="Fit quality versus N")

    q=source_candidates(field,nv,src)
    ss=unique_candidate_sites(q)
    if ss:
        sx=np.array([s["f1"] for s in ss]); sy=np.array([s["f0"] for s in ss])
        sk=np.array([s["kappa"] for s in ss])
        sizes=8+30*np.nan_to_num(sk,nan=0.0)
        ax[1,0].scatter(sx,sy,s=sizes,alpha=.18,label="searched sites")
    marked=set()
    for order,row in sorted(orders.items()):
        for sid in full.canon(row.site_key):
            if sid in marked: continue
            marked.add(sid)
            hit=[s for s in ss if s["site_id"]==sid]
            if hit:
                s=hit[0]
                ax[1,0].scatter([s["f1"]],[s["f0"]],s=55)
                ax[1,0].annotate(str(sid),(s["f1"],s["f0"]),fontsize=7,
                                 xytext=(3,3),textcoords="offset points")
    # Make sure refined final sites are visible even if not present in source search.
    for s in final["sites"]:
        sid=int(s["site_id"])
        if sid in marked: continue
        ax[1,0].scatter([s["f1_kHz"]],[s["f0_kHz"]],s=75,marker="*")
        ax[1,0].annotate(f"{sid} final",(s["f1_kHz"],s["f0_kHz"]),fontsize=7,
                         xytext=(3,3),textcoords="offset points")
    ax[1,0].set(xlabel="f_minus (kHz)",ylabel="f_plus (kHz)",
                title="13C candidate-site frequency map")

    # Real-space projection of the same candidate lattice sites.
    ori=tuple(int(v) for v in final["orientation"])
    cc=cat[cat.ori.map(tuple)==ori].copy()
    candidate_ids={int(s["site_id"]) for s in ss}
    cp=cc[cc.site_id.astype(int).isin(candidate_ids)]
    if len(cp):
        ax[1,1].scatter(cp.x_A,cp.y_A,s=8+25*np.nan_to_num(cp.kappa),alpha=.18)
    detail=[]
    for sid in final["site_key"]:
        hit=cc[cc.site_id.astype(int)==int(sid)]
        if hit.empty: continue
        r=hit.iloc[0]
        ax[1,1].scatter([r.x_A],[r.y_A],s=80,marker="*")
        ax[1,1].annotate(str(int(sid)),(r.x_A,r.y_A),fontsize=8,
                         xytext=(3,3),textcoords="offset points")
        detail.append(
            f"{int(sid)}: xyz=({r.x_A:.2f},{r.y_A:.2f},{r.z_A:.2f}) A\n"
            f"    f-= {r.f1_kHz:.2f}, f+= {r.f0_kHz:.2f} kHz, "
            f"k={r.kappa:.3f}, r={r.distance_A:.2f} A"
        )
    ax[1,1].scatter([0],[0],marker="x",s=50,label="NV")
    ax[1,1].set_aspect("equal",adjustable="datalim")
    ax[1,1].set(xlabel="x (A)",ylabel="y (A)",title="13C lattice sites (xy projection)")

    ax[1,2].axis("off")
    text=[
        f"Final model: N{final['order']} {final['site_key']}",
        f"source: {final['source']}",
        f"background: {final['background']}",
        f"redchi2: {final['redchi2']:.3f}",
        f"BIC: {final['bic']:.2f}",
        f"scale: {final['scale']:.3f}" if np.isfinite(final["scale"]) else "scale: n/a",
        f"BIC margin: {final['margin']:.3f}" if np.isfinite(final["margin"]) else "BIC margin: inf",
        f"class: {final['confidence']}",
        "",
        "Final 13C sites:",
        *detail,
    ]
    ax[1,2].text(0,1,"\n".join(text),va="top",ha="left",
                 fontsize=7.5,family="monospace",transform=ax[1,2].transAxes)

    fig.suptitle(
        f"Final: N{final['order']} {final['site_key']} | "
        f"redchi2={final['redchi2']:.2f} | {final['confidence']}",
        fontsize=11
    )
    fig.tight_layout(rect=[0,0,.99,.95])
    pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
    return oi


def run_field(field,args):
    base,t,Y,E,cat,names=full.load_measurements(field)
    src=full.load_sources(field)
    ids=full.parse_nv_arg(args.nv,Y.shape[0])
    outdir=Path(args.output_dir)/field; outdir.mkdir(parents=True,exist_ok=True)
    pdfpath=outdir/f"spin_echo_order_compare_{field}_{full.SAMPLES[field]}.pdf"
    allrows=[]
    with PdfPages(pdfpath) as pdf:
        for nv in ids:
            oi=plot_page(pdf,field,nv,names[nv],base,t,Y[nv],E[nv],cat,src)
            for _,r in oi.iterrows():
                allrows.append(dict(field=field,nv_index=nv,**r.to_dict()))
    pd.DataFrame(allrows).to_csv(
        outdir/f"spin_echo_order_compare_{field}_{full.SAMPLES[field]}.csv",index=False)
    print(field,"order-comparison complete:",pdfpath)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=("49G","52G","both"),default="both")
    ap.add_argument("--nv",default=None)
    ap.add_argument("--output-dir",default=str(OUTROOT))
    args=ap.parse_args()
    for f in (["49G","52G"] if args.field=="both" else [args.field]):
        run_field(f,args)


if __name__=="__main__":
    main()
