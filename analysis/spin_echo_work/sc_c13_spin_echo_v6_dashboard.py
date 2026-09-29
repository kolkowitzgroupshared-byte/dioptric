"""Post-process a completed V6 spin-echo fit into a detailed dashboard PDF.

No fitting is performed.  The saved V6 candidate table is used to reconstruct
the actual shared-physical-amplitude V6 model for global and per-NV pages.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import sc_c13_spin_echo_physical_family_search_v6 as v6

DENSE_POINTS = 1600
TOP_RANK_TABLE = 10


def parse_nv_list(s):
    if not s or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def discover_candidate(base, field):

    _, _, _, prefix = base.discover_paths()
    root = Path(base.OUTPUT_DIR) if base.OUTPUT_DIR else prefix.parent
    pats = list(root.glob(
        f"*{field}*v6_physamp_snr0p5_tol12p0kHz_smax3p0_topS15_perF3_"
        "oriassigned_maxC3_candidate_fits.csv"))
    pats = [p for p in pats if "_subset_" not in p.name]
    if not pats:
        raise FileNotFoundError(f"No completed full V6 candidate file found in {root}")
    return max(pats, key=lambda p: p.stat().st_mtime)


def row_sites(row):
    out = []
    for j in range(1, int(row.model_order) + 1):
        out.append(dict(
            site_id=int(row[f"c13_{j}_site_id"]),
            f0_kHz=float(row[f"c13_{j}_f0_kHz"]),
            f1_kHz=float(row[f"c13_{j}_f1_kHz"]),
            kappa=float(row[f"c13_{j}_kappa"]),
            orientation=ast.literal_eval(str(row.orientation)),
        ))
    return out


def prediction(base, tt, row):
    theta = np.asarray(json.loads(row.theta_json), float)
    return v6.v6_model(base, np.asarray(tt, float), theta, row_sites(row))


def row_id(row):

    return (int(row.model_order), str(row.site_key), round(float(row.bic), 8))


def finite_float(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def family_size(x):
    try:
        z = ast.literal_eval(str(x))
        return len(z) if isinstance(z, (list, tuple)) else np.nan
    except Exception:
        return np.nan


def set_equal_3d(ax, pts):
    if not pts:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        return
    p = np.vstack([np.zeros(3), np.asarray(pts)])
    lo, hi = p.min(axis=0), p.max(axis=0)
    c = 0.5 * (lo + hi)
    r = max(1.0, 0.58 * float(np.max(hi - lo)))
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)


def selected_components(best):
    rows = []
    for _, r in best.iterrows():

        s = finite_float(r.get("visibility_scale"))
        for j in range(1, int(r.model_order)+1):
            k = finite_float(r.get(f"c13_{j}_kappa"))
            rows.append(dict(
                nv_index=int(r.nv_index), order=int(r.model_order),
                site_id=int(r[f"c13_{j}_site_id"]), scale=s, kappa=k,
                effective_kappa=s*k if np.isfinite(s*k) else np.nan,
                amp=finite_float(r.get(f"c13_{j}_amp")),
                f0=finite_float(r.get(f"c13_{j}_f0_kHz")),
                f1=finite_float(r.get(f"c13_{j}_f1_kHz")),
                distance=finite_float(r.get(f"c13_{j}_distance_A")),
                physical_snr=finite_float(r.get(f"c13_{j}_physical_snr")),
                family_size=family_size(r.get(f"c13_{j}_family_members")),
            ))
    return pd.DataFrame(rows)


def global_summary(pdf, cdf, field, base):
    best = cdf[cdf.rank_global_bic == 1].sort_values("nv_index")
    raw = cdf[cdf.rank_global_redchi == 1].sort_values("nv_index")
    aicc = cdf[cdf.rank_global_aicc == 1].sort_values("nv_index")
    ev = []
    for nv, g in cdf.groupby("nv_index"):
        _, _, _, _, e = base.select_bic_with_evidence(g)
        ev.append(dict(nv_index=int(nv), **e))
    ev = pd.DataFrame(ev)
    comp = selected_components(best)

    fig, axs = plt.subplots(2, 3, figsize=(20, 13.5))
    orders = np.arange(4)
    for name, tab, dx in [("BIC",best,-.22),("AICc",aicc,0),("raw",raw,.22)]:

        cnt = tab.model_order.value_counts().reindex(orders, fill_value=0)
        axs[0,0].bar(orders+dx, cnt.values, width=.21, label=name)
    axs[0,0].set_xticks(orders)
    axs[0,0].set(title="Model-order comparison", xlabel="Number of C13", ylabel="NV count")
    axs[0,0].legend(); axs[0,0].grid(alpha=.15, axis="y")

    z = best.red_chi2.to_numpy(float)
    hi = max(2.5, min(10, np.nanpercentile(z[np.isfinite(z)], 97)))
    axs[0,1].hist(np.clip(z, 0, hi), bins=35)
    axs[0,1].axvline(np.nanmedian(z), ls="--", label=f"median={np.nanmedian(z):.3f}")
    axs[0,1].axvline(2.5, ls=":", label="2.5")
    axs[0,1].set(title="Best-BIC fit quality", xlabel="Reduced chi-square", ylabel="NV count")
    axs[0,1].legend(); axs[0,1].grid(alpha=.15)

    gaps = pd.to_numeric(ev.delta_bic_other_order, errors="coerce").dropna()
    axs[0,2].hist(np.clip(gaps, 0, 30), bins=35)
    for x in [2,6,10]: axs[0,2].axvline(x, ls="--", lw=.8)
    axs[0,2].set(title="Model-order evidence", xlabel="Delta BIC to competing order", ylabel="NV count")
    axs[0,2].grid(alpha=.15)

    scales = best.loc[best.model_order>0, "visibility_scale"].dropna().to_numpy(float)
    axs[1,0].hist(scales, bins=30)
    axs[1,0].axvline(v6.VISIBILITY_SCALE_MAX, ls="--", label="V6 scale ceiling")

    nbound = int(best.get("visibility_scale_bound_hit", False).fillna(False).astype(bool).sum())
    axs[1,0].set(title=f"Shared visibility scale s_NV | bound hits={nbound}",
                 xlabel="s_NV", ylabel="NV count")
    axs[1,0].legend(); axs[1,0].grid(alpha=.15)

    if len(comp):
        axs[1,1].scatter(comp.kappa, comp.effective_kappa, c=comp.physical_snr,
                         s=18, alpha=.7)
        axs[1,1].plot([0,1],[0,1], ls="--", lw=1)
        axs[1,1].axhline(1, ls=":", lw=1)
        axs[1,1].set(title="Physical-amplitude diagnostic",
                     xlabel="Catalog kappa", ylabel="s_NV * kappa")
        axs[1,1].grid(alpha=.15)

    axs[1,2].axis("off")
    counts = best.model_order.value_counts().reindex(orders, fill_value=0)
    txt = [
        "V6 RUN SUMMARY",
        f"field                    : {field}",
        f"NVs                      : {len(best)}",
        f"best-BIC N=0/1/2/3       : {counts.tolist()}",
        f"median reduced chi2      : {best.red_chi2.median():.3f}",
        f"chi2r > 2.5              : {(best.red_chi2>2.5).sum()}",
        f"scale ceiling hits       : {nbound}",
        f"selected C13 components  : {len(comp)}",
        f"s*kappa > 1 components   : {(comp.effective_kappa>1).sum() if len(comp) else 0}",
        "",
        "Amplitude model:",
        "a_j = s_NV * contrast * kappa_j / 4",
        "one shared s_NV for every C13 in an NV.",

        "",
        "Frequency band:",
        f"{best.frequency_band_low_kHz.median():.2f} - "
        f"{best.frequency_band_high_kHz.median():.1f} kHz",
    ]
    axs[1,2].text(.02,.98,"\n".join(txt), va="top", family="monospace", fontsize=10)
    fig.suptitle(f"{field} V6 physics-constrained multi-C13 dashboard", fontsize=16)
    fig.subplots_adjust(left=.06,right=.98,bottom=.06,top=.94,hspace=.28,wspace=.25)
    pdf.savefig(fig); plt.close(fig)


def c13_text(row, j):
    if j > int(row.model_order):
        return ""
    k = finite_float(row.get(f"c13_{j}_kappa"))
    s = finite_float(row.get("visibility_scale"))
    vals = [
        f"C13 #{j}: site {int(row[f'c13_{j}_site_id'])}",
        f"f0/f1 = {finite_float(row.get(f'c13_{j}_f0_kHz')):.3f} / "
        f"{finite_float(row.get(f'c13_{j}_f1_kHz')):.3f} kHz",
        f"kappa = {k:.5f}    s*kappa = {s*k:.5f}",
        f"amp = {finite_float(row.get(f'c13_{j}_amp')):.6f}",
        f"r = {finite_float(row.get(f'c13_{j}_distance_A')):.3f} A",
        f"physical SNR = {finite_float(row.get(f'c13_{j}_physical_snr')):.3f}",
        f"matched dchi2 = {finite_float(row.get(f'c13_{j}_matched_delta_chi2')):.2f}",
        f"family = {row.get(f'c13_{j}_family_id','')}",
        f"family size = {family_size(row.get(f'c13_{j}_family_members')):.0f}",
    ]
    return "\n".join(vals)


def model_text():
    return (
        "V6 MODEL\n"
        "S(t) = b - C M(t) + M(t) Sum_j a_j [cos(2*pi*f0_j*t+phi0_j) + "
        "cos(2*pi*f1_j*t+phi1_j)]\n"
        "a_j = s_NV * C * kappa_j / 4     (one shared s_NV per NV)\n"
        "M(t) = exp[-(t/T2)^beta] R(t), with the old-protocol revival envelope R(t).\n"
        "f0/f1 and kappa are locked to the assigned-orientation lattice catalog; "
        "both frequencies are restricted to the experimental band."
    )


def plot_nv(pdf, base, field, nv, t, y, e, cdf):
    d = cdf[cdf.nv_index == nv].copy()
    selected, raw_best, conservative, path, evidence = base.select_bic_with_evidence(d)
    aicc_best = d.sort_values(["aicc","bic"]).iloc[0]
    top = d.sort_values(["bic","red_chi2"]).head(TOP_RANK_TABLE)

    fig = plt.figure(figsize=(20,13.5))
    gs = fig.add_gridspec(4,3,height_ratios=[1,1,1.12,.34],
                          width_ratios=[1.2,1.05,1],hspace=.34,wspace=.27)
    ax_full=fig.add_subplot(gs[0,:2]); ax_rank=fig.add_subplot(gs[0,2])
    ax_zoom=fig.add_subplot(gs[1,0]); ax_res=fig.add_subplot(gs[1,1])
    ax3d=fig.add_subplot(gs[1,2],projection="3d")
    ax_order=fig.add_subplot(gs[2,0]); ax_info=fig.add_subplot(gs[2,1:])
    ax_model=fig.add_subplot(gs[3,:])

    yy=np.asarray(y[nv],float); ee=base.safe_err(e[nv])
    td=np.linspace(float(t.min()),float(t.max()),DENSE_POINTS)
    ax_full.errorbar(t,yy,yerr=ee,fmt="o",ms=3,capsize=1,lw=.45,
                     label="data",zorder=10)
    plotted=set()

    def draw(row,label,ls="-",lw=2,alpha=1):
        rid=row_id(row)
        if rid in plotted:return
        plotted.add(rid)
        ax_full.plot(td,prediction(base,td,row),ls=ls,lw=lw,alpha=alpha,label=label)

    draw(selected,
         f"BIC #1 | N={int(selected.model_order)} {selected.site_key} | "
         f"chi2r={selected.red_chi2:.2f}",lw=2.5)
    if row_id(aicc_best)!=row_id(selected):
        draw(aicc_best,f"AICc #1 | N={int(aicc_best.model_order)} {aicc_best.site_key}",
             ls="-.",lw=1.7)
    if row_id(raw_best)!=row_id(selected):
        draw(raw_best,f"raw #1 | N={int(raw_best.model_order)} {raw_best.site_key}",
             ls=":",lw=1.7)
    if row_id(conservative) not in plotted:
        draw(conservative,
             f"conservative | N={int(conservative.model_order)} {conservative.site_key}",
             ls="--",lw=1.5)
    for _,r in top.head(4).iterrows():
        if row_id(r) not in plotted:
            draw(r,f"BIC rank {int(r.rank_global_bic)} | N={int(r.model_order)} {r.site_key}",
                 lw=.85,alpha=.4)
    ax_full.set(xlabel="Total evolution time (us)",ylabel="Normalized signal",
                title=f"NV {nv}: V6 data and ranked alternatives")
    ax_full.legend(fontsize=7,ncol=2);ax_full.grid(alpha=.2)

    center=float(selected.revival_time_us); window=12.5
    m=np.abs(t-center)<=window
    tz=np.linspace(max(float(t.min()),center-window),
                   min(float(t.max()),center+window),700)
    ax_zoom.errorbar(t[m],yy[m],yerr=ee[m],fmt="o",ms=3,capsize=1,lw=.45)
    ax_zoom.plot(tz,prediction(base,tz,selected),lw=2.2,label="BIC #1")
    if row_id(aicc_best)!=row_id(selected):
        ax_zoom.plot(tz,prediction(base,tz,aicc_best),ls="-.",lw=1.2,label="AICc #1")
    if row_id(raw_best)!=row_id(selected):
        ax_zoom.plot(tz,prediction(base,tz,raw_best),ls=":",lw=1.2,label="raw #1")
    ax_zoom.axvline(center,ls=":",lw=.8)
    ax_zoom.set(xlabel="Total evolution time (us)",ylabel="Normalized signal",
                title="First-revival zoom")
    ax_zoom.legend(fontsize=7);ax_zoom.grid(alpha=.2)

    pred=prediction(base,t,selected)

    resid=(yy-pred)/ee
    ax_res.axhline(0,lw=1)
    for q,ls in [(2,"--"),(-2,"--"),(3,":"),(-3,":")]:
        ax_res.axhline(q,ls=ls,lw=.7,alpha=.55)
    ax_res.plot(t,resid,"o",ms=3)
    ax_res.set(xlabel="Total evolution time (us)",
               ylabel="(data-fit)/STE",
               title=f"Residuals | RMS={np.sqrt(np.mean(resid**2)):.2f}")
    ax_res.grid(alpha=.2)

    db=top.bic.to_numpy(float)-float(d.bic.min())
    labels=[f"#{int(r.rank_global_bic)} N{int(r.model_order)} {r.site_key}"
            for _,r in top.iterrows()]
    yp=np.arange(len(top))
    ax_rank.barh(yp,db);ax_rank.set_yticks(yp);ax_rank.set_yticklabels(labels,fontsize=7)
    ax_rank.invert_yaxis();ax_rank.set(xlabel="Delta BIC",title="Top BIC candidates")
    for yi,(_,r) in zip(yp,top.iterrows()):
        ax_rank.text(db[yi]+.15,yi,f"chi2r={r.red_chi2:.2f}",va="center",fontsize=6.5)
    ax_rank.grid(alpha=.15,axis="x")

    byorder=(d.sort_values(["bic","red_chi2"])
               .groupby("model_order",as_index=False).first().sort_values("model_order"))
    dob=byorder.bic.to_numpy(float)-float(byorder.bic.min())
    xx=np.arange(len(byorder))
    ax_order.bar(xx,dob);ax_order.set_xticks(xx)
    ax_order.set_xticklabels([f"N={int(o)}" for o in byorder.model_order])
    ax_order.set(xlabel="Model order",ylabel="Delta BIC",
                 title=f"Best per order | selected N={int(selected.model_order)}")
    for xi,(_,r) in zip(xx,byorder.iterrows()):
        ax_order.text(xi,dob[xi]+.25,f"{r.site_key}\nchi2r={r.red_chi2:.2f}",
                      ha="center",fontsize=7)
    ax_order.grid(alpha=.15,axis="y")

    geom=selected
    if int(selected.model_order)==0:
        nz=d[d.model_order>0].sort_values(["bic","red_chi2"])
        if len(nz):geom=nz.iloc[0]
    pts=[];ax3d.scatter([0],[0],[0],marker="*",s=180,label="NV",depthshade=False)
    for j in range(1,int(geom.model_order)+1):
        xyz=np.array([geom.get(f"c13_{j}_x_A",np.nan),
                      geom.get(f"c13_{j}_y_A",np.nan),
                      geom.get(f"c13_{j}_z_A",np.nan)],float)
        if not np.all(np.isfinite(xyz)):continue
        pts.append(xyz)
        ax3d.plot([0,xyz[0]],[0,xyz[1]],[0,xyz[2]],lw=1,alpha=.65)
        ax3d.scatter(*[[x] for x in xyz],s=90,depthshade=False,
                     label=f"C{j}: S{int(geom[f'c13_{j}_site_id'])}")
        ax3d.text(xyz[0],xyz[1],xyz[2],f" S{int(geom[f'c13_{j}_site_id'])}",fontsize=7)
    set_equal_3d(ax3d,pts)
    ax3d.set(xlabel="x (A)",ylabel="y (A)",zlabel="z (A)",
             title=f"C13 geometry | orientation {geom.orientation}")
    ax3d.view_init(elev=23,azim=38)
    if int(geom.model_order)>0:ax3d.legend(fontsize=6)

    ax_info.axis("off")
    scale=finite_float(selected.get("visibility_scale"))
    bound=bool(selected.get("visibility_scale_bound_hit",False))
    info=[
        "PRIMARY GLOBAL-BIC RESULT",
        f"N_C13 = {int(selected.model_order)}    sites = {selected.site_key}",
        f"orientation = {selected.orientation}",
        f"chi2r = {selected.red_chi2:.4f}",
        f"BIC = {selected.bic:.2f}    AICc = {selected.aicc:.2f}",
        f"DeltaBIC other order = {finite_float(evidence.get('delta_bic_other_order')):.2f}",
        f"DeltaBIC same-order site = {finite_float(evidence.get('delta_bic_same_order_site')):.2f}",
        f"review flags = {evidence.get('review_flags','')}",
        "",
        "V6 PHYSICAL AMPLITUDE",
        f"s_NV = {scale:.5f} / {finite_float(selected.get('visibility_scale_max')):.2f}",

        f"scale-bound hit = {bound}",
        f"band = {selected.frequency_band_low_kHz:.2f}-"
        f"{selected.frequency_band_high_kHz:.1f} kHz",
        "",
        "BACKGROUND",
        f"baseline={selected.baseline:.6f}  contrast={selected.contrast:.6f}",
        f"Trev={selected.revival_time_us:.5f} us  width0={selected.width0_us:.5f} us",
        f"T2={selected.T2_us:.2f} us / limit {selected.T2_limit_us:.1f} us",
        f"beta={selected.beta:.4f}  taper={selected.amp_taper_alpha:.4f}",
        f"width_slope={selected.width_slope:.5f}  chirp={selected.revival_chirp:.6f}",
    ]
    ax_info.text(.00,.99,"\n".join(info),va="top",family="monospace",fontsize=6.8)
    xpos=[.40,.61,.81]
    for j,x in enumerate(xpos,1):
        txt=c13_text(selected,j)
        if txt:ax_info.text(x,.99,txt,va="top",family="monospace",fontsize=6.3)

    ax_model.axis("off")
    ax_model.text(.012,.96,model_text(),va="top",family="monospace",
                  fontsize=6.2,linespacing=1.15)
    eff=[]
    for j in range(1,int(selected.model_order)+1):
        eff.append(scale*finite_float(selected.get(f"c13_{j}_kappa")))
    physical_flag=" | s*kappa>1" if any(x>1 for x in eff if np.isfinite(x)) else ""
    bound_flag=" | scale at bound" if bound else ""
    fig.suptitle(
        f"{field} NV {nv} | V6 BIC N={int(selected.model_order)} {selected.site_key} | "
        f"chi2r={selected.red_chi2:.3f} | "
        f"order evidence={evidence.get('model_order_evidence','')}"
        f"{bound_flag}{physical_flag}", fontsize=13.2)
    fig.subplots_adjust(left=.05,right=.98,bottom=.04,top=.94,hspace=.34,wspace=.28)
    pdf.savefig(fig);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G"],required=True)
    ap.add_argument("--nv",type=str,default=None,
                    help="Optional comma-separated NV subset, e.g. 9,13,114")
    ap.add_argument("--input",type=str,default=None,
                    help="Optional explicit V6 candidate_fits.csv")
    ap.add_argument("--output",type=str,default=None,
                    help="Optional output PDF path")
    args=ap.parse_args()

    base=v6.load_backend(args.field)
    _,ck,_,_=base.discover_paths()
    t,y,e=base.load_data(ck)
    cand=Path(args.input) if args.input else discover_candidate(base,args.field)
    print("candidate:",cand)
    cdf=pd.read_csv(cand)
    req=parse_nv_list(args.nv)
    if req is not None:
        missing=sorted(set(req)-set(cdf.nv_index.astype(int)))
        if missing:raise ValueError(f"NVs not present in candidate file: {missing}")
        nvs=req
    else:
        nvs=sorted(cdf.nv_index.astype(int).unique().tolist())

    if args.output:
        out=Path(args.output)
    else:
        stem=str(cand)
        if stem.endswith("_candidate_fits.csv"):
            stem=stem[:-len("_candidate_fits.csv")]
        suffix="_dashboard_v6"
        if req is not None:
            suffix+="_subset_"+"-".join(map(str,nvs))
        out=Path(stem+suffix+".pdf")
    out.parent.mkdir(parents=True,exist_ok=True)

    print(f"NVs in dashboard: {len(nvs)}")
    print("output:",out)
    with PdfPages(out) as pdf:
        global_summary(pdf,cdf[cdf.nv_index.isin(nvs)].copy(),args.field,base)
        for ii,nv in enumerate(nvs,1):
            plot_nv(pdf,base,args.field,nv,t,y,e,cdf)
            if ii%20==0 or ii==len(nvs):
                print(f"  rendered {ii}/{len(nvs)}")
    print("WROTE",out)


if __name__=="__main__":
    main()
