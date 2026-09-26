"""V13 diagnostic: combine the two best-supported V12 simplifications.

Compare, on the exact same representative 10+10 NV set:
  M0: polished full V6 background
  M1: V6 with beta fixed to 2 and revival taper fixed to 0

All discrete-C13 sites and the V6 discrete architecture are unchanged.
All remaining parameters are refit.

The pair is optimized iteratively so the unrestricted full model is guaranteed
to have chi2 <= the nested simplified model.  This avoids the local-minimum
artifact exposed by V12.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v7_diagnostic as v7
import sc_spin_echo_v8_physical_bath_diagnostic as v8
import sc_spin_echo_v10_lattice_bath_diagnostic as v10
import sc_spin_echo_v12_v6_background_ablation as v12

FIXED_SIMPLE={5:2.0, 6:0.0}  # beta=2, taper=0
MODEL_SIMPLE="beta2_taper0"


def nested_pair_fit(base,t,y,e,row,sites,trev_phys,max_cycles=5,tol=1e-7):
    """Fit full and simplified models with broad V12 basin seeding.

    The comparison is still only full V6 vs beta=2,taper=0.  Other V12
    ablations are used only as deterministic seeds to locate better minima.
    """
    specs=v12.model_specs(trev_phys)

    # Basin discovery: use every V12 constrained family only as a seed source.
    prelim=[]
    for _name,fixed,_desc in specs:
        q=v12.fit_ablation(base,t,y,e,row,sites,fixed)
        if q is not None:
            prelim.append(q)
    if not prelim:
        return None,None,0

    qfull=v12.fit_ablation(
        base,t,y,e,row,sites,{},
        extra_full_seeds=[q["theta"] for q in prelim])
    qsimple=v12.fit_ablation(
        base,t,y,e,row,sites,FIXED_SIMPLE,
        extra_full_seeds=[q["theta"] for q in prelim] +
                         ([qfull["theta"]] if qfull is not None else []))
    if qfull is None or qsimple is None:
        return None,None,0

    # Alternate the two nested models until the ordering is closed.
    cycles=0
    for cycles in range(1,max_cycles+1):
        qs=v12.fit_ablation(
            base,t,y,e,row,sites,FIXED_SIMPLE,
            extra_full_seeds=[qfull["theta"],qsimple["theta"]] +
                             [q["theta"] for q in prelim])
        if qs is not None and qs["chi2"] < qsimple["chi2"]:
            qsimple=qs

        qf=v12.fit_ablation(
            base,t,y,e,row,sites,{},
            extra_full_seeds=[qfull["theta"],qsimple["theta"]] +
                             [q["theta"] for q in prelim])
        if qf is not None and qf["chi2"] < qfull["chi2"]:
            qfull=qf

        if qsimple["chi2"] >= qfull["chi2"]-tol:
            break
    return qfull,qsimple,cycles


def evaluate(field,outdir,n_each=10):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths()
    t,y,e=base.load_data(ck)
    cand=v7.latest_v6_candidate(base,field)
    cdf=pd.read_csv(cand)
    bestdf=cdf[cdf.rank_global_bic==1].sort_values("nv_index")
    nvs,roles=v7.choose_subset(bestdf,field,n_each)

    fL=v8.catalog_larmor_khz(base)
    trev_phys=2000.0/fL
    rows=[];details=[]
    print(f"\n{field}: V13 beta=2, taper=0 combined diagnostic")
    print(f"  broad basin seeds from V12; Trev_phys={trev_phys:.4f} us")
    for nv in nvs:
        row=bestdf[bestdf.nv_index==nv].iloc[0]
        sites=v10.sites_from_row(row)
        yy=np.asarray(y[nv],float); ee=base.safe_err(e[nv])

        qfull,qsimple,cycles=nested_pair_fit(
            base,t,yy,ee,row,sites,trev_phys)
        if qfull is None or qsimple is None:
            raise RuntimeError(f"{field} NV{nv}: pair fit failed")
        if qsimple["chi2"] < qfull["chi2"]-1e-5:
            raise RuntimeError(
                f"nested consistency failure {field} NV{nv}: "
                f"simple-full={qsimple['chi2']-qfull['chi2']}"
            )

        # Saved V6 reference, to identify optimization improvements.
        import json
        saved_th=np.asarray(json.loads(row.theta_json),float)
        saved_pred=v6.v6_model(base,t,saved_th,sites)
        saved=v7.stats(yy,ee,saved_pred,int(row.npar))

        rf=(yy-qfull["pred"])/ee
        rs=(yy-qsimple["pred"])/ee
        thf=qfull["theta"]; ths=qsimple["theta"]
        dchi=float(qsimple["chi2"]-qfull["chi2"])
        dbic=float(qsimple["bic"]-qfull["bic"])
        rec=dict(
            field=field,nv_index=int(nv),role=roles[nv],
            model_order=int(row.model_order),site_key=str(row.site_key),
            cycles=int(cycles),
            saved_v6_chi2=float(saved["chi2"]),
            full_chi2=float(qfull["chi2"]),full_redchi=float(qfull["red_chi2"]),
            full_bic=float(qfull["bic"]),full_npar=int(qfull["npar"]),
            simple_chi2=float(qsimple["chi2"]),simple_redchi=float(qsimple["red_chi2"]),
            simple_bic=float(qsimple["bic"]),simple_npar=int(qsimple["npar"]),
            delta_chi2=dchi,delta_bic=dbic,
            saved_to_full_delta_chi2=float(qfull["chi2"]-saved["chi2"]),
            full_resid_rms=float(np.sqrt(np.mean(rf*rf))),
            simple_resid_rms=float(np.sqrt(np.mean(rs*rs))),
            full_Trev_us=float(thf[2]),simple_Trev_us=float(ths[2]),
            full_width0_us=float(thf[3]),simple_width0_us=float(ths[3]),
            full_T2_us=float(1000*thf[4]),simple_T2_us=float(1000*ths[4]),
            full_beta=float(thf[5]),simple_beta=float(ths[5]),
            full_taper=float(thf[6]),simple_taper=float(ths[6]),
            full_slope=float(thf[7]),simple_slope=float(ths[7]),
            full_chirp=float(thf[8]),simple_chirp=float(ths[8]),
            full_scale=(float(thf[9]) if int(row.model_order)>0 else np.nan),
            simple_scale=(float(ths[9]) if int(row.model_order)>0 else np.nan),
        )
        rows.append(rec)
        details.append(dict(row=row,sites=sites,y=yy,e=ee,
                            full=qfull,simple=qsimple,role=roles[nv]))
        print(
            f"  NV{nv:3d} {roles[nv]:12s} N={int(row.model_order)} "
            f"chi2r {qfull['red_chi2']:.3f}->{qsimple['red_chi2']:.3f} "
            f"dchi2={dchi:+.2f} dBIC={dbic:+.2f} "
            f"saved->full={qfull['chi2']-saved['chi2']:+.2f}"
        )

    tab=pd.DataFrame(rows)
    summary=make_summary(tab,field,len(t))
    residual=common_residual_table(t,details,field)
    stem=f"v13_beta2_taper0_{field}"
    tab.to_csv(outdir/f"{stem}_per_nv.csv",index=False)
    summary.to_csv(outdir/f"{stem}_summary.csv",index=False)
    residual.to_csv(outdir/f"{stem}_common_residual.csv",index=False)
    make_pdf(outdir/f"{stem}.pdf",field,t,tab,summary,residual,details)
    return dict(field=field,table=tab,summary=summary,residual=residual)


def make_summary(tab,field,npoints):
    # M1 has exactly two fewer free parameters than M0.
    bic_reward=2.0*np.log(float(npoints))
    return pd.DataFrame([dict(
        field=field,n_nv=len(tab),npoints=int(npoints),
        parameters_removed=2,bic_reward_for_two_removed=float(bic_reward),
        bic_better_count=int((tab.delta_bic<0).sum()),
        bic_strong_better_count=int((tab.delta_bic<=-6).sum()),
        bic_ambiguous_count=int((tab.delta_bic.abs()<2).sum()),
        bic_strong_worse_count=int((tab.delta_bic>=6).sum()),
        median_delta_chi2=float(tab.delta_chi2.median()),
        mean_delta_chi2=float(tab.delta_chi2.mean()),
        max_delta_chi2=float(tab.delta_chi2.max()),
        median_delta_bic=float(tab.delta_bic.median()),
        mean_delta_bic=float(tab.delta_bic.mean()),
        median_full_redchi=float(tab.full_redchi.median()),
        median_simple_redchi=float(tab.simple_redchi.median()),
        median_resid_rms_change=float(
            (tab.simple_resid_rms-tab.full_resid_rms).median()),
        scale_bound_full_count=int((tab.full_scale>=2.94).fillna(False).sum()),
        scale_bound_simple_count=int((tab.simple_scale>=2.94).fillna(False).sum()),
        optimizer_improved_saved_count=int((tab.saved_to_full_delta_chi2<-1e-3).sum()),
    )])


def common_residual_table(t,details,field):
    RF=[];RS=[]
    for z in details:
        RF.append((z["y"]-z["full"]["pred"])/z["e"])
        RS.append((z["y"]-z["simple"]["pred"])/z["e"])
    RF=np.asarray(RF);RS=np.asarray(RS)
    return pd.DataFrame(dict(
        field=field,time_us=np.asarray(t,float),
        full_median=np.nanmedian(RF,axis=0),
        simple_median=np.nanmedian(RS,axis=0),
        full_mean=np.nanmean(RF,axis=0),
        simple_mean=np.nanmean(RS,axis=0),
    ))

def make_pdf(path,field,t,tab,summary,residual,details):
    with PdfPages(path) as pdf:
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(
            f"{field}: V6 vs beta=2, taper=0 (same discrete C13 sites)",
            fontsize=15)
        x=np.arange(len(tab))
        axs[0,0].bar(x,tab.delta_bic)
        axs[0,0].axhline(0,lw=.8);axs[0,0].axhline(6,ls="--",lw=.8)
        axs[0,0].axhline(-6,ls="--",lw=.8)
        axs[0,0].set_xticks(x)
        axs[0,0].set_xticklabels(tab.nv_index.astype(str),rotation=60)
        axs[0,0].set(xlabel="NV",ylabel="Delta BIC (simple-full)",
                     title="Model evidence")
        axs[0,0].grid(alpha=.15,axis="y")

        axs[0,1].bar(x,tab.delta_chi2)
        axs[0,1].axhline(2*np.log(summary.npoints.iloc[0]),ls="--",lw=.9,
                         label="BIC break-even")
        axs[0,1].set_xticks(x)
        axs[0,1].set_xticklabels(tab.nv_index.astype(str),rotation=60)
        axs[0,1].set(xlabel="NV",ylabel="Raw Delta chi-square",
                     title="Fit loss from removing beta+taper")
        axs[0,1].legend(fontsize=8);axs[0,1].grid(alpha=.15,axis="y")

        axs[1,0].plot(t,residual.full_median,label="full V6",lw=1.4)
        axs[1,0].plot(t,residual.simple_median,label="beta=2,taper=0",lw=1.4)
        axs[1,0].axhline(0,lw=.7)
        axs[1,0].set(xlabel="Total evolution time (us)",
                     ylabel="Median normalized residual",
                     title="Common residual across representative NVs")
        axs[1,0].legend();axs[1,0].grid(alpha=.15)

        axs[1,1].axis("off")
        s=summary.iloc[0]
        lines=[
            f"N NVs = {int(s.n_nv)}",
            f"parameters removed = {int(s.parameters_removed)}",
            f"BIC reward = {s.bic_reward_for_two_removed:.3f}",
            f"BIC better = {int(s.bic_better_count)}/{int(s.n_nv)}",
            f"strongly better = {int(s.bic_strong_better_count)}/{int(s.n_nv)}",
            f"strongly worse = {int(s.bic_strong_worse_count)}/{int(s.n_nv)}",
            f"median dchi2 = {s.median_delta_chi2:.3f}",
            f"median dBIC = {s.median_delta_bic:.3f}",
            f"median chi2r full/simple = "
            f"{s.median_full_redchi:.3f}/{s.median_simple_redchi:.3f}",
            f"scale-bound full/simple = "
            f"{int(s.scale_bound_full_count)}/{int(s.scale_bound_simple_count)}",
            "",
            "Simple model fixes:",
            "  beta = 2",
            "  revival taper = 0",
            "All other V6 parameters remain free.",
        ]
        axs[1,1].text(.02,.98,"\n".join(lines),va="top",
                      family="monospace",fontsize=9)
        fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

        # Parameter movement page.
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(f"{field}: parameter compensation after simplification",fontsize=15)
        pairs=[
            ("Trev_us","Trev (us)"),
            ("width0_us","width0 (us)"),
            ("T2_us","T2 (us)"),
            ("slope","width slope"),
        ]
        for ax,(key,label) in zip(axs.ravel(),pairs):
            a=tab[f"full_{key}"];b=tab[f"simple_{key}"]
            ax.scatter(a,b,c=tab.model_order,s=55)
            lo=min(a.min(),b.min());hi=max(a.max(),b.max())
            ax.plot([lo,hi],[lo,hi],ls="--",lw=1)
            ax.set(xlabel=f"full {label}",ylabel=f"simple {label}")
            ax.grid(alpha=.15)
        fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)

        # Per-NV pages.
        for rec,z in zip(tab.to_dict("records"),details):
            fig,axs=plt.subplots(2,2,figsize=(11,8.5))
            row=z["row"]
            fig.suptitle(
                f"{field} NV {rec['nv_index']} | {rec['role']} | "
                f"N={rec['model_order']} {rec['site_key']}",fontsize=14)
            axs[0,0].errorbar(t,z["y"],yerr=z["e"],fmt="o",ms=3,lw=.4,label="data")
            axs[0,0].plot(t,z["full"]["pred"],label="polished full V6",lw=1.4)
            axs[0,0].plot(t,z["simple"]["pred"],label="beta=2,taper=0",lw=1.4)
            axs[0,0].set(xlabel="Time (us)",ylabel="Signal")
            axs[0,0].legend(fontsize=7);axs[0,0].grid(alpha=.15)

            rf=(z["y"]-z["full"]["pred"])/z["e"]
            rs=(z["y"]-z["simple"]["pred"])/z["e"]
            axs[0,1].plot(t,rf,"o-",ms=2,lw=.7,label="full")
            axs[0,1].plot(t,rs,"o-",ms=2,lw=.7,label="simple")
            axs[0,1].axhline(0,lw=.7)
            axs[0,1].set(xlabel="Time (us)",ylabel="Normalized residual")
            axs[0,1].legend(fontsize=7);axs[0,1].grid(alpha=.15)

            axs[1,0].bar([0,1],[rec["full_redchi"],rec["simple_redchi"]])
            axs[1,0].set_xticks([0,1]);axs[1,0].set_xticklabels(["full","simple"])
            axs[1,0].set_ylabel("Reduced chi-square")
            axs[1,0].set_title(
                f"dchi2={rec['delta_chi2']:+.2f}, dBIC={rec['delta_bic']:+.2f}")
            axs[1,0].grid(alpha=.15,axis="y")

            axs[1,1].axis("off")
            lines=[
                f"saved->polished dchi2 = {rec['saved_to_full_delta_chi2']:+.3f}",
                f"full/simple Trev = {rec['full_Trev_us']:.3f} / "
                f"{rec['simple_Trev_us']:.3f} us",
                f"full/simple width0 = {rec['full_width0_us']:.3f} / "
                f"{rec['simple_width0_us']:.3f} us",
                f"full/simple T2 = {rec['full_T2_us']:.2f} / "
                f"{rec['simple_T2_us']:.2f} us",
                f"full beta = {rec['full_beta']:.3f} -> fixed 2",
                f"full taper = {rec['full_taper']:.3f} -> fixed 0",
                f"full/simple slope = {rec['full_slope']:.3f} / "
                f"{rec['simple_slope']:.3f}",
                f"full/simple chirp = {rec['full_chirp']:+.5f} / "
                f"{rec['simple_chirp']:+.5f}",
            ]
            if rec["model_order"]>0:
                lines.append(
                    f"full/simple sNV = {rec['full_scale']:.3f} / "
                    f"{rec['simple_scale']:.3f}")
            axs[1,1].text(.02,.98,"\n".join(lines),va="top",
                          family="monospace",fontsize=8.5)
            fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--n-each",type=int,default=10)
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    ym=datetime.now().strftime("%Y_%m")
    outdir=(Path(args.output_dir) if args.output_dir else
            Path(r"G:\nvdata\pc_NVOffice\branch_master")/
            "spin_echo_v13_beta2_taper0"/ym)
    outdir.mkdir(parents=True,exist_ok=True)

    fields=["49G","52G"] if args.field=="both" else [args.field]
    res=[evaluate(f,outdir,args.n_each) for f in fields]
    if len(res)>1:
        pd.concat([x["table"] for x in res],ignore_index=True).to_csv(
            outdir/"v13_beta2_taper0_both_per_nv.csv",index=False)
        pd.concat([x["summary"] for x in res],ignore_index=True).to_csv(
            outdir/"v13_beta2_taper0_both_summary.csv",index=False)
        pd.concat([x["residual"] for x in res],ignore_index=True).to_csv(
            outdir/"v13_beta2_taper0_both_common_residual.csv",index=False)

    print("\nCOMPLETE")
    for x in res:
        print("\n",x["field"])
        print(x["summary"].to_string(index=False))


if __name__=="__main__":
    main()
