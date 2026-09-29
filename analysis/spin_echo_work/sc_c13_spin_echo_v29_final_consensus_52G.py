"""V29 consolidated 212-NV QNami 52G consensus after V22-V28b.

Priority of evidence:
  base V22 for all NVs
  -> V24 targeted-heavy replacements for the 54 serious-QC NVs
  -> V25 background ablation when strongly BIC-favored
  -> V26b targeted N=4 upgrades
  -> V28 multiplicative cross-term upgrades when strongly BIC-favored
  -> V27/V28b scale-profile identifiability labels

Outputs one final per-NV table plus summary/PDF.  Different-sample 49G data are
not used.
"""
from __future__ import annotations
import ast,json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo")
V22=ROOT/r"c13_spin_echo_v22_full_physical_rerank\2026_09\52G\smax30_tol12_pool8_sub1_topO2_topG4_cap40"
V24=ROOT/r"spin_echo_v24_targeted_heavy_52G\2026_09"
V25=ROOT/r"spin_echo_v25_background_ablation_52G\2026_09"
V26=ROOT/r"spin_echo_v26_residual_error_diagnostics_52G\2026_09"
V26B=ROOT/r"spin_echo_v26b_targeted_n4_52G\2026_09"
V27=ROOT/r"spin_echo_v27_scale_profile_52G\2026_09"
V28=ROOT/r"spin_echo_v28_nonlinear_eSEEM_52G\2026_09"
V28B=ROOT/r"spin_echo_v28b_cross_scale_profile_52G\2026_09"
OUT=ROOT/r"spin_echo_v29_final_consensus_52G\2026_09"
def canon(x):
    try:
        a=ast.literal_eval(str(x))
        return tuple(sorted(int(v) for v in a))
    except Exception:
        return ()


def site_evidence(margin,order):
    if int(order)==0:
        return "no_site"
    if not np.isfinite(margin):
        return "robust"
    if margin>=6: return "robust"
    if margin>=2: return "moderate"
    return "ambiguous"


def fit_status(red):
    if red<=2: return "good"
    if red<=3: return "marginal"
    return "poor"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    w22=pd.read_csv(V22/"v22_winners.csv").set_index("nv_index")
    q22=pd.read_csv(V22/"v22_qc_summary.csv").set_index("nv_index")
    w24=pd.read_csv(V24/"v22_winners.csv").set_index("nv_index")
    q24=pd.read_csv(V24/"v22_qc_summary.csv").set_index("nv_index")
    v25=pd.read_csv(V25/"v25_background_ablation.csv").set_index("nv_index")
    v26=pd.read_csv(V26/"v26_residual_summary.csv").set_index("nv_index")
    n4all=pd.read_csv(V26B/"v26b_targeted_n4_candidates.csv")
    n4best=pd.read_csv(V26B/"v26b_targeted_n4_best.csv").set_index("nv_index")
    v27=pd.read_csv(V27/"v27_scale_profile_summary.csv").set_index("nv_index")
    v28=pd.read_csv(V28/"v28_nonlinear_eSEEM_summary.csv").set_index("nv_index")
    crossprof=pd.read_csv(V28B/"v28b_cross_scale_profiles.csv")
    v28b=pd.read_csv(V28B/"v28b_cross_scale_summary.csv").set_index("nv_index")
    records=[]
    for nv in w22.index.astype(int):
        use24=nv in w24.index
        win=w24.loc[nv] if use24 else w22.loc[nv]
        qc=q24.loc[nv] if use24 else q22.loc[nv]
        rec=dict(
            nv_index=nv,base_source="V24_targeted_heavy" if use24 else "V22_full",
            final_source="V24_targeted_heavy" if use24 else "V22_full",
            final_model_form="additive_shared",
            final_background="beta2_taper0",
            final_model_order=int(win.model_order),
            final_site_key=str(canon(win.site_key)),
            orientation=str(win.orientation),
            final_bic=float(win.bic),final_redchi2=float(win.red_chi2),
            final_scale=float(win.visibility_scale) if pd.notna(win.visibility_scale) else np.nan,
            candidate_bic_margin=float(qc.delta_bic_to_second),
            original_qc_flags=str(qc.qc_flags) if pd.notna(qc.qc_flags) else "",
            special_delta_bic=np.nan,special_note="",
            background_delta_bic=np.nan,
            file_median_birge=np.nan,
            n4_added_site=np.nan,
        )

        # Evidence-supported background relaxation on the fixed V24 assignment.
        if nv in v25.index:
            q=v25.loc[nv]
            models=["reduced","beta_free","taper_free","full_bg"]
            best=min(models,key=lambda m:float(q[f"{m}_bic"]))
            db=float(q[f"{best}_bic"]-q["reduced_bic"])
            rec["background_delta_bic"]=db
            if db<=-6:
                rec["final_background"]=best
                rec["final_bic"]=float(q[f"{best}_bic"])
                rec["final_redchi2"]=float(q[f"{best}_redchi2"])
                rec["final_source"]="V25_background"
                rec["special_note"]=f"background {best} favored"
        if nv in v26.index:
            rec["file_median_birge"]=float(v26.loc[nv].median_birge)

        records.append(rec)

    d=pd.DataFrame(records).set_index("nv_index")
    # Strong targeted N=4 upgrades.
    for nv,q in n4best.iterrows():
        nv=int(nv)
        if not bool(q.n4_strongly_supported):
            continue
        old=canon(q.incumbent_site_key)
        new=tuple(sorted(old+(int(q.added_site_id),)))
        group=n4all[n4all.nv_index==nv].sort_values("n4_bic")
        margin=float(group.iloc[1].n4_bic-group.iloc[0].n4_bic) if len(group)>1 else np.inf
        d.loc[nv,"final_source"]="V26b_targeted_N4"
        d.loc[nv,"final_model_form"]="additive_shared_N4"
        bg_label=str(q.background)
        if bg_label=="reduced":
            bg_label="beta2_taper0"
        d.loc[nv,"final_background"]=bg_label
        d.loc[nv,"final_model_order"]=4
        d.loc[nv,"final_site_key"]=str(new)
        d.loc[nv,"final_bic"]=float(q.n4_bic)
        d.loc[nv,"final_redchi2"]=float(q.n4_redchi2)
        d.loc[nv,"final_scale"]=float(q.visibility_scale)
        d.loc[nv,"candidate_bic_margin"]=margin
        d.loc[nv,"special_delta_bic"]=float(q.delta_bic_n4_vs_n3)
        d.loc[nv,"special_note"]=f"N4 strongly favored; added site {int(q.added_site_id)}"
        d.loc[nv,"n4_added_site"]=int(q.added_site_id)

    # V28 cross-term upgrades; exact-product was never strongly favored.
    for nv,q in v28.iterrows():
        nv=int(nv)
        if not bool(q.cross_strongly_favored):
            continue
        d.loc[nv,"final_source"]="V28_cross_term"
        d.loc[nv,"final_model_form"]="multiplicative_cross"
        d.loc[nv,"final_background"]="beta2_taper0"
        d.loc[nv,"final_bic"]=float(q.cross_bic)
        d.loc[nv,"final_redchi2"]=float(q.cross_redchi2)
        d.loc[nv,"final_scale"]=float(q.cross_scale)
        d.loc[nv,"special_delta_bic"]=float(q.delta_bic_cross_vs_add)
        d.loc[nv,"special_note"]="phase-preserving cross terms strongly favored"
    # Amplitude-identifiability status.
    d["amplitude_status"]="not_flagged"
    d["scale95_low"]=np.nan; d["scale95_high"]=np.nan
    d["line_gt_contrast_required_95"]=False
    for nv,q in v27.iterrows():
        nv=int(nv)
        if nv in v28b.index and d.loc[nv,"final_model_form"]=="multiplicative_cross":
            cp=crossprof[crossprof.nv_index==nv].copy()
            cp["dchi"]=cp.chi2-cp.chi2.min()
            ok=cp[cp.dchi<=3.84]
            lo=float(ok.scale.min()); hi=float(ok.scale.max())
            # Reuse physical scale limit from additive-site geometry.
            lim=float(q.scale_for_line_over_contrast_1)
            gp=cp[cp.scale<=lim]
            dphys=float(gp.dchi.min()) if len(gp) else np.inf
            high=bool(lo>10); line=bool(dphys>3.84)
        else:
            lo=float(q.scale95_low); hi=float(q.scale95_high)
            high=bool(q.high_scale_required_95)
            line=bool(q.line_gt_contrast_required_95)
        d.loc[nv,"scale95_low"]=lo; d.loc[nv,"scale95_high"]=hi
        d.loc[nv,"line_gt_contrast_required_95"]=line
        if high and line:
            status="large_modulation_required"
        elif high:
            status="large_scale_low_kappa"
        else:
            status="scale_ambiguous_moderate_allowed"
        d.loc[nv,"amplitude_status"]=status

    d["fit_status"]=[fit_status(x) for x in d.final_redchi2]
    d["site_evidence"]=[
        site_evidence(m,o) for m,o in zip(d.candidate_bic_margin,d.final_model_order)
    ]
    # Combined confidence class: fit adequacy first, then amplitude, then assignment.
    classes=[]
    reasons=[]
    for nv,r in d.iterrows():
        if r.fit_status=="poor":
            c="D_model_inadequate"
            why=f"redchi2={r.final_redchi2:.2f}>3 after validated refinements"
        elif int(r.final_model_order)==0:
            c="E_no_resolved_C13"
            why="N=0 preferred with acceptable fit"
        elif r.amplitude_status in (
            "large_modulation_required","large_scale_low_kappa",
            "scale_ambiguous_moderate_allowed",
        ):
            c="B_amplitude_limited"
            why=r.amplitude_status
        elif r.site_evidence!="robust":
            c="C_assignment_ambiguous"
            why=f"candidate BIC margin {r.candidate_bic_margin:.2f}"
        else:
            c="A_robust_assignment"
            why="adequate fit, robust candidate margin, no amplitude warning"
        classes.append(c); reasons.append(why)
    d["confidence_class"]=classes
    d["confidence_reason"]=reasons
    d=d.reset_index()
    d.to_csv(OUT/"v29_final_consensus_212NV.csv",index=False)

    counts=d.confidence_class.value_counts()
    metadata={
        "n_nv":int(len(d)),
        "confidence_counts":{str(k):int(v) for k,v in counts.items()},
        "final_model_forms":{str(k):int(v) for k,v in d.final_model_form.value_counts().items()},
        "N4_upgrades":d[d.final_model_form=="additive_shared_N4"].nv_index.astype(int).tolist(),
        "cross_term_upgrades":d[d.final_model_form=="multiplicative_cross"].nv_index.astype(int).tolist(),
        "poor_fit_nvs":d[d.fit_status=="poor"].nv_index.astype(int).tolist(),
        "note_49G":"Johnson 49G excluded because it is a different physical sample.",
    }
    with open(OUT/"v29_metadata.json","w",encoding="utf-8") as f:
        json.dump(metadata,f,indent=2)
    with PdfPages(OUT/"v29_final_consensus_report.pdf") as pdf:
        fig,ax=plt.subplots(1,2,figsize=(11,5))
        ax[0].bar(np.arange(len(counts)),counts.values)
        ax[0].set_xticks(np.arange(len(counts)),counts.index,rotation=30,ha="right")
        ax[0].set(title="Final confidence classes",ylabel="NV count")
        mc=d.final_model_form.value_counts()
        ax[1].bar(np.arange(len(mc)),mc.values)
        ax[1].set_xticks(np.arange(len(mc)),mc.index,rotation=25,ha="right")
        ax[1].set(title="Final model forms",ylabel="NV count")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        special=d[d.final_source.isin(["V25_background","V26b_targeted_N4","V28_cross_term"])]
        for start in range(0,max(1,len(special)),20):
            q=special.iloc[start:start+20]
            if q.empty: break
            cols=["nv_index","final_source","final_model_order","final_site_key",
                  "final_background","final_redchi2","special_delta_bic","confidence_class"]
            fig,ax=plt.subplots(figsize=(11,8.5)); ax.axis("off")
            show=q[cols].copy()
            for c in ("final_redchi2","special_delta_bic"):
                show[c]=show[c].map(lambda x:"" if pd.isna(x) else f"{x:.2f}")
            tab=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center")
            tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1,1.3)
            ax.set_title("Evidence-supported model changes")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for start in range(0,len(d),30):
            q=d.iloc[start:start+30]
            cols=["nv_index","final_model_order","final_site_key","final_redchi2",
                  "site_evidence","amplitude_status","confidence_class"]
            fig,ax=plt.subplots(figsize=(11,8.5)); ax.axis("off")
            show=q[cols].copy()
            show["final_redchi2"]=show.final_redchi2.map(lambda x:f"{x:.2f}")
            tab=ax.table(cellText=show.values,colLabels=show.columns,loc="center",cellLoc="center")
            tab.auto_set_font_size(False); tab.set_fontsize(6.5); tab.scale(1,1.2)
            ax.set_title("Final 212-NV consensus")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print("\nV29 FINAL CONSENSUS COMPLETE")
    print("confidence classes:",metadata["confidence_counts"])
    print("model forms:",metadata["final_model_forms"])
    print("N4 upgrades:",metadata["N4_upgrades"])
    print("cross-term upgrades:",metadata["cross_term_upgrades"])
    print("poor-fit count:",len(metadata["poor_fit_nvs"]))
    print("poor-fit NVs:",metadata["poor_fit_nvs"])
    print("output:",OUT)


if __name__=="__main__":
    main()
