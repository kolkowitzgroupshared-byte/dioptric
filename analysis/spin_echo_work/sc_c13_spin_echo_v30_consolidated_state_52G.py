"""V30 consolidated 52G model state after V22--V29 diagnostics.

Conservative promotion rules:
* V24 replaces V22 only for its targeted heavy-search NVs.
* V25 background variants are adopted only when best-vs-reduced dBIC <= -6.
* V27/V28 discrete-site changes are accepted only if V29 says replicated_strong.
* replicated_moderate changes are retained as provisional, not final.
* non-replicated search gains revert to the pre-V27 model state.
"""
from __future__ import annotations
import ast, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import sc_c13_spin_echo_v26_residual_error_diagnostics_52G as v26
import sc_c13_spin_echo_v27_residual_driven_add_site_52G as v27
import sc_c13_spin_echo_v28_parsimony_rerank_52G as v28

ROOT22=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v22_full_physical_rerank\2026_09\52G\smax30_tol12_pool8_sub1_topO2_topG4_cap40")
ROOT24=v26.ROOT24
ROOT25=v26.ROOT25
ROOT29=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v29_replication_validation_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v30_consolidated_state_52G\2026_09")
def load_pre_v27():
    w22=pd.read_csv(ROOT22/"v22_winners.csv").set_index("nv_index")
    q22=pd.read_csv(ROOT22/"v22_qc_summary.csv").set_index("nv_index")
    w24=pd.read_csv(ROOT24/"v22_winners.csv").set_index("nv_index")
    q24=pd.read_csv(ROOT24/"v22_qc_summary.csv").set_index("nv_index")
    cur=w22.copy()
    qc=q22.copy()
    source=pd.Series("v22",index=cur.index,dtype=object)
    for nv,row in w24.iterrows():
        cur.loc[int(nv)]=row
        qc.loc[int(nv)]=q24.loc[int(nv)]
        source.loc[int(nv)]="v24"
    return cur.sort_index(),qc.sort_index(),source


def load_background_state(cur):
    p25=pd.read_csv(ROOT25/"v25_background_ablation.csv").set_index("nv_index")
    bics=["reduced_bic","beta_free_bic","taper_free_bic","full_bg_bic"]
    p25["best_bg"]=p25[bics].idxmin(axis=1).str.replace("_bic","",regex=False)
    p25["best_bic"]=p25[bics].min(axis=1)
    p25["dBIC_best_vs_reduced"]=p25.best_bic-p25.reduced_bic
    return p25


def proposal_record(nv,s27,f27,s28,f28):
    rr=s27.loc[nv]
    if nv in s28.index and s28.loc[nv].classification=="lower_order_competitive":
        sr=s28.loc[nv]
        hit=f28[
            (f28.nv_index==nv)
            & (f28.site_key.astype(str)==str(sr.best_lower_site_key))
        ].sort_values("bic").iloc[0]
        return dict(
            source="v28_lower",model_order=int(hit.model_order),
            site_key=str(hit.site_key),theta_json=str(hit.theta_json),
            bic=float(hit.bic),red_chi2=float(hit.redchi2),
            visibility_scale=float(hit.visibility_scale) if pd.notna(hit.visibility_scale) else np.nan,
        )
    hit=f27[
        (f27.nv_index==nv)
        & (f27.new_site_key.astype(str)==str(rr.best_new_site_key))
    ].sort_values("new_bic").iloc[0]
    return dict(
        source="v27_add",model_order=int(hit.new_order),
        site_key=str(hit.new_site_key),theta_json=str(hit.theta_json),
        bic=float(hit.new_bic),red_chi2=float(hit.new_redchi2),
        visibility_scale=float(hit.visibility_scale),
    )
def run():
    OUT.mkdir(parents=True,exist_ok=True)
    cur,qc,base_source=load_pre_v27()
    p25=load_background_state(cur)
    s27=pd.read_csv(v27.OUT/"v27_add_site_summary.csv").set_index("nv_index")
    f27=pd.read_csv(v27.OUT/"v27_add_site_candidate_fits.csv.gz")
    s28=pd.read_csv(v28.OUT/"v28_parsimony_summary.csv").set_index("nv_index")
    f28=pd.read_csv(v28.OUT/"v28_lower_order_candidate_fits.csv.gz")
    s29=pd.read_csv(ROOT29/"v29_replication_summary.csv").set_index("nv_index")

    rows=[]
    for nv,row in cur.iterrows():
        nv=int(nv)
        bg="reduced"; bg_source="production_reduced"; bg_dbic=0.0
        theta_json=str(row.theta_json)
        red=float(row.red_chi2); bic=float(row.bic)
        scale=float(row.visibility_scale) if pd.notna(row.visibility_scale) else np.nan
        state_source=str(base_source.loc[nv])
        discrete_status="pre_v27"
        replication_class="not_tested"

        if nv in p25.index and float(p25.loc[nv].dBIC_best_vs_reduced)<=-6:
            pr=p25.loc[nv]; bg=str(pr.best_bg)
            bg_source="v25_strong_background"
            bg_dbic=float(pr.dBIC_best_vs_reduced)
            theta_json=str(pr[f"{bg}_theta_json"])
            red=float(pr[f"{bg}_redchi2"]); bic=float(pr[f"{bg}_bic"])
            th=np.asarray(json.loads(theta_json),float)
            scale=float(th[9]) if int(row.model_order)>0 else np.nan
            state_source="v25"

        current_order=int(row.model_order)
        current_sites=str(row.site_key)
        search_gain_rejected=False

        if nv in s29.index:
            rep=str(s29.loc[nv].replication_class)
            replication_class=rep
            prop=proposal_record(nv,s27,f27,s28,f28)
            if rep=="replicated_strong":
                discrete_status="accepted_new_assignment"
                state_source=prop["source"]
                current_order=prop["model_order"]; current_sites=prop["site_key"]
                theta_json=prop["theta_json"]; red=prop["red_chi2"]; bic=prop["bic"]
                scale=prop["visibility_scale"]
            elif rep=="replicated_moderate":
                discrete_status="provisional_new_assignment"
                state_source=prop["source"]
                current_order=prop["model_order"]; current_sites=prop["site_key"]
                theta_json=prop["theta_json"]; red=prop["red_chi2"]; bic=prop["bic"]
                scale=prop["visibility_scale"]
            else:
                discrete_status="search_gain_not_replicated"
                search_gain_rejected=True
        flags=str(qc.loc[nv].qc_flags) if nv in qc.index and pd.notna(qc.loc[nv].qc_flags) else ""
        high_scale=bool(np.isfinite(scale) and scale>10)
        bad_fit=bool(red>=3.0)
        bic_ambiguous="bic_ambiguous" in flags.split(";")

        if discrete_status=="accepted_new_assignment":
            confidence="accepted_replication_supported"
        elif discrete_status=="provisional_new_assignment":
            confidence="provisional_replication_supported"
        elif search_gain_rejected and bad_fit:
            confidence="model_inadequate_search_gain_not_replicated"
        elif bad_fit:
            confidence="model_inadequate"
        elif high_scale:
            confidence="amplitude_identifiability_needed"
        elif bic_ambiguous:
            confidence="discrete_assignment_ambiguous"
        else:
            confidence="baseline_supported"

        rows.append(dict(
            nv_index=nv,model_order=current_order,site_key=current_sites,
            state_source=state_source,discrete_status=discrete_status,
            replication_class=replication_class,
            background_model=bg,background_source=bg_source,
            background_dBIC_vs_reduced=bg_dbic,
            bic=bic,red_chi2=red,visibility_scale=scale,
            high_scale_gt10=high_scale,bad_fit_redchi2_ge3=bad_fit,
            prior_qc_flags=flags,confidence_class=confidence,
            theta_json=theta_json,
        ))

    d=pd.DataFrame(rows).sort_values("nv_index")
    d.to_csv(OUT/"v30_consolidated_52G_state.csv",index=False)

    summary=dict(
        n_nvs=int(len(d)),
        order_counts={str(k):int(v) for k,v in d.model_order.value_counts().sort_index().items()},
        confidence_counts={str(k):int(v) for k,v in d.confidence_class.value_counts().items()},
        n_bad_fit=int(d.bad_fit_redchi2_ge3.sum()),
        n_high_scale=int(d.high_scale_gt10.sum()),
        n_accepted_new=int((d.discrete_status=="accepted_new_assignment").sum()),
        n_provisional_new=int((d.discrete_status=="provisional_new_assignment").sum()),
        n_nonreplicated_gain=int((d.discrete_status=="search_gain_not_replicated").sum()),
    )
    with open(OUT/"v30_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print("\nV30 COMPLETE")
    for k,v in summary.items(): print(f"{k}: {v}")
    print("output:",OUT)
    print("\nNon-baseline states:")
    print(d[d.discrete_status!="pre_v27"][
        ["nv_index","model_order","site_key","discrete_status","replication_class",
         "red_chi2","visibility_scale","confidence_class"]
    ].to_string(index=False))


if __name__=="__main__":
    run()
