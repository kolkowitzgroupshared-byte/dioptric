"""V23 cross-field consistency analysis for completed V22 49G/52G fits.

No nonlinear refitting. For each NV present in both fields:
  * compare raw V22 winners
  * compare model order and canonical lattice-site sets
  * find candidate site sets represented in BOTH fields
  * compute per-field DeltaBIC relative to each field's own winner
  * rank shared assignments by joint DeltaBIC = DeltaBIC_49 + DeltaBIC_52
  * report reciprocal support for each field's winner in the other field

The joint score is relative evidence across independent datasets, not a new
single-field BIC.
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

ROOT=Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v22_full_physical_rerank\2026_09"
)
CONFIG="smax30_tol12_pool8_sub1_topO2_topG4_cap40"
DEFAULT_OUT=Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v23_cross_field_consistency\2026_09"
)
def canon(value):
    if value is None or (isinstance(value,float) and np.isnan(value)):
        return ()
    try:
        x=ast.literal_eval(str(value))
    except Exception:
        x=value
    if isinstance(x,(tuple,list,np.ndarray)):
        return tuple(sorted(int(v) for v in x))
    s=str(value).strip()
    if s in ("","()","nan","None"): return ()
    return tuple(sorted(int(v) for v in s.strip("()[] ").split(",") if str(v).strip()))


def load_field(field,config):
    root=ROOT/field/config
    cand=pd.read_csv(root/"v22_candidate_fits.csv.gz")
    qc=pd.read_csv(root/"v22_qc_summary.csv")
    if "site_key_canonical" not in cand:
        cand["site_key_canonical"]=cand.site_key.map(lambda x:str(canon(x)))
    else:
        cand["site_key_canonical"]=cand.site_key_canonical.map(lambda x:str(canon(x)))
    cand["cross_key"]=cand.model_order.astype(int).astype(str)+"|"+cand.site_key_canonical
    return root,cand,qc


def best_by_key(g):
    q=g.sort_values("bic").drop_duplicates("cross_key",keep="first").copy()
    best=float(q.bic.min())
    q["delta_bic"]=q.bic-best
    return q.set_index("cross_key",drop=False)


def winner_row(g):
    return g.sort_values("bic").iloc[0]


def overlap_metrics(a,b):
    A=set(canon(a)); B=set(canon(b))
    inter=len(A&B); union=len(A|B)
    return inter,(inter/union if union else 1.0)
def analyze(c49,c52):
    common=sorted(set(c49.nv_index.astype(int)) & set(c52.nv_index.astype(int)))
    rows=[]; shared_rows=[]
    for nv in common:
        a=best_by_key(c49[c49.nv_index==nv])
        b=best_by_key(c52[c52.nv_index==nv])
        w49=winner_row(a); w52=winner_row(b)
        shared=sorted(set(a.index)&set(b.index))
        ranked=[]
        for key in shared:
            r49=a.loc[key]; r52=b.loc[key]
            joint=float(r49.delta_bic+r52.delta_bic)
            ranked.append((joint,key,r49,r52))
        ranked.sort(key=lambda z:z[0])

        if ranked:
            joint,key,cr49,cr52=ranked[0]
            consensus_order=int(cr49.model_order)
            consensus_sites=str(canon(cr49.site_key_canonical))
            d49=float(cr49.delta_bic); d52=float(cr52.delta_bic)
        else:
            joint=np.nan; consensus_order=-1; consensus_sites=""
            d49=np.nan; d52=np.nan

        key49=str(w49.cross_key); key52=str(w52.cross_key)
        d49win_in52=float(b.loc[key49].delta_bic) if key49 in b.index else np.nan
        d52win_in49=float(a.loc[key52].delta_bic) if key52 in a.index else np.nan
        overlap,jacc=overlap_metrics(w49.site_key_canonical,w52.site_key_canonical)

        exact=bool(key49==key52)
        same_order=bool(int(w49.model_order)==int(w52.model_order))
        if exact:
            status="exact_winner_agreement"
        elif ranked and d49<=6 and d52<=6:
            status="shared_assignment_near_best_both"
        elif ranked:
            status="shared_assignment_but_field_tension"
        else:
            status="no_shared_candidate"
        rows.append(dict(
            nv_index=nv,status=status,
            exact_winner_agreement=exact,same_winner_order=same_order,
            winner_49_order=int(w49.model_order),
            winner_49_sites=str(canon(w49.site_key_canonical)),
            winner_49_bic=float(w49.bic),
            winner_49_margin=float(
                a.sort_values("bic").iloc[1].bic-w49.bic if len(a)>1 else np.inf
            ),
            winner_52_order=int(w52.model_order),
            winner_52_sites=str(canon(w52.site_key_canonical)),
            winner_52_bic=float(w52.bic),
            winner_52_margin=float(
                b.sort_values("bic").iloc[1].bic-w52.bic if len(b)>1 else np.inf
            ),
            winner_site_overlap=overlap,winner_site_jaccard=jacc,
            deltaBIC_49winner_in_52=d49win_in52,
            deltaBIC_52winner_in_49=d52win_in49,
            consensus_order=consensus_order,consensus_sites=consensus_sites,
            consensus_deltaBIC_49=d49,consensus_deltaBIC_52=d52,
            consensus_joint_deltaBIC=joint,
            num_shared_candidates=len(shared),
        ))

        for rank,(joint,key,r49,r52) in enumerate(ranked[:10],1):
            shared_rows.append(dict(
                nv_index=nv,joint_rank=rank,cross_key=key,
                model_order=int(r49.model_order),
                sites=str(canon(r49.site_key_canonical)),
                deltaBIC_49=float(r49.delta_bic),
                deltaBIC_52=float(r52.delta_bic),
                joint_deltaBIC=float(joint),
                bic_49=float(r49.bic),bic_52=float(r52.bic),
            ))
    return pd.DataFrame(rows),pd.DataFrame(shared_rows)
def attach_qc(summary,q49,q52):
    keep=[
        "nv_index","red_chi2","shared_scale","max_line_to_contrast",
        "amp_diag_delta_bic","delta_bic_to_second","qc_flags"
    ]
    a=q49[[c for c in keep if c in q49.columns]].copy()
    b=q52[[c for c in keep if c in q52.columns]].copy()
    a=a.rename(columns={c:f"{c}_49" for c in a.columns if c!="nv_index"})
    b=b.rename(columns={c:f"{c}_52" for c in b.columns if c!="nv_index"})
    return summary.merge(a,on="nv_index",how="left").merge(b,on="nv_index",how="left")


def report(path,summary):
    with PdfPages(path) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(11,8.5))
        counts=summary.status.value_counts()
        axes[0,0].bar(np.arange(len(counts)),counts.values)
        axes[0,0].set_xticks(np.arange(len(counts)),counts.index,rotation=25,ha="right")
        axes[0,0].set(title="Cross-field assignment status",ylabel="NV count")

        axes[0,1].bar([0,1],[
            int(summary.same_winner_order.sum()),
            int(summary.exact_winner_agreement.sum())
        ])
        axes[0,1].set_xticks([0,1],["same order","exact same sites"])
        axes[0,1].set(title=f"Agreement across {len(summary)} common NVs",ylabel="count")

        x=summary.consensus_deltaBIC_49
        y=summary.consensus_deltaBIC_52
        m=np.isfinite(x)&np.isfinite(y)
        axes[1,0].scatter(x[m],y[m],s=14)
        axes[1,0].axvline(6,ls="--",lw=.8); axes[1,0].axhline(6,ls="--",lw=.8)
        axes[1,0].set(xlabel="Consensus ΔBIC in 49G",ylabel="Consensus ΔBIC in 52G",
                      title="Best shared assignment support")

        vals=summary.consensus_joint_deltaBIC.replace([np.inf,-np.inf],np.nan).dropna()
        axes[1,1].hist(vals,bins=30)
        axes[1,1].set(xlabel="joint ΔBIC",ylabel="NV count",
                      title="Best shared-assignment joint evidence")
        fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
        # Compact pages of the most conflicted NVs.
        q=summary.copy()
        q["conflict_score"]=q[["consensus_deltaBIC_49","consensus_deltaBIC_52"]].max(axis=1)
        q=q.sort_values(["exact_winner_agreement","conflict_score"],ascending=[True,False])
        for start in range(0,len(q),28):
            s=q.iloc[start:start+28]
            fig,ax=plt.subplots(figsize=(11,8.5)); ax.axis("off")
            cols=["nv_index","status","winner_49_sites","winner_52_sites",
                  "consensus_sites","consensus_deltaBIC_49","consensus_deltaBIC_52"]
            show=s[cols].copy()
            for c in ["consensus_deltaBIC_49","consensus_deltaBIC_52"]:
                show[c]=show[c].map(lambda v:"" if pd.isna(v) else f"{v:.1f}")
            tbl=ax.table(cellText=show.values,colLabels=show.columns,
                         loc="center",cellLoc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(6.5); tbl.scale(1,1.25)
            ax.set_title("49G ↔ 52G cross-field assignments")
            fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def dataset_signature(field):
    mod=importlib.import_module(
        f"analysis.spin_echo_work.sc_c13_spin_echo_old_protocol_ranked_{field}"
    )
    _,nv_list,_,_,_,_=mod.load_data()
    names=[str(nv.name) for nv in nv_list]
    sample=names[0].split("-nv")[0] if names else ""
    return sample,names


def run(args):
    sample49,names49=dataset_signature("49G")
    sample52,names52=dataset_signature("52G")
    if sample49 != sample52:
        raise RuntimeError(
            "Cross-field comparison aborted: datasets are different samples: "
            f"49G={sample49!r} ({len(names49)} NVs), "
            f"52G={sample52!r} ({len(names52)} NVs). "
            "Match/acquire the same physical NV sample before cross-field analysis."
        )
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    r49,c49,q49=load_field("49G",args.config)
    r52,c52,q52=load_field("52G",args.config)
    summary,shared=analyze(c49,c52)
    summary=attach_qc(summary,q49,q52)

    summary.to_csv(out/"v23_cross_field_summary.csv",index=False)
    shared.to_csv(out/"v23_shared_candidate_ranking.csv",index=False)
    report(out/"v23_cross_field_report.pdf",summary)

    meta=dict(
        config=args.config,source_49G=str(r49),source_52G=str(r52),
        n_common_nvs=int(len(summary)),
        n_49_only=int(len(set(c49.nv_index)-set(c52.nv_index))),
        n_52_only=int(len(set(c52.nv_index)-set(c49.nv_index))),
        status_counts={str(k):int(v) for k,v in summary.status.value_counts().items()},
        same_order=int(summary.same_winner_order.sum()),
        exact_winner_agreement=int(summary.exact_winner_agreement.sum()),
        note="joint DeltaBIC is sum of within-field DeltaBIC values for the same discrete assignment",
    )
    with open(out/"v23_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)

    print("\nV23 CROSS-FIELD COMPLETE")
    print("common NVs:",len(summary))
    print("status:",meta["status_counts"])
    print("same model order:",meta["same_order"])
    print("exact winner agreement:",meta["exact_winner_agreement"])
    print("output:",out)
    return summary,shared


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default=CONFIG)
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=ap.parse_args(); run(args)


if __name__=="__main__":
    main()
