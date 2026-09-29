"""V33 targeted N3 continuation of the completed V32 Johnson four-field fit.

Baseline:
  V32 N<=2, 204 Johnson NVs, fields 49/59/62/65 G.

Default selection:
  V32 global winner is N2 AND N2 beats N1 by >= 10 BIC units.

For each selected NV, V33:
  * reuses the exact V32 physical-site pool
  * keeps several near-best N2 parents (not only the winner)
  * adds one third physical 13C site
  * fits the SAME 3 site IDs to all four fields
  * keeps field-specific background/contrast/T2/scale/phases independent
  * compares N3 against the V32 N2 baseline by summed four-field BIC
  * checkpoints every NV
  * creates a fitted-plot PDF with N2 vs N3 overlays for all four fields
"""
from __future__ import annotations

import argparse, ast, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_c13_spin_echo_v32_joint_multifield_johnson as v32
from spin_echo_paths import existing_or_canonical
ROOT=v32.ROOT
V32ROOT=ROOT/r"c13_spin_echo_v32_joint_multifield_johnson\2026_09"
V32TAG="N2_pool30_freq6_beam10_cap120_reduced_smax30p0"
BASE=V32ROOT/V32TAG
OUTROOT=existing_or_canonical(
    "c13_spin_echo_v33_targeted_n3_multifield_johnson", "2026_09"
)


def parse_tuple(x):
    z=ast.literal_eval(str(x))
    return tuple(int(v) for v in z)


def row_to_parent(row):
    ids=parse_tuple(row.site_key)
    fieldfits={}
    for cfg in v32.FIELDS:
        lab=cfg["label"]
        fieldfits[lab]={
            "theta":np.asarray(json.loads(str(row[f"{lab}_theta_json"])),float),
            "bic":float(row[f"{lab}_bic"]),
            "red_chi2":float(row[f"{lab}_redchi2"]),
        }
    return {
        "site_ids":ids,
        "fieldfits":fieldfits,
        "joint_bic_sum":float(row.joint_bic_sum),
        "joint_redchi2":float(row.joint_redchi2),
    }


def load_v32():
    winners=pd.read_csv(BASE/"v32_joint_winners.csv")
    orderbest=pd.read_csv(BASE/"v32_joint_best_by_order.csv")
    cand=pd.read_csv(BASE/"v32_joint_candidates.csv.gz")
    return winners,orderbest,cand
def selection_table(winners,orderbest,args):
    piv=orderbest.pivot(index="nv_index",columns="model_order",
                        values="joint_bic_sum")
    rows=[]
    for _,w in winners.iterrows():
        nv=int(w.nv_index)
        if int(w.model_order)!=2 or nv not in piv.index or 1 not in piv.columns:
            continue
        db=float(piv.loc[nv,2]-piv.loc[nv,1])
        rows.append(dict(
            nv_index=nv,site_key=str(w.site_key),
            v32_joint_bic=float(w.joint_bic_sum),
            v32_joint_redchi2=float(w.joint_redchi2),
            dBIC_N2_vs_N1=db,
            strong_n2=bool(db<=-abs(args.min_n2_vs_n1)),
        ))
    d=pd.DataFrame(rows)

    if args.nv:
        wanted=set(v32.parse_nv_arg(args.nv,204))
        d=d[d.nv_index.isin(wanted)]
    elif args.selection=="strong":
        d=d[d.strong_n2]
    elif args.selection=="all-n2":
        pass
    elif args.selection=="poor-fit":
        d=d[d.v32_joint_redchi2>=args.poor_fit_threshold]
    return d.sort_values("nv_index").reset_index(drop=True)


def load_pool(nv):
    p=BASE/"checkpoints"/f"nv_{nv:04d}_pool.csv"
    d=pd.read_csv(p)
    return sorted({int(x) for x in d.site_id.tolist()})
def n0fits_for_nv(nv,cand):
    q=cand[(cand.nv_index==nv)&(cand.model_order==0)]
    if q.empty:
        raise RuntimeError(f"NV{nv}: missing V32 N0 row")
    return row_to_parent(q.sort_values("joint_bic_sum").iloc[0])["fieldfits"]


def n2_parents_for_nv(nv,cand,args):
    q=cand[(cand.nv_index==nv)&(cand.model_order==2)].copy()
    if q.empty:
        return []
    q=q.sort_values("joint_bic_sum")
    best=float(q.iloc[0].joint_bic_sum)
    q=q[q.joint_bic_sum<=best+args.parent_delta_bic].head(args.parent_count)
    return [row_to_parent(r) for _,r in q.iterrows()]


def candidate_rows_from_fit(nv,ori,parent,fit,added_sid):
    row=dict(
        nv_index=int(nv),orientation=str(tuple(ori)),model_order=3,
        site_key=str(tuple(fit["site_ids"])),added_site_id=int(added_sid),
        parent_site_key=str(tuple(parent["site_ids"])),
        parent_joint_bic=float(parent["joint_bic_sum"]),
        joint_bic_sum=float(fit["joint_bic_sum"]),
        joint_chi2=float(fit["joint_chi2"]),
        joint_redchi2=float(fit["joint_redchi2"]),
    )
    for cfg in v32.FIELDS:
        lab=cfg["label"]; ff=fit["fieldfits"][lab]
        th=np.asarray(ff["theta"],float)
        row[f"{lab}_bic"]=float(ff["bic"])
        row[f"{lab}_redchi2"]=float(ff["red_chi2"])
        row[f"{lab}_scale"]=float(th[9])
        row[f"{lab}_theta_json"]=json.dumps(th.tolist())
    return row
def fit_one_nv(nv,data,cats,cand,args,ckdir):
    cp=ckdir/f"nv_{nv:04d}_n3.csv.gz"
    if cp.exists():
        return str(cp)

    base=v6.load_backend("49G")
    ori=tuple(int(v) for v in data["49G"]["ori"][nv])
    pool=load_pool(nv)
    n0fits=n0fits_for_nv(nv,cand)
    parents=n2_parents_for_nv(nv,cand,args)
    if not parents:
        raise RuntimeError(f"NV{nv}: no eligible N2 parents")

    proposals={}
    for parent in parents:
        have=set(parent["site_ids"])
        for sid in pool:
            if int(sid) in have:
                continue
            ids=tuple(sorted((*have,int(sid))))
            old=proposals.get(ids)
            if old is None or parent["joint_bic_sum"]<old["joint_bic_sum"]:
                proposals[ids]=parent

    keys=sorted(proposals,key=lambda k:proposals[k]["joint_bic_sum"])
    keys=keys[:args.candidate_cap]
    rows=[]
    for ids in keys:
        parent=proposals[ids]
        added=list(set(ids)-set(parent["site_ids"]))
        added_sid=int(added[0]) if len(added)==1 else -1
        fit=v32.fit_joint_candidate(
            base,nv,ori,ids,data,cats,n0fits,args,parent=parent
        )
        if fit is not None:
            rows.append(candidate_rows_from_fit(
                nv,ori,parent,fit,added_sid
            ))

    if not rows:
        raise RuntimeError(f"NV{nv}: all N3 fits failed")
    d=pd.DataFrame(rows).sort_values("joint_bic_sum").reset_index(drop=True)
    d["rank_n3_joint_bic"]=np.arange(1,len(d)+1)
    d.to_csv(cp,index=False,compression="gzip")
    w=d.iloc[0]
    print(
        f"[NV {nv:3d}] N3 fits={len(d)} WIN {w.site_key} "
        f"jointBIC={w.joint_bic_sum:.1f} redchi={w.joint_redchi2:.2f}"
    )
    return str(cp)
def build_outputs(selected,winners,orderbest,cand,ckdir):
    n3parts=[]
    for nv in selected.nv_index.astype(int):
        p=ckdir/f"nv_{nv:04d}_n3.csv.gz"
        if p.exists():
            n3parts.append(pd.read_csv(p))
    n3=pd.concat(n3parts,ignore_index=True) if n3parts else pd.DataFrame()

    base2=winners.set_index("nv_index")
    rows=[]
    evidence=[]
    for nv in range(204):
        b=base2.loc[nv]
        final=b.to_dict()
        source="V32"
        db=np.nan
        if not n3.empty and nv in set(n3.nv_index.astype(int)):
            q=n3[n3.nv_index==nv].sort_values("joint_bic_sum")
            r=q.iloc[0]
            db=float(r.joint_bic_sum-b.joint_bic_sum)
            if db<0:
                for k,v in r.items():
                    final[k]=v
                source="V33_N3"
        final["final_source"]=source
        final["dBIC_N3_vs_V32_winner"]=db
        rows.append(final)
        evidence.append(dict(
            nv_index=nv,
            v32_order=int(b.model_order),
            v32_site_key=str(b.site_key),
            v32_bic=float(b.joint_bic_sum),
            v32_redchi2=float(b.joint_redchi2),
            n3_tested=bool(not np.isnan(db)),
            dBIC_N3_vs_V32_winner=db,
            n3_preferred=bool(np.isfinite(db) and db<0),
            n3_strong=bool(np.isfinite(db) and db<=-6),
        ))
    return n3,pd.DataFrame(rows),pd.DataFrame(evidence)
def predict_from_row(row,lab,data,cats):
    base=v6.load_backend("49G")
    ori=parse_tuple(row.orientation)
    ids=parse_tuple(row.site_key)
    th=np.asarray(json.loads(str(row[f"{lab}_theta_json"])),float)
    sites=[v32.site_record(cats,lab,ori,sid) for sid in ids]
    dense=np.linspace(float(data[lab]["t"].min()),
                      float(data[lab]["t"].max()),1400)
    pred=v6.v6_model(base,dense,th,sites)
    return dense,pred


def plot_page(pdf,nv,v32row,n3row,data,cats):
    fig,ax=plt.subplots(3,3,figsize=(14,10.5))
    ori=parse_tuple(v32row.orientation)
    n2ids=parse_tuple(v32row.site_key)
    n3ids=parse_tuple(n3row.site_key) if n3row is not None else ()

    for k,cfg in enumerate(v32.FIELDS):
        lab=cfg["label"]; dd=data[lab]
        a=ax[k//2,k%2]
        a.errorbar(dd["t"],dd["y"][nv],yerr=dd["e"][nv],
                   fmt=".",ms=2.4,lw=.45,alpha=.5,label="data")
        x2,y2=predict_from_row(v32row,lab,data,cats)
        a.plot(x2,y2,lw=1.0,label=f"N2 {n2ids}")
        if n3row is not None:
            x3,y3=predict_from_row(n3row,lab,data,cats)
            a.plot(x3,y3,lw=1.35,label=f"N3 {n3ids}")
        a.set(title=f"{lab}",xlabel="evolution time (us)",
              ylabel="normalized signal")
        a.legend(fontsize=6)

    # BIC evidence.
    a=ax[0,2]
    vals=[float(v32row.joint_bic_sum)]
    labs=["N2"]
    if n3row is not None:
        vals.append(float(n3row.joint_bic_sum)); labs.append("N3")
    db=np.asarray(vals)-min(vals)
    a.bar(labs,db)
    a.set(ylabel="Delta joint BIC",title="N2 vs N3 evidence")
    # Per-field fit quality.
    a=ax[1,2]
    x=np.arange(4); width=.34
    r2=[float(v32row[f"{c['label']}_redchi2"]) for c in v32.FIELDS]
    a.bar(x-width/2,r2,width,label="N2")
    if n3row is not None:
        r3=[float(n3row[f"{c['label']}_redchi2"]) for c in v32.FIELDS]
        a.bar(x+width/2,r3,width,label="N3")
    a.axhline(2,ls="--",lw=.7); a.axhline(3,ls="--",lw=.7)
    a.set_xticks(x,[c["label"] for c in v32.FIELDS])
    a.set(ylabel="reduced chi-square",title="Per-field fit quality")
    a.legend(fontsize=7)

    # Frequency trajectories for accepted N3 (or N2 if no N3).
    a=ax[2,0]
    ids=n3ids if n3ids else n2ids
    bm=[np.linalg.norm(c["B_G"]) for c in v32.FIELDS]
    for sid in ids:
        fm=[]; fp=[]
        for cfg in v32.FIELDS:
            s=v32.site_record(cats,cfg["label"],ori,sid)
            fm.append(s["f1_kHz"]); fp.append(s["f0_kHz"])
        a.plot(bm,fm,"o-",ms=3,label=f"{sid} f-")
        a.plot(bm,fp,"o--",ms=3,label=f"{sid} f+")
    a.set(xlabel="|B| (G)",ylabel="frequency (kHz)",
          title="Same-site field trajectories")
    a.legend(fontsize=5.5,ncol=2)

    # Real-space sites.
    a=ax[2,1]
    cc=v32.catalog_orientation(cats["49G"],ori)
    for sid in ids:
        q=cc[cc.site_id.astype(int)==sid].iloc[0]
        a.scatter([q.x_A],[q.y_A],s=80,marker="*")
        a.annotate(str(sid),(q.x_A,q.y_A),fontsize=8,
                   xytext=(3,3),textcoords="offset points")
    a.scatter([0],[0],marker="x",s=50)
    a.set_aspect("equal",adjustable="datalim")
    a.set(xlabel="x (A)",ylabel="y (A)",title="Selected C13 sites")

    # Text panel.
    a=ax[2,2]; a.axis("off")
    lines=[
        f"NV {nv}",
        f"orientation: {ori}",
        f"V32 N2: {n2ids}",
        f"V32 joint redchi2: {float(v32row.joint_redchi2):.3f}",
    ]
    if n3row is not None:
        delta=float(n3row.joint_bic_sum-v32row.joint_bic_sum)
        lines += [
            f"V33 N3: {n3ids}",
            f"N3 joint redchi2: {float(n3row.joint_redchi2):.3f}",
            f"dBIC(N3-N2): {delta:.2f}",
        ]
    a.text(0,1,"\n".join(lines),va="top",family="monospace",fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
def make_report(pdfpath,selected,winners,n3,data,cats):
    win=winners.set_index("nv_index")
    n3best=(n3.sort_values("joint_bic_sum")
            .groupby("nv_index",as_index=False).first()
            .set_index("nv_index")) if len(n3) else pd.DataFrame()

    with PdfPages(pdfpath) as pdf:
        fig,ax=plt.subplots(1,3,figsize=(14,4.5))
        if len(n3):
            merged=selected.merge(
                n3best[["joint_bic_sum","joint_redchi2"]],
                left_on="nv_index",right_index=True,how="left",
                suffixes=("_v32","_n3")
            )
            db=merged.joint_bic_sum-merged.v32_joint_bic
            ax[0].hist(db.dropna(),bins=30)
            ax[0].axvline(-6,ls="--",lw=.8)
            ax[0].set(title="N3 evidence",xlabel="dBIC N3-N2")
            ax[1].hist(merged.joint_redchi2.dropna(),bins=30)
            ax[1].set(title="Best N3 joint fit",xlabel="joint redchi2")
            ax[2].bar(
                ["N3 better","N3 strong","N2 retained"],
                [int((db<0).sum()),int((db<=-6).sum()),int((db>=0).sum())]
            )
            ax[2].set(title="V33 outcomes")
        fig.suptitle(f"V33 targeted N3 | selected NVs={len(selected)}")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for nv in selected.nv_index.astype(int):
            v32row=win.loc[nv]
            r3=n3best.loc[nv] if len(n3best) and nv in n3best.index else None
            plot_page(pdf,nv,v32row,r3,data,cats)
def run(args):
    winners,orderbest,cand=load_v32()
    selected=selection_table(winners,orderbest,args)
    if selected.empty:
        raise RuntimeError("No NVs selected for V33")

    data,seeds,cats=v32.load_all()
    args._data=data
    tag=(
        f"{args.selection}_db{str(args.min_n2_vs_n1).replace('.','p')}_"
        f"parents{args.parent_count}_pdb{str(args.parent_delta_bic).replace('.','p')}_"
        f"cap{args.candidate_cap}_{args.background}_smax"
        f"{str(args.scale_max).replace('.','p')}"
    )
    outdir=Path(args.output_dir)/tag
    ckdir=outdir/"checkpoints"
    outdir.mkdir(parents=True,exist_ok=True)
    ckdir.mkdir(parents=True,exist_ok=True)
    selected.to_csv(outdir/"v33_selected_nvs.csv",index=False)

    print("="*100)
    print("V33 TARGETED N3 JOHNSON FOUR-FIELD")
    print("selected:",len(selected),
          "criterion:",args.selection,
          "min N2-vs-N1:",args.min_n2_vs_n1)
    print("parents:",args.parent_count,
          "parent delta BIC:",args.parent_delta_bic,
          "candidate cap:",args.candidate_cap)

    nvs=selected.nv_index.astype(int).tolist()
    def task(nv):
        return fit_one_nv(nv,data,cats,cand,args,ckdir)

    with threadpool_limits(limits=1):
        Parallel(n_jobs=max(1,args.workers),backend="loky",verbose=10)(
            delayed(task)(nv) for nv in nvs
        )

    n3,final,evidence=build_outputs(
        selected,winners,orderbest,cand,ckdir
    )
    n3.to_csv(outdir/"v33_n3_candidates.csv.gz",
              index=False,compression="gzip")
    final.to_csv(outdir/"v33_final_winners_204.csv",index=False)
    evidence.to_csv(outdir/"v33_order_evidence.csv",index=False)

    pdfpath=outdir/"v33_targeted_n3_multifield_report.pdf"
    make_report(pdfpath,selected,winners,n3,data,cats)
    tested=evidence[evidence.n3_tested]
    print("V33 COMPLETE")
    print("tested:",len(tested))
    print("N3 preferred dBIC<0:",int(tested.n3_preferred.sum()))
    print("N3 strong dBIC<=-6:",int(tested.n3_strong.sum()))
    if len(tested):
        vals=tested.dBIC_N3_vs_V32_winner.dropna()
        print("median dBIC N3-N2:",float(vals.median()))
    print("output:",outdir)

    meta=vars(args).copy(); meta.pop("_data",None)
    meta.update(
        v32_source=str(BASE),
        selected_nv_count=int(len(selected)),
        n3_preferred_count=int(tested.n3_preferred.sum()),
        n3_strong_count=int(tested.n3_strong.sum()),
    )
    with open(outdir/"v33_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--selection",
                    choices=("strong","all-n2","poor-fit"),
                    default="strong")
    ap.add_argument("--nv",default=None,
                    help="Explicit NV list/range overrides selection, e.g. 13,48,178-181")
    ap.add_argument("--min-n2-vs-n1",type=float,default=10.0,
                    help="For --selection strong: require dBIC(N2-N1) <= -this value.")
    ap.add_argument("--poor-fit-threshold",type=float,default=2.0)
    ap.add_argument("--parent-count",type=int,default=5,
                    help="Near-best V32 N2 parents to expand.")
    ap.add_argument("--parent-delta-bic",type=float,default=20.0,
                    help="Only N2 parents within this BIC of best.")
    ap.add_argument("--candidate-cap",type=int,default=150,
                    help="Maximum unique N3 discrete site sets per NV.")
    ap.add_argument("--background",
                    choices=("reduced","full","beta-free","taper-free"),
                    default="reduced")
    ap.add_argument("--scale-max",type=float,default=30.0)
    ap.add_argument("--robust-max-nfev",type=int,default=1000)
    ap.add_argument("--final-max-nfev",type=int,default=1800)
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--output-dir",default=str(OUTROOT))
    args=ap.parse_args()
    if args.parent_count<1 or args.candidate_cap<1:
        ap.error("parent/candidate limits must be positive")
    run(args)


if __name__=="__main__":
    main()
