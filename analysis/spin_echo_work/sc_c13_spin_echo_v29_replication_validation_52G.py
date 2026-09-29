"""V29 independent-acquisition replication validation for V27/V28 models.

The discrete site sets discovered from the combined 52 G trace are held fixed.
For each of the 9 original acquisitions, continuous parameters are refit
independently and BIC is compared against the pre-V27 baseline model.

This tests whether the new lattice assignments replicate across independent
raw acquisitions rather than only improving one combined trace.
"""
from __future__ import annotations

import ast, json, sys, gc
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

from utils import data_manager as dm
from utils import widefield
import analysis.spin_echo_work.sc_c13_spin_echo_old_protocol_ranked_52G as old52
import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v14_beta2_taper0_rerank as v14
import sc_c13_spin_echo_v26_residual_error_diagnostics_52G as v26
import sc_c13_spin_echo_v27_residual_driven_add_site_52G as v27
import sc_c13_spin_echo_v28_parsimony_rerank_52G as v28

OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v29_replication_validation_52G\2026_09")
def seed_variants(theta):
    th=np.asarray(theta,float)
    out=[th.copy()]
    if len(th)>9:
        for sc in (1.0,3.0,10.0):
            q=th.copy(); q[9]=sc; out.append(q)
    return out


def proposal_for_nv(nv,cur,v25,s27,f27,s28,f28,catalog,omap):
    row=cur.loc[nv]
    theta0,source,bg,dbic=v26.current_theta(row,v25.loc[nv])
    sites0=v14.sites_from_record(row)
    info=dict(
        nv_index=int(nv),background_model=bg,
        baseline_sites=sites0,baseline_theta=theta0,
        baseline_order=len(sites0),baseline_site_key=str(row.site_key),
    )

    rr=s27.loc[nv]
    use_lower=False
    if nv in s28.index and s28.loc[nv].classification=="lower_order_competitive":
        use_lower=True

    if use_lower:
        sr=s28.loc[nv]
        hit=f28[
            (f28.nv_index==nv)
            & (f28.site_key.astype(str)==str(sr.best_lower_site_key))
        ].sort_values("bic").iloc[0]
        ids=tuple(int(x) for x in ast.literal_eval(str(hit.site_key)))
        ori=tuple(omap[nv]); sites=[]
        base_by_id={int(s["site_id"]):s for s in sites0}
        for s in sites0:
            if int(s["site_id"]) in ids: sites.append(s)
        for sid in ids:
            if sid not in base_by_id:
                sites.append(v28.get_catalog_site(catalog,ori,sid))
        theta=np.asarray(json.loads(str(hit.theta_json)),float)
        source="v28_lower"
        full_class=str(sr.classification)
    else:
        hit=f27[
            (f27.nv_index==nv)
            & (f27.new_site_key.astype(str)==str(rr.best_new_site_key))
        ].sort_values("new_bic").iloc[0]
        sid=int(hit.added_site_id)
        ori=tuple(omap[nv])
        sites=sites0+[v28.get_catalog_site(catalog,ori,sid)]
        theta=np.asarray(json.loads(str(hit.theta_json)),float)
        source="v27_add"
        full_class=(str(s28.loc[nv].classification)
                    if nv in s28.index else "add_from_N0")

    info.update(
        proposal_sites=sites,proposal_theta=theta,
        proposal_order=len(sites),
        proposal_site_key=str(tuple(sorted(int(s["site_id"]) for s in sites))),
        proposal_source=source,full_data_classification=full_class,
    )
    return info


def fit_pair(info,t,y,e):
    base=v6.load_backend("52G")
    bg=info["background_model"]
    b=v28.fit_sites(
        base,t,y,e,info["baseline_sites"],
        seed_variants(info["baseline_theta"]),bg,max_nfev=3000
    )
    p=v28.fit_sites(
        base,t,y,e,info["proposal_sites"],
        seed_variants(info["proposal_theta"]),bg,max_nfev=3000
    )
    if b is None or p is None:
        return dict(nv_index=info["nv_index"],status="fit_failed")
    return dict(
        nv_index=info["nv_index"],status="ok",
        baseline_bic=float(b["bic"]),proposal_bic=float(p["bic"]),
        delta_bic=float(p["bic"]-b["bic"]),
        baseline_redchi2=float(b["red_chi2"]),
        proposal_redchi2=float(p["red_chi2"]),
        baseline_chi2=float(b["chi2"]),proposal_chi2=float(p["chi2"]),
        delta_chi2=float(p["chi2"]-b["chi2"]),
        proposal_scale=float(p["theta"][9]) if info["proposal_sites"] else np.nan,
    )
def process_file(stem,selected_ids,t_ref):
    raw=dm.get_raw_data(file_stem=stem,load_npz=True,use_cache=False)
    ids=np.asarray(selected_ids,int)
    counts=np.asarray(raw["counts"])
    nvsub=[raw["nv_list"][int(i)] for i in ids]
    sig=np.asarray(counts[0,ids],dtype=np.float32)
    ref=np.asarray(counts[1,ids],dtype=np.float32)
    norm,ste=widefield.process_counts(nvsub,sig,ref,threshold=True)
    tf=2.0*np.asarray(raw["taus"],float).ravel()/1e3
    order=np.argsort(tf)
    if not np.allclose(tf[order],t_ref,rtol=0,atol=1e-9):
        raise ValueError(f"tau mismatch: {stem}")
    out=(np.asarray(norm,float)[:,order],np.asarray(ste,float)[:,order],
         int(raw["num_runs"]),int(raw["num_reps"]))
    del raw,counts,sig,ref,norm,ste
    gc.collect()
    return out


def run(workers=10):
    OUT.mkdir(parents=True,exist_ok=True)
    cur,v25=v26.load_current_rows()
    s27=pd.read_csv(v27.OUT/"v27_add_site_summary.csv").set_index("nv_index")
    f27=pd.read_csv(v27.OUT/"v27_add_site_candidate_fits.csv.gz")
    s28=pd.read_csv(v28.OUT/"v28_parsimony_summary.csv").set_index("nv_index")
    f28=pd.read_csv(v28.OUT/"v28_lower_order_candidate_fits.csv.gz")
    strong=s27[(s27.status=="ok")&(s27.delta_bic<=-6)]
    ids=sorted(strong.index.astype(int).tolist())

    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)
    catalog=v6.load_catalog(base); omap=v6.load_assigned_orientations("52G",Y.shape[0])
    infos={nv:proposal_for_nv(nv,cur,v25,s27,f27,s28,f28,catalog,omap)
           for nv in ids}

    data,_,_,_,_,_=old52.load_data()
    stems=list(data["source_file_stems"])
    print(f"V29 replication validation: {len(ids)} NVs x {len(stems)} files")
    rows=[]
    for fi,stem in enumerate(stems,1):
        print(f"Source {fi}/{len(stems)}: {stem}")
        norm,ste,nruns,nreps=process_file(stem,ids,t)
        results=Parallel(n_jobs=workers,backend="loky",verbose=5)(
            delayed(fit_pair)(infos[nv],t,norm[j],ste[j])
            for j,nv in enumerate(ids)
        )
        for r in results:
            r.update(file_index=fi,file_stem=stem,num_runs=nruns,num_reps=nreps)
            rows.append(r)

    d=pd.DataFrame(rows)
    d.to_csv(OUT/"v29_per_file_model_comparison.csv",index=False)
    summaries=[]
    for nv in ids:
        g=d[(d.nv_index==nv)&(d.status=="ok")].copy()
        info=infos[nv]
        k=int((g.delta_bic<0).sum())
        strongk=int((g.delta_bic<=-6).sum())
        med=float(g.delta_bic.median())
        p=float(binomtest(k,len(g),0.5,alternative="greater").pvalue) if len(g) else np.nan
        if len(g)>=7 and k>=7 and med<=-6:
            rep="replicated_strong"
        elif len(g)>=6 and k>=6 and med<=-2:
            rep="replicated_moderate"
        else:
            rep="not_consistently_replicated"
        summaries.append(dict(
            nv_index=nv,baseline_order=info["baseline_order"],
            baseline_site_key=info["baseline_site_key"],
            proposal_order=info["proposal_order"],
            proposal_site_key=info["proposal_site_key"],
            proposal_source=info["proposal_source"],
            full_data_classification=info["full_data_classification"],
            n_files=len(g),n_proposal_better=k,n_strong_deltaBIC_le_m6=strongk,
            median_delta_bic=med,mean_delta_bic=float(g.delta_bic.mean()),
            min_delta_bic=float(g.delta_bic.min()),max_delta_bic=float(g.delta_bic.max()),
            sign_test_p=p,replication_class=rep,
        ))
    s=pd.DataFrame(summaries).sort_values(["replication_class","median_delta_bic"])
    s.to_csv(OUT/"v29_replication_summary.csv",index=False)

    print("\nV29 COMPLETE")
    print(s.replication_class.value_counts().to_dict())
    print("strongly replicated:",int((s.replication_class=="replicated_strong").sum()))
    print("output:",OUT)
    print(s.to_string(index=False))


if __name__=="__main__":
    run()
