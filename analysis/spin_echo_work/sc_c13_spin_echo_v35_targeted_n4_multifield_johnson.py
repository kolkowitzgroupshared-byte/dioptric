"""V35 targeted N4 continuation for V34-priority Johnson NVs."""
from __future__ import annotations
import argparse, ast, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
REPO_ROOT=Path(__file__).resolve().parents[2]
for q in (HERE,REPO_ROOT):
    if str(q) not in sys.path: sys.path.insert(0,str(q))
import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_c13_spin_echo_v32_joint_multifield_johnson as v32

V33=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo_v33_targeted_n3_multifield_johnson\2026_09\strong_db10p0_parents5_pdb20p0_cap150_reduced_smax30p0")
OUTROOT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v35_targeted_n4_multifield_johnson\2026_09")
PRIORITY=[48,60,167,187,191]
def tup(x): return tuple(int(v) for v in ast.literal_eval(str(x)))
def parent_from_row(r):
    ff={}
    for cfg in v32.FIELDS:
        lab=cfg["label"]
        ff[lab]={"theta":np.asarray(json.loads(str(r[f"{lab}_theta_json"])),float),
                 "bic":float(r[f"{lab}_bic"]),"red_chi2":float(r[f"{lab}_redchi2"])}
    return {"site_ids":tup(r.site_key),"fieldfits":ff,
            "joint_bic_sum":float(r.joint_bic_sum),"joint_redchi2":float(r.joint_redchi2)}

def load_pool(nv):
    p=v32.BASE/"checkpoints"/f"nv_{nv:04d}_pool.csv" if hasattr(v32,"BASE") else None
    if p is None or not p.exists():
        p=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v32_joint_multifield_johnson\2026_09\N2_pool30_freq6_beam10_cap120_reduced_smax30p0\checkpoints")/f"nv_{nv:04d}_pool.csv"
    d=pd.read_csv(p)
    return sorted(set(d.site_id.astype(int)))
def n0fits(nv,cand):
    q=cand[(cand.nv_index==nv)&(cand.model_order==0)].sort_values("joint_bic_sum")
    r=q.iloc[0]; ff={}
    for cfg in v32.FIELDS:
        lab=cfg["label"]
        ff[lab]={"theta":np.asarray(json.loads(str(r[f"{lab}_theta_json"])),float),
                 "bic":float(r[f"{lab}_bic"]),"red_chi2":float(r[f"{lab}_redchi2"])}
    return ff

def row_from_fit(nv,ori,parent,fit,added):
    z={"nv_index":nv,"orientation":str(tuple(ori)),"model_order":4,
       "site_key":str(tuple(fit["site_ids"])),"added_site_id":int(added),
       "parent_site_key":str(tuple(parent["site_ids"])),
       "parent_joint_bic":parent["joint_bic_sum"],
       "joint_bic_sum":fit["joint_bic_sum"],"joint_chi2":fit["joint_chi2"],
       "joint_redchi2":fit["joint_redchi2"]}
    for cfg in v32.FIELDS:
        lab=cfg["label"]; f=fit["fieldfits"][lab]; th=np.asarray(f["theta"],float)
        z[f"{lab}_bic"]=f["bic"]; z[f"{lab}_redchi2"]=f["red_chi2"]
        z[f"{lab}_scale"]=th[9]; z[f"{lab}_theta_json"]=json.dumps(th.tolist())
    return z
def fit_nv(nv,data,cats,v32cand,n3,args,ckdir):
    cp=ckdir/f"nv_{nv:04d}_n4.csv.gz"
    if cp.exists(): return str(cp)
    base=v6.load_backend("49G")
    ori=tuple(int(x) for x in data["49G"]["ori"][nv])
    pool=load_pool(nv)
    q=n3[n3.nv_index==nv].sort_values("joint_bic_sum")
    best=float(q.iloc[0].joint_bic_sum)
    q=q[q.joint_bic_sum<=best+args.parent_delta_bic].head(args.parent_count)
    parents=[parent_from_row(r) for _,r in q.iterrows()]
    proposals={}
    for par in parents:
        have=set(par["site_ids"])
        for sid in pool:
            if sid in have: continue
            ids=tuple(sorted((*have,sid)))
            if ids not in proposals or par["joint_bic_sum"]<proposals[ids]["joint_bic_sum"]:
                proposals[ids]=par
    keys=sorted(proposals,key=lambda k:proposals[k]["joint_bic_sum"])[:args.candidate_cap]
    rows=[]; n0=n0fits(nv,v32cand)
    for ids in keys:
        par=proposals[ids]; added=list(set(ids)-set(par["site_ids"]))[0]
        fit=v32.fit_joint_candidate(base,nv,ori,ids,data,cats,n0,args,parent=par)
        if fit is not None: rows.append(row_from_fit(nv,ori,par,fit,added))
    if not rows: raise RuntimeError(f"NV{nv}: all N4 fits failed")
    d=pd.DataFrame(rows).sort_values("joint_bic_sum").reset_index(drop=True)
    d["rank_n4_joint_bic"]=np.arange(1,len(d)+1)
    d.to_csv(cp,index=False,compression="gzip")
    print(f"NV{nv}: N4 {d.iloc[0].site_key} BIC={d.iloc[0].joint_bic_sum:.2f} redchi={d.iloc[0].joint_redchi2:.3f}")
    return str(cp)
def run(args):
    data,seeds,cats=v32.load_all()
    v32cand=pd.read_csv(Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v32_joint_multifield_johnson\2026_09\N2_pool30_freq6_beam10_cap120_reduced_smax30p0\v32_joint_candidates.csv.gz"))
    n3=pd.read_csv(V33/"v33_n3_candidates.csv.gz")
    n3win=pd.read_csv(V33/"v33_final_winners_204.csv")
    out=OUTROOT/f"priority5_parents{args.parent_count}_pdb{args.parent_delta_bic:g}_cap{args.candidate_cap}_{args.background}_smax{args.scale_max:g}"
    ck=out/"checkpoints"; ck.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        Parallel(n_jobs=args.workers,backend="loky",verbose=10)(
            delayed(fit_nv)(nv,data,cats,v32cand,n3,args,ck) for nv in PRIORITY)
    parts=[pd.read_csv(ck/f"nv_{nv:04d}_n4.csv.gz") for nv in PRIORITY]
    all4=pd.concat(parts,ignore_index=True); all4.to_csv(out/"v35_n4_candidates.csv.gz",index=False,compression="gzip")
    rows=[]
    for nv in PRIORITY:
        b=n3win[n3win.nv_index==nv].iloc[0]; q=all4[all4.nv_index==nv].sort_values("joint_bic_sum"); r=q.iloc[0]
        rows.append({"nv_index":nv,"n3_site_key":b.site_key,"n4_site_key":r.site_key,
          "n3_bic":b.joint_bic_sum,"n4_bic":r.joint_bic_sum,"dBIC_N4_vs_N3":r.joint_bic_sum-b.joint_bic_sum,
          "n3_redchi2":b.joint_redchi2,"n4_redchi2":r.joint_redchi2,"n4_added_site_id":r.added_site_id})
    pd.DataFrame(rows).to_csv(out/"v35_n4_vs_n3_summary.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False)); print("OUTPUT",out)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent-count",type=int,default=5)
    ap.add_argument("--parent-delta-bic",type=float,default=20.0)
    ap.add_argument("--candidate-cap",type=int,default=150)
    ap.add_argument("--background",choices=("reduced","full","beta-free","taper-free"),default="reduced")
    ap.add_argument("--scale-max",type=float,default=30.0)
    ap.add_argument("--robust-max-nfev",type=int,default=1000)
    ap.add_argument("--final-max-nfev",type=int,default=1800)
    ap.add_argument("--workers",type=int,default=5)
    args=ap.parse_args(); run(args)
if __name__=="__main__": main()
