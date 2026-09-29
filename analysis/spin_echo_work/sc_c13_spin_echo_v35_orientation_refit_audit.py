"""Conditional alternate-orientation refit audit for final V33 winners."""
from pathlib import Path
import sys, ast, json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent; ROOT=Path(__file__).resolve().parents[2]
for q in (HERE,ROOT):
    if str(q) not in sys.path: sys.path.insert(0,str(q))
import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_c13_spin_echo_v32_joint_multifield_johnson as v32

V32=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v32_joint_multifield_johnson\2026_09\N2_pool30_freq6_beam10_cap120_reduced_smax30p0")
V33=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo_v33_targeted_n3_multifield_johnson\2026_09\strong_db10p0_parents5_pdb20p0_cap150_reduced_smax30p0")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v35_orientation_field_audit\2026_09")
OUT.mkdir(parents=True,exist_ok=True)
class Args:
    background="reduced"; scale_max=30.0; robust_max_nfev=1000; final_max_nfev=1800
args=Args()
def tup(x): return tuple(int(v) for v in ast.literal_eval(str(x)))
def parent(r):
    ff={}
    for cfg in v32.FIELDS:
        lab=cfg["label"]
        ff[lab]={"theta":np.asarray(json.loads(str(r[f"{lab}_theta_json"])),float),
                 "bic":float(r[f"{lab}_bic"]),"red_chi2":float(r[f"{lab}_redchi2"])}
    return {"site_ids":tup(r.site_key),"fieldfits":ff,
            "joint_bic_sum":float(r.joint_bic_sum),"joint_redchi2":float(r.joint_redchi2)}
def n0fits(nv,cand):
    r=cand[(cand.nv_index==nv)&(cand.model_order==0)].sort_values("joint_bic_sum").iloc[0]
    return parent(r)["fieldfits"]
data,seeds,cats=v32.load_all()
win=pd.read_csv(V33/"v33_final_winners_204.csv")
win["nv_index"]=np.arange(len(win),dtype=int)
cand=pd.read_csv(V32/"v32_joint_candidates.csv.gz")
observed=sorted({tuple(map(int,x)) for x in data["49G"]["ori"]})
base=v6.load_backend("49G")
ck=OUT/"orientation_refit_checkpoints"; ck.mkdir(exist_ok=True)

def task(nv):
    cp=ck/f"nv_{nv:04d}.json"
    if cp.exists(): return json.load(open(cp))
    r=win[win.nv_index==nv].iloc[0]
    order=int(r.model_order); assigned=tup(r.orientation)
    if order==0:
        z={"nv_index":nv,"model_order":0,"assigned_orientation":str(assigned),
           "status":"uninformative_N0","assigned_bic":float(r.joint_bic_sum)}
    else:
        alt=[o for o in observed if o!=assigned]
        if len(alt)!=1: raise RuntimeError((nv,assigned,observed))
        alt=alt[0]; ids=tup(r.site_key)
        fit=v32.fit_joint_candidate(base,nv,alt,ids,data,cats,n0fits(nv,cand),args,parent=parent(r))
        if fit is None:
            z={"nv_index":nv,"model_order":order,"assigned_orientation":str(assigned),
               "alternate_orientation":str(alt),"status":"alternate_fit_failed",
               "assigned_bic":float(r.joint_bic_sum),"alternate_bic":np.nan,"dBIC_alt_minus_assigned":np.nan}
        else:
            z={"nv_index":nv,"model_order":order,"assigned_orientation":str(assigned),
               "alternate_orientation":str(alt),"status":"ok",
               "assigned_bic":float(r.joint_bic_sum),"alternate_bic":float(fit["joint_bic_sum"]),
               "dBIC_alt_minus_assigned":float(fit["joint_bic_sum"]-r.joint_bic_sum),
               "assigned_redchi2":float(r.joint_redchi2),"alternate_redchi2":float(fit["joint_redchi2"])}
    json.dump(z,open(cp,"w"),indent=2); return z
nvs=win.nv_index.astype(int).tolist()
with threadpool_limits(limits=1):
    rows=Parallel(n_jobs=10,backend="loky",verbose=10)(delayed(task)(nv) for nv in nvs)
res=pd.DataFrame(rows).sort_values("nv_index")
res.to_csv(OUT/"orientation_conditional_alt_refit_204nv.csv",index=False)
ok=res[res.status.eq("ok")].copy()
def cls(x):
    if x>=10:return "strong_assigned"
    if x>=6:return "moderate_assigned"
    if x>=2:return "weak_assigned"
    if x>=0:return "ambiguous_assigned"
    if x>-2:return "ambiguous_alternate"
    return "alternate_preferred"
ok["orientation_evidence_class"]=ok.dBIC_alt_minus_assigned.map(cls)
ok.to_csv(OUT/"orientation_conditional_alt_refit_informative.csv",index=False)
print("informative",len(ok),"N0",int((res.model_order==0).sum()))
print(ok.orientation_evidence_class.value_counts().to_dict())
print("median dBIC alt-assigned",float(ok.dBIC_alt_minus_assigned.median()))
print("alternate preferred",ok.loc[ok.dBIC_alt_minus_assigned<0,"nv_index"].astype(int).tolist())
print("OUTPUT",OUT)
