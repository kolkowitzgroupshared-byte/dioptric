"""Audit orientation labels and magnetic-field geometry used by V32-V35."""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[2]
for q in (HERE,ROOT):
    if str(q) not in sys.path: sys.path.insert(0,str(q))
import sc_c13_c13_spin_echo_v32_joint_multifield_johnson as v32

OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v35_orientation_field_audit\2026_09")
OUT.mkdir(parents=True,exist_ok=True)
data,seeds,cats=v32.load_all()
ref=data["49G"]["ori"]
assert ref.shape==(204,3)
for cfg in v32.FIELDS:
    assert np.array_equal(data[cfg["label"]]["ori"],ref)
oris=[tuple(map(int,x)) for x in ref]
counts=pd.Series(oris).value_counts()
print("Orientation counts:",counts.to_dict())
# Field geometry relative to every stored orientation class
uoris={o:np.asarray(o,float)/np.linalg.norm(o) for o in sorted(set(oris))}
rows=[]
for cfg in v32.FIELDS:
    b=np.asarray(cfg["B_G"],float); bm=np.linalg.norm(b); bh=b/bm
    for o,u in uoris.items():
        par=float(np.dot(b,u))
        rows.append({"field":cfg["label"],"Bx_G":b[0],"By_G":b[1],"Bz_G":b[2],
          "Bmag_G":bm,"orientation":str(o),"Bparallel_G":par,
          "abs_Bparallel_G":abs(par),"angle_deg":float(np.degrees(np.arccos(np.clip(abs(np.dot(bh,u)),-1,1)))),
          "electron_projection_MHz":2.8025*par})
geom=pd.DataFrame(rows)
geom.to_csv(OUT/"field_orientation_geometry.csv",index=False)

# Pairwise geometric separation between observed orientation classes
sep=[]
labels=list(uoris)
for cfg in v32.FIELDS:
    b=np.asarray(cfg["B_G"],float)
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            p1=abs(np.dot(b,uoris[labels[i]])); p2=abs(np.dot(b,uoris[labels[j]]))
            sep.append({"field":cfg["label"],"ori_a":str(labels[i]),"ori_b":str(labels[j]),
              "delta_abs_Bparallel_G":abs(p1-p2),
              "approx_delta_electron_projection_MHz":2.8025*abs(p1-p2)})
pd.DataFrame(sep).to_csv(OUT/"orientation_geometric_separation.csv",index=False)
# Per-NV audit table: provenance/consistency plus geometry for assigned class
nvrows=[]
for nv,o in enumerate(oris):
    z={"nv_index":nv,"orientation":str(o),"orientation_same_all_4_fields":True}
    for cfg in v32.FIELDS:
        lab=cfg["label"]; b=np.asarray(cfg["B_G"],float); u=uoris[o]
        z[f"{lab}_Bmag_G"]=np.linalg.norm(b)
        z[f"{lab}_Bparallel_G"]=np.dot(b,u)
        z[f"{lab}_angle_deg"]=np.degrees(np.arccos(np.clip(abs(np.dot(b/np.linalg.norm(b),u)),-1,1)))
    nvrows.append(z)
audit=pd.DataFrame(nvrows)
audit.to_csv(OUT/"orientation_field_audit_204nv.csv",index=False)

fig,ax=plt.subplots(figsize=(7,4.8))
for o,g in geom.groupby("orientation"):
    ax.plot(g.field,g.abs_Bparallel_G,marker="o",label=o)
ax.set_ylabel("|B parallel| (G)"); ax.set_xlabel("Dataset")
ax.set_title("Field projection onto observed NV axes"); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"field_projection_by_orientation.png",dpi=250); plt.close(fig)
meta={"n_nv":204,"orientation_counts":{str(k):int(v) for k,v in counts.items()},
 "orientation_map_identical_all_four_fields":True,
 "field_vectors":{c["label"]:np.asarray(c["B_G"],float).tolist() for c in v32.FIELDS},
 "field_magnitudes_G":{c["label"]:float(np.linalg.norm(c["B_G"])) for c in v32.FIELDS},
 "important_limitation":"Stored-label consistency and geometric separability are not a posterior probability of orientation. A fair confidence test requires alternate-orientation refitting/search or independent ODMR resonance evidence."}
with open(OUT/"orientation_field_audit_summary.json","w") as f: json.dump(meta,f,indent=2)
print(json.dumps(meta,indent=2))
print(pd.DataFrame(sep).to_string(index=False))
print("OUTPUT",OUT)
