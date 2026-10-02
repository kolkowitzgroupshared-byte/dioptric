"""V31b targeted seed-stability check for V31 profile anomalies. Preserves V30/V31."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v31_scale_profile_52G as v31
R30=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v30_consolidated_state_52G\2026_09")
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v31b_profile_seedcheck_52G\2026_09")
IDS=[12,37,78]
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    st=pd.read_csv(R30/"v30_consolidated_52G_state.csv").set_index("nv_index")
    base=v6.load_backend("52G"); _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck); cat=v6.load_catalog(base)
    om=v6.load_assigned_orientations("52G",Y.shape[0])
    out=[]
    with threadpool_limits(limits=1):
        for nv in IDS:
            r=st.loc[nv]; ori=tuple(om[nv])
            sites=v31.sites_from_ids(cat,ori,r.site_key)
            th0=np.asarray(json.loads(str(r.theta_json)),float)
            for s in v31.S_GRID:
                trials=[]
                for seed in (th0,):
                    q=v31.fit_fixed_scale(base,t,Y[nv],E[nv],sites,seed,str(r.background_model),float(s),max_nfev=7000)
                    if q is not None: trials.append(q)
                if trials:
                    th,stats=min(trials,key=lambda q:q[1]["chi2"])
                    out.append(dict(nv_index=nv,scale=float(s),chi2=float(stats["chi2"]),theta_json=json.dumps([float(x) for x in th])))
            print("done",nv)
    d=pd.DataFrame(out); d.to_csv(OUT/"v31b_profiles.csv",index=False)
    for nv,g in d.groupby("nv_index"):
        b=g.loc[g.chi2.idxmin()]; at10=g.iloc[np.argmin(np.abs(g.scale-10))]
        print(nv,"best",float(b.scale),float(b.chi2),"dchi10",float(at10.chi2-b.chi2))
if __name__=="__main__": main()
