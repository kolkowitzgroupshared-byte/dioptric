"""V26b: corrected excess between-file variance for QNami 52 G.

V26's raw between-file SEM contains ordinary within-file measurement noise.
This script subtracts the expected contribution from each file's own STE
before adding only excess slow-drift variance to the final combined STE.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT=Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0,str(REPO_ROOT))

import sc_c13_spin_echo_v26_residual_error_diagnostics_52G as v26
import sc_c13_spin_echo_physical_family_search_v6 as v6
import analysis.spin_echo_work.sc_c13_spin_echo_old_protocol_ranked_52G as old52

OUT=v26.OUT
def run():
    cur,v25=v26.load_current_rows()
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths()
    t,Y,E=base.load_data(ck)
    bad_ids=sorted(v25.index.astype(int).tolist())

    data,_,_,_,_,_=old52.load_data()
    stems=list(data["source_file_stems"])
    traces,stes,weights,names=v26.process_source_files(stems,bad_ids,t)

    a=weights/weights.sum()
    D=1.0-float(np.sum(a*a))
    wmean=np.tensordot(a,traces,axes=(0,0))
    diff=traces-wmean[None,:,:]
    obs_var=np.tensordot(a,diff*diff,axes=(0,0))/D

    # Expected weighted file-to-file variance from each file's reported
    # measurement variance alone.
    coeff=a*(1.0-a)
    meas_num=np.tensordot(coeff,stes*stes,axes=(0,0))
    meas_var=meas_num/D
    excess_var=np.maximum(obs_var-meas_var,0.0)

    # If the excess is independent from acquisition to acquisition, its
    # contribution to the weighted combined mean is tau^2 * sum(a_i^2).
    extra_sem=np.sqrt(excess_var*float(np.sum(a*a)))
    rows=[]
    for j,nv in enumerate(bad_ids):
        row=cur.loc[nv]
        vr=v25.loc[nv]
        theta,source,bg,dbic=v26.current_theta(row,vr)
        pred=v26.model_for_row(base,t,row,theta)
        raw=Y[nv]-pred
        e0=base.safe_err(E[nv])
        e1=np.sqrt(e0*e0+extra_sem[j]*extra_sem[j])

        fixed=2 if bg=="reduced" else (1 if bg in ("beta_free","taper_free") else 0)
        npar=max(1,len(theta)-fixed)
        dof=max(1,len(t)-npar)
        red0=float(np.sum((raw/e0)**2)/dof)
        red1=float(np.sum((raw/e1)**2)/dof)

        ratio=extra_sem[j]/np.maximum(e0,1e-12)
        positive=excess_var[j]>0
        obs_to_meas=obs_var[j]/np.maximum(meas_var[j],1e-18)
        rows.append(dict(
            nv_index=int(nv),background_model=bg,
            original_redchi2=red0,excess_corrected_redchi2=red1,
            median_extra_sem=float(np.median(extra_sem[j])),
            median_reported_ste=float(np.median(e0)),
            median_extra_to_reported=float(np.median(ratio)),
            p90_extra_to_reported=float(np.quantile(ratio,.9)),
            frac_timepoints_excess_positive=float(np.mean(positive)),
            median_obs_to_expected_file_variance=float(np.median(obs_to_meas)),
            frac_obs_variance_gt_expected=float(np.mean(obs_to_meas>1)),
        ))
    d=pd.DataFrame(rows).sort_values("nv_index")
    d.to_csv(OUT/"v26b_excess_variance_bad_nvs.csv",index=False)

    summary=dict(
        n_bad=int(len(d)),
        median_extra_to_reported=float(d.median_extra_to_reported.median()),
        n_median_extra_gt_reported=int((d.median_extra_to_reported>1).sum()),
        median_fraction_timepoints_with_excess=float(
            d.frac_timepoints_excess_positive.median()
        ),
        n_corrected_redchi2_lt3=int((d.excess_corrected_redchi2<3).sum()),
        n_corrected_redchi2_lt2=int((d.excess_corrected_redchi2<2).sum()),
        median_original_redchi2=float(d.original_redchi2.median()),
        median_corrected_redchi2=float(d.excess_corrected_redchi2.median()),
    )
    with open(OUT/"v26b_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)

    print("\nV26b COMPLETE")
    for k,v in summary.items(): print(f"{k}: {v}")
    print("\nPer-NV:")
    print(d.to_string(index=False))
    print("output:",OUT)


if __name__=="__main__":
    run()
