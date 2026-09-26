"""Choose promising/representative NVs for the full V8b site search."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
CONTROLS=[0,16,28,54,93,135,150,168]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--summary",required=True)
    ap.add_argument("--near-limit",type=int,default=10);args=ap.parse_args()
    path=Path(args.summary);base=str(path).removesuffix("_nv_summary.csv")
    s=pd.read_csv(path);o=pd.read_csv(base+"_model_orders.csv")
    alts=o[o.model_order>0].groupby("nv_index").bic.min()
    s["best_site_bic_gain"]=s.background_bic-s.nv_index.map(alts)
    selected=set(s.loc[s.selected_order>0,"nv_index"].astype(int))
    near=s[(s.selected_order==0)&(s.best_site_bic_gain>=5)].nlargest(args.near_limit,"best_site_bic_gain")
    near_ids=set(near.nv_index.astype(int));controls=set(CONTROLS)&set(s.nv_index.astype(int))
    chosen=selected|near_ids|controls
    r=s[s.nv_index.isin(chosen)][["nv_index","selected_order","best_site_bic_gain",
                  "selected_red_chi2","model_adequacy","site_inference_status"]].copy()
    r["selection_reason"]=r.nv_index.map(lambda n:
       "+".join(x for x,keep in (("selected_site",n in selected),
                                     ("near_BIC_threshold",n in near_ids),("control",n in controls)) if keep))
    out=Path(base+"_refinement_selection.csv");r.sort_values("nv_index").to_csv(out,index=False)
    print("COUNT",len(chosen));print("NVS",",".join(map(str,sorted(chosen))));print("OUTPUT",out)
if __name__=="__main__":main()
