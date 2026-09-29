"""Merge a full quick pass with deeper NV refinements and regenerate all-NV diagnostics."""
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from analysis.spin_echo_work import sc_c13_spin_echo_physics_fit_52G_v8b_multic13 as m
SUFFIXES={"summary":"_nv_summary.csv","candidates":"_candidate_fits.csv",
          "orders":"_model_orders.csv","profiles":"_t2_profiles.csv"}
def read_run(summary_path):
    base=str(Path(summary_path)).removesuffix("_nv_summary.csv")
    cfg=json.loads(Path(base+"_config.json").read_text(encoding="utf-8"))
    frames={key:pd.read_csv(base+suffix) for key,suffix in SUFFIXES.items()}
    return base,cfg,frames
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--quick-summary",required=True)
    ap.add_argument("--deep-summary",required=True);args=ap.parse_args()
    qb,qc,q=read_run(args.quick_summary);db,dc,d=read_run(args.deep_summary)
    for key in ("script_sha256","checkpoint_sha256","orientation_sha256","catalog_sha256"):
        if qc[key]!=dc[key]:raise ValueError(f"Incompatible source {key}")
    full=set(q["summary"].nv_index.astype(int)); refined=set(d["summary"].nv_index.astype(int))
    if len(full)!=212 or not refined or not refined.issubset(full):raise ValueError("Expected 212-NV quick pass and subset refinement")
    merged={}
    for key in SUFFIXES:
        a=q[key][~q[key].nv_index.isin(refined)].copy();a["search_pass"]="quick"
        b=d[key].copy();b["search_pass"]="deep"
        merged[key]=pd.concat([a,b],ignore_index=True).sort_values("nv_index")
        if key=="summary" and merged[key].nv_index.nunique()!=212:raise ValueError("Duplicate/missing NV")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base=Path(qb).parent/("v8b_final_"+stamp+"_n212")
    for key,suffix in SUFFIXES.items():merged[key].to_csv(str(base)+suffix,index=False)
    cfg=dict(qc);cfg.update(utc_stamp=stamp,quick=None,fit_mode="staged_quick_and_deep",
                         quick_source=qb,deep_source=db,
                         refined_nv=sorted(refined),nvs=sorted(full),
                         merged_search_passes={"quick":212-len(refined),"deep":len(refined)})
    Path(str(base)+"_config.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    m.CHECKPOINT_PATH=qc["checkpoint"];m.ORIENTATION_ASSIGNMENTS_CSV=qc["orientation_csv"]
    m.V6_ALL_SITE_FITS=qc["v6_seed_csv"];m.CATALOG_PATH=Path(qc["catalog"])
    paths=m.discover_paths();t,y,e,_,_,_,_,catalog=m.load_inputs(paths)
    m.DENSE_POINTS=1200
    with PdfPages(str(base)+"_dashboard.pdf") as pdf:
        for j,nv in enumerate(sorted(full),1):
            m.plot_dashboard(pdf,nv,t,y,e,merged["summary"],merged["candidates"],
                             merged["orders"],merged["profiles"],catalog)
            if j%20==0:print(f"Dashboard {j}/212",flush=True)
    fig=m.global_summary_figure(merged["summary"])
    fig.savefig(str(base)+"_global_summary.png",dpi=300,bbox_inches="tight")
    fig.savefig(str(base)+"_global_summary.pdf",bbox_inches="tight");plt.close(fig)
    print("MERGED",base,"refined",len(refined));print("orders",merged["summary"].selected_order.value_counts().to_dict())
if __name__=="__main__":main()
