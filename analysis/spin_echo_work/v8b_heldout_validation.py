"""Nested, stratified held-out validation of V8b model orders on selected NVs."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from analysis.spin_echo_work import sc_spin_echo_physics_fit_52G_v8b_multic13 as m
NVS=[0,16,28,54,93,135,150,168]
def run_nv(nv,t,y,e,ori,cat):
    rows=[]
    for fold in (0,1):
        # Keep sparse short-time points in training to determine nuisance amplitude.
        # Hold out the later echo structure where a resolved site is informative.
        hold=((np.arange(len(t))%3)==fold)&(t>=15.0); train=~hold
        tt,yy,ee=t[train],y[nv,train],e[nv,train]
        bg=m.fit_background(tt,yy,ee,m.data_seed(yy))
        fit0=m.make_order0_record(bg)
        pool=m.catalog_for_nv(cat,ori,tt)
        screen=m.screen_single_sites(tt,yy,ee,bg["bg"],pool)
        singles=m.refit_singles(tt,yy,ee,bg["bg"],screen[:50],10)
        best={0:fit0}
        if singles:best[1]=singles[0]
        pairs=m.search_next_order(tt,yy,ee,singles,pool,2,8,12) if singles else []
        if pairs:best[2]=pairs[0]
        for order,fit in sorted(best.items()):
            b,v,dt=m.unpack_theta(fit["theta"],order)
            pr=m.model_from_parts(t[hold],b,fit["sites"],v,dt)
            testchi=float(np.sum(((y[nv,hold]-pr)/e[nv,hold])**2))
            rows.append(dict(nv_index=nv,fold=fold,orientation=str(ori),
                order=order,sites=str(fit["site_key"]),train_chi2=fit["chi2"],
                train_bic=fit["bic"],heldout_chi2=testchi,
                heldout_chi2_per_point=testchi/hold.sum(),
                train_points=int(train.sum()),heldout_points=int(hold.sum())))
    return rows
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nv",default=",".join(map(str,NVS)))
    ap.add_argument("--workers",type=int,default=6)
    args=ap.parse_args()
    paths=m.discover_paths()
    t,y,e,odf,ori_map,quality,seeds,cat=m.load_inputs(paths)
    nvlist=[int(x) for x in args.nv.split(",")]
    with threadpool_limits(limits=1):
        out=Parallel(n_jobs=args.workers,backend="loky",verbose=5)(
            delayed(run_nv)(nv,t,y,e,ori_map[nv],cat) for nv in nvlist)
    df=pd.DataFrame([r for group in out for r in group])
    output=paths.prefix.parent/"v8b_heldout_validation_late.csv"
    df.to_csv(output,index=False)
    print(df.pivot_table(index="nv_index",columns="order",values="heldout_chi2_per_point").round(3))
    print("OUTPUT",output)
if __name__=="__main__":main()

