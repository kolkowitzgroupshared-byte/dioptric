"""Block-residual perturbation stability conditional on V8b catalog shortlists."""
import argparse,ast,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from analysis.spin_echo_work import sc_spin_echo_physics_fit_52G_v8b_multic13 as m
def sample_blocks(resid,rng,block=5):
    n=len(resid);out=[]
    while len(out)<n:
        i=int(rng.integers(n));out.extend(resid[(i+np.arange(block))%n])
    return np.asarray(out[:n],float)
def run_nv(nv,reps,t,y,e,candidates,lookup):
    rng=np.random.default_rng(20260923+nv)
    rows=candidates[candidates.nv_index==nv]
    ori=m.canonical_orientation(rows.iloc[0].orientation)
    specs=[]
    for order in (0,1,2):
        for _,r in rows[rows.model_order==order].head(5).iterrows():
            sites=[lookup[(ori,int(r[f"c13_{j}_site_id"]))] for j in range(1,order+1)]
            specs.append((order,sites,np.asarray(json.loads(r.theta_json),float)))
    bestrows={o:g.iloc[0] for o,g in rows.groupby("model_order",sort=True)}
    chosen,_fit,_dec=m.choose_model_order({o:dict(bic=float(r.bic)) for o,r in bestrows.items()})
    observed=bestrows[chosen]
    theta=np.asarray(json.loads(observed.theta_json),float)
    bg,etas,dt=m.unpack_theta(theta,chosen)
    sites=[lookup[(ori,int(observed[f"c13_{j}_site_id"]))] for j in range(1,chosen+1)]
    baseline=m.model_from_parts(t,bg,sites,etas,dt)
    standard=(y[nv]-baseline)/e[nv]
    out=[]
    for rep in range(reps):
        yb=baseline+e[nv]*sample_blocks(standard,rng)
        fitted={}
        for order,sites,theta in specs:
            fit=m.fit_site_set(t,yb,e[nv],sites,[theta])
            if fit is None:continue
            if order not in fitted or fit["bic"]<fitted[order]["bic"]:
                fitted[order]=fit
        if 0 not in fitted:continue
        selected_order,selected,decisions=m.choose_model_order(fitted)
        out.append(dict(nv_index=nv,replicate=rep,orientation=str(ori),
            selected_order=selected_order,selected_sites=str(selected["site_key"]),
            selected_red_chi2=selected["red_chi2"],
            best_0_bic=fitted[0]["bic"],
            best_1_bic=fitted.get(1,{}).get("bic",np.nan),
            best_2_bic=fitted.get(2,{}).get("bic",np.nan),
            shortlist_only=True,block_length=5))
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate-csv",required=True)
    ap.add_argument("--nv",default="28,54,93,135,150,168")
    ap.add_argument("--reps",type=int,default=12)
    ap.add_argument("--workers",type=int,default=6)
    args=ap.parse_args()
    paths=m.discover_paths()
    t,y,e,odf,ori_map,quality,seeds,cat=m.load_inputs(paths)
    lookup={(tuple(s["orientation_tuple"]),int(s["site_id"])):s for s in cat}
    cdf=pd.read_csv(args.candidate_csv)
    nvlist=[int(x) for x in args.nv.split(",")]
    with threadpool_limits(limits=1):
        out=Parallel(n_jobs=args.workers,backend="loky",verbose=5)(
            delayed(run_nv)(nv,args.reps,t,y,e,cdf,lookup) for nv in nvlist)
    df=pd.DataFrame([r for batch in out for r in batch])
    output=Path(str(args.candidate_csv).replace("_candidate_fits.csv","_block_residual_stability.csv"))
    df["candidate_source"]=str(args.candidate_csv)
    df.to_csv(output,index=False)
    print(df.groupby(["nv_index","selected_order"]).size().to_string())
    print("OUTPUT",output)
if __name__=="__main__":main()
