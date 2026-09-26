"""Group near-degenerate catalog signatures within each V8b searched shortlist."""
import argparse,ast,json
from pathlib import Path
import numpy as np,pandas as pd
def signature(row,order,bin_khz):
    parts=[]
    for j in range(1,order+1):
        fm=float(row[f"c13_{j}_fminus_kHz"])
        fp=float(row[f"c13_{j}_fplus_kHz"])
        parts.append((int(np.rint(fm/bin_khz)),int(np.rint(fp/bin_khz))))
    return tuple(sorted(parts))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates",required=True)
    ap.add_argument("--bin-khz",type=float,default=10.)
    args=ap.parse_args()
    inp=Path(args.candidates); df=pd.read_csv(inp)
    rows=[]
    for (nv,order),group in df[df.model_order>0].groupby(["nv_index","model_order"]):
        groups={}
        for _,r in group.iterrows():
            sig=signature(r,int(order),args.bin_khz)
            groups.setdefault(sig,[]).append(r)
        for sig,members in groups.items():
            rank=sorted(members,key=lambda r:r.aicc)
            best=rank[0]
            rows.append(dict(nv_index=int(nv),orientation=best.orientation,model_order=int(order),
                frequency_bin_kHz=args.bin_khz,signature=json.dumps(sig),
                family_weight_within_shortlist=sum(float(r.akaike_weight_within_order) for r in members),
                best_site_key=best.site_key,best_aicc=float(best.aicc),
                members=json.dumps([str(r.site_key) for r in rank]),
                n_members=len(members)))
    out=pd.DataFrame(rows)
    out["family_rank"]=out.groupby(["nv_index","model_order"])["family_weight_within_shortlist"].rank(
        ascending=False,method="first").astype(int)
    out=out.sort_values(["nv_index","model_order","family_rank"])
    label="_site_families.csv" if args.bin_khz==10 else f"_site_families_{args.bin_khz:g}kHz.csv"
    output=Path(str(inp).replace("_candidate_fits.csv",label))
    out.to_csv(output,index=False)
    print("OUTPUT",output,"rows",len(out))
    print(out[out.family_rank==1].groupby("model_order").family_weight_within_shortlist.median())
if __name__=="__main__":main()

