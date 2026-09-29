from pathlib import Path
import sys, json, numpy as np, pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import KMeans
REPO=Path(r"C:\Users\saroj\Github\dioptric")
sys.path.insert(0,str(REPO))
import analysis.sc_resonance_analysis_optimized as ra
from utils import widefield
OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v38_empirical_resonance_bfield\2026_09")
OUT.mkdir(parents=True,exist_ok=True)
DATA=[
("49G","2025_11_01-07_35_08-johnson-nv0_2025_10_21",[2.7666,2.7851,2.8222,2.8406]),
("65G","2025_11_20-09_14_44-johnson-nv0_2025_10_21",[2.7252,2.7464,2.8275,2.8487]),
("59G","2025_11_28-01_53_35-johnson-nv0_2025_10_21",[2.7081,2.8083,2.8251,2.8536]),
("62G","2025_12_10-10_28_25-johnson-nv0_2025_10_21",[2.7098,2.7859,2.8169,2.8706]),
]
rows=[]
rng=np.random.default_rng(12345)
for label,stem,hist in DATA:
    print("\n===",label,stem,"===")
    _,nv_list,freqs,sig,ref=ra.load_and_combine([stem])
    avg,ste=widefield.process_counts(nv_list,sig,ref,threshold=True)
    avg=np.asarray(avg,float); ste=np.asarray(ste,float)
    dense=np.linspace(np.nanmin(freqs),np.nanmax(freqs),45)
    with parallel_backend("loky",inner_max_num_threads=1):
        fits=Parallel(n_jobs=-1,verbose=3)(delayed(ra.fit_one_nv)(i,freqs,avg,ste,dense) for i in range(len(nv_list)))
    f1=np.array([x["popt"][2] if x["success"] else np.nan for x in fits],float)
    f2=np.array([x["popt"][3] if x["success"] else np.nan for x in fits],float)
    good=np.isfinite(f1)&np.isfinite(f2)&(f1<2.8785)&(f2>2.8785)
    x=f1[good].reshape(-1,1)
    km=KMeans(n_clusters=4,random_state=0,n_init=30).fit(x)
    centers=[]
    for c in range(4):
        vals=x[km.labels_==c,0]
        med=float(np.median(vals))
        boots=np.median(rng.choice(vals,(1000,len(vals)),replace=True),axis=1)
        se=float(np.std(boots,ddof=1))
        mad=float(1.4826*np.median(np.abs(vals-med)))
        centers.append((med,se,mad,len(vals)))
    centers=sorted(centers)
    hs=sorted(hist)
    for j,(med,se,mad,n) in enumerate(centers):
        rows.append(dict(field=label,raw_resonance=stem,branch=j,center_GHz=med,bootstrap_se_MHz=se*1000,robust_spread_MHz=mad*1000,n_nv=n,historical_GHz=hs[j],delta_hist_MHz=(med-hs[j])*1000))
    print("good",good.sum(),"/",len(nv_list))
    print("centers",centers)
    print("hist",hs)
df=pd.DataFrame(rows)
df.to_csv(OUT/"v38_empirical_odmr_centers.csv",index=False)
print("\n",df.to_string(index=False))
