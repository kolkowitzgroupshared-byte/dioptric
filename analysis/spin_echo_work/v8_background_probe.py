"""Diagnostic: compare empirical backgrounds with ideal single-site coherence."""
import ast, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
REPO=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO))
from analysis.spin_echo_work import sc_c13_spin_echo_physics_fit_52G_v8_multic13 as v8
ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\spin_echo\sc_c13_spin_echo_physics_fit_52G_nv_pillar_array\2026_09")
z=np.load(next(ROOT.glob("*ranked_52G_fit_checkpoint.npz")),allow_pickle=True)
t=z["times_us"].astype(float); Y=z["norm_counts"].astype(float); E=np.maximum(z["norm_counts_ste"].astype(float),1e-4)
V=pd.read_csv(next(ROOT.glob("*v6_all_equal_footing_sites.csv")))
O=pd.read_csv(next(ROOT.glob("*v6_orientation_assignments.csv"))).set_index("nv_index")
cat=json.load(open(REPO/"analysis/spin_echo_work/essem_freq_kappa_catalog_22A_52G.json"))
for s in cat:
    s["orientation_tuple"]=tuple(s["orientation"]);s["site_id"]=int(s["site_index"])
    s["fI_kHz"]=s["fI_Hz"]/1e3;s["fm_kHz"]=s["omega_ms_Hz"]/1e3
    s["fplus_kHz"]=s["f_plus_Hz"]/1e3;s["fminus_kHz"]=s["f_minus_Hz"]/1e3
NVS=[0,16,28,54,93]
def early(tt):
    tt=np.asarray(tt,float)
    return np.exp(-((tt-2.8)/1.4)**2)-.35*np.exp(-((tt-5.2)/1.4)**2)
def curve(tt,bg,sites=(),etas=(),dt=0):
    b,c,T,w,T2,beta,slope=bg[:7]
    a=bg[7] if len(bg)>7 else 0.
    alpha=bg[8] if len(bg)>8 else 0.
    tt=np.asarray(tt,float);rev=np.zeros_like(tt)
    for k in range(5):
        rev+=(1+k)**(-alpha)*np.exp(-((tt-k*T)/(w*(1+k*slope)))**4)
    env=np.exp(-((np.maximum(tt,0)/T2)**beta))
    L=np.ones_like(tt)
    for site,eta in zip(sites,etas): L*=1-eta*v8.site_q(tt,site,dt)
    return b-c*env*rev*L+a*early(tt)
def get_bounds(variant):
    lo=np.array([0,0,32,1,3,.6,0],float)
    hi=np.array([1.1,.95,38,15,600,4,.8],float)
    if variant>=1: lo=np.r_[lo,-.8];hi=np.r_[hi,.8]
    if variant>=2: lo=np.r_[lo,0];hi=np.r_[hi,4.]
    return lo,hi
def fit0(nv,variant):
    y,e=Y[nv],E[nv];lb,ub=get_bounds(variant)
    old=V[(V.nv_index==nv)&(V.site_rank==1)].iloc[0]
    seed=np.array([old.baseline,old.comb_contrast,old.revival_time_us,
                   old.width0_us,old.T2_us,old.T2_exp,old.width_slope],float)
    if variant>=1:seed=np.r_[seed,0.]
    if variant>=2:seed=np.r_[seed,old.amp_taper_alpha]
    out=[]
    for T2 in (15.,60.,200.):
      for beta in (1.,2.,3.):
        s=np.clip(seed,lb+1e-8,ub-1e-8);s[4]=T2;s[5]=beta
        r=least_squares(lambda x:(y-curve(t,x))/e,s,bounds=(lb,ub),max_nfev=1500)
        out.append((r.fun@r.fun,r.x))
    return min(out,key=lambda q:q[0])
def screen(nv,bg,ori):
    y,e=Y[nv],E[nv];base=curve(t,bg);b,c,T,w,T2,beta,slope=bg[:7]
    scores=[]
    for s in cat:
      if s["orientation_tuple"]!=ori or not (1.4<=s["distance_A"]<=22):continue
      if s["kappa"]<1e-5:continue
      basis=curve(t,bg,[s],[1.])-base
      r=y-base
      wgt=1/e**2
      eta=np.clip(np.sum(wgt*basis*r)/max(1e-15,np.sum(wgt*basis*basis)),0,1)
      chi=np.sum(((r-eta*basis)/e)**2)
      scores.append((chi,eta,s))
    return sorted(scores,key=lambda q:q[0])
def fit1(nv,bg,site,eta):
    y,e=Y[nv],E[nv];lb,ub=get_bounds(len(bg)-7)
    lb=np.r_[lb,0.,-.25];ub=np.r_[ub,1.,.25]
    fits=[]
    for e0 in (max(.03,float(eta)),.65):
      s=np.r_[bg,e0,0.]
      res=least_squares(lambda x:(y-curve(t,x[:-2],[site],[x[-2]],x[-1]))/e,
          np.clip(s,lb+1e-8,ub-1e-8),bounds=(lb,ub),max_nfev=2500)
      fits.append((res.fun@res.fun,res.x))
    return min(fits,key=lambda q:q[0])
rows=[]
for variant,name in [(0,"base"),(1,"early"),(2,"early_taper")]:
  for nv in NVS:
    o=tuple(ast.literal_eval(O.loc[nv,"orientation"])); bgchi,bg=fit0(nv,variant)
    sites=screen(nv,bg,o);best=None
    for _,eta,s in sites[:15]:
      r=fit1(nv,bg,s,eta)
      if best is None or r[0]<best[0]:best=(r[0],r[1],s)
    old=V[(V.nv_index==nv)&(V.site_rank==1)].iloc[0]
    row=dict(variant=name,nv_index=nv,v6_red=float(old.red_chi2),
      N0_red=bgchi/(94-len(bg)),N1_red=best[0]/(94-len(bg)-2),
      N1_site=best[2]["site_id"],kappa=best[2]["kappa"],
      eta=best[1][-2],T2_us=best[1][4],beta=best[1][5],
      early_amp=best[1][7] if variant>=1 else np.nan,
      taper=best[1][8] if variant>=2 else np.nan)
    rows.append(row);print(row,flush=True)
out=ROOT/"2026_09_23_v8_background_probe.csv"
pd.DataFrame(rows).to_csv(out,index=False);print("OUTPUT",out)
