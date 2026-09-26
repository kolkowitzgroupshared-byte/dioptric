
import sys, json, ast
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, r"C:\Users\saroj\Github\dioptric")
from analysis.spin_echo_work import fitter_module_for_spin_echo as oldfit

ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\sc_spin_echo_physics_fit_52G_nv_pillar_array\2026_09")
CK=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_fit_checkpoint.npz"
ORI=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_orientation_locked_confidence_v6_orientation_assignments.csv"
V6=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_orientation_locked_confidence_v6_nv_confidence_summary.csv"
CAT=Path(r"C:\Users\saroj\Github\dioptric\analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json")

z=np.load(CK,allow_pickle=True)
t=np.asarray(z["times_us"],float); y=np.asarray(z["norm_counts"],float); e=np.maximum(np.asarray(z["norm_counts_ste"],float),1e-4)
odf=pd.read_csv(ORI); om={int(r.nv_index):tuple(ast.literal_eval(r.orientation)) for r in odf.itertuples()}
v6=pd.read_csv(V6).set_index("nv_index")
cat=json.load(open(CAT))
for r in cat:
    r["ori"]=tuple(r["orientation"])
    r["site_id"]=int(r["site_index"])
    r["fI"]=float(r["fI_Hz"])/1e6
    r["fm"]=float(r["omega_ms_Hz"])/1e6
    r["fminus"]=float(r["f_minus_Hz"])/1e3
    r["fplus"]=float(r["f_plus_Hz"])/1e3
    r["fminus_cyc"]=float(r["f_minus_Hz"])/1e6
    r["fplus_cyc"]=float(r["f_plus_Hz"])/1e6

LB=np.array([0,0,28,1,.003,.6,0,0,-.02,0,-.20],float)
UB=np.array([1.1,.95,40,14,.30,3.0,2.0,.5,.02,1.25,.20],float)

def comb(t,T,w,a,ws,ch):
    n=max(1,min(64,int(np.ceil(1.2*np.max(t)/T))+1))
    return oldfit._comb_quartic_powerlaw(np.asarray(t,float),T,w,a,ws,ch,n)

def model(p, sites, tt=t):
    b,C,T,w,T2,beta,a,ws,ch,eta,dt=p
    env=np.exp(-((tt/(1000*T2))**beta))
    car=env*comb(tt,T,w,a,ws,ch)
    teff=tt+dt
    M=np.ones_like(tt)
    for s in sites:
        q=np.sin(np.pi*s["fplus_cyc"]*teff)**2 * np.sin(np.pi*s["fminus_cyc"]*teff)**2
        M*=1-eta*q
    return b-C*car*M

def core_model(p9,tt=t):
    b,C,T,w,T2,beta,a,ws,ch=p9
    env=np.exp(-((tt/(1000*T2))**beta))
    return b-C*env*comb(tt,T,w,a,ws,ch)

def stats(yy,ee,pred,k):
    chi=float(np.sum(((yy-pred)/ee)**2)); dof=max(1,len(yy)-k); return chi,chi/dof

def fit_core(nv):
    yy=y[nv]; ee=e[nv]
    p0=np.array([np.percentile(yy,90),min(.6,max(.05,np.percentile(yy,90)-np.percentile(yy,5))),35.66,5.5,.05,1.5,.4,.1,0])
    lb=LB[:9]; ub=UB[:9]
    best=None
    for T2 in [.02,.04,.08,.15]:
      for beta in [1.,1.5,2.]:
        q=p0.copy(); q[4]=T2;q[5]=beta
        res=least_squares(lambda p:(yy-core_model(p))/ee,np.clip(q,lb+1e-8,ub-1e-8),bounds=(lb,ub),loss="linear",max_nfev=8000)
        chi,red=stats(yy,ee,core_model(res.x),9)
        if best is None or chi<best[0]:best=(chi,red,res.x)
    return best

def fit_site(nv,site,core):
    yy=y[nv];ee=e[nv]
    p0=np.r_[core, .5, 0.]
    best=None
    for eta in [.15,.4,.8,1.0]:
      for dt in [-.08,0,.08]:
        q=p0.copy();q[9]=eta;q[10]=dt
        try:
          res=least_squares(lambda p:(yy-model(p,[site]))/ee,np.clip(q,LB+1e-8,UB-1e-8),bounds=(LB,UB),loss="linear",max_nfev=12000)
        except Exception:
          continue
        pred=model(res.x,[site]);chi,red=stats(yy,ee,pred,11)
        if best is None or chi<best[0]:best=(chi,red,res.x)
    return best

for nv in [0,5,8,15,16,28,30,51,54,75,93,101,151,164]:
    core=fit_core(nv)
    ori=om[nv]
    sites=[r for r in cat if r["ori"]==ori and 1.4<=float(r["distance_A"])<=22 and float(r["kappa"])>1e-4 and r["fplus"]<1300]
    scores=[]
    pbase=np.r_[core[2],.5,0.]
    yy=y[nv];ee=e[nv]
    for s in sites:
      bests=1e99
      for eta in np.linspace(0,1.2,13):
        q=pbase.copy();q[9]=eta
        chi=np.sum(((yy-model(q,[s]))/ee)**2)
        if chi<bests:bests=chi
      scores.append((bests,s))
    scores.sort(key=lambda x:x[0])
    fitbest=None; bestsite=None
    for _,s in scores[:30]:
      f=fit_site(nv,s,core[2])
      if f is not None and (fitbest is None or f[0]<fitbest[0]):
        fitbest=f;bestsite=s
    print("NV",nv,"ori",ori,"core",round(core[1],3),"hyb",round(fitbest[1],3),"V6",round(float(v6.loc[nv,"best_red_chi2"]),3),
          "site",bestsite["site_id"],"k",round(bestsite["kappa"],3),"T2us",round(fitbest[2][4]*1000,1),"eta",round(fitbest[2][9],3),"dt",round(fitbest[2][10],3))
