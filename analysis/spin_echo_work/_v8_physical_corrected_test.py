
import sys, json, ast
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, r"C:\Users\saroj\Github\dioptric")
from analysis.spin_echo_work import fitter_module_for_spin_echo as oldfit

ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\spin_echo\sc_c13_spin_echo_physics_fit_52G_nv_pillar_array\2026_09")
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

LB=np.array([0,0,15,1,.001,.6,0,0,-.06,0,-.30],float)
UB=np.array([1.05,.95,40,20,.60,4.0,4.0,.8,.06,1.00,.30],float)

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
        # Exact one-spin I=1/2 Hahn-echo factor.  The saved x-axis is total
        # evolution time t=2*tau, hence the 1/2 in the sine arguments.
        q=2.0*float(s["kappa"])*np.sin(0.5*np.pi*s["fI"]*teff)**2 * np.sin(0.5*np.pi*s["fm"]*teff)**2
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
    for T2 in [.01,.02,.04,.08,.15,.30,.55]:
      for beta in [.7,1.2,2.0,3.0,3.8]:
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

for nv in [0,16,28,54]:
    core=fit_core(nv)
    ori=om[nv]
    nyq=500.0/np.min(np.diff(np.unique(t)))
    # Catalog frequencies are conditional-precession frequencies.  On the
    # total-time axis the observable components are at half those values.
    sites=[r for r in cat if r["ori"]==ori and 1.4<=float(r["distance_A"])<=22 and float(r["kappa"])>1e-4 and 0.5*r["fplus"]<0.995*nyq]
    scores=[]
    pbase=np.r_[core[2],.5,0.]
    yy=y[nv];ee=e[nv]
    for s in sites:
      bests=1e99
      for eta in np.linspace(0,1.0,7):
        q=pbase.copy();q[9]=eta
        chi=np.sum(((yy-model(q,[s]))/ee)**2)
        if chi<bests:bests=chi
      scores.append((bests,s))
    scores.sort(key=lambda x:x[0])
    fitbest=None; bestsite=None
    for _,s in scores[:12]:
      f=fit_site(nv,s,core[2])
      if f is not None and (fitbest is None or f[0]<fitbest[0]):
        fitbest=f;bestsite=s
    print("NV",nv,"ori",ori,"core",round(core[1],3),"hyb",round(fitbest[1],3),"V6",round(float(v6.loc[nv,"best_red_chi2"]),3),
          "site",bestsite["site_id"],"k",round(bestsite["kappa"],3),"T2us",round(fitbest[2][4]*1000,1),"eta",round(fitbest[2][9],3),"dt",round(fitbest[2][10],3))

# ---- independent-spin multi-carbon physical test -----------------------------
def modelN(p, sites, tt=t):
    n=len(sites); bg=p[:9]; dt=p[9]; vis=p[10:10+n]
    b,C,T,w,T2,beta,a,ws,ch=bg
    env=np.exp(-((tt/(1000*T2))**beta)); car=env*comb(tt,T,w,a,ws,ch)
    teff=tt+dt; M=np.ones_like(tt,float)
    for v,s in zip(vis,sites):
        q=2.0*float(s["kappa"])*np.sin(0.5*np.pi*s["fI"]*teff)**2*np.sin(0.5*np.pi*s["fm"]*teff)**2
        M*=1-v*q
    return b-C*car*M

def fitN(nv,sites,bg0,dt0=0.0,vis0=None,maxn=12000):
    yy=y[nv]; ee=e[nv]; n=len(sites)
    if vis0 is None: vis0=np.repeat(.35,n)
    lb=np.r_[LB[:9],-.30,np.zeros(n)]; ub=np.r_[UB[:9],.30,np.ones(n)]
    base=np.r_[bg0,dt0,vis0]; fits=[]
    for t2m in [.7,1.,1.5]:
        for eta0 in [0.0,.25,.6]:
            q=base.copy(); q[4]=np.clip(bg0[4]*t2m,lb[4]+1e-8,ub[4]-1e-8)
            if n: q[10:]=np.clip(np.maximum(q[10:],eta0),0,1)
            try:
                r=least_squares(lambda x:(yy-modelN(x,sites))/ee,np.clip(q,lb+1e-8,ub-1e-8),
                                bounds=(lb,ub),loss="linear",max_nfev=maxn)
                pred=modelN(r.x,sites); chi,red=stats(yy,ee,pred,10+n)
                aic=chi+2*(10+n); aicc=aic+2*(10+n)*(11+n)/(len(yy)-(10+n)-1)
                fits.append((aicc,chi,red,r.x))
            except Exception: pass
    return min(fits,key=lambda z:(z[0],z[1])) if fits else None

def physical_screen(nv,core,keep=12):
    ori=om[nv]; nyq=500.0/np.min(np.diff(np.unique(t)))
    sites=[r for r in cat if r["ori"]==ori and 1.4<=float(r["distance_A"])<=22
           and float(r["kappa"])>1e-4 and 0.5*r["fplus"]<0.995*nyq]
    yy=y[nv]; ee=e[nv]; bg=core[2]; base=core_model(bg)
    scores=[]
    for s in sites:
        best=1e99
        for dt in [-.25,0,.25]:
            for eta in [0,.25,.5,.75,1.]:
                p=np.r_[bg,dt,eta]; chi=np.sum(((yy-modelN(p,[s]))/ee)**2)
                best=min(best,chi)
        scores.append((best,s))
    scores.sort(key=lambda z:z[0])
    return [s for _,s in scores[:keep]]

print("\nPHYSICAL MULTI-C13 TEST")
for nv in [0,16,28,54]:
    core=fit_core(nv); pool=physical_screen(nv,core,12)
    singles=[]
    for s in pool:
        f=fitN(nv,[s],core[2],0,[.35],8000)
        if f: singles.append((f[0],s,f))
    singles.sort(key=lambda z:z[0])
    best0_chi=core[0]; k0=9; npt=len(t)
    aic0=best0_chi+2*k0; aicc0=aic0+2*k0*(k0+1)/(npt-k0-1)

    pairs=[]
    top=[s for _,s,_ in singles[:7]]
    for s1,s2 in __import__("itertools").combinations(top,2):
        f=fitN(nv,[s1,s2],core[2],0,[.3,.3],9000)
        if f: pairs.append((f[0],(s1,s2),f))
    pairs.sort(key=lambda z:z[0])
    triples=[]
    top3=top[:5]
    for ss in __import__("itertools").combinations(top3,3):
        f=fitN(nv,list(ss),core[2],0,[.25,.25,.25],9000)
        if f: triples.append((f[0],ss,f))
    triples.sort(key=lambda z:z[0])
    candidates=[("N0",aicc0,None,None)]
    if singles: candidates.append(("N1",singles[0][0],[singles[0][1]],singles[0][2]))
    if pairs: candidates.append(("N2",pairs[0][0],list(pairs[0][1]),pairs[0][2]))
    if triples: candidates.append(("N3",triples[0][0],list(triples[0][1]),triples[0][2]))
    print("\nNV",nv,"V6",round(float(v6.loc[nv,"best_red_chi2"]),3),"core",round(core[1],3))
    for name,aicc,ss,f in candidates:
        if f is None: print(name,"aicc",round(aicc,1)); continue
        print(name,"aicc",round(aicc,1),"red",round(f[2],3),
              "sites",[x["site_id"] for x in ss],"vis",np.round(f[3][10:],3),
              "T2us",round(1000*f[3][4],1))
