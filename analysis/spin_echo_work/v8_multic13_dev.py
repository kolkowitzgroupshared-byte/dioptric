import ast, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import least_squares

ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\spin_echo\sc_c13_spin_echo_physics_fit_52G_nv_pillar_array\2026_09")
CK=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_fit_checkpoint.npz"
V6=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_orientation_locked_confidence_v6_all_equal_footing_sites.csv"
ORI=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_orientation_locked_confidence_v6_orientation_assignments.csv"
CAT=Path(r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json")
TEST=[28]
B=np.array([-48.551229,-18.748242,-5.973533])
FI=1.0705*np.linalg.norm(B)
TREV=2000/FI
print("FI",FI,"Trev total",TREV)

z=np.load(CK,allow_pickle=True); t=z["times_us"].astype(float); Y=z["norm_counts"].astype(float); E=np.maximum(z["norm_counts_ste"].astype(float),1e-4)
v6=pd.read_csv(V6); best6=v6[v6.site_rank==1].set_index("nv_index")
odf=pd.read_csv(ORI)
ori={int(r.nv_index):tuple(ast.literal_eval(r.orientation)) for r in odf.itertuples()}
cat=json.load(open(CAT))
for r in cat:
    r["ori"]=tuple(r["orientation"]); r["site"]=int(r["site_index"])
    r["fI"]=r["fI_Hz"]/1e3; r["fm"]=r["omega_ms_Hz"]/1e3
    r["fminus"]=r["f_minus_Hz"]/1e3; r["fplus"]=r["f_plus_Hz"]/1e3

# background params:
# baseline, contrast, Trev, width0, T2us, beta, alpha, width_slope, chirp, tau_offset
LB=np.array([0.0,0.0,15.0,1.0,1.0,0.6,0.0,0.0,-0.06,-0.50])
UB=np.array([1.10,0.95,40.0,20.0,600.0,4.0,4.0,0.8,0.06,0.50])

def comb(t,T,w,a,ws,ch):
    out=np.zeros_like(t)
    n=max(2,int(np.ceil(t.max()/T))+3)
    for k in range(n):
        mu=k*T*(1+k*ch); wk=w*(1+k*ws)
        if wk<=0: continue
        out += (1/(1+k)**a)*np.exp(-((t-mu)/wk)**4)
    return out

def qsite(t,rec,tauoff):
    # t is total echo evolution 2*tau; tau_eff=t/2 + tauoff
    tau=0.5*t + tauoff
    a=np.sin(np.pi*(rec["fI"]/1000.0)*tau)**2
    b=np.sin(np.pi*(rec["fm"]/1000.0)*tau)**2
    return 2*rec["kappa"]*a*b

def pred(t,p,sites=(),etas=()):
    b,c,T,w,T2,beta,alpha,ws,ch,t0=p[:10]
    env=np.exp(-(np.maximum(t,0)/T2)**beta)
    carrier=env*comb(t,T,w,alpha,ws,ch)
    M=np.ones_like(t)
    for rec,eta in zip(sites,etas):
        M*=1-np.clip(eta,0,1)*qsite(t,rec,t0)
    return b-c*carrier*M

def stats(y,e,yp,k):
    rr=(y-yp)/e; chi=float(rr@rr); n=len(y); red=chi/max(1,n-k)
    aic=chi+2*k; aicc=aic+2*k*(k+1)/max(1,n-k-1); bic=chi+k*np.log(n)
    return chi,red,aicc,bic

def seed_from_v6(nv):
    r=best6.loc[nv]
    return np.array([r.baseline,r.comb_contrast,r.revival_time_us,r.width0_us,r.T2_us,r.T2_exp,r.amp_taper_alpha,r.width_slope,r.revival_chirp,0.0],float)

def fit_bg(nv):
    y=Y[nv];e=E[nv]; s=np.clip(seed_from_v6(nv),LB+1e-8,UB-1e-8)
    starts=[]
    for T2 in [max(8,min(250,s[4])),25,50,100]:
      for beta in [max(.7,min(3,s[5])),1.0,2.0]:
        x=s.copy(); x[4]=T2;x[5]=beta
        try:
          res=least_squares(lambda p:(y-pred(t,p))/e,x,bounds=(LB,UB),loss="soft_l1",f_scale=1,max_nfev=4000)
          # polish with linear loss for valid likelihood
          res2=least_squares(lambda p:(y-pred(t,p))/e,res.x,bounds=(LB,UB),loss="linear",max_nfev=4000)
          st=stats(y,e,pred(t,res2.x),10); starts.append((st[0],res2.x,st))
        except Exception: pass
    return min(starts,key=lambda x:x[0])

def eta_screen(nv,p0,rec):
    y=Y[nv];e=E[nv]
    base=pred(t,p0)
    b,c,T,w,T2,beta,alpha,ws,ch,t0=p0
    carrier=np.exp(-(np.maximum(t,0)/T2)**beta)*comb(t,T,w,alpha,ws,ch)
    x=c*carrier*qsite(t,rec,t0)
    ww=1/e**2
    eta=np.clip(np.sum(ww*x*(y-base))/max(1e-15,np.sum(ww*x*x)),0,1)
    yp=base+eta*x
    return stats(y,e,yp,11)[0],eta

def fit_sites(nv,pbg,sites,eta0):
    y=Y[nv];e=E[nv]; nsite=len(sites)
    lb=np.r_[LB,np.zeros(nsite)]; ub=np.r_[UB,np.ones(nsite)]
    starts=[]
    for scale in [0.7,1.0,1.5]:
      x=np.r_[pbg,eta0].astype(float); x[4]=np.clip(pbg[4]*scale,LB[4]+1e-6,UB[4]-1e-6)
      def fun(v): return (y-pred(t,v[:10],sites,v[10:]))/e
      try:
        r=least_squares(fun,np.clip(x,lb+1e-8,ub-1e-8),bounds=(lb,ub),loss="soft_l1",f_scale=1,max_nfev=6000)
        r2=least_squares(fun,r.x,bounds=(lb,ub),loss="linear",max_nfev=6000)
        st=stats(y,e,pred(t,r2.x[:10],sites,r2.x[10:]),10+nsite)
        starts.append((st[0],r2.x,st))
      except Exception: pass
    return min(starts,key=lambda x:x[0]) if starts else None

for nv in TEST:
    print("\nNV",nv,"ori",ori[nv],"V6 red",best6.loc[nv].red_chi2,"site",best6.loc[nv].site_id)
    bg=fit_bg(nv); print(" BG red",bg[2][1],"bic",bg[2][3],"p",np.round(bg[1],3))
    pool=[r for r in cat if r["ori"]==ori[nv] and r["distance_A"]>1.4 and r["distance_A"]<22 and r["fplus"]<1400]
    scr=sorted([(eta_screen(nv,bg[1],r)[0],eta_screen(nv,bg[1],r)[1],r) for r in pool],key=lambda x:x[0])[:12]
    singles=[]
    for _,eta,r in scr:
        ff=fit_sites(nv,bg[1],[r],[eta])
        if ff: singles.append((ff[2][2],ff[2][3],r,ff))
    singles.sort(key=lambda x:x[0])
    for j,x in enumerate(singles[:5]):
        print(" S",j+1,x[2]["site"],"k",round(x[2]["kappa"],3),"fm/fp",round(x[2]["fminus"],1),round(x[2]["fplus"],1),"eta",round(x[3][1][10],3),"red",round(x[3][2][1],3),"bic",round(x[1],2))
    bestS=singles[0]
    # add second site by screening all sites around the full single solution; fully fit top 20
    p1=bestS[3][1][:10]; eta1=bestS[3][1][10]; s1=bestS[2]
    pair_scr=[]
    y=Y[nv];e=E[nv]
    base1=pred(t,p1,[s1],[eta1])
    b,c,T,w,T2,beta,alpha,ws,ch,t0=p1
    carrier=np.exp(-(np.maximum(t,0)/T2)**beta)*comb(t,T,w,alpha,ws,ch)
    M1=1-eta1*qsite(t,s1,t0)
    for r in pool:
      if r["site"]==s1["site"]: continue
      # reject nearly identical spectral twins
      if np.hypot(r["fminus"]-s1["fminus"],r["fplus"]-s1["fplus"])<2.0: continue
      x=c*carrier*M1*qsite(t,r,t0)
      ww=1/e**2; eta2=np.clip(np.sum(ww*x*(y-base1))/max(1e-15,np.sum(ww*x*x)),0,1)
      chi=stats(y,e,base1+eta2*x,12)[0]
      pair_scr.append((chi,eta2,r))
    pair_scr.sort(key=lambda x:x[0])
    pairs=[]
    for _,eta2,r in pair_scr[:8]:
      ff=fit_sites(nv,p1,[s1,r],[eta1,eta2])
      if ff:pairs.append((ff[2][2],ff[2][3],r,ff))
    pairs.sort(key=lambda x:x[0])
    if pairs:
      bp=pairs[0]
      s2=bp[2]; p2=bp[3][1][:10]; etas2=bp[3][1][10:]
      print(" PAIR",s1["site"],s2["site"],"etas",np.round(etas2,3),"red",round(bp[3][2][1],3),"bic",round(bp[1],2),"dBIC vs single",round(bp[1]-bestS[1],2))
      # third-site forward selection
      y=Y[nv]; e=E[nv]
      base2=pred(t,p2,[s1,s2],etas2)
      b,c,T,w,T2,beta,alpha,ws,ch,t0=p2
      carrier=np.exp(-(np.maximum(t,0)/T2)**beta)*comb(t,T,w,alpha,ws,ch)
      M12=(1-etas2[0]*qsite(t,s1,t0))*(1-etas2[1]*qsite(t,s2,t0))
      tri_scr=[]
      for r in pool:
        if r["site"] in (s1["site"],s2["site"]): continue
        if min(np.hypot(r["fminus"]-ss["fminus"],r["fplus"]-ss["fplus"]) for ss in [s1,s2])<2.0: continue
        x=c*carrier*M12*qsite(t,r,t0)
        ww=1/e**2
        eta3=np.clip(np.sum(ww*x*(y-base2))/max(1e-15,np.sum(ww*x*x)),0,1)
        chi=stats(y,e,base2+eta3*x,13)[0]
        tri_scr.append((chi,eta3,r))
      tri_scr.sort(key=lambda x:x[0])
      triples=[]
      for _,eta3,r in tri_scr[:8]:
        ff=fit_sites(nv,p2,[s1,s2,r],[etas2[0],etas2[1],eta3])
        if ff: triples.append((ff[2][2],ff[2][3],r,ff))
      triples.sort(key=lambda x:x[0])
      if triples:
        bt=triples[0]
        print(" TRIPLE",s1["site"],s2["site"],bt[2]["site"],"etas",np.round(bt[3][1][10:],3),"red",round(bt[3][2][1],3),"bic",round(bt[1],2),"dBIC vs pair",round(bt[1]-bp[1],2))
