import ast,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import least_squares
from analysis.spin_echo_work import fitter_module_for_spin_echo as oldfit

ROOT=Path(r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\sc_spin_echo_physics_fit_52G_nv_pillar_array\2026_09")
CK=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_fit_checkpoint.npz"
V6=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_orientation_locked_confidence_v6_all_equal_footing_sites.csv"
ORI=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_orientation_locked_confidence_v6_orientation_assignments.csv"
ATT=ROOT/"2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_all_attempts.csv.gz"
CAT=Path(r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json")
TEST=[0,28,54,16]
z=np.load(CK,allow_pickle=True);t=z["times_us"].astype(float);Y=z["norm_counts"].astype(float);E=np.maximum(z["norm_counts_ste"].astype(float),1e-4)
v6=pd.read_csv(V6); best6=v6[v6.site_rank==1].set_index("nv_index")
odf=pd.read_csv(ORI);ori={int(r.nv_index):tuple(ast.literal_eval(r.orientation)) for r in odf.itertuples()}
att=pd.read_csv(ATT)
att["orit"]=att.orientation.map(lambda x:tuple(ast.literal_eval(x)))
cat=json.load(open(CAT)); lookup={(tuple(r["orientation"]),int(r["site_index"])):r for r in cat}

# background params in exact old order minus oscillator:
# b,c,T,w,T2ms,beta,alpha,ws,ch
LBbg=np.array([0,0,15,1,.001,.6,0,0,-.06],float)
UBbg=np.array([1.05,.95,40,20,.6,4,4,.8,.06],float)

def carrier(t,p):
    b,c,T,w,T2,beta,alpha,ws,ch=p
    env=np.exp(-(t/(1000*T2))**beta)
    comb=oldfit._comb_quartic_powerlaw(np.asarray(t,float),T,w,alpha,ws,ch,max(1,min(64,int(np.ceil(1.2*t.max()/T))+1)))
    return env*comb

def model(t,bg,sites,pars):
    b,c,*_=bg
    car=carrier(t,bg)
    osc=np.zeros_like(t,float)
    for rec,(amp,pm,pp) in zip(sites,pars):
        fm=rec["f_minus_Hz"]/1e6
        fp=rec["f_plus_Hz"]/1e6
        osc += amp*(np.cos(2*np.pi*fm*t+pm)+np.cos(2*np.pi*fp*t+pp))
    return b-c*car+car*osc

def stats(y,e,yp,k):
    r=(y-yp)/e;chi=float(r@r);n=len(y);red=chi/max(1,n-k)
    aic=chi+2*k;aicc=aic+2*k*(k+1)/max(1,n-k-1);bic=chi+k*np.log(n)
    return chi,red,aicc,bic

def row_seed(nv):
    r=best6.loc[nv];p=np.array(json.loads(r.popt_json),float)
    bg=p[:9]; site=lookup[(ori[nv],int(r.site_id))]
    pars=[(abs(p[9]),p[11],p[13])] # old order amp,f0,phi0,f1,phi1; f0=plus, f1=minus likely phases accordingly
    # map phase minus=phi1, plus=phi0
    pars=[(abs(p[9]),p[13],p[11])]
    return bg,site,pars

def fit_combo(nv,sites,bg0,pars0,maxn=12000):
    y=Y[nv];e=E[nv];n=len(sites)
    x0=list(bg0)
    lb=list(LBbg);ub=list(UBbg)
    for amp,pm,pp in pars0:
        x0 += [max(0,min(2,amp)),pm,pp]
        lb += [0,-np.pi,-np.pi];ub += [2,np.pi,np.pi]
    x0=np.array(x0,float);lb=np.array(lb,float);ub=np.array(ub,float)
    def unpack(x):
        bg=x[:9]; ps=[tuple(x[9+3*i:12+3*i]) for i in range(n)]
        return bg,ps
    def fun(x):
        bg,ps=unpack(x);return (y-model(t,bg,sites,ps))/e
    best=None
    starts=[x0.copy()]
    for shift in [0.0,np.pi/2,-np.pi/2]:
        xx=x0.copy()
        for i in range(n):
            xx[10+3*i]=np.clip(xx[10+3*i]+shift,-np.pi,np.pi)
            xx[11+3*i]=np.clip(xx[11+3*i]-shift,-np.pi,np.pi)
        starts.append(xx)
    for xx in starts:
        try:
            rr=least_squares(fun,np.clip(xx,lb+1e-8,ub-1e-8),bounds=(lb,ub),loss="soft_l1",f_scale=1,max_nfev=maxn)
            rr2=least_squares(fun,rr.x,bounds=(lb,ub),loss="linear",max_nfev=maxn)
            bg,ps=unpack(rr2.x);st=stats(y,e,model(t,bg,sites,ps),9+3*n)
            if best is None or st[0]<best[0]:best=(st[0],rr2.x,st,bg,ps)
        except Exception as ex: pass
    return best

def residual_screen(nv,bg,sites,pars,pool):
    y=Y[nv];e=E[nv];base=model(t,bg,sites,pars);r=y-base;car=carrier(t,bg);W=1/e
    used={int(s["site_index"]) for s in sites};out=[]
    for rec in pool:
        sid=int(rec["site_index"])
        if sid in used:continue
        fm=rec["f_minus_Hz"]/1e6;fp=rec["f_plus_Hz"]/1e6
        X=np.c_[car*np.cos(2*np.pi*fm*t),car*np.sin(2*np.pi*fm*t),car*np.cos(2*np.pi*fp*t),car*np.sin(2*np.pi*fp*t)]
        Xw=X*W[:,None];rw=r*W
        coef=np.linalg.lstsq(Xw,rw,rcond=None)[0]
        pred=X@coef;chi=np.sum(((r-pred)/e)**2)
        # convert independent coeffs into common amp + two phases seed
        am=np.hypot(coef[0],coef[1]);ap=np.hypot(coef[2],coef[3]);amp=min(2,0.5*(am+ap))
        pm=np.arctan2(-coef[1],coef[0]);pp=np.arctan2(-coef[3],coef[2])
        out.append((chi,rec,(amp,pm,pp)))
    out.sort(key=lambda x:x[0]);return out

for nv in TEST:
    print("\nNV",nv,"ori",ori[nv],"V6",best6.loc[nv].red_chi2,"site",best6.loc[nv].site_id)
    bg,s1,p1=row_seed(nv)
    f1=fit_combo(nv,[s1],bg,p1,8000)
    print(" single reproduce",int(s1["site_index"]),"red",round(f1[2][1],4),"bic",round(f1[2][3],2),"pars",np.round(f1[4],3))
    # pool from correct orientation original screen + catalog top; collapse sites
    d=att[(att.nv_index==nv)&(att.stage=="screen")&(att.status=="ok")&(att.orit==ori[nv])].sort_values("red_chi2")
    ids=list(dict.fromkeys(d.site_id.astype(int).tolist()))[:250]
    pool=[lookup[(ori[nv],sid)] for sid in ids if (ori[nv],sid) in lookup]
    scr2=residual_screen(nv,f1[3],[s1],f1[4],pool)[:15]
    pairs=[]
    for _,s2,p2 in scr2:
        ff=fit_combo(nv,[s1,s2],f1[3],[f1[4][0],p2],7000)
        if ff:pairs.append((ff[2][3],s2,ff))
    pairs.sort(key=lambda x:x[0])
    if pairs:
        b2=pairs[0];s2=b2[1];f2=b2[2]
        print(" pair",int(s1["site_index"]),int(s2["site_index"]),"red",round(f2[2][1],4),"bic",round(f2[2][3],2),"dBIC",round(f2[2][3]-f1[2][3],2),"pars",np.round(f2[4],3))
        scr3=residual_screen(nv,f2[3],[s1,s2],f2[4],pool)[:10]
        tri=[]
        for _,s3,p3 in scr3:
            ff=fit_combo(nv,[s1,s2,s3],f2[3],[f2[4][0],f2[4][1],p3],6000)
            if ff:tri.append((ff[2][3],s3,ff))
        tri.sort(key=lambda x:x[0])
        if tri:
            b3=tri[0];print(" triple",int(s1["site_index"]),int(s2["site_index"]),int(b3[1]["site_index"]),"red",round(b3[2][2][1],4),"bic",round(b3[2][2][3],2),"dBIC",round(b3[2][2][3]-f2[2][3],2),"pars",np.round(b3[2][4],3))
