# -*- coding: utf-8 -*-
"""
V8 multi-13C spin-echo fitter.

Strategy
--------
* Keep the current V6 slow revival/envelope model because it describes the data.
* Lock each NV to its independently assigned crystallographic orientation.
* Refit only catalog sites from that orientation.
* Extend the current single-site f-/f+ oscillator to 0, 1, 2, or 3 catalog 13C.
* Add sites by residual forward selection, then jointly refit all parameters.
* Use ordinary weighted chi2/AICc/BIC for final ranking.
* Require Delta BIC <= -6 before accepting an additional carbon.
* Profile T2; if the upper interval stays open, report a bound instead of the
  arbitrary optimizer ceiling.

The primary model intentionally preserves the current fitter frequency
convention (catalog f-/f+ evaluated directly against total_evolution_times).
This lets V8 reproduce V6 for one carbon before asking whether more carbons
are warranted.  A strict ideal-Hahn product model should be used as a later
physics cross-check, not forced on the data before this model is validated.

Run from repository root:
    python analysis/spin_echo_work/sc_c13_spin_echo_multic13_v8.py
"""

from __future__ import annotations
import ast, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.spin_echo_work import fitter_module_for_spin_echo as oldfit

# ---------------- user settings ----------------------------------------------
MAPPED_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo")
UNC_ROOT = Path(r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\spin_echo")
RESULT_TAG = "spin_echo_old_protocol_ranked_52G"
CATALOG_PATH = REPO / r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json"

ALLOWED_ORIENTATIONS = ((1, 1, -1), (-1, 1, 1))
NV_INDICES = [0, 16, 28, 54]   # validation first; set None for all 212
MAX_C13 = 3

# Preserve the successful current fitter convention.
CATALOG_TIME_FREQ_SCALE = 1.0

FIRST_DEEP_KEEP = 16
FIRST_SCREEN_KEEP = 20
FIRST_MAX = 28
FORWARD_POOL = 300
PAIR_KEEP = 12
TRIPLE_KEEP = 8
SPECTRAL_REDUNDANCY_KHZ = 2.0
BIC_ADD_THRESHOLD = -6.0

ROBUST_NFEV = 9000
POLISH_NFEV = 14000
PHASE_STARTS = ((0.0, 0.0), (np.pi/2, -np.pi/2))

RUN_T2_PROFILE = True
T2_GRID_US = np.geomspace(5.0, 600.0, 18)
PROFILE_NFEV = 6000

CPU = os.cpu_count() or 4
N_JOBS = max(1, min(14, CPU - 2))
BLAS_THREADS = 1

# V6 background order:
# baseline, contrast, Trev, width, T2_ms, beta, alpha, width_slope, chirp
BG_LB = np.array([0.0, 0.0, 15.0, 1.0, .001, .6, 0.0, 0.0, -.06])
BG_UB = np.array([1.05, .95, 40.0, 20.0, .600, 4.0, 4.0, .80, .06])
SITE_LB = np.array([0.0, -np.pi, -np.pi])  # amp, phi-, phi+
SITE_UB = np.array([2.0, +np.pi, +np.pi])

def root_dir():
    if MAPPED_ROOT.exists():
        return MAPPED_ROOT
    if UNC_ROOT.exists():
        return UNC_ROOT
    raise FileNotFoundError("nvdata root is not available")

def newest(root, pattern):
    m = list(root.rglob(pattern))
    if not m:
        raise FileNotFoundError(pattern)
    return max(m, key=lambda p: p.stat().st_mtime)

def cori(v):
    if isinstance(v, str):
        v = ast.literal_eval(v)
    a = np.asarray(v, int).ravel()
    return tuple(int(x) for x in a)

def load_all():
    root = root_dir()
    att_path = newest(root, f"*{RESULT_TAG}_all_attempts.csv.gz")
    suffix = "_all_attempts.csv.gz"
    prefix = Path(str(att_path)[:-len(suffix)])
    ck_path = Path(str(prefix) + "_fit_checkpoint.npz")
    v6_path = newest(root, "*orientation_locked_confidence_v6_all_equal_footing_sites.csv")
    ori_path = newest(root, "*orientation_locked_confidence_v6_orientation_assignments.csv")

    z = np.load(ck_path, allow_pickle=True)
    t = np.asarray(z["times_us"], float)
    Y = np.asarray(z["norm_counts"], float)
    E = np.maximum(np.abs(np.asarray(z["norm_counts_ste"], float)), 1e-4)

    odf = pd.read_csv(ori_path)
    omap = {int(r.nv_index): cori(r.orientation) for r in odf.itertuples()}

    v6 = pd.read_csv(v6_path)
    v6["orit"] = v6.orientation.map(cori)

    cols = ["nv_index","stage","status","site_id","orientation","red_chi2","aicc","popt_json"]
    att = pd.read_csv(att_path, usecols=cols)
    att["orit"] = att.orientation.map(cori)

    raw = json.load(open(CATALOG_PATH, "r", encoding="utf-8"))
    lookup = {}
    byori = {o: [] for o in ALLOWED_ORIENTATIONS}
    for rr in raw:
        o = cori(rr["orientation"])
        if o not in byori:
            continue
        r = dict(rr)
        r["orit"] = o
        r["site"] = int(r["site_index"])
        r["fm_kHz"] = float(r["f_minus_Hz"]) / 1e3
        r["fp_kHz"] = float(r["f_plus_Hz"]) / 1e3
        lookup[(o, r["site"])] = r
        byori[o].append(r)
    return root, prefix, t, Y, E, odf, omap, v6, att, lookup, byori

def carrier(t, bg):
    b,c,T,w,T2,beta,alpha,ws,ch = np.asarray(bg,float)
    env = np.exp(-(np.asarray(t,float)/(1000.0*max(T2,1e-9)))**beta)
    n = max(1, min(64, int(np.ceil(1.2*np.max(t)/max(T,1e-9)))+1))
    comb = oldfit._comb_quartic_powerlaw(np.asarray(t,float),T,w,alpha,ws,ch,n)
    return env*comb

def model(t, bg, sites=(), pars=()):
    b,c,*_ = bg
    car = carrier(t,bg)
    osc = np.zeros_like(np.asarray(t,float))
    for rec,(amp,pm,pp) in zip(sites,pars):
        fm = CATALOG_TIME_FREQ_SCALE*rec["fm_kHz"]/1000.0
        fp = CATALOG_TIME_FREQ_SCALE*rec["fp_kHz"]/1000.0
        osc += amp*(np.cos(2*np.pi*fm*t+pm)+np.cos(2*np.pi*fp*t+pp))
    return b - c*car + car*osc

def stats(y,e,yp,k):
    r=(np.asarray(y)-np.asarray(yp))/np.maximum(np.asarray(e),1e-12)
    chi=float(r@r); n=len(y); red=chi/max(1,n-k)
    aic=chi+2*k
    aicc=aic+2*k*(k+1)/max(1,n-k-1)
    bic=chi+k*np.log(max(n,2))
    return dict(chi2=chi,red_chi2=red,aicc=aicc,bic=bic)

def bounds(nsite):
    return (np.r_[BG_LB, np.tile(SITE_LB,nsite)],
            np.r_[BG_UB, np.tile(SITE_UB,nsite)])

def pack(bg,pars):
    x=list(np.asarray(bg,float))
    for p in pars: x.extend(map(float,p))
    return np.asarray(x,float)

def unpack(x,nsite):
    x=np.asarray(x,float)
    return x[:9], [tuple(x[9+3*i:12+3*i]) for i in range(nsite)]

def fit_combo(t,y,e,sites,bg0,pars0, fixed_T2_us=None, nfev_scale=1.0):
    nsite=len(sites)
    lb,ub=bounds(nsite)
    x0=np.clip(pack(bg0,pars0),lb+1e-9,ub-1e-9)

    if fixed_T2_us is None:
        keep=np.arange(len(x0))
        fixed=None
    else:
        fixed=float(fixed_T2_us)/1000.0
        if not (BG_LB[4] <= fixed <= BG_UB[4]):
            return None
        keep=np.array([i for i in range(len(x0)) if i!=4])

    qlb,qub=lb[keep],ub[keep]
    q0=x0[keep]

    def assemble(q):
        if fixed is None: return np.asarray(q,float)
        x=np.empty_like(x0); x[keep]=q; x[4]=fixed; return x

    def resid(q):
        x=assemble(q); bg,ps=unpack(x,nsite)
        return (y-model(t,bg,sites,ps))/e

    starts=[q0]
    if fixed is None:
        for dm,dp in PHASE_STARTS:
            xx=x0.copy()
            for i in range(nsite):
                xx[10+3*i]=np.clip(xx[10+3*i]+dm,-np.pi,np.pi)
                xx[11+3*i]=np.clip(xx[11+3*i]+dp,-np.pi,np.pi)
            starts.append(xx[keep])

    out=[]
    for q in starts:
        try:
            r1=least_squares(resid,np.clip(q,qlb+1e-9,qub-1e-9),bounds=(qlb,qub),
                             loss="soft_l1",f_scale=1,max_nfev=int(ROBUST_NFEV*nfev_scale))
            r2=least_squares(resid,r1.x,bounds=(qlb,qub),loss="linear",
                             max_nfev=int(POLISH_NFEV*nfev_scale))
            x=assemble(r2.x); bg,ps=unpack(x,nsite); yp=model(t,bg,sites,ps)
            k=9+3*nsite-(1 if fixed is not None else 0)
            st=stats(y,e,yp,k)
            out.append(dict(x=x,bg=bg,pars=ps,pred=yp,**st))
        except Exception:
            pass
    return min(out,key=lambda q:q["chi2"]) if out else None

def old_seed(popt,rec):
    p=np.asarray(json.loads(popt) if isinstance(popt,str) else popt,float)
    bg=p[:9].copy(); amp=float(p[9]); f0=1000*p[10]; ph0=p[11]; f1=1000*p[12]; ph1=p[13]
    direct=abs(f0-rec["fm_kHz"])+abs(f1-rec["fp_kHz"])
    swap=abs(f0-rec["fp_kHz"])+abs(f1-rec["fm_kHz"])
    pm,pp=(ph0,ph1) if direct<=swap else (ph1,ph0)
    if amp<0:
        amp=-amp; pm+=np.pi; pp+=np.pi
    pm=np.arctan2(np.sin(pm),np.cos(pm)); pp=np.arctan2(np.sin(pp),np.cos(pp))
    return bg,(amp,pm,pp)

def generic_bg(y):
    b=float(np.clip(np.nanpercentile(y,90),.2,1.0))
    c=float(np.clip(b-np.nanpercentile(y,5),.03,.7))
    return np.array([b,c,35.66,6,.05,1.5,.5,.1,0],float)

def best_seed(nv,o,sid,v6,att,lookup,y):
    rec=lookup[(o,sid)]
    q=v6[(v6.nv_index==nv)&(v6.orit==o)&(v6.site_id.astype(int)==sid)]
    if q.empty:
        q=att[(att.nv_index==nv)&(att.orit==o)&(att.site_id.astype(int)==sid)&(att.status=="ok")]
    if not q.empty:
        q=q.sort_values(["red_chi2","aicc"]).iloc[0]
        try: return old_seed(q.popt_json,rec)
        except Exception: pass
    return generic_bg(y),(0.15,0,0)

def uniq(seq):
    out=[]; seen=set()
    for x in seq:
        x=int(x)
        if x not in seen: seen.add(x); out.append(x)
    return out

def first_pool(nv,o,v6,att,lookup):
    d=v6[(v6.nv_index==nv)&(v6.orit==o)].sort_values(["aicc","red_chi2"])
    s=att[(att.nv_index==nv)&(att.stage=="screen")&(att.status=="ok")&(att.orit==o)].sort_values(["red_chi2","aicc"])
    ids=uniq(d.site_id.tolist())[:FIRST_DEEP_KEEP]+uniq(s.site_id.tolist())[:FIRST_SCREEN_KEEP]
    ids=uniq(ids)[:FIRST_MAX]
    return [lookup[(o,i)] for i in ids if (o,i) in lookup]

def forward_pool(nv,o,att,lookup):
    s=att[(att.nv_index==nv)&(att.stage=="screen")&(att.status=="ok")&(att.orit==o)].sort_values(["red_chi2","aicc"])
    ids=uniq(s.site_id.tolist())[:FORWARD_POOL]
    return [lookup[(o,i)] for i in ids if (o,i) in lookup]

def specdist(a,b):
    return float(np.hypot(a["fm_kHz"]-b["fm_kHz"],a["fp_kHz"]-b["fp_kHz"]))

def residual_screen(t,y,e,fit,selected,pool,keep):
    base=model(t,fit["bg"],selected,fit["pars"]); r=y-base; car=carrier(t,fit["bg"]); w=1/e
    used={s["site"] for s in selected}; out=[]
    for rec in pool:
        if rec["site"] in used: continue
        if any(specdist(rec,s)<SPECTRAL_REDUNDANCY_KHZ for s in selected): continue
        fm=CATALOG_TIME_FREQ_SCALE*rec["fm_kHz"]/1000; fp=CATALOG_TIME_FREQ_SCALE*rec["fp_kHz"]/1000
        X=np.c_[car*np.cos(2*np.pi*fm*t),car*np.sin(2*np.pi*fm*t),
                car*np.cos(2*np.pi*fp*t),car*np.sin(2*np.pi*fp*t)]
        try: co=np.linalg.lstsq(X*w[:,None],r*w,rcond=None)[0]
        except Exception: continue
        chi=float(np.sum(((r-X@co)/e)**2))
        amp=float(np.clip(.5*(np.hypot(co[0],co[1])+np.hypot(co[2],co[3])),0,2))
        pm=float(np.arctan2(-co[1],co[0])); pp=float(np.arctan2(-co[3],co[2]))
        out.append((chi,rec,(amp,pm,pp)))
    out.sort(key=lambda z:z[0])
    return out[:keep]

def profile_t2(nv,t,y,e,sites,fit):
    if not RUN_T2_PROFILE: return [],("not_run","",np.nan,np.nan)
    rows=[]
    for T2 in np.unique(np.r_[T2_GRID_US,1000*fit["bg"][4]]):
        rr=fit_combo(t,y,e,sites,fit["bg"],fit["pars"],fixed_T2_us=T2,nfev_scale=.5)
        if rr is not None: rows.append(dict(nv_index=nv,T2_us=T2,chi2=rr["chi2"]))
    if not rows: return [],("failed","",np.nan,np.nan)
    mn=min(r["chi2"] for r in rows)
    for r in rows: r["delta_chi2"]=r["chi2"]-mn
    g=[r["T2_us"] for r in rows if r["delta_chi2"]<=3.84]
    lo,hi=min(g),max(g)
    if hi>=.98*T2_GRID_US[-1]: status,report="lower_bound_only",f"> {lo:.1f} us (95% profile)"
    elif lo<=1.02*T2_GRID_US[0]: status,report="upper_bound_only",f"< {hi:.1f} us (95% profile)"
    else: status,report="bounded",f"[{lo:.1f}, {hi:.1f}] us (95% profile)"
    return rows,(status,report,lo,hi)

def fit_nv(nv,o,t,Y,E,v6,att,lookup):
    y,e=Y[nv],E[nv]
    fpool=first_pool(nv,o,v6,att,lookup)
    if not fpool: raise RuntimeError(f"NV {nv}: no candidates")

    single=[]
    for rec in fpool:
        bg0,p0=best_seed(nv,o,rec["site"],v6,att,lookup,y)
        ff=fit_combo(t,y,e,[rec],bg0,[p0])
        if ff is not None: single.append((ff["bic"],rec,ff))
    single.sort(key=lambda z:z[0])
    if not single: raise RuntimeError(f"NV {nv}: single fits failed")
    s1,f1=single[0][1],single[0][2]

    # background-only, seeded by winning single background
    f0=fit_combo(t,y,e,[],f1["bg"],[])

    models={0:([],f0),1:([s1],f1)}
    pool=forward_pool(nv,o,att,lookup)

    if MAX_C13>=2:
        cand=[]
        for _,s2,p2 in residual_screen(t,y,e,f1,[s1],pool,PAIR_KEEP):
            ff=fit_combo(t,y,e,[s1,s2],f1["bg"],[f1["pars"][0],p2])
            if ff is not None: cand.append((ff["bic"],[s1,s2],ff))
        cand.sort(key=lambda z:z[0])
        if cand: models[2]=(cand[0][1],cand[0][2])

    if MAX_C13>=3 and 2 in models:
        ss,ff2=models[2]; cand=[]
        for _,s3,p3 in residual_screen(t,y,e,ff2,ss,pool,TRIPLE_KEEP):
            pars=list(ff2["pars"])+[p3]
            ff=fit_combo(t,y,e,ss+[s3],ff2["bg"],pars)
            if ff is not None: cand.append((ff["bic"],ss+[s3],ff))
        cand.sort(key=lambda z:z[0])
        if cand: models[3]=(cand[0][1],cand[0][2])

    pref=0
    for order in range(1,MAX_C13+1):
        if order not in models: break
        if models[order][1]["bic"]-models[pref][1]["bic"] <= BIC_ADD_THRESHOLD: pref=order
        else: break

    prows,pinfo=profile_t2(nv,t,y,e,models[pref][0],models[pref][1])
    status,report,tlo,thi=pinfo

    order_rows=[]
    for order,(sites,ff) in models.items():
        row=dict(nv_index=nv,orientation=str(o),order=order,chi2=ff["chi2"],red_chi2=ff["red_chi2"],
                 aicc=ff["aicc"],bic=ff["bic"],preferred=(order==pref),
                 T2_us=1000*ff["bg"][4],T2_exp=ff["bg"][5],revival_time_us=ff["bg"][2],
                 width0_us=ff["bg"][3],site_ids=json.dumps([s["site"] for s in sites]),
                 site_params=json.dumps([[float(v) for v in p] for p in ff["pars"]]))
        order_rows.append(row)

    single_rows=[]
    a=np.array([z[2]["aicc"] for z in single]); w=np.exp(-.5*(a-a.min())); w/=w.sum()
    for rank,(z,ww) in enumerate(zip(single[:10],w[:10]),1):
        rec,ff=z[1],z[2]; p=ff["pars"][0]
        single_rows.append(dict(nv_index=nv,orientation=str(o),rank=rank,site_id=rec["site"],
                                kappa=rec["kappa"],distance_A=rec["distance_A"],
                                fminus_kHz=rec["fm_kHz"],fplus_kHz=rec["fp_kHz"],
                                red_chi2=ff["red_chi2"],aicc=ff["aicc"],bic=ff["bic"],
                                akaike_weight=ww,amplitude=p[0],phase_minus=p[1],phase_plus=p[2],
                                x_A=rec.get("x_A",np.nan),y_A=rec.get("y_A",np.nan),z_A=rec.get("z_A",np.nan)))

    sites,ff=models[pref]
    summary=dict(nv_index=nv,orientation=str(o),preferred_order=pref,
                 preferred_sites=json.dumps([s["site"] for s in sites]),
                 preferred_red_chi2=ff["red_chi2"],preferred_bic=ff["bic"],
                 preferred_T2_us=1000*ff["bg"][4],preferred_T2_exp=ff["bg"][5],
                 T2_profile_status=status,T2_profile_report=report,T2_95_low_us=tlo,T2_95_high_us=thi)
    print(f"[NV {nv:3d}] ori={o} order={pref} sites={[s['site'] for s in sites]} redchi={ff['red_chi2']:.3f} BIC={ff['bic']:.1f} T2={1000*ff['bg'][4]:.1f} {status}")
    return dict(nv=nv,summary=summary,orders=order_rows,singles=single_rows,profiles=prows,models=models,pref=pref)

def plot_page(pdf,res,t,Y,E):
    nv=res["nv"]; y,e=Y[nv],E[nv]; sites,ff=res["models"][res["pref"]]
    fig,axs=plt.subplots(2,2,figsize=(14,9))
    td=np.linspace(t.min(),t.max(),2500)
    axs[0,0].errorbar(t,y,yerr=e,fmt="o",ms=3,lw=.5,capsize=1)
    axs[0,0].plot(td,model(td,ff["bg"],sites,ff["pars"]),lw=1.7)
    axs[0,0].set_title(f"NV {nv}: preferred {res['pref']} 13C"); axs[0,0].set_xlabel("Total evolution time (us)"); axs[0,0].grid(alpha=.2)
    c=ff["bg"][2]; m=(t>c-12.5)&(t<c+12.5); tz=np.linspace(c-12.5,c+12.5,1600)
    axs[0,1].errorbar(t[m],y[m],yerr=e[m],fmt="o",ms=3,lw=.5,capsize=1); axs[0,1].plot(tz,model(tz,ff["bg"],sites,ff["pars"]),lw=1.7)
    axs[0,1].set_title("First revival"); axs[0,1].grid(alpha=.2)
    od=pd.DataFrame(res["orders"]).sort_values("order"); db=od.bic-od.bic.min()
    axs[1,0].bar(od.order.astype(str),db); axs[1,0].axhline(6,ls="--",lw=.8); axs[1,0].set_ylabel("Delta BIC"); axs[1,0].set_xlabel("Number of 13C"); axs[1,0].grid(alpha=.2,axis="y")
    axs[1,1].axis("off")
    lines=[f"orientation: {res['summary']['orientation']}",f"preferred sites: {res['summary']['preferred_sites']}",
           f"red chi2: {res['summary']['preferred_red_chi2']:.3f}",f"BIC: {res['summary']['preferred_bic']:.2f}",
           f"T2 point: {res['summary']['preferred_T2_us']:.1f} us",f"T2 profile: {res['summary']['T2_profile_report']}",""]
    for i,(s,p) in enumerate(zip(sites,ff["pars"]),1):
        lines += [f"13C #{i}: site {s['site']}  r={s['distance_A']:.2f} A  kappa={s['kappa']:.3f}",
                  f"  f-/f+={s['fm_kHz']:.2f}/{s['fp_kHz']:.2f} kHz",
                  f"  amp={p[0]:.3f}  phi-={p[1]:.2f}  phi+={p[2]:.2f}"]
    axs[1,1].text(.02,.98,"\n".join(lines),va="top",family="monospace",fontsize=9)
    fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

def main():
    root,prefix,t,Y,E,odf,omap,v6,att,lookup,byori=load_all()
    nvs=list(range(Y.shape[0])) if NV_INDICES is None else [int(v) for v in NV_INDICES]
    jobs=[(nv,omap[nv]) for nv in nvs if omap.get(nv) in ALLOWED_ORIENTATIONS]
    print("="*90); print("V8 MULTI-13C FIT"); print("data:",prefix); print("NVs:",len(jobs),"workers:",N_JOBS,"freq scale:",CATALOG_TIME_FREQ_SCALE); print("="*90)
    with threadpool_limits(limits=BLAS_THREADS):
        results=Parallel(n_jobs=N_JOBS,backend="loky",batch_size=1,verbose=5)(
            delayed(fit_nv)(nv,o,t,Y,E,v6,att,lookup) for nv,o in jobs)

    orders=pd.DataFrame([x for r in results for x in r["orders"]])
    singles=pd.DataFrame([x for r in results for x in r["singles"]])
    profiles=pd.DataFrame([x for r in results for x in r["profiles"]])
    summary=pd.DataFrame([r["summary"] for r in results]).sort_values("nv_index")

    base=prefix.parent/(prefix.name+"_v8_multic13")
    orders.to_csv(str(base)+"_model_orders.csv",index=False)
    singles.to_csv(str(base)+"_top_single_sites.csv",index=False)
    profiles.to_csv(str(base)+"_t2_profiles.csv",index=False)
    summary.to_csv(str(base)+"_nv_summary.csv",index=False)
    odf.to_csv(str(base)+"_orientation_assignments.csv",index=False)

    with PdfPages(str(base)+"_dashboard.pdf") as pdf:
        for r in sorted(results,key=lambda q:q["nv"]): plot_page(pdf,r,t,Y,E)

    print("\nV8 complete")
    print("order counts:",summary.preferred_order.value_counts().sort_index().to_dict())
    print("median redchi:",summary.preferred_red_chi2.median())
    print("summary:",str(base)+"_nv_summary.csv")
    print("dashboard:",str(base)+"_dashboard.pdf")
    return dict(summary=summary,orders=orders,singles=singles,profiles=profiles)

if __name__=="__main__":
    main()
