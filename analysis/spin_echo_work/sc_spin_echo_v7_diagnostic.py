"""V7 diagnostic: isolate missing bath physics vs multiplicative C13 physics.

This script DOES NOT perform a new lattice-site search.  It takes the completed
V6 best-BIC site assignment for a representative subset and compares five models:

  V6       : saved V6 additive-C13 + local quartic revival background
  A-local  : additive C13 + per-NV flexible revival shape/bounds
  A-shared : additive C13 + field-shared/flexible revival bath
  B        : exact multiplicative Hahn C13 + original local revival background
  AB       : multiplicative C13 + field-shared/flexible revival bath

The goal is model diagnosis before any N=4 search.
"""
from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import sc_spin_echo_physical_family_search_v6 as v6

MODEL_NAMES = ["V6", "A_local_flex", "A_shared_bath", "B_multiplicative", "AB_both"]
SHARED_LB = np.array([25.0, -0.12, 1.5], float)   # Trev us, chirp, revival power
SHARED_UB = np.array([45.0,  0.12, 8.0], float)
ROBUST_NFEV = 1200
FINAL_NFEV = 3500


def latest_v6_candidate(base, field):
    _, _, _, prefix = base.discover_paths()
    root = prefix.parent
    hits = list(root.glob(
        f"*{field}*v6_physamp_snr0p5_tol12p0kHz_smax3p0_topS15_perF3_"
        "oriassigned_maxC3_candidate_fits.csv"))
    hits = [p for p in hits if "_subset_" not in p.name]
    if not hits:
        raise FileNotFoundError(f"No completed full V6 candidate table under {root}")
    return max(hits, key=lambda p: p.stat().st_mtime)


def sites_from_row(r):
    out=[]
    ori=ast.literal_eval(str(r.orientation)) if int(r.model_order)>0 else ()
    for j in range(1, int(r.model_order)+1):
        out.append(dict(
            site_id=int(r[f"c13_{j}_site_id"]),
            f0_kHz=float(r[f"c13_{j}_f0_kHz"]),
            f1_kHz=float(r[f"c13_{j}_f1_kHz"]),
            kappa=float(r[f"c13_{j}_kappa"]),
            orientation=ori,
        ))
    return out

def choose_subset(best, field, n_total=10):
    """Deterministic representative subset spanning failure and success modes."""
    b=best.copy()
    if field=="52G":
        b=b[b.nv_index!=0].copy()
    chosen=[]; role={}

    def add(tab, label, n):
        count=0
        for _,r in tab.iterrows():
            nv=int(r.nv_index)
            if nv in chosen:
                continue
            chosen.append(nv); role[nv]=label; count+=1
            if count>=n:
                break

    add(b[b.model_order==0].sort_values("red_chi2",ascending=False),
        "poor_N0",2)
    add(b[(b.model_order==3) &
          b.visibility_scale_bound_hit.fillna(False).astype(bool)]
        .sort_values("red_chi2",ascending=False),
        "saturated_N3",3)

    z=b[(b.model_order.isin([1,2])) &
        (~b.visibility_scale_bound_hit.fillna(False).astype(bool))].copy()
    z["_d1"]=abs(z.red_chi2-1)
    add(z.sort_values("_d1"),"good_N12",2)

    z=b[(b.model_order==3) &
        (~b.visibility_scale_bound_hit.fillna(False).astype(bool))].copy()
    z["_d1"]=abs(z.red_chi2-1)
    add(z.sort_values("_d1"),"good_N3",2)

    anchor=114 if field=="49G" else 9
    if anchor in set(b.nv_index) and anchor not in chosen:
        chosen.append(anchor);role[anchor]="anchor"

    while len(chosen)<n_total:
        z=b[~b.nv_index.isin(chosen)].copy()
        z["_d1"]=abs(z.red_chi2-1)
        nv=int(z.sort_values("_d1").iloc[0].nv_index)
        chosen.append(nv);role[nv]="fill"
    return chosen[:n_total], role


def comb_power(t, trev, width0, taper, slope, chirp, power):
    tt=np.asarray(t,float)
    out=np.zeros_like(tt)
    nrev=max(1,min(64,int(np.ceil(1.2*float(tt.max())/max(trev,1e-9)))+1))
    for k in range(nrev):
        mu=k*trev*(1.0+k*chirp)
        width=width0*(1.0+k*slope)
        if width<=0:
            continue
        if mu>float(tt.max())+5.0*width:
            break
        amp=1.0/((1.0+k)**taper)
        out += amp*np.exp(-np.abs((tt-mu)/width)**power)
    return out

def shared_carrier(t, local7, shared3):
    baseline,contrast,width0,T2_ms,beta,taper,slope=np.asarray(local7,float)
    trev,chirp,power=np.asarray(shared3,float)
    tt=np.asarray(t,float)
    T2_us=max(1e-9,1000.0*T2_ms)
    env=np.exp(-np.power(np.maximum(tt,0)/T2_us,beta))
    comb=comb_power(tt,trev,width0,taper,slope,chirp,power)
    return float(baseline),float(contrast),env*comb


def additive_shared_model(t, local, shared, sites):
    n=len(sites)
    bg=local[:7]
    baseline,contrast,carrier=shared_carrier(t,bg,shared)
    tt=np.asarray(t,float)
    osc=np.zeros_like(tt)
    if n:
        scale=float(local[7]); j=8
        for s in sites:
            phi0,phi1=local[j:j+2];j+=2
            amp=scale*contrast*float(s["kappa"])/4.0
            f0=float(s["f0_kHz"])/1000.0
            f1=float(s["f1_kHz"])/1000.0
            osc += amp*(np.cos(2*np.pi*f0*tt+phi0)+
                        np.cos(2*np.pi*f1*tt+phi1))
    return baseline-contrast*carrier+carrier*osc


def additive_local_flex_model(t, theta, sites):
    """V6 additive C13 with a local, flexible revival shape exponent."""
    th=np.asarray(theta,float)
    # bg10 = old 9 background params plus revival power at index 9
    baseline,contrast,trev,width0,T2_ms,beta,taper,slope,chirp,power=th[:10]
    tt=np.asarray(t,float)
    T2_us=max(1e-9,1000.0*T2_ms)
    env=np.exp(-np.power(np.maximum(tt,0)/T2_us,beta))
    carrier=env*comb_power(tt,trev,width0,taper,slope,chirp,power)
    osc=np.zeros_like(tt)
    if sites:
        scale=float(th[10]); j=11
        for s in sites:
            phi0,phi1=th[j:j+2];j+=2
            amp=scale*contrast*float(s["kappa"])/4.0
            f0=float(s["f0_kHz"])/1000.0
            f1=float(s["f1_kHz"])/1000.0
            osc += amp*(np.cos(2*np.pi*f0*tt+phi0)+
                        np.cos(2*np.pi*f1*tt+phi1))
    return baseline-contrast*carrier+carrier*osc


def local_flex_seed(row):
    th=np.asarray(json.loads(row.theta_json),float)
    return np.r_[th[:9],4.0,th[9:]]


def local_flex_bounds(base,t,row):
    n=int(row.model_order)
    t2max=base.t2_upper_us(t)/1000.0
    lb=[0.0,0.0,25.0,0.5,0.001,0.4,0.0,-0.4,-0.12,1.5]
    ub=[1.05,0.95,45.0,25.0,t2max,8.0,6.0,1.2,0.12,8.0]
    if n:
        lb += [0.0] + [-np.pi,-np.pi]*n
        ub += [3.0] + [ np.pi, np.pi]*n
    return np.asarray(lb,float),np.asarray(ub,float)


def fit_A_local(base,t,y,e,row,sites):
    seed=local_flex_seed(row)
    lb,ub=local_flex_bounds(base,t,row)
    seed=np.clip(seed,lb+1e-8,ub-1e-8)
    ee=base.safe_err(e)
    fun=lambda th:(np.asarray(y,float)-additive_local_flex_model(t,th,sites))/ee
    seeds=[]
    for power in [2.0,4.0,6.0]:
        s=seed.copy();s[9]=power;seeds.append(s)
    robust=[]
    for s in seeds:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="soft_l1",f_scale=1,
                            max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((np.sum(fun(q.x)**2),q.x))
        except Exception:
            pass
    if not robust:return None
    robust.sort(key=lambda z:z[0])
    finals=[]
    for _,s in robust[:2]:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="linear",
                            max_nfev=FINAL_NFEV,ftol=1e-9,xtol=1e-9,
                            gtol=1e-9,x_scale="jac")
            pred=additive_local_flex_model(t,q.x,sites)
            st=stats(y,ee,pred,len(q.x))
            finals.append(dict(theta=q.x,pred=pred,**st))
        except Exception:
            pass
    return min(finals,key=lambda z:z["chi2"]) if finals else None


def exact_c13_coherence(t, sites, visibility):
    tt=np.asarray(t,float)
    tau=tt/2.0
    coh=np.ones_like(tt)
    v=float(visibility)
    for s in sites:
        fa=float(s["f0_kHz"])/1000.0
        fb=float(s["f1_kHz"])/1000.0
        fI=abs(fa-fb)/2.0
        fm=(fa+fb)/2.0
        mod=2.0*v*float(s["kappa"])
        mod *= np.sin(np.pi*fI*tau)**2 * np.sin(np.pi*fm*tau)**2
        coh *= (1.0-mod)
    return coh


def multiplicative_local_model(base,t,theta,sites):
    bg=np.asarray(theta[:9],float)
    baseline,contrast,carrier=base.carrier_from_bg(t,bg)
    if not sites:
        return baseline-contrast*carrier
    visibility=float(theta[9])
    coh=exact_c13_coherence(t,sites,visibility)
    return baseline-contrast*carrier*coh


def multiplicative_shared_model(t,local,shared,sites):
    baseline,contrast,carrier=shared_carrier(t,local[:7],shared)
    if not sites:
        return baseline-contrast*carrier
    visibility=float(local[7])
    coh=exact_c13_coherence(t,sites,visibility)
    return baseline-contrast*carrier*coh

def stats(y,e,pred,k):
    r=(np.asarray(y,float)-np.asarray(pred,float))/np.asarray(e,float)
    chi2=float(np.sum(r*r)); n=len(r); k=int(k)
    dof=max(1,n-k)
    aic=chi2+2*k
    aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf
    bic=chi2+k*np.log(max(n,2))
    return dict(chi2=chi2,red_chi2=chi2/dof,aicc=float(aicc),bic=float(bic),
                npar=k,residual=r)


def total_stats(chi2,n,k):
    dof=max(1,n-k)
    aic=chi2+2*k
    aicc=aic+2*k*(k+1)/(n-k-1) if n>k+1 else np.inf
    bic=chi2+k*np.log(max(n,2))
    return dict(chi2=float(chi2),red_chi2=float(chi2/dof),
                aicc=float(aicc),bic=float(bic),npar=int(k),ndata=int(n))


def local_shared_seed(row):
    th=np.asarray(json.loads(row.theta_json),float)
    # baseline, contrast, width0, T2_ms, beta, taper, width_slope
    loc=np.array([th[0],th[1],th[3],th[4],th[5],th[6],th[7]],float)
    n=int(row.model_order)
    if n:
        loc=np.r_[loc, th[9], th[10:10+2*n]]
    return loc


def shared_local_bounds(base,t,row,kind):
    n=int(row.model_order)
    t2max=base.t2_upper_us(t)/1000.0
    lb=[0.0,0.0,0.5,0.001,0.4,0.0,-0.4]
    ub=[1.05,0.95,25.0,t2max,8.0,6.0,1.2]
    if n:
        if kind=="additive":
            lb += [0.0] + [-np.pi,-np.pi]*n
            ub += [3.0] + [ np.pi, np.pi]*n
        else:
            lb += [0.0]
            ub += [1.0]
    return np.asarray(lb,float),np.asarray(ub,float)


def oldbg_mult_seed(row):
    th=np.asarray(json.loads(row.theta_json),float)
    if int(row.model_order):
        return np.r_[th[:9], min(1.0,max(0.0,float(row.visibility_scale)))]
    return th[:9].copy()


def oldbg_mult_bounds(base,t,row):
    lb=np.asarray(base.BG_LB,float).copy()
    ub=np.asarray(base.BG_UB,float).copy()
    ub[4]=min(float(ub[4]),base.t2_upper_us(t)/1000.0)
    ub[1]=min(float(ub[1]),max(0.05,float(row.baseline)-0.01))
    if int(row.model_order):
        lb=np.r_[lb,0.0];ub=np.r_[ub,1.0]
    return lb,ub

def fit_B_independent(base,t,y,e,row,sites):
    seed=oldbg_mult_seed(row)
    lb,ub=oldbg_mult_bounds(base,t,row)
    seed=np.clip(seed,lb+1e-8,ub-1e-8)
    ee=base.safe_err(e)
    fun=lambda th:(np.asarray(y,float)-multiplicative_local_model(base,t,th,sites))/ee
    seeds=[seed]
    if sites:
        for vv in [0.25,0.5,0.8,0.98]:
            s=seed.copy();s[9]=vv;seeds.append(s)
    robust=[]
    for s in seeds:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="soft_l1",f_scale=1,
                            max_nfev=ROBUST_NFEV,x_scale="jac")
            robust.append((np.sum(fun(q.x)**2),q.x))
        except Exception:
            pass
    if not robust:
        return None
    robust.sort(key=lambda z:z[0])
    finals=[]
    for _,s in robust[:2]:
        try:
            q=least_squares(fun,s,bounds=(lb,ub),loss="linear",
                            max_nfev=FINAL_NFEV,ftol=1e-10,xtol=1e-10,
                            gtol=1e-10,x_scale="jac")
            pred=multiplicative_local_model(base,t,q.x,sites)
            st=stats(y,ee,pred,len(q.x))
            finals.append(dict(theta=q.x,pred=pred,**st))
        except Exception:
            pass
    return min(finals,key=lambda z:z["chi2"]) if finals else None


def pack_joint_seed(rows,kind):
    trev=np.median([float(r.revival_time_us) for r in rows])
    chirp=np.median([float(r.revival_chirp) for r in rows])
    shared=np.array([np.clip(trev,SHARED_LB[0]+1e-5,SHARED_UB[0]-1e-5),
                     np.clip(chirp,SHARED_LB[1]+1e-5,SHARED_UB[1]-1e-5),
                     4.0],float)
    pieces=[shared]
    for r in rows:
        if kind=="additive":
            pieces.append(local_shared_seed(r))
        else:
            th=np.asarray(json.loads(r.theta_json),float)
            loc=np.array([th[0],th[1],th[3],th[4],th[5],th[6],th[7]],float)
            if int(r.model_order):
                loc=np.r_[loc,min(1.0,max(0.0,float(r.visibility_scale)))]
            pieces.append(loc)
    return np.concatenate(pieces)

def fit_joint_shared(base,t,Y,E,rows,site_lists,kind):
    seed=pack_joint_seed(rows,kind)
    lbs=[SHARED_LB];ubs=[SHARED_UB]
    slices=[];cursor=3
    for r in rows:
        lb,ub=shared_local_bounds(base,t,r,kind)
        lbs.append(lb);ubs.append(ub)
        slices.append(slice(cursor,cursor+len(lb)));cursor+=len(lb)
    lb=np.concatenate(lbs);ub=np.concatenate(ubs)
    seed=np.clip(seed,lb+1e-8,ub-1e-8)
    errs=[base.safe_err(e) for e in E]
    # Residual block i depends only on the 3 shared bath parameters and
    # that NV's local block.  Supplying this sparsity pattern makes
    # finite-difference Jacobians much faster for 10-NV joint fits.
    nres=sum(len(q) for q in Y)
    sparsity=lil_matrix((nres,len(seed)),dtype=int)
    r0=0
    for yy,sl in zip(Y,slices):
        r1=r0+len(yy)
        sparsity[r0:r1,0:3]=1
        sparsity[r0:r1,sl]=1
        r0=r1
    sparsity=sparsity.tocsr()

    def preds(theta):
        shared=theta[:3]; out=[]
        for sl,s,y in zip(slices,site_lists,Y):
            loc=theta[sl]
            if kind=="additive":
                out.append(additive_shared_model(t,loc,shared,s))
            else:
                out.append(multiplicative_shared_model(t,loc,shared,s))
        return out

    def resid(theta):
        pp=preds(theta)
        return np.concatenate([(np.asarray(y,float)-p)/ee
                               for y,p,ee in zip(Y,pp,errs)])

    # multiple starting powers; keep shared timing seeded from V6.
    seeds=[]
    for power in [2.0,3.0,4.0,5.5,7.0]:
        s=seed.copy();s[2]=power;seeds.append(s)
    robust=[]
    for s in seeds:
        try:
            q=least_squares(resid,s,bounds=(lb,ub),loss="soft_l1",f_scale=1,
                            max_nfev=ROBUST_NFEV,x_scale="jac",
                            jac_sparsity=sparsity,tr_solver="lsmr")
            robust.append((np.sum(resid(q.x)**2),q.x))
        except Exception:
            pass
    if not robust:
        return None
    robust.sort(key=lambda z:z[0])
    finals=[]
    for _,s in robust[:2]:
        try:
            q=least_squares(resid,s,bounds=(lb,ub),loss="linear",
                            max_nfev=FINAL_NFEV,ftol=3e-9,xtol=3e-9,
                            gtol=3e-9,x_scale="jac",
                            jac_sparsity=sparsity,tr_solver="lsmr")
            rr=resid(q.x);pp=preds(q.x)
            finals.append((float(np.sum(rr*rr)),q.x,pp))
        except Exception:
            pass
    if not finals:
        return None
    chi2,theta,pp=min(finals,key=lambda z:z[0])
    return dict(theta=theta,preds=pp,slices=slices,
                shared=theta[:3],chi2=chi2,npar=len(theta))

def evaluate_field(field,outdir,n_each=10):
    base=v6.load_backend(field)
    _,ck,_,_=base.discover_paths()
    t,y,e=base.load_data(ck)
    cand=latest_v6_candidate(base,field)
    cdf=pd.read_csv(cand)
    best=cdf[cdf.rank_global_bic==1].sort_values("nv_index").copy()
    nvs,roles=choose_subset(best,field,n_each)
    rows=[best[best.nv_index==nv].iloc[0] for nv in nvs]
    sites=[sites_from_row(r) for r in rows]
    Y=[np.asarray(y[nv],float) for nv in nvs]
    E=[base.safe_err(e[nv]) for nv in nvs]

    print(f"\n{field} diagnostic NVs:")
    for nv,r in zip(nvs,rows):
        print(f"  NV{nv:3d} {roles[nv]:12s} N={int(r.model_order)} "
              f"chi2r={r.red_chi2:.3f} sites={r.site_key}")

    # Saved V6 reference
    ref_pred=[];ref_chi=0.0;ref_k=0
    per=[]
    for nv,r,s,yy,ee in zip(nvs,rows,sites,Y,E):
        th=np.asarray(json.loads(r.theta_json),float)
        pp=v6.v6_model(base,t,th,s)
        st=stats(yy,ee,pp,int(r.npar))
        ref_pred.append(pp);ref_chi+=st["chi2"];ref_k+=int(r.npar)

    print(f"{field}: fitting A-local flexible bath ...")
    Alocal=[]
    for nv,r,s,yy,ee in zip(nvs,rows,sites,Y,E):
        q=fit_A_local(base,t,yy,ee,r,s)
        if q is None:
            raise RuntimeError(f"A-local fit failed for {field} NV{nv}")
        Alocal.append(q)

    print(f"{field}: fitting B multiplicative/local ...")
    Bres=[]
    for nv,r,s,yy,ee in zip(nvs,rows,sites,Y,E):
        q=fit_B_independent(base,t,yy,ee,r,s)
        if q is None:
            raise RuntimeError(f"B fit failed for {field} NV{nv}")
        Bres.append(q)

    print(f"{field}: fitting A shared/flexible bath ...")
    A=fit_joint_shared(base,t,Y,E,rows,sites,"additive")
    if A is None: raise RuntimeError(f"A joint fit failed for {field}")

    print(f"{field}: fitting AB shared bath + multiplicative C13 ...")
    AB=fit_joint_shared(base,t,Y,E,rows,sites,"multiplicative")
    if AB is None: raise RuntimeError(f"AB joint fit failed for {field}")

    ndata=sum(len(q) for q in Y)
    totals={}
    totals["V6"]=total_stats(ref_chi,ndata,ref_k)
    alchi=sum(q["chi2"] for q in Alocal);alk=sum(q["npar"] for q in Alocal)
    totals["A_local_flex"]=total_stats(alchi,ndata,alk)
    bchi=sum(q["chi2"] for q in Bres);bk=sum(q["npar"] for q in Bres)
    totals["B_multiplicative"]=total_stats(bchi,ndata,bk)
    totals["A_shared_bath"]=total_stats(A["chi2"],ndata,A["npar"])
    totals["AB_both"]=total_stats(AB["chi2"],ndata,AB["npar"])

    # Per-NV diagnostics.  For shared models, report chi2 and RMS; total BIC is
    # the valid model-comparison statistic because three parameters are shared.
    for i,(nv,r,yy,ee) in enumerate(zip(nvs,rows,Y,E)):
        predA=A["preds"][i]; predB=Bres[i]["pred"]; predAB=AB["preds"][i]
        models={"V6":ref_pred[i],"A_local_flex":Alocal[i]["pred"],
                "A_shared_bath":predA,
                "B_multiplicative":predB,"AB_both":predAB}
        for name,pred in models.items():
            rr=(yy-pred)/ee
            per.append(dict(
                field=field,nv_index=nv,role=roles[nv],
                model_order=int(r.model_order),site_key=str(r.site_key),
                model=name,chi2=float(np.sum(rr*rr)),
                residual_rms=float(np.sqrt(np.mean(rr*rr))),
                residual_mean=float(np.mean(rr)),
                residual_lag1=float(np.corrcoef(rr[:-1],rr[1:])[0,1])
                    if len(rr)>2 and np.std(rr[:-1])>0 and np.std(rr[1:])>0 else np.nan,
            ))

    sdf=pd.DataFrame([
        dict(field=field,model=k,**v,
             delta_bic=np.nan,
             shared_trev_us=(A["shared"][0] if k=="A_shared_bath" else
                             AB["shared"][0] if k=="AB_both" else np.nan),
             shared_chirp=(A["shared"][1] if k=="A_shared_bath" else
                           AB["shared"][1] if k=="AB_both" else np.nan),
             shared_revival_power=(A["shared"][2] if k=="A_shared_bath" else
                                   AB["shared"][2] if k=="AB_both" else np.nan))
        for k,v in totals.items()
    ])
    sdf["delta_bic"]=sdf.bic-sdf.bic.min()
    pdf=outdir/f"v7_diagnostic_{field}.pdf"
    csv=outdir/f"v7_diagnostic_{field}_per_nv.csv"
    scsv=outdir/f"v7_diagnostic_{field}_summary.csv"
    pd.DataFrame(per).to_csv(csv,index=False)
    sdf.to_csv(scsv,index=False)
    make_pdf(pdf,field,t,Y,E,nvs,roles,rows,sites,ref_pred,Alocal,A,Bres,AB,sdf)
    return dict(field=field,summary=sdf,per=pd.DataFrame(per),
                pdf=pdf,csv=csv,summary_csv=scsv)

def make_pdf(path,field,t,Y,E,nvs,roles,rows,sites,ref,Alocal,A,Bres,AB,sdf):
    with PdfPages(path) as pdf:
        fig,axs=plt.subplots(2,2,figsize=(11,8.5))
        fig.suptitle(f"{field} V7 diagnostic: which missing physics matters?",fontsize=16)
        order=["V6","A_local_flex","A_shared_bath","B_multiplicative","AB_both"]
        q=sdf.set_index("model").loc[order]
        axs[0,0].bar(range(5),q.delta_bic)
        axs[0,0].set_xticks(range(5));axs[0,0].set_xticklabels(
            ["V6","A local","A shared","B mult.","AB both"],rotation=15)
        axs[0,0].set_ylabel("Delta BIC (total subset)");axs[0,0].grid(alpha=.15,axis="y")
        axs[0,1].bar(range(5),q.red_chi2)
        axs[0,1].set_xticks(range(5));axs[0,1].set_xticklabels(
            ["V6","A local","A shared","B mult.","AB both"],rotation=15)
        axs[0,1].set_ylabel("Global reduced chi-square");axs[0,1].grid(alpha=.15,axis="y")

        # population common residual for each model
        for name,preds in [
            ("V6",ref),("A local",[x["pred"] for x in Alocal]),
            ("A shared",A["preds"]),
            ("B",[x["pred"] for x in Bres]),("AB",AB["preds"])]:
            R=np.asarray([(yy-pp)/ee for yy,pp,ee in zip(Y,preds,E)])
            axs[1,0].plot(t,np.nanmedian(R,axis=0),label=name,lw=1.4)
        axs[1,0].axhline(0,lw=.8);axs[1,0].set(
            xlabel="Total evolution time (us)",ylabel="Median normalized residual",
            title="Common residual across representative NVs")
        axs[1,0].legend();axs[1,0].grid(alpha=.15)

        axs[1,1].axis("off")
        lines=[
            "MODELS",
            "V6      : saved additive C13 + local quartic bath",
            "A local  : additive C13 + local flexible bath shape/bounds",
            "A shared : additive C13 + field-shared/flexible bath",
            "B        : exact multiplicative Hahn C13 + original local bath",
            "AB       : shared/flexible bath + multiplicative Hahn C13",
            "",
            "A/AB shared bath parameters:",
            f"A : Trev={A['shared'][0]:.3f} us  chirp={A['shared'][1]:+.5f}  "
            f"power={A['shared'][2]:.3f}",
            f"AB: Trev={AB['shared'][0]:.3f} us  chirp={AB['shared'][1]:+.5f}  "
            f"power={AB['shared'][2]:.3f}",
            "",
            "Exact C13 factor:",
            "L_j = 1 - 2 v*kappa_j sin^2(pi fI tau) sin^2(pi fm tau)",
            "L_total = product_j L_j,  0 <= v <= 1",
            "",
            "Site assignments are FIXED to V6 for this diagnostic.",
        ]
        axs[1,1].text(.01,.99,"\n".join(lines),va="top",family="monospace",fontsize=8.2)
        fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);plt.close(fig)

        for i,(nv,role,r,yy,ee,s) in enumerate(zip(nvs,[roles[x] for x in nvs],rows,Y,E,sites)):
            fig,axs=plt.subplots(2,2,figsize=(11,8.5))
            fig.suptitle(f"{field} NV {nv} | {role} | V6 N={int(r.model_order)} {r.site_key}",
                         fontsize=14)
            td=np.linspace(float(t.min()),float(t.max()),1400)
            # Use fitted model functions for dense curves.
            v6dense=v6.v6_model(v6.load_backend(field),td,
                                np.asarray(json.loads(r.theta_json),float),s)
            Alocaldense=additive_local_flex_model(td,Alocal[i]["theta"],s)
            locA=A["theta"][A["slices"][i]]
            Adense=additive_shared_model(td,locA,A["shared"],s)
            Bdense=multiplicative_local_model(v6.load_backend(field),td,Bres[i]["theta"],s)
            locAB=AB["theta"][AB["slices"][i]]
            ABdense=multiplicative_shared_model(td,locAB,AB["shared"],s)

            axs[0,0].errorbar(t,yy,yerr=ee,fmt="o",ms=3,lw=.4,label="data")
            for name,pred,ls in [
                ("V6",v6dense,"-"),("A local",Alocaldense,"--"),
                ("A shared",Adense,"-."),
                ("B mult.",Bdense,":"),("AB both",ABdense,(0,(3,1,1,1)))]:
                axs[0,0].plot(td,pred,ls=ls,lw=1.35,label=name)
            axs[0,0].set(xlabel="Time (us)",ylabel="Signal",title="Full trace")
            axs[0,0].legend(fontsize=7);axs[0,0].grid(alpha=.15)

            models=[
                ("V6",ref[i]),("A local",Alocal[i]["pred"]),
                ("A shared",A["preds"][i]),
                ("B",Bres[i]["pred"]),("AB",AB["preds"][i])]
            for name,pred in models:
                axs[0,1].plot(t,(yy-pred)/ee,marker="o",ms=2,lw=.8,label=name)
            axs[0,1].axhline(0,lw=.7);axs[0,1].set(
                xlabel="Time (us)",ylabel="Normalized residual",title="Residual structure")
            axs[0,1].legend(fontsize=7);axs[0,1].grid(alpha=.15)

            vals=[]
            for name,pred in models:
                rr=(yy-pred)/ee
                vals.append(np.sqrt(np.mean(rr*rr)))
            axs[1,0].bar(range(5),vals)
            axs[1,0].set_xticks(range(5));axs[1,0].set_xticklabels(
                ["V6","A local","A shared","B","AB"],rotation=15)
            axs[1,0].set_ylabel("Residual RMS");axs[1,0].grid(alpha=.15,axis="y")

            axs[1,1].axis("off")
            bvis=(Bres[i]["theta"][9] if int(r.model_order)>0 else np.nan)
            abloc=AB["theta"][AB["slices"][i]]
            abvis=(abloc[7] if int(r.model_order)>0 else np.nan)
            lines=[
                f"V6 chi2r = {float(r.red_chi2):.3f}",
                f"V6 s_NV  = {float(r.visibility_scale):.3f}" if int(r.model_order)>0 else "V6 s_NV = n/a",
                f"B visibility v = {bvis:.3f}" if np.isfinite(bvis) else "B visibility = n/a",
                f"AB visibility v = {abvis:.3f}" if np.isfinite(abvis) else "AB visibility = n/a",
                "",
                "sites:",
            ]
            for sj in s:
                lines.append(f"  S{sj['site_id']}: kappa={sj['kappa']:.3f}, "
                             f"f={sj['f0_kHz']:.1f}/{sj['f1_kHz']:.1f} kHz")
            axs[1,1].text(.02,.98,"\n".join(lines),va="top",
                          family="monospace",fontsize=9)
            fig.tight_layout(rect=[0,0,1,.94]);pdf.savefig(fig);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",choices=["49G","52G","both"],default="both")
    ap.add_argument("--n-each",type=int,default=10)
    ap.add_argument("--output-dir",type=str,default=None)
    args=ap.parse_args()
    ym=datetime.now().strftime("%Y_%m")
    outdir=(Path(args.output_dir) if args.output_dir else
            Path(r"G:\nvdata\pc_NVOffice\branch_master")/"spin_echo_v7_diagnostic"/ym)
    outdir.mkdir(parents=True,exist_ok=True)
    fields=["49G","52G"] if args.field=="both" else [args.field]
    results=[]
    for fld in fields:
        results.append(evaluate_field(fld,outdir,args.n_each))
    if len(results)>1:
        allsum=pd.concat([x["summary"] for x in results],ignore_index=True)
        allsum.to_csv(outdir/"v7_diagnostic_both_summary.csv",index=False)
        allper=pd.concat([x["per"] for x in results],ignore_index=True)
        allper.to_csv(outdir/"v7_diagnostic_both_per_nv.csv",index=False)
    print("\nCOMPLETE")
    for x in results:
        print(x["field"],x["pdf"])
        print(x["summary"].to_string(index=False))


if __name__=="__main__":
    main()
