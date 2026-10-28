# -*- coding: utf-8 -*-
"""Models for fixed-T randomized 13C quantum-bath sensing.

(1) Weak stationary classical Gaussian-noise screening using exact ideal
    filter functions, nonnegative spectral-bin weights and nuisance terms.
(2) One unpolarized quantum-1/2 13C conditional-evolution coherence model,
    with an effective background and pulse-error term; phenomenological
    exploratory candidate fitting only, not validated hyperfine spectroscopy.

Ideal, instantaneous pi-pulse centers are used; finite 96ns pulses must be
tested and modeled before quantitative hyperfine estimates are trusted.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares,nnls
GAMMA_13C_HZ_PER_G=1070.5


def toggling_filter(ticks, clock_ns, T_ns, freqs_hz):
    """Complex ideal sensitivity integral Y(f) = int y(t) exp(i 2 pi f t) dt."""
    f=np.asarray(freqs_hz,float).reshape(-1)
    t=np.r_[0.,np.asarray(ticks,float)*clock_ns,T_ns]*1e-9
    if any(t[1:]<=t[:-1]) or abs(t[-1]-T_ns*1e-9)>1e-14:
        raise ValueError("Pulse centers must be strictly inside the fixed sensing time")
    signs=(-1.)**np.arange(len(t)-1)
    omega=2*np.pi*f
    out=np.empty(len(f),dtype=complex)
    positive=np.abs(omega)>1e-12
    w=omega[positive,None]
    out[positive]=np.sum(
        ((np.exp(1j*w*t[1:])-np.exp(1j*w*t[:-1]))/(1j*w))*signs,
        axis=1)
    out[~positive]=np.sum(signs*np.diff(t))
    return out


def filter_bins(steps, bin_edges_hz, points_per_bin=65):
    """Integrate |Y(f)|² averaged across each bin; units ~ s²."""
    edges=np.asarray(bin_edges_hz,float)
    if np.any(np.diff(edges)<=0) or edges[0]<=0:
        raise ValueError("Require positive increasing frequency-bin edges")
    F=[]
    for s in steps:
        row=[]
        for a,b in zip(edges[:-1],edges[1:]):
            freqs=np.linspace(a,b,points_per_bin)
            Y=toggling_filter(s["switch_ticks"],4,s["T_ns"],freqs)
            row.append(float(np.trapezoid(abs(Y)**2,freqs)/(b-a)))
        F.append(row)
    return np.asarray(F,dtype=float)


def _fit_nnls(F, counts, c, sem=None, *, spectral=True):
    """chi(C) = b0 + beta*Npi + sum s_k*F_k; all coefficients >= 0."""
    F=np.asarray(F,float)
    counts=np.asarray(counts,float)
    c=np.asarray(c,float)
    valid=np.isfinite(c)&(c>.04)&(c<.99)
    if sem is not None:
        sem=np.asarray(sem,float)
        valid &= np.isfinite(sem)&(sem>0)
    nparam=2+(F.shape[1] if spectral else 0)
    if valid.sum()<max(16,3*nparam):
        return {"status":"insufficient","usable":int(valid.sum()),"required":max(16,3*nparam)}
    # F / 1e-9 for numerical stability. Constant and n_pi are nuisance.
    A=np.column_stack([np.ones(len(c)),counts/10.])
    if spectral:A=np.column_stack([A,F/1e-9])
    use=A[valid]
    y=-np.log(c[valid])
    if sem is None:
        weight=np.ones(len(y))
    else:
        # uncertainty of -log(C), capped to avoid extreme individual weights
        errors=np.clip(sem[valid]/c[valid],.05,5.)
        weight=1/errors
        weight=np.clip(weight,np.percentile(weight,10),
                       np.percentile(weight,90))
    Aw=use*weight[:,None]
    yw=y*weight
    scales=np.linalg.norm(Aw,axis=0)
    if np.any(scales<1e-12):
        return {"status":"zero_column","usable":int(valid.sum())}
    normalized=Aw/scales
    rank=np.linalg.matrix_rank(normalized,tol=1e-8)
    if rank<nparam:
        return {"status":"rank_deficient","rank":int(rank),"usable":int(valid.sum())}
    # Light smoothness regularization on adjacent frequency bins.
    if spectral and F.shape[1]>1:
        P=np.zeros((F.shape[1]-1,nparam))
        for j in range(F.shape[1]-1):
            P[j,2+j]=.04
            P[j,2+j+1]=-.04
        augmented=np.vstack([normalized,P])
        targets=np.r_[yw,np.zeros(len(P))]
    else:
        augmented=normalized
        targets=yw
    coef,_=nnls(augmented,targets,maxiter=250*nparam)
    coef=coef/scales
    full_pred=A@coef
    measured_pred=full_pred[valid]
    rmse=float(np.sqrt(np.mean((y-measured_pred)**2)))
    return {"status":"ok","usable":int(valid.sum()),
            "rank":int(rank),"coef":coef,"predicted_chi":full_pred,
            "rms_chi":rmse}


def compare_models(F, npi, coherence, sem=None):
    full=_fit_nnls(F,npi,coherence,sem,spectral=True)
    null=_fit_nnls(F,npi,coherence,sem,spectral=False)
    return full,null


def heldout_cv(F, npi, coherence, sem=None, nfold=4):
    """Prediction of randomly held-out *masks*, not of averaged shots.

    Control steps should be removed before this comparison.
    """
    n=len(coherence)
    valid=np.isfinite(coherence)&(coherence>.04)&(coherence<.99)
    if sem is not None:
        valid &= np.isfinite(sem)&(np.asarray(sem)>0)
    if valid.sum()<max(24,4*(F.shape[1]+2)):
        return {"status":"insufficient_cv","usable":int(valid.sum())}
    prediction=[]; baseline=[]; real=[]
    for fold in range(nfold):
        # Seeded randomized folds rather than sorting by pulse count.
        rng=np.random.default_rng(109)
        fold_ids=rng.permutation(n)%nfold if False else None
        # fixed assignment shared across fit and null
        order=np.random.default_rng(109).permutation(n)
        fold_id=np.empty(n,int);fold_id[order]=np.arange(n)%nfold
        testing=valid&(fold_id==fold)
        training=valid&~testing
        if testing.sum()<2:
            continue
        err=sem[training] if sem is not None else None
        full=_fit_nnls(F[training],npi[training],coherence[training],
                       err,spectral=True)
        null=_fit_nnls(F[training],npi[training],coherence[training],
                       err,spectral=False)
        if full["status"]!="ok" or null["status"]!="ok":
            continue
        A=np.column_stack([np.ones(testing.sum()),npi[testing]/10.,
                           F[testing]/1e-9])
        B=A[:,:2]
        prediction.extend((A@full["coef"]).tolist())
        baseline.extend((B@null["coef"]).tolist())
        real.extend((-np.log(coherence[testing])).tolist())
    if len(real)<8:
        return {"status":"insufficient_cv","usable":len(real)}
    y=np.asarray(real)
    ps=np.asarray(prediction)
    p0=np.asarray(baseline)
    rmse=float(np.sqrt(np.mean((y-ps)**2)))
    rmse0=float(np.sqrt(np.mean((y-p0)**2)))
    return {"status":"ok","n_test":len(real),
            "spectral_rmse":rmse,"baseline_rmse":rmse0,
            "improvement":float(1-rmse/rmse0) if rmse0>0 else np.nan}


I2=np.eye(2,dtype=complex)
SIG_X=np.array([[0,1],[1,0]],dtype=complex)
SIG_Z=np.diag([1.,-1.]).astype(complex)


def _evolve(dt_s,z_hz,x_hz=0.):
    """2x2 unitary exp(-i*2*pi*(z Iz+x Ix)*dt) exactly."""
    h=np.hypot(z_hz,x_hz)
    angle=np.pi*h*dt_s
    if h==0:return I2.copy()
    return np.cos(angle)*I2 - 1j*np.sin(angle)*(
        z_hz*SIG_Z+x_hz*SIG_X)/h


def quantum_c13_coherence(steps,*,field_g=31.1,
                          a_parallel_hz=0.,a_perp_hz=0.):
    """Unpolarized single 13C: trace(U0 U1^dagger)/2.

    H_0/h = fL I_z, H_1/h = (fL + A_parallel) I_z + A_perp I_x.
    Signed A_parallel includes the electron-manifold sign convention.
    Real X projection returned. No finite-duration microwave simulation.
    """
    f_larmor=GAMMA_13C_HZ_PER_G*field_g
    predictions=[]
    for s in steps:
        t=np.r_[0.,np.asarray(s["pi_centers_ns"],float),
                float(s["T_ns"])]*1e-9
        if np.any(np.diff(t)<0):raise ValueError("Invalid pulse centers")
        U0=I2.copy()
        U1=I2.copy()
        for k,dt in enumerate(np.diff(t)):
            if k%2==0:
                U0=_evolve(dt,f_larmor)@U0
                U1=_evolve(dt,f_larmor+a_parallel_hz,a_perp_hz)@U1
            else:
                U0=_evolve(dt,f_larmor+a_parallel_hz,a_perp_hz)@U0
                U1=_evolve(dt,f_larmor)@U1
        L=np.trace(U0@U1.conj().T)/2
        predictions.append(float(np.real(L)))
    return np.asarray(predictions)


def quantum_fit_candidate(steps,coherence,sem=None,*,field_g=31.1,
                          max_hyperfine_khz=80.,seed=10):
    """Optional illustrative single-spin model with background/pulse-loss.

    Returns candidate conditional couplings and held-out diagnostics; fitted
    A parameters can be non-identifiable, so not a confirmed nucleus.
    """
    obs=np.asarray(coherence,float)
    npi=np.asarray([s["n_pi"] for s in steps],float)
    good=np.isfinite(obs)&(np.abs(obs)<=1.2)
    if sem is not None:good &= np.isfinite(sem)&(np.asarray(sem)>0)
    if good.sum()<max(20,4):
        return {"status":"insufficient","usable":int(good.sum())}
    sel=[s for s,ok in zip(steps,good) if ok]
    yy=obs[good];nn=npi[good]
    err=np.clip(np.asarray(sem)[good] if sem is not None else
                np.ones(len(yy))*.1,.03,1.)
    order=np.random.default_rng(seed).permutation(len(yy))
    ntest=max(4,len(yy)//5)
    train=np.ones(len(yy),bool);train[order[:ntest]]=False
    def residual(theta,keep):
        apar,aperp,contrast,pulse_loss=theta
        sim=quantum_c13_coherence([sel[k] for k in np.flatnonzero(keep)],
                field_g=field_g,a_parallel_hz=apar*1000,
                a_perp_hz=aperp*1000)
        return (contrast*np.exp(-pulse_loss*nn[keep])*sim-yy[keep])/err[keep]
    xmax=max_hyperfine_khz
    candidates=[]
    for apar in (-.7*xmax,-.2*xmax,0.,.2*xmax,.7*xmax):
        for aperp in (.05*xmax,.25*xmax,.7*xmax):
            fit=least_squares(residual,
               x0=[apar,aperp,.7,.02],args=(train,),
               bounds=([-xmax,0.,0.,0.],[xmax,xmax,1.,.35]),
               max_nfev=160)
            candidates.append(fit)
    best=min(candidates,key=lambda r:np.dot(r.fun,r.fun))
    test=~train
    test_rms=float(np.sqrt(np.mean(residual(best.x,test)**2)))
    base=float(np.sqrt(np.mean(((yy[test]-np.median(yy[train]))/
                                  err[test])**2)))
    return {"status":"ok","usable":int(good.sum()),
            "a_parallel_khz":float(best.x[0]),
            "a_perp_khz":float(best.x[1]),
            "contrast":float(best.x[2]),
            "pulse_loss":float(best.x[3]),
            "train_weighted_rms":float(np.sqrt(np.mean(best.fun**2))),
            "heldout_weighted_rms":test_rms,
            "constant_null_weighted_rms":base,
            "warning":"Illustrative conditional single-spin fit; degeneracies/"
                      "many-spin contributions and finite pi widths unresolved."}
