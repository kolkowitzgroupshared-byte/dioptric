# -*- coding: utf-8 -*-
"""Fixed-tau, random-multisine sign-switch RQS compiler and moment estimator.

Protocol derived from supplied RQS_toolkit (October 2026). Uses its random-
multisine sign-control design and first-order covariance estimator including
the per-tone Bessel radial correction; NOT an installation of the source
package, and NOT an estimator for unpolarized stochastic 13C spin baths.
Units: f in Hz, times s, coupling_hz=gamma*B0 in Hz.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import j0, j1


@dataclass
class Bank:
    frequencies_hz: np.ndarray
    duration_s: float
    coupling_hz: float
    pulses_ticks: list
    initial_sign: np.ndarray
    responses: np.ndarray
    sigma: np.ndarray
    condition: float
    clock_s: float
    seed: int
    max_pulses: int
    def serialize(self):
        return {"frequencies_hz": self.frequencies_hz.tolist(),
                "duration_s": float(self.duration_s),
                "coupling_hz": float(self.coupling_hz),
                "initial_sign": self.initial_sign.astype(int).tolist(),
                "pulses_ticks": [list(map(int,x)) for x in self.pulses_ticks],
                "condition": float(self.condition),
                "clock_s": float(self.clock_s),
                "seed": int(self.seed),
                "max_pulses": int(self.max_pulses)}


def exact_response(freq_hz, ticks, clock_s, total_ticks, coupling_hz, sign=1):
    """Source's response-vector ordering: all cosine then all sine integrals."""
    f=np.asarray(freq_hz,dtype=float)
    t=np.r_[0., np.asarray(ticks,dtype=float)*clock_s,
             float(total_ticks)*clock_s]
    signs=sign*(-1.)**np.arange(len(t)-1)
    w=2*np.pi*f
    cos=np.sum(signs*(np.sin(w[:,None]*t[1:])-
                      np.sin(w[:,None]*t[:-1]))/w[:,None],axis=1)
    sin=np.sum(signs*(np.cos(w[:,None]*t[:-1])-
                      np.cos(w[:,None]*t[1:]))/w[:,None],axis=1)
    return 2*np.pi*coupling_hz*np.r_[cos,sin]


def _repair_signs(s, min_cells):
    # Same nonadjacent-short-run merge convention as supplied toolkit.
    for _ in range(256):
        edges=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1,len(s)]
        short=np.flatnonzero(np.diff(edges)<min_cells)
        if not len(short) or len(edges)==2:
            return s
        picks=short[np.r_[True,np.diff(short)>1]]
        flips=np.ones(len(edges)-1,dtype=np.int8)
        flips[picks]=-1
        s*=np.repeat(flips,np.diff(edges))
    raise RuntimeError("Dwell-time repair did not converge")


def compile_bank(freq_hz, *, duration_us=8., coupling_hz=2800.,
                 seed=37, pool_size=8192, clock_ns=4.,
                 pi_ns=96., buffer_ns=16., max_pulses=48,
                 condition_limit=1000., batch_size=32):
    """Compile half the pool and implicit sign-negative twins.

    Unlike arbitrary random waiting times, switching derives from
    sign(sum_j cos(2*pi*f_j*t + random_phase_j)).
    """
    freq=np.asarray(freq_hz,dtype=float)
    if freq.ndim!=1 or not len(freq) or np.any(freq<=0) or np.unique(freq).size!=len(freq):
        raise ValueError("Frequencies must be distinct positive values")
    tick=clock_ns*1e-9
    total_ticks=round(duration_us*1e-6/tick)
    if total_ticks<40 or abs(total_ticks*tick-duration_us*1e-6)>1e-12:
        raise ValueError("Sensing time must be an integer multiple of 4ns")
    if len(freq)>1 and np.diff(np.sort(freq)).min()*(total_ticks*tick)<1:
        raise ValueError("Tone spacing below 1/T; reduce frequency bins or lengthen T")
    if freq.max()*tick >= .5:
        raise ValueError("Tone frequency exceeds clock Nyquist")
    if pool_size%2 or pool_size<4*len(freq)+2:
        raise ValueError("Pool size must be even and exceed 4*N+2")
    minimum=int(np.ceil((pi_ns+buffer_ns)/clock_ns))
    edge=minimum  # conservative, also keeps pi/2 projection away from pi
    if total_ticks < 2*edge:
        raise ValueError("Interrogation too short for finite-pulse margins")
    rng=np.random.default_rng(seed)
    ts=(np.arange(total_ticks)+.5)*tick
    cos_table=np.cos(2*np.pi*freq[:,None]*ts[None,:])
    sin_table=np.sin(2*np.pi*freq[:,None]*ts[None,:])
    pulses=[]; starts=[]; vectors=[]
    attempts=0
    while len(pulses)<pool_size//2 and attempts<pool_size*20:
        count=min(batch_size,pool_size//2-len(pulses))
        angles=rng.uniform(0,2*np.pi,(count,len(freq)))
        patterns=np.where(
            np.cos(angles)@cos_table - np.sin(angles)@sin_table >= 0,
            1,-1
        ).astype(np.int8)
        for x in patterns:
            attempts+=1
            x=_repair_signs(x,minimum)
            times=np.flatnonzero(x[1:]!=x[:-1])+1
            if len(times)>max_pulses:
                continue
            if len(times) and (times[0]<edge or
                               total_ticks-times[-1]<edge or
                               np.any(np.diff(times)<minimum)):
                raise RuntimeError("Pulse-spacing repair produced invalid timing")
            pulses.append(tuple(map(int,times)))
            starts.append(int(x[0]))
            vectors.append(exact_response(freq,times,tick,total_ticks,
                                          coupling_hz,sign=int(x[0])))
            if len(pulses)>=pool_size//2: break
    if len(pulses)!=pool_size//2:
        raise RuntimeError("Too many masks violate max_pulses; increase max_pulses")
    vec=np.asarray(vectors,float)
    sigma=vec.T@vec/len(vec)
    eig=np.linalg.eigvalsh(sigma)
    if eig[0]<=0 or eig[-1]/eig[0]>condition_limit:
        raise ValueError("RQS covariance poorly conditioned: "+
                         str(eig[-1]/max(eig[0],1e-300)))
    return Bank(freq,total_ticks*tick,coupling_hz,pulses,
                np.asarray(starts,np.int8),vec,sigma,
                float(eig[-1]/eig[0]),tick,seed,max_pulses)


def choose_masks(bank, n_settings=96, x_fraction=.2, seed=39):
    """Uniform-with-replacement signed bank draws, X/Y basis assignment."""
    if not 0<x_fraction<1 or n_settings<5:
        raise ValueError("Need 0<x_fraction<1 and at least five settings")
    rng=np.random.default_rng(seed)
    size=len(bank.pulses_ticks)
    choices=rng.integers(2*size,size=n_settings)
    bases=np.full(n_settings, "Y",dtype="<U1")
    bases[rng.choice(n_settings,max(1,round(x_fraction*n_settings)),
                     replace=False)]="X"
    rows=[]
    for i,(k,basis) in enumerate(zip(choices,bases)):
        base=int(k%size)
        init=int(bank.initial_sign[base])*(1 if k<size else -1)
        count=len(bank.pulses_ticks[base])
        # Same physical pulse pattern for both signed entries.
        outcome_sign=int(init*(-1)**count) if basis=="Y" else 1
        rows.append({"step":int(i),"pool_index":int(k),
                     "base_index":base,"basis":str(basis),
                     "initial_sign":init,
                     "outcome_sign":outcome_sign,
                     "switch_ticks":list(bank.pulses_ticks[base]),
                     "n_pi":int(count)})
    return rows


def correct_radii(bank, a, *, bound=1., grid_points=160):
    """Copied moment map from supplied RQS toolkit's Bessel correction."""
    n=len(bank.frequencies_hz)
    r=np.hypot(bank.responses[:,:n],bank.responses[:,n:])
    var=np.mean(r*r,axis=0)/2
    grid=np.linspace(0,max(2.4,1.2*bound),grid_points)
    H=np.empty((grid_points,n))
    for st in range(0,n,8):
        rr=r[:,st:st+8]
        arg=grid[:,None,None]*rr[None]
        top=np.einsum("pj,xpj->xj",rr,j1(arg))/len(rr)
        bottom=var[st:st+8]*np.mean(j0(arg),axis=1)
        H[:,st:st+8]=top/np.maximum(bottom,1e-12)
    H[0]=0
    H=np.maximum.accumulate(H,axis=0)
    mag=np.hypot(a[:n],a[n:])
    fixed=np.asarray([np.interp(mag[j],H[:,j],grid) for j in range(n)])
    scale=np.divide(fixed,mag,out=np.ones_like(mag),where=mag>1e-15)
    return a*np.r_[scale,scale]


def estimate(bank, rows, measured, *, contrast_floor=.02,
             bound=1., min_x=3,min_y=None):
    """Signed X/Y moments, Sigma whitening, radial bias correction.

    measured[i] is the physical SCC-normalized +/-1 mean, NOT a
    probability. Repeats of each row should be averaged separately before
    this function. NO free-running phase timestamp support here.
    """
    m=np.asarray(measured,float)
    if m.shape!=(len(rows),):
        raise ValueError("One physical X/Y mean per row required")
    index=np.asarray([r["pool_index"] for r in rows],int)
    y=np.asarray([r["basis"]=="Y" for r in rows],bool)
    valid=np.isfinite(m)&(abs(m)<=1.0)
    n=len(bank.frequencies_hz)
    if min_y is None:min_y=max(6*n,16)
    if np.sum(valid&~y)<min_x or np.sum(valid&y)<min_y:
        return {"status":"insufficient_good_xy","x":int(np.sum(valid&~y)),
                "y":int(np.sum(valid&y))}
    vectors=bank.responses[index%len(bank.pulses_ticks)]
    sign_pool=np.where(index<len(bank.pulses_ticks),1,-1)[:,None]
    vectors=vectors*sign_pool
    y_sign=np.asarray([r["outcome_sign"] for r in rows],float)
    # Toolkit uses X outcome independent of signed modulation and signed Y.
    mu=float(np.mean(m[valid&~y]))
    v=np.mean((m[valid&y]*y_sign[valid&y])[:,None]*vectors[valid&y],axis=0)
    floor=max(1e-12,float(contrast_floor))
    amplitude=np.linalg.solve(bank.sigma,v)/max(mu,floor)
    amplitude=correct_radii(bank,amplitude,bound=bound)
    return {"status":"ok","amplitudes":amplitude,
            "mu":mu,"floor_active":bool(mu<=floor),
            "n_x":int(np.sum(valid&~y)),"n_y":int(np.sum(valid&y)),
            "bank_condition":bank.condition}
