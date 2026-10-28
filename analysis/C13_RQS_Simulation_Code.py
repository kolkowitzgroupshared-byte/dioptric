#!/usr/bin/env python3
"""Self-contained example: RQS-inspired random-multisine filters sensing a 13C bath.

Synthetic experiment -- NOT data from Purcell. Each NV has a few unpolarized,
independent spin-1/2 13C nuclei. The conditional electron-state nuclear
Hamiltonians are H0/h=f_L Iz and H1/h=(f_L+Apar) Iz + Aperp Ix.

The code generates pulse centers from sign(sum cos(2 pi f_j t + random_phi_j)),
uses ideal instantaneous pi_X flips for the quantum dynamics and filter,
then adds an illustrative envelope, pulse-count-dependent attenuation,
and Poisson SCC readout shot noise. It uses the same 4 ns sampling and
112 ns minimum free evolution/pulse spacing as the Purcell pilot.

Assumptions: independent unpolarized 13C spins, no internuclear dipolar
coupling, ideal instantaneous microwave pulses in the spin model, selected
conditional hyperfine couplings, 31.1 G field, calibrated X readout.
Simulated spectral-power estimates are only MODEL-MISMATCH DIAGNOSTICS,
not recovered hyperfine couplings; the exact coherent quantum model is
used to generate the ground truth.

Run:
  python simulate_rqs_c13_bath.py --output-dir RQS_C13_SIM
  python simulate_rqs_c13_bath.py --n-nvs 24 --n-masks 32 --seed 37
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.optimize import nnls

GAMMA_C13_HZ_G = 1070.5
CLOCK_NS = 4.0
PI_NS = 96.0
MIN_GAP_NS = 112.0
GRIDS = {
    40.: [8000., 33292.55, 58585.1],
    60.: [15000., 33292.55, 51600.],
    80.: [20000., 33292.55, 46600.],
}


def _repair_short_dwell(sign, min_cells):
    sign = np.asarray(sign, dtype=np.int8).copy()
    for _ in range(256):
        edges = np.r_[0, np.flatnonzero(sign[1:] != sign[:-1])+1, len(sign)]
        short = np.flatnonzero(np.diff(edges) < min_cells)
        if not len(short) or len(edges) == 2:
            return sign
        picks = short[np.r_[True, np.diff(short) > 1]]
        flips = np.ones(len(edges)-1, dtype=np.int8)
        flips[picks] = -1
        sign *= np.repeat(flips, np.diff(edges))
    raise RuntimeError('Could not repair short dwell times')


def compile_multisine_masks(frequencies_hz, T_us, count, seed, pool_size=512):
    """Same multisine-sign selection rule as the Purcell RQS core."""
    f = np.asarray(frequencies_hz, float)
    tick = CLOCK_NS * 1e-9
    cells = round(T_us * 1e-6 / tick)
    margin = int(np.ceil(MIN_GAP_NS / CLOCK_NS))
    times = (np.arange(cells) + 0.5)*tick
    cost = np.cos(2*np.pi*f[:,None]*times)
    sint = np.sin(2*np.pi*f[:,None]*times)
    rng = np.random.default_rng(seed)
    pool = []
    while len(pool) < pool_size:
        ph = rng.uniform(0, 2*np.pi, (min(32, pool_size-len(pool)),len(f)))
        g = np.cos(ph) @ cost - np.sin(ph) @ sint
        for row in g:
            s = _repair_short_dwell(np.where(row>=0, 1,-1), margin)
            flips = np.flatnonzero(s[1:] != s[:-1])+1
            if len(flips) > 48:
                continue
            if len(flips) and (flips[0]<margin or cells-flips[-1]<margin or
                                np.any(np.diff(flips)<margin)):
                continue
            pool.append(flips)
            if len(pool) >= pool_size:break
    # Select distinct original sequences. The full Purcell experiment uses
    # an 8192-signed bank; this smaller pool gives an illustrative sample.
    inds = np.random.default_rng(seed+117).choice(len(pool), count, replace=False)
    return [pool[i] for i in inds]


def periodic_centers(T_us,n):
    return np.rint((np.arange(n)+0.5)*T_us*1000/(n*CLOCK_NS)).astype(int)


def make_steps(T_us,n_masks,seed):
    grid = GRIDS[float(T_us)]
    centers = compile_multisine_masks(grid,T_us,n_masks,seed)
    rows=[dict(kind='random', T_us=T_us, ticks=np.asarray(v,int),
               n_pi=len(v)) for v in centers]
    for kind,n in [('XY4',4),('XY8',8),('Hahn',1)]:
        ticks = periodic_centers(T_us,n)
        rows.append(dict(kind=kind,T_us=T_us,ticks=ticks,n_pi=n))
    return rows


def unitaries(dt, z_hz, x_hz):
    """U = exp[-i*2*pi*(z Iz+x Ix)*dt], with I=sigma/2."""
    h = np.hypot(z_hz,x_hz)
    theta = np.pi*h*dt
    cosine = np.cos(theta)
    sinc = np.divide(np.sin(theta),h,out=np.zeros_like(h,dtype=float),where=h>0)
    alpha = -1j*sinc*z_hz
    beta = -1j*sinc*x_hz
    out = np.empty(z_hz.shape+(2,2),dtype=np.complex128)
    out[...,0,0] = cosine+alpha
    out[...,1,1] = cosine-alpha
    out[...,0,1] = beta
    out[...,1,0] = beta
    return out


def simulate_quantum_bath(rows,field_g, apar_hz, aperp_hz, t2bg_s,
                           pulse_error=0.006,stretch=1.7):
    """Exact conditional evolution for independent, initially maximally mixed nuclei."""
    nv, nspin = apar_hz.shape
    ident = np.broadcast_to(np.eye(2,dtype=np.complex128),(nv,nspin,2,2)).copy()
    zbase = np.full((nv,nspin), GAMMA_C13_HZ_G*field_g)
    xbase = np.zeros_like(zbase)
    zshift = zbase+apar_hz
    out = np.empty((nv,len(rows)),float)
    for j,row in enumerate(rows):
        t = np.r_[0,row['ticks']*CLOCK_NS,row['T_us']*1000]*1e-9
        evolution_0=ident.copy(); evolution_1=ident.copy()
        for segment, dt in enumerate(np.diff(t)):
            ub=unitaries(dt,zbase,xbase)
            uh=unitaries(dt,zshift,aperp_hz)
            # The two NV electron branches exchange their conditional
            # nuclear Hamiltonians at every electron pi pulse.
            if segment%2 == 0:
                evolution_0=ub@evolution_0
                evolution_1=uh@evolution_1
            else:
                evolution_0=uh@evolution_0
                evolution_1=ub@evolution_1
        per_nucleus=np.sum(evolution_0*np.conj(evolution_1),axis=(-2,-1))/2
        L=np.real(np.prod(per_nucleus,axis=1))
        envelope=np.exp(-(row['T_us']*1e-6/t2bg_s)**stretch-
                        pulse_error*row['n_pi'])
        out[:,j]=np.clip(L*envelope,-1.,1.)
    return out


def sample_hyperfine(rng,n_nv,n_spin):
    """Deliberately illustrative couplings, not inferred from the user's sample."""
    apar = rng.normal(0,4_000,(n_nv,n_spin))
    aperp = rng.lognormal(np.log(2_700),.62,(n_nv,n_spin))
    # About a quarter of NVs contain a moderately coupled 13C in this toy model.
    strong = rng.random(n_nv)<.28
    apar[strong,0] += rng.choice([-1,1],strong.sum())*rng.uniform(11_000,33_000,strong.sum())
    aperp[strong,0] += rng.uniform(9_000,24_000,strong.sum())
    t2 = rng.uniform(145e-6,280e-6,n_nv)
    return apar,aperp,t2,strong


def sample_scc(rng,true_C,shots,reference_bright,reference_dark):
    """Poisson total counts, with signal/ref each having shots frames.

    Signal = (Rbright+Rdark)/2 + C*(Rbright-Rdark)/2. References have
    shots/2 even and shots/2 odd frames. Sem uses delta-method.
    """
    bright=reference_bright[:,None]
    dark=reference_dark[:,None]
    signal=(bright+dark)/2 + (bright-dark)/2*true_C
    if np.any(signal<=0):raise ValueError('Negative expected photons')
    sigmean=rng.poisson(shots*signal)/shots
    bhat=rng.poisson((shots/2)*bright,size=signal.shape)/(shots/2)
    dhat=rng.poisson((shots/2)*dark,size=signal.shape)/(shots/2)
    delta=bhat-dhat
    with np.errstate(divide='ignore',invalid='ignore'):
        C=(2*sigmean-bhat-dhat)/delta
        sC=np.sqrt((4*sigmean/shots)/delta**2+
             ((C+1)**2 * 2*bhat/shots + (C-1)**2 * 2*dhat/shots)/delta**2)
    return C,sC,delta


def filter_transform(row,frequency_hz):
    f=np.atleast_1d(np.asarray(frequency_hz,float))
    tt=np.r_[0.,row['ticks']*CLOCK_NS,row['T_us']*1000]*1e-9
    flip=(-1.)**np.arange(len(tt)-1)
    w=2*np.pi*f
    y=np.sum(flip[None,:]*(
            np.exp(1j*w[:,None]*tt[1:])-np.exp(1j*w[:,None]*tt[:-1]))/
             (1j*w[:,None]),axis=1)
    return y


def filters_for_rows(rows,edges,nfreq=41):
    F=np.empty((len(rows),len(edges)-1))
    for i,row in enumerate(rows):
        for k,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
            f=np.linspace(a,b,nfreq)
            F[i,k]=np.trapezoid(abs(filter_transform(row,f))**2,f)/(b-a)/1e-9
    return F


def fit_coherence(C,SEM,F,npi,train):
    train=train & np.isfinite(C)&np.isfinite(SEM)&(C>.04)&(C<.99)
    npar=F.shape[1]+2
    if train.sum()<max(12,3*npar):return None
    A=np.column_stack([np.ones(len(C)),npi/10.,F])[train]
    y=-np.log(C[train]); sig=np.maximum(SEM[train]/C[train],.05)
    w=1/sig
    w=np.clip(w,np.percentile(w,10),np.percentile(w,90))
    Aw=A*w[:,None];yw=y*w
    norm=np.linalg.norm(Aw,axis=0)
    if np.any(norm<1e-8):return None
    U=Aw/norm
    if np.linalg.matrix_rank(U)<npar:return None
    beta,_=nnls(U,yw,maxiter=100*npar)
    return beta/norm


def cross_validate(C,SEM,F,npi,seed=109):
    good=np.isfinite(C)&np.isfinite(SEM)&(C>.04)&(C<.99)
    if good.sum()<32:return np.nan,np.nan,np.nan
    N=len(npi)
    permutation=np.random.default_rng(seed).permutation(N)
    folds=np.empty(N,int);folds[permutation]=np.arange(N)%4
    full_predictions=[];null_predictions=[];observed=[]
    for fold in range(4):
        tr=good&(folds!=fold);te=good&(folds==fold)
        if te.sum()<3:continue
        coef=fit_coherence(C,SEM,F,npi,tr)
        # duration and pulse-count only; pulse count is nuisance,
        # important because patterns contain different pi pulse totals.
        trn=np.column_stack([np.ones(N),npi/10.])[tr]
        y=-np.log(C[tr]); w=np.clip(C[tr]/np.maximum(SEM[tr],.001),1,30)
        b0,_=nnls(trn*w[:,None],y*w)
        if coef is None:continue
        full_predictions.extend((np.column_stack([np.ones(N),npi/10.,F])[te]@coef).tolist())
        null_predictions.extend((np.column_stack([np.ones(N),npi/10.])[te]@b0).tolist())
        observed.extend((-np.log(C[te])).tolist())
    if len(observed)<8:return np.nan,np.nan,np.nan
    obs=np.array(observed)
    r1=np.sqrt(np.mean((obs-full_predictions)**2))
    r0=np.sqrt(np.mean((obs-null_predictions)**2))
    return r1,r0,1-r1/r0 if r0>0 else np.nan


def analyze(C,sC,rows,edges):
    ids=[i for i,r in enumerate(rows) if r['kind']=='random']
    R=[rows[i] for i in ids]
    F=filters_for_rows(R,edges)
    npi=np.array([r['n_pi'] for r in R],float)
    coeff=np.full((len(C),F.shape[1]),np.nan)
    cv=np.full(len(C),np.nan)
    residual=np.full(len(C),np.nan)
    for i in range(len(C)):
        obs=C[i,ids];sem=sC[i,ids]
        co=fit_coherence(obs,sem,F,npi,np.ones(len(npi),bool))
        if co is not None:
            coeff[i]=co[2:]
            good=np.isfinite(obs)&(obs>.04)&(obs<.99)
            pred=np.column_stack([np.ones(len(npi)),npi/10.,F])@co
            residual[i]=np.sqrt(np.mean((-np.log(obs[good])-pred[good])**2))
        cv[i]=cross_validate(obs,sem,F,npi)[-1]
    return coeff,cv,residual,F


def set_plot_theme():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
              'axes.titlesize':12,'axes.labelsize':10,
              'figure.facecolor':'white','axes.facecolor':'white',
              'savefig.facecolor':'white','pdf.fonttype':42})


def generate(args):
    rng=np.random.default_rng(args.seed)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    T0=60.
    field=args.field_g
    fl=GAMMA_C13_HZ_G*field
    apar,aperp,t2,strong=sample_hyperfine(rng,args.n_nvs,args.n_spins)
    steps={}
    ideal={};measured={};sems={};deltas={}
    b0=rng.uniform(7.2,9.0,args.n_nvs)
    contrast=rng.uniform(2.8,5.8,args.n_nvs)
    contrast[rng.random(args.n_nvs)<.08]=2.0
    d0=b0-contrast
    for T in (40.,60.,80.):
        maskseed=args.seed+int(T)
        steps[T]=make_steps(T,args.n_masks,maskseed)
        ideal[T]=simulate_quantum_bath(steps[T],field,apar,aperp,t2,
                                       pulse_error=args.pulse_error)
        measured[T],sems[T],deltas[T]=sample_scc(
             rng,ideal[T],args.shots,b0,d0)
    # We visually show invalid SCC references as missing values.
    for T in steps:
        valid=np.abs(deltas[T])>=args.min_reference
        measured[T][~valid]=np.nan
        sems[T][~valid]=np.nan
    edges=np.asarray([5853.725,24146.275,42446.275,60753.725],float)
    coeff,cv,residual,F=analyze(measured[T0],sems[T0],steps[T0],edges)
    mask=steps[T0][:args.n_masks]
    npi=np.asarray([x['n_pi'] for x in mask])
    fl_overlap=abs(np.asarray([filter_transform(r,[fl])[0] for r in mask]))**2/1e-9
    scored=np.nanmedian(ideal[T0][:,:args.n_masks],axis=1)
    typical=int(np.nanargmin(abs(scored-np.nanmedian(scored))))
    strongids=np.flatnonzero(strong)
    exceptional=int(strongids[0]) if len(strongids) else int(np.nanargmin(scored))
    chosen=[typical,exceptional]
    set_plot_theme()
    pages=[]

    # Figure 1 pulse schedule and the corresponding frequency filters.
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    for j,i in enumerate([0,1,2,3]):
        ts=mask[i]['ticks']*CLOCK_NS/1000
        axs[0].hlines(j,0,T0,color='#c5cbd5',linewidth=1.1)
        axs[0].vlines(ts,j-.34,j+.34,color=plt.cm.tab10(j),linewidth=2)
        axs[0].text(T0+.6,j,f'Mask {i+1}: {len(ts)} pi',va='center',fontsize=9)
    axs[0].set_xlim(-1,T0+16);axs[0].set_ylim(-.7,3.7)
    axs[0].set_yticks(range(4),['mask 1','mask 2','mask 3','mask 4'])
    axs[0].set_xlabel('Sensing time (microseconds)')
    axs[0].set_title('A. Different random-multisine pi_X pulse patterns, all at T=60 us')
    ff=np.linspace(5000,70000,500)
    for i in [0,1,2,3]:
        filt=abs(filter_transform(mask[i],ff))**2
        axs[1].plot(ff/1000,filt/filt.max(),label=f'mask {i+1} ({mask[i]["n_pi"]} pi)')
    axs[1].axvline(fl/1000,ls='--',c='k',label=f'bare 13C {fl/1000:.2f} kHz')
    axs[1].set_xlabel('Nuclear-frequency hypothesis (kHz)')
    axs[1].set_ylabel('Normalized ideal filter power |Y(f)|²')
    axs[1].set_title('B. Every random pulse sequence probes a different spectral filter')
    axs[1].legend(ncol=3,fontsize=8);axs[1].grid(alpha=.2)
    fig.suptitle('Synthetic RQS / 13C bath: pulse patterns and sensitivity',fontweight='bold')
    pages.append(('01_pulse_masks_and_filters',fig))

    # Figure 2 what the SCC camera would measure across a sample of 119 NVs.
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    for ax,data,title in [(axs[0],ideal[T0][:,:args.n_masks],
                  'A. Ground-truth quantum-spin coherence (before SCC noise)'),
                  (axs[1],measured[T0][:,:args.n_masks],
                  'B. Simulated SCC-normalized coherence, noisy and contrast-filtered')]:
        m=ax.imshow(data,aspect='auto',vmin=-1,vmax=1,cmap='RdBu_r',
                    origin='lower',interpolation='nearest')
        ax.set_xlabel(f'Random mask index (0–{args.n_masks-1})');ax.set_ylabel('NV index')
        ax.set_title(title)
        fig.colorbar(m,ax=ax,label='Normalized X coherence')
    fig.suptitle('What a multiplexed 13C-bath RQS measurement can look like',fontweight='bold')
    pages.append(('02_nv_mask_coherence_heatmaps',fig))

    # Figure 3 individual NV response with measurement shot-noise errorbars.
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    for ax,idx,name in zip(axs,chosen,['typical synthetic NV','more strongly coupled synthetic NV']):
        ax.errorbar(np.arange(args.n_masks),measured[T0][idx,:args.n_masks],
                    yerr=sems[T0][idx,:args.n_masks],fmt='.',alpha=.7,
                    color='#286da8',label='Simulated SCC measurement ±1 SEM')
        ax.plot(ideal[T0][idx,:args.n_masks],lw=1.7,color='#d34b3f',
                label='True quantum-bath coherence')
        ax.set(ylabel='X coherence',title=f'{name} (NV {idx}, {args.n_spins} effective nuclei)')
        ax.axhline(0,ls=':',c='grey');ax.grid(alpha=.18)
        ax.legend(fontsize=8,loc='best')
    axs[-1].set_xlabel('Random pulse mask (T=60 us)')
    fig.suptitle('The measured observable: coherence versus random mask',fontweight='bold')
    pages.append(('03_individual_nv_mask_response',fig))

    # Figure 4 test response versus 13C filter overlap; colour by pulse count.
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,idx in zip(axs,chosen):
        sc=ax.scatter(fl_overlap,measured[T0][idx,:args.n_masks],c=npi,
                    cmap='viridis',s=42,vmin=npi.min(),vmax=npi.max())
        ax.set_xlabel('Ideal filter |Y(f_L)|² / 1e-9 s²')
        ax.set_ylabel('SCC-normalized X coherence')
        ax.set_title(f'Synthetic NV {idx}: modulation near f_L')
        ax.grid(alpha=.2)
    fig.colorbar(sc,ax=axs,label='Refocusing pi count',shrink=.8)
    fig.suptitle('Look for filter-dependent contrast, not just pulse-count decay',fontweight='bold')
    pages.append(('04_coherence_vs_33khz_overlap',fig))

    # Figure 5 candidate frequency bins and crossvalidated prediction.
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    cm=np.log10(1+np.maximum(coeff,0))
    im=axs[0].imshow(cm,aspect='auto',interpolation='nearest',origin='lower',cmap='viridis',
               extent=[edges[0]/1000,edges[-1]/1000,-.5,args.n_nvs-.5])
    axs[0].axvline(fl/1000,color='white',ls='--',label='Bare 13C')
    axs[0].set_title('A. Gaussian-PSD-proxy reconstruction from a QUANTUM bath (model mismatch possible)')
    axs[0].set_xlabel('Coarse frequency bin (kHz)');axs[0].set_ylabel('NV index')
    axs[0].legend(fontsize=8)
    fig.colorbar(im,ax=axs[0],label='log10(1+ fitted nonnegative PSD proxy)')
    valid=np.isfinite(cv)
    axs[1].hist(cv[valid],bins=np.linspace(-1,1,31),color='#5078a5',edgecolor='white')
    axs[1].axvline(0,ls='--',c='#bc3b3b',label='No predictive gain')
    axs[1].set(xlabel='Held-out mask improvement vs pulse-count-only baseline',
               ylabel='Number of NVs',title=f'B. {np.sum(valid)} NVs testable; positive means better generalization')
    axs[1].legend();axs[1].grid(alpha=.18)
    fig.suptitle('Do not interpret spectral-proxy maxima as measured hyperfine couplings',fontweight='bold')
    pages.append(('05_spectral_proxy_and_validation',fig))

    # Figure 6 echo and periodic XY4 / XY8 vs T.
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    Ts=np.array([40.,60.,80.])
    labelcols={'random':'#287aaf','XY4':'#d48c27','XY8':'#9a558b','Hahn':'#3b8c6e'}
    for kind in ('random','XY4','XY8','Hahn'):
        med=[];q25=[];q75=[]
        for T in Ts:
            idx=[j for j,r in enumerate(steps[T]) if r['kind']==kind]
            X=measured[T][:,idx]
            vals=np.nanmedian(X,axis=1)
            med.append(np.nanmedian(vals))
            q25.append(np.nanpercentile(vals,25));q75.append(np.nanpercentile(vals,75))
        axs[0].plot(Ts,med,'o-',color=labelcols[kind],label=kind)
        axs[0].fill_between(Ts,q25,q75,color=labelcols[kind],alpha=.15)
    axs[0].set(xlabel='Fixed sensing time T (microseconds)',ylabel='Median SCC coherence',
               title='A. Different pulse protocols have different sensitivity and errors')
    axs[0].legend(ncol=4);axs[0].grid(alpha=.2)
    et=np.linspace(5,180,71)
    echo_rows=[dict(kind='Hahn',T_us=float(T),ticks=periodic_centers(T,1),n_pi=1)
               for T in et]
    EC=simulate_quantum_bath(echo_rows,field,apar,aperp,t2,args.pulse_error)
    for idx,col in zip(chosen,['#286da8','#d34b3f']):
        axs[1].plot(et,EC[idx],label=f'synthetic NV {idx}',color=col)
    axs[1].plot(et,np.nanmedian(EC,axis=0),'k--',lw=2,label='ensemble median')
    axs[1].axvline(60,ls=':',c='gray')
    axs[1].set(xlabel='Hahn-echo total evolution time (microseconds)',
               ylabel='Quantum-bath coherence',
               title='B. Synthetic Hahn echo: collapse/revival features depend on couplings')
    axs[1].legend(ncol=3,fontsize=8);axs[1].grid(alpha=.2)
    fig.suptitle('Controls needed to interpret a 13C bath response',fontweight='bold')
    pages.append(('06_controls_and_spin_echo',fig))

    # Output.
    pdfpath=out/'RQS_13C_bath_simulation.pdf'
    with PdfPages(pdfpath) as pdf:
        for name,fig in pages:
            fig.savefig(out/(name+'.png'),dpi=155,bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)
    summary={
        'simulation_only':True,
        'field_g':field,'bare_c13_larmor_hz':fl,
        'n_nvs':args.n_nvs,'effective_spins_per_nv':args.n_spins,
        'T_us':[40,60,80], 'random_masks_per_T':args.n_masks,
        'shots_per_setting':args.shots,'pi_ns_physical_constraint':PI_NS,
        'pulse_spacing_ns_min':MIN_GAP_NS,
        'mean_median_coherence_by_T':{str(int(T)):float(np.nanmedian(measured[T][:,:args.n_masks])) for T in steps},
        'NV_fit_fraction_60us':float(np.isfinite(coeff).all(axis=1).mean()),
        'NV_cv_positive_fraction_of_testable_60us':float(np.mean(cv[np.isfinite(cv)]>0)) if np.isfinite(cv).any() else None,
        'mean_strong_spin_fraction':float(strong.mean()),
        'notes':['Exact independent-unpolarized-spin conditional quantum dynamics.',
                 'Independent synthetic hyperfine couplings chosen only to illustrate a bath.',
                 'Finite 96ns pulse width excluded from dynamics; included only via min spacing.',
                 'True SCC contrast and pulse infidelity must be calibrated from the actual apparatus.',
                 'Gaussian PSD reconstruction is intentionally misspecified for coherent few-spin baths.'],
        'plots':[name+'.png' for name,_ in pages]
    }
    (out/'simulation_parameters_and_summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'simulated_NV_parameters.csv').open('w',newline='') as io:
        writer=csv.writer(io);writer.writerow(['nv_index','has_stronger_spin','T2_background_us',
                      'A_parallel_hz_each_spin','A_perp_hz_each_spin','bright_mean','dark_mean'])
        for i in range(args.n_nvs):
            writer.writerow([i,int(strong[i]),round(t2[i]*1e6,1),
                      ';'.join(f'{a:.1f}' for a in apar[i]),
                      ';'.join(f'{a:.1f}' for a in aperp[i]),
                      round(b0[i],2),round(d0[i],2)])
    np.savez_compressed(out/'simulated_RQS_13C_data.npz',
          T60_quantum_coherence=ideal[T0],T60_noisy_SCC_coherence=measured[T0],
          T60_sem=sems[T0],T60_npi=npi,T60_filter_bin_weights=F,
          T60_reconstructed_PSD_proxy=coeff,T60_CV_improvement=cv,
          A_parallel_hz=apar,A_perp_hz=aperp,T2_background_s=t2,
          T40_noisy_C=measured[40.],T80_noisy_C=measured[80.])
    print('Created:',pdfpath)
    print('Larmor fL:',fl,'Hz')
    print('Median noisy coherence at 40/60/80 us:',[round(summary['mean_median_coherence_by_T'][str(int(t))],3) for t in (40.,60.,80.)])
    print('CV testable:',int(np.sum(np.isfinite(cv))),'of',args.n_nvs,
           'positive:',int(np.sum(cv>0)))
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',default='G:\\nvdata\\pc_NVOffice\\branch_master\\RQS_C13_simulation')
    ap.add_argument('--n-nvs',type=int,default=119)
    ap.add_argument('--n-masks',type=int,default=64)
    ap.add_argument('--n-spins',type=int,default=10)
    ap.add_argument('--seed',type=int,default=37)
    ap.add_argument('--field-g',type=float,default=31.1)
    ap.add_argument('--shots',type=int,default=2000)
    ap.add_argument('--min-reference',type=float,default=0.0)
    ap.add_argument('--pulse-error',type=float,default=.006)
    a=ap.parse_args()
    if a.n_masks>512 or a.n_nvs<2 or a.n_masks<16 or a.shots<2 or a.shots%2:
        ap.error('Use 16<=n-masks<=512, n-nvs>=2, even shots>=2')
    generate(a)

if __name__=='__main__':
    main()
