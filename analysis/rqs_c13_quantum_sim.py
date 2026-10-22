# -*- coding: utf-8 -*-
"""Joint electron-13C quantum simulation of randomized DD and XY8-8.

Electron sensor |0>, |-1>; one or more independent spin-1/2 13C nuclei.
Hamiltonians conditional on electron state (frequency units Hz):
  H0/hbar = 2*pi * f_L * I_z
  H1/hbar = 2*pi * ((f_L + A_parallel)*I_z + A_perp*I_x)
Electron pi pulses (ideal/instantaneous) swap the two paths at saved centers.

C=Re[prod_j Tr(U_branchB_j^dagger U_branchA_j)/2] is the signed spin coherence
for a maximally mixed, unpolarized nuclear bath. This is not a fitted model.
Missing: finite MW pulses, remote 14N, bath dynamics, T2 envelope, optical errors.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from majorroutines.widefield.rqs_c13 import build_protocol
from analysis.rqs_c13_analysis import band_overlaps


_ID = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)


def nuclear_u(dt_s, fz_hz, fx_hz):
    norm = np.hypot(fz_hz, fx_hz)
    if norm == 0:
        return _ID.copy()
    angle = np.pi * norm * dt_s
    return np.cos(angle) * _ID - 1j*np.sin(angle)*(
        (fz_hz*_Z + fx_hz*_X) / norm
    )


def one_nucleus_coherence(step, larmor_hz, a_parallel_hz, a_perp_hz):
    boundaries = np.r_[0., np.asarray(step["pi_centers_ns"])*1e-9,
                       step["T_ns"]*1e-9]
    ua = _ID.copy()
    ub = _ID.copy()
    for seg, dt in enumerate(np.diff(boundaries)):
        # branch A begins in m_s=0, B in m_s=-1, then they exchange at pi pulses.
        a_is_zero = (seg % 2 == 0)
        u0 = nuclear_u(dt, larmor_hz, 0.)
        u1 = nuclear_u(dt, larmor_hz+a_parallel_hz, a_perp_hz)
        ua = (u0 if a_is_zero else u1) @ ua
        ub = (u1 if a_is_zero else u0) @ ub
    return np.trace(ub.conj().T @ ua)/2


def multi_nucleus_coherence(step, larmor_hz, hyperfine_sites_hz):
    coherence = 1.0 + 0j
    for apar, aperp in hyperfine_sites_hz:
        coherence *= one_nucleus_coherence(step, larmor_hz, apar, aperp)
    return float(np.real(coherence))


def simulate(proto, sites_hz):
    fL=proto["bare_c13_larmor_hz_expected"]
    return np.array([multi_nucleus_coherence(s, fL, sites_hz)
                     for s in proto["steps"]])


def run(outdir="analysis/rqs_c13_quantum_preview", field_g=33.4):
    out=Path(outdir); out.mkdir(parents=True, exist_ok=True)
    low=build_protocol(field_g=field_g, band="low", masks_per_duration=10)
    high=build_protocol(field_g=field_g, band="high", masks_per_duration=10)
    # ILLUSTRATIVE ONLY. The real diamond's Azz/Aperp must be fitted.
    low_sites_hz=[(3e3, 16e3), (17e3, 11e3)]
    high_sites_hz=[(300e3, 65e3)]
    clow=simulate(low, low_sites_hz)
    chigh=simulate(high, high_sites_hz)
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,(proto,c,sites) in enumerate([(low,clow,low_sites_hz),
                                          (high,chigh,high_sites_hz)]):
        xy=[i for i,s in enumerate(proto["steps"]) if s["kind"]=="xy8"]
        rnd=[i for i,s in enumerate(proto["steps"]) if s["kind"]=="random"]
        f=[proto["steps"][i]["target_f_hz"]/1e3 for i in xy]
        ax[0,col].plot(f,1-c[xy],"o-",label="XY8-8",ms=4)
        ax[0,col].set_xlabel("Nominal XY8 frequency (kHz)")
        ax[0,col].set_ylabel("Nuclear-induced loss 1-Re(L)")
        ax[0,col].set_title(proto["band"]+" bank / synthetic A="+
                            str([(round(a/1e3), round(b/1e3)) for a,b in sites])+
                            " kHz")
        ax[0,col].grid(alpha=.2)
        overlaps=band_overlaps(proto, proto["bare_c13_larmor_hz_expected"]
                              if col==0 else 330e3)
        t=[proto["steps"][i]["T_ns"]/1000 for i in rnd]
        sc=ax[1,col].scatter(overlaps[rnd]/1e-9,1-c[rnd],c=t,cmap="plasma")
        ax[1,col].set_xlabel("Ideal target-band filter overlap (1e-9 s²)")
        ax[1,col].set_ylabel("Nuclear-induced loss 1-Re(L)")
        ax[1,col].set_title("Random-timing XY8-8, same 8 microwave pi pulses")
        fig.colorbar(sc,ax=ax[1,col],label="Total sensing time (us)")
    fig.suptitle("INDEPENDENT SPIN-1/2 13C QUANTUM EXAMPLE — NOT EXPERIMENTAL DATA")
    path=out/"rqs_c13_joint_spin_simulation.png"
    fig.savefig(path,dpi=175); plt.close(fig)
    print("Saved",path)
    print("Low bank: XY8 loss span",float(np.ptp(1-clow[[i for i,s in enumerate(low["steps"]) if s["kind"]=="xy8"]])))
    print("High bank: XY8 loss span",float(np.ptp(1-chigh[[i for i,s in enumerate(high["steps"]) if s["kind"]=="xy8"]])))
    return path


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field-g",type=float,default=33.4)
    p.add_argument("--outdir",default="analysis/rqs_c13_quantum_preview")
    a=p.parse_args()
    # elementary consistency checks
    p0=build_protocol(masks_per_duration=2)
    s=p0["steps"][0]; f=p0["bare_c13_larmor_hz_expected"]
    assert abs(one_nucleus_coherence(s,f,0.,0.)-1)<1e-10
    assert np.isfinite(one_nucleus_coherence(s,f,10e3,20e3))
    print("PASS: decoupled nucleus and finite-coupling coherence checks")
    run(a.outdir,a.field_g)
