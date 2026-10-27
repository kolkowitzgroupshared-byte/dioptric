"""Synthetic phase-quadrature estimator test, no hardware."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from majorroutines.widefield.rqs_c13 import build_protocol
from analysis.sc_rqs_c13_random_x_estimator import phase_fit, GAMMA_NV_RAD_PER_S_PER_NT


def test():
    protocol=build_protocol(band="high",seed=37,masks_per_duration=16,
                            pi_ns=96,pi2_ns=48)
    steps=[]
    for j,s in enumerate(protocol["steps"]):
        if s["kind"]!="random":
            continue
        for final in (0,90):
            step=dict(s,kind="random_x",mask_id=j,
                      final_pi2_phase_deg=final,phase_deg=[0]*8)
            steps.append(step)
    edges=np.linspace(150e3,1.5e6,7)
    freq=np.linspace(edges[0],edges[-1],6)
    amps=np.array([10,-15,4,1,7,2,5,1,1,2,3,0],float)
    C=np.zeros(len(steps))
    for j,s in enumerate(steps):
        t=np.r_[0,np.asarray(s["pi_centers_ns"])*1e-9,
                s["T_ns"]*1e-9]
        w=2*np.pi*freq
        Y=np.sum(((-1)**np.arange(len(t)-1))[None,:]*(
            np.exp(1j*w[:,None]*t[1:])-np.exp(1j*w[:,None]*t[:-1])),
                 axis=1)/(1j*w)
        phase=(GAMMA_NV_RAD_PER_S_PER_NT*np.r_[Y.real,Y.imag])@amps
        C[j]=.83*(np.cos(phase) if s["final_pi2_phase_deg"]==0
                  else np.sin(phase))
    fit=phase_fit(C,steps,edges,nfreq=6)
    assert fit["status"]=="ok",fit
    assert fit["rms"]<0.01,fit
    print("PASS synthetic phase quadrature inversion, rmse:",fit["rms"])


if __name__=="__main__":
    test()
