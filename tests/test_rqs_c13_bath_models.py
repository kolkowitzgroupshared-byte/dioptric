"""Mathematical tests of 13C bath filter and conditional quantum dynamics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.rqs_multisine_core import compile_bank
from analysis.rqs_c13_bath_models import (
    toggling_filter,filter_bins,compare_models,heldout_cv,
    quantum_c13_coherence,
)


def test():
    T=60_000
    freqs=[15000.,33292.55,51600.]
    bank=compile_bank(freqs,duration_us=60.,seed=37,pool_size=256)
    ids=np.arange(64)
    rng=np.random.default_rng(44)
    rng.shuffle(ids)
    steps=[{"switch_ticks":list(bank.pulses_ticks[k]),
            "pi_centers_ns":[4*x for x in bank.pulses_ticks[k]],
            "T_ns":T,
            "n_pi":len(bank.pulses_ticks[k])}
           for k in ids]
    edges=[5853.725,24146.275,42446.275,60753.725]
    F=filter_bins(steps,edges)
    assert F.shape==(64,3)
    Y=toggling_filter([],4,T,[0,33e3])
    assert abs(Y[0]-60e-6)<1e-10
    assert np.all(F>=0)
    # Synthetic Gaussian-noise screening, with pulse-dependent nuisance.
    chi=.06+.025*np.asarray([s["n_pi"] for s in steps])
    chi+= (F/1e-9)@np.asarray([.02,.35,.02])
    C=np.exp(-chi)
    C+=rng.normal(0,.005,len(C))
    spectral,baseline=compare_models(
        F,np.asarray([s["n_pi"] for s in steps]),C)
    assert spectral["status"]=="ok",spectral
    assert baseline["status"]=="ok"
    cv=heldout_cv(F,np.asarray([s["n_pi"] for s in steps]),C)
    assert cv["status"]=="ok",cv
    assert cv["improvement"]>.0,cv
    print("PASS Gaussian spectral inference",cv)
    # Static z detuning refocuses under ideal Hahn echo.
    echo=[{"pi_centers_ns":[T//2],"T_ns":T,"n_pi":1}]
    echo_C=quantum_c13_coherence(echo,field_g=31.1,
                 a_parallel_hz=80_000,a_perp_hz=0)
    assert abs(echo_C[0]-1)<1e-10,echo_C
    ramsey=[{"pi_centers_ns":[],"T_ns":T,"n_pi":0}]
    val=quantum_c13_coherence(ramsey,field_g=31.1,
                 a_parallel_hz=80_000,a_perp_hz=0)[0]
    assert abs(val-np.cos(np.pi*80_000*T*1e-9))<1e-10,(val,T)
    arr=quantum_c13_coherence(steps,field_g=31.1,
                  a_parallel_hz=20_000,a_perp_hz=15_000)
    assert np.all(np.abs(arr)<=1+1e-12)
    print("PASS quantum 13C conditional evolutions (echo, Ramsey, bath)")
if __name__=="__main__":
    test()
