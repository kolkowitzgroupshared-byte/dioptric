"""Quick synthetic candidate conditional single-spin fit test."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.rqs_multisine_core import compile_bank
from analysis.rqs_c13_bath_models import quantum_c13_coherence,quantum_fit_candidate
bank=compile_bank([15000,33292.55,51600],duration_us=60,
                  seed=17,pool_size=128)
steps=[{"pi_centers_ns":[int(k*4) for k in s],
        "T_ns":60000,"n_pi":len(s)} for s in bank.pulses_ticks[:24]]
npi=np.asarray([r["n_pi"] for r in steps],float)
C=.85*np.exp(-.015*npi)*quantum_c13_coherence(
       steps,field_g=31.1,a_parallel_hz=18000,a_perp_hz=12000)
fit=quantum_fit_candidate(steps,C,np.full(len(C),.04),field_g=31.1)
assert fit["status"]=="ok",fit
assert np.isfinite(fit["heldout_weighted_rms"]),fit
print("PASS exploratory quantum-spin fit",fit)
