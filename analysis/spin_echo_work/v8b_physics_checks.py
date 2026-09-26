"""Numerical Hahn-echo propagator checks for the V8b total-time convention."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from analysis.spin_echo_work import sc_spin_echo_physics_fit_52G_v8b_multic13 as m
raw=json.load(open(ROOT/"analysis/spin_echo_work/essem_freq_kappa_catalog_22A_52G.json"))
records=[]
for s in raw:
    if tuple(s["orientation"])==(1,1,-1) and s["site_index"] in (45,95,736):
        s["site_id"]=int(s["site_index"])
        s["orientation_tuple"]=tuple(s["orientation"])
        s["fI_kHz"]=s["fI_Hz"]/1e3;s["fm_kHz"]=s["omega_ms_Hz"]/1e3
        s["fplus_kHz"]=s["f_plus_Hz"]/1e3;s["fminus_kHz"]=s["f_minus_Hz"]/1e3
        records.append(s)
assert len(records)==3
x=np.array([[0,1],[1,0]],complex);z=np.diag([1,-1]).astype(complex);eye=np.eye(2,dtype=complex)
times=np.linspace(.4,83.32,113)
for site in records:
    f0=site["fI_Hz"]/1e6; f1=site["omega_ms_Hz"]/1e6
    kappa=site["kappa"];n=np.sqrt(kappa)*x+np.sqrt(1-kappa)*z
    for t in times:
        tau=t/2
        a,b=np.pi*f0*tau,np.pi*f1*tau
        u0=np.cos(a)*eye-1j*np.sin(a)*z
        u1=np.cos(b)*eye-1j*np.sin(b)*n
        exact=np.trace(u0@u1@u0.conj().T@u1.conj().T).real/2
        approximate=1-m.site_q(np.array([t]),site)[0]
        assert abs(exact-approximate)<3e-12,(site["site_id"],t,exact,approximate)
pool=m.catalog_for_nv(records,(1,1,-1),times)
assert all(tuple(s["orientation_tuple"])==(1,1,-1) for s in pool)
p=m.c13_coherence(times,records[:2],[.3,.7],0.)
q=(1-.3*m.site_q(times,records[0]))*(1-.7*m.site_q(times,records[1]))
assert np.allclose(p,q)
print("PASS: propagator, total-time conversion, independent product, orientation lock")

