"""Synthetic SCC-data to 13C bath PDF/CSV/NPZ pipeline, no instruments."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile,json
import numpy as np
from utils.rqs_multisine_core import compile_bank
from analysis.sc_rqs_c13_bath_analysis import load_file,pool_files,analyze
from analysis.rqs_c13_bath_models import filter_bins


def build(seed,base):
    bank=compile_bank([15000.,33292.55,51600.],duration_us=60.,
                      seed=seed,pool_size=256)
    rows=[]
    idx=np.random.default_rng(seed+11).choice(len(bank.pulses_ticks),64,replace=False)
    for k in idx:
        centers=list(bank.pulses_ticks[k])
        rows.append({"kind":"random_multisine",
                "bank_index":int(k),"switch_ticks":centers,
                "pi_centers_ns":[4*int(c) for c in centers],
                "n_pi":len(centers),"T_ns":60_000,"phase_mode":0})
    t=15000
    xy=np.rint((np.arange(8)+.5)*t/8).astype(int)
    rows.append({"kind":"periodic_xy8","switch_ticks":xy.tolist(),
         "pi_centers_ns":(xy*4).tolist(),"n_pi":8,
         "T_ns":60000,"phase_mode":1})
    xy4=np.rint((np.arange(4)+.5)*t/4).astype(int)
    rows.append({"kind":"periodic_xy4","switch_ticks":xy4.tolist(),
         "pi_centers_ns":(xy4*4).tolist(),"n_pi":4,
         "T_ns":60000,"phase_mode":2})
    rows.append({"kind":"spin_echo","switch_ticks":[t//2],
         "pi_centers_ns":[30000],"n_pi":1,"T_ns":60000,"phase_mode":0})
    edges=[5853.725,24146.275,42446.275,60753.725]
    F=filter_bins(rows[:64],edges)/1e-9
    npi=np.array([r["n_pi"] for r in rows[:64]])
    rng=np.random.default_rng(seed)
    C=np.empty((5,len(rows)))
    for j in range(5):
        spect=np.array([.03,.15+.025*j,.035])
        C[j,:64]=np.exp(-.09-.005*npi-F@spect)
        C[j,64]=.87
        C[j,65]=.76
        C[j,66]=.64
    counts=np.empty((2,5,3,len(rows),20),float)
    counts[0]=150+50*C[:,None,:,None]+rng.normal(0,1,
                                               (5,3,len(rows),20))
    counts[1,:,:,:,::2]=200+rng.normal(0,1,(5,3,len(rows),10))
    counts[1,:,:,:,1::2]=100+rng.normal(0,1,(5,3,len(rows),10))
    protocol={"experiment":"rqs_c13_bath_fixed_T",
        "T_us":60.,"field_g_expected":31.1,
        "c13_larmor_hz_expected":33292.55,
        "frequency_centers_hz":[15000.,33292.55,51600.],
        "frequency_edges_hz":edges,
        "pi_ns":96,"pi2_ns":48,
        "steps":rows}
    raw={"rqs_complete":True,"rqs_c13_bath_protocol":protocol,
         "nv_list":[{"name":"test_NV_"+str(j)} for j in range(5)],
         "num_runs":3,"num_reps":20}
    meta=base.with_suffix(".txt");npz=base.with_suffix(".npz")
    meta.write_text(json.dumps(raw))
    np.savez_compressed(npz,counts=counts)
    return meta


def test():
    with tempfile.TemporaryDirectory() as root:
        root=Path(root)
        f=build(31,root/"sample1")
        d=load_file(f,reference_min=2.5)
        assert d["coherence"].shape==(5,67)
        assert np.allclose(d["coherence"][:,:64].mean(axis=0),d["coherence"][:,:64][0],atol=.12)
        assert np.isfinite(d["coherence_sem"]).all()
        report=analyze(d,outdir=root/"analysis",per_nv=3)
        assert report["fit_count"]==5,report
        assert report["CV_testable_NV"]>=3,report
        assert report["CV_spectral_better_NV"]>=3,report
        assert (root/"analysis"/"05_heldout_validation.pdf").is_file()
        assert (root/"analysis"/"07_NV_diagnostics.pdf").is_file()
        f2=build(32,root/"sample2")
        both=pool_files([d,load_file(f2)])
        assert len(both["steps"])==131,len(both["steps"])
        print("PASS load, pooling independent masks, filter tests, CV, PDF/PNG/CSV/NPZ")
if __name__=="__main__":
    test()
