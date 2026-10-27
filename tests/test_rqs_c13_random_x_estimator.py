"""End-to-end synthetic SCC test (never operates lab hardware)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import numpy as np
from majorroutines.widefield.rqs_c13 import build_protocol
from analysis import sc_rqs_c13_random_x_estimator as estimator


def test():
    p=build_protocol(field_g=31.1,band="high",seed=37,
                     masks_per_duration=16,pi_ns=96,pi2_ns=48)
    for j,s in enumerate(p["steps"]):
        if s["kind"]=="random":
            s.update(kind="random_x",mask_id=j,
                     phase_deg=[0]*8,final_pi2_phase_deg=0)
        else:
            s.update(kind="periodic_xy8",mask_id=j,
                     final_pi2_phase_deg=0)
    p["experiment"]="rqs_c13_random_pi_x_with_periodic_xy8_reference"
    steps=p["steps"]; n=len(steps)
    rng=np.random.default_rng(41)
    # Coherence from synthetic Gaussian noise with positive power bins.
    random=[j for j,s in enumerate(steps) if s["kind"]=="random_x"]
    edges=np.linspace(150e3,1.5e6,9)
    F=estimator.filter_design([steps[j] for j in random],edges)
    t=np.asarray([steps[j]["T_ns"] for j in random])
    dur=(t[:,None]==np.unique(t)[None,:]).astype(float)
    C=np.full((2,n),0.85)
    C[:,random]=np.exp(-F@np.array([.01,.01,.02,.08,.02,.01,.01,.01])-
                       dur@np.full(4,.04))[None,:]
    counts=np.zeros((2,2,4,n,10))
    for i in range(2):
        for j in range(n):
            counts[0,i,:,j,:]=150+50*C[i,j]+rng.normal(0,.5,(4,10))
            counts[1,i,:,j,::2]=200
            counts[1,i,:,j,1::2]=100
    raw={"rqs_c13_protocol":p,"rqs_complete":True,
         "nv_list":[{"name":"synthetic_nv0"},{"name":"synthetic_nv1"}]}
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp)/"synthetic_rqs"
        base.with_suffix(".txt").write_text(json.dumps(raw))
        np.savez_compressed(base.with_suffix(".npz"),counts=counts)
        d=estimator._load(str(base.with_suffix(".txt")),min_ref=2.5)
        assert d["C"].shape==(2,n)
        out=Path(tmp)/"out"
        report=estimator.analyze(d,nbin=8,outdir=out)
        assert report["fit_count"]==2,report
        assert (out/"rqs_pi_x_estimator.pdf").exists()
        assert (out/"fit_arrays.npz").exists()
    print("PASS end-to-end synthetic SCC, PSD fit and PDF export")


if __name__=="__main__":
    test()
