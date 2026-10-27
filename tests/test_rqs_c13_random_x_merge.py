"""Test that two different seeds add independent filters rather than averaging."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from majorroutines.widefield.rqs_c13 import build_protocol
from analysis.sc_rqs_c13_random_x_estimator import _pool_different_masks


def fake(seed):
    p=build_protocol(field_g=31.1,band="high",seed=seed,
                     masks_per_duration=16,pi_ns=96,pi2_ns=48)
    q=[]
    for j,s in enumerate(p["steps"]):
        d=dict(s,kind="random_x" if s["kind"]=="random"
               else "periodic_xy8",
               phase_deg=[0]*8 if s["kind"]=="random"
               else [0,90,0,90,90,0,90,0],
               final_pi2_phase_deg=0,mask_id=j)
        q.append(d)
    return {"stem":str(seed),"proto":p,"steps":q,
            "C":np.full((2,len(q)),.8),
            "names":["nv0","nv1"],"shot_count":1000}


def test():
    a,b=fake(37),fake(38)
    d=_pool_different_masks([a,b])
    assert len(d["steps"])==144
    assert d["C"].shape==(2,144)
    assert sum(s["kind"]=="random_x" for s in d["steps"])==128
    assert sum(s["kind"]=="periodic_xy8" for s in d["steps"])==16
    assert np.allclose(d["C"],.8)
    e=_pool_different_masks([a,fake(37)])
    assert len(e["steps"])==80
    assert np.allclose(e["C"],.8)
    print("PASS: different seeds => 128 masks; matching seeds => 64 averaged masks")


if __name__=="__main__":
    test()
