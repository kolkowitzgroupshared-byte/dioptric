"""Offline fixed-T 13C bath random-multisine test (never shoots hardware)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from majorroutines.widefield.rqs_c13_bath import build_protocol
from servers.timing.sequencelibrary.QM_opx.camera.rqs_c13_bath import get_seq

def run(t):
    p=build_protocol(duration_us=t,n_masks=16,bank_size=128)
    rows=p["steps"]; m=p["max_pi"]
    assert len(rows)==19
    assert sum(r["kind"]=="random_multisine" for r in rows)==16
    assert sum(r["kind"]=="periodic_xy8" for r in rows)==1
    assert sum(r["kind"]=="periodic_xy4" for r in rows)==1
    assert sum(r["kind"]=="spin_echo" for r in rows)==1
    assert len(set(r["bank_index"] for r in rows if r["bank_index"] is not None))==16
    for r in rows:
        assert len(r["gaps_cc_padded"])==m+1
        assert min(r["gaps_cc_padded"])>=4
        assert sum(r["gaps_cc_padded"][:r["n_pi"]+1])+r["n_pi"]*24==int(t*250)
        assert r["T_ns"]==int(t*1000)
    base=[[[107.72,107.70]],[164],[1.0],[[72.44,73.22]],[88],[1.0],[False],[1]]
    gaps=[[int(r["gaps_cc_padded"][k]) for r in rows] for k in range(m+1)]
    npi=[int(r["n_pi"]) for r in rows]
    phase=[int(r["phase_mode"]) for r in rows]
    parity=[1 if k%2==0 else 0 for k in npi]
    seq,_=get_seq(base,gaps,npi,phase,parity,num_reps=2)
    assert seq is not None
    print("PASS bath QUA",t,"us",len(rows),"settings",m,"max pi","condition",p["bank_covariance_condition"])
if __name__=="__main__":
    for duration in (40,60,80):
        run(duration)
