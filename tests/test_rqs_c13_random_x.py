"""Hardware-free tests of randomized all-X RQS sensing pilot."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from majorroutines.widefield.rqs_c13_random_x import build_protocol
from servers.timing.sequencelibrary.QM_opx.camera.rqs_c13_random_x import get_seq


def test_mode(readout="both", masks=2):
    p=build_protocol(masks_per_duration=masks,readout=readout,pi_ns=96,pi2_ns=48)
    q=p["steps"]
    nrand=4*masks*(2 if readout=="both" else 1)
    assert sum(s["kind"]=="random_x" for s in q)==nrand
    assert sum(s["kind"]=="periodic_xy8" for s in q)==16
    assert all(s["phase_deg"]==[0]*8 for s in q if s["kind"]=="random_x")
    assert all(s["phase_deg"]==[0,90,0,90,90,0,90,0]
               for s in q if s["kind"]=="periodic_xy8")
    assert all(abs(s["T_ns"]-s["requested_T_ns"])<=4 for s in q)
    waits=[[s["waits_cc"][j] for s in q] for j in range(9)]
    axes=[s["pulse_axis_mode"] for s in q]
    basis=[s["final_pi2_phase_deg"] for s in q]
    base=[[[107.72,107.70]],[164],[1.0],[[72.44,73.22]],[88],[1.0],[False],[1]]
    seq,_=get_seq(base,waits,axes,basis,2)
    assert seq is not None
    print("PASS QUA",readout,"steps",len(q))


if __name__=="__main__":
    test_mode("cos",2)
    test_mode("both",2)
    test_mode("cos",16)
