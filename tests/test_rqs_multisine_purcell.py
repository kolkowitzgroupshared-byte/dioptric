"""Offline Purcell RQS tests: compiler, finite pulse widths and QUA."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from majorroutines.widefield import rqs_multisine as rqs
from servers.timing.sequencelibrary.QM_opx.camera.rqs_multisine import get_seq
from utils.rqs_multisine_core import exact_response
import numpy as np

def main():
    bank,rows=rqs.construct(duration_us=8,bank_size=128,n_settings=24)
    assert len(rows)==24
    assert len({s["basis"] for s in rows})==2
    assert len({s["n_pi"] for s in rows})>2
    for step in rows:
        assert sum(step["gaps_cc"][:step["n_pi"]+1])+24*step["n_pi"]==2000
        assert min(step["gaps_cc"])>=4
    for i in range(8):
        v=exact_response(bank.frequencies_hz,bank.pulses_ticks[i],
                         bank.clock_s,2000,bank.coupling_hz,
                         sign=int(bank.initial_sign[i]))
        np.testing.assert_allclose(v,bank.responses[i],atol=1e-8)
    args=[[[107.72,107.70]],[164],[1.0],[[72.44,73.22]],[88],[1.0],[False],[1]]
    lim=max(s["n_pi"] for s in rows)
    gaps=[[int(s["gaps_cc"][j]) for s in rows] for j in range(lim+1)]
    counts=[int(s["n_pi"]) for s in rows]
    basis=[0 if s["basis"]=="X" else 1 for s in rows]
    program,_=get_seq(args,gaps,counts,basis,[1 if k%2==0 else 0 for k in counts],num_reps=2)
    assert program is not None
    print("PASS variable-pi QUA and 4ns timing; bank condition",bank.condition)

if __name__=="__main__":
    main()
