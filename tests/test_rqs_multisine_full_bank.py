import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from majorroutines.widefield.rqs_multisine import construct
b,rows=construct(bank_size=8192,n_settings=96)
print("FULL_BANK_OK",len(b.pulses_ticks),len(rows),
      "condition",b.condition,
      "min_pi",min(s["n_pi"] for s in rows),
      "max_pi",max(s["n_pi"] for s in rows))

from servers.timing.sequencelibrary.QM_opx.camera.rqs_multisine import get_seq
base=[[[107.72,107.70]],[164],[1.0],[[72.44,73.22]],[88],[1.0],[False],[1]]
lim=max(x["n_pi"] for x in rows)
gaps=[[int(x["gaps_cc"][i]) for x in rows] for i in range(lim+1)]
counts=[int(x["n_pi"]) for x in rows]
basis=[0 if x["basis"]=="X" else 1 for x in rows]
even=[1 if x["n_pi"]%2==0 else 0 for x in rows]
seq,_=get_seq(base,gaps,counts,basis,even,2)
assert seq is not None
print("FULL_QAU_BUILT",len(rows),"max_pulses",lim)
