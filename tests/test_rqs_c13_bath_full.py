import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from majorroutines.widefield.rqs_c13_bath import build_protocol
from servers.timing.sequencelibrary.QM_opx.camera.rqs_c13_bath import get_seq
p=build_protocol(duration_us=60.,bank_size=8192,n_masks=64,
                 field_g=31.1,seed=37,pi_ns=96,pi2_ns=48)
steps=p["steps"]
assert len(steps)==67
assert p["bank_covariance_condition"]<20
m=p["max_pi"]
g=[[int(s["gaps_cc_padded"][i]) for s in steps] for i in range(m+1)]
n=[int(s["n_pi"]) for s in steps]
ph=[int(s["phase_mode"]) for s in steps]
pa=[1 if k%2==0 else 0 for k in n]
base=[[[107.72,107.70]],[164],[1.0],[[72.44,73.22]],[88],[1.0],[False],[1]]
seq,_=get_seq(base,g,n,ph,pa,2)
assert seq is not None
print("PASS: full 13C bath bank, real 4ns QUA construction",
      len(steps),"settings",
      "max pulses",m,
      "condition",p["bank_covariance_condition"],
      "C13 Larmor",p["c13_larmor_hz_expected"],"Hz")
