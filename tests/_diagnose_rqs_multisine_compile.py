"""Compile/try OPX QUA in the QM simulator ONLY. No hardware acquisition."""
import argparse,sys,traceback,logging
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qm import QuantumMachinesManager,SimulationConfig,CompilerOptionArguments
from qm.logger import logger
from utils import common
from majorroutines.widefield import rqs_multisine
from servers.timing.sequencelibrary.QM_opx.camera.rqs_multisine import get_seq

a=argparse.ArgumentParser()
a.add_argument("--settings",type=int,default=96)
a.add_argument("--nv-count",type=int,default=1)
a.add_argument("--reps",type=int,default=6)
a.add_argument("--flags",choices=["default","server"],default="server")
args=a.parse_args()
logger.setLevel(logging.INFO)
bank,rows=rqs_multisine.construct(duration_us=8.,n_settings=args.settings,
    bank_size=8192,seed=37,pi_ns=96,pi2_ns=48,coupling_hz=2800.)
largest=max(x["n_pi"] for x in rows)
gaps=[[int(r["gaps_cc"][k]) for r in rows] for k in range(largest+1)]
counts=[int(r["n_pi"]) for r in rows]
bases=[0 if r["basis"]=="X" else 1 for r in rows]
parity=[1 if r["n_pi"]%2==0 else 0 for r in rows]
nv=args.nv_count
# Same widefield SCC macro shape as physical acquisition, using harmless
# static coordinates. No polarizing laser is turned on by simulator.
base_args=[[[107.72,107.70] for _ in range(nv)],
           [164 for _ in range(nv)],[1.0 for _ in range(nv)],
           [[72.44,73.22] for _ in range(nv)],[88 for _ in range(nv)],
           [1.0 for _ in range(nv)],[False for _ in range(nv)],[1]]
print("MAKE_QUA settings=",len(rows),"nv_count=",nv,"max_pi=",largest,flush=True)
seq,_=get_seq(base_args,gaps,counts,bases,parity,num_reps=args.reps)
print("QUA_CONSTRUCTED",flush=True)
flags=CompilerOptionArguments(flags=["skip-loop-unrolling","skip-loop-rolling"]) if args.flags=="server" else None
cfg=common.get_config_dict()
print("SIMULATE_ONLY connecting to QOP simulator; no OPX output",flush=True)
qmm=QuantumMachinesManager(**cfg["DeviceIDs"]["QM_opx_args"])
try:
    job=qmm.simulate(common.get_opx_config_dict(),seq,
                     SimulationConfig(duration=100),compiler_options=flags)
    print("SUCCESS QM SIMULATION COMPILE",flush=True)
except Exception as e:
    print("SIMULATION_EXCEPTION",repr(e),flush=True)
    traceback.print_exc()
    sys.exit(1)
