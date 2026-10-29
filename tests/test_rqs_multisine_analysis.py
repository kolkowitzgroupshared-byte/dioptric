import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.rqs_multisine_core import compile_bank,choose_masks,estimate
from analysis.sc_rqs_multisine_analysis import analyze
import tempfile

def test():
    bank=compile_bank(np.linspace(234375,1415625,8),
                      pool_size=512,seed=37)
    rows=choose_masks(bank,n_settings=160,seed=43)
    truth=np.zeros(16)
    truth[3]=.25
    truth[10]=-.12
    idx=np.asarray([r["pool_index"] for r in rows])
    half=len(bank.responses)
    v=bank.responses[idx%half]*np.where(idx<half,1.,-1.)[:,None]
    phase=v@truth
    obs=np.asarray([.73*np.cos(p) if r["basis"]=="X"
                    else .73*r["outcome_sign"]*np.sin(p)
                    for p,r in zip(phase,rows)])
    result=estimate(bank,rows,obs)
    print("Recovered",result.get("amplitudes"),"fit status",result["status"])
    assert result["status"]=="ok"
    assert abs(result["amplitudes"][3]-.25)<.12
    with tempfile.TemporaryDirectory() as d:
        report=analyze(bank,rows,["nv0","nv1"],np.tile(obs,(2,1)),
                       np.full((2,len(rows)),6.),{"fixed_T_us":8.},
                       outdir=d,readout_calibrated=True)
        assert report["fit_count"]==2
        assert (Path(d)/"01_amplitude_map.pdf").exists()
    print("PASS coherent-field simulation and PDFs")
if __name__=="__main__":
    test()
