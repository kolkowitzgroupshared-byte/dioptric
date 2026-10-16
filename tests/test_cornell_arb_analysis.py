import unittest
from pathlib import Path
import numpy as np
from analysis.cornell_arb_sqr import analyze
from utils.cornell_arb_su2 import compile_gates

CSV=Path(__file__).resolve().parents[1]/"calibrations"/"cornell_arb_sqr"/"arbSQR.csv"

class TestExperimentAnalysis(unittest.TestCase):
    def test_recover_ideal_bloch_with_reference_parity(self):
        manifest=compile_gates(CSV,96,0)
        gates=manifest["gates"]
        ng=len(gates); n_nv=3; n_runs=3; n_reps=8
        raw_counts=np.empty((3*ng+1,n_nv,n_runs,1,n_reps))
        # ms0 1200 photons, ms1 800 photons; alternate the reference
        for r in range(n_runs):
            ms0=1200+5*r
            ms1=800+5*r
            for rep in range(n_reps):
                raw_counts[-1,:,r,0,rep]=ms0 if rep%2==0 else ms1
                for g,gate in enumerate(gates):
                    for b,axis in enumerate((2,0,1)):  # sequence ordering Z,X,Y
                        val=gate["expected_bloch_xyz"][axis]
                        raw_counts[3*g+b,:,r,0,rep]=(ms0+ms1+val*(ms0-ms1))/2
        out=analyze({"counts":raw_counts,"cornell_manifest":manifest})
        target=np.asarray([x["expected_bloch_xyz"] for x in gates])
        np.testing.assert_allclose(out["measured_xyz"],np.repeat(target[:,None,:],n_nv,axis=1),atol=1e-11)
        self.assertEqual(out["reference_valid_fraction"],1.0)

if __name__=="__main__": unittest.main()
