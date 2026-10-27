"""Z-only Cornell experiment offline tests: no acquisition is performed."""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from utils.single_qubit_gate_compiler import compile_gates
from utils.virtual_z_gate_compiler import compile_virtual_z_manifest
from majorroutines.widefield import arbitrary_gate_tomography as routine
from analysis import arbitrary_gate_z_only as analysis

CSV = (Path(__file__).resolve().parents[1] /
       "calibrations/purcell/arbitrary_gate_tomography/arbSQR.csv")


class TestCornellZOnly(unittest.TestCase):

    def test_manifest_and_reference_target(self):
        # Mode compilation does not overwrite the prior virtual_z manifest.
        manifest,path = routine.prepare(uwave_ind=1,mode="virtual_z_z_only")
        baseline=compile_gates(CSV,manifest["pi_pulse_ns"],1)
        old=compile_virtual_z_manifest(baseline,manifest["pi_pulse_ns"])
        self.assertTrue(path.name.endswith("_virtual_z_z_only.json"))
        self.assertEqual(manifest["readout_protocol"],"virtual_z_z_only")
        self.assertEqual(manifest["input_state_order"],["+Z"])
        self.assertEqual(manifest["num_exps_per_gate"],2)
        self.assertEqual(len(manifest["gates"]),10)
        for gate,orig in zip(manifest["gates"],old["gates"]):
            self.assertEqual(gate["pulses"],orig["pulses"])
            self.assertEqual(gate["preparation_phases_deg"],
                             orig["preparation_phases_deg"])
            self.assertAlmostEqual(gate["expected_direct_z"],
                                   orig["expected_projected_z"][0])
            self.assertAlmostEqual(gate["predicted_direct_z"],
                                   orig["predicted_projected_z"][0])

    def test_qua_constructs_two_experiments_no_pi_over_2(self):
        from servers.timing.sequencelibrary.QM_opx.camera import (
            arbitrary_gate_tomography as qua_sequence
        )
        manifest,_=routine.prepare(uwave_ind=1,mode="virtual_z_z_only")
        args = [
            [[107.715,107.724],[107.438,105.983]],
            [132,132],[1.,1.],
            [[72.439,73.236],[72.175,71.822]],
            [40,88],[1.,1.],[False,False],[1],
        ]
        for gate in manifest["gates"]:
            with self.subTest(gate=gate["gate"]):
                with patch.object(qua_sequence.seq_utils,"macro_pi_on_2_pulse",
                                  side_effect=AssertionError("Unexpected pi/2")):
                    program,_=qua_sequence.get_seq(args,[gate],num_reps=2)
                self.assertIsNotNone(program)

    def test_acquisition_shape_guard(self):
        nvs=[{"name":"NV0"},{"name":"NV1"}]
        raw={"counts":np.zeros((2,2,3,1,4)),
             "crash_counter":[0,0,0], "nv_list":nvs}
        routine._check_gate_raw(raw,0,3,num_exps=2)
        with self.assertRaises(RuntimeError):
            routine._check_gate_raw(raw,0,3,num_exps=4)

    def _fake_data(self):
        rng=np.random.default_rng(987)
        manifest,_=routine.prepare(uwave_ind=1,mode="virtual_z_z_only")
        G,N,R,P=len(manifest["gates"]),3,6,100
        counts=np.empty((G,2,N,R,1,P),dtype=np.float64)
        ideal=np.array([g["predicted_direct_z"] for g in manifest["gates"]])
        for i in range(G):
            for j in range(N):
                counts[i,0,j,:,0,:]=rng.poisson(
                    42 + 10*(1-ideal[i]),(R,P))
        counts[:,1,:,:,0,::2]=rng.poisson(42,(G,N,R,P//2))
        counts[:,1,:,:,0,1::2]=rng.poisson(62,(G,N,R,P//2))
        return {"nv_list":[{"name":f"NV{i}"} for i in range(N)],
                "counts":counts,"tomography_mode":"virtual_z_z_only",
                "cornell_manifest":manifest}

    def test_scc_normalization_errors_and_one_pdf(self):
        raw=self._fake_data()
        res=analysis.analyze(raw,reference_scope="per_gate")
        self.assertEqual(res["measured_z"].shape,(10,3))
        self.assertEqual(res["z_sem"].shape,(10,3))
        self.assertTrue(res["valid_ref"].all())
        self.assertLess(np.nanmedian(res["compiled_error"]),.12)
        self.assertTrue(np.all(res["z_sem"]>0))
        pooled=analysis.analyze(raw,reference_scope="all_gates")
        self.assertEqual(pooled["measured_z"].shape,(10,3))
        with tempfile.TemporaryDirectory() as td:
            filename=Path(td)/"all_NVs_z_only.pdf"
            analysis.save_individual_pdf(res,filename)
            self.assertTrue(filename.exists())
            data=filename.read_bytes()
            self.assertTrue(data.startswith(b"%PDF"))
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b",data)),3)
            a,b=analysis.plot_results(res)
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            import matplotlib.pyplot as plt
            plt.close(a);plt.close(b)

    def test_rejects_old_three_input_data(self):
        raw=self._fake_data()
        raw["counts"]=raw["counts"][:,:1]
        with self.assertRaises(ValueError):
            analysis.analyze(raw)
        raw=self._fake_data()
        raw["cornell_manifest"]["readout_protocol"]="virtual_z_mixed_projection"
        with self.assertRaises(ValueError):
            analysis.analyze(raw)


if __name__ == "__main__":
    unittest.main()
