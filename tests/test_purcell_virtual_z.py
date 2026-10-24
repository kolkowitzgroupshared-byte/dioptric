"""Offline regression checks for Purcell virtual-Z Cornell experiment.

No QUA server or microwave hardware is required.
Run: python -m unittest tests.test_purcell_virtual_z -v
"""
import unittest
from pathlib import Path

import numpy as np

from analysis import arbitrary_gate_tomography as analysis
from utils.single_qubit_gate_compiler import compile_gates
from utils.virtual_z_gate_compiler import compile_virtual_z_manifest, PHASE_GRID_DEG

CSV = (Path(__file__).resolve().parents[1] / "calibrations" /
       "purcell" / "arbitrary_gate_tomography" / "arbSQR.csv")


class TestPurcellVirtualZ(unittest.TestCase):
    def test_all_ten_compiled_without_mutating_baseline(self):
        for pi_ns in (88, 96):
            with self.subTest(pi_ns=pi_ns):
                old = compile_gates(CSV, pi_ns, 1)
                old_pulses = [list(g["pulses"]) for g in old["gates"]]
                new = compile_virtual_z_manifest(old, pi_ns)
                self.assertEqual(len(new["gates"]), 10)
                self.assertLessEqual(new["max_rotation_error_rad"], 0.15)
                self.assertFalse(new["hardware_warning"])
                self.assertEqual(new["input_state_order"], ["+Z", "+X", "+Y"])
                self.assertEqual(old_pulses, [g["pulses"] for g in old["gates"]])
                for gate in new["gates"]:
                    self.assertLessEqual(len(gate["pulses"]), 1)
                    self.assertEqual(set(gate["preparation_phases_deg"]),
                                     {"+Z", "+X", "+Y"})
                    self.assertEqual(gate["preparation_phases_deg"]["+Z"], None)
                    self.assertTrue(all(
                        phase in range(0, 360, PHASE_GRID_DEG)
                        for phase in gate["preparation_phases_deg"].values()
                        if phase is not None
                    ))
                    self.assertTrue(all(p["phase_deg"] in range(0, 360, PHASE_GRID_DEG)
                                        for p in gate["pulses"]))
                    np.testing.assert_allclose(
                        gate["expected_projected_z"][0],
                        gate["expected_bloch_xyz"][1], atol=1e-11
                    )

    def test_reconstruct_all_three_projected_z_responses(self):
        baseline = compile_gates(CSV, 96, 1)
        manifest = compile_virtual_z_manifest(baseline, 96)
        rng = np.random.default_rng(12)
        N, R, P = 5, 8, 200
        counts = np.empty((10, 4, N, R, 1, P), dtype=np.float64)
        for i, gate in enumerate(manifest["gates"]):
            for j, expected in enumerate(gate["predicted_projected_z"]):
                counts[i, j, :, :, 0, :] = rng.poisson(
                    40 + 10 * (1 - expected), (N, R, P)
                )
            counts[i, 3, :, :, 0, ::2] = rng.poisson(
                40, (N, R, P // 2)
            )
            counts[i, 3, :, :, 0, 1::2] = rng.poisson(
                60, (N, R, P // 2)
            )
        result = analysis.analyze(
            {"counts": counts, "cornell_manifest": manifest,
             "tomography_mode": "virtual_z"},
            min_ref_contrast=2.5, min_ref_snr=3.0
        )
        self.assertEqual(result["measurement_kind"],
                         "projected_z_responses")
        self.assertEqual(result["measurement_components"],
                         ["+Z", "+X", "+Y"])
        self.assertEqual(result["measured_projected_z"].shape, (10, N, 3))
        self.assertLess(float(np.nanmedian(result["compiled_error_norm"])),
                        0.12)


if __name__ == "__main__":
    unittest.main()
