"""Offline checks of one-pulse phase-tracked Cornell Experiment 0.

Run from the repository root:
    python -m unittest tests.test_purcell_virtual_z_frame -v

No hardware or server calls are made.
"""
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from analysis import arbitrary_gate_tomography as analysis
from utils.single_qubit_gate_compiler import compile_gates, source_rotation
from utils.virtual_z_gate_compiler import compile_virtual_z_manifest, PHASE_GRID_DEG
from utils.virtual_z_frame_gate_compiler import compile_virtual_z_frame_manifest

CSV = (Path(__file__).resolve().parents[1] / "calibrations" /
       "purcell" / "arbitrary_gate_tomography" / "arbSQR.csv")


def _rxy(phase_deg, angle_rad):
    p = np.deg2rad(phase_deg)
    return Rotation.from_rotvec(angle_rad*np.array([np.cos(p), np.sin(p), 0]))


def _rz(phase_deg):
    return Rotation.from_rotvec([0., 0., np.deg2rad(phase_deg)])


class TestCornellVirtualZFrame(unittest.TestCase):
    def test_ten_gate_equivalence_and_fixed_preparation(self):
        for pi_ns in (88, 96):
            with self.subTest(pi_ns=pi_ns):
                base = compile_gates(CSV, pi_ns, 1)
                baseline_pulses = [[dict(p) for p in g["pulses"]]
                                   for g in base["gates"]]
                old = compile_virtual_z_manifest(base, pi_ns)
                frame = compile_virtual_z_frame_manifest(base, pi_ns)
                self.assertEqual(frame["experiment_mode"], "virtual_z_frame")
                self.assertEqual(len(frame["gates"]), 10)
                self.assertFalse(frame["hardware_warning"])
                self.assertEqual(baseline_pulses,
                                 [g["pulses"] for g in base["gates"]])
                self.assertEqual(frame["iq_phase_grid_deg"], PHASE_GRID_DEG)
                inputs = np.array([[0,0,1],[1,0,0],[0,1,0]])
                for o, f, b in zip(old["gates"], frame["gates"], base["gates"]):
                    self.assertEqual(f["preparation_phases_deg"],
                                     {"+Z":None, "+X":90, "+Y":180})
                    self.assertEqual(len(f["pulses"]), 1)
                    self.assertEqual(f["pulses"][0]["duration_ns"],
                                     o["pulses"][0]["duration_ns"])
                    self.assertEqual(
                        f["pulses"][0]["phase_deg"],
                        (o["pulses"][0]["phase_deg"] -
                         o["virtual_z_phase_compiled_deg"]) % 360
                    )
                    self.assertEqual(
                        f["projection_phase_deg"],
                        (-o["virtual_z_phase_compiled_deg"]) % 360
                    )
                    self.assertTrue(all(
                        (v % PHASE_GRID_DEG) == 0 for v in
                        [f["pulses"][0]["phase_deg"], f["projection_phase_deg"]]
                    ))
                    drive = _rxy(f["pulses"][0]["phase_deg"],
                                 f["pulses"][0]["actual_angle_rad"])
                    logical = _rz(f["frame_final_deg"]) * drive
                    target = source_rotation(b)
                    self.assertAlmostEqual(
                        (target.inv()*logical).magnitude(),
                        f["rotation_error_rad"], places=10
                    )
                    measured = [
                        _rxy(f["projection_phase_deg"], np.pi/2).apply(
                            drive.apply(v))[2] for v in inputs
                    ]
                    np.testing.assert_allclose(
                        measured, f["predicted_projected_z"], atol=1e-11
                    )
                    np.testing.assert_allclose(
                        f["predicted_projected_z"],
                        o["predicted_projected_z"], atol=1e-11
                    )
                    self.assertEqual(
                        (f["frame_before_gate_deg"] +
                         f["frame_after_gate_update_deg"]) % 360,
                        f["frame_final_deg"]
                    )

    def test_scc_analysis_mode_recognized(self):
        base = compile_gates(CSV, 96, 1)
        frame = compile_virtual_z_frame_manifest(base, 96)
        rng = np.random.default_rng(123)
        nvs, runs, reps = 4, 8, 200
        counts = np.empty((10,4,nvs,runs,1,reps),float)
        for i,g in enumerate(frame["gates"]):
            for j, z in enumerate(g["predicted_projected_z"]):
                counts[i,j,:, :,0,:] = rng.poisson(
                    40 + 10*(1-z), (nvs,runs,reps)
                )
            counts[i,3,:,:,0,::2] = rng.poisson(
                40, (nvs,runs,reps//2)
            )
            counts[i,3,:,:,0,1::2] = rng.poisson(
                60, (nvs,runs,reps//2)
            )
        result = analysis.analyze(
            {"counts": counts, "cornell_manifest": frame,
             "tomography_mode": "virtual_z_frame"},
            min_ref_contrast=2.5, min_ref_snr=3.
        )
        self.assertEqual(result["measurement_kind"], "projected_z_responses")
        self.assertEqual(result["measurement_components"], ["+Z","+X","+Y"])
        self.assertEqual(result["measured_projected_z"].shape, (10,nvs,3))
        self.assertLess(np.nanmedian(result["compiled_error_norm"]),0.15)

    def test_one_pulse_qua_program_is_constructible(self):
        from servers.timing.sequencelibrary.QM_opx.camera import (
            arbitrary_gate_tomography as qua_sequence
        )
        base = compile_gates(CSV, 96, 1)
        frame = compile_virtual_z_frame_manifest(base,96)
        args = [
            [[107.715,107.724],[107.438,105.983]],
            [132,132],[1.0,1.0],
            [[72.439,73.236],[72.175,71.822]],
            [40,88],[1.0,1.0],[False,False],[1],
        ]
        for gate in frame["gates"]:
            with self.subTest(gate=gate["gate"]):
                program, _ = qua_sequence.get_seq(args,[gate],num_reps=2)
                self.assertIsNotNone(program)

    def test_qua_rejects_inconsistent_projection_frame(self):
        from servers.timing.sequencelibrary.QM_opx.camera import (
            arbitrary_gate_tomography as qua_sequence
        )
        base = compile_gates(CSV,96,1)
        frame = compile_virtual_z_frame_manifest(base,96)
        g = dict(frame["gates"][0])
        g["projection_phase_deg"] = (g["projection_phase_deg"] + 1) % 360
        args = [
            [[107.715,107.724],[107.438,105.983]],
            [132,132],[1.,1.],
            [[72.439,73.236],[72.175,71.822]],
            [40,88],[1.,1.],[False,False],[1],
        ]
        with self.assertRaises(ValueError):
            qua_sequence.get_seq(args,[g],num_reps=2)


if __name__ == "__main__":
    unittest.main()
