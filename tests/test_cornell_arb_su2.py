import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from utils.cornell_arb_su2 import load_cornell_csv,compile_gates

CSV=Path(__file__).resolve().parents[1]/"calibrations"/"cornell_arb_sqr"/"arbSQR.csv"

class TestCornellArb(unittest.TestCase):
    def test_source_all_10_gates_match(self):
        self.assertEqual(len(load_cornell_csv(CSV)),10)

    def test_purcell_compiler(self):
        for pi_ns in (48,96,128,192):
            m=compile_gates(CSV,pi_ns,0)
            self.assertEqual(len(m["gates"]),10)
            for gate in m["gates"]:
                self.assertLess(gate["rotation_error_rad"],0.25)
                self.assertLessEqual(len(gate["pulses"]),3)
                self.assertTrue(all(p["phase_deg"] in (0,90,180,270) for p in gate["pulses"]))
                self.assertTrue(all(p["duration_cc"]>=4 for p in gate["pulses"]))

if __name__=="__main__": unittest.main()
