"""Offline checks for fixed-duration IQ-amplitude Rabi (no real acquisition)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from majorroutines.widefield.rabi_iq_amplitude import make_sweep
from analysis.sc_rabi_iq_amplitude_analysis import (
    process, fit_trace, cosine_model, save_analysis
)
from servers.timing.sequencelibrary.QM_opx.camera.rabi_iq_amplitude import (
    get_seq, validate_amplitudes
)


class TestIQAmpRabi(unittest.TestCase):
    def test_amplitude_range(self):
        np.testing.assert_allclose(make_sweep(41),np.linspace(0.,0.5,41))
        self.assertEqual(validate_amplitudes([0.0,0.25,0.5]),
                         [0.,0.25,0.5])
        for bad in ([-0.001,0.1],[0.25,0.501],[float("nan"),0.25]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_amplitudes(bad)

    def test_qua_builds_all_steps_both_phases(self):
        args = [
            [[107.715,107.724],[107.438,105.983]],
            [132,132],[1.,1.],
            [[72.439,73.236],[72.175,71.822]],
            [40,88],[1.,1.],[False,False],[1],
        ]
        for phase in (0,90):
            with self.subTest(phase=phase):
                seq,_ = get_seq(args,make_sweep(41),384,phase_deg=phase,
                                num_reps=2)
                self.assertIsNotNone(seq)
        with self.assertRaises(ValueError):
            get_seq(args,[0.,0.5],382,num_reps=2)
        with self.assertRaises(ValueError):
            get_seq(args,[0.,0.5],384,num_reps=3)

    def test_reference_and_fit_from_synthetic_counts(self):
        rng=np.random.default_rng(987)
        amps=make_sweep(41)
        nv,runs,reps=4,8,200
        counts=np.empty((2,nv,runs,len(amps),reps),dtype=np.float64)
        ideal_p1=cosine_model(amps,0.02,0.83,0.25,0.0)
        for i in range(nv):
            for j,pr in enumerate(ideal_p1):
                counts[0,i,:,j,:]=rng.poisson(30+20*pr,(runs,reps))
        counts[1,:,:,:,::2]=rng.poisson(
            30,(nv,runs,len(amps),reps//2))
        counts[1,:,:,:,1::2]=rng.poisson(
            50,(nv,runs,len(amps),reps//2))
        raw={"counts":counts,"iq_amplitudes_v":amps,
             "pulse_duration_ns":384,"phase_deg":0}
        result=process(raw)
        self.assertEqual(result["p1"].shape,(nv,len(amps)))
        self.assertTrue(result["valid"].all())
        self.assertLess(np.nanmean(np.abs(result["p1"]-ideal_p1)),0.05)
        fit=fit_trace(amps,np.nanmean(result["p1"],axis=0),
                      np.full(len(amps),0.02))
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["period_v"],0.25,delta=0.035)
        self.assertAlmostEqual(fit["pi_amplitude_v"],0.125,delta=0.035)
        with TemporaryDirectory() as td:
            saved,_=save_analysis(result,Path(td)/"synthetic",outdir=Path(td)/"out")
            for name in ("mean_rabi_amplitude.png","per_nv_heatmap.png",
                         "per_nv_fit.csv","normalized_response.csv","summary.json"):
                self.assertTrue((saved/name).exists(),name)

if __name__ == "__main__":
    unittest.main()
