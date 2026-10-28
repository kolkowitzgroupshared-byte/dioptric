"""Synthetic batch averaging checks for IQ-amplitude Rabi; no hardware use."""
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from analysis import sc_rabi_iq_amplitude_multifile_analysis as mf


def write_batch(folder,stem,*,names,seed,r0,r1,phase=0,pulse_ns=384):
    rng=np.random.default_rng(seed)
    x=np.linspace(0,0.5,24)
    N,R,P=len(names),5,80
    counts=np.empty((2,N,R,len(x),P),dtype=float)
    from analysis.sc_rabi_iq_amplitude_analysis import cosine_model
    for i,name in enumerate(names):
        curve=cosine_model(x,.035,0.75,0.25,0)+{
            "NV_A":.01,"NV_B":-.02,"NV_C":0}[name]
        for j,p1 in enumerate(curve):
            counts[0,i,:,j,:]=rng.poisson(r0+(r1-r0)*p1,(R,P))
    counts[1,:,:,:,::2]=rng.poisson(r0,(N,R,len(x),P//2))
    counts[1,:,:,:,1::2]=rng.poisson(r1,(N,R,len(x),P//2))
    raw=dict(
        counts="saved separately", nv_list=[{"name":n} for n in names],
        num_runs=R,num_reps=P,uwave_ind_list=[1],
        phase_deg=phase,pulse_duration_ns=pulse_ns,
        iq_amplitudes_v=x.tolist(), num_exps_per_step=2
    )
    (folder/(stem+".txt")).write_text(json.dumps(raw),encoding="utf8")
    np.savez_compressed(folder/(stem+".npz"),counts=counts)


class TestIQAmpBatch(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.root=Path(self.t.name)
        self.a="2026_10_27-18_08_16-fake"
        self.b="2026_10_27-18_40_00-fake"
        write_batch(self.root,self.a,names=["NV_A","NV_B","NV_C"],
                    seed=31,r0=35,r1=65)
        write_batch(self.root,self.b,names=["NV_C","NV_A","NV_B"],
                    seed=32,r0=40,r1=75)
    def tearDown(self):
        self.t.cleanup()

    def test_matching_normalization_and_sem(self):
        res=mf.combine([self.a,self.b],data_dir=self.root)
        self.assertEqual(res["nv_names"],["NV_A","NV_B","NV_C"])
        self.assertEqual(res["p1"].shape,(3,24))
        self.assertTrue(res["valid"].all())
        self.assertTrue(np.all(res["n_files_contributing"]==2))
        self.assertTrue(np.all(res["p1_sem"]>0))
        self.assertAlmostEqual(np.nanmean(res["p1"][:,0]),.04,delta=.08)

    def test_multipage_pdf(self):
        out=self.root/"out"
        summary=mf.analyze_many([self.a,self.b],data_dir=self.root,
                                outdir=out)
        self.assertEqual(summary["summary"]["num_batches"],2)
        pdf=summary["pdf"]
        self.assertTrue(pdf.is_file())
        data=pdf.read_bytes()
        self.assertTrue(data.startswith(b"%PDF"))
        self.assertEqual(len(re.findall(rb"/Type\s*/Page\b",data)),4)
        self.assertTrue(any(out.glob("*-per_nv_fits.csv")))
        self.assertTrue(any(out.glob("*-normalized_response.csv")))
        self.assertTrue(any(out.glob("*-summary.json")))

    def test_incompatible_rejected(self):
        write_batch(self.root,self.b,names=["NV_C","NV_A","NV_B"],
                    seed=32,r0=40,r1=75,pulse_ns=96)
        with self.assertRaisesRegex(ValueError,"duration differs"):
            mf.combine([self.a,self.b],data_dir=self.root)

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate file stem"):
            mf.combine([self.a,self.a],data_dir=self.root)


if __name__=="__main__":
    unittest.main()
