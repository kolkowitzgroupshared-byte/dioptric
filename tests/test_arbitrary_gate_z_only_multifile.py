"""Offline synthetic two-file batch averaging for Cornell Z-only experiments."""
from __future__ import annotations

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import arbitrary_gate_z_only_multifile_analysis as mf
from majorroutines.widefield import arbitrary_gate_tomography as routine


def _write_batch(folder, stem, manifest, *, seed, order, runs=6, reps=100,
                 r0=35, r1=65):
    rng=np.random.default_rng(seed)
    gates=manifest["gates"]
    G=len(gates);N=len(order)
    counts=np.empty((G,2,N,runs,1,reps),dtype=float)
    ideal=np.array([g["predicted_direct_z"] for g in gates])
    for k in range(G):
        for i,name in enumerate(order):
            shift = {"NV0":0.0,"NV1":0.01,"NV2":-0.02}[name]
            p1=(1-ideal[k])/2+shift
            counts[k,0,i,:,0,:]=rng.poisson(r0+(r1-r0)*p1,(runs,reps))
    counts[:,1,:,:,0,::2]=rng.poisson(r0,(G,N,runs,reps//2))
    counts[:,1,:,:,0,1::2]=rng.poisson(r1,(G,N,runs,reps//2))
    obj={
        "experiment":"Cornell Experiment 0: virtual-Z +Z preparation, direct Z SCC",
        "timestamp":stem[:19],"tomography_mode":"virtual_z_z_only",
        "readout_protocol":"virtual_z_z_only",
        "nv_list":[{"name":n} for n in order],
        "num_runs":runs,"num_reps":reps,
        "cornell_manifest":manifest,
    }
    (folder/(stem+".txt")).write_text(json.dumps(obj),encoding="utf8")
    np.savez_compressed(folder/(stem+".npz"),counts=counts)


class TestZOnlyBatchAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,_=routine.prepare(uwave_ind=1,mode="virtual_z_z_only")

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.names=["NV0","NV1","NV2"]
        self.stem1="2026_10_27-16_12_00-synthetic"
        self.stem2="2026_10_27-16_20_00-synthetic"
        _write_batch(self.folder,self.stem1,self.manifest,seed=91,
                     order=["NV0","NV1","NV2"],r0=35,r1=65)
        # Change background/reference and NV order; independent normalization
        # and matching by name must still recover the correct response.
        _write_batch(self.folder,self.stem2,self.manifest,seed=92,
                     order=["NV2","NV0","NV1"],r0=43,r1=74)

    def tearDown(self):
        self.temp.cleanup()

    def test_combine_two_batches_name_alignment(self):
        one=mf.combine([self.stem1],data_dir=self.folder)
        two=mf.combine([self.stem1,self.stem2],data_dir=self.folder)
        self.assertEqual(two["num_files"],2)
        self.assertEqual(two["nv_names"],self.names)
        self.assertEqual(two["measured_z"].shape,(10,3))
        self.assertTrue(np.isfinite(two["z_sem"]).all())
        self.assertTrue((two["z_sem"]>0).all())
        self.assertTrue(two["valid_ref"].all())
        self.assertTrue(np.all(two["n_files_contributing"]==2))
        self.assertLess(float(np.median(two["compiled_error"])),.18)
        self.assertLess(float(np.max(np.abs(two["measured_z"]-one["measured_z"]))),.25)

    def test_one_pdf_and_csv_after_batch_average(self):
        result=mf.analyze_many([self.stem1,self.stem2],
                               data_dir=self.folder,
                               outdir=self.folder/"out")
        pdf=result["pdf"]
        self.assertTrue(pdf.is_file())
        binary=pdf.read_bytes()
        self.assertTrue(binary.startswith(b"%PDF-"))
        self.assertEqual(len(re.findall(rb"/Type\s*/Page\b",binary)),3)
        outputs=list((self.folder/"out").iterdir())
        self.assertTrue(any(f.name.endswith("-per_nv_gate.csv") for f in outputs))
        self.assertTrue(any(f.name.endswith("-source_files.csv") for f in outputs))
        self.assertTrue(any(f.name.endswith("-summary.json") for f in outputs))
        self.assertEqual(result["summary"]["num_input_files"],2)
        self.assertEqual(result["summary"]["matched_nvs"],3)

    def test_incompatible_compilation_rejected(self):
        manifest=copy.deepcopy(self.manifest)
        manifest["gates"][0]["pulses"][0]["phase_deg"]=(manifest["gates"][0]["pulses"][0]["phase_deg"]+1)%360
        _write_batch(self.folder,self.stem2,manifest,seed=92,
                     order=["NV2","NV0","NV1"],r0=43,r1=74)
        with self.assertRaisesRegex(ValueError,"different Cornell gates"):
            mf.combine([self.stem1,self.stem2],data_dir=self.folder)

    def test_duplicate_and_wrong_protocol_rejected(self):
        with self.assertRaisesRegex(ValueError,"Repeated file stem"):
            mf.combine([self.stem1,self.stem1],data_dir=self.folder)
        path=self.folder/(self.stem2+".txt")
        obj=json.loads(path.read_text(encoding="utf8"))
        obj["tomography_mode"]="virtual_z"
        path.write_text(json.dumps(obj),encoding="utf8")
        with self.assertRaisesRegex(ValueError,"not virtual_z_z_only"):
            mf.combine([self.stem1,self.stem2],data_dir=self.folder)


if __name__=="__main__":
    unittest.main()
