"""Offline code-generation and mathematical checks for rqs_c13.

NO OPX session is opened, no device initialized, no microwave command issued.
Run from repo root: python -m tests.test_rqs_c13_purcell
If tests/ has no __init__.py, use: python tests/test_rqs_c13_purcell.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from majorroutines.widefield.rqs_c13 import build_protocol
from analysis.rqs_c13_analysis import filter_y2


def test_schedules():
    for band in ("low", "high"):
        p = build_protocol(band=band, masks_per_duration=3)
        assert len(p["steps"]) > 15
        for s in p["steps"]:
            waits = s["waits_cc"]
            assert len(waits) == 9 and all(isinstance(w, int) and w >= 8 for w in waits)
            assert len(s["pi_centers_ns"]) == 8
            assert 0 < s["pi_centers_ns"][0] < s["pi_centers_ns"][-1] < s["T_ns"]
            assert abs(s["T_ns"] - s["requested_T_ns"]) < 5
            assert 4 * sum(waits) + 8 * p["pi_ns"] + p["pi2_ns"] == s["T_ns"]
        p2 = build_protocol(band=band, masks_per_duration=3)
        assert p["protocol_sha256"] == p2["protocol_sha256"]
    print("PASS schedule/timing/determinism")


def test_xy8_frequency():
    p = build_protocol(masks_per_duration=2)
    s = next(s for s in p["steps"] if s["kind"] == "xy8")
    target = s["target_f_hz"]
    freq = np.linspace(0.65*target, 1.35*target, 1001)
    power = filter_y2(s, freq)
    peak = freq[np.argmax(power)]
    assert abs(peak-target)/target < .06, (peak, target)
    print("PASS XY8 nominal center frequency, relative difference:", (peak-target)/target)


def test_qua_builder_without_qmm():
    from servers.timing.sequencelibrary.QM_opx.camera.rqs_c13 import get_seq
    p = build_protocol(masks_per_duration=10)
    wait_tables = [[p["steps"][i]["waits_cc"][j]
                    for i in range(len(p["steps"]))] for j in range(9)]
    # Synthetic NV coordinates, only used for in-memory QUA program construction.
    base = [
        [[107.72, 107.70]], [164], [1.0],
        [[72.44, 73.22]], [88], [1.0], [False], [0]
    ]
    seq, _ = get_seq(base, wait_tables, 2)
    assert seq is not None
    print("PASS in-memory QUA program construction, NO HARDWARE")


def test_synthetic_analysis():
    """Exercise full save-to-figure analysis with clearly synthetic counts."""
    from analysis.rqs_c13_analysis import analyze, band_overlaps
    from tempfile import TemporaryDirectory
    p = build_protocol(masks_per_duration=3)
    overlap = band_overlaps(p, p["bare_c13_larmor_hz_expected"])/1e-9
    t = np.array([s["T_ns"] for s in p["steps"]])*1e-9
    c = np.exp(-0.07 - 300*t - 0.03*overlap)
    nsteps = len(p["steps"])
    data = np.empty((2, 2, 2, nsteps, 4), float)
    data[0, :, :, :, :] = (150 - 50*c)[None,None,:,None]
    data[1, :, :, :, ::2] = 200.
    data[1, :, :, :, 1::2] = 100.
    raw = {"counts": data, "rqs_complete": True,
           "rqs_c13_protocol": p}
    with TemporaryDirectory() as tmp:
        fit = analyze(raw, tmp)
        assert (Path(tmp)/"rqs_c13_diagnostic.png").exists()
        assert (Path(tmp)/"rqs_c13_per_nv.csv").exists()
        assert np.isfinite(fit[0][0]) and fit[0][0] > 0
    # Exercise the *real* dm.save_* methods, but point dm.get_file_path at
    # a disposable local folder so a test NEVER writes to G:/nvdata.
    import json
    from unittest.mock import patch
    from utils import data_manager as dm
    with TemporaryDirectory() as tmp:
        fake_base = Path(tmp) / "2026_10_22-12_00_00-rqs_c13_analysis.txt"
        with patch.object(dm, "get_file_path", return_value=fake_base) as path_mock:
            analyze(raw, source_stem="synthetic-raw-dataset")
        assert path_mock.call_args.kwargs["subfolder"] == "synthetic-raw-dataset"
        for suffix in (".png", ".csv", ".json", ".npz"):
            assert fake_base.with_suffix(suffix).is_file(), suffix
        summary = json.loads(fake_base.with_suffix(".json").read_text())
        assert summary["source_experiment_stem"] == "synthetic-raw-dataset"
        assert summary["number_of_nvs"] == 2
        with np.load(fake_base.with_suffix(".npz")) as saved:
            assert saved["normalized_coherence"].shape == (2, nsteps)
    print("PASS synthetic SCC analysis and dm-backed save (NO G-drive writes)")


if __name__ == "__main__":
    test_schedules()
    test_xy8_frequency()
    test_qua_builder_without_qmm()
    test_synthetic_analysis()
