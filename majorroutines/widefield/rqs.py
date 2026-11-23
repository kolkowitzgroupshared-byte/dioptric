# -*- coding: utf-8 -*-
"""
Widefield randomized quantum sensing (RQS)

Plays the RQS acquisition plan (OPX timing files from rqs_opx_export_with_pi2) on all
NVs at once and saves the raw data, optionally with a phase-locked test field from OPX
analog output 1 (the tones argument of main). write_averages_files turns one or more
saved runs into the per-NV averages files that step 7 of the RQS notebook reads.

Adapted from spin_echo.py on November 3rd, 2026; test-field tones added on November 5th, 2026

@author: STERNFELD
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield as widefield


def get_timing_dir(bundle_dir):
    bundle_dir = Path(bundle_dir)
    return bundle_dir.parent / (bundle_dir.name + "_opx_timing")


def build_steps(acquisition, rows, phase_cycled):
    """One step per planned shot of each row, for each analysis phase.

    step = 2 * j + phase, where j indexes `rows` and phase 1 is the analysis pi/2
    shifted by 180 deg.
    """
    phases = (0, 1) if phase_cycled else (0,)
    steps = []
    for j, row in enumerate(rows):
        for _ in range(int(acquisition["shots"].iloc[row])):
            steps.extend(2 * j + phase for phase in phases)
    return steps


def import_rqs_toolkit(bundle_dir):
    """randomized_sensing from the toolkit that holds the bundle (<toolkit>/outputs/<bundle>)."""
    src = Path(bundle_dir).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import randomized_sensing

    return randomized_sensing


def write_averages_files(data_list, nv_inds=None, ms0_is_nv_minus=False):
    """Write the per-NV averages files that step 7 of the RQS notebook reads.

    data_list: raw data from main, or a list of them (for example one per block of
        rows); together they must cover every acquisition row, in both phases if
        phase cycled.
    nv_inds: NVs of nv_list to write, in order; nv_list[nv_inds[k]] becomes
        measured_averages_nv{k}[_plus|_minus].csv in the bundle folder.
    ms0_is_nv_minus: True if m_s = 0 reads as NV-. The +1 outcome is the m_s = 0
        outcome; calibration A confirms the choice (the visibility must be positive).
    """
    if isinstance(data_list, dict):
        data_list = [data_list]
    if not all(data.get("phase_locked", True) for data in data_list):
        raise ValueError(
            "Free-running data can't be averaged per row: each shot sees its own signal phase, "
            "so it needs per-shot outcomes with each shot's time"
        )
    bundle_dir = Path(data_list[0]["bundle_dir"])
    R = import_rqs_toolkit(bundle_dir)
    plan = R.load_bundle(bundle_dir)
    phase_cycled = data_list[0]["phase_cycled"]
    phases = ("plus", "minus") if phase_cycled else ("",)
    num_nvs = len(data_list[0]["nv_list"])
    nv_inds = list(range(num_nvs)) if nv_inds is None else list(nv_inds)
    row_index = {r["row_id"]: ind for ind, r in enumerate(plan.rows)}

    sums = np.zeros((len(nv_inds), len(phases), len(plan.rows)))
    shots = np.zeros((len(phases), len(plan.rows)))
    for data in data_list:
        if data["plan_id"] != plan.plan_id or data["phase_cycled"] != phase_cycled:
            raise ValueError("Every file must come from this bundle's plan, with the same phase cycling")
        if "states" not in data:
            raise ValueError("No 'states' in the raw data: threshold the counts per NV first")
        states = np.asarray(data["states"])[0]  # signal experiment: [nv, run, step, rep]
        outcomes = np.where(states.astype(bool) == ms0_is_nv_minus, 1, -1)
        per_step = outcomes[nv_inds].sum(axis=(1, 3))  # [nv, step]
        steps = np.asarray(data["steps"])
        plan_rows = np.array([row_index[row_id] for row_id in data["row_ids"]])[steps >> 1]
        for p in range(len(phases)):
            sel = (steps & 1) == p
            np.add.at(sums[:, p], (slice(None), plan_rows[sel]), per_step[:, sel])
            np.add.at(shots[p], plan_rows[sel], outcomes.shape[1] * outcomes.shape[3])
    if np.any(shots == 0):
        missing = int(np.sum(shots == 0))
        raise ValueError(f"{missing} row/phase combinations have no shots yet: run the remaining rows")

    files = []
    is_x = np.array([r["basis"] == "X" for r in plan.rows])
    print("file   nv_list index   X contrast   readout offset")
    for k, nv_ind in enumerate(nv_inds):
        for p, phase in enumerate(phases):
            suffix = f"_{phase}" if phase else ""
            path = bundle_dir / f"measured_averages_nv{k}{suffix}.csv"
            R.write_averages(path, plan, sums[k, p] / shots[p], repetitions=shots[p].astype(int))
            files.append(path)
        x_means = [sums[k, p, is_x].sum() / shots[p, is_x].sum() for p in range(len(phases))]
        if phase_cycled:
            contrast, offset = (x_means[0] - x_means[1]) / 2, f"{(x_means[0] + x_means[1]) / 2:+.3f}"
        else:
            contrast, offset = x_means[0], "not separated"
        print(f"nv{k:<4d} {nv_ind:13d}   {contrast:+10.3f}   {offset:>14s}")
    print(f"Wrote {len(files)} files in {bundle_dir}")
    return files


def main(nv_list, num_reps, num_runs, bundle_dir, rows=None, phase_cycled=True, tones=None, phase_locked=True):
    """Play the RQS plan and save the raw data.

    tones: test field on OPX analog output 1, as a list of (frequency_hz, amp_v, phase_deg);
        None or [] for no test field (calibration B, or a real external signal). amp_v is the
        amplitude into 50 ohm at the OPX output, before the attenuators. phase_deg 0 gives a
        pure cosine, 90 a pure sine.
    phase_locked: True restarts the tones at the same phase every shot (phase-locked plans).
        False lets them run freely on the OPX clock, so each shot catches them at a different
        phase, and the OPX records each shot's time (free-running plans).
    """
    ### Some initial setup

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "rqs.py"

    uwave_ind_list = [1]  # iq modulated: needed for the +Y preparation and the analysis phases
    tones = [[float(freq), float(amp), float(phase)] for freq, amp, phase in (tones or [])]
    phase_locked = bool(phase_locked)
    if not phase_locked:
        print(
            "Free-running: the OPX records each shot's time in its result stream "
            "'rqs_shot_times', but base_routine does not save that stream yet, so this "
            "run cannot be decoded until it does."
        )

    timing_dir = get_timing_dir(bundle_dir)
    acquisition = pd.read_csv(timing_dir / "opx_ready_acquisition.csv")
    if rows is None:
        rows = range(len(acquisition))
    rows = [int(row) for row in rows]
    steps = build_steps(acquisition, rows, phase_cycled)
    num_steps = len(steps)

    ### Collect the data
    def run_fn(shuffled_step_inds):
        shuffled_steps = [steps[ind] for ind in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            shuffled_steps,
            str(timing_dir),
            rows,
            tones,
            phase_locked,
        ]
        # print(seq_args)
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,
    )

    ### Process and plot

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "bundle_dir": str(bundle_dir),
        "plan_id": str(acquisition["plan_id"].iloc[0]),
        "row_ids": [str(acquisition["row_id"].iloc[row]) for row in rows],
        "steps": steps,
        "phase_cycled": phase_cycled,
        "tones": tones,
        "phase_locked": phase_locked,
    }

    # save data
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(raw_data, file_path)

    ### Clean up and return
    tb.reset_cfm()
    kpl.show()
    return raw_data


if __name__ == "__main__":
    from pathlib import Path

    kpl.init_kplotlib()

    DATA_FOLDER = Path(r"G:\nvdata\pc_Purcell\branch_master\rqs\2026_11")
    BUNDLE_NAME = "rqs_16k7_phase_locked"
    AMP_V = 0.0  # 0.0 for the 0 V run, 0.04 for the tone run
    FIRST_GOOD = "2026_11_18-15_39_13-johnson-nv0_2026_10_02" #"2026_11_18-12_13_47-johnson-nv0_2026_10_02"  # tone first block saved with IQ mode on

    file_stems = []
    for path in sorted(DATA_FOLDER.glob("*.txt")):
        if path.stem < FIRST_GOOD:
            continue
        info = dm.get_raw_data(file_stem=path.stem, load_npz=False)
        amp = info["tones"][0][1] if info.get("tones") else 0.0
        if (str(info.get("bundle_dir", "")).endswith(BUNDLE_NAME)
                and len(info.get("row_ids", [])) > 20 and abs(amp - AMP_V) < 1e-9):
            file_stems.append(path.stem)
    print(f"{len(file_stems)} block files at {AMP_V} V (expect 9):", *file_stems, sep="\n  ")

    def otsu_threshold(values):
        """Count level that best splits the values into two groups (Otsu's method)."""
        hist, edges = np.histogram(values, bins=128)
        centers = 0.5 * (edges[1:] + edges[:-1])
        w0 = np.cumsum(hist)
        w1 = w0[-1] - w0
        s0 = np.cumsum(hist * centers)
        m0 = s0 / np.maximum(w0, 1)
        m1 = (s0[-1] - s0) / np.maximum(w1, 1)
        between = w0 * w1 * (m0 - m1) ** 2
        return float(edges[1 + int(np.argmax(between[:-1]))])

    print("loading the block files...")
    data_list = [dm.get_raw_data(file_stem=stem, load_npz=True) for stem in file_stems]

    # Charge state per shot: each NV's threshold from all of its counts in these blocks
    counts = [np.asarray(d["counts"], dtype=np.float64) for d in data_list]  # [exp, nv, run, step, rep]
    num_nvs = counts[0].shape[1]
    thresholds = []
    for nv in range(num_nvs):
        values = np.concatenate([c[:, nv].ravel() for c in counts])
        thresholds.append(otsu_threshold(values[np.isfinite(values)]))
    thresholds = np.array(thresholds)
    bad = 0
    for d, c in zip(data_list, counts):
        bad += int(np.sum(~np.isfinite(c)))
        c = np.where(np.isfinite(c), c, thresholds[None, :, None, None, None] - 1)  # invalid -> dark (rare)
        d["states"] = (c > thresholds[None, :, None, None, None]).astype(np.int8)  # 1 = bright = NV-
    print(f"thresholds: median {np.median(thresholds):.1f} counts; {bad} invalid count values set dark")

    nv_inds = [i for i in range(num_nvs) if i not in (0, 6)]
        # Diagnostic: the spin signal in the raw counts, before any thresholding (NV average)
    acq_basis = pd.read_csv(Path(data_list[0]["bundle_dir"]) / "acquisition.csv").set_index("row_id")["basis"]
    sums = {}
    for d in data_list:
        c = np.asarray(d["counts"], dtype=np.float64)[0][nv_inds]   # signal experiment: [nv, run, step, rep]
        c = np.where(np.isfinite(c), c, np.nan)
        per_step = np.nanmean(c, axis=(0, 1, 3))                    # NV-average counts per step
        steps = np.asarray(d["steps"])
        bases = acq_basis.loc[np.asarray(d["row_ids"])[steps >> 1]].to_numpy()
        for b in ("X", "Y"):
            for p, name in ((0, "plus"), (1, "minus")):
                sel = (bases == b) & ((steps & 1) == p)
                s, n = sums.get((b, name), (0.0, 0))
                sums[(b, name)] = (s + per_step[sel].sum(), n + int(sel.sum()))
    for key, (s, n) in sorted(sums.items()):
        print(f"{key[0]} rows, {key[1]:5s}: mean counts {s / n:7.3f} over {n} steps")
        

    # Reference shots (the second experiment of every step), for comparison
    ref_means = []
    for d in data_list:
        c = np.asarray(d["counts"], dtype=np.float64)[1][nv_inds]   # reference experiment: [nv, run, step, rep]
        ref_means.append(np.nanmean(np.where(np.isfinite(c), c, np.nan)))
    print(f"reference shots: mean counts {np.mean(ref_means):7.3f}")     
    # print("writing the averages files...")
    # write_averages_files(data_list, nv_inds=nv_inds, ms0_is_nv_minus=False)
    # print("done")