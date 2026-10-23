# -*- coding: utf-8 -*-
"""Compare/average multiple Purcell Cornell arbitrary-gate acquisition files.

Edit FILE_STEMS, then run from the dioptric repo root:
    python -m analysis.arbitrary_gate_multifile_analysis

Or supply the source files on the command line:
    python -m analysis.arbitrary_gate_multifile_analysis --files STEM1 STEM2 ...

The script auto-groups 'virtual_z' and 'projection' runs, normalizes each
independently using its OWN dual-rail reference, then matches NVs by name.
It intentionally does not concatenate raw counts across different files.
All standard outputs use data_manager to save into the configured G: nvdata.
Use --data-dir and --outdir for an explicit testing/debugging override.

This is a projected-Z *response* comparison, NOT a gate-fidelity measurement.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # do not block acquisition or console workflow
import matplotlib.pyplot as plt
import numpy as np

from analysis import arbitrary_gate_tomography as arb

# ---------------------------------------------------------------------------
# USER SETTINGS: add as many complete timestamp-NV file stems as desired.
# Files may be interleaved; their saved tomography_mode determines grouping.
# ---------------------------------------------------------------------------
FILE_STEMS = [
    # "2026_10_21-20_36_54-johnson-nv0_2026_10_02",  # virtual_z
    "2026_10_21-22_42_30-johnson-nv0_2026_10_02",  # projection
    # "2026_10_22-...-johnson-nv0_2026_10_02",
]
DATA_DIR = None              # None -> normal dm/NAS lookup
OUTDIR = None                # None -> dm saves to G: (recommended)
# Same Purcell NAS share for sessions where Windows G: is not mapped.
NAS_UNC_ROOT = r"\\192.168.0.197\G"
REFERENCE_SCOPE = "all_gates"  # "per_gate" also supported
MIN_REFERENCE_CONTRAST = 2.5  # photons
MIN_REFERENCE_SNR = 3.0
SHOW_PLOTS = False
BOOTSTRAP_SAMPLES = 4000
SEED = 20261021

MODES = ("virtual_z", "projection")
MODE_LABEL = {"virtual_z": "Virtual-Z (one gate pulse)",
              "projection": "Three-pulse projection"}
INPUT_LABELS = ("+Z", "+X", "+Y")


def _name(nv):
    name = nv.get("name") if isinstance(nv, dict) else getattr(nv, "name", None)
    if not name:
        raise ValueError("An NV has no stable .name; cannot match files safely")
    return str(name)


def _stem(filename):
    p = Path(str(filename))
    return p.with_suffix("").name if p.suffix in (".txt", ".npz") else p.name


def _load_one(stem, data_dir, min_contrast, min_snr, reference_scope):
    if data_dir is None and not Path("G:/").is_dir():
        # In remote/background Purcell shells the Windows drive mapping and
        # data_manager search index may be unavailable. Resolve the NORMAL
        # experiment output folder, then access the same NAS via UNC.
        from utils import data_manager as dm
        source = Path(__file__).resolve().parents[1] / "majorroutines" / "widefield" / "arbitrary_gate_tomography.py"
        stamp = _stem(stem)[:10] + "-00_00_00"
        expected = dm.get_output_folder(source, time_stamp=stamp)
        if str(expected).upper().startswith("G:"):
            candidate = Path(str(Path(NAS_UNC_ROOT)) + str(expected)[2:])
            if not candidate.is_dir():
                raise RuntimeError(f"G: is unmapped; Purcell NAS input folder is unavailable: {candidate}")
            data_dir = candidate
            print("G: unmapped; reading", _stem(stem), "through NAS path:", data_dir)
    raw, _ = arb.load_run(stem, data_dir=data_dir)
    if not isinstance(raw.get("counts"), np.ndarray):
        raw["counts"] = np.asarray(raw["counts"], dtype=float)
    mode = raw.get("tomography_mode", raw.get("cornell_manifest", {}).get("experiment_mode"))
    if mode not in MODES:
        raise ValueError(
            f"{stem}: mode={mode!r}. This comparison expects {MODES}, "
            "not standard XYZ tomography or known-state calibration."
        )
    protocol = raw["cornell_manifest"].get("readout_protocol")
    expected = "virtual_z_projection" if mode == "virtual_z" else "fixed_projection"
    if protocol != expected:
        raise ValueError(f"{stem}: mode/protocol mismatch ({mode}/{protocol})")
    names = [_name(nv) for nv in raw["nv_list"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{stem}: duplicate NV names")
    counts = np.asarray(raw["counts"])
    if counts.ndim != 6 or counts.shape[1] != 4 or counts.shape[4] != 1:
        raise ValueError(f"{stem}: wrong counts layout {counts.shape}")
    if counts.shape[2] != len(names):
        raise ValueError(f"{stem}: NV/counts mismatch")
    if counts.shape[3] != int(raw["num_runs"]) or counts.shape[-1] != int(raw["num_reps"]):
        raise ValueError(f"{stem}: incomplete counts dimensions")
    if counts.shape[-1] < 4 or counts.shape[-1] % 2:
        raise ValueError(f"{stem}: reference needs even >=4 reps")
    for i, attempts in enumerate(raw.get("gate_crash_counter", [])):
        if len(attempts) != int(raw["num_runs"]) or any(a is None for a in attempts):
            raise ValueError(f"{stem}: incomplete gate {i+1} acquisition")
    res = arb.analyze(raw, min_ref_contrast=min_contrast, min_ref_snr=min_snr,
                      reference_scope=reference_scope)
    return {"stem": _stem(stem), "mode": mode, "raw": raw, "result": res,
            "names": names, "index": {name: i for i, name in enumerate(names)},
            "shots": int(raw["num_runs"]) * int(raw["num_reps"])}


def _align_and_combine(records, common_names):
    """Combine individually normalized runs with shot-count weights.

    Errors preserve both within-file propagated SEM and the between-file
    variation where there is replication. Individual reference pools are NEVER
    merged across files. Invalid gate/NV pairs have zero weight.
    """
    ideal = records[0]["result"]["expected_xyz"]
    gate_count = ideal.shape[0]
    num_nvs = len(common_names)
    response = np.full((gate_count, num_nvs, 3), np.nan)
    err_sem = np.full_like(response, np.nan)
    prediction = np.full((gate_count, 3), np.nan)
    valid = np.zeros((gate_count, num_nvs), dtype=bool)
    contributing = np.zeros((gate_count, num_nvs), dtype=int)
    observed_stack, sem_stack, valid_stack, weights = [], [], [], []
    for record in records:
        res = record["result"]
        index = [record["index"][x] for x in common_names]
        observed_stack.append(res["measured_xyz"][:, index, :])
        sem_stack.append(res["ste_xyz"][:, index, :])
        valid_stack.append(res["valid_ref"][:, index])
        weights.append(record["shots"])
    obs = np.asarray(observed_stack, dtype=float)  # [file,G,N,3]
    sem = np.asarray(sem_stack, dtype=float)
    mask = np.asarray(valid_stack, dtype=bool)
    shot_w = np.asarray(weights, dtype=float)[:, None, None]
    # Per observable require finite mean and SEM; preserve a common mask for
    # three responses in a gate/NV pair.
    mask &= np.all(np.isfinite(obs) & np.isfinite(sem), axis=-1)
    w = np.where(mask, shot_w, 0.0)
    ws = np.sum(w, axis=0)
    valid = ws > 0
    contributing = mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        response = np.divide(np.sum(np.where(mask[..., None], obs, 0.) * w[..., None], axis=0),
                             ws[..., None], out=np.full_like(response, np.nan),
                             where=ws[..., None] > 0)
        # Propagated SEM from each file's own reference and signal noise.
        within = np.divide(
            np.sqrt(np.sum(np.where(mask[..., None], sem, 0.)**2 * w[..., None]**2, axis=0)),
            ws[..., None], out=np.full_like(response, np.nan),
            where=ws[..., None] > 0)
        # Run-to-run dispersion of the response mean (used conservatively).
        n_eff = np.divide(ws**2, np.sum(w**2, axis=0),
                          out=np.zeros_like(ws), where=np.sum(w**2, axis=0) > 0)
        dif = np.where(mask[..., None], obs - response[None, ...], 0.)
        ss = np.sum(w[..., None] * dif**2, axis=0)
        denom = ws - np.divide(np.sum(w**2, axis=0), ws,
                              out=np.zeros_like(ws), where=ws > 0)
        between = np.sqrt(np.divide(ss, denom[..., None] * n_eff[..., None],
                                    out=np.zeros_like(response),
                                    where=(denom * n_eff)[..., None] > 0))
        err_sem = np.where(valid[..., None], np.maximum(within, between), np.nan)
    # Each saved manifest has its own quantized pulse predictions.
    pred = np.array([r["result"]["compiled_xyz"] for r in records])
    prediction = np.average(pred, axis=0, weights=weights)
    return {"response": response, "sem": err_sem, "valid": valid,
            "n_files_contributing": contributing, "compiled": prediction,
            "ideal": ideal,
            "error": np.linalg.norm(response - ideal[:, None, :], axis=-1),
            "names": list(common_names), "records": records}


def _bootstrap_difference(dif, samples=4000, seed=20261021):
    """Two-way bootstrap resampling gates and NVs (descriptive, not causal)."""
    if dif.size == 0 or samples < 100:
        return [None, None]
    rng = np.random.default_rng(seed)
    g, n = dif.shape
    gate_inds = rng.integers(0, g, size=(samples, g))
    nv_inds = rng.integers(0, n, size=(samples, n))
    vals = dif[gate_inds[:, :, None], nv_inds[:, None, :]].mean(axis=(1, 2))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def analyze_many(file_stems=FILE_STEMS, *, data_dir=DATA_DIR,
                 outdir=OUTDIR, reference_scope=REFERENCE_SCOPE,
                 min_ref_contrast=MIN_REFERENCE_CONTRAST,
                 min_ref_snr=MIN_REFERENCE_SNR,
                 show_plots=SHOW_PLOTS, bootstrap_samples=BOOTSTRAP_SAMPLES):
    """Analyze one or more files; aggregate compatible runs by mode."""
    file_stems = [_stem(s) for s in file_stems]
    if not file_stems:
        raise ValueError("FILE_STEMS is empty")
    if len(set(file_stems)) != len(file_stems):
        raise ValueError("Repeated file stem: each source file should appear only once")
    loaded = [
        _load_one(stem, data_dir, min_ref_contrast, min_ref_snr, reference_scope)
        for stem in file_stems
    ]
    # Guard against mixing different physical gates / experimental protocols.
    target = loaded[0]["result"]["expected_xyz"]
    gates = [g["gate"] for g in loaded[0]["raw"]["cornell_manifest"]["gates"]]
    for record in loaded[1:]:
        m = record["raw"]["cornell_manifest"]
        if ([g["gate"] for g in m["gates"]] != gates or
                record["result"]["expected_xyz"].shape != target.shape or
                not np.allclose(record["result"]["expected_xyz"], target, atol=1e-7)):
            raise ValueError(f"{record['stem']}: Cornell target gates differ")
        if m.get("projection_phase_deg") != 0:
            raise ValueError(f"{record['stem']}: expected fixed +X projection")
    common = set(loaded[0]["names"])
    for record in loaded[1:]:
        common &= set(record["names"])
    # Preserve first file's spatial order; do not align by index blindly.
    common_names = [name for name in loaded[0]["names"] if name in common]
    if not common_names:
        raise ValueError("No common NV names across the selected files")
    excluded = {r["stem"]: len(r["names"]) - len(common_names) for r in loaded}
    groups = {mode: _align_and_combine([r for r in loaded if r["mode"] == mode],
                                        common_names)
              for mode in MODES if any(r["mode"] == mode for r in loaded)}
    paired = None
    paired_all = None
    if len(groups) == 2:
        zv, pv = groups["virtual_z"], groups["projection"]
        paired = zv["valid"] & pv["valid"]
        paired_all = paired.all(axis=0)
        # Balanced comparison uses only NVs accepted for ALL ten gates.
        if not paired_all.any():
            warnings.warn("No NV passes all gates in both modes; only per-gate comparisons available")
    summaries = {}
    for mode, group in groups.items():
        valid = group["valid"]
        summaries[mode] = {
            "num_files": len(group["records"]),
            "file_stems": [r["stem"] for r in group["records"]],
            "valid_gate_nv_pairs": int(valid.sum()),
            "nvs_valid_all_gates": int(valid.all(axis=0).sum()),
            "median_error": float(np.nanmedian(group["error"][valid])) if valid.any() else None,
            "mean_error": float(np.nanmean(group["error"][valid])) if valid.any() else None,
            "out_of_range_fraction": float(np.mean(np.abs(group["response"][valid]) > 1)) if valid.any() else None,
        }
    comparison = None
    if paired_all is not None and paired_all.any():
        ev = groups["virtual_z"]["error"][:, paired_all]
        ep = groups["projection"]["error"][:, paired_all]
        dif = ev - ep
        comparison = {
            "matched_nvs_valid_all_gates": int(paired_all.sum()),
            "median_virtual_z_error": float(np.median(ev)),
            "median_projection_error": float(np.median(ep)),
            "mean_virtual_z_error": float(np.mean(ev)),
            "mean_projection_error": float(np.mean(ep)),
            "mean_difference_vz_minus_projection": float(np.mean(dif)),
            "two_way_bootstrap_95ci": _bootstrap_difference(dif, bootstrap_samples, SEED),
            "number_nvs_with_lower_gate_avg_vz_error": int(np.sum(dif.mean(axis=0) < 0)),
            "per_gate": [
                {"gate": g+1, "median_virtual_z": float(np.median(ev[g])),
                 "median_projection": float(np.median(ep[g])),
                 "mean_difference": float(np.mean(dif[g])),
                 "fraction_nvs_virtual_z_better": float(np.mean(dif[g] < 0))}
                for g in range(len(gates))
            ],
        }
    summary = {
        "experiment": "Cornell Experiment 0: multi-file projected-Z analysis",
        "warning": "Projected-Z response error, NOT a gate-fidelity estimate",
        "source_files": [{"file_stem": r["stem"], "mode": r["mode"],
                           "nvs": len(r["names"]), "num_runs": int(r["raw"]["num_runs"]),
                           "num_reps": int(r["raw"]["num_reps"]),
                           "reference_valid_fraction": float(r["result"]["reference_valid_fraction"])}
                          for r in loaded],
        "analysis": {"reference_scope": reference_scope,
                     "min_ref_contrast_photons": min_ref_contrast,
                     "min_ref_snr": min_ref_snr,
                     "all_file_common_nv_count": len(common_names),
                     "excluded_nvs_per_file": excluded,
                     "combination": "shot-count-weighted normalized responses; each file normalized separately"},
        "per_mode": summaries, "paired": comparison,
    }
    _save_output(groups, loaded, paired_all, comparison, summary, outdir,
                 show_plots=show_plots)
    print("\n=== CORNELL MULTI-FILE ANALYSIS ===")
    for mode, item in summaries.items():
        print(f"{mode:11s}: {item['num_files']} file(s), "
              f"{item['valid_gate_nv_pairs']} accepted gate/NV pairs; "
              f"median response error={item['median_error']}")
    if comparison:
        print("Common fully accepted NVs:", comparison["matched_nvs_valid_all_gates"])
        print("Paired medians: virtual_z={:.4f}, projection={:.4f}".format(
            comparison["median_virtual_z_error"], comparison["median_projection_error"]))
        print("Paired mean VZ minus projection={:.4f}; gate+NV bootstrap CI={}".format(
            comparison["mean_difference_vz_minus_projection"],
            comparison["two_way_bootstrap_95ci"]))
    return {"summary": summary, "groups": groups, "common_nv_names": common_names,
            "paired_nv_mask": paired_all}


def _save_output(groups, records, paired_all, comparison, summary, outdir, show_plots):
    """Save figures, CSV, summary and NPZ using dm or explicit output override."""
    from utils import data_manager as dm
    if outdir is None:
        timestamp = dm.get_time_stamp()
        base = dm.get_file_path(__file__, timestamp, "cornell_multifile_comparison")
        # Background/remote PowerShell shells on Purcell may not inherit G:.
        # This is the SAME NAS root, never an unannounced local C: fallback.
        if str(base).upper().startswith("G:") and not Path("G:/").is_dir():
            root = Path(NAS_UNC_ROOT)
            if not root.is_dir():
                raise RuntimeError(
                    "G: is unmapped and the configured Purcell NAS UNC root "
                    f"is unreachable: {root}. No files were saved."
                )
            base = Path(str(root) + str(base)[2:])
            print("G: not mapped in this session; using the SAME NAS through", root)
        print("Saving analysis via data_manager to:", base.parent)
        def fig_save(fig, suffix):
            path = base.with_name(base.stem + "-" + suffix + ".txt")
            dm.save_figure(fig, path)
        def text_save(content, suffix, ext):
            path = base.with_name(base.stem + "-" + suffix + ".txt")
            dm.save_text(content, path, suffix=ext)
        def npz_save(**arrays):
            dm.save_npz(base, compressed=True, **arrays)
    else:
        folder = Path(outdir)
        folder.mkdir(parents=True, exist_ok=True)
        print("Saving analysis to explicit output directory:", folder)
        def fig_save(fig, suffix):
            fig.savefig(folder / (suffix + ".png"), dpi=190)
        def text_save(content, suffix, ext):
            (folder / (suffix + ext)).write_text(content, encoding="utf-8")
        def npz_save(**arrays):
            np.savez_compressed(folder / "combined_results.npz", **arrays)
    text_save(json.dumps(summary, indent=2), "summary", ".json")
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["file_stem", "mode", "nv_count", "run_count", "repetitions",
                "valid_ref_fraction", "median_response_error", "median_ref_contrast_photons"])
    for r in records:
        valid = r["result"]["valid_ref"]
        w.writerow([r["stem"], r["mode"], len(r["names"]), r["raw"]["num_runs"],
                    r["raw"]["num_reps"], float(valid.mean()),
                    float(np.nanmedian(r["result"]["bloch_error_norm"][valid])) if valid.any() else "",
                    float(np.nanmedian(r["result"]["reference_contrast"][valid])) if valid.any() else ""])
    text_save(buf.getvalue(), "source_files", ".csv")
    gate_csv = io.StringIO(newline="")
    gw = csv.writer(gate_csv)
    gw.writerow(["gate", "mode", "nv_count", "median_error", "mean_error",
                 "compiled_so3_error_rad", "source_files"])
    for mode, group in groups.items():
        for k in range(group["ideal"].shape[0]):
            v = group["valid"][k]
            gw.writerow([k+1, mode, int(v.sum()),
                         float(np.median(group["error"][k, v])) if v.any() else "",
                         float(np.mean(group["error"][k, v])) if v.any() else "",
                         ";".join(str(round(float(r["raw"]["cornell_manifest"]["gates"][k]["rotation_error_rad"]), 5)) for r in group["records"]),
                         ";".join(r["stem"] for r in group["records"])])
    text_save(gate_csv.getvalue(), "per_gate", ".csv")
    nv_csv = io.StringIO(newline="")
    nw = csv.writer(nv_csv)
    nw.writerow(["mode", "nv_index_common", "nv_name", "gate", "initial_state",
                 "measured_projected_z", "ste", "ideal", "compiled", "valid", "files_contributing"])
    for mode, group in groups.items():
        for i, name in enumerate(group["names"]):
            for k in range(group["ideal"].shape[0]):
                for j, state in enumerate(INPUT_LABELS):
                    nw.writerow([mode, i, name, k+1, state,
                                 group["response"][k, i, j], group["sem"][k, i, j],
                                 group["ideal"][k, j], group["compiled"][k, j],
                                 bool(group["valid"][k, i]), int(group["n_files_contributing"][k, i])])
    text_save(nv_csv.getvalue(), "per_nv_gate_state", ".csv")
    arrays = {"nv_names": np.asarray(next(iter(groups.values()))["names"], dtype=str),
              "ideal_projected_z": next(iter(groups.values()))["ideal"]}
    for mode, group in groups.items():
        arrays[f"{mode}_projected_z"] = group["response"]
        arrays[f"{mode}_projected_z_ste"] = group["sem"]
        arrays[f"{mode}_valid"] = group["valid"]
        arrays[f"{mode}_error_norm"] = group["error"]
        arrays[f"{mode}_files_contributing"] = group["n_files_contributing"]
    if paired_all is not None:
        arrays["paired_all_gates_nv_mask"] = paired_all
    npz_save(**arrays)
    figures = _make_figures(groups, records, paired_all, comparison)
    for name, fig in figures.items():
        fig_save(fig, name)
        if not show_plots:
            plt.close(fig)
    if show_plots:
        plt.show(block=False)
    print("Saved", len(figures), "figures, 3 CSV tables, JSON summary and compressed NPZ")


def _make_figures(groups, records, paired_all, comparison):
    figs = {}
    ng = next(iter(groups.values()))["ideal"].shape[0]
    x = np.arange(1, ng+1)
    # A: per-gate paired error across identical NVs if available.
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    width = 0.36 if len(groups) > 1 else 0.6
    for mi, (mode, group) in enumerate(groups.items()):
        y = []
        for i in range(ng):
            mask = (paired_all if paired_all is not None and np.any(paired_all)
                    else group["valid"][i])
            y.append(float(np.nanmedian(group["error"][i, mask])) if np.any(mask) else np.nan)
        dx = (mi - (len(groups)-1)/2) * width
        ax.bar(x+dx, y, width=width, label=MODE_LABEL[mode])
    ax.set(xlabel="Cornell gate", ylabel="Median 3-response L2 error",
           title="Cornell arbitrary rotations | normalized SCC response error")
    ax.set_xticks(x); ax.grid(axis="y", alpha=.2); ax.legend()
    figs["error_by_gate"] = fig
    # B: all three initial-state response curves.
    fig, axes = plt.subplots(3, 1, figsize=(10.7, 10.0), sharex=True,
                             layout="constrained")
    ideal = next(iter(groups.values()))["ideal"]
    for j, state in enumerate(INPUT_LABELS):
        ax = axes[j]
        ax.plot(x, ideal[:, j], "k--o", label="Ideal")
        for mode, group in groups.items():
            mask = paired_all if paired_all is not None and np.any(paired_all) else group["valid"].all(axis=0)
            if not np.any(mask):
                continue
            ax.plot(x, np.nanmedian(group["response"][:, mask, j], axis=1),
                    "-o", label=MODE_LABEL[mode])
        ax.set_ylabel(f"Final <Z> | input {state}")
        ax.set_ylim(-1.7, 1.7); ax.axhline(0, color="grey", linewidth=.6)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    axes[-1].set(xlabel="Cornell gate")
    axes[-1].set_xticks(x)
    fig.suptitle("Projected-Z responses after fixed +X pi/2")
    figs["projected_z_three_inputs"] = fig
    # C: gate vs spatially matched NV error heatmaps.
    fig, axes = plt.subplots(1, len(groups), figsize=(6.3*len(groups), 6),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (mode, group) in zip(axes, groups.items()):
        im = ax.imshow(np.where(group["valid"], group["error"], np.nan).T,
                       origin="lower", aspect="auto", vmin=0, vmax=2,
                       interpolation="nearest")
        ax.set_title(MODE_LABEL[mode]); ax.set_xlabel("Gate index")
        ax.set_xticks(np.arange(ng), labels=[str(v) for v in x])
        ax.set_ylabel("Common NV index")
    fig.colorbar(im, ax=axes.tolist(), label="3-response L2 error", shrink=.8)
    figs["gate_nv_error_heatmaps"] = fig
    # D: reference contrast of each individual input file (never merged first).
    fig, ax = plt.subplots(figsize=(10.5, 5.3), layout="constrained")
    for rec in records:
        r = rec["result"]
        contrast = r["raw_gate_reference_1"] - r["raw_gate_reference_0"]
        ax.plot(x, np.nanmedian(contrast, axis=1), "-o", linewidth=1.5,
                label=f"{rec['mode']} | {rec['stem'][:19]}")
    ax.set(xlabel="Gate acquisition order", ylabel="Median R1-R0 (photons)",
           title="Per-file SCC reference contrast: investigate drift")
    ax.set_xticks(x); ax.grid(alpha=.2); ax.legend(fontsize=8)
    figs["reference_contrast_sources"] = fig
    # E: distribution of per-NV gate-mean matched error difference.
    if paired_all is not None and np.any(paired_all):
        dif = (groups["virtual_z"]["error"][:, paired_all] -
               groups["projection"]["error"][:, paired_all])
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        xval = dif.mean(axis=0)
        ax.hist(xval, bins=25)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(np.mean(xval), linestyle=":", linewidth=2, label="Mean difference")
        ax.set(xlabel="Mean over gates: virtual-Z error minus projection error",
               ylabel="NV count", title="Matched-NV comparison (negative favors virtual-Z)")
        ax.legend()
        figs["paired_nv_difference"] = fig
    return figs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", default=None,
                        help="Saved file stems; overrides FILE_STEMS at the top of this script")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Optional explicit directory of raw .txt/.npz files")
    parser.add_argument("--outdir", default=OUTDIR,
                        help="Optional debug output directory; None saves using data_manager to G:")
    parser.add_argument("--reference-scope", default=REFERENCE_SCOPE,
                        choices=("all_gates", "per_gate"))
    parser.add_argument("--min-contrast", type=float, default=MIN_REFERENCE_CONTRAST)
    parser.add_argument("--min-snr", type=float, default=MIN_REFERENCE_SNR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--show", action="store_true", help="Nonblocking plot display")
    args = parser.parse_args()
    analyze_many(FILE_STEMS if args.files is None else args.files,
                 data_dir=args.data_dir, outdir=args.outdir,
                 reference_scope=args.reference_scope,
                 min_ref_contrast=args.min_contrast, min_ref_snr=args.min_snr,
                 bootstrap_samples=args.bootstrap, show_plots=args.show)


if __name__ == "__main__":
    main()
