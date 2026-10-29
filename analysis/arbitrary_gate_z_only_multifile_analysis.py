# -*- coding: utf-8 -*-
"""Combine arbitrarily many Cornell virtual_z_z_only acquisition batches.

Edit FILE_STEMS and run:
    python -m analysis.arbitrary_gate_z_only_multifile_analysis

Or:
    python -m analysis.arbitrary_gate_z_only_multifile_analysis --files STEM1 STEM2 ...
    python -m analysis.arbitrary_gate_z_only_multifile_analysis --files STEM1 STEM2 --data-dir PATH --outdir PATH

All input files must be virtual_z_z_only with compatible ten gate pulses.
Each file is normalized using its OWN dual-rail SCC reference (per gate by
default) before shot-weighted averaging; raw counts are not concatenated.
NVs are matched by saved name, not position. Rejected reference points get
zero weight; plotted +/-1 SEM includes propagated count noise and, when larger,
the observed between-batch dispersion.

OUTPUT: one combined multipage PDF (one page per common NV, all ten gates),
summary/heatmap PNG, source/per-gate/per-NV CSVs, JSON, and compressed NPZ.
This is a direct R_zz population response, NOT a full gate fidelity metric.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import arbitrary_gate_z_only as z
from utils import data_manager as dm

# Add complete timestamp-NV stems, one entry per independent saved batch.
FILE_STEMS = [
    # "2026_10_27-16_51_38-johnson-nv0_2026_10_02",
    "2026_10_28-00_20_49-johnson-nv0_2026_10_02"
]

DATA_DIR = None          # None = normal Purcell G:/NAS lookup.
OUTDIR = None            # None = save to configured data_manager folder.
REFERENCE_SCOPE = "per_gate"
MIN_REFERENCE_CONTRAST = 0.0
MIN_REFERENCE_SNR = 0.0


def _file_stem(arg):
    p = Path(str(arg))
    if p.suffix.lower() in (".txt", ".npz"):
        p = p.with_suffix("")
    return p.name


def _resolve_input(arg, data_dir=None):
    """Load one TXT/NPZ pair using data_manager's real experiment directory."""
    src = Path(str(arg))
    if src.suffix.lower() in (".txt", ".npz"):
        src = src.with_suffix("")
    if data_dir is not None:
        folder = Path(data_dir)
        return z.load_run(src.name, data_dir=folder)
    if src.is_absolute() or src.parent != Path("."):
        return z.load_run(str(src))
    # Prefer the mounted G: path, with same NAS via UNC if unmapped.
    source = Path(__file__).resolve().parents[1] / (
        "majorroutines/widefield/arbitrary_gate_tomography.py"
    )
    if not src.name.startswith("20") or len(src.name) < 10:
        raise ValueError(f"Use complete timestamp-NV file stem: {src.name}")
    stamp = src.name[:10] + "-00_00_00"
    folder = dm.get_output_folder(source, time_stamp=stamp)
    folder = z.resolve_pdf_path(folder)
    return z.load_run(src.name, data_dir=folder)


def _name(nv):
    name = nv.get("name") if isinstance(nv, dict) else getattr(nv, "name", None)
    if not name:
        raise ValueError("NV has no stable name; cannot match across files")
    return str(name)


def _load_one(arg, data_dir, reference_scope, min_ref_contrast, min_ref_snr):
    raw, path = _resolve_input(arg, data_dir=data_dir)
    if raw.get("tomography_mode") != "virtual_z_z_only":
        raise ValueError(f"{arg}: not virtual_z_z_only mode")
    manifest = raw["cornell_manifest"]
    if manifest.get("readout_protocol") != "virtual_z_z_only":
        raise ValueError(f"{arg}: inconsistent Z-only protocol")
    names = [_name(n) for n in raw["nv_list"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{arg}: duplicate NV names")
    counts = np.asarray(raw["counts"])
    if counts.ndim != 6 or counts.shape[1] != 2 or counts.shape[2] != len(names):
        raise ValueError(f"{arg}: unexpected counts layout {counts.shape}")
    if counts.shape[3] != int(raw["num_runs"]) or counts.shape[5] != int(raw["num_reps"]):
        raise ValueError(f"{arg}: incomplete counts dimensions")
    if counts.shape[5] < 4 or counts.shape[5] % 2:
        raise ValueError(f"{arg}: requires even repetitions >= 4")
    for i, attempts in enumerate(raw.get("gate_crash_counter", [])):
        if len(attempts) != int(raw["num_runs"]) or any(v is None for v in attempts):
            raise ValueError(f"{arg}: incomplete gate {i+1} acquisition")
    result = z.analyze(
        raw, min_ref_contrast=min_ref_contrast, min_ref_snr=min_ref_snr,
        reference_scope=reference_scope,
    )
    return dict(
        stem=_file_stem(arg), path=path, manifest=manifest,
        result=result, names=names, index={name: i for i, name in enumerate(names)},
        shots=int(raw["num_runs"])*int(raw["num_reps"]),
        num_runs=int(raw["num_runs"]), num_reps=int(raw["num_reps"]),
        microwave_freq=raw.get("uwave_freq"),
        microwave_power_dbm=raw.get("uwave_power_dbm"),
    )


def _signature(manifest):
    """Compare exact microwave gate compilation, rather than only target cos(beta)."""
    return dict(
        csv_sha256=manifest.get("csv_sha256"),
        mw_chain=manifest.get("mw_chain"),
        pi_pulse_ns=manifest.get("pi_pulse_ns"),
        input_state_order=manifest.get("input_state_order"),
        gates=[{
            "gate": g["gate"],
            "pulses": g["pulses"],
            "expected_direct_z": g["expected_direct_z"],
            "predicted_direct_z": g["predicted_direct_z"],
        } for g in manifest["gates"]],
    )


def combine(file_stems=FILE_STEMS, *, data_dir=DATA_DIR,
            reference_scope=REFERENCE_SCOPE,
            min_ref_contrast=MIN_REFERENCE_CONTRAST,
            min_ref_snr=MIN_REFERENCE_SNR):
    stems = [str(s) for s in file_stems]
    if not stems:
        raise ValueError("FILE_STEMS empty: list saved Z-only files or use --files")
    if len({_file_stem(s) for s in stems}) != len(stems):
        raise ValueError("Repeated file stem; do not count the same batch twice")

    records = [
        _load_one(stem, data_dir, reference_scope, min_ref_contrast, min_ref_snr)
        for stem in stems
    ]
    first = records[0]
    sig = _signature(first["manifest"])
    for r in records[1:]:
        if _signature(r["manifest"]) != sig:
            raise ValueError(
                f"{r['stem']}: different Cornell gates, pulse compilation, "
                "microwave chain or calibration; analyze separately"
            )
        # New acquisitions save SG394 frequency/power for cross-batch checks.
        for key in ("microwave_freq", "microwave_power_dbm"):
            a, b = first[key], r[key]
            if (a is None) != (b is None):
                raise ValueError(
                    f"{r['stem']}: {key} missing from only one file; "
                    "cannot verify equal hardware settings"
                )
            if a is not None and not np.array_equal(np.asarray(a), np.asarray(b)):
                raise ValueError(f"{r['stem']}: {key} differs between batches")
    if first["microwave_freq"] is None or first["microwave_power_dbm"] is None:
        warnings.warn(
            "Microwave frequency/power absent from older acquisition metadata. "
            "Verify identical SG394 settings before interpreting batch average."
        )
    common = set(first["names"])
    for r in records[1:]:
        common &= set(r["names"])
    names = [n for n in first["names"] if n in common]
    if not names:
        raise ValueError("No common NV names across selected acquisitions")
    G = len(first["manifest"]["gates"])
    N = len(names)
    obs, sem, masks, shots = [], [], [], []
    for r in records:
        inds = [r["index"][n] for n in names]
        res = r["result"]
        obs.append(res["measured_z"][:, inds])
        sem.append(res["z_sem"][:, inds])
        masks.append(res["valid_ref"][:, inds])
        shots.append(r["shots"])
    obs = np.asarray(obs, dtype=float)          # [batch, gate, NV]
    sem = np.asarray(sem, dtype=float)
    masks = np.asarray(masks, dtype=bool)
    masks &= np.isfinite(obs) & np.isfinite(sem) & (sem >= 0)
    w = np.where(masks, np.asarray(shots, dtype=float)[:, None, None], 0.)
    sum_w = np.sum(w, axis=0)
    valid = sum_w > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.divide(
            np.sum(np.where(masks, obs, 0) * w, axis=0),
            sum_w, out=np.full((G, N), np.nan), where=valid,
        )
        within = np.divide(
            np.sqrt(np.sum(np.where(masks, sem, 0)**2 * w**2, axis=0)),
            sum_w, out=np.full((G,N), np.nan), where=valid,
        )
        sum_w2 = np.sum(w**2, axis=0)
        n_eff = np.divide(sum_w**2, sum_w2, out=np.zeros_like(sum_w),
                          where=sum_w2 > 0)
        denominator = sum_w - np.divide(sum_w2, sum_w,
                                       out=np.zeros_like(sum_w), where=valid)
        differences = np.where(masks, obs - mean[None, :, :], 0.)
        scatter = np.sum(w * differences**2, axis=0)
        between = np.sqrt(np.divide(
            scatter, denominator * n_eff, out=np.zeros_like(mean),
            where=(denominator*n_eff)>0,
        ))
        ste = np.where(valid, np.maximum(within, between), np.nan)

    ideal = np.array([g["expected_direct_z"] for g in first["manifest"]["gates"]])
    compiled = np.array([g["predicted_direct_z"] for g in first["manifest"]["gates"]])
    return {
        "measured_z": mean, "z_sem": ste,
        "expected_z": ideal, "compiled_z": compiled,
        "error": np.abs(mean - ideal[:, None]),
        "compiled_error": np.abs(mean - compiled[:, None]),
        "valid_ref": valid,
        "n_files_contributing": np.sum(masks, axis=0),
        "shots_contributing": sum_w,
        "nv_names": names, "num_nvs": N,
        "gate_labels": [str(g["gate"]) for g in first["manifest"]["gates"]],
        "reference_scope": reference_scope,
        "min_ref_contrast": min_ref_contrast,
        "min_ref_snr": min_ref_snr,
        "reference_valid_fraction": float(np.mean(valid)),
        "readout_protocol": "virtual_z_z_only",
        "observable": "R_zz = <sigma_z> after Cornell gate on optical +Z",
        "records": records,
        "num_files": len(records),
    }


def _save(result, outdir):
    from utils import data_manager as dm
    if outdir is None:
        from datetime import datetime
        timestamp = dm.get_time_stamp()
        base = dm.get_file_path(__file__, timestamp, "cornell_z_only_multifile")
        folder = z.resolve_pdf_path(base).parent
        basename = Path(base).stem
    else:
        folder = Path(outdir)
        basename = "cornell_z_only_multifile"
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / (basename + "-all_individual_nvs.pdf")
    z.save_individual_pdf(result, pdf)
    figs = z.plot_results(result)
    for fig, suffix in zip(figs, ("mean_response", "error_heatmap")):
        fig.savefig(folder/(basename+"-"+suffix+".png"),dpi=190)
        plt.close(fig)

    with (folder/(basename+"-source_files.csv")).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_stem","num_runs","num_reps","shots","nv_count",
                         "reference_valid_fraction"])
        for r in result["records"]:
            writer.writerow([
                r["stem"],r["num_runs"],r["num_reps"],r["shots"],
                len(r["names"]),r["result"]["reference_valid_fraction"]
            ])
    with (folder/(basename+"-per_nv_gate.csv")).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer=csv.writer(handle)
        writer.writerow(["nv_name","gate","measured_z","sem","ideal_z",
                         "compiled_z","absolute_error","valid",
                         "batches_contributing","shots_contributing"])
        for i,name in enumerate(result["nv_names"]):
            for k,gate in enumerate(result["gate_labels"]):
                writer.writerow([
                    name,gate,result["measured_z"][k,i],
                    result["z_sem"][k,i],result["expected_z"][k],
                    result["compiled_z"][k],result["error"][k,i],
                    bool(result["valid_ref"][k,i]),
                    int(result["n_files_contributing"][k,i]),
                    int(result["shots_contributing"][k,i]),
                ])
    with (folder/(basename+"-per_gate.csv")).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer=csv.writer(handle)
        writer.writerow(["gate","accepted_nvs","ideal_z","compiled_z",
                         "median_measured_z","median_absolute_error"])
        for k,gate in enumerate(result["gate_labels"]):
            valid = result["valid_ref"][k]
            writer.writerow([
                gate,int(valid.sum()),result["expected_z"][k],
                result["compiled_z"][k],
                float(np.median(result["measured_z"][k,valid])) if valid.any() else "",
                float(np.median(result["error"][k,valid])) if valid.any() else "",
            ])
    np.savez_compressed(
        folder/(basename+"-combined.npz"),
        nv_names=np.array(result["nv_names"], dtype=str),
        measured_z=result["measured_z"], ste=result["z_sem"],
        expected_z=result["expected_z"],compiled_z=result["compiled_z"],
        valid=result["valid_ref"],
        files_contributing=result["n_files_contributing"],
    )
    valid=result["valid_ref"]
    summary={
        "experiment":"Cornell Experiment 0: multi-batch virtual_z_z_only",
        "readout_protocol":"virtual_z_z_only",
        "num_input_files":result["num_files"],
        "microwave_freq":result["records"][0]["microwave_freq"],
        "microwave_power_dbm":result["records"][0]["microwave_power_dbm"],
        "file_stems":[r["stem"] for r in result["records"]],
        "matched_nvs":result["num_nvs"],
        "accepted_gate_nv_pairs":int(valid.sum()),
        "total_gate_nv_pairs":int(valid.size),
        "reference_scope":result["reference_scope"],
        "min_ref_contrast_photons":result["min_ref_contrast"],
        "min_ref_snr":result["min_ref_snr"],
        "combination":"Shot-weighted independently normalized means; SEM=max(propagated, between-batch dispersion)",
        "median_absolute_z_error":float(np.median(result["error"][valid])) if valid.any() else None,
        "pdf_path":str(pdf),
        "warning":"Z-only population measurement is not arbitrary gate fidelity; virtual-Z phases are invisible.",
    }
    (folder/(basename+"-summary.json")).write_text(
        json.dumps(summary,indent=2),encoding="utf-8"
    )
    print(f"Combined {result['num_files']} Z-only batches; {result['num_nvs']} common NVs")
    print(f"Accepted gate/NV pairs: {valid.sum()}/{valid.size}")
    print("Single multipage PDF:",pdf)
    print("Other outputs:",folder)
    return pdf, summary


def analyze_many(file_stems=FILE_STEMS, *, data_dir=DATA_DIR, outdir=OUTDIR,
                 reference_scope=REFERENCE_SCOPE,
                 min_ref_contrast=MIN_REFERENCE_CONTRAST,
                 min_ref_snr=MIN_REFERENCE_SNR):
    combined=combine(file_stems,data_dir=data_dir,
                     reference_scope=reference_scope,
                     min_ref_contrast=min_ref_contrast,min_ref_snr=min_ref_snr)
    pdf, summary=_save(combined,outdir)
    return {"combined":combined,"summary":summary,"pdf":pdf}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files",nargs="+",default=None,
                   help="Timestamp-NV stems of separate Z-only batches")
    p.add_argument("--data-dir",default=DATA_DIR)
    p.add_argument("--outdir",default=OUTDIR)
    p.add_argument("--reference-scope",choices=("per_gate","all_gates"),
                   default=REFERENCE_SCOPE)
    p.add_argument("--min-contrast",type=float,default=MIN_REFERENCE_CONTRAST)
    p.add_argument("--min-snr",type=float,default=MIN_REFERENCE_SNR)
    args=p.parse_args()
    return analyze_many(FILE_STEMS if args.files is None else args.files,
                        data_dir=args.data_dir,outdir=args.outdir,
                        reference_scope=args.reference_scope,
                        min_ref_contrast=args.min_contrast,
                        min_ref_snr=args.min_snr)


if __name__=="__main__":
    main()
