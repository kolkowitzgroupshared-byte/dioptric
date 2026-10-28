# -*- coding: utf-8 -*-
"""Cornell virtual-Z: optical +Z input and direct SCC Z readout only.

Raw layout [gate, signal/reference, nv, run, one_step, repetition].
Even reference repetitions prepare ms=0, odd repetitions prepare ms=1.
No preparatory microwave pi/2 or final analysis pi/2 appears in the signal.

The measured value is R_zz = cos(beta), *not* a full SU(2) gate fidelity:
virtual-Z phases are invisible to +Z -> Z-only population measurements.

Command-line:
    python -m analysis.arbitrary_gate_z_only --file-stem TIMESTAMP-NV
    python -m analysis.arbitrary_gate_z_only --file-stem TIMESTAMP-NV --reference-scope all_gates
    python -m analysis.arbitrary_gate_z_only --file-stem TIMESTAMP-NV --data-dir G:\\...\\2026_10
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from analysis import arbitrary_gate_tomography as arb

REFERENCE_SCOPE = "per_gate"
MIN_REFERENCE_CONTRAST = 2.5
MIN_REFERENCE_SNR = 3.0
NAS_UNC_ROOT = r"\\192.168.0.197\G"


def _stats(array, axes):
    """Mean and SEM of scalar SCC count samples (float64 prevents overflow)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.sum(np.isfinite(array), axis=axes)
        mean = np.nanmean(array, axis=axes)
        sem = np.nanstd(array, axis=axes, ddof=1) / np.sqrt(
            np.maximum(n, 1)
        )
    return mean, np.where(n > 1, sem, np.nan)


def analyze(raw_data, min_ref_contrast=MIN_REFERENCE_CONTRAST,
            min_ref_snr=MIN_REFERENCE_SNR, reference_scope=REFERENCE_SCOPE):
    """Compute <Z>, propagated one-SEM uncertainties, and per-NV gate errors."""
    manifest = raw_data["cornell_manifest"]
    if manifest.get("readout_protocol") != "virtual_z_z_only":
        raise ValueError("Expected a virtual_z_z_only manifest; won't mix protocols")
    if raw_data.get("tomography_mode") not in (None, "virtual_z_z_only"):
        raise ValueError("Raw tomography mode does not match the Z-only manifest")
    if reference_scope not in ("per_gate", "all_gates"):
        raise ValueError("reference_scope must be 'per_gate' or 'all_gates'")
    gates = manifest["gates"]
    a = np.asarray(raw_data["counts"], dtype=np.float64)
    if a.ndim != 6 or a.shape[0] != len(gates) or a.shape[1] != 2 or a.shape[4] != 1:
        raise ValueError(
            f"Z-only count layout must be (gate,2,nv,run,1,rep); got {a.shape}"
        )
    G, _, N, R, _, P = a.shape
    if P < 2 or P % 2:
        raise ValueError("Even num_reps >=2 required for dual-rail references")
    if R < 1:
        raise ValueError("Expected one or more runs")
    sig = a[:, 0, :, :, 0, :]  # G,N,R,P
    ref = a[:, 1, :, :, 0, :]
    r0_samples = ref[..., 0::2]
    r1_samples = ref[..., 1::2]
    S, s_sem = _stats(sig, (2,3))
    if reference_scope == "per_gate":
        R0, r0_sem = _stats(r0_samples, (2,3))
        R1, r1_sem = _stats(r1_samples, (2,3))
    else:
        r0, e0 = _stats(r0_samples, (0,2,3))
        r1, e1 = _stats(r1_samples, (0,2,3))
        R0, R1 = np.broadcast_to(r0, (G,N)), np.broadcast_to(r1, (G,N))
        r0_sem, r1_sem = np.broadcast_to(e0, (G,N)), np.broadcast_to(e1,(G,N))
    D = R1 - R0
    Dsem = np.hypot(r0_sem, r1_sem)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = D/Dsem
    valid = (np.isfinite(D) & (D >= min_ref_contrast)
             & np.isfinite(snr) & (snr >= min_ref_snr))
    denominator = np.where(valid, D, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = 1 - 2 * (S-R0) / denominator
        # Derivatives dZ/dS, dZ/dR0 and dZ/dR1.
        z_var = (2*s_sem/denominator)**2 + (
            2*(R1-S)*r0_sem/denominator**2
        )**2 + (
            2*(S-R0)*r1_sem/denominator**2
        )**2
        z_sem = np.sqrt(z_var)

    ideal = np.asarray(
        [g["expected_direct_z"] for g in gates], dtype=float
    )
    compiled = np.asarray(
        [g["predicted_direct_z"] for g in gates], dtype=float
    )
    if not np.allclose(
        ideal, [g["expected_bloch_xyz"][2] for g in gates], atol=1e-12
    ):
        raise ValueError("Z-only manifest has inconsistent ideal targets")
    if not np.allclose(
        compiled, [g["predicted_actual_bloch"][2] for g in gates], atol=1e-12
    ):
        raise ValueError("Z-only manifest has inconsistent compiled targets")
    # Keep signal masked for rejected reference pairs; never clip to [-1,1].
    error = np.abs(z - ideal[:,None])
    compiled_error = np.abs(z - compiled[:,None])
    rawR0, rawR0sem = _stats(r0_samples, (2,3))
    rawR1, rawR1sem = _stats(r1_samples, (2,3))
    return {
        "measured_z": z, "z_sem": z_sem, "expected_z": ideal,
        "compiled_z": compiled,
        "error": error, "compiled_error": compiled_error,
        "valid_ref": valid, "reference_scope": reference_scope,
        "reference_0": R0, "reference_1": R1,
        "reference_contrast": D, "reference_snr": snr,
        "raw_gate_reference_0": rawR0, "raw_gate_reference_1": rawR1,
        "raw_gate_reference_ste": np.hypot(rawR0sem, rawR1sem),
        "reference_valid_fraction": float(np.mean(valid)),
        "num_nvs": N, "num_runs": R, "num_reps": P,
        "gate_labels": [str(g["gate"]) for g in gates],
        "nv_names": [
            (n["name"] if isinstance(n, dict) else n.name)
            for n in raw_data["nv_list"]
        ],
        "observable": "R_zz = <sigma_z> after Cornell gate on optical +Z",
        "readout_protocol": "virtual_z_z_only",
        "min_ref_contrast": min_ref_contrast,
        "min_ref_snr": min_ref_snr,
    }


def plot_results(result):
    """Summary of direct-Z targets/data and spatial NV residuals."""
    x = np.arange(1, len(result["expected_z"])+1)
    z = np.where(result["valid_ref"], result["measured_z"], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(z, axis=1)
        qlow = np.nanpercentile(z, 25, axis=1)
        qhi = np.nanpercentile(z, 75, axis=1)
    fig, ax = plt.subplots(figsize=(10,5.4),layout="constrained")
    ax.plot(x,result["expected_z"],"k--o",label="Ideal Z")
    ax.plot(x,result["compiled_z"],":s",label="Compiled Z")
    ax.errorbar(x,med,yerr=[np.maximum(0,med-qlow),np.maximum(0,qhi-med)],
                fmt="o",capsize=3,label="Median NV response (25-75% range)")
    ax.set(xlabel="Cornell gate",ylabel=r"Direct $\langle\sigma_z\rangle$",
           title="Cornell +Z input / direct-Z SCC readout")
    ax.set_xticks(x);ax.grid(alpha=.2);ax.legend()

    fig2,ax2 = plt.subplots(figsize=(10,5.5),layout="constrained")
    image=np.where(result["valid_ref"], result["error"],np.nan)
    im=ax2.imshow(image.T,origin="lower",aspect="auto",
                  interpolation="nearest",vmin=0,vmax=2)
    ax2.set(xlabel="Cornell gate",ylabel="NV index",
            title="Z-only absolute population-response error (invalid references masked)")
    ax2.set_xticks(np.arange(len(x)),labels=result["gate_labels"])
    fig2.colorbar(im,ax=ax2,label=r"$|Z_{measured} - Z_{ideal}|$")
    return fig,fig2


def resolve_pdf_path(path):
    """Keep output on configured G: drive, via equivalent NAS UNC if unmapped."""
    p=Path(path)
    if str(p).upper().startswith("G:") and not Path("G:/").is_dir():
        p=Path(NAS_UNC_ROOT + str(p)[2:])
    return p


def save_individual_pdf(result, pdf_path):
    """Single multipage vector PDF: each NV gets ideal, compiled, 1-SEM bars."""
    path=resolve_pdf_path(pdf_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    z=result["measured_z"]
    sem=result["z_sem"]
    valid=result["valid_ref"]
    x=np.arange(1,len(result["expected_z"])+1)
    with PdfPages(path,metadata={
        "Title":"Cornell Z-only direct-SCC response, all individual NVs",
        "Subject":"Optical +Z preparation, direct Z readout and 1 SEM error bars"
    }) as pdf:
        for i,name in enumerate(result["nv_names"]):
            fig,ax=plt.subplots(figsize=(9.0,5.8),layout="constrained")
            ax.plot(x,result["expected_z"],"k--o",lw=1.4,
                    label="Cornell ideal",zorder=2)
            ax.plot(x,result["compiled_z"],":s",lw=1.3,
                    label="Compiled IQ pulse",zorder=2)
            keep=valid[:,i] & np.isfinite(z[:,i]) & np.isfinite(sem[:,i])
            if keep.any():
                ax.errorbar(x[keep],z[keep,i],yerr=sem[keep,i],
                            fmt="o",capsize=3,ms=5,lw=1.2,
                            label="Measured direct Z (1 SEM)",zorder=4)
            rejected=~keep
            if rejected.any():
                ax.scatter(x[rejected],np.full(np.sum(rejected),-1.35),
                           c="tab:red",marker="x",label="Rejected SCC reference")
            ax.axhline(1,lw=.6,color="gray")
            ax.axhline(0,lw=.6,color="gray")
            ax.axhline(-1,lw=.6,color="gray")
            ymax=1.6
            if keep.any():
                ymax=max(ymax,float(np.nanmax(np.abs(z[keep,i])+sem[keep,i]))+.25)
            ax.set_ylim(-min(5,ymax),min(5,ymax))
            ax.set_xticks(x)
            ax.set(xlabel="Cornell gate",ylabel=r"Direct $\langle\sigma_z\rangle$",
                   title=(f"Z-only Cornell experiment | NV {i+1}/{len(result['nv_names'])}: {name}"
                          + (f" | averaged {result['num_files']} batches"
                             if result.get("num_files", 1) > 1 else "")))
            ax.grid(alpha=.18)
            ax.legend(loc="best",fontsize=9)
            errors=result["error"][:,i][keep]
            median=(float(np.median(errors)) if errors.size else float("nan"))
            fig.text(.12,.035,
                     f"Accepted: {np.sum(keep)}/{len(x)} gates | "
                     f"Median |Z-Zideal|: {median:.3f} | "
                     f"Reference: {result['reference_scope']}, "
                     f"SNR >= {result['min_ref_snr']}, "
                     f"contrast >= {result['min_ref_contrast']} photons",
                     ha="left",fontsize=8.5)
            pdf.savefig(fig,bbox_inches="tight",pad_inches=.22)
            plt.close(fig)
    print(f"Saved {len(result['nv_names'])}-page Z-only PDF: {path}")
    return path


def load_run(stem,data_dir=None):
    """Load the normal saved TXT/NPZ dataset even if remote G: is unmapped."""
    if data_dir is not None:
        return arb.load_run(stem,data_dir=data_dir)
    name=Path(str(stem))
    if name.is_absolute():
        return arb.load_run(stem)
    try:
        return arb.load_run(stem)
    except (FileNotFoundError, OSError):
        from utils import data_manager as dm
        source=Path(__file__).resolve().parents[1] / (
            "majorroutines/widefield/arbitrary_gate_tomography.py"
        )
        stamp=name.name[:10]+"-00_00_00"
        directory=dm.get_output_folder(source,time_stamp=stamp)
        if str(directory).upper().startswith("G:"):
            directory=resolve_pdf_path(directory)
        return arb.load_run(name.name,data_dir=directory)


def analyze_file(stem,data_dir=None,outdir=None,reference_scope=REFERENCE_SCOPE,
                 min_ref_contrast=MIN_REFERENCE_CONTRAST,
                 min_ref_snr=MIN_REFERENCE_SNR):
    """Offline reanalysis and single multipage PDF from an acquired Z-only run."""
    raw,path=load_run(stem,data_dir=data_dir)
    result=analyze(raw,min_ref_contrast=min_ref_contrast,
                   min_ref_snr=min_ref_snr,reference_scope=reference_scope)
    dest=(Path(outdir) if outdir else resolve_pdf_path(
        path.with_name(path.stem+"-z-only-analysis")
    ))
    dest.mkdir(parents=True,exist_ok=True)
    pdf=save_individual_pdf(result,dest/"all_individual_nvs.pdf")
    figs=plot_results(result)
    for fig,label in zip(figs,("summary","heatmap")):
        fig.savefig(dest/(label+".png"),dpi=190)
        plt.close(fig)
    summary={
        "stem":Path(str(path)).stem,
        "readout_protocol":result["readout_protocol"],
        "observable":result["observable"],
        "num_nvs":result["num_nvs"],
        "num_runs":result["num_runs"],
        "num_reps":result["num_reps"],
        "reference_scope":result["reference_scope"],
        "valid_gate_nv_pairs":int(np.sum(result["valid_ref"])),
        "total_gate_nv_pairs":int(result["valid_ref"].size),
        "median_abs_z_error":(
            float(np.nanmedian(result["error"][result["valid_ref"]]))
            if np.any(result["valid_ref"]) else None
        ),
        "pdf_path":str(pdf),
        "warning":"Direct Z on +Z cannot validate virtual-Z phases or full gate fidelity."
    }
    (dest/"summary.json").write_text(json.dumps(summary,indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return result,pdf


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file-stem",required=True)
    p.add_argument("--data-dir",default=None)
    p.add_argument("--outdir",default=None)
    p.add_argument("--reference-scope",choices=("per_gate","all_gates"),
                   default=REFERENCE_SCOPE)
    p.add_argument("--min-contrast",type=float,default=MIN_REFERENCE_CONTRAST)
    p.add_argument("--min-snr",type=float,default=MIN_REFERENCE_SNR)
    a=p.parse_args()
    analyze_file(a.file_stem,data_dir=a.data_dir,outdir=a.outdir,
                 reference_scope=a.reference_scope,
                 min_ref_contrast=a.min_contrast,min_ref_snr=a.min_snr)
