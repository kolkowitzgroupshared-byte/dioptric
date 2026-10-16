"""Postprocess Cornell Experiment 0 arbitrary-gate tomography.

Supported count layouts
-----------------------
Preferred/batched:
    [gate, experiment=4, NV, run, step=1, repetition]
    experiment order for each gate: Z, X, Y, dual-rail reference

Legacy:
    [experiment=3*n_gates+1, NV, run, step=1, repetition]
    order: gate1 Z,X,Y; ...; gateN Z,X,Y; one shared reference
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt


def _analyze_batched_counts(counts, n_gates, min_ref_contrast):
    if (
        counts.ndim != 6
        or counts.shape[0] != n_gates
        or counts.shape[1] != 4
        or counts.shape[4] != 1
    ):
        raise ValueError(
            "Batched counts must have shape "
            "(n_gates, 4, nv, run, 1, rep)."
        )
    if counts.shape[-1] < 2:
        raise ValueError("Reference requires both even and odd repetitions.")

    # [gate, exp, nv, run, rep]
    data = counts[:, :, :, :, 0, :]

    # First three experiments are Z, X, Y signal readouts.
    sig = np.nanmean(data[:, :3], axis=-1)  # [gate,basis,nv,run]

    # Fourth experiment is the alternating dual-rail reference.
    ref = data[:, 3]  # [gate,nv,run,rep]
    ref0 = np.nanmean(ref[:, :, :, 0::2], axis=-1)
    ref1 = np.nanmean(ref[:, :, :, 1::2], axis=-1)

    contrast = ref0 - ref1
    valid = np.isfinite(contrast) & (np.abs(contrast) >= min_ref_contrast)
    denominator = np.where(valid, contrast, np.nan)

    sz = (
        2 * sig
        - ref0[:, None, :, :]
        - ref1[:, None, :, :]
    ) / denominator[:, None, :, :]

    # [gate,basis,nv,run] -> [gate,nv,run,basis]
    per_run = sz.transpose(0, 2, 3, 1)

    # Acquisition basis order is Z,X,Y. Store XYZ.
    per_run = per_run[:, :, :, [1, 2, 0]]

    return per_run, ref0, ref1, contrast, valid


def _analyze_legacy_counts(counts, n_gates, min_ref_contrast):
    if (
        counts.ndim != 5
        or counts.shape[0] != 3 * n_gates + 1
        or counts.shape[3] != 1
    ):
        raise ValueError(
            "Legacy counts must have shape "
            "(3*n_gates+1, nv, run, 1, rep)."
        )
    if counts.shape[-1] < 2:
        raise ValueError("Reference requires both even and odd repetitions.")

    data = counts[:, :, :, 0, :]
    ref = data[-1]

    ref0 = np.nanmean(ref[:, :, 0::2], axis=-1)
    ref1 = np.nanmean(ref[:, :, 1::2], axis=-1)

    contrast = ref0 - ref1
    valid = np.isfinite(contrast) & (np.abs(contrast) >= min_ref_contrast)
    denominator = np.where(valid, contrast, np.nan)

    sig = np.nanmean(data[:-1], axis=-1)
    sz = (
        2 * sig
        - ref0[None, :, :]
        - ref1[None, :, :]
    ) / denominator[None, :, :]

    # [gate*basis,nv,run] -> [gate,nv,run,basis]
    per_run = (
        sz.reshape(n_gates, 3, counts.shape[1], counts.shape[2])
        .transpose(0, 2, 3, 1)
    )
    per_run = per_run[:, :, :, [1, 2, 0]]

    return per_run, ref0, ref1, contrast, valid


def analyze(raw_data, min_ref_contrast=0.5):
    counts = np.asarray(raw_data["counts"], dtype=float)
    gates = raw_data["cornell_manifest"]["gates"]
    n_gates = len(gates)

    if counts.ndim == 6:
        per_run, ref0, ref1, contrast, valid = _analyze_batched_counts(
            counts, n_gates, min_ref_contrast
        )
    elif counts.ndim == 5:
        per_run, ref0, ref1, contrast, valid = _analyze_legacy_counts(
            counts, n_gates, min_ref_contrast
        )
    else:
        raise ValueError(
            f"Unexpected counts ndim={counts.ndim}; expected 6 (batched) "
            "or 5 (legacy)."
        )

    with np.errstate(invalid="ignore"):
        measured = np.nanmean(per_run, axis=2)

    if per_run.shape[2] >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            n_valid = np.sum(np.isfinite(per_run), axis=2)
            ste = np.nanstd(per_run, axis=2, ddof=1) / np.sqrt(n_valid)
    else:
        ste = np.full_like(measured, np.nan)

    expected = np.array([g["expected_bloch_xyz"] for g in gates], dtype=float)
    compiled_expected = np.array(
        [g["predicted_actual_bloch"] for g in gates], dtype=float
    )

    residual = measured - expected[:, None, :]
    residual_compiled = measured - compiled_expected[:, None, :]
    norm = np.linalg.norm(residual, axis=-1)

    return {
        "measured_xyz": measured,
        "ste_xyz": ste,
        "expected_xyz": expected,
        "compiled_xyz": compiled_expected,
        "residual_xyz": residual,
        "residual_compiled_xyz": residual_compiled,
        "bloch_error_norm": norm,
        "reference_0": ref0,
        "reference_1": ref1,
        "reference_contrast": contrast,
        "valid_ref": valid,
        "reference_valid_fraction": float(np.mean(valid)),
        "per_run_xyz": per_run,
    }


def plot_results(raw_data, analysis, nv_ind=0):
    exp = analysis["expected_xyz"]
    obs = analysis["measured_xyz"]
    err = analysis["ste_xyz"]

    if nv_ind >= obs.shape[1]:
        raise ValueError("nv_ind out of range")

    n = len(exp)
    x = np.arange(1, n + 1)

    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for j, (name, ax) in enumerate(zip(("X", "Y", "Z"), axs)):
        ax.plot(x, exp[:, j], "k--", marker="o", label="Cornell predicted")
        ax.plot(
            x,
            analysis["compiled_xyz"][:, j],
            ".",
            label="OPX quantized expected",
        )
        yerr = err[:, nv_ind, j]
        if not np.any(np.isfinite(yerr)):
            yerr = None
        ax.errorbar(
            x,
            obs[:, nv_ind, j],
            yerr=yerr,
            fmt="o",
            label="Measured",
        )
        ax.set_ylabel(f"<sigma_{name.lower()}>")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(alpha=0.2)

    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel("Cornell gate index")
    fig.suptitle(f"Cornell Experiment 0, NV index {nv_ind}")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    im = ax2.imshow(
        analysis["bloch_error_norm"].T,
        aspect="auto",
        origin="lower",
        vmin=0,
    )
    ax2.set_xlabel("Gate index (0 based)")
    ax2.set_ylabel("NV index")
    ax2.set_title("Full Bloch-vector distance to Cornell prediction")
    fig2.colorbar(im, ax=ax2, label="|r_measured - r_target|")

    return fig, fig2


def analyze_file(file_stem, nv_ind=0):
    from utils import data_manager as dm

    raw = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
        use_cache=False,
    )
    results = analyze(raw)
    fig, fig2 = plot_results(raw, results, nv_ind=nv_ind)

    print("Reference valid fraction:", results["reference_valid_fraction"])
    print(
        "Mean |Bloch error| for each gate:",
        np.nanmean(results["bloch_error_norm"], axis=1),
    )
    plt.show(block=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file-stem", required=True)
    p.add_argument("--nv-ind", type=int, default=0)
    args = p.parse_args()
    analyze_file(args.file_stem, args.nv_ind)
