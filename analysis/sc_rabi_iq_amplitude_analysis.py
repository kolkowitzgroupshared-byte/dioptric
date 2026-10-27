# -*- coding: utf-8 -*-
"""Analyze fixed-duration IQ-amplitude Rabi on Purcell widefield NVs.

Raw counts: [signal/ref, NV, run, step, rep].
Each step has its own alternating SCC reference:
  even repetitions: ms=0; odd repetitions: ms=1.
The zero-IQ-amplitude point skips the microwave TTL.

Usage:
    python -m analysis.sc_rabi_iq_amplitude_analysis --file TIMESTAMP-NV
    python -m analysis.sc_rabi_iq_amplitude_analysis --file TIMESTAMP-NV --data-dir G:/nvdata/pc_Purcell/branch_master/rabi_iq_amplitude/2026_10
    python -m analysis.sc_rabi_iq_amplitude_analysis --file TIMESTAMP-NV --nv-indices 1 2 3 4
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

NAS_EXPERIMENT_ROOT = Path(
    r"\\192.168.0.197\G\nvdata\pc_Purcell\branch_master\rabi_iq_amplitude"
)


def cosine_model(amplitude_v, baseline, contrast, period_v, phase_rad):
    """P(ms=-1) versus IQ voltage; a pi rotation occurs about half a period."""
    a = np.asarray(amplitude_v, dtype=float)
    return baseline + 0.5 * contrast * (
        1.0 - np.cos(2 * np.pi * a / period_v + phase_rad)
    )


def fit_trace(amps, y, y_sem=None):
    amps = np.asarray(amps, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(amps) & np.isfinite(y)
    if y_sem is not None:
        y_sem = np.asarray(y_sem, dtype=float)
        good &= np.isfinite(y_sem) & (y_sem > 0)
    x = amps[good]
    z = y[good]
    if x.size < 8:
        return None
    order = np.argsort(x)
    x, z = x[order], z[order]
    spacing = float(np.median(np.diff(x)))
    span = float(np.ptp(x))
    if spacing <= 0 or span <= 0:
        return None
    sigma = (y_sem[good][order] if y_sem is not None else np.ones_like(z))
    floor = max(float(np.nanmedian(sigma)) * 0.3, 1e-4)
    sigma = np.maximum(sigma, floor)

    # Seed from FFT, then multistart to avoid locking onto a harmonic.
    fft = np.fft.rfft(z - np.mean(z))
    freqs = np.fft.rfftfreq(z.size, spacing)
    peak = int(np.argmax(np.abs(fft[1:])) + 1)
    fft_period = 1.0 / freqs[peak] if len(freqs) > 1 and freqs[peak] > 0 else span
    contrast0 = max(float(np.ptp(z)), 0.1)
    period_seeds = np.unique(np.clip(
        [fft_period, fft_period*0.5, fft_period*2,
         span/5, span/4, span/3, span/2, span, span*2],
        max(2*spacing, 0.01), max(2*span, 0.02)
    ))
    best = None
    for p0_period in period_seeds:
        for phase in (0., np.pi):
            try:
                popt, cov = curve_fit(
                    cosine_model, x, z,
                    p0=(float(np.nanmin(z)), contrast0, p0_period, phase),
                    sigma=sigma, absolute_sigma=True,
                    bounds=([-0.5, 0, max(2*spacing, 0.005), -2*np.pi],
                            [1.5, 2.0, max(2*span, 0.02), 2*np.pi]),
                    maxfev=20000,
                )
                resid = (cosine_model(x, *popt)-z)/sigma
                score = float(np.sum(resid*resid))
                if best is None or score < best["chi2"]:
                    period = float(popt[2])
                    # First positive local maximum of P1(a).
                    a_pi = (0.5 - float(popt[3])/(2*np.pi))*period
                    a_pi %= period
                    if a_pi == 0:
                        a_pi = period
                    best = dict(
                        baseline=float(popt[0]), contrast=float(popt[1]),
                        period_v=period, phase_rad=float(popt[3]),
                        pi_amplitude_v=float(a_pi),
                        chi2=score, fit_params=popt,
                        period_ste_v=(float(np.sqrt(cov[2,2]))
                                      if np.isfinite(cov[2,2]) and cov[2,2]>=0
                                      else float("nan")),
                    )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
    return best


def load_data(file_stem, data_dir=None):
    candidate = Path(str(file_stem))
    stem = candidate.with_suffix("") if candidate.suffix in (".txt", ".npz") else candidate
    if data_dir is not None:
        stem = Path(data_dir) / stem.name
    elif stem.parent == Path("."):
        month = stem.name[:7]
        stem = NAS_EXPERIMENT_ROOT / month / stem.name
    with stem.with_suffix(".txt").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as f:
        raw["counts"] = np.asarray(f["counts"])
    return raw, stem


def process(raw, nv_indices=None, min_contrast=2.5, min_snr=3.0):
    counts = np.asarray(raw["counts"], dtype=np.float64)
    if counts.ndim != 5 or counts.shape[0] != 2:
        raise ValueError(f"Expected [2,nv,run,step,rep], got {counts.shape}")
    _, nvs, runs, steps, reps = counts.shape
    if reps < 2 or reps % 2:
        raise ValueError("Dual-rail reference requires an even number of repetitions")
    amplitudes = np.asarray(raw["iq_amplitudes_v"], dtype=float)
    if len(amplitudes) != steps:
        raise ValueError("Amplitude-axis length does not match count steps")
    inds = (np.arange(nvs) if nv_indices is None else
            np.asarray(nv_indices, dtype=int))
    if inds.size == 0 or np.any((inds < 0) | (inds >= nvs)):
        raise ValueError("NV indices must be within the acquired list")
    sig, ref = counts[0, inds], counts[1, inds]
    r0raw, r1raw = ref[..., ::2], ref[..., 1::2]

    def stats(a):
        mean = np.nanmean(a, axis=(1,3))
        n = np.sum(np.isfinite(a), axis=(1,3))
        sem = np.nanstd(a, axis=(1,3), ddof=1) / np.sqrt(np.maximum(n,1))
        return mean, sem

    s, ssem = stats(sig)
    r0, sem0 = stats(r0raw)
    r1, sem1 = stats(r1raw)
    d = r1-r0
    dsem = np.hypot(sem0, sem1)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = d/dsem
    valid = (np.isfinite(d) & (d > min_contrast) &
             np.isfinite(snr) & (snr >= min_snr))
    den = np.where(valid, d, np.nan)
    p1 = (s-r0)/den
    with np.errstate(invalid="ignore", divide="ignore"):
        p1sem = np.sqrt(
            (ssem / den)**2
            + ((r1-s)*sem0 / den**2)**2
            + ((s-r0)*sem1 / den**2)**2
        )
    # Do not clip to [0,1]; out-of-range values diagnose reference problems.
    return {
        "amplitudes_v": amplitudes, "nv_indices": inds,
        "p1": p1, "p1_sem": p1sem, "sigma_z": 1-2*p1,
        "valid": valid, "r0": r0, "r1": r1, "snr": snr,
        "pulse_duration_ns": raw["pulse_duration_ns"],
        "phase_deg": raw["phase_deg"], "num_runs": runs, "num_reps": reps,
    }


def save_analysis(result, stem, outdir=None):
    folder = (Path(outdir) if outdir is not None
              else stem.parent / (stem.name+"-iqamp-analysis"))
    folder.mkdir(parents=True, exist_ok=True)
    x, arr, ste = result["amplitudes_v"], result["p1"], result["p1_sem"]
    fits = [fit_trace(x, a, b) for a,b in zip(arr,ste)]
    mean = np.nanmean(arr, axis=0)
    means_sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(np.isfinite(arr),axis=0),1)
    )
    mean_fit = fit_trace(x, mean, means_sem)

    fig, ax = plt.subplots(figsize=(8.5,5.2),layout="constrained")
    for curve in arr[:min(len(arr),30)]:
        ax.plot(x, curve, linewidth=.6, alpha=.20)
    ax.errorbar(x,mean,yerr=means_sem,fmt="o",ms=3,capsize=2,
                label=f"Mean of {len(arr)} selected NVs")
    if mean_fit is not None:
        xf=np.linspace(float(x.min()),float(x.max()),800)
        ax.plot(xf,cosine_model(xf,*mean_fit["fit_params"]),lw=2,
                label=f'Fit: period={mean_fit["period_v"]:.3g} V; '
                      f'first pi~{mean_fit["pi_amplitude_v"]:.3g} V')
    ax.set(xlabel="IQ vector amplitude at OPX DAC (V)",
           ylabel="SCC-normalized P(ms=-1)",
           title=f'Fixed-duration IQ Rabi | {result["pulse_duration_ns"]} ns | '
                 f'phase {result["phase_deg"]} deg')
    ax.grid(alpha=.2);ax.legend()
    fig.savefig(folder/"mean_rabi_amplitude.png",dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5.8),layout="constrained")
    im=ax.imshow(arr,origin="lower",aspect="auto",interpolation="nearest",
                 extent=[float(x[0]),float(x[-1]),-0.5,len(arr)-0.5],
                 vmin=-0.1,vmax=1.1)
    ax.set(xlabel="IQ amplitude (V)",ylabel="Selected NV row",
           title="SCC-normalized Rabi response by NV")
    fig.colorbar(im,ax=ax,label="P(ms=-1)")
    fig.savefig(folder/"per_nv_heatmap.png",dpi=200)
    plt.close(fig)

    with (folder/"per_nv_fit.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["nv_index","fit_period_v","first_pi_amplitude_v",
                    "period_sem_v","contrast","fit_chi2","valid_steps"])
        for nv,fit,valid in zip(result["nv_indices"],fits,result["valid"]):
            w.writerow([int(nv)] +
                       ([fit["period_v"],fit["pi_amplitude_v"],
                         fit["period_ste_v"],fit["contrast"],fit["chi2"]]
                        if fit is not None else [""]*5) +
                       [int(np.sum(valid))])
    with (folder/"normalized_response.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["nv_index","amplitude_v","p1","p1_sem",
                                    "sigma_z","reference_valid"])
        for i,nv in enumerate(result["nv_indices"]):
            for j,a in enumerate(x):
                w.writerow([int(nv),float(a),arr[i,j],ste[i,j],
                            1-2*arr[i,j],bool(result["valid"][i,j])])
    summary = {
        "source":str(stem), "fixed_pulse_duration_ns":result["pulse_duration_ns"],
        "phase_deg":result["phase_deg"],
        "num_selected_nvs":len(arr),
        "num_runs":result["num_runs"],"num_reps":result["num_reps"],
        "reference_valid_fraction":float(np.mean(result["valid"])),
        "mean_fit": ({k:v for k,v in mean_fit.items() if k!="fit_params"}
                     if mean_fit else None),
        "warning":"Fitted pi voltage depends on this fixed pulse duration and SG394 power.",
    }
    (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Analysis saved to",folder)
    print("Mean fit:",summary["mean_fit"])
    return folder,summary


def main(file_stem, data_dir=None, outdir=None, nv_indices=None):
    raw, stem = load_data(file_stem,data_dir)
    result = process(raw,nv_indices=nv_indices)
    return save_analysis(result,stem,outdir)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file",required=True,help="Run stem or full TXT path")
    parser.add_argument("--data-dir",default=None)
    parser.add_argument("--outdir",default=None)
    parser.add_argument("--nv-indices",nargs="+",type=int,default=None,
                        help="Zero-based positions in the saved nv_list")
    args=parser.parse_args()
    main(args.file,args.data_dir,args.outdir,args.nv_indices)
