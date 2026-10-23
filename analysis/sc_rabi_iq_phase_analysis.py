# -*- coding: utf-8 -*-
"""
Analyze IQ-phase Rabi data.

Supports:
- multiple raw files
- automatic grouping by phase_deg
- concatenation of repeated files along the RUN axis
- all-NV or selected-NV analysis
- group-averaged damped-cosine fit
- per-NV Rabi-period fits
- comparison of Rabi period / pi time versus commanded IQ phase

Raw count layout:
    counts[experiment=(signal, reference), nv, run, step, rep]

The reference is dual-rail:
    even reps -> ms=0
    odd reps  -> ms=1

Created Oct 2026
@author: Saroj Chand
"""

from __future__ import annotations

import csv
import io
import json
import traceback
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield
from utils.constants import NVSig


# =============================================================================
# USER SETTINGS
# =============================================================================

# Direct raw-data folder. Reading the TXT/NPZ pair directly avoids relying on
# the data-manager SQLite search index, which can fail on the NAS/UNC path.
DATA_DIR = (
    r"\\192.168.0.197\G\nvdata\pc_Purcell\branch_master"
    r"\rabi_iq_phase\2026_10"
)

# Newest IQ-phase Rabi runs.
# Add the 270-degree file here when it is acquired.
FILE_STEMS = [
    "2026_10_22-20_13_55-johnson-nv0_2026_10_02",  # 0 deg
    "2026_10_22-21_28_28-johnson-nv0_2026_10_02",  # 90 deg
    "2026_10_22-22_43_41-johnson-nv0_2026_10_02",  # 180 deg
    "2026_10_22-23_58_20-johnson-nv0_2026_10_02",  # 270 deg
]

# If you want to combine repeated acquisitions at the same phase, simply add
# more files. They will be concatenated along the run axis automatically.
#
# Example:
# FILE_STEMS = [
#     "phase0_run1",
#     "phase0_run2",
#     "phase90_run1",
#     "phase90_run2",
#     "phase180_run1",
#     "phase270_run1",
# ]

# O2 orientation used by the chain-1 Cornell experiment.
O2_INDICES = [
    1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 20, 21, 23, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 37, 38, 41, 44, 45, 46, 47, 50, 51, 53, 54,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 71, 73, 74, 77,
    81, 84, 85, 86, 90, 92, 95, 96, 98, 101, 102, 105, 106, 107, 108,
    111, 112, 113, 114, 115, 117,
]

# None -> all NVs.
# O2_INDICES -> use the 71-NV O2 subset.
NV_INDICES = O2_INDICES

THRESHOLD = True

# Physically reasonable search window for these data.
MIN_RABI_PERIOD_NS = 80.0
MAX_RABI_PERIOD_NS = 400.0

SAVE_RESULTS = True
SAVE_FIGURES = True
SHOW_FIGURES = True

SAVE_BASENAME = "rabi_iq_phase_analysis"


# =============================================================================
# MODELS / FITTING
# =============================================================================

def damped_cosine(t, offset, amp, period, decay, phase_rad):
    return (
        offset
        + amp
        * np.exp(-t / decay)
        * np.cos(2 * np.pi * t / period + phase_rad)
    )


def _fft_period_guess(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]

    if len(t) < 5:
        return 190.0

    dt = float(np.median(np.diff(t)))
    centered = y - np.mean(y)

    transform = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=dt)

    if len(freqs) <= 1:
        return 190.0

    valid_freq = freqs > 0
    valid_freq &= freqs >= 1.0 / MAX_RABI_PERIOD_NS
    valid_freq &= freqs <= 1.0 / MIN_RABI_PERIOD_NS

    inds = np.flatnonzero(valid_freq)
    if len(inds) == 0:
        return 190.0

    best = inds[np.argmax(np.abs(transform[inds]))]
    freq = float(freqs[best])

    if freq <= 0:
        return 190.0

    return float(1.0 / freq)


def fit_rabi(t, y, yerr=None):
    """Robust damped-cosine fit. Returns dict or None."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)

    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        valid &= np.isfinite(yerr)

    tfit = t[valid]
    yfit = y[valid]

    if len(tfit) < 8:
        return None

    period0 = _fft_period_guess(tfit, yfit)

    offset0 = float(np.nanmean(yfit))
    amp0 = 0.5 * float(np.nanmax(yfit) - np.nanmin(yfit))
    if not np.isfinite(amp0) or amp0 < 0.02:
        amp0 = 0.45

    # Choose the sign/phase from the first point approximately.
    phase0 = 0.0
    if yfit[0] < offset0:
        amp0 = -abs(amp0)
    else:
        amp0 = abs(amp0)

    span = max(float(np.ptp(tfit)), 1.0)
    decay0 = max(4 * period0, 2 * span)

    p0 = [
        offset0,
        amp0,
        period0,
        decay0,
        phase0,
    ]

    lower = [
        -0.5,
        -1.5,
        MIN_RABI_PERIOD_NS,
        40.0,
        -4 * np.pi,
    ]

    upper = [
        1.5,
        1.5,
        MAX_RABI_PERIOD_NS,
        1e5,
        4 * np.pi,
    ]

    sigma = None
    if yerr is not None:
        sigma = yerr[valid]
        positive = sigma[np.isfinite(sigma) & (sigma > 0)]
        floor = np.nanmedian(positive) if len(positive) else 1.0
        sigma = np.where(
            np.isfinite(sigma) & (sigma > 0),
            sigma,
            floor,
        )

    try:
        popt, pcov = curve_fit(
            damped_cosine,
            tfit,
            yfit,
            p0=p0,
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            bounds=(lower, upper),
            maxfev=50000,
        )

        perr = np.sqrt(np.diag(pcov))

        period = float(popt[2])
        period_ste = float(perr[2])
        pi_ns = period / 2.0
        pi_ste = period_ste / 2.0

        residual = yfit - damped_cosine(tfit, *popt)
        rms = float(np.sqrt(np.mean(residual**2)))

        return {
            "popt": popt,
            "perr": perr,
            "period_ns": period,
            "period_ste_ns": period_ste,
            "pi_ns": pi_ns,
            "pi_ste_ns": pi_ste,
            "rms": rms,
        }

    except Exception:
        return None


# =============================================================================
# LOADING / COMBINING
# =============================================================================

def load_one(file_stem, data_dir=DATA_DIR):
    """Load one IQ-Rabi run directly from its TXT/NPZ pair."""
    stem = Path(str(file_stem))

    if stem.suffix in (".txt", ".npz"):
        stem = stem.with_suffix("")

    if data_dir is not None and not stem.is_absolute():
        stem = Path(data_dir) / stem.name

    txt_path = stem.with_suffix(".txt")
    npz_path = stem.with_suffix(".npz")

    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with txt_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with np.load(npz_path, allow_pickle=False) as compressed:
        if "counts" not in compressed:
            raise KeyError(f"{npz_path}: missing 'counts' array")
        data["counts"] = np.asarray(
            compressed["counts"],
            dtype=np.float64,
        )

    # Convert saved NV dictionaries back to NVSig objects because the
    # widefield threshold/normalization routines expect NVSig entries.
    valid_fields = {field.name for field in fields(NVSig)}
    data["nv_list"] = [
        NVSig(
            **{
                key: nv[key]
                for key in nv
                if key in valid_fields
            }
        )
        if isinstance(nv, dict)
        else nv
        for nv in data["nv_list"]
    ]

    if "phase_deg" not in data:
        raise ValueError(
            f"{file_stem}: missing phase_deg metadata."
        )

    counts = np.asarray(data["counts"], dtype=np.float64)

    if counts.ndim != 5:
        raise ValueError(
            f"{file_stem}: expected counts[exp,nv,run,step,rep], got {counts.shape}"
        )

    if counts.shape[0] < 2:
        raise ValueError(
            f"{file_stem}: expected signal + reference, got {counts.shape}"
        )

    return data


def _nv_names(nv_list):
    return [
        getattr(nv, "name", str(nv))
        for nv in nv_list
    ]


def combine_same_phase(entries):
    """Combine compatible files at the same phase along run axis."""
    base_stem, base = entries[0]

    phase = int(base["phase_deg"]) % 360
    taus0 = np.asarray(base["taus"], dtype=float)
    nv_names0 = _nv_names(base["nv_list"])

    counts_list = [
        np.asarray(base["counts"], dtype=np.float64)
    ]

    source_files = [base_stem]

    for stem, data in entries[1:]:
        this_phase = int(data["phase_deg"]) % 360

        if this_phase != phase:
            raise ValueError(
                f"Internal grouping error: {stem} has phase {this_phase}, expected {phase}."
            )

        taus = np.asarray(data["taus"], dtype=float)
        if taus.shape != taus0.shape or not np.allclose(taus, taus0):
            raise ValueError(
                f"{stem}: tau grid differs from {base_stem}."
            )

        if _nv_names(data["nv_list"]) != nv_names0:
            raise ValueError(
                f"{stem}: NV list/order differs from {base_stem}."
            )

        c = np.asarray(data["counts"], dtype=np.float64)

        # All axes except run axis 2 must match.
        for axis in (0, 1, 3, 4):
            if c.shape[axis] != counts_list[0].shape[axis]:
                raise ValueError(
                    f"{stem}: incompatible counts shape {c.shape}; "
                    f"first file is {counts_list[0].shape}."
                )

        counts_list.append(c)
        source_files.append(stem)

    combined = dict(base)
    combined["counts"] = np.concatenate(
        counts_list,
        axis=2,
    )
    combined["num_runs"] = int(
        combined["counts"].shape[2]
    )
    combined["source_files"] = source_files

    return combined


def load_and_group(file_stems):
    groups = defaultdict(list)

    for stem in file_stems:
        data = load_one(stem)
        phase = int(data["phase_deg"]) % 360
        groups[phase].append((stem, data))

    combined = {}
    for phase, entries in groups.items():
        combined[phase] = combine_same_phase(entries)

    return dict(sorted(combined.items()))


# =============================================================================
# PROCESS EACH PHASE
# =============================================================================

def process_phase(data, nv_indices=None):
    counts = np.asarray(data["counts"], dtype=np.float64)

    sig_counts = counts[0]
    ref_counts = counts[1]

    norm_counts, norm_counts_ste = widefield.process_counts(
        data["nv_list"],
        sig_counts,
        ref_counts,
        threshold=THRESHOLD,
    )

    norm_counts = np.asarray(norm_counts, dtype=float)
    norm_counts_ste = np.asarray(norm_counts_ste, dtype=float)

    num_nvs = norm_counts.shape[0]

    if nv_indices is None:
        inds = np.arange(num_nvs, dtype=int)
    else:
        inds = np.asarray(nv_indices, dtype=int)
        inds = inds[(inds >= 0) & (inds < num_nvs)]

    if len(inds) == 0:
        raise ValueError("No valid NV indices selected.")

    sel = norm_counts[inds]
    sel_ste = norm_counts_ste[inds]

    # Mean over selected NVs.
    group_mean = np.nanmean(sel, axis=0)

    # Propagate the individual measurement STEs into the mean.
    n_eff = np.sum(np.isfinite(sel_ste), axis=0)
    group_ste = np.sqrt(
        np.nansum(sel_ste**2, axis=0)
    ) / np.maximum(n_eff, 1)

    taus = np.asarray(data["taus"], dtype=float)

    group_fit = fit_rabi(
        taus,
        group_mean,
        group_ste,
    )

    # Per-NV fits.
    per_nv_period = np.full(num_nvs, np.nan)
    per_nv_period_ste = np.full(num_nvs, np.nan)
    per_nv_pi = np.full(num_nvs, np.nan)
    per_nv_rms = np.full(num_nvs, np.nan)

    for nv_ind in inds:
        fit = fit_rabi(
            taus,
            norm_counts[nv_ind],
            norm_counts_ste[nv_ind],
        )

        if fit is None:
            continue

        per_nv_period[nv_ind] = fit["period_ns"]
        per_nv_period_ste[nv_ind] = fit["period_ste_ns"]
        per_nv_pi[nv_ind] = fit["pi_ns"]
        per_nv_rms[nv_ind] = fit["rms"]

    valid_periods = per_nv_period[inds]
    valid_periods = valid_periods[np.isfinite(valid_periods)]

    if len(valid_periods):
        median_period = float(np.nanmedian(valid_periods))
        q25, q75 = np.nanpercentile(valid_periods, [25, 75])
        median_pi = median_period / 2.0
    else:
        median_period = np.nan
        q25 = np.nan
        q75 = np.nan
        median_pi = np.nan

    return {
        "phase_deg": int(data["phase_deg"]) % 360,
        "taus": taus,
        "norm_counts": norm_counts,
        "norm_counts_ste": norm_counts_ste,
        "selected_inds": inds,
        "group_mean": group_mean,
        "group_ste": group_ste,
        "group_fit": group_fit,
        "per_nv_period_ns": per_nv_period,
        "per_nv_period_ste_ns": per_nv_period_ste,
        "per_nv_pi_ns": per_nv_pi,
        "per_nv_rms": per_nv_rms,
        "median_period_ns": median_period,
        "median_pi_ns": median_pi,
        "period_q25_ns": float(q25),
        "period_q75_ns": float(q75),
        "num_valid_fits": int(len(valid_periods)),
        "num_selected_nvs": int(len(inds)),
        "num_runs": int(data["counts"].shape[2]),
        "num_reps": int(data["counts"].shape[4]),
        "source_files": list(data.get("source_files", [])),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_group_traces(results_by_phase):
    fig, ax = plt.subplots(
        figsize=(8.5, 5.5),
        layout="constrained",
    )

    for phase, res in results_by_phase.items():
        t = res["taus"]
        y = res["group_mean"]
        ye = res["group_ste"]

        ax.errorbar(
            t,
            y,
            yerr=ye,
            fmt="o",
            ms=4,
            capsize=2,
            label=f"{phase} deg data",
        )

        fit = res["group_fit"]
        if fit is not None:
            tt = np.linspace(
                float(np.min(t)),
                float(np.max(t)),
                1200,
            )
            yy = damped_cosine(
                tt,
                *fit["popt"],
            )

            ax.plot(
                tt,
                yy,
                lw=1.5,
                label=(
                    f"{phase} deg fit: "
                    f"T_R={fit['period_ns']:.1f} ns, "
                    f"pi={fit['pi_ns']:.1f} ns"
                ),
            )

    ax.set_xlabel("IQ pulse duration (ns)")
    ax.set_ylabel("Normalized population")
    ax.set_title(
        "IQ-phase Rabi: selected-NV average"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    return fig


def plot_period_vs_phase(results_by_phase):
    phases = np.array(
        sorted(results_by_phase),
        dtype=float,
    )

    med = np.array(
        [
            results_by_phase[int(p)]["median_period_ns"]
            for p in phases
        ],
        dtype=float,
    )

    q25 = np.array(
        [
            results_by_phase[int(p)]["period_q25_ns"]
            for p in phases
        ],
        dtype=float,
    )

    q75 = np.array(
        [
            results_by_phase[int(p)]["period_q75_ns"]
            for p in phases
        ],
        dtype=float,
    )

    yerr = np.vstack(
        [
            med - q25,
            q75 - med,
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.8),
        layout="constrained",
    )

    ax.errorbar(
        phases,
        med,
        yerr=yerr,
        fmt="o",
        capsize=4,
        label="Median per-NV Rabi period (IQR)",
    )

    ax.set_xlabel("Commanded IQ phase (deg)")
    ax.set_ylabel("Rabi period (ns)")
    ax.set_title(
        "Rabi-rate equality across IQ phase"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    return fig


def plot_per_nv_periods(results_by_phase):
    fig, ax = plt.subplots(
        figsize=(9, 5.2),
        layout="constrained",
    )

    for phase, res in results_by_phase.items():
        inds = res["selected_inds"]
        periods = res["per_nv_period_ns"][inds]

        valid = np.isfinite(periods)

        ax.plot(
            inds[valid],
            periods[valid],
            ".",
            ms=6,
            label=f"{phase} deg",
        )

    ax.set_xlabel("NV index")
    ax.set_ylabel("Fitted Rabi period (ns)")
    ax.set_title(
        "Per-NV IQ-phase Rabi periods"
    )
    ax.grid(alpha=0.25)
    ax.legend()

    return fig


# =============================================================================
# SUMMARY / SAVE
# =============================================================================

def print_summary(results_by_phase):
    print()
    print("=" * 84)
    print("IQ PHASE RABI SUMMARY")
    print("=" * 84)
    print(
        "phase  files  runs  selectedNV  validFit  "
        "group_period(ns)  group_pi(ns)  median_period(ns)  median_pi(ns)"
    )

    for phase, res in results_by_phase.items():
        fit = res["group_fit"]

        if fit is None:
            group_period = np.nan
            group_pi = np.nan
        else:
            group_period = fit["period_ns"]
            group_pi = fit["pi_ns"]

        print(
            f"{phase:5d}  "
            f"{len(res['source_files']):5d}  "
            f"{res['num_runs']:4d}  "
            f"{res['num_selected_nvs']:10d}  "
            f"{res['num_valid_fits']:8d}  "
            f"{group_period:16.2f}  "
            f"{group_pi:12.2f}  "
            f"{res['median_period_ns']:17.2f}  "
            f"{res['median_pi_ns']:13.2f}"
        )

    print("=" * 84)

    # Very useful direct diagnostics.
    if 0 in results_by_phase and 90 in results_by_phase:
        p0 = results_by_phase[0]["median_period_ns"]
        p90 = results_by_phase[90]["median_period_ns"]
        mismatch = 100 * (p90 - p0) / np.nanmean([p0, p90])

        print(
            f"0 vs 90 deg median-period mismatch: {mismatch:+.2f}%"
        )

    if 0 in results_by_phase and 180 in results_by_phase:
        p0 = results_by_phase[0]["median_period_ns"]
        p180 = results_by_phase[180]["median_period_ns"]
        mismatch = 100 * (p180 - p0) / np.nanmean([p0, p180])

        print(
            f"0 vs 180 deg median-period mismatch: {mismatch:+.2f}%"
        )

    if 90 in results_by_phase and 270 in results_by_phase:
        p90 = results_by_phase[90]["median_period_ns"]
        p270 = results_by_phase[270]["median_period_ns"]
        mismatch = 100 * (p270 - p90) / np.nanmean([p90, p270])

        print(
            f"90 vs 270 deg median-period mismatch: {mismatch:+.2f}%"
        )


def save_summary_csv(results_by_phase, output_base):
    """Save summary CSV through the normal data backend (works on NAS)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(
            [
                "phase_deg",
                "num_files",
                "num_runs",
                "num_reps",
                "num_selected_nvs",
                "num_valid_nv_fits",
                "group_period_ns",
                "group_period_ste_ns",
                "group_pi_ns",
                "group_pi_ste_ns",
                "median_nv_period_ns",
                "period_q25_ns",
                "period_q75_ns",
                "median_nv_pi_ns",
                "source_files",
            ]
        )

    for phase, res in results_by_phase.items():
        fit = res["group_fit"]

        writer.writerow(
            [
                phase,
                len(res["source_files"]),
                res["num_runs"],
                res["num_reps"],
                res["num_selected_nvs"],
                res["num_valid_fits"],
                np.nan if fit is None else fit["period_ns"],
                np.nan if fit is None else fit["period_ste_ns"],
                np.nan if fit is None else fit["pi_ns"],
                np.nan if fit is None else fit["pi_ste_ns"],
                res["median_period_ns"],
                res["period_q25_ns"],
                res["period_q75_ns"],
                res["median_pi_ns"],
                ";".join(res["source_files"]),
            ]
        )

    csv_path = Path(str(output_base) + "-summary")
    dm.save_text(
        buffer.getvalue(),
        csv_path,
        suffix=".csv",
    )
    print(f"Saved CSV: {csv_path.with_suffix('.csv')}")


def save_analysis_record(results_by_phase, file_stems):
    timestamp = dm.get_time_stamp()

    output_path = dm.get_file_path(
        __file__,
        timestamp,
        SAVE_BASENAME,
    )

    record = {
        "timestamp": timestamp,
        "source_file_stems": list(file_stems),
        "selected_nv_indices": (
            None if NV_INDICES is None else list(NV_INDICES)
        ),
        "threshold": bool(THRESHOLD),
        "phase_results": {},
    }

    for phase, res in results_by_phase.items():
        fit = res["group_fit"]

        record["phase_results"][str(phase)] = {
            "source_files": res["source_files"],
            "num_runs": res["num_runs"],
            "num_reps": res["num_reps"],
            "num_selected_nvs": res["num_selected_nvs"],
            "num_valid_fits": res["num_valid_fits"],
            "group_period_ns": (
                None if fit is None else float(fit["period_ns"])
            ),
            "group_period_ste_ns": (
                None if fit is None else float(fit["period_ste_ns"])
            ),
            "group_pi_ns": (
                None if fit is None else float(fit["pi_ns"])
            ),
            "group_pi_ste_ns": (
                None if fit is None else float(fit["pi_ste_ns"])
            ),
            "median_nv_period_ns": float(res["median_period_ns"]),
            "period_q25_ns": float(res["period_q25_ns"]),
            "period_q75_ns": float(res["period_q75_ns"]),
            "median_nv_pi_ns": float(res["median_pi_ns"]),
            "per_nv_period_ns": res["per_nv_period_ns"].tolist(),
            "per_nv_pi_ns": res["per_nv_pi_ns"].tolist(),
        }

    # Save analysis metadata/results through the normal data manager.
    dm.save_raw_data(
        record.copy(),
        output_path,
    )
    print(f"Saved analysis record: {output_path}")

    return output_path.with_suffix("")


# =============================================================================
# MAIN
# =============================================================================

def main(file_stems=FILE_STEMS, nv_indices=NV_INDICES):
    groups = load_and_group(file_stems)

    print()
    print("Files grouped by phase:")
    for phase, data in groups.items():
        print(
            f"  {phase:3d} deg : "
            f"{len(data['source_files'])} file(s), "
            f"{data['counts'].shape[2]} total runs"
        )

    results = {}

    for phase, data in groups.items():
        print()
        print(f"Processing phase {phase} deg...")

        results[phase] = process_phase(
            data,
            nv_indices=nv_indices,
        )

    print_summary(results)

    fig1 = plot_group_traces(results)
    fig2 = plot_period_vs_phase(results)
    fig3 = plot_per_nv_periods(results)

    if SAVE_RESULTS:
        output_base = save_analysis_record(
            results,
            file_stems,
        )
    else:
        timestamp = dm.get_time_stamp()
        output_base = dm.get_file_path(
            __file__,
            timestamp,
            SAVE_BASENAME,
        ).with_suffix("")

    if SAVE_FIGURES:
        dm.save_figure(
            fig1,
            Path(
                str(output_base)
                + "-group-traces"
            ),
        )

        dm.save_figure(
            fig2,
            Path(
                str(output_base)
                + "-period-vs-phase"
            ),
        )

        dm.save_figure(
            fig3,
            Path(
                str(output_base)
                + "-per-nv-periods"
            ),
        )

        print("Saved figures.")

    # CSV is written locally only if output_base is an accessible filesystem path.
    # If this path is cloud-backed and not mounted, skip without failing analysis.
    try:
        save_summary_csv(
            results,
            output_base,
        )
    except Exception:
        print("CSV save skipped:")
        print(traceback.format_exc())

    if SHOW_FIGURES:
        kpl.show(block=True)
    else:
        plt.close("all")

    return results


if __name__ == "__main__":
    kpl.init_kplotlib()
    main()
