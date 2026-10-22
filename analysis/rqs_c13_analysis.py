# -*- coding: utf-8 -*-
"""Analyze Purcell RQS-inspired 13C randomized DD vs periodic XY8 SCC data.

This script is DIAGNOSTIC: a classical Gaussian-noise filter overlap is not
a full quantum 13C hyperfine spectrum. It does not claim Kannan's optimal RQS.
Run from dioptric root: python analysis/rqs_c13_analysis.py --latest
Or: python analysis/rqs_c13_analysis.py --file-stem TIMESTAMP-NVNAME
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path

# Allow direct execution from the repository root, not only python -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def filter_y2(step, f_hz):
    """Ideal instantaneous-pulse toggling-filter power |Y(f)|^2 [s^2]."""
    pulse_t = np.asarray(step["pi_centers_ns"], float) * 1e-9
    t_end = float(step["T_ns"]) * 1e-9
    boundaries = np.r_[0.0, pulse_t, t_end]
    f_hz = np.asarray(f_hz, float)
    omega = 2 * np.pi * f_hz
    ph1 = np.exp(1j * omega[:, None] * boundaries[1:][None, :])
    ph0 = np.exp(1j * omega[:, None] * boundaries[:-1][None, :])
    signs = (-1.0) ** np.arange(len(boundaries) - 1)
    y = ((ph1 - ph0) * signs[None, :]).sum(axis=1) / (1j * omega)
    return np.abs(y) ** 2


def normalize_scc(raw_data):
    """Dual-rail calibration: ref exp alternates bright/dark with rep parity.

    Returns per-NV per-step apparent contrast. Auto-orients sign so short
    random measurements tend toward +1; final XY8 pi/2 may invert the sign.
    """
    counts = np.asarray(raw_data["counts"], float)
    if counts.ndim != 5 or counts.shape[0] != 2:
        raise ValueError("Expected counts[exp, nv, run, step, rep] with exp=2")
    if counts.shape[-1] % 2:
        raise ValueError("Even num_reps needed for dual-rail reference")
    sig = np.mean(counts[0], axis=(1, 3))
    ref0 = np.mean(counts[1, :, :, :, ::2], axis=(1, 3))
    ref1 = np.mean(counts[1, :, :, :, 1::2], axis=(1, 3))
    diff = ref0 - ref1
    raw_c = np.divide(
        2 * sig - ref0 - ref1, diff,
        out=np.full_like(sig, np.nan),
        where=np.abs(diff) > 0.2,
    )
    steps = raw_data["rqs_c13_protocol"]["steps"]
    tvals = np.asarray([s["T_ns"] for s in steps])
    idx = [i for i, s in enumerate(steps) if s["kind"] == "random"]
    idx = sorted(idx, key=lambda i: tvals[i])[:max(4, min(15, len(idx)))]
    orient = np.sign(np.nanmedian(raw_c[:, idx], axis=1))
    orient[~np.isfinite(orient) | (orient == 0)] = 1
    return raw_c * orient[:, None], diff


def band_overlaps(proto, f_target_hz, fractional_width=0.20):
    """Integrate ideal |Y(f)|^2 over a band centered on f_target_hz."""
    width = max(f_target_hz * fractional_width, 5e3)
    low = max(f_target_hz - width, 1e3)
    hi = f_target_hz + width
    f = np.linspace(low, hi, 101)
    # normalize by bandwidth for numerical conditioning; units remain s^2
    return np.array([np.trapz(filter_y2(s, f), f) / (hi - low)
                     for s in proto["steps"]])


def regression_by_duration(contrast, protocol, overlaps):
    """Fit chi_i = intercept_by_duration + slope * filtered-power proxy.

    Only random steps. Intercepts suppress generic time-dependent decoherence.
    Slope sign/strength is descriptive; this is not calibrated nuclear PSD.
    """
    steps = protocol["steps"]
    inds = [i for i, s in enumerate(steps) if s["kind"] == "random"]
    times = np.array([int(round(steps[i]["T_ns"])) for i in inds])
    x = np.asarray(overlaps[inds], float)
    good_x = np.isfinite(x)
    # 1e-9 s^2 chosen only to display a human-scale fitted coefficient
    x = x / 1e-9
    c = np.asarray(contrast[inds], float)
    valid = np.isfinite(c) & (c > 0.025) & (c < 0.999) & good_x
    y = -np.log(np.clip(c, 0.025, 0.999))
    xcent = np.full_like(x, np.nan)
    ycent = np.full_like(y, np.nan)
    for t in np.unique(times):
        keep = (times == t) & valid
        if np.sum(keep) > 1:
            xcent[keep] = x[keep] - np.mean(x[keep])
            ycent[keep] = y[keep] - np.mean(y[keep])
    good = np.isfinite(xcent) & np.isfinite(ycent)
    if good.sum() < 8 or np.sum(xcent[good] ** 2) < 1e-9:
        return np.nan, np.nan, int(good.sum())
    slope = np.dot(xcent[good], ycent[good]) / np.dot(xcent[good], xcent[good])
    fitted = slope * xcent[good]
    resid = ycent[good] - fitted
    se = np.sqrt(np.sum(resid**2) / max(1, good.sum() - 2) /
                 np.sum(xcent[good]**2))
    return float(slope), float(se), int(good.sum())


def analyze(raw_data, outdir=None, target_hz=None, source_stem=None):
    """Analyze counts and save through dm to G:\\nvdata by default.

    outdir is an explicit local-test override only. In ordinary lab analysis,
    outdir=None uses dm.get_file_path/save_figure/save_text/save_npz.
    """
    if not raw_data.get("rqs_complete", False):
        raise RuntimeError("Acquisition incomplete; will not analyze partial counts.")
    proto = raw_data["rqs_c13_protocol"]
    steps = proto["steps"]
    f_l = float(proto["bare_c13_larmor_hz_expected"])
    if target_hz is None:
        if proto["band"] == "high":
            raise ValueError("High band needs a hyperfine candidate --target-khz.")
        target_hz = f_l
    c, refdiff = normalize_scc(raw_data)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    overlaps = band_overlaps(proto, target_hz)
    slopes = [regression_by_duration(c[i], proto, overlaps)
              for i in range(c.shape[0])]
    random_inds = [j for j, s in enumerate(steps) if s["kind"] == "random"]
    xy_inds = [j for j, s in enumerate(steps) if s["kind"] == "xy8"]
    freqs = np.geomspace(7e3 if proto["band"] == "low" else 8e4,
                        1.6e5 if proto["band"] == "low" else 2.2e6, 350)
    filt = np.asarray([filter_y2(s, freqs) for s in steps])
    filt /= np.maximum(filt.max(axis=1, keepdims=True), 1e-40)
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    axs[0].imshow(filt, extent=[freqs.min()/1e3, freqs.max()/1e3,
                               len(steps)-0.5, -0.5],
                  aspect="auto", interpolation="nearest", cmap="viridis")
    axs[0].axvline(target_hz/1e3, color="white", linestyle="--",
                   label="target 13C candidate")
    axs[0].set_xlabel("Frequency (kHz) [displayed linear; grid log-spaced]")
    # Correct x-axis coordinate for log-spaced grid:
    axs[0].images[-1].set_extent([0, len(freqs)-1, len(steps)-0.5, -0.5])
    positions = [np.argmin(abs(freqs-v)) for v in np.geomspace(freqs.min(),freqs.max(),6)]
    axs[0].set_xticks(positions, [f"{freqs[p]/1e3:.0f}" for p in positions])
    axs[0].lines[-1].remove()
    axs[0].axvline(int(np.argmin(abs(freqs-target_hz))), color="white",
                   linestyle="--")
    axs[0].set_ylabel("Protocol step")
    axs[0].set_title("Ideal instantaneous-pulse filter power (per-row scaled)")
    avg = np.nanmedian(c, axis=0)
    avg_xy = avg[xy_inds]
    x_xy = [steps[i]["target_f_hz"]/1e3 for i in xy_inds]
    axs[1].plot(x_xy, avg_xy, "o-", label="XY8-8")
    axs[1].axvline(target_hz/1e3, color="tab:red", ls="--", label="13C target")
    axs[1].set_xlabel("Nominal XY8 filter frequency (kHz)")
    axs[1].set_ylabel("Median SCC-normalized coherence")
    axs[1].legend(loc="best")
    xs = overlaps[random_inds] / 1e-9
    xr = np.array([steps[i]["T_ns"] for i in random_inds])/1000
    sc = axs[2].scatter(xs, avg[random_inds], c=xr, cmap="plasma")
    fig.colorbar(sc, ax=axs[2], label="Sensing time (us)")
    axs[2].set_xlabel("13C candidate band filter overlap (1e-9 s²)")
    axs[2].set_ylabel("Median SCC-normalized coherence")
    axs[2].set_title("Random-mask response: correlation, not automatic identification")
    # Build per-NV result table independently of the output backend.
    output_csv = StringIO(newline="")
    wr = csv.writer(output_csv)
    wr.writerow(["nv_index", "regression_slope_arbitrary", "slope_se",
                 "random_steps_used", "mean_dualrail_ref_diff_photons"])
    for i, (slope, se, n) in enumerate(slopes):
        wr.writerow([i, slope, se, n, float(np.nanmean(refdiff[i]))])

    source_stem = Path(source_stem or raw_data.get("timestamp", "unknown_run")).name
    valid_slopes = [float(s) for s, _, _ in slopes if np.isfinite(s)]
    summary = {
        "source_experiment_stem": source_stem,
        "method": "RQS-inspired randomized timing vs periodic XY8-8",
        "warning": "Diagnostic correlation; not proof of 13C or a calibrated PSD",
        "target_frequency_khz": float(target_hz / 1e3),
        "expected_bare_13c_larmor_khz": float(f_l / 1e3),
        "field_g_from_acquisition": float(proto["field_g_expected"]),
        "band": proto["band"],
        "protocol_sha256": proto.get("protocol_sha256"),
        "number_of_nvs": int(c.shape[0]),
        "number_of_steps": int(c.shape[1]),
        "number_of_random_steps": len(random_inds),
        "number_of_xy8_steps": len(xy_inds),
        "nvs_with_finite_regression_slope": len(valid_slopes),
        "median_regression_slope_arbitrary": (
            float(np.median(valid_slopes)) if valid_slopes else None
        ),
    }

    if outdir is None:
        # The same NAS-backed dm workflow as other Purcell analysis scripts.
        # No silent fallback to C: when the mapped NAS is unavailable.
        from utils import data_manager as dm

        stamp = dm.get_time_stamp()
        base = dm.get_file_path(
            __file__, stamp, "rqs_c13_analysis", subfolder=source_stem
        )
        dm.save_figure(fig, base)
        dm.save_text(output_csv.getvalue(), base, suffix=".csv")
        dm.save_text(json.dumps(summary, indent=2), base, suffix=".json")
        dm.save_npz(
            base,
            compressed=True,
            normalized_coherence=c,
            dual_rail_reference_separation_photons=refdiff,
            filter_overlap_s2=overlaps,
            sensing_time_ns=np.asarray([s["T_ns"] for s in steps], dtype=float),
            target_frequency_hz=np.asarray([target_hz]),
            xy8_nominal_frequency_hz=np.asarray(
                [steps[j]["target_f_hz"] for j in xy_inds]
            ),
        )
        print("Saved RQS C13 analysis through data_manager:")
        for ext in (".png", ".csv", ".json", ".npz"):
            print(" ", base.with_suffix(ext))
    else:
        # Explicit local override, mainly for synthetic tests and debugging.
        fig.savefig(outdir / "rqs_c13_diagnostic.png", dpi=175)
        (outdir / "rqs_c13_per_nv.csv").write_text(
            output_csv.getvalue(), encoding="utf-8"
        )
        (outdir / "rqs_c13_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("Saved local test analysis to:", outdir)
    plt.close(fig)
    print("Caution: regression needs 13C-free/magnetic-field controls to identify peak.")
    return slopes


def self_test():
    from majorroutines.widefield.rqs_c13 import build_protocol
    proto = build_protocol()
    overlaps = band_overlaps(proto, proto["bare_c13_larmor_hz_expected"])
    step_ts = np.array([s["T_ns"] for s in proto["steps"]])*1e-9
    x = overlaps / 1e-9
    contrasts = np.exp(-0.05 - 200 * step_ts - 0.02 * x)
    slope, se, n = regression_by_duration(contrasts, proto, overlaps)
    assert n > 20 and abs(slope - 0.02) < 1e-6, (slope, se, n)
    print("PASS: synthetic positive 13C bandpower regression slope =", slope,
          "(not real nuclear data)")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--file-stem", "--file-id", dest="file_stem", default=None,
                   help="Exact saved filename WITHOUT .txt/.npz extension")
    p.add_argument("--latest", action="store_true",
                   help="Analyze newest RQS-C13 saved experiment from the NAS")
    p.add_argument("--outdir", default=None,
                   help="Optional LOCAL output directory for debugging; "
                        "default saves into G: through data_manager")
    p.add_argument("--target-khz", type=float, default=None)
    args=p.parse_args()
    if args.self_test:
        self_test()
    elif args.latest or args.file_stem:
        from utils import data_manager as dm
        if args.latest and args.file_stem:
            p.error("Use either --latest or --file-stem, not both.")
        if args.latest:
            src = Path(__file__).resolve().parents[1] / "majorroutines" / "widefield" / "rqs_c13.py"
            folder = dm.get_output_folder(src)
            if not folder.is_dir():
                p.error("NAS experiment folder unavailable: " + str(folder)
                        + ". Run from your normal Purcell session with G: mounted, "
                        + "or supply --file-stem if the NAS index is accessible.")
            candidates = list(folder.glob("*.txt"))
            candidates = [c for c in candidates if c.name.startswith("2026_")]
            if not candidates:
                p.error("No RQS-C13 raw .txt files in " + str(folder))
            most_recent = max(candidates, key=lambda c: c.name)
            stem = most_recent.stem
            print("Newest RQS-C13 file:", most_recent)
        else:
            stem = Path(args.file_stem).stem
            if not stem or stem.isdigit():
                p.error("Supply the complete timestamp-NV filename stem, not a numeric ID.")
        print("Loading counts and metadata for:", stem)
        raw = dm.get_raw_data(file_stem=stem, load_npz=True)
        if not isinstance(raw.get("counts"), np.ndarray):
            raise RuntimeError("Counts not loaded as an array. Verify .npz availability.")
        outdir = Path(args.outdir) / stem if args.outdir else None
        analyze(raw, outdir=outdir, source_stem=stem,
                target_hz=None if args.target_khz is None else args.target_khz*1e3)
    else:
        p.error("Provide --latest, --file-stem STEM, or --self-test.")
