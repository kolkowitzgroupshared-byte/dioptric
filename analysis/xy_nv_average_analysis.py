# -*- coding: utf-8 -*-
"""
Analyze an XY run (majorroutines/widefield/xy.py, no tone): does XY4 in IQ mode keep its
coherence at the operating point of the XY4 tone calibration (tau = 2.5 us, i.e. pi pulses
5 us apart)?

The NV-averaged signal is plotted against tau (center to center, pi pulses 2 tau apart). Three
levels are compared:
  - short taus (up to SHORT_TAU_MAX_NS): the coherent level, before the bath acts
  - the operating point (OPERATING_TAU_NS)
  - the 13C dip, where the pulse spacing 2 tau equals half the 13C precession period
    (tau ~ 1 / (4 f_13C) ~ 7.6 us): the 13C field scrambles the phase there, so the dip
    stands in for the decohered level
"Coherence kept" at the operating point = (S_op - S_dip) / (S_short - S_dip). If the dip
is not clearly below the short-tau level, that ratio is not reported.

A map of every NV against tau shows whether the dip (and any loss) is common to all NVs.
Several files can be combined: list them all.

Run from the dioptric folder in the dioptric environment:
    python xy_nv_average_analysis.py

Created on November 16th, 2026
"""

import re
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FILE_STEMS = ["2026_11_16-18_51_19-johnson-nv0_2026_10_02"]  # the saved .txt names without extension
EXCLUDE_NVS = [0, 6]  # another orientation
THRESHOLD = True
OPERATING_TAU_NS = 2500  # the XY4 tone calibration's pulse spacing is 2 tau = 5 us
SHORT_TAU_MAX_NS = 1500  # taus up to this define the coherent level
F_13C_HZ = 33e3  # 13C precession (spin echo revivals every ~60 us)
DIP_WINDOW_NS = (6000, 9000)  # where to look for the 13C dip
XY_PULSES = {"hahn": 1, "xy2": 2, "xy4": 4, "xy8": 8, "xy16": 16}


def number_of_pi_pulses(xy_seq):
    match = re.match(r"([a-zA-Z]+\d*)(?:-(\d+))?", str(xy_seq).lower())
    return XY_PULSES[match.group(1)] * (int(match.group(2)) if match.group(2) else 1)


def _tidy(fig):
    """tight_layout unless the plotting style already set a layout engine (kplotlib)."""
    get_engine = getattr(fig, "get_layout_engine", None)
    if get_engine is not None and get_engine() is not None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except RuntimeError:
            pass


def _clean(counts):
    """Replace scattered invalid counts (inf/NaN) with the median of the same experiment, NV
    and step over the other runs and reps. Returns counts and the number replaced."""
    bad = ~np.isfinite(counts)
    if not bad.any():
        return counts, 0
    clean = np.where(bad, np.nan, counts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = np.nanmedian(clean, axis=(2, 4), keepdims=True)
    return np.where(bad, np.broadcast_to(fill, counts.shape), counts), int(bad.sum())


def load(file_stems):
    from utils import data_manager as dm

    first, parts = None, []
    for stem in file_stems:
        data = dm.get_raw_data(file_stem=stem, load_npz=True)
        if first is None:
            first = data
        elif not np.allclose(np.asarray(data["taus"], float), np.asarray(first["taus"], float)):
            raise ValueError(f"{stem} has different taus")
        # Saved files store counts as float16 (max 65504): convert first
        parts.append(np.asarray(data["counts"], dtype=np.float64))
    return first, np.concatenate(parts, axis=2)


def nv_average(curves, errors):
    curves, errors = np.asarray(curves, float), np.asarray(errors, float)
    ok = np.isfinite(curves) & np.isfinite(errors) & (errors > 0)
    w = np.where(ok, 1 / np.where(errors > 0, errors, 1) ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(ok, curves, 0) * w).sum(axis=0) / w.sum(axis=0)
        err = 1 / np.sqrt(w.sum(axis=0))
    return mean, err


def levels(taus_ns, mean, err):
    """Coherent level, operating point, 13C dip, and the coherence kept at the operating point."""
    taus_ns = np.asarray(taus_ns, float)
    short = taus_ns <= SHORT_TAU_MAX_NS
    s_short = float(np.average(mean[short], weights=1 / err[short] ** 2))
    e_short = float(1 / np.sqrt(np.sum(1 / err[short] ** 2)))
    op = int(np.argmin(np.abs(taus_ns - OPERATING_TAU_NS)))
    window = (taus_ns >= DIP_WINDOW_NS[0]) & (taus_ns <= DIP_WINDOW_NS[1])
    out = {"short": (s_short, e_short), "op": (float(mean[op]), float(err[op]), float(taus_ns[op]))}
    if window.any():
        # the dip's extreme in the direction away from the coherent level
        idx = np.flatnonzero(window)
        away = np.abs(mean[idx] - s_short)
        d = idx[int(np.argmax(away))]
        out["dip"] = (float(mean[d]), float(err[d]), float(taus_ns[d]))
        depth = s_short - mean[d]
        depth_err = np.hypot(e_short, err[d])
        out["dip_significance"] = float(abs(depth) / depth_err)
        if abs(depth) > 5 * depth_err:
            kept = (mean[op] - mean[d]) / depth
            kept_err = abs(kept) * np.hypot(np.hypot(err[op], err[d]) / max(abs(mean[op] - mean[d]), 1e-12),
                                            depth_err / abs(depth))
            out["kept"] = (float(kept), float(kept_err))
    return out


def report(info, n_nvs, n_runs, n_pi, xy_seq, replaced):
    print(f"\n{xy_seq}: {n_pi} pi pulses, {n_nvs} NVs (excluded {EXCLUDE_NVS}), {n_runs} runs")
    if replaced:
        print(f"  Data: {replaced} invalid count values replaced by medians of the same NV and step")
    s, e = info["short"]
    print(f"  Coherent level (tau up to {SHORT_TAU_MAX_NS / 1e3:g} us): {s:.3f} +/- {e:.3f}")
    v, ve, t = info["op"]
    print(f"  Operating point (tau {t / 1e3:g} us, pulses {2 * t / 1e3:g} us apart, total {2 * n_pi * t / 1e3:g} us): "
          f"{v:.3f} +/- {ve:.3f}")
    if "dip" in info:
        d, de, dt = info["dip"]
        print(f"  13C dip: {d:.3f} +/- {de:.3f} at tau {dt / 1e3:g} us "
              f"(expected near {1e6 / (4 * F_13C_HZ):.1f} us), {info['dip_significance']:.1f} sigma from the coherent level")
    if "kept" in info:
        k, ke = info["kept"]
        print(f"  Coherence kept at the operating point: {100 * k:.0f} +/- {100 * ke:.0f} % "
              f"(of the way from the dip to the coherent level)")
        if k > 0.8:
            print("  -> XY4 keeps its coherence at 2.5 us: its silence at 100 kHz was not about the pulses.")
        elif k < 0.5:
            print("  -> XY4 loses much of its coherence at 2.5 us: the pulses (or IQ timing) are the likely problem.")
    else:
        print("  The dip is not clearly below the coherent level, so the coherence kept can't be quantified.")


def plot(taus_ns, mean, err, curves, nv_numbers, info, n_pi, xy_seq):
    import matplotlib.pyplot as plt

    t_us = np.asarray(taus_ns, float) * 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    ax.errorbar(t_us, mean, yerr=err, fmt="o-", ms=3, lw=0.8, label="NV average")
    ax.axvspan(t_us.min(), SHORT_TAU_MAX_NS / 1e3, color="0.9", label="coherent level region")
    ax.axhline(info["short"][0], color="0.5", lw=0.8)
    ax.axvline(OPERATING_TAU_NS / 1e3, color="tab:green", ls="--", lw=1, label="operating point (XY4 tone run)")
    ax.axvline(1e6 / (4 * F_13C_HZ), color="tab:red", ls=":", lw=1, label="13C spacing (dip expected)")
    ax.set_xlabel("tau (us), center to center; pi pulses are 2 tau apart")
    ax.set_ylabel("Normalized NV population, NV average")
    title = f"{xy_seq}: {n_pi} pi pulses"
    if "kept" in info:
        title += f"; coherence kept at 2.5 us: {100 * info['kept'][0]:.0f} %"
    ax.set_title(title)
    ax.legend(fontsize=8)
    top = ax.secondary_xaxis("top", functions=(lambda t: 2 * n_pi * t, lambda T: T / (2 * n_pi)))
    top.set_xlabel("total time (us)")

    ax = axes[1]
    # each NV relative to its own coherent level, so the map shows changes, not offsets
    short = np.asarray(taus_ns, float) <= SHORT_TAU_MAX_NS
    rel = curves - np.nanmedian(curves[:, short], axis=1, keepdims=True)
    lim = np.nanpercentile(np.abs(rel), 98)
    im = ax.imshow(rel, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                   extent=[t_us.min(), t_us.max(), len(curves) - 0.5, -0.5])
    ax.axvline(OPERATING_TAU_NS / 1e3, color="tab:green", ls="--", lw=1)
    ax.axvline(1e6 / (4 * F_13C_HZ), color="tab:red", ls=":", lw=1)
    ax.set_xlabel("tau (us)")
    ax.set_ylabel("NV (row; in order of the NV list)")
    ax.set_title("Each NV relative to its own short-tau level")
    fig.colorbar(im, ax=ax, label="Change in normalized population")
    ticks = np.linspace(0, len(nv_numbers) - 1, min(8, len(nv_numbers))).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(nv_numbers[i]) for i in ticks])
    _tidy(fig)
    return fig


def analyze(file_stems=FILE_STEMS):
    import matplotlib.pyplot as plt

    from utils import data_manager as dm
    from utils import widefield

    data, counts = load(file_stems)
    counts, replaced = _clean(counts)
    nv_list, taus = data["nv_list"], np.asarray(data["taus"], float)
    xy_seq = data.get("xy_seq", "xy4-3")
    n_pi = number_of_pi_pulses(xy_seq)
    keep = [i for i in range(len(nv_list)) if i not in set(EXCLUDE_NVS)]
    processed = widefield.process_counts([nv_list[i] for i in keep], counts[0][keep], counts[1][keep],
                                         threshold=THRESHOLD)
    curves, errors = np.asarray(processed[0], float), np.asarray(processed[1], float)
    mean, err = nv_average(curves, errors)
    info = levels(taus, mean, err)
    report(info, len(keep), counts.shape[2], n_pi, xy_seq, replaced)
    fig = plot(taus, mean, err, curves, keep, info, n_pi, xy_seq)

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, f"{file_stems[0]}-xy-analysis")
    summary = {
        "source_files": list(file_stems),
        "xy_seq": xy_seq,
        "n_pi": n_pi,
        "nv_indices": keep,
        "taus_ns": taus.tolist(),
        "nv_average": mean.tolist(),
        "nv_average_err": err.tolist(),
        "coherent_level": info["short"],
        "operating_point": info["op"],
        "dip": info.get("dip"),
        "coherence_kept": info.get("kept"),
        "invalid_values_replaced": replaced,
    }
    dm.save_raw_data(summary, file_path)
    dm.save_figure(fig, file_path)
    print(f"\nSaved to {file_path}")
    plt.show(block=True)
    return info


if __name__ == "__main__":
    analyze()
