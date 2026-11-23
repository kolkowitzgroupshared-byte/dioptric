# -*- coding: utf-8 -*-
"""
Does any echo survive in an RQS run? A picture from the raw camera counts (no thresholding).

At 0 V, every X-basis row of a working echo ends in m_s = 0 in the normal analysis phase
("plus") and in m_s = +/-1 with the analysis pi/2 flipped by 180 deg ("minus"), whatever the
row's pulses. So the X rows' minus-minus-plus count difference is the echo signal. Y-basis
rows read the sine part of the phase, which is zero without a tone, so they show nothing
either way. The reference shots (the second experiment of every step) give the m_s = 0 level.

Panel 1: mean counts of X and Y rows in both phases, relative to the reference shots, with
    rough markers for a working echo: plus at the reference, minus one spin-flip step above.
    The step is estimated as twice the signal shots' offset from the reference (the offset of
    a fully mixed spin), so it is only a guide.
Panel 2: each NV's echo signal (X minus minus X plus) against that rough expectation.
Panel 3: the echo signal of the X rows against the time of their first pi pulse, for rows
    with two pi pulses (binned) and with one. If the 14N lines with slow pulses were the
    cause, some pulse positions should still show an echo.

Run from the dioptric folder in the dioptric environment:
    python rqs_contrast_check.py

Created on November 18th, 2026
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings: which block files
# ---------------------------------------------------------------------------
DATA_FOLDER = Path(r"G:\nvdata\pc_Purcell\branch_master\rqs\2026_11")
BUNDLE_NAME = "rqs_16k7_phase_locked"
AMP_V = 0.0
FIRST_GOOD = "2026_11_18-11_22_23-johnson-nv0_2026_10_02"  # the long 0 V run: first block
LAST_GOOD = "2026_11_18-12_07_15-johnson-nv0_2026_10_02"   # ... and after its last block (12_07_15)
EXCLUDE_NVS = (0, 6)
N_POSITION_BINS = 12


def select_files(dm):
    stems = []
    for path in sorted(DATA_FOLDER.glob("*.txt")):
        if path.stem < FIRST_GOOD or path.stem > LAST_GOOD:
            continue
        info = dm.get_raw_data(file_stem=path.stem, load_npz=False)
        amp = info["tones"][0][1] if info.get("tones") else 0.0
        if (str(info.get("bundle_dir", "")).endswith(BUNDLE_NAME)
                and len(info.get("row_ids", [])) > 20 and abs(amp - AMP_V) < 1e-9):
            stems.append(path.stem)
    return stems


def collect(data_list):
    """NV-averaged counts per (row, phase) for the signal shots, per NV per phase for X rows,
    and the reference level."""
    bundle = Path(data_list[0]["bundle_dir"])
    acq = pd.read_csv(bundle / "acquisition.csv").set_index("row_id")
    num_nvs = np.asarray(data_list[0]["counts"]).shape[1]
    nv_inds = [i for i in range(num_nvs) if i not in EXCLUDE_NVS]
    rows = {}                                   # row_id -> [[sum, n] plus, [sum, n] minus]
    per_nv = np.zeros((len(nv_inds), 2, 2))     # NV, phase, (sum, n) for X rows
    groups = {}                                 # (basis, phase) -> list of per-step NV averages
    ref_sum = ref_n = 0.0
    for d in data_list:
        counts = np.asarray(d["counts"], dtype=np.float64)
        counts = np.where(np.isfinite(counts), counts, np.nan)
        sig = counts[0][nv_inds]                # [nv, run, step, rep]
        per_step = np.nanmean(sig, axis=(0, 1, 3))
        per_step_nv = np.nanmean(sig, axis=(1, 3))   # [nv, step]
        steps = np.asarray(d["steps"])
        row_ids = np.asarray(d["row_ids"])[steps >> 1]
        phases = steps & 1
        bases = acq.loc[row_ids, "basis"].to_numpy()
        for rid, ph, val in zip(row_ids, phases, per_step):
            entry = rows.setdefault(rid, [[0.0, 0], [0.0, 0]])
            entry[ph][0] += val
            entry[ph][1] += 1
        for b in ("X", "Y"):
            for ph in (0, 1):
                sel = (bases == b) & (phases == ph)
                groups.setdefault((b, ph), []).extend(per_step[sel].tolist())
                if b == "X":
                    per_nv[:, ph, 0] += np.nansum(per_step_nv[:, sel], axis=1)
                    per_nv[:, ph, 1] += sel.sum()
        if counts.shape[0] > 1:
            ref = counts[1][nv_inds]
            ref_sum += np.nansum(ref)
            ref_n += np.sum(np.isfinite(ref))
    reference = ref_sum / ref_n if ref_n else np.nan
    return bundle, acq, rows, per_nv, groups, reference, nv_inds


def plot(bundle, acq, rows, per_nv, groups, reference):
    pulses = pd.read_csv(bundle / "pulses.csv")
    first = pulses[pulses["pulse_index"] == 0].set_index("sequence_index")["time_s"]
    count = pulses.groupby("sequence_index").size()

    signal_mean = np.mean(np.concatenate([np.asarray(v) for v in groups.values()]))
    step = 2 * (signal_mean - reference)        # rough spin-flip step (fully mixed = half of it)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})

    # 1. mean counts per basis and phase, relative to the reference shots
    ax = axes[0]
    labels, means, errs = [], [], []
    for b in ("X", "Y"):
        for ph, name in ((0, "plus"), (1, "minus")):
            v = np.asarray(groups.get((b, ph), []), float)
            labels.append(f"{b} {name}")
            means.append(v.mean() - reference if v.size else np.nan)
            errs.append(v.std() / np.sqrt(v.size) if v.size > 1 else np.nan)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, color=["tab:blue", "tab:orange", "0.6", "0.75"], capsize=4, label="measured")
    ax.plot([0, 1], [0, step], "_", color="tab:red", ms=34, mew=3, label="working echo (rough)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(min(-0.4, np.nanmin(means) - 0.4), max(step, np.nanmax(means)) * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean counts minus reference shots")
    ax.set_title("X rows: plus and minus should differ", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # 2. each NV's echo signal
    ax = axes[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        nv_signal = per_nv[:, 1, 0] / per_nv[:, 1, 1] - per_nv[:, 0, 0] / per_nv[:, 0, 1]
    nv_signal = nv_signal[np.isfinite(nv_signal)]
    ax.hist(nv_signal, bins=25, color="tab:blue", alpha=0.8, label=f"{nv_signal.size} NVs")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(step, color="k", ls="--", lw=1.2, label=f"working echo (rough): {step:.1f}")
    ax.set_xlabel("echo signal: X minus minus X plus (counts)")
    ax.set_ylabel("NVs")
    ax.set_title(f"per NV: median {np.median(nv_signal):+.2f} counts", fontsize=10)
    ax.legend(fontsize=8)

    # 3. echo signal against the first pi pulse's time
    ax = axes[2]
    t1, diff, npi = [], [], []
    for rid, ((sp, n_p), (sm, n_m)) in rows.items():
        if acq.loc[rid, "basis"] != "X" or n_p == 0 or n_m == 0:
            continue
        seq = acq.loc[rid, "sequence_index"]
        t1.append(float(first.get(seq, np.nan)) * 1e6)
        diff.append(sm / n_m - sp / n_p)
        npi.append(int(count.get(seq, 0)))
    t1, diff, npi = np.array(t1), np.array(diff), np.array(npi)
    two = npi == 2
    edges = np.linspace(np.nanmin(t1[two]), np.nanmax(t1[two]), N_POSITION_BINS + 1)
    idx = np.clip(np.digitize(t1[two], edges) - 1, 0, N_POSITION_BINS - 1)
    centers, m, e = [], [], []
    for k in range(N_POSITION_BINS):
        v = diff[two][idx == k]
        if v.size > 3:
            centers.append(0.5 * (edges[k] + edges[k + 1]))
            m.append(v.mean())
            e.append(v.std() / np.sqrt(v.size))
    ax.errorbar(centers, m, yerr=e, fmt="o", color="tab:blue", label="two pi pulses (binned rows)")
    one = npi == 1
    if one.any():
        ax.errorbar(np.nanmean(t1[one]), diff[one].mean(), yerr=diff[one].std() / np.sqrt(one.sum()),
                    fmt="s", color="tab:orange", label=f"one pi pulse ({one.sum()} rows)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(step, color="k", ls="--", lw=1.2, label="working echo (rough)")
    ax.set_xlabel("time of the first pi pulse in the window (us)")
    ax.set_ylabel("echo signal (counts)")
    ax.set_title("no pulse position keeps an echo?", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"RQS at {AMP_V:g} V: reference shots {reference:.2f} counts, signal shots {signal_mean:.2f}", fontsize=11)
    fig.tight_layout()
    return fig


def main():
    from utils import data_manager as dm

    stems = select_files(dm)
    print(f"{len(stems)} block files:", *stems, sep="\n  ")
    data_list = [dm.get_raw_data(file_stem=stem, load_npz=True) for stem in stems]
    bundle, acq, rows, per_nv, groups, reference, nv_inds = collect(data_list)
    fig = plot(bundle, acq, rows, per_nv, groups, reference)
    out = Path(stems[-1] + "-echo-check.png")
    fig.savefig(out, dpi=150)
    print(f"Figure saved as {out.resolve()}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
