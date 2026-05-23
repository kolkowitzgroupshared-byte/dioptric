# -*- coding: utf-8 -*-
"""
Plot saved stationary-count data.

Loads the tab-separated text file streamed by
majorroutines/confocal/confocal_stationary_count.py (when save_data=True) and
plots raw counts vs. time alongside a kcps panel computed from the readout
duration recorded in the file header.

Creator: chemistatcode
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(path):
    """Load a stationary-count text file.

    Returns
    -------
    sample_index : np.ndarray
    time_s : np.ndarray
    counts : np.ndarray
    metadata : dict
        Parsed key/value pairs from the `# key: value` header lines.
    """
    path = Path(path)
    metadata = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" in body:
                key, val = body.split(":", 1)
                metadata[key.strip()] = val.strip()

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    sample_index = data[:, 0].astype(int)
    time_s = data[:, 1]
    counts = data[:, 2]
    return sample_index, time_s, counts, metadata


def main():
    # Edit these two for the run you want to plot
    data_dir = r"G:\nvdata\pc_cryo\branch_master\confocal_stationary_count\2026_05"
    base_name = "2026_05_14-12_00_00-(nv)"

    path = Path(data_dir) / f"{base_name}.txt"
    sample_index, time_s, counts, metadata = load_data(path)

    nv_name = metadata.get("nv_name", "nv")
    timestamp = metadata.get("timestamp", "")
    readout_ns = float(metadata.get("readout_ns", "nan"))

    print(f"Loaded {len(counts)} samples from {path.name}")
    print(f"  nv_name = {nv_name}")
    print(f"  timestamp = {timestamp}")
    print(f"  readout_ns = {readout_ns}")
    if len(counts) > 0:
        print(f"  duration = {time_s[-1]:.3f} s")
        print(f"  mean counts = {np.nanmean(counts):.3f}")
        print(f"  std counts = {np.nanstd(counts):.3f}")

    readout_s = readout_ns * 1e-9
    if np.isfinite(readout_s) and readout_s > 0:
        kcps = counts / readout_s / 1e3
        has_kcps = True
    else:
        kcps = None
        has_kcps = False

    n_panels = 2 if has_kcps else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, 6 if has_kcps else 4), sharex=True
    )
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(time_s, counts, "-", color="navy", linewidth=0.8)
    axes[0].set_ylabel("Raw counts")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title(f"Stationary count — {nv_name} ({timestamp})")

    if has_kcps:
        axes[1].plot(time_s, kcps, "-", color="darkorange", linewidth=0.8)
        axes[1].set_ylabel("kcps")
        axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
