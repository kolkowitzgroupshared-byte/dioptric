"""Plot a saved camera image twice: on its full color scale, and with the color scale
capped so dim light (such as the edge of the yellow beam) becomes visible.

Run from the dioptric folder in the dioptric environment:
    python plot_saved_image.py
        newest image in FOLDER
    python plot_saved_image.py "G:\\nvdata\\...\\some_image.npz"
        a specific saved image
    python plot_saved_image.py "G:\\nvdata\\...\\some_image.npz" 0.05
        cap the color scale at 5 % of the brightest pixel instead of 10 %
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FOLDER = Path(r"G:\nvdata\pc_Purcell\branch_master\image_sample\2026_11")
path = Path(r"G:\nvdata\pc_Purcell\branch_master\image_sample\2026_11\2026_11_09-11_46_10-johnson-nv0_2026_10_02.npz")
CAP_FRACTION = 0.1

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    files = sorted(FOLDER.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"No saved images (.npz) in {FOLDER}")
    path = files[-1]
cap = float(sys.argv[2]) if len(sys.argv) > 2 else CAP_FRACTION

data = np.load(path, allow_pickle=True)
key = "img_array" if "img_array" in data.files else data.files[0]
img = np.asarray(data[key], dtype=float)
print(f"{path.name}: '{key}', min {img.min():.0f}, median {np.median(img):.0f}, max {img.max():.0f}")

low = np.percentile(img, 1)
capped = low + cap * (img.max() - low)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, vmax, label in ((axes[0], img.max(), "full scale"), (axes[1], capped, f"capped at {cap:.0%}")):
    shown = ax.imshow(img, cmap="inferno", vmin=low, vmax=vmax)
    fig.colorbar(shown, ax=ax, label="Counts")
    ax.set_title(f"{path.stem[:19]}, {label}")
fig.tight_layout()
plt.show()
