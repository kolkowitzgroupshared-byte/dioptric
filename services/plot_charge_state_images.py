"""Plot the Nov 9 charge-state images (ref, sig, diff) on the same color scales as the
October 22 run, so the two can be compared directly.

Run from the dioptric folder in the dioptric environment:
    python plot_charge_state_images.py
or give another run's .npz file:
    python plot_charge_state_images.py "G:\\nvdata\\...\\some_run.npz"
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

path = (
    r"G:\nvdata\pc_Purcell\branch_master\charge_state_histograms\2026_11"
    r"\2026_11_09-10_59_43-johnson-nv0_2026_10_02.npz"
)
if len(sys.argv) > 1:
    path = sys.argv[1]

data = np.load(path, allow_pickle=True)
print("Arrays in the file:", data.files)

# key, label, color range in photons (as in the October images)
panels = [
    ("ref_img_array", "ref", 1.5, 10),
    ("sig_img_array", "sig", 1.5, 7),
    ("diff_img_array", "diff", -5.5, 0.8),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (key, label, vmin, vmax) in zip(axes, panels):
    if key not in data.files:
        ax.set_title(f"{label}: not in this file")
        ax.axis("off")
        continue
    image = ax.imshow(data[key], cmap="inferno", vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label="Photons")
    ax.set_title(f"{label}, Nov 9 (October color scale)")
fig.tight_layout()
plt.show()
