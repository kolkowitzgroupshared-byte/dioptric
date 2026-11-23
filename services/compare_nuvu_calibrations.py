"""Compare the two NUVU-camera-to-SLM coordinate calibrations: when each was saved,
whether they are identical, and how far the SLM spots would move between them.

Run from the dioptric folder in the dioptric environment:
    python compare_nuvu_calibrations.py
"""

import os
import time

import numpy as np

PATHS = [
    "slmsuite/calibration/nuvu_to_thorcam_slm.npz",  # used by slm_control_panel_purcell.py
    "calibrations/purcell/current/nuvu_to_thorcam_slm.npz",  # listed in the config
]
KEY = "M_nuvu_to_thorcam_slm"
# Reference NV, and two points near opposite corners of the NV area (NUVU pixels)
TEST_POINTS = np.array([[151.893, 147.049], [30.0, 30.0], [260.0, 260.0]])

matrices = []
for path in PATHS:
    data = np.load(path, allow_pickle=True)
    if KEY not in data.files:
        raise KeyError(f"{path} has no '{KEY}'; it contains {data.files}")
    matrix = np.asarray(data[KEY], dtype=float)
    matrices.append(matrix)
    saved = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
    print(f"{path}\n  last modified {saved}\n  matrix:\n{np.round(matrix, 5)}\n")

points = np.c_[TEST_POINTS, np.ones(len(TEST_POINTS))]
spots = [points @ matrix.T for matrix in matrices]
shift = np.linalg.norm(spots[0] - spots[1], axis=1)
print("Identical:", np.allclose(matrices[0], matrices[1]))
print("Spot position difference (SLM camera pixels) at the reference NV and two corners:",
      np.round(shift, 2))
