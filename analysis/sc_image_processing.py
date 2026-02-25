# -*- coding: utf-8 -*-
"""
Image sampl
@auhtor : Saroj Chand
"""

import sys
import traceback
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from numpy.linalg import lstsq
from scipy.optimize import curve_fit, least_squares
from scipy.stats import pearsonr

from utils import _cloud_box as box_cloud
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield


if __name__ == "__main__":
    kpl.init_kplotlib()
    data = dm.get_raw_data(
        file_stem="2026_02_24-15_44_06-qnami-nv0_2026_02_20", load_npz=True
    )
    img_adus = np.array(data["img_array"], dtype=float)

    # 1) Convert ADUs -> photons (estimated)
    img_ph = widefield.adus_to_photons(img_adus)

    # 2) Saturate the top end to reveal low-level features
    vmin = np.nanpercentile(img_ph, 1.0)      # or 0.5
    vmax = np.nanpercentile(img_ph, 99.5)     # try 99.0–99.8

    fig, ax = plt.subplots()
    kpl.imshow(
        ax,
        img_ph,
        cbar_label="Estimated photons",
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    kpl.show(block=True)