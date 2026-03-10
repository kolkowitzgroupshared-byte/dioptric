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
    # data = dm.get_raw_data(
    #     file_stem="2026_02_25-14_17_38-qnami-nv0_2026_02_20", load_npz=True
    # )
    # data = dm.get_raw_data(
    #     file_stem="2026_03_03-17_50_36-qnami-nv0_2026_02_20", load_npz=True
    # )
    # data = dm.get_raw_data(
    #     file_stem="2026_03_08-11_19_47-qnami-nv0_2026_02_20", load_npz=True
    # )
    
    # img_adus = np.array(data["ref_img_array"], dtype=float)

    # # 1) Convert ADUs -> photons (estimated)
    # img_ph = widefield.adus_to_photons(img_adus)

    # # 2) Saturate the top end to reveal low-level features
    # vmin = np.nanpercentile(img_ph,0.5)      # or 0.5
    # vmax = np.nanpercentile(img_ph, 99.9)     # try 99.0–99.8

    # fig, ax = plt.subplots()
    # kpl.imshow(
    #     ax,
    #     img_ph,
    #     cbar_label="Estimated photons",
    # )

    data = dm.get_raw_data(
        file_stem="2026_03_08-11_19_47-qnami-nv0_2026_02_20", load_npz=True
    )

    img_ph = np.array(data["ref_img_array"], dtype=float)

    # 1) Convert ADUs -> photons
    # img_ph = widefield.adus_to_photons(img_adus)

    # -------------------------------------------------
    # Mask the bright center spot with NaN
    # -------------------------------------------------
    img_plot = img_ph.copy()

    # Option A: use the image center
    y0 = 204.083
    x0 = 221.37

    # Option B: use the brightest pixel instead
    # y0, x0 = np.unravel_index(np.nanargmax(img_plot), img_plot.shape)

    radius = 12  # adjust as needed

    yy, xx = np.ogrid[:img_plot.shape[0], :img_plot.shape[1]]
    mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius ** 2

    img_plot[mask] = np.nan

    # 2) Compute display range after masking
    vmin = np.nanpercentile(img_plot, 0.5)
    vmax = np.nanpercentile(img_plot, 99.9)

    # 3) Plot
    fig, ax = plt.subplots()
    kpl.imshow(
        ax,
        img_plot,
        cbar_label="Estimated photons",
    )

    # Optional: draw the removed region
    circ = plt.Circle((x0, y0), radius, edgecolor="None", facecolor="black", linewidth=1.5)
    ax.add_patch(circ)

    # save high-resolution figure
    fig.savefig(
        "center_spot_removed.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show(block=True)

    sys.exit()
    file_stems = [
        "2026_03_03-17_45_13-qnami-nv0_2026_02_20",
        "2026_03_03-17_37_30-qnami-nv0_2026_02_20",
        "2026_03_03-17_32_09-qnami-nv0_2026_02_20",
        "2026_03_03-17_45_13-qnami-nv0_2026_02_20",
        "2026_03_03-18_26_27-qnami-nv0_2026_02_20",
        "2026_03_03-17_50_36-qnami-nv0_2026_02_20",
        "2026_03_03-18_26_27-qnami-nv0_2026_02_20"
    ]
    

    img_list = []

    for stem in file_stems:
        try:
            data = dm.get_raw_data(file_stem=stem, load_npz=True)

            if "img_array" not in data:
                print(f"[skip] {stem}: no 'img_array' key")
                continue

            img = np.asarray(data["img_array"], dtype=float)

            # If img has extra dimensions, squeeze them (optional; remove if you need them)
            img = np.squeeze(img)

            if img.ndim != 2:
                print(f"[skip] {stem}: img_array has shape {img.shape}, expected 2D")
                continue

            img_list.append(img)

        except Exception as e:
            print(f"[skip] {stem}: {e}")

    if len(img_list) == 0:
        print("No valid images found.")
    else:
        # Stack -> (n_images, H, W), then max-project
        img_stack = np.stack(img_list, axis=0)
        combined_img = np.max(img_stack, axis=0)

        vmin = np.nanpercentile(img_ph,96.0)      # or 0.5
        vmax = np.nanpercentile(img_ph, 99)     # try 99.0–99.8

        fig, ax = plt.subplots()
        kpl.imshow(
            ax,
            img_ph,
            cbar_label="Estimated photons",
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            no_cbar=True,
        )
        ax.axis("off")
        kpl.show(block=True)