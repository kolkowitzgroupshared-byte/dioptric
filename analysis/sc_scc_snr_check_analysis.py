# -*- coding: utf-8 -*-
"""
Lightweight check of SCC SNR.

Plots SCC SNR versus:
    - SCC AOD amplitude multiplier, or
    - SCC duration

Created Fall 2024
Updated Sep 2026

@author: Saroj Chand
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.constants import CoordsKey, VirtualLaserKey
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import widefield


# =============================================================================
# USER SETTINGS
# =============================================================================

# FILE_STEM = "2026_09_15-22_34_35-qnami-nv0_2026_02_20"
# FILE_STEM = "2026_09_15-23_47_59-qnami-nv0_2026_02_20"
# FILE_STEM = "2026_09_16-10_27_40-qnami-nv0_2026_02_20"
# FILE_STEM = "2026_09_16-11_34_19-qnami-nv0_2026_02_20"
# FILE_STEM = "2026_09_16-13_07_07-qnami-nv0_2026_02_20"
FILE_STEM = "2026_09_16-14_02_21-qnami-nv0_2026_02_20"

X_AXIS = "scc_duration"       # "scc_amp" or "scc_duration"
STEP_IND = 0
APPLY_THRESHOLD = True

SAVE_RESULTS = True
SAVE_CSV = False
SAVE_FIGURE = True

SAVE_BASENAME = "scc_snr_check_analysis"


ROI_CENTER = (125, 125)


# =============================================================================
# ANALYSIS
# =============================================================================


def process_data(data):

    nv_list = data["nv_list"]
    num_nvs = len(nv_list)

    counts = np.asarray(data["counts"])
    sig_counts = counts[0]
    ref_counts = counts[1]

    # Threshold charge-state counts
    if APPLY_THRESHOLD:
        sig_counts, ref_counts = widefield.threshold_counts(
            nv_list,
            sig_counts,
            ref_counts,
            dynamic_thresh=False,
        )

    # Calculate metrics
    avg_sig, avg_sig_ste, _ = widefield.average_counts(sig_counts)
    avg_ref, avg_ref_ste, _ = widefield.average_counts(ref_counts)

    avg_snr, avg_snr_ste = widefield.calc_snr(
        sig_counts,
        ref_counts,
    )

    avg_contrast, avg_contrast_ste = widefield.calc_contrast(
        sig_counts,
        ref_counts,
    )

    # Select one step
    avg_sig = avg_sig[:, STEP_IND]
    avg_sig_ste = avg_sig_ste[:, STEP_IND]

    avg_ref = avg_ref[:, STEP_IND]
    avg_ref_ste = avg_ref_ste[:, STEP_IND]

    avg_snr = avg_snr[:, STEP_IND]
    avg_snr_ste = avg_snr_ste[:, STEP_IND]

    avg_contrast = avg_contrast[:, STEP_IND]
    avg_contrast_ste = avg_contrast_ste[:, STEP_IND]

    # -------------------------------------------------------------------------
    # Per-NV parameters
    # -------------------------------------------------------------------------

    scc_durations = []
    scc_amps = []

    pixel_x = []
    pixel_y = []
    distances = []

    cx, cy = ROI_CENTER

    for nv in nv_list:

        coords = pos.get_nv_coords(
            nv,
            coords_key=CoordsKey.PIXEL,
            drift_adjust=False,
        )

        x = float(coords[0])
        y = float(coords[1])

        pixel_x.append(x)
        pixel_y.append(y)

        distances.append(
            np.hypot(x - cx, y - cy)
        )

        # SCC duration
        scc_durations.append(
            pos.get_nv_pulse_duration(
                nv,
                VirtualLaserKey.SCC,
            )
        )

        # SCC AOD multiplier
        amp = nv.pulse_amps.get(
            VirtualLaserKey.SCC,
            np.nan,
        )

        scc_amps.append(amp)

    # -------------------------------------------------------------------------
    # Yellow laser power
    # -------------------------------------------------------------------------

    waveforms = data["opx_config"]["waveforms"]

    readout_sample = waveforms[
        "yellow_charge_readout"
    ]["sample"]

    spin_pol_sample = waveforms[
        "yellow_spin_pol"
    ]["sample"]

    a, b, c = 1.5133e4, 2.6976, -38.63

    readout_power = a * readout_sample**b + c
    spin_pol_power = a * spin_pol_sample**b + c

    # -------------------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------------------

    df = pd.DataFrame(
        {
            "NV Index": np.arange(num_nvs),

            "Signal Counts": avg_sig,
            "Signal STE": avg_sig_ste,

            "Reference Counts": avg_ref,
            "Reference STE": avg_ref_ste,

            "SNR": avg_snr,
            "SNR STE": avg_snr_ste,

            "Contrast": avg_contrast,
            "Contrast STE": avg_contrast_ste,

            "SCC Duration (ns)": scc_durations,
            "SCC AOD Multiplier": scc_amps,

            "Pixel X": pixel_x,
            "Pixel Y": pixel_y,
            "Distance": distances,
        }
    )

    metadata = {
        "num_nvs": num_nvs,
        "readout_power_uw": float(readout_power),
        "spin_pol_power_uw": float(spin_pol_power),
    }

    return df, metadata


# =============================================================================
# PLOT
# =============================================================================


def plot_snr(df, metadata):

    if X_AXIS == "scc_amp":
        x = df["SCC AOD Multiplier"]
        xlabel = "SCC AOD amplitude multiplier"

    elif X_AXIS == "scc_duration":
        x = df["SCC Duration (ns)"]
        xlabel = "SCC duration (ns)"

    else:
        raise ValueError(
            "X_AXIS must be 'scc_amp' or 'scc_duration'"
        )

    snr = df["SNR"].to_numpy()
    snr_ste = df["SNR STE"].to_numpy()

    valid = (
        np.isfinite(x)
        & np.isfinite(snr)
        & np.isfinite(snr_ste)
    )

    x = np.asarray(x)[valid]
    snr = snr[valid]
    snr_ste = snr_ste[valid]

    median_snr = np.nanmedian(snr)
    mean_snr = np.nanmean(snr)

    print()
    print("=" * 55)
    print("SCC SNR SUMMARY")
    print("=" * 55)
    print(f"Number of NVs:     {metadata['num_nvs']}")
    print(f"Valid NVs:         {len(snr)}")
    print(f"Median SNR:        {median_snr:.4f}")
    print(f"Mean SNR:          {mean_snr:.4f}")
    print(
        f"Readout power:     "
        f"{metadata['readout_power_uw']:.1f} uW"
    )
    print(
        f"Spin-pol power:    "
        f"{metadata['spin_pol_power_uw']:.1f} uW"
    )
    print("=" * 55)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        x,
        snr,
        yerr=snr_ste,
        fmt="o",
        markersize=4,
        capsize=2,
        alpha=0.7,
        label=f"Median SNR = {median_snr:.3f}",
    )

    ax.axhline(
        median_snr,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("SCC SNR")

    ax.set_title(
        f"SCC SNR of {metadata['num_nvs']} NVs\n"
        f"Readout = {metadata['readout_power_uw']:.0f} µW, "
        f"spin pol = {metadata['spin_pol_power_uw']:.0f} µW"
    )

    ax.grid(alpha=0.25)
    ax.legend()

    return fig


# =============================================================================
# SAVE
# =============================================================================


def save_results(df, metadata, fig):

    timestamp = dm.get_time_stamp()

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        SAVE_BASENAME,
    )

    results = {
        "source_file_stem": FILE_STEM,
        "num_nvs": metadata["num_nvs"],

        "nv_index": df["NV Index"].tolist(),

        "scc_aod_multiplier":
            df["SCC AOD Multiplier"].tolist(),

        "scc_duration_ns":
            df["SCC Duration (ns)"].tolist(),

        "snr": df["SNR"].tolist(),
        "snr_ste": df["SNR STE"].tolist(),

        "contrast": df["Contrast"].tolist(),
        "contrast_ste": df["Contrast STE"].tolist(),

        "signal_counts":
            df["Signal Counts"].tolist(),

        "reference_counts":
            df["Reference Counts"].tolist(),

        "readout_power_uw":
            metadata["readout_power_uw"],

        "spin_pol_power_uw":
            metadata["spin_pol_power_uw"],

        "median_snr":
            float(np.nanmedian(df["SNR"])),

        "mean_snr":
            float(np.nanmean(df["SNR"])),
    }

    if SAVE_RESULTS:
        dm.save_raw_data(
            results,
            file_path,
        )
        print(f"\nSaved analysis: {file_path}")

    if SAVE_CSV:
        df.to_csv(
            f"{file_path}.csv",
            index=False,
        )

    if SAVE_FIGURE:
        fig.savefig(
            f"{file_path}.png",
            dpi=300,
            bbox_inches="tight",
        )


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":

    kpl.init_kplotlib()

    print(f"Loading: {FILE_STEM}")

    data = dm.get_raw_data(
        file_stem=FILE_STEM,
        load_npz=True,
    )

    df, metadata = process_data(data)

    print()
    print(
        df[
            [
                "NV Index",
                "SCC AOD Multiplier",
                "SCC Duration (ns)",
                "SNR",
                "SNR STE",
            ]
        ]
    )

    fig = plot_snr(
        df,
        metadata,
    )

    if SAVE_RESULTS or SAVE_CSV or SAVE_FIGURE:
        save_results(
            df,
            metadata,
            fig,
        )

    kpl.show(block=True)