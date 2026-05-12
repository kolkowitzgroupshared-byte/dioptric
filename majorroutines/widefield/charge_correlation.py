# -*- coding: utf-8 -*-
"""
Equal-time charge-state correlation measurement for widefield NV arrays.

Goal
----
Repeatedly prepare/drive/read charge states for many NVs, threshold each NV into
NV-/NV0, and compute equal-time spatial correlations:

    C_ij = <s_i s_j> - <s_i><s_j>
    R_ij = C_ij / (sigma_i sigma_j)

where s_i is the binary charge state of NV i.

This is the first step toward spatially resolved charge dynamics / charge-bath
correlation measurements.

Created: 2026-05
"""

import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, VirtualLaserKey


# =============================================================================
# Basic helpers
# =============================================================================


def _get_pixel_coords(nv_list):
    coords = []
    for nv in nv_list:
        coords.append(pos.get_nv_coords(nv, CoordsKey.PIXEL, drift_adjust=False))
    return np.asarray(coords, dtype=float)


def _flatten_states(states):
    """
    Convert states to shape:
        shots x num_nvs

    Expected threshold_counts output is usually:
        states[nv_ind, run_ind, step_ind, rep_ind]
    """
    states = np.asarray(states)

    if states.ndim == 4:
        # nv, run, step, rep -> shots, nv
        num_nvs = states.shape[0]
        return np.moveaxis(states, 0, -1).reshape(-1, num_nvs)

    if states.ndim == 3:
        # nv, run, rep -> shots, nv
        num_nvs = states.shape[0]
        return np.moveaxis(states, 0, -1).reshape(-1, num_nvs)

    if states.ndim == 2:
        # already nv x shots or shots x nv; assume nv x shots if first dim is smaller
        if states.shape[0] < states.shape[1]:
            return states.T
        return states

    raise ValueError(f"Unexpected states shape: {states.shape}")


def _threshold_counts_safe(nv_list, counts, dynamic_thresh=False):
    """
    Threshold charge-readout counts into binary states.

    First tries your normal widefield.threshold_counts path.
    If thresholds are missing and dynamic thresholding fails, falls back to
    per-NV median thresholding.
    """
    try:
        states = widefield.threshold_counts(
            nv_list,
            counts,
            dynamic_thresh=dynamic_thresh,
        )
        return np.asarray(states, dtype=bool), "widefield.threshold_counts"
    except Exception:
        print("widefield.threshold_counts failed; using per-NV median threshold fallback.")
        print(traceback.format_exc())

    counts = np.asarray(counts)
    num_nvs = counts.shape[0]
    flat = counts.reshape(num_nvs, -1)

    thresholds = np.nanmedian(flat, axis=1)
    states_flat = flat > thresholds[:, None]

    states = states_flat.reshape(counts.shape)
    return np.asarray(states, dtype=bool), "median_fallback"


def _pairwise_distances(pixel_coords, pixel_size_um=None):
    """
    Return pairwise distances in pixels or microns.
    """
    pixel_coords = np.asarray(pixel_coords, dtype=float)
    diff = pixel_coords[:, None, :] - pixel_coords[None, :, :]
    dist_px = np.linalg.norm(diff, axis=-1)

    if pixel_size_um is None:
        return dist_px, "pixels"

    return dist_px * float(pixel_size_um), "um"


def _bin_correlation_vs_distance(corr, dist, num_bins=30, min_pairs_per_bin=10):
    """
    Bin upper-triangular correlation values versus distance.
    """
    corr = np.asarray(corr, dtype=float)
    dist = np.asarray(dist, dtype=float)

    iu = np.triu_indices_from(corr, k=1)
    r_vals = dist[iu]
    c_vals = corr[iu]

    finite = np.isfinite(r_vals) & np.isfinite(c_vals)
    r_vals = r_vals[finite]
    c_vals = c_vals[finite]

    bins = np.linspace(np.nanmin(r_vals), np.nanmax(r_vals), num_bins + 1)

    r_centers = []
    c_mean = []
    c_sem = []
    n_pairs = []

    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (r_vals >= b0) & (r_vals < b1)
        vals = c_vals[mask]

        if len(vals) < min_pairs_per_bin:
            continue

        r_centers.append(0.5 * (b0 + b1))
        c_mean.append(np.nanmean(vals))
        c_sem.append(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
        n_pairs.append(len(vals))

    return {
        "r_centers": np.asarray(r_centers),
        "corr_mean": np.asarray(c_mean),
        "corr_sem": np.asarray(c_sem),
        "n_pairs": np.asarray(n_pairs, dtype=int),
    }


def _shuffle_states_per_nv(states_shots_by_nv, rng=None):
    """
    Shuffle time axis independently for each NV.

    This destroys true equal-time correlations while preserving each NV's
    single-NV charge statistics.
    """
    if rng is None:
        rng = np.random.default_rng()

    states = np.asarray(states_shots_by_nv).copy()
    shuffled = np.empty_like(states)

    for nv_ind in range(states.shape[1]):
        shuffled[:, nv_ind] = rng.permutation(states[:, nv_ind])

    return shuffled


# =============================================================================
# Correlation analysis
# =============================================================================


def compute_charge_correlations(
    states_shots_by_nv,
    pixel_coords,
    pixel_size_um=None,
    num_bins=30,
    do_shuffle=True,
):
    """
    Compute equal-time charge correlations and distance-binned correlation.
    """
    S = np.asarray(states_shots_by_nv, dtype=float)

    # Remove shots with all NaNs if needed.
    valid_shots = np.all(np.isfinite(S), axis=1)
    S = S[valid_shots]

    p_nvm = np.nanmean(S, axis=0)
    sigma = np.nanstd(S, axis=0, ddof=1)

    # Pearson correlation matrix.
    corr = np.corrcoef(S, rowvar=False)

    dist, dist_units = _pairwise_distances(pixel_coords, pixel_size_um=pixel_size_um)
    corr_vs_r = _bin_correlation_vs_distance(
        corr,
        dist,
        num_bins=num_bins,
    )

    out = {
        "states_shots_by_nv": S,
        "p_nvm": p_nvm,
        "sigma": sigma,
        "corr": corr,
        "dist": dist,
        "dist_units": dist_units,
        "corr_vs_r": corr_vs_r,
    }

    if do_shuffle:
        S_shuf = _shuffle_states_per_nv(S)
        corr_shuf = np.corrcoef(S_shuf, rowvar=False)
        corr_vs_r_shuf = _bin_correlation_vs_distance(
            corr_shuf,
            dist,
            num_bins=num_bins,
        )

        out |= {
            "states_shuffled": S_shuf,
            "corr_shuffled": corr_shuf,
            "corr_vs_r_shuffled": corr_vs_r_shuf,
        }

    return out


def plot_charge_correlations(corr_data):
    """
    Generate standard charge-correlation plots.
    """
    figs = []

    corr = corr_data["corr"]
    p_nvm = corr_data["p_nvm"]
    corr_vs_r = corr_data["corr_vs_r"]
    dist_units = corr_data["dist_units"]

    # Correlation matrix.
    fig, ax = plt.subplots()
    im = ax.imshow(corr, vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xlabel("NV index")
    ax.set_ylabel("NV index")
    ax.set_title("Equal-time charge-state correlation matrix")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    figs.append(fig)

    # Correlation versus distance.
    fig, ax = plt.subplots()
    ax.errorbar(
        corr_vs_r["r_centers"],
        corr_vs_r["corr_mean"],
        yerr=corr_vs_r["corr_sem"],
        fmt="o-",
        label="Data",
    )

    if "corr_vs_r_shuffled" in corr_data:
        shuf = corr_data["corr_vs_r_shuffled"]
        ax.errorbar(
            shuf["r_centers"],
            shuf["corr_mean"],
            yerr=shuf["corr_sem"],
            fmt="o--",
            label="Shuffle control",
        )

    ax.axhline(0, linestyle="--")
    ax.set_xlabel(f"NV-NV separation ({dist_units})")
    ax.set_ylabel("Mean correlation")
    ax.set_title("Charge correlation versus distance")
    ax.legend()
    figs.append(fig)

    # NV- probability distribution.
    fig, ax = plt.subplots()
    ax.hist(p_nvm, bins=30)
    ax.set_xlabel("NV- probability")
    ax.set_ylabel("Number of NVs")
    ax.set_title("Distribution of charge-state probability")
    figs.append(fig)

    # Per-shot global charge fraction.
    S = corr_data["states_shots_by_nv"]
    global_frac = np.nanmean(S, axis=1)

    fig, ax = plt.subplots()
    ax.plot(global_frac, ".-")
    ax.set_xlabel("Shot index")
    ax.set_ylabel("Mean NV- fraction")
    ax.set_title("Global charge fraction versus shot")
    figs.append(fig)

    return figs


def process_and_plot(raw_data):
    """
    Process raw_data from this experiment and plot charge correlations.
    """
    nv_list = raw_data["nv_list"]
    counts = np.asarray(raw_data["counts"])[0]

    dynamic_thresh = raw_data.get("dynamic_thresh", False)
    pixel_size_um = raw_data.get("pixel_size_um", None)
    num_bins = raw_data.get("num_distance_bins", 30)

    states, threshold_method = _threshold_counts_safe(
        nv_list,
        counts,
        dynamic_thresh=dynamic_thresh,
    )

    states_shots_by_nv = _flatten_states(states)
    pixel_coords = _get_pixel_coords(nv_list)

    corr_data = compute_charge_correlations(
        states_shots_by_nv,
        pixel_coords,
        pixel_size_um=pixel_size_um,
        num_bins=num_bins,
        do_shuffle=True,
    )

    corr_data["threshold_method"] = threshold_method
    figs = plot_charge_correlations(corr_data)

    raw_data |= {
        "states": states,
        "states_shots_by_nv": states_shots_by_nv,
        "threshold_method": threshold_method,
        "p_nvm": corr_data["p_nvm"],
        "charge_corr": corr_data["corr"],
        "charge_corr_shuffled": corr_data.get("corr_shuffled", None),
        "pairwise_dist": corr_data["dist"],
        "dist_units": corr_data["dist_units"],
        "corr_vs_r_r_centers": corr_data["corr_vs_r"]["r_centers"],
        "corr_vs_r_mean": corr_data["corr_vs_r"]["corr_mean"],
        "corr_vs_r_sem": corr_data["corr_vs_r"]["corr_sem"],
        "corr_vs_r_n_pairs": corr_data["corr_vs_r"]["n_pairs"],
    }

    if "corr_vs_r_shuffled" in corr_data:
        shuf = corr_data["corr_vs_r_shuffled"]
        raw_data |= {
            "corr_vs_r_shuffle_r_centers": shuf["r_centers"],
            "corr_vs_r_shuffle_mean": shuf["corr_mean"],
            "corr_vs_r_shuffle_sem": shuf["corr_sem"],
            "corr_vs_r_shuffle_n_pairs": shuf["n_pairs"],
        }

    return figs, corr_data


# =============================================================================
# Main experiment
# =============================================================================


def main(
    nv_list,
    num_reps,
    num_runs,
    do_drive=True,
    targeted_drive=False,
    dynamic_thresh=False,
    pixel_size_um=None,
    num_distance_bins=30,
    save_images=False,
):
    """
    Run equal-time charge-state correlation experiment.

    Parameters
    ----------
    nv_list : list[NVSig]
        NVs to measure.

    num_reps, num_runs : int
        Number of repeated charge snapshots.

    do_drive : bool
        If True, apply charge-polarization/drive pulse before each readout.

    targeted_drive : bool
        If True, use targeted polarization coordinates.
        For first uniform correlation measurement, usually False.

    dynamic_thresh : bool
        Passed to widefield.threshold_counts during analysis.

    pixel_size_um : float or None
        Pixel-to-micron calibration for distance axis.
        If None, distance is plotted in pixels.

    save_images : bool
        For many shots, set False unless debugging.
    """
    seq_file = "charge_correlation.py"
    num_steps = 1
    num_exps = 1

    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(
                nv_list,
                VirtualLaserKey.CHARGE_POL,
            )
        )

        seq_args = [
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            bool(do_drive),
            bool(targeted_drive),
        ]

        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=save_images,
        save_images_avg_reps=False,
        charge_prep_fn=None,
        num_exps=num_exps,
    )

    raw_data |= {
        "experiment": "charge_correlation",
        "num_reps": num_reps,
        "num_runs": num_runs,
        "do_drive": bool(do_drive),
        "targeted_drive": bool(targeted_drive),
        "dynamic_thresh": bool(dynamic_thresh),
        "pixel_size_um": pixel_size_um,
        "num_distance_bins": num_distance_bins,
        "img_array-units": "photons",
    }

    try:
        figs, corr_data = process_and_plot(raw_data)
    except Exception:
        print(traceback.format_exc())
        figs = []

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        f"{repr_nv_name}-charge-correlation",
    )

    keys_to_compress = [
        "states",
        "states_shots_by_nv",
        "p_nvm",
        "charge_corr",
        "charge_corr_shuffled",
        "pairwise_dist",
        "corr_vs_r_r_centers",
        "corr_vs_r_mean",
        "corr_vs_r_sem",
        "corr_vs_r_n_pairs",
        "corr_vs_r_shuffle_r_centers",
        "corr_vs_r_shuffle_mean",
        "corr_vs_r_shuffle_sem",
        "corr_vs_r_shuffle_n_pairs",
    ]
    keys_to_compress = [key for key in keys_to_compress if key in raw_data]

    dm.save_raw_data(raw_data, file_path, keys_to_compress)

    for ind, fig in enumerate(figs):
        fig_path = dm.get_file_path(
            __file__,
            timestamp,
            f"{repr_nv_name}-charge-correlation-{ind}",
        )
        dm.save_figure(fig, fig_path)

    tb.reset_cfm()

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Reload example:
    raw_data = dm.get_raw_data(file_stem="2026_05_11-16_48_56-qnami-nv0_2026_02_20-charge-correlation", 
                               load_npz=True, 
                               allow_pickle=True)
    process_and_plot(raw_data)
    kpl.show(block=True)

    print("Import this module and call charge_correlation.main(nv_list, ...).")