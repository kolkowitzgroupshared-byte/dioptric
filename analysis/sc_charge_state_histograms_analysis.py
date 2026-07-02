# -*- coding: utf-8 -*-
"""
Reference-only multi-NV charge-state histogram analysis.

This script analyzes widefield charge-state histogram data where:

    counts[0] = signal branch, with ionization pulse
    counts[1] = reference branch, without ionization pulse

Important:
    All fitted parameters come from the reference/no-ionization branch only.

The signal/ionization branch is only plotted for visual comparison. It is not used
to estimate:
    - number of NVs per pillar
    - thresholds
    - p_minus
    - rate0
    - delta
    - fidelities
    - feedback parameters

This is useful when one pillar/spot may contain more than one NV.

Created: Fall 2024
Updated: June 2026
@author: Saroj Chand
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import traceback
from typing import Optional
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
from math import comb
import numpy as np

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    analyze_charge_histogram_multinv_binomial,
)

from utils import widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


from analysis import bimodal_histogram
from analysis.bimodal_histogram import ProbDist
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils import widefield
from utils.constants import VirtualLaserKey

# =============================================================================
# Plotting
# =============================================================================

def plot_histograms(
    sig_counts_list,
    ref_counts_list,
    no_title=True,
    ax=None,
    density=False,
):
    """
    Plot signal/reference histograms.

    sig_counts_list:
        With ionization pulse. Used only visually.

    ref_counts_list:
        Without ionization pulse. Used for analysis/fitting.

    No background subtraction is applied.
    """

    laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout = laser_dict["duration"]
    readout_ms = int(readout / 1e6)

    labels = [
        "With ionization pulse",
        "Without ionization pulse",
    ]

    # Your previous color code
    colors = [
        kpl.KplColors.RED,
        kpl.KplColors.GREEN,
    ]

    counts_lists = [
        sig_counts_list,
        ref_counts_list,
    ]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if not no_title:
        ax.set_title(f"Charge-state histograms, readout = {readout_ms} ms")

    ax.set_xlabel("Integrated counts")

    if density:
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("Number of occurrences")

    for ind in range(2):
        counts_list = counts_lists[ind]

        if counts_list is None or len(counts_list) == 0:
            continue

        kpl.histogram(
            ax,
            counts_list,
            color=colors[ind],
            density=density,
            label=labels[ind],
        )

    ax.set_xlim(-0.5, None)
    ax.legend(loc=kpl.Loc.UPPER_RIGHT)

    if fig is not None:
        return fig



# =============================================================================
# Helper functions
# =============================================================================

def safe_float(x):
    """Convert to float safely."""
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


def make_json_safe(obj):
    """
    Convert analysis objects to JSON/orjson-safe objects.

    This avoids recursion/serialization problems from numpy arrays,
    numpy scalars, and nested objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Avoid saving arbitrary Python objects.
    return str(obj)

def classify_multinv_counts(counts, thresholds):
    """
    Classify each shot into k = number of NV- at this pillar.

    thresholds length = N, separating:
        k = 0 | k = 1 | k = 2 | ... | k = N

    Example:
        thresholds = [t01, t12, t23]

        counts <= t01              -> k = 0
        t01 < counts <= t12        -> k = 1
        t12 < counts <= t23        -> k = 2
        counts > t23               -> k = 3

    Returns
    -------
    k_est : ndarray[int]
        Estimated number of NV- for each shot.
    """

    counts = np.asarray(counts, dtype=float).flatten()

    if thresholds is None:
        return np.full(counts.shape, -1, dtype=int)

    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.size == 0 or np.any(~np.isfinite(thresholds)):
        return np.full(counts.shape, -1, dtype=int)

    return np.searchsorted(thresholds, counts, side="right").astype(int)


def summarize_ref_classification(
    ref_counts,
    threshold_any,
    thresholds_multiclass,
    n_nvs,
):
    """
    Analyze only the reference/no-ionization branch.

    Returns
    -------
    dict with:
        p_any_minus:
            Probability that at least one NV is NV-.

        mean_num_minus:
            Mean estimated number of NV- in this pillar.

        prob_k:
            Probability of k NV- for k = 0, 1, ..., N.

        k_est:
            Shot-by-shot estimated number of NV-.
    """

    ref_counts = np.asarray(ref_counts, dtype=float).flatten()
    n_nvs = int(n_nvs)

    # Binary classification: at least one NV-?
    if threshold_any is None or not np.isfinite(threshold_any):
        p_any_minus = np.nan
    else:
        p_any_minus = float(np.mean(ref_counts > threshold_any))

    # Multi-class classification: k = 0, 1, ..., N NV-.
    k_est = classify_multinv_counts(ref_counts, thresholds_multiclass)

    prob_k = np.zeros(n_nvs + 1, dtype=float)
    for k in range(n_nvs + 1):
        prob_k[k] = float(np.mean(k_est == k))

    good = k_est >= 0
    if np.any(good):
        mean_num_minus = float(np.mean(k_est[good]))
    else:
        mean_num_minus = np.nan

    return {
        "p_any_minus": p_any_minus,
        "mean_num_minus": mean_num_minus,
        "prob_k": prob_k,
        "k_est": k_est,
    }


def feedback_classify_count(counts, threshold_any, thresholds_multiclass, n_nvs_est):
    """
    Helper for future feedback experiments.

    Parameters
    ----------
    counts : float or array
        Measured integrated counts for one pillar.

    threshold_any : float
        Binary threshold. counts > threshold_any means at least one NV is NV-.

    thresholds_multiclass : list or array
        Multi-class thresholds.

    n_nvs_est : int
        Estimated total number of NVs in this pillar.

    Returns
    -------
    dict
    """

    counts = np.asarray(counts, dtype=float)

    is_any_nv_minus = counts > threshold_any
    k_est = np.searchsorted(thresholds_multiclass, counts, side="right")
    is_fully_initialized = k_est >= int(n_nvs_est)

    return {
        "is_any_nv_minus": is_any_nv_minus,
        "k_est": k_est,
        "is_fully_initialized": is_fully_initialized,
    }


# =============================================================================
# Main reference-only multi-NV analysis
# =============================================================================
def fit_one_pillar_multinv_job(
    ind,
    ref_counts_list,
    prob_dist_name,
    max_nvs_per_position,
    force_nvs,
    bic_extra_nv_penalty,
):
    """
    Parallel worker for one pillar.

    Important:
    This function must be defined at top level, outside process_and_plot,
    especially on Windows.
    """

    try:
        prob_dist = ProbDist[prob_dist_name]

        ref_counts_list = np.asarray(ref_counts_list, dtype=float).flatten()

        fit = analyze_charge_histogram_multinv_binomial(
            ref_counts_list,
            prob_dist=prob_dist,
            max_nvs=max_nvs_per_position,
            force_nvs=force_nvs,
            bic_extra_nv_penalty=bic_extra_nv_penalty,
            seed=ind,
        )

        if not fit.get("ok", False):
            return {
                "ind": int(ind),
                "ok": False,
                "fit": None,
                "ref_summary": None,
                "error": None,
            }

        n_est = int(fit["n_nvs"])
        threshold_any = safe_float(fit["threshold_any"])
        thresholds_multiclass = np.asarray(fit["thresholds"], dtype=float)

        ref_summary = summarize_ref_classification(
            ref_counts=ref_counts_list,
            threshold_any=threshold_any,
            thresholds_multiclass=thresholds_multiclass,
            n_nvs=n_est,
        )

        return {
            "ind": int(ind),
            "ok": True,
            "fit": fit,
            "ref_summary": ref_summary,
            "error": None,
        }

    except Exception:
        return {
            "ind": int(ind),
            "ok": False,
            "fit": None,
            "ref_summary": None,
            "error": traceback.format_exc(),
        }

def process_and_plot(
    raw_data,
    do_plot_histograms=False,
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
    max_nvs_per_position: int = 3,
    force_nvs: Optional[int] = None,
    bic_extra_nv_penalty: float = 2.0,
    save_analysis: bool = False,
    save_hist_figs: bool = False,
    n_jobs: int = 12,
    joblib_verbose: int = 10,
):
    """
    Reference-only multi-NV charge-state histogram analysis.

    Important
    ---------
    All fit parameters, thresholds, N estimates, and feedback parameters are
    extracted from the reference/no-ionization branch only.

    Signal branch:
        counts[0] = with ionization pulse.
        Used only for plotting.

    Reference branch:
        counts[1] = without ionization pulse.
        Used for all analysis.

    Outputs are saved into:
        raw_data["charge_hist_multinv_binomial"]

    Parallelization
    ---------------
    The expensive per-pillar fitting is parallelized using joblib.

    For your i9-12900K:
        start with n_jobs=12
        then test n_jobs=8, 10, 14, 16
    """

    nv_list = raw_data["nv_list"]
    num_positions = len(nv_list)

    counts = np.asarray(raw_data["counts"])

    if counts.shape[0] < 2:
        raise ValueError(
            "Expected raw_data['counts'][0] = signal and "
            "raw_data['counts'][1] = reference."
        )

    sig_counts_lists = [counts[0, ind].flatten() for ind in range(num_positions)]
    ref_counts_lists = [counts[1, ind].flatten() for ind in range(num_positions)]

    # -------------------------------------------------------------------------
    # Fixed-size arrays aligned with original pillar / nv_list index.
    # -------------------------------------------------------------------------
    ok_arr = np.full(num_positions, False, dtype=bool)

    n_nvs_est_arr = np.full(num_positions, np.nan)
    threshold_any_arr = np.full(num_positions, np.nan)

    fidelity_any_arr = np.full(num_positions, np.nan)
    fidelity_multiclass_arr = np.full(num_positions, np.nan)
    prep_fidelity_any_ref_arr = np.full(num_positions, np.nan)

    p_minus_arr = np.full(num_positions, np.nan)
    bg_arr = np.full(num_positions, np.nan)
    rate0_arr = np.full(num_positions, np.nan)
    delta_arr = np.full(num_positions, np.nan)
    red_chi_sq_arr = np.full(num_positions, np.nan)

    model_list = [None for _ in range(num_positions)]
    best_candidate_model_list = [None for _ in range(num_positions)]
    best_candidate_bic_arr = np.full(num_positions, np.nan)
    best_equal_bic_arr = np.full(num_positions, np.nan)
    unequal_2nv_beats_equal_arr = np.full(num_positions, False, dtype=bool)
    candidate_results_list = [None for _ in range(num_positions)]

    ref_p_any_minus_arr = np.full(num_positions, np.nan)
    ref_mean_num_minus_arr = np.full(num_positions, np.nan)

    # Variable-length outputs.
    thresholds_multiclass_list = [None for _ in range(num_positions)]
    weights_list = [None for _ in range(num_positions)]
    ref_prob_k_list = [None for _ in range(num_positions)]
    ref_k_est_list = [None for _ in range(num_positions)]

    # Compact list of dicts for future feedback routines.
    feedback_params = [None for _ in range(num_positions)]

    hist_figs = [None for _ in range(num_positions)]

    # -------------------------------------------------------------------------
    # Parallel per-pillar fitting
    # -------------------------------------------------------------------------
    print("\n=== Starting parallel multi-NV reference-only fits ===")
    print(f"Number of pillars: {num_positions}")
    print(f"n_jobs: {n_jobs}")
    print(f"prob_dist: {prob_dist.name}")
    print(f"max_nvs_per_position: {max_nvs_per_position}")
    print(f"force_nvs: {force_nvs}")
    print(f"bic_extra_nv_penalty: {bic_extra_nv_penalty}")

    if n_jobs is None or int(n_jobs) == 1:
        parallel_results = [
            fit_one_pillar_multinv_job(
                ind,
                ref_counts_lists[ind],
                prob_dist.name,
                max_nvs_per_position,
                force_nvs,
                bic_extra_nv_penalty,
            )
            for ind in range(num_positions)
        ]
    else:
        parallel_results = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            verbose=joblib_verbose,
            batch_size=1,
        )(
            delayed(fit_one_pillar_multinv_job)(
                ind,
                ref_counts_lists[ind],
                prob_dist.name,
                max_nvs_per_position,
                force_nvs,
                bic_extra_nv_penalty,
            )
            for ind in range(num_positions)
        )

    # Keep results ordered by original pillar index.
    parallel_results = sorted(parallel_results, key=lambda x: x["ind"])

    # -------------------------------------------------------------------------
    # Fill arrays from fit results.
    # -------------------------------------------------------------------------
    for result in parallel_results:
        ind = int(result["ind"])

        sig_counts_list = sig_counts_lists[ind]
        ref_counts_list = ref_counts_lists[ind]

        if result["error"] is not None:
            print(f"\nMulti-NV fit failed for pillar index {ind}")
            print(result["error"])
            continue

        if not result["ok"]:
            print(f"\nMulti-NV fit not OK for pillar index {ind}")
            continue

        fit = result["fit"]
        ref_summary = result["ref_summary"]

        ok_arr[ind] = True

        # ---------------------------------------------------------------------
        # Extract reference-fit results.
        # ---------------------------------------------------------------------
        n_est = int(fit["n_nvs"])

        threshold_any = safe_float(fit["threshold_any"])
        thresholds_multiclass = np.asarray(fit["thresholds"], dtype=float)

        weights = np.asarray(fit["weights"], dtype=float)

        fidelity_any = safe_float(fit["fidelity_any"])
        fidelity_multiclass = safe_float(fit["fidelity_multiclass"])
        red_chi_sq = safe_float(fit.get("red_chi_sq", np.nan))

        p_minus = safe_float(fit.get("p_minus", np.nan))
        bg = safe_float(fit.get("bg", np.nan))
        rate0 = safe_float(fit.get("rate0", np.nan))
        delta = safe_float(fit.get("delta", np.nan))

        # P(any NV-) from fitted reference weights.
        # For N>1, this generalizes old prep fidelity = 1 - P(k=0).
        prep_fidelity_any_ref = 1.0 - float(weights[0])

        n_nvs_est_arr[ind] = n_est
        threshold_any_arr[ind] = threshold_any

        thresholds_multiclass_list[ind] = thresholds_multiclass
        weights_list[ind] = weights

        fidelity_any_arr[ind] = fidelity_any
        fidelity_multiclass_arr[ind] = fidelity_multiclass
        prep_fidelity_any_ref_arr[ind] = prep_fidelity_any_ref

        p_minus_arr[ind] = p_minus
        bg_arr[ind] = bg
        rate0_arr[ind] = rate0
        delta_arr[ind] = delta
        red_chi_sq_arr[ind] = red_chi_sq

        model_list[ind] = fit.get("model", None)
        best_candidate_model_list[ind] = fit.get("best_candidate_model", None)
        best_candidate_bic_arr[ind] = safe_float(
            fit.get("best_candidate_bic", np.nan)
        )
        best_equal_bic_arr[ind] = safe_float(fit.get("best_equal_bic", np.nan))
        unequal_2nv_beats_equal_arr[ind] = bool(
            fit.get("unequal_2nv_beats_equal", False)
        )
        candidate_results_list[ind] = fit.get("candidate_results", None)

        # ---------------------------------------------------------------------
        # Reference classification only.
        # This was already computed inside the worker.
        # ---------------------------------------------------------------------
        ref_p_any_minus_arr[ind] = ref_summary["p_any_minus"]
        ref_mean_num_minus_arr[ind] = ref_summary["mean_num_minus"]
        ref_prob_k_list[ind] = ref_summary["prob_k"]
        ref_k_est_list[ind] = ref_summary["k_est"]

        # ---------------------------------------------------------------------
        # Feedback-ready compact parameters.
        # ---------------------------------------------------------------------
        feedback_params[ind] = {
            "pillar_index": int(ind),
            "nv_name": getattr(nv_list[ind], "name", str(ind)),

            # Estimated total number of NVs in this pillar.
            "n_nvs_est": int(n_est),

            # Binary feedback:
            # counts > threshold_any means at least one NV is NV-.
            "threshold_any": float(threshold_any),
            "fidelity_any": float(fidelity_any),

            # Multi-class feedback:
            # k_est = searchsorted(thresholds_multiclass, counts)
            "thresholds_multiclass": thresholds_multiclass.tolist(),
            "fidelity_multiclass": float(fidelity_multiclass),

            # Fit model parameters from reference branch.
            "p_minus": float(p_minus),
            "bg": float(bg),
            "rate0": float(rate0),
            "delta": float(delta),
            "weights_k": weights.tolist(),

            # Model-selection diagnostics.
            "model": fit.get("model", None),
            "best_candidate_model": fit.get("best_candidate_model", None),
            "best_candidate_bic": safe_float(
                fit.get("best_candidate_bic", np.nan)
            ),
            "best_equal_bic": safe_float(fit.get("best_equal_bic", np.nan)),
            "unequal_2nv_beats_equal": bool(
                fit.get("unequal_2nv_beats_equal", False)
            ),

            # Reference-branch statistics only.
            "ref_p_any_minus": float(ref_summary["p_any_minus"]),
            "ref_mean_num_minus": float(ref_summary["mean_num_minus"]),
            "ref_prob_k": ref_summary["prob_k"].tolist(),

            "prep_fidelity_any_ref": float(prep_fidelity_any_ref),
            "red_chi_sq": float(red_chi_sq),
        }

        print(
            f"Pillar {ind}: "
            f"model={fit.get('model', None)}, "
            f"best_candidate={fit.get('best_candidate_model', None)}, "
            f"2nv_unequal_beats_equal={fit.get('unequal_2nv_beats_equal', False)}, "
            f"N_est={n_est}, "
            f"bg={bg:.3f}, "
            f"rate0={rate0:.3f}, "
            f"delta={delta:.3f}, "
            f"threshold_any={threshold_any:.3f}, "
            f"thresholds_multi={np.round(thresholds_multiclass, 3)}, "
            f"fid_any={fidelity_any:.3f}, "
            f"fid_multi={fidelity_multiclass:.3f}, "
            f"ref P(any NV-)={ref_summary['p_any_minus']:.3f}, "
            f"ref mean k={ref_summary['mean_num_minus']:.3f}, "
            f"weights={np.round(weights, 3)}"
        )

        # ---------------------------------------------------------------------
        # Plot: signal branch shown only visually; fit is reference only.
        # This remains serial. Do not plot inside parallel workers.
        # ---------------------------------------------------------------------
        if do_plot_histograms:
            fig = plot_histograms(
                sig_counts_list,
                ref_counts_list,
                density=True,
            )
            ax = fig.gca()

            x_max = max(
                np.nanmax(sig_counts_list),
                np.nanmax(ref_counts_list),
            )
            x_vals = np.linspace(0, x_max, 1000)

            single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)

            # Model used by analyze_charge_histogram_multinv_binomial:
            # k NV- gives mode centered around base + k * delta.
            #
            # For N NVs in the pillar:
            #     k = 0, 1, ..., N
            #
            # The no-NV-minus mode is centered near:
            #     base = bg + N * rate0
            #
            # Each additional NV- adds approximately delta counts.
            base = bg + n_est * rate0
            combined = np.zeros_like(x_vals, dtype=float)

            for k in range(n_est + 1):
                lam_k = base + k * delta
                comp = float(weights[k]) * single_pdf(x_vals, lam_k)
                combined += comp

                kpl.plot_line(
                    ax,
                    x_vals,
                    comp,
                    # label=f"fit k={k}",
                )

            kpl.plot_line(
                ax,
                x_vals,
                combined,
                color=kpl.KplColors.BLUE,
                # label="combined fit",
            )

            # Multi-class thresholds from reference branch.
            for t in thresholds_multiclass:
                ax.axvline(
                    t,
                    color=kpl.KplColors.GRAY,
                    ls="dashed",
                    lw=1,
                )

            # Binary any-NV- threshold.
            ax.axvline(
                threshold_any,
                color="black",
                ls="dashed",
                lw=2,
                label="any NV- threshold",
            )

            try:
                nv_num = widefield.get_nv_num(nv_list[ind])
            except Exception:
                nv_num = ind

            txt = (
                f"Pillar/NV {nv_num}\n"
                f"N_est = {n_est}\n"
                f"fid_any = {fidelity_any:.3f}\n"
                f"fid_multi = {fidelity_multiclass:.3f}\n"
                f"P(any NV-) = {ref_summary['p_any_minus']:.3f}\n"
                f"mean k = {ref_summary['mean_num_minus']:.2f}\n"
            )

            kpl.anchored_text(
                ax,
                txt,
                kpl.Loc.CENTER_RIGHT,
                size=kpl.Size.SMALL,
            )

            ax.legend(loc=kpl.Loc.UPPER_RIGHT)
            hist_figs[ind] = fig

            if save_hist_figs:
                timestamp = dm.get_time_stamp()
                nv_name = getattr(nv_list[ind], "name", f"pillar-{ind}")
                file_path = dm.get_file_path(
                    __file__,
                    timestamp,
                    f"{nv_name}-ref-only-multinv-charge-hist",
                )
                dm.save_figure(fig, file_path)

            kpl.show(block=True)

    # -------------------------------------------------------------------------
    # Save analysis back into raw_data.
    # -------------------------------------------------------------------------
    analysis_dict = {
        "analysis_type": "multi_nv_binomial_ref_only_no_background_subtraction",
        "note": (
            "All fit parameters, thresholds, estimated number of NVs per pillar, "
            "and feedback parameters are from the reference/no-ionization branch "
            "only. The signal/ionization branch is excluded from multi-NV analysis "
            "and is used only for visual comparison."
        ),
        "prob_dist": prob_dist.name,
        "max_nvs_per_position": int(max_nvs_per_position),
        "force_nvs": None if force_nvs is None else int(force_nvs),
        "bic_extra_nv_penalty": float(bic_extra_nv_penalty),

        # Parallel settings.
        "n_jobs": None if n_jobs is None else int(n_jobs),

        # Validity and number of NVs per pillar.
        "ok": ok_arr,
        "n_nvs_est": n_nvs_est_arr,

        # Feedback thresholds from reference branch.
        "threshold_any": threshold_any_arr,
        "thresholds_multiclass": thresholds_multiclass_list,

        # Fidelities from reference fit.
        "readout_fidelity_any": fidelity_any_arr,
        "fidelity_multiclass": fidelity_multiclass_arr,
        "prep_fidelity_any_ref": prep_fidelity_any_ref_arr,

        # Fit parameters from reference branch.
        "p_minus": p_minus_arr,
        "bg": bg_arr,
        "rate0": rate0_arr,
        "delta": delta_arr,
        "weights_k": weights_list,
        "red_chi_sq": red_chi_sq_arr,

        # Model-selection diagnostics.
        "model": model_list,
        "best_candidate_model": best_candidate_model_list,
        "best_candidate_bic": best_candidate_bic_arr,
        "best_equal_bic": best_equal_bic_arr,
        "unequal_2nv_beats_equal": unequal_2nv_beats_equal_arr,
        "candidate_results": candidate_results_list,

        # Reference classification only.
        "ref_p_any_minus": ref_p_any_minus_arr,
        "ref_mean_num_minus": ref_mean_num_minus_arr,
        "ref_prob_k": ref_prob_k_list,
        "ref_k_est": ref_k_est_list,

        # Compact dict list for future feedback routines.
        "feedback_params": feedback_params,
    }

    raw_data["charge_hist_multinv_binomial"] = analysis_dict

    # -------------------------------------------------------------------------
    # Summary printout.
    # -------------------------------------------------------------------------
    print("\n=== Multi-NV reference-only histogram summary ===")
    print("Good fits:", int(np.sum(ok_arr)), "/", num_positions)

    for n in range(1, max_nvs_per_position + 1):
        num_n = int(np.sum(ok_arr & (n_nvs_est_arr == n)))
        print(f"Estimated pillars with {n} NV(s): {num_n}")

    print("Median threshold_any:", np.nanmedian(threshold_any_arr))
    print("Median fidelity_any:", np.nanmedian(fidelity_any_arr))
    print("Median fidelity_multiclass:", np.nanmedian(fidelity_multiclass_arr))
    print("Median ref mean k:", np.nanmedian(ref_mean_num_minus_arr))
    print("Median ref P(any NV-):", np.nanmedian(ref_p_any_minus_arr))

    # -------------------------------------------------------------------------
    # Summary plot.
    # -------------------------------------------------------------------------
    good = (
        ok_arr
        & np.isfinite(fidelity_any_arr)
        & np.isfinite(fidelity_multiclass_arr)
    )

    if np.any(good):
        fig, ax = plt.subplots()
        kpl.plot_points(
            ax,
            fidelity_any_arr[good],
            fidelity_multiclass_arr[good],
        )
        ax.set_xlabel("Binary any-NV$^{-}$ fidelity")
        ax.set_ylabel("Multi-class fidelity")
        ax.set_title("charge-readout fidelity by pillar")

    # -------------------------------------------------------------------------
    # Optional save of analysis-only result.
    # -------------------------------------------------------------------------
    # if save_analysis:
    #     timestamp = dm.get_time_stamp()

    #     try:
    #         repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    #         repr_nv_name = repr_nv_sig.name
    #     except Exception:
    #         repr_nv_name = "multinv-charge-analysis"

    #     analysis_raw_data = {
    #         "timestamp": timestamp,
    #         "source_timestamp": raw_data.get("timestamp", None),
    #         "source_file_id": raw_data.get("file_id", None),
    #         # "nv_list": nv_list,
    #         "charge_hist_multinv_binomial": analysis_dict,
    #     }

    #     file_path = dm.get_file_path(
    #         __file__,
    #         timestamp,
    #         f"{repr_nv_name}-ref-only-multinv-charge-analysis",
    #     )

    #     dm.save_raw_data(
    #         analysis_raw_data,
    #         file_path,
    #         keys_to_compress=[],
    #     )

    #     print("Saved reference-only multi-NV charge analysis:", file_path)
    if save_analysis:
        timestamp = dm.get_time_stamp()

        try:
            repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
            repr_nv_name = repr_nv_sig.name
        except Exception:
            repr_nv_name = "multinv-charge-analysis"

        # Make a lightweight save copy.
        # Keep the full analysis in raw_data, but avoid saving recursive/heavy objects.
        analysis_dict_for_save = dict(analysis_dict)

        # These are the most likely to cause recursion / huge files.
        analysis_dict_for_save["candidate_results"] = None
        analysis_dict_for_save["ref_k_est"] = None

        # Save simple NV names instead of full NV objects.
        nv_names = [
            getattr(nv, "name", str(ind))
            for ind, nv in enumerate(nv_list)
        ]

        analysis_raw_data = {
            "timestamp": timestamp,
            "source_timestamp": raw_data.get("timestamp", None),
            "source_file_id": raw_data.get("file_id", None),
            "nv_names": nv_names,
            "charge_hist_multinv_binomial": make_json_safe(analysis_dict_for_save),
        }

        file_path = dm.get_file_path(
            __file__,
            timestamp,
            f"{repr_nv_name}-ref-only-multinv-charge-analysis",
        )

        dm.save_raw_data(
            analysis_raw_data,
            file_path,
            keys_to_compress=[],
        )

        print("Saved reference-only multi-NV charge analysis:", file_path)
        print("Saved 1-NV pillars:", len(one_nv_inds))

    return hist_figs

# =============================================================================
# Compact plotting style + metric explanations
# =============================================================================

SCATTER_FIGSIZE = (6.5, 5)
SPATIAL_FIGSIZE = (6.5, 5)
BAR_FIGSIZE = (6.5, 5)
HIST_FIGSIZE = (6.5, 5)

POINT_SIZE = 10
POINT_ALPHA = 0.75

METRIC_INFO = {
    "n_nvs_est": (
        "Estimated NVs per pillar",
        "Estimated total number of NVs contributing to this pillar/spot.",
    ),
    "threshold_any": (
        "Any-NV$^{-}$ threshold",
        "Binary threshold: counts above this mean at least one NV is NV$^{-}$.",
    ),
    "readout_fidelity_any": (
        "Binary fidelity",
        "Readout fidelity for distinguishing k=0 from k≥1 NV$^{-}$.",
    ),
    "fidelity_multiclass": (
        "Multi-class fidelity",
        "Readout fidelity for distinguishing k=0,1,...,N NV$^{-}$.",
    ),
    "prep_fidelity_any_ref": (
        "P(any NV$^{-}$), fit",
        "From fitted weights: 1 - P(k=0).",
    ),
    "ref_p_any_minus": (
        "P(any NV$^{-}$), classified",
        "Fraction of reference shots classified as having at least one NV$^{-}$.",
    ),
    "ref_mean_num_minus": (
        "Mean k",
        "Mean estimated number of NV$^{-}$ in the reference branch.",
    ),
    "p_minus": (
        "p_minus",
        "Fitted single-NV probability of being NV$^{-}$ in the reference branch.",
    ),
    "rate0": (
        "rate0",
        "Fitted per-NV NV$^{0}$ readout counts.",
    ),
    "delta": (
        "delta",
        "Extra counts when one NV changes from NV$^{0}$ to NV$^{-}$.",
    ),
    "red_chi_sq": (
        "Reduced $\\chi^2$",
        "Goodness of fit; smaller is generally better.",
    ),
}


def get_metric_label(key):
    return METRIC_INFO.get(key, (key, ""))[0]


def get_metric_note(key):
    return METRIC_INFO.get(key, (key, ""))[1]


def add_metric_note(ax, key, fontsize=7):
    note = get_metric_note(key)
    if note:
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )


def print_metric_definitions():
    print("\n=== Metric definitions ===")
    for key, (label, note) in METRIC_INFO.items():
        print(f"{key:25s} : {label} — {note}")


# =============================================================================
# Basic analysis accessors
# =============================================================================

def get_charge_analysis(raw_data):
    if "charge_hist_multinv_binomial" not in raw_data:
        raise KeyError(
            "raw_data does not contain 'charge_hist_multinv_binomial'. "
            "Run process_and_plot(...) first or load the saved analysis file."
        )
    return raw_data["charge_hist_multinv_binomial"]


def arr_from_analysis(analysis, key):
    val = analysis[key]
    return np.asarray(val, dtype=float)


def get_good_mask(analysis):
    return np.asarray(analysis["ok"], dtype=bool)


def extract_multiclass_threshold_array(analysis, max_nvs_per_position=None):
    """
    Convert variable-length threshold lists into fixed arrays.

    threshold_mat[:, 0] = threshold separating k=0 | k=1
    threshold_mat[:, 1] = threshold separating k=1 | k=2
    threshold_mat[:, 2] = threshold separating k=2 | k=3
    """

    thresholds_list = analysis["thresholds_multiclass"]
    num_pillars = len(thresholds_list)

    if max_nvs_per_position is None:
        max_nvs_per_position = int(analysis.get("max_nvs_per_position", 3))

    threshold_mat = np.full((num_pillars, max_nvs_per_position), np.nan)

    for ind, thresholds in enumerate(thresholds_list):
        if thresholds is None:
            continue

        thresholds = np.asarray(thresholds, dtype=float).flatten()
        num_t = min(len(thresholds), max_nvs_per_position)
        threshold_mat[ind, :num_t] = thresholds[:num_t]

    return threshold_mat


def extract_weights_array(analysis, max_nvs_per_position=None):
    """
    Convert variable-length weights_k lists into fixed arrays.

    weights_mat[:, k] = fitted probability of k NV- in reference branch.
    """

    weights_list = analysis["weights_k"]
    num_pillars = len(weights_list)

    if max_nvs_per_position is None:
        max_nvs_per_position = int(analysis.get("max_nvs_per_position", 3))

    weights_mat = np.full((num_pillars, max_nvs_per_position + 1), np.nan)

    for ind, weights in enumerate(weights_list):
        if weights is None:
            continue

        weights = np.asarray(weights, dtype=float).flatten()
        num_w = min(len(weights), max_nvs_per_position + 1)
        weights_mat[ind, :num_w] = weights[:num_w]

    return weights_mat


def get_xy_from_nv_list(nv_list, coords_key=None):
    """
    Try to get 2D coordinates for spatial scatter plots.

    Example:
        coords_key = "laser_INTE_520_aod"
        coords_key = "laser_COBO_638_aod"
        coords_key = "pixel"
        coords_key = "coords"
    """

    if coords_key is None:
        return None

    xy = []

    for nv in nv_list:
        coord = None

        try:
            coord = pos.get_nv_coords(nv, coords_key, drift_adjust=False)
        except Exception:
            pass

        if coord is None:
            try:
                coord = getattr(nv, coords_key)
            except Exception:
                pass

        if coord is None:
            try:
                coord = nv[coords_key]
            except Exception:
                pass

        if coord is None:
            xy.append([np.nan, np.nan])
            continue

        coord = np.asarray(coord, dtype=float).flatten()

        if len(coord) < 2:
            xy.append([np.nan, np.nan])
        else:
            xy.append([coord[0], coord[1]])

    return np.asarray(xy, dtype=float)


# =============================================================================
# Histogram plotting
# =============================================================================

def plot_histograms(
    sig_counts_list,
    ref_counts_list,
    no_title=True,
    ax=None,
    density=False,
):
    """
    Plot signal/reference histograms.

    Red:
        signal branch, with ionization pulse, visual only.

    Green:
        reference branch, without ionization pulse, used for fitting.

    This version does not require the global config["Optics"] entry.
    """

    # Try to get readout duration from config, but do not require it.
    readout_ms = None
    try:
        laser_key = VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
        laser_dict = tb.get_virtual_laser_dict(laser_key)
        readout = laser_dict.get("duration", None)
        if readout is not None:
            readout_ms = int(readout / 1e6)
    except Exception:
        readout_ms = None

    labels = [
        "With ionization pulse",
        "Without ionization pulse",
    ]

    colors = [
        kpl.KplColors.RED,
        kpl.KplColors.GREEN,
    ]

    counts_lists = [
        sig_counts_list,
        ref_counts_list,
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=HIST_FIGSIZE)
    else:
        fig = None

    if not no_title:
        if readout_ms is None:
            ax.set_title("Charge-state histograms")
        else:
            ax.set_title(f"Charge-state histograms, readout = {readout_ms} ms")

    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Probability" if density else "Number of occurrences")

    for ind in range(2):
        counts_list = counts_lists[ind]

        if counts_list is None or len(counts_list) == 0:
            continue

        kpl.histogram(
            ax,
            counts_list,
            color=colors[ind],
            density=density,
            label=labels[ind],
        )

    ax.set_xlim(-0.5, None)
    ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=7)

    if fig is not None:
        return fig

# =============================================================================
# Compact scatter plots
# =============================================================================

def scatter_metric_vs_index(
    raw_data,
    key,
    ylabel=None,
    title=None,
    good_only=True,
    add_note=True,
):
    """
    Compact scatter plot of one metric versus pillar index.
    """

    analysis = get_charge_analysis(raw_data)
    vals = arr_from_analysis(analysis, key)
    inds = np.arange(len(vals))

    good = np.isfinite(vals)
    if good_only:
        good &= get_good_mask(analysis)

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)

    ax.scatter(
        inds[good],
        vals[good],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
    )

    label = ylabel if ylabel is not None else get_metric_label(key)

    ax.set_xlabel("Pillar index")
    ax.set_ylabel(label)
    ax.set_title(title if title is not None else label, fontsize=15)

    if add_note:
        add_metric_note(ax, key)

    return fig, ax


def scatter_metric_spatial(
    raw_data,
    key,
    coords_key,
    cbar_label=None,
    title=None,
    good_only=True,
    marker_size=9,
    add_note=True,
):
    analysis = get_charge_analysis(raw_data)
    nv_list = raw_data["nv_list"]

    vals = arr_from_analysis(analysis, key)
    xy = get_xy_from_nv_list(nv_list, coords_key=coords_key)

    if xy is None:
        raise ValueError("Could not get xy coordinates. Provide a valid coords_key.")

    good = (
        np.isfinite(vals)
        & np.isfinite(xy[:, 0])
        & np.isfinite(xy[:, 1])
    )

    if good_only:
        good &= get_good_mask(analysis)

    fig, ax = plt.subplots(figsize=SPATIAL_FIGSIZE)

    sc = ax.scatter(
        xy[good, 0],
        xy[good, 1],
        c=vals[good],
        s=marker_size,
        alpha=POINT_ALPHA,
    )

    label = cbar_label if cbar_label is not None else get_metric_label(key)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel(f"{coords_key} x")
    ax.set_ylabel(f"{coords_key} y")
    ax.set_title(title if title is not None else label, fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    # Image-style coordinates: x increases to the right, y increases downward.
    # This makes the spatial scatter match camera/image display orientation.
    ax.invert_yaxis()

    if add_note:
        add_metric_note(ax, key)

    return fig, ax


def scatter_two_metrics(
    raw_data,
    x_key,
    y_key,
    xlabel=None,
    ylabel=None,
    title=None,
    good_only=True,
):
    """
    Compact scatter plot of one analysis parameter against another.
    """

    analysis = get_charge_analysis(raw_data)

    x = arr_from_analysis(analysis, x_key)
    y = arr_from_analysis(analysis, y_key)

    good = np.isfinite(x) & np.isfinite(y)
    if good_only:
        good &= get_good_mask(analysis)

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)

    ax.scatter(
        x[good],
        y[good],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
    )

    xlabel = xlabel if xlabel is not None else get_metric_label(x_key)
    ylabel = ylabel if ylabel is not None else get_metric_label(y_key)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title is not None else f"{ylabel} vs {xlabel}", fontsize=15)

    return fig, ax


# =============================================================================
# Summary plots
# =============================================================================

def plot_n_nv_count_summary(raw_data):
    """
    Bar plot: how many pillars are estimated to contain 1, 2, 3, ... NVs.
    """

    analysis = get_charge_analysis(raw_data)

    ok = get_good_mask(analysis)
    n_nvs = arr_from_analysis(analysis, "n_nvs_est")

    max_n = int(np.nanmax(n_nvs[ok])) if np.any(ok) else 0

    ns = np.arange(1, max_n + 1)
    counts = np.array([np.sum(ok & (n_nvs == n)) for n in ns], dtype=int)

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    ax.bar(ns, counts)

    ax.set_xlabel("Estimated NVs per pillar")
    ax.set_ylabel("Number of pillars")
    ax.set_title("Multi-NV occupancy", fontsize=15)

    for n, c in zip(ns, counts):
        ax.text(n, c, str(c), ha="center", va="bottom", fontsize=8)

    return fig, ax


def plot_thresholds_vs_index(raw_data):
    """
    Plot threshold_any and all multi-class thresholds versus pillar index.
    """

    analysis = get_charge_analysis(raw_data)
    good = get_good_mask(analysis)

    threshold_any = arr_from_analysis(analysis, "threshold_any")
    threshold_mat = extract_multiclass_threshold_array(analysis)

    inds = np.arange(len(threshold_any))

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)

    mask = good & np.isfinite(threshold_any)
    ax.scatter(
        inds[mask],
        threshold_any[mask],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="any NV$^{-}$",
    )

    for t_ind in range(threshold_mat.shape[1]):
        vals = threshold_mat[:, t_ind]
        mask = good & np.isfinite(vals)

        if not np.any(mask):
            continue

        ax.scatter(
            inds[mask],
            vals[mask],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            label=f"k={t_ind}|{t_ind + 1}",
        )

    ax.set_xlabel("Pillar index")
    ax.set_ylabel("Integrated counts")
    ax.set_title("Feedback thresholds", fontsize=15)
    ax.legend(fontsize=7)

    return fig, ax


def plot_weights_vs_index(raw_data):
    """
    Plot fitted weights P(k NV-) versus pillar index.
    """

    analysis = get_charge_analysis(raw_data)
    good = get_good_mask(analysis)

    weights_mat = extract_weights_array(analysis)
    inds = np.arange(weights_mat.shape[0])

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)

    for k in range(weights_mat.shape[1]):
        vals = weights_mat[:, k]
        mask = good & np.isfinite(vals)

        if not np.any(mask):
            continue

        ax.scatter(
            inds[mask],
            vals[mask],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            label=f"P(k={k})",
        )

    ax.set_xlabel("Pillar index")
    ax.set_ylabel("Fitted probability")
    ax.set_title("Reference charge-state weights", fontsize=15)
    ax.legend(fontsize=7)

    return fig, ax


def plot_fidelity_any_vs_multiclass(raw_data):
    """
    x = binary any-NV- readout fidelity
    y = multi-class fidelity
    """

    analysis = get_charge_analysis(raw_data)

    ok = get_good_mask(analysis)
    fidelity_any = arr_from_analysis(analysis, "readout_fidelity_any")
    fidelity_multiclass = arr_from_analysis(analysis, "fidelity_multiclass")

    good = (
        ok
        & np.isfinite(fidelity_any)
        & np.isfinite(fidelity_multiclass)
    )

    fig, ax = plt.subplots(figsize=SCATTER_FIGSIZE)

    ax.scatter(
        fidelity_any[good],
        fidelity_multiclass[good],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
    )

    ax.set_xlabel("Binary any-NV$^{-}$ fidelity")
    ax.set_ylabel("Multi-class fidelity")
    ax.set_title("Readout fidelity by pillar", fontsize=15)

    return fig, ax


def plot_one_pillar_hist_and_fit(
    raw_data,
    pillar_ind,
    density=True,
):
    """
    Plot one selected pillar with:
        red histogram   = sig branch, visual only
        green histogram = ref branch, fitted
        fit components  = reference-only multi-NV model
        dashed lines    = thresholds
    """

    analysis = get_charge_analysis(raw_data)

    if "counts" not in raw_data:
        raise KeyError(
            "raw_data does not contain 'counts'. "
            "Load the original raw data and attach the saved analysis dictionary."
        )

    counts = np.asarray(raw_data["counts"])
    nv_list = raw_data["nv_list"]

    sig_counts = counts[0, pillar_ind].flatten()
    ref_counts = counts[1, pillar_ind].flatten()

    ok = bool(np.asarray(analysis["ok"])[pillar_ind])
    if not ok:
        raise ValueError(f"Pillar {pillar_ind} does not have a good fit.")

    n_est = int(np.asarray(analysis["n_nvs_est"])[pillar_ind])
    threshold_any = float(np.asarray(analysis["threshold_any"])[pillar_ind])
    thresholds = np.asarray(analysis["thresholds_multiclass"][pillar_ind], dtype=float)
    weights = np.asarray(analysis["weights_k"][pillar_ind], dtype=float)

    p_minus = float(np.asarray(analysis["p_minus"])[pillar_ind])
    bg = float(np.asarray(analysis["bg"])[pillar_ind])
    rate0 = float(np.asarray(analysis["rate0"])[pillar_ind])
    delta = float(np.asarray(analysis["delta"])[pillar_ind])

    fidelity_any = float(np.asarray(analysis["readout_fidelity_any"])[pillar_ind])
    fidelity_multi = float(np.asarray(analysis["fidelity_multiclass"])[pillar_ind])
    ref_p_any = float(np.asarray(analysis["ref_p_any_minus"])[pillar_ind])
    ref_mean_k = float(np.asarray(analysis["ref_mean_num_minus"])[pillar_ind])

    prob_dist_name = analysis.get("prob_dist", "COMPOUND_POISSON")
    prob_dist = ProbDist[prob_dist_name]

    fig = plot_histograms(
        sig_counts,
        ref_counts,
        density=density,
    )
    ax = fig.gca()

    x_max = max(np.nanmax(sig_counts), np.nanmax(ref_counts))
    x_vals = np.linspace(0, x_max, 1000)

    single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)

    base = bg + n_est * rate0
    combined = np.zeros_like(x_vals, dtype=float)

    for k in range(n_est + 1):
        lam_k = base + k * delta
        comp = float(weights[k]) * single_pdf(x_vals, lam_k)
        combined += comp

        kpl.plot_line(
            ax,
            x_vals,
            comp,
            label=f"fit k={k}",
        )

    kpl.plot_line(
        ax,
        x_vals,
        combined,
        color=kpl.KplColors.BLUE,
        label="combined ref fit",
    )

    for t in thresholds:
        ax.axvline(
            t,
            color=kpl.KplColors.GRAY,
            ls="dashed",
            lw=1,
        )

    ax.axvline(
        threshold_any,
        color="black",
        ls="dashed",
        lw=2,
        label="any NV- threshold",
    )

    try:
        nv_num = widefield.get_nv_num(nv_list[pillar_ind])
    except Exception:
        nv_num = pillar_ind

    txt = (
        f"Pillar/NV {nv_num}\n"
        f"index = {pillar_ind}\n"
        f"N_est = {n_est}\n"
        f"fid_any = {fidelity_any:.3f}\n"
        f"fid_multi = {fidelity_multi:.3f}\n"
        f"P(any NV-) = {ref_p_any:.3f}\n"
        f"mean k = {ref_mean_k:.2f}\n"
        f"p_minus = {p_minus:.3f}\n"
        f"rate0 = {rate0:.2f}\n"
        f"delta = {delta:.2f}"
    )

    kpl.anchored_text(
        ax,
        txt,
        kpl.Loc.CENTER_RIGHT,
        size=kpl.Size.SMALL,
    )

    ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=7)

    return fig, ax


def plot_all_charge_multinv_summaries(
    raw_data,
    coords_key=None,
    save_figs=False,
):
    """
    Generate all main compact summary scatter plots.

    If coords_key is provided, also makes spatial maps.

    Example:
        coords_key = "laser_INTE_520_aod"
        coords_key = "laser_COBO_638_aod"
        coords_key = "pixel"
    """

    figs = []

    # ---------------------------------------------------------------------
    # Index scatter plots
    # ---------------------------------------------------------------------
    keys_to_plot = [
        ("n_nvs_est", get_metric_label("n_nvs_est")),
        ("threshold_any", get_metric_label("threshold_any")),
        ("readout_fidelity_any", get_metric_label("readout_fidelity_any")),
        ("fidelity_multiclass", get_metric_label("fidelity_multiclass")),
        ("prep_fidelity_any_ref", get_metric_label("prep_fidelity_any_ref")),
        ("ref_p_any_minus", get_metric_label("ref_p_any_minus")),
        ("ref_mean_num_minus", get_metric_label("ref_mean_num_minus")),
        ("p_minus", get_metric_label("p_minus")),
        ("rate0", get_metric_label("rate0")),
        ("delta", get_metric_label("delta")),
        ("red_chi_sq", get_metric_label("red_chi_sq")),
    ]

    for key, label in keys_to_plot:
        try:
            fig, ax = scatter_metric_vs_index(
                raw_data,
                key,
                ylabel=label,
                title=label,
            )
            figs.append(fig)
        except Exception:
            print(f"Could not plot {key}")
            print(traceback.format_exc())

    # ---------------------------------------------------------------------
    # Thresholds, weights, occupancy, fidelity summary
    # ---------------------------------------------------------------------
    for plot_fn, name in [
        (plot_thresholds_vs_index, "thresholds"),
        (plot_weights_vs_index, "weights"),
        (plot_n_nv_count_summary, "NV occupancy summary"),
        (plot_fidelity_any_vs_multiclass, "fidelity_any vs fidelity_multiclass"),
    ]:
        try:
            fig, ax = plot_fn(raw_data)
            figs.append(fig)
        except Exception:
            print(f"Could not plot {name}")
            print(traceback.format_exc())

    # ---------------------------------------------------------------------
    # Pair scatter plots
    # ---------------------------------------------------------------------
    pair_plots = [
        ("delta", "readout_fidelity_any", "delta", "Binary fidelity"),
        ("delta", "fidelity_multiclass", "delta", "Multi-class fidelity"),
        ("p_minus", "fidelity_multiclass", "p_minus", "Multi-class fidelity"),
        ("threshold_any", "delta", "Any-NV$^{-}$ threshold", "delta"),
        ("n_nvs_est", "threshold_any", "Estimated NVs", "Any-NV$^{-}$ threshold"),
        ("n_nvs_est", "ref_mean_num_minus", "Estimated NVs", "Mean k"),
    ]

    for x_key, y_key, xlabel, ylabel in pair_plots:
        try:
            fig, ax = scatter_two_metrics(
                raw_data,
                x_key,
                y_key,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            figs.append(fig)
        except Exception:
            print(f"Could not plot {y_key} vs {x_key}")
            print(traceback.format_exc())

    # ---------------------------------------------------------------------
    # Spatial maps
    # ---------------------------------------------------------------------
    if coords_key is not None:
        spatial_keys = [
            ("n_nvs_est", get_metric_label("n_nvs_est")),
            ("threshold_any", get_metric_label("threshold_any")),
            ("readout_fidelity_any", get_metric_label("readout_fidelity_any")),
            ("fidelity_multiclass", get_metric_label("fidelity_multiclass")),
            ("ref_mean_num_minus", get_metric_label("ref_mean_num_minus")),
            ("p_minus", get_metric_label("p_minus")),
            ("rate0", get_metric_label("rate0")),
            ("delta", get_metric_label("delta")),
            ("red_chi_sq", get_metric_label("red_chi_sq")),
        ]

        for key, label in spatial_keys:
            try:
                fig, ax = scatter_metric_spatial(
                    raw_data,
                    key,
                    coords_key=coords_key,
                    cbar_label=label,
                    title=f"{label} map",
                )
                figs.append(fig)
            except Exception:
                print(f"Could not make spatial map for {key}")
                print(traceback.format_exc())

    # ---------------------------------------------------------------------
    # Optional save
    # ---------------------------------------------------------------------
    if save_figs:
        timestamp = dm.get_time_stamp()
        for ind, fig in enumerate(figs):
            file_path = dm.get_file_path(
                __file__,
                timestamp,
                f"charge-multinv-summary-{ind:02d}",
            )
            dm.save_figure(fig, file_path)

    return figs

def save_selected_pillar_histograms(
    raw_data,
    pillar_inds,
    label,
    density=True,
    close_figs=True,
):
    """
    Plot and save one-pillar histogram+fit figures for selected pillars.

    Parameters
    ----------
    raw_data : dict
        Must contain raw_data["counts"], raw_data["nv_list"],
        and raw_data["charge_hist_multinv_binomial"].

    pillar_inds : array-like
        Pillar indices to plot/save.

    label : str
        Label used in saved filename, e.g. "best-2nv" or "best-3nv".

    density : bool
        If True, plot probability-density histograms.

    close_figs : bool
        If True, close figures after saving to avoid too many open windows.
    """

    timestamp = dm.get_time_stamp()
    saved_paths = []

    analysis = raw_data["charge_hist_multinv_binomial"]
    n_nvs_est = np.asarray(analysis["n_nvs_est"], dtype=float)
    fidelity_multi = np.asarray(analysis["fidelity_multiclass"], dtype=float)
    fidelity_any = np.asarray(analysis["readout_fidelity_any"], dtype=float)

    for pillar_ind in pillar_inds:
        pillar_ind = int(pillar_ind)

        fig, ax = plot_one_pillar_hist_and_fit(
            raw_data,
            pillar_ind=pillar_ind,
            density=density,
        )

        n_est = int(n_nvs_est[pillar_ind])
        fid_multi = fidelity_multi[pillar_ind]
        fid_any = fidelity_any[pillar_ind]

        file_label = (
            f"{label}-pillar-{pillar_ind:04d}"
            f"-N{n_est}"
            f"-fidmulti-{fid_multi:.3f}"
            f"-fidany-{fid_any:.3f}"
        )

        file_path = dm.get_file_path(
            __file__,
            timestamp,
            file_label,
        )

        dm.save_figure(fig, file_path)
        saved_paths.append(file_path)

        print("Saved:", file_path)

        if close_figs:
            plt.close(fig)

    return saved_paths


def print_three_nv_statistical_counts(analysis, pillar_ind, num_shots=None):
    """
    Print expected Poisson peak means, widths, probabilities, and shot counts
    for a pillar classified as 3 NVs.
    """

    n_est = int(np.asarray(analysis["n_nvs_est"])[pillar_ind])
    if n_est != 3:
        print(f"Warning: pillar {pillar_ind} has n_est={n_est}, not 3.")

    p_minus = float(np.asarray(analysis["p_minus"])[pillar_ind])
    bg = float(np.asarray(analysis["bg"])[pillar_ind])
    rate0 = float(np.asarray(analysis["rate0"])[pillar_ind])
    delta = float(np.asarray(analysis["delta"])[pillar_ind])

    if num_shots is None:
        num_shots = 1.0

    print(f"\nPillar {pillar_ind}")
    print(f"N_est = {n_est}")
    print(f"rate0 = {rate0:.3f}")
    print(f"delta = {delta:.3f}")
    print(f"p_minus = {p_minus:.3f}")
    print()

    print("Expected 3-NV Poisson components:")
    print("k   mean lambda_k   sigma=sqrt(lambda)   P(k)      expected shots")

    for k in range(4):
        lam_k = bg + 3 * rate0 + k * delta
        sigma_k = np.sqrt(lam_k)

        prob_k = comb(3, k) * (p_minus ** k) * ((1 - p_minus) ** (3 - k))
        shots_k = num_shots * prob_k

        print(
            f"{k}   "
            f"{lam_k:10.2f}   "
            f"{sigma_k:10.2f}        "
            f"{prob_k:7.3f}   "
            f"{shots_k:10.1f}"
        )

    print("\nPeak separability:")
    for k in range(3):
        lam0 = bg + 3 * rate0 + k * delta
        lam1 = bg + 3 * rate0 + (k + 1) * delta

        # Separation in units of combined shot noise
        dprime = delta / np.sqrt(lam0 + lam1)

        print(
            f"k={k} to k={k+1}: "
            f"separation={delta:.2f} counts, "
            f"d'={dprime:.2f}"
        )
# =============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()

    raw_data = dm.get_raw_data(
        # file_stem="2026_03_02-17_30_11-qnami-nv0_2026_02_20", ## 1277 working NVs
        # file_stem="2026_06_18-14_53_41-qnami-nv0_2026_02_20", ## 1176 working NVs
        # file_stem="2026_06_14-18_44_06-qnami-nv0_2026_02_20",
        # file_stem="2026_06_20-19_22_52-qnami-nv0_2026_02_20",
        # file_stem="2026_06_23-12_05_32-qnami-nv0_2026_02_20",
        file_stem="2026_06_23-15_11_05-qnami-nv0_2026_02_20", ## 1176 working NVs

        load_npz=True,
    )

    process_and_plot(
    raw_data,
    do_plot_histograms=False,
    prob_dist=ProbDist.COMPOUND_POISSON,
    max_nvs_per_position=3,
    force_nvs=None,
    bic_extra_nv_penalty=2.0,
    save_analysis=True,
    save_hist_figs=False,
    n_jobs=12,
    )

    # kpl.show(block=True)
    # sys.exit()
    # =============================================================================
    # Analyzed data: load saved analysis and attach it to original raw data
    # =============================================================================
    analysis_data = dm.get_raw_data(
        # file_stem="2026_06_15-01_24_30-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis",  #1176NVs
        # file_stem="2026_06_22-18_39_58-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", #1176NVs
        # file_stem="2026_06_22-18_39_58-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", #814NVs
        file_stem="2026_06_22-18_39_58-qnami-nv0_2026_02_20-ref-only-multinv-charge-analysis", #814NVs
        load_npz=True,
    )

    # print_metric_definitions()

    # figs = plot_all_charge_multinv_summaries( 
    #     analysis_data,
    #     coords_key="pixel",
    #     save_figs=True,
    # )

    # # Attach saved analysis to original raw data so histogram plotting works.
    # raw_data["charge_hist_multinv_binomial"] = analysis_data[
    #     "charge_hist_multinv_binomial"
    # ]

    # # print_metric_definitions()

    # analysis = analysis_data["charge_hist_multinv_binomial"]

    # num_reps = 2000   # replace with your real num_reps * num_runs if known
    # print_three_nv_statistical_counts(
    #     analysis,
    #     pillar_ind=16,
    #     num_shots=num_reps,
    # )
    # kpl.show(block=True)
    # sys.exit()



    # New lightweight saved file has this structure:
    # analysis_data["charge_hist_multinv_binomial"] = actual analysis dictionary
    analysis = analysis_data["charge_hist_multinv_binomial"]

    # Attach saved analysis to original raw_data so histogram plotting can use:
    #   raw_data["counts"]
    #   raw_data["nv_list"]
    #   raw_data["charge_hist_multinv_binomial"]
    raw_data["charge_hist_multinv_binomial"] = analysis

    # Optional: print available keys
    print("\nLoaded analysis keys:")
    print(list(analysis.keys()))

    # =============================================================================
    # Plot summary figures from saved analysis
    # =============================================================================
    figs = plot_all_charge_multinv_summaries(
        raw_data,
        coords_key="pixel",
        save_figs=True,
    )

    # =============================================================================
    # Find selected 2-NV and 3-NV pillars
    # =============================================================================

    ok = np.asarray(analysis["ok"], dtype=bool)
    n_nvs_est = np.asarray(analysis["n_nvs_est"], dtype=float)
    fidelity_multi = np.asarray(analysis["fidelity_multiclass"], dtype=float)
    fidelity_any = np.asarray(analysis["readout_fidelity_any"], dtype=float)
    threshold_any = np.asarray(analysis["threshold_any"], dtype=float)
    ref_mean_k = np.asarray(analysis["ref_mean_num_minus"], dtype=float)

    one_nv_inds = np.where(ok & (np.rint(n_nvs_est).astype(int) == 1))[0]

    print("\n1-NV pillar indices:")
    print(one_nv_inds)

    print("\nNumber of 1-NV pillars:", len(one_nv_inds))
    
    sys.exit()
    
    two_nv_inds = np.where(ok & (n_nvs_est == 2))[0]
    three_nv_inds = np.where(ok & (n_nvs_est == 3))[0]

    print("\n2-NV pillar indices:")
    print(two_nv_inds)

    print("\n3-NV pillar indices:")
    print(three_nv_inds)

    print("\nNumber of 2-NV pillars:", len(two_nv_inds))
    print("Number of 3-NV pillars:", len(three_nv_inds))

    # =============================================================================
    # Choose best examples by multi-class fidelity
    # =============================================================================
    num_examples = 10

    two_nv_best = two_nv_inds[
        np.argsort(fidelity_multi[two_nv_inds])[::-1]
    ][:num_examples]

    three_nv_best = three_nv_inds[
        np.argsort(fidelity_multi[three_nv_inds])[::-1]
    ][:num_examples]

    print("\nBest 2-NV examples by multi-class fidelity:")
    for ind in two_nv_best:
        print(
            f"pillar {ind}: "
            f"fid_multi={fidelity_multi[ind]:.3f}, "
            f"fid_any={fidelity_any[ind]:.3f}, "
            f"threshold_any={threshold_any[ind]:.1f}, "
            f"mean_k={ref_mean_k[ind]:.2f}"
        )

    print("\nBest 3-NV examples by multi-class fidelity:")
    for ind in three_nv_best:
        print(
            f"pillar {ind}: "
            f"fid_multi={fidelity_multi[ind]:.3f}, "
            f"fid_any={fidelity_any[ind]:.3f}, "
            f"threshold_any={threshold_any[ind]:.1f}, "
            f"mean_k={ref_mean_k[ind]:.2f}"
        )

    # =============================================================================
    # Plot histograms for selected 2-NV and 3-NV pillars
    # =============================================================================

    for pillar_ind in two_nv_best:
        plot_one_pillar_hist_and_fit(
            raw_data,
            pillar_ind=int(pillar_ind),
            density=True,
        )

    for pillar_ind in three_nv_best:
        plot_one_pillar_hist_and_fit(
            raw_data,
            pillar_ind=int(pillar_ind),
            density=True,
        )


    # # Optional: save selected histogram figures
    # save_selected_pillar_histograms(
    #     raw_data,
    #     two_nv_best,
    #     label="best-2nv-charge-hist",
    #     density=True,
    #     close_figs=True,
    # )

    # save_selected_pillar_histograms(
    #     raw_data,
    #     three_nv_best,
    #     label="best-3nv-charge-hist",
    #     density=True,
    #     close_figs=True,
    # )

    kpl.show(block=True)