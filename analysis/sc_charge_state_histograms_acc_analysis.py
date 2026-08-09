# -*- coding: utf-8 -*-
"""
Single-step charge-state histogram analysis.

This script is for ONE histogram condition only. There is NO sweep and no
optimal-step search.

Supported modes
---------------
MODEL_KIND = "single"
    Ordinary one-NV / bimodal histogram fit for each NV/pillar.

MODEL_KIND = "multi"
    Multi-NV binomial histogram fit for each pillar.

BACKEND = "cpu"
    Reliable CPU/joblib version.

BACKEND = "gpu"
    Optional GPU version if your lab GPU fitting module is available.
    If unavailable or parsing fails, the script falls back to CPU.

Expected counts shape
---------------------
The script is intentionally permissive. It assumes the first axis is experiment
branch and the second axis is NV/pillar index:

    counts[exp, nv, ...]

Everything after nv is flattened into one histogram.

Typical branch convention
-------------------------
    counts[0] = signal / with ionization pulse      -> plotted in red only
    counts[1] = reference / without ionization pulse -> fitted and plotted green

For single-NV mode, the reference branch is fitted with a bimodal distribution.
For multi-NV mode, the reference branch is fitted with a binomial multi-NV model.

Created July 2026
@author: Saroj Chand
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

# Compatibility patch for old labrad with newer NumPy.
# Do not use hasattr(np, "bool8") because newer NumPy can warn on access.
if "bool8" not in np.__dict__:
    np.bool8 = np.bool_

from analysis import bimodal_histogram
from analysis.bimodal_histogram import (
    ProbDist,
    determine_threshold,
    fit_bimodal_histogram,
)

try:
    from analysis.bimodal_histogram import analyze_charge_histogram_multinv_binomial
    MULTI_CPU_AVAILABLE = True
except Exception:
    analyze_charge_histogram_multinv_binomial = None
    MULTI_CPU_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# Optional lab GPU fitting module. Change this import if your GPU fitter lives
# under a different filename.
try:
    from analysis.sc_gpu_bimodal_fitting import (
        GpuMultimodeFitConfig,
        fit_charge_histograms_gpu_batch,
    )
    GPU_FITTER_AVAILABLE = True
except Exception:
    GpuMultimodeFitConfig = None
    fit_charge_histograms_gpu_batch = None
    GPU_FITTER_AVAILABLE = False

from utils import data_manager as dm
from utils import kplotlib as kpl

try:
    from utils import widefield
except Exception:
    widefield = None




# =============================================================================
# Generic helpers
# =============================================================================


def now():
    return time.perf_counter()


def print_elapsed(label, t0):
    dt = now() - t0
    print(f"{label}: {dt:.2f} s = {dt / 60:.2f} min")
    return dt


def print_runtime_header(label, backend, n_jobs):
    print(f"\n=== {label} ===")
    print("backend:", backend)
    print("CPU count:", os.cpu_count())
    print("n_jobs:", n_jobs)
    print("CuPy available:", CUPY_AVAILABLE)
    print("GPU fitter import available:", GPU_FITTER_AVAILABLE)

    if CUPY_AVAILABLE:
        try:
            num_gpus = cp.cuda.runtime.getDeviceCount()
            print("Number of CUDA GPUs:", num_gpus)
            if num_gpus > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                print("GPU 0 name:", props["name"].decode())
        except Exception:
            print("Could not query CUDA GPU.")


def make_json_safe(obj):
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
    return str(obj)


def safe_float(x):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


def get_prob_dist(prob_dist_name):
    try:
        return ProbDist[prob_dist_name]
    except Exception:
        return getattr(ProbDist, str(prob_dist_name))


def get_nv_label(nv_list, ind):
    if nv_list is None:
        return str(ind)

    try:
        if widefield is not None:
            return str(widefield.get_nv_num(nv_list[ind]))
    except Exception:
        pass

    try:
        return str(getattr(nv_list[ind], "name", ind))
    except Exception:
        return str(ind)


def flatten_branch_counts(counts, exp_ind, nv_ind):
    """
    Return 1D histogram counts for one exp branch and one NV/pillar.

    Works for shapes like:
        counts[exp, nv, rep]
        counts[exp, nv, run, rep]
        counts[exp, nv, run, something, rep]
    """
    return np.asarray(counts[exp_ind, nv_ind], dtype=float).flatten()


def extract_ref_and_sig_counts(raw_data, fit_exp_ind=1, sig_exp_ind=0):
    counts = np.asarray(raw_data["counts"], dtype=float)
    nv_list = raw_data["nv_list"]
    num_positions = len(nv_list)

    if counts.ndim < 3:
        raise ValueError(
            "Expected counts shape counts[exp, nv, ...]. "
            f"Got shape {counts.shape}."
        )

    if counts.shape[0] <= fit_exp_ind:
        raise ValueError(
            f"FIT_EXP_IND={fit_exp_ind} does not exist. counts.shape={counts.shape}"
        )

    ref_counts_lists = [
        flatten_branch_counts(counts, fit_exp_ind, ind)
        for ind in range(num_positions)
    ]

    if sig_exp_ind is not None and counts.shape[0] > sig_exp_ind:
        sig_counts_lists = [
            flatten_branch_counts(counts, sig_exp_ind, ind)
            for ind in range(num_positions)
        ]
    else:
        sig_counts_lists = [None for _ in range(num_positions)]

    return sig_counts_lists, ref_counts_lists


# =============================================================================
# Single-NV CPU fitting
# =============================================================================


def fit_single_nv_cpu_job(ind, ref_counts, prob_dist_name):
    try:
        prob_dist = get_prob_dist(prob_dist_name)
        ref_counts = np.asarray(ref_counts, dtype=float).flatten()
        ref_counts = ref_counts[np.isfinite(ref_counts)]

        popt, pcov, red_chi_sq = fit_bimodal_histogram(
            ref_counts,
            prob_dist,
            no_plot=True,
        )

        if popt is None:
            return {
                "ind": int(ind),
                "ok": False,
                "threshold": np.nan,
                "readout_fidelity": np.nan,
                "prep_fidelity": np.nan,
                "red_chi_sq": np.nan,
                "fit_params": None,
                "error": None,
            }

        threshold, readout_fidelity = determine_threshold(
            popt,
            prob_dist,
            dark_mode_weight=0.5,
            ret_fidelity=True,
        )

        # In your older single-NV script this is 1 - popt[0].
        prep_fidelity = 1.0 - float(popt[0])

        return {
            "ind": int(ind),
            "ok": True,
            "threshold": float(threshold),
            "readout_fidelity": float(readout_fidelity),
            "prep_fidelity": float(prep_fidelity),
            "red_chi_sq": float(red_chi_sq),
            "fit_params": np.asarray(popt, dtype=float),
            "error": None,
        }

    except Exception:
        return {
            "ind": int(ind),
            "ok": False,
            "threshold": np.nan,
            "readout_fidelity": np.nan,
            "prep_fidelity": np.nan,
            "red_chi_sq": np.nan,
            "fit_params": None,
            "error": traceback.format_exc(),
        }


def fit_single_cpu(ref_counts_lists, prob_dist, n_jobs=12, joblib_verbose=10):
    num_positions = len(ref_counts_lists)

    print("\n=== CPU single-NV bimodal fitting ===")
    print("num positions:", num_positions)
    print("prob_dist:", prob_dist.name)

    tasks = [
        (ind, ref_counts_lists[ind], prob_dist.name)
        for ind in range(num_positions)
    ]

    if n_jobs is None or int(n_jobs) == 1:
        results = [fit_single_nv_cpu_job(*task) for task in tasks]
    else:
        results = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            verbose=joblib_verbose,
            batch_size="auto",
        )(
            delayed(fit_single_nv_cpu_job)(*task)
            for task in tasks
        )

    results = sorted(results, key=lambda x: x["ind"])
    return build_single_analysis_from_job_results(results, prob_dist)


def build_single_analysis_from_job_results(job_results, prob_dist):
    num_positions = len(job_results)

    ok = np.zeros(num_positions, dtype=bool)
    threshold = np.full(num_positions, np.nan)
    readout_fidelity = np.full(num_positions, np.nan)
    prep_fidelity = np.full(num_positions, np.nan)
    red_chi_sq = np.full(num_positions, np.nan)
    fit_params = [None for _ in range(num_positions)]

    for res in job_results:
        ind = int(res["ind"])

        if res.get("error") is not None:
            print(f"\nSingle-NV fit failed for index {ind}")
            print(res["error"])

        ok[ind] = bool(res.get("ok", False))
        threshold[ind] = safe_float(res.get("threshold", np.nan))
        readout_fidelity[ind] = safe_float(res.get("readout_fidelity", np.nan))
        prep_fidelity[ind] = safe_float(res.get("prep_fidelity", np.nan))
        red_chi_sq[ind] = safe_float(res.get("red_chi_sq", np.nan))

        if res.get("fit_params") is not None:
            fit_params[ind] = np.asarray(res["fit_params"], dtype=float).ravel()

    return {
        "analysis_type": "single_step_single_nv_bimodal",
        "backend_requested": BACKEND,
        "prob_dist": prob_dist.name,
        "ok": ok,
        "threshold": threshold,
        "readout_fidelity": readout_fidelity,
        "prep_fidelity": prep_fidelity,
        "red_chi_sq": red_chi_sq,
        "fit_params": fit_params,
    }


# =============================================================================
# Multi-NV CPU fitting
# =============================================================================


def classify_multinv_counts(counts, thresholds):
    counts = np.asarray(counts, dtype=float).flatten()

    if thresholds is None:
        return np.full(counts.shape, -1, dtype=int)

    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.size == 0 or np.any(~np.isfinite(thresholds)):
        return np.full(counts.shape, -1, dtype=int)

    return np.searchsorted(thresholds, counts, side="right").astype(int)


def summarize_ref_classification(ref_counts, threshold_any, thresholds_multiclass, n_nvs):
    ref_counts = np.asarray(ref_counts, dtype=float).flatten()
    n_nvs = int(n_nvs)

    if threshold_any is None or not np.isfinite(threshold_any):
        p_any_minus = np.nan
    else:
        p_any_minus = float(np.mean(ref_counts > threshold_any))

    k_est = classify_multinv_counts(ref_counts, thresholds_multiclass)

    prob_k = np.full(n_nvs + 1, np.nan, dtype=float)
    for k in range(n_nvs + 1):
        prob_k[k] = float(np.mean(k_est == k))

    good = k_est >= 0
    mean_num_minus = float(np.mean(k_est[good])) if np.any(good) else np.nan

    return {
        "p_any_minus": p_any_minus,
        "mean_num_minus": mean_num_minus,
        "prob_k": prob_k,
        "k_est": k_est,
    }


def fit_multi_nv_cpu_job(
    ind,
    ref_counts,
    prob_dist_name,
    max_nvs_per_position,
    force_nvs,
    bic_extra_nv_penalty,
):
    if not MULTI_CPU_AVAILABLE:
        return {
            "ind": int(ind),
            "ok": False,
            "fit": None,
            "ref_summary": None,
            "error": "analyze_charge_histogram_multinv_binomial not importable.",
        }

    try:
        prob_dist = get_prob_dist(prob_dist_name)
        ref_counts = np.asarray(ref_counts, dtype=float).flatten()
        ref_counts = ref_counts[np.isfinite(ref_counts)]

        fit = analyze_charge_histogram_multinv_binomial(
            ref_counts,
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
            ref_counts=ref_counts,
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


def fit_multi_cpu(
    ref_counts_lists,
    prob_dist,
    max_nvs_per_position=3,
    force_nvs=None,
    bic_extra_nv_penalty=2.0,
    n_jobs=12,
    joblib_verbose=10,
):
    num_positions = len(ref_counts_lists)

    print("\n=== CPU multi-NV binomial fitting ===")
    print("num positions:", num_positions)
    print("prob_dist:", prob_dist.name)
    print("max_nvs_per_position:", max_nvs_per_position)
    print("force_nvs:", force_nvs)

    tasks = [
        (
            ind,
            ref_counts_lists[ind],
            prob_dist.name,
            max_nvs_per_position,
            force_nvs,
            bic_extra_nv_penalty,
        )
        for ind in range(num_positions)
    ]

    if n_jobs is None or int(n_jobs) == 1:
        results = [fit_multi_nv_cpu_job(*task) for task in tasks]
    else:
        results = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            verbose=joblib_verbose,
            batch_size=1,
        )(
            delayed(fit_multi_nv_cpu_job)(*task)
            for task in tasks
        )

    results = sorted(results, key=lambda x: x["ind"])
    return build_multi_analysis_from_job_results(
        results,
        prob_dist,
        max_nvs_per_position=max_nvs_per_position,
        force_nvs=force_nvs,
        bic_extra_nv_penalty=bic_extra_nv_penalty,
    )


def build_multi_analysis_from_job_results(
    job_results,
    prob_dist,
    max_nvs_per_position=3,
    force_nvs=None,
    bic_extra_nv_penalty=2.0,
):
    num_positions = len(job_results)

    ok = np.zeros(num_positions, dtype=bool)
    n_nvs_est = np.full(num_positions, np.nan)
    threshold_any = np.full(num_positions, np.nan)
    readout_fidelity_any = np.full(num_positions, np.nan)
    fidelity_multiclass = np.full(num_positions, np.nan)
    prep_fidelity_any_ref = np.full(num_positions, np.nan)
    p_minus = np.full(num_positions, np.nan)
    bg = np.full(num_positions, np.nan)
    rate0 = np.full(num_positions, np.nan)
    delta = np.full(num_positions, np.nan)
    red_chi_sq = np.full(num_positions, np.nan)
    ref_p_any_minus = np.full(num_positions, np.nan)
    ref_mean_num_minus = np.full(num_positions, np.nan)

    thresholds_multiclass = [None for _ in range(num_positions)]
    weights_k = [None for _ in range(num_positions)]
    model = [None for _ in range(num_positions)]
    best_candidate_model = [None for _ in range(num_positions)]
    feedback_params = [None for _ in range(num_positions)]

    for res in job_results:
        ind = int(res["ind"])

        if res.get("error") is not None:
            print(f"\nMulti-NV fit failed for index {ind}")
            print(res["error"])
            continue

        if not res.get("ok", False):
            continue

        fit = res["fit"]
        ref_summary = res["ref_summary"]

        n_est = int(fit["n_nvs"])
        weights = np.asarray(fit["weights"], dtype=float)
        thresholds = np.asarray(fit["thresholds"], dtype=float)

        ok[ind] = True
        n_nvs_est[ind] = n_est
        threshold_any[ind] = safe_float(fit.get("threshold_any", np.nan))
        readout_fidelity_any[ind] = safe_float(fit.get("fidelity_any", np.nan))
        fidelity_multiclass[ind] = safe_float(fit.get("fidelity_multiclass", np.nan))

        # Multi-NV equivalent of prep fidelity: probability of at least one NV-.
        prep_fidelity_any_ref[ind] = 1.0 - float(weights[0])

        p_minus[ind] = safe_float(fit.get("p_minus", np.nan))
        bg[ind] = safe_float(fit.get("bg", np.nan))
        rate0[ind] = safe_float(fit.get("rate0", np.nan))
        delta[ind] = safe_float(fit.get("delta", np.nan))
        red_chi_sq[ind] = safe_float(fit.get("red_chi_sq", np.nan))
        ref_p_any_minus[ind] = safe_float(ref_summary.get("p_any_minus", np.nan))
        ref_mean_num_minus[ind] = safe_float(ref_summary.get("mean_num_minus", np.nan))

        thresholds_multiclass[ind] = thresholds
        weights_k[ind] = weights
        model[ind] = fit.get("model", None)
        best_candidate_model[ind] = fit.get("best_candidate_model", None)

        feedback_params[ind] = {
            "pillar_index": int(ind),
            "n_nvs_est": int(n_est),
            "threshold_any": safe_float(fit.get("threshold_any", np.nan)),
            "thresholds_multiclass": thresholds.tolist(),
            "readout_fidelity_any": safe_float(fit.get("fidelity_any", np.nan)),
            "fidelity_multiclass": safe_float(fit.get("fidelity_multiclass", np.nan)),
            "prep_fidelity_any_ref": float(prep_fidelity_any_ref[ind]),
            "p_minus": safe_float(fit.get("p_minus", np.nan)),
            "bg": safe_float(fit.get("bg", np.nan)),
            "rate0": safe_float(fit.get("rate0", np.nan)),
            "delta": safe_float(fit.get("delta", np.nan)),
            "weights_k": weights.tolist(),
            "red_chi_sq": safe_float(fit.get("red_chi_sq", np.nan)),
            "model": fit.get("model", None),
            "best_candidate_model": fit.get("best_candidate_model", None),
            "ref_p_any_minus": safe_float(ref_summary.get("p_any_minus", np.nan)),
            "ref_mean_num_minus": safe_float(ref_summary.get("mean_num_minus", np.nan)),
        }

    return {
        "analysis_type": "single_step_multi_nv_binomial",
        "backend_requested": BACKEND,
        "prob_dist": prob_dist.name,
        "max_nvs_per_position": int(max_nvs_per_position),
        "force_nvs": None if force_nvs is None else int(force_nvs),
        "bic_extra_nv_penalty": float(bic_extra_nv_penalty),
        "ok": ok,
        "n_nvs_est": n_nvs_est,
        "threshold_any": threshold_any,
        "thresholds_multiclass": thresholds_multiclass,
        "readout_fidelity_any": readout_fidelity_any,
        "fidelity_multiclass": fidelity_multiclass,
        "prep_fidelity_any_ref": prep_fidelity_any_ref,
        "p_minus": p_minus,
        "bg": bg,
        "rate0": rate0,
        "delta": delta,
        "red_chi_sq": red_chi_sq,
        "ref_p_any_minus": ref_p_any_minus,
        "ref_mean_num_minus": ref_mean_num_minus,
        "weights_k": weights_k,
        "model": model,
        "best_candidate_model": best_candidate_model,
        "feedback_params": feedback_params,
    }


# =============================================================================
# Optional GPU fitting wrapper
# =============================================================================


def make_gpu_config():
    if GpuMultimodeFitConfig is None:
        return None

    try:
        return GpuMultimodeFitConfig(
            max_nvs=MAX_NVS_PER_POSITION,
            fit_chunk_size=GPU_FIT_CHUNK_SIZE,
            candidate_chunk_size=GPU_CANDIDATE_CHUNK_SIZE,
            refine_fit_chunk_size=GPU_REFINE_FIT_CHUNK_SIZE,
        )
    except TypeError:
        # Your local config class may have different fields.
        try:
            return GpuMultimodeFitConfig(max_nvs=MAX_NVS_PER_POSITION)
        except Exception:
            return None


def fit_gpu_best_effort(ref_counts_lists, model_kind, prob_dist):
    """
    Best-effort GPU call.

    Because the exact return schema of your GPU fitter may differ between files,
    this wrapper tries to call it and then maps common output keys.

    If this fails, caller should fall back to CPU.
    """

    if not (CUPY_AVAILABLE and GPU_FITTER_AVAILABLE):
        raise RuntimeError("CuPy or fit_charge_histograms_gpu_batch is unavailable.")

    counts_mat = np.stack([
        np.asarray(c, dtype=float).flatten()
        for c in ref_counts_lists
    ], axis=0)

    model_mode = GPU_MODEL_MODE_SINGLE if model_kind == "single" else GPU_MODEL_MODE_MULTI
    config = make_gpu_config()

    print("\n=== GPU fitting best-effort ===")
    print("counts_mat shape:", counts_mat.shape)
    print("model_kind:", model_kind)
    print("model_mode:", model_mode)

    kwargs = {
        "prob_dist": prob_dist,
        "model_mode": model_mode,
    }

    if config is not None:
        kwargs["config"] = config

    try:
        gpu_result = fit_charge_histograms_gpu_batch(counts_mat, **kwargs)
    except TypeError:
        # Some versions use prob_dist_name or omit config.
        kwargs2 = {
            "prob_dist_name": prob_dist.name,
            "model_mode": model_mode,
        }
        gpu_result = fit_charge_histograms_gpu_batch(counts_mat, **kwargs2)

    if not isinstance(gpu_result, dict):
        raise RuntimeError(
            "GPU fitter returned a non-dict object. "
            "Please adapt fit_gpu_best_effort() to your local GPU return schema."
        )

    # If your GPU fitter already returns a complete analysis-like dictionary,
    # preserve it and add a marker.
    gpu_result = dict(gpu_result)
    gpu_result["analysis_type"] = f"single_step_{model_kind}_gpu_raw_output"
    gpu_result["backend_used"] = "gpu"
    gpu_result["prob_dist"] = prob_dist.name

    return gpu_result


# =============================================================================
# Analysis coordinator
# =============================================================================


def process_single_step_charge_histograms(
    raw_data,
    model_kind="single",
    backend="cpu",
    fit_exp_ind=1,
    sig_exp_ind=0,
    prob_dist=ProbDist.COMPOUND_POISSON,
    n_jobs=12,
    joblib_verbose=10,
    max_nvs_per_position=3,
    force_nvs=None,
    bic_extra_nv_penalty=2.0,
    save_analysis=True,
):
    t_total = now()

    model_kind = str(model_kind).lower()
    backend = str(backend).lower()

    if model_kind not in {"single", "multi"}:
        raise ValueError("model_kind must be 'single' or 'multi'.")

    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'.")

    print_runtime_header(
        f"Single-step charge histogram analysis: {model_kind}",
        backend=backend,
        n_jobs=n_jobs,
    )

    nv_list = raw_data["nv_list"]
    num_positions = len(nv_list)

    print("num positions:", num_positions)
    print("fit_exp_ind:", fit_exp_ind)
    print("sig_exp_ind:", sig_exp_ind)
    print("counts shape:", np.asarray(raw_data["counts"]).shape)

    t_counts = now()
    sig_counts_lists, ref_counts_lists = extract_ref_and_sig_counts(
        raw_data,
        fit_exp_ind=fit_exp_ind,
        sig_exp_ind=sig_exp_ind,
    )
    print_elapsed("Count extraction", t_counts)

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------
    t_fit = now()
    backend_used = backend

    if backend == "gpu":
        try:
            analysis = fit_gpu_best_effort(
                ref_counts_lists,
                model_kind=model_kind,
                prob_dist=prob_dist,
            )
        except Exception:
            print("\nGPU fitting failed or unavailable. Falling back to CPU.")
            print(traceback.format_exc())
            backend_used = "cpu_fallback"

            if model_kind == "single":
                analysis = fit_single_cpu(
                    ref_counts_lists,
                    prob_dist=prob_dist,
                    n_jobs=n_jobs,
                    joblib_verbose=joblib_verbose,
                )
            else:
                analysis = fit_multi_cpu(
                    ref_counts_lists,
                    prob_dist=prob_dist,
                    max_nvs_per_position=max_nvs_per_position,
                    force_nvs=force_nvs,
                    bic_extra_nv_penalty=bic_extra_nv_penalty,
                    n_jobs=n_jobs,
                    joblib_verbose=joblib_verbose,
                )
    else:
        if model_kind == "single":
            analysis = fit_single_cpu(
                ref_counts_lists,
                prob_dist=prob_dist,
                n_jobs=n_jobs,
                joblib_verbose=joblib_verbose,
            )
        else:
            analysis = fit_multi_cpu(
                ref_counts_lists,
                prob_dist=prob_dist,
                max_nvs_per_position=max_nvs_per_position,
                force_nvs=force_nvs,
                bic_extra_nv_penalty=bic_extra_nv_penalty,
                n_jobs=n_jobs,
                joblib_verbose=joblib_verbose,
            )

    fit_elapsed = print_elapsed("Fit time", t_fit)

    analysis["backend_used"] = backend_used
    analysis["model_kind"] = model_kind
    analysis["fit_exp_ind"] = int(fit_exp_ind)
    analysis["sig_exp_ind"] = None if sig_exp_ind is None else int(sig_exp_ind)
    analysis["num_positions"] = int(num_positions)
    analysis["fit_elapsed_s"] = float(fit_elapsed)

    if "ok" in analysis:
        ok = np.asarray(analysis["ok"], dtype=bool)
        print("Good fits:", int(np.sum(ok)), "/", int(ok.size))

    # -------------------------------------------------------------------------
    # Attach and save
    # -------------------------------------------------------------------------
    raw_data["single_step_charge_histogram"] = analysis

    if save_analysis:
        timestamp = dm.get_time_stamp()
        base_name = (
            raw_data.get("file_stem")
            or raw_data.get("file_name")
            or raw_data.get("timestamp")
            or "raw_data"
        )
        if isinstance(base_name, (list, tuple)):
            base_name = "_".join(map(str, base_name))
        base_name = str(base_name).replace(" ", "_")

        file_name = (
            f"single_step_charge_hist_{model_kind}_{backend_used}_{base_name}"
        )
        file_path = dm.get_file_path(__file__, timestamp, file_name)

        save_dict = {
            "timestamp": timestamp,
            "source_file": base_name,
            "single_step_charge_histogram": make_json_safe(analysis),
        }

        dm.save_raw_data(save_dict, file_path, keys_to_compress=[])
        analysis["saved_file_path"] = str(file_path)
        print("Saved analysis:", file_path)

    total_elapsed = print_elapsed("Total analysis time", t_total)
    analysis["total_elapsed_s"] = float(total_elapsed)

    return analysis


# =============================================================================
# Plotting helpers
# =============================================================================


def get_analysis(raw_data_or_analysis):
    if isinstance(raw_data_or_analysis, dict) and "single_step_charge_histogram" in raw_data_or_analysis:
        return raw_data_or_analysis["single_step_charge_histogram"]
    return raw_data_or_analysis


def plot_prep_vs_readout_single_step(raw_data_or_analysis, use_multiclass=False):
    """
    Simple single-step scatter.

    For MODEL_KIND='single':
        x = prep_fidelity
        y = readout_fidelity

    For MODEL_KIND='multi':
        x = prep_fidelity_any_ref
        y = readout_fidelity_any, or fidelity_multiclass if use_multiclass=True
    """
    analysis = get_analysis(raw_data_or_analysis)
    model_kind = analysis.get("model_kind", "single")

    if model_kind == "single":
        y = np.asarray(analysis["readout_fidelity"], dtype=float)
        x = np.asarray(analysis["prep_fidelity"], dtype=float)
        xlabel = "Prep fidelity"
        ylabel = "Readout fidelity"
    else:
        x = np.asarray(analysis["prep_fidelity_any_ref"], dtype=float)
        if use_multiclass:
            y = np.asarray(analysis["fidelity_multiclass"], dtype=float)
            ylabel = "Multi-class readout fidelity"
        else:
            y = np.asarray(analysis["readout_fidelity_any"], dtype=float)
            ylabel = "Binary any-NV$^{-}$ readout fidelity"
        xlabel = "Prep fidelity, 1 - P(k=0)"

    ok = np.asarray(analysis.get("ok", np.ones_like(x, dtype=bool)), dtype=bool)
    good = ok & np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6.5, 5))

    kpl.plot_points(
        ax,
        x[good],
        y[good],
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title(f"Single-step {model_kind}: prep vs readout fidelity")

    if np.any(good):
        txt = (
            f"N good = {int(np.sum(good))}\n"
            f"median x = {np.nanmedian(x[good]):.3f}\n"
            f"median y = {np.nanmedian(y[good]):.3f}"
        )
        kpl.anchored_text(ax, txt, kpl.Loc.LOWER_RIGHT, size=kpl.Size.SMALL)

    return fig, ax


def plot_histograms_red_green(sig_counts, ref_counts, density=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 5))
    else:
        fig = None

    if sig_counts is not None:
        kpl.histogram(
            ax,
            np.asarray(sig_counts, dtype=float).flatten(),
            color=kpl.KplColors.RED,
            density=density,
            label="With ionization pulse",
        )

    kpl.histogram(
        ax,
        np.asarray(ref_counts, dtype=float).flatten(),
        color=kpl.KplColors.GREEN,
        density=density,
        label="Reference / fitted",
    )

    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Probability" if density else "Number of occurrences")
    ax.set_xlim(-0.5, None)

    if fig is not None:
        return fig, ax
    return None, ax


def plot_single_nv_hist_and_fit(raw_data, ind, density=True):
    analysis = get_analysis(raw_data)
    counts = np.asarray(raw_data["counts"], dtype=float)

    fit_exp_ind = int(analysis.get("fit_exp_ind", FIT_EXP_IND))
    sig_exp_ind = analysis.get("sig_exp_ind", SIG_EXP_IND)

    ref_counts = flatten_branch_counts(counts, fit_exp_ind, ind)
    sig_counts = None if sig_exp_ind is None else flatten_branch_counts(counts, int(sig_exp_ind), ind)

    fig, ax = plot_histograms_red_green(sig_counts, ref_counts, density=density)

    ok = bool(np.asarray(analysis["ok"])[ind])
    if not ok:
        ax.set_title(f"Index {ind}: fit not OK")
        return fig, ax

    threshold = safe_float(np.asarray(analysis["threshold"], dtype=float)[ind])
    readout_fidelity = safe_float(np.asarray(analysis["readout_fidelity"], dtype=float)[ind])
    prep_fidelity = safe_float(np.asarray(analysis["prep_fidelity"], dtype=float)[ind])
    red_chi_sq = safe_float(np.asarray(analysis["red_chi_sq"], dtype=float)[ind])
    fit_params = analysis["fit_params"][ind]

    prob_dist = get_prob_dist(analysis.get("prob_dist", "COMPOUND_POISSON"))

    if np.isfinite(threshold):
        ax.axvline(
            threshold,
            color=kpl.KplColors.GRAY,
            ls="dashed",
            label="Threshold",
        )

    if fit_params is not None:
        popt = np.asarray(fit_params, dtype=float).ravel()
        x_max = max(np.nanmax(ref_counts), threshold if np.isfinite(threshold) else 0)
        x_vals = np.linspace(0, x_max + 1, 1000)

        single_num = bimodal_histogram.get_single_mode_num_params(prob_dist)
        single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)
        bimodal_pdf = bimodal_histogram.get_bimodal_pdf(prob_dist)

        dark = popt[0] * single_pdf(x_vals, *popt[1 : 1 + single_num])
        bright = (1.0 - popt[0]) * single_pdf(x_vals, *popt[1 + single_num :])
        combined = bimodal_pdf(x_vals, *popt)

        kpl.plot_line(ax, x_vals, dark, color=kpl.KplColors.RED, label="NV$^{0}$ mode")
        kpl.plot_line(ax, x_vals, bright, color=kpl.KplColors.GREEN, label="NV$^{-}$ mode")
        kpl.plot_line(ax, x_vals, combined, color=kpl.KplColors.BLUE, label="Combined fit")

    txt = (
        f"index = {ind}\n"
        f"prep = {prep_fidelity:.3f}\n"
        f"readout = {readout_fidelity:.3f}\n"
        f"red χ² = {red_chi_sq:.3f}"
    )
    kpl.anchored_text(ax, txt, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)

    ax.set_title(f"Single-NV histogram: index {ind}")
    ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=7)
    return fig, ax


def plot_multi_nv_hist_and_fit(raw_data, ind, density=True):
    analysis = get_analysis(raw_data)
    counts = np.asarray(raw_data["counts"], dtype=float)

    fit_exp_ind = int(analysis.get("fit_exp_ind", FIT_EXP_IND))
    sig_exp_ind = analysis.get("sig_exp_ind", SIG_EXP_IND)

    ref_counts = flatten_branch_counts(counts, fit_exp_ind, ind)
    sig_counts = None if sig_exp_ind is None else flatten_branch_counts(counts, int(sig_exp_ind), ind)

    fig, ax = plot_histograms_red_green(sig_counts, ref_counts, density=density)

    ok = bool(np.asarray(analysis["ok"])[ind])
    if not ok:
        ax.set_title(f"Index {ind}: fit not OK")
        return fig, ax

    n_est = int(np.asarray(analysis["n_nvs_est"], dtype=float)[ind])
    threshold_any = safe_float(np.asarray(analysis["threshold_any"], dtype=float)[ind])
    thresholds = np.asarray(analysis["thresholds_multiclass"][ind], dtype=float)
    weights = np.asarray(analysis["weights_k"][ind], dtype=float)

    readout_fidelity = safe_float(np.asarray(analysis["readout_fidelity_any"], dtype=float)[ind])
    fidelity_multi = safe_float(np.asarray(analysis["fidelity_multiclass"], dtype=float)[ind])
    prep = safe_float(np.asarray(analysis["prep_fidelity_any_ref"], dtype=float)[ind])
    p_minus = safe_float(np.asarray(analysis["p_minus"], dtype=float)[ind])
    bg = safe_float(np.asarray(analysis["bg"], dtype=float)[ind])
    rate0 = safe_float(np.asarray(analysis["rate0"], dtype=float)[ind])
    delta = safe_float(np.asarray(analysis["delta"], dtype=float)[ind])
    red_chi_sq = safe_float(np.asarray(analysis["red_chi_sq"], dtype=float)[ind])

    prob_dist = get_prob_dist(analysis.get("prob_dist", "COMPOUND_POISSON"))
    single_pdf = bimodal_histogram.get_single_mode_pdf(prob_dist)

    x_max = max(
        np.nanmax(ref_counts),
        np.nanmax(sig_counts) if sig_counts is not None else 0,
        threshold_any if np.isfinite(threshold_any) else 0,
    )
    x_vals = np.linspace(0, x_max + 1, 1000)

    base = bg + n_est * rate0
    combined = np.zeros_like(x_vals, dtype=float)

    for k in range(n_est + 1):
        lam_k = base + k * delta
        comp = float(weights[k]) * single_pdf(x_vals, lam_k)
        combined += comp
        kpl.plot_line(ax, x_vals, comp, label=f"fit k={k}")

    kpl.plot_line(ax, x_vals, combined, color=kpl.KplColors.BLUE, label="combined ref fit")

    for t in thresholds:
        if np.isfinite(t):
            ax.axvline(t, color=kpl.KplColors.GRAY, ls="dashed", lw=1)

    if np.isfinite(threshold_any):
        ax.axvline(
            threshold_any,
            color="black",
            ls="dashed",
            lw=2,
            label="any NV- threshold",
        )

    txt = (
        f"index = {ind}\n"
        f"N_est = {n_est}\n"
        f"prep = {prep:.3f}\n"
        f"fid_any = {readout_fidelity:.3f}\n"
        f"fid_multi = {fidelity_multi:.3f}\n"
        f"p_minus = {p_minus:.3f}\n"
        f"rate0 = {rate0:.2f}\n"
        f"delta = {delta:.2f}\n"
        f"red χ² = {red_chi_sq:.3f}"
    )
    kpl.anchored_text(ax, txt, kpl.Loc.CENTER_RIGHT, size=kpl.Size.SMALL)

    ax.set_title(f"Multi-NV histogram: index {ind}")
    ax.legend(loc=kpl.Loc.UPPER_RIGHT, fontsize=7)
    return fig, ax


def plot_example_histograms(raw_data, inds, density=True):
    analysis = get_analysis(raw_data)
    model_kind = analysis.get("model_kind", MODEL_KIND)

    for ind in inds:
        ind = int(ind)
        if ind >= int(analysis.get("num_positions", len(raw_data["nv_list"]))):
            continue

        try:
            if model_kind == "single":
                plot_single_nv_hist_and_fit(raw_data, ind, density=density)
            else:
                plot_multi_nv_hist_and_fit(raw_data, ind, density=density)
        except Exception:
            print(f"Could not plot histogram for index {ind}")
            print(traceback.format_exc())


def load_saved_single_step_analysis(raw_data, analysis_file_stem):
    """
    Load saved single-step charge histogram analysis and attach it to raw_data.

    This avoids rerunning the CPU/GPU fitting.
    """
    analysis_data = dm.get_raw_data(
        file_stem=analysis_file_stem,
        load_npz=True,
    )

    print("\nLoaded saved analysis file:")
    print(analysis_file_stem)

    print("\nTop-level saved keys:")
    print(list(analysis_data.keys()))

    if "single_step_charge_histogram" in analysis_data:
        analysis = analysis_data["single_step_charge_histogram"]
    else:
        # fallback if saved file is already the analysis dictionary
        analysis = analysis_data

    # Convert common fields back to numpy arrays.
    array_keys = [
        "ok",
        "threshold",
        "readout_fidelity",
        "prep_fidelity",
        "red_chi_sq",
        "n_nvs_est",
        "threshold_any",
        "readout_fidelity_any",
        "fidelity_multiclass",
        "prep_fidelity_any_ref",
        "p_minus",
        "bg",
        "rate0",
        "delta",
        "ref_p_any_minus",
        "ref_mean_num_minus",
    ]

    for key in array_keys:
        if key in analysis:
            analysis[key] = np.asarray(analysis[key])

    raw_data["single_step_charge_histogram"] = analysis

    print("\nAttached saved analysis to raw_data['single_step_charge_histogram']")
    print("Analysis keys:")
    print(list(analysis.keys()))

    if "ok" in analysis:
        ok = np.asarray(analysis["ok"], dtype=bool)
        print("Good fits:", int(np.sum(ok)), "/", len(ok))

    return analysis, analysis_data


def print_single_step_saved_summary(raw_data):
    """
    Print useful numbers from saved single-step analysis.
    """
    analysis = get_analysis(raw_data)
    model_kind = analysis.get("model_kind", "single")

    print("\n=== Saved single-step analysis summary ===")
    print("model_kind:", model_kind)
    print("backend_used:", analysis.get("backend_used", None))
    print("prob_dist:", analysis.get("prob_dist", None))
    print("fit_exp_ind:", analysis.get("fit_exp_ind", None))
    print("sig_exp_ind:", analysis.get("sig_exp_ind", None))
    print("num_positions:", analysis.get("num_positions", None))

    if "ok" in analysis:
        ok = np.asarray(analysis["ok"], dtype=bool)
        print("Good fits:", int(np.sum(ok)), "/", len(ok))

    if model_kind == "single":
        prep = np.asarray(analysis["prep_fidelity"], dtype=float)
        readout = np.asarray(analysis["readout_fidelity"], dtype=float)
        threshold = np.asarray(analysis["threshold"], dtype=float)

        good = np.asarray(analysis.get("ok", np.ones_like(prep, dtype=bool)), dtype=bool)
        good = good & np.isfinite(prep) & np.isfinite(readout)

        print("median prep fidelity:", np.nanmedian(prep[good]))
        print("median readout fidelity:", np.nanmedian(readout[good]))
        print("median threshold:", np.nanmedian(threshold[good]))

    else:
        prep = np.asarray(analysis["prep_fidelity_any_ref"], dtype=float)
        fid_any = np.asarray(analysis["readout_fidelity_any"], dtype=float)
        fid_multi = np.asarray(analysis["fidelity_multiclass"], dtype=float)
        n_est = np.asarray(analysis["n_nvs_est"], dtype=float)

        good = np.asarray(analysis.get("ok", np.ones_like(prep, dtype=bool)), dtype=bool)
        good = good & np.isfinite(prep)

        print("median prep any-NV-:", np.nanmedian(prep[good]))
        print("median readout fidelity any:", np.nanmedian(fid_any[good]))
        print("median multiclass fidelity:", np.nanmedian(fid_multi[good]))

        for n in sorted(np.unique(n_est[np.isfinite(n_est)])):
            n = int(n)
            num = int(np.sum(good & (np.rint(n_est).astype(int) == n)))
            print(f"num estimated {n}-NV pillars:", num)


def get_best_single_step_inds(raw_data, num_examples=12, sort_key="readout_fidelity"):
    """
    Return best indices from saved analysis.

    sort_key options for single:
        readout_fidelity
        prep_fidelity
        red_chi_sq   # lower is better

    sort_key options for multi:
        fidelity_multiclass
        readout_fidelity_any
        prep_fidelity_any_ref
        red_chi_sq   # lower is better
    """
    analysis = get_analysis(raw_data)

    ok = np.asarray(analysis.get("ok", []), dtype=bool)
    if ok.size == 0:
        ok = np.ones(int(analysis.get("num_positions", len(raw_data["nv_list"]))), dtype=bool)

    vals = np.asarray(analysis[sort_key], dtype=float)

    good = ok & np.isfinite(vals)
    valid_inds = np.where(good)[0]

    if sort_key == "red_chi_sq":
        order = valid_inds[np.argsort(vals[valid_inds])]
    else:
        order = valid_inds[np.argsort(vals[valid_inds])[::-1]]

    return order[:num_examples]


def save_selected_single_step_histograms(
    raw_data,
    inds,
    label="saved-single-step-hist",
    density=True,
    close_figs=True,
):
    """
    Save histogram figures for selected NV/pillar indices using saved analysis.
    """
    analysis = get_analysis(raw_data)
    timestamp = raw_data.get("timestamp", dm.get_time_stamp())
    saved_paths = []

    for ind in inds:
        ind = int(ind)

        try:
            model_kind = analysis.get("model_kind", "single")

            if model_kind == "single":
                fig, ax = plot_single_nv_hist_and_fit(
                    raw_data,
                    ind,
                    density=density,
                )
            else:
                fig, ax = plot_multi_nv_hist_and_fit(
                    raw_data,
                    ind,
                    density=density,
                )

            file_path = dm.get_file_path(
                __file__,
                timestamp,
                f"{label}-ind{ind}",
            )

            dm.save_figure(fig, file_path)
            saved_paths.append(file_path)

            print("Saved:", file_path)

            if close_figs:
                plt.close(fig)

        except Exception:
            print(f"Could not save histogram for index {ind}")
            print(traceback.format_exc())

    return saved_paths

if __name__ == "__main__":
    kpl.init_kplotlib()
    # =============================================================================
    # User settings
    # =============================================================================
    # FILE_ID = "2026_07_10-12_06_57-qnami-nv0_2026_02_20"
    # FILE_ID = "2026_07_10-16_57_47-qnami-nv0_2026_02_20", ## 814 working NVs 200ms readout
    # FILE_ID = "2026_07_13-17_00_15-qnami-nv0_2026_02_20", ## 814 working NVs 100ms readout
    # FILE_ID = "2026_07_14-13_06_11-qnami-nv0_2026_02_20", ## 814 working NVs 100ms readout
    # FILE_ID = "2026_07_15-16_51_54-qnami-nv0_2026_02_20", ## 631 working NVs 100ms readout
    # FILE_ID = "2026_07_16-22_48_08-qnami-nv0_2026_02_20", ## 631 working NVs 100ms readout
    # FILE_ID = "2026_07_17-19_02_51-qnami-nv0_2026_02_20", ## 631 working NVs 100ms readout
    # FILE_ID = "2026_07_17-22_35_56-qnami-nv0_2026_02_20", ## 631 working NVs 100ms readout
    # FILE_ID = "2026_07_19-00_17_00-qnami-nv0_2026_02_20", ## 631 working NVs 500ms readout
    # FILE_ID = "2026_07_20-16_19_40-qnami-nv0_2026_02_20", ## 631 working NVs 1000ms readout
    # FILE_ID = "2026_07_20-17_04_32-qnami-nv0_2026_02_20", ## 631 working NVs 1000ms readout
    # FILE_ID = "2026_07_21-15_39_42-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_07_21-16_08_28-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_07_22-16_28_38-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_07_22-16_56_52-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_07_22-17_30_06-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_07_22-21_22_04-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_07_22-22_20_35-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_08_04-13_21_07-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_08_04-16_25_34-qnami-nv0_2026_02_20" ## 631 working NVs 500ms readout
    # FILE_ID = "2026_08_08-15_17_54-qnami-nv0_2026_02_20" ## 415 working NVs 500ms readout
    # FILE_ID = "2026_08_08-16_00_34-qnami-nv0_2026_02_20" ## 415 working NVs 500ms readout
    # FILE_ID = "2026_08_08-16_44_44-qnami-nv0_2026_02_20" ## 415 working NVs 500ms readout
    # FILE_ID = "2026_08_08-17_21_48-qnami-nv0_2026_02_20" ## 402 working NVs 500ms readout
    # FILE_ID = "2026_08_08-18_15_06-qnami-nv0_2026_02_20" ## 404 working NVs 500ms readout
    # FILE_ID = "2026_08_08-18_48_51-qnami-nv0_2026_02_20" ## 402 working NVs 500ms readout
    # FILE_ID = "2026_08_08-19_19_02-qnami-nv0_2026_02_20" ## 366 working NVs 500ms readout
    # FILE_ID = "2026_08_08-20_30_31-qnami-nv0_2026_02_20" ## 366 working NVs 500ms readout
    # FILE_ID = "2026_08_08-22_18_32-qnami-nv0_2026_02_20" ## 351 working NVs 500ms readout
    FILE_ID = "2026_08_08-22_52_19-qnami-nv0_2026_02_20" ## 366 working NVs 500ms readout

    # SAVED_ANALYSIS_FILE_ID = "2026_07_15-19_48_48-single_step_charge_hist_single_cpu_2026_07_15-19_42_19-qnami-nv0_2026_02_20"
    # SAVED_ANALYSIS_FILE_ID = "2026_07_21-16_11_27-single_step_charge_hist_single_cpu_2026_07_21-16_08_28-qnami-nv0_2026_02_20"
    SAVED_ANALYSIS_FILE_ID = "2026_08_04-13_25_01-single_step_charge_hist_single_cpu_2026_08_04-13_21_07-qnami-nv0_2026_02_20"
 
    RUN_NEW_PROCESSING =True

    MODEL_KIND = "single"
    BACKEND = "cpu"

    SIG_EXP_IND = 0
    FIT_EXP_IND = 1

    PROB_DIST = ProbDist.COMPOUND_POISSON
    N_JOBS = 12
    JOBLIB_VERBOSE = 10

    MAX_NVS_PER_POSITION = 3
    FORCE_NVS: Optional[int] = None
    BIC_EXTRA_NV_PENALTY = 2.0

    GPU_MODEL_MODE_SINGLE = "bimodal"
    GPU_MODEL_MODE_MULTI = "strict_auto"
    GPU_FIT_CHUNK_SIZE = 128
    GPU_CANDIDATE_CHUNK_SIZE = 128
    GPU_REFINE_FIT_CHUNK_SIZE = 32

    SAVE_ANALYSIS = True

    DO_PLOT_SUMMARY = True
    DO_PLOT_EXAMPLE_HISTS = True
    DO_SAVE_SELECTED_HISTS = True

    NUM_EXAMPLES = 12

    # For single-NV:
    SORT_KEY = "readout_fidelity"
    # SORT_KEY = "prep_fidelity"
    # SORT_KEY = "red_chi_sq"

    # Manual examples if desired.
    EXAMPLE_INDS = [0, 1, 2, 3, 10, 50,  100, 200, 300, 400, 500, 600]

    # EXAMPLE_INDS =[8, 303, 364, 422, 463, 536]    #Lost in every run
    # EXAMPLE_INDS = [13, 25, 37, 51, 69, 87, 99, 149, 181, 222, 248, 249, 294, 323, 332, 340, 413, 459, 460, 480, 491, 510, 619] 
    # =============================================================================
    # Load original raw data
    # =============================================================================
    raw_data = dm.get_raw_data(
        file_stem=FILE_ID,
        load_npz=True,
    )

    raw_data["file_stem"] = FILE_ID

    # =============================================================================
    # Either run new processing or load saved analysis
    # =============================================================================
    if RUN_NEW_PROCESSING:
        analysis = process_single_step_charge_histograms(
            raw_data,
            model_kind=MODEL_KIND,
            backend=BACKEND,
            fit_exp_ind=FIT_EXP_IND,
            sig_exp_ind=SIG_EXP_IND,
            prob_dist=PROB_DIST,
            n_jobs=N_JOBS,
            joblib_verbose=JOBLIB_VERBOSE,
            max_nvs_per_position=MAX_NVS_PER_POSITION,
            force_nvs=FORCE_NVS,
            bic_extra_nv_penalty=BIC_EXTRA_NV_PENALTY,
            save_analysis=SAVE_ANALYSIS,
        )
    else:
        analysis, analysis_data = load_saved_single_step_analysis(
            raw_data,
            SAVED_ANALYSIS_FILE_ID,
        )

    # =============================================================================
    # Print summary
    # =============================================================================
    print_single_step_saved_summary(raw_data)

    # =============================================================================
    # Plot summary
    # =============================================================================
    if DO_PLOT_SUMMARY:
        plot_prep_vs_readout_single_step(
            raw_data,
            use_multiclass=False,
        )

        if analysis.get("model_kind", MODEL_KIND) == "multi":
            plot_prep_vs_readout_single_step(
                raw_data,
                use_multiclass=True,
            )

    # =============================================================================
    # Choose examples from saved analysis
    # =============================================================================
    # best_inds = get_best_single_step_inds(
    #     raw_data,
    #     num_examples=NUM_EXAMPLES,
    #     sort_key=SORT_KEY,
    # )

    # print(f"\nBest examples by {SORT_KEY}:")
    # print(best_inds)

    # Use best indices by default.
    # inds_to_plot = best_inds
    # Or use manual examples:
    inds_to_plot = EXAMPLE_INDS

    # =============================================================================
    # Plot example histograms
    # =============================================================================
    if DO_PLOT_EXAMPLE_HISTS:
        plot_example_histograms(
            raw_data,
            inds_to_plot,
            density=True,
        )

    # =============================================================================
    # Save selected histograms
    # =============================================================================

    # if DO_SAVE_SELECTED_HISTS:
    #     save_selected_single_step_histograms(
    #         raw_data,
    #         inds_to_plot,
    #         label=f"best-{SORT_KEY}-single-step",
    #         density=True,
    #         close_figs=True,
    #     )

    kpl.show(block=True)
