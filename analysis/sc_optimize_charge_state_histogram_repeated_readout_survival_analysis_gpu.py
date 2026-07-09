# -*- coding: utf-8 -*-
"""
GPU repeated charge-readout survival analysis.

Experiment order for repeated_readout=True:
    exp 0 = ionized branch, readout 1
    exp 1 = ionized branch, readout 2
    exp 2 = reference/no-ionization branch, readout 1
    exp 3 = reference/no-ionization branch, readout 2

Goal:
    Optimize readout amplitude/duration using:
        1. Single-shot charge classification fidelity.
        2. Charge-state survival between readout 1 and readout 2 with no re-prep.

GPU path:
    - Batched CuPy coarse bimodal fits for R1 and R2 histograms.
    - GPU threshold search from the fitted compound-Poisson rates.
    - GPU survival metrics from raw repeated-readout counts.

This is meant to be fast. Keep the CPU SciPy script as the reference check for
important final numbers.

Created July 2026
@author: Saroj Chand
"""

from __future__ import annotations

import os
import sys
import math

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.special import gammaln as scipy_gammaln
except Exception:
    scipy_gammaln = None

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

try:
    import cupy as cp
    from cupyx.scipy.special import gammaln as cp_gammaln

    GPU_AVAILABLE = True
except Exception:
    cp = None
    cp_gammaln = None
    GPU_AVAILABLE = False

from analysis.sc_gpu_bimodal_fitting import (
    GpuFitConfig,
    GpuMultimodeFitConfig,
    determine_thresholds_any_minus_gpu,
    fit_charge_histograms_gpu_batch,
    summarize_gpu,
)
from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# Basic helpers
# =============================================================================


def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def _base_file_stem(raw_data):
    stem = (
        raw_data.get("file_stem")
        or raw_data.get("file_name")
        or raw_data.get("timestamp")
        or "raw_data"
    )
    if isinstance(stem, (list, tuple)):
        stem = "_".join(map(str, stem))
    return str(stem).replace(" ", "_")


def _norm01(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out
    xmin = np.nanmin(x[finite])
    xmax = np.nanmax(x[finite])
    out[finite] = (x[finite] - xmin) / (xmax - xmin + 1e-12)
    out[~finite] = np.nan
    return out


def _get_readout_axis(raw_data):
    min_step_val = raw_data["min_step_val"]
    max_step_val = raw_data["max_step_val"]
    num_steps = int(raw_data["num_steps"])

    step_vals_raw = np.linspace(min_step_val, max_step_val, num_steps)
    optimize_pol_or_readout = raw_data["optimize_pol_or_readout"]
    optimize_duration_or_amp = raw_data["optimize_duration_or_amp"]

    a, b, c = 1.5133e04, 2.6976, -38.63

    if optimize_pol_or_readout:
        x_label = (
            "Polarization duration (ns)"
            if optimize_duration_or_amp
            else "Polarization amplitude"
        )
        return {
            "step_vals_raw": step_vals_raw,
            "step_vals": step_vals_raw,
            "x_label": x_label,
            "power_fit_a": a,
            "power_fit_b": b,
            "power_fit_c": c,
            "yellow_charge_readout_amp": np.nan,
        }

    if optimize_duration_or_amp:
        return {
            "step_vals_raw": step_vals_raw,
            "step_vals": step_vals_raw * 1e-6,
            "x_label": "Readout duration (ms)",
            "power_fit_a": a,
            "power_fit_b": b,
            "power_fit_c": c,
            "yellow_charge_readout_amp": np.nan,
        }

    yellow_charge_readout_amp = raw_data["opx_config"]["waveforms"][
        "yellow_charge_readout"
    ]["sample"]
    aom_voltage = step_vals_raw * yellow_charge_readout_amp
    readout_power_uW = a * (aom_voltage**b) + c

    return {
        "step_vals_raw": step_vals_raw,
        "step_vals": readout_power_uW,
        "x_label": "Readout amplitude (uW)",
        "power_fit_a": a,
        "power_fit_b": b,
        "power_fit_c": c,
        "yellow_charge_readout_amp": float(yellow_charge_readout_amp),
    }


def _aom_voltage_from_step_value(step_val, axis_info):
    if axis_info["x_label"] != "Readout amplitude (uW)":
        return np.nan

    a = float(axis_info["power_fit_a"])
    b = float(axis_info["power_fit_b"])
    c = float(axis_info["power_fit_c"])

    if not np.isfinite(step_val) or step_val <= c:
        return np.nan

    return float(((step_val - c) / a) ** (1 / b))




def _pminus_to_any_minus_population(p_minus_arr, n_nvs_arr):
    """
    Convert per-NV NV- probability p_minus into probability of at least one NV-
    in a pillar.

    For an N-NV pillar:
        P(any NV-) = 1 - (1 - p_minus)**N

    This is the physically useful binary population for the all-NV0 vs any-NV-
    threshold used in this repeated-readout analysis.
    """
    p = np.asarray(p_minus_arr, dtype=float)
    n = np.asarray(n_nvs_arr, dtype=float)
    n = np.where(np.isfinite(n) & (n >= 1), n, 1.0)
    p = np.clip(p, 0.0, 1.0)
    out = 1.0 - (1.0 - p) ** n
    out[~np.isfinite(p)] = np.nan
    return out



def _as_float(value, default=np.nan):
    """Convert saved JSON values such as None to a plotting-safe float."""
    if value is None:
        return float(default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out


def _as_int(value, default=None):
    """Convert saved JSON values to int, returning default for None/nan."""
    if value is None:
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return int(out)
# =============================================================================
# kplotlib style helpers
# =============================================================================


def _kpl_color(name, fallback):
    try:
        return getattr(kpl.KplColors, name)
    except Exception:
        return fallback


def _plot_line(ax, x, y, label=None, color=None, linestyle="-", linewidth=2):
    try:
        return kpl.plot_line(
            ax,
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    except TypeError:
        try:
            return kpl.plot_line(
                ax,
                x,
                y,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        except Exception:
            return ax.plot(
                x,
                y,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
    except Exception:
        return ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )


def _plot_points(ax, x, y, label=None, color=None, size=18, alpha=0.65):
    try:
        return kpl.plot_points(
            ax,
            x,
            y,
            label=label,
            color=color,
            size=size,
            alpha=alpha,
        )
    except TypeError:
        try:
            return kpl.plot_points(ax, x, y, label=label, color=color)
        except Exception:
            return ax.scatter(x, y, label=label, color=color, s=size, alpha=alpha)
    except Exception:
        return ax.scatter(x, y, label=label, color=color, s=size, alpha=alpha)


def _plot_histogram(
    ax,
    data,
    bins=60,
    density=True,
    label=None,
    color=None,
    alpha=0.45,
):
    try:
        return kpl.histogram(
            ax,
            data,
            bins=bins,
            density=density,
            label=label,
            color=color,
            alpha=alpha,
        )
    except TypeError:
        try:
            return kpl.histogram(
                ax,
                data,
                density=density,
                label=label,
                color=color,
            )
        except Exception:
            return ax.hist(
                data,
                bins=bins,
                density=density,
                label=label,
                color=color,
                alpha=alpha,
            )
    except Exception:
        return ax.hist(
            data,
            bins=bins,
            density=density,
            label=label,
            color=color,
            alpha=alpha,
        )


def _mark_optimal(ax, opt_val, label=None):
    if not np.isfinite(opt_val):
        return
    if label is None:
        label = f"Optimal = {opt_val:.3g}"
    ax.axvline(
        opt_val,
        color=_kpl_color("RED", "red"),
        linestyle="--",
        linewidth=1.8,
        label=label,
    )


def _style_axis(ax):
    ax.grid(True, linestyle="--", alpha=0.45)


# =============================================================================
# CPU fit-curve helpers for plotting saved GPU fits
# =============================================================================


def _gammaln_cpu(x):
    if scipy_gammaln is not None:
        return scipy_gammaln(x)
    return np.vectorize(math.lgamma)(x)


def _poisson_pdf_cpu(x, rate, eps=1e-12):
    x = np.asarray(x, dtype=float)
    rate = max(float(rate), eps)
    log_pdf = np.where(x == 0, 0.0, x * np.log(rate)) - rate - _gammaln_cpu(x + 1.0)
    return np.exp(log_pdf)


def _compound_poisson_pdf_cpu(z, rate, nsig=5.0, min_lim=10, max_lim=50_000):
    z = np.asarray(z, dtype=float)
    rate = max(float(rate), 1e-12)
    upper = int(
        min(
            max(int(np.ceil(rate + nsig * np.sqrt(max(rate, 0.0)))), int(min_lim)),
            int(max_lim),
        )
    )
    k = np.arange(0, upper, dtype=float)
    p_z_given_k = np.zeros((k.size, z.size), dtype=float)

    if k.size > 0:
        p_z_given_k[0, :] = 0.0
        p_z_given_k[0, np.isclose(z, 0.0)] = 1.0

    if k.size > 1:
        k_nonzero = k[1:]
        log_pdf = (
            z[None, :] * np.log(k_nonzero[:, None])
            - k_nonzero[:, None]
            - _gammaln_cpu(z[None, :] + 1.0)
        )
        p_z_given_k[1:, :] = np.exp(log_pdf)

    p_k = _poisson_pdf_cpu(k, rate)
    return p_k @ p_z_given_k


def _binom_weights_cpu(n_nvs, p_minus):
    p_minus = float(np.clip(p_minus, 0.0, 1.0))
    ks = np.arange(int(n_nvs) + 1, dtype=float)
    coeff = np.asarray([math.comb(int(n_nvs), int(k)) for k in ks], dtype=float)
    weights = coeff * (p_minus**ks) * ((1.0 - p_minus) ** (int(n_nvs) - ks))
    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(total):
        weights = np.zeros(int(n_nvs) + 1, dtype=float)
        weights[0] = 1.0
        return weights
    return weights / total


def _gpu_equal_model_fit_curves(x_vals, popt, n_nvs):
    """
    Build k=0..N component curves for saved GPU equal-brightness fit params.

    popt = [p_minus, bg, rate0, delta]
    lambda_k = bg + N*rate0 + k*delta
    """
    p_minus, bg, rate0, delta = [float(v) for v in np.asarray(popt, dtype=float)]
    n_nvs = int(max(n_nvs, 1))
    weights = _binom_weights_cpu(n_nvs, p_minus)
    ks = np.arange(n_nvs + 1, dtype=int)
    rates = bg + n_nvs * max(rate0, 1e-12) + ks * max(delta, 0.0)

    components = []
    combined = np.zeros_like(np.asarray(x_vals, dtype=float), dtype=float)
    for k, weight, rate in zip(ks, weights, rates):
        pdf = _compound_poisson_pdf_cpu(x_vals, rate)
        weighted = float(weight) * pdf
        components.append(
            {
                "k": int(k),
                "weight": float(weight),
                "rate": float(rate),
                "pdf": pdf,
                "weighted_pdf": weighted,
            }
        )
        combined += weighted

    return combined, components


# =============================================================================
# GPU threshold helpers
# =============================================================================


def _poisson_pdf_table_gpu(x_vals_gpu, rates_gpu, eps=1e-12):
    rates_gpu = cp.maximum(rates_gpu.astype(cp.float64), eps)
    x_vals_gpu = x_vals_gpu.astype(cp.float64)

    log_rate = cp.log(rates_gpu)[:, None]
    x = x_vals_gpu[None, :]
    log_pdf = (
        cp.where(x == 0, 0.0, x * log_rate)
        - rates_gpu[:, None]
        - cp_gammaln(x + 1.0)
    )
    return cp.exp(log_pdf)


def _compound_poisson_pdf_table_gpu(x_vals_gpu, rates_gpu, eps=1e-12):
    rmax = float(cp.asnumpy(cp.nanmax(rates_gpu)))
    upper = int(max(10, np.ceil(rmax + 5 * np.sqrt(max(rmax, 0.0)))))
    k_vals_gpu = cp.arange(0, upper, dtype=cp.float64)

    k_safe = cp.maximum(k_vals_gpu, eps)
    z = x_vals_gpu.astype(cp.float64)[None, :]
    log_k = cp.log(k_safe)[:, None]

    log_p_z_given_k = (
        cp.where(z == 0, 0.0, z * log_k)
        - k_vals_gpu[:, None]
        - cp_gammaln(z + 1.0)
    )

    if upper > 0:
        log_p_z_given_k = log_p_z_given_k.copy()
        log_p_z_given_k[0, :] = -cp.inf
        log_p_z_given_k[0, 0] = 0.0

    p_z_given_k = cp.exp(log_p_z_given_k)
    p_k_given_rate = _poisson_pdf_table_gpu(k_vals_gpu, rates_gpu, eps=eps)
    return p_k_given_rate @ p_z_given_k


def _fit_results_to_arrays(gpu_fit_results, num_nvs, num_steps):
    popt_arr = np.full((num_nvs, num_steps, 3), np.nan, dtype=float)
    chi_arr = np.full((num_nvs, num_steps), np.nan, dtype=float)
    success_arr = np.zeros((num_nvs, num_steps), dtype=bool)

    for flat_ind, fit_res in enumerate(gpu_fit_results):
        nv_ind = flat_ind // num_steps
        step_ind = flat_ind % num_steps
        popt, pcov, red_chi_sq = fit_res
        if popt is None:
            continue
        popt_arr[nv_ind, step_ind, :] = np.asarray(popt, dtype=float)
        chi_arr[nv_ind, step_ind] = float(red_chi_sq)
        success_arr[nv_ind, step_ind] = True

    return popt_arr, chi_arr, success_arr


def _charge_fit_results_to_arrays(gpu_fit_results, num_nvs, num_steps):
    """
    Convert fit_charge_histograms_gpu_batch() dictionaries to analysis arrays.

    fit_params_arr columns:
        0 p_minus
        1 bg
        2 rate0
        3 delta

    For N>1, thresholding is interpreted as all NV0 vs any NV-.
    """
    popt_arr = np.full((num_nvs, num_steps, 4), np.nan, dtype=float)
    chi_arr = np.full((num_nvs, num_steps), np.nan, dtype=float)
    bic_arr = np.full((num_nvs, num_steps), np.nan, dtype=float)
    nll_arr = np.full((num_nvs, num_steps), np.nan, dtype=float)
    n_nvs_arr = np.zeros((num_nvs, num_steps), dtype=int)
    model_code_arr = np.zeros((num_nvs, num_steps), dtype=int)
    success_arr = np.zeros((num_nvs, num_steps), dtype=bool)

    model_codes = {
        "bimodal": 1,
        "1nv_equal": 11,
        "2nv_equal": 12,
        "3nv_equal": 13,
        "4nv_equal": 14,
    }

    for flat_ind, fit_res in enumerate(gpu_fit_results):
        nv_ind = flat_ind // num_steps
        step_ind = flat_ind % num_steps

        if not isinstance(fit_res, dict) or not fit_res.get("ok", False):
            continue

        popt = np.asarray(fit_res.get("popt", []), dtype=float)
        if popt.size != 4 or not np.all(np.isfinite(popt)):
            continue

        popt_arr[nv_ind, step_ind, :] = popt
        chi_arr[nv_ind, step_ind] = float(fit_res.get("red_chi_sq", np.nan))
        bic_arr[nv_ind, step_ind] = float(fit_res.get("bic", np.nan))
        nll_arr[nv_ind, step_ind] = float(fit_res.get("nll", np.nan))
        n_nvs_arr[nv_ind, step_ind] = int(fit_res.get("n_nvs", 1))
        model = str(fit_res.get("model", ""))
        model_code_arr[nv_ind, step_ind] = model_codes.get(model, 99)
        success_arr[nv_ind, step_ind] = True

    return {
        "popt": popt_arr,
        "chi": chi_arr,
        "bic": bic_arr,
        "nll": nll_arr,
        "n_nvs": n_nvs_arr,
        "model_code": model_code_arr,
        "success": success_arr,
    }


def _thresholds_from_gpu_fit(
    popt_arr,
    x_max,
    chunk_size=4096,
):
    """
    GPU threshold search for popt = [dark_weight, dark_rate, bright_rate].

    The search maximizes:
        0.5 * P_dark(count <= threshold)
      + 0.5 * P_bright(count > threshold)

    Returns:
        threshold_arr, readout_fidelity_arr
    """

    num_nvs, num_steps, _ = popt_arr.shape
    flat = popt_arr.reshape(-1, 3)
    valid = np.all(np.isfinite(flat), axis=1)

    threshold_flat = np.full(flat.shape[0], np.nan, dtype=float)
    fidelity_flat = np.full(flat.shape[0], np.nan, dtype=float)

    if not np.any(valid):
        return (
            threshold_flat.reshape(num_nvs, num_steps),
            fidelity_flat.reshape(num_nvs, num_steps),
        )

    rates = np.concatenate([flat[valid, 1], flat[valid, 2]])
    unique_rates, inverse = np.unique(rates, return_inverse=True)
    num_valid = int(np.sum(valid))
    dark_rate_ind = inverse[:num_valid]
    bright_rate_ind = inverse[num_valid:]

    x_vals_gpu = cp.arange(0, int(x_max) + 1, dtype=cp.float64)
    rates_gpu = cp.asarray(unique_rates, dtype=cp.float64)

    pdf = _compound_poisson_pdf_table_gpu(x_vals_gpu, rates_gpu)
    cdf = cp.cumsum(pdf, axis=1)
    cdf = cp.clip(cdf, 0.0, 1.0)

    zeros = cp.zeros((cdf.shape[0], 1), dtype=cp.float64)
    cdf_ext = cp.concatenate([zeros, cdf], axis=1)
    threshold_candidates = np.arange(-0.5, int(x_max) + 0.5 + 1e-12, 1.0)

    valid_indices = np.where(valid)[0]

    for start in range(0, num_valid, int(chunk_size)):
        stop = min(start + int(chunk_size), num_valid)

        dark_idx_gpu = cp.asarray(dark_rate_ind[start:stop], dtype=cp.int32)
        bright_idx_gpu = cp.asarray(bright_rate_ind[start:stop], dtype=cp.int32)

        dark_cdf = cdf_ext[dark_idx_gpu, :]
        bright_cdf = cdf_ext[bright_idx_gpu, :]

        fid = 0.5 * dark_cdf + 0.5 * (1.0 - bright_cdf)
        best = cp.argmax(fid, axis=1)
        best_fid = fid[cp.arange(fid.shape[0]), best]

        threshold_flat[valid_indices[start:stop]] = threshold_candidates[
            cp.asnumpy(best)
        ]
        fidelity_flat[valid_indices[start:stop]] = cp.asnumpy(best_fid)

    return (
        threshold_flat.reshape(num_nvs, num_steps),
        fidelity_flat.reshape(num_nvs, num_steps),
    )


# =============================================================================
# GPU processing
# =============================================================================


def _build_gpu_fit_batches(counts):
    """
    Build CPU lists of flattened count arrays for the GPU coarse fitter.

    R1 fit population:
        ion R1 + ref R1 = exp 0 + exp 2
    R2 fit population:
        ion R2 + ref R2 = exp 1 + exp 3
    """

    num_nvs = counts.shape[1]
    num_steps = counts.shape[3]

    r1_batch = []
    r2_batch = []

    for nv_ind in range(num_nvs):
        for step_ind in range(num_steps):
            ion_r1 = counts[0, nv_ind, :, step_ind, :].reshape(-1)
            ion_r2 = counts[1, nv_ind, :, step_ind, :].reshape(-1)
            ref_r1 = counts[2, nv_ind, :, step_ind, :].reshape(-1)
            ref_r2 = counts[3, nv_ind, :, step_ind, :].reshape(-1)
            r1_batch.append(np.concatenate([ion_r1, ref_r1]))
            r2_batch.append(np.concatenate([ion_r2, ref_r2]))

    return r1_batch, r2_batch


def _compute_survival_gpu(counts, threshold_arr):
    counts_gpu = cp.asarray(counts, dtype=cp.float64)
    threshold_gpu = cp.asarray(threshold_arr, dtype=cp.float64)
    threshold_gpu = threshold_gpu[:, None, :, None]

    ion_r1 = counts_gpu[0]
    ion_r2 = counts_gpu[1]
    ref_r1 = counts_gpu[2]
    ref_r2 = counts_gpu[3]

    ion_s1 = ion_r1 > threshold_gpu
    ion_s2 = ion_r2 > threshold_gpu
    ref_s1 = ref_r1 > threshold_gpu
    ref_s2 = ref_r2 > threshold_gpu

    axes = (1, 3)

    ref_same = cp.mean(ref_s1 == ref_s2, axis=axes)
    ion_same = cp.mean(ion_s1 == ion_s2, axis=axes)

    ref_nvm_num = cp.sum(ref_s1 & ref_s2, axis=axes)
    ref_nvm_den = cp.sum(ref_s1, axis=axes)
    ref_nvm = cp.where(ref_nvm_den > 0, ref_nvm_num / ref_nvm_den, cp.nan)

    ref_nv0_num = cp.sum((~ref_s1) & (~ref_s2), axis=axes)
    ref_nv0_den = cp.sum(~ref_s1, axis=axes)
    ref_nv0 = cp.where(ref_nv0_den > 0, ref_nv0_num / ref_nv0_den, cp.nan)

    total_ref = cp.asarray(ref_s1.shape[1] * ref_s1.shape[3], dtype=cp.float64)
    ref_nvm_population = ref_nvm_den / total_ref
    ref_nv0_population = ref_nv0_den / total_ref

    ion_nv0_num = cp.sum((~ion_s1) & (~ion_s2), axis=axes)
    ion_nv0_den = cp.sum(~ion_s1, axis=axes)
    ion_nv0 = cp.where(ion_nv0_den > 0, ion_nv0_num / ion_nv0_den, cp.nan)

    mean_ion_r1 = cp.mean(ion_r1, axis=axes)
    mean_ion_r2 = cp.mean(ion_r2, axis=axes)
    mean_ref_r1 = cp.mean(ref_r1, axis=axes)
    mean_ref_r2 = cp.mean(ref_r2, axis=axes)

    return {
        "ref_same_state_survival_arr": cp.asnumpy(ref_same),
        "ref_nvm_survival_arr": cp.asnumpy(ref_nvm),
        "ref_nv0_survival_arr": cp.asnumpy(ref_nv0),
        "ref_nvm_population_arr": cp.asnumpy(ref_nvm_population),
        "ref_nv0_population_arr": cp.asnumpy(ref_nv0_population),
        "ref_nvm_to_nv0_prob_arr": cp.asnumpy(1.0 - ref_nvm),
        "ref_nv0_to_nvm_prob_arr": cp.asnumpy(1.0 - ref_nv0),
        "ion_same_state_survival_arr": cp.asnumpy(ion_same),
        "ion_nv0_survival_arr": cp.asnumpy(ion_nv0),
        "ion_nv0_to_nvm_prob_arr": cp.asnumpy(1.0 - ion_nv0),
        "mean_ion_r1_arr": cp.asnumpy(mean_ion_r1),
        "mean_ion_r2_arr": cp.asnumpy(mean_ion_r2),
        "mean_ref_r1_arr": cp.asnumpy(mean_ref_r1),
        "mean_ref_r2_arr": cp.asnumpy(mean_ref_r2),
    }


def _choose_optimal_step(
    step_vals,
    readout1_fidelity,
    readout2_fidelity,
    ref_same,
    ref_nvm,
    ref_nv0,
    ref_nvm_population,
    ref_nv0_population,
    gof1,
    gof2,
    criteria,
    score_weights,
):
    good = (
        np.isfinite(readout1_fidelity)
        & np.isfinite(readout2_fidelity)
        & np.isfinite(ref_same)
        & np.isfinite(ref_nvm)
        & (readout1_fidelity >= criteria["min_readout1_fidelity"])
        & (readout2_fidelity >= criteria["min_readout2_fidelity"])
        & (ref_same >= criteria["min_ref_same_state_survival"])
        & (ref_nvm >= criteria["min_ref_nvm_survival"])
    )

    min_ref_nvm_population = criteria.get("min_ref_nvm_population", None)
    if min_ref_nvm_population is not None:
        good = (
            good
            & np.isfinite(ref_nvm_population)
            & (ref_nvm_population >= min_ref_nvm_population)
        )

    min_ref_nv0 = criteria.get("min_ref_nv0_survival", None)
    if min_ref_nv0 is not None:
        good = good & np.isfinite(ref_nv0) & (ref_nv0 >= min_ref_nv0)

    min_ref_nv0_population = criteria.get("min_ref_nv0_population", None)
    if min_ref_nv0_population is not None:
        good = (
            good
            & np.isfinite(ref_nv0_population)
            & (ref_nv0_population >= min_ref_nv0_population)
        )

    fit_quality = 0.5 * (1.0 - _norm01(gof1)) + 0.5 * (1.0 - _norm01(gof2))

    w_r1, w_r2, w_same, w_nvm, w_low = score_weights
    score = (
        w_r1 * _norm01(readout1_fidelity)
        + w_r2 * _norm01(readout2_fidelity)
        + w_same * _norm01(ref_same)
        + w_nvm * _norm01(ref_nvm)
        + w_low * (1.0 - _norm01(step_vals))
        + 0.10 * fit_quality
    )

    if np.any(good):
        return int(np.where(good)[0][0]), "lowest step satisfying thresholds", score

    score_safe = np.where(np.isfinite(score), score, -np.inf)
    if np.all(score_safe == -np.inf):
        return None, "no finite score", score

    return int(np.nanargmax(score_safe)), "fallback max score", score


def _compute_optima(arrays, step_vals, axis_info, criteria, score_weights):
    num_nvs, num_steps = arrays["readout1_fidelity_arr"].shape

    median = {
        key: np.nanmedian(val, axis=0)
        for key, val in arrays.items()
        if key.endswith("_arr") and np.asarray(val).ndim == 2
    }

    pass_mask = (
        np.isfinite(arrays["readout1_fidelity_arr"])
        & np.isfinite(arrays["readout2_fidelity_arr"])
        & np.isfinite(arrays["ref_same_state_survival_arr"])
        & np.isfinite(arrays["ref_nvm_survival_arr"])
        & (arrays["readout1_fidelity_arr"] >= criteria["min_readout1_fidelity"])
        & (arrays["readout2_fidelity_arr"] >= criteria["min_readout2_fidelity"])
        & (
            arrays["ref_same_state_survival_arr"]
            >= criteria["min_ref_same_state_survival"]
        )
        & (arrays["ref_nvm_survival_arr"] >= criteria["min_ref_nvm_survival"])
    )

    if criteria.get("min_ref_nvm_population", None) is not None:
        pass_mask = (
            pass_mask
            & np.isfinite(arrays["ref_nvm_population_arr"])
            & (
                arrays["ref_nvm_population_arr"]
                >= criteria["min_ref_nvm_population"]
            )
        )

    if criteria.get("min_ref_nv0_survival", None) is not None:
        pass_mask = (
            pass_mask
            & np.isfinite(arrays["ref_nv0_survival_arr"])
            & (arrays["ref_nv0_survival_arr"] >= criteria["min_ref_nv0_survival"])
        )

    if criteria.get("min_ref_nv0_population", None) is not None:
        pass_mask = (
            pass_mask
            & np.isfinite(arrays["ref_nv0_population_arr"])
            & (
                arrays["ref_nv0_population_arr"]
                >= criteria["min_ref_nv0_population"]
            )
        )

    pass_fraction = np.nanmean(pass_mask.astype(float), axis=0)

    pop_ind, pop_reason, pop_score = _choose_optimal_step(
        step_vals,
        median["readout1_fidelity_arr"],
        median["readout2_fidelity_arr"],
        median["ref_same_state_survival_arr"],
        median["ref_nvm_survival_arr"],
        median["ref_nv0_survival_arr"],
        median["ref_nvm_population_arr"],
        median["ref_nv0_population_arr"],
        median["goodness1_of_fit_arr"],
        median["goodness2_of_fit_arr"],
        criteria,
        score_weights,
    )

    min_pass_fraction = criteria.get("min_pass_fraction", None)
    if min_pass_fraction is not None and np.any(pass_fraction >= min_pass_fraction):
        pop_ind = int(np.where(pass_fraction >= min_pass_fraction)[0][0])
        pop_reason = f"lowest step with pass_fraction >= {min_pass_fraction:.3f}"

    per_nv_score_arr = np.full((num_nvs, num_steps), np.nan)
    per_nv_step_inds = np.full(num_nvs, -1, dtype=int)
    per_nv_step_vals = np.full(num_nvs, np.nan, dtype=float)
    per_nv_values = []

    for nv_ind in range(num_nvs):
        step_ind, reason, score = _choose_optimal_step(
            step_vals,
            arrays["readout1_fidelity_arr"][nv_ind],
            arrays["readout2_fidelity_arr"][nv_ind],
            arrays["ref_same_state_survival_arr"][nv_ind],
            arrays["ref_nvm_survival_arr"][nv_ind],
            arrays["ref_nv0_survival_arr"][nv_ind],
            arrays["ref_nvm_population_arr"][nv_ind],
            arrays["ref_nv0_population_arr"][nv_ind],
            arrays["goodness1_of_fit_arr"][nv_ind],
            arrays["goodness2_of_fit_arr"][nv_ind],
            criteria,
            score_weights,
        )
        per_nv_score_arr[nv_ind] = score

        if step_ind is None:
            per_nv_values.append(
                {
                    "nv_ind": int(nv_ind),
                    "optimal_step_ind": None,
                    "optimal_step_val": np.nan,
                    "reason": reason,
                    "score": np.nan,
                }
            )
            continue

        step_val = float(step_vals[step_ind])
        per_nv_step_inds[nv_ind] = int(step_ind)
        per_nv_step_vals[nv_ind] = step_val
        per_nv_values.append(
            {
                "nv_ind": int(nv_ind),
                "optimal_step_ind": int(step_ind),
                "optimal_step_val": step_val,
                "reason": reason,
                "score": float(score[step_ind]) if np.isfinite(score[step_ind]) else np.nan,
                "readout1_fidelity": float(arrays["readout1_fidelity_arr"][nv_ind, step_ind]),
                "readout2_fidelity": float(arrays["readout2_fidelity_arr"][nv_ind, step_ind]),
                "ref_same_state_survival": float(
                    arrays["ref_same_state_survival_arr"][nv_ind, step_ind]
                ),
                "ref_nvm_survival": float(
                    arrays["ref_nvm_survival_arr"][nv_ind, step_ind]
                ),
                "ref_nv0_survival": float(
                    arrays["ref_nv0_survival_arr"][nv_ind, step_ind]
                ),
                "ref_nvm_population": float(
                    arrays["ref_nvm_population_arr"][nv_ind, step_ind]
                ),
                "ref_nv0_population": float(
                    arrays["ref_nv0_population_arr"][nv_ind, step_ind]
                ),
                "fit1_n_nvs": int(arrays["fit1_n_nvs_arr"][nv_ind, step_ind]),
                "fit2_n_nvs": int(arrays["fit2_n_nvs_arr"][nv_ind, step_ind]),
                "fit1_bic": float(arrays["fit1_bic_arr"][nv_ind, step_ind]),
                "fit2_bic": float(arrays["fit2_bic_arr"][nv_ind, step_ind]),
                "aom_voltage": _aom_voltage_from_step_value(step_val, axis_info),
            }
        )

    return {
        "median": median,
        "population_step_ind": pop_ind,
        "population_reason": pop_reason,
        "population_score": pop_score,
        "pass_mask": pass_mask,
        "pass_fraction": pass_fraction,
        "per_nv_score_arr": per_nv_score_arr,
        "per_nv_step_inds": per_nv_step_inds,
        "per_nv_step_vals": per_nv_step_vals,
        "per_nv_values": per_nv_values,
    }


def process_repeated_readout_survival_gpu(
    raw_data,
    do_plot=True,
    save_data=True,
    gpu_fit_config=None,
    model_mode="auto",
    max_nvs=3,
    force_nvs=None,
    min_readout1_fidelity=0.85,
    min_readout2_fidelity=0.85,
    min_ref_same_state_survival=0.95,
    min_ref_nvm_survival=0.95,
    min_ref_nv0_survival=None,
    min_ref_nvm_population=0.02,
    min_ref_nv0_population=None,
    min_pass_fraction=0.50,
    score_weights=(0.25, 0.20, 0.20, 0.30, 0.05),
):
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy/GPU is not available in this Python environment.")

    if gpu_fit_config is None:
        gpu_fit_config = GpuMultimodeFitConfig(
            max_nvs=int(max_nvs),
            num_p=11,
            num_bg=4,
            num_rate0=12,
            num_delta=12,
            fit_chunk_size=512,
            candidate_chunk_size=512,
        )

    print("\n=== GPU repeated-readout survival analysis ===")
    print("GPU info:", summarize_gpu())

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    num_steps = int(raw_data["num_steps"])
    counts = np.asarray(raw_data["counts"], dtype=float)

    if counts.shape[0] < 4:
        raise ValueError(
            f"Expected at least 4 experiments for repeated readout, got {counts.shape[0]}"
        )

    axis_info = _get_readout_axis(raw_data)
    step_vals = np.asarray(axis_info["step_vals"], dtype=float)

    print(f"num_nvs: {num_nvs}")
    print(f"num_steps: {num_steps}")
    print(f"total GPU fits per readout: {num_nvs * num_steps}")
    print("exp order: 0=ion R1, 1=ion R2, 2=ref R1, 3=ref R2")
    print(f"charge fit model_mode: {model_mode}, max_nvs: {max_nvs}")

    r1_batch, r2_batch = _build_gpu_fit_batches(counts)

    print("\nFitting R1 histograms on GPU...")
    r1_fit_results, r1_debug = fit_charge_histograms_gpu_batch(
        r1_batch,
        prob_dist="COMPOUND_POISSON",
        model_mode=model_mode,
        multimode_config=gpu_fit_config
        if isinstance(gpu_fit_config, GpuMultimodeFitConfig)
        else None,
        bimodal_config=gpu_fit_config
        if isinstance(gpu_fit_config, GpuFitConfig)
        else None,
        max_nvs=max_nvs,
        force_nvs=force_nvs,
        return_debug=True,
    )
    print("R1 GPU fit debug:", r1_debug)

    print("\nFitting R2 histograms on GPU...")
    r2_fit_results, r2_debug = fit_charge_histograms_gpu_batch(
        r2_batch,
        prob_dist="COMPOUND_POISSON",
        model_mode=model_mode,
        multimode_config=gpu_fit_config
        if isinstance(gpu_fit_config, GpuMultimodeFitConfig)
        else None,
        bimodal_config=gpu_fit_config
        if isinstance(gpu_fit_config, GpuFitConfig)
        else None,
        max_nvs=max_nvs,
        force_nvs=force_nvs,
        return_debug=True,
    )
    print("R2 GPU fit debug:", r2_debug)

    r1_fit = _charge_fit_results_to_arrays(
        r1_fit_results,
        num_nvs,
        num_steps,
    )
    r2_fit = _charge_fit_results_to_arrays(
        r2_fit_results,
        num_nvs,
        num_steps,
    )

    x_max = int(np.nanmax(counts))
    print("\nComputing all-NV0 vs any-NV- thresholds on GPU...")
    threshold1, readout1_fid = determine_thresholds_any_minus_gpu(
        r1_fit["popt"],
        r1_fit["n_nvs"],
        prob_dist="COMPOUND_POISSON",
        x_max=x_max,
        config=gpu_fit_config
        if isinstance(gpu_fit_config, GpuMultimodeFitConfig)
        else GpuMultimodeFitConfig(max_nvs=int(max_nvs)),
    )
    threshold2, readout2_fid = determine_thresholds_any_minus_gpu(
        r2_fit["popt"],
        r2_fit["n_nvs"],
        prob_dist="COMPOUND_POISSON",
        x_max=x_max,
        config=gpu_fit_config
        if isinstance(gpu_fit_config, GpuMultimodeFitConfig)
        else GpuMultimodeFitConfig(max_nvs=int(max_nvs)),
    )

    prep1 = _pminus_to_any_minus_population(
        r1_fit["popt"][:, :, 0],
        r1_fit["n_nvs"],
    )
    prep2 = _pminus_to_any_minus_population(
        r2_fit["popt"][:, :, 0],
        r2_fit["n_nvs"],
    )

    print("Computing repeated-readout survival metrics on GPU...")
    survival = _compute_survival_gpu(counts, threshold1)

    arrays = {
        "threshold_arr": threshold1,
        "threshold2_arr": threshold2,
        "readout1_fidelity_arr": readout1_fid,
        "readout2_fidelity_arr": readout2_fid,
        "prep1_fidelity_arr": prep1,
        "prep2_fidelity_arr": prep2,
        "goodness1_of_fit_arr": r1_fit["chi"],
        "goodness2_of_fit_arr": r2_fit["chi"],
        "fit1_success_arr": r1_fit["success"],
        "fit2_success_arr": r2_fit["success"],
        "fit1_params_arr": r1_fit["popt"],
        "fit2_params_arr": r2_fit["popt"],
        "fit1_bic_arr": r1_fit["bic"],
        "fit2_bic_arr": r2_fit["bic"],
        "fit1_nll_arr": r1_fit["nll"],
        "fit2_nll_arr": r2_fit["nll"],
        "fit1_n_nvs_arr": r1_fit["n_nvs"],
        "fit2_n_nvs_arr": r2_fit["n_nvs"],
        "fit1_model_code_arr": r1_fit["model_code"],
        "fit2_model_code_arr": r2_fit["model_code"],
    }
    arrays.update(survival)

    criteria = {
        "min_readout1_fidelity": float(min_readout1_fidelity),
        "min_readout2_fidelity": float(min_readout2_fidelity),
        "min_ref_same_state_survival": float(min_ref_same_state_survival),
        "min_ref_nvm_survival": float(min_ref_nvm_survival),
        "min_ref_nv0_survival": (
            None
            if min_ref_nv0_survival is None
            else float(min_ref_nv0_survival)
        ),
        "min_ref_nvm_population": (
            None
            if min_ref_nvm_population is None
            else float(min_ref_nvm_population)
        ),
        "min_ref_nv0_population": (
            None
            if min_ref_nv0_population is None
            else float(min_ref_nv0_population)
        ),
        "min_pass_fraction": (
            None
            if min_pass_fraction is None
            else float(min_pass_fraction)
        ),
    }

    opt = _compute_optima(arrays, step_vals, axis_info, criteria, score_weights)
    pop_ind = opt["population_step_ind"]

    if pop_ind is None:
        pop_step = np.nan
        pop_aom_voltage = np.nan
    else:
        pop_step = float(step_vals[pop_ind])
        pop_aom_voltage = _aom_voltage_from_step_value(pop_step, axis_info)

    valid_step_vals = opt["per_nv_step_vals"][np.isfinite(opt["per_nv_step_vals"])]
    if valid_step_vals.size > 0:
        total_power = float(np.nanmean(valid_step_vals))
        optimal_weights = valid_step_vals / total_power
    else:
        total_power = np.nan
        optimal_weights = np.asarray([], dtype=float)

    results = {
        "analysis_type": "gpu_repeated_readout_survival",
        "file_stem_source": _base_file_stem(raw_data),
        "used_gpu": True,
        "gpu_fit_config": make_json_safe(gpu_fit_config.__dict__),
        "charge_fit_model_mode": str(model_mode),
        "charge_fit_max_nvs": int(max_nvs),
        "charge_fit_force_nvs": None if force_nvs is None else int(force_nvs),
        "threshold_meaning": "all NV0 if counts <= threshold; any NV- if counts > threshold",
        "r1_fit_debug": make_json_safe(r1_debug),
        "r2_fit_debug": make_json_safe(r2_debug),
        "num_nvs": int(num_nvs),
        "num_steps": int(num_steps),
        "step_vals_raw": axis_info["step_vals_raw"].tolist(),
        "step_vals": step_vals.tolist(),
        "x_label": axis_info["x_label"],
        "power_fit_a": float(axis_info["power_fit_a"]),
        "power_fit_b": float(axis_info["power_fit_b"]),
        "power_fit_c": float(axis_info["power_fit_c"]),
        "yellow_charge_readout_amp": float(axis_info["yellow_charge_readout_amp"])
        if np.isfinite(axis_info["yellow_charge_readout_amp"])
        else None,
        "exp_order": {
            "0": "ionized_readout_1",
            "1": "ionized_readout_2",
            "2": "reference_readout_1",
            "3": "reference_readout_2",
        },
        "criteria": criteria,
        "score_weights": tuple(float(v) for v in score_weights),
        "population_optimal_step_ind": None if pop_ind is None else int(pop_ind),
        "population_optimal_step_val": float(pop_step)
        if np.isfinite(pop_step)
        else None,
        "population_aom_voltage": float(pop_aom_voltage)
        if np.isfinite(pop_aom_voltage)
        else None,
        "population_optimal_reason": opt["population_reason"],
        "population_score": opt["population_score"].tolist(),
        "per_nv_optimal_values": make_json_safe(opt["per_nv_values"]),
        "per_nv_optimal_step_inds": opt["per_nv_step_inds"].tolist(),
        "per_nv_optimal_step_vals": opt["per_nv_step_vals"].tolist(),
        "per_nv_score_arr": opt["per_nv_score_arr"].tolist(),
        "pass_fraction": opt["pass_fraction"].tolist(),
        "pass_mask": opt["pass_mask"].tolist(),
        "valid_step_vals": valid_step_vals.tolist(),
        "total_power": float(total_power) if np.isfinite(total_power) else None,
        "optimal_weights": optimal_weights.tolist(),
        "median": {
            key: val.tolist()
            for key, val in opt["median"].items()
        },
    }

    for key, val in arrays.items():
        results[key] = make_json_safe(val)

    print("\n=== GPU repeated-readout survival optimum ===")
    print("Population optimal step index:", results["population_optimal_step_ind"])
    print(f"Population optimal {axis_info['x_label']}: {pop_step:.4g}")
    print("Population AOM voltage:", results["population_aom_voltage"])
    print("Reason:", opt["population_reason"])

    if pop_ind is not None:
        med = opt["median"]
        print(
            "At population optimum: "
            f"R1 fid={med['readout1_fidelity_arr'][pop_ind]:.3f}, "
            f"R2 fid={med['readout2_fidelity_arr'][pop_ind]:.3f}, "
            f"ref same={med['ref_same_state_survival_arr'][pop_ind]:.3f}, "
            f"ref NV- survival={med['ref_nvm_survival_arr'][pop_ind]:.3f}, "
            f"ref NV0 survival={med['ref_nv0_survival_arr'][pop_ind]:.3f}, "
            f"pass fraction={opt['pass_fraction'][pop_ind]:.3f}"
        )

    if valid_step_vals.size > 0:
        print("Per-NV mean optimal step:", float(np.nanmean(valid_step_vals)))
        print("Per-NV median optimal step:", float(np.nanmedian(valid_step_vals)))
        print("Number valid NVs:", int(valid_step_vals.size))

    if save_data:
        timestamp = dm.get_time_stamp()
        file_name = f"gpu_repeated_readout_survival_processed_{_base_file_stem(raw_data)}"
        file_path = dm.get_file_path(__file__, timestamp, file_name)
        dm.save_raw_data(make_json_safe(results), file_path)
        results["saved_file_path"] = str(file_path)
        print("Saved GPU repeated-readout survival analysis:", file_path)

    if do_plot:
        plot_gpu_repeated_readout_summary(results)
        plot_gpu_selected_model_summary(results)
        plot_gpu_repeated_readout_all_nv_scatters(results)
        plot_gpu_per_nv_optimal_step_distribution(results)
        plot_gpu_optimum_metric_scatter(results)

    return results


# =============================================================================
# Plots
# =============================================================================


def _population_optimum(results):
    opt_ind = results.get("population_optimal_step_ind", None)
    opt_val = results.get("population_optimal_step_val", None)
    if opt_ind is None or opt_val is None:
        return None, np.nan
    return int(opt_ind), float(opt_val)


def plot_gpu_repeated_readout_summary(results):
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    opt_ind, opt_val = _population_optimum(results)
    med = results["median"]

    r1 = np.asarray(med["readout1_fidelity_arr"], dtype=float)
    r2 = np.asarray(med["readout2_fidelity_arr"], dtype=float)
    same = np.asarray(med["ref_same_state_survival_arr"], dtype=float)
    nvm = np.asarray(med["ref_nvm_survival_arr"], dtype=float)
    nv0 = np.asarray(med["ref_nv0_survival_arr"], dtype=float)
    ion = np.asarray(med["mean_ion_r1_arr"], dtype=float)
    ref1 = np.asarray(med["mean_ref_r1_arr"], dtype=float)
    ref2 = np.asarray(med["mean_ref_r2_arr"], dtype=float)
    score = np.asarray(results["population_score"], dtype=float)
    pass_fraction = np.asarray(results.get("pass_fraction", []), dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)

    for ax, arr, label, color in [
        (axes[0], r1, "R1 fidelity", _kpl_color("BLUE", "tab:blue")),
        (axes[0], r2, "R2 fidelity", _kpl_color("ORANGE", "tab:orange")),
    ]:
        _plot_points(ax, step_vals, arr, label=label, color=color)
        _plot_line(ax, step_vals, arr, color=color)
    axes[0].set_ylabel("Fidelity")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()

    for ax, arr, label, color in [
        (axes[1], same, "Ref same-state survival", _kpl_color("PURPLE", "tab:purple")),
        (axes[1], nvm, "Ref NV- survival", _kpl_color("GREEN", "tab:green")),
        (axes[1], nv0, "Ref NV0 survival", _kpl_color("GRAY", "tab:gray")),
    ]:
        _plot_points(ax, step_vals, arr, label=label, color=color)
        _plot_line(ax, step_vals, arr, color=color)
    axes[1].set_ylabel("Survival")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    for ax, arr, label, color in [
        (axes[2], ion, "Median ion R1 counts", _kpl_color("RED", "tab:red")),
        (axes[2], ref1, "Median ref R1 counts", _kpl_color("BLUE", "tab:blue")),
        (axes[2], ref2, "Median ref R2 counts", _kpl_color("GREEN", "tab:green")),
    ]:
        _plot_points(ax, step_vals, arr, label=label, color=color)
        _plot_line(ax, step_vals, arr, color=color)
    axes[2].set_ylabel("Counts")
    axes[2].legend()

    _plot_points(
        axes[3],
        step_vals,
        score,
        label="Population score",
        color=_kpl_color("BLACK", "black"),
    )
    _plot_line(axes[3], step_vals, score, color=_kpl_color("BLACK", "black"))
    if pass_fraction.size == step_vals.size:
        _plot_points(
            axes[3],
            step_vals,
            pass_fraction,
            label="NV pass fraction",
            color=_kpl_color("RED", "tab:red"),
        )
        _plot_line(
            axes[3],
            step_vals,
            pass_fraction,
            color=_kpl_color("RED", "tab:red"),
            linestyle=":",
        )
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel(x_label)
    axes[3].legend()

    for ax in axes:
        _mark_optimal(ax, opt_val)
        _style_axis(ax)

    title = "GPU repeated-readout optimization"
    if opt_ind is not None:
        title += f"\nchosen index {opt_ind}, {x_label}={opt_val:.4g}"
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_gpu_selected_model_summary(results):
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    opt_ind, opt_val = _population_optimum(results)

    n1 = np.asarray(results["fit1_n_nvs_arr"], dtype=float)
    n2 = np.asarray(results["fit2_n_nvs_arr"], dtype=float)

    max_n = int(
        max(
            1,
            np.nanmax(n1) if np.isfinite(n1).any() else 1,
            np.nanmax(n2) if np.isfinite(n2).any() else 1,
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    med_n1 = np.nanmedian(np.where(n1 > 0, n1, np.nan), axis=0)
    med_n2 = np.nanmedian(np.where(n2 > 0, n2, np.nan), axis=0)

    _plot_points(axes[0], step_vals, med_n1, label="R1 median selected N", color=_kpl_color("BLUE", "tab:blue"))
    _plot_line(axes[0], step_vals, med_n1, color=_kpl_color("BLUE", "tab:blue"))
    _plot_points(axes[0], step_vals, med_n2, label="R2 median selected N", color=_kpl_color("ORANGE", "tab:orange"))
    _plot_line(axes[0], step_vals, med_n2, color=_kpl_color("ORANGE", "tab:orange"))
    axes[0].set_ylabel("Selected NV count")
    axes[0].set_ylim(0.8, max_n + 0.3)
    axes[0].legend()

    for n_val in range(1, max_n + 1):
        frac = np.nanmean(n1 == n_val, axis=0)
        _plot_line(
            axes[1],
            step_vals,
            frac,
            label=f"R1 fraction N={n_val}",
            linewidth=2,
        )

    axes[1].set_ylabel("Fraction of NVs")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    for ax in axes:
        _mark_optimal(ax, opt_val)
        _style_axis(ax)

    title = "GPU model-selection diagnostic"
    if opt_ind is not None:
        title += f"\nchosen index {opt_ind}, {x_label}={opt_val:.4g}"
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_gpu_repeated_readout_all_nv_scatters(results, alpha=0.28, size=12):
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    opt_ind, opt_val = _population_optimum(results)

    r1 = np.asarray(results["readout1_fidelity_arr"], dtype=float)
    r2 = np.asarray(results["readout2_fidelity_arr"], dtype=float)
    same = np.asarray(results["ref_same_state_survival_arr"], dtype=float)
    nvm = np.asarray(results["ref_nvm_survival_arr"], dtype=float)
    nvm_to_nv0 = np.asarray(results["ref_nvm_to_nv0_prob_arr"], dtype=float)
    nvm_population = np.asarray(results["ref_nvm_population_arr"], dtype=float)
    score = np.asarray(results["per_nv_score_arr"], dtype=float)
    pass_mask = np.asarray(results.get("pass_mask", np.zeros_like(score)), dtype=float)

    x = np.tile(step_vals, r1.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()

    panels = [
        (r1, "R1 fidelity", "Fidelity", _kpl_color("BLUE", "tab:blue")),
        (r2, "R2 fidelity", "Fidelity", _kpl_color("ORANGE", "tab:orange")),
        (same, "Reference same-state survival", "Survival", _kpl_color("PURPLE", "tab:purple")),
        (nvm, "Reference NV- survival", "Survival", _kpl_color("GREEN", "tab:green")),
        (nvm_to_nv0, "Reference NV- -> NV0 probability", "Ionization probability", _kpl_color("RED", "tab:red")),
        (nvm_population, "Reference NV- population fraction", "Fraction", _kpl_color("GRAY", "tab:gray")),
        (pass_mask, "NVs passing all criteria", "Pass = 1", _kpl_color("GREEN", "tab:green")),
        (score, "Per-NV combined score", "Score", _kpl_color("BLACK", "black")),
    ]

    for ax, (arr, title, ylabel, color) in zip(axes, panels):
        y = arr.reshape(-1)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=size, alpha=alpha, color=color)
        med = np.nanmedian(arr, axis=0)
        _plot_line(
            ax,
            step_vals,
            med,
            color=_kpl_color("BLACK", "black"),
            label="Median",
            linewidth=2.5,
        )
        _mark_optimal(ax, opt_val, label="Optimal")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        _style_axis(ax)
        ax.legend(fontsize=8)

    fig.suptitle("GPU all-NV repeated-readout scatter diagnostics")
    plt.tight_layout()
    return fig


def plot_gpu_per_nv_optimal_step_distribution(results):
    step_vals = np.asarray(results["per_nv_optimal_step_vals"], dtype=float)
    valid = step_vals[np.isfinite(step_vals)]
    x_label = results["x_label"]
    opt_ind, opt_val = _population_optimum(results)

    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_histogram(
        ax,
        valid,
        bins=40,
        density=False,
        label="Per-NV optima",
        color=_kpl_color("BLUE", "tab:blue"),
        alpha=0.75,
    )

    if valid.size > 0:
        mean_val = float(np.nanmean(valid))
        median_val = float(np.nanmedian(valid))
        ax.axvline(
            mean_val,
            color=_kpl_color("BLACK", "black"),
            linestyle=":",
            linewidth=1.8,
            label=f"Mean = {mean_val:.3g}",
        )
        ax.axvline(
            median_val,
            color=_kpl_color("GREEN", "tab:green"),
            linestyle="-.",
            linewidth=1.8,
            label=f"Median = {median_val:.3g}",
        )

    _mark_optimal(ax, opt_val, label=f"Population optimal = {opt_val:.3g}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of NVs")
    ax.set_title("Distribution of per-NV optimal readout settings")
    ax.legend()
    _style_axis(ax)
    plt.tight_layout()
    return fig


def plot_gpu_optimum_metric_scatter(results):
    opt_vals = results["per_nv_optimal_values"]
    r1 = np.asarray([v.get("readout1_fidelity", np.nan) for v in opt_vals], dtype=float)
    r2 = np.asarray([v.get("readout2_fidelity", np.nan) for v in opt_vals], dtype=float)
    same = np.asarray([v.get("ref_same_state_survival", np.nan) for v in opt_vals], dtype=float)
    nvm = np.asarray([v.get("ref_nvm_survival", np.nan) for v in opt_vals], dtype=float)
    step = np.asarray([v.get("optimal_step_val", np.nan) for v in opt_vals], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    sc0 = axes[0].scatter(r1, nvm, c=step, s=18, alpha=0.75, cmap="viridis")
    axes[0].set_xlabel("R1 fidelity")
    axes[0].set_ylabel("Ref NV- survival")
    axes[0].set_title("Readout fidelity vs ionization survival")
    plt.colorbar(sc0, ax=axes[0], label=results["x_label"])

    sc1 = axes[1].scatter(r2, nvm, c=step, s=18, alpha=0.75, cmap="viridis")
    axes[1].set_xlabel("R2 fidelity")
    axes[1].set_ylabel("Ref NV- survival")
    axes[1].set_title("R2 fidelity vs ionization survival")
    plt.colorbar(sc1, ax=axes[1], label=results["x_label"])

    sc2 = axes[2].scatter(same, nvm, c=step, s=18, alpha=0.75, cmap="viridis")
    axes[2].set_xlabel("Ref same-state survival")
    axes[2].set_ylabel("Ref NV- survival")
    axes[2].set_title("Survival consistency")
    plt.colorbar(sc2, ax=axes[2], label=results["x_label"])

    for ax in axes:
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        _style_axis(ax)

    fig.suptitle("All NVs at their own selected GPU optimum")
    # Do not call tight_layout here: kplotlib/matplotlib layout engines can
    # conflict after colorbars are created.
    fig.subplots_adjust(
        left=0.07,
        right=0.94,
        bottom=0.13,
        top=0.84,
        wspace=0.42,
    )
    return fig


def plot_gpu_repeated_readout_nv_histograms(
    raw_data,
    results,
    nv_ind,
    step_ind=None,
    bins=60,
    density=True,
    plot_fit=True,
):
    counts = np.asarray(raw_data["counts"], dtype=float)

    if step_ind is None:
        opt = results["per_nv_optimal_values"][nv_ind]
        step_ind = opt.get("optimal_step_ind", None)

        if step_ind is None:
            step_ind = results.get("population_optimal_step_ind", None)

        if step_ind is None:
            raise ValueError("No valid step index available for histogram plot.")

        step_ind = int(step_ind)

    # Safety in case step_ind was saved as float/string
    step_ind = _as_int(step_ind, default=None)
    if step_ind is None:
        raise ValueError("Invalid step_ind.")

    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]

    ion_r1 = counts[0, nv_ind, :, step_ind, :].flatten()
    ion_r2 = counts[1, nv_ind, :, step_ind, :].flatten()
    ref_r1 = counts[2, nv_ind, :, step_ind, :].flatten()
    ref_r2 = counts[3, nv_ind, :, step_ind, :].flatten()

    threshold = _as_float(results["threshold_arr"][nv_ind][step_ind])
    r1_fid = _as_float(results["readout1_fidelity_arr"][nv_ind][step_ind])
    r2_fid = _as_float(results["readout2_fidelity_arr"][nv_ind][step_ind])
    same = _as_float(results["ref_same_state_survival_arr"][nv_ind][step_ind])
    nvm = _as_float(results["ref_nvm_survival_arr"][nv_ind][step_ind])
    nv0 = _as_float(results["ref_nv0_survival_arr"][nv_ind][step_ind])
    nvm_pop = _as_float(results["ref_nvm_population_arr"][nv_ind][step_ind])
    nv0_pop = _as_float(results["ref_nv0_population_arr"][nv_ind][step_ind])
    score = _as_float(results["per_nv_score_arr"][nv_ind][step_ind])

    try:
        fit_n = _as_int(
            results.get("fit1_n_nvs_arr", [[1]])[nv_ind][step_ind],
            default=1,
        )
    except Exception:
        fit_n = 1

    try:
        fit_bic = _as_float(
            results.get("fit1_bic_arr", [[np.nan]])[nv_ind][step_ind]
        )
    except Exception:
        fit_bic = np.nan

    try:
        fit_params = np.asarray(
            results.get("fit1_params_arr", [])[nv_ind][step_ind],
            dtype=float,
        )
    except Exception:
        fit_params = np.asarray([], dtype=float)

    # IMPORTANT: create figure/axes outside try/except
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    blue = _kpl_color("BLUE", "tab:blue")
    orange = _kpl_color("ORANGE", "tab:orange")
    red = _kpl_color("RED", "tab:red")
    green = _kpl_color("GREEN", "tab:green")
    gray = _kpl_color("GRAY", "tab:gray")
    black = _kpl_color("BLACK", "black")

    # -------------------------------------------------------------------------
    # Panel 0: ionized branch R1/R2
    # -------------------------------------------------------------------------
    _plot_histogram(
        axes[0],
        ion_r1,
        bins=bins,
        density=density,
        label="Ion R1",
        color=red,
    )
    _plot_histogram(
        axes[0],
        ion_r2,
        bins=bins,
        density=density,
        label="Ion R2",
        color=orange,
    )

    if np.isfinite(threshold):
        axes[0].axvline(
            threshold,
            color=gray,
            linestyle="--",
            linewidth=1.8,
            label="Threshold",
        )

    axes[0].set_title("Ionized branch")
    axes[0].set_xlabel("Integrated counts")
    axes[0].set_ylabel("Probability" if density else "Counts")
    axes[0].legend(fontsize=9)
    _style_axis(axes[0])

    # -------------------------------------------------------------------------
    # Panel 1: reference branch R1/R2
    # -------------------------------------------------------------------------
    _plot_histogram(
        axes[1],
        ref_r1,
        bins=bins,
        density=density,
        label="Ref R1",
        color=blue,
    )
    _plot_histogram(
        axes[1],
        ref_r2,
        bins=bins,
        density=density,
        label="Ref R2",
        color=green,
    )

    if np.isfinite(threshold):
        axes[1].axvline(
            threshold,
            color=gray,
            linestyle="--",
            linewidth=1.8,
            label="Threshold",
        )

    axes[1].set_title("Reference branch: no re-prep between R1 and R2")
    axes[1].set_xlabel("Integrated counts")
    axes[1].set_ylabel("Probability" if density else "Counts")
    axes[1].legend(fontsize=9)
    _style_axis(axes[1])

    # -------------------------------------------------------------------------
    # Panel 2: threshold fit populations with model overlay
    # -------------------------------------------------------------------------
    _plot_histogram(
        axes[2],
        ion_r1,
        bins=bins,
        density=density,
        label="Ion R1",
        color=red,
    )
    _plot_histogram(
        axes[2],
        ref_r1,
        bins=bins,
        density=density,
        label="Ref R1",
        color=blue,
    )

    if np.isfinite(threshold):
        axes[2].axvline(
            threshold,
            color=gray,
            linestyle="--",
            linewidth=1.8,
            label="Threshold",
        )

    if plot_fit and fit_params.size == 4 and np.all(np.isfinite(fit_params)):
        x_max = max(
            float(np.nanmax(ion_r1)) if ion_r1.size else 0.0,
            float(np.nanmax(ref_r1)) if ref_r1.size else 0.0,
            threshold if np.isfinite(threshold) else 0.0,
        )

        x_fit = np.arange(0, int(np.ceil(x_max)) + 1, dtype=float)

        try:
            combined, components = _gpu_equal_model_fit_curves(
                x_fit,
                fit_params,
                fit_n,
            )

            component_colors = [
                _kpl_color("GRAY", "tab:gray"),
                _kpl_color("GREEN", "tab:green"),
                _kpl_color("ORANGE", "tab:orange"),
                _kpl_color("PURPLE", "tab:purple"),
                _kpl_color("RED", "tab:red"),
            ]

            for comp in components:
                color = component_colors[comp["k"] % len(component_colors)]
                label = (
                    f"k={comp['k']} component"
                    if fit_n <= 3
                    else f"k={comp['k']}"
                )

                _plot_line(
                    axes[2],
                    x_fit,
                    comp["weighted_pdf"],
                    label=label,
                    color=color,
                    linestyle=":",
                    linewidth=1.4,
                )

            _plot_line(
                axes[2],
                x_fit,
                combined,
                label="GPU fit combined",
                color=black,
                linewidth=2.2,
            )

        except Exception as exc:
            axes[2].text(
                0.03,
                0.95,
                f"Fit overlay failed:\n{exc}",
                transform=axes[2].transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )

    axes[2].set_title("R1 threshold fit populations")
    axes[2].set_xlabel("Integrated counts")
    axes[2].set_ylabel("Probability" if density else "Counts")
    axes[2].legend(fontsize=8)
    _style_axis(axes[2])

    # -------------------------------------------------------------------------
    # Panel 3: text summary
    # -------------------------------------------------------------------------
    axes[3].axis("off")

    step_val = (
        float(step_vals[step_ind])
        if step_ind < step_vals.size and np.isfinite(step_vals[step_ind])
        else np.nan
    )

    text = (
        f"NV {nv_ind}\n"
        f"step index = {step_ind}\n"
        f"{x_label} = {step_val:.4g}\n\n"
        f"threshold = {threshold:.2f}\n"
        f"R1 selected model = {fit_n} NV equal\n"
        f"R1 BIC = {fit_bic:.1f}\n"
        f"R1 fidelity = {r1_fid:.3f}\n"
        f"R2 fidelity = {r2_fid:.3f}\n"
        f"ref same-state survival = {same:.3f}\n"
        f"ref NV- survival = {nvm:.3f}\n"
        f"ref NV0 survival = {nv0:.3f}\n"
        f"ref NV- population = {nvm_pop:.3f}\n"
        f"ref NV0 population = {nv0_pop:.3f}\n"
        f"score = {score:.3f}\n\n"
        f"mean ion R1 = {np.nanmean(ion_r1):.1f}\n"
        f"mean ion R2 = {np.nanmean(ion_r2):.1f}\n"
        f"mean ref R1 = {np.nanmean(ref_r1):.1f}\n"
        f"mean ref R2 = {np.nanmean(ref_r2):.1f}"
    )

    axes[3].text(
        0.04,
        0.96,
        text,
        va="top",
        ha="left",
        fontsize=11,
    )

    fig.suptitle("GPU repeated-readout histogram overlay")

    try:
        fig.tight_layout()
    except Exception:
        pass

    return fig

def pick_representative_nvs(results):
    opt_vals = results["per_nv_optimal_values"]
    score = np.asarray([v.get("score", np.nan) for v in opt_vals], dtype=float)
    nvm = np.asarray([v.get("ref_nvm_survival", np.nan) for v in opt_vals], dtype=float)
    r1 = np.asarray([v.get("readout1_fidelity", np.nan) for v in opt_vals], dtype=float)

    valid = np.isfinite(score)
    valid_inds = np.where(valid)[0]
    if valid_inds.size == 0:
        raise ValueError("No valid NVs for representative selection.")

    good_nv = int(valid_inds[np.nanargmax(score[valid_inds])])
    low_survival_nv = int(valid_inds[np.nanargmin(nvm[valid_inds])])
    low_fidelity_nv = int(valid_inds[np.nanargmin(r1[valid_inds])])
    median_score = np.nanmedian(score[valid_inds])
    median_nv = int(valid_inds[np.nanargmin(np.abs(score[valid_inds] - median_score))])

    return {
        "good_nv": good_nv,
        "low_nv_minus_survival_nv": low_survival_nv,
        "low_readout_fidelity_nv": low_fidelity_nv,
        "median_nv": median_nv,
    }


def print_nv_optimum_summary(results, nv_ind):
    opt = results["per_nv_optimal_values"][nv_ind]
    print("\n=== GPU NV repeated-readout optimum ===")
    print("NV:", int(nv_ind))
    print("step index:", opt.get("optimal_step_ind", None))
    print(f"{results['x_label']}:", opt.get("optimal_step_val", np.nan))
    print("reason:", opt.get("reason", None))
    print("score:", opt.get("score", np.nan))
    print("R1 fidelity:", opt.get("readout1_fidelity", np.nan))
    print("R2 fidelity:", opt.get("readout2_fidelity", np.nan))
    print("R1 selected NV count:", opt.get("fit1_n_nvs", np.nan))
    print("R2 selected NV count:", opt.get("fit2_n_nvs", np.nan))
    print("R1 BIC:", opt.get("fit1_bic", np.nan))
    print("R2 BIC:", opt.get("fit2_bic", np.nan))
    print("ref same-state survival:", opt.get("ref_same_state_survival", np.nan))
    print("ref NV- survival:", opt.get("ref_nvm_survival", np.nan))
    print("ref NV0 survival:", opt.get("ref_nv0_survival", np.nan))
    print("ref NV- population fraction:", opt.get("ref_nvm_population", np.nan))
    print("ref NV0 population fraction:", opt.get("ref_nv0_population", np.nan))
    print("aom voltage:", opt.get("aom_voltage", np.nan))


def plot_gpu_repeated_readout_nv_metrics(results, nv_ind):
    """
    Plot all repeated-readout metrics for one NV from the processed results file.

    This does not need raw_data. Use plot_gpu_repeated_readout_nv_histograms()
    when you also want the raw count histograms.
    """
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]
    opt = results["per_nv_optimal_values"][nv_ind]

    opt_step_ind = opt.get("optimal_step_ind", None)
    opt_step_val = opt.get("optimal_step_val", np.nan)
    pop_ind, pop_val = _population_optimum(results)

    r1 = np.asarray(results["readout1_fidelity_arr"][nv_ind], dtype=float)
    r2 = np.asarray(results["readout2_fidelity_arr"][nv_ind], dtype=float)
    same = np.asarray(results["ref_same_state_survival_arr"][nv_ind], dtype=float)
    nvm = np.asarray(results["ref_nvm_survival_arr"][nv_ind], dtype=float)
    nv0 = np.asarray(results["ref_nv0_survival_arr"][nv_ind], dtype=float)
    nvm_pop = np.asarray(results["ref_nvm_population_arr"][nv_ind], dtype=float)
    score = np.asarray(results["per_nv_score_arr"][nv_ind], dtype=float)
    fit_n = np.asarray(results.get("fit1_n_nvs_arr", []), dtype=float)

    if fit_n.size:
        fit_n = fit_n[nv_ind]

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    _plot_points(axes[0], step_vals, r1, label="R1 fidelity", color=_kpl_color("BLUE", "tab:blue"))
    _plot_line(axes[0], step_vals, r1, color=_kpl_color("BLUE", "tab:blue"))
    _plot_points(axes[0], step_vals, r2, label="R2 fidelity", color=_kpl_color("ORANGE", "tab:orange"))
    _plot_line(axes[0], step_vals, r2, color=_kpl_color("ORANGE", "tab:orange"))
    axes[0].set_ylabel("Fidelity")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()

    _plot_points(axes[1], step_vals, same, label="Ref same-state", color=_kpl_color("PURPLE", "tab:purple"))
    _plot_line(axes[1], step_vals, same, color=_kpl_color("PURPLE", "tab:purple"))
    _plot_points(axes[1], step_vals, nvm, label="Ref NV- survival", color=_kpl_color("GREEN", "tab:green"))
    _plot_line(axes[1], step_vals, nvm, color=_kpl_color("GREEN", "tab:green"))
    _plot_points(axes[1], step_vals, nv0, label="Ref NV0 survival", color=_kpl_color("GRAY", "tab:gray"))
    _plot_line(axes[1], step_vals, nv0, color=_kpl_color("GRAY", "tab:gray"))
    axes[1].set_ylabel("Survival")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    _plot_points(axes[2], step_vals, nvm_pop, label="Ref NV- population", color=_kpl_color("BLUE", "tab:blue"))
    _plot_line(axes[2], step_vals, nvm_pop, color=_kpl_color("BLUE", "tab:blue"))
    if fit_n.size:
        ax2 = axes[2].twinx()
        _plot_points(ax2, step_vals, fit_n, label="Selected model N", color=_kpl_color("RED", "tab:red"))
        _plot_line(ax2, step_vals, fit_n, color=_kpl_color("RED", "tab:red"), linestyle=":")
        ax2.set_ylabel("Selected N")
        ax2.set_ylim(0.8, max(3.2, np.nanmax(fit_n) + 0.3))
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(0, 1.02)
    axes[2].legend(loc="upper left")

    _plot_points(axes[3], step_vals, score, label="Per-NV score", color=_kpl_color("BLACK", "black"))
    _plot_line(axes[3], step_vals, score, color=_kpl_color("BLACK", "black"))
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel(x_label)
    axes[3].legend()

    for ax in axes:
        if np.isfinite(pop_val):
            ax.axvline(
                pop_val,
                color=_kpl_color("GRAY", "tab:gray"),
                linestyle=":",
                linewidth=1.7,
                label="Population opt",
            )
        _mark_optimal(ax, opt_step_val, label="NV opt")
        _style_axis(ax)

    title = f"NV {nv_ind} repeated-readout metrics"
    if opt_step_ind is not None:
        title += f"\nNV opt index {opt_step_ind}, {x_label}={float(opt_step_val):.4g}"
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def pick_more_nvs_to_inspect(
    results,
    num_best=5,
    num_low_survival=5,
    num_low_fidelity=5,
    num_median=5,
    num_random=0,
    seed=0,
):
    """
    Pick several useful NV groups from a processed result file.

    Returns a dict of label -> list of NV indices.
    """
    opt_vals = results["per_nv_optimal_values"]
    score = np.asarray([v.get("score", np.nan) for v in opt_vals], dtype=float)
    nvm = np.asarray([v.get("ref_nvm_survival", np.nan) for v in opt_vals], dtype=float)
    r1 = np.asarray([v.get("readout1_fidelity", np.nan) for v in opt_vals], dtype=float)
    nvm_pop = np.asarray([v.get("ref_nvm_population", np.nan) for v in opt_vals], dtype=float)

    valid = np.isfinite(score)
    valid_inds = np.where(valid)[0]
    if valid_inds.size == 0:
        return {}

    groups = {}
    groups["best_score"] = valid_inds[np.argsort(score[valid_inds])[-num_best:][::-1]].tolist()

    survival_valid = valid & np.isfinite(nvm) & np.isfinite(nvm_pop) & (nvm_pop >= 0.02)
    survival_inds = np.where(survival_valid)[0]
    if survival_inds.size:
        groups["low_nv_minus_survival"] = survival_inds[
            np.argsort(nvm[survival_inds])[:num_low_survival]
        ].tolist()

    fidelity_valid = valid & np.isfinite(r1)
    fidelity_inds = np.where(fidelity_valid)[0]
    if fidelity_inds.size:
        groups["low_r1_fidelity"] = fidelity_inds[
            np.argsort(r1[fidelity_inds])[:num_low_fidelity]
        ].tolist()

    median_score = np.nanmedian(score[valid_inds])
    groups["near_median_score"] = valid_inds[
        np.argsort(np.abs(score[valid_inds] - median_score))[:num_median]
    ].tolist()

    if num_random > 0:
        rng = np.random.default_rng(seed)
        groups["random"] = rng.choice(
            valid_inds,
            size=min(num_random, valid_inds.size),
            replace=False,
        ).astype(int).tolist()

    return groups


def plot_many_gpu_nvs_after_processing(
    results,
    raw_data=None,
    nv_inds=None,
    groups=None,
    plot_metrics=True,
    plot_histograms=False,
    bins=60,
    density=True,
):
    """
    Inspect many NVs after processing.

    Pass only results for metric curves. Pass raw_data too if you want histogram
    overlays at each NV's selected optimum.
    """
    if groups is None:
        if nv_inds is None:
            groups = pick_more_nvs_to_inspect(results)
        else:
            groups = {"selected": [int(v) for v in nv_inds]}

    figs = []
    seen = set()

    for label, inds in groups.items():
        print(f"\n=== {label} ===")
        for nv_ind in inds:
            nv_ind = int(nv_ind)
            if nv_ind in seen:
                continue
            seen.add(nv_ind)

            print_nv_optimum_summary(results, nv_ind)

            if plot_metrics:
                figs.append(plot_gpu_repeated_readout_nv_metrics(results, nv_ind))

            if plot_histograms:
                if raw_data is None:
                    print("Skipping histograms: raw_data was not provided.")
                else:
                    figs.append(
                        plot_gpu_repeated_readout_nv_histograms(
                            raw_data,
                            results,
                            nv_ind=nv_ind,
                            step_ind=None,
                            bins=bins,
                            density=density,
                        )
                    )

    return figs


def plot_representative_nv_histograms(raw_data, results, bins=60, density=True):
    reps = pick_representative_nvs(results)
    figs = []

    for label, nv_ind in reps.items():
        print("\n", label)
        print_nv_optimum_summary(results, nv_ind)
        figs.append(
            plot_gpu_repeated_readout_nv_histograms(
                raw_data,
                results,
                nv_ind=nv_ind,
                step_ind=None,
                bins=bins,
                density=density,
            )
        )

    return figs



# =============================================================================
# Extra after-processing inspection helpers
# =============================================================================


def load_gpu_repeated_readout_results(processed_file, raw_file=None):
    """
    Load a saved GPU repeated-readout processed file, with optional raw data.

    Parameters
    ----------
    processed_file : str
        File stem for the saved processed result.
    raw_file : str | None
        Original raw data file stem. Required only for charge histogram overlays.

    Returns
    -------
    results : dict
    raw_data : dict | None
    """
    results = dm.get_raw_data(file_stem=processed_file, load_npz=True)

    raw_data = None
    if raw_file is not None:
        raw_data = dm.get_raw_data(file_stem=raw_file, load_npz=True)
        raw_data["file_stem"] = raw_file

    return results, raw_data


def make_gpu_per_nv_summary_table(results):
    """
    Make a compact per-NV summary table from processed results.

    Returns a list of dictionaries, one per NV. This avoids requiring pandas.
    """
    opt_vals = results.get("per_nv_optimal_values", [])
    table = []

    for nv_ind, opt in enumerate(opt_vals):
        row = {
            "nv_ind": int(nv_ind),
            "step_ind": opt.get("optimal_step_ind", None),
            "step_val": opt.get("optimal_step_val", np.nan),
            "score": opt.get("score", np.nan),
            "reason": opt.get("reason", None),
            "r1_fid": opt.get("readout1_fidelity", np.nan),
            "r2_fid": opt.get("readout2_fidelity", np.nan),
            "same": opt.get("ref_same_state_survival", np.nan),
            "nvm_surv": opt.get("ref_nvm_survival", np.nan),
            "nv0_surv": opt.get("ref_nv0_survival", np.nan),
            "nvm_pop": opt.get("ref_nvm_population", np.nan),
            "nv0_pop": opt.get("ref_nv0_population", np.nan),
            "fit1_n": opt.get("fit1_n_nvs", np.nan),
            "fit2_n": opt.get("fit2_n_nvs", np.nan),
            "fit1_bic": opt.get("fit1_bic", np.nan),
            "fit2_bic": opt.get("fit2_bic", np.nan),
            "aom_voltage": opt.get("aom_voltage", np.nan),
        }
        table.append(row)

    return table


def _finite_or_default(val, default=np.nan):
    try:
        val = float(val)
    except Exception:
        return default
    return val if np.isfinite(val) else default


def print_gpu_per_nv_summary_table(results, nv_inds=None, sort_by="score", reverse=True, max_rows=30):
    """
    Print a readable per-NV table after processing.

    Examples
    --------
    print_gpu_per_nv_summary_table(results, sort_by="nvm_surv", reverse=False)
    print_gpu_per_nv_summary_table(results, nv_inds=[121, 639, 1048])
    """
    table = make_gpu_per_nv_summary_table(results)

    if nv_inds is not None:
        keep = set(int(v) for v in nv_inds)
        table = [row for row in table if int(row["nv_ind"]) in keep]

    if sort_by is not None and len(table) > 0 and sort_by in table[0]:
        table = sorted(
            table,
            key=lambda r: _finite_or_default(r.get(sort_by, np.nan), -np.inf if reverse else np.inf),
            reverse=bool(reverse),
        )

    if max_rows is not None:
        table = table[: int(max_rows)]

    print("\n=== GPU per-NV processed summary ===")
    print(
        " NV | step | step_val | score | R1 | R2 | same | NV- surv | NV- pop | N1/N2 | reason"
    )
    print("-" * 108)

    for row in table:
        step_ind = row["step_ind"]
        step_str = "None" if step_ind is None else str(int(step_ind))
        print(
            f"{row['nv_ind']:4d} | "
            f"{step_str:>4s} | "
            f"{_finite_or_default(row['step_val']):8.3g} | "
            f"{_finite_or_default(row['score']):5.3f} | "
            f"{_finite_or_default(row['r1_fid']):5.3f} | "
            f"{_finite_or_default(row['r2_fid']):5.3f} | "
            f"{_finite_or_default(row['same']):5.3f} | "
            f"{_finite_or_default(row['nvm_surv']):8.3f} | "
            f"{_finite_or_default(row['nvm_pop']):7.3f} | "
            f"{row['fit1_n']}/{row['fit2_n']} | "
            f"{row['reason']}"
        )

    return table


def find_gpu_nvs_to_debug(
    results,
    min_score=None,
    max_score=None,
    min_nvm_survival=None,
    max_nvm_survival=None,
    min_r1_fidelity=None,
    max_r1_fidelity=None,
    min_r2_fidelity=None,
    max_r2_fidelity=None,
    min_nvm_population=None,
    max_nvm_population=None,
    fit1_n=None,
    fit2_n=None,
    reason_contains=None,
):
    """
    Return NV indices matching physical/QC conditions after processing.

    Useful examples:
        find_gpu_nvs_to_debug(results, max_nvm_survival=0.8, min_nvm_population=0.02)
        find_gpu_nvs_to_debug(results, max_r1_fidelity=0.85)
        find_gpu_nvs_to_debug(results, fit1_n=2)
        find_gpu_nvs_to_debug(results, reason_contains="fallback")
    """
    table = make_gpu_per_nv_summary_table(results)
    out = []

    for row in table:
        keep = True

        def check_minmax(key, vmin, vmax):
            v = _finite_or_default(row.get(key, np.nan))
            if vmin is not None and not (np.isfinite(v) and v >= float(vmin)):
                return False
            if vmax is not None and not (np.isfinite(v) and v <= float(vmax)):
                return False
            return True

        keep &= check_minmax("score", min_score, max_score)
        keep &= check_minmax("nvm_surv", min_nvm_survival, max_nvm_survival)
        keep &= check_minmax("r1_fid", min_r1_fidelity, max_r1_fidelity)
        keep &= check_minmax("r2_fid", min_r2_fidelity, max_r2_fidelity)
        keep &= check_minmax("nvm_pop", min_nvm_population, max_nvm_population)

        if fit1_n is not None:
            keep &= int(row.get("fit1_n", -1)) == int(fit1_n)
        if fit2_n is not None:
            keep &= int(row.get("fit2_n", -1)) == int(fit2_n)

        if reason_contains is not None:
            reason = str(row.get("reason", ""))
            keep &= str(reason_contains).lower() in reason.lower()

        if keep:
            out.append(int(row["nv_ind"]))

    return out


def print_gpu_population_quality_summary(results):
    """
    Print population-level quality at the chosen global optimum and list useful NV groups.
    """
    opt_ind, opt_val = _population_optimum(results)
    print("\n=== GPU population quality summary ===")
    print("population optimal step index:", opt_ind)
    print(f"population optimal {results.get('x_label', 'step')}:", opt_val)
    print("reason:", results.get("population_optimal_reason", None))
    print("aom voltage:", results.get("population_aom_voltage", None))

    if opt_ind is not None:
        med = results.get("median", {})
        for key in [
            "readout1_fidelity_arr",
            "readout2_fidelity_arr",
            "ref_same_state_survival_arr",
            "ref_nvm_survival_arr",
            "ref_nv0_survival_arr",
            "ref_nvm_population_arr",
            "ref_nv0_population_arr",
        ]:
            if key in med:
                arr = np.asarray(med[key], dtype=float)
                print(f"{key}[opt] =", float(arr[opt_ind]))

        pass_fraction = np.asarray(results.get("pass_fraction", []), dtype=float)
        if pass_fraction.size > opt_ind:
            print("pass_fraction[opt] =", float(pass_fraction[opt_ind]))

    low_surv = find_gpu_nvs_to_debug(
        results,
        max_nvm_survival=0.80,
        min_nvm_population=0.02,
    )
    low_r1 = find_gpu_nvs_to_debug(results, max_r1_fidelity=0.85)
    fallback = find_gpu_nvs_to_debug(results, reason_contains="fallback")

    print("NVs with NV- survival <= 0.80 and NV- pop >= 0.02:", low_surv[:30])
    print("NVs with R1 fidelity <= 0.85:", low_r1[:30])
    print("NVs using fallback optimum:", fallback[:30])

    return {
        "low_survival_nvs": low_surv,
        "low_r1_fidelity_nvs": low_r1,
        "fallback_nvs": fallback,
    }


def plot_gpu_step_quality_distribution(results, step_ind=None, bins=50):
    """
    Show distribution of per-NV metrics at one step, defaulting to population optimum.
    """
    if step_ind is None:
        step_ind, _ = _population_optimum(results)
    if step_ind is None:
        raise ValueError("No valid step_ind supplied and no population optimum found.")

    step_ind = int(step_ind)
    step_vals = np.asarray(results["step_vals"], dtype=float)
    x_label = results["x_label"]

    panels = [
        ("readout1_fidelity_arr", "R1 fidelity"),
        ("readout2_fidelity_arr", "R2 fidelity"),
        ("ref_same_state_survival_arr", "Same-state survival"),
        ("ref_nvm_survival_arr", "NV- survival"),
        ("ref_nvm_population_arr", "NV- population"),
        ("per_nv_score_arr", "Score"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, panels):
        arr = np.asarray(results[key], dtype=float)[:, step_ind]
        arr = arr[np.isfinite(arr)]
        _plot_histogram(
            ax,
            arr,
            bins=bins,
            density=False,
            label=title,
            color=_kpl_color("BLUE", "tab:blue"),
            alpha=0.75,
        )
        if arr.size:
            ax.axvline(np.nanmedian(arr), color=_kpl_color("RED", "red"), linestyle="--", label="median")
        ax.set_title(title)
        ax.legend()
        _style_axis(ax)

    fig.suptitle(f"Metric distributions at step {step_ind}, {x_label}={step_vals[step_ind]:.4g}")
    plt.tight_layout()
    return fig


def plot_gpu_model_count_heatmap(results, readout=1):
    """
    Heatmap of selected model count N for all NVs vs step.

    readout=1 uses fit1_n_nvs_arr, readout=2 uses fit2_n_nvs_arr.
    """
    key = "fit1_n_nvs_arr" if int(readout) == 1 else "fit2_n_nvs_arr"
    n_arr = np.asarray(results[key], dtype=float)
    step_vals = np.asarray(results["step_vals"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        n_arr,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Selected N")
    ax.set_xlabel(f"Step index ({results['x_label']})")
    ax.set_ylabel("NV index")
    ax.set_title(f"Selected model size heatmap, readout {readout}")

    # Show a few x tick labels as physical values
    if step_vals.size > 1:
        tick_inds = np.linspace(0, step_vals.size - 1, min(8, step_vals.size)).astype(int)
        ax.set_xticks(tick_inds)
        ax.set_xticklabels([f"{step_vals[i]:.2g}" for i in tick_inds], rotation=45, ha="right")

    return fig


def plot_gpu_nv_dashboard(
    results,
    nv_ind,
    raw_data=None,
    step_ind=None,
    bins=60,
    density=True,
):
    """
    One-call dashboard for a selected NV after processing.

    Always plots metric curves from processed data.
    If raw_data is provided, also plots charge histograms with fit overlay.
    """
    print_nv_optimum_summary(results, nv_ind)
    figs = [plot_gpu_repeated_readout_nv_metrics(results, nv_ind)]

    if raw_data is not None:
        figs.append(
            plot_gpu_repeated_readout_nv_histograms(
                raw_data,
                results,
                nv_ind=nv_ind,
                step_ind=step_ind,
                bins=bins,
                density=density,
                plot_fit=True,
            )
        )

    return figs


def plot_gpu_nv_dashboards(
    results,
    raw_data=None,
    nv_inds=None,
    groups=None,
    max_nvs=20,
    bins=60,
    density=True,
):
    """
    Plot dashboards for many NVs after processing.

    You can pass explicit nv_inds or the output of pick_more_nvs_to_inspect().
    """
    if nv_inds is None:
        if groups is None:
            groups = pick_more_nvs_to_inspect(
                results,
                num_best=5,
                num_low_survival=5,
                num_low_fidelity=5,
                num_median=5,
                num_random=0,
            )

        nv_inds = []
        for _label, vals in groups.items():
            nv_inds.extend([int(v) for v in vals])

    # Deduplicate while preserving order
    seen = set()
    nv_inds_unique = []
    for v in nv_inds:
        if int(v) not in seen:
            seen.add(int(v))
            nv_inds_unique.append(int(v))

    if max_nvs is not None:
        nv_inds_unique = nv_inds_unique[: int(max_nvs)]

    figs = []
    for nv_ind in nv_inds_unique:
        figs.extend(
            plot_gpu_nv_dashboard(
                results,
                nv_ind=nv_ind,
                raw_data=raw_data,
                bins=bins,
                density=density,
            )
        )

    return figs


def save_gpu_per_nv_summary_csv(results, csv_path):
    """
    Save the compact per-NV summary table to CSV using only the Python standard library.
    """
    import csv

    table = make_gpu_per_nv_summary_table(results)
    if len(table) == 0:
        raise ValueError("No per-NV rows available.")

    fieldnames = list(table[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow(row)

    print("Saved per-NV summary CSV:", csv_path)
    return csv_path



# =============================================================================
# Example usage
# =============================================================================


if __name__ == "__main__":
    kpl.init_kplotlib()

    run_new_analysis = False

    raw_file = "2026_06_26-21_58_16-qnami-nv0_2026_02_20"
    processed_file = (
        "2026_07_01-14_11_53-"
        "gpu_repeated_readout_survival_processed_"
        "2026_06_26-21_58_16-qnami-nv0_2026_02_20"
    )

    if run_new_analysis:
        raw_data = dm.get_raw_data(file_stem=raw_file, load_npz=True)
        raw_data["file_stem"] = raw_file

        cfg = GpuMultimodeFitConfig(
            max_nvs=3,
            num_p=11,
            num_bg=4,
            num_rate0=12,
            num_delta=12,
            fit_chunk_size=512,
            candidate_chunk_size=512,
        )

        results = process_repeated_readout_survival_gpu(
            raw_data,
            do_plot=True,
            save_data=True,
            gpu_fit_config=cfg,
            model_mode="auto",
            max_nvs=3,
            force_nvs=None,
            min_readout1_fidelity=0.85,
            min_readout2_fidelity=0.85,
            min_ref_same_state_survival=0.95,
            min_ref_nvm_survival=0.95,
            min_ref_nv0_survival=None,
            min_ref_nvm_population=0.02,
            min_ref_nv0_population=None,
            min_pass_fraction=0.50,
            score_weights=(0.25, 0.20, 0.20, 0.30, 0.05),
        )

        plot_representative_nv_histograms(
            raw_data,
            results,
            bins=60,
            density=True,
        )

        kpl.show(block=True)
        sys.exit()

    # After-processing viewer. This does not rerun the GPU fits.
    results, raw_data = load_gpu_repeated_readout_results(
        processed_file=processed_file,
        raw_file=raw_file,  # set None if you only want metric plots, no histograms
    )

    plot_gpu_repeated_readout_summary(results)
    plot_gpu_selected_model_summary(results)
    plot_gpu_repeated_readout_all_nv_scatters(results)
    plot_gpu_per_nv_optimal_step_distribution(results)
    plot_gpu_optimum_metric_scatter(results)
    plot_gpu_step_quality_distribution(results)
    plot_gpu_model_count_heatmap(results, readout=1)

    debug_groups = print_gpu_population_quality_summary(results)

    # Automatically inspect useful groups
    groups = pick_more_nvs_to_inspect(
        results,
        num_best=5,
        num_low_survival=5,
        num_low_fidelity=5,
        num_median=5,
        num_random=5,
        seed=0,
    )

    # Add targeted debug NVs from quality filters
    targeted = []
    targeted.extend(debug_groups["low_survival_nvs"][:5])
    targeted.extend(debug_groups["low_r1_fidelity_nvs"][:5])
    targeted.extend(debug_groups["fallback_nvs"][:5])
    targeted.extend([0, 10, 50, 121, 639, 1048, 1100])

    print_gpu_per_nv_summary_table(
        results,
        nv_inds=targeted,
        sort_by=None,
        max_rows=50,
    )

    plot_gpu_nv_dashboards(
        results,
        raw_data=raw_data,
        groups=groups,
        max_nvs=20,
        bins=60,
        density=True,
    )

    # Hand-picked NVs
    plot_gpu_nv_dashboards(
        results,
        raw_data=raw_data,
        nv_inds=[0, 10, 50, 121, 639, 1048, 1100],
        max_nvs=None,
        bins=60,
        density=True,
    )

    kpl.show(block=True)
