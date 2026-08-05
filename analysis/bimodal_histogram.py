# -*- coding: utf-8 -*-
"""
Analysis functions for bimodal histograms, the kind you get with single-shot readout.
Includes fitting functions and threshold determination

Created on November 11th, 2024

@author: mccambria
"""

import inspect
import sys
import time
from enum import Enum, auto
from functools import cache
from inspect import signature
import warnings
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.special import factorial, gammainc, gammaincc, gammaln, xlogy
from scipy.stats import norm, poisson, skewnorm

from utils import kplotlib as kpl
from utils.tool_belt import curve_fit

inv_root_2_pi = 1 / np.sqrt(2 * np.pi)

# region Probability distributions


class ProbDist(Enum):
    POISSON = auto()
    BROADENED_POISSON = auto()
    COMPOUND_POISSON = auto()  # See wiki 11/14
    BROADENED_COMPOUND_POISSON = auto()
    GAUSSIAN = auto()
    SKEW_GAUSSIAN = auto()
    COMPOUND_POISSON_WITH_IONIZATION = auto()  # See Cambria PRX 2025


def get_single_mode_num_params(prob_dist: ProbDist):
    single_mode_pdf = get_single_mode_pdf(prob_dist)
    sig = signature(single_mode_pdf)
    # Loop through params, count only non-optional
    num_params = 0
    for param in sig.parameters.values():
        if param.default is param.empty:
            num_params += 1

    # Exclude first param, x, the point to evaluate at
    return num_params - 1


def get_single_mode_pdf(prob_dist: ProbDist):
    fn_name = f"{prob_dist.name.lower()}_pdf"
    return eval(fn_name)


def get_single_mode_cdf(prob_dist: ProbDist):
    fn_name = f"{prob_dist.name.lower()}_cdf"
    return eval(fn_name)


def get_bimodal_pdf(prob_dist: ProbDist):
    if prob_dist is ProbDist.COMPOUND_POISSON_WITH_IONIZATION:
        dark_mode_fn = get_single_mode_pdf(ProbDist.COMPOUND_POISSON)
        bright_mode_fn = get_single_mode_pdf(ProbDist.COMPOUND_POISSON_WITH_IONIZATION)

        def bimodal_fn(x, dark_mode_weight, *params):
            bright_mode_weight = 1 - dark_mode_weight
            first_mode_val = dark_mode_fn(x, params[0])
            second_mode_val = bright_mode_fn(x, *params)
            return (
                dark_mode_weight * first_mode_val + bright_mode_weight * second_mode_val
            )

    else:
        single_mode_fn = get_single_mode_pdf(prob_dist)
        bimodal_fn = _get_bimodal_fn(single_mode_fn)

    return bimodal_fn


def get_bimodal_cdf(prob_dist: ProbDist):
    single_mode_fn = get_single_mode_cdf(prob_dist)
    return _get_bimodal_fn(single_mode_fn)


def _get_bimodal_fn(single_mode_fn):
    def bimodal_fn(x, dark_mode_weight, *params):
        bright_mode_weight = 1 - dark_mode_weight
        half_num_params = len(params) // 2
        first_mode_val = single_mode_fn(x, *params[:half_num_params])
        second_mode_val = single_mode_fn(x, *params[half_num_params:])
        return dark_mode_weight * first_mode_val + bright_mode_weight * second_mode_val

    return bimodal_fn


# @cache
def poisson_pdf(x, rate):
    # return poisson(mu=rate).pmf(x)
    # return (rate**x) * np.exp(-rate) / factorial(x)
    # Computing the pdf directly tends to overflow. Compute exp(ln(pdf)) instead
    return np.exp(xlogy(x, rate) - rate - gammaln(x + 1))


def poisson_cdf(x, rate):
    return _calc_cdf(ProbDist.POISSON, x, rate)


# def _calc_cdf(prob_dist, x, *params):
#     """Cumulative distribution function for poisson pdf. Integrates
#     up to and including x"""
#     pdf = get_single_mode_pdf(prob_dist)
#     x_floor = int(np.floor(x))
#     val = 0
#     for ind in range(x_floor):
#         val += pdf(ind, *params)
#     return val

def _calc_cdf(prob_dist, x, *params):
    """
    CDF for discrete count distributions.
    Returns P(X <= floor(x)).
    """
    pdf = get_single_mode_pdf(prob_dist)
    x_floor = int(np.floor(x))

    if x_floor < 0:
        return 0.0

    val = 0.0
    for ind in range(x_floor + 1):
        val += pdf(ind, *params)

    return float(np.clip(val, 0.0, 1.0))


def _safe_upper_lim(rate, nsig=5, min_lim=10, max_lim=50_000):
    """Pick an inclusive integer upper bound for k-sums based on rate."""
    r = np.asarray(rate, dtype=float)
    if not np.isfinite(r).any():
        raise ValueError("rate has no finite values")
    rmax = np.nanmax(r)
    if rmax < 0:
        raise ValueError(f"rate must be >= 0; got max={rmax}")
    upper_cont = rmax + nsig * np.sqrt(max(rmax, 0.0))
    # inclusive integer bound with a reasonable floor/ceiling
    return int(min(max(int(np.ceil(upper_cont)), min_lim), max_lim))

def _make_integer_histogram(counts_list):
    """
    Make a probability histogram for integer count data.

    Returns:
        x_vals      : integer count centers
        hist        : probability per count value
        hist_errs   : uncertainty of each probability bin
        bin_counts  : raw shot counts in each bin
    """
    counts_list = np.asarray(counts_list, dtype=float).ravel()
    counts_list = counts_list[np.isfinite(counts_list)]
    counts_list = counts_list[counts_list >= 0]

    xi = np.rint(counts_list).astype(int)
    max_count = int(np.max(xi))

    bin_counts = np.bincount(xi, minlength=max_count + 1).astype(float)
    num_samples = float(np.sum(bin_counts))

    x_vals = np.arange(max_count + 1, dtype=float)
    hist = bin_counts / num_samples

    # Probability uncertainty from shot noise
    hist_errs = np.sqrt(np.maximum(bin_counts, 1.0)) / num_samples

    return x_vals, hist, hist_errs, bin_counts


def compound_poisson_pdf(z, rate):
    if isinstance(z, list):
        z = np.array(z)
    z_not_array = not isinstance(z, np.ndarray)
    # If z is not an array, turn it into one so we can use the same code.
    # Convert back at the end.
    if z_not_array:
        z = np.array([z])
    z = z[np.newaxis, :]

    lower_lim = 0
    # upper_lim = round(rate + 5 * np.sqrt(rate))
    upper_lim = _safe_upper_lim(rate)  # <<< fixed upper limit
    integral_points = np.arange(lower_lim, upper_lim, 1, dtype=np.float64)
    integral_points = integral_points[:, np.newaxis]

    integrand = poisson_pdf(z, integral_points) * poisson_pdf(integral_points, rate)
    ret_val = np.sum(integrand, axis=0)
    if z_not_array:
        return ret_val[0]
    else:
        return ret_val


def compound_poisson_with_ionization_pdf(z, lambda_0, lambda_m, ion):
    if isinstance(z, list):
        z = np.array(z)
    z_not_array = not isinstance(z, np.ndarray)
    # If z is not an array, turn it into one so we can use the same code.
    # Convert back at the end.
    if z_not_array:
        z = np.array([z])
    z = z[np.newaxis, :]

    lower_lim = 0
    upper_lim = round(lambda_m + 5 * np.sqrt(lambda_m))
    integral_points = np.arange(lower_lim, upper_lim, 1, dtype=np.float64)
    integral_points = integral_points[:, np.newaxis]

    lambda_diff = lambda_m - lambda_0
    term_1 = poisson_pdf(integral_points, lambda_m) * (1 - ion + (1 / 2) * ion**2)
    coeff_23 = ion * (lambda_diff + ion * lambda_0) / (lambda_diff**2)
    term_2 = gammaincc(integral_points + 1, lambda_0)
    term_3 = gammaincc(integral_points + 1, lambda_m)
    coeff_45 = (ion**2) / (lambda_diff**2)
    term_4 = (integral_points + 1) * gammaincc(integral_points + 2, lambda_m)
    term_5 = (integral_points + 1) * gammaincc(integral_points + 2, lambda_0)
    integrand = poisson_pdf(z, integral_points) * (
        term_1 + coeff_23 * (term_2 - term_3) + coeff_45 * (term_4 - term_5)
    )

    ret_val = np.sum(integrand, axis=0)
    if z_not_array:
        return ret_val[0]
    else:
        return ret_val


def compound_poisson_cdf(x, rate):
    return _calc_cdf(ProbDist.COMPOUND_POISSON, x, rate)



def broadened_compound_poisson_pdf(z, rate, sigma):
    """
    Discrete broadened compound-Poisson PMF.

    Physical meaning:
        First generate ideal compound-Poisson photon counts m.
        Then add Gaussian technical broadening with width sigma.
        Finally bin back into integer count bin z.

    P(z | rate, sigma)
        = sum_m P_CP(m | rate) *
          integral_{z-0.5}^{z+0.5} Normal(y | m, sigma) dy

    This lets a single charge-state peak be broader than pure compound-Poisson,
    so the fitter is less tempted to invent fake extra NV modes.
    """

    if isinstance(z, list):
        z = np.array(z)

    z_not_array = not isinstance(z, np.ndarray)
    if z_not_array:
        z = np.array([z], dtype=float)

    z = np.asarray(z, dtype=float).ravel()

    rate = float(max(rate, 1e-12))
    sigma = float(max(sigma, 1e-9))

    # Sum over ideal integer counts m.
    upper_lim = _safe_upper_lim(
        rate + 6.0 * sigma,
        nsig=7,
        min_lim=20,
        max_lim=50_000,
    )

    m_vals = np.arange(0, upper_lim + 1, dtype=float)

    # Ideal compound-Poisson probability for m.
    p_m = compound_poisson_pdf(m_vals, rate)
    p_m = np.asarray(p_m, dtype=float)
    p_m = np.where(np.isfinite(p_m), p_m, 0.0)
    p_m = np.clip(p_m, 0.0, None)

    s = float(np.sum(p_m))
    if s > 0 and np.isfinite(s):
        p_m = p_m / s

    # Probability that Gaussian-broadened value lands in integer bin z.
    # Shape: len(m_vals) x len(z)
    z_lo = z[None, :] - 0.5
    z_hi = z[None, :] + 0.5
    m = m_vals[:, None]

    bin_prob = norm.cdf((z_hi - m) / sigma) - norm.cdf((z_lo - m) / sigma)
    bin_prob = np.where(np.isfinite(bin_prob), bin_prob, 0.0)
    bin_prob = np.clip(bin_prob, 0.0, 1.0)

    ret_val = p_m @ bin_prob

    if z_not_array:
        return float(ret_val[0])

    return ret_val


def broadened_compound_poisson_cdf(x, rate, sigma):
    return _calc_cdf(ProbDist.BROADENED_COMPOUND_POISSON, x, rate, sigma)


def compound_poisson_with_ionization_cdf(x, lambda_0, lambda_m, ion):
    return _calc_cdf(
        ProbDist.COMPOUND_POISSON_WITH_IONIZATION, x, lambda_0, lambda_m, ion
    )


def broadened_poisson_pdf(x, rate, sigma, do_norm=True):
    if isinstance(x, (list, np.ndarray)):
        ret_vals = [broadened_poisson_pdf(el, rate, sigma) for el in x]
        return np.array(ret_vals)

    def integrand(y):
        return poisson_pdf(y, rate) * gaussian_pdf(x - y, 0, sigma)

    lower_lim = round(max(0, x - 4 * sigma))
    upper_lim = round(x + 4 * sigma)

    integral_points = np.arange(lower_lim, upper_lim, 1, dtype=np.float64)
    ret_val = np.sum(integrand(integral_points))
    return ret_val


def broadened_poisson_cdf(x, rate, sigma):
    return _calc_cdf(ProbDist.BROADENED_POISSON, x, rate, sigma)


# @cache
def gaussian_pdf(x, mean, std):
    return inv_root_2_pi * (1 / std) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    # return norm(loc=mean, scale=std).pdf(x)


def gaussian_cdf(x, mean, std):
    return norm(loc=mean, scale=std).cdf(x)


def skew_gaussian_pdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).pdf(x)


def skew_gaussian_cdf(x, mean, std, skew):
    return skewnorm(a=skew, loc=mean, scale=std).cdf(x)


def exponential_integral(nu, z):
    return (z ** (nu - 1)) * gammainc(1 - nu, z)


# endregion
def fit_bimodal_histogram(
    counts_list, prob_dist: ProbDist, no_print=True, no_plot=True
):
    """Fit the passed probability distribution to a histogram of the passed counts_list.
    counts_list should have some population in both modes

    Parameters
    ----------
    counts_list : list | np.ndarray
        Array-like of recorded counts from an NV
    prob_dist : ProbDist
        Probability distribution to use for the fit
    no_print : bool, optional
        Whether to skip printing out the results of the fit, by default True
    no_plot : bool, optional
        Whether to skip plotting out the histogram and fit, by default True

    Returns
    -------
    np.ndarray(float)
        popt, the optimized fit parameters
    """

    # no_plot = False

    counts_list = np.asarray(counts_list).flatten()

    # Remove outliers
    median = np.median(counts_list)
    std = np.std(counts_list)
    if np.isfinite(std) and std > 0:
        counts_list = counts_list[counts_list < median + 10 * std]

    num_samples = len(counts_list)
    if num_samples < 50:
        return None, None, None

    # Integer-count histogram
    x_vals, hist, hist_errs, bin_counts = _make_integer_histogram(counts_list)
    max_count = int(x_vals[-1])
    num_samples = int(np.sum(bin_counts))

    ### Fit the histogram
    # Get guess params
    mean_dark_guess = round(np.quantile(counts_list, 0.15))
    mean_bright_guess = round(np.quantile(counts_list, 0.65))
    mean_dark_min = round(np.quantile(counts_list, 0.02))
    mean_bright_max = round(np.quantile(counts_list, 0.98))
    ratio_guess = 0.3
    
    bounds = (-np.inf, np.inf)  # Default bounds
    if prob_dist is ProbDist.SKEW_GAUSSIAN:
        guess_params = [ratio_guess]
        guess_params.extend([mean_dark_guess, 2 * np.sqrt(mean_dark_guess), 2])
        guess_params.extend([mean_bright_guess, 2 * np.sqrt(mean_bright_guess), -2])
        skew_lim = 5
        bounds = (
            (0, mean_dark_min, 0, -skew_lim, mean_dark_min, 0, -skew_lim),
            (1, mean_bright_max, np.inf, skew_lim, mean_bright_max, np.inf, skew_lim),
        )
    elif prob_dist is ProbDist.POISSON:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess)
    elif prob_dist is ProbDist.BROADENED_POISSON:
        guess_params = (ratio_guess, mean_dark_guess, 3, mean_bright_guess, 3)
        bounds = (
            (0, mean_dark_min, 1, mean_dark_min, 1),
            (1, mean_bright_max, mean_dark_guess, mean_bright_max, mean_dark_guess),
        )
    elif prob_dist is ProbDist.COMPOUND_POISSON:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess)
        bounds = (
            (0, mean_dark_min, mean_dark_min),
            (1, mean_bright_max, mean_bright_max),
        )

    elif prob_dist is ProbDist.BROADENED_COMPOUND_POISSON:
        sigma_dark_guess = max(2.0, 0.25 * np.sqrt(max(mean_dark_guess, 1.0)))
        sigma_bright_guess = max(2.0, 0.25 * np.sqrt(max(mean_bright_guess, 1.0)))

        guess_params = (
            ratio_guess,
            mean_dark_guess,
            sigma_dark_guess,
            mean_bright_guess,
            sigma_bright_guess,
        )

        bounds = (
            (
                0.0,
                mean_dark_min,
                0.25,
                mean_dark_min,
                0.25,
            ),
            (
                1.0,
                mean_bright_max,
                max(3.0, 0.75 * mean_bright_guess),
                mean_bright_max,
                max(3.0, 0.75 * mean_bright_guess),
            ),
        )
    elif prob_dist is ProbDist.COMPOUND_POISSON_WITH_IONIZATION:
        guess_params = (ratio_guess, mean_dark_guess, mean_bright_guess, 0.0)
        bounds = (
            (0, mean_dark_min, mean_dark_min, 0.0),
            (1, mean_bright_max, mean_bright_max, 0.5),
        )

    # return guess_params

    # Fit
    fit_fn = get_bimodal_pdf(prob_dist)
    try:
        popt, pcov, red_chi_sq = curve_fit(
            fit_fn,
            x_vals,
            hist,
            guess_params,
            hist_errs,
            bounds=bounds,
            # ftol=1e-6,
            # xtol=1e-6,
        )
        if not no_print:
            print(f"Fit Parameters: {popt}")
            print(f"Reduced chi squared: {red_chi_sq}")

        if not no_plot:
            fig, ax = plt.subplots()
            ax.set_xlabel("Integrated counts")
            ax.set_ylabel("Probability")
            kpl.histogram(ax, counts_list, density=True)
            x_vals = np.linspace(0, np.max(counts_list), 1000)

            # Dark mode
            dark_ratio = popt[0]
            single_mode_fn = get_single_mode_pdf(prob_dist)
            num_params = get_single_mode_num_params(prob_dist)
            line = dark_ratio * single_mode_fn(x_vals, *popt[1 : 1 + num_params])
            kpl.plot_line(
                ax, x_vals, line, color=kpl.KplColors.RED, label=r"NV$^{0}$ mode"
            )

            # Bright mode
            num_params = get_single_mode_num_params(prob_dist)
            line = (1 - dark_ratio) * single_mode_fn(x_vals, *popt[1 + num_params :])
            kpl.plot_line(
                ax, x_vals, line, color=kpl.KplColors.GREEN, label=r"NV$^{-}$ mode"
            )

            # Both modes
            line = fit_fn(x_vals, *popt)
            kpl.plot_line(ax, x_vals, line, color=kpl.KplColors.BLUE, label="Combined")

            ax.legend(loc=kpl.Loc.UPPER_RIGHT)
            kpl.show(block=True)
        return popt, pcov, red_chi_sq
    except Exception as exc:
        return None, None, None


def determine_threshold(
    popt,
    prob_dist: "ProbDist",
    dark_mode_weight=None,
    do_print=False,
    ret_fidelity=False,
):
    """Determine the optimal threshold for assigning a state based on measured counts.
    Returns None (and None fidelity) gracefully if thresholding can't be determined.
    """

    # ---------- Basic validation ----------
    if popt is None:
        return (None, None) if ret_fidelity else None

    popt = np.asarray(popt, dtype=float)
    if popt.size < 3 or np.any(~np.isfinite(popt)):
        warnings.warn(
            f"determine_threshold: invalid popt (size={popt.size}, finite={np.all(np.isfinite(popt))}). Returning None."
        )
        return (None, None) if ret_fidelity else None

    # ---------- Weights ----------
    if dark_mode_weight is None:
        dark_mode_weight = popt[0]
    try:
        dark_mode_weight = float(dark_mode_weight)
    except Exception:
        dark_mode_weight = 0.5

    if not np.isfinite(dark_mode_weight):
        dark_mode_weight = 0.5
    dark_mode_weight = float(np.clip(dark_mode_weight, 0.0, 1.0))
    bright_mode_weight = 1.0 - dark_mode_weight

    # ---------- Params / means ----------
    num_single_mode_params = get_single_mode_num_params(prob_dist)
    single_mode_cdf = get_single_mode_cdf(prob_dist)

    # NOTE: You currently use a "hack" mean definition:
    # mean_counts_dark = popt[1], mean_counts_bright = popt[2]
    # Keep it, but guard it.
    mean_counts_dark = popt[1]
    mean_counts_bright = popt[2]

    if not (np.isfinite(mean_counts_dark) and np.isfinite(mean_counts_bright)):
        warnings.warn("determine_threshold: non-finite mean counts. Returning None.")
        return (None, None) if ret_fidelity else None

    # Build threshold candidates robustly (never empty)
    lo = min(mean_counts_dark, mean_counts_bright)
    hi = max(mean_counts_dark, mean_counts_bright)

    # Candidate thresholds are half-integers spanning the region between modes
    start = np.floor(lo) - 0.5
    stop = np.ceil(hi) + 0.5 + 1e-12  # +eps to avoid arange edge emptiness
    thresh_options = np.arange(start, stop, 1.0, dtype=float)

    # If means are extremely close (or weird), ensure at least one option
    if thresh_options.size == 0:
        thresh_options = np.array(
            [(mean_counts_dark + mean_counts_bright) / 2.0], dtype=float
        )

    # ---------- Search for best threshold ----------
    best_fid = -np.inf
    best_thresh = None

    for val in thresh_options:
        try:
            dark_left_prob = float(
                single_mode_cdf(val, *popt[1 : 1 + num_single_mode_params])
            )
            bright_left_prob = float(
                single_mode_cdf(val, *popt[1 + num_single_mode_params :])
            )
        except Exception:
            continue

        if not (np.isfinite(dark_left_prob) and np.isfinite(bright_left_prob)):
            continue

        # Keep probabilities in [0, 1] in case numerical routines drift slightly
        dark_left_prob = float(np.clip(dark_left_prob, 0.0, 1.0))
        bright_left_prob = float(np.clip(bright_left_prob, 0.0, 1.0))

        bright_right_prob = 1.0 - bright_left_prob

        fid = dark_mode_weight * dark_left_prob + bright_mode_weight * bright_right_prob
        if np.isfinite(fid) and fid > best_fid:
            best_fid = fid
            best_thresh = val

    # ---------- Fallback if everything failed ----------
    if best_thresh is None:
        best_thresh = (mean_counts_dark + mean_counts_bright) / 2.0
        best_fid = np.nan
        warnings.warn(
            "determine_threshold: could not evaluate any candidate thresholds (CDF failures / bad params). "
            f"Falling back to midpoint threshold={best_thresh} with fidelity=nan."
        )

    if do_print:
        if np.isfinite(best_fid):
            print(
                f"Optimum readout fidelity {round(best_fid, 3)} achieved at threshold {best_thresh}"
            )
        else:
            print(f"Using fallback threshold {best_thresh} (fidelity unavailable)")

    return (best_thresh, best_fid) if ret_fidelity else best_thresh


# -----------------------------
# Structured N-NV binomial model
# -----------------------------
def _binom_coeffs(n: int) -> np.ndarray:
    return np.array([math.comb(n, k) for k in range(n + 1)], dtype=float)


def _binom_weights(n: int, p: float) -> np.ndarray:
    p = float(np.clip(p, 0.0, 1.0))
    ks = np.arange(n + 1, dtype=float)
    coeff = _binom_coeffs(n)
    w = coeff * (p**ks) * ((1.0 - p) ** (n - ks))
    s = float(np.sum(w))
    if s <= 0 or not np.isfinite(s):
        w = np.zeros(n + 1, dtype=float)
        w[0] = 1.0
        return w
    return w / s


def get_binomial_multinv_pdf(prob_dist: ProbDist, n_nvs: int):
    """
    Mixture over k=0..N with binomial weights and

        lambda_k = bg + N*rate0 + k*delta

    Parameters:
        p_minus : probability each NV is NV-
        bg      : background/local offset counts
        rate0   : per-NV NV0 contribution
        delta   : extra counts per NV changing NV0 -> NV-
    """
    if prob_dist is ProbDist.COMPOUND_POISSON_WITH_IONIZATION:
        raise ValueError(
            "Binomial multinv model is for reference histograms without ionization."
        )

    single_pdf = get_single_mode_pdf(prob_dist)
    coeff = _binom_coeffs(n_nvs)
    K = n_nvs + 1

    def fn(x, p_minus, bg, rate0, delta):
        p_minus = float(np.clip(p_minus, 0.0, 1.0))
        bg = float(max(bg, 0.0))
        rate0 = float(max(rate0, 1e-9))
        delta = float(max(delta, 0.0))

        ks = np.arange(K, dtype=float)
        w = coeff * (p_minus**ks) * ((1.0 - p_minus) ** (n_nvs - ks))
        s = float(np.sum(w))
        if s <= 0 or not np.isfinite(s):
            w = np.zeros(K, dtype=float)
            w[0] = 1.0
        else:
            w = w / s

        x_arr = np.asarray(x, dtype=float)
        y = np.zeros_like(x_arr, dtype=float)

        base = bg + n_nvs * rate0

        for k in range(K):
            lam_k = base + k * delta
            y += w[k] * single_pdf(x_arr, lam_k)

        return y

    return fn


def get_two_nv_unequal_pdf(prob_dist: ProbDist):
    """
    Two-NV unequal-brightness model.

    Four configurations:
        00, 10, 01, 11

    Means:
        bg + 2*rate0
        bg + 2*rate0 + delta1
        bg + 2*rate0 + delta2
        bg + 2*rate0 + delta1 + delta2
    """
    single_pdf = get_single_mode_pdf(prob_dist)

    def fn(x, p1, p2, bg, rate0, delta1, delta2):
        p1 = float(np.clip(p1, 0.0, 1.0))
        p2 = float(np.clip(p2, 0.0, 1.0))

        bg = float(max(bg, 0.0))
        rate0 = float(max(rate0, 1e-9))
        delta1 = float(max(delta1, 0.0))
        delta2 = float(max(delta2, 0.0))

        x_arr = np.asarray(x, dtype=float)
        y = np.zeros_like(x_arr, dtype=float)

        base = bg + 2 * rate0

        weights = [
            (1 - p1) * (1 - p2),
            p1 * (1 - p2),
            (1 - p1) * p2,
            p1 * p2,
        ]

        means = [
            base,
            base + delta1,
            base + delta2,
            base + delta1 + delta2,
        ]

        for w, lam in zip(weights, means):
            y += w * single_pdf(x_arr, lam)

        return y

    return fn


def fit_two_nv_unequal_histogram(
    counts_list,
    prob_dist: ProbDist,
    no_print=True,
    no_plot=True,
    n_restarts: int = 8,
    seed: int = 0,
):
    """
    Fit two-NV unequal-brightness model.

    popt = [p1, p2, bg, rate0, delta1, delta2]
    """
    from utils.tool_belt import curve_fit

    counts_list = np.asarray(counts_list, dtype=float).ravel()
    counts_list = counts_list[np.isfinite(counts_list)]
    counts_list = counts_list[counts_list >= 0]

    if counts_list.size < 50:
        return None, None, None

    median = np.median(counts_list)
    std = np.std(counts_list)
    if np.isfinite(std) and std > 0:
        counts_list = counts_list[counts_list < median + 10 * std]

    if counts_list.size < 50:
        return None, None, None

    x_vals, hist, hist_errs, bin_counts = _make_integer_histogram(counts_list)

    q02 = float(np.quantile(counts_list, 0.02))
    q15 = float(np.quantile(counts_list, 0.15))
    q50 = float(np.quantile(counts_list, 0.50))
    q85 = float(np.quantile(counts_list, 0.85))
    q98 = float(np.quantile(counts_list, 0.98))

    bg_guess = max(0.0, 0.25 * q02)
    rate0_guess = max(1e-3, (q15 - bg_guess) / 2.0)

    span = max(q98 - q15, 1e-3)
    delta1_guess = max(1e-3, 0.4 * span)
    delta2_guess = max(1e-3, 0.7 * span)

    p1_guess = 0.7
    p2_guess = 0.7

    bounds = (
        (0.0, 0.0, 0.0, 1e-6, 0.0, 0.0),
        (1.0, 1.0, max(1.0, q15), max(1e-3, q98), max(1e-3, q98), max(1e-3, q98)),
    )

    fit_fn = get_two_nv_unequal_pdf(prob_dist)

    rng = np.random.default_rng(seed)
    best = None

    for r in range(max(1, n_restarts)):
        p0 = np.array(
            [p1_guess, p2_guess, bg_guess, rate0_guess, delta1_guess, delta2_guess],
            dtype=float,
        )

        if r > 0:
            p0[0] = float(np.clip(p0[0] + 0.15 * rng.standard_normal(), 0.05, 0.95))
            p0[1] = float(np.clip(p0[1] + 0.15 * rng.standard_normal(), 0.05, 0.95))
            p0[2] = float(np.clip(p0[2] * (1.0 + 0.30 * rng.standard_normal()), bounds[0][2], bounds[1][2]))
            p0[3] = float(np.clip(p0[3] * (1.0 + 0.25 * rng.standard_normal()), bounds[0][3], bounds[1][3]))
            p0[4] = float(np.clip(p0[4] * (1.0 + 0.35 * rng.standard_normal()), bounds[0][4], bounds[1][4]))
            p0[5] = float(np.clip(p0[5] * (1.0 + 0.35 * rng.standard_normal()), bounds[0][5], bounds[1][5]))

        try:
            popt, pcov, red_chi_sq = curve_fit(
                fit_fn,
                x_vals,
                hist,
                p0,
                hist_errs,
                bounds=bounds,
            )

            # Sort delta1/delta2 so output is easier to read
            p1, p2, bg, rate0, d1, d2 = [float(v) for v in popt]
            if d2 < d1:
                popt = np.array([p2, p1, bg, rate0, d2, d1], dtype=float)

            if (best is None) or (red_chi_sq < best[2]):
                best = (popt, pcov, red_chi_sq)

        except Exception:
            continue

    if best is None:
        return None, None, None

    popt, pcov, red_chi_sq = best

    if not no_print:
        print(f"[2NV unequal] popt={popt} red_chi_sq={red_chi_sq}")

    return popt, pcov, red_chi_sq

def fit_binomial_multinv_histogram(
    counts_list,
    prob_dist: ProbDist,
    n_nvs: int,
    no_print=True,
    no_plot=True,
    n_restarts: int = 5,
    seed: int = 0,
):
    """
    Fits (p_minus, rate0, delta) using the same histogram + curve_fit style as fit_bimodal_histogram().
    Returns: popt, pcov, red_chi_sq
    """
    from utils.tool_belt import curve_fit  # same wrapper you already use

    counts_list = np.asarray(counts_list).flatten()

    # Remove outliers (same logic as your current fitter) :contentReference[oaicite:2]{index=2}
    median = np.median(counts_list)
    std = np.std(counts_list)
    if np.isfinite(std) and std > 0:
        counts_list = counts_list[counts_list < median + 10 * std]
    num_samples = len(counts_list)
    if num_samples < 50:
        return None, None, None
    
    x_vals, hist, hist_errs, bin_counts = _make_integer_histogram(counts_list)
    max_count = int(x_vals[-1])
    num_samples = int(np.sum(bin_counts))

    # ---- Initial guesses from quantiles ----
    # ---- Initial guesses from quantiles ----
    q02 = float(np.quantile(counts_list, 0.02))
    q15 = float(np.quantile(counts_list, 0.15))
    q65 = float(np.quantile(counts_list, 0.65))
    q98 = float(np.quantile(counts_list, 0.98))
    mean_tot = float(np.mean(counts_list))

    # background is a small baseline offset
    bg_guess = max(0.0, 0.25 * q02)

    # low mode approximately bg + N*rate0
    rate0_guess = max(1e-3, (q15 - bg_guess) / max(n_nvs, 1))

    # spacing between charge manifolds
    delta_guess = max(1e-3, (q98 - q15) / max(n_nvs, 1))

    # estimate p from mean_total ≈ bg + N*rate0 + N*p*delta
    p_guess = (mean_tot - bg_guess - n_nvs * rate0_guess) / max(
        n_nvs * delta_guess, 1e-9
    )
    p_guess = float(np.clip(p_guess, 0.05, 0.95))

    # Bounds
    bg_min = 0.0
    bg_max = max(1.0, q15)

    rate0_min = 1e-6
    rate0_max = max(rate0_min + 1e-3, q98 / max(n_nvs, 1))

    delta_min = 0.0
    delta_max = max(1e-3, q98 - q02)

    bounds = (
        (0.0, bg_min, rate0_min, delta_min),
        (1.0, bg_max, rate0_max, delta_max),
    )

    fit_fn = get_binomial_multinv_pdf(prob_dist, n_nvs)

    # Multi-start to avoid local minima
    rng = np.random.default_rng(seed)
    best = None
    for r in range(max(1, n_restarts)):
        p0 = np.array([p_guess, bg_guess, rate0_guess, delta_guess], dtype=float)
        if r > 0:
            p0[0] = float(np.clip(p0[0] + 0.10 * rng.standard_normal(), 0.05, 0.95))

            p0[1] = float(
                np.clip(
                    p0[1] * (1.0 + 0.30 * rng.standard_normal()),
                    bounds[0][1],
                    bounds[1][1],
                )
            )

            p0[2] = float(
                np.clip(
                    p0[2] * (1.0 + 0.20 * rng.standard_normal()),
                    bounds[0][2],
                    bounds[1][2],
                )
            )

            p0[3] = float(
                np.clip(
                    p0[3] * (1.0 + 0.20 * rng.standard_normal()),
                    bounds[0][3],
                    bounds[1][3],
                )
            )
        try:
            popt, pcov, red_chi_sq = curve_fit(
                fit_fn, x_vals, hist, p0, hist_errs, bounds=bounds
            )
            if (best is None) or (red_chi_sq < best[2]):
                best = (popt, pcov, red_chi_sq)
        except Exception:
            continue

    if best is None:
        return None, None, None

    popt, pcov, red_chi_sq = best

    if not no_print:
        print(f"[binomial multinv] N={n_nvs} popt={popt} red_chi_sq={red_chi_sq}")

    return popt, pcov, red_chi_sq


def determine_multithreshold_binomial_multinv(
    popt,
    prob_dist: ProbDist,
    n_nvs: int,
    x_max: int,
    ret_fidelity: bool = True,
):
    """
    Multi-class Bayes-optimal thresholds for classes k=0..N (k = #NV-).
    Returns thresholds (length N) at half-integers and overall fidelity.
    """
    if popt is None:
        return (None, None) if ret_fidelity else None

    single_cdf = get_single_mode_cdf(prob_dist)

    p_minus, bg, rate0, delta = [float(v) for v in popt]
    weights = _binom_weights(n_nvs, p_minus)  # length N+1
    K = n_nvs + 1

    # sort by k already monotonic in lambda_k = N*rate0 + k*delta
    T = np.arange(-0.5, float(x_max) + 0.5 + 1e-12, 1.0, dtype=float)
    Ngrid = T.size

    # CDF table: (K, Ngrid)
    cdfs = np.zeros((K, Ngrid), dtype=float)
    base = max(bg, 0.0) + n_nvs * max(rate0, 1e-9)
    delta = max(delta, 0.0)
    for k in range(K):
        lam_k = base + k * delta
        # vectorize CDF eval
        cdfs[k, :] = np.array([single_cdf(t, lam_k) for t in T], dtype=float)
    cdfs = np.clip(cdfs, 0.0, 1.0)

    # DP over ordered thresholds
    dp = np.full((K - 1, Ngrid), -np.inf, dtype=float)
    back = np.full((K - 1, Ngrid), -1, dtype=int)

    dp[0, :] = weights[0] * cdfs[0, :]

    for i in range(1, K - 1):
        for j in range(Ngrid):
            best_val = -np.inf
            best_k = -1
            for k0 in range(j):
                val = dp[i - 1, k0] + weights[i] * (cdfs[i, j] - cdfs[i, k0])
                if val > best_val:
                    best_val = val
                    best_k = k0
            dp[i, j] = best_val
            back[i, j] = best_k

    best_total = -np.inf
    best_j = 0
    for j in range(Ngrid):
        val = dp[K - 2, j] + weights[K - 1] * (1.0 - cdfs[K - 1, j])
        if val > best_total:
            best_total = float(val)
            best_j = j

    idxs = [best_j]
    for i in range(K - 2, 0, -1):
        best_j = back[i, best_j]
        idxs.append(best_j)
    idxs = list(reversed(idxs))
    thresholds = [float(T[ii]) for ii in idxs]  # length K-1 = N

    return (thresholds, float(best_total)) if ret_fidelity else thresholds


def determine_threshold_any_minus_binomial_multinv(
    popt,
    prob_dist: ProbDist,
    n_nvs: int,
    x_max: int,
    ret_fidelity: bool = True,
):
    """
    Binary threshold: class 0 (k=0) vs class >=1 (any NV-).
    This keeps the meaning closest to your original NV0 vs NV- thresholding.
    """
    if popt is None:
        return (None, None) if ret_fidelity else None

    single_cdf = get_single_mode_cdf(prob_dist)
    single_pdf = get_single_mode_pdf(prob_dist)

    p_minus, bg, rate0, delta = [float(v) for v in popt]
    weights = _binom_weights(n_nvs, p_minus)
    w0 = float(weights[0])
    wrest = 1.0 - w0
    if wrest <= 1e-12:
        # effectively always k=0; threshold irrelevant
        t = 0.5
        fid = 1.0
        return (t, fid) if ret_fidelity else t

    base = max(bg, 0.0) + n_nvs * max(rate0, 1e-9)
    delta = max(delta, 0.0)

    # Build mixture CDF for rest (k>=1), normalized
    T = np.arange(-0.5, float(x_max) + 0.5 + 1e-12, 1.0, dtype=float)

    cdf0 = np.array([single_cdf(t, base + 0 * delta) for t in T], dtype=float)
    cdf_rest = np.zeros_like(T, dtype=float)
    for k in range(1, n_nvs + 1):
        lam_k = base + k * delta
        cdf_k = np.array([single_cdf(t, lam_k) for t in T], dtype=float)
        cdf_rest += float(weights[k]) * cdf_k
    cdf_rest = cdf_rest / wrest

    # Bayes accuracy for threshold t:
    # correct = w0*P0(left) + wrest*Prest(right)
    best_fid = -np.inf
    best_t = None
    for i, t in enumerate(T):
        p0_left = float(np.clip(cdf0[i], 0.0, 1.0))
        prest_left = float(np.clip(cdf_rest[i], 0.0, 1.0))
        fid = w0 * p0_left + wrest * (1.0 - prest_left)
        if fid > best_fid:
            best_fid = float(fid)
            best_t = float(t)

    return (best_t, best_fid) if ret_fidelity else best_t


def analyze_charge_histogram_multinv_binomial(
    counts_list,
    prob_dist: ProbDist = ProbDist.COMPOUND_POISSON,
    max_nvs: int = 3,
    force_nvs: int | None = None,
    bic_extra_nv_penalty: float = 1.0,
    seed: int = 0,
    include_unequal_2nv: bool = True,
):
    """
    Try N=1..max_nvs, or force_nvs if given.
    Fit binomial-structured multi-NV equal-brightness model and select N by BIC.

    Main equal-brightness model:
        lambda_k = bg + N*rate0 + k*delta

    Extra diagnostic model:
        2-NV unequal-brightness model

    For compatibility, this returns the best equal-brightness model as the main fit.
    The 2-NV unequal model is stored in candidate_results for comparison.
    """

    x = np.asarray(counts_list, dtype=float).ravel()
    x = x[np.isfinite(x)]
    x = x[x >= 0]

    if x.size < 50:
        return {"ok": False, "reason": "too_few_samples"}

    # Outlier trim
    med = np.median(x)
    std = np.std(x)
    if np.isfinite(std) and std > 0:
        x = x[x < med + 10 * std]

    if x.size < 50:
        return {"ok": False, "reason": "too_few_samples_after_trim"}

    # Integer-count histogram for likelihood
    xi = np.rint(np.clip(x, 0, None)).astype(int)
    max_count = int(np.max(xi))
    xs_int = np.arange(0, max_count + 1, dtype=float)

    bin_counts = np.bincount(xi, minlength=max_count + 1).astype(float)
    n_samp = float(np.sum(bin_counts))

    Ns = [int(force_nvs)] if force_nvs is not None else list(range(1, int(max_nvs) + 1))

    best_equal = None
    candidate_results = []

    # --------------------------------------------------
    # Equal-brightness models: 1nv_equal, 2nv_equal, 3nv_equal
    # --------------------------------------------------
    for N in Ns:
        popt, pcov, red = fit_binomial_multinv_histogram(
            x,
            prob_dist,
            N,
            no_print=True,
            no_plot=True,
            n_restarts=5,
            seed=seed + 19 * N,
        )

        if popt is None:
            continue

        # popt = [p_minus, bg, rate0, delta]
        p_minus, bg, rate0, delta = [float(v) for v in popt]

        pdf_fn = get_binomial_multinv_pdf(prob_dist, N)
        p = np.asarray(pdf_fn(xs_int, *popt), dtype=float)

        # Defensive normalization
        p = np.clip(p, 1e-300, None)
        p = p / float(np.sum(p))

        ll = float(np.sum(bin_counts * np.log(p)))

        # Free parameters: p_minus, bg, rate0, delta
        # plus structural penalty for larger N
        k_free = 4 + (N - 1)

        bic = float(
            k_free * np.log(max(n_samp, 1.0))
            - 2.0 * ll
            + bic_extra_nv_penalty * (N - 1)
        )

        thresholds, fid_multi = determine_multithreshold_binomial_multinv(
            popt,
            prob_dist,
            N,
            x_max=max_count,
            ret_fidelity=True,
        )

        thr_any, fid_any = determine_threshold_any_minus_binomial_multinv(
            popt,
            prob_dist,
            N,
            x_max=max_count,
            ret_fidelity=True,
        )

        weights = _binom_weights(N, p_minus)

        res = dict(
            ok=True,
            model=f"{N}nv_equal",
            prob_dist=prob_dist,
            n_nvs=int(N),

            # Raw fit
            popt=popt,
            pcov=pcov,

            # Explicit physical parameters
            p_minus=p_minus,
            bg=bg,
            rate0=rate0,
            delta=delta,

            # Fit quality
            red_chi_sq=red,
            ll=ll,
            bic=bic,
            k_free=k_free,
            n_samp=n_samp,

            # Classification
            weights=weights,
            thresholds=thresholds,
            threshold_any=thr_any,
            fidelity_any=fid_any,
            fidelity_multiclass=fid_multi,
            x_max=max_count,
        )

        candidate_results.append(res)

        if (best_equal is None) or (res["bic"] < best_equal["bic"]):
            best_equal = res

    if best_equal is None:
        return {"ok": False, "reason": "fit_failed"}

    # --------------------------------------------------
    # Extra diagnostic candidate: 2-NV unequal-brightness model
    # --------------------------------------------------
    if include_unequal_2nv and (force_nvs is None or int(force_nvs) == 2):
        popt, pcov, red = fit_two_nv_unequal_histogram(
            x,
            prob_dist,
            no_print=True,
            no_plot=True,
            n_restarts=8,
            seed=seed + 777,
        )

        if popt is not None:
            pdf_fn = get_two_nv_unequal_pdf(prob_dist)
            p = np.asarray(pdf_fn(xs_int, *popt), dtype=float)

            p = np.clip(p, 1e-300, None)
            p = p / float(np.sum(p))

            ll = float(np.sum(bin_counts * np.log(p)))

            # parameters: p1, p2, bg, rate0, delta1, delta2
            k_free = 6

            bic = float(
                k_free * np.log(max(n_samp, 1.0))
                - 2.0 * ll
            )

            p1, p2, bg, rate0, delta1, delta2 = [float(v) for v in popt]

            res_unequal = dict(
                ok=True,
                model="2nv_unequal",
                prob_dist=prob_dist,
                n_nvs=2,

                popt=popt,
                pcov=pcov,

                p1=p1,
                p2=p2,
                bg=bg,
                rate0=rate0,
                delta1=delta1,
                delta2=delta2,

                red_chi_sq=red,
                ll=ll,
                bic=bic,
                k_free=k_free,
                n_samp=n_samp,
                x_max=max_count,

                # No thresholds yet for unequal model
                thresholds=None,
                threshold_any=None,
                fidelity_any=None,
                fidelity_multiclass=None,
            )

            candidate_results.append(res_unequal)

    # --------------------------------------------------
    # Diagnostics: best candidate among all models
    # --------------------------------------------------
    best_any = min(candidate_results, key=lambda r: r["bic"])

    best_equal["candidate_results"] = candidate_results
    best_equal["best_candidate_model"] = best_any["model"]
    best_equal["best_candidate_bic"] = best_any["bic"]
    best_equal["best_equal_model"] = best_equal["model"]
    best_equal["best_equal_bic"] = best_equal["bic"]
    best_equal["unequal_2nv_beats_equal"] = (
        best_any["model"] == "2nv_unequal"
        and best_any["bic"] < best_equal["bic"]
    )

    return best_equal

if __name__ == "__main__":
    kpl.init_kplotlib()
    # (z, lambda_0, lambda_m, ion)
    line_fn = compound_poisson_with_ionization_pdf
    fig, ax = plt.subplots()
    x_vals = np.linspace(0, 80, 1000)
    line_vals = line_fn(x_vals, 20, 40, 0.0)
    print(np.sum(line_vals) * x_vals[1] - x_vals[0])
    kpl.plot_line(ax, x_vals, line_vals)
    kpl.show(block=True)
