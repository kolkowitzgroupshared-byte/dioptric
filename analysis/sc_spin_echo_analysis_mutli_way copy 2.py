# -*- coding: utf-8 -*-
"""
Spin-echo: finer fit + fitted-figure + parameter panels

- Physics-y comb with quartic lobes, amplitude taper, width growth, chirp
- Optional two-frequency sin^2 beating with phases
- Smoothly plugs into your plotting + data pipeline
"""

import os
import sys
import traceback
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import inspect
import re, hashlib, datetime as dt
import itertools, random
import pandas as pd

# --- Optional numba (falls back gracefully) ----------------------------------
try:
    from numba import njit
except Exception:

    def njit(*_args, **_kwargs):
        def wrap(fn):
            return fn

        return wrap


# --- Your utilities (assumed in PYTHONPATH) ----------------------------------
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield as widefield
from utils.tool_belt import curve_fit
from scipy.optimize import least_squares

# =============================================================================
# Finer model
#    - COMB has NO overall comb_contrast
#    - MOD carries the overall amplitude (and optional beating)
# =============================================================================


def fine_decay(
    tau_us,
    baseline=1.0,
    comb_contrast=0.6,
    revival_time=37.0,
    width0_us=6.0,
    T2_ms=0.08,
    T2_exp=1.0,
    amp_taper_alpha=0.0,
    width_slope=0.0,
    revival_chirp=0.0,
    # NEW (additive, signed; zero-mean cos carrier(s))
    osc_amp=0.0,
    osc_f0=0.0,
    osc_phi0=0.0,
    osc_f1=0.0,
    osc_phi1=0.0,
):
    """
    signal(τ) = baseline
                - comb_contrast * envelope(τ) * COMB(τ)               [dip]
                +                 envelope(τ) * COMB(τ) * OSC(τ)      [signed oscillation]

    envelope(τ) = exp[-(τ / (1000*T2_ms))^T2_exp]
    COMB(τ)     = Σ_k [ 1/(1+k)^amp_taper_alpha ] * exp(-((τ-μ_k)/w_k)^4)
                    μ_k = k * revival_time * (1 + k*revival_chirp)
                    w_k = width0_us * (1 + k*width_slope)

    OSC(τ)      = osc_amp * [ cos(2π f0 τ + φ0) + cos(2π f1 τ + φ1) ]   # zero mean
    τ in microseconds, f in cycles/μs, phases in rad.
    """
    amp_taper_alpha = 0.0 if amp_taper_alpha is None else float(amp_taper_alpha)
    width_slope = 0.0 if width_slope is None else float(width_slope)
    revival_chirp = 0.0 if revival_chirp is None else float(revival_chirp)
    osc_amp = 0.0 if osc_amp is None else float(osc_amp)
    osc_f0 = 0.0 if osc_f0 is None else float(osc_f0)
    osc_f1 = 0.0 if osc_f1 is None else float(osc_f1)
    osc_phi0 = 0.0 if osc_phi0 is None else float(osc_phi0)
    osc_phi1 = 0.0 if osc_phi1 is None else float(osc_phi1)

    tau = np.asarray(tau_us, dtype=float).ravel()
    width0_us = max(1e-9, float(width0_us))
    revival_time = max(1e-9, float(revival_time))
    T2_us = max(1e-9, 1000.0 * float(T2_ms))
    T2_exp = float(T2_exp)

    # envelope
    envelope = np.exp(-((tau / T2_us) ** T2_exp))

    # number of revivals to include
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / revival_time)) + 1))

    comb = _comb_quartic_powerlaw(
        tau,
        revival_time,
        width0_us,
        amp_taper_alpha,
        width_slope,
        revival_chirp,
        n_guess,
    )

    carrier = envelope * comb
    # baseline minus revival dip
    dip = comb_contrast * carrier
    # additive, zero-mean oscillation (can push above baseline)
    osc = 0.0
    if osc_amp != 0.0:
        if osc_f0 != 0.0:
            osc += np.cos(2 * np.pi * osc_f0 * tau + osc_phi0)
        if osc_f1 != 0.0:
            osc += np.cos(2 * np.pi * osc_f1 * tau + osc_phi1)

    return baseline - dip + carrier * (osc_amp * osc)


def fine_decay_fixed_revival(
    tau,
    baseline,
    comb_contrast,
    width0_us,
    T2_ms,
    T2_exp,
    amp_taper_alpha=None,
    width_slope=None,
    revival_chirp=None,
    osc_amp=None,
    osc_f0=None,
    osc_f1=None,
    osc_phi0=None,
    osc_phi1=None,
    _fixed_rev_time_us=37.0,
):
    return fine_decay(
        tau,
        baseline,
        comb_contrast,
        _fixed_rev_time_us,
        width0_us,
        T2_ms,
        T2_exp,
        amp_taper_alpha,
        width_slope,
        revival_chirp,
        osc_amp,  # was osc_contrast
        osc_f0,
        osc_phi0,  # <-- swap order
        osc_f1,  # <-- swap order
        osc_phi1,
    )


@njit
def _comb_quartic_powerlaw(
    tau, revival_time, width0_us, amp_taper_alpha, width_slope, revival_chirp, n_guess
):
    """
    NOTE: no overall amplitude factor here (no comb_contrast).
    """
    n = tau.shape[0]
    out = np.zeros(n, dtype=np.float64)
    tmax = 0.0
    for i in range(n):
        if tau[i] > tmax:
            tmax = tau[i]

    for k in range(n_guess):
        mu_k = k * revival_time * (1.0 + k * revival_chirp)
        w_k = width0_us * (1.0 + k * width_slope)
        if w_k <= 0.0:
            continue
        if mu_k > tmax + 5.0 * w_k:
            break

        amp_k = 1.0 / ((1.0 + k) ** amp_taper_alpha)  # <- amplitude taper only
        inv_w4 = 1.0 / (w_k**4)

        for i in range(n):
            x = tau[i] - mu_k
            out[i] += amp_k * np.exp(-(x * x) * (x * x) * inv_w4)

    return out


# =============================================================================
# Fitting helpers
# =============================================================================


def _smart_percentiles(y, p_low=5, p_high=90):
    y = np.asarray(y, float)
    if y.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(y, p_low))
    hi = float(np.nanpercentile(y, p_high))
    return lo, hi


def _uniformize(times_us, y):
    """Resample to a uniform grid if needed (for FFT seeding). Returns (tu, yu)."""
    t = np.asarray(times_us, float)
    y = np.asarray(y, float)
    if t.size < 8:
        return t, y
    dt = np.diff(t)
    if np.allclose(dt, dt.mean(), rtol=1e-2, atol=1e-3):
        return t, y
    n = max(256, int(2 ** np.ceil(np.log2(len(t)))))
    tu = np.linspace(t.min(), t.max(), n)
    yu = np.interp(tu, t, y)
    return tu, yu


def _fft_peak_freq(times_us, resid, fmin, fmax):
    """Find dominant frequency (cycles/μs) in [fmin, fmax] from FFT of residual."""
    t, r = _uniformize(times_us, resid)
    if t.size < 16:
        return None
    # detrend & window
    r = r - np.nanmedian(r)
    r = np.nan_to_num(r, nan=0.0)
    r = r * np.hanning(len(r))
    dt = np.diff(t).mean()
    freqs = np.fft.rfftfreq(len(r), d=dt)  # cycles per μs if t is μs
    Y = np.fft.rfft(r)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band):
        return None
    k = np.argmax(np.abs(Y[band]))
    f_hat = float(freqs[band][k])
    return f_hat if np.isfinite(f_hat) and f_hat > 0 else None


def _initial_guess_and_bounds(times_us, y, enable_extras=True, fixed_rev_time=None):
    """
    Smart p0 and bounds for fine_decay that (a) respect your 0→0.6→1 behavior
    and (b) allow both slow and fast beats without violating Nyquist.
    """
    times_us = np.asarray(times_us, float)
    y = np.asarray(y, float)

    # ------- robust baseline & contrast -------
    y_lo, y_hi = _smart_percentiles(y, 5, 90)
    baseline_guess = float(y_hi) if np.isfinite(y_hi) else (y[0] if y.size else 0.6)
    # we want ~0.4 contrast if min near 0 and baseline ~0.6
    comb_contrast_guess = max(
        0.05, min(baseline_guess - y_lo, baseline_guess - 0.02, 0.9)
    )
    # enforce min >= 0 later via bounds tying comb_contrast to baseline

    # ------- envelope rough seed -------
    # take a late window
    if times_us.size:
        j0 = max(0, len(times_us) - max(7, len(times_us) // 10))
        y_late = float(np.nanmean(y[j0:])) if j0 < len(y) else float(y[-1])
        ratio = (baseline_guess - y_late) / max(1e-9, comb_contrast_guess)
        ratio = min(max(ratio, 1e-6), 0.999999)
        T2_exp_guess = 2.0  # slightly softer than 3
        # T2_ms guess scaled by total span; keep conservative
        tspan_us = (
            max(1.0, float(times_us.max() - times_us.min())) if times_us.size else 100.0
        )
        T2_ms_guess = max(
            0.01,
            0.25
            * (tspan_us / 1000.0)
            / max(1e-6, (-np.log(ratio)) ** (1.0 / T2_exp_guess)),
        )
    else:
        T2_exp_guess, T2_ms_guess = 2.0, 0.1

    width0_guess = 6.0
    revival_guess = 38.0 if fixed_rev_time is None else fixed_rev_time

    # ------- base vector & bounds -------
    if fixed_rev_time is None:
        p0 = [
            baseline_guess,
            comb_contrast_guess,
            revival_guess,
            width0_guess,
            T2_ms_guess,
            T2_exp_guess,
        ]
        # tie comb_contrast to baseline: ub later adjusted below
        lb = [0.0, 0.00, 25.0, 1.0, 0.001, 0.6]
        ub = [1.05, 0.95, 55.0, 20.0, 0.6, 4.0]
    else:
        p0 = [
            baseline_guess,
            comb_contrast_guess,
            width0_guess,
            T2_ms_guess,
            T2_exp_guess,
        ]
        lb = [0.0, 0.00, 1.0, 0.001, 0.6]
        ub = [1.05, 0.95, 20.0, 0.6, 4.0]

    # tighten comb_contrast upper bound so min >= 0:
    # comb_contrast <= baseline - eps
    eps_min = 0.01
    if fixed_rev_time is None:
        ub[1] = min(ub[1], max(0.05, p0[0] - eps_min))
    else:
        ub[1] = min(ub[1], max(0.05, p0[0] - eps_min))

    if not enable_extras:
        return np.array(p0, float), np.array(lb, float), np.array(ub, float)

    # extra_p0 = [0.3, 0.02, 0.0,  0.30, 0.10, 0.01, 0.0, 0.0]
    # extra_lb = [0.0, 0.00, -0.01, -1.00, 0.00, 0.00, -np.pi, -np.pi]
    # extra_ub = [2.0, 0.20,  0.01,  1.00, 0.40, 0.20,  np.pi,  np.pi]

    extra_p0 = [0.3, 0.02, 0.0, 0.30, 0.10, 0.01, 0.0, 0.0]
    extra_lb = [0.0, 0.00, -0.06, -1.00, 0.00, 0.00, -np.pi, -np.pi]
    extra_ub = [4.0, 0.80, 0.06, 1.00, 6.00, 6.00, np.pi, np.pi]

    p0.extend(extra_p0)
    lb.extend(extra_lb)
    ub.extend(extra_ub)
    return np.array(p0, float), np.array(lb, float), np.array(ub, float)


def _seed_osc_from_grid(
    times_us,
    y,
    yerr,
    core_params,
    fit_fn_core,
    f_lo=0.02,
    f_hi=0.15,
    n_grid=60,
    phase_grid=(0.0, np.pi / 2, np.pi, -np.pi / 2),
):
    """
    Given a core fit (no oscillation), find a good (osc_amp, f, phi) seed by
    linear regression over a coarse frequency grid. Returns (amp, f, phi).
    """
    t = np.asarray(times_us, float)
    yy = np.asarray(y, float)
    ee = np.maximum(1e-9, np.asarray(yerr, float))
    carrier = fit_fn_core(
        t, *core_params
    )  # this is baseline - dip if you use your core-only fn
    # We want residual of data after removing baseline - dip:
    resid = yy - carrier

    best = (0.0, None, None)  # (amp, f, phi)

    for f in np.linspace(f_lo, f_hi, n_grid):
        c = np.cos(2 * np.pi * f * t)
        s = np.sin(2 * np.pi * f * t)
        # weighted linear regression resid ≈ carrier * (A*c + B*s)
        X = np.column_stack([carrier * c, carrier * s])
        w = 1.0 / ee
        Xw = X * w[:, None]
        yw = resid * w
        # solve (Xw^T Xw) beta = Xw^T yw
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        A, B = beta
        amp = np.hypot(A, B)
        phi = np.arctan2(-B, A)  # so that A cos + B sin = amp cos(· + phi)
        if amp > best[0]:
            best = (float(amp), float(f), float(phi))

    return best  # (amp, f, phi)


def core_only(
    tau,
    baseline,
    comb_contrast,
    revival_time,
    width0_us,
    T2_ms,
    T2_exp,
    amp_taper_alpha=0.0,
    width_slope=0.0,
    revival_chirp=0.0,
):
    tau = np.asarray(tau, float)
    env = np.exp(-((tau / (1000.0 * T2_ms)) ** T2_exp))
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / max(1e-9, revival_time))) + 1))
    comb = _comb_quartic_powerlaw(
        tau,
        revival_time,
        width0_us,
        amp_taper_alpha or 0.0,
        width_slope or 0.0,
        revival_chirp or 0.0,
        n_guess,
    )
    return baseline - comb_contrast * env * comb


def _chi2_red(y, yerr, yfit, npar):
    resid = (y - yfit) / np.maximum(1e-12, yerr)
    chi2 = float(np.sum(resid**2))
    dof = max(1, len(y) - npar)
    return chi2 / dof


def _sanitize_trace(times_us, y, yerr, err_floor=1e-3):
    """Remove non-finite rows; enforce an error floor."""
    m = np.isfinite(times_us) & np.isfinite(y) & np.isfinite(yerr)
    t = times_us[m].astype(float)
    yy = y[m].astype(float)
    ee = np.maximum(err_floor, np.abs(yerr[m].astype(float)))
    # drop duplicates in time (keep first)
    if t.size:
        _, uniq_idx = np.unique(t, return_index=True)
        t, yy, ee = t[uniq_idx], yy[uniq_idx], ee[uniq_idx]
    return t, yy, ee


def _revival_weights(t, Trev, w0, width_slope=0.0, nrev=8, boost=2.0):
    w = np.ones_like(t, float)
    for k in range(nrev):
        mu_k = k * Trev
        wk = w0 * (1.0 + k * width_slope)
        w *= 1.0 + (boost - 1.0) * np.exp(-(((t - mu_k) / (1.5 * wk + 1e-9)) ** 2))
    return w


def _fit_curve_fit(fit_fn, times_us, y, yerr, p0, lb, ub, maxfev):
    popt, pcov, _ = curve_fit(
        fit_fn,
        times_us,
        y,
        p0,
        yerr,
        bounds=[lb, ub],
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
        maxfev=maxfev,  # <- IMPORTANT
    )
    yfit = fit_fn(times_us, *popt)
    red = _chi2_red(y, yerr, yfit, len(popt))
    return popt, pcov, red


def _fit_least_squares(fit_fn, times_us, y, yerr, p0, lb, ub, max_nfev, weights=None):
    """Robust fallback using soft-L1; optional per-point weights."""
    t = np.asarray(times_us, float)
    yy = np.asarray(y, float)
    ee = np.maximum(1e-12, np.asarray(yerr, float))
    ww = None if weights is None else np.asarray(weights, float)

    def resid(p):
        r = (yy - fit_fn(t, *p)) / ee
        return r if ww is None else r * ww

    res = least_squares(
        resid,
        x0=np.asarray(p0, float),
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    popt = res.x
    yfit = fit_fn(t, *popt)
    red = _chi2_red(yy, ee, yfit, len(popt))
    return popt, None, red


def _pick_best(cands):
    """cands is list of tuples (name, popt, pcov, red, fit_fn, note)"""
    good = [(i, c) for i, c in enumerate(cands) if np.isfinite(c[3])]
    if not good:
        return None
    # pick smallest reduced chi^2
    i_best = min(good, key=lambda ic: ic[1][3])[0]
    return cands[i_best]


# ---------------- Amp-bound sweep helpers ----------------


def _core_len_for_fn(fit_fn):
    """Number of core params before 'extras' start."""
    # fine_decay: [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp] -> 6
    # fine_decay_fixed_revival: [baseline, comb_contrast, width0_us, T2_ms, T2_exp] -> 5
    return 6 if fit_fn is fine_decay else 5


def _set_osc_amp_bounds(lb, ub, fit_fn, new_min, new_max):
    """In-place change of osc_amp bounds in [amp_taper_alpha, width_slope, revival_chirp, osc_amp, ...]."""
    k0 = _core_len_for_fn(fit_fn)  # start of extras
    idx_amp = k0 + 3  # extras: α, slope, chirp, osc_amp, ...
    if idx_amp >= len(lb):
        # extras disabled; nothing to set
        return False
    lb[idx_amp] = float(new_min)
    ub[idx_amp] = float(new_max)
    return True


# === NEW: parameter indexing helpers =========================================
def _param_index_map(fit_fn):
    """
    Return a dict mapping parameter names -> indices for the chosen fit_fn.
    Order (fine_decay):
      [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp, osc_contrast,
       osc_f0, osc_f1, osc_phi0, osc_phi1]

    For fine_decay_fixed_revival, revival_time is absent from the core.
    """
    if fit_fn is fine_decay:
        names = [
            "baseline",
            "comb_contrast",
            "revival_time",
            "width0_us",
            "T2_ms",
            "T2_exp",
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_contrast",
            "osc_f0",
            "osc_f1",
            "osc_phi0",
            "osc_phi1",
        ]
    else:
        names = [
            "baseline",
            "comb_contrast",
            "width0_us",
            "T2_ms",
            "T2_exp",
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_contrast",
            "osc_f0",
            "osc_f1",
            "osc_phi0",
            "osc_phi1",
        ]
    return {n: i for i, n in enumerate(names)}


def _set_bounds(lb, ub, idx, vmin, vmax):
    lb[idx] = float(vmin)
    ub[idx] = float(vmax)


def _set_initial(p0, idx, val):
    p0[idx] = float(val)


def _get_val(vec, idx, default=None):
    return float(vec[idx]) if 0 <= idx < len(vec) else default


# === NEW: frequency band inference from sampling =============================
def _freq_band_from_times(times_us, margin=0.05, fmax_cap=None):
    """
    Infer a sensible [fmin, fmax] band (cycles/µs) from time span and min spacing.
    fmin ~ 1/(span), fmax ~ 1/(2*dt_min). Shrink/expand slightly via 'margin'.
    """
    t = np.asarray(times_us, float)
    if t.size < 2:
        return 0.0, 1.0
    span = max(1e-9, float(t.max() - t.min()))
    dt_min = max(1e-9, float(np.diff(np.unique(t)).min()))
    fmin = 1.0 / span
    fmax = 0.5 / dt_min
    if fmax_cap is not None:
        fmax = min(fmax, float(fmax_cap))
    # apply margin
    fmin = max(0.0, (1.0 - margin) * fmin)
    fmax = (1.0 + margin) * fmax
    return fmin, fmax


# === NEW: FFT-based frequency seeding ========================================
def _seed_freqs_fft(
    times_us,
    y,
    yerr,
    fit_fn_core,
    p0_core,
    lb_core,
    ub_core,
    band=None,
    n_peaks=3,
    include_harmonics=True,
):
    """
    1) Fit a core (no-extras) model to capture comb/envelope
    2) FFT residual -> pick top 'n_peaks' in band
    3) Optionally add harmonics (2f, 3f) if within band
    Returns sorted unique candidate frequencies (list of floats).
    """
    # fit core with curve_fit; if it fails, least_squares fallback
    try:
        popt, _, _ = _fit_curve_fit(
            fit_fn_core, times_us, y, yerr, p0_core, lb_core, ub_core, maxfev=120000
        )
    except Exception:
        try:
            popt, _, _ = _fit_least_squares(
                fit_fn_core,
                times_us,
                y,
                yerr,
                p0_core,
                lb_core,
                ub_core,
                max_nfev=120000,
            )
        except Exception:
            # fallback: no detrending
            popt = None

    if popt is not None:
        resid = y - fit_fn_core(times_us, *popt)
    else:
        resid = y - np.nanmedian(y)

    fmin, fmax = band if band is not None else _freq_band_from_times(times_us)
    # use your FFT helper
    t, r = _uniformize(times_us, resid)
    if t.size < 16:
        return []

    r = r - np.nanmedian(r)
    r = np.nan_to_num(r, nan=0.0)
    r = r * np.hanning(len(r))
    dt = np.diff(t).mean()
    freqs = np.fft.rfftfreq(len(r), d=dt)
    Y = np.fft.rfft(r)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        return []

    mags = np.abs(Y[band_mask])
    fband = freqs[band_mask]
    order = np.argsort(mags)[::-1]  # descending
    picked = []
    for k in order[: max(1, n_peaks)]:
        f = float(fband[k])
        if f > 0 and np.isfinite(f):
            picked.append(f)

    # add harmonics if desired
    cands = set(picked)
    if include_harmonics:
        for f in list(picked):
            for m in (2, 3, 0.5, 1 / 3):
                fm = m * f
                if fmin <= fm <= fmax:
                    cands.add(fm)

    return sorted(cands)


def _normalize_freq_band(times_us, band=None, fmax_cap=None):
    fmin_samp, fmax_samp = _freq_band_from_times(
        times_us, margin=0.0, fmax_cap=fmax_cap
    )
    if band is None:
        fmin, fmax = fmin_samp, fmax_samp
    else:
        fmin, fmax = map(float, band)
        if fmin > fmax:  # swap if inverted
            fmin, fmax = fmax, fmin
        # intersect with sampling band (Nyquist-safe)
        fmin = max(0.0, max(fmin, fmin_samp))
        fmax = min(fmax, fmax_samp)
    # ensure non-empty, add tiny slack
    eps = 1e-9
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin + eps:
        # fallback to safe band from sampling
        fmin, fmax = fmin_samp, fmax_samp
        if fmax <= fmin + eps:
            # last-ditch: pick a tiny positive window
            fmin, fmax = 0.0, max(1e-3, fmax_samp)
    return float(fmin), float(fmax)


# === NEW: generic multi-sweeper ==============================================
def _clone_vecs(*vecs):
    return [np.array(v, float) if v is not None else None for v in vecs]


def _grid_product(dict_of_lists):
    """
    dict_of_lists: {"osc_f0":[...], "osc_f1":[...], "osc_phi0":[...], ...}
    Yields dicts with one selection per key. Empty dict if input empty.
    """
    if not dict_of_lists:
        yield {}
        return
    keys = list(dict_of_lists.keys())
    lists = [dict_of_lists[k] for k in keys]
    sizes = [len(L) for L in lists]
    if any(s == 0 for s in sizes):
        return
    import itertools

    for combo in itertools.product(*lists):
        yield {k: v for k, v in zip(keys, combo)}


def _apply_param_overrides(p0, lb, ub, fit_fn, overrides=None, bound_boxes=None):
    """
    overrides: {"osc_f0": value, "revival_time": value, ...} -> set initial p0
    bound_boxes: {"osc_f0": (min,max), ...} -> set bounds
    Returns updated (p0, lb, ub).
    """
    pmap = _param_index_map(fit_fn)
    p0, lb, ub = _clone_vecs(p0, lb, ub)
    overrides = overrides or {}
    bound_boxes = bound_boxes or {}
    for name, val in overrides.items():
        if name in pmap:
            _set_initial(p0, pmap[name], val)
    for name, box in bound_boxes.items():
        if (name in pmap) and (box is not None):
            _set_bounds(lb, ub, pmap[name], box[0], box[1])
    return p0, lb, ub


def _sanitize_bound_boxes(bound_boxes, band):
    if not isinstance(bound_boxes, dict):
        return {}
    bmin, bmax = band
    out = {}
    for k, box in bound_boxes.items():
        if box is None:
            continue
        lo, hi = box
        if lo > hi:
            lo, hi = hi, lo
        lo = max(0.0, max(lo, bmin))
        hi = min(hi, bmax)
        if hi <= lo + 1e-9:
            # collapse to a tiny valid window around the midpoint
            mid = 0.5 * (bmin + bmax)
            lo, hi = max(0.0, mid * 0.98), mid * 1.02
        out[k] = (float(lo), float(hi))
    return out


def build_freq_pairs(freqs, band, max_pairs=40, min_sep=0.01):
    """Return a small set of (f0,f1) with f1 < f0, well separated, in-band."""
    lo, hi = band
    fs = [
        f
        for f in sorted(set(round(float(x), 9) for x in freqs))
        if np.isfinite(f) and lo <= f <= hi
    ]
    # Prefer the top-K by FFT magnitude if you have that; otherwise take spaced picks:
    if len(fs) > max_pairs:
        step = max(1, len(fs) // max_pairs)
        fs = fs[::step]
    out = []
    for i, f0 in enumerate(fs):
        for f1 in fs[:i]:  # enforce f1 < f0 (no permutations, no equals)
            if abs(f0 - f1) < min_sep:  # avoid nearly identical pairs
                continue
            out.append((f0, f1))
    return out[:max_pairs]


def harmonic_candidates(f0, kmax=4, tol=0.003):
    # Return plausible companions around simple ratios: 1/2, 2/3, 3/4, 1, 4/3, 3/2, 2
    ratios = [(1, 2), (2, 3), (3, 4), (1, 1), (4, 3), (3, 2), (2, 1)]
    cand = []
    for p, q in ratios:
        f = f0 * (p / q)
        if f > 0:
            cand.append(f)
    # Collapse near-duplicates
    cand = sorted(set(round(c, 6) for c in cand))
    return cand


# ---- parameter indexing helpers --------------------------------------------


def _param_index_map(fit_fn):
    """
    Map parameter names to indices for the chosen fit function.

    For fine_decay (free revival_time), the expected order is:
      [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp,
       osc_amp, osc_f0, osc_phi0, osc_f1, osc_phi1]

    For fine_decay_fixed_revival (no revival_time in core), the order is:
      [baseline, comb_contrast, width0_us, T2_ms, T2_exp,
       amp_taper_alpha, width_slope, revival_chirp,
       osc_amp, osc_f0, osc_phi0, osc_f1, osc_phi1]
    """
    if fit_fn is fine_decay:
        names = [
            "baseline",
            "comb_contrast",
            "revival_time",
            "width0_us",
            "T2_ms",
            "T2_exp",
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_amp",
            "osc_f0",
            "osc_phi0",
            "osc_f1",
            "osc_phi1",
        ]
    else:  # fine_decay_fixed_revival
        names = [
            "baseline",
            "comb_contrast",
            "width0_us",
            "T2_ms",
            "T2_exp",
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_amp",
            "osc_f0",
            "osc_phi0",
            "osc_f1",
            "osc_phi1",
        ]
    return {n: i for i, n in enumerate(names)}


def _set_bounds(lb, ub, idx, vmin, vmax):
    idx = int(idx)
    lb[idx] = float(vmin)
    ub[idx] = float(vmax)


def _set_initial(p0, idx, val):
    idx = int(idx)
    p0[idx] = float(val)


def _as_float_list(xs):
    out = []
    for x in xs:
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


# ---- main fast fitter -------------------------------------------------------

# ================= Bound-hit repair utilities ================================


def _param_index_map(fit_fn):
    """Parameter order map matching your fine_decay / fine_decay_fixed_revival."""
    if fit_fn is fine_decay:
        names = [
            "baseline",
            "comb_contrast",
            "revival_time",
            "width0_us",
            "T2_ms",
            "T2_exp",
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_amp",
            "osc_f0",
            "osc_phi0",
            "osc_f1",
            "osc_phi1",
        ]
    else:  # fine_decay_fixed_revival
        names = [
            "baseline",
            "comb_contrast",
            "width0_us",
            "T2_ms",
            "T2_exp",
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_amp",
            "osc_f0",
            "osc_phi0",
            "osc_f1",
            "osc_phi1",
        ]
    return {n: i for i, n in enumerate(names)}


def _set_bounds(lb, ub, idx, vmin, vmax):
    idx = int(idx)
    lb[idx] = float(vmin)
    ub[idx] = float(vmax)


def _set_initial(p0, idx, val):
    idx = int(idx)
    p0[idx] = float(val)


def _as_float_list(xs):
    out = []
    for x in xs:
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def _bound_hits(popt, lb, ub, frac_tol=0.01, abs_tol=1e-9):
    """
    Return {idx: 'low'|'high'} for params within tol of bounds.
    """
    hits = {}
    popt = np.asarray(popt, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    span = np.maximum(ub - lb, 0.0)
    low_tol = np.maximum(frac_tol * span, abs_tol)
    high_tol = np.maximum(frac_tol * span, abs_tol)
    for i, (p, lo, hi, ltol, htol) in enumerate(zip(popt, lb, ub, low_tol, high_tol)):
        if not np.isfinite(p):
            continue
        if (p - lo) <= ltol:
            hits[i] = "low"
        elif (hi - p) <= htol:
            hits[i] = "high"
    return hits


def _repair_bounds_for_hits(
    lb,
    ub,
    hits,
    pmap,
    *,
    physics_caps=None,  # optional dict: name -> (hard_lo, hard_hi)
    freq_band=None,  # optional (flo, fhi) in cycles/µs for osc_f*
    scale=1.8,
    min_span=1e-6,
):
    """
    Widen only the parameters that hit bounds by 'scale', respecting physics caps.
    Guarantees non-negative frequencies for osc_f0/osc_f1 and keeps them within freq_band if given.
    Returns (lb2, ub2).
    """
    import numpy as np

    lb2 = np.array(lb, float)
    ub2 = np.array(ub, float)
    name_by_idx = {j: i for i, j in pmap.items()}  # idx -> name

    # ---- sensible defaults if not provided ----
    # NB: you can tweak these to your experiment
    if physics_caps is None:
        physics_caps = {}

    flo, fhi = (None, None)
    if freq_band is not None and len(freq_band) == 2:
        flo, fhi = float(freq_band[0]), float(freq_band[1])

    def _caps_for(name, cur_lo, cur_hi):
        # start from provided caps (if any)
        hard_lo, hard_hi = physics_caps.get(name, (-np.inf, np.inf))

        # non-negativity for frequencies, and keep within band if known
        if name in ("osc_f0", "osc_f1"):
            if np.isfinite(hard_lo):
                hard_lo = max(hard_lo, 0.0)
            else:
                hard_lo = 0.0  # enforce non-negative freq
            if flo is not None:
                hard_lo = max(hard_lo, 0.0)  # keep ≥0 even if band is tiny
            if fhi is not None and np.isfinite(hard_hi):
                hard_hi = min(hard_hi, fhi)
            elif fhi is not None:
                hard_hi = fhi

        # a few gentle, safe caps (only if not already provided)
        if name == "T2_ms" and not np.isfinite(
            physics_caps.get("T2_ms", (np.nan, np.nan))[0]
        ):
            hard_lo = max(hard_lo, 1e-6)  # >0
        if name == "width0_us" and not np.isfinite(
            physics_caps.get("width0_us", (np.nan, np.nan))[0]
        ):
            hard_lo = max(hard_lo, 1e-6)  # >0
        if name == "T2_exp" and not np.isfinite(
            physics_caps.get("T2_exp", (np.nan, np.nan))[0]
        ):
            # typical range; adjust to your prior
            hard_lo = max(hard_lo, 0.25)
            hard_hi = min(hard_hi, 4.0) if np.isfinite(hard_hi) else 4.0

        return hard_lo, hard_hi

    for idx, side in hits.items():
        name = name_by_idx.get(idx, None)
        if name is None:
            continue

        lo, hi = float(lb2[idx]), float(ub2[idx])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        span = max(min_span, hi - lo)
        hard_lo, hard_hi = _caps_for(name, lo, hi)

        if side == "low":
            lo_new = lo - (scale - 1.0) * span
            hi_new = hi + 0.25 * span
        else:  # side == "high"
            lo_new = lo - 0.25 * span
            hi_new = hi + (scale - 1.0) * span

        # clamp to hard caps
        lo_new = max(hard_lo, lo_new)
        hi_new = min(hard_hi, hi_new)

        # enforce non-negative for frequencies explicitly (belt & suspenders)
        if name in ("osc_f0", "osc_f1"):
            lo_new = max(0.0, lo_new)

        # keep min span
        if not np.isfinite(lo_new) or not np.isfinite(hi_new) or hi_new <= lo_new:
            mid = 0.5 * (lo + hi)
            lo_new, hi_new = mid - 0.5 * min_span, mid + 0.5 * min_span
            # re-apply caps
            lo_new = max(hard_lo, lo_new)
            hi_new = min(hard_hi, hi_new)
            if hi_new - lo_new < min_span:
                # final guarantee
                hi_new = lo_new + min_span

        lb2[idx], ub2[idx] = float(lo_new), float(hi_new)

    return lb2, ub2


def _retie_contrast_to_baseline(p0, lb, ub, pmap, eps=0.01):
    """Ensure comb_contrast ≤ baseline - eps by adjusting its UB."""
    if "baseline" in pmap and "comb_contrast" in pmap:
        ib = pmap["baseline"]
        ic = pmap["comb_contrast"]
        ub[ic] = min(ub[ic], max(0.0, p0[ib] - eps))
    return p0, lb, ub


def _reseed_to_center(p0, lb, ub, only_idxs=None):
    """Set p0 to centers (optionally only for indices in only_idxs)."""
    p0 = np.array(p0, float)
    lb = np.array(lb, float)
    ub = np.array(ub, float)
    if only_idxs is None:
        mask = np.ones_like(p0, dtype=bool)
    else:
        mask = np.zeros_like(p0, dtype=bool)
        mask[list(only_idxs)] = True
    centers = 0.5 * (lb + ub)
    p0[mask] = centers[mask]
    return p0


# Physics-aware hard caps (tune to your platform)
_PHYS_CAPS = {
    "baseline": (0.0, 1.1),
    "comb_contrast": (0.0, 1.0),
    "revival_time": (20.0, 70.0),  # µs; adjust to your B-field
    "width0_us": (0.2, 50.0),
    "T2_ms": (0.003, 5.0),
    "T2_exp": (0.6, 3.0),
    "amp_taper_alpha": (0.0, 3.0),
    "width_slope": (-1.0, 1.0),
    "revival_chirp": (-0.01, 0.01),
    "osc_amp": (-10.0, 10.0),  # amp windows still govern effective range
    "osc_f0": (0.0, 5.0),
    "osc_f1": (0.0, 5.0),
    "osc_phi0": (-np.pi, np.pi),
    "osc_phi1": (-np.pi, np.pi),
}


def _uniq_in_band(freqs, lo, hi, r=9):
    s, out = set(), []
    for f in freqs:
        try:
            f = float(f)
        except:
            continue
        if np.isfinite(f) and lo <= f <= hi:
            k = round(f, r)
            if k not in s:
                s.add(k)
                out.append(f)
    return sorted(out)


def _maxmin(vals, k):
    if not vals:
        return []
    if k >= len(vals):
        return list(vals)
    picked = [vals[0], vals[-1]]
    cand = vals[1:-1]
    while len(picked) < k and cand:
        best = max(cand, key=lambda v: min(abs(v - c) for c in picked))
        picked.append(best)
        cand.remove(best)
    return sorted(picked)


def diversified_prior_overrides(
    freq_seeds,
    band,
    *,
    min_sep=0.01,
    nyq_guard=0.9,
    n_single=8,
    n_pairs=20,
    nbuckets=5,
    jitter=0.02,
    seed=1234,
):
    random.seed(seed)
    lo, hi = band
    hi *= nyq_guard
    seeds = _uniq_in_band(freq_seeds, lo, hi)
    if not seeds:
        return []

    # ---- singles (f1=0): bucketed + max-min + light jitter ----
    edges = np.linspace(seeds[0], seeds[-1], nbuckets + 1)
    buckets = [[] for _ in range(nbuckets)]
    bi = 0
    for v in seeds:
        while bi < nbuckets - 1 and v > edges[bi + 1]:
            bi += 1
        buckets[bi].append(v)

    singles = []
    for B in buckets:
        singles += _maxmin(B, min(2, len(B)))
    if len(singles) < n_single:
        singles = _uniq_in_band(singles + _maxmin(seeds, n_single), lo, hi)

    def apart(x, S):
        return all(abs(x - s) >= min_sep for s in S)

    for v in list(singles)[: max(1, n_single // 3)]:
        j = v * (1 + random.uniform(-jitter, jitter))
        if lo <= j <= hi and apart(j, singles):
            singles.append(j)
    singles = _uniq_in_band(singles, lo, hi)[:n_single]

    # ---- pairs: low↔high + harmonics + max-min fallback ----
    ovs, seen = [], set()

    def push(a, b):
        a, b = (a, b) if a <= b else (b, a)
        if abs(a - b) < min_sep:
            return False
        k = (round(a, 6), round(b, 6))
        if k in seen:
            return False
        seen.add(k)
        ovs.append({"osc_f0": a, "osc_f1": b})
        return True

    for f0 in singles:
        push(f0, 0.0)
    need = max(0, n_pairs)

    # low↔high
    mid = nbuckets // 2 or 1
    lows = [b for b in buckets[:mid] if b]
    highs = [b for b in buckets[mid:] if b]
    for BL in lows:
        for BH in highs:
            for a in _maxmin(BL, min(2, len(BL))):
                for b in _maxmin(BH, min(2, len(BH))):
                    if need and push(a, b):
                        need -= 1
                    if need == 0:
                        break
                if need == 0:
                    break
        if need == 0:
            break

    # harmonics (½, ⅔, ¾)
    if need:
        for f0 in _maxmin(seeds, min(8, len(seeds))):
            for r in (0.5, 2 / 3, 0.75):
                f1 = f0 * r
                if lo <= f1 <= hi and need and push(f0, f1):
                    need -= 1
            if need == 0:
                break

    # max-min fallback by separation
    if need:
        cand = sorted(
            (
                (abs(a - b), a, b)
                for a, b in itertools.combinations(seeds, 2)
                if abs(a - b) >= min_sep
            ),
            reverse=True,
        )
        for _, a, b in cand:
            if need and push(a, b):
                need -= 1
            if need == 0:
                break

    return ovs


def _score_tuple(popt, redchi, lb, ub, pmap, *, amp_weight=1e-3, bound_weight=0.1):
    """
    Return a tuple for min()-selection:
      ( redchi + bound_penalty, amp_norm )
    so we first pick good χ² that are NOT at bounds; then prefer smaller osc_amp.

    - bound_penalty adds 'bound_weight' if a key param (T2_ms, T2_exp, osc_f*) is near its bound.
    - amp_norm is |osc_amp| (or 0 if absent), scaled by amp_weight so it is a tie-breaker only.
    """
    pen = 0.0
    if pmap:  # protect if pmap is None

        def _near_wall(idx):
            if idx is None or idx >= len(popt):
                return False
            lo, hi, x = lb[idx], ub[idx], popt[idx]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return False
            span = max(1e-12, hi - lo)
            return (x - lo) <= 0.01 * span or (hi - x) <= 0.01 * span

        iT2 = pmap.get("T2_ms")
        inT2 = pmap.get("T2_exp")
        if iT2 is not None and _near_wall(iT2):
            pen += bound_weight * 2.0  # extra-punish T2 at wall
        if inT2 is not None and _near_wall(inT2):
            pen += bound_weight

        for k in ("osc_f0", "osc_f1"):
            ik = pmap.get(k)
            if ik is not None and _near_wall(ik):
                pen += 0.5 * bound_weight

    iA = pmap.get("osc_amp") if pmap else None
    amp_norm = abs(float(popt[iA])) if (iA is not None and iA < len(popt)) else 0.0
    return (float(redchi) + pen, amp_weight * amp_norm)


def _amp_windows_and_seeds():
    # small→large windows; seeds inside each
    return [
        ((-0.3, 0.3), [0.10, 0.20]),
        ((-0.6, 0.6), [0.15, 0.35, 0.55]),
        ((-1.0, 1.0), [0.25, 0.50, 0.85]),
        ((-2.0, 2.0), [0.40, 0.80, 1.20]),
    ]


# ================= Updated fitter with bound-repair ===========================


def fit_one_nv_with_freq_sweeps(
    times_us,
    y,
    yerr,
    amp_bound_grid=((-1, 1),),  # start narrow; add (-3,3) in a retry if needed
    *,
    # Frequency handling
    freq_bound_boxes=None,  # e.g. {"osc_f0": (0.04, 0.36), "osc_f1": (0.0, 0.36)}
    freq_seed_band=None,  # (fmin, fmax) in cycles/µs; if None -> inferred
    freq_seed_n_peaks=2,  # keep small for speed
    seed_include_harmonics=False,  # enable only if residuals remain high
    # Extra multi-starts (any param in your model)
    extra_overrides_grid=None,  # e.g. {"osc_phi0":[0, np.pi/2], "osc_phi1":[0]}
    # Model choice
    use_fixed_revival=False,
    enable_extras=True,
    fixed_rev_time=37.6,
    # Prior harmonic pass (fast pre-fit using simple (f0,f1) ratios)
    prior_enable=True,
    prior_max_pairs=24,  # small, fast
    prior_min_sep=0.01,
    early_stop_redchi=None,  # stop after prior if already excellent
    # Optimizer budgets (progressive)
    small_maxfev=40000,
    small_max_nfev=60000,
    big_maxfev=120000,
    big_max_nfev=180000,
    refine_target_red=1.05,  # if best > this, refine the winner with big budgets
    # Coarse screening
    coarse_K=8,  # keep top-K seeds per amp window
    coarse_max_nfev=200000,
    # Data cleaning
    err_floor=1e-3,
    # Logging
    verbose=True,
):
    """
    Robust & fast single-NV fitter with:
      • FFT-derived frequency seeds (band-limited)
      • Optional harmonic prior pass (quick (f0,f1) trials)
      • Coarse→fine screening: keep only top-K survivors for heavy fits
      • Early stop if a great χ² is reached
      • Progressive budgets: cheap pass first, refine the winner only if needed
      • NEW: bound-hit repair — widen only the wall-hitting params, reseed, refit

    Returns
    -------
    popt_best, pcov_best, redchi_best, fn_best, amp_window_best, overrides_used

    Requires (already in your code base):
      _sanitize_trace, _initial_guess_and_bounds, _core_len_for_fn,
      _normalize_freq_band, _sanitize_bound_boxes, _seed_freqs_fft,
      _clone_vecs, _set_osc_amp_bounds, _apply_param_overrides, _grid_product,
      _fit_curve_fit, _fit_least_squares, _pick_best, _chi2_red,
      fine_decay, fine_decay_fixed_revival, harmonic_candidates
    """

    # ---------- local cheap coarse scorer (core-only LSQ) ----------
    def _coarse_redchi(fit_fn, t, yy, ee, p0, lb, ub, kcore):
        try:
            popt, _, _ = _fit_least_squares(
                fit_fn,
                t,
                yy,
                ee,
                p0[:kcore],
                lb[:kcore],
                ub[:kcore],
                max_nfev=coarse_max_nfev,
            )
            yfit = fit_fn(t, *popt)
            return float(_chi2_red(yy, ee, yfit, len(popt)))
        except Exception:
            return float("inf")

    # -----------------------------
    # 1) Sanitize input trace
    # -----------------------------
    t, yy, ee = _sanitize_trace(times_us, y, yerr, err_floor=err_floor)
    if len(t) < 8:
        raise RuntimeError("Too few valid points after sanitization.")

    # -----------------------------
    # 2) Base model & initial bounds
    # -----------------------------
    fit_fn_base = fine_decay if not use_fixed_revival else fine_decay_fixed_revival
    p0_base, lb_base, ub_base = _initial_guess_and_bounds(
        t,
        yy,
        enable_extras,
        fixed_rev_time=(None if fit_fn_base is fine_decay else fixed_rev_time),
    )
    pmap = _param_index_map(fit_fn_base)

    def _try_anti_cap(popt, lb, ub, used_fn, overrides, *, tol_rel=0.10):
        pmap = _param_index_map(used_fn)
        iT2 = pmap.get("T2_ms", None)
        if iT2 is None:
            return (popt, None, None)

        # is T2 near upper wall?
        if not (
            np.isfinite(ub[iT2])
            and abs(ub[iT2] - popt[iT2]) <= 0.01 * max(1e-12, ub[iT2] - lb[iT2])
        ):
            return (popt, None, None)

        best = (np.inf, popt, None)  # (redchi, popt, pcov)
        for (amin, amax), a_seeds in [
            ((-0.3, 0.3), [0.1, 0.2]),
            ((-0.6, 0.6), [0.15, 0.35]),
        ]:
            base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
            _set_osc_amp_bounds(base_lb, base_ub, used_fn, amin, amax)
            for a0 in a_seeds:
                ov2 = dict(overrides)
                ov2["osc_amp"] = np.clip(a0, amin, amax)
                p0_try, lb_try, ub_try = _apply_param_overrides(
                    base_p0, base_lb, base_ub, used_fn, ov2, None
                )
                try:
                    popt2, pcov2, red2 = _fit_least_squares(
                        used_fn,
                        t,
                        yy,
                        ee,
                        p0_try,
                        lb_try,
                        ub_try,
                        max_nfev=small_max_nfev // 2,
                    )
                except Exception:
                    continue
                score = _score_tuple(popt2, red2, lb_try, ub_try, pmap)
                if score[0] < best[0]:
                    best = (score[0], popt2, pcov2)

        # accept if χ² increased only mildly
        _, popt_new, pcov_new = best
        if popt_new is None:
            return (popt, None, None)

        yfit_old = used_fn(t, *popt)
        red_old = _chi2_red(yy, ee, yfit_old, len(popt))
        yfit_new = used_fn(t, *popt_new)
        red_new = _chi2_red(yy, ee, yfit_new, len(popt_new))

        if (red_new - red_old) <= tol_rel * max(1e-9, red_old):
            if verbose:
                print(
                    "[ANTI-CAP] Accepted alt solution: similar χ², \(T_2\) off the wall"
                )
            return (popt_new, pcov_new, red_new)
        return (popt, None, None)

    # Core-only vectors (for FFT detrending / quick trials)
    if enable_extras:
        kcore = _core_len_for_fn(fit_fn_base)
        p0_core, lb_core, ub_core = p0_base[:kcore], lb_base[:kcore], ub_base[:kcore]
    else:
        kcore = len(p0_base)
        p0_core, lb_core, ub_core = p0_base, lb_base, ub_base

    # -----------------------------
    # 3) Frequency band + bound boxes
    # -----------------------------
    band = _normalize_freq_band(t, freq_seed_band, fmax_cap=None)  # (lo, hi)
    lo, hi = band

    bound_boxes = {}
    if enable_extras and "osc_f0" in pmap:
        bound_boxes["osc_f0"] = (max(0.0, lo * 0.8), hi * 1.1)
    if enable_extras and "osc_f1" in pmap:
        bound_boxes["osc_f1"] = (0.0, hi * 1.1)

    if isinstance(freq_bound_boxes, dict):
        for k, v in freq_bound_boxes.items():
            if k in pmap and v is not None:
                bound_boxes[k] = tuple(v)

    bound_boxes = _sanitize_bound_boxes(bound_boxes, band)

    # -----------------------------
    # 4) FFT frequency seeds
    # -----------------------------
    fit_fn_core = fit_fn_base
    freq_seeds = _seed_freqs_fft(
        t,
        yy,
        ee,
        fit_fn_core,
        p0_core,
        lb_core,
        ub_core,
        band=band,
        n_peaks=freq_seed_n_peaks,
        include_harmonics=seed_include_harmonics,
    )
    if verbose:
        try:
            print(
                f"FFT freq seeds (cycles/µs): {np.round(freq_seeds, 6)} in band {np.round(band, 6)}"
            )
        except Exception:
            print(f"FFT freq seeds (cycles/µs): {freq_seeds} in band {band}")

    # ---------------------------------------------
    # 5) PRIOR (diversified harmonic/paired seeds)
    # ---------------------------------------------
    all_candidates = []
    if prior_enable and enable_extras and ("osc_f0" in pmap) and len(freq_seeds) > 0:
        # distinct seeds within band
        seeds = _uniq_in_band(freq_seeds, lo, hi)

        # build diversified overrides: singles (f1=0) + wide/close pairs + harmonics
        prior_ovrs = diversified_prior_overrides(
            seeds,
            (lo, hi),
            min_sep=prior_min_sep,
            nyq_guard=0.95,
            n_single=min(12, max(4, len(seeds))),
            n_pairs=prior_max_pairs,
            nbuckets=5,
            jitter=0.015,
            seed=1729,
        )

        base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
        _set_osc_amp_bounds(base_lb, base_ub, fit_fn_base, -1.0, 1.0)

        start_best = np.inf
        best_prior = None

        for ov in prior_ovrs:  # your diversified f0/f1 overrides
            f0 = float(ov.get("osc_f0", 0.0))
            f1 = float(ov.get("osc_f1", 0.0))
            if not (lo <= f0 <= hi):
                continue
            if not (0.0 <= f1 <= hi):
                continue

            for (amin, amax), a_seeds in _amp_windows_and_seeds():
                base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
                _set_osc_amp_bounds(base_lb, base_ub, fit_fn_base, amin, amax)

                for a0 in a_seeds:
                    ov2 = dict(ov)
                    ov2["osc_amp"] = np.clip(a0, amin, amax)

                    p0_try, lb_try, ub_try = _apply_param_overrides(
                        base_p0,
                        base_lb,
                        base_ub,
                        fit_fn_base,
                        overrides=ov2,
                        bound_boxes=None,
                    )

                    # quick LSQ (+extras preferred)
                    tried = False
                    try:
                        if enable_extras and len(p0_try) > kcore:
                            if verbose:
                                print(
                                    f"  [priors] lsq_extras f0={f0:.6g}, f1={f1:.6g}, A~{a0:.2f} @[{amin},{amax}]"
                                )
                            popt, pcov, red = _fit_least_squares(
                                fit_fn_base,
                                t,
                                yy,
                                ee,
                                p0_try,
                                lb_try,
                                ub_try,
                                max_nfev=small_max_nfev // 4,
                            )
                            tried = True
                        else:
                            raise RuntimeError
                    except Exception:
                        try:
                            if verbose:
                                print(
                                    f"  [priors] lsq_noextras f0={f0:.6g}, f1={f1:.6g}, A~{a0:.2f} @[{amin},{amax}]"
                                )
                            popt, pcov, red = _fit_least_squares(
                                fit_fn_base,
                                t,
                                yy,
                                ee,
                                p0_try[:kcore],
                                lb_try[:kcore],
                                ub_try[:kcore],
                                max_nfev=small_max_nfev // 4,
                            )
                        except Exception:
                            continue

                    # score with penalties (don’t accept T2 at wall)
                    sc = _score_tuple(
                        popt, red, lb_try, ub_try, _param_index_map(fit_fn_base)
                    )
                    if sc[0] < start_best:
                        start_best = sc[0]
                        best_prior = (
                            "lsq_prior",
                            (amin, amax),
                            ov2,
                            popt,
                            pcov,
                            red,
                            fit_fn_base,
                        )

        if best_prior is not None:
            # normalize tuple schema to match sweep candidates:
            name, ab, overrides, popt, pcov, red, used_fn = best_prior
            all_candidates.append((ab, overrides, name, popt, pcov, red, used_fn))
            if verbose:
                print(
                    f"[PRIOR-BEST] amp={ab}, mode={name}, redχ²={red:.4g}, overrides={overrides}"
                )

            if isinstance(early_stop_redchi, (int, float)) and red <= early_stop_redchi:
                if verbose:
                    print(f"[EARLY-STOP] Using prior candidate with redχ²={red:.4g}")
                return popt, pcov, red, used_fn, ab, overrides
    # 6) Seed grid for the full sweep (amp × …)
    # ---------------------------------------------------
    seed_grid = {}
    if enable_extras and len(freq_seeds) > 0:
        seed_grid["osc_f0"] = _as_float_list(freq_seeds)
        seed_grid["osc_f1"] = [0.0] + _as_float_list(
            freq_seeds
        )  # allow single-frequency

    if extra_overrides_grid:
        for k, v in extra_overrides_grid.items():
            seed_grid.setdefault(k, [])
            seed_grid[k].extend(list(v))

    # Keep only params that exist in this model
    seed_grid = {k: v for k, v in seed_grid.items() if k in pmap}

    # ---------------------------------------
    # 7) Full sweep over amp windows & seeds
    #    (coarse→fine survivor screening)
    # ---------------------------------------
    def _run_attempts_with_budget(p0_try, lb_try, ub_try, overrides, maxfev, max_nfev):
        # type-guard
        if overrides is not None and not isinstance(overrides, dict):
            raise TypeError(
                f"overrides must be dict or None, got {type(overrides)}: {overrides}"
            )

        attempts = []

        def _log(tag):
            if verbose:
                keys = (
                    ", ".join([f"{k}:{overrides[k]:.6g}" for k in overrides.keys()])
                    if overrides
                    else ""
                )
                print(f"  [overrides={{ {keys} }}] {tag}")

        # LSQ core-only
        try:
            _log("least_squares no-extras (soft_l1)")
            popt_LB, pcov_LB, red_LB = _fit_least_squares(
                fit_fn_base,
                t,
                yy,
                ee,
                p0_try[:kcore],
                lb_try[:kcore],
                ub_try[:kcore],
                max_nfev=max_nfev,
            )
            attempts.append(
                ("lsq_noextras", popt_LB, pcov_LB, red_LB, fit_fn_base, "ok")
            )
        except Exception as e:
            attempts.append(
                ("lsq_noextras", None, None, np.inf, fit_fn_base, f"fail: {e}")
            )

        # LSQ extras
        if enable_extras:
            try:
                _log("least_squares + extras (soft_l1)")
                popt_L, pcov_L, red_L = _fit_least_squares(
                    fit_fn_base, t, yy, ee, p0_try, lb_try, ub_try, max_nfev=max_nfev
                )
                attempts.append(
                    ("lsq_extras", popt_L, pcov_L, red_L, fit_fn_base, "ok")
                )
            except Exception as e:
                attempts.append(
                    ("lsq_extras", None, None, np.inf, fit_fn_base, f"fail: {e}")
                )

        # curve_fit core-only
        try:
            _log("curve_fit no-extras")
            popt_B, pcov_B, red_B = _fit_curve_fit(
                fit_fn_base,
                t,
                yy,
                ee,
                p0_try[:kcore],
                lb_try[:kcore],
                ub_try[:kcore],
                maxfev=maxfev,
            )
            attempts.append(
                ("curve_fit_noextras", popt_B, pcov_B, red_B, fit_fn_base, "ok")
            )
        except Exception as e:
            attempts.append(
                ("curve_fit_noextras", None, None, np.inf, fit_fn_base, f"fail: {e}")
            )

        # curve_fit extras
        if enable_extras:
            try:
                _log("curve_fit + extras")
                popt, pcov, red = _fit_curve_fit(
                    fit_fn_base, t, yy, ee, p0_try, lb_try, ub_try, maxfev=maxfev
                )
                attempts.append(
                    ("curve_fit_extras", popt, pcov, red, fit_fn_base, "ok")
                )
            except Exception as e:
                attempts.append(
                    ("curve_fit_extras", None, None, np.inf, fit_fn_base, f"fail: {e}")
                )

        # fixed-revival try if base is free-revival
        if fit_fn_base is fine_decay:
            try:
                _log("curve_fit fixed-revival")
                p0_C, lb_C, ub_C = _initial_guess_and_bounds(
                    t, yy, enable_extras=False, fixed_rev_time=fixed_rev_time
                )
                popt_C, pcov_C, red_C = _fit_curve_fit(
                    fine_decay_fixed_revival, t, yy, ee, p0_C, lb_C, ub_C, maxfev=maxfev
                )
                attempts.append(
                    (
                        "curve_fit_fixedrev",
                        popt_C,
                        pcov_C,
                        red_C,
                        fine_decay_fixed_revival,
                        "ok",
                    )
                )
            except Exception as e:
                attempts.append(
                    (
                        "curve_fit_fixedrev",
                        None,
                        None,
                        np.inf,
                        fine_decay_fixed_revival,
                        f"fail: {e}",
                    )
                )

        best_here = _pick_best(attempts)
        if best_here is None:
            return None
        mode, popt, pcov, red, used_fn, _note = best_here
        yfit = used_fn(times_us, *popt)
        red = _chi2_red(y, yerr, yfit, len(popt))
        return (mode, popt, pcov, red, used_fn)

    # main sweep
    for ab in amp_bound_grid:
        a_min, a_max = float(ab[0]), float(ab[1])

        base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
        _set_osc_amp_bounds(base_lb, base_ub, fit_fn_base, a_min, a_max)
        base_p0, base_lb, base_ub = _apply_param_overrides(
            base_p0,
            base_lb,
            base_ub,
            fit_fn_base,
            overrides=None,
            bound_boxes=bound_boxes,
        )

        # ---- coarse screening over all overrides for this amp window ----
        coarse_pool = []  # (coarse_red, (p0_try, lb_try, ub_try, overrides))
        for overrides in _grid_product(seed_grid):
            p0_try, lb_try, ub_try = _apply_param_overrides(
                base_p0,
                base_lb,
                base_ub,
                fit_fn_base,
                overrides=overrides,
                bound_boxes=None,
            )
            cscore = _coarse_redchi(
                fit_fn_base, t, yy, ee, p0_try, lb_try, ub_try, kcore
            )
            coarse_pool.append((cscore, (p0_try, lb_try, ub_try, overrides)))

        coarse_pool.sort(key=lambda x: x[0])
        survivors = [x[1] for x in coarse_pool[: max(1, int(coarse_K))]]

        # ---- run heavy attempts on survivors with SMALL budgets ----
        for p0_try, lb_try, ub_try, overrides in survivors:
            res = _run_attempts_with_budget(
                p0_try,
                lb_try,
                ub_try,
                overrides,
                maxfev=small_maxfev,
                max_nfev=small_max_nfev,
            )
            if res is None:
                continue
            mode, popt, pcov, red, used_fn = res
            all_candidates.append((ab, overrides, mode, popt, pcov, red, used_fn))

            if isinstance(early_stop_redchi, (int, float)) and red <= early_stop_redchi:
                if verbose:
                    print(f"[EARLY-STOP] Sweep reached redχ²={red:.3g}")
                return popt, pcov, red, used_fn, ab, overrides

    # -----------------------------
    # 8) Select best (and refine if needed)
    # -----------------------------
    if not all_candidates:
        raise RuntimeError("All frequency/amp sweep attempts failed.")

    # abest, overrides_best, mode_best, popt_best, pcov_best, red_best, fn_best = \
    #     min(all_candidates, key=lambda c: c[5])

    def _pick_best_with_penalty(cands, lb_ref, ub_ref, pmap_ref):
        # each c: (ab, overrides, mode, popt, pcov, red, used_fn)
        scored = []
        for c in cands:
            ab, ov, mode, popt, pcov, red, used_fn = c
            sc = _score_tuple(popt, red, lb_ref, ub_ref, pmap_ref)
            scored.append((sc, c))
        scored.sort(key=lambda z: z[0])  # lexicographic: (chi+pen, amp_tie)
        return scored[0][1]

    abest, overrides_best, mode_best, popt_best, pcov_best, red_best, fn_best = (
        _pick_best_with_penalty(
            all_candidates, lb_base, ub_base, _param_index_map(fit_fn_base)
        )
    )

    # after picking (abest, overrides_best, mode_best, popt_best, pcov_best, red_best, fn_best)
    popt_alt, pcov_alt, red_alt = _try_anti_cap(
        popt_best, lb_base, ub_base, fn_best, overrides_best
    )
    if popt_alt is not None and popt_alt is not popt_best:
        popt_best, pcov_best, red_best = (
            popt_alt,
            pcov_alt,
            red_alt if red_alt is not None else red_best,
        )

    if verbose:
        print(
            f"[BEST] amp={abest}, mode={mode_best}, redχ²={red_best:.4g}, overrides={overrides_best}"
        )

    # --- Early exit if already good enough ---
    if (
        np.isfinite(red_best)
        and (early_stop_redchi is not None)
        and (red_best <= early_stop_redchi)
    ):
        if verbose:
            print(
                f"[EARLY-STOP] Keeping prior/sweep best (redχ²={red_best:.4g} ≤ {early_stop_redchi})"
            )
        return popt_best, pcov_best, red_best, fn_best, abest, overrides_best

    if not np.isfinite(red_best):
        return popt_best, pcov_best, red_best, fn_best, abest, overrides_best

    if verbose:
        print("[REFINE] Re-running best candidate with bigger budgets")

    # Rebuild bounds around the winning amp window
    base_p0, base_lb, base_ub = _clone_vecs(p0_base, lb_base, ub_base)
    _set_osc_amp_bounds(base_lb, base_ub, fn_best, float(abest[0]), float(abest[1]))
    base_p0, base_lb, base_ub = _apply_param_overrides(
        base_p0,
        base_lb,
        base_ub,
        fn_best,
        overrides=None,
        bound_boxes=bound_boxes,
    )
    p0_try, lb_try, ub_try = _apply_param_overrides(
        base_p0,
        base_lb,
        base_ub,
        fn_best,
        overrides=overrides_best,
        bound_boxes=None,
    )

    # Warm start from previous best params
    p0_try = np.array(popt_best, dtype=float)

    # Optionally freeze f0,f1 during refine
    pidx = _param_index_map(fn_best)
    if "osc_f0" in pidx:
        lb_try[pidx["osc_f0"]] = ub_try[pidx["osc_f0"]] = p0_try[pidx["osc_f0"]]
    if "osc_f1" in pidx and overrides_best.get("osc_f1", 0.0) != 0.0:
        lb_try[pidx["osc_f1"]] = ub_try[pidx["osc_f1"]] = p0_try[pidx["osc_f1"]]

    # Run big budgets
    res = _run_attempts_with_budget(
        p0_try, lb_try, ub_try, overrides_best, maxfev=big_maxfev, max_nfev=big_max_nfev
    )
    if res is not None:
        mode_r, popt_r, pcov_r, red_r, fn_r = res
        if np.isfinite(red_r) and (red_r < red_best):
            if verbose:
                print(
                    f"[REFINE-BEST] mode={mode_r}, redχ²={red_r:.4g} < {red_best:.4g} (improved)"
                )
            popt_best, pcov_best, red_best, fn_best = popt_r, pcov_r, red_r, fn_r
        else:
            if verbose:
                print(f"[REFINE] did not improve (staying at redχ²={red_best:.4g})")

    # --- Bound-hit repair pass ------------------------------------------------
    pmap_win = _param_index_map(fn_best)
    hits = _bound_hits(popt_best, lb_try, ub_try, frac_tol=0.01, abs_tol=1e-10)
    # If osc_amp is pressed against window edge, escalate window and repair too
    if "osc_amp" in pmap_win:
        iA = pmap_win["osc_amp"]
        if iA in hits:
            # escalate amp window once (e.g., (-1,1) → (-3,3))
            a0, a1 = float(abest[0]), float(abest[1])
            if (a1 - a0) <= 2.01:  # was likely (-1,1)
                if verbose:
                    print("[REPAIR] Escalating osc_amp window to (-3,3)")
                abest = (-3.0, 3.0)

    if hits:
        if verbose:
            # make readable names of hit params
            inv = {j: i for i, j in pmap_win.items()}
            hit_names = [inv[i] for i in hits.keys()]
            print(
                f"[REPAIR] Params at bounds: {hit_names} → widening boxes & reseeding"
            )

        # 1) widen only those hit params
        lb_rep, ub_rep = _repair_bounds_for_hits(
            lb_try, ub_try, hits, pmap, freq_band=band
        )

        # 2) re-tie contrast to baseline
        p0_rep = np.array(popt_best, float)
        p0_rep, lb_rep, ub_rep = _retie_contrast_to_baseline(
            p0_rep, lb_rep, ub_rep, pmap_win, eps=0.01
        )

        # 3) reseed hit indices at the centers of widened boxes
        p0_rep = _reseed_to_center(p0_rep, lb_rep, ub_rep, only_idxs=list(hits.keys()))

        # 4) run heavy attempt
        res_rep = _run_attempts_with_budget(
            p0_rep,
            lb_rep,
            ub_rep,
            overrides_best,
            maxfev=big_maxfev,
            max_nfev=big_max_nfev,
        )
        if res_rep is not None:
            mode_r, popt_r, pcov_r, red_r, fn_r = res_rep
            if np.isfinite(red_r) and (red_r < red_best):
                if verbose:
                    print(
                        f"[REPAIR-BEST] mode={mode_r}, redχ²={red_r:.4g} < {red_best:.4g}"
                    )
                popt_best, pcov_best, red_best, fn_best = popt_r, pcov_r, red_r, fn_r
            else:
                if verbose:
                    print(f"[REPAIR] did not improve (keeping redχ²={red_best:.4g})")

    return popt_best, pcov_best, red_best, fn_best, abest, overrides_best


# === NEW: top-level runner that includes freq sweeps ==========================
def run_with_amp_and_freq_sweeps(
    nv_list,
    norm_counts,
    norm_counts_ste,
    times_us,
    nv_inds=None,
    amp_bound_grid=((-0.5, 0.5), (-1, 1), (-2, 2), (-3, 3), (-4, 4)),
    freq_bound_boxes=None,
    freq_seed_band=None,
    freq_seed_n_peaks=3,
    seed_include_harmonics=True,
    extra_overrides_grid=None,
    use_fixed_revival=False,
    enable_extras=True,
    fixed_rev_time=39.2,
    verbose=True,
):
    if nv_inds is None:
        nv_inds = list(range(len(nv_list)))
    popts, pcovs, chis, fit_fns, chosen_amp_bounds, chosen_overrides = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for lbl in nv_inds:
        print(f"Fitting (amp+freq sweeps) for NV {lbl}")
        try:
            pi, cov, chi, fn, ab, ov = fit_one_nv_with_freq_sweeps(
                times_us,
                norm_counts[lbl],
                norm_counts_ste[lbl],
                amp_bound_grid=amp_bound_grid,
                freq_bound_boxes=freq_bound_boxes,
                freq_seed_band=freq_seed_band,
                freq_seed_n_peaks=freq_seed_n_peaks,
                seed_include_harmonics=seed_include_harmonics,
                extra_overrides_grid=extra_overrides_grid,
                use_fixed_revival=use_fixed_revival,
                enable_extras=enable_extras,
                fixed_rev_time=fixed_rev_time,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[WARN] Fit failed for NV {lbl}: {e}")
            pi = cov = fn = None
            chi = np.nan
            ab = None
            ov = {}
        popts.append(pi)
        pcovs.append(cov)
        chis.append(chi)
        fit_fns.append(fn)
        chosen_amp_bounds.append(ab)
        chosen_overrides.append(ov)
    return popts, pcovs, chis, fit_fns, nv_inds


# ==========================================
#  helper: decide which NVs to keep based on T2_ms
# ==========================================


# --- helper: decide which NVs to keep based on T2_ms ---
def _t2_keep_mask(
    t2_ms,
    method="iqr",  # "iqr" | "mad" | "z" | None (combine with abs_range if you want)
    iqr_k=1.5,  # IQR multiplier (1.5 classic, 3.0 stricter)
    mad_k=3.5,  # MAD multiplier (≈3–4 is common)
    z_thresh=4.0,  # |z| threshold
    abs_range=None,  # e.g. (0.01, 50.0) ms   -> keep only inside this range
    finite_only=True,
):
    """
    Returns a boolean mask of same length as t2_ms where True = keep.
    Combines: finite filter, optional abs_range, and one robust method.
    """
    t2 = np.asarray(t2_ms, float)
    keep = np.ones_like(t2, dtype=bool)

    if finite_only:
        keep &= np.isfinite(t2)

    if abs_range is not None:
        lo, hi = abs_range
        keep &= (t2 >= lo) & (t2 <= hi)

    # Robust method (computed only on currently "keep" values)
    x = t2[keep]
    if x.size == 0:
        return keep  # nothing left anyway

    if method == "iqr":
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        keep_subset = (x >= lo) & (x <= hi)

    elif method == "mad":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + 1e-12
        # robust z ~ 0.6745*(x-med)/MAD   -> use |robust z| <= mad_k
        rz = 0.6745 * (x - med) / mad
        keep_subset = np.abs(rz) <= mad_k

    elif method == "z":
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-12
        z = (x - mu) / sd
        keep_subset = np.abs(z) <= z_thresh

    else:
        # No robust method; rely only on finite/abs_range
        return keep

    # write back into full mask at the positions that were "keep" so far
    idx = np.where(keep)[0]
    keep[idx] = keep_subset
    return keep


_UNIFIED_KEYS = [
    "baseline",
    "comb_contrast",
    "revival_time_us",
    "width0_us",
    "T2_ms",
    "T2_exp",
    "amp_taper_alpha",
    "width_slope",
    "revival_chirp",
    "osc_contrast",
    "osc_f0",
    "osc_f1",
    "osc_phi0",
    "osc_phi1",
]


def _normalize_popt_to_unified(p):
    q = np.full(14, np.nan, float)
    L = len(p)
    p = np.asarray(p, float)
    if L == 6:  # variable core, no extras
        q[0:6] = p[0:6]
    elif L == 5:  # fixed core, no extras
        q[0] = p[0]
        q[1] = p[1]
        q[2] = np.nan
        q[3] = p[2]
        q[4] = p[3]
        q[5] = p[4]
    elif L == 14:  # variable + extras
        q[0:6] = p[0:6]
        q[6:] = p[6:14]
    elif L == 13:  # fixed + extras
        q[0] = p[0]
        q[1] = p[1]
        q[2] = np.nan
        q[3] = p[2]
        q[4] = p[3]
        q[5] = p[4]
        q[6:] = p[5:13]
    else:
        if L >= 6:
            q[0:6] = p[0:6]
            if L > 6:
                m = min(8, L - 6)
                q[6 : 6 + m] = p[6 : 6 + m]
        elif L >= 5:
            q[0] = p[0]
            q[1] = p[1]
            q[2] = np.nan
            q[3] = p[2]
            q[4] = p[3]
            q[5] = p[4]
    return q


# ==========================================
# 2) Parameter panels with full-length mask
#    (T2-based outlier rejection)
# ==========================================


def plot_each_param_separately(
    popts,
    chi2_list,
    fit_nv_labels,
    save_prefix=None,
    include_trend=True,
    bins=30,
    t2_policy=dict(
        method="iqr",
        iqr_k=1.5,
        abs_range=None,
        mad_k=3.5,
        z_thresh=4.0,
        finite_only=True,
    ),
):
    valid = [
        (i, p, chi2_list[i] if i < len(chi2_list) else np.nan)
        for i, p in enumerate(popts)
        if p is not None
    ]
    if not valid:
        print("No successful fits.")
        return [], np.zeros(len(popts), bool), np.array([], int)

    uni_rows, x_labels, chi2_ok, positions = [], [], [], []
    for i, p, chi in valid:
        uni_rows.append(_normalize_popt_to_unified(p))
        x_labels.append(fit_nv_labels[i])  # <— use the provided labels!
        chi2_ok.append(chi)
        positions.append(i)

    arr = np.vstack(uni_rows)
    x_labels = np.asarray(x_labels)
    chi2_ok = np.asarray(chi2_ok, float)
    positions = np.asarray(positions, int)

    # T2 filter on the valid subset
    t2 = arr[:, 4]
    keep_valid = _t2_keep_mask(
        t2,
        method=t2_policy.get("method", "iqr"),
        iqr_k=t2_policy.get("iqr_k", 1.5),
        mad_k=t2_policy.get("mad_k", 3.5),
        z_thresh=t2_policy.get("z_thresh", 4.0),
        abs_range=t2_policy.get("abs_range", None),
        finite_only=t2_policy.get("finite_only", True),
    )

    full_mask = np.zeros(len(popts), dtype=bool)
    full_mask[positions] = keep_valid

    arr_f = arr[keep_valid]
    labels_f = x_labels[keep_valid]
    chi2_f = chi2_ok[keep_valid]

    def _one(vec, name, ylabel):
        fig, axes = plt.subplots(
            1, 2 if include_trend else 1, figsize=(10 if include_trend else 5, 4)
        )
        if include_trend:
            axh, axt = axes
        else:
            axh = axes
        axh.hist(vec[np.isfinite(vec)], bins=bins)
        axh.set_title(f"{name} histogram")
        axh.set_xlabel(ylabel)
        axh.set_ylabel("count")
        if include_trend:
            axt.plot(labels_f, vec, ".", ms=4)
            axt.set_title(f"{name} vs NV label")
            axt.set_xlabel("NV label")
            axt.set_ylabel(ylabel)
        if save_prefix:
            file_path = dm.get_file_path(
                __file__, timestamp, f"{save_prefix}_{name}.png"
            )
            dm.save_figure(fig, file_path)
        return fig

    figs = []
    units = [
        "arb.",
        "arb.",
        "µs",
        "µs",
        "ms",
        "–",
        "–",
        "– per revival",
        "fraction",
        "arb.",
        "1/µs",
        "1/µs",
        "rad",
        "rad",
    ]
    for col, (name, unit) in enumerate(zip(_UNIFIED_KEYS, units)):
        figs.append((name, _one(arr_f[:, col], name, unit)))

    fig_chi, axes = plt.subplots(
        1, 2 if include_trend else 1, figsize=(10 if include_trend else 5, 4)
    )
    if include_trend:
        axh, axt = axes
    else:
        axh = axes
    axh.hist(chi2_f[np.isfinite(chi2_f)], bins=bins)
    axh.set_title("reduced χ² histogram")
    axh.set_xlabel("χ²_red")
    axh.set_ylabel("count")
    if include_trend:
        axt.plot(labels_f, chi2_f, ".", ms=4)
        axt.set_title("reduced χ² vs NV label")
        axt.set_xlabel("NV label")
        axt.set_ylabel("χ²_red")
    fig_chi.tight_layout()
    if save_prefix:
        fig_chi.savefig(f"{save_prefix}-chi2_red.png", dpi=220)
        file_path = dm.get_file_path(__file__, timestamp, f"{save_prefix}-chi2_red.png")
        dm.save_figure(fig_chi, file_path)

    figs.append(("chi2_red", fig_chi))

    kept_labels = labels_f.astype(int)
    return figs, full_mask, kept_labels


# ==========================================
# 3) Individual NV fit plots
#    - keep passed nv_inds
#    - safely handle shorter popts (fixed-revival)
# ==========================================


def _coerce_to_core6(p, default_rev=39.2):
    p = np.asarray(p, float)
    if (
        len(p) == 5
    ):  # fixed-revival core -> inject revival_time for plotting with fine_decay
        b, cc, w0, t2, exp = p
        return np.array([b, cc, default_rev, w0, t2, exp], float)
    return p


def _safe_call_fit_fn(fit_fn, t, p, default_rev=39.2):
    try:
        return fit_fn(t, *p)
    except TypeError:
        return fit_fn(t, *_coerce_to_core6(p, default_rev=default_rev))


# --- helpers you already have ---
def _coerce_to_core6(p, default_rev=39.2):
    p = np.asarray(p, float)
    if (
        len(p) == 5
    ):  # fixed-revival core -> inject revival_time for plotting with fine_decay
        b, cc, w0, t2, exp = p
        return np.array([b, cc, default_rev, w0, t2, exp], float)
    return p


def _safe_call_fit_fn(fit_fn, t, p, default_rev=39.2):
    try:
        return fit_fn(t, *p)
    except TypeError:
        return fit_fn(t, *_coerce_to_core6(p, default_rev=default_rev))


# --- NEW: map p -> dict using the fit function signature (best effort) ---
def _params_to_dict(fit_fn, p, default_rev=39.2):
    """
    Turn a parameter vector 'p' into a name->value dict using the fit_fn signature.
    If lengths don't match (e.g., coercion paths), fall back to a sensible mapping.
    """
    p = np.asarray(p, float).tolist()
    try:
        sig = inspect.signature(fit_fn)
        names = [k for k in sig.parameters.keys()][1:]  # drop 'tau'
    except Exception:
        names = []

    d = {}
    if names and len(p) <= len(names):
        for name, val in zip(names, p):
            d[name] = float(val)
        # In case of fixed-revival core (no 'revival_time'):
        if ("revival_time" not in d) and ("width0_us" in d) and ("T2_ms" in d):
            d["revival_time"] = float(default_rev)
        # Back-compat: normalize osc names
        if "osc_contrast" in d and "osc_amp" not in d:
            d["osc_amp"] = d["osc_contrast"]
    else:
        # Fallback by length heuristics
        # Core-6: [baseline, comb_contrast, revival_time, width0_us, T2_ms, T2_exp]
        if len(p) >= 6:
            d.update(
                dict(
                    baseline=p[0],
                    comb_contrast=p[1],
                    revival_time=p[2],
                    width0_us=p[3],
                    T2_ms=p[4],
                    T2_exp=p[5],
                )
            )
        elif len(p) == 5:
            d.update(
                dict(
                    baseline=p[0],
                    comb_contrast=p[1],
                    revival_time=default_rev,
                    width0_us=p[2],
                    T2_ms=p[3],
                    T2_exp=p[4],
                )
            )
        # Try to place extras in a common order if present beyond core-6
        # [amp_taper_alpha, width_slope, revival_chirp, osc_amp (or contrast),
        #  osc_f0, osc_f1, osc_phi0, osc_phi1, mu0_us]
        extras = p[6:] if len(p) > 6 else []
        keys_extras = [
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_amp",
            "osc_f0",
            "osc_f1",
            "osc_phi0",
            "osc_phi1",
        ]
        for k, v in zip(keys_extras, extras):
            d[k] = float(v)

    # Final tidy: ensure consistent fields exist (even if missing)
    for k in [
        "baseline",
        "comb_contrast",
        "revival_time",
        "width0_us",
        "T2_ms",
        "T2_exp",
        "amp_taper_alpha",
        "width_slope",
        "revival_chirp",
        "osc_amp",
        "osc_f0",
        "osc_f1",
        "osc_phi0",
        "osc_phi1",
    ]:
        d.setdefault(k, None)
    return d


def _echo_summary_lines(t_us, y):
    if len(y) == 0:
        return []
    arr = np.asarray(y, float)
    n = max(3, len(arr) // 6)
    early = float(np.nanmean(arr[:n]))
    late = float(np.nanmean(arr[-n:]))
    return [
        f"range: {arr.min():.3f}…{arr.max():.3f}",
        f"⟨early⟩→⟨late⟩: {early:.3f}→{late:.3f}",
    ]


def _format_param_box(pdct):
    """Make a compact, readable box for the most relevant parameters."""

    def fmt(v, nd=3):
        return "—" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))

    lines = []
    lines.append(
        f"baseline: {fmt(pdct['baseline'])}, comb_contrast: {fmt(pdct['comb_contrast'])}"
    )
    lines.append(
        f"Trev (μs): {fmt(pdct['revival_time'])}, rev_width (μs): {fmt(pdct['width0_us'])}"
    )
    lines.append(f"T2 (ms): {fmt(pdct['T2_ms'])}, T2_exp (n): {fmt(pdct['T2_exp'])}")
    # Oscillation terms (show only if present / non-zero)
    if (pdct.get("osc_amp") is not None) and (abs(pdct.get("osc_amp", 0.0)) > 1e-6):
        lines.append(f"osc_amp: {fmt(pdct['osc_amp'])}")
        if pdct.get("osc_f0", None) is not None:
            lines.append(
                f"f0 (cyc/μs): {fmt(pdct['osc_f0'])}, f1 (cyc/μs): {fmt(pdct['osc_f1'])}"
            )
        if pdct.get("osc_phi0", None) is not None:
            lines.append(
                f"φ0 (rad): {fmt(pdct['osc_phi0'])}, φ1 (rad): {fmt(pdct['osc_phi1'])}"
            )
    # Comb shaping
    if any(
        pdct.get(k, None) not in (None, 0.0)
        for k in ("amp_taper_alpha", "width_slope", "revival_chirp")
    ):
        lines.append(
            f"α: {fmt(pdct['amp_taper_alpha'])}, slope: {fmt(pdct['width_slope'])}, chirp: {fmt(pdct['revival_chirp'])}"
        )
    return lines


# --- UPDATED: now annotates each subplot with a fit-parameter box (and optional χ²_red) ---
def plot_individual_fits(
    norm_counts,
    norm_counts_ste,
    total_evolution_times,
    popts,
    nv_inds,  # labels same order as popts
    fit_fn_per_nv,  # per-NV fit function
    keep_mask=None,
    show_residuals=True,
    n_fit_points=1000,
    save_prefix=None,
    block=False,
    default_rev_for_plot=39.2,
    red_chi2_list=None,  # OPTIONAL: pass list of reduced-χ² (same order as popts)
    show_param_box=True,  # toggle the on-plot parameter box
):
    N = len(popts)
    assert len(nv_inds) == N, "nv_inds must be same length/order as popts"
    t_all = np.asarray(total_evolution_times, float)

    positions = np.arange(N)
    if keep_mask is not None:
        positions = positions[np.asarray(keep_mask, bool)]

    figs = []
    for pos in positions:
        lbl = nv_inds[pos]
        p = popts[pos]
        if p is None:
            continue

        fit_fn = fit_fn_per_nv[pos] or fine_decay

        y = np.asarray(norm_counts[lbl], float)
        e = np.asarray(norm_counts_ste[lbl], float)

        if show_residuals:
            fig, (ax, axr) = plt.subplots(
                2,
                1,
                figsize=(7, 6),
                sharex=True,
                gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06),
            )
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.6))

        # data
        ax.errorbar(t_all, y, yerr=e, fmt="o", ms=3.5, lw=0.8, alpha=0.9, capsize=2)
        ax.set_ylabel("Normalized NV$^{-}$ population")

        # model curve (dense grid)
        t_fit = np.linspace(np.nanmin(t_all), np.nanmax(t_all), n_fit_points)
        y_fit = _safe_call_fit_fn(fit_fn, t_fit, p, default_rev=default_rev_for_plot)
        ax.plot(t_fit, y_fit, "-", lw=2)

        ax.set_title(f"NV {lbl}")
        ymin = min(np.nanmin(y) - 0.1, -0.1)
        ymax = max(np.nanmax(y) + 0.1, 1.2)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)

        # residuals
        if show_residuals:
            y_model = _safe_call_fit_fn(
                fit_fn, t_all, p, default_rev=default_rev_for_plot
            )
            res = y - y_model
            axr.axhline(0.0, ls="--", lw=1.0)
            axr.plot(t_all, res, ".", ms=3.5)
            axr.set_xlabel("Total evolution time (µs)")
            axr.set_ylabel("res.")
            axr.grid(True, alpha=0.25)
        else:
            ax.set_xlabel("Total evolution time (µs)")

        # --- NEW: add a fit-parameter box + echo summary + optional χ²_red ---
        if show_param_box:
            pdict = _params_to_dict(fit_fn, p, default_rev=default_rev_for_plot)
            box_lines = _format_param_box(pdict)
            # top-right: parameter box
            ax.text(
                0.99,
                0.98,
                "\n".join(box_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6
                ),
            )
            # top-left: quick data summary + χ²_red (if provided)
            # left_lines = _echo_summary_lines(t_all, y)
            left_lines = []
            if red_chi2_list is not None and np.isfinite(red_chi2_list[pos]):
                left_lines.append(f"χ²_red: {red_chi2_list[pos]:.3g}")
            if left_lines:
                ax.text(
                    0.01,
                    0.98,
                    "\n".join(left_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6
                    ),
                )

        # optional save (uses your dm/timestamp if present)
        if save_prefix:
            try:
                file_path = dm.get_file_path(
                    __file__, timestamp, f"{save_prefix}-nv{int(lbl):03d}"
                )
                dm.save_figure(fig, file_path, f"nv{int(lbl):03d}")
            except Exception:
                pass

        figs.append((lbl, fig))

    if figs:
        plt.show(block=block)
    return figs


# =============================================================================
# CLI / Example
# =============================================================================
# ---- helpers ----
def _index_map(unified_keys):
    return {k: i for i, k in enumerate(unified_keys)}


def _safe_sigma(pcov, idx):
    try:
        if pcov is None:
            return np.nan
        pcov = np.asarray(pcov, float)
        if idx is None or idx >= pcov.shape[0]:
            return np.nan
        v = pcov[idx, idx]
        return np.sqrt(v) if (np.isfinite(v) and v >= 0) else np.nan
    except Exception:
        return np.nan


def extract_T2_freqs_and_errors(fit_dict, *, pick_freq="max", chi2_fail_thresh=None):
    """
    Returns
    -------
    nv_labels, T2_us, f0_kHz, f1_kHz, A_pick_kHz, chis, fit_fail,
    sT2_us, sf0_kHz, sf1_kHz, sA_pick_kHz
    """
    keys = fit_dict["unified_keys"]
    kmap = _index_map(keys)
    popts = fit_dict["popts"]
    pcovs = fit_dict.get("pcovs", [None] * len(popts))
    chis = np.array(fit_dict.get("red_chi2", [np.nan] * len(popts)), float)
    nvlbl = np.asarray(fit_dict["nv_labels"], int)

    idx_T2 = kmap.get("T2_ms", None)
    idx_f0 = kmap.get("osc_f0", None)
    idx_f1 = kmap.get("osc_f1", None)

    N = len(popts)
    T2_us = np.full(N, np.nan)
    f0_kHz = np.full(N, np.nan)
    f1_kHz = np.full(N, np.nan)
    A_pick_kHz = np.full(N, np.nan)
    sT2_us = np.full(N, np.nan)
    sf0_kHz = np.full(N, np.nan)
    sf1_kHz = np.full(N, np.nan)
    sA_pick_kHz = np.full(N, np.nan)
    fit_fail = np.zeros(N, bool)

    for i, (p, C) in enumerate(zip(popts, pcovs)):
        if not isinstance(p, (list, tuple)):
            fit_fail[i] = True
            continue

        # chi2 filter (optional)
        if chi2_fail_thresh is not None:
            try:
                if float(chis[i]) > float(chi2_fail_thresh):
                    fit_fail[i] = True
            except Exception:
                pass

        # T2 (ms -> µs) + sigma
        if idx_T2 is not None and idx_T2 < len(p):
            try:
                T2_us[i] = float(p[idx_T2]) * 1000.0
                sT2_ms = _safe_sigma(C, idx_T2)
                sT2_us[i] = (sT2_ms * 1000.0) if np.isfinite(sT2_ms) else np.nan
            except Exception:
                pass

        # f0, f1 (cycles/µs = MHz) -> kHz + sigmas
        cand = []
        tags = []  # keep which index produced which
        if idx_f0 is not None and idx_f0 < len(p):
            try:
                f0 = float(p[idx_f0])
                if np.isfinite(f0) and f0 > 0:
                    f0_kHz[i] = 1000.0 * f0
                    s0 = _safe_sigma(C, idx_f0)
                    sf0_kHz[i] = (1000.0 * s0) if np.isfinite(s0) else np.nan
                    cand.append(f0)
                    tags.append("f0")
            except Exception:
                pass
        if idx_f1 is not None and idx_f1 < len(p):
            try:
                f1 = float(p[idx_f1])
                if np.isfinite(f1) and f1 > 0:
                    f1_kHz[i] = 1000.0 * f1
                    s1 = _safe_sigma(C, idx_f1)
                    sf1_kHz[i] = (1000.0 * s1) if np.isfinite(s1) else np.nan
                    cand.append(f1)
                    tags.append("f1")
            except Exception:
                pass

        if cand:
            if pick_freq == "min":
                j = int(np.argmin(cand))
            elif pick_freq == "nonzero_first":
                j = 0
            else:  # "max"
                j = int(np.argmax(cand))
            f_pick = cand[j]
            tag = tags[j]
            A_pick_kHz[i] = 1000.0 * f_pick
            if tag == "f0":
                sA_pick_kHz[i] = sf0_kHz[i]
            else:
                sA_pick_kHz[i] = sf1_kHz[i]

    return (
        nvlbl,
        T2_us,
        f0_kHz,
        f1_kHz,
        A_pick_kHz,
        chis,
        fit_fail,
        sT2_us,
        sf0_kHz,
        sf1_kHz,
        sA_pick_kHz,
    )


# --- keep your existing extract_T2_freqs_and_errors(...) ---


def _mask_huge_errors(values, sigmas, *, rel_cap=None, pct_cap=None):
    """
    Returns a copy of 'sigmas' with too-large bars set to NaN (so matplotlib won't draw them).
      rel_cap:   max allowed sigma/value (e.g., 0.75). If value<=0 or NaN, rel test is skipped.
      pct_cap:   clip absolute sigma above this percentile to NaN (e.g., 95).
    """
    if sigmas is None:
        return None
    v = np.asarray(values, float)
    s = np.asarray(sigmas, float).copy()

    # relative cap
    if rel_cap is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = s / v
        bad_rel = ~np.isfinite(rel) | (rel > float(rel_cap))
        s[bad_rel] = np.nan

    # percentile cap (absolute)
    if pct_cap is not None:
        finite = np.isfinite(s) & (s > 0)
        if np.any(finite):
            thresh = np.percentile(s[finite], float(pct_cap))
            s[(s > thresh)] = np.nan

    return s


def plot_sorted_panels_with_err(
    nv_labels,
    T2_us,
    sT2_us,
    A_pick_kHz,
    sA_pick_kHz,
    *,
    mask_no_decay=None,
    mask_fit_fail=None,
    title_prefix="Spin-Echo",
    t2_units="µs",
    # error-bar pruning controls (tune as you like)
    t2_rel_cap=1.0,  # hide T2 bars with σ > 100% of value
    t2_pct_cap=99,  # and hide top 5% largest absolute T2 sigmas
    A_rel_cap=0.75,  # hide A bars with σ > 75% of value
    A_pct_cap=99,  # and hide top 5% largest absolute A sigmas
):
    N = len(nv_labels)
    mask_no_decay = np.zeros(N, bool) if mask_no_decay is None else mask_no_decay
    mask_fit_fail = np.zeros(N, bool) if mask_fit_fail is None else mask_fit_fail

    # ---- (a) T2 sorted with pruned error bars ----
    valid_t2 = np.isfinite(T2_us) & (~mask_no_decay) & (~mask_fit_fail)
    if np.any(valid_t2):
        idx = np.where(valid_t2)[0]
        order = idx[np.argsort(T2_us[idx])]
        x = np.arange(1, order.size + 1)

        if t2_units.lower().startswith("ms"):
            y = T2_us[order] / 1000.0
            yerr_raw = (sT2_us[order] / 1000.0) if sT2_us is not None else None
        else:
            y = T2_us[order]
            yerr_raw = sT2_us[order] if sT2_us is not None else None

        yerr = _mask_huge_errors(y, yerr_raw, rel_cap=t2_rel_cap, pct_cap=t2_pct_cap)

        plt.figure(figsize=(10, 5))
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            ms=3,
            lw=0.8,
            capsize=2,
            elinewidth=0.8,
            alpha=0.95,
        )
        plt.grid(alpha=0.3)
        plt.xlabel("NV index (sorted)")
        plt.ylabel(
            r"$T_2$ (" + ("ms" if t2_units.lower().startswith("ms") else "µs") + ")"
        )
        plt.yscale("log")
        plt.title(f"{title_prefix}: $T_2$ (sorted)")
        note = f"Excluded: no-decay={mask_no_decay.sum()}, fit-fail={mask_fit_fail.sum()}; Used={order.size}/{N}"
        plt.text(
            0.01,
            0.98,
            note,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    else:
        print("[plot] No valid T2 to plot.")

    # ---- (b) Ahfs sorted with pruned error bars ----
    valid_A = np.isfinite(A_pick_kHz) & (A_pick_kHz > 0)
    if np.any(valid_A):
        idx = np.where(valid_A)[0]
        order = idx[np.argsort(A_pick_kHz[idx])]
        x = np.arange(1, order.size + 1)
        y = A_pick_kHz[order]
        yerr_raw = sA_pick_kHz[order] if sA_pick_kHz is not None else None

        yerr = _mask_huge_errors(y, yerr_raw, rel_cap=A_rel_cap, pct_cap=A_pct_cap)

        plt.figure(figsize=(10, 5))
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            ms=3,
            lw=0.8,
            capsize=2,
            elinewidth=0.8,
            alpha=0.95,
            label="Spin-echo derived (picked)",
        )
        plt.grid(alpha=0.3)
        plt.xlabel("NV index (sorted)")
        plt.ylabel(r"$A_{\mathrm{hfs}}$ (kHz)")
        plt.title(f"{title_prefix}: $A_{{\\rm hfs}}$ (sorted)")
        plt.yscale("log")
        note = f"Excluded here (no valid freq): {(~valid_A).sum()}"
        plt.text(
            0.01,
            0.98,
            note,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    else:
        print("[plot] No hyperfine points to plot.")


def build_nv_orientation_map(table=None):
    nv_to_ori = {}
    for ori, idx_list in table.items():
        for nv_idx in idx_list:
            if nv_idx in nv_to_ori:
                raise ValueError(f"NV {nv_idx} appears in multiple orientations!")
            nv_to_ori[nv_idx] = ori
    return nv_to_ori


if __name__ == "__main__":
    kpl.init_kplotlib()
    # --- Load your data------------------------------------
    file_stems = [
        "2025_10_10-11_29_40-rubin-nv0_2025_09_08",
        "2025_10_10-08_55_59-rubin-nv0_2025_09_08",
        "2025_10_10-06_28_12-rubin-nv0_2025_09_08",
        "2025_10_10-03_59_48-rubin-nv0_2025_09_08",
        "2025_10_10-01_31_59-rubin-nv0_2025_09_08",
        "2025_10_09-23_03_41-rubin-nv0_2025_09_08",
        "2025_10_10-14_23_58-rubin-nv0_2025_09_08",
        "2025_10_10-17_04_27-rubin-nv0_2025_09_08",
    ]

    # file_stems = ["2025_10_29-10_33_01-johnson-nv0_2025_10_21",
    #             "2025_10_29-02_21_07-johnson-nv0_2025_10_21",
    #             ]

    # --- Magnetic field (crystal axes) ---
    B_G = [-46.27557688 - 17.16599864 - 5.70139829]
    B_G_mag = 49.685072884712
    B_hat = [-0.93137786 - 0.34549609 - 0.11475073]
    ###204NVs
    file_stems = [
        "2025_10_31-23_53_21-johnson-nv0_2025_10_21",
        "2025_10_31-15_40_56-johnson-nv0_2025_10_21",
        "2025_10_31-07_42_45-johnson-nv0_2025_10_21",
    ]

    ###204NVs dataset 2
    file_stems_1 = [
        "2025_11_03-01_47_09-johnson-nv0_2025_10_21",
        "2025_11_02-14_49_57-johnson-nv0_2025_10_21",
        "2025_11_02-04_46_56-johnson-nv0_2025_10_21",
    ]
    ###204NVs
    file_stems_2 = [
        "2025_11_11-23_52_50-johnson-nv0_2025_10_21",
        "2025_11_11-14_53_37-johnson-nv0_2025_10_21",
        "2025_11_11-06_02_04-johnson-nv0_2025_10_21",
        "2025_11_10-20_58_00-johnson-nv0_2025_10_21",
        "2025_11_10-11_36_39-johnson-nv0_2025_10_21",
        "2025_11_10-03_06_14-johnson-nv0_2025_10_21",
    ]

    file_stems = file_stems_1 + file_stems_2

    ###

    ### New B field:
    # --- Magnetic field (crystal axes) ---
    B_G = [-31.61263115 - 56.58135644 - 6.5512002]
    B_G = 65.143891267575
    B_G = [-0.48527391 - 0.86855967 - 0.10056507]
    # --- Quartet frequency shifts per NV axis ---

    axes = np.array(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ],
        dtype=int,
    )

    # ref / new in GHz, Δf in MHz
    f_ref_GHz = np.array([2.766625, 2.840625, 2.822175, 2.785075])
    f_new_GHz = np.array([2.725200, 2.827500, 2.848700, 2.746400])
    delta_f_MHz = np.array([-41.425, -13.125, +26.525, -38.675])

    quartet_freq_shifts = {
        "axes": axes,  # (4, 3) int
        "f_ref_GHz": f_ref_GHz,  # (4,)
        "f_new_GHz": f_new_GHz,  # (4,)
        "delta_f_MHz": delta_f_MHz,  # (4,)
    }
    file_stems = [
        "2025_11_25-21_57_00-johnson-nv0_2025_10_21",
        "2025_11_25-13_19_23-johnson-nv0_2025_10_21",
        "2025_11_25-04_44_08-johnson-nv0_2025_10_21",
        "2025_11_24-20_06_54-johnson-nv0_2025_10_21",
        "2025_11_24-11_36_56-johnson-nv0_2025_10_21",
        "2025_11_24-03_03_41-johnson-nv0_2025_10_21",
        ### New Sets
        "2025_12_08-19_42_33-johnson-nv0_2025_10_21",
        "2025_12_09-04_14_28-johnson-nv0_2025_10_21",
        "2025_12_09-12_53_18-johnson-nv0_2025_10_21",
        "2025_12_09-21_31_09-johnson-nv0_2025_10_21",
    ]

    # ######## Johnson B 59
    # B_G = [-41.57848995 - 32.77145194 - 27.5799348]
    # B_G = 59.694151242944
    # B_G = [-0.69652536 - 0.54898933 - 0.46202072]
    # file_stems = [
    #     "2025_12_04-19_43_15-johnson-nv0_2025_10_21",
    #     "2025_12_04-11_08_28-johnson-nv0_2025_10_21",
    #     "2025_12_04-02_39_13-johnson-nv0_2025_10_21",
    #     "2025_12_03-17_39_36-johnson-nv0_2025_10_21",
    #     "2025_12_03-08_56_17-johnson-nv0_2025_10_21",
    #     "2025_12_02-23_42_45-johnson-nv0_2025_10_21",
    #     # "2025_12_02-02_01_34-johnson-nv0_2025_10_21",
    #     # "2025_12_01-17_26_42-johnson-nv0_2025_10_21",
    #     # "2025_12_01-08_53_38-johnson-nv0_2025_10_21",
    #     "2025_12_01-00_24_57-johnson-nv0_2025_10_21",
    #     "2025_11_30-07_21_26-johnson-nv0_2025_10_21",
    #     "2025_11_30-15_56_42-johnson-nv0_2025_10_21",
    # ]
    ######## Johnson B 62G
    # B_G = [-48.67047318, -32.07615947, 22.49657427]
    # B_G_mag = 62.480323463718
    # B_hat = [-0.77897281, -0.51338018, 0.36005854]
    # file_stems = [
    #     "2025_12_22-17_20_47-johnson-nv0_2025_10_21",
    #     "2025_12_22-08_45_30-johnson-nv0_2025_10_21",
    #     "2025_12_22-00_08_14-johnson-nv0_2025_10_21",
    #     "2025_12_21-15_34_13-johnson-nv0_2025_10_21",
    #     "2025_12_21-07_05_27-johnson-nv0_2025_10_21",
    #     "2025_12_20-22_36_09-johnson-nv0_2025_10_21",
    #     ### New Sets
    #     "2025_12_31-21_21_36-johnson-nv0_2025_10_21",
    #     "2025_12_31-12_41_36-johnson-nv0_2025_10_21",
    #     "2025_12_31-03_59_43-johnson-nv0_2025_10_21",
    #     "2025_12_30-19_22_41-johnson-nv0_2025_10_21",
    #     "2025_12_30-10_47_11-johnson-nv0_2025_10_21",
    #     "2025_12_30-02_17_17-johnson-nv0_2025_10_21",
    # ]

    # ref / new in GHz, Δf in MHz
    f_ref_GHz = np.array([2.766625, 2.840625, 2.822175, 2.785075])
    f_new_GHz = np.array([2.784250, 2.822175, 2.840625, 2.711450])

    # quartet_freq_shifts = {
    #     "axes": axes,  # (4, 3) int
    #     "f_ref_GHz": f_ref_GHz,  # (4,)
    #     "f_new_GHz": f_new_GHz,  # (4,)
    # }
    ####
    ####Target 2.788 GHz NV indices
    # "[1, 1, -1]"  =  [0, 1, 3, 5, 6, 7, 9, 10, 13, 18, 19, 21, 24, 25, 27, 28, 30, 32, 34, 36, 40, 41, 43, 44, 46, 48, 49, 51, 52, 53, 56, 57, 64, 65, 66, 67, 68, 69, 73, 75, 77, 80, 82, 84, 86, 88, 91, 98, 100, 101, 102, 103, 106, 107, 109, 110, 111, 113, 115, 116, 118, 119, 120, 121, 123, 124, 127, 129, 130, 131, 132, 133, 134, 135, 141, 142, 146, 149, 150, 152, 153, 156, 157, 158, 162, 163, 165, 167, 168, 171, 174, 177, 179, 184, 185, 186, 187, 189, 190, 191, 192, 193, 195, 198, 201, 203]
    # # ####Target 2.841 GHz -> NV indices
    # "[-1, 1, 1]" = [2, 4, 8, 11, 12, 14, 15, 16, 17, 20, 22, 23, 26, 29, 31, 33, 35, 37, 38, 39, 42, 45, 47, 50, 54, 55, 58, 59, 60, 61, 62, 63, 70, 71, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 90, 92, 93, 94, 95, 96, 97, 99, 104, 105, 108, 112, 114, 117, 122, 125, 126, 128, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 151, 154, 155, 159, 160, 161, 164, 166, 169, 170, 172, 173, 175, 176, 178, 180, 181, 182, 183, 188, 194, 196, 197, 199, 200, 202]

    # Map orientation → list of NV indices (from your note)
    # Your orientation lists (as you gave them)
    # fmt:off
    ORI_11m1 = [0, 1, 3, 5, 6, 7, 9, 10, 13, 18, 19, 21, 24, 25, 27, 28, 30, 32, 34, 36, 40, 41, 43, 44, 46, 48, 49, 51, 52, 53, 56, 57, 64, 65, 66, 67, 68, 69, 73, 75, 77, 80, 82, 84, 86, 88, 91, 98, 100, 101, 102, 103, 106, 107, 109, 110, 111, 113, 115, 116, 118, 119, 120, 121, 123, 124, 127, 129, 130, 131, 132, 133, 134, 135, 141, 142, 146, 149, 150, 152, 153, 156, 157, 158, 162, 163, 165, 167, 168, 171, 174, 177, 179, 184, 185, 186, 187, 189, 190, 191, 192, 193, 195, 198, 201, 203]
    ORI_m111 = [2, 4, 8, 11, 12, 14, 15, 16, 17, 20, 22, 23, 26, 29, 31, 33, 35, 37, 38, 39, 42, 45, 47, 50, 54, 55, 58, 59, 60, 61, 62, 63, 70, 71, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 90, 92, 93, 94, 95, 96, 97, 99, 104, 105, 108, 112, 114, 117, 122, 125, 126, 128, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 151, 154, 155, 159, 160, 161, 164, 166, 169, 170, 172, 173, 175, 176, 178, 180, 181, 182, 183, 188, 194, 196, 197, 199, 200, 202]
    # fmt:on
    # print(ORI_11m1, ORI_m111)
    # sys.exit()
    data = widefield.process_multiple_files(file_stems, load_npz=True)
    nv_list = data["nv_list"]
    taus = data["taus"]
    total_evolution_times = 2 * np.array(taus) / 1e3
    counts = np.array(data["counts"])
    sig = counts[0]
    ref = counts[1]
    norm_counts, norm_counts_ste = widefield.process_counts(
        nv_list, sig, ref, threshold=True
    )

    # --- build per-NV orientations ----
    ORI_11m1_set = set(ORI_11m1)
    ORI_m111_set = set(ORI_m111)

    # After you get nv_list from widefield.ocess_multiple_files(...)
    n_nv = len(nv_list)

    # Just “global indices” = position in the list
    nv_indices_global = np.arange(n_nv, dtype=int)

    # Orientation array (N × 3) and labels
    orientations = np.zeros((n_nv, 3), dtype=int)
    ori_labels = []

    for i in range(n_nv):
        if i in ORI_11m1_set:
            ori = (1, 1, -1)
        elif i in ORI_m111_set:
            ori = (-1, 1, 1)
        else:
            ori = (0, 0, 0)  # unknown / unused

        orientations[i, :] = ori
        ori_labels.append(f"[{ori[0]}, {ori[1]}, {ori[2]}]")

    timestamp = dm.get_time_stamp()
    processed_data = {
        "timestamp": timestamp,
        "dataset_ids": file_stems,
        "nv_list": nv_list,
        "norm_counts": norm_counts,
        "norm_counts_ste": norm_counts_ste,
        "total_evolution_times": total_evolution_times,
        "nv_indices_global": nv_indices_global,  # 0..N-1
        "orientations": orientations,  # shape (N, 3)
        # "orientation_labels": np.array(ori_labels, dtype=object),
    }
    # --- Add to your processed_data dict ---
    processed_data.update(
        {
            "B_G": B_G,  # Gauss, shape (3,)
            "B_G_mag": B_G_mag,  # |B| in Gauss
            "B_hat": B_hat,  # unit vector
        }
    )
    tokens = []
    for s in file_stems:
        m = re.search(r"-([A-Za-z0-9]+)-nv", s)  # e.g. "...-johnson-nv0_..."
        if m:
            tokens.append(m.group(1))
    sample = max(set(tokens), key=tokens.count) if tokens else "sample"
    srcsig = f"s{len(file_stems)}-{hashlib.sha1('|'.join(file_stems).encode()).hexdigest()[:6]}"
    # --- tiny signature of the source list ---
    name = f"{sample}_{len(nv_list)}nv_{srcsig}"
    # name   = f"{sample}_{len(fit_nv_labels)}nv_{date}_{rev}_{model}_{sweep}_{srcsig}"
    # print(name)
    file_path = dm.get_file_path(__file__, timestamp, name)
    dm.save_raw_data(processed_data, file_path)

    sys.exit()
    ### get proceeded data data
    # file_stem = "2025_11_10-16_17_03-johnson_204nv_s3-003c56" #dataset 1
    # file_stem = "2025_11_11-01_05_17-johnson_204nv_s3-0e14ae" #dataset 3
    file_stem = "2025_11_11-01_15_45-johnson_204nv_s6-6d8f5c"  # dataset2 + dataset3
    data = dm.get_raw_data(file_stem=file_stem)
    nv_list = data["nv_list"]
    norm_counts = np.array(data["norm_counts"])
    norm_counts_ste = np.array(data["norm_counts_ste"])
    total_evolution_times = np.array(data["total_evolution_times"])
    # print(norm_counts[172])
    # Example NV filtering
    # split_esr = [12, 13, 14, 61, 116]
    # broad_esr = [52, 11]
    # weak_esr  = [72, 64, 55, 96, 112, 87, 12, 58, 36]
    # skip_inds = list(set(split_esr + broad_esr + weak_esr))
    # skip_inds = []
    # nv_inds = [ind for ind in range(len(data["nv_list"])) if ind not in skip_inds]
    # nv_inds = [2, 10, 22, 25, 55, 112]
    # nv_inds = [196]
    nv_inds = None
    # --- Run and plot ---------------------------------------------------------
    ts = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    repr_nv_sig = widefield.get_repr_nv_sig(data["nv_list"])
    save_prefix = dm.get_file_path(__file__, ts, f"{repr_nv_sig.name}-finefit")

    # Toggle these as you wish:
    USE_FIXED_REVIVAL = False  # True -> uses fine_decay_fixed_revival
    ENABLE_EXTRAS = True  # enable alpha/width_slope/chirp + beating + phases
    DEFAULT_REV_US = 37.2
    # # 1) FIT
    popts, pcovs, chis, fit_fns, fit_nv_labels = run_with_amp_and_freq_sweeps(
        nv_list,
        norm_counts,
        norm_counts_ste,
        total_evolution_times,
        # nv_inds= [4, 7, 10, 15, 16, 18, 21, 24, 26, 27, 28, 33, 39, 43, 48, 52, 53, 57, 59, 61, 64, 65, 66, 68, 72, 73, 77, 83, 97, 102, 106, 109, 121, 123, 127, 129, 132, 135, 136, 139, 147, 152, 157, 163, 167, 173, 185, 189, 190, 193, 194, 195, 197, 198, 201, 202],
        # nv_inds= [10, 15],
        # amp_bound_grid=((-0.6, 0.6),(-1.0, 1.0),(-2.0, 2.0),),
        # Optional: tighten frequency boxes
        freq_bound_boxes={"osc_f0": (0.001, 6.0), "osc_f1": (0.001, 6.0)},
        # Optional: force a band (else inferred from sampling)
        freq_seed_band=(0.001, 6.0),
        # Try more seeds if signals are messy
        freq_seed_n_peaks=6,
        # Add custom overrides (e.g., probe a specific revival_time or phase grid)
        # extra_overrides_grid={"osc_phi0": [0.0, np.pi/3, 2*np.pi/3]},
        extra_overrides_grid=None,
        use_fixed_revival=False,
        enable_extras=True,
        fixed_rev_time=37.6,
        verbose=True,
    )

    timestamp = dm.get_time_stamp()
    fit_dict = {
        "timestamp": timestamp,
        "dataset_ids": file_stems,  # provenance
        "default_rev_us": float(DEFAULT_REV_US),
        "run_flags": {
            "use_fixed_revival": bool(USE_FIXED_REVIVAL),
            "enable_extras": bool(ENABLE_EXTRAS),
        },
        # NV indexing/order and sampling grid (so red_chi2 aligns with your original time axis if needed)
        "nv_labels": list(map(int, fit_nv_labels)),
        "times_us": np.asarray(total_evolution_times, float).tolist(),
        # Core results you don't want to recompute
        "popts": [p.tolist() if p is not None else None for p in popts],
        "pcovs": [c.tolist() if c is not None else None for c in pcovs],
        "red_chi2": [float(c) if c is not None else None for c in chis],
        "fit_fn_names": [fn.__name__ if fn is not None else None for fn in fit_fns],
        # Parameter key order (so you can interpret popts later without code spelunking)
        "unified_keys": [
            "baseline",
            "comb_contrast",
            "revival_time_us",
            "width0_us",
            "T2_ms",
            "T2_exp",
            "amp_taper_alpha",
            "width_slope",
            "revival_chirp",
            "osc_amp",
            "osc_f0",
            "osc_f1",
            "osc_phi0",
            "osc_phi1",
        ],
    }
    # repr_nv_sig = widefield.get_repr_nv_sig(nv_list)

    # sample = (re.search(r"-([A-Za-z0-9]+)-nv", file_stems[0]) or [None,"sample"])[1]
    # srcsig = f"s{len(file_stems)}-{hashlib.sha1('|'.join(file_stems).encode()).hexdigest()[:6]}"
    tokens = []
    for s in file_stems:
        m = re.search(r"-([A-Za-z0-9]+)-nv", s)  # e.g. "...-johnson-nv0_..."
        if m:
            tokens.append(m.group(1))
    sample = max(set(tokens), key=tokens.count) if tokens else "sample"
    srcsig = f"s{len(file_stems)}-{hashlib.sha1('|'.join(file_stems).encode()).hexdigest()[:6]}"
    # --- tiny signature of the source list ---
    name = f"{sample}_{len(fit_nv_labels)}nv_{srcsig}"
    # name   = f"{sample}_{len(fit_nv_labels)}nv_{date}_{rev}_{model}_{sweep}_{srcsig}"
    # print(name)
    file_path = dm.get_file_path(__file__, timestamp, name)
    dm.save_raw_data(fit_dict, file_path)
    # print(file_path )
    # NV labels: [ 96, 97, 101, 114, 115, 119, 127, 129, 130, 135, 148, 150, 153, 154, 158, 159, 161, 168, 173, 174, 185, 194, 195]

    plot_individual_fits(
        norm_counts,
        norm_counts_ste,
        total_evolution_times,
        popts,
        nv_inds=fit_nv_labels,
        fit_fn_per_nv=fit_fns,
        # keep_mask=keep_mask,
        show_residuals=True,
        block=False,
    )
    kpl.show(block=True)
    sys.exit()
    # ## laod analysed data
    # timestamp = dm.get_time_stamp()
    # # file_stem= "2025_11_01-16_57_48-rubin-nv0_2025_09_08"
    # file_stem= "2025_11_10-19_33_17-johnson_204nv_s3-003c56"
    # file_stem= "2025_11_10-21_38_55-johnson_204nv_s3-003c56"
    # file_stem= "2025_11_11-01_46_41-johnson_204nv_s3-003c56"
    file_stem = "2025_11_11-06_23_14-johnson_204nv_s6-6d8f5c"

    data = dm.get_raw_data(file_stem=file_stem)
    popts = data["popts"]
    chis = data["red_chi2"]
    fit_nv_labels = data["nv_labels"]
    fit_fn_names = data["fit_fn_names"]
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    # 2) PARAM PANELS (T2 outlier filter)
    # figs, keep_mask, kept_labels = plot_each_param_separately(
    #     popts, chis, fit_nv_labels,
    #     save_prefix= "rubin-spin_echo-2025_09_08",
    #     t2_policy=dict(method="iqr", iqr_k=5, abs_range=(0.00, 1.0))
    # )

    fit_nv_labels = list(map(int, data["nv_labels"]))
    fit_fn_names = data["fit_fn_names"]

    # 1) Map stored names -> real callables
    _fn_map = {
        "fine_decay": fine_decay,
        "fine_decay_fixed_revival": fine_decay_fixed_revival,
    }
    fit_fns = []
    for name in fit_fn_names:
        if name is None:
            fit_fns.append(None)
        else:
            fn = _fn_map.get(name)
            if fn is None:
                fn = fine_decay
            fit_fns.append(fn)

    # 3) INDIVIDUAL FITS — PASS THE SAME LABELS + PER-NV FIT FUNCTIONS
    _ = plot_individual_fits(
        norm_counts,
        norm_counts_ste,
        total_evolution_times,
        popts,
        nv_inds=fit_nv_labels,
        fit_fn_per_nv=fit_fns,
        # keep_mask=keep_mask,
        show_residuals=True,
        block=False,
    )

    # # --------------------------
    # # Example usage
    # # --------------------------
    (
        nv,
        T2_us,
        f0_kHz,
        f1_kHz,
        A_pick_kHz,
        chis,
        fit_fail,
        sT2_us,
        sf0_kHz,
        sf1_kHz,
        sA_pick_kHz,
    ) = extract_T2_freqs_and_errors(data, pick_freq="max", chi2_fail_thresh=3.0)
    # plot_sorted_panels_with_err(
    #     nv, T2_us, sT2_us, A_pick_kHz, sA_pick_kHz,
    #     mask_fit_fail=fit_fail,
    #     # tweak caps if needed:
    #     t2_rel_cap=1.0, t2_pct_cap=95,
    #     A_rel_cap=0.75, A_pct_cap=95
    # )
    # # print(T2_us)
    # THRESH_US = 400.0

    # # # Base validity mask
    # valid = np.isfinite(T2_us) & (~fit_fail)
    # mask = valid & (T2_us >= THRESH_US)

    # # Core table
    # df = pd.DataFrame({
    #     "nv": nv,
    #     "T2_us": T2_us,
    #     "T2_ms": T2_us/1000.0,
    #     "sT2_us": sT2_us,
    #     "f0_kHz": f0_kHz,
    #     "sf0_kHz": sf0_kHz,
    #     "f1_kHz": f1_kHz,
    #     "sf1_kHz": sf1_kHz,
    #     "A_pick_kHz": A_pick_kHz,
    #     "sA_pick_kHz": sA_pick_kHz,
    #     "red_chi2": chis,
    #     "fit_fail": fit_fail,
    # })
    # sel = df.loc[mask].sort_values("T2_us", ascending=False).reset_index(drop=True)

    # print(f"NVs with T2 >= {THRESH_US:.0f} µs: {len(sel)}")
    # print(sel[["nv","T2_us","sT2_us","A_pick_kHz","sA_pick_kHz","f0_kHz","f1_kHz","red_chi2"]].to_string(index=False))

    # 1) Strictly at the cap (allow tiny float noise)
    CAP_US = 200
    THRESH_US = 200
    # mask_cap = np.isfinite(T2_us) & np.isclose(T2_us, CAP_US, atol=1e-6)
    mask_cap = np.isfinite(T2_us) & np.isclose(T2_us, CAP_US, atol=1e-6)
    mask_cap = np.isfinite(T2_us) & (T2_us > THRESH_US)
    cap_indices = np.where(mask_cap)[0]  # 0-based positions in your arrays
    cap_labels = nv[mask_cap]  # NV labels corresponding to those positions

    print("Count:", mask_cap.sum())
    print("Indices:", cap_indices.tolist())
    print("NV labels:", cap_labels.tolist())
    kpl.show(block=True)
