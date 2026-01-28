# -*- coding: utf-8 -*-
"""
Spin-echo: finer fit + fitted-figure + parameter panels

- Physics-y comb with quartic lobes, amplitude taper, width growth, chirp
- Optional two-frequency sin^2 beating with phases
- Smoothly plugs into your plotting + data pipeline

Author: @saroj chand
"""

import os
import sys
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import re, hashlib, datetime as dt
from analysis.spin_echo_work.echo_fit_models import fine_decay, fine_decay_fixed_revival

import matplotlib.ticker as mticker

# --- Optional numba (falls back gracefully) ----------------------------------
try:
    from numba import njit
except Exception:

    def njit(*_args, **_kwargs):
        def wrap(fn):
            return fn

        return wrap


# --- utilities----------------------------------
from utils import data_manager as dm


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


# --- NEW: map p -> dict using the fit function signature (best effort) ---
def params_to_dict(fit_fn, p, default_rev=39.2):
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
            pdict = params_to_dict(fit_fn, p, default_rev=default_rev_for_plot)
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
                timestamp = dm.get_time_stamp()
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
# Plotting
# =============================================================================
def set_axes_equal_3d(ax):
    """Make 3D axes have equal scale (so spheres look like spheres)."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])
    max_range = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(xmid - max_range, xmid + max_range)
    ax.set_ylim3d(ymid - max_range, ymid + max_range)
    ax.set_zlim3d(zmid - max_range, zmid + max_range)


def _echo_summary_lines(taus_us, echo):
    if len(echo) == 0:
        return []
    arr = np.asarray(echo, float)
    n = max(3, len(arr) // 3)
    early = float(np.nanmean(arr[:n]))
    late = float(np.nanmean(arr[-n:]))
    return [
        f"Echo range: {arr.min():.3f} … {arr.max():.3f}",
        f"⟨early⟩→⟨late⟩: {early:.3f} → {late:.3f}",
    ]


def _fine_param_lines(fine_params):
    if not fine_params:
        return []
    pretty = {
        "revival_time": "T_rev (μs)",
        "width0_us": "width₀ (μs)",
        "T2_ms": "T₂ (ms)",
        "T2_exp": "stretch n",
        "amp_taper_alpha": "amp taper α",
        "width_slope": "width slope",
        "revival_chirp": "rev chirp",
    }
    keys = [
        "revival_time",
        "width0_us",
        "T2_ms",
        "T2_exp",
        "amp_taper_alpha",
        "width_slope",
        "revival_chirp",
    ]
    out = []
    for k in keys:
        if k in fine_params:
            v = fine_params[k]
            sval = f"{v:.3g}" if isinstance(v, (int, float)) else f"{v}"
            out.append(f"{pretty[k]}: {sval}")
    return out


# def _site_table_lines(site_info, max_rows=8):
#     if not site_info:
#         return ["(no annotated realization)"]
#     rows = sorted(site_info, key=lambda d: -abs(d.get("Apar_kHz", 0.0)))
#     lines = ["site  |A∥|(kHz)   r", "------------------------"]
#     for d in rows[:max_rows]:
#         sid = d.get("site_id", "?")
#         apar = float(abs(d.get("Apar_kHz", 0.0)))
#         rmag = float(d.get("r", np.nan))
#         lines.append(f"{sid:<5} {apar:>8.0f}  {rmag:>6.2f}")
#     if len(rows) > max_rows:
#         lines.append(f"... (+{len(rows)-max_rows} more)")
#     return lines


def _site_table_lines(site_info, max_rows: int = 4):
    """
    Compact human-readable summary of matched 13C sites.

    Expected keys in each dict of site_info:
      - site_id
      - r           (Å)
      - theta_deg
      - kappa
      - fI_kHz
      - fm_kHz, fp_kHz   (optional)
      - orientation      (tuple/list of 3 ints, e.g. (1,1,1))
    """
    if not site_info:
        return ["(no annotated realization)"]

    # Sort by |kappa| (or fall back to |Apar| if present)
    def sort_key(d):
        k = d.get("kappa", None)
        if k is not None and np.isfinite(k):
            return -abs(k)
        return -abs(d.get("Apar_kHz", 0.0))

    rows = sorted(site_info, key=sort_key)

    lines = ["Matched ¹³C sites:", "------------------"]
    for d in rows[:max_rows]:
        sid = d.get("site_id", "?")
        rmag = float(d.get("r", np.nan))
        theta = float(d.get("theta_deg", np.nan))
        kappa = float(d.get("kappa", np.nan))
        fI = float(d.get("fI_kHz", np.nan))
        fm = d.get("fm_kHz", None)
        fp = d.get("fp_kHz", None)
        ori = d.get("orientation", None)
        if isinstance(ori, (list, tuple)) and len(ori) == 3:
            ori_str = f"({int(ori[0])},{int(ori[1])},{int(ori[2])})"
        else:
            ori_str = "?"

        if fm is not None or fp is not None:
            try:
                fm_f = float(fm) if fm is not None else float("nan")
                fp_f = float(fp) if fp is not None else float("nan")
            except Exception:
                pass
        # First line: ID + orientation + angle
        lines.append(f"site={sid}  nv_ori={ori_str}  θ={theta:5.1f}°")

        # Second line: radius, kappa, fI
        lines.append(
            f"κ={kappa:4.2f}, fI={fI:2.0f} kHz, fm/fp={fm_f:2.0f}/{fp_f:2.0f} kHz"
        )

    return lines


def _env_only_curve(taus_us, fine_params):
    """baseline - envelope(τ); ignores COMB/MOD so you see pure T2 envelope."""
    if not fine_params:
        return None
    baseline = float(fine_params.get("baseline", 1.0))
    T2_ms = float(fine_params.get("T2_ms", 1.0))
    T2_exp = float(fine_params.get("T2_exp", 1.0))
    # envelope(τ) = exp[-(τ/(1000*T2_ms))^T2_exp]
    env = np.exp(-((np.asarray(taus_us, float) / (1000.0 * T2_ms)) ** T2_exp))
    # multiply by comb_contrast if you want to visualize the amplitude scale
    contrast = float(fine_params.get("comb_contrast", 1.0))
    return baseline - contrast * env


def _comb_only_curve(taus_us, fine_params):
    """
    Very light-weight comb sketch (Gaussian revivals); ignores oscillations and width slope.
    Useful if you want to also show envelope×comb (set show_env_times_comb=True).
    """
    if not fine_params:
        return None
    T_rev = float(
        fine_params.get("revival_time", fine_params.get("revival_time_us", 0.0))
    )
    width0 = float(fine_params.get("width0_us", 0.0))
    alpha = float(fine_params.get("amp_taper_alpha", 0.0))
    if T_rev <= 0 or width0 <= 0:
        return np.ones_like(taus_us, dtype=float)

    τ = np.asarray(taus_us, float)
    mmax = int(max(1, np.ceil(τ.max() / T_rev) + 2))
    comb = np.zeros_like(τ, float)
    # sum of Gaussians centered at m*T_rev with amplitude taper ~ exp(-alpha*m)
    for m in range(mmax + 1):
        amp = np.exp(-alpha * m) if alpha > 0 else 1.0
        comb += amp * np.exp(-0.5 * ((τ - m * T_rev) / width0) ** 2)
    # normalize to [0,1] peak
    mx = comb.max()
    if mx > 0:
        comb = comb / mx
    return comb


def plot_echo_with_sites(
    taus_us,
    echo,
    aux,
    title="Spin Echo (single NV)",
    rmax=None,
    fine_params=None,
    units_label="(arb units)",
    nv_label=None,  # show NV id
    sim_info=None,  # dict with sim settings to display
    show_env=True,  # overlay envelope-only
    show_env_times_comb=False,  # optionally overlay envelope×comb
    # --- NEW: experimental & fit extras ---
    echo_ste=None,  # optional 1σ errors on echo
    fit_fn=None,  # optional fit function (fine_decay / fine_decay_fixed_revival)
    fit_params=None,  # parameter vector for fit_fn
    tau_is_half_time=True,  # if True, model is evaluated at t = 2*tau
    default_rev_for_plot=39.2,
):
    fig = plt.figure(figsize=(12, 5))

    # ---------------- Echo panel ----------------
    ax0 = fig.add_subplot(1, 2, 1)

    # --- NEW: plot data with or without errorbars ---
    if echo_ste is not None:
        echo_ste = np.asarray(echo_ste, float)
        ax0.errorbar(
            taus_us,
            echo,
            yerr=echo_ste,
            fmt="o",
            ms=3.0,
            lw=0.8,
            capsize=2,
            alpha=0.9,
            label="echo (data)",
        )
    else:
        ax0.plot(taus_us, echo, lw=1.0, label="echo (data)")

    ax0.set_xlabel("τ (μs)")
    ax0.set_ylabel(f"Coherence {units_label}")

    # Title: include NV label if provided
    if nv_label is not None:
        ax0.set_title(f"{title} — NV {nv_label}")
    else:
        ax0.set_title(title)

    ax0.grid(True, alpha=0.3)

    # Vertical revival guide lines (if provided)
    revs = aux.get("revivals_us", None)
    if revs is not None:
        for t in np.atleast_1d(revs):
            ax0.axvline(t, ls="--", lw=0.7, alpha=0.35)

    # --- NEW: plot full fit curve (using your fine_decay model) ---
    if (fit_fn is not None) and (fit_params is not None):
        taus_arr = np.asarray(taus_us, float)

        # make a denser grid for smooth curve
        if taus_arr.size > 1:
            tau_min = float(np.nanmin(taus_arr))
            tau_max = float(np.nanmax(taus_arr))
            # e.g. 4× more points than data
            taus_dense = np.linspace(tau_min, tau_max, 4 * taus_arr.size)
        else:
            taus_dense = taus_arr

        if tau_is_half_time:
            t_model = 2.0 * taus_dense  # model expects total evolution time
        else:
            t_model = taus_dense

        y_fit = _safe_call_fit_fn(
            fit_fn,
            t_model,
            fit_params,
            default_rev=default_rev_for_plot,
        )

        ax0.plot(
            taus_dense,
            y_fit,
            "-",
            lw=1.8,
            alpha=0.9,
            label="fit (full model)",
        )
    # --- overlay envelope(s) ---
    env_line = None
    if show_env and fine_params:
        y_env = _env_only_curve(taus_us, fine_params)
        if y_env is not None:
            (env_line,) = ax0.plot(
                taus_us,
                y_env,
                ls="--",
                lw=1.2,
                label="envelope (T2)",
                alpha=0.9,
            )

    if show_env_times_comb and fine_params:
        comb = _comb_only_curve(taus_us, fine_params)
        if comb is not None:
            baseline = float(fine_params.get("baseline", 1.0))
            contrast = float(fine_params.get("comb_contrast", 1.0))
            T2_ms = float(fine_params.get("T2_ms", 1.0))
            T2_exp = float(fine_params.get("T2_exp", 1.0))
            env = np.exp(-((np.asarray(taus_us, float) / (1000.0 * T2_ms)) ** T2_exp))
            y_env_comb = baseline - contrast * env * comb
            ax0.plot(
                taus_us,
                y_env_comb,
                ls=":",
                lw=1.2,
                label="envelope×comb (no osc)",
                alpha=0.9,
            )

    # Existing stats box
    stats = aux.get("stats", {}) or {}

    # ---- Combined NV/sim + fine-params box (single box, right-top) ----
    combined_lines = []

    # Header & flags
    if nv_label is not None:
        flag_bits = []
        if show_env:
            flag_bits.append("Env")
        if show_env_times_comb:
            flag_bits.append("Comb")
        hdr = f"NV: {nv_label}"
        if flag_bits:
            hdr += f"  [{'+'.join(flag_bits)} shown]"
        combined_lines.append(hdr)

    # Build a meta dict from sim_info with fallbacks to aux
    meta = {} if sim_info is None else dict(sim_info)
    meta.setdefault("distance_cutoff", aux.get("distance_cutoff"))
    meta.setdefault("Ak_min_kHz", aux.get("Ak_min_kHz"))
    meta.setdefault("Ak_max_kHz", aux.get("Ak_max_kHz"))
    meta.setdefault("T2_fit_us", None)  # you can set this upstream if desired

    # Pretty labels
    pretty_sim = {
        "distance_cutoff": "d matched (Å)",
        "Ak_min_kHz": "Ak_min (kHz)",
        "Ak_max_kHz": "Ak_max (kHz)",
        "T2_fit_us": "T2_fit (μs)",
    }

    def _fmt_meta(k, v):
        if v is None:
            return None
        lab = pretty_sim.get(k, k)
        if k == "hyperfine_path":
            from pathlib import Path

            v = Path(str(v)).stem
        if isinstance(v, float):
            v = f"{v:.3g}"
        return f"{lab}: {v}"

    # Collect sim/meta lines (only those that exist)
    sim_lines = []
    for k in ["distance_cutoff", "Ak_min_kHz", "Ak_max_kHz", "T2_fit_us"]:
        line = _fmt_meta(k, meta.get(k))
        if line:
            sim_lines.append(line)

    # Fine-parameter lines
    fp_lines = _fine_param_lines(fine_params) if fine_params else []
    if fp_lines and show_env:
        fp_lines = ["Exp Params."] + fp_lines

    # Merge sections with a thin separator if both present
    if sim_lines and fp_lines:
        combined_lines.extend(sim_lines + ["—"] + fp_lines)
    elif sim_lines:
        combined_lines.extend(sim_lines)
    elif fp_lines:
        combined_lines.extend(fp_lines)

    # Render the single box (right-top)
    if combined_lines:
        ax0.text(
            0.99,
            0.02,
            "\n".join(combined_lines),
            transform=ax0.transAxes,
            fontsize=9,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.5, lw=0.6),
        )

    # Legend if we drew extra curves
    if (
        (show_env and fine_params)
        or show_env_times_comb
        or (fit_fn is not None and fit_params is not None)
    ):
        ax0.legend(loc="best", fontsize=9, framealpha=0.8)

    # ---------------- 3D positions panel ----------------
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    bg = aux.get("all_candidate_positions", None)
    if bg is not None and len(bg) > 0:
        ax1.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=8, alpha=0.15)

    pos = aux.get("positions", None)
    info = aux.get("site_info", [])
    if pos is not None and len(pos) > 0:
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, depthshade=True)
        for pnt, meta_site in zip(pos, info):
            rmag = meta_site.get("r", np.nan)
            label = f"r={rmag:.2f}"
            ax1.text(pnt[0], pnt[1], pnt[2], label, fontsize=8, ha="left", va="bottom")

    ax1.scatter([0], [0], [0], s=70, marker="*", zorder=5)
    ax1.text(0, 0, 0, "NV", fontsize=9, ha="right", va="top")
    ax1.set_title("¹³C positions (NV frame)")
    ax1.set_xlabel("x (Å)")
    ax1.set_ylabel("y (Å)")
    ax1.set_zlabel("z (Å)")

    if rmax is None:
        if bg is not None and len(bg) > 0:
            rmax = float(np.max(np.linalg.norm(bg, axis=1)))
        elif pos is not None and len(pos) > 0:
            rmax = float(np.max(np.linalg.norm(pos, axis=1)))
        else:
            rmax = 1.0
    rpad = 0.05 * rmax
    ax1.set_xlim(-rmax - rpad, rmax + rpad)
    ax1.set_ylim(-rmax - rpad, rmax + rpad)
    ax1.set_zlim(-rmax - rpad, rmax + rpad)
    set_axes_equal_3d(ax1)

    picked_all = aux.get("picked_ids_per_realization", [])
    n_real = len(picked_all) if picked_all is not None else 0
    n_chosen = len(info) if info is not None else 0
    left_box = [f"Chosen Sites: {n_chosen}", f"Realizations: {n_real}"]
    if stats.get("N_candidates") is not None:
        left_box.append(f"Candidates: {stats['N_candidates']}")
    # if "abundance_fraction" in stats:
    #     left_box.append(f"Abundance p: {100*stats['abundance_fraction']:.2f}%")
    # ax1.text2D(
    #     0.01,
    #     0.02,
    #     "\n".join(left_box),
    #     transform=ax1.transAxes,
    #     fontsize=9,
    #     va="bottom",
    #     ha="left",
    #     bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.6),
    # )

    table_lines = _site_table_lines(info, max_rows=8)
    ax1.text2D(
        0.99,
        0.02,
        "\n".join(table_lines),
        transform=ax1.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.6),
    )

    return fig


def _normalize_orientation_array(orientation):
    """
    Take an array-like of orientation entries and return a 1D object array
    of tuples like (-1,1,1) or None.
    """
    norm = []
    for o in orientation:
        if o is None or (isinstance(o, float) and not np.isfinite(o)):
            norm.append(None)
            continue
        try:
            # e.g. o is something like np.array([-1,1,1]) or list
            t = tuple(int(v) for v in np.ravel(o))
        except Exception:
            # fall back to tuple(o)
            t = tuple(o)
        norm.append(t)
    return np.array(norm, dtype=object)


def plot_branch_pairs(
    f0_kHz,
    f1_kHz,
    title="Spin-echo branch pairs (f0, f1)",
    f_range_kHz=(10, 6000),
    exp_freqs=True,
    orientation=None,  # optional per-NV orientation
    ori_to_str=None,  # optional converter to nice labels
):
    """
    Pairwise plot: for each NV, draw a small vertical 'stick'
    connecting f0 and f1.

    If `orientation` is None:
        - Use markers:
            f0 -> circle ("o")
            f1 -> square ("s")

    If `orientation` is provided:
        - Markers encode ORIENTATION:
            first orientation  -> "o"
            second orientation -> "s"
          (f0 and f1 for a given NV share the same marker)
    """
    f0 = np.asarray(f0_kHz, float)
    f1 = np.asarray(f1_kHz, float)

    # keep only finite, >0 pairs
    mask = np.isfinite(f0) & np.isfinite(f1) & (f0 > 0) & (f1 > 0)
    f0 = f0[mask]
    f1 = f1[mask]

    ori_list = None
    if orientation is not None:
        # slice orientation with same mask
        ori_arr = np.asarray(orientation, dtype=object)[mask]

    if f0.size == 0:
        raise ValueError("No valid (f0, f1) pairs to plot.")

    # sort by mid-frequency
    mid_freq = (f0 + f1) / 2.0
    order = np.argsort(mid_freq)
    f0 = f0[order]
    f1 = f1[order]

    if orientation is not None:
        ori_arr = ori_arr[order]

        # normalize each entry to a tuple like (-1,1,1) or None
        ori_list = []
        for o in ori_arr:
            if o is None:
                ori_list.append(None)
                continue

            arr = np.asarray(o)
            if arr.size == 0:
                ori_list.append(None)
                continue

            flat = arr.ravel()
            # typical case: 3-vector
            if flat.size == 3:
                tup = (int(flat[0]), int(flat[1]), int(flat[2]))
            else:
                tup = tuple(int(v) for v in flat)
            ori_list.append(tup)

    x = np.arange(1, len(f0) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # vertical sticks
    for xi, y0, y1 in zip(x, f0, f1):
        ax.vlines(xi, min(y0, y1), max(y0, y1), lw=0.7, alpha=0.6)

    # -----------------------------------------
    # CASE 1: no orientation info (old behavior)
    # -----------------------------------------
    if ori_list is None:
        if exp_freqs:
            ax.scatter(x, f0, s=12, marker="o", label="f0")
            ax.scatter(x, f1, s=12, marker="s", label="f1")
        else:
            ax.scatter(x, f0, s=12, marker="o", label="f_plus")
            ax.scatter(x, f1, s=12, marker="s", label="f_minus")

    # -----------------------------------------
    # CASE 2: orientation-aware markers
    # -----------------------------------------
    else:
        if ori_to_str is None:

            def ori_to_str(o):
                return str(o)

        # get unique orientations as tuples via plain Python
        keys_seen = set()
        unique_oris = []
        for t in ori_list:
            if t is None:
                continue
            if t not in keys_seen:
                keys_seen.add(t)
                unique_oris.append(t)

        # sort deterministically
        unique_oris.sort()

        # assign markers: first ori -> circle, second -> square, others '^'
        marker_map = {}
        for idx, ori_val in enumerate(unique_oris):
            if idx == 0:
                marker_map[ori_val] = "o"
            elif idx == 1:
                marker_map[ori_val] = "s"
            else:
                marker_map[ori_val] = "^"

        handled = set()
        for ori_val in unique_oris:
            m = marker_map[ori_val]

            # Python-level mask: ori_list is list of tuples / None
            mask_ori = np.array([o == ori_val for o in ori_list], dtype=bool)

            x_sub = x[mask_ori]
            f0_sub = f0[mask_ori]
            f1_sub = f1[mask_ori]

            ori_label = ori_to_str(ori_val)

            ax.scatter(
                x_sub,
                f0_sub,
                s=12,
                marker=m,
                label=f"{ori_label} (f0,f1)" if ori_val not in handled else None,
            )
            ax.scatter(
                x_sub,
                f1_sub,
                s=12,
                marker=m,
            )
            handled.add(ori_val)

    ax.set_yscale("log")
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("NV index (sorted by mid-frequency)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(framealpha=0.85)

    fig.tight_layout()
    return fig, ax


def compare_two_fields(
    matches_df: pd.DataFrame,
    field_labels=None,
    title_prefix: str = "C13 bath",
):
    """
    Compare the same 13C sites between TWO fields and make 'meaningful' plots
    that show what changed.

    Requirements:
      - matches_df has columns:
          'orientation', 'site_index', 'field_label'
        and ideally:
          'distance_A', 'f_minus_kHz', 'f_plus_kHz', 'kappa'
          optional: 'T2_us'
      - Exactly two fields in 'field_label' (or specify them via field_labels).

    What it does:
      1) Aggregates per (orientation, site_index, field_label):
           - distance_A (mean)
           - f_minus_kHz (mean)
           - f_plus_kHz (mean)
           - kappa (mean)
           - T2_us (mean, if present)
           - n_matches (count)
      2) Pivots to a wide table with separate columns per field.
      3) Makes comparison plots:
           - f_- (field1 vs field2) + identity line
           - f_+ (field1 vs field2)
           - kappa (field1 vs field2)
           - multiplicity (n_matches field1 vs field2)
           - histograms of Δf_- , Δf_+ , Δkappa
           - |Δf_-| and |Δf_+| vs distance
    """

    df = matches_df.copy()

    if "field_label" not in df.columns:
        raise ValueError(
            "matches_df must have a 'field_label' column to compare fields."
        )

    # ------------------------ 0) choose which two fields ------------------------
    unique_fields = list(df["field_label"].unique())
    if field_labels is None:
        if len(unique_fields) != 2:
            raise ValueError(
                f"Expected exactly 2 fields, found {len(unique_fields)}: {unique_fields}"
            )
        field_labels = sorted(unique_fields)
    else:
        # ensure they are actually present
        for fld in field_labels:
            if fld not in unique_fields:
                raise ValueError(
                    f"Requested field_label '{fld}' not found in matches_df "
                    f"(present: {unique_fields})"
                )
        # if user passed more than 2, just take first two
        field_labels = list(field_labels[:2])

    f1, f2 = field_labels
    print(f"Comparing fields: {f1} vs {f2}")

    # ------------------------ 1) aggregate per site + field ------------------------
    agg_dict = {
        "distance_A": "mean",
        "f_minus_kHz": "mean",
        "f_plus_kHz": "mean",
        "kappa": "mean",
        "field_label": "first",
    }

    if "T2_us" in df.columns:
        agg_dict["T2_us"] = "mean"

    # Count how many NVs matched this site at each field
    grouped = df.groupby(
        ["orientation", "site_index", "field_label"], as_index=False
    ).agg(agg_dict)
    grouped["n_matches"] = (
        df.groupby(["orientation", "site_index", "field_label"])["nv_label"].transform(
            "nunique"
        )
        if "nv_label" in df.columns
        else df.groupby(["orientation", "site_index", "field_label"])[
            "distance_A"
        ].transform("size")
    )

    # ------------------------ 2) pivot to wide format ------------------------
    # index = (orientation, site_index), columns = field_label
    wide = grouped.set_index(["orientation", "site_index", "field_label"]).unstack(
        "field_label"
    )

    # Flatten MultiIndex columns to e.g. "distance_A_f1"
    wide.columns = [f"{col[0]}_{col[1]}" for col in wide.columns.values]
    wide = wide.reset_index()

    # Restrict to only the two fields we care about, and drop sites missing either field
    needed_cols = []
    for base in ["distance_A", "f_minus_kHz", "f_plus_kHz", "kappa", "n_matches"]:
        needed_cols.extend([f"{base}_{f1}", f"{base}_{f2}"])
    if "T2_us" in agg_dict:
        needed_cols.extend([f"T2_us_{f1}", f"T2_us_{f2}"])

    # Keep only rows where both fields exist (non-NaN) for at least the frequencies
    mask_both = wide[[f"f_minus_kHz_{f1}", f"f_minus_kHz_{f2}"]].notna().all(axis=1)
    wide_both = wide.loc[mask_both].copy()

    print(f"Number of sites with data at BOTH fields: {len(wide_both)}")

    # Distance: they should be the same across fields; just take f1's
    rA = wide_both[f"distance_A_{f1}"].to_numpy(float)

    # ------------------------ 3) convenience arrays ------------------------
    f_minus_1 = wide_both[f"f_minus_kHz_{f1}"].to_numpy(float)
    f_minus_2 = wide_both[f"f_minus_kHz_{f2}"].to_numpy(float)
    f_plus_1 = wide_both[f"f_plus_kHz_{f1}"].to_numpy(float)
    f_plus_2 = wide_both[f"f_plus_kHz_{f2}"].to_numpy(float)

    kappa_1 = wide_both[f"kappa_{f1}"].to_numpy(float)
    kappa_2 = wide_both[f"kappa_{f2}"].to_numpy(float)

    nmatch_1 = wide_both[f"n_matches_{f1}"].to_numpy(float)
    nmatch_2 = wide_both[f"n_matches_{f2}"].to_numpy(float)

    # differences
    d_fm = f_minus_2 - f_minus_1
    d_fp = f_plus_2 - f_plus_1
    d_kappa = kappa_2 - kappa_1
    d_nmatch = nmatch_2 - nmatch_1

    # optional T2
    has_T2 = ("T2_us" in agg_dict) and (f"T2_us_{f1}" in wide_both.columns)
    if has_T2:
        T2_1 = wide_both[f"T2_us_{f1}"].to_numpy(float)
        T2_2 = wide_both[f"T2_us_{f2}"].to_numpy(float)
        d_T2 = T2_2 - T2_1

    # ------------------------ 4) f_minus and f_plus comparison plots ------------------------
    max_fm = max(f_minus_1.max(), f_minus_2.max())
    max_fp = max(f_plus_1.max(), f_plus_2.max())

    # f_minus: B1 vs B2
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(f_minus_1, f_minus_2, s=20, alpha=0.7)
    ax.plot([0, max_fm], [0, max_fm], "k--", linewidth=1)
    ax.set_xlabel(f"$f_-({f1})$  (kHz)")
    ax.set_ylabel(f"$f_-({f2})$  (kHz)")
    ax.set_title(f"{title_prefix}: $f_-$ shift between {f1} and {f2}")
    ax.grid(True, alpha=0.3)

    # f_plus: B1 vs B2
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(f_plus_1, f_plus_2, s=20, alpha=0.7)
    ax.plot([0, max_fp], [0, max_fp], "k--", linewidth=1)
    ax.set_xlabel(f"$f_+({f1})$  (kHz)")
    ax.set_ylabel(f"$f_+({f2})$  (kHz)")
    ax.set_title(f"{title_prefix}: $f_+$ shift between {f1} and {f2}")
    ax.grid(True, alpha=0.3)

    # ------------------------ 5) Δf histograms & Δf vs radius ------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        d_fm, bins=50, histtype="step", label=r"$\Delta f_- = f_-^{(2)} - f_-^{(1)}$"
    )
    ax.hist(
        d_fp, bins=50, histtype="step", label=r"$\Delta f_+ = f_+^{(2)} - f_+^{(1)}$"
    )
    ax.set_xlabel("Δf (kHz)")
    ax.set_ylabel("Count of sites")
    ax.set_title(f"{title_prefix}: frequency shifts between fields")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(rA, np.abs(d_fm), s=20, alpha=0.7, label=r"|Δf_-|")
    ax.scatter(rA, np.abs(d_fp), s=20, alpha=0.7, label=r"|Δf_+|")
    ax.set_xlabel("Distance NV–13C (Å)")
    ax.set_ylabel("|Δf| (kHz)")
    ax.set_title(f"{title_prefix}: magnitude of frequency shift vs radius")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # ------------------------ 6) kappa comparison ------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(kappa_1, kappa_2, s=20, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel(f"$\\kappa({f1})$")
    ax.set_ylabel(f"$\\kappa({f2})$")
    ax.set_title(f"{title_prefix}: ESEEM misalignment change")
    ax.grid(True, alpha=0.3)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d_kappa, bins=40, histtype="stepfilled", alpha=0.6)
    ax.set_xlabel(r"Δκ = κ(2) − κ(1)")
    ax.set_ylabel("Count of sites")
    ax.set_title(f"{title_prefix}: change in ESEEM misalignment between fields")
    ax.grid(True, alpha=0.3)

    # ------------------------ 7) multiplicity change ------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(nmatch_1, nmatch_2, s=20, alpha=0.7)
    max_nm = max(nmatch_1.max(), nmatch_2.max())
    ax.plot([0, max_nm], [0, max_nm], "k--", linewidth=1)
    ax.set_xlabel(f"$n_\\mathrm{{matches}}({f1})$")
    ax.set_ylabel(f"$n_\\mathrm{{matches}}({f2})$")
    ax.set_title(f"{title_prefix}: site multiplicity change between fields")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d_nmatch, bins=np.arange(d_nmatch.min() - 0.5, d_nmatch.max() + 1.5, 1.0))
    ax.set_xlabel(r"Δn_matches = n_matches(2) − n_matches(1)")
    ax.set_ylabel("Count of sites")
    ax.set_title(f"{title_prefix}: change in site occupancy between fields")
    ax.grid(True, alpha=0.3)

    # ------------------------ 8) T2 comparison (if available) ------------------------
    if has_T2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(T2_1, T2_2, s=20, alpha=0.7)
        max_T2 = max(T2_1.max(), T2_2.max())
        ax.plot([1e-1, max_T2], [1e-1, max_T2], "k--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"$T_2({f1})$ (µs)")
        ax.set_ylabel(f"$T_2({f2})$ (µs)")
        ax.set_title(f"{title_prefix}: $T_2$ change between fields")
        ax.grid(True, which="both", alpha=0.3)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(d_T2, bins=40, histtype="stepfilled", alpha=0.6)
        ax.set_xlabel(r"Δ$T_2$ = $T_2(2)$ − $T_2(1)$ (µs)")
        ax.set_ylabel("Count of sites")
        ax.set_title(f"{title_prefix}: change in $T_2$ between fields")
        ax.grid(True, alpha=0.3)

    print("compare_two_fields: finished. Call plt.show() (or kpl.show()) to view.")
    return wide_both


def plot_branch_correlation_by_orientation(
    f0_kHz,
    f1_kHz,
    orientation=None,
    ori_to_str=None,
    title="Branch correlation: f0 vs f1",
    f_range_kHz=(10, 6000),
    filter_to_range=False,
    x_label="f0 (kHz)",
    y_label="f1 (kHz)",
):
    """
    Scatter plot of f0 vs f1, colored/marked by NV orientation.

    - One point per NV: (f0, f1) in kHz.
    - If `orientation` is provided, points are grouped by orientation
      and assigned different markers.
    - If `filter_to_range` is True, only keep points with BOTH f0 and f1
      inside [f_min, f_max] from `f_range_kHz`.
    """

    f0 = np.asarray(f0_kHz, float)
    f1 = np.asarray(f1_kHz, float)

    # basic finite + positive filter
    mask = np.isfinite(f0) & np.isfinite(f1) & (f0 > 0) & (f1 > 0)

    if filter_to_range:
        fmin, fmax = f_range_kHz
        mask &= (f0 >= fmin) & (f0 <= fmax) & (f1 >= fmin) & (f1 <= fmax)

    f0 = f0[mask]
    f1 = f1[mask]

    ori_list = None
    if orientation is not None:
        ori_arr = np.asarray(orientation, dtype=object)[mask]

    if f0.size == 0:
        raise ValueError("No valid (f0, f1) pairs to plot.")

    # normalize orientations to tuples
    if orientation is not None:
        ori_list = []
        for o in ori_arr:
            if o is None:
                ori_list.append(None)
                continue
            arr = np.asarray(o)
            if arr.size == 0:
                ori_list.append(None)
                continue
            flat = arr.ravel()
            if flat.size == 3:
                tup = (int(flat[0]), int(flat[1]), int(flat[2]))
            else:
                tup = tuple(int(v) for v in flat)
            ori_list.append(tup)

    fig, ax = plt.subplots(figsize=(6, 5))

    # no orientation info → single scatter
    if ori_list is None:
        ax.scatter(f0, f1, s=12, alpha=0.7)
    else:
        if ori_to_str is None:

            def ori_to_str(o):
                return str(o)

        # Build a *Python* set of unique orientations
        keys_seen = set()
        unique_oris = []
        for t in ori_list:
            if t is None:
                continue
            if t not in keys_seen:
                keys_seen.add(t)
                unique_oris.append(t)
        unique_oris.sort()

        marker_cycle = ["o", "s", "^", "D", "v"]
        marker_map = {
            ori_val: marker_cycle[i % len(marker_cycle)]
            for i, ori_val in enumerate(unique_oris)
        }

        handled = set()

        for ori_val in unique_oris:
            m = marker_map[ori_val]
            # 1D boolean mask built from Python list → always right shape
            mask_ori = np.array([o == ori_val for o in ori_list], dtype=bool)

            f0_sub = f0[mask_ori]
            f1_sub = f1[mask_ori]

            label = ori_to_str(ori_val)
            ax.scatter(
                f0_sub,
                f1_sub,
                s=16,
                marker=m,
                alpha=0.8,
                label=label if ori_val not in handled else None,
            )
            handled.add(ori_val)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if ori_list is not None:
        ax.legend(fontsize="x-small", framealpha=0.85)

    fig.tight_layout()
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt


def set_axes_equal_3d(ax):
    """Make 3D axes have equal scale (so spheres look like spheres)."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])

    xr = abs(xlim[1] - xlim[0])
    yr = abs(ylim[1] - ylim[0])
    zr = abs(zlim[1] - zlim[0])
    r = 0.5 * max(xr, yr, zr)

    ax.set_xlim3d(xmid - r, xmid + r)
    ax.set_ylim3d(ymid - r, ymid + r)
    ax.set_zlim3d(zmid - r, zmid + r)


def plot_c13_candidates_only(
    aux,
    title="¹³C candidate positions (NV frame)",
    rmax=None,
    s=8,
    alpha=0.18,
):
    bg = aux.get("all_candidate_positions", None)
    if bg is None or len(bg) == 0:
        raise ValueError("aux['all_candidate_positions'] is missing or empty.")

    bg = np.asarray(bg, float)

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=s, alpha=alpha, depthshade=False)

    # NV at origin
    ax.scatter([0], [0], [0], s=90, marker="*", zorder=5)
    ax.text(0, 0, 0, "NV", fontsize=10, ha="right", va="top")

    ax.set_title(title)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")

    if rmax is None:
        rmax = float(np.max(np.linalg.norm(bg, axis=1)))
    rpad = 0.05 * rmax
    ax.set_xlim(-rmax - rpad, rmax + rpad)
    ax.set_ylim(-rmax - rpad, rmax + rpad)
    ax.set_zlim(-rmax - rpad, rmax + rpad)
    set_axes_equal_3d(ax)

    return fig, ax

