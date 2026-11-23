# -*- coding: utf-8 -*-
"""
Fit the Ramsey fringes: drive frequency error, each NV's offset, and T2*.

The drive sits DETUNING below the configured frequency, so each NV shows fringes at
f0 = (line - drive), near the detuning, plus the two 14N side components at
f0 -/+ HYPERFINE_MHZ. The model is
    y(tau) = c + exp(-(tau / T2*)^p) * sum_k [a_k cos(2 pi f_k tau) + b_k sin(2 pi f_k tau)],
    f_k = f0 + k * HYPERFINE_MHZ, k = -1, 0, +1,
with free amplitudes and phases per component (the off-resonant pi/2 pulses weight the
three differently). For trial (f0, T2*) the a_k, b_k and c follow from a linear fit, so
only f0 and T2* are searched. The line (m_I = 0) is at drive + f0.

The NV average gives the line position (to set the drive); each NV is then fitted on its
own for its offset and T2*. Several files can be combined: list them all.

Run from the dioptric folder in the dioptric environment:
    python ramsey_nv_average_fit.py

Created on November 13th, 2026; per-NV uncertainties without refitting, November 15th;
offset map across the field of view, November 16th
"""

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FILE_STEMS = ["2026_11_13-15_28_34-johnson-nv0_2026_10_02"]  # the saved .txt names without extension
EXCLUDE_NVS = [0, 6]  # another orientation
THRESHOLD = True  # as in the other analyses; False uses raw counts
UWAVE_IND = 1  # signal generator that drove the NVs
DETUNING_MHZ = None  # None reads it from the data file (3 MHz in do_ramsey)
HYPERFINE_MHZ = 2.16  # 14N splitting
DECAY_POWER = 2.0  # 2 = Gaussian decay, typical for NVs
ACCEPT_KHZ = 5.0  # NVs within this of the median offset are listed as usable for RQS
N_BOOTSTRAP = 200
LATE_US = 1.5  # the late part of the decay, for the check below
MAKE_MAP = True  # also map each NV's offset across the field of view, with a plane fit
RQS_SEQUENCE_RMS_US = 7.6  # rms of the integral of s(t) for the current RQS plan (tau = 100 us)


# ---------------------------------------------------------------------------
# Model and fits (tau in us, frequencies in MHz)
# ---------------------------------------------------------------------------
def _basis(tau_us, f0_mhz, t2_us):
    """Design matrix: constant, then cos/sin of the three components under the decay."""
    tau_us = np.asarray(tau_us, float)
    decay = np.exp(-((tau_us / t2_us) ** DECAY_POWER))
    columns = [np.ones_like(tau_us)]
    for k in (-1, 0, 1):
        phase = 2 * np.pi * (f0_mhz + k * HYPERFINE_MHZ) * tau_us
        columns += [decay * np.cos(phase), decay * np.sin(phase)]
    return np.column_stack(columns)


def ramsey_model(tau_us, f0_mhz, t2_us, *linear):
    return _basis(tau_us, f0_mhz, t2_us) @ np.asarray(linear, float)


def _linear_solve(tau_us, y, w, f0_mhz, t2_us):
    design = _basis(tau_us, f0_mhz, t2_us)
    coef, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
    return coef, (design @ coef - y) * w


def fit_ramsey(tau_us, y, y_err, f0_guess, t2_guess, f0_window=0.5):
    """Returns dict with f0, t2, their errors, reduced chi^2 and all parameters."""
    tau_us, y, y_err = (np.asarray(v, float) for v in (tau_us, y, y_err))
    ok = np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
    t, v, w = tau_us[ok], y[ok], 1 / y_err[ok]

    def residuals(p):
        return _linear_solve(t, v, w, p[0], p[1])[1]

    lo = [f0_guess - f0_window, 0.1]
    hi = [f0_guess + f0_window, 100.0]
    best = None
    for t2_start in (0.5 * t2_guess, t2_guess, 2 * t2_guess):  # a few starts, keep the best
        p0 = [f0_guess, float(np.clip(t2_start, lo[1] * 1.01, hi[1] * 0.99))]
        sol = least_squares(residuals, p0, bounds=(lo, hi), x_scale=[0.01, 1.0])
        if best is None or sol.cost < best.cost:
            best = sol
    f0, t2 = best.x
    coef, _ = _linear_solve(t, v, w, f0, t2)

    # Uncertainties from the Jacobian at the solution, without re-optimizing, so the
    # estimate stays inside the search window (an unbounded refit can jump to another
    # fringe component on noisy single NVs)
    popt = np.array([f0, t2, *coef])
    base = ramsey_model(t, *popt) * w
    jac = np.empty((len(t), len(popt)))
    for i in range(len(popt)):
        step = max(abs(popt[i]) * 1e-6, 1e-9)
        dp = np.zeros_like(popt)
        dp[i] = step
        jac[:, i] = (ramsey_model(t, *(popt + dp)) - ramsey_model(t, *(popt - dp))) * w / (2 * step)
    resid = base - v * w
    red_chi2 = float(np.sum(resid**2) / max(len(t) - len(popt), 1))
    cov = np.linalg.pinv(jac.T @ jac)
    perr = np.sqrt(np.abs(np.diag(cov))) * np.sqrt(max(red_chi2, 1.0))
    # A fit that ends on the edge of the frequency window found no fringe of its own there
    at_edge = abs(f0 - f0_guess) > 0.95 * f0_window
    return {"f0_mhz": float(popt[0]), "f0_err_mhz": float(perr[0]),
            "t2_us": float(abs(popt[1])), "t2_err_us": float(perr[1]),
            "reduced_chi2": red_chi2, "popt": popt, "at_edge": bool(at_edge)}


def fringe_guess(tau_us, y, near_mhz, span_mhz=1.0):
    """Strongest Fourier component within span_mhz of near_mhz (zero-padded spectrum)."""
    tau_us = np.asarray(tau_us, float)
    step = np.median(np.diff(tau_us))
    n_pad = 16 * len(y)
    spectrum = np.abs(np.fft.rfft(np.asarray(y) - np.mean(y), n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=step)
    near = np.abs(freqs - near_mhz) <= span_mhz
    return float(freqs[near][np.argmax(spectrum[near])])


def nv_average(curves, errors):
    """Inverse-variance weighted average over NVs, per delay."""
    curves, errors = np.asarray(curves, float), np.asarray(errors, float)
    bad = ~np.isfinite(curves) | ~np.isfinite(errors) | (errors <= 0)
    weights = np.where(bad, 0.0, 1.0 / np.where(bad, 1.0, errors) ** 2)
    total = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(bad, 0.0, curves) * weights).sum(axis=0) / total
        err = 1 / np.sqrt(total)
    return mean, err


def rqs_contrast_factor(t2_us, rms_us=RQS_SEQUENCE_RMS_US):
    """Rough fraction of contrast kept by the RQS plan under quasi-static noise of this T2*."""
    return 1 / np.sqrt(1 + 2 * (rms_us / t2_us) ** 2)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def fit_all(taus_ns, curves, errors, detuning_mhz, n_boot=N_BOOTSTRAP, seed=0):
    tau_us = np.asarray(taus_ns, float) * 1e-3
    mean, err = nv_average(curves, errors)
    f0_guess = fringe_guess(tau_us, mean, detuning_mhz)
    avg = fit_ramsey(tau_us, mean, err, f0_guess, t2_guess=0.5 * tau_us.max())

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(curves), len(curves))
        m, e = nv_average(curves[pick], errors[pick])
        try:
            boot.append(fit_ramsey(tau_us, m, e, avg["f0_mhz"], avg["t2_us"], 0.2)["f0_mhz"])
        except Exception:
            pass
    avg["f0_bootstrap_err_mhz"] = float(np.std(boot)) if len(boot) > 10 else np.nan

    per_nv = []
    for y, e in zip(curves, errors):
        try:
            fit = fit_ramsey(tau_us, y, e, avg["f0_mhz"], avg["t2_us"], 0.3)
            per_nv.append(None if fit["at_edge"] else fit)
        except Exception:
            per_nv.append(None)
    # Weighted sum of the single-NV fits, and the data averaged over the same NVs: if the
    # sum follows the data where the average fit does not, that structure is real signal
    # from NVs that differ; if not, it is noise shared by all NVs
    fitted = [i for i, fit in enumerate(per_nv) if fit is not None]
    fit_sum = sub_mean = sub_err = fit_sum_dense = None
    tau_dense = np.linspace(tau_us.min(), tau_us.max(), 3000)
    if fitted:
        e = np.asarray(errors, float)[fitted]
        w = np.where(np.isfinite(e) & (e > 0), 1 / np.where(e > 0, e, 1) ** 2, 0.0)
        # at the measured delays, for the residual check
        models = np.array([ramsey_model(tau_us, *per_nv[i]["popt"]) for i in fitted])
        fit_sum = (models * w).sum(axis=0) / w.sum(axis=0)
        sub_mean, sub_err = nv_average(np.asarray(curves, float)[fitted], e)
        # on a fine grid, for a smooth plot (each NV weighted by its mean weight)
        w_nv = w.mean(axis=1)
        models_dense = np.array([ramsey_model(tau_dense, *per_nv[i]["popt"]) for i in fitted])
        fit_sum_dense = (models_dense * w_nv[:, None]).sum(axis=0) / w_nv.sum()
    return {"tau_us": tau_us, "mean": mean, "err": err, "avg": avg, "per_nv": per_nv,
            "fit_sum": fit_sum, "sub_mean": sub_mean, "sub_err": sub_err,
            "tau_dense": tau_dense, "fit_sum_dense": fit_sum_dense}


def late_residuals(res, after_us=LATE_US):
    """Rms residual after after_us, in error bars: (single model fit, sum of single-NV fits)."""
    late = res["tau_us"] > after_us
    avg_resid = (res["mean"] - ramsey_model(res["tau_us"], *res["avg"]["popt"])) / res["err"]
    out = [float(np.sqrt(np.mean(avg_resid[late] ** 2)))]
    if res["fit_sum"] is not None:
        sum_resid = (res["sub_mean"] - res["fit_sum"]) / res["sub_err"]
        out.append(float(np.sqrt(np.mean(sum_resid[late] ** 2))))
    else:
        out.append(np.nan)
    return out


def summarize(res, drive_ghz, detuning_mhz, keep):
    avg = res["avg"]
    offsets = np.array([np.nan if p is None else (p["f0_mhz"] - detuning_mhz) * 1e3 for p in res["per_nv"]])
    offset_errs = np.array([np.nan if p is None else p["f0_err_mhz"] * 1e3 for p in res["per_nv"]])
    t2s = np.array([np.nan if p is None else p["t2_us"] for p in res["per_nv"]])
    good = np.isfinite(offsets) & np.isfinite(offset_errs) & (offset_errs < 100)
    median_offset = float(np.median(offsets[good])) if good.any() else np.nan
    spread = 1.4826 * np.median(np.abs(offsets[good] - median_offset)) if good.any() else np.nan
    usable = [int(keep[i]) for i in np.flatnonzero(good & (np.abs(offsets - median_offset) <= ACCEPT_KHZ))]
    return {
        "drive_ghz": drive_ghz,
        "detuning_mhz": detuning_mhz,
        "line_ghz": drive_ghz + avg["f0_mhz"] * 1e-3,
        "avg_offset_khz": (avg["f0_mhz"] - detuning_mhz) * 1e3,
        "avg_offset_fit_err_khz": avg["f0_err_mhz"] * 1e3,
        "avg_offset_bootstrap_err_khz": avg["f0_bootstrap_err_mhz"] * 1e3,
        "avg_t2_us": avg["t2_us"],
        "avg_reduced_chi2": avg["reduced_chi2"],
        "median_offset_khz": median_offset,
        "offset_spread_khz": float(spread),
        "median_offset_err_khz": float(np.median(offset_errs[good])) if good.any() else np.nan,
        # spread between NVs with the single-NV fit noise removed (in quadrature)
        "offset_true_spread_khz": float(np.sqrt(max(spread**2 - np.median(offset_errs[good]) ** 2, 0.0)))
        if good.any() else np.nan,
        "median_t2_us": float(np.nanmedian(t2s[good])) if good.any() else np.nan,
        "per_nv_offset_khz": offsets,
        "per_nv_offset_err_khz": offset_errs,
        "per_nv_t2_us": t2s,
        "nv_indices": list(keep),
        "usable_nvs": usable,
        "late_check": late_residuals(res),
    }


def report(s, num_used, num_excluded):
    print(f"\nRamsey: {num_used} NVs used, {num_excluded} excluded; drive {s['drive_ghz']:.6f} GHz "
          f"({s['detuning_mhz']:g} MHz below the configured frequency)")
    print(f"  NV average: line at {s['line_ghz']:.6f} GHz, i.e. {s['avg_offset_khz']:+.1f} kHz from the "
          f"configured frequency (+/- {s['avg_offset_fit_err_khz']:.1f} fit, "
          f"+/- {s['avg_offset_bootstrap_err_khz']:.1f} bootstrap); reduced chi2 {s['avg_reduced_chi2']:.2f}")
    print(f"  Per NV: median offset {s['median_offset_khz']:+.1f} kHz, spread {s['offset_spread_khz']:.1f} kHz "
          f"(typical single-NV uncertainty {s['median_offset_err_khz']:.1f} kHz; "
          f"spread between NVs without that noise about {s['offset_true_spread_khz']:.1f} kHz)")
    print(f"  T2*: {s['median_t2_us']:.2f} us median per NV ({s['avg_t2_us']:.2f} us from the average, "
          f"which also includes the spread between NVs)")
    factor = rqs_contrast_factor(s["median_t2_us"])
    print(f"  Rough RQS contrast kept at this T2* (current plan, tau = 100 us): {100 * factor:.0f} %")
    print(f"  NVs within +/- {ACCEPT_KHZ:g} kHz of the median: {len(s['usable_nvs'])} -> {s['usable_nvs']}")
    if s["median_offset_err_khz"] > ACCEPT_KHZ:
        print(f"  (Single-NV uncertainties exceed {ACCEPT_KHZ:g} kHz, so this list is only indicative.)")
    if "late_check" in s:
        single, summed = s["late_check"]
        print(f"  Late part (after {LATE_US:g} us), rms residual in error bars: average fit {single:.2f}, "
              f"sum of single-NV fits {summed:.2f}")
        if single > 1.5 and summed > 1.5:
            print("    Neither follows it: the late wiggles look like noise shared by all NVs, "
                  "so the error bars of the average are too small.")
        elif single > 1.5:
            print("    The single-NV fits follow it: the late wiggles are real signal from NVs that differ.")
    new_freq = s["drive_ghz"] + s["detuning_mhz"] * 1e-3 + s["median_offset_khz"] * 1e-6
    print(f"\n  -> Config frequency for signal generator {UWAVE_IND} (NVs' median line): {new_freq:.6f} GHz")


def plot(res, s):
    import matplotlib.pyplot as plt

    tau_us, avg = res["tau_us"], res["avg"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [2, 1, 1]})
    dense = np.linspace(tau_us.min(), tau_us.max(), 3000)
    axes[0].errorbar(tau_us, res["mean"], yerr=res["err"], fmt="o", ms=3, label="NV average")
    axes[0].plot(dense, ramsey_model(dense, *avg["popt"]), "-", lw=1,
                 label=f"fit: f0 = {avg['f0_mhz']:.4f} MHz, T2* = {avg['t2_us']:.2f} us")
    if res["fit_sum_dense"] is not None:
        axes[0].plot(res["tau_dense"], res["fit_sum_dense"], "-", lw=1, color="tab:green",
                     label="sum of single-NV fits")
    axes[0].set_xlabel("Delay (us)")
    axes[0].set_ylabel("Normalized NV population (NV average)")
    axes[0].legend(loc="upper right", fontsize=8)

    step = np.median(np.diff(tau_us))
    n_pad = 16 * len(tau_us)
    spectrum = np.abs(np.fft.rfft(res["mean"] - np.mean(res["mean"]), n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=step)
    axes[1].plot(freqs, spectrum)
    for k in (-1, 0, 1):
        axes[1].axvline(avg["f0_mhz"] + k * HYPERFINE_MHZ, color="k", ls="--", lw=0.8)
    axes[1].set_xlim(0, min(freqs.max(), avg["f0_mhz"] + 2 * HYPERFINE_MHZ + 1))
    axes[1].set_xlabel("Fringe frequency (MHz)")
    axes[1].set_ylabel("Fourier magnitude")

    offsets = s["per_nv_offset_khz"][np.isfinite(s["per_nv_offset_khz"])]
    if offsets.size:
        axes[2].hist(offsets, bins=25)
        axes[2].axvline(s["median_offset_khz"], color="k", ls="--", lw=1)
    axes[2].set_xlabel("Offset from configured frequency (kHz)")
    axes[2].set_ylabel("Number of NVs")
    fig.tight_layout()
    return fig


def nv_pixel(nv):
    """Camera pixel (x, y) of an NV entry from the data file (dict or NVSig object)."""
    coords = nv.get("coords") if isinstance(nv, dict) else getattr(nv, "coords", None)
    if coords is None:
        return (np.nan, np.nan)
    items = coords.items() if isinstance(coords, dict) else vars(coords).items()
    for key, value in items:
        name = getattr(key, "value", None) or getattr(key, "name", None) or key
        if "pixel" in str(name).lower() and value is not None and len(value) >= 2:
            return (float(value[0]), float(value[1]))
    return (np.nan, np.nan)


def plane_fit(xy, z, err):
    """Weighted fit z = a + b (x - x0) + c (y - y0), then refit without points more than
    3 robust spreads off the plane. Returns a dict (gradient in kHz per pixel, direction,
    explained fraction of the spread, residual rms, off-plane points)."""
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(z) & np.isfinite(err) & (err > 0)
    idx = np.flatnonzero(ok)
    if idx.size < 5:
        return None
    center = np.median(xy[idx], axis=0)
    use = np.ones(idx.size, bool)
    for _ in range(2):
        X = np.column_stack([np.ones(idx.size), xy[idx, 0] - center[0], xy[idx, 1] - center[1]])
        w = 1 / err[idx]
        coef = np.linalg.lstsq(X[use] * w[use, None], z[idx][use] * w[use], rcond=None)[0]
        resid = z[idx] - X @ coef
        spread = 1.4826 * np.median(np.abs(resid[use] - np.median(resid[use])))
        use = np.abs(resid) <= 3 * max(spread, 1e-9)
    total = np.var(z[idx][use])
    return {
        "center_px": center,
        "coef": coef,
        "gradient_khz_per_px": float(np.hypot(coef[1], coef[2])),
        "direction_deg": float(np.degrees(np.arctan2(coef[2], coef[1]))),
        "explained": float(1 - np.var(resid[use]) / total) if total > 0 else np.nan,
        "residual_rms_khz": float(np.sqrt(np.mean(resid[use] ** 2))),
        "off_plane": [int(i) for i in idx[~use]],
        "used": idx[use],
    }


def offset_map(nv_list, s):
    """Map of each NV's offset across the field of view; returns (figure, plane fit)."""
    import matplotlib.pyplot as plt

    keep = list(s["nv_indices"])
    xy = np.array([nv_pixel(nv_list[i]) for i in keep])
    z = np.asarray(s["per_nv_offset_khz"], float)
    err = np.asarray(s["per_nv_offset_err_khz"], float)
    if not np.isfinite(xy).any():
        print("\nOffset map: no pixel positions found in the NV list, so no map.")
        return None, None
    fit = plane_fit(xy, z, err)
    if fit is not None:
        fit["off_plane"] = [int(keep[i]) for i in fit["off_plane"]]  # positions -> NV numbers
    good = np.isfinite(z) & np.isfinite(xy).all(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    span = np.nanpercentile(np.abs(z[good] - s["median_offset_khz"]), 95) if good.any() else 1.0
    sc = ax.scatter(xy[good, 0], xy[good, 1], c=z[good], cmap="coolwarm", s=45, edgecolors="k", linewidths=0.4,
                    vmin=s["median_offset_khz"] - span, vmax=s["median_offset_khz"] + span)
    bad = ~good & np.isfinite(xy).all(axis=1)
    if bad.any():
        ax.scatter(xy[bad, 0], xy[bad, 1], marker="x", color="gray", s=30, label="no reliable fit")
        ax.legend(loc="lower right", fontsize=8)
    for i, (x, y) in zip(keep, xy):
        if np.isfinite(x):
            ax.annotate(str(i), (x, y), fontsize=5, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="Offset from configured frequency (kHz)")
    if fit is not None:
        cx, cy = fit["center_px"]
        size = 0.25 * np.ptp(xy[good, 0]) if good.sum() > 1 else 20
        ang = np.radians(fit["direction_deg"])
        ax.annotate("", (cx + size * np.cos(ang), cy + size * np.sin(ang)), (cx, cy),
                    arrowprops=dict(arrowstyle="->", lw=2, color="k"))
        ax.set_title(f"Offsets rise along the arrow: {100 * fit['gradient_khz_per_px']:.0f} kHz per 100 px")
    ax.set_xlabel("Camera x (pixel)")
    ax.set_ylabel("Camera y (pixel)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image convention: y grows downward

    ax = axes[1]
    if fit is not None:
        u = np.array([np.cos(np.radians(fit["direction_deg"])), np.sin(np.radians(fit["direction_deg"]))])
        along = (xy - fit["center_px"]) @ u
        on = np.zeros(len(keep), bool)
        on[fit["used"]] = True
        ax.errorbar(along[good & on], z[good & on], yerr=err[good & on], fmt="o", ms=4, alpha=0.8, label="NVs")
        off = good & ~on
        if off.any():
            ax.errorbar(along[off], z[off], yerr=err[off], fmt="s", ms=5, color="tab:red", label="off the plane")
        line = np.linspace(np.nanmin(along[good]), np.nanmax(along[good]), 50)
        ax.plot(line, fit["coef"][0] + fit["gradient_khz_per_px"] * line, "k-", label="plane fit")
        ax.set_xlabel("Position along the arrow (pixel)")
        ax.legend(fontsize=8)
    ax.set_ylabel("Offset from configured frequency (kHz)")
    fig.tight_layout()
    return fig, fit


def report_map(fit, s):
    if fit is None:
        return
    print("\nOffset map across the field of view:")
    print(f"  Gradient: {fit['gradient_khz_per_px']:.2f} kHz per pixel "
          f"({100 * fit['gradient_khz_per_px']:.0f} kHz per 100 pixels), offsets rising toward "
          f"{fit['direction_deg']:.0f} deg (0 = +x, 90 = +y, image y pointing down)")
    print(f"  The plane explains {100 * fit['explained']:.0f} % of the spread between NVs; "
          f"{fit['residual_rms_khz']:.0f} kHz rms remains, against about "
          f"{s['median_offset_err_khz']:.0f} kHz of single-NV fit noise")
    if fit["off_plane"]:
        print(f"  NVs more than 3 robust spreads off the plane: {fit['off_plane']}")


def load(file_stems):
    from utils import data_manager as dm

    sig_parts, ref_parts, first = [], [], None
    for stem in file_stems:
        data = dm.get_raw_data(file_stem=stem, load_npz=True, use_cache=True)
        if first is None:
            first = data
        elif not np.allclose(np.asarray(data["taus"], float), np.asarray(first["taus"], float)):
            raise ValueError(f"{stem} has different delays")
        counts = np.asarray(data["counts"])
        sig_parts.append(counts[0].astype(np.float32))
        ref_parts.append(counts[1].astype(np.float32))
    return first, np.concatenate(sig_parts, axis=1), np.concatenate(ref_parts, axis=1)


def drive_and_detuning(data):
    """Drive frequency (GHz) and detuning (MHz) used, from the data file."""
    detuning = DETUNING_MHZ if DETUNING_MHZ is not None else float(data.get("detuning", 3.0))
    freq_list = data.get("uwave_freq_list")
    if freq_list is not None and len(freq_list) > UWAVE_IND:
        return float(freq_list[UWAVE_IND]), detuning
    sig_gens = data["config"]["Microwaves"]["VirtualSigGens"]
    sig_gen = sig_gens.get(str(UWAVE_IND), sig_gens.get(UWAVE_IND))
    return float(sig_gen["frequency"]) - detuning * 1e-3, detuning


def analyze(file_stems=FILE_STEMS):
    import matplotlib.pyplot as plt

    from utils import data_manager as dm
    from utils import widefield

    data, sig, ref = load(file_stems)
    nv_list, taus = data["nv_list"], np.asarray(data["taus"], float)
    drive_ghz, detuning = drive_and_detuning(data)
    processed = widefield.process_counts(nv_list, sig, ref, threshold=THRESHOLD)
    curves, errors = np.asarray(processed[0], float), np.asarray(processed[1], float)
    keep = [ind for ind in range(len(nv_list)) if ind not in set(EXCLUDE_NVS)]

    res = fit_all(taus, curves[keep], errors[keep], detuning)
    s = summarize(res, drive_ghz, detuning, keep)
    report(s, len(keep), len(nv_list) - len(keep))
    fig = plot(res, s)

    fig_map, plane = offset_map(nv_list, s) if MAKE_MAP else (None, None)
    report_map(plane, s)

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, "nv_average_ramsey")
    summary = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in s.items()}
    summary["source_files"] = list(file_stems)
    if plane is not None:
        summary |= {
            "nv_pixels": [list(nv_pixel(nv_list[i])) for i in s["nv_indices"]],
            "gradient_khz_per_px": plane["gradient_khz_per_px"],
            "gradient_direction_deg": plane["direction_deg"],
            "plane_explained_fraction": plane["explained"],
            "plane_residual_rms_khz": plane["residual_rms_khz"],
            "nvs_off_plane": plane["off_plane"],
        }
    dm.save_raw_data(summary, file_path)
    dm.save_figure(fig, file_path)
    if fig_map is not None:
        dm.save_figure(fig_map, dm.get_file_path(__file__, timestamp, "nv_average_ramsey_map"))
    print(f"\nSaved to {file_path}")
    plt.show(block=True)
    return res, s


if __name__ == "__main__":
    analyze()
