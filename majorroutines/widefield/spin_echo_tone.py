# -*- coding: utf-8 -*-
"""
Widefield spin echo with a phase-locked test tone: calibrates the AC field per OPX volt
at each NV, at the tone frequency and through the same chain RQS uses.

main() sweeps the tone amplitude A at two tone phases 90 deg apart (default 90 and 0).
Each NV's echo signal follows c + b cos(k A) at each phase, with a shared c and b. The
tone's true phase at the NVs doesn't matter: the two phases give
    k_max = sqrt(k_90^2 + k_0^2)  (rad per volt)
and with the echo phase gamma B 2T/pi of a sine tone (T = 1/f, gamma = 2 pi x 28.025
Hz/nT), the field along the NV axis per volt of OPX amplitude (into 50 ohm) is
    field per volt = k_max / (gamma 2T/pi).
That value holds for the attenuation in the chain during the measurement (attenuation_db
is only recorded). atan2(k_0, k_90) is how far the tone's phase at the NVs is from the
set phase, up to its sign.

Control panel function (after: from majorroutines.widefield import spin_echo_tone):

def do_spin_echo_tone(nv_list):
    amps_v = np.linspace(0, 0.5, 21).tolist()
    spin_echo_tone.main(nv_list, num_reps=10, num_runs=30, amps_v=amps_v,
                        freq_hz=16.7e3, attenuation_db=30)

With IQ pulse phases (first pi/2, pi, last pi/2), e.g. x,y,x and x,y,y:

def do_spin_echo_tone_xyx(nv_list):
    amps_v = np.linspace(0, 0.5, 21).tolist()
    spin_echo_tone.main(nv_list, num_reps=10, num_runs=30, amps_v=amps_v,
                        freq_hz=16.7e3, attenuation_db=20, pulse_phases_deg=(0, 90, 0))

(0, 90, 90) gives x,y,y; (0, 0, 0) gives x,x,x through the IQ path.

The echo lasts one tone period (1/f). At this low bias field the 13C nuclei collapse the
echo within ~10 us and revive it every ~60 us of echo time (spin echo of Oct 16, 2026),
so choose f so that 1/f lands on a revival: 1/(60 us) = 16.7 kHz. A 100 kHz echo (10 us)
would sit in the collapse. At 16.7 kHz each nT gives 6x more echo phase than at 100 kHz,
hence the 30 dB attenuator: the 0.025 V amplitude steps resolve up to ~19,000 nT/V.

Created on November 9th, 2026; 16.7 kHz settings added on November 15th, 2026;
IQ pulse phases and the sin readout added on November 17th, 2026

@author: schand
"""

import matplotlib.pyplot as plt
import warnings

import numpy as np
from scipy.optimize import curve_fit as scipy_curve_fit

GAMMA_RAD_PER_S_PER_NT = 2 * np.pi * 28.025  # NV gyromagnetic ratio


# Thresholded charge states, normalized like the ESR, Rabi and Ramsey analyses; raw photon
# counts (False) can blow up when the two reference levels are close (few shots)
THRESHOLD = True


def phase_per_nt(freq_hz, sensing_time_s=None):
    """Echo phase (rad) per nT of tone amplitude: gamma * (2/pi) * sensing time.

    The Hahn echo senses for one tone period (sensing_time_s None -> 1/f); a multi-pulse
    sequence locked to the tone (pi pulses half a period apart) senses for its whole
    length, n_pi half periods (see xy_tone.py)."""
    sensing = 1 / float(freq_hz) if sensing_time_s is None else float(sensing_time_s)
    return GAMMA_RAD_PER_S_PER_NT * 2 * sensing / np.pi


def _tidy(fig):
    """Tidy a figure's layout, unless the plotting style already set a layout engine:
    kplotlib turns on matplotlib's constrained layout, and switching such a figure to
    tight_layout after a colorbar exists raises a RuntimeError."""
    get_engine = getattr(fig, "get_layout_engine", None)
    if get_engine is not None and get_engine() is not None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except RuntimeError:
            pass


def _k_limit(amps):
    """Largest k the amplitude steps can resolve; above it, cosines alias back to slower ones."""
    return np.pi / np.min(np.diff(np.unique(np.asarray(amps, float))))


def _wave(x, quadrature="cos"):
    """The echo signal's shape against echo phase x: cos for the usual readout (last pi/2
    along the first), sin when the last pi/2 is 90 deg from the first (e.g. x,y,y)."""
    return np.sin(x) if quadrature == "sin" else np.cos(x)


def _grid_free(amps, y, quadrature="cos"):
    """(k, c, b) for y = c + b wave(k A), by a grid search over k with c and b free."""
    best = (np.inf, 0.0, float(np.mean(y)), 0.0)
    for k in np.linspace(0, _k_limit(amps), 500)[1:]:
        design = np.column_stack([np.ones_like(amps), _wave(k * amps, quadrature)])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        res = np.sum((y - design @ coef) ** 2)
        if res < best[0]:
            best = (res, k, coef[0], coef[1])
    return best[1:]


def _grid_fixed(amps, y, c, b, quadrature="cos"):
    """k for y = c + b wave(k A) with c and b already known (from the strong phase).
    With the sin readout, k can be negative (the response flips sign between tone phases)."""
    k_lim = _k_limit(amps)
    k_grid = np.linspace(-k_lim if quadrature == "sin" else 0, k_lim, 1000 if quadrature == "sin" else 500)
    res = [np.sum((y - c - b * _wave(k * amps, quadrature)) ** 2) for k in k_grid]
    return k_grid[int(np.argmin(res))]


def fit_nv(amps, y, ste=None, return_all_errors=False, quadrature="cos"):
    """Joint fit of one NV. y and ste: shape (num_phases, num_amps).

    The phase whose signal changes most fixes the offset c and contrast b; the other
    phases share them. Returns |k| per phase (rad/V), their uncertainties, and the fit.
    """
    amps = np.asarray(amps, float)
    y = np.asarray(y, float)
    num_phases = y.shape[0]
    smoothed = [np.convolve(row, np.ones(3) / 3, mode="valid") for row in y]
    strong = int(np.argmax([np.ptp(row) for row in smoothed]))
    k_strong, c0, b0 = _grid_free(amps, y[strong], quadrature)
    k0 = [k_strong if ind == strong else _grid_fixed(amps, row, c0, b0, quadrature) for ind, row in enumerate(y)]

    def model(_, c, b, *ks):
        return np.concatenate([c + b * _wave(k * amps, quadrature) for k in ks])

    sigma = None
    if ste is not None:
        ste = np.asarray(ste, float).ravel()
        if np.all(np.isfinite(ste)) and np.all(ste > 0):
            sigma = ste
    k_max = _k_limit(amps)
    k_low = -k_max if quadrature == "sin" else 0.0
    lower = [-np.inf, -np.inf] + [k_low] * num_phases
    upper = [np.inf, np.inf] + [k_max] * num_phases
    p0 = [c0, b0] + [min(max(k, k_low), k_max) for k in k0]
    popt, pcov = scipy_curve_fit(
        model, np.zeros(y.size), y.ravel(), p0=p0, sigma=sigma,
        absolute_sigma=sigma is not None, bounds=(lower, upper), maxfev=20000,
    )
    ks = np.abs(popt[2:2 + num_phases])
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    errs = perr[2:2 + num_phases]
    if return_all_errors:
        return ks, errs, popt, perr
    return ks, errs, popt


def field_from_ks(ks, errs, phases_deg, freq_hz, sensing_time_s=None):
    """Field per volt (nT/V), its uncertainty, and the tone's phase error (deg)."""
    ks, errs = np.asarray(ks, float), np.asarray(errs, float)
    quarter = len(ks) == 2 and np.isclose(abs(phases_deg[0] - phases_deg[1]) % 180, 90)
    if quarter:
        k_max = np.hypot(ks[0], ks[1])
        k_err = np.hypot(ks[0] * errs[0], ks[1] * errs[1]) / k_max if k_max > 0 else np.nan
        phase_error = np.degrees(np.arctan2(ks[1], ks[0]))
    else:
        ind = int(np.argmax(ks))
        k_max, k_err, phase_error = ks[ind], errs[ind], np.nan
    per_nt = phase_per_nt(freq_hz, sensing_time_s)
    return k_max / per_nt, k_err / per_nt, phase_error


EXCLUDE_NVS = (0, 6)  # another orientation: no echo at this drive frequency
MIN_CONTRAST_SIGMA = 3.0  # single-NV fit kept only if its contrast b exceeds this many errors
MIN_CONTRAST_FRACTION = 0.25  # ... and is at least this fraction of the NV-average contrast
MAX_K_FRACTION = 0.6  # ... and its k stays below this fraction of what the steps resolve
DETECTION_DCHI2 = 25.0  # NV-average response counts as detected above this chi^2 improvement


def _clean_counts(counts, nv_list, exclude_nvs):
    """Drop excluded NVs; replace scattered invalid counts (inf or NaN, e.g. values that
    overflowed when saved) with the median of the same experiment, NV and step over the
    other runs and reps; drop NVs left with no valid data for some step.

    counts: (2, nvs, runs, steps, reps). Returns counts, kept NV indices, notes."""
    num_nvs = counts.shape[1]
    keep_nv = np.array([ind not in set(exclude_nvs) for ind in range(num_nvs)])
    bad = ~np.isfinite(counts)
    notes = []
    if bad.any():
        notes.append(f"{int(bad.sum())} invalid count values (inf or NaN) of {counts.size} in the file; "
                     f"replaced by the median of the same NV and step from the other runs")
        clean = np.where(bad, np.nan, counts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmedian(clean, axis=(2, 4), keepdims=True)
        counts = np.where(bad, np.broadcast_to(fill, counts.shape), counts)
        empty = ~np.isfinite(counts).all(axis=(0, 2, 3, 4))
        if np.any(empty & keep_nv):
            notes.append(f"dropped NVs with no valid data for some step: {[int(i) for i in np.flatnonzero(empty & keep_nv)]}")
            keep_nv &= ~empty
    nv_idx = np.flatnonzero(keep_nv)
    if nv_idx.size == 0:
        raise ValueError("No NVs left to analyze")
    return counts[:, nv_idx], nv_idx, notes


def _significance(amps, y, ste, popt, quadrature="cos"):
    """chi^2 improvement of the shared-k model over a constant, for the NV-average curves."""
    w = 1 / np.where(np.isfinite(ste) & (ste > 0), ste, np.inf) ** 2
    c_const = np.sum(w * y) / np.sum(w)
    chi_const = np.sum(w * (y - c_const) ** 2)
    model = np.array([popt[0] + popt[1] * _wave(k * amps, quadrature) for k in popt[2:]])
    chi_fit = np.sum(w * (y - model) ** 2)
    return float(chi_const - chi_fit)


def analyze(raw_data, plot=True, exclude_nvs=EXCLUDE_NVS, details=False):
    """Field per volt from a saved or fresh run; returns a dict of results.

    Headline: a joint fit (shared contrast, one k per tone phase) to the NV-averaged signal,
    which failed single-NV fits cannot pull around. Single NVs are fitted too and kept only
    if their contrast is clearly nonzero and their k is resolvable."""
    from utils import widefield

    nv_list = raw_data["nv_list"]
    amps = np.asarray(raw_data["amps_v"], float)
    phases = list(raw_data["phases_deg"])
    freq_hz = float(raw_data["freq_hz"])
    sensing_time_s = raw_data.get("sensing_time_s")  # None for the Hahn echo (one period)
    att = raw_data.get("attenuation_db")
    n_pi = raw_data.get("n_pi")
    quadrature = raw_data.get("readout_quadrature", "cos")
    pulse_phases = raw_data.get("pulse_phases_deg")
    # Saved files store counts as float16 (max 65504): convert before computing means and spreads
    counts = np.asarray(raw_data["states"] if "states" in raw_data else raw_data["counts"], dtype=np.float64)
    counts, nv_idx, notes = _clean_counts(counts, nv_list, exclude_nvs)
    num_runs = counts.shape[2]
    nvs = [nv_list[i] for i in nv_idx]
    # Some versions return (avg, ste), others (avg, ste, norms)
    processed = widefield.process_counts(nvs, counts[0], counts[1], threshold=THRESHOLD)
    avg, ste = np.asarray(processed[0], float), np.asarray(processed[1], float)
    shape = (len(nvs), len(phases), len(amps))
    avg, ste = np.reshape(avg, shape), np.reshape(ste, shape)

    # NV average (inverse-variance weighted) and its shared-k fit
    w = np.where(np.isfinite(ste) & (ste > 0) & np.isfinite(avg), 1 / np.where(ste > 0, ste, 1) ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(w > 0, avg, 0) * w).sum(axis=0) / w.sum(axis=0)
        mean_err = 1 / np.sqrt(w.sum(axis=0))
    ks_avg, errs_avg, popt_avg, perr_avg = fit_nv(amps, mean, mean_err, return_all_errors=True, quadrature=quadrature)
    field_avg, field_avg_err, phase_err_avg = field_from_ks(ks_avg, errs_avg, phases, freq_hz, sensing_time_s)
    dchi2 = _significance(amps, mean, mean_err, popt_avg, quadrature)
    detected = dchi2 > DETECTION_DCHI2
    b_avg = abs(popt_avg[1])

    # Single NVs, with a quality cut
    k_lim = _k_limit(amps)
    fields = np.full(len(nvs), np.nan)
    errors = np.full(len(nvs), np.nan)
    phase_errors = np.full(len(nvs), np.nan)
    good = np.zeros(len(nvs), bool)
    ks_all = []
    popts = []
    for i, (y, s_) in enumerate(zip(avg, ste)):
        try:
            ks, errs, popt, perr = fit_nv(amps, y, s_, return_all_errors=True, quadrature=quadrature)
            field, error, phase_error = field_from_ks(ks, errs, phases, freq_hz, sensing_time_s)
            fields[i], errors[i], phase_errors[i] = field, error, phase_error
            b, b_err = abs(popt[1]), perr[1]
            good[i] = (np.isfinite(field) and b > MIN_CONTRAST_SIGMA * b_err
                       and b > MIN_CONTRAST_FRACTION * b_avg and np.max(ks) < MAX_K_FRACTION * k_lim)
            ks_all.append([float(k) for k in ks])
            popts.append(popt)
        except Exception:
            ks_all.append([np.nan] * len(phases))
            popts.append(None)
    if detected:
        # single NVs must also agree with the NV average within a factor of 3
        good &= (fields > field_avg / 3) & (fields < 3 * field_avg)
    else:
        good[:] = False  # with no response in the average, single-NV fits only follow noise
    median_good = float(np.median(fields[good])) if good.any() else np.nan
    spread_good = float(np.std(fields[good])) if good.sum() > 1 else np.nan

    # Report
    sequence = f", {n_pi} pi pulses over {sensing_time_s * 1e6:g} us" if n_pi is not None else ""
    if pulse_phases is not None:
        names = {0: "x", 90: "y", 180: "-x", 270: "-y"}
        sequence += ", pulses " + ",".join(names.get(int(round(p)) % 360, f"{p:g}") for p in pulse_phases)
        sequence += f" (IQ on, {quadrature} readout)"
    print(f"Run at {freq_hz / 1e3:g} kHz{sequence}" + (f", with {att} dB in the chain" if att is not None else ""))
    for note in notes:
        print(f"  Data: {note}")
    print(f"  NVs analyzed: {len(nvs)} (excluded {list(exclude_nvs)}), runs in the file: {num_runs}")
    if num_runs < 5:
        print(f"  WARNING: the file has only {num_runs} runs, far too few for a meaningful "
              f"calibration (a test run?). The numbers below are not reliable.")
    if detected:
        print(f"  NV average: {field_avg:.1f} +/- {field_avg_err:.1f} nT/V along the NV axes "
              f"(response significance: chi2 improvement {dchi2:.0f})")
        print(f"  Tone phase at the NVs is off from the set phase by about {phase_err_avg:.0f} deg")
        reached = max(ks_avg) * amps.max()
        if reached < np.pi / 2:
            print(f"  Warning: the largest amplitude reaches only about {reached:.1f} rad of echo phase; "
                  f"raise the amplitudes or use less attenuation (aim for 3-10 rad).")
        elif max(ks_avg) > MAX_K_FRACTION * k_lim:
            print("  Warning: the echo phase changes too much between amplitude steps; use finer steps.")
        if att is not None:
            print(f"  At 40 dB this becomes about {field_avg * 10 ** (-(40 - att) / 20):.2f} nT/V "
                  f"(nominal attenuation; use the measured value when you have it)")
    else:
        print(f"  NV average: no clear response to the tone (chi2 improvement {dchi2:.1f}, "
              f"below {DETECTION_DCHI2:g}).")
        print(f"  The largest echo phase over the sweep stays below about 1 rad, i.e. under roughly "
              f"{1 / (phase_per_nt(freq_hz, sensing_time_s) * amps.max()):.0f} nT/V.")
    if detected:
        print(f"  Single NVs passing the quality cut: {int(good.sum())} of {len(nvs)}"
              + (f"; median {median_good:.1f} nT/V, spread (std) {spread_good:.1f} nT/V" if good.any() else ""))
    else:
        print("  Single NVs: not reported, since the NV average shows no response.")

    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        dense = np.linspace(amps.min(), amps.max(), 400)
        for p_ind, phase in enumerate(phases):
            axes[0].errorbar(amps, mean[p_ind], yerr=mean_err[p_ind], fmt="o", ms=4,
                             label=f"tone phase {phase:g} deg")
            if detected:
                axes[0].plot(dense, popt_avg[0] + popt_avg[1] * _wave(popt_avg[2 + p_ind] * dense, quadrature), "-",
                             color=axes[0].lines[-1].get_color() if axes[0].lines else None)
        axes[0].set_xlabel("Tone amplitude (V at the OPX)")
        axes[0].set_ylabel("Signal, NV average")
        axes[0].set_title("NV average" + (f": {field_avg:.0f} nT/V" if detected else ": no clear response"))
        axes[0].legend()
        if good.any():
            axes[1].hist(fields[good], bins=20)
            axes[1].axvline(median_good, color="k", ls="--", label=f"median {median_good:.0f} nT/V")
        if detected:
            axes[1].axvline(field_avg, color="tab:red", ls=":", label=f"NV average {field_avg:.0f} nT/V")
        axes[1].set_xlabel("Field per volt along each NV axis (nT/V), single NVs passing the cut")
        axes[1].set_ylabel("Number of NVs")
        if good.any() or detected:
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "no single-NV results\n(no response in the NV average)",
                         ha="center", va="center", transform=axes[1].transAxes)
        _tidy(fig)

    detail = {}
    if details:
        detail = _details(nvs, nv_idx, amps, phases, avg, ste, popts, fields, errors, good,
                          field_avg if detected else np.nan, plot, quadrature)

    return {
        **detail,
        "nv_indices_analyzed": [int(i) for i in nv_idx],
        "runs_used": int(num_runs),
        "field_per_volt_nT": [float(x) for x in fields],
        "field_per_volt_err_nT": [float(x) for x in errors],
        "tone_phase_error_deg": [float(x) for x in phase_errors],
        "k_per_phase_rad_per_v": ks_all,
        "single_nv_fit_passed": [bool(x) for x in good],
        "median_field_per_volt_nT": median_good,
        "nv_average_field_per_volt_nT": float(field_avg) if detected else np.nan,
        "nv_average_field_err_nT": float(field_avg_err) if detected else np.nan,
        "nv_average_dchi2": dchi2,
        "figure": fig,
    }


def _nv_pixel(nv):
    """Camera pixel (x, y) of an NV entry (dict or NVSig object)."""
    coords = nv.get("coords") if isinstance(nv, dict) else getattr(nv, "coords", None)
    if coords is None:
        return (np.nan, np.nan)
    items = coords.items() if isinstance(coords, dict) else vars(coords).items()
    for key, value in items:
        name = getattr(key, "value", None) or getattr(key, "name", None) or key
        if "pixel" in str(name).lower() and value is not None and len(value) >= 2:
            return (float(value[0]), float(value[1]))
    return (np.nan, np.nan)


def _details(nvs, nv_idx, amps, phases, avg, ste, popts, fields, errors, good, field_avg, plot, quadrature="cos"):
    """Closer look at the single NVs that pass the cut: robust statistics of their field per
    volt, every single-NV fit (pages, pass/fail marked), and a map across the field of view."""
    out = {}
    passed = fields[good]
    if passed.size:
        rng = np.random.default_rng(0)
        boot = [np.median(rng.choice(passed, passed.size)) for _ in range(2000)]
        q25, q50, q75 = np.percentile(passed, [25, 50, 75])
        mad = 1.4826 * np.median(np.abs(passed - q50))
        print("\n  Closer look at the single NVs passing the cut:")
        print(f"    {passed.size} NVs: median {q50:.1f} nT/V +/- {np.std(boot):.1f} (bootstrap); "
              f"middle half {q25:.0f}-{q75:.0f} nT/V; robust spread {mad:.0f} nT/V "
              f"({100 * mad / q50:.0f} %); range {passed.min():.0f}-{passed.max():.0f}")
        print(f"    Typical single-NV uncertainty {np.median(errors[good]):.0f} nT/V, so the spread "
              + ("is mostly real variation between NVs" if mad > 2 * np.median(errors[good])
                 else "is largely fit noise"))
        if np.isfinite(field_avg):
            print(f"    NV-average fit {field_avg:.1f} nT/V, {100 * (field_avg / q50 - 1):+.1f} % from the median")
        out |= {"passed_median_nT": float(q50), "passed_median_err_nT": float(np.std(boot)),
                "passed_iqr_nT": [float(q25), float(q75)], "passed_robust_spread_nT": float(mad)}
    if not plot:
        return out

    # Every single-NV fit, 24 per page
    pages = []
    per_page = 24
    dense = np.linspace(amps.min(), amps.max(), 300)
    for start in range(0, len(nvs), per_page):
        fig, axes = plt.subplots(4, 6, figsize=(18, 11), sharex=True)
        for ax, i in zip(axes.flat, range(start, min(start + per_page, len(nvs)))):
            for p in range(len(phases)):
                line = ax.errorbar(amps, avg[i, p], yerr=ste[i, p], fmt="o", ms=2.5, lw=0.8)
                if popts[i] is not None:
                    ax.plot(dense, popts[i][0] + popts[i][1] * _wave(popts[i][2 + p] * dense, quadrature), "-", lw=1,
                            color=line[0].get_color())
            verdict = "pass" if good[i] else "fail"
            value = f"{fields[i]:.0f} nT/V" if np.isfinite(fields[i]) else "no fit"
            ax.set_title(f"NV {nv_idx[i]}: {value} ({verdict})", fontsize=8,
                         color="tab:green" if good[i] else "tab:red")
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[min(per_page, len(nvs) - start):]:
            ax.axis("off")
        fig.supxlabel("Tone amplitude (V at the OPX)")
        fig.supylabel("Signal (both tone phases)")
        _tidy(fig)
        pages.append(fig)
    out["single_nv_figures"] = pages

    # Map of the passing NVs' field per volt across the field of view
    xy = np.array([_nv_pixel(nv) for nv in nvs])
    if np.isfinite(xy).any() and good.any():
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ok = good & np.isfinite(xy).all(axis=1)
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=fields[ok], cmap="viridis", s=45, edgecolors="k", linewidths=0.4)
        bad = ~good & np.isfinite(xy).all(axis=1)
        ax.scatter(xy[bad, 0], xy[bad, 1], marker="x", color="gray", s=25, label="failed the cut")
        for i, (x, y) in enumerate(xy):
            if np.isfinite(x):
                ax.annotate(str(nv_idx[i]), (x, y), fontsize=5, xytext=(3, 3), textcoords="offset points")
        fig.colorbar(sc, ax=ax, label="Field per volt along the NV axis (nT/V)")
        ax.set_xlabel("Camera x (pixel)")
        ax.set_ylabel("Camera y (pixel)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Field per volt across the field of view")
        _tidy(fig)
        out["map_figure"] = fig
    return out


def save_detail_figures(details, file_path):
    """Save the field map (figure) and the single-NV fits (PDF) next to a run's data, then
    close the PDF pages so they don't pile up as windows. Never raises."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from utils import data_manager as dm

    try:
        if details.get("map_figure") is not None:
            dm.save_figure(details["map_figure"], type(file_path)(str(file_path) + "-field-map"))
        pages = details.get("single_nv_figures") or []
        if pages:
            pdf_path = str(file_path) + "-single-nvs.pdf"
            with PdfPages(pdf_path) as pdf:
                for page in pages:
                    pdf.savefig(page)
            for page in pages:
                plt.close(page)
            print(f"Single-NV fits: {pdf_path}")
    except Exception as exc:
        print(f"Saving the field map or single-NV fits failed ({exc}); the data is saved.")


def save_reanalysis(results, file_stem, source_file, report=None):
    """Save a re-analysis next to the routine's data: the numbers (data file), the main figure,
    the field map, and the single-NV fits (PDF), all under one timestamp. Returns the path."""
    from matplotlib.backends.backend_pdf import PdfPages

    from utils import data_manager as dm

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(source_file, timestamp, file_stem + "-reanalysis")
    figure_keys = ("figure", "map_figure", "single_nv_figures")
    summary = {key: value for key, value in results.items() if key not in figure_keys}
    summary |= {
        "timestamp": timestamp,
        "source_file_stem": file_stem,
        "cut_settings": {
            "exclude_nvs": list(EXCLUDE_NVS),
            "min_contrast_sigma": MIN_CONTRAST_SIGMA,
            "min_contrast_fraction": MIN_CONTRAST_FRACTION,
            "max_k_fraction": MAX_K_FRACTION,
            "detection_dchi2": DETECTION_DCHI2,
        },
    }
    if report:
        summary["report"] = report.splitlines()
    dm.save_raw_data(summary, file_path)
    if results.get("figure") is not None:
        dm.save_figure(results["figure"], file_path)
    if results.get("map_figure") is not None:
        dm.save_figure(results["map_figure"], dm.get_file_path(source_file, timestamp, file_stem + "-reanalysis-field-map"))
    pages = results.get("single_nv_figures") or []
    if pages:
        with PdfPages(str(file_path) + "-single-nvs.pdf") as pdf:
            for page in pages:
                pdf.savefig(page)
    print(f"\nSaved the re-analysis (numbers, figure{', field map' if results.get('map_figure') is not None else ''}"
          f"{', single-NV fits PDF' if pages else ''}): {file_path}")
    return file_path


def reanalyze(file_stem, source_file, details=True):
    """Load a saved run, analyze it (printing the report), and save everything."""
    import contextlib
    import io

    from utils import data_manager as dm

    data = dm.get_raw_data(file_stem=file_stem, load_npz=True)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        results = analyze(data, details=details)
    report = buffer.getvalue()
    print(report, end="")
    save_reanalysis(results, file_stem, source_file, report)
    return results


def main(nv_list, num_reps, num_runs, amps_v, freq_hz=100e3, phases_deg=(90.0, 0.0), pulse_phases_deg=None,
         attenuation_db=None, uwave_ind_list=(1,)):
    """amps_v: tone amplitudes (V into 50 ohm at the OPX output), together within 0.5 V.
    attenuation_db: what is in the chain during this run, recorded with the data.
    pulse_phases_deg: None for the plain echo (x,x,x without IQ, as before), or the IQ phases
    of (first pi/2, pi, last pi/2) in degrees, which turns IQ mode on: (0, 0, 0) is x,x,x
    through the IQ path, (0, 90, 0) is x,y,x, and (0, 90, 90) is x,y,y. When the last pi/2 is
    90 deg from the first, the echo reads the other quadrature (signal c + b sin(k A)), which
    the analysis handles."""
    from majorroutines.widefield import base_routine
    from utils import data_manager as dm
    from utils import kplotlib as kpl
    from utils import tool_belt as tb
    from utils import widefield

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "spin_echo_tone.py"
    uwave_ind_list = [int(ind) for ind in uwave_ind_list]
    amps_v = [float(amp) for amp in amps_v]
    phases_deg = [float(phase) for phase in phases_deg]
    num_steps = len(phases_deg) * len(amps_v)
    pulse_phases = None if pulse_phases_deg is None else [float(p) for p in pulse_phases_deg]
    if pulse_phases is not None and len(pulse_phases) != 3:
        raise ValueError("pulse_phases_deg needs three phases: first pi/2, pi, last pi/2")
    quadrature = "cos"
    if pulse_phases is not None and int(round(pulse_phases[2] - pulse_phases[0])) % 180 == 90:
        quadrature = "sin"

    def run_fn(shuffled_step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            [int(ind) for ind in shuffled_step_inds],
            float(freq_hz),
            phases_deg,
            amps_v,
            pulse_phases,
        ]
        pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args), num_reps)

    # IQ mode only when pulse phases are given, so the plain echo runs exactly as before
    raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, run_fn, uwave_ind_list=uwave_ind_list,
        load_iq=pulse_phases is not None,
    )

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "amps_v": amps_v,
        "amps_v-units": "V into 50 ohm at the OPX output",
        "phases_deg": phases_deg,
        "freq_hz": float(freq_hz),
        "attenuation_db": attenuation_db,
        "pulse_phases_deg": pulse_phases,
        "readout_quadrature": quadrature,
        "step_order": "step = phase_index * len(amps_v) + amplitude_index",
    }

    fig = None
    details = {}
    try:
        results = analyze(raw_data, details=True)
        fig = results.pop("figure")
        details = {key: results.pop(key) for key in ("map_figure", "single_nv_figures") if key in results}
        raw_data |= results
    except Exception as exc:
        print(f"Analysis failed ({exc}); the raw data is still saved.")

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_sig.name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)
    save_detail_figures(details, file_path)

    tb.reset_cfm()
    kpl.show()
    return raw_data



if __name__ == "__main__":
    # Re-analyze a saved run with a closer look at the single NVs, and save it: put its file stem here
    from utils import kplotlib as kpl

    FILE_STEM = "2026_11_16-05_03_55-johnson-nv0_2026_10_02"  # Hahn echo at 16.7 kHz, 30 dB

    kpl.init_kplotlib()
    reanalyze(FILE_STEM, __file__)
    kpl.show(block=True)
