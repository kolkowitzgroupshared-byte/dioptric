# -*- coding: utf-8 -*-
"""
Analyze a calibrate_iq_delay run: how well the IQ waveforms and the microwave switch line up,
separately for the I channel (x pulses) and the Q channel (y pulses).

What the routine does (servers/.../camera/calibrate_iq_delay.py): at each step it plays a pi
pulse on the I element (experiment 0, an x pulse) or the Q element (experiment 1, a y pulse)
starting at time 0, and opens the switch tau later for the same length. The switch element
already carries the config's iq_delay (140 ns) in every play, so tau is EXTRA delay on top of
iq_delay, and tau = 0 is exactly how normal IQ pulses are played. There are no reference shots.

As tau grows, the IQ waveform slides out of the switch window: the overlap shrinks by tau and
the pulse's rotation with it. Each channel is fitted with
    signal = c + a sin^2(theta_full/2 * overlap/T),   overlap = clip(T - |tau - tau0|, 0, T)
with T the pi pulse length from the config. tau0 is where the two line up (tau0 = 0 means
iq_delay is right; tau0 < 0 means the switch should open earlier, iq_delay + tau0). The sweep
starts at 20 ns (the shortest QUA wait), so tau0 is extrapolated and depends somewhat on the
assumed full-pulse rotation; both pi and the Rabi-implied value are reported.

Comparing the channels: if the y pulses were weaker (a_Q < a_I) or shifted (tau0_Q != tau0_I),
XY sequences would fail while x-only sequences work.

Run from the dioptric folder in the dioptric environment:
    python iq_delay_analysis.py

Created on November 17th, 2026; rewritten for the two-channel measurement the same day
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FILE_STEMS = ["2026_11_17-12_33_18-johnson-nv0_2026_10_02"]  # the saved .txt names without extension
EXCLUDE_NVS = [0, 6]  # another orientation
THRESHOLD = True


def _tidy(fig):
    """tight_layout unless the plotting style already set a layout engine (kplotlib)."""
    get_engine = getattr(fig, "get_layout_engine", None)
    if get_engine is not None and get_engine() is not None:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except RuntimeError:
            pass


def _clean(counts):
    """Replace scattered invalid counts (inf/NaN) with the median of the same experiment, NV
    and step over the other runs and reps. Returns counts and the number replaced."""
    bad = ~np.isfinite(counts)
    if not bad.any():
        return counts, 0
    clean = np.where(bad, np.nan, counts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = np.nanmedian(clean, axis=(2, 4), keepdims=True)
    return np.where(bad, np.broadcast_to(fill, counts.shape), counts), int(bad.sum())


def configured_delay(data):
    """The iq_delay in the run's saved config, if it was saved."""
    try:
        return float(data["config"]["CommonDurations"]["iq_delay"])
    except (KeyError, TypeError, ValueError):
        return None


def load(file_stems):
    from utils import data_manager as dm

    first, parts = None, []
    for stem in file_stems:
        data = dm.get_raw_data(file_stem=stem, load_npz=True)
        if first is None:
            first = data
        elif not np.allclose(np.asarray(data["taus"], float), np.asarray(first["taus"], float)):
            raise ValueError(f"{stem} has different delays")
        # Saved files store counts as float16 (max 65504): convert first
        parts.append(np.asarray(data["counts"], dtype=np.float64))
    return first, np.concatenate(parts, axis=2)


def nv_average(curves, errors):
    curves, errors = np.asarray(curves, float), np.asarray(errors, float)
    ok = np.isfinite(curves) & np.isfinite(errors) & (errors > 0)
    w = np.where(ok, 1 / np.where(errors > 0, errors, 1) ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(ok, curves, 0) * w).sum(axis=0) / w.sum(axis=0)
        err = 1 / np.sqrt(w.sum(axis=0))
    return mean, err


CHANNELS = ("x pulse (I)", "y pulse (Q)")


def pulse_length_ns(data):
    """The pi pulse length used, from the run's saved config (fallback 160 ns)."""
    try:
        ind = data.get("uwave_ind", [1])[0]
        sig_gens = data["config"]["Microwaves"]["VirtualSigGens"]
        sg = sig_gens.get(str(ind), sig_gens.get(ind)) if isinstance(sig_gens, dict) else sig_gens[ind]
        return float(sg["pi_pulse"])
    except (KeyError, TypeError, ValueError, IndexError):
        return 160.0


def rabi_rotation(data, t_pulse):
    """Full-pulse rotation implied by the config's Rabi period (in units of pi), if saved."""
    try:
        ind = data.get("uwave_ind", [1])[0]
        sig_gens = data["config"]["Microwaves"]["VirtualSigGens"]
        sg = sig_gens.get(str(ind), sig_gens.get(ind)) if isinstance(sig_gens, dict) else sig_gens[ind]
        return min(2 * t_pulse / float(sg["rabi_period"]), 1.0)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def overlap_model(t, c, a, t0, theta_full, t_pulse):
    ov = np.clip(t_pulse - np.abs(t - t0), 0, t_pulse)
    return c + a * np.sin(theta_full * ov / t_pulse / 2) ** 2


def fit_channel(taus, mean, err, t_pulse, theta_full):
    """Fit c, a, tau0 with the full-pulse rotation fixed. Returns dict or None."""
    f = lambda t, c, a, t0: overlap_model(t, c, a, t0, theta_full, t_pulse)
    hi_first = mean[: max(2, len(mean) // 4)].mean() > mean[-max(2, len(mean) // 4):].mean()
    c0 = float(np.median(mean[-max(3, len(mean) // 3):])) if hi_first else float(np.median(mean[: max(3, len(mean) // 3)]))
    a0 = float(mean.max() - mean.min()) * (1 if hi_first else -1)
    try:
        popt, pcov = curve_fit(f, taus, mean, p0=[c0, a0, 0.0], sigma=err, absolute_sigma=True,
                               bounds=([-np.inf, -np.inf, taus.min() - 2 * t_pulse], [np.inf, np.inf, taus.max()]),
                               maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((mean - f(taus, *popt)) / err) ** 2))
    return {"c": float(popt[0]), "a": float(popt[1]), "a_err": float(perr[1]), "tau0": float(popt[2]),
            "tau0_err": float(perr[2]), "chi2_dof": chi2 / max(len(taus) - 3, 1), "theta_full": theta_full,
            "popt": popt}


def report(fits, taus, cfg_delay, t_pulse, n_nvs, n_runs, replaced):
    print(f"\nIQ delay calibration: {n_nvs} NVs (excluded {EXCLUDE_NVS}), {n_runs} runs, extra switch delay "
          f"{taus.min():g} to {taus.max():g} ns; pulses {t_pulse:g} ns")
    if replaced:
        print(f"  Data: {replaced} invalid count values replaced by medians of the same NV and step")
    if cfg_delay is not None:
        print(f"  Configured iq_delay (on the switch): {cfg_delay:g} ns; tau = 0 is how normal IQ pulses are played")
    for assumption, per_channel in fits.items():
        print(f"  Assuming the full pulse rotates by {assumption}:")
        for name, fit in per_channel.items():
            if fit is None:
                print(f"    {name}: fit failed")
                continue
            print(f"    {name}: aligned at tau0 = {fit['tau0']:+.1f} +/- {fit['tau0_err']:.1f} ns, "
                  f"pulse effect a = {fit['a']:.3f} +/- {fit['a_err']:.3f}, chi2/dof {fit['chi2_dof']:.1f}")
        fi, fq = per_channel.get(CHANNELS[0]), per_channel.get(CHANNELS[1])
        if fi is not None and fq is not None:
            dt = fq["tau0"] - fi["tau0"]
            dt_err = np.hypot(fq["tau0_err"], fi["tau0_err"])
            ratio = fq["a"] / fi["a"] if fi["a"] != 0 else np.nan
            ratio_err = abs(ratio) * np.hypot(fq["a_err"] / fq["a"], fi["a_err"] / fi["a"]) if fq["a"] != 0 else np.nan
            print(f"    y against x: timing {dt:+.1f} +/- {dt_err:.1f} ns, strength ratio {ratio:.3f} +/- {ratio_err:.3f}")
    # verdict from the pi assumption
    base = fits.get("pi") or next(iter(fits.values()))
    fi, fq = base.get(CHANNELS[0]), base.get(CHANNELS[1])
    if fi is not None and fq is not None:
        t0 = 0.5 * (fi["tau0"] + fq["tau0"])
        print(f"  -> The switch window and the IQ waveforms line up {abs(t0):.0f} ns "
              f"{'earlier' if t0 < 0 else 'later'} than now"
              + (f": iq_delay of about {cfg_delay + t0:.0f} ns instead of {cfg_delay:g}" if cfg_delay is not None else "")
              + f". A {abs(t0):.0f} ns misalignment leaves {100 * max(0, 1 - abs(t0) / t_pulse):.0f} % of each pulse inside the window.")
        ratio = fq["a"] / fi["a"] if fi["a"] != 0 else np.nan
        dt = fq["tau0"] - fi["tau0"]
        if abs(ratio - 1) < 0.1 and abs(dt) < 10:
            print("  -> x and y pulses agree in strength and timing: the IQ channels are balanced.")
        else:
            print("  -> x and y pulses differ" + (f" in strength (ratio {ratio:.2f})" if abs(ratio - 1) >= 0.1 else "")
                  + (f" in timing ({dt:+.0f} ns)" if abs(dt) >= 10 else "") + ": this would hurt XY sequences.")


def plot(taus, means, errs, curves_i, nv_numbers, fits, cfg_delay, t_pulse):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    dense = np.linspace(min(taus.min(), -40), taus.max(), 500)
    base = fits.get("pi") or next(iter(fits.values()))
    for (name, mean), err, color in zip(means.items(), errs.values(), ("tab:blue", "tab:orange")):
        ax.errorbar(taus, mean, yerr=err, fmt="o", ms=4, color=color, label=name)
        fit = base.get(name)
        if fit is not None:
            ax.plot(dense, overlap_model(dense, fit["c"], fit["a"], fit["tau0"], fit["theta_full"], t_pulse), "-",
                    lw=1, color=color, alpha=0.8)
            ax.axvline(fit["tau0"], color=color, ls="--", lw=0.8)
    ax.axvline(0, color="0.4", ls=":", lw=1, label="tau = 0: how pulses are played now")
    ax.axvspan(dense.min(), taus.min(), color="0.92", label="not measured (extrapolated)")
    ax.set_xlabel(f"Extra switch delay tau (ns), on top of iq_delay {cfg_delay:g} ns" if cfg_delay is not None
                  else "Extra switch delay tau (ns)")
    ax.set_ylabel("Signal, NV average")
    ax.set_title("IQ waveform against switch window, x and y pulses")
    ax.legend(fontsize=8)

    ax = axes[1]
    rel = curves_i - np.nanmedian(curves_i, axis=1, keepdims=True)
    lim = np.nanpercentile(np.abs(rel), 98)
    step = float(np.median(np.diff(taus)))
    im = ax.imshow(rel, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                   extent=[taus.min() - step / 2, taus.max() + step / 2, len(curves_i) - 0.5, -0.5])
    ax.set_xlabel("Extra switch delay tau (ns)")
    ax.set_ylabel("NV (row; in order of the NV list)")
    ax.set_title("x pulses: each NV relative to its own typical level")
    fig.colorbar(im, ax=ax, label="Change in signal")
    ticks = np.linspace(0, len(nv_numbers) - 1, min(8, len(nv_numbers))).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(nv_numbers[i]) for i in ticks])
    _tidy(fig)
    return fig


def _process(widefield, nvs, counts):
    """NV averages per step for one experiment, without a reference (this routine has none)."""
    try:
        processed = widefield.process_counts(nvs, counts, threshold=THRESHOLD)
        return np.asarray(processed[0], float), np.asarray(processed[1], float)
    except TypeError:
        # older process_counts that needs a reference: fall back to raw counts
        avg = counts.mean(axis=(1, 3))
        ste = counts.std(axis=(1, 3)) / np.sqrt(counts.shape[1] * counts.shape[3])
        return avg, ste


def analyze(file_stems=FILE_STEMS):
    import matplotlib.pyplot as plt

    from utils import data_manager as dm
    from utils import widefield

    data, counts = load(file_stems)
    counts, replaced = _clean(counts)
    nv_list, taus = data["nv_list"], np.asarray(data["taus"], float)
    keep = [i for i in range(len(nv_list)) if i not in set(EXCLUDE_NVS)]
    nvs = [nv_list[i] for i in keep]
    means, errs, curves = {}, {}, {}
    for k, name in enumerate(CHANNELS):
        c, e = _process(widefield, nvs, counts[k][keep])
        curves[name] = c
        means[name], errs[name] = nv_average(c, e)
    cfg_delay = configured_delay(data)
    t_pulse = pulse_length_ns(data)
    assumptions = {"pi": np.pi}
    rr = rabi_rotation(data, t_pulse)
    if rr is not None and abs(rr - 1) > 0.02:
        assumptions[f"{rr:.2f} pi (Rabi period)"] = rr * np.pi
    fits = {label: {name: fit_channel(taus, means[name], errs[name], t_pulse, theta)
                    for name in CHANNELS} for label, theta in assumptions.items()}
    report(fits, taus, cfg_delay, t_pulse, len(keep), counts.shape[2], replaced)
    fig = plot(taus, means, errs, curves[CHANNELS[0]], keep, fits, cfg_delay, t_pulse)

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, f"{file_stems[0]}-iq-delay-analysis")
    summary = {
        "source_files": list(file_stems),
        "nv_indices": keep,
        "taus_ns": taus.tolist(),
        "configured_iq_delay_ns": cfg_delay,
        "pulse_length_ns": t_pulse,
        "nv_average": {name: means[name].tolist() for name in CHANNELS},
        "nv_average_err": {name: errs[name].tolist() for name in CHANNELS},
        "fits": {label: {name: (None if f is None else {k: v for k, v in f.items() if k != "popt"})
                         for name, f in per.items()} for label, per in fits.items()},
        "invalid_values_replaced": replaced,
    }
    dm.save_raw_data(summary, file_path)
    dm.save_figure(fig, file_path)
    print(f"\nSaved to {file_path}")
    plt.show(block=True)
    return fits


if __name__ == "__main__":
    analyze()
