# -*- coding: utf-8 -*-
"""
Fit the NV-averaged ESR spectrum, to set the drive frequency for Rabi.

All NVs on one orientation share the same lines, so their spectra are averaged
(weighted by each point's uncertainty), which cuts the noise by roughly the square root
of the number of NVs. Each line is then fitted with the pulsed-ESR line shape: the
spin-flip probability after a pi pulse, against detuning,
    P(d) = R^2 / (R^2 + d^2) * sin^2(pi/2 * sqrt(1 + d^2 / R^2)),
with R the Rabi frequency (R = 1 / (2 * pi length)), summed over the three 14N
hyperfine lines HYPERFINE_MHZ apart. Free parameters per line: center, R, height and
background. The center's uncertainty comes from the fit and from a bootstrap over NVs
(resampling which NVs go into the average and refitting).

Lines outside the scanned range are skipped, so a fine scan around one line works too.
If PULSE_NS is set, the line shape uses that pulse length (exact for any rotation);
if it is None, the pulse is assumed to be a pi pulse at the fitted Rabi frequency.

Run from the dioptric folder in the dioptric environment:
    python esr_nv_average_fit.py

Control panel function for a fine scan of the ms = -1 line (0.5 MHz steps over 20 MHz):

def do_resonance_fine(nv_list):
    freq_center = 2.79151
    freq_range = 0.020
    num_steps = 41
    num_reps = 3
    num_runs = 100
    freqs = calculate_freqs(freq_center, freq_range, num_steps)
    freqs = sorted(set(freqs))
    num_steps = len(freqs)
    resonance.main(nv_list, num_steps, num_reps, num_runs, freqs=freqs, uwave_ind_list=[1])

Created on November 10th, 2026; one-line scans and PULSE_NS added on November 11th, 2026
"""

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
FILE_STEM = "2026_11_12-15_45_07-johnson-nv0_2026_10_02"
EXCLUDE_NVS = [0, 6]  # another orientation (no Rabi oscillation at 2.7915 GHz)
THRESHOLD = True  # as in sc_resonance_analysis_optimized.py; False uses raw counts
LINE_GUESSES_GHZ = (2.79151, 2.95078)  # ms = -1 (the drive line) and ms = +1; lines outside the scan are skipped
FIT_WINDOW_MHZ = 25.0  # each line is fitted within this distance of its guess
HYPERFINE_MHZ = 2.16  # 14N splitting; 0 fits a single line
PULSE_NS = None  # ESR pulse length in ns (e.g. 172); None assumes a pi pulse at the fitted Rabi frequency
N_BOOTSTRAP = 200
GAMMA_MHZ_PER_MT = 28.025  # NV gyromagnetic ratio


# ---------------------------------------------------------------------------
# Line shape and fit
# ---------------------------------------------------------------------------
def pi_pulse_line(freq_ghz, center_ghz, rabi_mhz):
    """Spin-flip probability after the ESR pulse, at Rabi frequency rabi_mhz.

    The pulse lasts PULSE_NS, or a pi pulse (1 / (2 * rabi_mhz)) if PULSE_NS is None.
    """
    detuning = (np.asarray(freq_ghz, float) - center_ghz) * 1e3  # MHz
    pulse_us = 1 / (2 * rabi_mhz) if PULSE_NS is None else PULSE_NS * 1e-3
    effective = np.sqrt(rabi_mhz**2 + detuning**2)  # MHz
    return (rabi_mhz / effective) ** 2 * np.sin(np.pi * pulse_us * effective) ** 2


def line_model(freq_ghz, background, height, center_ghz, rabi_mhz):
    if HYPERFINE_MHZ:
        shifts = np.array([-1.0, 0.0, 1.0]) * HYPERFINE_MHZ * 1e-3
        shape = sum(pi_pulse_line(freq_ghz, center_ghz + s, rabi_mhz) for s in shifts) / 3
    else:
        shape = pi_pulse_line(freq_ghz, center_ghz, rabi_mhz)
    return background + height * shape


def fit_line(freqs_ghz, y, y_err, guess_ghz, window_mhz=FIT_WINDOW_MHZ):
    """Fit one line near guess_ghz. Returns (popt, perr, reduced chi^2, window mask)."""
    freqs_ghz = np.asarray(freqs_ghz, float)
    near = np.abs(freqs_ghz - guess_ghz) * 1e3 <= window_mhz
    near &= np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
    if near.sum() < 6:
        raise ValueError(f"Only {near.sum()} usable points near {guess_ghz} GHz")
    f, v, e = freqs_ghz[near], np.asarray(y)[near], np.asarray(y_err)[near]
    background0 = float(np.median(np.concatenate([v[:2], v[-2:]])))
    peak = int(np.argmax(np.abs(v - background0)))
    if PULSE_NS is None:
        rabi0, rabi_lo, rabi_hi = 5.0, 0.5, 30.0
    else:
        # Rotation between 0.2 pi and 1.8 pi: start at pi, and keep away from the 3 pi solution
        pulse_us = PULSE_NS * 1e-3
        rabi0, rabi_lo, rabi_hi = 0.5 / pulse_us, 0.1 / pulse_us, 0.9 / pulse_us
    p0 = [background0, float(v[peak] - background0), float(f[peak]), rabi0]
    half = window_mhz * 1e-3
    bounds = ([-np.inf, -np.inf, guess_ghz - half, rabi_lo], [np.inf, np.inf, guess_ghz + half, rabi_hi])
    popt, pcov = curve_fit(line_model, f, v, p0=p0, sigma=e, absolute_sigma=True,
                           bounds=bounds, maxfev=20000)
    red_chi2 = float(np.sum(((v - line_model(f, *popt)) / e) ** 2) / max(len(f) - len(popt), 1))
    # Inflate the errors if the scatter is larger than the error bars claim
    perr = np.sqrt(np.diag(pcov)) * np.sqrt(max(red_chi2, 1.0))
    return popt, perr, red_chi2, near


def nv_average(spectra, errors):
    """Inverse-variance weighted average over NVs, per frequency."""
    spectra = np.asarray(spectra, float)
    errors = np.asarray(errors, float)
    bad = ~np.isfinite(spectra) | ~np.isfinite(errors) | (errors <= 0)
    weights = np.where(bad, 0.0, 1.0 / np.where(bad, 1.0, errors) ** 2)
    total = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(bad, 0.0, spectra) * weights).sum(axis=0) / total
        err = 1 / np.sqrt(total)
    return mean, err


def bootstrap_center(freqs_ghz, spectra, errors, guess_ghz, n_boot=N_BOOTSTRAP, seed=0):
    """Spread of the fitted center when the averaged NVs are resampled."""
    rng = np.random.default_rng(seed)
    num_nvs = len(spectra)
    centers = []
    for _ in range(n_boot):
        pick = rng.integers(0, num_nvs, num_nvs)
        mean, err = nv_average(spectra[pick], errors[pick])
        try:
            centers.append(fit_line(freqs_ghz, mean, err, guess_ghz)[0][2])
        except Exception:
            pass
    return float(np.std(centers)) if len(centers) > 10 else np.nan


def split_blocks(counts, num_steps):
    """Resonance layout [sig0 | sig1 | ref0 | ref1], as in sc_resonance_analysis_optimized.py."""
    counts = np.asarray(counts)
    n = num_steps // 4
    sig = np.concatenate((counts[:, :, 0:n, :], counts[:, :, n:2 * n, :]), axis=3)
    ref = np.empty((counts.shape[0], counts.shape[1], n, 2 * counts.shape[3]), dtype=counts.dtype)
    ref[:, :, :, 0::2] = counts[:, :, 2 * n:3 * n, :]
    ref[:, :, :, 1::2] = counts[:, :, 3 * n:, :]
    return sig, ref


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def fit_average(freqs_ghz, spectra, errors, n_boot=N_BOOTSTRAP):
    """Fit both lines of the NV-averaged spectrum; returns a dict of results."""
    mean, err = nv_average(spectra, errors)
    freqs_ghz = np.asarray(freqs_ghz, float)
    lines = {}
    for name, guess in zip(("ms_minus1", "ms_plus1"), LINE_GUESSES_GHZ):
        if not freqs_ghz.min() <= guess <= freqs_ghz.max():
            continue  # this line is outside the scanned range
        popt, perr, red_chi2, near = fit_line(freqs_ghz, mean, err, guess)
        boot = bootstrap_center(freqs_ghz, spectra, errors, guess, n_boot) if n_boot else np.nan
        lines[name] = {
            "center_ghz": float(popt[2]),
            "center_fit_err_mhz": float(perr[2] * 1e3),
            "center_bootstrap_err_mhz": float(boot * 1e3),
            "rabi_mhz": float(popt[3]),
            "rabi_err_mhz": float(perr[3]),
            "height": float(popt[1]),
            "background": float(popt[0]),
            "reduced_chi2": red_chi2,
            "popt": popt,
            "window": near,
        }
    if not lines:
        raise ValueError("None of LINE_GUESSES_GHZ is inside the scanned range")
    results = {"mean": mean, "err": err, "lines": lines}
    if "ms_minus1" in lines and "ms_plus1" in lines:
        f1, f2 = lines["ms_minus1"]["center_ghz"], lines["ms_plus1"]["center_ghz"]
        splitting_mhz = (f2 - f1) * 1e3
        results |= {
            "center_ghz": (f1 + f2) / 2,
            "splitting_mhz": splitting_mhz,
            "field_mt": splitting_mhz / (2 * GAMMA_MHZ_PER_MT),
        }
    return results


def report(results, num_used, num_excluded):
    print(f"\nNV-averaged ESR: {num_used} NVs averaged, {num_excluded} excluded")
    for label, key in (("ms = -1 (drive line)", "ms_minus1"), ("ms = +1", "ms_plus1")):
        if key not in results["lines"]:
            continue
        line = results["lines"][key]
        print(f"  {label}: {line['center_ghz']:.6f} GHz  "
              f"+/- {line['center_fit_err_mhz']:.3f} MHz (fit), "
              f"+/- {line['center_bootstrap_err_mhz']:.3f} MHz (bootstrap over NVs); "
              f"reduced chi2 {line['reduced_chi2']:.2f}")
        print(f"      Rabi frequency from the line shape {line['rabi_mhz']:.2f} +/- {line['rabi_err_mhz']:.2f} MHz "
              f"(pi pulse about {1e3 / (2 * line['rabi_mhz']):.0f} ns)")
    if "center_ghz" in results:
        print(f"  Center {results['center_ghz']:.6f} GHz, splitting {results['splitting_mhz']:.3f} MHz, "
              f"field along the NV axis {results['field_mt']:.4f} mT")
    if "ms_minus1" in results["lines"]:
        print(f"\n  -> Drive frequency for Rabi (signal generator 1): "
              f"{results['lines']['ms_minus1']['center_ghz']:.6f} GHz")


def plot(freqs_ghz, results):
    import matplotlib.pyplot as plt

    keys = [key for key in ("ms_minus1", "ms_plus1") if key in results["lines"]]
    fig, axes = plt.subplots(1, 1 + len(keys), figsize=(7.5 + 3.75 * len(keys), 4.2),
                             gridspec_kw={"width_ratios": [2] + [1] * len(keys)})
    dense = np.linspace(freqs_ghz.min(), freqs_ghz.max(), 2000)
    axes[0].errorbar(freqs_ghz, results["mean"], yerr=results["err"], fmt="o", ms=3, label="NV average")
    for key, ax in zip(keys, axes[1:]):
        line = results["lines"][key]
        near = line["window"]
        zoom = np.linspace(freqs_ghz[near].min(), freqs_ghz[near].max(), 500)
        axes[0].plot(dense, line_model(dense, *line["popt"]), "-", lw=1)
        ax.errorbar(freqs_ghz[near], results["mean"][near], yerr=results["err"][near], fmt="o", ms=4)
        ax.plot(zoom, line_model(zoom, *line["popt"]), "-")
        ax.axvline(line["center_ghz"], color="k", ls="--", lw=1)
        ax.set_title(f"{key.replace('_', ' ')}: {line['center_ghz']:.5f} GHz")
        ax.set_xlabel("Frequency (GHz)")
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_ylabel("Normalized NV population (NV average)")
    axes[0].legend()
    fig.tight_layout()
    return fig


def analyze(file_stem=FILE_STEM):
    import matplotlib.pyplot as plt

    from utils import data_manager as dm
    from utils import widefield

    data = dm.get_raw_data(file_stem=file_stem, load_npz=True, use_cache=True)
    nv_list = data["nv_list"]
    freqs = np.asarray(data["freqs"], float)
    counts = np.asarray(data["counts"])[0]
    sig, ref = split_blocks(counts, int(data["num_steps"]))
    sig, ref = sig.astype(np.float32), ref.astype(np.float32)

    processed = widefield.process_counts(nv_list, sig, ref, threshold=THRESHOLD)
    spectra, errors = np.asarray(processed[0], float), np.asarray(processed[1], float)

    keep = [ind for ind in range(len(nv_list)) if ind not in set(EXCLUDE_NVS)]
    results = fit_average(freqs, spectra[keep], errors[keep])
    report(results, len(keep), len(nv_list) - len(keep))
    fig = plot(freqs, results)

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, "nv_average_esr")
    summary = {
        "source_file": file_stem,
        "nvs_used": keep,
        "nvs_excluded": list(EXCLUDE_NVS),
        "hyperfine_mhz": HYPERFINE_MHZ,
        "pulse_ns": PULSE_NS,
    }
    summary |= {key: results[key] for key in ("center_ghz", "splitting_mhz", "field_mt") if key in results}
    for key, line in results["lines"].items():
        summary |= {f"{key}_{name}": value for name, value in line.items() if name not in ("popt", "window")}
    dm.save_raw_data(summary, file_path)
    dm.save_figure(fig, file_path)
    print(f"\nSaved to {file_path}")
    plt.show(block=True)
    return results


if __name__ == "__main__":
    analyze()
