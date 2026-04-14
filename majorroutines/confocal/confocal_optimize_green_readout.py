# -*- coding: utf-8 -*-
"""
Sweep the green-laser spin-readout pulse duration and measure ODMR contrast
at each duration to find the readout time that maximizes NV contrast on the
cryo setup.

For each readout_ns in `readout_times_ns`:
    - override nv_sig.pulse_durations[VirtualLaserKey.SPIN_READOUT]
    - run confocal_resonance.main over a narrow frequency window that
      contains a single Zeeman line (e.g., the m_s=-1 peak)
    - fit a single Lorentzian (reuses fit_one_line from
      figures/SingletChar/confocal_resonance_fit.py) and record the fitted
      contrast with its standard error

At the end: save a summary + per-point raw runs, plot contrast vs readout
duration, and print the best readout time.

Created on April 14th, 2026

@author: chemistatcode
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.confocal.confocal_resonance as resonance
from figures.SingletChar.confocal_resonance_fit import fit_one_line
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils.constants import VirtualLaserKey


def main(
    nv_sig,
    readout_times_ns,
    freq_center_ghz,
    freq_span_mhz,
    num_steps,
    num_reps,
    num_runs,
    uwave_ind,
    uwave_power_dbm=None,
    laser_power=None,
    optimize_between_runs=True,
):
    """Optimize the green spin-readout pulse duration via ODMR contrast.

    The scan window (`freq_center_ghz`, `freq_span_mhz`) must contain only
    one Zeeman line -- the fit routine expects a single Lorentzian dip.
    """
    kpl.init_kplotlib()

    readout_times_ns = np.asarray(readout_times_ns, dtype=int).ravel()
    if readout_times_ns.size == 0:
        raise ValueError("readout_times_ns must contain at least one value")

    n = readout_times_ns.size
    contrasts = np.full(n, np.nan)
    contrast_stes = np.full(n, np.nan)
    centers_ghz = np.full(n, np.nan)
    fwhms_ghz = np.full(n, np.nan)
    per_point_runs = []

    timestamp = dm.get_time_stamp()

    for i, readout_ns in enumerate(readout_times_ns):
        readout_ns = int(readout_ns)
        print("\n" + "=" * 64)
        print(f"[{i + 1}/{n}] readout_ns = {readout_ns}")
        print("=" * 64)

        # Deepcopy so the caller's nv_sig isn't mutated across points.
        nv_sig_run = copy.deepcopy(nv_sig)
        nv_sig_run.pulse_durations[VirtualLaserKey.SPIN_READOUT] = readout_ns

        run_data = resonance.main(
            nv_sig_run,
            freq_center_ghz=freq_center_ghz,
            freq_span_mhz=freq_span_mhz,
            num_steps=num_steps,
            num_reps=num_reps,
            num_runs=num_runs,
            uwave_ind=uwave_ind,
            uwave_power_dbm=uwave_power_dbm,
            laser_power=laser_power,
            optimize_between_runs=optimize_between_runs,
            do_plot=False,
        )
        per_point_runs.append(run_data)

        freqs = np.asarray(run_data["freqs_ghz"], dtype=float)
        norm = np.asarray(run_data["norm_mean"], dtype=float)
        valid = np.isfinite(freqs) & np.isfinite(norm)
        freqs, norm = freqs[valid], norm[valid]

        if freqs.size < 4:
            print("  Not enough finite points to fit a Lorentzian.")
            continue

        try:
            popt, pcov = fit_one_line(freqs, norm)
            perr = np.sqrt(np.diag(pcov))
            _, f0, fwhm, contrast = popt
            contrasts[i] = contrast
            contrast_stes[i] = perr[3]
            centers_ghz[i] = f0
            fwhms_ghz[i] = fwhm
            print(
                f"  Fit OK: f0={f0:.6f} GHz, FWHM={fwhm * 1e3:.2f} MHz, "
                f"contrast={contrast:.4f} +/- {perr[3]:.4f}"
            )
        except Exception as e:
            print(f"  Fit failed at readout_ns={readout_ns}: {e}")

    # Save summary + per-point raw runs
    summary = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "readout_times_ns": readout_times_ns.tolist(),
        "contrasts": contrasts.tolist(),
        "contrast_stes": contrast_stes.tolist(),
        "centers_ghz": centers_ghz.tolist(),
        "fwhms_ghz": fwhms_ghz.tolist(),
        "freq_center_ghz": float(freq_center_ghz),
        "freq_span_mhz": float(freq_span_mhz),
        "num_steps": int(num_steps),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_power_dbm": uwave_power_dbm,
        "per_readout_runs": per_point_runs,
    }

    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))

    fig, ax = plt.subplots(figsize=(8, 6))
    ok = np.isfinite(contrasts)
    if np.any(ok):
        ax.errorbar(
            readout_times_ns[ok], contrasts[ok],
            yerr=contrast_stes[ok], fmt="o-",
            color="darkorange", markersize=6, capsize=3, linewidth=1.5,
            label="Fitted ODMR contrast",
        )
        i_best = int(np.nanargmax(contrasts))
        ax.axvline(
            readout_times_ns[i_best], color="gray", linestyle=":",
            linewidth=1,
            label=(
                f"Best: {readout_times_ns[i_best]} ns "
                f"(contrast {contrasts[i_best]:.3f})"
            ),
        )
    ax.set_xlabel("Green readout pulse duration (ns)")
    ax.set_ylabel("ODMR contrast (fitted Lorentzian depth)")
    ax.set_title("Green readout optimization")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()

    dm.save_raw_data(summary, file_path)
    dm.save_figure(fig, file_path)
    print(f"\nSaved sweep to {file_path}")

    print("\n" + "=" * 64)
    print("GREEN READOUT SWEEP SUMMARY")
    print("=" * 64)
    print(f"{'readout (ns)':>14} {'contrast':>12} {'+/- err':>10} {'f0 (GHz)':>12}")
    for t, c, ce, f0 in zip(readout_times_ns, contrasts, contrast_stes, centers_ghz):
        if np.isfinite(c):
            print(f"{int(t):>14d} {c:>12.4f} {ce:>10.4f} {f0:>12.6f}")
        else:
            print(f"{int(t):>14d} {'--':>12} {'--':>10} {'--':>12}")
    if np.any(ok):
        i_best = int(np.nanargmax(contrasts))
        print("-" * 64)
        print(
            f"Best readout: {int(readout_times_ns[i_best])} ns, "
            f"contrast = {contrasts[i_best]:.4f} "
            f"+/- {contrast_stes[i_best]:.4f}"
        )
    print("=" * 64)

    plt.show()
    return summary
