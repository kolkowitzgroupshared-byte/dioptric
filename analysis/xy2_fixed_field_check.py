# -*- coding: utf-8 -*-
"""
How much echo does the XY2 run keep? The NV-averaged signal is fitted with the field fixed at
the spin echo's value (1,870 nT/V at 16.7 kHz, 20 dB), leaving only the offset, the swing and
the tone's phase free. The swing is compared with the IQ Hahn echo's (about 0.58 on the same
normalized scale). A noisy free fit can lock onto wiggles; with the field fixed, the fit only
asks how large the expected oscillation is.

Run from the dioptric folder in the dioptric environment:
    python xy2_fixed_field_check.py

Created on November 18th, 2026
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit

FILE_STEM = "2026_11_18-16_10_31-johnson-nv0_2026_10_02"  # XY2 at 16.7 kHz, stopped by an OPX timeout
RUNS_DONE = 6  # runs 0-5 finished before the timeout (None = use all runs)
FIELD_NT_PER_V = 1870.0  # spin echo at 16.7 kHz with 20 dB in the chain
HAHN_SWING = 0.58  # IQ Hahn echo, x,x,x, peak to peak on the normalized scale


def fixed_field_fit(raw_data, runs_done=None, field_nt_per_v=FIELD_NT_PER_V, hahn_swing=HAHN_SWING):
    from majorroutines.widefield import spin_echo_tone as T
    from utils import widefield

    counts = np.asarray(raw_data["counts"], dtype=np.float64)
    if runs_done is not None:
        counts = counts[:, :, :runs_done]
    counts, nv_idx, _ = T._clean_counts(counts, raw_data["nv_list"], T.EXCLUDE_NVS)
    nvs = [raw_data["nv_list"][i] for i in nv_idx]
    amps = np.asarray(raw_data["amps_v"], float)
    phases = np.asarray(raw_data["phases_deg"], float)
    processed = widefield.process_counts(nvs, counts[0], counts[1], threshold=T.THRESHOLD)
    avg = np.reshape(np.asarray(processed[0], float), (len(nvs), len(phases), len(amps)))
    ste = np.reshape(np.asarray(processed[1], float), (len(nvs), len(phases), len(amps)))
    w = np.where(np.isfinite(ste) & (ste > 0) & np.isfinite(avg), 1 / np.where(ste > 0, ste, 1) ** 2, 0.0)
    mean = (np.where(w > 0, avg, 0) * w).sum(axis=0) / w.sum(axis=0)
    err = 1 / np.sqrt(w.sum(axis=0))

    K = field_nt_per_v * T.phase_per_nt(float(raw_data["freq_hz"]), raw_data.get("sensing_time_s"))  # rad per V

    def model(_, c, b, psi):
        phi = K * np.outer(np.cos(np.radians(phases - psi)), amps)
        return (c + b * np.cos(phi)).ravel()

    best = None
    for psi0 in (0.0, 45.0, 90.0, 135.0):
        for b0 in (-0.2, 0.2):
            try:
                popt, pcov = curve_fit(model, None, mean.ravel(), p0=[np.mean(mean), b0, psi0],
                                       sigma=err.ravel(), absolute_sigma=True, maxfev=20000)
            except RuntimeError:
                continue
            chi2 = np.sum(((mean.ravel() - model(None, *popt)) / err.ravel()) ** 2)
            if best is None or chi2 < best[0]:
                best = (chi2, popt, np.sqrt(np.diag(pcov)))
    chi2, popt, perr = best
    swing, swing_err = 2 * abs(popt[1]), 2 * perr[1]
    print(f"Field fixed at {field_nt_per_v:g} nT/V ({K:.2f} rad/V), {counts.shape[2]} runs, {len(nvs)} NVs:")
    print(f"  NV-average level {popt[0]:.2f}; swing {swing:.3f} +/- {swing_err:.3f} -> "
          f"{100 * swing / hahn_swing:.0f} +/- {100 * swing_err / hahn_swing:.0f} % of the IQ Hahn echo's {hahn_swing}")
    print(f"  tone phase {popt[2] % 180:.0f} deg; chi2 per degree of freedom {chi2 / (mean.size - 3):.2f}")
    return swing, swing_err


if __name__ == "__main__":
    from utils import data_manager as dm

    data = dm.get_raw_data(file_stem=FILE_STEM, load_npz=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fixed_field_fit(data, runs_done=RUNS_DONE)
